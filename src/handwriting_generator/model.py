import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.modules import LSTM

from handwriting_generator.loss import HandwritingLoss
from handwriting_generator.modules import GaussianWindow, MixtureDensityNetwork

__all__ = ["HandwritingGenerator"]


class HandwritingGenerator(pl.LightningModule):
    def __init__(
        self,
        alphabet_size: int,
        hidden_size: int,
        n_window_components: int,
        n_mixture_components: int,
        *,
        learning_rate: float = 1e-4,
    ):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.n_window_components = n_window_components
        self.n_mixture_components = n_mixture_components
        self.learning_rate = learning_rate

        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        # Gaussian Window layer
        self.window_layer = GaussianWindow(
            input_size=hidden_size, n_components=n_window_components
        )
        # Second LSTM layer, takes as input the concatenation of the input,
        # the output of the first LSTM layer
        # and the output of the Window layer
        self.lstm2_layer = LSTM(
            input_size=3 + hidden_size + alphabet_size + 1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Third LSTM layer, takes as input the concatenation of the output of the first LSTM layer,
        # the output of the second LSTM layer
        # and the output of the Window layer
        self.lstm3_layer = LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        # Mixture Density Network Layer
        self.output_layer = MixtureDensityNetwork(
            input_size=hidden_size, n_mixtures=n_mixture_components
        )

        # Loss function
        self.loss = HandwritingLoss()

    def forward(self, strokes, onehot, bias=None):
        # First LSTM Layer
        input_ = strokes
        output1, _ = self.lstm1_layer(input_)
        # Gaussian Window Layer
        window, phi = self.window_layer(output1, onehot)
        # Second LSTM Layer
        output2, _ = self.lstm2_layer(
            torch.cat((strokes, output1, window), dim=2),
        )
        # Third LSTM Layer
        output3, _ = self.lstm3_layer(output2)
        # MixtureDensityNetwork Layer
        eos, pi, mu1, mu2, sigma1, sigma2, rho = self.output_layer(output3, bias)
        return (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi)

    @staticmethod
    def sample_bivariate_gaussian(pi, mu1, mu2, sigma1, sigma2, rho):
        # Pick distribution from the MixtureDensityNetwork
        p = pi.data[0, 0, :].numpy()
        idx = np.random.choice(p.shape[0], p=p)
        m1 = mu1.data[0, 0, idx]
        m2 = mu2.data[0, 0, idx]
        s1 = sigma1.data[0, 0, idx]
        s2 = sigma2.data[0, 0, idx]
        r = rho.data[0, 0, idx]
        mean = [m1, m2]
        covariance = [[s1**2, r * s1 * s2], [r * s1 * s2, s2**2]]
        Z = torch.autograd.Variable(
            sigma1.data.new(np.random.multivariate_normal(mean, covariance, 1))
        ).unsqueeze(0)
        X = Z[:, :, 0:1]
        Y = Z[:, :, 1:2]
        return X, Y

    def training_step(self, batch, batch_idx):
        # Split data tuple
        strokes, onehot, strokes_lens, onehot_lengths = batch
        # Move inputs to correct device
        # onehot, strokes = onehot.to(self.device), strokes.to(self.device)
        # Main Model Forward Step
        (eos, pi, mu1, mu2, sigma1, sigma2, rho), _ = self(strokes, onehot)

        loss = self.loss(
            eos[:, :-1],
            pi[:, :-1],
            mu1[:, :-1],
            mu2[:, :-1],
            sigma1[:, :-1],
            sigma2[:, :-1],
            rho[:, :-1],
            torch.roll(strokes, -1, dims=1)[:, :-1],
        ) / strokes.size(1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)
        self.log("val_loss", val_loss)

    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num

    @classmethod
    def load_model(cls, parameters: dict, state_dict: dict):
        model = cls(**parameters)
        model.load_state_dict(state_dict)
        return model

    def __deepcopy__(self, *args, **kwargs):
        model = HandwritingGenerator(
            self.alphabet_size,
            self.hidden_size,
            self.n_window_components,
            self.n_mixture_components,
        )
        return model

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return optimizer
