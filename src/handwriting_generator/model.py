import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.distributions
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch.nn.modules import LSTM, BatchNorm1d, Linear, Sequential, Sigmoid
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from handwriting_generator.loss import HandwritingLoss
from handwriting_generator.modules import GaussianWindow, MixtureDensityNetwork
from handwriting_generator.utils import (
    batch_index_select,
    plot_strokes,
    plot_window_weights,
)

__all__ = ["HandwritingGenerator"]


class HandwritingGenerator(pl.LightningModule, HyperparametersMixin):
    def __init__(
        self,
        alphabet_size: int,
        hidden_size: int = 400,
        n_window_components: int = 10,
        n_mixture_components: int = 20,
        *,
        learning_rate: float = 1e-3,
        probability_bias: float = 1.0,
    ):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.n_window_components = n_window_components
        self.n_mixture_components = n_mixture_components
        self.learning_rate = learning_rate
        self.probability_bias = probability_bias

        self.save_hyperparameters()

        # Batch normalization layer
        # This is used to standardize the x and y values
        self.batch_norm = BatchNorm1d(num_features=2)

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
        # Third LSTM layer, takes as input the concatenation of the input,
        # the output of the second LSTM layer
        # and the output of the Window layer
        self.lstm3_layer = LSTM(
            input_size=3 + hidden_size + alphabet_size + 1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Mixture Density Network Layer
        self.mdn_layer = MixtureDensityNetwork(
            input_size=hidden_size, n_mixtures=n_mixture_components
        )

        # EOS Layer
        self.eos_layer = Sequential(
            Linear(in_features=hidden_size, out_features=1),
            Sigmoid(),
        )

        # Loss function
        self.loss = HandwritingLoss()

    def forward(
        self,
        strokes,
        onehot,
        strokes_lengths: torch.Tensor,
        onehot_lengths: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, ...] | None = None,
    ):
        if hidden is None:
            hidden = (None, None, None)
        hidden1, hidden2, hidden3 = hidden
        # Batch normalization
        input_ = torch.clone(strokes)
        assert input_.shape[2] == 3
        input_ = torch.swapaxes(input_, 1, 2)
        input_[:, :2] = self.batch_norm(input_[:, :2])
        assert input_.shape[1] == 3
        input_ = torch.swapaxes(input_, 1, 2)
        # First LSTM Layer
        assert input_.shape[2] == 3
        input_ = pack_padded_sequence(input_, strokes_lengths, batch_first=True)
        out, hidden1 = self.lstm1_layer(input_, hidden1)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Gaussian Window Layer
        window, phi = self.window_layer(out, onehot)
        # Mask Phi values outside of length
        phi_mask = torch.zeros_like(phi, dtype=torch.bool)
        for i, length in enumerate(strokes_lengths):
            phi_mask[i, length:] = True
        for i, length in enumerate(onehot_lengths):
            phi_mask[i, :, length:] = True
        phi = phi.masked_fill(phi_mask, 0.0)
        # Second LSTM Layer
        out = torch.cat((strokes, out, window), dim=2)
        out = pack_padded_sequence(out, strokes_lengths, batch_first=True)
        out, hidden2 = self.lstm2_layer(out, hidden2)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # Third LSTM Layer
        out = torch.cat((strokes, out, window), dim=2)
        out = pack_padded_sequence(out, strokes_lengths, batch_first=True)
        out, hidden3 = self.lstm3_layer(out, hidden3)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # MixtureDensityNetwork Layer
        pi, mu1, mu2, sigma1, sigma2, rho = self.mdn_layer(out, bias)
        # EOS Layer
        eos = self.eos_layer(out)
        return (
            (eos, pi, mu1, mu2, sigma1, sigma2, rho),
            (window, phi),
            (hidden1, hidden2, hidden3),
        )

    @staticmethod
    def sample_point(eos, pi, mu1, mu2, sigma1, sigma2, rho):
        # Pick distribution from the MixtureDensityNetwork
        indices = torch.distributions.Categorical(pi).sample()
        m1 = batch_index_select(mu1, dim=1, indices=indices)
        m2 = batch_index_select(mu2, dim=1, indices=indices)
        s1 = batch_index_select(sigma1, dim=1, indices=indices)
        s2 = batch_index_select(sigma2, dim=1, indices=indices)
        r = batch_index_select(rho, dim=1, indices=indices)
        mean = torch.cat([m1, m2], dim=1)
        covariance = torch.stack(
            [
                torch.cat([s1**2, r * s1 * s2], dim=1),
                torch.cat([r * s1 * s2, s2**2], dim=1),
            ],
            dim=2,
        )
        Z = torch.distributions.MultivariateNormal(mean, covariance).sample()
        x = Z[:, [0]]
        y = Z[:, [1]]
        eos = torch.distributions.Bernoulli(eos).sample()[:, :, 0]
        return x, y, eos

    def _generate_strokes_and_window_weights(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        # Split data tuple
        strokes, onehot, _, _, transcriptions = batch
        strokes_lengths = [1] * strokes.shape[0]
        onehot_lengths = [1] * onehot.shape[0]
        # Main Model Forward Step
        generated_points = []
        input_ = strokes[:, :1, :]
        finish = False
        counter = 0
        hidden = None
        while not finish and counter <= 1000:
            counter += 1
            (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi), hidden = self(
                input_,
                onehot,
                strokes_lengths,
                onehot_lengths,
                bias=self.probability_bias,
                hidden=hidden,
            )
            finish = phi[0, 0, -1].ge(torch.max(phi[0, 0, :])).item()
            x, y, eos = self.sample_point(eos, pi, mu1, mu2, sigma1, sigma2, rho)
            input_ = torch.stack((x, y, eos), dim=2)
            generated_points.append(input_)
        generated_strokes = torch.cat((strokes[:, 0:1], *generated_points), dim=1)
        return (
            generated_strokes.cpu().numpy(),
            phi.cpu().numpy(),
            strokes.cpu().numpy(),
            transcriptions,
        )

    def _step(self, batch, batch_idx):
        # Split data tuple
        strokes, onehot, strokes_lengths, onehot_lengths, _ = batch
        # Main Model Forward Step
        (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi), _ = self(
            strokes, onehot, strokes_lengths, onehot_lengths
        )
        # Compute loss
        loss = self.loss(eos, pi, mu1, mu2, sigma1, sigma2, rho, strokes)
        return loss, (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, _, (window, phi) = self._step(batch, batch_idx)
        self.log("val_loss", loss)
        strokes, _, strokes_lengths, _, transcriptions = batch
        idx = 0
        length = strokes_lengths[idx]
        fig = plot_window_weights(
            strokes[idx][:length].cpu().numpy(),
            phi[idx][:length].cpu().numpy(),
            transcriptions[idx],
        )
        self.logger.experiment.add_figure(
            f"val_window_weights_{batch_idx}_{idx}", fig, self.global_step
        )
        # Plot Strokes
        (
            generated_strokes,
            phi,
            strokes,
            transcriptions,
        ) = self._generate_strokes_and_window_weights(batch)
        # Plot Strokes
        fig, axes = plt.subplots(2, 1)
        strokes_array = strokes[idx]
        generated_strokes_array = generated_strokes[idx]
        transcription = transcriptions[idx]
        plot_strokes(strokes_array, ax=axes[0], transcription=transcription)
        plot_strokes(generated_strokes_array, ax=axes[1], transcription=transcription)
        fig.tight_layout()
        self.logger.experiment.add_figure(
            f"val_ground_truth_and_generated_strokes_{batch_idx}_{idx}",
            fig,
            self.global_step,
        )

    def test_step(self, batch, batch_idx):
        (
            generated_strokes,
            phi,
            strokes,
            transcriptions,
        ) = self._generate_strokes_and_window_weights(batch)
        # Plot Strokes
        idx = 0
        fig, axes = plt.subplots(2, 1)
        strokes_array = strokes[idx]
        generated_strokes_array = generated_strokes[idx]
        window_weights = phi[idx]
        transcription = transcriptions[idx]
        plot_strokes(strokes_array, ax=axes[0], transcription=transcription)
        plot_strokes(generated_strokes_array, ax=axes[1], transcription=transcription)
        fig.tight_layout()
        self.logger.experiment.add_figure(
            f"test_ground_truth_and_generated_strokes_{batch_idx}_{idx}",
            fig,
            self.global_step,
        )
        # Plot window weights
        fig = plot_window_weights(strokes_array, window_weights, transcription)
        self.logger.experiment.add_figure(
            f"test_window_weights_{batch_idx}_{idx}", fig, self.global_step
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            alpha=0.95,
            weight_decay=1e-4,
        )
        return {
            "optimizer": optimizer,
        }
