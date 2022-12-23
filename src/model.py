import torch
from torch.nn.modules import Module, LSTM
from src.modules import GaussianWindow, MDN
import numpy as np


class HandwritingGenerator(Module):
    def __init__(
        self, alphabet_size, hidden_size, num_window_components, num_mixture_components
    ):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.num_window_components = num_window_components
        self.num_mixture_components = num_mixture_components
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        # Gaussian Window layer
        self.window_layer = GaussianWindow(
            input_size=hidden_size, num_components=num_window_components
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
        self.output_layer = MDN(
            input_size=hidden_size, num_mixtures=num_mixture_components
        )

        # Hidden State Variables
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        # Initiliaze parameters
        self.reset_parameters()

    def forward(self, strokes, onehot, bias=None):
        # First LSTM Layer
        input_ = strokes
        self.lstm1_layer.flatten_parameters()
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # Gaussian Window Layer
        window, self.prev_kappa, phi = self.window_layer(
            output1, onehot, self.prev_kappa
        )
        # Second LSTM Layer
        output2, self.hidden2 = self.lstm2_layer(
            torch.cat((strokes, output1, window), dim=2), self.hidden2
        )
        # Third LSTM Layer
        output3, self.hidden3 = self.lstm3_layer(output2, self.hidden3)
        # MDN Layer
        eos, pi, mu1, mu2, sigma1, sigma2, rho = self.output_layer(output3, bias)
        return (eos, pi, mu1, mu2, sigma1, sigma2, rho), (window, phi)

    @staticmethod
    def sample_bivariate_gaussian(pi, mu1, mu2, sigma1, sigma2, rho):
        # Pick distribution from the MDN
        p = pi.data[0, 0, :].numpy()
        idx = np.random.choice(p.shape[0], p=p)
        m1 = mu1.data[0, 0, idx]
        m2 = mu2.data[0, 0, idx]
        s1 = sigma1.data[0, 0, idx]
        s2 = sigma2.data[0, 0, idx]
        r = rho.data[0, 0, idx]
        mean = [m1, m2]
        covariance = [[s1 ** 2, r * s1 * s2], [r * s1 * s2, s2 ** 2]]
        Z = torch.autograd.Variable(
            sigma1.data.new(np.random.multivariate_normal(mean, covariance, 1))
        ).unsqueeze(0)
        X = Z[:, :, 0:1]
        Y = Z[:, :, 1:2]
        return X, Y

    def reset_state(self):
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

    def reset_parameters(self):
        for parameter in self.parameters():
            if len(parameter.size()) == 2:
                torch.nn.init.xavier_uniform_(parameter, gain=1.0)
            else:
                stdv = 1.0 / parameter.size(0)
                torch.nn.init.uniform_(parameter, -stdv, stdv)

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
            self.num_window_components,
            self.num_mixture_components,
        )
        return model
