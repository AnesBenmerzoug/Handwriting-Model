import torch
from torch.autograd.variable import Variable
from torch.nn.modules import Module, LSTM
from src.modules import GaussianWindow, MDN
import random


class HandwritingGenerator(Module):
    def __init__(self, alphabet_size, hidden_size, num_window_components, num_mixture_components):
        super(HandwritingGenerator, self).__init__()
        self.alphabet_size = alphabet_size
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3 + alphabet_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        # Gaussian Window layer
        self.window_layer = GaussianWindow(input_size=hidden_size,
                                           num_components=num_window_components)
        # Second LSTM layer, takes as input the concatenation of the input,
        # the output of the first LSTM layer
        # and the output of the Window layer
        self.lstm2_layer = LSTM(input_size=3 + hidden_size + alphabet_size,
                                hidden_size=hidden_size,
                                batch_first=True)

        # Third LSTM layer, takes as input the concatenation of the output of the first LSTM layer,
        # the output of the second LSTM layer
        # and the output of the Window layer
        self.lstm3_layer = LSTM(input_size=2 * hidden_size + alphabet_size,
                                hidden_size=hidden_size,
                                batch_first=True)

        # Mixture Density Network Layer
        self.output_layer = MDN(input_size=hidden_size,
                                num_mixtures=num_mixture_components)

        # Hidden State Variables
        self.prev_window = None
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

        # Initiliaze parameters
        self.reset_parameters()

    def forward(self, strokes, onehot, bias=None):
        if self.prev_window is None:
            self.prev_window = Variable(torch.zeros((strokes.size(0), strokes.size(1), self.alphabet_size)))
        # First LSTM Layer
        input_ = torch.cat((strokes, self.prev_window), dim=2)
        output1, self.hidden1 = self.lstm1_layer(input_, self.hidden1)
        # Gaussian Window Layer
        self.prev_window, self.prev_kappa = self.window_layer(output1, onehot, self.prev_kappa)
        # Second LSTM Layer
        output2, self.hidden2 = self.lstm2_layer(torch.cat((strokes, output1, self.prev_window), dim=2), self.hidden2)
        # Third LSTM Layer
        output3, self.hidden3 = self.lstm3_layer(torch.cat((output1, output2, self.prev_window), dim=2), self.hidden3)
        # MDN Layer
        eos, pi, mu1, mu2, sigma1, sigma2, rho = self.output_layer(output3, bias)
        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def sample_bivariate_gaussian(self, pi, mu1, mu2, sigma1, sigma2, rho):
        # Pick the distribution with the highest proportion from the MDN
        _, idx = torch.max(pi, dim=2)
        X = torch.normal(means=torch.cat((mu1[:, :, idx], mu2[:, :, idx]), dim=2),
                         std=torch.cat((sigma1[:, :, idx], sigma2[:, :, idx]), dim=2)).squeeze(3)
        return X[:, :, 0:1], X[:, :, 1:2]

    def reset_state(self):
        self.prev_window = None
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None
        self.hidden3 = None

    def reset_parameters(self):
        for parameter in self.parameters():
            if len(parameter.size()) == 2:
                torch.nn.init.xavier_uniform(parameter, gain=1.0)
            else:
                torch.nn.init.uniform(parameter, -1.0, 1.0)

    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num

    @classmethod
    def load_model(cls, package, useGPU=False):
        params = package['params']
        model = cls(alphabet_size=package['alphabet_size'],
                    hidden_size=params['hidden_size'],
                    num_window_components=params['num_window_components'],
                    num_mixture_components=params['num_mixture_components'])
        model.load_state_dict(package['state_dict'])
        if useGPU is True:
            model = model.cuda()
        return model