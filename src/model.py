import torch
from torch.nn.modules import Module, LSTM, Linear
from src.modules import GaussianWindow, MDN


class HandwritingGenerator(Module):
    def __init__(self, alphabet_size, hidden_size, num_window_components, num_mixture_components):
        super(HandwritingGenerator, self).__init__()
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3,
                                hidden_size=hidden_size)
        # Gaussian Window layer
        self.window_layer = GaussianWindow(input_size=hidden_size,
                                           num_components=num_window_components)
        # Second LSTM layer, takes as input the concatenation of the input,
        # the output of the first LSTM layer
        # and the output of the Window layer
        self.lstm2_layer = LSTM(input_size=3 + hidden_size + alphabet_size,
                                hidden_size=hidden_size)
        # Mixture Density Network Layer
        self.output_layer = MDN(input_size=hidden_size,
                                num_mixtures=num_mixture_components)

        # Hidden State Variables
        self.prev_kappa = None
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, strokes, onehot):
        print(strokes.size())
        print(onehot.size())
        output, self.hidden1 = self.lstm1_layer(strokes, self.hidden1)
        print(output.size())
        window, self.prev_kappa = self.window_layer(output, onehot, self.prev_kappa)
        print(window.size())
        output, self.hidden2 = self.lstm2_layer(torch.cat((strokes, output, window), dim=2), self.hidden2)
        print(output.size())
        mdn_parameters = self.output_layer(output)  # eos, pi, mu1, mu2, sigma1, sigma2, rho
        print(mdn_parameters)
        raise KeyboardInterrupt
        return mdn_parameters

    def sample_bivariate_gaussian(self, pi, mu1, mu2, sigma1, sigma2, rho):
        pass

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
        params = package['parameters']
        model = cls()
        model.load_state_dict(package['state_dict'])
        if useGPU is True:
            model = model.cuda()
        return model