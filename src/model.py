import torch
from torch.nn.modules import Module, LSTM, Linear
from src.modules import GaussianWindow, MDN


class HandwritingGenerator(Module):
    def __init__(self, alphabet_size, num_window_components, hidden_size):
        super(HandwritingGenerator, self).__init__()
        # First LSTM layer, takes as input a tuple (x, y, eol)
        self.lstm1_layer = LSTM(input_size=3,
                                hidden_size=hidden_size)
        # Gaussian Window layer
        self.window_layer = GaussianWindow(input_size=hidden_size,
                                           num_components=num_window_components,
                                           window_size=alphabet_size)
        # Second LSTM layer, takes as input the concatenation of the input,
        # the output of the first LSTM layer
        # and the output of the Window layer
        self.lstm2_layer = LSTM(input_size=3 + hidden_size + alphabet_size,
                                hidden_size=hidden_size)
        # Mixture Density Network Layer
        self.output_layer = MDN()

    def forward(self, onehot):
        pass

    def reset_parameters(self):
        for parameter in self.parameters():
            if parameter.size() == 2:
                torch.nn.init.xavier_uniform(parameter, gain=1.0)
            else:
                torch.nn.init.uniform(parameter, -1.0, 1.0)

    def num_parameters(self):
        num = 0
        for weight in self.parameters():
            num = num + weight.numel()
        return num
