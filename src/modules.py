import torch
from torch.nn.modules import Module, Linear


class GaussianWindow(Module):
    def __init__(self, input_size, num_components, window_size):
        super(GaussianWindow, self).__init__()
        self.parameter_layer = Linear(in_features=input_size, out_features=3*num_components)
        self.num_components = num_components
        self.window_size = window_size

    def forward(self, input, onehot):
        abk_hats = self.parameter_layer(input)
        abk = torch.exp(abk_hats)
        alpha, beta, kappa = torch.chunk(abk, dim=1)
        for i in range(kappa.size(0) - 1):
            kappa[i+1] = kappa[i+1] + kappa[i]
        u = torch.arange(0, self.window_size - 1)
        phi = torch.sum(alpha * torch.exp(-beta * torch.pow(kappa - u, 2)))
        window = phi * onehot
        return window


class MDN(Module):
    def __init__(self):
        super(MDN, self).__init__()

    def forward(self, input):
        pass
