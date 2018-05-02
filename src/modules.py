import torch
from torch.nn.modules import Module, Linear


class GaussianWindow(Module):
    def __init__(self, input_size, num_components, window_size):
        super(GaussianWindow, self).__init__()
        self.parameter_layer = Linear(in_features=input_size, out_features=3*num_components)
        self.num_components = num_components
        self.window_size = window_size

    def forward(self, input_, onehot, prev_kappa=None):
        abk_hats = self.parameter_layer(input_)
        abk = torch.exp(abk_hats)
        alpha, beta, kappa = torch.chunk(abk, 3, dim=2)
        if prev_kappa is None:
            prev_kappa = kappa
        else:
            kappa, prev_kappa = kappa + prev_kappa, kappa
        u = torch.autograd.Variable(torch.arange(0, onehot.size(1)).view(-1, 1).expand(-1, kappa.size(2)))
        phi = torch.sum(alpha * torch.exp(-beta * torch.pow(kappa - u, 2)), dim=2).view(1, -1, 1)
        window = phi * onehot.float()
        return window, prev_kappa


class MDN(Module):
    def __init__(self):
        super(MDN, self).__init__()

    def forward(self, input_):
        pass
