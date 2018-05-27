import torch
from torch.nn.modules import Module, Linear
import torch.nn.functional as F


class GaussianWindow(Module):
    def __init__(self, input_size, num_components):
        super(GaussianWindow, self).__init__()
        self.input_size = input_size
        self.num_components = num_components
        self.parameter_layer = Linear(in_features=input_size, out_features=3*num_components)

    def forward(self, input_, onehot, prev_kappa=None):
        abk_hats = self.parameter_layer(input_)
        abk = torch.exp(abk_hats).unsqueeze(3)
        alpha, beta, kappa = abk.chunk(3, dim=2)
        if prev_kappa is not None:
            kappa = kappa + prev_kappa
        else:
            kappa = kappa.cumsum(dim=2)
        u = torch.autograd.Variable(torch.arange(1, onehot.size(1) + 1))
        phi = torch.sum(alpha * torch.exp(-beta * ((kappa - u) ** 2)), dim=2)
        window = torch.matmul(phi, onehot)
        return window, kappa, phi

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_components={num_components})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MDN(Module):
    def __init__(self, input_size, num_mixtures):
        super(MDN, self).__init__()
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.parameter_layer = Linear(in_features=input_size, out_features=1 + 6*num_mixtures)

    def forward(self, input_, bias=None):
        mixture_parameters = self.parameter_layer(input_)
        eos_hat = mixture_parameters[:, :, 0:1]
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.chunk(mixture_parameters[:, :, 1:], 6, dim=2)
        eos = F.sigmoid(-eos_hat)
        mu1 = mu1_hat
        mu2 = mu2_hat
        rho = F.tanh(rho_hat)
        if bias is None:
            bias = torch.zeros_like(rho)
        pi = F.softmax(pi_hat * (1 + bias), dim=2)
        sigma1 = torch.exp(sigma1_hat - bias)
        sigma2 = torch.exp(sigma2_hat - bias)
        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_mixtures={num_mixtures})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
