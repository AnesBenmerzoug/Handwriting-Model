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
        abk = torch.exp(abk_hats)
        alpha, beta, kappa = abk.chunk(3, dim=2)
        if prev_kappa is not None:
            kappa = kappa + prev_kappa
        u = torch.autograd.Variable(torch.arange(0, onehot.size(1)).view(-1, 1).expand(-1, kappa.size(2)))
        phi = torch.sum(alpha * torch.exp(-beta * ((kappa - u) ** 2)), dim=2).view(1, -1)
        window = torch.matmul(phi, onehot)
        return window, kappa

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_components={num_components})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MDN(Module):
    def __init__(self, input_size, num_mixtures):
        super(MDN, self).__init__()
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.parameter_layer = Linear(in_features=input_size, out_features=1 + 6*num_mixtures)

    def forward(self, input_):
        parameters_hats = self.parameter_layer(input_)
        eos_hat = parameters_hats[:, :, 0:1]
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = torch.chunk(parameters_hats[:, :, 1:], 6, dim=2)
        eos = F.sigmoid(-eos_hat)
        pi = F.softmax(pi_hat, dim=2)
        mu1 = mu1_hat
        mu2 = mu2_hat
        sigma1 = torch.exp(sigma1_hat)
        sigma2 = torch.exp(sigma2_hat)
        rho = F.tanh(rho_hat)
        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_mixtures={num_mixtures})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
