from __future__ import print_function, division
import torch
from torch.nn import Module
import numpy as np


class HandwritingLoss(Module):
    def __init__(self, parameters):
        super(HandwritingLoss, self).__init__()
        self.params = parameters

    def forward(self, mdn_parameters, stroke):
        eos, pi, mu1, mu2, sigma1, sigma2, rho = mdn_parameters
        x_data = stroke[0, 0, 0]
        y_data = stroke[0, 0, 1]
        eos_data = stroke[0, 0, 2]
        N = self.bivariateGaussian(x_data, y_data, mu1, mu2, sigma1, sigma2, rho)
        Pr = pi * N
        Pr = torch.clamp(Pr, 1e-20, torch.max(Pr).data[0])
        loss = - torch.log(Pr) \
               - torch.log(eos * eos_data + (1 - eos) * (1 - eos_data))
        return torch.sum(loss)

    def bivariateGaussian(self, x, y, mu1, mu2, sigma1, sigma2, rho):
        x_mu1 = x - mu1
        y_mu2 = y - mu2
        Z = torch.pow(x_mu1, 2)/sigma1 \
            + torch.pow(y_mu2, 2)/sigma2 \
            - 2 * rho * x_mu1 * y_mu2 / (sigma1 * sigma2)
        N = torch.exp(-Z / (2*(1 - torch.pow(rho, 2)))) \
            / (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - torch.pow(rho, 2)))
        return N
