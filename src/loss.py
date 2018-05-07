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
        Pr = torch.sum(pi * N)
        loss = - torch.log(Pr + 1e-20) \
               - torch.log(eos * eos_data + (1.0 - eos) * (1.0 - eos_data))
        return loss.view(1)

    def bivariateGaussian(self, x, y, mu1, mu2, sigma1, sigma2, rho):
        Z = ((x - mu1) / sigma1) ** 2.0 \
            + ((y - mu2) / sigma2) ** 2.0 \
            - 2.0 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
        N = torch.exp(- 0.5 * Z / (1.0 - rho ** 2.0)) \
            / (2.0 * np.pi * sigma1 * sigma2 * torch.sqrt(1.0 - rho ** 2.0))
        return N
