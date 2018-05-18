from __future__ import print_function, division
import torch
from torch.nn import Module
import numpy as np
from functools import reduce


class HandwritingLoss(Module):
    def __init__(self, parameters):
        super(HandwritingLoss, self).__init__()
        self.params = parameters

    def forward(self, mdn_parameters, stroke):
        eos, pi, mu1, mu2, sigma1, sigma2, rho = mdn_parameters
        x_data, y_data, eos_data = stroke.chunk(3, dim=2)
        N = self.bivariateGaussian(x_data, y_data, mu1, mu2, sigma1, sigma2, rho)
        term1 = -((pi * N).sum(dim=2, keepdim=True) + 1e-20).log()
        term2 = -(0.7 * eos * eos_data + 0.3 * (1.0 - eos) * (1.0 - eos_data) + 1e-20).log()
        loss = term1 + term2
        reduction_factor = reduce((lambda x, y: x * y), loss.size())
        return loss.sum() / reduction_factor

    def bivariateGaussian(self, x, y, mu1, mu2, sigma1, sigma2, rho):
        Z = ((x - mu1) / sigma1) ** 2.0 \
            + ((y - mu2) / sigma2) ** 2.0 \
            - 2.0 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
        N = torch.exp(- 0.5 * Z / (1.0 - rho ** 2.0)) \
            / (2.0 * np.pi * sigma1 * sigma2 * torch.sqrt(1.0 - rho ** 2.0))
        return N
