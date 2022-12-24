from functools import reduce

import numpy as np
import torch
from torch.nn import Module


class HandwritingLoss(Module):
    def __init__(self, parameters):
        super(HandwritingLoss, self).__init__()
        self.params = parameters

    def forward(self, eos, pi, mu1, mu2, sigma1, sigma2, rho, strokes):
        x_data, y_data, eos_data = strokes.chunk(3, dim=2)
        N = self.bivariate_gaussian(x_data, y_data, mu1, mu2, sigma1, sigma2, rho)
        epsilon = 1e-20
        term1 = -((pi * N).sum(dim=2, keepdim=True).clamp(min=epsilon)).log()
        term2 = (
            -(eos * eos_data + (1.0 - eos) * (1.0 - eos_data)).clamp(min=epsilon).log()
        )
        loss = term1 + term2
        reduction_factor = reduce((lambda x, y: x * y), loss.size())
        return loss.sum() / reduction_factor

    @staticmethod
    def bivariate_gaussian(x, y, mu1, mu2, sigma1, sigma2, rho):
        Z = (
            ((x - mu1) / sigma1) ** 2.0
            + ((y - mu2) / sigma2) ** 2.0
            - 2.0 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
        )
        N = torch.exp(-0.5 * Z / (1.0 - (rho**2.0))) / (
            2.0 * np.pi * sigma1 * sigma2 * torch.sqrt(1.0 - rho**2.0)
        )
        return N
