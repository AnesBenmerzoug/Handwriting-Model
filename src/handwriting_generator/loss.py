from functools import reduce

import numpy as np
import torch
from torch.nn import Module


class HandwritingLoss(Module):
    def forward(
        self, eos, pi, mu1, mu2, sigma1, sigma2, rho, strokes, *, epsilon: float = 1e-20
    ):
        eos = eos[:, :-1]
        pi = pi[:, :-1]
        mu1 = mu1[:, :-1]
        mu2 = mu2[:, :-1]
        sigma1 = sigma1[:, :-1]
        sigma2 = sigma2[:, :-1]
        rho = rho[:, :-1]
        strokes = torch.roll(strokes, -1, dims=1)[:, :-1]
        x_gt, y_gt, eos_gt = strokes.chunk(3, dim=2)
        N = self.bivariate_gaussian(x_gt, y_gt, mu1, mu2, sigma1, sigma2, rho)
        # clamp is used to make we don't end up with negative input values for log
        term1 = -((pi * N).sum(dim=2, keepdim=True).clamp(min=epsilon)).log()
        term2 = -(eos * eos_gt + (1.0 - eos) * (1.0 - eos_gt)).clamp(min=epsilon).log()
        loss = term1 + term2
        return loss.mean()

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
