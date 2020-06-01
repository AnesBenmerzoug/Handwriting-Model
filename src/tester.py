from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from .dataset import IAMDataset
from .model import HandwritingGenerator
from .loss import HandwritingLoss
from .utils import plotstrokes, plotwindow
import numpy as np
import random


class Tester(object):
    def __init__(self, parameters):
        self.params = parameters

        # Initialize datasets
        self.testset = IAMDataset(self.params)

        # Initialize loaders
        self.testloader = DataLoader(
            self.testset,
            batch_size=1,
            shuffle=True,
            num_workers=self.params.num_workers,
        )

        # Initialize model
        path = self.params.model_dir / "trained_model.pt"
        self.model = self.load_model(path)

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def test_model(self):
        self.model.eval()
        losses = 0.0
        inf = float("inf")
        for data in self.testloader:
            # Split data tuple
            onehot, strokes = data
            # Main Model Forward Step
            self.model.reset_state()
            loss = None
            for idx in range(strokes.size(1) - 1):
                output = self.model(strokes[:, idx : idx + 1, :], onehot)
                # Loss Computation
                if loss is None:
                    loss = self.criterion(
                        output, strokes[:, idx : idx + 1, :]
                    ) / strokes.size(1)
                else:
                    loss = loss + self.criterion(
                        output, strokes[:, idx : idx + 1, :]
                    ) / strokes.size(1)
            if loss.data.item() == inf or loss.data.item() == -inf:
                print("Warning, received inf loss. Skipping it")
            elif loss.data.item() != loss.data.item():
                print("Warning, received NaN loss.")
            else:
                losses = losses + loss.data.item()
        return losses / len(self.testloader)

    def test_random_sample(self):
        self.model.eval()
        index = np.random.randint(0, len(self.testset))
        data = self.testset[index]
        # Split data tuple
        onehot, strokes = data
        onehot, strokes = onehot.unsqueeze(0), strokes.unsqueeze(0)
        print(self.testset.ascii[index])
        # Main Model Forward Step
        self.model.reset_state()
        all_outputs = []
        phis = []
        windows = []
        input_ = strokes[:, 0:1]
        finish = False
        counter = 0
        while not finish:
            output, (window, phi) = self.model(
                input_, onehot, self.params.probability_bias
            )
            windows.append(window.data[0].numpy())
            phis.append(phi.data[0].numpy())
            finish = phi[0, 0, -1].data.ge(torch.max(phi.data[0, 0, :])).item()
            eos, pi, mu1, mu2, sigma1, sigma2, rho = output
            x, y = self.model.sample_bivariate_gaussian(
                pi, mu1, mu2, sigma1, sigma2, rho
            )
            eos_data = eos.data
            threshold = eos_data.new([random.random()])
            mask = eos_data.ge(threshold).float()
            eos = (eos * mask).ceil()
            input_ = torch.cat((x, y, eos), dim=2)
            all_outputs.append(input_)
            counter += 1
        phis = np.vstack(phis)
        windows = np.vstack(windows)
        generated_strokes = torch.cat((strokes[:, 0:1], *all_outputs), dim=1).data
        plotstrokes(strokes.data, generated_strokes)
        plotwindow(phis, windows)

    @staticmethod
    def load_model(path: Union[Path, str]):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        parameters = package["parameters"]
        state_dict = package["state_dict"]
        return HandwritingGenerator.load_model(parameters, state_dict)
