import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from handwriting_generator.config import Parameters
from handwriting_generator.constants import OUTPUT_DIR
from handwriting_generator.dataset import IAMDataset
from handwriting_generator.loss import HandwritingLoss
from handwriting_generator.model import HandwritingGenerator
from handwriting_generator.utils import collate_fn, plot_strokes, plotwindow

__all__ = ["Tester"]

logger = logging.getLogger(__name__)


class Tester:
    def __init__(self, parameters: Parameters):
        self.params = parameters

        # Initialize datasets
        self.testset = IAMDataset()

        # Initialize loaders
        self.test_loader = DataLoader(
            self.testset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Initialize model
        path = OUTPUT_DIR / "trained_model.pt"
        self.model = self.load_model(path)

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def test_model(self):
        self.model.eval()
        losses = 0.0
        inf = float("inf")
        with logging_redirect_tqdm():
            for data in tqdm(self.test_loader):
                # Split data tuple
                onehot, strokes, onehot_lengths, strokes_lens = data
                # Main Model Forward Step
                with torch.no_grad():
                    (eos, pi, mu1, mu2, sigma1, sigma2, rho), _ = self.model(
                        strokes, onehot
                    )

                    loss = self.criterion(
                        eos[:, :-1],
                        pi[:, :-1],
                        mu1[:, :-1],
                        mu2[:, :-1],
                        sigma1[:, :-1],
                        sigma2[:, :-1],
                        rho[:, :-1],
                        torch.roll(strokes, -1, dims=1)[:, :-1],
                    ) / strokes.size(1)

                if loss.data.item() == inf or loss.data.item() == -inf:
                    logger.info("Warning, received inf loss. Skipping it")
                elif loss.data.item() != loss.data.item():
                    logger.info("Warning, received NaN loss.")
                else:
                    losses = losses + loss.data.item()
        return losses / len(self.test_loader)

    def test_random_sample(self):
        self.model.eval()
        index = np.random.randint(0, len(self.testset))
        data = self.testset[index]
        # Split data tuple
        onehot, strokes = data
        onehot, strokes = onehot.unsqueeze(0).float(), strokes.unsqueeze(0).float()
        logger.info(self.testset.transcriptions[index])
        # Main Model Forward Step
        all_outputs = []
        phis = []
        windows = []
        input_ = strokes[:, 0:1]
        finish = False
        counter = 0
        while not finish and counter <= 1000:
            counter += 1
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
        phis = np.vstack(phis)
        windows = np.vstack(windows)
        generated_strokes = torch.cat((strokes[:, 0:1], *all_outputs), dim=1).data
        # Plot Strokes
        fig, axes = plt.subplots(2, 1)
        plot_strokes(strokes.numpy(), ax=axes[0])
        plot_strokes(generated_strokes.numpy(), ax=axes[1])
        fig.show()
        # Plot Gaussian Window
        plotwindow(phis, windows)

    @staticmethod
    def load_model(path: Path | str) -> HandwritingGenerator:
        package = torch.load(path, map_location=lambda storage, loc: storage)
        parameters = package["parameters"]
        state_dict = package["state_dict"]
        return HandwritingGenerator.load_model(parameters, state_dict)
