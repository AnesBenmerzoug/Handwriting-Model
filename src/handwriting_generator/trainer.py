import logging
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from handwriting_generator.config import Parameters
from handwriting_generator.constants import OUTPUT_DIR
from handwriting_generator.dataset import IAMDataset
from handwriting_generator.loss import HandwritingLoss
from handwriting_generator.model import HandwritingGenerator
from handwriting_generator.utils import collate_fn

__all__ = ["Trainer"]

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, parameters: Parameters):

        self.params = parameters

        # Initialize datasets
        self.trainset = IAMDataset()

        self.alphabet = self.trainset.alphabet
        alphabet_size = len(self.alphabet)

        # Initialize loaders
        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Initialize model
        self.model = HandwritingGenerator(
            alphabet_size=alphabet_size,
            hidden_size=self.params.hidden_size,
            n_window_components=self.params.n_window_components,
            n_mixture_components=self.params.n_mixture_components,
        )
        self.model.to(self.device)

        logger.info(self.model)

        logger.info("Number of parameters = {}".format(self.model.num_parameters()))

        # Optimizer setup
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def train_model(self):
        min_loss = None
        best_model = self.model.state_dict()
        avg_losses = np.zeros(self.params.n_epochs)
        path = OUTPUT_DIR / "trained_model.pt"

        with logging_redirect_tqdm():
            for epoch in trange(self.params.n_epochs):
                try:
                    logger.info("Epoch {}".format(epoch + 1))

                    # Set mode to training
                    self.model.train()

                    # Go through the training set
                    avg_losses[epoch] = self.train_epoch()

                    logger.info("Average loss = {:.3f}".format(avg_losses[epoch]))

                    if min_loss is None or min_loss >= avg_losses[epoch]:
                        min_loss = avg_losses[epoch]
                        best_model = self.model.state_dict()

                    if (epoch + 1) % 5 == 0:
                        self.save_model(best_model, path)

                except KeyboardInterrupt:
                    logger.info("Training was interrupted")
                    break
        # Saving trained model
        self.save_model(best_model, path)
        return avg_losses

    def train_epoch(self):
        losses = 0.0
        inf = float("inf")
        for batch_index, (data) in tqdm(
            enumerate(self.train_loader, 1), total=len(self.train_loader), leave=False
        ):
            if batch_index % 20 == 0:
                logger.info(f"Step {batch_index}")
                logger.info(f"Average Loss so far: {losses / batch_index}")
            # Split data tuple
            onehot, strokes, onehot_lengths, strokes_lens = data
            # Move inputs to correct device
            onehot, strokes = onehot.to(self.device), strokes.to(self.device)
            # Main Model Forward Step
            self.model.reset_state()
            loss = None
            for idx in range(strokes.size(1) - 1):
                output, _ = self.model(strokes[:, idx : idx + 1, :], onehot)
                # Loss Computation
                loss = (
                    self.criterion(output, strokes[:, idx + 1 : idx + 2, :])
                    / strokes.size(1)
                    if loss is None
                    else loss
                    + self.criterion(output, strokes[:, idx + 1 : idx + 2, :])
                    / strokes.size(1)
                )
            if loss.data.item() == inf or loss.data.item() == -inf:
                logger.info("Warning, received inf loss. Skipping it")
            elif loss.data.item() != loss.data.item():
                logger.info("Warning, received NaN loss.")
            else:
                losses = losses + loss.data.item()
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            loss.backward()
            # Clip gradients
            clip_grad_norm_(self.model.parameters(), self.params.max_norm)
            # Weight Update
            self.optimizer.step()
            if self.use_gpu is True:
                torch.cuda.synchronize()
            del onehot, strokes, data
        # Compute the average loss for this epoch
        avg_loss = losses / len(self.train_loader)
        return avg_loss

    def save_model(self, model_parameters, path):
        model = deepcopy(self.model)
        model.load_state_dict(model_parameters)
        torch.save(
            self.serialize(model),
            path,
        )

    def serialize(self, model):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.cpu() if model_is_cuda else self.model
        package = {
            "state_dict": model.state_dict(),
            "optim_dict": self.optimizer.state_dict(),
            "parameters": {
                "alphabet_size": self.model.alphabet_size,
                "hidden_size": self.model.hidden_size,
                "n_window_components": self.model.n_window_components,
                "n_mixture_components": self.model.n_mixture_components,
            },
        }
        return package
