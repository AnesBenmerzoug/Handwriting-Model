import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import clip_grad_norm
from .dataset import IAMDataset
from .model import HandwritingGenerator
from .loss import HandwritingLoss
from copy import deepcopy
from .utils import plotstrokes


class Trainer:
    def __init__(self, parameters):

        self.params = parameters

        # Initialize datasets
        self.trainset = IAMDataset(self.params)

        self.alphabet = self.trainset.alphabet
        alphabet_size = len(self.alphabet)

        # Initialize loaders
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            sampler=RandomSampler(self.trainset),
        )

        # Checking for GPU
        self.use_gpu = self.params.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")

        # Initialize model
        self.model = HandwritingGenerator(
            alphabet_size=alphabet_size,
            hidden_size=self.params.hidden_size,
            num_window_components=self.params.num_window_components,
            num_mixture_components=self.params.num_mixture_components,
        )
        self.model.to(self.device)

        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        # Optimizer setup
        self.optimizer = self.optimizer_select()

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def train_model(self):
        min_loss = None
        best_model = self.model.state_dict()
        avg_losses = np.zeros(self.params.num_epochs)
        self.params.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.params.model_dir / "trained_model.pt"
        for epoch in range(self.params.num_epochs):
            try:
                print("Epoch {}".format(epoch + 1))

                # Set mode to training
                self.model.train()

                # Go through the training set
                avg_losses[epoch] = self.train_epoch()

                print("Average loss = {:.3f}".format(avg_losses[epoch]))

                if min_loss is None or min_loss >= avg_losses[epoch]:
                    min_loss = avg_losses[epoch]
                    best_model = self.model.state_dict()

                if (epoch + 1) % 5 == 0:
                    self.save_model(best_model, path)

            except KeyboardInterrupt:
                print("Training was interrupted")
                break
        # Saving trained model
        self.save_model(best_model, path)
        return avg_losses

    def train_epoch(self):
        losses = 0.0
        inf = float("inf")
        for batch_index, (data) in enumerate(self.trainloader, 1):
            if batch_index % 20 == 0:
                print("Step {}".format(batch_index))
                print("Average Loss so far: {}".format(losses / batch_index))
            # Split data tuple
            onehot, strokes = data
            # Plot strokes
            # plotstrokes(strokes)
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
                print("Warning, received inf loss. Skipping it")
            elif loss.data.item() != loss.data.item():
                print("Warning, received NaN loss.")
            else:
                losses = losses + loss.data.item()
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            loss.backward()
            # Clip gradients
            clip_grad_norm(self.model.parameters(), self.params.max_norm)
            # Weight Update
            self.optimizer.step()
            if self.use_gpu is True:
                torch.cuda.synchronize()
            del onehot, strokes, data
        # Compute the average loss for this epoch
        avg_loss = losses / len(self.trainloader)
        return avg_loss

    def save_model(self, model_parameters, path):
        model = deepcopy(self.model)
        model.load_state_dict(model_parameters)
        torch.save(
            self.serialize(model), path,
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
                "num_window_components": self.model.num_window_components,
                "num_mixture_components": self.model.num_mixture_components,
            },
        }
        return package

    def optimizer_select(self):
        if self.params.optimizer == "Adam":
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == "Adadelta":
            return optim.Adadelta(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == "SGD":
            return optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                nesterov=self.params.nesterov,
            )
        elif self.params.optimizer == "RMSprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
            )
        else:
            raise NotImplementedError
