import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import clip_grad_norm
from src.dataset import IAMDataset
from src.optimizer import SVRG
from src.scheduler import CosineAnnealingWarmRestartLR
from src.model import HandwritingGenerator
from collections import namedtuple
from copy import deepcopy
import time
import os


class Trainer(object):
    def __init__(self, parameters):

        self.params = parameters

        # Initialize datasets
        self.trainset = IAMDataset(self.params, setType='training')
        self.validationset = IAMDataset(self.params, setType='validation')

        self.alphabet = self.trainset.alphabet

        # Initialize loaders
        self.trainloader = DataLoader(self.trainset, batch_size=self.params.batch_size,
                                      shuffle=False, num_workers=self.params.num_workers,
                                      sampler=RandomSampler(self.trainset))

        self.validationloader = DataLoader(self.validationset, batch_size=self.params.batch_size,
                                           shuffle=False, num_workers=self.params.num_workers)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

        # Initialize model
        if self.params.resumeTraining is False:
            print("Training New Model")
            self.model = HandwritingGenerator(alphabet_size=len(self.alphabet),
                                              hidden_size=self.params.hidden_size,
                                              num_window_components=self.params.num_window_components,
                                              num_mixture_components=self.params.num_mixture_components)
        else:
            print("Resuming Training")
            self.load_model(self.useGPU)
        print(self.model)

        print("Number of parameters = {}".format(self.model.num_parameters()))

        if self.params.optimizer == 'SVRG':
            self.snapshot_model = deepcopy(self.model)

        if self.useGPU is True:
            print("Using GPU")
            try:
                self.model.cuda()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cuda()
            except RuntimeError:
                print("Failed to find GPU. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cpu()
            except UserWarning:
                print("GPU is too old. Using CPU instead")
                self.useGPU = False
                self.model.cpu()
                if self.params.optimizer == 'SVRG':
                    self.snapshot_model.cpu()
        else:
            print("Using CPU")


        # Setup optimizer
        self.optimizer = self.optimizer_select()

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestartLR(self.optimizer,
                                                      T_max=self.params.T_max,
                                                      eta_min=self.params.eta_min,
                                                      T_mult=self.params.T_mult)

    def train_model(self):
        min_Loss = None
        best_model = self.model.state_dict()
        avg_losses = np.zeros(self.params.num_epochs)
        for epoch in range(self.params.num_epochs):
            print("Epoch {}".format(epoch + 1))
            # Update learning rate
            self.scheduler.step()

            if self.params.optimizer == 'SVRG':
                # Update SVRG snapshot
                self.optimizer.update_snapshot(dataloader=self.trainloader, closure=self.snapshot_closure())

            print("Learning Rate= {}".format(self.optimizer.param_groups[0]['lr']))

            # Set mode to training
            self.model.train()

            # Go through the training set
            avg_losses[epoch] = self.train_epoch()

            print("Average loss= {:.3f}".format(avg_losses[epoch]))

            # Switch to eval and go through the test set
            self.model.eval()

            # Go through the test set
            test_loss = self.test_epoch()
            print("In Epoch {}, Obtained Loss {:.3f}".format(epoch + 1, test_loss))
            if min_Loss is None or min_Loss >= test_loss:
                min_Loss = test_loss
                best_model = self.model.state_dict()
        # Saving trained model
        self.save_model(best_model, min_Loss * 100)
        return avg_losses

    def train_epoch(self):
        losses = 0.0
        for batch_index, (data) in enumerate(self.trainloader, 1):
            if batch_index % 1 == 0:
                print("Step {}".format(batch_index))
            # Split data tuple
            onehot, strokes = data
            # Wrap it in Variables
            if self.useGPU is True:
                onehot, strokes = onehot.cuda(), strokes.cuda()
            onehot, strokes = Variable(onehot), Variable(strokes)
            # Main Model Forward Step
            all_output = []
            print(strokes.size())
            for idx in range(strokes.size(1)):
                output = self.model(strokes[:, idx:idx+1, :], onehot)
                all_output.append(output)
            # Loss Computation
            loss = self.criterion(output, strokes)
            print("loss = {:.3f}".format(loss.data[0]))
            inf = float("inf")
            if loss.data[0] == inf or loss.data[0] == -inf:
                print("Warning, received inf loss. Skipping it")
            elif loss.data[0] != loss.data[0]:
                print("Warning, received NaN loss.")
            else:
                losses = losses + loss.data[0]
            # Zero the optimizer gradient
            self.optimizer.zero_grad()
            # Backward step
            loss.backward()
            # Clip gradients
            clip_grad_norm(self.model.parameters(), self.params.max_norm)
            if self.params.optimizer == 'SVRG':
                # Snapshot Model Forward Backward
                snapshot_output = self.snapshot_model(onehot)
                snapshot_loss = self.criterion(snapshot_output, strokes)
                self.snapshot_model.zero_grad()
                snapshot_loss.backward()
                clip_grad_norm(self.snapshot_model.parameters(), self.params.max_norm)
            # Weight Update
            self.optimizer.step()
            if self.useGPU is True:
                torch.cuda.synchronize()
            del onehot, strokes, data, loss, output
        # Compute the average loss for this epoch
        avg_loss = losses / len(self.trainloader)
        if self.params.optimizer == 'SVRG':
            # Take a snapshot of the latest parameters
            self.optimizer.take_snapshot()
        return avg_loss

    def test_epoch(self):
        pass

    def optimizer_select(self):
        if self.params.optimizer == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'Adadelta':
            return optim.Adadelta(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimizer == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.params.learning_rate,
                             momentum=self.params.momentum, nesterov=self.params.nesterov)
        elif self.params.optimizer == 'SVRG':
            return SVRG(self.model.parameters(), self.snapshot_model.parameters(),
                        lr=self.params.learning_rate)
        else:
            raise NotImplementedError

    def save_model(self, model_parameters, model_accuracy):
        self.model.load_state_dict(model_parameters)
        torch.save(self.serialize(),
                   os.path.join(self.params.savedModelDir, 'Trained_Model_{}'.format(int(model_accuracy))
                                + '_' + time.strftime("%d.%m.20%y_%H.%M")))

    def load_model(self, useGPU=False):
        package = torch.load(self.params.trainedModelPath, map_location=lambda storage, loc: storage)
        self.model = HandwritingGenerator.load_model(package, useGPU)
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())
        self.optimizer = self.optimizer_select()

    def serialize(self):
        model_is_cuda = next(self.model.parameters()).is_cuda
        model = self.model.cpu() if model_is_cuda else self.model
        package = {
            'state_dict': model.state_dict(),
            'params': self.params._asdict(),
            'optim_dict': self.optimizer.state_dict()
        }
        return package
