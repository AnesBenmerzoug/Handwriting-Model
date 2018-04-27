import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.nn.utils import clip_grad_norm
from src.dataset import IAMDataset
from copy import deepcopy
import time
import os


class Trainer(object):
    def __init__(self, parameters):

        self.params = parameters

        # Initialize datasets
        self.trainset = IAMDataset(self.params, setType='train')
        self.validationset = IAMDataset(self.params, setType='validate')

        # Initialize loaders
        self.trainloader = DataLoader(self.trainset, batch_size=self.params.batch_size,
                                      shuffle=False, num_workers=self.params.num_workers,
                                      sampler=RandomSampler(self.trainset))

        self.validationloader = DataLoader(self.validationset, batch_size=self.params.batch_size,
                                           shuffle=False, num_workers=self.params.num_workers)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()

