import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.dataset import IAMDataset
import random


class Tester(object):
    def __init__(self, parameters):
        self.params = parameters

        # Initialize datasets
        self.testset = IAMDataset(self.params, setType='test')

        # Initialize loaders

        self.testloader = DataLoader(self.testset, batch_size=self.params.batch_size,
                                     shuffle=False, num_workers=self.params.num_workers)

        # Checking for GPU
        self.useGPU = self.params.useGPU and torch.cuda.is_available()
