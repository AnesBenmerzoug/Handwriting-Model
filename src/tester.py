import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.dataset import IAMDataset
from src.model import HandwritingGenerator
from src.loss import HandwritingLoss
from src.utils import plotstrokes
from collections import namedtuple
import random


class Tester(object):
    def __init__(self, parameters):
        self.params = parameters

        # Initialize datasets
        self.testset = IAMDataset(self.params, setType='testing')

        # Initialize loaders
        self.testloader = DataLoader(self.testset, batch_size=self.params.batch_size,
                                     shuffle=False, num_workers=self.params.num_workers)

        # Initialize model
        self.load_model()

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def test_model(self):
        losses = 0.0
        for data in self.testloader:
            # Split data tuple
            onehot, strokes = data
            # Wrap it in Variables
            onehot, strokes = Variable(onehot, volatile=True), Variable(strokes, volatile=True)
            # Main Model Forward Step
            self.model.reset_state()
            loss = None
            for idx in range(strokes.size(1) - 1):
                output = self.model(strokes[:, idx:idx + 1, :], onehot)
                # Loss Computation
                if loss is None:
                    loss = self.criterion(output, strokes[:, idx:idx + 1, :]) / strokes.size(1)
                else:
                    loss = loss + self.criterion(output, strokes[:, idx:idx + 1, :]) / strokes.size(1)
            inf = float("inf")
            if loss.data[0] == inf or loss.data[0] == -inf:
                print("Warning, received inf loss. Skipping it")
            elif loss.data[0] != loss.data[0]:
                print("Warning, received NaN loss.")
            else:
                losses = losses + loss.data[0]
        return losses / len(self.testloader)

    def test_random_sample(self):
        data = iter(self.testloader).next()
        # Split data tuple
        onehot, strokes = data
        # Add an initial (0.0, 0.0, 1.0) point to strokes
        strokes = torch.cat((torch.FloatTensor([[[0.0, 0.0, 1.0]]]), strokes), dim=1)
        # Wrap it in Variables
        onehot, strokes = Variable(onehot, volatile=True), Variable(strokes, volatile=True)
        # Main Model Forward Step
        self.model.reset_state()
        all_outputs = []
        input_ = strokes[:, 0:1]
        for idx in range(strokes.size(1) - 1):
            output = self.model(input_, onehot)
            eos, pi, mu1, mu2, sigma1, sigma2, rho = output
            x, y = self.model.sample_bivariate_gaussian(pi, mu1, mu2, sigma1, sigma2, rho)
            eos = eos.round()
            input_ = torch.cat((x, y, eos), dim=2)
            all_outputs.append(input_)
        generated_strokes = torch.cat(all_outputs, dim=1).data
        # Add an initial (0.0, 0.0, 1.0) point to strokes
        generated_strokes = torch.cat((torch.FloatTensor([[[0.0, 0.0, 1.0]]]), generated_strokes), dim=1)
        print(strokes)
        print(generated_strokes)
        plotstrokes(generated_strokes)

    def load_model(self, useGPU=False):
        package = torch.load(self.params.testModelPath, map_location=lambda storage, loc: storage)
        self.model = HandwritingGenerator.load_model(package, useGPU)
        parameters = package['params']
        self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())