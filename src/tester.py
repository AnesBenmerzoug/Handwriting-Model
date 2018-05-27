import torch
from torch.autograd import Variable
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
        self.testloader = DataLoader(self.testset, batch_size=1,
                                     shuffle=True, num_workers=self.params.num_workers)

        # Initialize model
        self.load_model()

        # Criterion
        self.criterion = HandwritingLoss(self.params)

    def test_model(self):
        self.model.eval()
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
        self.model.eval()
        dataiter = iter(self.testloader)
        data = dataiter.next()
        # Split data tuple
        onehot, strokes = data
        # Wrap it in Variables
        onehot, strokes = Variable(onehot, volatile=True), Variable(strokes, volatile=True)
        # Main Model Forward Step
        self.model.reset_state()
        all_outputs = []
        phis = []
        windows = []
        input_ = strokes[:, 0:1]
        finish = False
        counter = 0
        while not finish:
            output, (window, phi) = self.model(input_, onehot, self.params.probability_bias)
            windows.append(window.data[0].numpy())
            phis.append(phi.data[0].numpy())
            finish = phi[0, 0, -1].data.ge(torch.max(phi.data[0, 0, :]))[0]
            eos, pi, mu1, mu2, sigma1, sigma2, rho = output
            x, y = self.model.sample_bivariate_gaussian(pi, mu1, mu2, sigma1, sigma2, rho)
            eos_data = eos.data
            threshold = eos_data.new([random.random()])
            mask = Variable(eos_data.ge(threshold).float(), volatile=True)
            eos = (eos * mask).ceil()
            input_ = torch.cat((x, y, eos), dim=2)
            all_outputs.append(input_)
            counter += 1
        phis = np.vstack(phis)
        windows = np.vstack(windows)
        generated_strokes = torch.cat((strokes[:, 0:1], *all_outputs), dim=1).data
        plotstrokes(strokes.data, generated_strokes)
        print(strokes.data)
        print(generated_strokes)
        plotwindow(phis, windows)

    def load_model(self, useGPU=False):
        package = torch.load(self.params.testModelPath, map_location=lambda storage, loc: storage)
        self.model = HandwritingGenerator.load_model(package, useGPU)
        #parameters = package['params']
        #self.params = namedtuple('Parameters', (parameters.keys()))(*parameters.values())
