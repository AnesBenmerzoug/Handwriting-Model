from __future__ import print_function, division
import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
from torch.nn.utils import clip_grad_norm


class SVRG(Optimizer):
    r"""Implementation of the optimizer described in :
    https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf
    """
    def __init__(self, params, snapshot_params, lr=required):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)
        # State Initialization
        for group in self.param_groups:
            group['snapshot_params'] = list(snapshot_params)
            group['average_gradient'] = list()
            for p in group['params']:
                group['average_gradient'].append(torch.zeros_like(p.data))

    def step(self, closure=None):
        r"""Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss, snapshot_loss = None, None
        if closure is not None:
            self.zero_grad()
            loss, snapshot_loss = closure()

        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                snapshot_params = group['snapshot_params'][idx]
                average_gradient = group['average_gradient'][idx]
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("SVRG doesn't support sparse gradients")
                # gradient data
                d_p = p.grad.data
                # subtract the average gradient
                d_p.add_(-1, average_gradient)
                # add the snapshot gradient
                if snapshot_params.grad is not None:
                    d_p.add_(snapshot_params.grad.data)

                p.data.add_(-group['lr'], d_p)

        return loss

    def update_snapshot(self, dataloader, closure):
        r"""Updates the parameter snapshot and the average gradient
        Arguments:
            dataloader : A dataloader used to get the training samples.
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError("A closure has to be given")
        if dataloader is None:
            raise RuntimeError("Dataloader has to be given")
        # Zero the gradient of the snapshot parameters
        self.zero_grad()
        # Iterate over all the dataset to compute the average gradient
        for i, (data, target) in enumerate(dataloader):
            closure(data, target)
            for group in self.param_groups:
                for idx, p in enumerate(group['snapshot_params']):
                    if p.grad is None:
                        continue
                    if i == 0:
                        group['average_gradient'][idx].zero_()
                    group['average_gradient'][idx].add_(1/len(dataloader), p.grad.data)

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
                if sp.grad is not None:
                    sp.grad.detach_()
                    sp.grad.zero_()

    def take_snapshot(self):
        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                sp.data.copy_(p.data)
