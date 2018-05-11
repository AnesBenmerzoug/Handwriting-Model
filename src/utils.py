import torch
import matplotlib.pyplot as plt
import numpy as np


def plotlosses(losses, title='', xlabel='', ylabel=''):
    epochs = np.arange(losses.size) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plotstrokes(strokes):
    # Cumulative sum, because they are represented as relative displacement
    x = torch.cumsum(strokes[:, :, 0], dim=1).numpy()
    y = torch.cumsum(strokes[:, :, 1], dim=1).numpy()
    eos = strokes[:, :, 2]
    eos_indices = (eos.nonzero()[:, 1]).numpy()
    plt.figure(figsize=(20, 2))
    idx = 0
    while idx != eos_indices.shape[0]:
        start_index = eos_indices[idx]+1
        try:
            end_index = eos_indices[idx+1]
        except IndexError:
            end_index = x.shape[1]
        plt.plot(x[0, start_index:end_index], y[0, start_index:end_index], 'b-', linewidth=2.0)
        idx += 1
    plt.gca().invert_yaxis()
    plt.show()
    pass

