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
    x = strokes[:, :, 0].numpy()
    y = strokes[:, :, 1].numpy()
    # Cumulative sum, because they are represented as relative displacement
    for idx in range(1, x.shape[1]):
        x[:, idx] += x[:, idx-1]
        y[:, idx] += y[:, idx-1]
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

