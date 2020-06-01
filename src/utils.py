import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plotlosses(losses, title="", xlabel="", ylabel=""):
    epochs = np.arange(losses.size) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plotstrokes(strokes, other_strokes=None):
    if other_strokes is not None:
        plt.figure(figsize=(40, 2))
        plt.subplot(211)
    else:
        plt.figure(figsize=(20, 2))
    # Cumulative sum, because they are represented as relative displacement
    x = torch.cumsum(strokes[:, :, 0], dim=1).numpy()
    y = torch.cumsum(strokes[:, :, 1], dim=1).numpy()
    eos = strokes[:, :, 2]
    eos_indices = (eos.nonzero()[:, 1]).numpy()
    idx = 0
    while idx != eos_indices.shape[0]:
        start_index = eos_indices[idx] + 1
        try:
            end_index = eos_indices[idx + 1]
        except IndexError:
            end_index = x.shape[1]
        plt.plot(
            x[0, start_index:end_index],
            y[0, start_index:end_index],
            "b-",
            linewidth=2.0,
        )
        idx += 1
    plt.gca().invert_yaxis()
    if other_strokes is not None:
        plt.subplot(212)
        x = torch.cumsum(other_strokes[:, :, 0], dim=1).numpy()
        y = torch.cumsum(other_strokes[:, :, 1], dim=1).numpy()
        eos = other_strokes[:, :, 2]
        eos_indices = (eos.nonzero()[:, 1]).numpy()
        idx = 0
        while idx != eos_indices.shape[0]:
            start_index = eos_indices[idx] + 1
            try:
                end_index = eos_indices[idx + 1]
            except IndexError:
                end_index = x.shape[1]
            plt.plot(
                x[0, start_index:end_index],
                y[0, start_index:end_index],
                "b-",
                linewidth=2.0,
            )
            idx += 1
        plt.gca().invert_yaxis()
    plt.show()
    pass


def plotwindow(phis, windows):
    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plt.title("Phis", fontsize=20)
    plt.xlabel("Ascii #", fontsize=15)
    plt.ylabel("Time steps", fontsize=15)
    plt.imshow(phis, interpolation="nearest", aspect="auto", cmap=cm.jet)
    plt.subplot(122)
    plt.title("Soft attention window", fontsize=20)
    plt.xlabel("One-hot vector", fontsize=15)
    plt.ylabel("Time steps", fontsize=15)
    plt.imshow(windows, interpolation="nearest", aspect="auto", cmap=cm.jet)
    plt.show()
