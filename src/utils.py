from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def imgshow(img):
    img = make_grid(img).numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()


def plotlosses(losses, title='', xlabel='', ylabel=''):
    epochs = np.arange(losses.size, dtype=np.uint8) + 1
    plt.plot(epochs, losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plotaccuracy(accuracy, classes, title='', xlabel='', ylabel=''):
    indices = np.arange(len(classes), dtype=np.uint8)
    width = 0.35
    plt.bar(indices, accuracy, width)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(indices, classes)
    plt.show()

