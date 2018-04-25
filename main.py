from __future__ import print_function
from collections import namedtuple
import faulthandler
import time
import yaml

from src import *


if __name__ == "__main__":
    print("Starting time: {}".format(time.asctime()))

    # To have a more verbose output in case of an exception
    faulthandler.enable()

    with open('parameters.yaml', 'r') as params_file:
        parameters = yaml.safe_load(params_file)
        parameters = namedtuple('Parameters', (parameters.keys()))(*parameters.values())

    if parameters.trainModel is True:
        # Instantiating the trainer
        trainer = Trainer(parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(avg_losses, title='Average Loss per Epoch', xlabel='Epoch', ylabel='Average Loss')

    else:
        # Instantiating the tester
        tester = Tester(parameters)
        # Testing the model
        tester.test_random_sample()
        # Testing the model accuracy
        #total_accuracy, class_accuracy = tester.test_model()

    print("Finishing time: {}".format(time.asctime()))


