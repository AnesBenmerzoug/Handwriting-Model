import faulthandler
import logging
import time

import click

from handwriting_generator import Tester, Trainer, plotlosses
from handwriting_generator.config import Parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--n-epochs", type=int, default=100)
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
def main(n_epochs: int, train: bool, gpu: bool):
    # To have a more verbose output in case of an exception
    faulthandler.enable()

    parameters = Parameters(n_epochs=n_epochs, train_model=train, use_gpu=gpu)

    if parameters.train_model is True:
        # Instantiating the trainer
        trainer = Trainer(parameters)
        # Training the model
        avg_losses = trainer.train_model()
        # Plot losses
        plotlosses(
            avg_losses,
            title="Average Loss per Epoch",
            xlabel="Epoch",
            ylabel="Average Loss",
        )

    # Instantiating the tester
    tester = Tester(parameters)
    # Testing the model
    tester.test_random_sample()
    # Testing the model accuracy
    # test_losses = tester.test_model()


if __name__ == "__main__":
    main()
