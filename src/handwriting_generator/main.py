import faulthandler
import logging

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    TQDMProgressBar,
)

from handwriting_generator.config import Parameters
from handwriting_generator.constants import OUTPUT_DIR
from handwriting_generator.dataset import IAMDataModule
from handwriting_generator.model import HandwritingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

pl.seed_everything(16)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


@click.command()
@click.option("--n-epochs", type=int, default=100)
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
def main(n_epochs: int, train: bool, gpu: bool):
    # To have a more verbose output in case of an exception
    faulthandler.enable()

    parameters = Parameters(n_epochs=n_epochs, train_model=train, use_gpu=gpu)

    datamodule = IAMDataModule(batch_size=parameters.batch_size, num_workers=2)
    datamodule.prepare_data()

    model = HandwritingGenerator(
        alphabet_size=len(datamodule.alphabet),
        hidden_size=parameters.hidden_size,
        n_window_components=parameters.n_window_components,
        n_mixture_components=parameters.n_mixture_components,
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir=OUTPUT_DIR)

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=0.5,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            LearningRateMonitor("epoch"),
            DeviceStatsMonitor(),
        ],
        logger=tensorboard,
        profiler="simple",
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
