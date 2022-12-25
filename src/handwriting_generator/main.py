import faulthandler
import logging

import click
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from rich.logging import RichHandler

from handwriting_generator.constants import OUTPUT_DIR
from handwriting_generator.dataset import IAMDataModule
from handwriting_generator.model import HandwritingGenerator
from handwriting_generator.utils import find_best_model_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[click])],
    force=True,
)
logger = logging.getLogger(__name__)

pl.seed_everything(16)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


@click.command()
@click.option("--n-epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=64)
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--gpu/--no-gpu", is_flag=True, default=True)
def main(n_epochs: int, batch_size: int, train: bool, gpu: bool):
    # To have a more verbose output in case of an exception
    faulthandler.enable()

    datamodule = IAMDataModule(batch_size=batch_size, num_workers=4)
    datamodule.prepare_data()

    if train:
        model = HandwritingGenerator(alphabet_size=len(datamodule.alphabet))
    else:
        logs_dir = OUTPUT_DIR / "lightning_logs"
        checkpoint_path = find_best_model_checkpoint(logs_dir)
        logger.info(f"Loading checkpoint from: '{checkpoint_path}'")
        model = HandwritingGenerator.load_from_checkpoint(
            alphabet_size=len(datamodule.alphabet),
            checkpoint_path=checkpoint_path,
            map_location=lambda storage, loc: storage,
        )

    trainer = pl.Trainer(
        default_root_dir=OUTPUT_DIR,
        max_epochs=n_epochs,
        auto_lr_find=True,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=0.5,
        callbacks=[
            LearningRateMonitor("epoch"),
            ModelCheckpoint(
                save_last=True,
                save_top_k=2,
                monitor="val_loss",
                filename="{epoch}-{val_loss:.5f}",
            ),
            RichModelSummary(max_depth=2),
            RichProgressBar(refresh_rate=10),
        ],
        logger=pl_loggers.TensorBoardLogger(save_dir=OUTPUT_DIR),
        profiler="simple",
    )

    if train:
        trainer.tune(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
