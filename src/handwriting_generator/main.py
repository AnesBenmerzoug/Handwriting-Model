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
from pytorch_lightning.plugins.precision import (
    NativeMixedPrecisionPlugin as MixedPrecisionPlugin,
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

pl.seed_everything(16, workers=True)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


@click.command()
@click.option("--n-epochs", type=int, default=200)
@click.option("--batch-size", type=int, default=64)
@click.option("--train/--no-train", is_flag=True, default=True)
@click.option("--auto-lr-find/--no-auto-lr-find", is_flag=True, default=False)
@click.option("--use-gpu/--no-use-gpu", is_flag=True, default=True)
def main(
    n_epochs: int, batch_size: int, train: bool, auto_lr_find: bool, use_gpu: bool
):
    use_gpu = torch.cuda.is_available() and use_gpu
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

    kwargs = dict(
        default_root_dir=OUTPUT_DIR,
        max_epochs=n_epochs,
        auto_lr_find=auto_lr_find,
        accelerator="auto",
        gradient_clip_val=0.3,
        gradient_clip_algorithm="value",
        track_grad_norm=2,
        callbacks=[
            LearningRateMonitor("epoch", log_momentum=True),
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
    )

    if use_gpu:
        kwargs.update(
            dict(
                devices=1,
                plugins=[
                    MixedPrecisionPlugin(precision=16, device="cuda"),
                ],
            )
        )

    trainer = pl.Trainer(**kwargs)

    if train:
        trainer.tune(model=model, datamodule=datamodule)
        trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
