"""Main function

Execute full ml pipeline from data preparation to testing
"""

from pathlib import Path

import click
import mlflow.pytorch

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils.log import config_logger
from data import LitDataModule
from model import LitNet


@click.command(name="pipeline")
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to dataset directory",
)
def pipeline(data_dir: Path) -> None:
    """
    Full ml pipeline, from data retrieval to evaluation.

    Args:
        data_dir: path to dataset directory
    """
    # logging
    logger = config_logger()
    logger.info("Starting ml pipeline.")

    logger.info("Lightning setup:")
    datamodule: LitDataModule = LitDataModule(
        data_dir, batch_size=64, testing=False
    )
    datamodule.setup(stage="fit")

    # initialize net
    net = LitNet()

    logger.info("Done. Training step:")
    mlflow.pytorch.autolog()

    trainer = pl.Trainer(
        max_epochs=50,
        callbacks=[EarlyStopping(monitor="eval_loss", mode="min")],
        enable_checkpointing=False,
    )

    # training our model
    trainer.fit(net, datamodule=datamodule)

    logger.info("Done. Testing step:")
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    pipeline()
