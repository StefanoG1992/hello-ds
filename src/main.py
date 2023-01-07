"""Main function.

Execute full ml pipeline from data preparation to testing
"""

from pathlib import Path

import click
import mlflow
import mlflow.pytorch

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data import LitDataModule
from model import LitNet

from utils.log import config_logger


@click.command(name="pipeline")
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to dataset directory",
)
@click.option(
    "-T",
    "--testing",
    is_flag=True,
    default=False,
    help="If passed, use only a short excerpt of the dataset",
)
def pipeline(data_dir: Path, testing: bool) -> None:
    """Full ml pipeline, from data retrieval to evaluation.

    Args:
        data_dir (path): path to dataset directory
        testing (bool): if true, use only a small fraction of the data
    """
    # logging
    logger = config_logger()
    logger.info("Starting ml pipeline.")

    with mlflow.start_run(run_name="basic-run"):
        # sync mlflow logger with custom one
        pytorch_logger = mlflow.pytorch._logger
        logger.addHandler(pytorch_logger)

        logger.info("Lightning setup:")
        datamodule: LitDataModule = LitDataModule(
            data_dir, batch_size=32, num_workers=8, testing=testing
        )
        datamodule.setup(stage="fit")

        # initialize net
        net = LitNet(lr=0.001)

        logger.info("Done. Training step:")
        mlflow.pytorch.autolog()

        trainer = pl.Trainer(
            max_epochs=10,
            auto_lr_find=True,
            callbacks=[EarlyStopping(monitor="eval_loss", mode="min")],
        )

        # training our model
        trainer.fit(net, datamodule=datamodule)

        logger.info("Done. Testing step:")
        trainer.test(datamodule=datamodule, ckpt_path="best")

        logger.info("Training finished. Logging model params")
        mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    pipeline()
