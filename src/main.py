"""Main function.

Execute full ml pipeline from data preparation to testing
"""

from pathlib import Path
from typing import Any

import logging
import click
import mlflow
import mlflow.pytorch

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from mlflow.entities.experiment import Experiment

from data import LitDataModule
from model import LitNet
from utils.info import config_logger, get_tags


@click.command(name="pipeline")
@click.option(
    "-d",
    "--data-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to dataset directory",
)
@click.option(
    "-e",
    "--experiment-name",
    type=str,
    default="Default",
    help="Experiment name",
)
@click.option(
    "-r",
    "--run-name",
    type=str,
    default="basic-run",
    help="Run name",
)
@click.option(
    "-T",
    "--testing",
    is_flag=True,
    default=False,
    help="If passed, use only a short excerpt of the dataset",
)
def pipeline(
    data_dir: Path,
    experiment_name: str,
    run_name: str,
    testing: bool,
) -> None:
    """Full ml pipeline, from data retrieval to evaluation.

    Args:
        data_dir (path): path to dataset directory
        experiment_name (str): experiment name. If it does not exist,
            it gets created. Default is 'Default' (exp_id = '0')
        run_name (str): Run name. Default is 'basic-run'
        testing (bool): if true, use only a small fraction of the data
    """
    # logging
    logger: logging.RootLogger = config_logger()
    logger.info("Starting ml pipeline.")

    # set experiment - create if not exist
    exp: Experiment = mlflow.set_experiment(experiment_name)

    # get tags - require git
    tags: dict[str, Any] = get_tags()

    with mlflow.start_run(
        experiment_id=exp.experiment_id, run_name=run_name, tags=tags
    ):
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
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
        )

        # training our model
        trainer.fit(net, datamodule=datamodule)

        logger.info("Done. Testing step:")
        trainer.test(model=net, datamodule=datamodule)

        logger.info("Training finished. Logging model params")
        mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    pipeline()
