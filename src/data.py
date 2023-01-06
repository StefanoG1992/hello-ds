"""Data preparation class"""

from pathlib import Path

import torch

import pytorch_lightning as pl

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import random_split


class LitDataModule(pl.LightningDataModule):
    """Lightning data module"""

    def __init__(
        self, data_dir: Path, batch_size: int = 64, testing: bool = True
    ):
        """
        Init lightning datamodule
        Args:
            data_dir (Path): directory where to download datasets
            batch_size (int): batch size, default is 64
            testing (bool): if True, limit datasets size to 64
        """
        super().__init__()
        self.data_dir: Path = data_dir
        self.batch_size: int = batch_size
        self.testing: bool = testing
        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage: str | None) -> None:
        """
        Setup step for lightning datamodule.

        Args:
            stage (str | None): used by lightning
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                ),
            ]
        )
        # extract data
        data_test: Dataset = datasets.CIFAR10(
            self.data_dir, train=False, download=False, transform=transform
        )
        data_full: Dataset = datasets.CIFAR10(
            self.data_dir, train=True, download=False, transform=transform
        )

        # split
        split_size: tuple[int, int] = (
            int(len(data_full) * 0.8),
            int(len(data_full) * 0.2),
        )
        data_train, data_val = random_split(data_full, split_size)

        # returning results
        if self.testing is True:
            idxs: torch.LongTensor = torch.arange(64)
            self.data_train = torch.utils.data.Subset(data_train, idxs)
            self.data_val = torch.utils.data.Subset(data_val, idxs)
            self.data_test = torch.utils.data.Subset(data_test, idxs)
        else:
            self.data_train = data_train
            self.data_val = data_val
            self.data_test = data_test

    def train_dataloader(self) -> DataLoader:
        """Create dataloader for train dataset"""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self) -> DataLoader:
        """Generate dataloader for validation dataset"""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self) -> DataLoader:
        """Generate dataloader for test dataset"""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )
