"""Data preparation class."""

from pathlib import Path

import torch

import pytorch_lightning as pl

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import random_split


class LitDataModule(pl.LightningDataModule):
    """Lightning data module."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        pin_memory: bool = True,
        num_workers: int = 8,
        persistent_workers: bool = True,
        testing: bool = False,
    ):
        """
        Init lightning datamodule
        Args:
            data_dir (Path): directory where to download datasets
            batch_size (int): batch size, default is 64
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
            num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
            testing (bool): if True, limit datasets size to 64
        """
        super().__init__()
        self.data_dir: Path = data_dir
        self.batch_size: int = batch_size
        self.testing: bool = testing
        self.pin_memory: bool = pin_memory
        self.num_workers: int = num_workers
        self.persistent_workers: bool = persistent_workers
        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage: str | None) -> None:
        """Setup step for lightning datamodule.

        Args:
            stage (str | None): used to log which stage are we in, e.g. train
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
        """Create dataloader for train dataset."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Generate dataloader for validation dataset."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Generate dataloader for test dataset."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
