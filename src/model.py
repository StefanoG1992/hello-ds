"""ML model."""

from typing import Literal

import torch

import pytorch_lightning as pl

from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from torchmetrics.functional import accuracy


class Net(nn.Module):
    """Convolutional Neural Network to predict labeled images."""

    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, padding=1
        )
        self.conv2d_3 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1
        )
        self.dropout_2d = nn.Dropout2d(p=0.3)
        self.dense_1 = nn.Linear(8 * 8 * 8, 32)
        self.dense_2 = nn.Linear(32, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Forward step."""
        out = torch.tanh(self.conv2d_1(x))  # B * 16 * 32 * 32
        out = self.dropout_2d(out)  # 2d dropout
        out = torch.tanh(self.conv2d_2(out))
        out = F.max_pool2d(out, 2)  # B * 8 * 16 * 16
        out = self.dropout_2d(out)  # 2d dropout
        out = F.max_pool2d(F.relu(self.conv2d_3(out)), 2)  # B * 8 * 8 * 8
        out = nn.Flatten(end_dim=-1)(out)  # B * (8 * 8 * 8)
        out = torch.tanh(self.dense_1(out))
        out = self.dense_2(out)
        return out


class LitNet(pl.LightningModule):
    """Lightning rewriting of CNN Net class."""

    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.model = Net()
        self.lr = lr

    def evaluate_step(
        self,
        data: Tensor,
        target: Tensor,
        step: Literal["train", "eval", "test"],
    ) -> tuple[Tensor, Tensor]:
        """Evaluation step. It is the same for train, validation, test.

        Args:
            data (Tensor): data to evaluate. Must have shape B x C x H x W
            target (Tensor): target for testing. Must have shape B x N_labels
            step (str): descriptive step. Must be one of
                - train
                - eval
                - test

        Returns:
            loss (Tensor): cross-entropy loss
            accuracy (Tensor): accuracy
        """
        logits: Tensor = self(data)
        pred: Tensor = logits.argmax(dim=1)

        # compute accuracy
        acc: Tensor = accuracy(pred, target, task="multiclass", num_classes=10)

        # compute loss
        # recall cross_entropy requires inputs as logits, target as indices
        loss: Tensor = F.cross_entropy(logits, target)

        # logging
        self.log(f"{step}_loss", loss, on_epoch=True)
        self.log(f"{step}_accuracy", acc, on_epoch=True)

        return loss, acc

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Lightning-automated training step.

        batch-idx is used only for lightning internal purposes
        """
        data, target = batch

        # evaluate loss, accuracy
        loss, _ = self.evaluate_step(data, target, step="train")
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        """Lightning-automated validation step.

        batch-idx is used only for lightning internal purposes
        """
        data, target = batch

        # evaluate loss, accuracy
        _, acc = self.evaluate_step(data, target, step="eval")
        return acc

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Lightning-automated testing step.

        batch-idx is used only for lightning internal purposes
        """
        data, target = batch

        # evaluate loss, accuracy
        loss, _ = self.evaluate_step(data, target, step="train")
        return loss

    def forward(self, data: Tensor) -> Tensor:
        """Overriding abstract lightning forward."""
        return self.model(data)

    def configure_optimizers(self) -> optim.Optimizer:
        """Optimizer for training step."""
        return optim.Adam(self.parameters(), lr=self.lr)
