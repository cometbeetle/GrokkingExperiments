from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from grokking.data_utils import DataTracker


__all__ = ["BaseModel", "BaseClassifierCE", "BaseClassifierMSE"]


class BaseModel(nn.Module, ABC):
    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor: ...

    @staticmethod
    @abstractmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float: ...

    @staticmethod
    @abstractmethod
    def compute_loss(logits: Tensor, labels: Tensor) -> Tensor: ...

    @property
    def current_dev(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def train_epoch(
        self, loader: DataLoader, optimizer: Optimizer
    ) -> tuple[float, float]:
        self.train()
        epoch_loss: float = 0.0
        epoch_acc: float = 0.0
        n: int = 0

        for data, labels in loader:
            data = data.to(self.current_dev)
            labels = labels.to(self.current_dev)

            # Zero gradients.
            optimizer.zero_grad()

            # Model forward pass.
            logits = self(data)

            # Compute loss.
            loss = self.compute_loss(logits, labels)

            # Perform backpropagation.
            loss.backward()
            optimizer.step()

            # Keep track of training metrics.
            batch_size = data.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_acc += self.accuracy(logits.detach(), labels) * batch_size
            n += batch_size

        return epoch_loss / n, epoch_acc / n

    @torch.no_grad()
    def run_eval(
        self, loader: DataLoader, adv_eps: float | None = None
    ) -> tuple[float, float]:
        self.eval()
        total_loss: float = 0.0
        total_acc: float = 0.0
        n: int = 0

        for data, labels in loader:
            data = data.to(self.current_dev)
            labels = labels.to(self.current_dev)

            # Whether to enable adversarial evaluation example generation.
            if adv_eps is not None:
                with torch.enable_grad():
                    data_adv = data.clone().detach().requires_grad_(True)
                    logits = self(data_adv)
                    loss = self.compute_loss(logits, labels)
                    self.zero_grad()
                    loss.backward()
                    data_adv = data_adv + adv_eps * data_adv.grad.sign()
                data_adv = torch.clamp(data_adv, 0.0, 1.0).detach()
                data = data_adv

            logits = self(data)
            loss = self.compute_loss(logits, labels)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            total_acc += self.accuracy(logits, labels) * batch_size
            n += batch_size

        return total_loss / n, total_acc / n

    def run_train(
        self,
        epochs: int,
        batch_size: int,
        dataset: Dataset,
        optimizer: Optimizer,
        train_frac: float = 0.5,
        data_tracker: DataTracker | None = None,
        *,
        evaluate: bool = True,
        adv_eps: float | None = None,
    ) -> None:
        train_dataset, val_dataset = random_split(dataset, [train_frac, 1 - train_frac])
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        loop = tqdm(
            range(epochs),
            unit="epoch",
            desc=f"Training {self.name}",
        )

        for _ in loop:
            epoch_loss, epoch_acc = self.train_epoch(train_loader, optimizer)

            if data_tracker is not None:
                data_tracker.epoch_train_losses.append(epoch_loss)
                data_tracker.epoch_train_accs.append(epoch_acc)
                data_tracker.total_epochs += 1

            if evaluate:
                val_loss, val_acc = self.run_eval(val_loader, adv_eps)
                loop.set_postfix(
                    {
                        "Train Loss": f"{epoch_loss:.3f}",
                        "Train Acc": f"{epoch_acc:.3f}",
                        "Val Loss": f"{val_loss:.3f}",
                        "Val Acc": f"{val_acc:.3f}",
                    }
                )
                if data_tracker is not None:
                    data_tracker.val_losses.append(val_loss)
                    data_tracker.val_accs.append(val_acc)
            else:
                loop.set_postfix(
                    {"Train Loss": f"{epoch_loss:.3f}", "Train Acc": f"{epoch_acc:.3f}"}
                )


class BaseClassifierCE(BaseModel, ABC):
    def predict(self, x: Tensor) -> Tensor:
        return self(x).argmax(dim=-1)

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()

    @staticmethod
    def compute_loss(logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels)


class BaseClassifierMSE(BaseModel, ABC):
    def predict(self, x: Tensor) -> Tensor:
        return self(x).argmax(dim=-1)

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        preds = logits.argmax(dim=-1)
        labels = F.one_hot(labels, num_classes=logits.shape[1]).argmax(dim=-1)
        return (preds == labels).float().mean().item()

    @staticmethod
    def compute_loss(logits: Tensor, labels: Tensor) -> Tensor:
        labels = F.one_hot(labels, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, labels)
