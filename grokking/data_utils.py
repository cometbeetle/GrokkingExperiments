from __future__ import annotations

import random
import os
from dataclasses import dataclass, field

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from typing import Literal, Self, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from grokking.models import BaseModel
    from torch.optim import Optimizer


__all__ = [
    "Checkpoint",
    "DataTracker",
    "ModOpDataset",
    "set_seed",
    "save_checkpoint",
    "load_checkpoint",
]


def set_seed(seed: int = 246) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(
    model: BaseModel, optimizer: Optimizer, data_tracker: DataTracker, filename: Path
) -> None:
    """
    Save a checkpoint of the model and its current optimizer state.

    The model state dictionary, along with the optimizer state dictionary and
    data tracker, are saved using `torch.save`.
    """
    checkpoint: Checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_tracker": data_tracker,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename: Path) -> Checkpoint:
    """
    Load a model checkpoint from a file using `torch.load`.
    """
    return torch.load(filename, weights_only=False)


class Checkpoint(TypedDict):
    model_state_dict: dict
    optimizer_state_dict: dict
    data_tracker: DataTracker


@dataclass
class DataTracker:
    total_epochs: int = 0
    epoch_train_losses: list[float] = field(default_factory=list)
    epoch_train_accs: list[float] = field(default_factory=list)
    step_train_losses: list[float] = field(default_factory=list)
    step_train_accs: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)


class ModOpDataset(Dataset):
    p: int
    op: Literal["+", "-", "*", "/"]
    seed: int | None
    squeeze_labels: bool
    data: list[tuple[int, int, int]]

    def __init__(
        self,
        p: int = 97,
        op: Literal["+", "-", "*", "/"] = "/",
        seed: int | None = None,
        *,
        squeeze_labels: bool = False,
        generate: bool = True,
    ):
        self.p = p
        self.op = op
        self.seed = seed
        self.squeeze_labels = squeeze_labels
        self.data = []
        if generate:
            self.generate()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        def t(x) -> Tensor:
            return torch.tensor(x, dtype=torch.long)

        a, b, c = self.data[index]
        data = t([a, b])
        labels = t(c) if self.squeeze_labels else t([c])
        return data, labels

    def generate(self) -> None:
        # Generate all (a, b, c) triplets for 0 <= a,b < p.
        pairs = [(a, b) for a in range(self.p) for b in range(self.p)]
        random.shuffle(pairs)

        # Compute c = a `op` b (mod p).
        def compute(a: int, b: int) -> int:
            if self.op == "+":
                return (a + b) % self.p
            elif self.op == "-":
                return (a - b) % self.p
            elif self.op == "*":
                return (a * b) % self.p
            elif self.op == "/":
                if b == 0:
                    return 0
                return (a * pow(b, -1, self.p)) % self.p
            else:
                msg = f"Unknown operation '{self.op}'"
                raise ValueError(msg)

        self.data = [(a, b, compute(a, b)) for (a, b) in pairs]

    def split(self, data: Literal["train", "val"], train_frac: float = 0.5) -> Self:
        n_train = int(len(self.data) * train_frac)
        inst = self.__class__(self.p, self.op, self.seed, generate=False)

        if data == "train":
            inst.data = self.data[:n_train]
        elif data == "val":
            inst.data = self.data[n_train:]
        else:
            msg = f"Split must be either 'train' or 'val' (got '{data}')"
            raise ValueError(msg)

        return inst
