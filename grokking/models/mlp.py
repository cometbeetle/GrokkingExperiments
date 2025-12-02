from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from grokking.models import BaseModel, BaseClassifierCE, BaseClassifierMSE

__all__ = [
    "GrokEmbedMLP",
    "GrokMLP",
    "DeepGrokMLPMSE",
    "DeepGrokMLPCE",
    "DeepGrokMLPMixin",
]


class GrokEmbedMLP(BaseClassifierCE):
    n_emb: int
    hidden_units: int
    seq_len: int
    token_emb: nn.Embedding
    stack: nn.Sequential

    def __init__(self, n_emb: int, hidden_units: int = 288, seq_len: int = 2) -> None:
        super().__init__("GrokEmbedMLP")
        self.n_emb = n_emb
        self.hidden_units = hidden_units
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(n_emb, hidden_units)
        self.stack = nn.Sequential(
            nn.Linear(seq_len * hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_emb),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x).flatten(1)
        return self.stack(x)


class GrokMLP(BaseModel):
    in_features: int
    hidden_units: int
    out_features: int
    stack: nn.Sequential

    def __init__(
        self, in_features: int = 2, hidden_units: int = 2048, out_features: int = 1
    ) -> None:
        super().__init__("GrokMLP")
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.stack = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x.float())

    def predict(self, x: Tensor) -> Tensor:
        return self(x).long()

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        return (logits.long() == labels).float().mean().item()

    @staticmethod
    def compute_loss(logits: Tensor, labels: Tensor) -> Tensor:
        return F.mse_loss(logits, labels.float())


class DeepGrokMLPMixin(nn.Module):
    in_features: int
    hidden_units: int
    out_features: int
    n_layers: int
    in_layer: nn.Linear
    hidden_layers: nn.ModuleList
    out_layer: nn.Linear

    def __init__(
        self,
        in_features: int = 784,
        hidden_units: int = 200,
        out_features: int = 10,
        n_layers: int = 12,
        weight_scale: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_features = out_features
        self.n_layers = n_layers
        self.in_layer = nn.Linear(in_features, hidden_units)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_units, hidden_units) for _ in range(n_layers - 2)
        )
        self.out_layer = nn.Linear(hidden_units, out_features)

        @torch.no_grad()
        def scale_weights(p):
            if isinstance(p, nn.Linear):
                p.weight *= weight_scale

        self.apply(scale_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.in_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.out_layer(x)


class DeepGrokMLPMSE(DeepGrokMLPMixin, BaseClassifierMSE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="DeepGrokMLPMSE", *args, **kwargs)


class DeepGrokMLPCE(DeepGrokMLPMixin, BaseClassifierCE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(name="DeepGrokMLPCE", *args, **kwargs)
