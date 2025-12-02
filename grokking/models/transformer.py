from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from grokking.models import BaseClassifierCE


__all__ = ["GrokTransformer"]


class GrokTransformer(BaseClassifierCE):
    token_emb: nn.Embedding
    pos_emb: Tensor
    transformer: nn.TransformerEncoder
    out_proj: nn.Linear

    def __init__(
        self,
        n_emb: int,
        width: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_hidden: int = 128,
        seq_len: int = 2,
    ):
        super().__init__("GrokTransformer")
        self.token_emb = nn.Embedding(n_emb, width)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, width) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=ff_hidden, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(width, n_emb)

        # Randomly initialize parameters.
        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_emb(x)
        x = x + self.pos_emb.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.out_proj(x)
        return x
