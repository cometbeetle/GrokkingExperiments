from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet18

from grokking.models import BaseClassifierCE, BaseClassifierMSE

__all__ = ["GrokResNetCE", "GrokResNetMSE"]


class GrokResNetCE(BaseClassifierCE):
    def __init__(self, num_classes: int = 10, weight_scale: float = 120.0) -> None:
        super().__init__("GrokResNetCE")
        self.resnet = resnet18(num_classes=num_classes)
        self.resnet._norm_layer = nn.Identity()

        @torch.no_grad()
        def scale_weights(p):
            if isinstance(p, nn.Linear | nn.Conv2d):
                p.weight *= weight_scale

        self.apply(scale_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(x)


class GrokResNetMSE(BaseClassifierMSE):
    def __init__(self, num_classes: int = 10, weight_scale: float = 20.0) -> None:
        super().__init__("GrokResNetMSE")
        self.resnet = resnet18(num_classes=num_classes)
        self.resnet._norm_layer = nn.Identity()

        @torch.no_grad()
        def scale_weights(p):
            if isinstance(p, nn.Linear | nn.Conv2d):
                p.weight *= weight_scale

        self.apply(scale_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.resnet(x)
