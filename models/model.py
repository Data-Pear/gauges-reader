from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tvm

BackboneName = Literal[
    "resnet18", "resnet34", "resnet50", "convnext_tiny", "mobilenet_v3_large"
]


@dataclass
class ModelConfig:
    backbone: BackboneName = "convnext_tiny"
    pretrained: bool = True
    dropout: float = 0.0  # можно 0.1-0.2 при переобучении


def _replace_head_with_regressor(
    model: nn.Module, in_features: int, dropout: float
) -> nn.Module:
    layers = []
    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, 1))
    model.regressor = nn.Sequential(*layers)  # attach for clarity
    return model


class GaugeRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            m = tvm.resnet18(weights=weights)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )

        elif cfg.backbone == "resnet34":
            weights = tvm.ResNet34_Weights.DEFAULT if cfg.pretrained else None
            m = tvm.resnet34(weights=weights)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )

        elif cfg.backbone == "resnet50":
            weights = tvm.ResNet50_Weights.DEFAULT if cfg.pretrained else None
            m = tvm.resnet50(weights=weights)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )

        elif cfg.backbone == "convnext_tiny":
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if cfg.pretrained else None
            m = tvm.convnext_tiny(weights=weights)
            # у convnext classifier = Sequential(LayerNorm2d, Flatten, Linear)
            in_features = m.classifier[-1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )

        elif cfg.backbone == "mobilenet_v3_large":
            weights = tvm.MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained else None
            m = tvm.mobilenet_v3_large(weights=weights)
            in_features = m.classifier[-1].in_features
            m.classifier = nn.Identity()
            self.backbone = m
            self.head = nn.Sequential(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(in_features, 1),
            )

        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        y = self.head(feats)  # [B, 1]
        return y.squeeze(1)  # [B]
