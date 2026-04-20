"""
Temporal contact classifier: EfficientNet-B3 (timm) backbone + swappable temporal module.

Tensor flow
-----------
Sequence input: ``x`` of shape ``(B, T, C, H, W)`` with ``T = window_size`` (default 8).

1. Flatten time into batch: ``(B*T, C, H, W)``.
2. Backbone returns per-frame features: ``(B*T, F)`` with ``F = backbone_feat_dim`` (1536 for B3).
3. Restore time: ``(B, T, F)``.
4. ``TemporalModule`` maps ``(B, T, F)`` → logits ``(B, 1)`` for the **center** time step.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import timm
import torch
import torch.nn as nn


# EfficientNet-B3 global-pooled features (timm, num_classes=0)
EFFICIENTNET_B3_FEATURE_DIM = 1536


@runtime_checkable
class TemporalModule(Protocol):
    """Replace with e.g. ``LSTMTemporalModule`` while keeping the same I/O contract."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(B, T, F)`` per-frame features.

        Returns
        -------
        Tensor
            ``(B, 1)`` logits for the center frame.
        """
        ...


class TemporalConvNet(nn.Module):
    """
    Lightweight 1D CNN over time (kernel 3, padding 1 → length preserved).

    Input ``(B, T, F)`` → internal ``(B, F, T)`` for ``Conv1d`` → output logits ``(B, 1)``.
    """

    def __init__(
        self,
        in_features: int,
        *,
        hidden_channels: int = 256,
        num_layers: int = 3,
        center_index: int | None = None,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        if center_index is None:
            center_index = (window_size - 1) // 2
        self.center_index = center_index
        self.window_size = window_size

        layers: list[nn.Module] = []
        c_in = in_features
        for i in range(num_layers):
            c_out = hidden_channels
            # Conv1d: (B, C_in, T) → (B, C_out, T)
            layers.append(nn.Conv1d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        center = x[:, self.center_index, :]  # (B, C)
        return self.head(center)  # (B, 1)


class LSTMTemporalModule(nn.Module):
    """
    Drop-in alternative to :class:`TemporalConvNet` (same I/O: ``(B, T, F)`` → ``(B, 1)``).

    Use by passing ``temporal_module=LSTMTemporalModule(feat_dim, ...)`` after the backbone exists.
    """

    def __init__(
        self,
        in_features: int,
        *,
        hidden_size: int = 256,
        num_layers: int = 1,
        window_size: int = 8,
        center_index: int | None = None,
    ) -> None:
        super().__init__()
        if center_index is None:
            center_index = (window_size - 1) // 2
        self.center_index = center_index
        self.lstm = nn.LSTM(
            in_features,
            hidden_size,
            num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h, _ = self.lstm(x)  # (B, T, hidden)
        return self.head(h[:, self.center_index, :])  # (B, 1)


class TemporalContactClassifier(nn.Module):
    """
    Full model: timm backbone (features only) + temporal module + center-frame logit.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        temporal_module: nn.Module | None = None,
        temporal_hidden_channels: int = 256,
        temporal_num_layers: int = 3,
        window_size: int = 8,
        backbone_feat_dim: int = EFFICIENTNET_B3_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        # num_classes=0 removes classifier; forward() returns pooled features (N, F).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = getattr(self.backbone, "num_features", backbone_feat_dim)

        if temporal_module is None:
            self.temporal = TemporalConvNet(
                feat_dim,
                hidden_channels=temporal_hidden_channels,
                num_layers=temporal_num_layers,
                window_size=window_size,
            )
        else:
            self.temporal = temporal_module

        self.backbone_feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        assert t == self.window_size, f"Expected T={self.window_size}, got {t}"

        flat = x.reshape(b * t, c, h, w)  # (B*T, C, H, W)
        feat = self.backbone(flat)  # (B*T, F)
        feat = feat.reshape(b, t, -1)  # (B, T, F)
        logits = self.temporal(feat)  # (B, 1)
        return logits

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Fine-tune toggle: set backbone gradients on/off."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def build_model(
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    window_size: int = 8,
    backbone_name: str = "efficientnet_b3",
    temporal_hidden: int = 256,
    temporal_layers: int = 3,
) -> TemporalContactClassifier:
    """Convenience builder with default ``TemporalConvNet``."""
    return TemporalContactClassifier(
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        temporal_module=None,
        temporal_hidden_channels=temporal_hidden,
        temporal_num_layers=temporal_layers,
        window_size=window_size,
    )
