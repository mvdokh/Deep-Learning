"""
Jaw keypoint tracker: EfficientNet-B2 backbone + temporal context + condition FiLM + heatmap head.

Tensor flow
-----------
Input ``x``: (B, T, C, H, W), ``condition_ids``: (B,)

1. Per-frame spatial features from backbone (features_only).
2. Global-pool per frame → (B, T, F) → TemporalConvNet → center (B, F).
3. Condition embedding + temporal context → FiLM modulates center spatial map.
4. HeatmapDecoder → (B, 2, H, W).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

EFFICIENTNET_B2_FEATURE_DIM = 1408
NUM_CONDITIONS = 3


class TemporalConvNet(nn.Module):
    """1D CNN over time; returns center-timestep features (B, C)."""

    def __init__(
        self,
        in_features: int,
        *,
        hidden_channels: int = 256,
        num_layers: int = 3,
        window_size: int = 8,
        center_index: int | None = None,
    ) -> None:
        super().__init__()
        if center_index is None:
            center_index = (window_size - 1) // 2
        self.center_index = center_index

        layers: list[nn.Module] = []
        c_in = in_features
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(c_in, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = hidden_channels
        self.conv = nn.Sequential(*layers)
        self.out_dim = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x[:, self.center_index, :]


class ConditionFiLM(nn.Module):
    """Condition embedding + temporal context → scale/shift on spatial features."""

    def __init__(
        self,
        spatial_channels: int,
        temporal_dim: int,
        num_conditions: int = NUM_CONDITIONS,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_conditions, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(temporal_dim + embed_dim, spatial_channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_channels * 2, spatial_channels * 2),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        temporal_ctx: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> torch.Tensor:
        # spatial: (B, C, h, w)
        cond = self.embed(condition_ids)
        ctx = torch.cat([temporal_ctx, cond], dim=1)
        gamma_beta = self.proj(ctx)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return spatial * (1.0 + gamma) + beta


class HeatmapDecoder(nn.Module):
    """Upsample spatial features to full-resolution 2-channel heatmaps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.mid_channels = hidden // 2
        self.out_conv = nn.Conv2d(hidden // 2, out_channels, 1)

    def forward(self, x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        x = self.refine(x)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return self.out_conv(x)


class JawKeypointTracker(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        window_size: int = 8,
        temporal_hidden: int = 256,
        temporal_layers: int = 3,
        decoder_hidden: int = 256,
        num_conditions: int = NUM_CONDITIONS,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.center_index = (window_size - 1) // 2

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.spatial_channels = self.backbone.feature_info.channels()[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = getattr(
            self.backbone, "num_features", EFFICIENTNET_B2_FEATURE_DIM
        )
        if hasattr(self.backbone, "feature_info"):
            feat_dim = self.backbone.feature_info.channels()[-1]

        self.temporal = TemporalConvNet(
            feat_dim,
            hidden_channels=temporal_hidden,
            num_layers=temporal_layers,
            window_size=window_size,
        )
        self.film = ConditionFiLM(
            self.spatial_channels,
            self.temporal.out_dim,
            num_conditions=num_conditions,
        )
        self.decoder = HeatmapDecoder(
            self.spatial_channels,
            out_channels=2,
            hidden=decoder_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        assert t == self.window_size

        flat = x.reshape(b * t, c, h, w)
        spatial_list = self.backbone(flat)
        spatial = spatial_list[0]  # (B*T, C, h', w')
        _, sc, sh, sw = spatial.shape

        pooled = self.global_pool(spatial).flatten(1)  # (B*T, C)
        pooled = pooled.reshape(b, t, -1)
        temporal_ctx = self.temporal(pooled)

        center_spatial = spatial.reshape(b, t, sc, sh, sw)[:, self.center_index]
        modulated = self.film(center_spatial, temporal_ctx, condition_ids)
        heatmaps = self.decoder(modulated, out_h=h, out_w=w)
        return heatmaps

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable


def build_model(
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    window_size: int = 8,
    temporal_hidden: int = 256,
    temporal_layers: int = 3,
    decoder_hidden: int = 256,
) -> JawKeypointTracker:
    return JawKeypointTracker(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        window_size=window_size,
        temporal_hidden=temporal_hidden,
        temporal_layers=temporal_layers,
        decoder_hidden=decoder_hidden,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
