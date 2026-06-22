"""Dual-keypoint heatmap loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class BalancedKeypointHeatmapLoss(nn.Module):
    """
    Mean of per-keypoint MSE losses (tip + line).
    Intermediate total when one keypoint is accurate and the other is not.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        pred, target : (B, 2, H, W)
        """
        loss_tip = nn.functional.mse_loss(pred[:, 0], target[:, 0])
        loss_line = nn.functional.mse_loss(pred[:, 1], target[:, 1])
        loss = 0.5 * (loss_tip + loss_line)
        return loss, {
            "loss_tip": float(loss_tip.detach().cpu()),
            "loss_line": float(loss_line.detach().cpu()),
        }
