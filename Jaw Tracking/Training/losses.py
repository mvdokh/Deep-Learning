"""Dual-keypoint heatmap + coordinate losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import heatmaps_to_coords


class JawKeypointLoss(nn.Module):
    """
    Combined objective for jaw tip + base keypoints.

    total = heatmap_focal + coord_weight * coordinate_smooth_l1

    Heatmap focal loss down-weights easy background pixels. Coordinate loss
    decodes predictions with soft-argmax and penalizes distance to targets in
    heatmap pixel space so loss tracks localization, not just near-zero backgrounds.
    """

    def __init__(
        self,
        *,
        coord_weight: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        coord_scale: float = 320.0,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.coord_scale = coord_scale

    def focal_heatmap_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        pos_loss = -((1.0 - pred) ** self.focal_alpha) * torch.log(pred) * target
        neg_weight = (1.0 - target) ** self.focal_beta
        neg_loss = -(pred**self.focal_alpha) * torch.log(1.0 - pred) * neg_weight
        return (pos_loss + neg_loss).mean()

    def coordinate_loss(
        self, pred: torch.Tensor, keypoints_target: torch.Tensor
    ) -> torch.Tensor:
        pred_coords = heatmaps_to_coords(pred)
        return F.smooth_l1_loss(pred_coords, keypoints_target) / self.coord_scale

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        keypoints_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        pred, target : (B, 2, H, W) predicted / target heatmaps
        keypoints_target : (B, 2, 2) tip and base xy in heatmap pixels
        """
        loss_tip_hm = self.focal_heatmap_loss(pred[:, 0], target[:, 0])
        loss_line_hm = self.focal_heatmap_loss(pred[:, 1], target[:, 1])
        loss_hm = 0.5 * (loss_tip_hm + loss_line_hm)

        loss_coord = self.coordinate_loss(pred, keypoints_target)
        loss = loss_hm + self.coord_weight * loss_coord

        return loss, {
            "loss_hm": float(loss_hm.detach().cpu()),
            "loss_tip": float(loss_tip_hm.detach().cpu()),
            "loss_line": float(loss_line_hm.detach().cpu()),
            "loss_coord": float(loss_coord.detach().cpu()),
        }


# Backward-compatible alias
BalancedKeypointHeatmapLoss = JawKeypointLoss
