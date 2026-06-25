"""Dual-keypoint heatmap + coordinate losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import heatmaps_to_coords


class JawKeypointLoss(nn.Module):
    """
    Combined objective for jaw tip + base keypoints.

    total = loss_hm + coord_weight * loss_coord + relative_offset_weight * loss_rel

    Tip supervision is scaled per-sample (TeLC=1.0, occluded BiPoles lower).
    Relative offset loss (tip − base) is full weight on all conditions.
    """

    def __init__(
        self,
        *,
        coord_weight: float = 1.0,
        tip_coord_weight: float = 2.0,
        relative_offset_weight: float = 0.5,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        coord_scale: float = 320.0,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.tip_coord_weight = tip_coord_weight
        self.relative_offset_weight = relative_offset_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.coord_scale = coord_scale

    def focal_heatmap_loss_per_sample(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample mean focal loss. pred/target: (B, H, W)."""
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        pos_loss = -((1.0 - pred) ** self.focal_alpha) * torch.log(pred) * target
        neg_weight = (1.0 - target) ** self.focal_beta
        neg_loss = -(pred**self.focal_alpha) * torch.log(1.0 - pred) * neg_weight
        per_pixel = pos_loss + neg_loss
        b = pred.shape[0]
        return per_pixel.reshape(b, -1).mean(dim=1)

    def weighted_mean(
        self, values: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        wsum = weights.sum().clamp(min=1e-6)
        return (values * weights).sum() / wsum

    def coordinate_loss(
        self,
        pred: torch.Tensor,
        keypoints_target: torch.Tensor,
        tip_supervision_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_coords = heatmaps_to_coords(pred)
        tip_w = tip_supervision_weight * self.tip_coord_weight
        base_w = torch.ones_like(tip_w)

        loss_tip = F.smooth_l1_loss(
            pred_coords[:, 0], keypoints_target[:, 0], reduction="none"
        ).mean(dim=-1)
        loss_base = F.smooth_l1_loss(
            pred_coords[:, 1], keypoints_target[:, 1], reduction="none"
        ).mean(dim=-1)

        loss_tip_mean = self.weighted_mean(loss_tip, tip_w)
        loss_base_mean = self.weighted_mean(loss_base, base_w)
        denom = tip_w.sum() + base_w.sum()
        loss = (tip_w * loss_tip).sum() + (base_w * loss_base).sum()
        loss = loss / denom.clamp(min=1e-6)

        scale = self.coord_scale
        return loss / scale, loss_tip_mean / scale, loss_base_mean / scale

    def relative_offset_loss(
        self, pred: torch.Tensor, keypoints_target: torch.Tensor
    ) -> torch.Tensor:
        pred_coords = heatmaps_to_coords(pred)
        pred_off = pred_coords[:, 0] - pred_coords[:, 1]
        gt_off = keypoints_target[:, 0] - keypoints_target[:, 1]
        return F.smooth_l1_loss(pred_off, gt_off) / self.coord_scale

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        keypoints_target: torch.Tensor,
        tip_supervision_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        pred, target : (B, 2, H, W) predicted / target heatmaps
        keypoints_target : (B, 2, 2) tip and base xy in heatmap pixels
        tip_supervision_weight : (B,) per-sample tip loss scale (1.0 TeLC, lower BiPoles)
        """
        tip_w = tip_supervision_weight.to(pred.device, dtype=pred.dtype)

        loss_tip_hm_ps = self.focal_heatmap_loss_per_sample(pred[:, 0], target[:, 0])
        loss_line_hm_ps = self.focal_heatmap_loss_per_sample(pred[:, 1], target[:, 1])
        loss_tip_hm = self.weighted_mean(loss_tip_hm_ps, tip_w)
        loss_line_hm = loss_line_hm_ps.mean()
        loss_hm = 0.5 * (loss_tip_hm + loss_line_hm)

        loss_coord, loss_coord_tip, loss_coord_base = self.coordinate_loss(
            pred, keypoints_target, tip_w
        )
        loss_rel = self.relative_offset_loss(pred, keypoints_target)

        loss = (
            loss_hm
            + self.coord_weight * loss_coord
            + self.relative_offset_weight * loss_rel
        )

        return loss, {
            "loss_hm": float(loss_hm.detach().cpu()),
            "loss_tip": float(loss_tip_hm.detach().cpu()),
            "loss_line": float(loss_line_hm.detach().cpu()),
            "loss_coord": float(loss_coord.detach().cpu()),
            "loss_coord_tip": float(loss_coord_tip.detach().cpu()),
            "loss_coord_base": float(loss_coord_base.detach().cpu()),
            "loss_rel": float(loss_rel.detach().cpu()),
        }


# Backward-compatible alias
BalancedKeypointHeatmapLoss = JawKeypointLoss
