"""Keypoint metrics: PCK and RMSE in original pixel space."""

from __future__ import annotations

import numpy as np
import torch


def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable coordinate decode from a single-channel heatmap.

    Parameters
    ----------
    heatmap : (B, H, W)

    Returns
    -------
    coords : (B, 2)  x, y in heatmap pixel coordinates
    """
    b, h, w = heatmap.shape
    flat = heatmap.reshape(b, -1)
    prob = torch.softmax(flat, dim=1).reshape(b, h, w)

    xs = torch.arange(w, device=heatmap.device, dtype=heatmap.dtype)
    ys = torch.arange(h, device=heatmap.device, dtype=heatmap.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    ex = (prob * xx).sum(dim=(1, 2))
    ey = (prob * yy).sum(dim=(1, 2))
    return torch.stack([ex, ey], dim=1)


def heatmaps_to_coords(pred: torch.Tensor) -> torch.Tensor:
    """Decode (B, 2, H, W) heatmaps to (B, 2, 2) — tip and line xy."""
    tip = soft_argmax_2d(pred[:, 0])
    line = soft_argmax_2d(pred[:, 1])
    return torch.stack([tip, line], dim=1)


def scale_coords_to_original(
    coords: torch.Tensor,
    img_w: int,
    img_h: int,
    orig_w: int = 640,
    orig_h: int = 480,
) -> torch.Tensor:
    """Scale heatmap-space coords to original image pixels."""
    scale = coords.new_tensor([img_w / orig_w, img_h / orig_h])
    return coords * scale


@torch.no_grad()
def compute_keypoint_metrics(
    pred_coords_hm: torch.Tensor,
    gt_coords_orig: torch.Tensor,
    *,
    img_w: int,
    img_h: int,
    orig_w: int = 640,
    orig_h: int = 480,
    pck_threshold: float = 10.0,
) -> dict[str, float]:
    """
    Parameters
    ----------
    pred_coords_hm : (B, 2, 2) predicted tip/line in heatmap space
    gt_coords_orig : (B, 2, 2) ground truth in original 640×480 space
    """
    pred_orig = scale_coords_to_original(
        pred_coords_hm, img_w, img_h, orig_w, orig_h
    )
    dist = torch.linalg.norm(pred_orig - gt_coords_orig, dim=-1)  # (B, 2)

    tip_dist = dist[:, 0].cpu().numpy()
    line_dist = dist[:, 1].cpu().numpy()
    all_dist = dist.cpu().numpy().ravel()

    return {
        "rmse_tip": float(np.sqrt(np.mean(tip_dist**2))),
        "rmse_line": float(np.sqrt(np.mean(line_dist**2))),
        "rmse_mean": float(np.sqrt(np.mean(all_dist**2))),
        "pck_tip": float(np.mean(tip_dist <= pck_threshold)),
        "pck_line": float(np.mean(line_dist <= pck_threshold)),
        "pck_mean": float(np.mean(all_dist <= pck_threshold)),
    }
