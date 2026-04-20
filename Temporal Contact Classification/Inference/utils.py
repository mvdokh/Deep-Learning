"""
Sliding-window inference: per-frame logits aligned to the **center** of each 8-frame window.

Video frames are indexed 0 … N-1. For center frame ``c``, the window uses video indices
``c - half_left … c + half_right`` with edge replication (same convention as training pad mode).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm

_TRAIN = Path(__file__).resolve().parent.parent / "Training"
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from model import build_model


def get_inference_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def load_temporal_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Rebuild the model from a training checkpoint (``train.py`` / ``final_model.pt``)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(
        pretrained=False,
        freeze_backbone=False,
        window_size=int(cfg.get("window_size", 8)),
        temporal_hidden=int(cfg.get("temporal_hidden", 256)),
        temporal_layers=int(cfg.get("temporal_layers", 3)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded temporal model from {checkpoint_path}")
    return model


def window_indices_for_center(
    center_pos: int,
    n: int,
    window_size: int,
    edge_mode: Literal["pad", "skip"] = "pad",
) -> np.ndarray | None:
    """``center_pos`` is the video-frame index (0 … n-1). Returns indices into ``[0..n-1]``."""
    half_left = (window_size - 1) // 2
    half_right = window_size - 1 - half_left
    lo = center_pos - half_left
    hi = center_pos + half_right
    if edge_mode == "skip":
        if lo < 0 or hi >= n:
            return None
        return np.arange(lo, hi + 1, dtype=np.int64)
    idx = np.arange(lo, hi + 1, dtype=np.int64)
    return np.clip(idx, 0, n - 1)


@torch.no_grad()
def sliding_window_inference_on_frames(
    model: nn.Module,
    frames_rgb: np.ndarray,
    transform: A.Compose,
    device: torch.device,
    *,
    window_size: int = 8,
    batch_size: int = 16,
    edge_mode: Literal["pad", "skip"] = "pad",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    frames_rgb : (N, H, W, 3) uint8 RGB array (entire video in order).

    Returns
    -------
    DataFrame with columns ``frame``, ``logit``, ``probability`` for each video frame index.
    """
    model.eval()
    n = len(frames_rgb)
    t = window_size
    rows: list[dict] = []

    centers = []
    batches: list[list[np.ndarray]] = []

    for c in range(n):
        widx = window_indices_for_center(c, n, t, edge_mode)
        if widx is None:
            continue
        seq = [frames_rgb[i] for i in widx]
        centers.append(c)
        batches.append(seq)

    logits_out = np.zeros(n, dtype=np.float32)
    filled = np.zeros(n, dtype=bool)

    for start in tqdm(range(0, len(batches), batch_size), desc="Sliding windows"):
        chunk = batches[start : start + batch_size]
        ch_centers = centers[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for seq in chunk:
            tens = [transform(image=im)["image"] for im in seq]
            tensors.append(torch.stack(tens, dim=0))
        x = torch.stack(tensors, dim=0).to(device)  # (B, T, C, H, W)
        logit = model(x).squeeze(1).cpu().numpy()
        for ci, lg in zip(ch_centers, logit):
            logits_out[ci] = lg
            filled[ci] = True

    prob = 1.0 / (1.0 + np.exp(-logits_out))
    df = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "logit": logits_out,
            "probability": prob,
            "predicted": (prob >= 0.5).astype(np.int64),
        }
    )
    if not filled.all():
        df.loc[~filled, ["logit", "probability", "predicted"]] = np.nan
    return df


def load_video_rgb_all_frames(video_path: str) -> np.ndarray:
    """Load entire video into (N, H, W, 3) uint8 RGB (memory: use for moderate-length clips)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0)


@torch.no_grad()
def run_sliding_window_on_video_file(
    model: nn.Module,
    video_path: str,
    device: torch.device,
    *,
    img_size: int = 256,
    window_size: int = 8,
    batch_size: int = 16,
) -> pd.DataFrame:
    """End-to-end: load video → sliding-window logits → DataFrame."""
    frames = load_video_rgb_all_frames(video_path)
    tfm = get_inference_transforms(img_size)
    return sliding_window_inference_on_frames(
        model,
        frames,
        tfm,
        device,
        window_size=window_size,
        batch_size=batch_size,
    )


def frames_to_intervals(
    df: pd.DataFrame,
    label_col: str = "predicted",
    label_val: int = 1,
) -> pd.DataFrame:
    """Collapse consecutive positive frames into [Start, End] intervals."""
    df = df.sort_values("frame").reset_index(drop=True)
    mask = df[label_col] == label_val
    frames = df.loc[mask, "frame"].values.astype(np.int64)
    if len(frames) == 0:
        return pd.DataFrame(columns=["Start", "End"])

    intervals: list[tuple[int, int]] = []
    start = int(frames[0])
    prev = int(frames[0])
    for f in frames[1:]:
        f = int(f)
        if f == prev + 1:
            prev = f
        else:
            intervals.append((start, prev))
            start = f
            prev = f
    intervals.append((start, prev))
    return pd.DataFrame(intervals, columns=["Start", "End"])
