"""
Sliding-window inference for jaw keypoint tracking.
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

from metrics import heatmaps_to_coords, scale_coords_to_original
from model import build_model

CONDITIONS = ("IRt_BiPoles", "IRt_TeLC", "PCRt_BiPoles")
CONDITION_TO_ID = {name: i for i, name in enumerate(CONDITIONS)}
ORIG_W, ORIG_H = 640, 480


def get_inference_transforms(img_h: int = 240, img_w: int = 320) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_h, img_w),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def load_jaw_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(
        pretrained=False,
        freeze_backbone=False,
        window_size=int(cfg.get("window_size", 8)),
        temporal_hidden=int(cfg.get("temporal_hidden", 256)),
        temporal_layers=int(cfg.get("temporal_layers", 3)),
        decoder_hidden=int(cfg.get("decoder_hidden", 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded jaw model from {checkpoint_path}")
    return model


def window_indices_for_center(
    center_pos: int,
    n: int,
    window_size: int,
    edge_mode: Literal["pad", "skip"] = "pad",
) -> np.ndarray | None:
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


def load_images_from_dir(
    input_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PNGs named ``{frame:07d}.png`` sorted by frame number.

    Returns
    -------
    frames_rgb : (N, H, W, 3) uint8
    frame_numbers : (N,) int original frame indices from filenames
    """
    paths = sorted(Path(input_dir).glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG files in {input_dir}")

    frames: list[np.ndarray] = []
    frame_numbers: list[int] = []
    for p in paths:
        stem = p.stem.lstrip("0") or "0"
        frame_numbers.append(int(stem))
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read {p}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    order = np.argsort(frame_numbers)
    frames_arr = np.stack(frames, axis=0)[order]
    fn_arr = np.asarray(frame_numbers, dtype=np.int64)[order]
    return frames_arr, fn_arr


@torch.no_grad()
def sliding_window_keypoint_inference(
    model: nn.Module,
    frames_rgb: np.ndarray,
    transform: A.Compose,
    device: torch.device,
    condition_id: int,
    *,
    window_size: int = 8,
    batch_size: int = 8,
    edge_mode: Literal["pad", "skip"] = "pad",
    img_w: int = 320,
    img_h: int = 240,
) -> pd.DataFrame:
    """
    Predict tip and line keypoints for each frame position in ``frames_rgb``.

    Returns DataFrame with columns:
    ``frame_idx``, ``tip_x``, ``tip_y``, ``line_x``, ``line_y`` in original 640×480 coords.
    """
    model.eval()
    n = len(frames_rgb)
    centers: list[int] = []
    batches: list[list[np.ndarray]] = []

    for c in range(n):
        widx = window_indices_for_center(c, n, window_size, edge_mode)
        if widx is None:
            continue
        seq = [frames_rgb[i] for i in widx]
        centers.append(c)
        batches.append(seq)

    cond = torch.tensor([condition_id], dtype=torch.long, device=device)
    tips = np.full((n, 2), np.nan, dtype=np.float32)
    lines = np.full((n, 2), np.nan, dtype=np.float32)

    for start in tqdm(range(0, len(batches), batch_size), desc="Inference"):
        chunk = batches[start : start + batch_size]
        ch_centers = centers[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for seq in chunk:
            tens = [transform(image=im)["image"] for im in seq]
            tensors.append(torch.stack(tens, dim=0))
        x = torch.stack(tensors, dim=0).to(device)
        exp_ids = cond.expand(x.size(0))
        pred_hm = model(x, exp_ids)
        coords = heatmaps_to_coords(pred_hm)
        coords_orig = scale_coords_to_original(
            coords, img_w, img_h, ORIG_W, ORIG_H
        ).cpu().numpy()

        for ci, coord in zip(ch_centers, coords_orig):
            tips[ci] = coord[0]
            lines[ci] = coord[1]

    return pd.DataFrame(
        {
            "frame_idx": np.arange(n, dtype=np.int64),
            "tip_x": tips[:, 0],
            "tip_y": tips[:, 1],
            "line_x": lines[:, 0],
            "line_y": lines[:, 1],
        }
    )


def run_inference_on_image_dir(
    checkpoint_path: str,
    input_dir: str,
    condition: str,
    output_csv: str,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    if condition not in CONDITION_TO_ID:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    img_h = int(cfg.get("img_h", 240))
    img_w = int(cfg.get("img_w", 320))
    window_size = int(cfg.get("window_size", 8))

    model = load_jaw_model(checkpoint_path, dev)
    frames, frame_numbers = load_images_from_dir(input_dir)
    transform = get_inference_transforms(img_h, img_w)

    df = sliding_window_keypoint_inference(
        model,
        frames,
        transform,
        dev,
        CONDITION_TO_ID[condition],
        window_size=window_size,
        img_w=img_w,
        img_h=img_h,
    )
    df["frame"] = frame_numbers
    df = df[
        ["frame", "tip_x", "tip_y", "line_x", "line_y"]
    ]
    df.to_csv(output_csv, index=False)
    print(f"Wrote predictions → {output_csv}")
    return df
