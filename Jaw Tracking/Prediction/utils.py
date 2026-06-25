"""
Sliding-window inference for jaw keypoint tracking (shared checkpoint).
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
from scipy.signal import savgol_filter
from tqdm import tqdm

_TRAIN = Path(__file__).resolve().parent.parent / "Training"
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from metrics import heatmaps_to_coords, scale_coords_to_original
from model import build_model

CONDITIONS = ("IRt_BiPoles", "IRt_TeLC", "PCRt_BiPoles")
DEFAULT_ORIG_W, DEFAULT_ORIG_H = 640, 480
TIP_CSV_NAME = "jaw_tip.csv"
BASE_CSV_NAME = "jaw_base.csv"


def default_checkpoint_path(
    checkpoints_root: str | Path = "../Training/checkpoints",
    filename: str = "best_model.pt",
) -> str:
    """Default checkpoint for the shared multi-condition model."""
    return str(Path(checkpoints_root) / filename)


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
        temporal_hidden=int(cfg.get("temporal_hidden", 384)),
        temporal_layers=int(cfg.get("temporal_layers", 3)),
        decoder_hidden=int(cfg.get("decoder_hidden", 512)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    cond = cfg.get("condition", "shared multi-condition")
    print(f"Loaded {cond} model from {checkpoint_path}")
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


def video_properties(video_path: str | Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or n <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    return width, height, n


def read_video_frame_slice(
    video_path: str | Path,
    start: int,
    end: int,
) -> np.ndarray:
    """Read inclusive frame range ``[start, end]`` as RGB uint8 array (T, H, W, 3)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(start, end + 1):
        ret, bgr = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(
                f"Failed reading frame while loading slice [{start}, {end}] from {video_path}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return np.stack(frames, axis=0)


def load_images_from_dir(
    input_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
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


def smooth_tip_trajectory(
    tip_x: np.ndarray,
    tip_y: np.ndarray,
    *,
    window: int = 7,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Savitzky–Golay smooth tip coords; leaves short sequences unchanged."""
    n = len(tip_x)
    if n < window:
        return tip_x.copy(), tip_y.copy()
    if window % 2 == 0:
        window += 1
    return (
        savgol_filter(tip_x, window_length=window, polyorder=polyorder),
        savgol_filter(tip_y, window_length=window, polyorder=polyorder),
    )


def apply_tip_smoothing_to_df(
    df: pd.DataFrame,
    *,
    smooth_tip_trajectory_enabled: bool = True,
    smooth_window: int = 7,
) -> pd.DataFrame:
    if not smooth_tip_trajectory_enabled:
        return df
    out = df.copy()
    tip_x, tip_y = smooth_tip_trajectory(
        out["tip_x"].to_numpy(dtype=np.float64),
        out["tip_y"].to_numpy(dtype=np.float64),
        window=smooth_window,
    )
    out["tip_x"] = tip_x.astype(np.float32)
    out["tip_y"] = tip_y.astype(np.float32)
    return out


def write_dataset_keypoint_csvs(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    tip_csv_name: str = TIP_CSV_NAME,
    base_csv_name: str = BASE_CSV_NAME,
) -> tuple[Path, Path]:
    """Write ``frame,x,y`` CSVs matching the dataset creation format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tip_path = out / tip_csv_name
    base_path = out / base_csv_name

    tip_df = pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "x": df["tip_x"].astype(float),
            "y": df["tip_y"].astype(float),
        }
    )
    base_df = pd.DataFrame(
        {
            "frame": df["frame"].astype(int),
            "x": df["line_x"].astype(float),
            "y": df["line_y"].astype(float),
        }
    )
    tip_df.to_csv(tip_path, index=False)
    base_df.to_csv(base_path, index=False)
    print(f"Wrote {tip_path}")
    print(f"Wrote {base_path}")
    return tip_path, base_path


@torch.no_grad()
def _run_window_batches(
    model: nn.Module,
    centers: list[int],
    batches: list[list[np.ndarray]],
    transform: A.Compose,
    device: torch.device,
    *,
    batch_size: int,
    img_w: int,
    img_h: int,
    orig_w: int,
    orig_h: int,
    tips: np.ndarray,
    lines: np.ndarray,
) -> None:
    for start in range(0, len(batches), batch_size):
        chunk = batches[start : start + batch_size]
        ch_centers = centers[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for seq in chunk:
            tens = [transform(image=im)["image"] for im in seq]
            tensors.append(torch.stack(tens, dim=0))
        x = torch.stack(tensors, dim=0).to(device)
        pred_hm = model(x)
        coords = heatmaps_to_coords(pred_hm)
        coords_orig = scale_coords_to_original(
            coords, img_w, img_h, orig_w, orig_h
        ).cpu().numpy()

        for ci, coord in zip(ch_centers, coords_orig):
            tips[ci] = coord[0]
            lines[ci] = coord[1]


@torch.no_grad()
def sliding_window_keypoint_inference(
    model: nn.Module,
    frames_rgb: np.ndarray,
    transform: A.Compose,
    device: torch.device,
    *,
    window_size: int = 8,
    batch_size: int = 8,
    edge_mode: Literal["pad", "skip"] = "pad",
    img_w: int = 320,
    img_h: int = 240,
    orig_w: int = DEFAULT_ORIG_W,
    orig_h: int = DEFAULT_ORIG_H,
    smooth_tip_trajectory_enabled: bool = True,
    smooth_window: int = 7,
) -> pd.DataFrame:
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

    tips = np.full((n, 2), np.nan, dtype=np.float32)
    lines = np.full((n, 2), np.nan, dtype=np.float32)

    for start in tqdm(range(0, len(batches), batch_size), desc="Inference"):
        chunk_centers = centers[start : start + batch_size]
        chunk_batches = batches[start : start + batch_size]
        _run_window_batches(
            model,
            chunk_centers,
            chunk_batches,
            transform,
            device,
            batch_size=batch_size,
            img_w=img_w,
            img_h=img_h,
            orig_w=orig_w,
            orig_h=orig_h,
            tips=tips,
            lines=lines,
        )

    df = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "tip_x": tips[:, 0],
            "tip_y": tips[:, 1],
            "line_x": lines[:, 0],
            "line_y": lines[:, 1],
        }
    )
    return apply_tip_smoothing_to_df(
        df,
        smooth_tip_trajectory_enabled=smooth_tip_trajectory_enabled,
        smooth_window=smooth_window,
    )


@torch.no_grad()
def sliding_window_keypoint_inference_from_video(
    model: nn.Module,
    video_path: str | Path,
    transform: A.Compose,
    device: torch.device,
    *,
    window_size: int = 8,
    batch_size: int = 8,
    edge_mode: Literal["pad", "skip"] = "pad",
    img_w: int = 320,
    img_h: int = 240,
    center_chunk: int = 512,
    orig_w: int = DEFAULT_ORIG_W,
    orig_h: int = DEFAULT_ORIG_H,
    smooth_tip_trajectory_enabled: bool = True,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """Stream a long video in chunks to avoid loading all frames into RAM."""
    model.eval()
    video_path = Path(video_path)
    _, _, n = video_properties(video_path)
    half_left = (window_size - 1) // 2
    half_right = window_size - 1 - half_left

    tips = np.full((n, 2), np.nan, dtype=np.float32)
    lines = np.full((n, 2), np.nan, dtype=np.float32)

    for chunk_start in tqdm(
        range(0, n, center_chunk), desc="Video chunks", unit="chunk"
    ):
        chunk_end = min(chunk_start + center_chunk, n)
        read_lo = max(0, chunk_start - half_left)
        read_hi = min(n - 1, chunk_end - 1 + half_right)
        frames = read_video_frame_slice(video_path, read_lo, read_hi)

        centers: list[int] = []
        batches: list[list[np.ndarray]] = []
        for c in range(chunk_start, chunk_end):
            widx = window_indices_for_center(c, n, window_size, edge_mode)
            if widx is None:
                continue
            seq = [frames[int(i - read_lo)] for i in widx]
            centers.append(c)
            batches.append(seq)

        _run_window_batches(
            model,
            centers,
            batches,
            transform,
            device,
            batch_size=batch_size,
            img_w=img_w,
            img_h=img_h,
            orig_w=orig_w,
            orig_h=orig_h,
            tips=tips,
            lines=lines,
        )

    df = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=np.int64),
            "tip_x": tips[:, 0],
            "tip_y": tips[:, 1],
            "line_x": lines[:, 0],
            "line_y": lines[:, 1],
        }
    )
    return apply_tip_smoothing_to_df(
        df,
        smooth_tip_trajectory_enabled=smooth_tip_trajectory_enabled,
        smooth_window=smooth_window,
    )


def _load_model_and_cfg(
    checkpoint_path: str | None,
    checkpoints_root: str | Path,
    device: str | None,
) -> tuple[nn.Module, torch.device, dict, str]:
    ckpt_path = checkpoint_path or default_checkpoint_path(checkpoints_root)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    model = load_jaw_model(ckpt_path, dev)
    return model, dev, cfg, ckpt_path


def run_inference_on_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint_path: str | None = None,
    checkpoints_root: str | Path = "../Training/checkpoints",
    device: str | None = None,
    batch_size: int = 8,
    center_chunk: int = 512,
    smooth_tip_trajectory_enabled: bool = True,
    smooth_window: int = 7,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Run inference on a video and save dataset-style CSVs.

    Writes ``jaw_tip.csv`` and ``jaw_base.csv`` (columns: ``frame,x,y``) to
    ``output_dir``. Frame numbers are 0-based video frame indices.
    """
    video_path = Path(video_path)
    model, dev, cfg, ckpt_path = _load_model_and_cfg(
        checkpoint_path, checkpoints_root, device
    )
    img_h = int(cfg.get("img_h", 240))
    img_w = int(cfg.get("img_w", 320))
    window_size = int(cfg.get("window_size", 8))
    transform = get_inference_transforms(img_h, img_w)
    orig_w, orig_h, n_frames = video_properties(video_path)

    print(f"Video: {video_path}")
    print(f"Frames: {n_frames:,}")
    print(f"Source size: {orig_w}x{orig_h}")
    print(f"Model input: {img_w}x{img_h}")
    print(f"Checkpoint: {ckpt_path}")

    df = sliding_window_keypoint_inference_from_video(
        model,
        video_path,
        transform,
        dev,
        window_size=window_size,
        batch_size=batch_size,
        img_w=img_w,
        img_h=img_h,
        center_chunk=center_chunk,
        orig_w=orig_w,
        orig_h=orig_h,
        smooth_tip_trajectory_enabled=smooth_tip_trajectory_enabled,
        smooth_window=smooth_window,
    )
    tip_path, base_path = write_dataset_keypoint_csvs(df, output_dir)
    return df, tip_path, base_path


def run_inference_on_image_dir(
    input_dir: str,
    output_dir: str | Path,
    *,
    checkpoint_path: str | None = None,
    checkpoints_root: str | Path = "../Training/checkpoints",
    device: str | None = None,
    batch_size: int = 8,
    smooth_tip_trajectory_enabled: bool = True,
    smooth_window: int = 7,
) -> tuple[pd.DataFrame, Path, Path]:
    """
    Run inference on a folder of PNG frames.

    Writes ``jaw_tip.csv`` and ``jaw_base.csv`` to ``output_dir``.
    """
    model, dev, cfg, ckpt_path = _load_model_and_cfg(
        checkpoint_path, checkpoints_root, device
    )
    img_h = int(cfg.get("img_h", 240))
    img_w = int(cfg.get("img_w", 320))
    window_size = int(cfg.get("window_size", 8))

    frames, frame_numbers = load_images_from_dir(input_dir)
    transform = get_inference_transforms(img_h, img_w)
    orig_h, orig_w = frames.shape[1], frames.shape[2]

    df = sliding_window_keypoint_inference(
        model,
        frames,
        transform,
        dev,
        window_size=window_size,
        batch_size=batch_size,
        img_w=img_w,
        img_h=img_h,
        orig_w=orig_w,
        orig_h=orig_h,
        smooth_tip_trajectory_enabled=smooth_tip_trajectory_enabled,
        smooth_window=smooth_window,
    )
    df["frame"] = frame_numbers
    df = df[["frame", "tip_x", "tip_y", "line_x", "line_y"]]
    tip_path, base_path = write_dataset_keypoint_csvs(df, output_dir)
    return df, tip_path, base_path
