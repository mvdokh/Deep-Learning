"""
Utilities for building jaw-tracking pickle datasets from per-condition folders.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

CONDITIONS = ("IRt_BiPoles", "IRt_TeLC", "PCRt_BiPoles")
CONDITION_TO_ID = {name: i for i, name in enumerate(CONDITIONS)}

TIP_CSV = "jaw_tip_side_clean.csv"
LINE_CSV = "jaw_line_side.csv"
IMAGE_DIR = "images"
EXPECTED_SHAPE = (480, 640, 3)


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path}  ({mb:.1f} MB, {len(data['frame_numbers']):,} frames)")


def _image_path(images_dir: Path, frame: int) -> Path:
    return images_dir / f"{int(frame):07d}.png"


def load_aligned_condition(
    condition_dir: Path,
    condition_name: str,
    experiment_id: int,
) -> dict:
    """Inner-join tip/line CSVs, load matching PNGs, return pickle-ready dict."""
    tip_df = pd.read_csv(condition_dir / TIP_CSV)
    line_df = pd.read_csv(condition_dir / LINE_CSV)
    merged = tip_df.merge(line_df, on="frame", suffixes=("_tip", "_line"))
    merged = merged.sort_values("frame").reset_index(drop=True)

    images_dir = condition_dir / IMAGE_DIR
    frames_list: list[np.ndarray] = []
    frame_numbers: list[int] = []
    tips: list[list[float]] = []
    lines: list[list[float]] = []

    for row in merged.itertuples(index=False):
        img_path = _image_path(images_dir, row.frame)
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image for frame {row.frame}: {img_path}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Failed to read {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape != EXPECTED_SHAPE:
            raise ValueError(f"{img_path} shape {rgb.shape}, expected {EXPECTED_SHAPE}")
        frames_list.append(rgb)
        frame_numbers.append(int(row.frame))
        tips.append([float(row.x_tip), float(row.y_tip)])
        lines.append([float(row.x_line), float(row.y_line)])

    if not frames_list:
        raise RuntimeError(f"No aligned frames in {condition_dir}")

    return {
        "frames": np.stack(frames_list, axis=0),
        "frame_numbers": np.asarray(frame_numbers, dtype=np.int64),
        "keypoints_tip": np.asarray(tips, dtype=np.float32),
        "keypoints_line": np.asarray(lines, dtype=np.float32),
        "condition": condition_name,
        "experiment_id": int(experiment_id),
    }


def merge_pickles(pkl_paths: Sequence[str], out_path: str) -> dict:
    """Concatenate per-condition pickles; add ``experiment_ids`` per row."""
    all_frames: list[np.ndarray] = []
    all_fn: list[np.ndarray] = []
    all_tip: list[np.ndarray] = []
    all_line: list[np.ndarray] = []
    all_exp: list[np.ndarray] = []
    conditions: list[str] = []

    for path in pkl_paths:
        d = load_pkl(path)
        n = len(d["frame_numbers"])
        all_frames.append(d["frames"])
        all_fn.append(np.asarray(d["frame_numbers"]))
        all_tip.append(np.asarray(d["keypoints_tip"]))
        all_line.append(np.asarray(d["keypoints_line"]))
        exp_id = int(d.get("experiment_id", len(conditions)))
        all_exp.append(np.full(n, exp_id, dtype=np.int64))
        conditions.append(str(d.get("condition", Path(path).stem)))

    data = {
        "frames": np.concatenate(all_frames, axis=0),
        "frame_numbers": np.concatenate(all_fn, axis=0),
        "keypoints_tip": np.concatenate(all_tip, axis=0),
        "keypoints_line": np.concatenate(all_line, axis=0),
        "experiment_ids": np.concatenate(all_exp, axis=0),
        "conditions": conditions,
    }
    save_pkl(data, out_path)
    return data


def split_train_val(
    merged: dict,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Temporal 80/20 split within each experiment_id.

    Frames are sorted by ``frame_numbers``; the last ``val_fraction`` of each
    clip goes to validation so both splits keep consecutive frame runs (needed
    for 8-frame windows with ``require_consecutive_frames=True``).
    """
    del seed  # temporal split is deterministic; kept for API compatibility
    n = len(merged["frame_numbers"])
    train_mask = np.zeros(n, dtype=bool)

    for exp_id in np.unique(merged["experiment_ids"]):
        idx = np.where(merged["experiment_ids"] == exp_id)[0]
        order = np.argsort(merged["frame_numbers"][idx], kind="mergesort")
        sorted_idx = idx[order]
        n_val = max(1, int(round(len(sorted_idx) * val_fraction)))
        train_mask[sorted_idx[:-n_val]] = True

    def subset(mask: np.ndarray) -> dict:
        return {
            "frames": merged["frames"][mask],
            "frame_numbers": merged["frame_numbers"][mask],
            "keypoints_tip": merged["keypoints_tip"][mask],
            "keypoints_line": merged["keypoints_line"][mask],
            "experiment_ids": merged["experiment_ids"][mask],
        }

    train_data = subset(train_mask)
    val_data = subset(~train_mask)
    return train_data, val_data
