"""
Prepare pickle files for **temporal** training: same keys as frame-wise data plus ``video_ids``.

Each row is still one extracted frame. Rows that share a ``video_id`` are treated as one
time-ordered clip (sorted by ``frame_numbers``) when building 8-frame windows.

You can reuse the Collision / Contact ``Create Dataset`` flow to produce per-experiment
``train.pkl`` / ``test.pkl``, then merge them with :func:`merge_pickles_with_video_ids`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


def merge_pickles_with_video_ids(
    pkl_paths: Sequence[str],
    out_path: str,
    start_video_id: int = 0,
) -> dict:
    """
    Concatenate several frame pickles and assign ``video_ids`` so each source file is one clip.

    Parameters
    ----------
    pkl_paths
        Each pickle must contain ``frames``, ``labels``, ``frame_numbers``.
    out_path
        Where to save the merged pickle.
    start_video_id
        First id to use (inclusive); incremented per input file.

    Returns
    -------
    dict
        The merged data dict (also written to ``out_path``).
    """
    all_frames: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_fn: list[np.ndarray] = []
    all_vid: list[np.ndarray] = []

    vid = start_video_id
    for path in pkl_paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        n = len(d["labels"])
        all_frames.append(d["frames"])
        all_labels.append(np.asarray(d["labels"]))
        all_fn.append(np.asarray(d["frame_numbers"]))
        all_vid.append(np.full(n, vid, dtype=np.int64))
        vid += 1

    data = {
        "frames": np.concatenate(all_frames, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "frame_numbers": np.concatenate(all_fn, axis=0),
        "video_ids": np.concatenate(all_vid, axis=0),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved merged temporal pickle → {out_path}  ({len(data['labels']):,} rows, video_ids {start_video_id}..{vid-1})")
    return data


def add_video_id_column(data: dict, video_id: int) -> dict:
    """Return a shallow copy of *data* with ``video_ids`` set to a constant."""
    out = dict(data)
    n = len(data["labels"])
    out["video_ids"] = np.full(n, int(video_id), dtype=np.int64)
    return out


def save_pkl(data: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path}  ({mb:.1f} MB)")


def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
