"""
Sequence dataset: 8 consecutive frames (by video frame index) with label at the center.

Pickle format (same as frame-wise contact data, plus optional ``video_ids``)::

    frames        : (N, H, W, 3) uint8 RGB
    labels        : (N,) int 0/1
    frame_numbers : (N,) int  video frame index
    video_ids     : (N,) int  optional; same id = same clip. If omitted, all rows are one video.

Edge handling
-------------
* ``edge_mode="pad"``: replicate first/last frame so every labeled frame can be a center.
* ``edge_mode="skip"``: only centers with full 8-frame support inside the sequence (no padding).

Augmentations
-------------
Training uses ``ReplayCompose`` so the same random transform is applied to all 8 frames.
"""

from __future__ import annotations

import pickle
from typing import Literal

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

EdgeMode = Literal["pad", "skip"]


def get_train_transforms(img_size: int = 256) -> A.ReplayCompose:
    """Random augmentations; use with ``apply_replay_to_sequence`` for time-consistent samples."""
    return A.ReplayCompose(
        [
            A.Resize(img_size, img_size),
            A.Rotate(limit=20, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.2
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Deterministic preprocessing (no replay needed)."""
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


def apply_replay_to_sequence(
    transform: A.ReplayCompose, images: list[np.ndarray]
) -> list[torch.Tensor]:
    """
    Apply one sampled augmentation to every frame (``images``: H×W×3 uint8).

    Returns list of ``(C, H, W)`` float tensors.
    """
    first = transform(image=images[0])
    replay = first["replay"]
    out = [first["image"]]
    for i in range(1, len(images)):
        o = A.ReplayCompose.replay(replay, image=images[i])
        out.append(o["image"])
    return out


def apply_val_to_sequence(
    transform: A.Compose, images: list[np.ndarray]
) -> list[torch.Tensor]:
    return [transform(image=im)["image"] for im in images]


def _center_window_indices(
    center_pos: int,
    window_size: int,
    n: int,
    edge_mode: EdgeMode,
) -> np.ndarray | None:
    """
    For a **sorted** video of length ``n``, return indices ``(window_size,)`` into that
    video for the window centered on sorted index ``center_pos``.

    Center index inside the window is ``(window_size - 1) // 2`` (e.g. 3 for window_size=8).

    Returns ``None`` if ``edge_mode=="skip"`` and the window would cross bounds.
    """
    half_left = (window_size - 1) // 2
    half_right = window_size - 1 - half_left

    lo = center_pos - half_left
    hi = center_pos + half_right
    if edge_mode == "skip":
        if lo < 0 or hi >= n:
            return None
        return np.arange(lo, hi + 1, dtype=np.int64)

    # pad: clamp indices to [0, n-1]
    idx = np.arange(lo, hi + 1, dtype=np.int64)
    idx = np.clip(idx, 0, n - 1)
    return idx


class TemporalContactSequenceDataset(Dataset):
    """
    Each item: sequence ``(T, C, H, W)`` and scalar label for the **center** frame.

    Windows are built **within each video** (``video_ids``), sorted by ``frame_numbers``.
    Optional ``require_consecutive_frames``: require ``frame[i+1] == frame[i] + 1`` across the window.
    """

    def __init__(
        self,
        pkl_path: str,
        transform: A.Compose | A.ReplayCompose | None = None,
        *,
        window_size: int = 8,
        edge_mode: EdgeMode = "pad",
        require_consecutive_frames: bool = True,
    ) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.frames: np.ndarray = data["frames"]
        self.labels: np.ndarray = data["labels"].astype(np.float32)
        self.frame_numbers: np.ndarray = np.asarray(data["frame_numbers"]).astype(np.int64)
        if "video_ids" in data:
            self.video_ids: np.ndarray = np.asarray(data["video_ids"]).astype(np.int64)
        else:
            self.video_ids = np.zeros(len(self.labels), dtype=np.int64)

        self.transform = transform
        self.window_size = window_size
        self.edge_mode = edge_mode
        self.require_consecutive_frames = require_consecutive_frames
        self.center_index = (window_size - 1) // 2

        self._centers: list[tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Populate ``_centers`` as list of ``(video_row_index_for_center, ...)`` — store flat indices."""
        unique_vids = np.unique(self.video_ids)
        for vid in unique_vids:
            mask = self.video_ids == vid
            inds = np.where(mask)[0]
            order = np.argsort(self.frame_numbers[inds], kind="mergesort")
            sorted_flat = inds[order]
            fn = self.frame_numbers[sorted_flat]
            n = len(sorted_flat)

            for j in range(n):
                widx = _center_window_indices(j, self.window_size, n, self.edge_mode)
                if widx is None:
                    continue
                sel = sorted_flat[widx]
                if self.require_consecutive_frames:
                    fns = self.frame_numbers[sel]
                    if not np.all(np.diff(fns) == 1):
                        continue
                self._centers.append((int(sel[self.center_index]),))

        if not self._centers:
            raise RuntimeError(
                "No valid temporal windows. Check video_ids/frame_numbers, "
                "or set require_consecutive_frames=False / edge_mode='pad'."
            )

    def __len__(self) -> int:
        return len(self._centers)

    def __getitem__(self, idx: int):
        (center_flat_idx,) = self._centers[idx]
        vid = self.video_ids[center_flat_idx]
        mask = self.video_ids == vid
        inds = np.where(mask)[0]
        order = np.argsort(self.frame_numbers[inds], kind="mergesort")
        sorted_flat = inds[order]
        pos = int(np.where(sorted_flat == center_flat_idx)[0][0])

        widx = _center_window_indices(
            pos, self.window_size, len(sorted_flat), self.edge_mode
        )
        assert widx is not None
        sel = sorted_flat[widx]
        label = self.labels[center_flat_idx]

        seq = [self.frames[i] for i in sel]  # list of H,W,3 uint8

        if self.transform is not None:
            if isinstance(self.transform, A.ReplayCompose):
                tensors = apply_replay_to_sequence(self.transform, seq)
            else:
                tensors = apply_val_to_sequence(self.transform, seq)
        else:
            tensors = [
                torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 for im in seq
            ]

        x = torch.stack(tensors, dim=0)  # (T, C, H, W)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
