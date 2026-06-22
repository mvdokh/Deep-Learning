"""
Temporal jaw keypoint dataset with dual heatmap targets at the center frame.

Pickle format::

    frames          : (N, H, W, 3) uint8 RGB
    frame_numbers   : (N,) int
    keypoints_tip   : (N, 2) float32  x, y in original pixels (640×480)
    keypoints_line  : (N, 2) float32
    experiment_ids  : (N,) int
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

ORIG_W, ORIG_H = 640, 480
KEYPOINT_SIGMA_DIAMETER_PX = 10.0


def get_img_size_hw(img_h: int = 240, img_w: int = 320) -> tuple[int, int]:
    return img_h, img_w


def get_train_transforms(img_h: int = 240, img_w: int = 320) -> A.ReplayCompose:
    return A.ReplayCompose(
        [
            A.Resize(img_h, img_w),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.2
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms(img_h: int = 240, img_w: int = 320) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_h, img_w),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def gaussian_heatmap(
    height: int,
    width: int,
    cx: float,
    cy: float,
    sigma: float,
) -> np.ndarray:
    """Single 2D Gaussian heatmap, peak 1.0 at (cx, cy)."""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return g.astype(np.float32)


def keypoints_to_heatmaps(
    keypoints: list[tuple[float, float]],
    height: int,
    width: int,
    *,
    sigma_diameter_px: float = KEYPOINT_SIGMA_DIAMETER_PX,
    orig_w: int = ORIG_W,
    orig_h: int = ORIG_H,
) -> np.ndarray:
    """
    Build (2, H, W) heatmaps for tip and line.

    ``sigma`` scales with resize so the Gaussian diameter stays ~10 px in original space.
    """
    scale_x = width / orig_w
    scale_y = height / orig_h
    sigma = (sigma_diameter_px / 6.0) * (scale_x + scale_y) / 2.0

    out = np.zeros((2, height, width), dtype=np.float32)
    for ch, (x, y) in enumerate(keypoints):
        cx = x * scale_x
        cy = y * scale_y
        out[ch] = gaussian_heatmap(height, width, cx, cy, sigma)
    return out


def _center_window_indices(
    center_pos: int,
    window_size: int,
    n: int,
    edge_mode: EdgeMode,
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


def apply_replay_to_sequence_with_keypoints(
    transform: A.ReplayCompose,
    images: list[np.ndarray],
    center_index: int,
    keypoints: list[tuple[float, float]],
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    """
    Apply one random augmentation to all frames; keypoints transformed via center frame.
    """
    kps = [tuple(kp) for kp in keypoints]
    first = transform(image=images[center_index], keypoints=kps)
    replay = first["replay"]
    transformed_kps = first["keypoints"]

    tensors: list[torch.Tensor] = []
    for i, im in enumerate(images):
        if i == center_index:
            tensors.append(first["image"])
        else:
            tensors.append(A.ReplayCompose.replay(replay, image=im)["image"])
    return tensors, transformed_kps


def apply_val_to_sequence_with_keypoints(
    transform: A.Compose,
    images: list[np.ndarray],
    center_index: int,
    keypoints: list[tuple[float, float]],
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    kps = [tuple(kp) for kp in keypoints]
    tensors: list[torch.Tensor] = []
    transformed_kps = kps
    for i, im in enumerate(images):
        if i == center_index:
            out = transform(image=im, keypoints=kps)
            transformed_kps = out["keypoints"]
            tensors.append(out["image"])
        else:
            tensors.append(transform(image=im)["image"])
    return tensors, transformed_kps


class JawKeypointSequenceDataset(Dataset):
    """
    Each item: sequence (T,C,H,W), heatmaps (2,H,W), keypoints_orig (2,2), experiment_id.
  """

    def __init__(
        self,
        pkl_path: str,
        transform: A.Compose | A.ReplayCompose | None = None,
        *,
        window_size: int = 8,
        edge_mode: EdgeMode = "pad",
        require_consecutive_frames: bool = True,
        img_h: int = 240,
        img_w: int = 320,
    ) -> None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.frames: np.ndarray = data["frames"]
        self.keypoints_tip: np.ndarray = data["keypoints_tip"].astype(np.float32)
        self.keypoints_line: np.ndarray = data["keypoints_line"].astype(np.float32)
        self.frame_numbers: np.ndarray = np.asarray(data["frame_numbers"]).astype(np.int64)
        if "experiment_ids" in data:
            self.experiment_ids: np.ndarray = np.asarray(data["experiment_ids"]).astype(np.int64)
        else:
            exp = int(data.get("experiment_id", 0))
            self.experiment_ids = np.full(len(self.frames), exp, dtype=np.int64)

        self.transform = transform
        self.window_size = window_size
        self.edge_mode = edge_mode
        self.require_consecutive_frames = require_consecutive_frames
        self.center_index = (window_size - 1) // 2
        self.img_h = img_h
        self.img_w = img_w

        self._centers: list[int] = []
        self._build_index()

    def _build_index(self) -> None:
        for vid in np.unique(self.experiment_ids):
            mask = self.experiment_ids == vid
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
                self._centers.append(int(sel[self.center_index]))

        if not self._centers:
            raise RuntimeError("No valid temporal windows in dataset.")

    @property
    def center_indices_for_sampler(self) -> np.ndarray:
        """experiment_id for each valid window (aligned with dataset indices)."""
        return np.asarray([self.experiment_ids[i] for i in self._centers], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._centers)

    def __getitem__(self, idx: int):
        center_flat = self._centers[idx]
        vid = self.experiment_ids[center_flat]
        mask = self.experiment_ids == vid
        inds = np.where(mask)[0]
        order = np.argsort(self.frame_numbers[inds], kind="mergesort")
        sorted_flat = inds[order]
        pos = int(np.where(sorted_flat == center_flat)[0][0])

        widx = _center_window_indices(
            pos, self.window_size, len(sorted_flat), self.edge_mode
        )
        assert widx is not None
        sel = sorted_flat[widx]
        seq = [self.frames[i] for i in sel]

        tip = tuple(self.keypoints_tip[center_flat].tolist())
        line = tuple(self.keypoints_line[center_flat].tolist())
        keypoints = [tip, line]
        keypoints_orig = torch.tensor(
            [[tip[0], tip[1]], [line[0], line[1]]], dtype=torch.float32
        )

        if self.transform is not None:
            if isinstance(self.transform, A.ReplayCompose):
                tensors, transformed_kps = apply_replay_to_sequence_with_keypoints(
                    self.transform, seq, self.center_index, keypoints
                )
            else:
                tensors, transformed_kps = apply_val_to_sequence_with_keypoints(
                    self.transform, seq, self.center_index, keypoints
                )
            heatmaps = keypoints_to_heatmaps(
                transformed_kps, self.img_h, self.img_w
            )
        else:
            tensors = [
                torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 for im in seq
            ]
            heatmaps = keypoints_to_heatmaps(keypoints, self.img_h, self.img_w)

        x = torch.stack(tensors, dim=0)
        hm = torch.from_numpy(heatmaps)
        exp_id = torch.tensor(int(vid), dtype=torch.long)
        frame_num = torch.tensor(int(self.frame_numbers[center_flat]), dtype=torch.long)
        return x, hm, keypoints_orig, exp_id, frame_num
