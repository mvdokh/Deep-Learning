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
KEYPOINT_SIGMA_DIAMETER_PX = 8.0

# experiment_id: 0=IRt_BiPoles, 1=IRt_TeLC, 2=PCRt_BiPoles
TELC_EXPERIMENT_ID = 1
BIPOLES_EXPERIMENT_IDS = frozenset({0, 2})


def tip_supervision_weight_for_experiment(
    experiment_id: int,
    *,
    occluded_tip_weight: float = 0.25,
) -> float:
    """Full tip supervision on visible TeLC; down-weight inferred BiPoles tips."""
    return 1.0 if int(experiment_id) == TELC_EXPERIMENT_ID else occluded_tip_weight


def get_img_size_hw(img_h: int = 240, img_w: int = 320) -> tuple[int, int]:
    return img_h, img_w


def get_geom_train_transforms(img_h: int = 240, img_w: int = 320) -> A.ReplayCompose:
    """
    Spatial augmentations + keypoint tracking (no normalize/tensor).

    Geometric ops update keypoints; image-only ops below are replayed across
    all 8 frames without moving labels (blur, noise, dust specks).

    Horizontal flip is disabled — side-view jaws always face the same direction.
    """
    return A.ReplayCompose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.2
            ),
            # Mild image-only corruptions (low p, small magnitude)
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.8), p=0.25),
            A.GaussNoise(std_range=(0.01, 0.035), p=0.25),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.12),
                p=0.15,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(1, 3),
                hole_width_range=(1, 3),
                fill="random",
                p=0.2,
            ),
            A.MotionBlur(blur_limit=3, p=0.1),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_geom_val_transforms(img_h: int = 240, img_w: int = 320) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_tensor_transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_train_transforms(img_h: int = 240, img_w: int = 320) -> A.ReplayCompose:
    """Backward-compatible alias (geom only; pair with :func:`get_tensor_transform`)."""
    return get_geom_train_transforms(img_h, img_w)


def get_val_transforms(img_h: int = 240, img_w: int = 320) -> A.Compose:
    """Backward-compatible alias (geom only; pair with :func:`get_tensor_transform`)."""
    return get_geom_val_transforms(img_h, img_w)


def scale_keypoints_orig_to_target(
    keypoints: list[tuple[float, float]],
    target_w: int,
    target_h: int,
    orig_w: int = ORIG_W,
    orig_h: int = ORIG_H,
) -> list[tuple[float, float]]:
    """Scale CSV keypoints (x in [0, orig_w], y in [0, orig_h]) to target resolution."""
    sx = target_w / orig_w
    sy = target_h / orig_h
    return [(float(x) * sx, float(y) * sy) for x, y in keypoints]


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
    in_target_pixels: bool = False,
) -> np.ndarray:
    """
    Build (2, H, W) heatmaps for tip and line.

  ``sigma`` scales with resize so the Gaussian diameter stays ~8 px in original space.

    Parameters
    ----------
    in_target_pixels
        If True, ``keypoints`` are already in ``(width, height)`` pixel coords
        (e.g. after albumentations). If False, they are in original 640×480 space.
    """
    scale_x = width / orig_w
    scale_y = height / orig_h
    sigma = (sigma_diameter_px / 6.0) * (scale_x + scale_y) / 2.0

    out = np.zeros((2, height, width), dtype=np.float32)
    for ch, (x, y) in enumerate(keypoints):
        if in_target_pixels:
            cx, cy = float(x), float(y)
        else:
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


def _normalize_keypoints(
    keypoints,
) -> list[tuple[float, float]]:
    return [(float(kp[0]), float(kp[1])) for kp in keypoints]


def apply_replay_to_sequence_with_keypoints(
    geom_transform: A.ReplayCompose,
    tensor_transform: A.Compose,
    images: list[np.ndarray],
    center_index: int,
    keypoints: list[tuple[float, float]],
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    """
    Apply geom augmentations (with keypoints on center frame), then normalize/tensor.
    """
    kps = [tuple(kp) for kp in keypoints]
    first = geom_transform(image=images[center_index], keypoints=kps)
    replay = first["replay"]
    transformed_kps = _normalize_keypoints(first["keypoints"])

    tensors: list[torch.Tensor] = []
    for i, im in enumerate(images):
        if i == center_index:
            aug = first["image"]
        else:
            aug = A.ReplayCompose.replay(replay, image=im)["image"]
        tensors.append(tensor_transform(image=aug)["image"])
    return tensors, transformed_kps


def apply_val_to_sequence_with_keypoints(
    geom_transform: A.Compose,
    tensor_transform: A.Compose,
    images: list[np.ndarray],
    center_index: int,
    keypoints: list[tuple[float, float]],
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    kps = [tuple(kp) for kp in keypoints]
    tensors: list[torch.Tensor] = []
    transformed_kps = kps
    for i, im in enumerate(images):
        if i == center_index:
            out = geom_transform(image=im, keypoints=kps)
            transformed_kps = _normalize_keypoints(out["keypoints"])
            aug = out["image"]
        else:
            aug = geom_transform(image=im)["image"]
        tensors.append(tensor_transform(image=aug)["image"])
    return tensors, transformed_kps


class JawKeypointSequenceDataset(Dataset):
    """
    Each item: sequence (T,C,H,W), heatmaps (2,H,W), keypoints_orig (2,2),
    keypoints_target (2,2), experiment_id, frame_num, tip_supervision_weight.
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
        occluded_tip_weight: float = 0.25,
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
        self.tensor_transform = get_tensor_transform()
        self.is_replay_geom = isinstance(transform, A.ReplayCompose) if transform else False
        self.window_size = window_size
        self.edge_mode = edge_mode
        self.require_consecutive_frames = require_consecutive_frames
        self.center_index = (window_size - 1) // 2
        self.img_h = img_h
        self.img_w = img_w
        self.occluded_tip_weight = occluded_tip_weight

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
            if self.is_replay_geom:
                tensors, transformed_kps = apply_replay_to_sequence_with_keypoints(
                    self.transform,
                    self.tensor_transform,
                    seq,
                    self.center_index,
                    keypoints,
                )
            else:
                tensors, transformed_kps = apply_val_to_sequence_with_keypoints(
                    self.transform,
                    self.tensor_transform,
                    seq,
                    self.center_index,
                    keypoints,
                )
            heatmaps = keypoints_to_heatmaps(
                transformed_kps, self.img_h, self.img_w, in_target_pixels=True
            )
            keypoints_target = torch.tensor(
                [[transformed_kps[0][0], transformed_kps[0][1]],
                 [transformed_kps[1][0], transformed_kps[1][1]]],
                dtype=torch.float32,
            )
        else:
            tensors = [
                torch.from_numpy(im).permute(2, 0, 1).float() / 255.0 for im in seq
            ]
            scaled = scale_keypoints_orig_to_target(
                keypoints, self.img_w, self.img_h
            )
            heatmaps = keypoints_to_heatmaps(
                scaled, self.img_h, self.img_w, in_target_pixels=True
            )
            keypoints_target = torch.tensor(
                [[scaled[0][0], scaled[0][1]], [scaled[1][0], scaled[1][1]]],
                dtype=torch.float32,
            )

        x = torch.stack(tensors, dim=0)
        hm = torch.from_numpy(heatmaps)
        exp_id = torch.tensor(int(vid), dtype=torch.long)
        frame_num = torch.tensor(int(self.frame_numbers[center_flat]), dtype=torch.long)
        tip_weight = torch.tensor(
            tip_supervision_weight_for_experiment(
                vid, occluded_tip_weight=self.occluded_tip_weight
            ),
            dtype=torch.float32,
        )
        return x, hm, keypoints_orig, keypoints_target, exp_id, frame_num, tip_weight
