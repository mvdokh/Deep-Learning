"""Optical flow + NCC patch refinement whisker tracker (classical CV)."""

from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np
from scipy.interpolate import splev, splprep

from .mask_utils import (
    dilate_skeleton_to_mask,
    get_control_points,
    skeletonize_mask,
)


CONFIG: dict = {
    "farneback": {
        "pyr_scale": 0.5,
        "levels": 5,
        "winsize": 15,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 1.5,
        "flags": 0,
    },
    "ncc_patch_size": 21,
    "ncc_search_radius": 30,
    "n_augmentations": 8,
    "use_augmentation_voting": False,
    "clahe_clip": 2.0,
    "clahe_tile": (8, 8),
}


def preprocess(image: np.ndarray) -> np.ndarray:
    """Convert ``image`` to grayscale uint8 and apply CLAHE."""
    if image is None:
        raise ValueError("preprocess received None image")

    img = image
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(CONFIG["clahe_clip"]),
        tileGridSize=tuple(CONFIG["clahe_tile"]),
    )
    return clahe.apply(img)


def compute_flow(ref: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow ``ref → tgt``. Returns ``(H, W, 2)``."""
    return cv2.calcOpticalFlowFarneback(ref, tgt, None, **CONFIG["farneback"])


def warp_mask(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Forward-warp a binary mask via flow (scatter + 1-px close)."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return out

    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]
    new_x = np.round(xs + fx).astype(np.int32)
    new_y = np.round(ys + fy).astype(np.int32)
    valid = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)
    out[new_y[valid], new_x[valid]] = 1

    kernel = np.ones((3, 3), dtype=np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


def warp_points(points: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Bilinearly sample flow at each ``(x, y)`` and return displaced points."""
    h, w = flow.shape[:2]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return pts.copy()

    x = pts[:, 0]
    y = pts[:, 1]
    x_clip = np.clip(x, 0.0, w - 1.0001)
    y_clip = np.clip(y, 0.0, h - 1.0001)
    x0 = np.floor(x_clip).astype(np.int32)
    y0 = np.floor(y_clip).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x_clip - x0).astype(np.float32)
    wy = (y_clip - y0).astype(np.float32)

    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]

    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    fx = w00 * f00[:, 0] + w01 * f01[:, 0] + w10 * f10[:, 0] + w11 * f11[:, 0]
    fy = w00 * f00[:, 1] + w01 * f01[:, 1] + w10 * f10[:, 1] + w11 * f11[:, 1]

    return np.column_stack([x + fx, y + fy]).astype(np.float32)


def _refine_with_ncc_impl(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    ref_points: np.ndarray,
    warped_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    patch_size = int(CONFIG["ncc_patch_size"])
    search_radius = int(CONFIG["ncc_search_radius"])
    half = patch_size // 2

    margin = half + search_radius
    pad_ref = cv2.copyMakeBorder(
        ref_image, margin, margin, margin, margin, cv2.BORDER_REFLECT_101
    )
    pad_tgt = cv2.copyMakeBorder(
        tgt_image, margin, margin, margin, margin, cv2.BORDER_REFLECT_101
    )

    ref_pts = np.asarray(ref_points, dtype=np.float32).reshape(-1, 2)
    warp_pts = np.asarray(warped_points, dtype=np.float32).reshape(-1, 2)

    refined = warp_pts.copy()
    scores = np.zeros(len(ref_pts), dtype=np.float32)

    for i in range(len(ref_pts)):
        rx, ry = int(round(ref_pts[i, 0])), int(round(ref_pts[i, 1]))
        wx, wy = int(round(warp_pts[i, 0])), int(round(warp_pts[i, 1]))

        prx = rx + margin
        pry = ry + margin
        patch = pad_ref[pry - half : pry - half + patch_size,
                        prx - half : prx - half + patch_size]

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            scores[i] = 0.0
            continue

        pwx = wx + margin
        pwy = wy + margin
        win_size = patch_size + 2 * search_radius
        win_x0 = pwx - half - search_radius
        win_y0 = pwy - half - search_radius
        window = pad_tgt[win_y0 : win_y0 + win_size, win_x0 : win_x0 + win_size]

        if window.shape[0] != win_size or window.shape[1] != win_size:
            scores[i] = 0.0
            continue

        try:
            result = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            scores[i] = 0.0
            continue

        peak_y, peak_x = np.unravel_index(int(np.argmax(result)), result.shape)
        scores[i] = float(result[peak_y, peak_x])

        refined[i, 0] = float(peak_x) + (wx - search_radius)
        refined[i, 1] = float(peak_y) + (wy - search_radius)

    return refined.astype(np.float32), scores


def refine_with_ncc(
    ref_image: np.ndarray,
    tgt_image: np.ndarray,
    ref_points: np.ndarray,
    warped_points: np.ndarray,
) -> np.ndarray:
    """Refine warped control points by NCC patch matching against the reference."""
    refined, _ = _refine_with_ncc_impl(ref_image, tgt_image, ref_points, warped_points)
    return refined


def generate_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    n: int = 8,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``n`` ``(aug_image, aug_mask)`` pairs.

    Applies random rotation (±8°), zoom (0.95–1.05×), brightness jitter (±15
    intensity), and a smooth elastic displacement field. The same geometric
    transform is applied to both image and mask.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rng = np.random.default_rng()

    elastic_sigma = 8.0
    elastic_amp = 4.0

    is_color = image.ndim == 3

    for _ in range(int(n)):
        theta = float(rng.uniform(-8.0, 8.0))
        scale = float(rng.uniform(0.95, 1.05))
        delta = float(rng.uniform(-15.0, 15.0))

        affine = cv2.getRotationMatrix2D(center, theta, scale)
        aug_img = cv2.warpAffine(
            image, affine, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        aug_mask = cv2.warpAffine(
            mask, affine, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        rand_dx = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
        rand_dy = rng.uniform(-1.0, 1.0, size=(h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(rand_dx, (0, 0), elastic_sigma) * elastic_amp
        dy = cv2.GaussianBlur(rand_dy, (0, 0), elastic_sigma) * elastic_amp
        x_grid, y_grid = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = (x_grid + dx).astype(np.float32)
        map_y = (y_grid + dy).astype(np.float32)

        aug_img = cv2.remap(
            aug_img, map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        aug_mask = cv2.remap(
            aug_mask, map_x, map_y,
            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        aug_img = np.clip(aug_img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        if is_color and aug_img.ndim == 2:
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)

        aug_mask = (aug_mask > 0).astype(np.uint8)

        yield aug_img, aug_mask


def _smooth_refit(points: np.ndarray, n: int = 60) -> np.ndarray:
    """Refit a smoothed cubic spline through ordered ``(x, y)`` points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return pts

    diffs = np.diff(pts, axis=0)
    keep = np.concatenate([[True], np.linalg.norm(diffs, axis=1) > 1e-6])
    pts = pts[keep]
    if len(pts) < 4:
        return pts

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if arclen[-1] <= 0:
        return pts
    u = arclen / arclen[-1]

    smoothing = float(len(pts))
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=smoothing, k=3)
        sample = np.linspace(0.0, 1.0, n)
        xs, ys = splev(sample, tck)
        return np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        return pts


def _build_confidence_map(
    refined_points: np.ndarray,
    scores: np.ndarray,
    image_shape: tuple,
    radius: int = 2,
) -> np.ndarray:
    h, w = image_shape[:2]
    cmap = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(refined_points, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    for i in range(len(pts)):
        x, y = pts[i]
        ix, iy = int(round(float(x))), int(round(float(y)))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(cmap, (ix, iy), int(radius), float(sc[i]), thickness=-1)
    return cmap


def _keep_best_component(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Keep the connected component most overlapping the reference mask.

    Falls back to the largest component if no component overlaps.
    """
    pm = (predicted_mask > 0).astype(np.uint8)
    if pm.sum() == 0:
        return pm

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pm, connectivity=8)
    if n_labels <= 2:
        return pm

    ref = (reference_mask > 0)
    best_label = 0
    best_overlap = -1
    for lbl in range(1, n_labels):
        overlap = int(((labels == lbl) & ref).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = lbl

    if best_overlap <= 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))

    return (labels == best_label).astype(np.uint8)


def track_whisker(
    ref_image: np.ndarray,
    ref_mask: np.ndarray,
    tgt_image: np.ndarray,
    config: dict = CONFIG,
) -> dict:
    """Track a whisker from ``ref_image`` to ``tgt_image``.

    Returns
    -------
    dict
        ``{"predicted_mask": uint8 (H, W),
            "confidence_map": float32 (H, W),
            "refined_points": float32 (N, 2)}``
    """
    n_ctrl = 60
    half_width = 1
    h, w = tgt_image.shape[:2]
    image_shape = (h, w)

    ref_p = preprocess(ref_image)
    tgt_p = preprocess(tgt_image)

    base_skel = skeletonize_mask(ref_mask)
    base_pts = get_control_points(base_skel, n=n_ctrl)

    base_flow = compute_flow(ref_p, tgt_p)
    base_warped_mask = warp_mask((ref_mask > 0).astype(np.uint8), base_flow)

    if config.get("use_augmentation_voting", False):
        n_aug = int(config.get("n_augmentations", 8))
        vote_tally = np.zeros(image_shape, dtype=np.int32)
        all_refined: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        for aug_img, aug_mask in generate_augmentations(ref_image, ref_mask, n=n_aug):
            aug_p = preprocess(aug_img)
            aug_skel = skeletonize_mask(aug_mask)
            aug_pts = get_control_points(aug_skel, n=n_ctrl)
            if len(aug_pts) < 2:
                continue

            flow = compute_flow(aug_p, tgt_p)
            warped_pts = warp_points(aug_pts, flow)
            refined_pts, scores = _refine_with_ncc_impl(aug_p, tgt_p, aug_pts, warped_pts)

            cand_mask = dilate_skeleton_to_mask(refined_pts, half_width, image_shape)
            vote_tally += cand_mask.astype(np.int32)

            all_refined.append(refined_pts)
            all_scores.append(scores)

        threshold = max(1, n_aug // 2)
        predicted_mask = (vote_tally >= threshold).astype(np.uint8)

        if all_refined:
            stack = np.stack(all_refined, axis=0)
            score_stack = np.stack(all_scores, axis=0)
            median_pts = np.median(stack, axis=0).astype(np.float32)
            max_scores = score_stack.max(axis=0).astype(np.float32)
            refined_points = _smooth_refit(median_pts, n=n_ctrl)
            confidence_map = _build_confidence_map(median_pts, max_scores, image_shape)
        else:
            refined_points = np.empty((0, 2), dtype=np.float32)
            confidence_map = np.zeros(image_shape, dtype=np.float32)

        if len(refined_points) >= 2:
            predicted_mask = np.maximum(
                predicted_mask,
                dilate_skeleton_to_mask(refined_points, half_width, image_shape),
            )

    else:
        warped_pts = warp_points(base_pts, base_flow)
        refined_pts, scores = _refine_with_ncc_impl(ref_p, tgt_p, base_pts, warped_pts)
        refined_points = _smooth_refit(refined_pts, n=n_ctrl)

        if len(refined_points) < 2:
            refined_points = refined_pts

        predicted_mask = dilate_skeleton_to_mask(
            refined_points, half_width, image_shape
        )
        confidence_map = _build_confidence_map(refined_pts, scores, image_shape)

    predicted_mask = _keep_best_component(predicted_mask, base_warped_mask)

    return {
        "predicted_mask": predicted_mask.astype(np.uint8),
        "confidence_map": confidence_map.astype(np.float32),
        "refined_points": np.asarray(refined_points, dtype=np.float32),
    }
