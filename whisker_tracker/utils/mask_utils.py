"""Mask construction, skeletonization, and centerline utilities."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from skimage.morphology import skeletonize


def points_to_mask(
    points: np.ndarray,
    image_shape: tuple,
    line_width: int = 3,
) -> np.ndarray:
    """Rasterize an ordered polyline of ``(x, y)`` points into a binary mask.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2)`` array of ordered ``(x, y)`` polyline vertices.
    image_shape : tuple
        Target shape ``(H, W)`` or ``(H, W, C)``; only the first two dims are
        used.
    line_width : int
        Stroke width in pixels (3 is the canonical whisker width here).

    Returns
    -------
    np.ndarray
        ``uint8`` binary mask (values 0 or 1) of shape ``(H, W)``.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if points is None or len(points) < 2:
        return mask

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts_int = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        mask,
        [pts_int],
        isClosed=False,
        color=1,
        thickness=int(max(1, line_width)),
        lineType=cv2.LINE_8,
    )
    return mask


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thin a binary mask to a 1-pixel centerline."""
    return skeletonize(mask > 0).astype(np.uint8)


def _order_skeleton(skel: np.ndarray) -> np.ndarray:
    """Return skeleton pixels ordered along the longest path (in pixel coords).

    Builds an 8-connected graph over skeleton pixels, then takes the diameter
    of that graph via two BFS passes. Naturally drops short branch spurs.
    Output is float32 ``(M, 2)`` in ``(x, y)`` order.
    """
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(ys) == 1:
        return np.array([[xs[0], ys[0]]], dtype=np.float32)

    coords = np.column_stack([ys, xs])
    coord_index = {(int(y), int(x)): i for i, (y, x) in enumerate(coords)}

    n = len(coords)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = coord_index.get((int(y) + dy, int(x) + dx))
                if j is not None:
                    neighbors[i].append(j)

    def bfs_farthest(src: int) -> tuple[int, list[int]]:
        dist = [-1] * n
        parent = [-1] * n
        dist[src] = 0
        q = deque([src])
        far = src
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] > dist[far]:
                        far = v
                    q.append(v)
        return far, parent

    degrees = np.array([len(nb) for nb in neighbors])
    endpoint_candidates = np.where(degrees == 1)[0]
    start = int(endpoint_candidates[0]) if len(endpoint_candidates) > 0 else int(np.argmin(degrees))

    far1, _ = bfs_farthest(start)
    far2, parent = bfs_farthest(far1)

    path: list[int] = []
    u = far2
    while u != -1:
        path.append(u)
        u = parent[u]
    path.reverse()

    ordered_yx = coords[path]
    ordered_xy = np.column_stack([ordered_yx[:, 1], ordered_yx[:, 0]]).astype(np.float32)
    return ordered_xy


def get_control_points(skeleton: np.ndarray, n: int = 60) -> np.ndarray:
    """Fit a cubic spline to a skeleton and return ``n`` evenly-spaced points.

    The points are evenly spaced in arc-length parameterization along the
    fitted curve.
    """
    ordered = _order_skeleton(skeleton)
    m = len(ordered)
    if m == 0:
        return np.empty((0, 2), dtype=np.float32)
    if m == 1:
        return np.repeat(ordered, n, axis=0).astype(np.float32)

    diffs = np.diff(ordered, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    keep_mask = np.concatenate([[True], seg_lens > 1e-6])
    ordered = ordered[keep_mask]
    m = len(ordered)
    if m < 2:
        return np.repeat(ordered, n, axis=0).astype(np.float32)

    diffs = np.diff(ordered, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(arclen[-1])
    if total <= 0:
        return np.repeat(ordered[:1], n, axis=0).astype(np.float32)
    u_params = arclen / total

    k = min(3, m - 1)
    if k < 1:
        return np.repeat(ordered[:1], n, axis=0).astype(np.float32)

    try:
        tck, _ = splprep([ordered[:, 0], ordered[:, 1]], u=u_params, s=0, k=k)
        sample_u = np.linspace(0.0, 1.0, n)
        xs, ys = splev(sample_u, tck)
        return np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        idx = np.linspace(0, m - 1, n)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, m - 1)
        t = (idx - i0)[:, None]
        return ((1.0 - t) * ordered[i0] + t * ordered[i1]).astype(np.float32)


def dilate_skeleton_to_mask(
    control_points: np.ndarray,
    half_width: int,
    image_shape: tuple,
) -> np.ndarray:
    """Re-rasterize control points back to a binary mask at the given width."""
    thickness = int(max(1, 2 * int(half_width) + 1))
    return points_to_mask(control_points, image_shape, line_width=thickness)
