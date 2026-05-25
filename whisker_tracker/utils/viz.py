"""Visualization and output saving for whisker predictions."""

from __future__ import annotations

import os

import cv2
import numpy as np


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Return a BGR image with ``mask`` overlaid as a semi-transparent fill.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale ``(H, W)`` or BGR ``(H, W, 3)``).
    mask : np.ndarray
        Binary mask ``(H, W)``; non-zero pixels are colored.
    color : tuple
        BGR color tuple (``(0, 255, 0)`` is green).
    alpha : float
        Blend weight in ``[0, 1]``; higher values produce a more opaque fill.

    Returns
    -------
    np.ndarray
        ``uint8`` BGR image of the same ``(H, W)`` as ``image``.
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        canvas = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        canvas = image.copy()
    else:
        raise ValueError(f"Unsupported image shape {image.shape}")

    if canvas.dtype != np.uint8:
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    out = canvas.copy()
    sel = mask > 0
    if not np.any(sel):
        return out

    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
    a = float(np.clip(alpha, 0.0, 1.0))
    blended = ((1.0 - a) * canvas[sel].astype(np.float32) + a * color_arr).astype(np.uint8)
    out[sel] = blended
    return out


def save_prediction(image: np.ndarray, mask: np.ndarray, output_path: str) -> str:
    """Write the green-overlay prediction image to ``output_path``.

    The output directory is created if missing. The output resolution matches
    the input image exactly.

    Returns
    -------
    str
        The absolute output path actually written.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    overlay = overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5)
    ok = cv2.imwrite(output_path, overlay)
    if not ok:
        raise IOError(f"cv2.imwrite failed for path: {output_path}")
    return os.path.abspath(output_path)
