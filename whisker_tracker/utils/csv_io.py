"""CSV parsing for whisker label files.

Each CSV file represents exactly ONE whisker line.
The CSV has NO header. Each row contains two values: x and y (one point per row).
"""

from __future__ import annotations

import re

import numpy as np

_SPLIT_RE = re.compile(r"[,;\s]+")


def load_whisker_csv(csv_path: str) -> np.ndarray:
    """Parse a whisker CSV file.

    Each row: ``x, y`` (one point per row, no header).

    Tolerates trailing whitespace, blank lines, and inconsistent delimiters
    (comma, semicolon, or whitespace). Rows that fail to parse or contain
    non-finite values are silently dropped.

    Parameters
    ----------
    csv_path : str
        Path to the whisker CSV file.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(N, 2)`` with rows ``[x, y]``.
    """
    pts: list[tuple[float, float]] = []

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue

            tokens = [t for t in _SPLIT_RE.split(line) if t]
            if len(tokens) < 2:
                continue

            try:
                x = float(tokens[0])
                y = float(tokens[1])
            except ValueError:
                continue

            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            pts.append((x, y))

    if not pts:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)
