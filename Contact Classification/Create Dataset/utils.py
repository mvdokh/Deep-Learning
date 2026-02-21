"""
Utility functions for creating the whisker-pole contact detection dataset.

Workflow:
    1. Parse contact / no-contact CSV files that contain frame ranges.
    2. Build a master label DataFrame (frame_number, label).
    3. Extract the corresponding frames from a video as images.
    4. Bundle images + labels into train / test pickle files (80:20 split).
"""

import cv2
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ──────────────────────────────── CSV parsing ────────────────────────────────

def parse_frame_ranges(csv_path: str) -> list[Tuple[int, int]]:
    """
    Read a CSV with columns ``Start`` and ``End`` and return a list of
    (start, end) tuples (inclusive on both sides).
    """
    df = pd.read_csv(csv_path)
    # Normalise column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    ranges = list(zip(df["start"].astype(int), df["end"].astype(int)))
    return ranges


def expand_ranges(ranges: list[Tuple[int, int]]) -> np.ndarray:
    """Expand a list of (start, end) ranges into a sorted array of unique frame numbers."""
    frames = []
    for s, e in ranges:
        frames.extend(range(s, e + 1))  # inclusive
    return np.unique(np.array(frames))


# ────────────────────────── Label DataFrame builder ──────────────────────────

def build_label_dataframe(
    contact_csv: str,
    no_contact_csv: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with columns ``frame`` and ``label`` from the two CSVs.
    
    - label = 1  →  contact
    - label = 0  →  no contact

    Overlapping frames (if any) are resolved in favour of the **contact** label.
    """
    contact_frames = expand_ranges(parse_frame_ranges(contact_csv))
    no_contact_frames = expand_ranges(parse_frame_ranges(no_contact_csv))

    df_contact = pd.DataFrame({"frame": contact_frames, "label": 1})
    df_no_contact = pd.DataFrame({"frame": no_contact_frames, "label": 0})

    df = pd.concat([df_contact, df_no_contact], ignore_index=True)
    # Drop duplicates – keep contact label when overlap exists
    df = df.sort_values("label", ascending=False).drop_duplicates(subset="frame", keep="first")
    df = df.sort_values("frame").reset_index(drop=True)
    return df


# ────────────────────────── Frame extraction ─────────────────────────────────

def extract_frames(
    video_path: str,
    frame_numbers: np.ndarray,
    resize: Tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Extract specific frames from *video_path* and return them as a dict
    ``{frame_number: image_array}``.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_numbers : array-like
        Sorted array of 0-indexed frame numbers to extract.
    resize : tuple (width, height), optional
        If provided, each frame is resized to this dimension.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_set = set(frame_numbers)
    images: dict[int, np.ndarray] = {}

    # Sort so we can seek forward efficiently
    sorted_frames = np.sort(frame_numbers)
    
    pbar = tqdm(total=len(sorted_frames), desc="Extracting frames")

    for target in sorted_frames:
        if target >= total_frames:
            print(f"⚠  Frame {target} exceeds video length ({total_frames}). Skipping.")
            pbar.update(1)
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠  Could not read frame {target}. Skipping.")
            pbar.update(1)
            continue

        # Convert BGR→RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        images[target] = frame
        pbar.update(1)

    pbar.close()
    cap.release()
    return images


# ─────────────────────────── Train / Test Split ──────────────────────────────

def split_dataset(
    label_df: pd.DataFrame,
    images: dict[int, np.ndarray],
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[dict, dict]:
    """
    Split the labelled frames into train and test sets (stratified by label).

    Returns
    -------
    train_data : dict  {"frames": np.ndarray, "labels": np.ndarray, "frame_numbers": np.ndarray}
    test_data  : dict  {"frames": np.ndarray, "labels": np.ndarray, "frame_numbers": np.ndarray}
    """
    # Only keep rows that were actually extracted
    available = label_df[label_df["frame"].isin(images.keys())].copy()
    available = available.reset_index(drop=True)

    X_idx_train, X_idx_test = train_test_split(
        available.index,
        test_size=test_size,
        random_state=random_state,
        stratify=available["label"],
    )

    def _pack(indices):
        rows = available.loc[indices]
        frame_nums = rows["frame"].values
        labels = rows["label"].values
        frames = np.stack([images[f] for f in frame_nums])
        return {
            "frames": frames,
            "labels": labels,
            "frame_numbers": frame_nums,
        }

    return _pack(X_idx_train), _pack(X_idx_test)


# ──────────────────────────── Pickle I/O ─────────────────────────────────────

def save_pkl(data: dict, path: str) -> None:
    """Save *data* dict to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {path}  ({size_mb:.1f} MB)")


def load_pkl(path: str) -> dict:
    """Load a pickle file and return its dict."""
    with open(path, "rb") as f:
        return pickle.load(f)
