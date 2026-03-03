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


# ──────────────────────── Multi‑experiment utilities ─────────────────────────

def build_full_dataset_from_root(
    root_dir: str,
    resize: Tuple[int, int] | None = None,
    contact_csv_name: str = "contact_agnostic",
    no_contact_csv_name: str = "no_contact",
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> dict:
    """
    Aggregate frames + labels from multiple experiment folders into one dataset.

    Expected layout under *root_dir*::

        root_dir/
            102725_1/
                contact_agnostic.csv
                no_contact.csv
                <video>.mp4
            102625_1/
                contact_agnostic.csv
                no_contact.csv
                <video>.mp4
            ...

    Parameters
    ----------
    root_dir : str
        Directory containing per‑experiment subfolders.
    resize : tuple (width, height), optional
        Passed to ``extract_frames``.
    contact_csv_name : str
        Base name (without extension) of the contact CSV.
    no_contact_csv_name : str
        Base name (without extension) of the no‑contact CSV.
    video_extensions : tuple of str
        Video file extensions to look for.

    Returns
    -------
    data : dict
        {"frames": np.ndarray, "labels": np.ndarray, "frame_numbers": np.ndarray}
        ready to be saved as a pickle file and consumed by ``ContactDataset``.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root dataset directory does not exist: {root_dir}")

    all_frames: list[np.ndarray] = []
    all_labels: list[int] = []
    all_frame_numbers: list[int] = []

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not exp_dirs:
        raise RuntimeError(f"No experiment subfolders found under: {root_dir}")

    print(f"Found {len(exp_dirs)} experiment folders under {root_dir}")

    for exp_dir in exp_dirs:
        # Locate CSVs
        contact_csv_candidates = list(exp_dir.glob(f"{contact_csv_name}*.csv"))
        no_contact_csv_candidates = list(exp_dir.glob(f"{no_contact_csv_name}*.csv"))

        if not contact_csv_candidates or not no_contact_csv_candidates:
            print(f"⚠  Missing CSVs in {exp_dir.name} — skipping this folder.")
            continue

        contact_csv = str(contact_csv_candidates[0])
        no_contact_csv = str(no_contact_csv_candidates[0])

        # Locate video
        video_files = [
            p for p in exp_dir.iterdir()
            if p.is_file() and p.suffix.lower() in video_extensions
        ]
        if not video_files:
            print(f"⚠  No video file with extensions {video_extensions} in {exp_dir.name} — skipping.")
            continue

        video_path = str(sorted(video_files)[0])

        print(f"\n▶ Processing experiment: {exp_dir.name}")
        print(f"   Video      : {Path(video_path).name}")
        print(f"   Contact CSV: {Path(contact_csv).name}")
        print(f"   No‑contact : {Path(no_contact_csv).name}")

        # Build labels and extract frames for this experiment
        label_df = build_label_dataframe(contact_csv, no_contact_csv)
        frame_numbers = label_df["frame"].values.astype(int)

        images = extract_frames(video_path, frame_numbers, resize=resize)
        if not images:
            print(f"⚠  No frames extracted for {exp_dir.name} — skipping.")
            continue

        # Some requested frames may be missing (beyond video length etc.),
        # so we only keep the intersection between label_df and images.keys().
        available_frames = sorted(set(label_df["frame"]).intersection(images.keys()))
        if not available_frames:
            print(f"⚠  No overlapping frames for {exp_dir.name} — skipping.")
            continue

        label_map = dict(zip(label_df["frame"].values, label_df["label"].values))

        for f in available_frames:
            all_frames.append(images[f])
            all_labels.append(int(label_map[f]))
            all_frame_numbers.append(int(f))

        print(f"   Collected {len(available_frames):,} frames from {exp_dir.name}")

    if not all_frames:
        raise RuntimeError("No frames collected from any experiment. "
                           "Check directory structure and CSV/video names.")

    frames_arr = np.stack(all_frames)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    frame_nums_arr = np.asarray(all_frame_numbers, dtype=np.int64)

    print(f"\n✅ Final aggregated dataset: {len(labels_arr):,} frames")

    return {
        "frames": frames_arr,
        "labels": labels_arr,
        "frame_numbers": frame_nums_arr,
    }


def build_datasets_per_experiment(
    root_dir: str,
    resize: Tuple[int, int] | None = None,
    contact_csv_name: str = "contact_agnostic",
    no_contact_csv_name: str = "no_contact",
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    output_name: str = "contact_dataset.pkl",
) -> dict[str, dict]:
    """
    Build and save one pickle *per experiment folder* under *root_dir*.

    For each experiment directory, this will:
        1. Find ``contact_agnostic*.csv`` and ``no_contact*.csv`` (configurable).
        2. Find a video file with one of *video_extensions*.
        3. Build labels via ``build_label_dataframe``.
        4. Extract the annotated frames via ``extract_frames`` (with optional resize).
        5. Pack into a dict compatible with ``ContactDataset`` and save as
           ``<exp_dir>/<output_name>``.

    Returns
    -------
    summary : dict
        Mapping ``exp_name -> {\"path\", \"n_frames\", \"n_pos\", \"n_neg\"}``.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root dataset directory does not exist: {root_dir}")

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not exp_dirs:
        raise RuntimeError(f"No experiment subfolders found under: {root_dir}")

    print(f"Found {len(exp_dirs)} experiment folders under {root_dir}")

    summary: dict[str, dict] = {}

    for exp_dir in exp_dirs:
        # Skip if pickle already exists
        out_path = exp_dir / output_name
        if out_path.exists():
            print(f"⏭  {exp_dir.name} already has {output_name} — skipping.")
            continue

        # Locate CSVs
        contact_csv_candidates = list(exp_dir.glob(f"{contact_csv_name}*.csv"))
        no_contact_csv_candidates = list(exp_dir.glob(f"{no_contact_csv_name}*.csv"))

        if not contact_csv_candidates or not no_contact_csv_candidates:
            print(f"⚠  Missing CSVs in {exp_dir.name} — skipping this folder.")
            continue

        contact_csv = str(contact_csv_candidates[0])
        no_contact_csv = str(no_contact_csv_candidates[0])

        # Locate video
        video_files = [
            p for p in exp_dir.iterdir()
            if p.is_file() and p.suffix.lower() in video_extensions
        ]
        if not video_files:
            print(f"⚠  No video file with extensions {video_extensions} in {exp_dir.name} — skipping.")
            continue

        video_path = str(sorted(video_files)[0])

        print(f"\n▶ Processing experiment: {exp_dir.name}")
        print(f"   Video      : {Path(video_path).name}")
        print(f"   Contact CSV: {Path(contact_csv).name}")
        print(f"   No‑contact : {Path(no_contact_csv).name}")

        # Build labels and extract frames for this experiment
        label_df = build_label_dataframe(contact_csv, no_contact_csv)
        frame_numbers = label_df["frame"].values.astype(int)

        images = extract_frames(video_path, frame_numbers, resize=resize)
        if not images:
            print(f"⚠  No frames extracted for {exp_dir.name} — skipping.")
            continue

        available_frames = sorted(set(label_df["frame"]).intersection(images.keys()))
        if not available_frames:
            print(f"⚠  No overlapping frames for {exp_dir.name} — skipping.")
            continue

        label_map = dict(zip(label_df["frame"].values, label_df["label"].values))

        frames = np.stack([images[f] for f in available_frames])
        labels = np.asarray([int(label_map[f]) for f in available_frames], dtype=np.int64)
        frame_nums = np.asarray(available_frames, dtype=np.int64)

        data = {
            "frames": frames,
            "labels": labels,
            "frame_numbers": frame_nums,
        }

        out_path = exp_dir / output_name
        save_pkl(data, str(out_path))

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())

        summary[exp_dir.name] = {
            "path": str(out_path),
            "n_frames": int(len(labels)),
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

        print(
            f"   → {out_path.name}: {len(labels):,} frames "
            f"(contact: {n_pos:,}, no-contact: {n_neg:,})"
        )

    if not summary:
        raise RuntimeError(
            "No datasets were created. Check directory structure and CSV/video names."
        )

    print(f"\n✅ Created pickles for {len(summary)} experiments.")
    return summary


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
