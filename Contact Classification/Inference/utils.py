"""
Inference utilities for the whisker-pole contact classifier.

Provides:
    - load_model()              : rebuild EfficientNet-B3 and load checkpoint
    - get_inference_transforms(): Resize + Normalize for inference
    - VideoFrameDataset         : reads frames directly from a video file
    - run_inference()           : classify all frames, return per-frame DataFrame
    - frames_to_intervals()     : collapse consecutive contact frames into (Start, End) ranges
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


# ─────────────────────────── Model loading ───────────────────────────────────

def _build_model_arch(dropout: float = 0.3) -> nn.Module:
    """Reconstruct the EfficientNet-B3 architecture (no pretrained weights)."""
    from torchvision.models import efficientnet_b3

    model = efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features  # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, 1),
    )
    return model


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained checkpoint and return the model in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model_arch()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")
    if "config" in ckpt:
        print(f"  Config: {ckpt['config']}")
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")
    if "val_loss" in ckpt:
        print(f"  Val loss: {ckpt['val_loss']:.4f}")
    return model


# ─────────────────────────── Transforms ──────────────────────────────────────

def get_inference_transforms(img_size: int = 256) -> A.Compose:
    """Same as val transforms: Resize + Normalize."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────── Video Dataset ───────────────────────────────────

class VideoFrameDataset(Dataset):
    """
    Lazily reads frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to the video.
    start_frame : int
        First frame to process (inclusive).
    end_frame : int
        Last frame to process (exclusive).
    transform : albumentations.Compose
        Preprocessing pipeline.
    """

    def __init__(
        self,
        video_path: str,
        start_frame: int = 0,
        end_frame: int | None = None,
        transform: A.Compose | None = None,
    ):
        self.video_path = video_path
        self.transform = transform
        # Lazily opened cv2.VideoCapture handle (per worker process).
        # Each DataLoader worker will get its own dataset copy and thus
        # its own VideoCapture instance. This avoids reopening the video
        # file on every __getitem__ call, which is very slow.
        self._cap = None
        # Track last frame index read for this dataset instance so that when
        # the DataLoader is iterating sequentially (shuffle=False) we can just
        # call cap.read() without an expensive cap.set(...) seek on every item.
        self._last_frame_idx = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.start_frame = max(0, start_frame)
        self.end_frame = min(total, end_frame) if end_frame is not None else total
        self.frame_indices = list(range(self.start_frame, self.end_frame))
        print(f"VideoFrameDataset: frames {self.start_frame}–{self.end_frame - 1} "
              f"({len(self.frame_indices):,} frames)")

    def _get_capture(self) -> cv2.VideoCapture:
        """
        Return an open VideoCapture for this dataset instance.

        In a multi-worker DataLoader, each worker process has its own copy
        of the dataset, so each will lazily open its own VideoCapture once.
        """
        if self._cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {self.video_path}")
            self._cap = cap
        return self._cap

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        frame_num = self.frame_indices[idx]

        cap = self._get_capture()
        # If this is not the immediate next frame after the last one we read
        # in this worker, perform an explicit seek. With shuffle=False and the
        # default sampler, each worker processes a contiguous, increasing chunk
        # of indices, so most of the time this branch is skipped and we just
        # read sequentially from the video file, which is much faster.
        if self._last_frame_idx is None or frame_num != self._last_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        self._last_frame_idx = frame_num

        if not ret:
            # Return a black frame if read fails
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            frame = self.transform(image=frame)["image"]

        return frame, frame_num

    def __del__(self):
        # Ensure VideoCapture is released when the dataset is garbage-collected.
        if getattr(self, "_cap", None) is not None:
            self._cap.release()
            self._cap = None


# ─────────────────────────── Inference runner ────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run inference on all batches and return a DataFrame with columns:
        frame, probability, contact

    Parameters
    ----------
    threshold : float
        Probability threshold for classifying as contact (1).
    """
    model.eval()
    all_frames, all_probs, all_preds = [], [], []

    for images, frame_nums in tqdm(dataloader, desc="Inference"):
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        all_frames.extend(frame_nums.numpy())
        all_probs.extend(probs)
        all_preds.extend(preds)

    df = pd.DataFrame({
        "frame": all_frames,
        "probability": all_probs,
        "contact": all_preds,
    })
    df = df.sort_values("frame").reset_index(drop=True)
    return df


# ─────────────────────────── Interval conversion ────────────────────────────

def frames_to_intervals(df: pd.DataFrame, label_col: str = "contact", label_val: int = 1) -> pd.DataFrame:
    """
    Convert per-frame predictions into contiguous intervals.

    Returns a DataFrame with columns ``Start`` and ``End`` where each row
    represents a run of consecutive frames with ``label_col == label_val``.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    mask = df[label_col] == label_val
    frames = df.loc[mask, "frame"].values

    if len(frames) == 0:
        return pd.DataFrame(columns=["Start", "End"])

    intervals = []
    start = frames[0]
    prev = frames[0]

    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            intervals.append((int(start), int(prev)))
            start = f
            prev = f
    intervals.append((int(start), int(prev)))

    return pd.DataFrame(intervals, columns=["Start", "End"])


# ─────────────────────────── Batch helpers ────────────────────────────────────

import os

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def find_video_folders(root: str, exts: tuple[str, ...] = VIDEO_EXTS) -> dict[str, list[str]]:
    """
    Walk ``root`` and return a mapping:

        folder_path -> [video_path1, video_path2, ...]

    Only folders that contain at least one video are included.
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Root path does not exist: {root}")

    folders: dict[str, list[str]] = {}

    for dirpath, _dirnames, filenames in os.walk(root):
        videos = [
            os.path.join(dirpath, fname)
            for fname in filenames
            if fname.lower().endswith(exts)
        ]
        if videos:
            folders[dirpath] = sorted(videos)

    return folders


def run_video_contact_intervals(
    model: nn.Module,
    device: torch.device,
    video_path: str,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    img_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run contact inference on a single video and return the intervals DataFrame.
    """
    print(f"Creating dataset for {video_path}")

    transform = get_inference_transforms(img_size)
    dataset = VideoFrameDataset(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"Running inference on {len(dataset):,} frames "
        f"({len(dataloader):,} batches) for {os.path.basename(video_path)}"
    )
    results_df = run_inference(model, dataloader, device, threshold=threshold)

    intervals_df = frames_to_intervals(results_df, label_col="contact", label_val=1)
    print(f"Found {len(intervals_df)} contact intervals in {os.path.basename(video_path)}")
    return intervals_df


def batch_contact_intervals(
    model: nn.Module,
    device: torch.device,
    data_root: str,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    img_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    threshold: float = 0.5,
) -> dict[str, list[str]]:
    """
    Run contact-interval inference over all videos under ``data_root``.

    For each folder that contains video(s):
      - If the folder has a single video, writes ``contact_intervals.csv``.
      - If the folder has multiple videos, writes ``<video_stem>_contact_intervals.csv``.

    Returns a mapping:
        folder_path -> [csv_path1, csv_path2, ...]
    """
    folders = find_video_folders(data_root)
    total_videos = sum(len(v) for v in folders.values())

    print(
        f"Discovered {total_videos} videos in {len(folders)} folders "
        f"under {os.path.abspath(data_root)}"
    )

    if not folders:
        print("No videos found. Nothing to do.")
        return {}

    outputs: dict[str, list[str]] = {}

    for folder, videos in sorted(folders.items()):
        print(f"\n=== Folder: {folder} ({len(videos)} video(s)) ===")
        multi_video = len(videos) > 1

        folder_outputs: list[str] = []
        for video_path in videos:
            print(f"\n--- Processing video: {video_path} ---")
            intervals_df = run_video_contact_intervals(
                model,
                device,
                video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                threshold=threshold,
            )

            if multi_video:
                stem = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(folder, f"{stem}_contact_intervals.csv")
            else:
                out_path = os.path.join(folder, "contact_intervals.csv")

            intervals_df.to_csv(out_path, index=False)
            print(f"Saved contact intervals to: {out_path}")
            folder_outputs.append(out_path)

        outputs[folder] = folder_outputs

    return outputs
