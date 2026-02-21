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

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        frame_num = self.frame_indices[idx]

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            # Return a black frame if read fails
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            frame = self.transform(image=frame)["image"]

        return frame, frame_num


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
