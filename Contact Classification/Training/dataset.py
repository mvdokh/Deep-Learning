"""
PyTorch Dataset for whisker-pole contact classification.

Loads train/test data from pickle files produced by the Create Dataset notebook.
Each pkl contains:
    - "frames"        : np.ndarray  (N, H, W, 3) uint8 RGB images
    - "labels"        : np.ndarray  (N,) int  0 or 1
    - "frame_numbers" : np.ndarray  (N,) int  original video frame indices
"""

import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────── Augmentation pipelines ──────────────────────────

def get_train_transforms(img_size: int = 256) -> A.Compose:
    """Augmentation pipeline for training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.1], contrast_limit=[-0.1, 0.1], p=0.2),
        A.RandomGravel(p=0.2),
        A.SaltAndPepper(amount=(0.01, 0.04), salt_vs_pepper=(0.4, 0.6), p=0.2),
        A.ZoomBlur(max_factor=(1, 1.0), step_factor=(0.01, 0.07), p=0.2),
        A.RandomRotate90(p=0.2),
        A.Sharpen(alpha=[0.2, 0.5], lightness=[0.5, 1], method="kernel", kernel_size=5, sigma=1, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """Minimal transforms for validation / inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────── Dataset class ───────────────────────────────────

class ContactDataset(Dataset):
    """
    Parameters
    ----------
    pkl_path : str
        Path to the pickle file (train.pkl or test.pkl).
    transform : albumentations.Compose, optional
        Augmentation pipeline applied to each image.
    """

    def __init__(self, pkl_path: str, transform: A.Compose | None = None):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.frames: np.ndarray = data["frames"]          # (N, H, W, 3) uint8
        self.labels: np.ndarray = data["labels"].astype(np.float32)  # (N,)
        self.frame_numbers: np.ndarray = data["frame_numbers"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.frames[idx]  # H, W, 3  uint8
        label = self.labels[idx]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]  # C, H, W  float32 tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label, dtype=torch.float32)
        return image, label
