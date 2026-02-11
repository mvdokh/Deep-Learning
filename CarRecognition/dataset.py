"""
PyTorch Dataset & DataLoader utilities for the car recognition task.

The processed/ folder is expected to be in ImageFolder layout:
    processed/
      train/<class_label>/img1.jpg ...
      val/<class_label>/img1.jpg ...
      test/<class_label>/img1.jpg ...
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image

import config


# ──────────────────────────────────────────────
# Albumentations transforms (better augmentation)
# ──────────────────────────────────────────────

train_transform = A.Compose([
    A.RandomResizedCrop(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

eval_transform = A.Compose([
    A.Resize(height=config.IMAGE_SIZE + 32, width=config.IMAGE_SIZE + 32),
    A.CenterCrop(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# ──────────────────────────────────────────────
# Albumentations-compatible ImageFolder wrapper
# ──────────────────────────────────────────────

class AlbumentationsDataset(torch.utils.data.Dataset):
    """Wraps a torchvision ImageFolder so we can apply Albumentations."""

    def __init__(self, root: str, transform: A.Compose):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform
        # expose class info
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image = cv2.imread(path)
        if image is None:
            # fallback to PIL
            image = np.array(Image.open(path).convert("RGB"))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        return augmented["image"], label


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def get_dataloaders(batch_size: int | None = None, num_workers: int | None = None):
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    train_ds = AlbumentationsDataset(str(config.PROCESSED_DIR / "train"), train_transform)
    val_ds   = AlbumentationsDataset(str(config.PROCESSED_DIR / "val"),   eval_transform)
    test_ds  = AlbumentationsDataset(str(config.PROCESSED_DIR / "test"),  eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names
