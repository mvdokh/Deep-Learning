"""
Training and evaluation utilities for the contact classifier.

Provides:
    - build_model()       : EfficientNet-B3 backbone with binary classification head
    - train_one_epoch()   : single epoch training loop
    - evaluate()          : validation / test evaluation
    - EarlyStopping       : patience-based early stopping helper
    - plot_history()      : loss / accuracy curves
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from tqdm import tqdm


# ─────────────────────────── Model builder ───────────────────────────────────

def build_model(
    num_classes: int = 1,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build an EfficientNet-B3 (~12 M backbone params) for binary classification.

    The final classifier is replaced with:
        Dropout → Linear(1536, 512) → ReLU → Dropout → Linear(512, 1)

    Total ≈ 13–14 M parameters.

    Parameters
    ----------
    num_classes : int
        Use 1 for binary (BCE loss).  Use 2+ for multi-class (CE loss).
    pretrained : bool
        Load ImageNet-pretrained weights.
    dropout : float
        Dropout probability in the classification head.
    freeze_backbone : bool
        If True, freeze all backbone (feature extractor) weights.
    """
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1536 for B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(512, num_classes),
    )

    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ─────────────────────────── Training loop ───────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> dict:
    """
    Train for one epoch.  Returns dict with 'loss' and 'accuracy'.
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)       # (B,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return {"loss": epoch_loss, "accuracy": epoch_acc}


# ─────────────────────────── Evaluation ──────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model.  Returns dict with 'loss', 'accuracy',
    'precision', 'recall', 'f1', 'preds', 'labels'.
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    return {
        "loss": epoch_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "preds": all_preds,
        "labels": all_labels,
        "probs": np.array(all_probs),
    }


# ─────────────────────────── Early Stopping ──────────────────────────────────

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'min' (for loss) or 'max' (for accuracy / f1).
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Returns True if the score improved (checkpoint was updated).
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return True

        improved = (
            (score < self.best_score - self.min_delta) if self.mode == "min"
            else (score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ─────────────────────────── Checkpoint I/O ──────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    history: dict,
    config: dict,
    save_path: str,
) -> None:
    """Save a training checkpoint to disk."""
    import os
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "history": history,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  💾 Checkpoint saved → {save_path}  ({size_mb:.1f} MB)")


# ─────────────────────────── Plotting ────────────────────────────────────────

def plot_history(history: dict, save_path: str | None = None):
    """
    Plot training & validation loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc' (lists).
    save_path : str, optional
        Save figure to this path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    plt.show()


def plot_confusion_matrix(labels, preds, save_path: str | None = None):
    """Display a confusion matrix."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Contact", "Contact"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_classification_report(labels, preds):
    """Print sklearn classification report."""
    print(classification_report(
        labels, preds,
        target_names=["No Contact", "Contact"],
        digits=4,
    ))
