"""
Training script for the car recognition model.

Usage
-----
    python train.py                         # train with defaults from config.py
    python train.py --epochs 50 --lr 3e-4   # override hyper-parameters
    python train.py --resume checkpoints/best_model.pth
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model, unfreeze_backbone


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="  train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs
        pbar.set_postfix(loss=f"{running_loss/n:.4f}", acc=f"{running_acc/n:.4f}")

    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        n += bs
    return running_loss / n, running_acc / n


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        batch_size=args.batch_size,
    )

    # Model
    model = build_model()
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    model = model.to(device)

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif config.SCHEDULER == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # Logging
    writer = SummaryWriter(log_dir=str(config.LOG_DIR))
    best_val_acc = 0.0
    patience_counter = 0

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Training
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Unfreeze backbone after N epochs
        if config.FREEZE_BACKBONE and epoch == config.UNFREEZE_AFTER_EPOCHS:
            unfreeze_backbone(model)
            # Reset optimizer to include all params with a lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=config.WEIGHT_DECAY,
            )
            if config.SCHEDULER == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  |  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  |  "
            f"lr={lr:.2e}  |  {elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", lr, epoch)

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch + 1, val_acc,
                            config.CHECKPOINT_DIR / "best_model.pth")
            print(f"  >>> New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # Save latest every epoch
        save_checkpoint(model, optimizer, epoch + 1, val_acc,
                        config.CHECKPOINT_DIR / "latest_model.pth")

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    writer.close()

    # Final test evaluation
    print("\n--- Test evaluation (best model) ---")
    model = build_model(pretrained=False, freeze_backbone=False).to(device)
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    _per_class_accuracy(model, test_loader, class_names, device)


@torch.no_grad()
def _per_class_accuracy(model, loader, class_names, device):
    model.eval()
    correct = torch.zeros(len(class_names))
    total = torch.zeros(len(class_names))
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for c in range(len(class_names)):
            mask = labels == c
            total[c] += mask.sum().item()
            correct[c] += (preds[mask] == c).sum().item()
    for c, name in enumerate(class_names):
        acc = correct[c] / total[c] if total[c] > 0 else 0
        print(f"  {name:35s}  {acc:.4f}  ({int(correct[c])}/{int(total[c])})")


if __name__ == "__main__":
    main()
