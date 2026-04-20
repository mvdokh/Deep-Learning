"""
Train the temporal contact classifier (BCEWithLogitsLoss, optional frozen backbone, t-SNE callback).

Run from this directory::

    python train.py --train_pkl /path/train.pkl --val_pkl /path/val.pkl --out_dir ./runs/exp1
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from callbacks import TSNECallback
from dataset import TemporalContactSequenceDataset, get_train_transforms, get_val_transforms
from metrics import compute_binary_metrics, confusion_metrics
from model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal contact classification training")
    p.add_argument("--train_pkl", type=str, required=True)
    p.add_argument("--val_pkl", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./training_output")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument(
        "--no_freeze_backbone",
        action="store_true",
        help="Train the EfficientNet backbone from the first epoch (default: backbone frozen)",
    )
    p.add_argument(
        "--unfreeze_backbone_epoch",
        type=int,
        default=0,
        help="1-based epoch index to enable backbone fine-tuning (0 = never)",
    )
    p.add_argument("--temporal_hidden", type=int, default=256)
    p.add_argument("--temporal_layers", type=int, default=3)
    p.add_argument("--edge_mode", type=str, default="pad", choices=["pad", "skip"])
    p.add_argument(
        "--no_require_consecutive",
        action="store_true",
        help="Allow windows where frame_numbers are not strictly +1 (e.g. padded edges with repeated frames)",
    )
    p.add_argument("--tsne_every", type=int, default=5, help="0 disables t-SNE callback")
    p.add_argument("--tsne_max_samples", type=int, default=2000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    loss_sum = 0.0
    n = 0
    all_preds: list[int] = []
    all_labels: list[float] = []
    all_probs: list[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(np.int64)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    labels = np.array(all_labels, dtype=np.int64)
    preds = np.array(all_preds, dtype=np.int64)
    probs = np.array(all_probs, dtype=np.float32)
    metrics = compute_binary_metrics(labels, preds, probs=probs)
    metrics.update(confusion_metrics(labels, preds))
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_tf = get_train_transforms(args.img_size)
    val_tf = get_val_transforms(args.img_size)

    train_ds = TemporalContactSequenceDataset(
        args.train_pkl,
        transform=train_tf,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
    )
    val_ds = TemporalContactSequenceDataset(
        args.val_pkl,
        transform=val_tf,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    freeze_bb = not args.no_freeze_backbone
    model = build_model(
        pretrained=True,
        freeze_backbone=freeze_bb,
        window_size=args.window_size,
        temporal_hidden=args.temporal_hidden,
        temporal_layers=args.temporal_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tsne_cb = TSNECallback(
        val_loader,
        device,
        out_dir=os.path.join(args.out_dir, "tsne"),
        every_n_epochs=args.tsne_every,
        max_samples=args.tsne_max_samples,
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
    }

    for epoch in range(args.epochs):
        if args.unfreeze_backbone_epoch > 0 and epoch + 1 == args.unfreeze_backbone_epoch:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            print(f"Epoch {epoch+1}: backbone unfrozen, optimizer re-initialized (lr={args.lr * 0.1})")

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_f1"].append(val_m["f1"])
        history["val_accuracy"].append(val_m["accuracy"])

        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={val_m['loss']:.4f}  "
            f"val_f1={val_m['f1']:.4f}  val_acc={val_m['accuracy']:.4f}  "
            f"roc_auc={val_m.get('roc_auc', float('nan')):.4f}"
        )

        tsne_cb(model, epoch)

        if val_m["loss"] < best_val - 1e-6:
            best_val = val_m["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            ckpt_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "val_metrics": val_m,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"  saved best checkpoint → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, final_path)
    print(f"Done. Final weights → {final_path}")


if __name__ == "__main__":
    main()
