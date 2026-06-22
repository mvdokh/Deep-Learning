"""
Train jaw keypoint heatmap model.

Run from this directory::

    python train.py --train_pkl ../data/train.pkl --val_pkl ../data/val.pkl --out_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import copy
import os
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import JawKeypointSequenceDataset, get_train_transforms, get_val_transforms
from losses import BalancedKeypointHeatmapLoss
from metrics import compute_keypoint_metrics, heatmaps_to_coords
from model import build_model, count_parameters
from sampler import ExperimentGroupedBatchSampler


def default_training_config(**overrides: Any) -> Namespace:
    """Default hyperparameters for notebook or programmatic training."""
    cfg = Namespace(
        train_pkl="../data/train.pkl",
        val_pkl="../data/val.pkl",
        out_dir="./checkpoints",
        epochs=80,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        num_workers=4,
        img_h=240,
        img_w=320,
        window_size=8,
        temporal_hidden=384,
        temporal_layers=3,
        decoder_hidden=512,
        edge_mode="pad",
        no_require_consecutive=False,
        no_freeze_backbone=False,
        unfreeze_backbone_epoch=0,
        patience=15,
        seed=42,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jaw keypoint tracking training")
    p.add_argument("--train_pkl", type=str, required=True)
    p.add_argument("--val_pkl", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--img_h", type=int, default=240)
    p.add_argument("--img_w", type=int, default=320)
    p.add_argument("--window_size", type=int, default=8)
    p.add_argument("--temporal_hidden", type=int, default=384)
    p.add_argument("--temporal_layers", type=int, default=3)
    p.add_argument("--decoder_hidden", type=int, default=512)
    p.add_argument("--edge_mode", type=str, default="pad", choices=["pad", "skip"])
    p.add_argument(
        "--no_require_consecutive",
        action="store_true",
        help="Allow windows where frame_numbers are not strictly consecutive",
    )
    p.add_argument(
        "--no_freeze_backbone",
        action="store_true",
        help="Train backbone from epoch 1 (default: frozen initially)",
    )
    p.add_argument("--unfreeze_backbone_epoch", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    xs, hms, kps_orig, kps_tgt, exp_ids, frames = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(hms, dim=0),
        torch.stack(kps_orig, dim=0),
        torch.stack(kps_tgt, dim=0),
        torch.stack(exp_ids, dim=0),
        torch.stack(frames, dim=0),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, img_w, img_h):
    model.eval()
    loss_sum = 0.0
    loss_tip_sum = 0.0
    loss_line_sum = 0.0
    n = 0
    metric_acc: dict[str, list[float]] = {
        "rmse_tip": [],
        "rmse_line": [],
        "rmse_mean": [],
        "pck_tip": [],
        "pck_line": [],
        "pck_mean": [],
    }

    for x, hm, kps_orig, _, exp_ids, _ in loader:
        x = x.to(device)
        hm = hm.to(device)
        kps_orig = kps_orig.to(device)
        exp_ids = exp_ids.to(device)

        pred = model(x, exp_ids)
        loss, parts = criterion(pred, hm)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        loss_tip_sum += parts["loss_tip"] * bs
        loss_line_sum += parts["loss_line"] * bs
        n += bs

        pred_coords = heatmaps_to_coords(pred)
        m = compute_keypoint_metrics(
            pred_coords, kps_orig, img_w=img_w, img_h=img_h
        )
        for k, v in m.items():
            metric_acc[k].append(v)

    out = {k: float(np.mean(v)) for k, v in metric_acc.items()}
    out["loss"] = loss_sum / max(n, 1)
    out["loss_tip"] = loss_tip_sum / max(n, 1)
    out["loss_line"] = loss_line_sum / max(n, 1)
    return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    n = 0
    for x, hm, _, _, exp_ids, _ in loader:
        x = x.to(device)
        hm = hm.to(device)
        exp_ids = exp_ids.to(device)
        optimizer.zero_grad()
        pred = model(x, exp_ids)
        loss, _ = criterion(pred, hm)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)


def run_training(args: Namespace) -> dict:
    """
    Full training loop (used by CLI and :mod:`train.ipynb`).

    Returns
    -------
    dict
        ``model``, ``history``, ``best_val_loss``, ``out_dir``, ``device``, ``config``
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    train_tf = get_train_transforms(args.img_h, args.img_w)
    val_tf = get_val_transforms(args.img_h, args.img_w)

    train_ds = JawKeypointSequenceDataset(
        args.train_pkl,
        transform=train_tf,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
        img_h=args.img_h,
        img_w=args.img_w,
    )
    val_ds = JawKeypointSequenceDataset(
        args.val_pkl,
        transform=val_tf,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
        img_h=args.img_h,
        img_w=args.img_w,
    )

    train_sampler = ExperimentGroupedBatchSampler(
        train_ds.center_indices_for_sampler,
        args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    freeze_bb = not args.no_freeze_backbone
    model = build_model(
        pretrained=True,
        freeze_backbone=freeze_bb,
        window_size=args.window_size,
        temporal_hidden=args.temporal_hidden,
        temporal_layers=args.temporal_layers,
        decoder_hidden=args.decoder_hidden,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    criterion = BalancedKeypointHeatmapLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_pck_mean": [],
        "val_rmse_mean": [],
    }

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        if args.unfreeze_backbone_epoch > 0 and epoch + 1 == args.unfreeze_backbone_epoch:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            print(f"Epoch {epoch+1}: backbone unfrozen")

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(
            model, val_loader, criterion, device, args.img_w, args.img_h
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_pck_mean"].append(val_m["pck_mean"])
        history["val_rmse_mean"].append(val_m["rmse_mean"])

        print(
            f"Epoch {epoch+1}/{args.epochs}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_m['loss']:.4f}  "
            f"val_pck_mean={val_m['pck_mean']:.4f}  "
            f"val_rmse_mean={val_m['rmse_mean']:.2f}px"
        )

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

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val,
        "out_dir": args.out_dir,
        "device": device,
        "config": args,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
