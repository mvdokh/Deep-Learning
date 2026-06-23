"""
Train one shared jaw keypoint model on all conditions.

Batches are shuffled within each condition only (never mixed across conditions).

Run from this directory::

    python train.py
"""

from __future__ import annotations

import argparse
import copy
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    JawKeypointSequenceDataset,
    get_geom_train_transforms,
    get_geom_val_transforms,
)
from losses import JawKeypointLoss
from metrics import compute_keypoint_metrics, heatmaps_to_coords
from model import build_model, count_parameters
from sampler import ExperimentGroupedBatchSampler

CONDITIONS = ("IRt_BiPoles", "IRt_TeLC", "PCRt_BiPoles")


def default_training_config(**overrides: Any) -> Namespace:
    """Default hyperparameters for notebook or programmatic training."""
    data_dir = overrides.get("data_dir", "../data")
    cfg = Namespace(
        data_dir=data_dir,
        train_pkl=str(Path(data_dir) / "train.pkl"),
        val_pkl=str(Path(data_dir) / "val.pkl"),
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
        unfreeze_backbone_epoch=15,
        coord_weight=1.0,
        patience=15,
        seed=42,
        show_progress=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    if "data_dir" in overrides:
        cfg.train_pkl = overrides.get("train_pkl", str(Path(cfg.data_dir) / "train.pkl"))
        cfg.val_pkl = overrides.get("val_pkl", str(Path(cfg.data_dir) / "val.pkl"))
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Jaw keypoint tracking training")
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--train_pkl", type=str, default=None)
    p.add_argument("--val_pkl", type=str, default=None)
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
    p.add_argument("--no_require_consecutive", action="store_true")
    p.add_argument("--no_freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_backbone_epoch", type=int, default=15)
    p.add_argument("--coord_weight", type=float, default=1.0)
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
def evaluate(
    model,
    loader,
    criterion,
    device,
    img_w,
    img_h,
    *,
    show_progress: bool = True,
    epoch: int | None = None,
):
    model.eval()
    loss_sum = 0.0
    loss_hm_sum = 0.0
    loss_coord_sum = 0.0
    n = 0
    metric_acc: dict[str, list[float]] = {
        "rmse_tip": [],
        "rmse_line": [],
        "rmse_mean": [],
        "pck_tip": [],
        "pck_line": [],
        "pck_mean": [],
    }

    desc = f"Epoch {epoch} val" if epoch is not None else "Val"
    batch_iter = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        leave=True,
        disable=not show_progress,
    )
    for x, hm, kps_orig, kps_tgt, _, _ in batch_iter:
        x = x.to(device)
        hm = hm.to(device)
        kps_orig = kps_orig.to(device)
        kps_tgt = kps_tgt.to(device)

        pred = model(x)
        loss, parts = criterion(pred, hm, kps_tgt)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        loss_hm_sum += parts["loss_hm"] * bs
        loss_coord_sum += parts["loss_coord"] * bs
        n += bs

        pred_coords = heatmaps_to_coords(pred)
        m = compute_keypoint_metrics(
            pred_coords, kps_orig, img_w=img_w, img_h=img_h
        )
        for k, v in m.items():
            metric_acc[k].append(v)

        batch_iter.set_postfix(
            loss=f"{loss.item():.3f}",
            coord=f"{parts['loss_coord']:.3f}",
            refresh=False,
        )

    out = {k: float(np.mean(v)) for k, v in metric_acc.items()}
    out["loss"] = loss_sum / max(n, 1)
    out["loss_hm"] = loss_hm_sum / max(n, 1)
    out["loss_coord"] = loss_coord_sum / max(n, 1)
    return out


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    show_progress: bool = True,
    epoch: int | None = None,
):
    model.train()
    loss_sum = 0.0
    n = 0

    desc = f"Epoch {epoch} train" if epoch is not None else "Train"
    batch_iter = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        leave=True,
        disable=not show_progress,
    )
    for x, hm, _, kps_tgt, _, _ in batch_iter:
        x = x.to(device)
        hm = hm.to(device)
        kps_tgt = kps_tgt.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss, parts = criterion(pred, hm, kps_tgt)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        batch_iter.set_postfix(
            loss=f"{loss.item():.3f}",
            coord=f"{parts['loss_coord']:.3f}",
            refresh=False,
        )

    return loss_sum / max(n, 1)


def run_training(args: Namespace) -> dict:
    """Train one shared model on merged train/val pickles."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Training shared model on all conditions")
    print(f"Train: {args.train_pkl}")
    print(f"Val:   {args.val_pkl}")
    print(f"{'='*60}")

    train_geom = get_geom_train_transforms(args.img_h, args.img_w)
    val_geom = get_geom_val_transforms(args.img_h, args.img_w)

    train_ds = JawKeypointSequenceDataset(
        args.train_pkl,
        transform=train_geom,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
        img_h=args.img_h,
        img_w=args.img_w,
    )
    val_ds = JawKeypointSequenceDataset(
        args.val_pkl,
        transform=val_geom,
        window_size=args.window_size,
        edge_mode=args.edge_mode,
        require_consecutive_frames=not args.no_require_consecutive,
        img_h=args.img_h,
        img_w=args.img_w,
    )

    train_sampler = ExperimentGroupedBatchSampler(
        train_ds.center_indices_for_sampler,
        batch_size=args.batch_size,
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

    criterion = JawKeypointLoss(
        coord_weight=args.coord_weight,
        coord_scale=(args.img_w + args.img_h) / 2.0,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_pck_mean": [],
        "val_rmse_mean": [],
    }

    show_progress = getattr(args, "show_progress", True)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        epoch_no = epoch + 1

        if args.unfreeze_backbone_epoch > 0 and epoch_no == args.unfreeze_backbone_epoch:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            if show_progress:
                tqdm.write(f"Epoch {epoch_no}: backbone unfrozen")

        tr_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=show_progress,
            epoch=epoch_no,
        )
        val_m = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args.img_w,
            args.img_h,
            show_progress=show_progress,
            epoch=epoch_no,
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_pck_mean"].append(val_m["pck_mean"])
        history["val_rmse_mean"].append(val_m["rmse_mean"])

        if val_m["rmse_mean"] < best_val_rmse - 1e-6:
            best_val_rmse = val_m["rmse_mean"]
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
            summary = (
                f"Epoch {epoch_no}/{args.epochs}  train_loss={tr_loss:.4f}  "
                f"val_loss={val_m['loss']:.4f}  val_coord={val_m['loss_coord']:.4f}  "
                f"val_rmse={val_m['rmse_mean']:.1f}px  "
                f"best_rmse={best_val_rmse:.1f}px  *saved*"
            )
        else:
            patience_counter += 1
            summary = (
                f"Epoch {epoch_no}/{args.epochs}  train_loss={tr_loss:.4f}  "
                f"val_loss={val_m['loss']:.4f}  val_coord={val_m['loss_coord']:.4f}  "
                f"val_rmse={val_m['rmse_mean']:.1f}px  "
                f"best_rmse={best_val_rmse:.1f}px"
            )
            if patience_counter >= args.patience:
                if show_progress:
                    tqdm.write(summary)
                    tqdm.write("Early stopping.")
                break

        if show_progress:
            tqdm.write(summary)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_path = os.path.join(args.out_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, final_path)
    print(f"Done. Final weights → {final_path}")

    return {
        "model": model,
        "history": history,
        "best_val_rmse": best_val_rmse,
        "out_dir": args.out_dir,
        "device": device,
        "config": args,
    }


def args_from_cli(cli: argparse.Namespace) -> Namespace:
    data_dir = cli.data_dir
    return Namespace(
        data_dir=data_dir,
        train_pkl=cli.train_pkl or str(Path(data_dir) / "train.pkl"),
        val_pkl=cli.val_pkl or str(Path(data_dir) / "val.pkl"),
        out_dir=cli.out_dir,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        lr=cli.lr,
        weight_decay=cli.weight_decay,
        num_workers=cli.num_workers,
        img_h=cli.img_h,
        img_w=cli.img_w,
        window_size=cli.window_size,
        temporal_hidden=cli.temporal_hidden,
        temporal_layers=cli.temporal_layers,
        decoder_hidden=cli.decoder_hidden,
        edge_mode=cli.edge_mode,
        no_require_consecutive=cli.no_require_consecutive,
        no_freeze_backbone=cli.no_freeze_backbone,
        unfreeze_backbone_epoch=cli.unfreeze_backbone_epoch,
        coord_weight=cli.coord_weight,
        patience=cli.patience,
        seed=cli.seed,
    )


def main() -> None:
    run_training(args_from_cli(parse_args()))


if __name__ == "__main__":
    main()
