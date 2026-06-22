"""
Build per-condition and merged pickle datasets for jaw keypoint tracking.

Run from this directory::

    python create_dataset.py --data_root /mnt/c/Users/wanglab/Desktop/Tip+Base
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    CONDITIONS,
    CONDITION_TO_ID,
    merge_pickles,
    save_pkl,
    split_train_val,
    load_aligned_condition,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create jaw tracking pickle datasets")
    p.add_argument(
        "--data_root",
        type=str,
        default="/mnt/c/Users/wanglab/Desktop/Tip+Base",
        help="Root folder containing IRt_BiPoles, IRt_TeLC, PCRt_BiPoles",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data"),
    )
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_paths: list[str] = []
    for name in CONDITIONS:
        cond_dir = data_root / name
        if not cond_dir.is_dir():
            raise FileNotFoundError(f"Missing condition folder: {cond_dir}")
        print(f"\n▶ Processing {name}")
        data = load_aligned_condition(cond_dir, name, CONDITION_TO_ID[name])
        out_path = out_dir / f"{name}.pkl"
        save_pkl(data, str(out_path))
        pkl_paths.append(str(out_path))

    merged_path = out_dir / "merged.pkl"
    merged = merge_pickles(pkl_paths, str(merged_path))

    train_data, val_data = split_train_val(merged, val_fraction=args.val_fraction, seed=args.seed)
    save_pkl(train_data, str(out_dir / "train.pkl"))
    save_pkl(val_data, str(out_dir / "val.pkl"))

    print(f"\n✅ Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
