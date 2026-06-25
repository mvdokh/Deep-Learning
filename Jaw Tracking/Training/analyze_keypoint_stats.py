"""
Per-condition keypoint statistics for jaw tracking datasets.

Run from Training/::

    python analyze_keypoint_stats.py --pkl ../data/train.pkl
    python analyze_keypoint_stats.py --pkl ../data/val.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

CONDITIONS = ("IRt_BiPoles", "IRt_TeLC", "PCRt_BiPoles")


def analyze(pkl_path: str) -> None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    tip = data["keypoints_tip"]
    line = data["keypoints_line"]
    exp = np.asarray(data["experiment_ids"])
    fn = np.asarray(data["frame_numbers"])

    print(f"\n{'=' * 60}")
    print(f"{pkl_path}  (N={len(tip):,})")
    print(f"{'=' * 60}")

    for eid, name in enumerate(CONDITIONS):
        m = exp == eid
        if m.sum() == 0:
            continue
        t, l = tip[m], line[m]
        frames = fn[m]

        dx = t[:, 0] - l[:, 0]
        dy = t[:, 1] - l[:, 1]
        dist = np.sqrt(dx**2 + dy**2)

        order = np.argsort(frames, kind="mergesort")
        t_sorted = t[order]
        vel = np.linalg.norm(np.diff(t_sorted, axis=0), axis=1)

        print(f"\n{name}  (n={m.sum():,})")
        print(f"  tip  mean x,y = ({t[:, 0].mean():.1f}, {t[:, 1].mean():.1f})")
        print(f"  base mean x,y = ({l[:, 0].mean():.1f}, {l[:, 1].mean():.1f})")
        print(
            f"  tip-base offset: dx={dx.mean():.1f}±{dx.std():.1f}  "
            f"dy={dy.mean():.1f}±{dy.std():.1f}  dist={dist.mean():.1f}±{dist.std():.1f}"
        )
        if len(vel):
            print(
                f"  tip frame-to-frame: mean={vel.mean():.2f}px  "
                f"std={vel.std():.2f}px  p95={np.percentile(vel, 95):.2f}px"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze jaw keypoint pickles")
    p.add_argument("--pkl", type=str, default="../data/train.pkl")
    args = p.parse_args()
    path = Path(args.pkl)
    if not path.is_file():
        raise FileNotFoundError(path)
    analyze(str(path))


if __name__ == "__main__":
    main()
