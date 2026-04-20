"""
CLI: sliding-window contact prediction on a video file.

Example::

    python inference.py --checkpoint ../Training/training_output/best_model.pt --video /path/video.mp4 --out_csv results.csv
"""

from __future__ import annotations

import argparse
import os

import torch

from utils import load_temporal_model, run_sliding_window_on_video_file


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal contact sliding-window inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--out_csv", type=str, default="temporal_contact_predictions.csv")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_temporal_model(args.checkpoint, device)
    df = run_sliding_window_on_video_file(
        model,
        args.video,
        device,
        img_size=args.img_size,
        batch_size=args.batch_size,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
