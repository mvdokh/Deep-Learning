import sys, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

INFERENCE_DIR = os.path.dirname(os.path.abspath("__file__"))
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from utils import (
    load_model,
    get_inference_transforms,
    VideoFrameDataset,
    run_inference,
    frames_to_intervals,
)

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========================  PATHS  ========================

VIDEO_PATH = "/home/mvdokh/data/b064_0913_1/output.mp4"


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(INFERENCE_DIR),
    "Training", "checkpoints", "contact_classifier.pt"
)

# Output CSVs will be saved here
OUTPUT_DIR = "/home/mvdokh/data/b064_0913_1/"
#OUTPUT_DIR = os.path.join(INFERENCE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================  SETTINGS  ========================

IMG_SIZE      = 256
BATCH_SIZE    = 64       # can be larger than training since no gradients
NUM_WORKERS   = 2        # try higher to parallelize video decoding
THRESHOLD     = 0.5      # probability threshold for contact
START_FRAME   = 117_750
END_FRAME     = 378_250  # only first 250k frames (second half has no pole)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Output dir: {OUTPUT_DIR}")

model = load_model(CHECKPOINT_PATH, DEVICE)

transform = get_inference_transforms(IMG_SIZE)

dataset = VideoFrameDataset(
    video_path=VIDEO_PATH,
    start_frame=START_FRAME,
    end_frame=END_FRAME,
    transform=transform,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Total frames to process: {len(dataset):,}")
print(f"Batches: {len(dataloader):,}")

results_df = run_inference(model, dataloader, DEVICE, threshold=THRESHOLD)

n_contact = (results_df["contact"] == 1).sum()
n_no_contact = (results_df["contact"] == 0).sum()
print(f"\nTotal frames   : {len(results_df):,}")
print(f"Contact frames : {n_contact:,} ({100*n_contact/len(results_df):.1f}%)")
print(f"No contact     : {n_no_contact:,} ({100*n_no_contact/len(results_df):.1f}%)")
print()
results_df.head(10)

# Per-frame CSV
per_frame_path = os.path.join(OUTPUT_DIR, "per_frame_predictions.csv")
results_df.to_csv(per_frame_path, index=False)
print(f"Saved per-frame predictions → {per_frame_path}")

# Contact intervals CSV
intervals_df = frames_to_intervals(results_df, label_col="contact", label_val=1)
intervals_path = os.path.join(OUTPUT_DIR, "contact_intervals.csv")
intervals_df.to_csv(intervals_path, index=False)
print(f"Saved contact intervals     → {intervals_path}")
print(f"\nFound {len(intervals_df)} contact intervals")
print()
intervals_df.head(20)