"""
Configuration for the Car Recognition pipeline.
Edit CAR_CLASSES to define the exact makes/models/years you want to recognize.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# ──────────────────────────────────────────────
# Car classes to recognize
# Each entry is a dict with make, model, year (or year range).
# The "label" key is the human-readable class name used everywhere.
# ──────────────────────────────────────────────
CAR_CLASSES = [
    {"label": "Toyota Camry 2020",     "make": "Toyota",  "model": "Camry",     "year": "2020"},
    {"label": "Honda Civic 2021",      "make": "Honda",   "model": "Civic",     "year": "2021"},
    {"label": "Ford Mustang 2022",     "make": "Ford",    "model": "Mustang",   "year": "2022"},
    {"label": "Tesla Model 3 2023",    "make": "Tesla",   "model": "Model 3",   "year": "2023"},
    {"label": "BMW 3 Series 2022",     "make": "BMW",     "model": "3 Series",  "year": "2022"},
    {"label": "Mercedes C-Class 2023", "make": "Mercedes","model": "C-Class",   "year": "2023"},
    {"label": "Chevrolet Corvette 2023","make":"Chevrolet","model": "Corvette", "year": "2023"},
    {"label": "Porsche 911 2023",      "make": "Porsche", "model": "911",       "year": "2023"},
]

# Convenience mappings
LABEL_TO_IDX = {entry["label"]: idx for idx, entry in enumerate(CAR_CLASSES)}
IDX_TO_LABEL = {idx: entry["label"] for idx, entry in enumerate(CAR_CLASSES)}
NUM_CLASSES = len(CAR_CLASSES)

# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────
IMAGES_PER_CLASS = 200          # target number of images to download per class
MAX_DOWNLOAD_THREADS = 4

# ──────────────────────────────────────────────
# Preprocessing / augmentation
# ──────────────────────────────────────────────
IMAGE_SIZE = 224                # model input resolution
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
MODEL_NAME = "efficientnet_b0"  # any model supported by `timm`
PRETRAINED = True
FREEZE_BACKBONE = True          # freeze backbone initially, fine-tune later
UNFREEZE_AFTER_EPOCHS = 5       # unfreeze backbone after N epochs

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
EARLY_STOP_PATIENCE = 7
SCHEDULER = "cosine"            # "cosine" | "step" | "plateau"

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.3      # minimum softmax confidence to report
TOP_K = 3                       # how many predictions to show

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
