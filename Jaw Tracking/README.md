# Jaw Tracking

Dual keypoint (jaw tip + jaw line) tracking across three experimental conditions using temporal heatmap regression with condition-aware inference.

## Overview

This pipeline predicts two keypoints per frame from side-view jaw images:

| Keypoint | CSV file |
|----------|----------|
| Jaw tip | `jaw_tip_side_clean.csv` |
| Jaw line | `jaw_line_side.csv` |

Three experimental conditions are supported:

| Condition | Aligned frames |
|-----------|----------------|
| `IRt_BiPoles` | 2,483 |
| `IRt_TeLC` | 3,213 |
| `PCRt_BiPoles` | 2,932 |

The model (~15M parameters) uses an EfficientNet-B2 backbone, 8-frame temporal context, condition FiLM embedding, and 2-channel Gaussian heatmap targets (10 px diameter). Training shuffles frames **within** each condition but never mixes conditions in a batch.

## Data layout

Each condition folder contains:

```
IRt_BiPoles/
├── jaw_tip_side_clean.csv    # frame,x,y
├── jaw_line_side.csv         # frame,x,y
└── images/
    ├── 0006970.png
    └── ...
```

- Images are **640×480** RGB PNGs.
- Frame numbers in filenames are zero-padded to 7 digits.
- Tip and line CSVs are inner-joined on `frame` (drops mismatched rows).

Default source path (WSL): `/mnt/c/Users/wanglab/Desktop/Tip+Base/`

## Environment setup

Use a dedicated conda environment in **WSL** (where this repo lives). Your RTX 5060 Ti is a **Blackwell** GPU (`sm_120`) — you must install PyTorch with **CUDA 12.8** (`cu128`). Older wheels (`cu124`, `cu126`) will not use the GPU correctly.

### 1. Confirm WSL sees the GPU

In a **WSL** terminal (not PowerShell):

```bash
nvidia-smi
```

You should see your RTX 5060 Ti. If this fails, update the NVIDIA driver on Windows (you already have 595.x) and ensure WSL2 is up to date (`wsl --update` from PowerShell).

### 2. Create the conda environment

```bash
cd "/home/wanglab/testing/Deep-Learning/Jaw Tracking"

conda create -n jaw-tracking python=3.10 -y
conda activate jaw-tracking

# RTX 5060 / 5060 Ti / 50-series — requires cu128 (NOT cu124)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

python -m ipykernel install --user --name jaw-tracking --display-name "Python (jaw-tracking)"
```

If `cu128` stable fails or you see `sm_120 is not compatible`, use the nightly build:

```bash
pip uninstall -y torch torchvision
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Verify GPU

```bash
conda activate jaw-tracking
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    x = torch.randn(4, 4, device='cuda')
    print('cuda tensor ok:', x.device)
"
```

Expected: `cuda available: True` and `NVIDIA GeForce RTX 5060 Ti`.

### 4. Open notebooks in Cursor

1. Open the project in WSL (not Windows path).
2. `conda activate jaw-tracking`
3. Open `Training/train.ipynb`
4. Select kernel **Python (jaw-tracking)** (top-right).
5. Ensure `../data/train.pkl` exists (run dataset creation below if not).

The training notebook will use GPU automatically when `torch.cuda.is_available()` is `True`.

### CPU-only fallback

Only if you cannot get WSL GPU working:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Quick start

```bash
cd "Jaw Tracking"
pip install -r requirements.txt

# 1. Create datasets (~2.3–3 GB per condition pickle)
cd "Create Dataset"
python create_dataset.py --data_root /mnt/c/Users/wanglab/Desktop/Tip+Base

# 2. Train
cd ../Training
# Option A: notebook (recommended)
#   Open train.ipynb, edit config cell, run all cells
# Option B: CLI
#   python train.py --train_pkl ../data/train.pkl --val_pkl ../data/val.pkl --out_dir ./checkpoints

# 3. Predict (see Prediction/predict.ipynb)
```

## Pickle schema

Per-condition and merged pickles contain:

```python
{
    "frames":          np.ndarray  # (N, 480, 640, 3) uint8 RGB
    "frame_numbers":   np.ndarray  # (N,) int
    "keypoints_tip":   np.ndarray  # (N, 2) float32 — x, y in original pixels
    "keypoints_line":  np.ndarray  # (N, 2) float32
    "condition":       str         # per-condition files only
    "experiment_id":   int         # 0=IRt_BiPoles, 1=IRt_TeLC, 2=PCRt_BiPoles
    "experiment_ids":  np.ndarray  # merged/train/val files
}
```

Outputs in `data/`:

- `IRt_BiPoles.pkl`, `IRt_TeLC.pkl`, `PCRt_BiPoles.pkl`
- `merged.pkl`, `train.pkl`, `val.pkl` (80/20 split within each condition)

## Training

Open [`Training/train.ipynb`](Training/train.ipynb) and edit the config cell, then run all cells. It calls `run_training()` from [`train.py`](Training/train.py) — same logic as the CLI, no terminal required.

| Setting | Default |
|---------|---------|
| Window size | 8 frames |
| Image size | 320×240 |
| Batch size | 8 |
| Epochs | 80 |
| Optimizer | AdamW, lr=1e-3 |
| Loss | `0.5 * MSE(tip) + 0.5 * MSE(line)` on heatmaps |
| Early stopping | patience 15 on val loss |

**Augmentations** (training): resize, ±15° rotation, horizontal flip, brightness/contrast. `ReplayCompose` applies the same random transform to all 8 frames; keypoints on the center frame are transformed jointly.

**Sampler**: `ExperimentGroupedBatchSampler` shuffles indices within each `experiment_id` only.

**Metrics**: PCK@10px and RMSE (per keypoint and mean), decoded via soft-argmax.

### Debugging augmentations

Open [`Training/visualize_augmentations.ipynb`](Training/visualize_augmentations.ipynb) to inspect batches, 8-frame sequences, heatmap overlays, and sampler behavior without training a model.

## Prediction

Open [`Prediction/predict.ipynb`](Prediction/predict.ipynb) and set:

```python
CONDITION = "IRt_BiPoles"   # or "IRt_TeLC" / "PCRt_BiPoles"
CHECKPOINT = "../Training/checkpoints/best_model.pt"
INPUT_DIR = "/path/to/images"
```

Output CSV columns: `frame, tip_x, tip_y, line_x, line_y` in original 640×480 coordinates.

## Future work

The following items are intentionally out of scope for v1:

**Raw video ingestion.** The pipeline assumes pre-extracted `images/` folders matching the training layout. A future `extract_frames.py` could read `.avi`/`.mp4`, write zero-padded PNGs, and optionally seed CSVs from an existing tracker.

**Per-condition checkpoints.** v1 trains one multi-condition model with a condition embedding (FiLM). Separate per-condition models or fine-tuning runs could be added for comparison.

**Parquet / lazy loading.** v1 stores uint8 pixel arrays in per-condition pickles (~2.3–3 GB each). Parquet with path-only metadata or memory-mapped `.npy` stacks would reduce RAM at the cost of I/O complexity.

**Cross-condition validation.** v1 uses an 80/20 random split within each experiment. Leave-one-condition-out (train on two conditions, validate on the third) would measure generalization across experimental setups.

**Online / streaming inference.** v1 runs batch inference over a PNG folder. A rolling 8-frame buffer could support webcam or live video.

**Keypoint visibility / occlusion.** Both keypoints are always supervised. Future CSV visibility flags could mask loss for missing annotations.

**Temporal model variants.** v1 uses a lightweight `TemporalConvNet`. LSTM/Transformer heads or longer windows (16/32 frames) are straightforward swaps.

**Export formats.** v1 writes CSV predictions. COCO keypoints JSON, DeepLabCut-compatible output, or overlay video export could be added.

**Hyperparameter search.** Defaults are fixed in `train.py`. Weights & Biases or Optuna integration would help tune learning rate, window size, and augmentation strength.

**Deployment.** v1 is local CLI + Jupyter. ONNX/TorchScript export or cluster batch jobs would support production pipelines.
