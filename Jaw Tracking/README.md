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

## Quick start

```bash
cd "Jaw Tracking"
pip install -r requirements.txt

# 1. Create datasets (~2.3–3 GB per condition pickle)
cd "Create Dataset"
python create_dataset.py --data_root /mnt/c/Users/wanglab/Desktop/Tip+Base

# 2. Train
cd ../Training
python train.py --train_pkl ../data/train.pkl --val_pkl ../data/val.pkl --out_dir ./checkpoints

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
