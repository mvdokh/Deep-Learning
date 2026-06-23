# Jaw Tracking

Dual keypoint (jaw tip + jaw line) tracking from side-view images. **One shared model** trained on all three conditions; batches shuffle within each condition only.

## Pipeline

```
Create Dataset/  →  data/{train,val}.pkl
Training/        →  checkpoints/best_model.pt
Prediction/      →  jaw_tip.csv, jaw_base.csv
```

| Keypoint | Source CSV | Prediction output |
|----------|------------|-------------------|
| Jaw tip | `jaw_tip_side_clean.csv` | `jaw_tip.csv` (`frame,x,y`) |
| Jaw line / base | `jaw_line_side.csv` | `jaw_base.csv` (`frame,x,y`) |

**Conditions:** `IRt_BiPoles`, `IRt_TeLC`, `PCRt_BiPoles` — images 640×480 PNGs, CSVs joined on `frame`.

Default source: `/mnt/c/Users/wanglab/Desktop/Tip+Base/`

## Setup (WSL + RTX 50-series)

RTX 5060 Ti needs PyTorch **CUDA 12.8** (`cu128`):

```bash
cd "/home/wanglab/testing/Deep-Learning/Jaw Tracking"
conda create -n jaw-tracking python=3.10 -y && conda activate jaw-tracking
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m ipykernel install --user --name jaw-tracking --display-name "Python (jaw-tracking)"
```

Verify: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`

## Quick start

```bash
cd "Create Dataset"
python create_dataset.py --data_root /mnt/c/Users/wanglab/Desktop/Tip+Base

cd ../Training
# Open train.ipynb (kernel: Python (jaw-tracking)) or: python train.py
```

## Model & training

EfficientNet-B2 backbone + 8-frame temporal conv + FiLM + 2-channel heatmap decoder (~15M params). Input 320×240; targets are 10 px Gaussian heatmaps.

| Setting | Default |
|---------|---------|
| Window | 8 frames |
| Batch size | 32 (notebook) / 8 (CLI) |
| Optimizer | AdamW, lr=1e-3 |
| Loss | focal heatmap + λ× coordinate (smooth L1, λ=1) |
| Backbone | frozen → unfreeze epoch 15 (lr×0.1) |
| Checkpoint | best `val_rmse_mean`; early stop patience 15 |

Coordinate loss decodes predictions via soft-argmax and directly optimizes location — fixes the old problem where heatmap MSE looked tiny while RMSE stayed ~120–180 px.

**Sampler:** `ExperimentGroupedBatchSampler` — never mixes conditions in a batch.

**Metrics:** RMSE and PCK@10px in original 640×480 space.

**Augmentations:** resize, ±15° rotation, flip, brightness/contrast, light blur/noise/dust/motion blur (`ReplayCompose` across all 8 frames).

Debug augmentations: [`Training/visualize_augmentations.ipynb`](Training/visualize_augmentations.ipynb)

## Prediction

Open [`Prediction/predict.ipynb`](Prediction/predict.ipynb). Loads `../Training/checkpoints/best_model.pt`. Supports `.mp4` or image folders. Coords scaled to actual video dimensions.

## Data pickles (`data/`)

`merged.pkl`, `train.pkl`, `val.pkl` — 80/20 **temporal** split per condition (last 20% → val). Schema:

```python
{
    "frames": np.ndarray,          # (N, 480, 640, 3) uint8
    "frame_numbers": np.ndarray,     # (N,) int
    "keypoints_tip": np.ndarray,     # (N, 2) float32
    "keypoints_line": np.ndarray,    # (N, 2) float32
    "experiment_ids": np.ndarray,    # 0/1/2 per condition
}
```
