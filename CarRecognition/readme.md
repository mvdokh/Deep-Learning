# Car Recognition — Make / Model / Year

A PyTorch-based image classification pipeline that downloads car images from the
web, trains an EfficientNet (or any `timm`-supported model) via transfer
learning, and provides a simple inference API to identify a car's make, model,
and year from a photo.

---

## Project Structure

```
CarRecognition/
├── config.py           # All configurable settings (classes, hyper-params, paths)
├── download_data.py    # Bulk image downloader + train/val/test splitter
├── dataset.py          # PyTorch Dataset & DataLoader with Albumentations
├── model.py            # Model builder (timm) with freeze/unfreeze utilities
├── train.py            # Full training loop with TensorBoard, early stopping
├── predict.py          # CLI & Python API for single-image inference
├── requirements.txt    # Python dependencies
├── car_classes.json    # (auto-generated) full car class list from NHTSA API
├── tools/
│   └── fetch_car_classes.py  # Fetches makes/models/years from NHTSA API
├── ClassGUI/
│   └── app.py                # Tkinter GUI for rapid manual image classification
├── data/
│   ├── raw/            # Downloaded images organised by class label
│   └── processed/      # train / val / test splits (ImageFolder layout)
├── checkpoints/        # Saved model weights
└── logs/               # TensorBoard logs
```

---

## Quick Start

### 1. Install dependencies

```bash
cd CarRecognition
pip install -r requirements.txt
```

### 2. Fetch car classes from the NHTSA API

Instead of manually listing cars, run the fetch tool to pull every
make/model/year from the US government's NHTSA database (free, no API key):

```bash
# Fetch popular makes, years 2018-2025 (writes car_classes.json)
python tools/fetch_car_classes.py

# Custom year range
python tools/fetch_car_classes.py --years 2020 2025

# Only specific makes
python tools/fetch_car_classes.py --makes Toyota Honda BMW

# Fetch ALL makes (warning: huge list, slow)
python tools/fetch_car_classes.py --all-makes --years 2023 2024

# Filter to passenger cars only (no trucks/SUVs)
python tools/fetch_car_classes.py --car-type passenger

# Preview without writing anything
python tools/fetch_car_classes.py --dry-run
```

This generates `car_classes.json` (or writes directly into `config.py`).
`config.py` auto-loads from `car_classes.json` if it exists.

You can also manually edit `CAR_CLASSES` in `config.py` if you prefer.

### 3. Download images

```bash
# Download images via Google (default)
python download_data.py

# Or use Bing as the image source
python download_data.py --engine bing

# Override the number of images per class
python download_data.py --per-class 500
```

This will:
1. Crawl the web for images of each car class
2. Save them to `data/raw/<class_label>/`
3. Validate and split into `data/processed/{train,val,test}/<class_label>/`

### 4. Train the model

```bash
python train.py

# Override hyper-parameters
python train.py --epochs 50 --lr 3e-4 --batch-size 64

# Resume from a checkpoint
python train.py --resume checkpoints/latest_model.pth
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/
```

### 5. Run inference

```bash
# Predict a single image
python predict.py path/to/car_photo.jpg

# Multiple images
python predict.py img1.jpg img2.jpg img3.jpg

# Show top-5 predictions
python predict.py photo.jpg --top-k 5
```

Example output:

```
============================================================
Image: my_car.jpg
============================================================
  1. Tesla Model 3 2023                 87.3%  ##########################
  2. BMW 3 Series 2022                   6.2%  #
  3. Mercedes C-Class 2023               3.1%
```

### 6. Manually classify images with the GUI

Launch the ClassGUI to rapidly label a folder of car images by hand:

```bash
python ClassGUI/app.py
```

- **Select Image Folder** — pick a folder of car photos
- **Search & Dropdown** — type to filter classes, pick from the dropdown
- **Enter** — classify and advance to the next image
- **Arrow keys** — navigate images, **S** to skip, **Ctrl/Cmd+Z** to undo
- Choose between **copy** or **move** mode
- Optionally set a separate output folder

The classes in the dropdown are loaded from `car_classes.json` (or the
inline fallback in `config.py`). Hit **Reload Classes** after running
`fetch_car_classes.py` to refresh the list without restarting.

### 7. Use as a Python library

```python
from predict import CarPredictor

predictor = CarPredictor("checkpoints/best_model.pth")
results = predictor.predict("my_car.jpg")

for r in results:
    print(f"{r['year']} {r['make']} {r['model']} — {r['confidence']*100:.1f}%")
```

---

## Key Design Decisions

| Decision | Detail |
|---|---|
| **Backbone** | EfficientNet-B0 via `timm` (swap to any model in `config.MODEL_NAME`) |
| **Transfer learning** | Backbone frozen for first N epochs, then unfrozen with lower LR |
| **Augmentation** | Albumentations (random crop, flip, color jitter, blur, cutout) |
| **Label smoothing** | 0.1 to prevent overconfident predictions |
| **Scheduler** | Cosine annealing (configurable to step or plateau) |
| **Early stopping** | Patience-based on validation accuracy |

---

## Tips for Better Accuracy

- **More data wins.** Increase `IMAGES_PER_CLASS` in config.py (500+ recommended).
- **Manually curate.** After downloading, scan `data/raw/` and delete irrelevant
  images (interiors, logos, diagrams).
- **Use a larger backbone.** Try `efficientnet_b3`, `convnext_small`, or
  `vit_small_patch16_224` in `config.MODEL_NAME`.
- **Add more classes gradually.** Start small (5-10 classes), verify good
  accuracy, then expand.
