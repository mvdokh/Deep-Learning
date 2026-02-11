"""
Inference script — pass in an image of a car and get predicted make/model/year.

Usage
-----
    # Single image
    python predict.py path/to/car_photo.jpg

    # Multiple images
    python predict.py img1.jpg img2.png img3.jpg

    # Specify a custom checkpoint
    python predict.py photo.jpg --checkpoint checkpoints/best_model.pth

    # Show top-5 predictions
    python predict.py photo.jpg --top-k 5

    # Use as a library
    from predict import CarPredictor
    predictor = CarPredictor("checkpoints/best_model.pth")
    results = predictor.predict("photo.jpg")
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dataset import eval_transform
from model import build_model


class CarPredictor:
    """
    High-level predictor that loads a checkpoint once and can run inference
    on as many images as needed.
    """

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.device = device or config.DEVICE
        checkpoint_path = checkpoint_path or str(config.CHECKPOINT_DIR / "best_model.pth")

        # Build model and load weights
        self.model = build_model(pretrained=False, freeze_backbone=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load class names — prefer the order from the training dataset folder
        processed_train = config.PROCESSED_DIR / "train"
        if processed_train.exists():
            self.class_names = sorted(
                [d.name for d in processed_train.iterdir() if d.is_dir()]
            )
        else:
            self.class_names = [config.IDX_TO_LABEL[i] for i in range(config.NUM_CLASSES)]

        print(f"CarPredictor ready  |  {len(self.class_names)} classes  |  device={self.device}")

    def _load_image(self, image_input) -> np.ndarray:
        """
        Accept a file path (str/Path), a PIL Image, or a numpy array.
        Returns an RGB numpy array.
        """
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise FileNotFoundError(f"Cannot open image: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")
        return image

    @torch.no_grad()
    def predict(self, image_input, top_k: int | None = None) -> list[dict]:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_input : str | Path | PIL.Image | np.ndarray
            The car image to classify.
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        list[dict]
            Each dict has keys: "label", "make", "model", "year", "confidence".
            Sorted by confidence descending.
        """
        top_k = top_k or config.TOP_K
        image = self._load_image(image_input)

        # Apply eval transforms
        augmented = eval_transform(image=image)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        # Top-K
        top_probs, top_idxs = probs.topk(min(top_k, len(self.class_names)))

        results = []
        for prob, idx in zip(top_probs.cpu(), top_idxs.cpu()):
            idx = idx.item()
            label = self.class_names[idx]
            # Try to look up structured info from config
            entry = next((c for c in config.CAR_CLASSES if c["label"] == label), None)
            results.append({
                "label": label,
                "make": entry["make"] if entry else "—",
                "model": entry["model"] if entry else "—",
                "year": entry["year"] if entry else "—",
                "confidence": round(prob.item(), 4),
            })

        return results

    def predict_batch(self, image_paths: list[str], top_k: int | None = None):
        """Run inference on a batch of images. Returns a list of result lists."""
        return [self.predict(p, top_k=top_k) for p in image_paths]


# ──────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────

def _print_results(image_path: str, results: list[dict]):
    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        conf_bar = "#" * int(r["confidence"] * 30)
        print(
            f"  {i}. {r['label']:35s}  "
            f"{r['confidence']*100:5.1f}%  {conf_bar}"
        )
        if r["confidence"] < config.CONFIDENCE_THRESHOLD and i == 1:
            print(f"     (low confidence — model is unsure)")
    print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict car make/model/year from an image")
    parser.add_argument("images", nargs="+", help="Path(s) to car image(s)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (default: checkpoints/best_model.pth)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help=f"Number of top predictions to show (default: {config.TOP_K})")
    args = parser.parse_args()

    predictor = CarPredictor(checkpoint_path=args.checkpoint)

    for img_path in args.images:
        results = predictor.predict(img_path, top_k=args.top_k)
        _print_results(img_path, results)


if __name__ == "__main__":
    main()
