"""
Car image downloader / data generator.

Uses icrawler (Google/Bing image crawlers) to bulk-download images for each
car class defined in config.CAR_CLASSES, then splits them into
train / val / test folders organised for torchvision.datasets.ImageFolder.

Usage
-----
    python download_data.py                     # download + organise
    python download_data.py --skip-download     # just re-organise existing raw images
    python download_data.py --engine bing       # use Bing instead of Google
    python download_data.py --per-class 300     # override images per class
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm

import config


def _build_query(entry: dict) -> str:
    """Build a search-engine query string from a car class entry."""
    return f"{entry['year']} {entry['make']} {entry['model']} car photo"


def download_images(engine: str = "google", per_class: int | None = None):
    """Download images for every class defined in config.CAR_CLASSES."""
    per_class = per_class or config.IMAGES_PER_CLASS

    if engine == "google":
        from icrawler.builtin import GoogleImageCrawler as Crawler
    elif engine == "bing":
        from icrawler.builtin import BingImageCrawler as Crawler
    else:
        raise ValueError(f"Unsupported engine: {engine!r}  (use 'google' or 'bing')")

    for entry in config.CAR_CLASSES:
        label = entry["label"]
        save_dir = config.RAW_DIR / label
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(save_dir.glob("*")))
        if existing >= per_class:
            print(f"[skip] {label}: already have {existing} images")
            continue

        query = _build_query(entry)
        need = per_class - existing
        print(f"\n>>> Downloading ~{need} images for '{label}'  (query: {query!r})")

        crawler = Crawler(
            storage={"root_dir": str(save_dir)},
            downloader_threads=config.MAX_DOWNLOAD_THREADS,
        )
        crawler.crawl(
            keyword=query,
            max_num=need,
            min_size=(200, 200),
            file_idx_offset=existing,
        )

    print("\n--- Download phase complete ---")
    _print_stats()


def _print_stats():
    """Print how many images exist per class."""
    print("\nRaw image counts:")
    for entry in config.CAR_CLASSES:
        d = config.RAW_DIR / entry["label"]
        count = len(list(d.glob("*"))) if d.exists() else 0
        print(f"  {entry['label']:35s}  {count}")


# ──────────────────────────────────────────────
# Train / Val / Test split
# ──────────────────────────────────────────────

def _is_valid_image(path: Path) -> bool:
    """Quick check: is the file a real image we can open?"""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def organise_splits(seed: int = 42):
    """
    Take raw/<class>/*.jpg  -->  processed/{train,val,test}/<class>/*.jpg

    Uses config.TRAIN_SPLIT / VAL_SPLIT / TEST_SPLIT ratios.
    Only copies files that are valid images.
    """
    random.seed(seed)

    for split in ("train", "val", "test"):
        (config.PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for entry in config.CAR_CLASSES:
        label = entry["label"]
        src = config.RAW_DIR / label
        if not src.exists():
            print(f"[warn] No raw images for {label}, skipping.")
            continue

        # Gather valid images
        images = sorted(src.iterdir())
        valid = [p for p in tqdm(images, desc=f"Validating {label}", leave=False) if _is_valid_image(p)]
        random.shuffle(valid)

        n = len(valid)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)

        splits = {
            "train": valid[:n_train],
            "val":   valid[n_train:n_train + n_val],
            "test":  valid[n_train + n_val:],
        }

        for split_name, file_list in splits.items():
            dest = config.PROCESSED_DIR / split_name / label
            dest.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                shutil.copy2(f, dest / f.name)
            total_copied += len(file_list)

        print(f"  {label}: train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    print(f"\nTotal images copied to processed/: {total_copied}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download & organise car images")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, only re-organise existing raw images")
    parser.add_argument("--engine", choices=["google", "bing"], default="google",
                        help="Search engine to use for crawling (default: google)")
    parser.add_argument("--per-class", type=int, default=None,
                        help=f"Images to download per class (default: {config.IMAGES_PER_CLASS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")
    args = parser.parse_args()

    if not args.skip_download:
        download_images(engine=args.engine, per_class=args.per_class)

    print("\n--- Organising train / val / test splits ---")
    organise_splits(seed=args.seed)
    print("\nDone!  Data is ready in:", config.PROCESSED_DIR)


if __name__ == "__main__":
    main()
