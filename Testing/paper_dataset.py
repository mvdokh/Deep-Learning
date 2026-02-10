"""
paper_dataset.py — PyTorch Dataset for scientific-paper PDF summarisation.

This module handles:
    1.  Text extraction from multi-page PDFs  (PyMuPDF / fitz)
    2.  Figure / image extraction from PDF pages
    3.  Automatic figure captioning via a pre-trained vision-language model
        (Salesforce BLIP) so that BART can "see" the visual content
    4.  Abstract extraction — used as the ground-truth summary target
    5.  Tokenisation and collation for BART fine-tuning

Directory layout expected
-------------------------
    papers_dir/
        paper_001.pdf
        paper_002.pdf
        ...

Each PDF is assumed to be a scientific paper whose **abstract** can be
located automatically (heuristic: text between "Abstract" and the next
section heading).  If no abstract is found, the first 512 tokens are
used as a fallback target.

Optionally, you can provide a CSV / JSON manifest that maps each PDF
filename to an explicit target summary — see ``PaperDataset.from_manifest``.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    BartTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Knobs for data processing and tokenisation."""

    # BART tokeniser
    bart_model_name: str = "facebook/bart-large-cnn"
    max_source_tokens: int = 1024
    max_target_tokens: int = 256

    # BLIP captioning model (generates text descriptions of figures)
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    max_caption_tokens: int = 48
    caption_device: Optional[str] = None  # auto-detected

    # Image extraction
    min_image_width: int = 100   # skip tiny icons / logos
    min_image_height: int = 100
    max_images_per_paper: int = 10

    # Abstract extraction fallback length (tokens)
    fallback_target_tokens: int = 512

    def __post_init__(self) -> None:
        if self.caption_device is None:
            if torch.cuda.is_available():
                self.caption_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.caption_device = "mps"
            else:
                self.caption_device = "cpu"


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

_ABSTRACT_RE = re.compile(
    r"(?i)\babstract\b[:\.\s]*\n?(.*?)(?=\n\s*\b("
    r"introduction|background|related\s+work|methods|methodology|"
    r"1[\.\s]|I[\.\s]|keywords"
    r")\b)",
    re.DOTALL,
)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Return concatenated page text."""
    doc = fitz.open(str(pdf_path))
    pages = [p.get_text("text").strip() for p in doc if p.get_text("text").strip()]
    doc.close()
    return "\n\n".join(pages)


def extract_abstract(full_text: str) -> Optional[str]:
    """Attempt to pull the abstract from the paper text."""
    m = _ABSTRACT_RE.search(full_text)
    if m:
        abstract = m.group(1).strip()
        if 30 < len(abstract) < 5000:
            return abstract
    return None


def extract_images_from_pdf(
    pdf_path: Path,
    min_w: int = 100,
    min_h: int = 100,
    max_images: int = 10,
) -> List[Image.Image]:
    """Extract raster images embedded in the PDF."""
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []

    for page in doc:
        if len(images) >= max_images:
            break
        for img_info in page.get_images(full=True):
            if len(images) >= max_images:
                break
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if base_image["width"] < min_w or base_image["height"] < min_h:
                continue

            try:
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                images.append(pil_img)
            except Exception:
                continue

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Figure captioning via BLIP
# ---------------------------------------------------------------------------

class FigureCaptioner:
    """Generates text descriptions of images using BLIP (PyTorch)."""

    def __init__(self, config: DatasetConfig) -> None:
        self.cfg = config
        print(f"[FigureCaptioner] Loading '{config.blip_model_name}' …")
        self.processor = BlipProcessor.from_pretrained(config.blip_model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.blip_model_name
        ).to(config.caption_device)
        self.model.eval()
        print(f"[FigureCaptioner] Ready on {config.caption_device}.")

    @torch.inference_mode()
    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.cfg.caption_device
        )
        ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_caption_tokens,
        )
        return self.processor.decode(ids[0], skip_special_tokens=True)

    def caption_batch(self, images: List[Image.Image]) -> List[str]:
        return [self.caption(img) for img in images]


# ---------------------------------------------------------------------------
# Single-paper processor (combines text + figure captions)
# ---------------------------------------------------------------------------

def process_paper(
    pdf_path: Path,
    captioner: Optional[FigureCaptioner],
    cfg: DatasetConfig,
) -> Dict[str, str]:
    """
    Extract everything from one PDF and return a dict with:
        source  — full text with figure captions inserted
        target  — abstract (or fallback)
    """
    full_text = extract_text_from_pdf(pdf_path)

    # --- figure captions ---
    figure_block = ""
    if captioner is not None:
        images = extract_images_from_pdf(
            pdf_path,
            min_w=cfg.min_image_width,
            min_h=cfg.min_image_height,
            max_images=cfg.max_images_per_paper,
        )
        if images:
            captions = captioner.caption_batch(images)
            lines = [
                f"[Figure {i + 1}]: {cap}" for i, cap in enumerate(captions)
            ]
            figure_block = (
                "\n\n--- Figures ---\n" + "\n".join(lines) + "\n--- End Figures ---\n"
            )

    source = full_text + figure_block

    # --- target (abstract) ---
    abstract = extract_abstract(full_text)
    if abstract is None:
        # Fallback: first N tokens decoded back to text
        tokenizer = BartTokenizer.from_pretrained(cfg.bart_model_name)
        ids = tokenizer.encode(full_text, add_special_tokens=False)[
            : cfg.fallback_target_tokens
        ]
        abstract = tokenizer.decode(ids, skip_special_tokens=True)

    return {"source": source, "target": abstract}


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PaperDataset(Dataset):
    """
    A map-style PyTorch Dataset that yields tokenised (input_ids,
    attention_mask, labels) tuples ready for BART fine-tuning.

    Construction
    ------------
    >>> ds = PaperDataset.from_directory("path/to/papers/")

    Or with an explicit manifest:
    >>> ds = PaperDataset.from_manifest("manifest.json", "path/to/papers/")
    """

    def __init__(
        self,
        records: List[Dict[str, str]],
        tokenizer: BartTokenizer,
        max_source_tokens: int = 1024,
        max_target_tokens: int = 256,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_source = max_source_tokens
        self.max_target = max_target_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        source_enc = self.tokenizer(
            rec["source"],
            max_length=self.max_source,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            rec["target"],
            max_length=self.max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        # Replace padding token ids with -100 so they are ignored by loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }

    # ---- factory methods --------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        papers_dir: str | Path,
        config: DatasetConfig | None = None,
        use_captioner: bool = True,
    ) -> "PaperDataset":
        """Build dataset by scanning a directory for *.pdf files."""
        cfg = config or DatasetConfig()
        papers_dir = Path(papers_dir)
        pdfs = sorted(papers_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {papers_dir}")

        print(f"[PaperDataset] Found {len(pdfs)} PDF(s) in '{papers_dir}'")

        captioner = FigureCaptioner(cfg) if use_captioner else None
        tokenizer = BartTokenizer.from_pretrained(cfg.bart_model_name)

        records: List[Dict[str, str]] = []
        for i, pdf in enumerate(pdfs):
            print(f"  Processing [{i + 1}/{len(pdfs)}] {pdf.name} …")
            try:
                rec = process_paper(pdf, captioner, cfg)
                records.append(rec)
            except Exception as e:
                print(f"    WARNING: skipping {pdf.name}: {e}")

        print(f"[PaperDataset] {len(records)} paper(s) processed successfully.")
        return cls(records, tokenizer, cfg.max_source_tokens, cfg.max_target_tokens)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        papers_dir: str | Path,
        config: DatasetConfig | None = None,
        use_captioner: bool = True,
    ) -> "PaperDataset":
        """
        Build dataset from a JSON manifest file.

        Expected format (list of objects):
            [
                {"pdf": "paper_001.pdf", "summary": "The paper proposes …"},
                …
            ]

        If "summary" is omitted for an entry, the abstract extractor is used.
        """
        cfg = config or DatasetConfig()
        papers_dir = Path(papers_dir)
        manifest_path = Path(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(f"[PaperDataset] Manifest has {len(manifest)} entries.")

        captioner = FigureCaptioner(cfg) if use_captioner else None
        tokenizer = BartTokenizer.from_pretrained(cfg.bart_model_name)

        records: List[Dict[str, str]] = []
        for i, entry in enumerate(manifest):
            pdf_path = papers_dir / entry["pdf"]
            print(f"  Processing [{i + 1}/{len(manifest)}] {pdf_path.name} …")
            try:
                rec = process_paper(pdf_path, captioner, cfg)
                # Override target if manifest provides an explicit summary
                if "summary" in entry and entry["summary"]:
                    rec["target"] = entry["summary"]
                records.append(rec)
            except Exception as e:
                print(f"    WARNING: skipping {pdf_path.name}: {e}")

        print(f"[PaperDataset] {len(records)} paper(s) processed successfully.")
        return cls(records, tokenizer, cfg.max_source_tokens, cfg.max_target_tokens)
