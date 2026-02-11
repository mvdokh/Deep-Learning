"""
PDF figure and caption extraction utilities.

Uses PyMuPDF (fitz) to:
  1. Extract embedded images from each page of a PDF.
  2. Locate the caption text immediately following each figure.
  3. Save figures as individual image files alongside metadata.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

import config


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class ExtractedFigure:
    """One figure extracted from a PDF."""
    figure_id: str              # e.g. "fig_1"
    page_number: int            # 0-indexed PDF page
    image_path: str             # path to saved image file
    caption: str                # extracted caption text (may be empty)
    bbox: list[float]           # [x0, y0, x1, y1] on the page
    width: int                  # image pixel width
    height: int                 # image pixel height
    source_pdf: str             # original PDF filename


@dataclass
class ExtractionResult:
    """All figures extracted from one PDF."""
    pdf_path: str
    pdf_name: str
    num_pages: int
    figures: list[ExtractedFigure] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pdf_path": self.pdf_path,
            "pdf_name": self.pdf_name,
            "num_pages": self.num_pages,
            "figures": [asdict(f) for f in self.figures],
        }

    def save_json(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


# ──────────────────────────────────────────────
# Caption extraction helpers
# ──────────────────────────────────────────────

def _extract_page_text_blocks(page: fitz.Page) -> list[dict]:
    """
    Return text blocks on a page sorted top-to-bottom.
    Each block: {"bbox": (x0,y0,x1,y1), "text": str, "type": int}
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    text_blocks = []
    for b in blocks:
        if b["type"] == 0:  # text block
            lines_text = []
            for line in b.get("lines", []):
                spans_text = "".join(span["text"] for span in line.get("spans", []))
                lines_text.append(spans_text)
            text_blocks.append({
                "bbox": b["bbox"],
                "text": "\n".join(lines_text).strip(),
            })
    # Sort top to bottom
    text_blocks.sort(key=lambda b: b["bbox"][1])
    return text_blocks


def _find_caption_for_image(
    image_bbox: tuple[float, float, float, float],
    text_blocks: list[dict],
    max_lines: int = 8,
) -> str:
    """
    Find the caption text that sits below (or overlaps with) the image bbox.

    Strategy:
      1. Collect text blocks whose top edge is near or below the image bottom.
      2. Look for one that starts with a caption prefix ("Fig.", "Figure", etc.).
      3. Accumulate consecutive lines until the caption ends.
    """
    img_bottom = image_bbox[3]
    img_top = image_bbox[1]
    candidates = []

    for tb in text_blocks:
        tb_top = tb["bbox"][1]
        # Text block starts near the bottom of the image or below it
        # Allow some overlap (caption can overlap the image slightly)
        if tb_top >= img_top + (img_bottom - img_top) * 0.5:
            candidates.append(tb)

    # Look for a block starting with a figure caption prefix
    caption_parts = []
    found = False
    for tb in candidates:
        text = tb["text"].strip()
        if not found:
            if any(text.startswith(prefix) for prefix in config.CAPTION_PREFIXES):
                caption_parts.append(text)
                found = True
        else:
            # Continue accumulating if it doesn't look like a new section
            if (any(text.startswith(prefix) for prefix in config.CAPTION_PREFIXES)
                    or text == "" or len(caption_parts) >= max_lines):
                break
            # Stop if this looks like body text (starts with a non-caption sentence
            # that's very long and not a continuation)
            if len(text) > 200 and not text[0].islower():
                break
            caption_parts.append(text)

    caption = " ".join(caption_parts).strip()

    # Clean up common artifacts
    caption = re.sub(r'\s+', ' ', caption)
    return caption


# ──────────────────────────────────────────────
# Image extraction
# ──────────────────────────────────────────────

def _is_figure_sized(width: int, height: int) -> bool:
    """Filter out small images (icons, logos, decorations)."""
    if width < config.MIN_IMAGE_WIDTH or height < config.MIN_IMAGE_HEIGHT:
        return False
    if width * height < config.MIN_IMAGE_AREA:
        return False
    return True


def _save_image(pixmap: fitz.Pixmap, output_path: Path) -> bool:
    """Save a fitz Pixmap as a PNG, handling CMYK conversion."""
    try:
        if pixmap.n > 4:  # CMYK or with alpha
            pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
        pixmap.save(str(output_path))
        return True
    except Exception as e:
        print(f"  [warn] Could not save image: {e}")
        return False


# ──────────────────────────────────────────────
# Main extraction function
# ──────────────────────────────────────────────

def extract_figures_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    render_dpi: int = 200,
) -> ExtractionResult:
    """
    Extract all figures and their captions from a PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    output_dir : str | Path | None
        Where to save extracted figure images.
        Defaults to config.OUTPUT_DIR / <pdf_stem>.
    render_dpi : int
        DPI for rendering page regions as images (fallback method).

    Returns
    -------
    ExtractionResult
        Contains metadata and paths to all extracted figures.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_stem = pdf_path.stem
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR / pdf_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    result = ExtractionResult(
        pdf_path=str(pdf_path),
        pdf_name=pdf_path.name,
        num_pages=len(doc),
    )

    fig_counter = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text_blocks = _extract_page_text_blocks(page)

        # Get images on this page
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            width = base_image["width"]
            height = base_image["height"]

            if not _is_figure_sized(width, height):
                continue

            fig_counter += 1
            fig_id = f"fig_{fig_counter}"
            img_filename = f"{fig_id}.png"
            img_path = output_dir / img_filename

            # Save the image
            img_bytes = base_image["image"]
            try:
                pix = fitz.Pixmap(img_bytes)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(img_path))
            except Exception:
                # Fallback: write raw bytes and try PIL
                try:
                    raw_path = output_dir / f"{fig_id}_raw.{base_image['ext']}"
                    raw_path.write_bytes(img_bytes)
                    pil_img = Image.open(raw_path).convert("RGB")
                    pil_img.save(str(img_path))
                    raw_path.unlink()
                except Exception as e:
                    print(f"  [warn] Skipping image on page {page_idx + 1}: {e}")
                    continue

            # Try to find the image's bbox on the page for caption matching
            img_rects = page.get_image_rects(xref)
            if img_rects:
                bbox = list(img_rects[0])
            else:
                # Approximate: full page width, centered vertically
                bbox = [0, 0, page.rect.width, page.rect.height]

            # Extract caption
            caption = _find_caption_for_image(tuple(bbox), text_blocks)

            figure = ExtractedFigure(
                figure_id=fig_id,
                page_number=page_idx,
                image_path=str(img_path),
                caption=caption,
                bbox=bbox,
                width=width,
                height=height,
                source_pdf=pdf_path.name,
            )
            result.figures.append(figure)

    doc.close()

    # Save metadata
    result.save_json(output_dir / "extraction_meta.json")

    print(f"Extracted {len(result.figures)} figures from {pdf_path.name}")
    return result


def extract_figures_from_directory(
    papers_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[ExtractionResult]:
    """
    Extract figures from every PDF in a directory.

    Returns a list of ExtractionResult (one per PDF).
    """
    papers_dir = Path(papers_dir) if papers_dir else config.DEFAULT_PAPERS_DIR
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR

    if not papers_dir.exists():
        raise FileNotFoundError(f"Papers directory not found: {papers_dir}")

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {papers_dir}")
        return []

    print(f"Found {len(pdfs)} PDFs in {papers_dir}\n")
    results = []
    for pdf in pdfs:
        try:
            res = extract_figures_from_pdf(pdf, output_dir / pdf.stem)
            results.append(res)
        except Exception as e:
            print(f"  [error] {pdf.name}: {e}")

    return results
