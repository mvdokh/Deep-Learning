"""
Book Summarizer — A PyTorch-based hierarchical summarization pipeline.

This module extracts text from a PDF (e.g. an entire book), splits it into
digestible chunks, and uses a pre-trained PyTorch transformer model
(facebook/bart-large-cnn) to produce abstractive summaries.  Because books
are far too long for a single forward pass, a *hierarchical* strategy is
used:

    1. Extract raw text from every page of the PDF.
    2. Split the text into overlapping chunks that fit within the model's
       max input length (1024 tokens for BART).
    3. Summarize each chunk independently.
    4. Concatenate the chunk-level summaries and repeat the process until
       the combined text is short enough to summarize in one pass.

The final output is a concise, coherent summary of the entire book.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SummarizerConfig:
    """All knobs for the summarization pipeline live here."""

    # HuggingFace model identifier (must be a PyTorch checkpoint)
    model_name: str = "facebook/bart-large-cnn"

    # Generation hyper-parameters
    max_input_tokens: int = 1024
    max_summary_tokens: int = 256
    min_summary_tokens: int = 56
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3

    # Chunking
    chunk_overlap_tokens: int = 64  # overlap between consecutive chunks

    # Device — auto-detected but can be overridden
    device: Optional[str] = None

    # Hierarchical summarization depth limit (safety valve)
    max_recursion_depth: int = 6

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Return the full text of a PDF, page by page."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
    doc.close()

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise ValueError("No extractable text found in the PDF.")
    return full_text


# ---------------------------------------------------------------------------
# Core summariser class
# ---------------------------------------------------------------------------

class BookSummarizer:
    """
    Wraps a PyTorch BART model and exposes a high-level ``summarize_pdf``
    method that handles everything from text extraction to hierarchical
    summarization.
    """

    def __init__(self, config: SummarizerConfig | None = None) -> None:
        self.cfg = config or SummarizerConfig()

        print(f"[BookSummarizer] Loading model '{self.cfg.model_name}' …")
        self.tokenizer = BartTokenizer.from_pretrained(self.cfg.model_name)
        self.model: BartForConditionalGeneration = (
            BartForConditionalGeneration.from_pretrained(self.cfg.model_name)
            .to(self.cfg.device)
        )
        self.model.eval()
        print(
            f"[BookSummarizer] Model loaded on {self.cfg.device} "
            f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params)"
        )

    # ---- public API -------------------------------------------------------

    def summarize_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> str:
        """End-to-end: PDF path in → summary string out.

        If *output_dir* is provided, intermediate and final summaries are
        written to numbered text files inside that directory.
        """
        print(f"\n[BookSummarizer] Extracting text from '{pdf_path}' …")
        raw_text = extract_text_from_pdf(pdf_path)
        print(f"[BookSummarizer] Extracted {len(raw_text):,} characters.")

        # Resolve output directory
        out = self._prepare_output_dir(output_dir, pdf_path)
        return self.summarize_text(raw_text, output_dir=out)

    def summarize_text(
        self,
        text: str,
        output_dir: Path | None = None,
    ) -> str:
        """Hierarchically summarize an arbitrarily long text string."""
        return self._hierarchical_summarize(text, depth=0, output_dir=output_dir)

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _prepare_output_dir(
        output_dir: str | Path | None,
        pdf_path: str | Path,
    ) -> Path:
        """Create (or default) the output directory for this run."""
        if output_dir is not None:
            out = Path(output_dir)
        else:
            stem = Path(pdf_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path(pdf_path).parent / f"summary_{stem}_{timestamp}"

        out.mkdir(parents=True, exist_ok=True)
        print(f"[BookSummarizer] Output directory: {out}")
        return out

    def _hierarchical_summarize(
        self,
        text: str,
        depth: int,
        output_dir: Path | None = None,
    ) -> str:
        """Recursively chunk → summarize → concatenate until short enough."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)

        indent = "  " * depth
        print(
            f"{indent}[depth {depth}] Input length: {n_tokens:,} tokens"
        )

        # Base case: text fits in a single model pass
        if n_tokens <= self.cfg.max_input_tokens:
            print(f"{indent}[depth {depth}] Summarising in a single pass …")
            final_summary = self._summarize_chunk(text)

            if output_dir is not None:
                self._write_final_summary(output_dir, depth, final_summary)

            return final_summary

        if depth >= self.cfg.max_recursion_depth:
            print(
                f"{indent}[depth {depth}] Max recursion depth reached — "
                "truncating to fit."
            )
            truncated = self.tokenizer.decode(
                tokens[: self.cfg.max_input_tokens], skip_special_tokens=True
            )
            final_summary = self._summarize_chunk(truncated)

            if output_dir is not None:
                self._write_final_summary(output_dir, depth, final_summary)

            return final_summary

        # Recursive case: split into chunks, summarize each, then combine
        chunks = self._split_into_chunks(tokens)
        print(f"{indent}[depth {depth}] Split into {len(chunks)} chunks.")

        summaries: list[str] = []
        for i, chunk_ids in enumerate(chunks):
            chunk_text = self.tokenizer.decode(
                chunk_ids, skip_special_tokens=True
            )
            print(
                f"{indent}  chunk {i + 1}/{len(chunks)} "
                f"({len(chunk_ids)} tokens) …"
            )
            summary = self._summarize_chunk(chunk_text)
            summaries.append(summary)

        combined = " ".join(summaries)
        combined_tokens = len(
            self.tokenizer.encode(combined, add_special_tokens=False)
        )
        print(
            f"{indent}[depth {depth}] Combined summaries: "
            f"{combined_tokens:,} tokens"
        )

        # ---- write this depth's chunk summaries to a file ----
        if output_dir is not None:
            self._write_depth_file(output_dir, depth, chunks, summaries)

        # Recurse on the combined summaries
        return self._hierarchical_summarize(
            combined, depth + 1, output_dir=output_dir
        )

    # ---- file-writing helpers ---------------------------------------------

    @staticmethod
    def _write_depth_file(
        output_dir: Path,
        depth: int,
        chunks: list[list[int]],
        summaries: list[str],
    ) -> None:
        """Write all chunk summaries for a given depth to a text file."""
        path = output_dir / f"depth_{depth}_summaries.txt"
        sep = "-" * 72

        lines: list[str] = []
        lines.append(f"{'=' * 72}")
        lines.append(f"  DEPTH {depth}  —  {len(summaries)} chunk(s) summarised")
        lines.append(f"{'=' * 72}\n")

        for i, summary in enumerate(summaries):
            lines.append(f"--- Chunk {i + 1}/{len(summaries)} "
                         f"({len(chunks[i])} input tokens) ---")
            lines.append(summary)
            lines.append("")  # blank line between chunks

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"    -> Saved depth {depth} summaries to '{path}'")

    @staticmethod
    def _write_final_summary(
        output_dir: Path,
        depth: int,
        summary: str,
    ) -> None:
        """Write the final (single-pass) summary to its own file."""
        path = output_dir / "final_summary.txt"

        lines: list[str] = []
        lines.append(f"{'=' * 72}")
        lines.append(f"  FINAL SUMMARY  (produced at depth {depth})")
        lines.append(f"{'=' * 72}\n")
        lines.append(summary)
        lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"    -> Saved final summary to '{path}'")

    def _split_into_chunks(self, token_ids: list[int]) -> list[list[int]]:
        """Split a token-id list into overlapping windows."""
        stride = self.cfg.max_input_tokens - self.cfg.chunk_overlap_tokens
        chunks: list[list[int]] = []
        for start in range(0, len(token_ids), stride):
            end = start + self.cfg.max_input_tokens
            chunk = token_ids[start:end]
            if len(chunk) < 32:  # skip tiny trailing fragment
                break
            chunks.append(chunk)
        return chunks

    @torch.inference_mode()
    def _summarize_chunk(self, text: str) -> str:
        """Run a single forward pass of the BART model on *text*."""
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        ).to(self.cfg.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.cfg.max_summary_tokens,
            min_length=self.cfg.min_summary_tokens,
            num_beams=self.cfg.num_beams,
            length_penalty=self.cfg.length_penalty,
            no_repeat_ngram_size=self.cfg.no_repeat_ngram_size,
            early_stopping=True,
        )

        return self.tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )


# ---------------------------------------------------------------------------
# Pretty-printing helper
# ---------------------------------------------------------------------------

def print_summary(summary: str, width: int = 80) -> None:
    """Print a nicely wrapped summary to the console."""
    border = "=" * width
    print(f"\n{border}")
    print("  BOOK SUMMARY")
    print(border)
    for line in textwrap.wrap(summary, width=width - 4):
        print(f"  {line}")
    print(f"{border}\n")
