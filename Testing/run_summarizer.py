#!/usr/bin/env python3
"""
CLI entry-point for the Book Summarizer.

Usage
-----
    # Summarise the default book PDF that ships with this repo:
    python run_summarizer.py

    # Summarise any PDF:
    python run_summarizer.py --pdf /path/to/book.pdf

    # Tweak generation parameters:
    python run_summarizer.py --pdf book.pdf --beams 6 --max-summary-tokens 300

    # Force CPU even if a GPU is available:
    python run_summarizer.py --device cpu

    # Save summary to a text file instead of (only) printing:
    python run_summarizer.py --output summary.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from summarizer import BookSummarizer, SummarizerConfig, print_summary


DEFAULT_PDF = (
    Path(__file__).parent
    / "the-true-creator-of-everything-how-the-human-brain-shaped-the-universe-"
      "as-we-know-it-0300244630-9780300244632_compress.pdf"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise an entire book (PDF) with a PyTorch transformer model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Path to the PDF file to summarise (default: the book bundled "
             "with this repo).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-cnn",
        help="HuggingFace model identifier (default: facebook/bart-large-cnn).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect).",
    )
    parser.add_argument(
        "--beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4).",
    )
    parser.add_argument(
        "--max-summary-tokens",
        type=int,
        default=256,
        help="Max tokens per chunk summary (default: 256).",
    )
    parser.add_argument(
        "--min-summary-tokens",
        type=int,
        default=56,
        help="Min tokens per chunk summary (default: 56).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional path to write the final summary text to a file.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.pdf.exists():
        print(f"Error: PDF not found at '{args.pdf}'", file=sys.stderr)
        sys.exit(1)

    config = SummarizerConfig(
        model_name=args.model,
        device=args.device,
        num_beams=args.beams,
        max_summary_tokens=args.max_summary_tokens,
        min_summary_tokens=args.min_summary_tokens,
    )

    summarizer = BookSummarizer(config)

    t0 = time.perf_counter()
    summary = summarizer.summarize_pdf(args.pdf)
    elapsed = time.perf_counter() - t0

    print_summary(summary)
    print(f"  [Finished in {elapsed:.1f}s]\n")

    if args.output:
        args.output.write_text(summary, encoding="utf-8")
        print(f"  Summary saved to '{args.output}'")


if __name__ == "__main__":
    main()
