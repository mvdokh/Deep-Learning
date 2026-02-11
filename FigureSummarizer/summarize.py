"""
End-to-end summarization pipeline.

Ties together extraction (extract.py) and inference (model.py):
  1. Extract figures + captions from a PDF
  2. Run each figure through the vision-language model
  3. Save summaries alongside the figures

Can also be used from the CLI:
    python summarize.py "path/to/paper.pdf"
    python summarize.py "path/to/papers_dir/" --all
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from extract import ExtractedFigure, ExtractionResult, extract_figures_from_pdf


@dataclass
class FigureSummary:
    """A figure plus its generated summary."""
    figure_id: str
    page_number: int
    image_path: str
    caption: str
    summary: str
    source_pdf: str


@dataclass
class PaperSummaryResult:
    """All figure summaries for one paper."""
    pdf_name: str
    num_figures: int
    figures: list[FigureSummary] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "num_figures": self.num_figures,
            "figures": [asdict(f) for f in self.figures],
        }

    def save_json(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    def to_markdown(self) -> str:
        """Render summaries as a readable Markdown document."""
        lines = [f"# Figure Summaries: {self.pdf_name}\n"]
        for fig in self.figures:
            lines.append(f"## {fig.figure_id} (page {fig.page_number + 1})\n")
            lines.append(f"![{fig.figure_id}]({fig.image_path})\n")
            if fig.caption:
                lines.append(f"**Caption:** {fig.caption}\n")
            lines.append(f"**Summary:** {fig.summary}\n")
            lines.append("---\n")
        return "\n".join(lines)

    def save_markdown(self, path: Path):
        path.write_text(self.to_markdown())


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def summarize_paper(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    summarizer=None,
) -> PaperSummaryResult:
    """
    Full pipeline: extract figures from a PDF, then summarize each one.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    output_dir : str | Path | None
        Where to save figures and summaries.
    summarizer : FigureSummarizer | None
        Pre-loaded model. If None, one is created (slow first call).

    Returns
    -------
    PaperSummaryResult
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR / pdf_path.stem

    # 1. Extract figures
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")
    extraction = extract_figures_from_pdf(pdf_path, output_dir)

    if not extraction.figures:
        print("No figures found in this PDF.")
        return PaperSummaryResult(pdf_name=pdf_path.name, num_figures=0)

    # 2. Load model if needed
    if summarizer is None:
        from model import FigureSummarizer
        summarizer = FigureSummarizer()

    # 3. Summarize each figure
    result = PaperSummaryResult(
        pdf_name=pdf_path.name,
        num_figures=len(extraction.figures),
    )

    for fig in tqdm(extraction.figures, desc="Summarizing figures"):
        print(f"\n--- {fig.figure_id} (page {fig.page_number + 1}) ---")
        if fig.caption:
            print(f"  Caption: {fig.caption[:120]}...")

        summary = summarizer.summarize(fig.image_path, caption=fig.caption)
        print(f"  Summary: {summary[:120]}...")

        result.figures.append(FigureSummary(
            figure_id=fig.figure_id,
            page_number=fig.page_number,
            image_path=fig.image_path,
            caption=fig.caption,
            summary=summary,
            source_pdf=fig.source_pdf,
        ))

    # 4. Save results
    result.save_json(output_dir / "summaries.json")
    result.save_markdown(output_dir / "summaries.md")
    print(f"\nSaved summaries to {output_dir}/")

    return result


def summarize_directory(
    papers_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[PaperSummaryResult]:
    """
    Summarize figures from all PDFs in a directory.

    Loads the model once and reuses it across all papers.
    """
    papers_dir = Path(papers_dir) if papers_dir else config.DEFAULT_PAPERS_DIR
    output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {papers_dir}")
        return []

    print(f"Found {len(pdfs)} PDFs. Loading model...\n")
    from model import FigureSummarizer
    summarizer = FigureSummarizer()

    results = []
    for pdf in pdfs:
        try:
            res = summarize_paper(pdf, output_dir / pdf.stem, summarizer=summarizer)
            results.append(res)
        except Exception as e:
            print(f"[error] {pdf.name}: {e}")

    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize figures from scientific PDFs")
    parser.add_argument("path", help="Path to a PDF file or directory of PDFs")
    parser.add_argument("--all", action="store_true",
                        help="Process all PDFs in the directory (path must be a directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: outputs/<pdf_stem>)")
    args = parser.parse_args()

    p = Path(args.path)

    if args.all or p.is_dir():
        summarize_directory(p, args.output)
    else:
        summarize_paper(p, args.output)


if __name__ == "__main__":
    main()
