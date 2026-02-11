"""
Configuration for the Figure Summarizer pipeline.
"""

import torch
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PAPERS_DIR = Path("/Users/martindokholyan/Desktop/Spring 26/Readings/papers")

# ──────────────────────────────────────────────
# Figure extraction settings
# ──────────────────────────────────────────────
MIN_IMAGE_WIDTH = 200       # px — skip tiny icons / logos
MIN_IMAGE_HEIGHT = 200      # px
MIN_IMAGE_AREA = 50_000     # px² — skip small decorative images
CAPTION_SEARCH_LINES = 8    # how many text lines below an image to scan for caption
CAPTION_PREFIXES = (        # strings that start a figure caption
    "Fig.", "Figure", "FIG.", "FIGURE",
    "Fig ", "Figure ",
)

# ──────────────────────────────────────────────
# Vision-language model
# ──────────────────────────────────────────────
# Using BLIP-2 via HuggingFace transformers.
# The model takes an image + optional text prompt and generates text.
# Only the figure image (+ its caption) enters the context window.
VLM_MODEL_ID = "Salesforce/blip2-opt-2.7b"      # good balance of quality vs. VRAM
# Alternatives:
#   "Salesforce/blip2-opt-6.7b"    — higher quality, needs ~16 GB VRAM
#   "Salesforce/blip2-flan-t5-xl"  — flan backbone, good for instruction following

# Generation settings
MAX_NEW_TOKENS = 256
NUM_BEAMS = 4
TEMPERATURE = 0.7
DO_SAMPLE = False           # greedy / beam search when False

# ──────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────
# {caption} is replaced with the extracted figure caption.
SUMMARIZE_PROMPT = (
    "This is a scientific figure. "
    "Caption: {caption} "
    "Provide a concise summary of what this figure shows, "
    "the key findings, and their significance."
)

SUMMARIZE_PROMPT_NO_CAPTION = (
    "This is a scientific figure from a research paper. "
    "Describe what this figure shows and summarize the key findings."
)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
