"""
Vision-language model wrapper for figure summarization.

Uses BLIP-2 (Salesforce) via HuggingFace transformers.
The model takes a single figure image + an optional text prompt
(containing the caption) and generates a natural-language summary.

The context window is limited to the figure itself — no full-paper text.
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import config


class FigureSummarizer:
    """
    Wraps a BLIP-2 model for figure-to-text summarization.

    Usage
    -----
        summarizer = FigureSummarizer()
        summary = summarizer.summarize("fig.png", caption="Fig 1. …")
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id or config.VLM_MODEL_ID
        self.device = device or config.DEVICE
        self.dtype = dtype or config.DTYPE

        print(f"Loading BLIP-2 processor from {self.model_id}...")
        self.processor = Blip2Processor.from_pretrained(self.model_id)

        print(f"Loading BLIP-2 model ({self.dtype}) on {self.device}...")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map={"": self.device} if self.device != "cpu" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        print("FigureSummarizer ready.\n")

    # ──────────────────────────────────────────
    # Core inference
    # ──────────────────────────────────────────

    def _build_prompt(self, caption: str | None) -> str:
        """Build the text prompt that accompanies the image."""
        if caption and caption.strip():
            return config.SUMMARIZE_PROMPT.format(caption=caption.strip())
        return config.SUMMARIZE_PROMPT_NO_CAPTION

    def _load_image(self, image_input) -> Image.Image:
        """Accept path, PIL Image, or numpy array."""
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert("RGB")
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        # numpy array
        return Image.fromarray(image_input).convert("RGB")

    @torch.no_grad()
    def summarize(
        self,
        image_input,
        caption: str | None = None,
        max_new_tokens: int | None = None,
        num_beams: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a summary for a single figure.

        Parameters
        ----------
        image_input : str | Path | PIL.Image
            The figure image.
        caption : str | None
            The figure caption (extracted from the PDF).
            If provided, it is included in the prompt.
        max_new_tokens : int
            Max tokens to generate.
        num_beams : int
            Beam search width.
        temperature : float
            Sampling temperature (only used if do_sample=True).

        Returns
        -------
        str
            Generated summary text.
        """
        max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        num_beams = num_beams or config.NUM_BEAMS
        temperature = temperature or config.TEMPERATURE

        image = self._load_image(image_input)
        prompt = self._build_prompt(caption)

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=config.DO_SAMPLE,
        )

        summary = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return summary

    def summarize_batch(
        self,
        image_inputs: list,
        captions: list[str | None] | None = None,
        **kwargs,
    ) -> list[str]:
        """
        Summarize multiple figures sequentially.
        (BLIP-2 doesn't batch images of different sizes well,
        so we loop for reliability.)
        """
        captions = captions or [None] * len(image_inputs)
        return [
            self.summarize(img, cap, **kwargs)
            for img, cap in zip(image_inputs, captions)
        ]
