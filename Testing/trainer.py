"""
trainer.py — PyTorch training framework for fine-tuning BART on scientific papers.

Features
--------
-   Standard seq2seq cross-entropy loss (label-smoothed optionally)
-   AdamW optimiser with linear warm-up + cosine decay schedule
-   Gradient accumulation for effective large batch sizes
-   Mixed-precision (fp16 / bf16) via ``torch.amp``
-   ROUGE-1 / ROUGE-2 / ROUGE-L evaluation after each epoch
-   Best-model checkpointing (by ROUGE-L)
-   TensorBoard logging (optional)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from rouge_score import rouge_scorer  # type: ignore
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Every hyper-parameter in one place."""

    # Model
    model_name: str = "facebook/bart-large-cnn"

    # Optimiser
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_ratio: float = 0.06  # fraction of total steps for warm-up
    num_epochs: int = 5

    # Batching
    batch_size: int = 2
    gradient_accumulation_steps: int = 8  # effective batch = 2 * 8 = 16
    num_workers: int = 0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Generation (for evaluation)
    eval_max_tokens: int = 256
    eval_min_tokens: int = 56
    eval_num_beams: int = 4

    # Checkpointing & logging
    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    log_every_n_steps: int = 10
    use_tensorboard: bool = True

    # Data split
    val_fraction: float = 0.1  # fraction of data held out for validation

    # Device
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # MPS does not support float16 AMP well — fall back
        if self.device == "mps" and self.amp_dtype == "float16":
            self.amp_dtype = "bfloat16"


# ---------------------------------------------------------------------------
# ROUGE evaluation
# ---------------------------------------------------------------------------

def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Return ROUGE-1 / 2 / L F-scores averaged over the batch."""
    if not _HAS_ROUGE:
        return {}
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1, r2, rl = 0.0, 0.0, 0.0
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rl += scores["rougeL"].fmeasure
    n = max(len(predictions), 1)
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BartTrainer:
    """
    Encapsulates the full training loop for BART fine-tuning.

    Usage
    -----
    >>> trainer = BartTrainer(dataset, config)
    >>> trainer.train()
    """

    def __init__(
        self,
        dataset: Dataset,
        config: TrainConfig | None = None,
    ) -> None:
        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)

        # ---- model & tokenizer ----
        print(f"[BartTrainer] Loading '{self.cfg.model_name}' …")
        self.tokenizer = BartTokenizer.from_pretrained(self.cfg.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.cfg.model_name
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[BartTrainer] Model on {self.device} ({n_params:.1f}M params)")

        # ---- data split ----
        val_size = max(1, int(len(dataset) * self.cfg.val_fraction))
        train_size = len(dataset) - val_size
        self.train_ds, self.val_ds = random_split(dataset, [train_size, val_size])
        print(
            f"[BartTrainer] Train: {train_size}  |  Val: {val_size}"
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        # ---- optimiser & schedule ----
        total_steps = (
            len(self.train_loader)
            // self.cfg.gradient_accumulation_steps
            * self.cfg.num_epochs
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            eps=self.cfg.adam_epsilon,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ---- AMP scaler ----
        self.use_amp = self.cfg.use_amp and self.device.type in ("cuda", "mps")
        self.amp_dtype = (
            torch.bfloat16 if self.cfg.amp_dtype == "bfloat16" else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )

        # ---- checkpointing ----
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_rouge_l = 0.0

        # ---- TensorBoard ----
        self.writer = None
        if self.cfg.use_tensorboard and _HAS_TB:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        global_step = 0
        for epoch in range(1, self.cfg.num_epochs + 1):
            global_step = self._train_one_epoch(epoch, global_step)
            val_metrics = self._evaluate(epoch)

            # checkpoint
            rouge_l = val_metrics.get("rougeL", 0.0)
            if rouge_l > self.best_rouge_l:
                self.best_rouge_l = rouge_l
                self._save_checkpoint("best_model")
                print(f"  ** New best ROUGE-L: {rouge_l:.4f} — saved best_model **")

            if epoch % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch}")

        print("\n[BartTrainer] Training complete.")
        if self.writer:
            self.writer.close()

    def _train_one_epoch(self, epoch: int, global_step: int) -> int:
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        t0 = time.perf_counter()
        for step, batch in enumerate(self.train_loader, 1):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(**batch)
                loss = outputs.loss / self.cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            epoch_loss += loss.item() * self.cfg.gradient_accumulation_steps

            if step % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                if global_step % self.cfg.log_every_n_steps == 0:
                    avg = epoch_loss / step
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  [epoch {epoch}  step {global_step}]  "
                        f"loss={avg:.4f}  lr={lr:.2e}  "
                        f"({elapsed:.0f}s elapsed)"
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg, global_step)
                        self.writer.add_scalar("train/lr", lr, global_step)

        avg_loss = epoch_loss / max(len(self.train_loader), 1)
        print(f"  Epoch {epoch} train loss: {avg_loss:.4f}")
        if self.writer:
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        return global_step

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[str] = []
        all_refs: List[str] = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(**batch)
            total_loss += outputs.loss.item()

            # Generate summaries for ROUGE
            gen_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.cfg.eval_max_tokens,
                min_length=self.cfg.eval_min_tokens,
                num_beams=self.cfg.eval_num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            preds = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_preds.extend(preds)

            # Decode labels (replace -100 → pad for decoding)
            labels = batch["labels"].clone()
            labels[labels == -100] = self.tokenizer.pad_token_id
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_refs.extend(refs)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        rouge = compute_rouge(all_preds, all_refs)

        print(
            f"  Epoch {epoch} val loss: {avg_loss:.4f}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in rouge.items())
        )
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
            for k, v in rouge.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

        return rouge

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        print(f"  Checkpoint saved → {path}")

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_summary(self, text: str) -> str:
        """Generate a summary from raw text using the current model weights."""
        self.model.eval()
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.cfg.eval_max_tokens,
            min_length=self.cfg.eval_min_tokens,
            num_beams=self.cfg.eval_num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)
