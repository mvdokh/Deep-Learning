"""
Utilities to export the trained contact classifier to a Torch 2.x
`.pt2` artifact using AOT export (AOTInductor backend).

The exported program can optionally use a dynamic batch dimension so
that you can run offline inference with variable batch sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


def _import_build_model() -> Callable[..., nn.Module]:
    """
    Import `build_model` from the Training utilities without requiring
    directory names to be valid Python package names.
    """
    this_dir = Path(__file__).resolve().parent
    training_dir = this_dir.parent / "Training"
    if str(training_dir) not in sys.path:
        sys.path.append(str(training_dir))

    from trainer import build_model  # type: ignore

    return build_model


def load_trained_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Rebuild the EfficientNet-B3 contact classifier and load weights
    from a training checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    build_model = _import_build_model()

    device = torch.device(device)
    model = build_model(num_classes=1, pretrained=False, dropout=dropout)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _get_export_api():
    """
    Return (export_fn, dynamic_dim) supporting multiple Torch versions.

    On older Torch versions where dynamic shapes are not available,
    `dynamic_dim` is returned as None and only static-shape export is
    supported.
    """
    # Prefer the public torch.export API when available.
    try:
        from torch import export as export_mod  # type: ignore[attr-defined]

        export_fn = export_mod.export  # type: ignore[attr-defined]
        dynamic_dim = getattr(export_mod, "dynamic_dim", None)
        return export_fn, dynamic_dim
    except Exception:  # pragma: no cover
        pass

    # Fallback to internal torch._export (API surface varies by version).
    try:
        import torch._export as _export  # type: ignore

        if hasattr(_export, "export"):
            export_fn = _export.export  # type: ignore[attr-defined]
        elif hasattr(_export, "aot_export"):
            export_fn = _export.aot_export  # type: ignore[attr-defined]
        else:
            raise ImportError("No suitable export function in torch._export")

        dynamic_dim = getattr(_export, "dynamic_dim", None)
        return export_fn, dynamic_dim
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not locate a compatible Torch export API. "
            "Please upgrade PyTorch to 2.1+ or disable AOT export."
        ) from exc


def export_contact_classifier_aot(
    checkpoint_path: str | Path,
    export_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    img_size: int = 256,
    channels: int = 3,
    static_batch_size: int = 1,
    dynamic_batch: bool = True,
    dynamic_min_batch: int = 1,
    dynamic_max_batch: int = 32,
    dropout: float = 0.3,
) -> Path:
    """
    Export the contact classifier as an AOTInductor `.pt2` program.

    Parameters
    ----------
    checkpoint_path:
        Path to the training checkpoint (`.pt`).
    export_path:
        Output path for the exported program (`.pt2` is recommended).
    device:
        Device used during export ("cuda" or "cpu").
    img_size:
        Input image size (H=W) expected by the model (default 256).
    channels:
        Number of image channels (default 3 for RGB).
    static_batch_size:
        Batch size to bake into the export when `dynamic_batch=False`.
    dynamic_batch:
        If True, export with a dynamic batch dimension.
    dynamic_min_batch / dynamic_max_batch:
        Allowed range for the dynamic batch dimension.
    dropout:
        Dropout used in the classifier head; should match training.
    """
    device = torch.device(device)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_trained_model(
        checkpoint_path=checkpoint_path,
        device=device,
        dropout=dropout,
    )

    export_fn, dynamic_dim = _get_export_api()

    if dynamic_batch:
        example_bs = max(dynamic_min_batch, 1)
    else:
        example_bs = max(static_batch_size, 1)

    example_input = torch.randn(
        example_bs,
        channels,
        img_size,
        img_size,
        device=device,
    )

    inputs = (example_input,)

    if dynamic_batch:
        if dynamic_dim is None:
            raise RuntimeError(
                "Dynamic batch export is not supported by this PyTorch version. "
                "Set dynamic_batch=False (USE_DYNAMIC_BATCH = False in the notebook) "
                "or upgrade to a newer Torch release with dynamic export support."
            )

        batch_dyn = dynamic_dim(example_input, 0, min=dynamic_min_batch, max=dynamic_max_batch)
        dynamic_shapes = {example_input: {0: batch_dyn}}
        ep = export_fn(model, inputs, dynamic_shapes=dynamic_shapes)
    else:
        ep = export_fn(model, inputs)

    ep.save(str(export_path))
    return export_path


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Export contact classifier to Torch 2.x AOT `.pt2` format.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for the exported program (.pt2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use during export (cuda or cpu).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
        help="Input image size (H=W).",
    )
    parser.add_argument(
        "--static-batch-size",
        type=int,
        default=1,
        help="Static batch size when not using dynamic batch.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch dimension.",
    )
    parser.add_argument(
        "--dynamic-min-batch",
        type=int,
        default=1,
        help="Minimum dynamic batch size.",
    )
    parser.add_argument(
        "--dynamic-max-batch",
        type=int,
        default=32,
        help="Maximum dynamic batch size.",
    )

    args = parser.parse_args()
    export_contact_classifier_aot(
        checkpoint_path=args.checkpoint,
        export_path=args.out,
        device=args.device,
        img_size=args.img_size,
        static_batch_size=args.static_batch_size,
        dynamic_batch=args.dynamic_batch,
        dynamic_min_batch=args.dynamic_min_batch,
        dynamic_max_batch=args.dynamic_max_batch,
    )
