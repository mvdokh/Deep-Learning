"""
Model builder for car recognition.

Uses the `timm` library to load any pretrained backbone (EfficientNet, ResNet,
ConvNeXt, ViT, etc.) and replaces the classification head to match the number
of car classes defined in config.py.
"""

import timm
import torch
import torch.nn as nn

import config


def build_model(
    model_name: str | None = None,
    num_classes: int | None = None,
    pretrained: bool | None = None,
    freeze_backbone: bool | None = None,
) -> nn.Module:
    """
    Build and return a classification model.

    Parameters
    ----------
    model_name : str
        Any model name supported by `timm.create_model`.
    num_classes : int
        Number of output classes.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freeze all layers except the final classifier head.

    Returns
    -------
    nn.Module
    """
    model_name = model_name or config.MODEL_NAME
    num_classes = num_classes or config.NUM_CLASSES
    pretrained = pretrained if pretrained is not None else config.PRETRAINED
    freeze_backbone = freeze_backbone if freeze_backbone is not None else config.FREEZE_BACKBONE

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    if freeze_backbone:
        _freeze_backbone(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}  |  Total params: {total:,}  |  Trainable: {trainable:,}")

    return model


def _freeze_backbone(model: nn.Module):
    """Freeze everything except the classifier head."""
    # timm models expose .get_classifier() which returns the head module
    classifier = model.get_classifier()
    classifier_params = set(id(p) for p in classifier.parameters())

    for param in model.parameters():
        if id(param) not in classifier_params:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters so the full model can be fine-tuned."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone unfrozen  |  Trainable params: {trainable:,}")


def load_checkpoint(path: str, model: nn.Module | None = None) -> nn.Module:
    """Load a saved checkpoint. If model is None, build one first."""
    checkpoint = torch.load(path, map_location=config.DEVICE, weights_only=False)

    if model is None:
        model = build_model(pretrained=False, freeze_backbone=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {path}  (epoch {checkpoint.get('epoch', '?')})")
    return model
