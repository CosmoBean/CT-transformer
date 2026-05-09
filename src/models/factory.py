"""
Model factory for supported classification backbones.
"""
from __future__ import annotations

import torch.nn as nn

from .sota_models import (
    EfficientNetClassifier,
    ResNetClassifier,
    SimpleCNNClassifier,
    SwinTransformerClassifier,
    VisionTransformerClassifier,
)


SUPPORTED_MODELS = {
    "simple_cnn",
    "efficientnet_b3",
    "resnet50",
    "vit_cnn_sized",
    "vit_base",
    "swin_base_patch4_window7_224",
}


def create_model(config: dict) -> nn.Module:
    model_name = config["model"]["name"]
    num_classes = int(config["model"]["num_classes"])
    pretrained = bool(config["model"].get("pretrained", True))
    image_size = int(config["data"]["image_size"])
    dropout = float(config["model"].get("dropout", 0.3))

    if model_name == "simple_cnn":
        return SimpleCNNClassifier(num_classes=num_classes, dropout=dropout)

    if model_name == "efficientnet_b3":
        return EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
        )

    if model_name == "resnet50":
        return ResNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            dropout=dropout,
        )

    if model_name == "vit_cnn_sized":
        return VisionTransformerClassifier(
            num_classes=num_classes,
            img_size=image_size,
            patch_size=16,
            embed_dim=128,
            depth=8,
            num_heads=8,
            mlp_ratio=2.0,
            dropout=dropout,
            pretrained=False,
            model_name="vit_custom",
        )

    if model_name == "vit_base":
        use_pretrained = pretrained and image_size == 224
        if pretrained and not use_pretrained:
            print(
                f"Warning: ViT pretrained weights require 224x224 input. "
                f"Using pretrained=False for {image_size}x{image_size}."
            )
        return VisionTransformerClassifier(
            num_classes=num_classes,
            img_size=image_size,
            pretrained=use_pretrained,
        )

    if model_name.startswith("swin"):
        return SwinTransformerClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            img_size=image_size,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Expected one of: {', '.join(sorted(SUPPORTED_MODELS))}"
    )
