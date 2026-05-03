#!/usr/bin/env python3
"""
Train one chest X-ray multi-label classifier on the VinBigData subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import ChestXRayDataset
from src.models import (
    EfficientNetClassifier,
    ResNetClassifier,
    SimpleCNNClassifier,
    SwinTransformerClassifier,
    VisionTransformerClassifier,
)
from src.training import Trainer
from src.utils import load_config


SUPPORTED_MODELS = {
    "simple_cnn",
    "efficientnet_b3",
    "resnet50",
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


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(config["data"]["data_dir"])
    train_csv_name = config["data"].get("train_csv", "train.csv")
    csv_path = data_dir / train_csv_name
    csv_path_str = str(csv_path) if csv_path.exists() else None

    dataset_kwargs = {
        "data_dir": str(data_dir),
        "csv_path": csv_path_str,
        "image_size": int(config["data"]["image_size"]),
        "train_split": float(config["data"].get("train_split", 0.8)),
        "val_split": float(config["data"].get("val_split", 0.2)),
        "seed": int(config.get("seed", 42)),
    }

    train_dataset = ChestXRayDataset(
        split="train",
        use_augmentation=bool(config["data"].get("use_augmentation", False)),
        **dataset_kwargs,
    )
    val_dataset = ChestXRayDataset(
        split="val",
        use_augmentation=False,
        **dataset_kwargs,
    )

    loader_kwargs = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def create_optimizer(config: dict, model: nn.Module) -> optim.Optimizer:
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer_name = str(config["training"]["optimizer"]).lower()

    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def create_scheduler(config: dict, optimizer: optim.Optimizer):
    scheduler_name = str(config["training"]["scheduler"]).lower()
    epochs = int(config["training"]["num_epochs"])

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one VinBigData classifier.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default=None, help="Override model name from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config.")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint directory.")
    parser.add_argument("--log-dir", default=None, help="Override log directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.save_dir:
        config["save_dir"] = args.save_dir
    if args.log_dir:
        config["log_dir"] = args.log_dir

    model_name = config["model"]["name"]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_MODELS))}"
        )

    device = resolve_device(config)
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(config, device)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(config)
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        save_dir=config["save_dir"],
        log_dir=config["log_dir"],
        use_multi_gpu=bool(config.get("use_multi_gpu", False)),
    )
    trainer.train(
        num_epochs=int(config["training"]["num_epochs"]),
        save_best=bool(config["training"].get("save_best", True)),
        metric_name=str(config["training"].get("metric_name", "auc_roc_macro")),
    )


if __name__ == "__main__":
    main()
