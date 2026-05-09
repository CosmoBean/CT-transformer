"""
Library-first classification training workflow.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import build_classification_dataloaders
from src.models import SUPPORTED_MODELS, create_model
from src.training import Trainer
from src.utils import load_config


def load_training_config(
    config_path: str,
    model_name: str | None = None,
    epochs: int | None = None,
    save_dir: str | None = None,
    log_dir: str | None = None,
) -> dict:
    config = load_config(config_path)
    if model_name:
        config["model"]["name"] = model_name
    if epochs is not None:
        config["training"]["num_epochs"] = epochs
    if save_dir:
        config["save_dir"] = save_dir
    if log_dir:
        config["log_dir"] = log_dir
    return config


def resolve_device(config: dict) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


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


def train_classifier(config: dict) -> None:
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

    train_loader, val_loader = build_classification_dataloaders(config, device)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = create_model(config)
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.BCEWithLogitsLoss(),
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


def train_classifier_from_args(
    config_path: str,
    model_name: str | None = None,
    epochs: int | None = None,
    save_dir: str | None = None,
    log_dir: str | None = None,
) -> None:
    config = load_training_config(
        config_path=config_path,
        model_name=model_name,
        epochs=epochs,
        save_dir=save_dir,
        log_dir=log_dir,
    )
    train_classifier(config)
