"""
High-level dataloader helpers for classification workflows.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import ChestXRayDataset


def build_classification_dataloaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
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
