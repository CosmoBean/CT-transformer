"""
Inference helpers for trained classification checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES, ChestXRayDataset
from src.models import create_model


def _resolve_device(device_name: str | torch.device | None) -> torch.device:
    if isinstance(device_name, torch.device):
        return device_name
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = str(device_name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def predict_classifier_outputs(
    config: dict,
    checkpoint_path: str | Path,
    split: str,
    threshold: float = 0.5,
    device_name: str | torch.device | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = _resolve_device(device_name)
    data_dir = Path(config["data"]["data_dir"])
    csv_path = data_dir / config["data"].get("train_csv", "train.csv")

    dataset = ChestXRayDataset(
        data_dir=str(data_dir),
        csv_path=str(csv_path),
        image_size=int(config["data"]["image_size"]),
        split=split,
        mode="classification",
        use_augmentation=False,
        train_split=float(config["data"].get("train_split", 0.8)),
        val_split=float(config["data"].get("val_split", 0.2)),
        seed=int(config.get("seed", 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(config["data"].get("batch_size", 32)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(config).to(device)
    model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    model.eval()

    probability_rows = []
    prediction_rows = []
    with torch.no_grad():
        for batch in loader:
            probabilities = torch.sigmoid(model(batch["image"].to(device))).cpu().numpy()
            predictions = (probabilities >= threshold).astype(int)
            for image_id, probability_values, predicted_values in zip(
                batch["image_id"],
                probabilities,
                predictions,
            ):
                probability_rows.append(
                    {
                        "image_id": image_id,
                        **{
                            class_name: float(value)
                            for class_name, value in zip(CLASS_NAMES, probability_values)
                        },
                    }
                )
                prediction_rows.append(
                    {
                        "image_id": image_id,
                        **{
                            class_name: int(value)
                            for class_name, value in zip(CLASS_NAMES, predicted_values)
                        },
                    }
                )
    return (
        pd.DataFrame(probability_rows).set_index("image_id"),
        pd.DataFrame(prediction_rows).set_index("image_id"),
    )


def predict_classifier_dataset(
    config: dict,
    checkpoint_path: str | Path,
    split: str,
    threshold: float = 0.5,
    device_name: str | torch.device | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> pd.DataFrame:
    _, prediction_df = predict_classifier_outputs(
        config=config,
        checkpoint_path=checkpoint_path,
        split=split,
        threshold=threshold,
        device_name=device_name,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return prediction_df
