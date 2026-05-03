"""
VinBigData-style datasets for classification and anomaly detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_classification_transform


CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    labels: np.ndarray


class ChestXRayDataset(Dataset):
    """
    Multi-label chest X-ray dataset backed by a folder of images and a CSV file.
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: Optional[str] = None,
        image_size: int = 224,
        split: str = "train",
        mode: str | None = None,
        use_augmentation: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        del mode  # Backward-compatible no-op: this repo now has one classification dataset path.
        self.data_dir = Path(data_dir)
        self.csv_path = Path(csv_path) if csv_path else None
        self.image_size = image_size
        self.split = split
        self.use_augmentation = use_augmentation
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed

        self.image_root = self._resolve_image_root(split)
        self.labels_df = self._load_labels(self.csv_path)
        self.samples = self._build_samples()
        self.transform = build_classification_transform(
            image_size=image_size,
            is_train=(split == "train"),
            use_augmentation=use_augmentation,
        )

    def _resolve_image_root(self, split: str) -> Path:
        requested = self.data_dir / split
        if requested.exists():
            return requested

        fallback = self.data_dir / "train"
        if fallback.exists():
            return fallback

        raise FileNotFoundError(
            f"Could not find image directory for split '{split}' in {self.data_dir}"
        )

    def _load_labels(self, csv_path: Optional[Path]) -> Optional[pd.DataFrame]:
        if csv_path is None:
            return None
        if not csv_path.exists():
            raise FileNotFoundError(f"Label CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "image_id" not in df.columns:
            raise ValueError("Label CSV must include an 'image_id' column.")

        label_columns = [column for column in df.columns if column != "image_id"]
        if not label_columns:
            raise ValueError("Label CSV must include at least one label column.")

        missing_columns = [column for column in CLASS_NAMES if column not in label_columns]
        if missing_columns:
            raise ValueError(
                "Label CSV is missing required class columns: "
                + ", ".join(missing_columns)
            )

        return df.set_index("image_id")[CLASS_NAMES]

    def _build_samples(self) -> list[Sample]:
        image_paths = sorted(
            [
                path
                for path in self.image_root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.image_root}")

        labels_by_id: dict[str, np.ndarray] = {}
        if self.labels_df is not None:
            for image_id, row in self.labels_df.iterrows():
                labels_by_id[str(image_id)] = row.astype(np.float32).to_numpy()

        all_samples = []
        for path in image_paths:
            image_id = path.stem
            labels = labels_by_id.get(
                image_id,
                np.zeros(len(CLASS_NAMES), dtype=np.float32),
            )
            all_samples.append(Sample(image_id=image_id, image_path=path, labels=labels))

        split_indices = self._split_indices(len(all_samples))
        return [all_samples[index] for index in split_indices]

    def _split_indices(self, num_samples: int) -> np.ndarray:
        indices = np.arange(num_samples)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        train_end = int(num_samples * self.train_split)
        if train_end <= 0 or train_end >= num_samples:
            return indices

        if self.split == "train":
            return indices[:train_end]
        if self.split == "val":
            return indices[train_end:]
        return indices

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "labels": torch.tensor(sample.labels, dtype=torch.float32),
            "image_id": sample.image_id,
        }
