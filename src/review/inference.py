"""
Shared Swin and YOLO inference helpers for the review workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.dataset import CLASS_NAMES
from src.data.detection import derive_image_level_labels_from_detections
from src.data.transforms import build_classification_transform
from src.models import create_model
from src.utils import load_config


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO-backed review. Install the project dependencies first."
        ) from exc
    return YOLO


def _resolve_device(device_name: str | int | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_name, int):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_name}")
        return torch.device("cpu")
    requested = str(device_name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class SwinInferenceResult:
    probabilities: dict[str, float]
    predicted_labels: list[str]


class SwinInferenceEngine:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        threshold: float = 0.5,
        device_name: str | int | None = None,
    ):
        self.config = load_config(config_path)
        self.config["model"]["name"] = "swin_base_patch4_window7_224"
        self.config["data"]["mode"] = "classification"
        self.threshold = threshold
        self.device = _resolve_device(device_name or self.config.get("device", "auto"))
        self.model = create_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.transform = build_classification_transform(
            image_size=self.config["data"]["image_size"],
            is_train=False,
            use_augmentation=False,
        )

    def predict(self, image_path: str | Path) -> SwinInferenceResult:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(image_tensor)).cpu().numpy()[0]
        probability_map = {
            label: float(probability)
            for label, probability in zip(CLASS_NAMES, probabilities)
        }
        predicted_labels = [
            label
            for label, probability in probability_map.items()
            if probability >= self.threshold
        ]
        return SwinInferenceResult(
            probabilities=probability_map,
            predicted_labels=predicted_labels,
        )


@dataclass
class YoloInferenceResult:
    detections: list[dict[str, Any]]
    predicted_labels: list[str]


class YoloInferenceEngine:
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        conf_threshold: float | None = None,
        device_name: str | int | None = None,
    ):
        self.config = load_config(config_path)
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else self.config["evaluation"].get("conf_threshold", 0.25)
        )
        self.device_name = device_name if device_name is not None else self.config.get("device", 0)
        YOLO = _load_ultralytics()
        self.model = YOLO(weights_path)

    def predict(self, image_path: str | Path) -> YoloInferenceResult:
        result = self.model.predict(
            source=str(image_path),
            imgsz=self.config["data"]["image_size"],
            conf=self.conf_threshold,
            batch=1,
            device=self.device_name,
            verbose=False,
        )[0]
        names = result.names
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for box, confidence, class_id in zip(xyxy, confs, classes):
                class_id = int(class_id)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": names[class_id],
                        "confidence": float(confidence),
                        "bbox_xyxy": [float(value) for value in box],
                    }
                )
        predicted_map = derive_image_level_labels_from_detections(
            detections,
            confidence_threshold=self.conf_threshold,
        )
        predicted_labels = [label for label, value in predicted_map.items() if value == 1]
        return YoloInferenceResult(
            detections=detections,
            predicted_labels=predicted_labels,
        )
