#!/usr/bin/env python3
"""
Run single-image inference with a YOLO checkpoint and derive image-level labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.detection import derive_image_level_labels_from_detections
from src.utils import load_config


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO inference. Install the project dependencies first."
        ) from exc
    return YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf-threshold", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    YOLO = _load_ultralytics()
    conf_threshold = args.conf_threshold
    if conf_threshold is None:
        conf_threshold = config["evaluation"].get("conf_threshold", 0.25)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.image,
        imgsz=config["data"]["image_size"],
        conf=conf_threshold,
        device=config.get("device", 0),
        verbose=False,
    )
    result = results[0]
    names = result.names
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        for box, confidence, class_id in zip(xyxy, confs, classes):
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": names[int(class_id)],
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )

    image_level = derive_image_level_labels_from_detections(
        detections,
        confidence_threshold=conf_threshold,
    )
    payload = {
        "image": str(Path(args.image)),
        "weights": str(Path(args.weights)),
        "confidence_threshold": conf_threshold,
        "detections": detections,
        "image_level_labels": image_level,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
