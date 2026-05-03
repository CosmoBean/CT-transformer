#!/usr/bin/env python3
"""
Train a YOLOv8 detector on the VinBigData bounding-box annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.detection import prepare_yolo_dataset
from src.utils import load_config, save_config


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO training. Install the project dependencies "
            "with `make install` or `source .venv/bin/activate && pip install -e .`."
        ) from exc
    return YOLO


def _extract_detection_metrics(metrics) -> dict:
    box_metrics = getattr(metrics, "box", None)
    return {
        "map50": float(getattr(box_metrics, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box_metrics, "map", 0.0) or 0.0),
        "maps": [float(value) for value in getattr(box_metrics, "maps", [])] if box_metrics is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VinBigData.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-output-dir", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    YOLO = _load_ultralytics()

    dataset_output_dir = args.dataset_output_dir or config["data"]["output_dir"]
    prepare_yolo_dataset(
        image_root=config["data"]["image_root"],
        raw_annotation_path=config["data"]["raw_annotation_path"],
        image_metadata_path=config["data"]["image_metadata_path"],
        output_dir=dataset_output_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
        merge_iou_threshold=config["data"].get("merge_iou_threshold", 0.3),
        link_mode=config["data"].get("link_mode", "symlink"),
        max_images_per_split=args.max_images_per_split,
    )

    dataset_yaml = Path(dataset_output_dir) / "dataset.yaml"
    weights = args.weights or config["model"]["weights"]
    epochs = args.epochs or config["training"]["num_epochs"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    project_dir = str(Path(args.project_dir or config["training"]["project_dir"]).resolve())
    run_name = args.run_name or config["training"]["run_name"]

    model = YOLO(weights)
    model.train(
        data=str(dataset_yaml),
        imgsz=config["data"]["image_size"],
        epochs=epochs,
        batch=batch_size,
        workers=config["training"].get("num_workers", 4),
        patience=config["training"].get("patience", 10),
        project=project_dir,
        name=run_name,
        device=config.get("device", 0),
        pretrained=True,
        verbose=True,
    )

    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=config["data"]["image_size"],
        batch=batch_size,
        device=config.get("device", 0),
        conf=config["evaluation"].get("conf_threshold", 0.25),
        verbose=False,
    )
    summary = _extract_detection_metrics(metrics)

    run_dir = Path(project_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detection_metrics.json").write_text(json.dumps(summary, indent=2))
    save_config(config, str(run_dir / "resolved_config.yaml"))

    print(json.dumps(summary, indent=2))
    print(f"YOLO training artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
