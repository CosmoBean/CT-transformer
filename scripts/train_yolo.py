#!/usr/bin/env python3
"""
Thin CLI wrapper for YOLO training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detection import train_yolo_run


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

    summary = train_yolo_run(
        config_path=args.config,
        epochs=args.epochs,
        weights=args.weights,
        batch_size=args.batch_size,
        project_dir=args.project_dir,
        run_name=args.run_name,
        dataset_output_dir=args.dataset_output_dir,
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps(summary, indent=2))
    print(f"YOLO training artifacts written to {summary['run_dir']}")


if __name__ == "__main__":
    main()
