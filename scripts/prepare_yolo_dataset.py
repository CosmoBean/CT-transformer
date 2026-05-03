#!/usr/bin/env python3
"""
Prepare a YOLO-formatted detection dataset from the raw VinBigData annotations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.detection import prepare_yolo_dataset
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO detection data for VinBigData.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["data"]["output_dir"]
    link_mode = args.link_mode or config["data"].get("link_mode", "symlink")

    metadata = prepare_yolo_dataset(
        image_root=config["data"]["image_root"],
        raw_annotation_path=config["data"]["raw_annotation_path"],
        image_metadata_path=config["data"]["image_metadata_path"],
        output_dir=output_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
        merge_iou_threshold=config["data"].get("merge_iou_threshold", 0.3),
        link_mode=link_mode,
        max_images_per_split=args.max_images_per_split,
    )

    print(json.dumps(metadata, indent=2))
    print(f"Prepared YOLO dataset at {output_dir}")


if __name__ == "__main__":
    main()
