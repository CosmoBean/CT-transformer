#!/usr/bin/env python3
"""
Thin CLI wrapper for YOLO evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detection import evaluate_yolo_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection and image-level metrics.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--compare-swin-checkpoint", default="experiments/agent_swin/checkpoints/best_model.pth")
    parser.add_argument("--output-dir", default="experiments/yolo_v8/reports")
    parser.add_argument("--dataset-output-dir", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    args = parser.parse_args()

    summary = evaluate_yolo_run(
        config_path=args.config,
        weights_path=args.weights,
        split=args.split,
        conf_threshold=args.conf_threshold,
        compare_swin_checkpoint=args.compare_swin_checkpoint,
        output_dir=args.output_dir,
        dataset_output_dir=args.dataset_output_dir,
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
