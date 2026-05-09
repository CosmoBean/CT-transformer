#!/usr/bin/env python3
"""
Thin CLI wrapper for YOLO single-image inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.detection import infer_yolo_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO inference on one image.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--conf-threshold", type=float, default=None)
    args = parser.parse_args()

    payload = infer_yolo_image(
        config_path=args.config,
        weights_path=args.weights,
        image_path=args.image,
        conf_threshold=args.conf_threshold,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
