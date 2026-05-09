#!/usr/bin/env python3
"""
Thin CLI wrapper for the single-image agentic review workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.review import run_review_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked agentic CXR reporting workflow.")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image (PNG/JPG/JPEG).")
    parser.add_argument("--image-id", default=None, help="Optional case id override. Defaults to file stem.")
    parser.add_argument("--review-config", default="configs/claude_review.yaml")
    parser.add_argument(
        "--output-dir",
        default="experiments/agentic_reports",
        help="Base output directory for generated case folders.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached Claude result and recompute.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    summary = run_review_case(
        image_path=str(image_path),
        review_config_path=args.review_config,
        output_dir=args.output_dir,
        image_id=args.image_id,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
