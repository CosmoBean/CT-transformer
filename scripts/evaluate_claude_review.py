#!/usr/bin/env python3
"""
Thin CLI wrapper for review workflow evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.review import evaluate_review_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Claude-backed review workflow.")
    parser.add_argument("--review-config", default="configs/claude_review.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    summary = evaluate_review_run(
        review_config_path=args.review_config,
        split=args.split,
        max_cases=args.max_cases,
        output_dir=args.output_dir,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
