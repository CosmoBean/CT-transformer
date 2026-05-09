#!/usr/bin/env python3
"""
Thin CLI wrapper for classification training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import train_classifier_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one VinBigData classifier.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default=None, help="Override model name from config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config.")
    parser.add_argument("--save-dir", default=None, help="Override checkpoint directory.")
    parser.add_argument("--log-dir", default=None, help="Override log directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_classifier_from_args(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
