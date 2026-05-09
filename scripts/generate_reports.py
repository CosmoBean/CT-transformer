#!/usr/bin/env python3
"""
Generate reproducible report examples from cached review outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_example_reports import generate_example_reports
from scripts.generate_presentation_comparison_reports import generate_comparison_reports


TEMPLATES = {
    "example": "Doctor-facing report without ground-truth comparison.",
    "comparison": "Presentation report comparing pipeline output against ground truth.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cached report examples with a named template.")
    parser.add_argument("--template", choices=sorted(TEMPLATES), default=None)
    parser.add_argument("--list-templates", action="store_true", help="Print available templates and exit.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/claude_review/cache"))
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("experiments/claude_review/eval_300/claude_vs_baselines_case_comparison.csv"),
    )
    parser.add_argument("--raw-annotation-path", type=Path, default=Path("data/_downloads/train_raw.csv"))
    parser.add_argument("--image-metadata-path", type=Path, default=Path("data/_downloads/vinbig_png/train_meta.csv"))
    return parser


def _print_templates() -> None:
    for name, description in TEMPLATES.items():
        print(f"{name}: {description}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_templates:
        _print_templates()
        return

    if args.template is None:
        parser.error("--template is required unless --list-templates is used.")

    if args.template == "example":
        output_dir = args.output_dir or Path("reports/example_reports")
        count = generate_example_reports(
            cache_dir=args.cache_dir,
            output_dir=output_dir,
            case_ids=args.case_ids,
        )
        print(f"Generated {count} example reports in {output_dir}")
        return

    output_dir = args.output_dir or Path("reports/comparision_reports")
    count = generate_comparison_reports(
        comparison_csv=args.comparison_csv,
        cache_dir=args.cache_dir,
        raw_annotation_path=args.raw_annotation_path,
        image_metadata_path=args.image_metadata_path,
        output_dir=output_dir,
    )
    print(f"Generated {count} comparison reports in {output_dir}")


if __name__ == "__main__":
    main()
