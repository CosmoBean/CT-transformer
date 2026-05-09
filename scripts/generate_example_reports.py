#!/usr/bin/env python3
"""
Generate doctor-facing example reports for selected cached review cases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.review.pdf import export_pdf


DEFAULT_CASE_IDS = [
    "302b0d070d5150b91bafc935e94a847b",
    "804bcde30d36e32d9429f00bed7a388d",
    "a8750c349b5dac834473304bad0f2877",
    "ba5e3409250a85483d6e39be759bc102",
    "305e4add9c72c91e9984305bf4e85aee",
]


def generate_example_reports(
    cache_dir: Path = Path("experiments/claude_review/cache"),
    output_dir: Path = Path("reports/example_reports"),
    case_ids: list[str] | None = None,
) -> int:
    case_ids = case_ids or DEFAULT_CASE_IDS
    output_dir.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        report_json = cache_dir / f"{case_id}_claude_review.json"
        if not report_json.exists():
            raise FileNotFoundError(f"Missing cached review JSON for case {case_id}: {report_json}")
        export_pdf(report_json, output_dir / f"{case_id}.pdf")
    return len(case_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate example doctor-facing reports from cached review outputs.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/claude_review/cache"),
        help="Directory containing *_claude_review.json cache files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/example_reports"),
        help="Destination for generated PDFs.",
    )
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        help="Specific case id to generate. Repeat to override the default five cases.",
    )
    args = parser.parse_args()

    count = generate_example_reports(cache_dir=args.cache_dir, output_dir=args.output_dir, case_ids=args.case_ids)
    print(f"Generated {count} example reports in {args.output_dir}")


if __name__ == "__main__":
    main()
