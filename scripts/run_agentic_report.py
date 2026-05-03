#!/usr/bin/env python3
"""
Locked end-to-end agentic workflow for a single chest X-ray image.

Input:
- chest X-ray image path (PNG/JPG/JPEG)

Outputs in one case folder:
- review_result.json
- report.md
- report.pdf
- *_annotated.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.export_review_pdfs import export_pdf
from src.review import ReviewOrchestrator, SwinInferenceEngine, YoloInferenceEngine
from src.utils import load_config


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

    review_config = load_config(args.review_config)
    paths_config = review_config["paths"]
    review_section = review_config["review"]

    image_id = args.image_id or image_path.stem
    case_dir = Path(args.output_dir) / image_id
    case_dir.mkdir(parents=True, exist_ok=True)

    swin_engine = SwinInferenceEngine(
        config_path=paths_config["swin_config"],
        checkpoint_path=paths_config["swin_checkpoint"],
        threshold=float(review_section.get("swin_threshold", 0.5)),
    )
    yolo_engine = YoloInferenceEngine(
        config_path=paths_config["yolo_config"],
        weights_path=paths_config["yolo_weights"],
        conf_threshold=float(review_section.get("yolo_conf_threshold", 0.25)),
    )
    orchestrator = ReviewOrchestrator(
        review_config=review_config,
        prompt_path=paths_config["prompt_path"],
        swin_engine=swin_engine,
        yolo_engine=yolo_engine,
    )

    result = orchestrator.review_case(
        image_id=image_id,
        image_path=image_path,
        cache_dir=Path(review_section["cache_dir"]),
        force_refresh=args.force_refresh,
    )

    review_json_path = case_dir / "review_result.json"
    report_md_path = case_dir / "report.md"
    review_json_path.write_text(json.dumps(result, indent=2))
    report_md_path.write_text(result["report_text"])

    export_pdf(review_json_path, case_dir / "report.pdf")

    summary = {
        "image_id": image_id,
        "image_path": str(image_path),
        "output_dir": str(case_dir),
        "final_labels": result["review"]["final_labels"],
        "supported_findings": result["review"]["supported_findings"],
        "uncertain_findings": result["review"]["uncertain_findings"],
        "review_recommendation": result["review"]["review_recommendation"],
        "confidence_band": result["review"]["confidence_band"],
        "artifacts": {
            "review_json": str(review_json_path),
            "report_markdown": str(report_md_path),
            "report_pdf": str(case_dir / "report.pdf"),
            "annotated_png": str(case_dir / "report_annotated.png"),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
