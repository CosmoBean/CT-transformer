#!/usr/bin/env python3
"""
Non-network smoke tests for Claude review schema, prompt rendering, and report output.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.review import build_case_packet, load_prompt_template, render_review_report, render_user_prompt
from src.review.schema import extract_json_object, validate_review_response


def main() -> None:
    prompt = load_prompt_template("prompts/claude_radiology_reviewer_prompt.md")
    packet = build_case_packet(
        image_id="demo",
        image_path="data/train/demo.png",
        swin_probabilities={
            "Aortic enlargement": 0.72,
            "Atelectasis": 0.05,
            "Calcification": 0.02,
            "Cardiomegaly": 0.81,
            "Consolidation": 0.01,
            "ILD": 0.03,
            "Infiltration": 0.07,
            "Lung Opacity": 0.11,
            "Nodule/Mass": 0.03,
            "Other lesion": 0.04,
            "Pleural effusion": 0.08,
            "Pleural thickening": 0.06,
            "Pneumothorax": 0.01,
            "Pulmonary fibrosis": 0.09,
            "No finding": 0.02,
        },
        swin_predicted_labels=["Aortic enlargement", "Cardiomegaly"],
        yolo_detections=[
            {
                "class_id": 0,
                "class_name": "Aortic enlargement",
                "confidence": 0.55,
                "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            }
        ],
        yolo_predicted_labels=["Aortic enlargement"],
        yolo_conf_threshold=0.25,
    )
    rendered_prompt = render_user_prompt(prompt, packet)
    assert "Aortic enlargement" in rendered_prompt
    assert "Cardiomegaly" in rendered_prompt
    assert "No acute abnormality" in rendered_prompt

    raw_text = json.dumps(
        {
            "final_labels": ["Aortic enlargement", "Cardiomegaly"],
            "supported_findings": ["Aortic enlargement"],
            "uncertain_findings": ["Cardiomegaly"],
            "localization_supported_findings": ["Aortic enlargement"],
            "supported_global_buckets": ["Cardiomediastinal abnormality"],
            "uncertain_global_buckets": [],
            "conflicts": ["YOLO did not localize cardiomegaly."],
            "review_recommendation": "uncertain",
            "confidence_band": "moderate",
            "findings_section": "Cardiomediastinal enlargement is suggested.",
            "impression_section": "1. Aortic enlargement. 2. Possible cardiomegaly.",
            "safety_note": "AI-generated decision-support summary; human review required for clinical use.",
        }
    )
    payload = extract_json_object(raw_text)
    validated = validate_review_response(payload)
    report = render_review_report(validated, case_packet=packet)
    assert "## Case Summary" in report
    assert "## Findings" in report
    assert "## Impression" in report
    assert "Supported global buckets" in report
    assert "Top Swin Probabilities" in report
    assert "YOLO boxes:" in report
    assert "human review required" in report.lower()

    print("Claude review pipeline tests passed")


if __name__ == "__main__":
    main()
