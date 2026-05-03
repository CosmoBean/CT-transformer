"""
Build structured case packets and prompts for Claude review.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from src.data.dataset import CLASS_NAMES
from src.review.taxonomy import DERIVED_GLOBAL_BUCKETS, LABEL_CATEGORY_BY_NAME, derive_global_bucket_hints


def load_prompt_template(prompt_path: str | Path) -> str:
    return Path(prompt_path).read_text()


def encode_image_as_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime_type = "image/png"
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _sorted_probabilities(probabilities: dict[str, float]) -> list[dict[str, float | str]]:
    return [
        {"label": label, "probability": float(score)}
        for label, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    ]


def build_case_packet(
    image_id: str,
    image_path: str | Path,
    swin_probabilities: dict[str, float],
    swin_predicted_labels: list[str],
    yolo_detections: list[dict[str, Any]],
    yolo_predicted_labels: list[str],
    yolo_conf_threshold: float,
) -> dict[str, Any]:
    swin_set = set(swin_predicted_labels)
    yolo_set = set(yolo_predicted_labels)
    overlap = sorted(swin_set & yolo_set)
    swin_only = sorted(swin_set - yolo_set)
    yolo_only = sorted(yolo_set - swin_set)
    bucket_hints = derive_global_bucket_hints(
        swin_predicted_labels=swin_predicted_labels,
        yolo_predicted_labels=yolo_predicted_labels,
    )

    return {
        "image_id": image_id,
        "image_path": str(Path(image_path)),
        "swin": {
            "probabilities": {label: float(swin_probabilities[label]) for label in CLASS_NAMES},
            "sorted_probabilities": _sorted_probabilities(swin_probabilities),
            "predicted_labels": list(swin_predicted_labels),
            "top_labels": [row["label"] for row in _sorted_probabilities(swin_probabilities)[:5]],
        },
        "yolo": {
            "confidence_threshold": float(yolo_conf_threshold),
            "detections": yolo_detections,
            "predicted_labels": list(yolo_predicted_labels),
        },
        "agreement_summary": {
            "overlap_labels": overlap,
            "swin_only_labels": swin_only,
            "yolo_only_labels": yolo_only,
            "models_agree_on_normal": "No finding" in swin_set and "No finding" in yolo_set,
            "models_disagree": bool(swin_only or yolo_only),
        },
        "label_taxonomy": {
            "label_categories": LABEL_CATEGORY_BY_NAME,
            "derived_global_buckets": DERIVED_GLOBAL_BUCKETS,
        },
        "derived_global_bucket_hints": bucket_hints,
    }


def render_user_prompt(prompt_template: str, case_packet: dict[str, Any]) -> str:
    return (
        prompt_template
        .replace("{{ALLOWED_LABELS}}", ", ".join(CLASS_NAMES))
        .replace("{{ALLOWED_GLOBAL_BUCKETS}}", ", ".join(DERIVED_GLOBAL_BUCKETS.keys()))
        .replace("{{CASE_PACKET_JSON}}", json.dumps(case_packet, indent=2))
    )
