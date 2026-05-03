"""
Schemas and validation helpers for Claude review responses.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from src.data.dataset import CLASS_NAMES


ALLOWED_LABELS = set(CLASS_NAMES)
ALLOWED_CONFIDENCE_BANDS = {"low", "moderate", "high"}
ALLOWED_REVIEW_RECOMMENDATIONS = {"supported", "uncertain", "needs_human_review"}
ALLOWED_GLOBAL_BUCKETS = {
    "No acute abnormality",
    "Cardiomediastinal abnormality",
    "Pleural abnormality",
    "Airspace or infectious-inflammatory pattern",
    "Chronic interstitial or fibrotic pattern",
    "Focal lesion or mass-like pattern",
    "Possible urgent thoracic abnormality",
}


@dataclass(frozen=True)
class ClaudeReviewResponse:
    final_labels: list[str]
    supported_findings: list[str]
    uncertain_findings: list[str]
    localization_supported_findings: list[str]
    supported_global_buckets: list[str]
    uncertain_global_buckets: list[str]
    conflicts: list[str]
    review_recommendation: str
    confidence_band: str
    findings_section: str
    impression_section: str
    safety_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Model response is empty.")

    fence_start = text.find("```json")
    if fence_start != -1:
        fence_end = text.find("```", fence_start + 7)
        if fence_end != -1:
            text = text[fence_start + 7:fence_end].strip()
    elif text.startswith("```"):
        fence_end = text.find("```", 3)
        if fence_end != -1:
            text = text[3:fence_end].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")

    return json.loads(text[start:end + 1])


def _normalize_label_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings.")
        item = item.strip()
        if not item:
            continue
        if item not in ALLOWED_LABELS:
            raise ValueError(
                f"{field_name} contains unsupported label '{item}'. "
                f"Allowed labels: {', '.join(CLASS_NAMES)}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def _normalize_bucket_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings.")
        item = item.strip()
        if not item:
            continue
        if item not in ALLOWED_GLOBAL_BUCKETS:
            raise ValueError(
                f"{field_name} contains unsupported bucket '{item}'. "
                f"Allowed buckets: {', '.join(sorted(ALLOWED_GLOBAL_BUCKETS))}"
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def validate_review_response(payload: dict[str, Any]) -> ClaudeReviewResponse:
    if not isinstance(payload, dict):
        raise ValueError("Reviewer payload must be a JSON object.")

    final_labels = _normalize_label_list(payload.get("final_labels"), "final_labels")
    supported_findings = _normalize_label_list(payload.get("supported_findings"), "supported_findings")
    uncertain_findings = _normalize_label_list(payload.get("uncertain_findings"), "uncertain_findings")
    localization_supported_findings = _normalize_label_list(
        payload.get("localization_supported_findings"),
        "localization_supported_findings",
    )
    supported_global_buckets = _normalize_bucket_list(
        payload.get("supported_global_buckets"),
        "supported_global_buckets",
    )
    uncertain_global_buckets = _normalize_bucket_list(
        payload.get("uncertain_global_buckets"),
        "uncertain_global_buckets",
    )

    conflicts_raw = payload.get("conflicts") or []
    if not isinstance(conflicts_raw, list) or any(not isinstance(item, str) for item in conflicts_raw):
        raise ValueError("conflicts must be a list of strings.")
    conflicts = [item.strip() for item in conflicts_raw if item and item.strip()]

    review_recommendation = str(payload.get("review_recommendation", "")).strip()
    if review_recommendation not in ALLOWED_REVIEW_RECOMMENDATIONS:
        raise ValueError(
            "review_recommendation must be one of: "
            + ", ".join(sorted(ALLOWED_REVIEW_RECOMMENDATIONS))
        )

    confidence_band = str(payload.get("confidence_band", "")).strip()
    if confidence_band not in ALLOWED_CONFIDENCE_BANDS:
        raise ValueError(
            "confidence_band must be one of: "
            + ", ".join(sorted(ALLOWED_CONFIDENCE_BANDS))
        )

    findings_section = str(payload.get("findings_section", "")).strip()
    impression_section = str(payload.get("impression_section", "")).strip()
    safety_note = str(payload.get("safety_note", "")).strip()

    if not findings_section:
        raise ValueError("findings_section is required.")
    if not impression_section:
        raise ValueError("impression_section is required.")
    if not safety_note:
        raise ValueError("safety_note is required.")

    return ClaudeReviewResponse(
        final_labels=final_labels,
        supported_findings=supported_findings,
        uncertain_findings=uncertain_findings,
        localization_supported_findings=localization_supported_findings,
        supported_global_buckets=supported_global_buckets,
        uncertain_global_buckets=uncertain_global_buckets,
        conflicts=conflicts,
        review_recommendation=review_recommendation,
        confidence_band=confidence_band,
        findings_section=findings_section,
        impression_section=impression_section,
        safety_note=safety_note,
    )
