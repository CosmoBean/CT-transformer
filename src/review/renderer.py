"""
Render validated review outputs into a doctor-facing support note.
"""
from __future__ import annotations

from typing import Any

from src.review.schema import ClaudeReviewResponse
from src.review.taxonomy import LABEL_CATEGORY_BY_NAME, derive_global_bucket_hints


def _best_yolo_confidence(case_packet: dict[str, Any], label: str) -> float | None:
    matches = [
        float(detection["confidence"])
        for detection in case_packet["yolo"]["detections"]
        if detection["class_name"] == label
    ]
    if not matches:
        return None
    return max(matches)


def _yolo_detections_for_label(case_packet: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return sorted(
        [
            detection
            for detection in case_packet["yolo"]["detections"]
            if detection["class_name"] == label
        ],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )


def _band_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "moderate"
    return "low"


def compute_finding_confidence_details(case_packet: dict[str, Any], label: str) -> dict[str, Any]:
    swin_probability = float(case_packet["swin"]["probabilities"].get(label, 0.0))
    yolo_confidence = _best_yolo_confidence(case_packet, label)
    category = LABEL_CATEGORY_BY_NAME.get(label, "diffuse_optional_localization")

    if label == "No finding":
        abnormal_localized = [
            item for item in case_packet["yolo"]["predicted_labels"] if item != "No finding"
        ]
        score = swin_probability if not abnormal_localized else min(swin_probability, 0.49)
        formula = (
            f"score = Swin({swin_probability:.2f})"
            if not abnormal_localized
            else f"score = min(Swin({swin_probability:.2f}), 0.49) because abnormal YOLO evidence exists"
        )
    elif category == "strong_localizable":
        if yolo_confidence is not None:
            blended = min(1.0, 0.6 * swin_probability + 0.4 * yolo_confidence)
            score = max(swin_probability, blended)
            formula = (
                f"score = max(Swin({swin_probability:.2f}), min(1.00, 0.6*{swin_probability:.2f} + 0.4*{yolo_confidence:.2f}))"
                f" = {score:.2f}"
            )
        else:
            score = min(swin_probability, 0.65)
            formula = f"score = min(Swin({swin_probability:.2f}), 0.65) because no YOLO support"
    else:
        if yolo_confidence is not None:
            blended = min(1.0, 0.8 * swin_probability + 0.2 * yolo_confidence)
            score = max(swin_probability, blended)
            formula = (
                f"score = max(Swin({swin_probability:.2f}), min(1.00, 0.8*{swin_probability:.2f} + 0.2*{yolo_confidence:.2f}))"
                f" = {score:.2f}"
            )
        else:
            score = swin_probability
            formula = f"score = Swin({swin_probability:.2f})"

    return {
        "label": label,
        "category": category,
        "swin_probability": swin_probability,
        "yolo_confidence": yolo_confidence,
        "score": score,
        "band": _band_from_score(score),
        "formula": formula,
    }


def _format_finding_confidence(case_packet: dict[str, Any], label: str) -> str:
    details = compute_finding_confidence_details(case_packet, label)
    yolo_confidence = details["yolo_confidence"]

    pieces = [f"{details['band']} (Swin {details['swin_probability']:.2f}"]
    if yolo_confidence is not None:
        pieces.append(f", YOLO {yolo_confidence:.2f}")
    pieces.append(")")
    return "".join(pieces)


def _format_box_coordinates(box: list[float]) -> str:
    return "[" + ", ".join(f"{float(value):.1f}" for value in box) + "]"


def _format_localization_summary(case_packet: dict[str, Any], label: str) -> str:
    category = LABEL_CATEGORY_BY_NAME.get(label, "diffuse_optional_localization")
    detections = _yolo_detections_for_label(case_packet, label)
    if detections:
        rendered_boxes = ", ".join(
            _format_box_coordinates(detection["bbox_xyxy"]) for detection in detections[:3]
        )
        return f"YOLO boxes: {rendered_boxes}"
    if category == "global_image_level_only":
        return "YOLO boxes: not expected for this label"
    if category == "strong_localizable":
        return "YOLO boxes: none"
    return "YOLO boxes: none (localization optional)"


def _format_support_line(case_packet: dict[str, Any], label: str) -> str:
    return (
        f"- {label}: {_format_finding_confidence(case_packet, label)}; "
        f"{_format_localization_summary(case_packet, label)}"
    )


def _normal_findings_text(case_packet: dict[str, Any]) -> str:
    no_finding_probability = float(case_packet["swin"]["probabilities"].get("No finding", 0.0))
    return (
        "No abnormal findings are strongly supported by the current model evidence. "
        f"The image-level classifier favors `No finding` with probability {no_finding_probability:.2f}, "
        "and the detector did not produce any abnormal localized findings above threshold."
    )


def _normal_impression_text() -> str:
    return (
        "No acute abnormality is supported by the current decision-support workflow. "
        "This is a normal-model summary rather than a diagnostic clearance, and human review remains required."
    )


def _join_labels(labels: list[str]) -> str:
    return ", ".join(labels) if labels else "None"


def _resolved_supported_global_buckets(
    response: ClaudeReviewResponse,
    case_packet: dict[str, Any],
) -> list[str]:
    derived_hints = case_packet.get("derived_global_bucket_hints") or derive_global_bucket_hints(
        swin_predicted_labels=response.final_labels,
        yolo_predicted_labels=case_packet["yolo"]["predicted_labels"],
    )
    buckets = response.supported_global_buckets or derived_hints.get("supported_buckets", [])
    return list(dict.fromkeys(buckets))


def _resolved_uncertain_global_buckets(
    response: ClaudeReviewResponse,
    case_packet: dict[str, Any],
) -> list[str]:
    derived_hints = case_packet.get("derived_global_bucket_hints") or derive_global_bucket_hints(
        swin_predicted_labels=response.final_labels,
        yolo_predicted_labels=case_packet["yolo"]["predicted_labels"],
    )
    buckets = response.uncertain_global_buckets or derived_hints.get("uncertain_buckets", [])
    return list(dict.fromkeys(buckets))


def _standard_findings_text(response: ClaudeReviewResponse, case_packet: dict[str, Any]) -> str:
    if response.final_labels == ["No finding"] and not response.uncertain_findings:
        return _normal_findings_text(case_packet)

    lines: list[str] = []
    abnormal_supported = [label for label in response.supported_findings if label != "No finding"]
    if abnormal_supported:
        lines.append(
            "Primary supported findings: "
            + "; ".join(
                f"{label} ({_format_finding_confidence(case_packet, label)})"
                for label in abnormal_supported
            )
            + "."
        )
    elif response.supported_findings:
        lines.append(
            "Primary supported findings: "
            + "; ".join(
                f"{label} ({_format_finding_confidence(case_packet, label)})"
                for label in response.supported_findings
            )
            + "."
        )
    else:
        lines.append("No abnormal findings are strongly supported by the current model evidence.")

    if response.uncertain_findings:
        lines.append(
            "Additional uncertain findings: "
            + "; ".join(
                f"{label} ({_format_finding_confidence(case_packet, label)})"
                for label in response.uncertain_findings
            )
            + "."
        )

    if response.localization_supported_findings:
        lines.append(
            "Localized YOLO support is present for: "
            + ", ".join(response.localization_supported_findings)
            + "."
        )

    supported_global_buckets = _resolved_supported_global_buckets(response, case_packet)
    if supported_global_buckets:
        lines.append(
            "Overall supported pattern summary: "
            + ", ".join(supported_global_buckets)
            + "."
        )

    return " ".join(lines)


def _standard_impression_text(response: ClaudeReviewResponse) -> str:
    if response.final_labels == ["No finding"] and not response.uncertain_findings:
        return _normal_impression_text()

    supported = [label for label in response.supported_findings if label != "No finding"]
    if supported:
        first_sentence = "Decision-support summary favors: " + ", ".join(supported) + "."
    elif response.final_labels:
        first_sentence = "Decision-support summary favors: " + ", ".join(response.final_labels) + "."
    else:
        first_sentence = "Decision-support summary is indeterminate."

    if response.uncertain_findings:
        second_sentence = (
            "Uncertain findings requiring correlation: "
            + ", ".join(response.uncertain_findings)
            + "."
        )
    elif response.review_recommendation == "needs_human_review":
        second_sentence = "Human review is recommended because the current evidence remains mixed."
    elif response.review_recommendation == "uncertain":
        second_sentence = "This output should be treated as supportive but not definitive."
    else:
        second_sentence = "This output is intended for decision support and still requires human review."

    return f"{first_sentence} {second_sentence}"


def render_review_report(response: ClaudeReviewResponse, case_packet: dict[str, Any]) -> str:
    supported_global_buckets = _resolved_supported_global_buckets(response, case_packet)
    uncertain_global_buckets = _resolved_uncertain_global_buckets(response, case_packet)
    supported_lines = [_format_support_line(case_packet, label) for label in response.supported_findings]
    uncertain_lines = [_format_support_line(case_packet, label) for label in response.uncertain_findings]
    top_probability_lines = [
        f"- {item['label']}: {float(item['probability']):.2f}"
        for item in case_packet["swin"]["sorted_probabilities"][:5]
    ]
    localized_lines = []
    for detection in sorted(
        case_packet["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )[:5]:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        localized_lines.append(
            f"- {detection['class_name']}: {float(detection['confidence']):.2f} @ [{box}]"
        )

    findings_text = _standard_findings_text(response, case_packet)
    impression_text = _standard_impression_text(response)

    lines = [
        "# AI Decision-Support Report",
        "",
        "## Case Summary",
        f"- Recommendation: {response.review_recommendation}",
        f"- Confidence: {response.confidence_band}",
        f"- Final labels: {_join_labels(response.final_labels)}",
        f"- Supported findings: {_join_labels(response.supported_findings)}",
        f"- Uncertain findings: {_join_labels(response.uncertain_findings)}",
        f"- Localization support: {_join_labels(response.localization_supported_findings)}",
        f"- Supported global buckets: {_join_labels(supported_global_buckets)}",
        f"- Uncertain global buckets: {_join_labels(uncertain_global_buckets)}",
    ]
    if response.conflicts:
        lines.extend(
            [
                "- Model disagreements:",
                *[f"  - {conflict}" for conflict in response.conflicts],
            ]
        )
    lines.extend(
        [
            "",
            "## Findings",
            findings_text,
            "",
            "## Impression",
            impression_text,
            "",
            "## Support Summary",
            "### Supported Findings",
            *(supported_lines or ["- None"]),
            "",
            "### Uncertain Findings",
            *(uncertain_lines or ["- None"]),
            "",
            "### Top Swin Probabilities",
            *(top_probability_lines or ["- None"]),
            "",
            "### YOLO Detections",
            *(localized_lines or ["- None"]),
        ]
    )
    lines.extend(
        [
            "",
            "## Safety",
            response.safety_note.strip(),
        ]
    )
    return "\n".join(lines)
