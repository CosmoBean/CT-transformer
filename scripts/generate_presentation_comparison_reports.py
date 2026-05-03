#!/usr/bin/env python3
"""
Generate presentation-focused comparison reports for selected CXR cases.

These reports are intentionally different from the locked clinical workflow PDF:
- they explicitly compare radiologist labels vs Swin vs YOLO
- they include an annotated YOLO overlay for presentation
- they summarize what each model got right, missed, and overcalled
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.detection import (
    load_image_metadata,
    load_raw_detection_annotations,
    merge_overlapping_boxes,
)
from src.review.renderer import compute_finding_confidence_details


DEFAULT_CASES: list[tuple[str, str]] = [
    ("302b0d070d5150b91bafc935e94a847b", "Clean no-finding case"),
    ("804bcde30d36e32d9429f00bed7a388d", "Single localized abnormality"),
    ("a8750c349b5dac834473304bad0f2877", "Disagreement case: Swin miss, YOLO partial support"),
    ("ba5e3409250a85483d6e39be759bc102", "Dense pleural and airspace multi-finding case"),
    ("305e4add9c72c91e9984305bf4e85aee", "Hard diffuse case with partial model capture"),
]

PAGE_SIZE = (8.5, 11)
HEADER_COLOR = "#102A43"
TEXT_COLOR = "#243B53"
MUTED_COLOR = "#5B7083"
GOOD_COLOR = "#1B7F5A"
MISS_COLOR = "#B54708"
EXTRA_COLOR = "#B42318"
BODY_SIZE = 9.2
SMALL_SIZE = 8.3
TITLE_SIZE = 18
SUBTITLE_SIZE = 11
HEADING_SIZE = 12
BOX_COLORS = ["#D62828", "#1D3557", "#2A9D8F", "#F4A261", "#7B2CBF"]
SWIN_THRESHOLD = 0.50
YOLO_CONF_THRESHOLD = 0.25
CLAUDE_BAND_REFERENCE = {
    "high": {"cases": 236, "exact": 214, "exact_acc": 0.9068},
    "moderate": {"cases": 58, "exact": 9, "exact_acc": 0.1552},
    "low": {"cases": 6, "exact": 2, "exact_acc": 0.3333},
}


@dataclass
class CaseRecord:
    image_id: str
    title: str
    true_labels: list[str]
    swin_predicted_labels: list[str]
    yolo_predicted_labels: list[str]
    claude_predicted_labels: list[str]
    supported_findings: list[str]
    uncertain_findings: list[str]
    review_recommendation: str
    confidence_band: str
    payload: dict
    ground_truth_boxes: list[dict]


def _parse_label_list(raw: str) -> list[str]:
    return list(ast.literal_eval(raw))


def _ordered_hits(reference: list[str], predicted: list[str]) -> list[str]:
    predicted_set = set(predicted)
    return [label for label in reference if label in predicted_set]


def _ordered_missing(reference: list[str], predicted: list[str]) -> list[str]:
    predicted_set = set(predicted)
    return [label for label in reference if label not in predicted_set]


def _ordered_extra(reference: list[str], predicted: list[str]) -> list[str]:
    reference_set = set(reference)
    return [label for label in predicted if label not in reference_set]


def _load_case_rows(csv_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["true_labels"] = _parse_label_list(row["true_labels"])
            row["swin_predicted_labels"] = _parse_label_list(row["swin_predicted_labels"])
            row["yolo_predicted_labels"] = _parse_label_list(row["yolo_predicted_labels"])
            row["claude_predicted_labels"] = _parse_label_list(row["claude_predicted_labels"])
            row["supported_findings"] = _parse_label_list(row["supported_findings"])
            row["uncertain_findings"] = _parse_label_list(row["uncertain_findings"])
            rows[row["image_id"]] = row
    return rows


def _load_case_records(
    image_ids: list[tuple[str, str]],
    comparison_csv: Path,
    cache_dir: Path,
    raw_annotation_path: Path,
    image_metadata_path: Path,
) -> list[CaseRecord]:
    raw_annotations = load_raw_detection_annotations(raw_annotation_path)
    metadata = load_image_metadata(image_metadata_path)
    rows = _load_case_rows(comparison_csv)
    records: list[CaseRecord] = []
    for image_id, title in image_ids:
        row = rows[image_id]
        payload = json.loads((cache_dir / f"{image_id}_claude_review.json").read_text())
        image = mpimg.imread(Path(payload["image_path"]))
        target_height, target_width = image.shape[:2]
        gt_boxes = _build_ground_truth_boxes(
            image_id=image_id,
            raw_annotations=raw_annotations,
            metadata=metadata,
            target_height=target_height,
            target_width=target_width,
        )
        records.append(
            CaseRecord(
                image_id=image_id,
                title=title,
                true_labels=row["true_labels"],
                swin_predicted_labels=row["swin_predicted_labels"],
                yolo_predicted_labels=row["yolo_predicted_labels"],
                claude_predicted_labels=row["claude_predicted_labels"],
                supported_findings=row["supported_findings"],
                uncertain_findings=row["uncertain_findings"],
                review_recommendation=row["review_recommendation"],
                confidence_band=row["confidence_band"],
                payload=payload,
                ground_truth_boxes=gt_boxes,
            )
        )
    return records


def _scale_box(
    box: tuple[float, float, float, float],
    original_height: float,
    original_width: float,
    target_height: int,
    target_width: int,
) -> tuple[float, float, float, float] | None:
    x_min, y_min, x_max, y_max = box
    if original_height <= 0 or original_width <= 0:
        return None
    scaled_x_min = max(0.0, min(target_width, x_min * target_width / original_width))
    scaled_y_min = max(0.0, min(target_height, y_min * target_height / original_height))
    scaled_x_max = max(0.0, min(target_width, x_max * target_width / original_width))
    scaled_y_max = max(0.0, min(target_height, y_max * target_height / original_height))
    if scaled_x_max <= scaled_x_min or scaled_y_max <= scaled_y_min:
        return None
    return scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max


def _build_ground_truth_boxes(
    image_id: str,
    raw_annotations,
    metadata,
    target_height: int,
    target_width: int,
) -> list[dict]:
    if image_id not in metadata.index:
        return []
    original_height = float(metadata.loc[image_id, "dim0"])
    original_width = float(metadata.loc[image_id, "dim1"])
    rows = raw_annotations[raw_annotations["image_id"] == image_id]
    if rows.empty:
        return []

    by_label: dict[str, list[tuple[float, float, float, float]]] = {}
    for _, row in rows.iterrows():
        label = row["class_name"]
        if label == "No finding":
            continue
        if any(value != value for value in [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]):
            continue
        by_label.setdefault(label, []).append(
            (float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"]))
        )

    scaled_boxes: list[dict] = []
    for label, boxes in sorted(by_label.items()):
        merged = merge_overlapping_boxes(boxes)
        for merged_box in merged:
            scaled = _scale_box(
                merged_box,
                original_height=original_height,
                original_width=original_width,
                target_height=target_height,
                target_width=target_width,
            )
            if scaled is None:
                continue
            scaled_boxes.append({"class_name": label, "bbox_xyxy": scaled})
    return scaled_boxes


def _draw_image(ax, image, title: str) -> None:
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title(title, fontsize=HEADING_SIZE, loc="left", pad=6, color=TEXT_COLOR)


def _draw_annotated(ax, image, detections: list[dict], title: str) -> None:
    _draw_image(ax, image, title)
    for index, detection in enumerate(detections[:5]):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor=color)
        )
        ax.text(
            x1,
            max(5, y1 - 6),
            f"{detection['class_name']} {float(detection['confidence']):.2f}",
            fontsize=SMALL_SIZE,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.2},
        )


def _draw_ground_truth(ax, image, detections: list[dict], title: str) -> None:
    _draw_image(ax, image, title)
    for index, detection in enumerate(detections[:8]):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor=color)
        )
        ax.text(
            x1,
            max(5, y1 - 6),
            detection["class_name"],
            fontsize=SMALL_SIZE,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.2},
        )


def _list_text(title: str, values: list[str], empty: str = "None") -> list[str]:
    lines = [title]
    if values:
        lines.extend(f"- {value}" for value in values)
    else:
        lines.append(f"- {empty}")
    return lines


def _draw_block(
    fig: plt.Figure,
    title: str,
    lines: list[str],
    x: float,
    y: float,
    width: int = 42,
    line_height: float = 0.021,
    title_color: str = HEADER_COLOR,
    text_color: str = TEXT_COLOR,
) -> None:
    wrapped: list[str] = []
    for line in lines:
        if line.startswith("- "):
            content = line[2:]
            chunks = textwrap.wrap(content, width=width - 2, break_long_words=False, break_on_hyphens=False)
            if not chunks:
                wrapped.append("-")
            else:
                wrapped.append(f"- {chunks[0]}")
                wrapped.extend(f"  {chunk}" for chunk in chunks[1:])
        else:
            wrapped.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""])

    fig.text(x, y, title, fontsize=HEADING_SIZE, fontweight="bold", ha="left", va="top", color=title_color)
    fig.text(x, y - 0.028, "\n".join(wrapped), fontsize=BODY_SIZE, ha="left", va="top", color=text_color, linespacing=1.35)


def _format_probabilities(case_packet: dict, limit: int = 6) -> list[str]:
    lines: list[str] = []
    for item in case_packet["swin"]["sorted_probabilities"][:limit]:
        lines.append(f"- {item['label']}: {float(item['probability']):.3f}")
    return lines or ["- None"]


def _format_detections(case_packet: dict, limit: int = 5) -> list[str]:
    detections = sorted(case_packet["yolo"]["detections"], key=lambda item: float(item["confidence"]), reverse=True)
    lines: list[str] = []
    for detection in detections[:limit]:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        lines.append(f"- {detection['class_name']}: {float(detection['confidence']):.2f} @ [{box}]")
    return lines or ["- None"]


def _format_ground_truth_detections(detections: list[dict], limit: int = 8) -> list[str]:
    lines = []
    for detection in detections[:limit]:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        lines.append(f"- {detection['class_name']} @ [{box}]")
    return lines or ["- None"]


def _confidence_calculation_lines(record: CaseRecord, limit: int = 4) -> list[str]:
    labels: list[str] = []
    for label in record.supported_findings + record.uncertain_findings:
        if label not in labels:
            labels.append(label)
    if not labels:
        labels = record.claude_predicted_labels[:]
    if not labels:
        labels = ["No finding"]

    lines = [
        "Per-finding band cutoffs: high >= 0.80, moderate >= 0.50, low < 0.50.",
        "Swin label threshold = 0.50. YOLO detection threshold = 0.25.",
    ]
    for label in labels[:limit]:
        details = compute_finding_confidence_details(record.payload["case_packet"], label)
        lines.append(
            f"{label}: {details['band']} from {details['formula']}"
        )

    band_ref = CLAUDE_BAND_REFERENCE.get(record.confidence_band)
    if band_ref:
        lines.append(
            f"Agentic AI report band reference: `{record.confidence_band}` occurred in {band_ref['cases']} / 300 cases with exact-match accuracy {band_ref['exact_acc']:.1%}."
        )
    return lines


def _write_markdown(record: CaseRecord, output_path: Path) -> None:
    true_labels = record.true_labels
    swin_hits = _ordered_hits(true_labels, record.swin_predicted_labels)
    swin_missing = _ordered_missing(true_labels, record.swin_predicted_labels)
    swin_extra = _ordered_extra(true_labels, record.swin_predicted_labels)
    yolo_hits = _ordered_hits(true_labels, record.yolo_predicted_labels)
    yolo_missing = _ordered_missing(true_labels, record.yolo_predicted_labels)
    yolo_extra = _ordered_extra(true_labels, record.yolo_predicted_labels)
    review = record.payload["review"]
    lines = [
        f"# {record.title}",
        "",
        f"**Case ID:** `{record.image_id}`",
        f"**Review Recommendation:** `{record.review_recommendation}`",
        f"**Confidence:** `{record.confidence_band}`",
        "",
        "## Ground Truth",
        ", ".join(true_labels) or "None",
        "",
        "## Swin Classification",
        f"Predicted: {', '.join(record.swin_predicted_labels) or 'None'}",
        f"Correctly captured: {', '.join(swin_hits) or 'None'}",
        f"Missed: {', '.join(swin_missing) or 'None'}",
        f"Overcalled: {', '.join(swin_extra) or 'None'}",
        "",
        "## YOLO Detection-Derived Labels",
        f"Predicted: {', '.join(record.yolo_predicted_labels) or 'None'}",
        f"Correctly captured: {', '.join(yolo_hits) or 'None'}",
        f"Missed: {', '.join(yolo_missing) or 'None'}",
        f"Overcalled: {', '.join(yolo_extra) or 'None'}",
        "",
        "## Agentic Summary",
        f"Supported findings: {', '.join(record.supported_findings) or 'None'}",
        f"Uncertain findings: {', '.join(record.uncertain_findings) or 'None'}",
        f"Final report labels: {', '.join(record.claude_predicted_labels) or 'None'}",
        "",
        "## Thresholds Used",
        f"- Swin positive-label threshold: {SWIN_THRESHOLD:.2f}",
        f"- YOLO detection confidence threshold: {YOLO_CONF_THRESHOLD:.2f}",
        "- Agentic AI confidence band (`high` / `moderate` / `low`) is qualitative and comes from the reviewer, not a fixed numeric threshold.",
        "",
        "## Confidence Calculation",
        *_confidence_calculation_lines(record),
        "",
        "## Findings",
        review["findings_section"],
        "",
        "## Impression",
        review["impression_section"],
        "",
        "## Top Swin Probabilities",
        *_format_probabilities(record.payload["case_packet"]),
        "",
        "## Top YOLO Detections",
        *_format_detections(record.payload["case_packet"]),
        "",
    ]
    output_path.write_text("\n".join(lines))


def _save_overlay(record: CaseRecord, output_path: Path) -> None:
    payload = record.payload
    image = mpimg.imread(Path(payload["image_path"]))
    detections = sorted(payload["case_packet"]["yolo"]["detections"], key=lambda item: float(item["confidence"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    _draw_annotated(ax, image, detections, "YOLO support overlay")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _write_pdf(record: CaseRecord, output_path: Path) -> None:
    payload = record.payload
    review = payload["review"]
    image = mpimg.imread(Path(payload["image_path"]))
    detections = sorted(payload["case_packet"]["yolo"]["detections"], key=lambda item: float(item["confidence"]), reverse=True)

    true_labels = record.true_labels
    swin_hits = _ordered_hits(true_labels, record.swin_predicted_labels)
    swin_missing = _ordered_missing(true_labels, record.swin_predicted_labels)
    swin_extra = _ordered_extra(true_labels, record.swin_predicted_labels)
    yolo_hits = _ordered_hits(true_labels, record.yolo_predicted_labels)
    yolo_missing = _ordered_missing(true_labels, record.yolo_predicted_labels)
    yolo_extra = _ordered_extra(true_labels, record.yolo_predicted_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.965, "Presentation Comparison Report", fontsize=TITLE_SIZE, fontweight="bold", color=HEADER_COLOR, ha="left", va="top")
        fig.text(0.07, 0.928, f"{record.title}    Case: {record.image_id}", fontsize=SUBTITLE_SIZE, color=TEXT_COLOR, ha="left", va="top")
        fig.text(0.07, 0.905, f"Review status: {record.review_recommendation}    Confidence: {record.confidence_band}", fontsize=10, color=MUTED_COLOR, ha="left", va="top")

        ax1 = fig.add_axes([0.05, 0.60, 0.27, 0.20])
        ax2 = fig.add_axes([0.365, 0.60, 0.27, 0.20])
        ax3 = fig.add_axes([0.68, 0.60, 0.27, 0.20])
        _draw_image(ax1, image, "Original CXR")
        _draw_ground_truth(ax2, image, record.ground_truth_boxes, "Ground-truth boxes")
        _draw_annotated(ax3, image, detections, "YOLO support overlay")

        _draw_block(fig, "Ground Truth", _list_text("", true_labels)[1:], 0.07, 0.50, width=38)
        _draw_block(fig, "Swin Classification", [
            f"Predicted: {', '.join(record.swin_predicted_labels) or 'None'}",
            f"Correct: {', '.join(swin_hits) or 'None'}",
            f"Missed: {', '.join(swin_missing) or 'None'}",
            f"Overcalled: {', '.join(swin_extra) or 'None'}",
        ], 0.39, 0.50, width=38)
        _draw_block(fig, "YOLO Labels", [
            f"Predicted: {', '.join(record.yolo_predicted_labels) or 'None'}",
            f"Correct: {', '.join(yolo_hits) or 'None'}",
            f"Missed: {', '.join(yolo_missing) or 'None'}",
            f"Overcalled: {', '.join(yolo_extra) or 'None'}",
        ], 0.70, 0.50, width=30)

        _draw_block(fig, "Agentic Workflow Summary", [
            f"Supported findings: {', '.join(record.supported_findings) or 'None'}",
            f"Uncertain findings: {', '.join(record.uncertain_findings) or 'None'}",
            f"Final report labels: {', '.join(record.claude_predicted_labels) or 'None'}",
            f"Pattern buckets: {', '.join(review.get('supported_global_buckets', [])) or 'None'}",
        ], 0.07, 0.21, width=110)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.965, "Evidence and Report Summary", fontsize=TITLE_SIZE, fontweight="bold", color=HEADER_COLOR, ha="left", va="top")
        _draw_block(fig, "Findings", [review["findings_section"]], 0.07, 0.90, width=110)
        _draw_block(fig, "Impression", [review["impression_section"]], 0.07, 0.63, width=110)
        _draw_block(fig, "Top Swin Probabilities", _format_probabilities(payload["case_packet"]), 0.07, 0.36, width=42)
        _draw_block(fig, "Ground-Truth Boxes", _format_ground_truth_detections(record.ground_truth_boxes), 0.42, 0.36, width=34)
        _draw_block(fig, "Top YOLO Detections", _format_detections(payload["case_packet"]), 0.66, 0.36, width=28)
        _draw_block(fig, "Thresholds Used", [
            f"Swin positive label cutoff: {SWIN_THRESHOLD:.2f}",
            f"YOLO detection confidence cutoff: {YOLO_CONF_THRESHOLD:.2f}",
            "Agentic AI confidence band is qualitative, not a fixed numeric threshold.",
        ], 0.42, 0.19, width=60)
        _draw_block(fig, "Presentation Takeaway", [
            f"Swin correct labels: {len(swin_hits)} / {len(true_labels)}",
            f"YOLO correct labels: {len(yolo_hits)} / {len(true_labels)}",
            f"Key miss for Swin: {', '.join(swin_missing[:3]) or 'None'}",
            f"Key miss for YOLO: {', '.join(yolo_missing[:3]) or 'None'}",
        ], 0.07, 0.16, width=38, title_color=HEADER_COLOR)
        fig.text(0.07, 0.08, review["safety_note"], fontsize=SMALL_SIZE, color=MUTED_COLOR, ha="left", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.965, "Confidence Calculation", fontsize=TITLE_SIZE, fontweight="bold", color=HEADER_COLOR, ha="left", va="top")
        _draw_block(
            fig,
            "How this case's confidence was calculated",
            _confidence_calculation_lines(record, limit=6),
            0.07,
            0.90,
            width=110,
            title_color=HEADER_COLOR,
        )
        fig.text(
            0.07,
            0.18,
            "Interpretation note: the report-level agentic AI confidence band is constrained to high / moderate / low in the schema, but the model chooses that band. The per-finding confidence lines above are deterministic and computed from the Swin probability, YOLO support, and the label taxonomy.",
            fontsize=BODY_SIZE,
            color=TEXT_COLOR,
            ha="left",
            va="top",
            wrap=True,
        )
        fig.text(0.07, 0.08, review["safety_note"], fontsize=SMALL_SIZE, color=MUTED_COLOR, ha="left", va="top")
        pdf.savefig(fig)
        plt.close(fig)


def _write_readme(records: list[CaseRecord], output_dir: Path) -> None:
    lines = ["# Presentation Comparison Examples", "", "These five cases compare radiologist ground truth against Swin classification, YOLO detection-derived labels, and the final agentic summary.", ""]
    for record in records:
        lines.append(f"- `{record.image_id}` — {record.title}")
        lines.append(f"  - [Markdown]({record.image_id}.md)")
        lines.append(f"  - [PDF]({record.image_id}.pdf)")
        lines.append(f"  - [Annotated overlay]({record.image_id}_annotated.png)")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate presentation comparison reports for selected cases.")
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=Path("experiments/claude_review/eval_300/claude_vs_baselines_case_comparison.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/claude_review/cache"),
    )
    parser.add_argument(
        "--raw-annotation-path",
        type=Path,
        default=Path("data/_downloads/train_raw.csv"),
    )
    parser.add_argument(
        "--image-metadata-path",
        type=Path,
        default=Path("data/_downloads/vinbig_png/train_meta.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/presentation_comparison_examples"),
    )
    args = parser.parse_args()

    records = _load_case_records(
        DEFAULT_CASES,
        args.comparison_csv,
        args.cache_dir,
        args.raw_annotation_path,
        args.image_metadata_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        _write_markdown(record, args.output_dir / f"{record.image_id}.md")
        _write_pdf(record, args.output_dir / f"{record.image_id}.pdf")
        _save_overlay(record, args.output_dir / f"{record.image_id}_annotated.png")

    _write_readme(records, args.output_dir)
    print(f"Generated {len(records)} comparison report examples in {args.output_dir}")


if __name__ == "__main__":
    main()
