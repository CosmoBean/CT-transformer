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
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    if title:
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


def _save_overlay(record: CaseRecord, output_path: Path) -> None:
    payload = record.payload
    image = mpimg.imread(Path(payload["image_path"]))
    detections = sorted(payload["case_packet"]["yolo"]["detections"], key=lambda item: float(item["confidence"]), reverse=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    _draw_annotated(ax, image, detections, "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _presentation_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor(HEADER_COLOR),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor(TEXT_COLOR),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor(HEADER_COLOR),
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor(HEADER_COLOR),
            spaceBefore=2,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12.5,
            textColor=colors.HexColor(TEXT_COLOR),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            textColor=colors.HexColor(MUTED_COLOR),
            spaceAfter=4,
        )
    )
    return styles


def _scaled_reportlab_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    width = float(image.imageWidth)
    height = float(image.imageHeight)
    scale = min(max_width / width, max_height / height)
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    return image


def _framed_image_cell(path: Path, title: str, styles) -> list:
    return [
        Paragraph(title, styles["SubHeading"]),
        _scaled_reportlab_image(path, 2.05 * inch, 2.05 * inch),
        Spacer(1, 0.02 * inch),
    ]


def _bullet_list(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, style)) for item in items],
        bulletType="bullet",
        leftIndent=14,
    )


def _compact_sentence(items: list[str], none_text: str = "none") -> str:
    return ", ".join(items) if items else none_text


def _concise_agent_summary(record: CaseRecord) -> list[str]:
    review = record.payload["review"]
    true_labels = record.true_labels
    supported = [label for label in record.supported_findings if label != "No finding"]
    uncertain = [label for label in record.uncertain_findings if label != "No finding"]
    final_labels = [label for label in record.claude_predicted_labels if label != "No finding"]
    summary = [
        f"The agentic report favors {_compact_sentence(final_labels or record.claude_predicted_labels)} with a {record.confidence_band} confidence band and a {record.review_recommendation} review status.",
        f"Primary AI-supported findings are {_compact_sentence(supported)}, while the remaining uncertain findings are {_compact_sentence(uncertain)}.",
        f"Relative to radiologist labels, Swin captured {_compact_sentence(_ordered_hits(true_labels, record.swin_predicted_labels))} and YOLO localized {_compact_sentence(_ordered_hits(true_labels, record.yolo_predicted_labels))}.",
    ]
    buckets = review.get("supported_global_buckets", [])
    if buckets:
        summary.append(f"Overall pattern emphasis is {_compact_sentence(buckets)}.")
    return summary[:4]


def _sentences_to_bullets(text: str, limit: int = 4) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]
    bullets = [part + "." for part in parts[:limit]]
    return bullets or [text.strip()]


def _section_block(title: str, content, styles) -> KeepTogether:
    return KeepTogether([Paragraph(title, styles["SectionHeading"]), content])


def _build_image_table(record: CaseRecord, styles, overlay_path: Path) -> Table:
    image = mpimg.imread(Path(record.payload["image_path"]))
    gt_path = overlay_path.with_name(f"{record.image_id}_gt.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    _draw_ground_truth(ax, image, record.ground_truth_boxes, "")
    fig.savefig(gt_path, dpi=180, facecolor="white")
    plt.close(fig)

    table = Table(
        [
            [
                _framed_image_cell(Path(record.payload["image_path"]), "Original CXR", styles),
                _framed_image_cell(gt_path, "Ground-Truth Boxes", styles),
                _framed_image_cell(overlay_path, "YOLO Support Overlay", styles),
            ],
        ],
        colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch],
        rowHeights=[2.45 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
            ]
        )
    )
    return table


def _comparison_table(record: CaseRecord, styles) -> Table:
    true_labels = record.true_labels
    swin_hits = _ordered_hits(true_labels, record.swin_predicted_labels)
    swin_missing = _ordered_missing(true_labels, record.swin_predicted_labels)
    yolo_hits = _ordered_hits(true_labels, record.yolo_predicted_labels)
    yolo_missing = _ordered_missing(true_labels, record.yolo_predicted_labels)
    rows = [
        [
            Paragraph("<b>Reference</b><br/>" + _compact_sentence(true_labels), styles["Body"]),
            Paragraph(
                "<b>Swin</b><br/>"
                f"Predicted: {_compact_sentence(record.swin_predicted_labels)}<br/>"
                f"Correct: {_compact_sentence(swin_hits)}<br/>"
                f"Missed: {_compact_sentence(swin_missing)}",
                styles["Body"],
            ),
            Paragraph(
                "<b>YOLO</b><br/>"
                f"Predicted: {_compact_sentence(record.yolo_predicted_labels)}<br/>"
                f"Correct: {_compact_sentence(yolo_hits)}<br/>"
                f"Missed: {_compact_sentence(yolo_missing)}",
                styles["Body"],
            ),
        ]
    ]
    table = Table(rows, colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F0F4F8")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#F3FAF7")),
                ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _evidence_table(record: CaseRecord, styles) -> Table:
    payload = record.payload
    true_labels = record.true_labels
    swin_missing = _ordered_missing(true_labels, record.swin_predicted_labels)
    yolo_missing = _ordered_missing(true_labels, record.yolo_predicted_labels)
    left = [
        Paragraph("Top Swin Probabilities", styles["SubHeading"]),
        _bullet_list(_format_probabilities(payload["case_packet"], limit=6), styles["Body"]),
    ]
    middle = [
        Paragraph("Top YOLO Detections", styles["SubHeading"]),
        _bullet_list(_format_detections(payload["case_packet"], limit=5), styles["Body"]),
    ]
    right = [
        Paragraph("Takeaways", styles["SubHeading"]),
        _bullet_list(
            [
                f"Swin correct labels: {len(_ordered_hits(true_labels, record.swin_predicted_labels))} / {len(true_labels)}",
                f"YOLO correct labels: {len(_ordered_hits(true_labels, record.yolo_predicted_labels))} / {len(true_labels)}",
                f"Swin key miss: {_compact_sentence(swin_missing[:3])}",
                f"YOLO key miss: {_compact_sentence(yolo_missing[:3])}",
            ],
            styles["Body"],
        ),
    ]
    table = Table([[left, middle, right]], colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return table


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(MUTED_COLOR))
    footer = "AI-generated presentation summary for research review only; human interpretation remains required."
    canvas.drawString(doc.leftMargin, 0.45 * inch, footer)
    canvas.restoreState()


def _write_pdf(record: CaseRecord, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _presentation_styles()
    review = record.payload["review"]

    with tempfile.TemporaryDirectory(prefix=f"{record.image_id}_presentation_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        overlay_path = tmp_path / f"{record.image_id}_annotated.png"
        _save_overlay(record, overlay_path)

        story = [
            Paragraph("Example Comparison Report", styles["ReportTitle"]),
            Paragraph(
                f"{record.title}<br/>Case: {record.image_id}<br/>Review status: {record.review_recommendation} &nbsp;&nbsp; Confidence: {record.confidence_band}",
                styles["ReportSubtitle"],
            ),
            _section_block("Image Comparison", _build_image_table(record, styles, overlay_path), styles),
            Spacer(1, 0.08 * inch),
            _section_block("Model Comparison", _comparison_table(record, styles), styles),
            Spacer(1, 0.05 * inch),
            _section_block("Findings", _bullet_list(_sentences_to_bullets(review["findings_section"], limit=4), styles["Body"]), styles),
            Spacer(1, 0.02 * inch),
            _section_block("Impression", _bullet_list(_sentences_to_bullets(review["impression_section"], limit=4), styles["Body"]), styles),
            PageBreak(),
            _section_block("Agentic Workflow Summary", Paragraph(" ".join(_concise_agent_summary(record)), styles["Body"]), styles),
            Spacer(1, 0.04 * inch),
            _section_block("Supporting Evidence", _evidence_table(record, styles), styles),
            Spacer(1, 0.06 * inch),
            _section_block(
                "Confidence Rules Used",
                _bullet_list(
                    [
                        f"Swin positive label cutoff: {SWIN_THRESHOLD:.2f}",
                        f"YOLO detection confidence cutoff: {YOLO_CONF_THRESHOLD:.2f}",
                        "Agentic AI confidence band is qualitative and selected by the reviewer.",
                    ],
                    styles["Body"],
                ),
                styles,
            ),
            Spacer(1, 0.04 * inch),
            Paragraph(review["safety_note"], styles["Small"]),
        ]

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.7 * inch,
            title="Example Comparison Report",
        )
        doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


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
        default=Path("reports/comparision_reports"),
    )
    args = parser.parse_args()

    count = generate_comparison_reports(
        comparison_csv=args.comparison_csv,
        cache_dir=args.cache_dir,
        raw_annotation_path=args.raw_annotation_path,
        image_metadata_path=args.image_metadata_path,
        output_dir=args.output_dir,
    )
    print(f"Generated {count} comparison report examples in {args.output_dir}")


def generate_comparison_reports(
    comparison_csv: Path = Path("experiments/claude_review/eval_300/claude_vs_baselines_case_comparison.csv"),
    cache_dir: Path = Path("experiments/claude_review/cache"),
    raw_annotation_path: Path = Path("data/_downloads/train_raw.csv"),
    image_metadata_path: Path = Path("data/_downloads/vinbig_png/train_meta.csv"),
    output_dir: Path = Path("reports/comparision_reports"),
    image_ids: list[tuple[str, str]] | None = None,
) -> int:
    records = _load_case_records(
        image_ids or DEFAULT_CASES,
        comparison_csv,
        cache_dir,
        raw_annotation_path,
        image_metadata_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        _write_pdf(record, output_dir / f"{record.image_id}.pdf")
    return len(records)


if __name__ == "__main__":
    main()
