#!/usr/bin/env python3
"""
Export Claude review JSON bundles to simple reviewable PDF files.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (8.5, 11)
TITLE_FONTSIZE = 18
SUBTITLE_FONTSIZE = 11
HEADING_FONTSIZE = 12
BODY_FONTSIZE = 9.5
SMALL_FONTSIZE = 8.5
LINE_HEIGHT = 0.025
LEFT_MARGIN = 0.07
TOP_MARGIN = 0.96
BOTTOM_MARGIN = 0.06
TEXT_WIDTH = 90
HEADER_COLOR = "#102A43"
TEXT_COLOR = "#243B53"
BOX_COLORS = ["#D62828", "#1D3557", "#2A9D8F", "#F4A261", "#7B2CBF"]


def _resolve_report_inputs(inputs: Iterable[str]) -> list[Path]:
    report_paths: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            path = path / "review_result.json"
        if not path.exists():
            raise FileNotFoundError(f"Review bundle not found: {path}")
        report_paths.append(path)
    return report_paths


def _strip_markdown(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        lines.append(line.replace("`", ""))
    return "\n".join(lines)


def _wrap_paragraphs(text: str, width: int = TEXT_WIDTH) -> list[str]:
    wrapped: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            wrapped.append("")
            continue
        if paragraph.lstrip().startswith("- "):
            content = paragraph.lstrip()[2:].strip()
            bullet_lines = textwrap.wrap(
                content,
                width=width - 2,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if not bullet_lines:
                wrapped.append("-")
                continue
            wrapped.append(f"- {bullet_lines[0]}")
            wrapped.extend(f"  {line}" for line in bullet_lines[1:])
            continue
        wrapped.extend(
            textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return wrapped


def _chunk_lines(lines: list[str], max_lines: int) -> list[list[str]]:
    if not lines:
        return [[]]
    return [lines[index:index + max_lines] for index in range(0, len(lines), max_lines)]


def _format_top_probabilities(case_packet: dict) -> str:
    items = case_packet["swin"]["sorted_probabilities"][:8]
    lines = ["Top Swin probabilities:"]
    lines.extend(
        f"- {item['label']}: {float(item['probability']):.3f}" for item in items
    )
    return "\n".join(lines)


def _format_yolo_summary(case_packet: dict) -> str:
    detections = sorted(
        case_packet["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )[:8]
    if not detections:
        return "Top YOLO detections:\n- None"

    lines = ["Top YOLO detections:"]
    for detection in detections:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        lines.append(
            f"- {detection['class_name']}: {float(detection['confidence']):.3f} @ [{box}]"
        )
    return "\n".join(lines)


def _render_original_image(fig: plt.Figure, image, bounds: list[float]) -> None:
    ax = fig.add_axes(bounds)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title("Original CXR", fontsize=HEADING_FONTSIZE, loc="left", pad=6, color=TEXT_COLOR)


def _render_annotated_image(fig: plt.Figure, image, detections: list[dict], bounds: list[float]) -> None:
    ax = fig.add_axes(bounds)
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title("YOLO Support Overlay", fontsize=HEADING_FONTSIZE, loc="left", pad=6, color=TEXT_COLOR)

    for index, detection in enumerate(detections[:5]):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor=color)
        ax.add_patch(rect)
        ax.text(
            x1,
            max(4, y1 - 6),
            f"{detection['class_name']} {float(detection['confidence']):.2f}",
            fontsize=SMALL_FONTSIZE,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.5},
        )


def _save_annotated_png(payload: dict, output_path: Path) -> None:
    image = mpimg.imread(Path(payload["image_path"]))
    detections = sorted(
        payload["case_packet"]["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    ax.set_title("YOLO Support Overlay", fontsize=HEADING_FONTSIZE, loc="left", pad=8, color=TEXT_COLOR)
    for index, detection in enumerate(detections[:5]):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        ax.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.2, edgecolor=color)
        )
        ax.text(
            x1,
            max(4, y1 - 6),
            f"{detection['class_name']} {float(detection['confidence']):.2f}",
            fontsize=SMALL_FONTSIZE,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.5},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def _format_case_summary(payload: dict) -> list[str]:
    review = payload["review"]
    return [
        f"Recommendation: {review['review_recommendation']}",
        f"Confidence: {review['confidence_band']}",
        f"Final labels: {', '.join(review['final_labels']) or 'None'}",
        f"Supported findings: {', '.join(review['supported_findings']) or 'None'}",
        f"Uncertain findings: {', '.join(review['uncertain_findings']) or 'None'}",
        f"Localization support: {', '.join(review['localization_supported_findings']) or 'None'}",
        f"Global buckets: {', '.join(review.get('supported_global_buckets', [])) or 'None'}",
    ]


def _format_yolo_legend(detections: list[dict]) -> list[str]:
    if not detections:
        return ["YOLO detections: None"]
    lines = ["YOLO detections:"]
    for detection in detections[:5]:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        lines.append(
            f"- {detection['class_name']} ({float(detection['confidence']):.2f}) @ [{box}]"
        )
    return lines


def _format_at_a_glance(payload: dict) -> list[str]:
    review = payload["review"]
    lines = [
        f"Review status: {review['review_recommendation']}",
        f"Confidence: {review['confidence_band']}",
        f"Key findings: {', '.join(review['supported_findings']) or 'None'}",
        f"Needs confirmation: {', '.join(review['uncertain_findings']) or 'None'}",
        f"YOLO localized: {', '.join(review['localization_supported_findings']) or 'None'}",
        f"Pattern summary: {', '.join(review.get('supported_global_buckets', [])) or 'None'}",
    ]
    conflicts = review.get("conflicts", [])
    if conflicts:
        lines.append(f"Model disagreement: {conflicts[0]}")
    return lines


def _format_support_lines(payload: dict) -> list[str]:
    case_packet = payload["case_packet"]
    review = payload["review"]
    lines: list[str] = []
    for label in review["supported_findings"]:
        lines.append(
            f"- {label}: {case_packet['swin']['probabilities'].get(label, 0.0):.2f}"
        )
    if not lines:
        lines.append("- None")
    return lines


def _format_uncertain_lines(payload: dict) -> list[str]:
    case_packet = payload["case_packet"]
    review = payload["review"]
    lines: list[str] = []
    for label in review["uncertain_findings"]:
        lines.append(
            f"- {label}: {case_packet['swin']['probabilities'].get(label, 0.0):.2f}"
        )
    if not lines:
        lines.append("- None")
    return lines


def _draw_text_block(
    fig: plt.Figure,
    title: str,
    lines: list[str],
    x: float,
    y: float,
    width: int = 52,
    max_lines: int | None = 12,
    font_size: float = BODY_FONTSIZE,
) -> None:
    wrapped: list[str] = []
    for line in lines:
        if not line.strip():
            wrapped.append("")
            continue
        if line.startswith("- "):
            content = line[2:]
            chunks = textwrap.wrap(
                content,
                width=width - 2,
                break_long_words=False,
                break_on_hyphens=False,
            )
            if chunks:
                wrapped.append(f"- {chunks[0]}")
                wrapped.extend(f"  {chunk}" for chunk in chunks[1:])
            else:
                wrapped.append("-")
        else:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    if max_lines is not None and len(wrapped) > max_lines:
        wrapped = wrapped[: max_lines - 1] + ["..."]

    fig.text(
        x,
        y,
        title,
        fontsize=HEADING_FONTSIZE,
        fontweight="bold",
        color=HEADER_COLOR,
        ha="left",
        va="top",
    )
    fig.text(
        x,
        y - 0.03,
        "\n".join(wrapped),
        fontsize=font_size,
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )


def _draw_first_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    image_path = Path(payload["image_path"])
    image = mpimg.imread(image_path)
    detections = sorted(
        payload["case_packet"]["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )

    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN,
        "AI Decision-Support Report",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
        color=HEADER_COLOR,
    )
    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN - 0.04,
        f"Case: {payload['image_id']}    Source image: {image_path.name}",
        fontsize=SUBTITLE_FONTSIZE,
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )

    _render_original_image(fig, image, [0.08, 0.57, 0.36, 0.26])
    _render_annotated_image(fig, image, detections, [0.56, 0.57, 0.36, 0.26])

    _draw_text_block(
        fig,
        "At A Glance",
        x=0.08,
        y=0.47,
        width=110,
        max_lines=None,
        font_size=9.0,
        lines=_format_at_a_glance(payload),
    )

    pdf.savefig(fig)
    plt.close(fig)


def _draw_clinical_summary_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    review = payload["review"]

    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN,
        "Clinical Summary",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
        color=HEADER_COLOR,
    )

    _draw_text_block(
        fig,
        "Findings",
        [review["findings_section"]],
        x=0.08,
        y=0.88,
        width=105,
        max_lines=None,
        font_size=10.0,
    )
    _draw_text_block(
        fig,
        "Impression",
        [review["impression_section"]],
        x=0.08,
        y=0.58,
        width=105,
        max_lines=None,
        font_size=10.0,
    )

    guidance_lines = [
        f"Recommendation: {review['review_recommendation']}",
        f"Confidence: {review['confidence_band']}",
        f"Supported findings: {', '.join(review['supported_findings']) or 'None'}",
        f"Uncertain findings: {', '.join(review['uncertain_findings']) or 'None'}",
    ]
    if review.get("conflicts"):
        guidance_lines.append(f"Disagreement: {review['conflicts'][0]}")
    _draw_text_block(
        fig,
        "Review Guidance",
        guidance_lines,
        x=0.08,
        y=0.30,
        width=105,
        max_lines=None,
        font_size=9.6,
    )

    safety_lines = textwrap.wrap(
        review["safety_note"],
        width=120,
        break_long_words=False,
        break_on_hyphens=False,
    )
    fig.text(
        LEFT_MARGIN,
        0.08,
        "\n".join(safety_lines),
        fontsize=SMALL_FONTSIZE,
        ha="left",
        va="top",
        color=TEXT_COLOR,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _draw_evidence_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    image = mpimg.imread(Path(payload["image_path"]))
    detections = sorted(
        payload["case_packet"]["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )

    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN,
        "Supporting Evidence",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
        color=HEADER_COLOR,
    )

    _render_annotated_image(fig, image, detections, [0.08, 0.48, 0.52, 0.38])

    _draw_text_block(
        fig,
        "YOLO Detections",
        _format_yolo_legend(detections),
        x=0.64,
        y=0.84,
        width=34,
        max_lines=None,
        font_size=9.0,
    )
    _draw_text_block(
        fig,
        "Supported Findings",
        _format_support_lines(payload),
        x=0.08,
        y=0.40,
        width=46,
        max_lines=None,
        font_size=9.0,
    )
    _draw_text_block(
        fig,
        "Uncertain Findings",
        _format_uncertain_lines(payload),
        x=0.54,
        y=0.40,
        width=46,
        max_lines=None,
        font_size=9.0,
    )
    _draw_text_block(
        fig,
        "Top Swin Probabilities",
        _format_top_probabilities(payload["case_packet"]).splitlines(),
        x=0.08,
        y=0.20,
        width=92,
        max_lines=None,
        font_size=9.0,
    )

    pdf.savefig(fig)
    plt.close(fig)


def export_pdf(report_path: Path, output_path: Path) -> None:
    payload = json.loads(report_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_png_path = output_path.with_name(f"{output_path.stem}_annotated.png")
    _save_annotated_png(payload, annotated_png_path)

    with PdfPages(output_path) as pdf:
        _draw_first_page(pdf, payload)
        _draw_clinical_summary_page(pdf, payload)
        _draw_evidence_page(pdf, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export review bundles to PDF.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Review JSON files or directories containing review_result.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write PDF files into",
    )
    args = parser.parse_args()

    report_paths = _resolve_report_inputs(args.inputs)
    output_dir = Path(args.output_dir)
    for report_path in report_paths:
        image_id = report_path.parent.name
        output_path = output_dir / f"{image_id}.pdf"
        export_pdf(report_path, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
