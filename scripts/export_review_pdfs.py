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
from matplotlib.backends.backend_pdf import PdfPages


PAGE_SIZE = (8.5, 11)
TITLE_FONTSIZE = 18
HEADING_FONTSIZE = 12
BODY_FONTSIZE = 10
LINE_HEIGHT = 0.027
LEFT_MARGIN = 0.08
TOP_MARGIN = 0.95
BOTTOM_MARGIN = 0.06
TEXT_WIDTH = 92


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


def _draw_first_page(pdf: PdfPages, payload: dict) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")

    image_path = Path(payload["image_path"])
    image = mpimg.imread(image_path)

    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN,
        "AI Decision-Support Report",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        LEFT_MARGIN,
        TOP_MARGIN - 0.04,
        f"Case: {payload['image_id']}",
        fontsize=BODY_FONTSIZE,
        ha="left",
        va="top",
    )

    ax_image = fig.add_axes([0.08, 0.46, 0.42, 0.38])
    ax_image.imshow(image, cmap="gray")
    ax_image.set_title("Chest X-ray", fontsize=HEADING_FONTSIZE)
    ax_image.axis("off")

    review = payload["review"]
    summary_lines = [
        f"Recommendation: {review['review_recommendation']}",
        f"Confidence: {review['confidence_band']}",
        "",
        "Supported findings:",
        ", ".join(review["supported_findings"]) or "None",
        "",
        "Uncertain findings:",
        ", ".join(review["uncertain_findings"]) or "None",
        "",
        "Supported global buckets:",
        ", ".join(review.get("supported_global_buckets", [])) or "None",
        "",
        "Localization support:",
        ", ".join(review["localization_supported_findings"]) or "None",
    ]
    fig.text(
        0.56,
        0.83,
        "\n".join(summary_lines),
        fontsize=BODY_FONTSIZE,
        ha="left",
        va="top",
    )

    report_excerpt = _wrap_paragraphs(_strip_markdown(payload["report_text"]), width=82)
    excerpt_lines = report_excerpt[:math.floor((0.36 - BOTTOM_MARGIN) / LINE_HEIGHT)]
    fig.text(
        LEFT_MARGIN,
        0.39,
        "\n".join(excerpt_lines),
        fontsize=BODY_FONTSIZE,
        ha="left",
        va="top",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _draw_text_pages(pdf: PdfPages, title: str, text: str) -> None:
    wrapped_lines = _wrap_paragraphs(_strip_markdown(text))
    max_lines = math.floor((TOP_MARGIN - BOTTOM_MARGIN - 0.05) / LINE_HEIGHT)
    for page_index, chunk in enumerate(_chunk_lines(wrapped_lines, max_lines), start=1):
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        heading = title if page_index == 1 else f"{title} (cont.)"
        fig.text(
            LEFT_MARGIN,
            TOP_MARGIN,
            heading,
            fontsize=HEADING_FONTSIZE,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            LEFT_MARGIN,
            TOP_MARGIN - 0.04,
            "\n".join(chunk),
            fontsize=BODY_FONTSIZE,
            ha="left",
            va="top",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def export_pdf(report_path: Path, output_path: Path) -> None:
    payload = json.loads(report_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_sections = [
        _format_top_probabilities(payload["case_packet"]),
        "",
        _format_yolo_summary(payload["case_packet"]),
    ]
    conflicts = payload["review"].get("conflicts", [])
    if conflicts:
        evidence_sections.extend(["", "Conflicts:"])
        evidence_sections.extend(f"- {conflict}" for conflict in conflicts)

    with PdfPages(output_path) as pdf:
        _draw_first_page(pdf, payload)
        _draw_text_pages(pdf, "Rendered Report", payload["report_text"])
        _draw_text_pages(pdf, "Model Evidence Summary", "\n".join(evidence_sections))


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
