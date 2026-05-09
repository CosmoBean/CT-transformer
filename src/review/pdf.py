"""
ReportLab-based PDF export helpers for review bundles.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus import KeepTogether


BOX_COLORS = ["#D62828", "#1D3557", "#2A9D8F", "#F4A261", "#7B2CBF"]
SAFETY_FOOTER = "AI-generated decision-support summary; human review required for clinical use."


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
    for index, detection in enumerate(detections[:5]):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1, y1, x2, y2 = [float(value) for value in detection["bbox_xyxy"]]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.2, edgecolor=color))
        ax.text(
            x1,
            max(4, y1 - 6),
            f"{detection['class_name']} {float(detection['confidence']):.2f}",
            fontsize=8.5,
            color="white",
            bbox={"facecolor": color, "edgecolor": color, "pad": 1.5},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _build_at_a_glance(payload: dict) -> list[str]:
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


def _format_yolo_legend(payload: dict) -> list[str]:
    detections = sorted(
        payload["case_packet"]["yolo"]["detections"],
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    if not detections:
        return ["No detections above threshold."]
    lines = []
    for detection in detections[:5]:
        box = ", ".join(f"{float(value):.1f}" for value in detection["bbox_xyxy"])
        lines.append(f"{detection['class_name']} ({float(detection['confidence']):.2f}) @ [{box}]")
    return lines


def _format_probability_lines(payload: dict, labels: list[str]) -> list[str]:
    probabilities = payload["case_packet"]["swin"]["probabilities"]
    if not labels:
        return ["None"]
    return [f"{label}: {float(probabilities.get(label, 0.0)):.2f}" for label in labels]


def _format_top_probabilities(payload: dict) -> list[str]:
    items = payload["case_packet"]["swin"]["sorted_probabilities"][:8]
    return [f"{item['label']}: {float(item['probability']):.3f}" for item in items]


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#102A43"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#243B53"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#102A43"),
            spaceBefore=6,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#102A43"),
            spaceBefore=4,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#243B53"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SafetyBanner",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12.5,
            textColor=colors.HexColor("#8A1C1C"),
            backColor=colors.HexColor("#FDECEC"),
            borderColor=colors.HexColor("#F5C2C7"),
            borderWidth=0.5,
            borderPadding=6,
            borderRadius=2,
            spaceAfter=10,
        )
    )
    return styles


def _bullet_list(items: list[str], body_style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, body_style)) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=16,
    )


def _scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    width = float(image.imageWidth)
    height = float(image.imageHeight)
    scale = min(max_width / width, max_height / height)
    image.drawWidth = width * scale
    image.drawHeight = height * scale
    return image


def _section_block(title: str, content, styles) -> KeepTogether:
    return KeepTogether([Paragraph(title, styles["SubHeading"]), content])


def _image_table(original_image: Path, annotated_image: Path, styles) -> Table:
    original = _scaled_image(original_image, max_width=3.0 * inch, max_height=3.0 * inch)
    annotated = _scaled_image(annotated_image, max_width=3.0 * inch, max_height=3.0 * inch)
    table = Table(
        [
            [Paragraph("Original CXR", styles["SubHeading"]), Paragraph("YOLO Support Overlay", styles["SubHeading"])],
            [original, annotated],
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return table


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5B7083"))
    canvas.drawString(doc.leftMargin, 0.45 * inch, SAFETY_FOOTER)
    canvas.restoreState()


def export_pdf(report_path: Path, output_path: Path) -> None:
    payload = json.loads(report_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    review = payload["review"]
    original_image_path = Path(payload["image_path"])

    with tempfile.TemporaryDirectory(prefix=f"{payload['image_id']}_report_") as tmp_dir:
        annotated_png_path = Path(tmp_dir) / f"{output_path.stem}_annotated.png"
        _save_annotated_png(payload, annotated_png_path)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title="AI Decision-Support Report",
        )

        story = [
            Paragraph("AI Decision-Support Report", styles["ReportTitle"]),
            Paragraph(
                f"Case: {payload['image_id']}<br/>Source image: {original_image_path.name}",
                styles["ReportSubtitle"],
            ),
            Paragraph(SAFETY_FOOTER, styles["SafetyBanner"]),
            HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D9E2EC")),
            Spacer(1, 0.15 * inch),
            Paragraph("Images", styles["SectionHeading"]),
            _image_table(original_image_path, annotated_png_path, styles),
            Spacer(1, 0.2 * inch),
            Paragraph("Clinical Summary", styles["SectionHeading"]),
            _bullet_list(_build_at_a_glance(payload), styles["Body"]),
            _section_block("Findings", Paragraph(review["findings_section"], styles["Body"]), styles),
            _section_block("Impression", Paragraph(review["impression_section"], styles["Body"]), styles),
            _section_block("Safety Note", Paragraph(review["safety_note"], styles["Body"]), styles),
            Spacer(1, 0.1 * inch),
            Paragraph("Supporting Evidence", styles["SectionHeading"]),
            _section_block("YOLO Detections", _bullet_list(_format_yolo_legend(payload), styles["Body"]), styles),
            _section_block("Supported Findings", _bullet_list(_format_probability_lines(payload, review["supported_findings"]), styles["Body"]), styles),
            _section_block("Uncertain Findings", _bullet_list(_format_probability_lines(payload, review["uncertain_findings"]), styles["Body"]), styles),
            _section_block("Top Swin Probabilities", _bullet_list(_format_top_probabilities(payload), styles["Body"]), styles),
        ]

        doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
