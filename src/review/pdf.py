"""
PyLaTeX-based PDF export helpers for review bundles.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pylatex import Command, Document, Figure, Itemize, MiniPage, NewPage, Package, Section, Subsection
from pylatex.utils import NoEscape, bold, escape_latex


BOX_COLORS = ["#D62828", "#1D3557", "#2A9D8F", "#F4A261", "#7B2CBF"]


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
    ax.set_title("YOLO Support Overlay", loc="left", pad=8)
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


def _append_bullet_list(container, items: list[str]) -> None:
    bullet_list = Itemize()
    for item in items:
        bullet_list.add_item(escape_latex(item))
    container.append(bullet_list)


def _append_text_block(container, title: str, body: str) -> None:
    container.append(bold(escape_latex(title)))
    container.append(NoEscape(r"\par"))
    container.append(escape_latex(body))
    container.append(NoEscape(r"\par\medskip"))


def _add_title_block(doc: Document, payload: dict) -> None:
    image_path = Path(payload["image_path"])
    doc.append(NoEscape(r"\begin{center}"))
    doc.append(NoEscape(r"{\LARGE \textbf{AI Decision-Support Report}\par}"))
    doc.append(NoEscape(r"\vspace{0.5em}"))
    doc.append(NoEscape(
        rf"{{\large Case: {escape_latex(payload['image_id'])} \quad Source image: {escape_latex(image_path.name)}\par}}"
    ))
    doc.append(NoEscape(r"\end{center}"))
    doc.append(NoEscape(r"\vspace{1em}"))


def _add_image_section(doc: Document, original_image: Path, annotated_image: Path) -> None:
    with doc.create(Section("Images")):
        with doc.create(MiniPage(width=NoEscape(r"0.48\textwidth"))) as left:
            left.append(bold("Original CXR"))
            left.append(NoEscape(r"\par"))
            with left.create(Figure(position="H")) as fig:
                fig.add_image(str(original_image.resolve()), width=NoEscape(r"0.95\linewidth"))

        doc.append(NoEscape(r"\hfill"))

        with doc.create(MiniPage(width=NoEscape(r"0.48\textwidth"))) as right:
            right.append(bold("YOLO Support Overlay"))
            right.append(NoEscape(r"\par"))
            with right.create(Figure(position="H")) as fig:
                fig.add_image(str(annotated_image.resolve()), width=NoEscape(r"0.95\linewidth"))


def _add_summary_section(doc: Document, payload: dict) -> None:
    review = payload["review"]
    with doc.create(Section("Clinical Summary")):
        _append_bullet_list(doc, _build_at_a_glance(payload))
        _append_text_block(doc, "Findings", review["findings_section"])
        _append_text_block(doc, "Impression", review["impression_section"])
        _append_text_block(doc, "Safety Note", review["safety_note"])


def _add_evidence_section(doc: Document, payload: dict) -> None:
    review = payload["review"]
    with doc.create(Section("Supporting Evidence")):
        with doc.create(Subsection("YOLO Detections")):
            _append_bullet_list(doc, _format_yolo_legend(payload))
        with doc.create(Subsection("Supported Findings")):
            _append_bullet_list(doc, _format_probability_lines(payload, review["supported_findings"]))
        with doc.create(Subsection("Uncertain Findings")):
            _append_bullet_list(doc, _format_probability_lines(payload, review["uncertain_findings"]))
        with doc.create(Subsection("Top Swin Probabilities")):
            _append_bullet_list(doc, _format_top_probabilities(payload))


def _create_document() -> Document:
    doc = Document(documentclass="article", geometry_options=["margin=1in"])
    doc.packages.append(Package("graphicx"))
    doc.packages.append(Package("float"))
    doc.packages.append(Package("parskip"))
    doc.packages.append(Package("hyperref"))
    doc.preamble.append(Command("title", "AI Decision-Support Report"))
    doc.preamble.append(Command("date", NoEscape(r"\today")))
    return doc


def _compile_latex(tex_path: Path) -> None:
    output_dir = tex_path.parent
    tectonic = shutil.which("tectonic")
    if tectonic:
        result = subprocess.run(
            [tectonic, "--keep-logs", "--keep-intermediates", "--outdir", str(output_dir), str(tex_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tectonic failed to compile report:\n{result.stdout}")
        return

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        for _ in range(2):
            subprocess.run(
                [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(output_dir),
                    str(tex_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return

    xelatex = shutil.which("xelatex")
    if xelatex:
        for _ in range(2):
            subprocess.run(
                [
                    xelatex,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(output_dir),
                    str(tex_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return

    raise RuntimeError(
        "No LaTeX compiler found. Install the Python dependency set from requirements.txt "
        "to get tectonic, or install pdflatex/xelatex locally."
    )


def export_pdf(report_path: Path, output_path: Path) -> None:
    payload = json.loads(report_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated_png_path = output_path.with_name(f"{output_path.stem}_annotated.png")
    _save_annotated_png(payload, annotated_png_path)

    doc = _create_document()
    _add_title_block(doc, payload)
    _add_image_section(doc, Path(payload["image_path"]), annotated_png_path)
    doc.append(NewPage())
    _add_summary_section(doc, payload)
    _add_evidence_section(doc, payload)

    tex_base = output_path.with_suffix("")
    doc.generate_tex(str(tex_base))
    _compile_latex(output_path.with_suffix(".tex"))
