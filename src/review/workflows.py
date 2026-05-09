"""
Library-first workflows for review generation and evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.dataset import CLASS_NAMES
from src.data.detection import build_split_image_ids, load_image_level_ground_truth
from src.evaluation import evaluate_multilabel_predictions, format_case_table, prediction_row_from_labels
from src.review.inference import SwinInferenceEngine, YoloInferenceEngine
from src.review.metrics import build_case_buckets
from src.review.orchestrator import ReviewOrchestrator
from src.utils import load_config


def build_review_orchestrator(review_config_path: str) -> tuple[dict, dict, ReviewOrchestrator]:
    review_config = load_config(review_config_path)
    paths_config = review_config["paths"]
    review_section = review_config["review"]
    orchestrator = ReviewOrchestrator(
        review_config=review_config,
        prompt_path=paths_config["prompt_path"],
        swin_engine=SwinInferenceEngine(
            config_path=paths_config["swin_config"],
            checkpoint_path=paths_config["swin_checkpoint"],
            threshold=float(review_section.get("swin_threshold", 0.5)),
        ),
        yolo_engine=YoloInferenceEngine(
            config_path=paths_config["yolo_config"],
            weights_path=paths_config["yolo_weights"],
            conf_threshold=float(review_section.get("yolo_conf_threshold", 0.25)),
        ),
    )
    return review_config, paths_config, orchestrator


def run_review_case(
    image_path: str,
    review_config_path: str,
    output_dir: str,
    image_id: str | None = None,
    force_refresh: bool = False,
) -> dict:
    review_config, _, orchestrator = build_review_orchestrator(review_config_path)
    review_section = review_config["review"]

    resolved_image_path = Path(image_path)
    resolved_image_id = image_id or resolved_image_path.stem
    case_dir = Path(output_dir) / resolved_image_id
    case_dir.mkdir(parents=True, exist_ok=True)

    result = orchestrator.review_case(
        image_id=resolved_image_id,
        image_path=resolved_image_path,
        cache_dir=Path(review_section["cache_dir"]),
        force_refresh=force_refresh,
    )

    review_json_path = case_dir / "review_result.json"
    report_md_path = case_dir / "report.md"
    review_json_path.write_text(json.dumps(result, indent=2))
    report_md_path.write_text(result["report_text"])
    from src.review.pdf import export_pdf

    export_pdf(review_json_path, case_dir / "report.pdf")

    return {
        "image_id": resolved_image_id,
        "image_path": str(resolved_image_path),
        "output_dir": str(case_dir),
        "final_labels": result["review"]["final_labels"],
        "supported_findings": result["review"]["supported_findings"],
        "uncertain_findings": result["review"]["uncertain_findings"],
        "review_recommendation": result["review"]["review_recommendation"],
        "confidence_band": result["review"]["confidence_band"],
        "artifacts": {
            "review_json": str(review_json_path),
            "report_markdown": str(report_md_path),
            "report_pdf": str(case_dir / "report.pdf"),
        },
    }


def evaluate_review_run(
    review_config_path: str,
    split: str = "val",
    max_cases: int | None = None,
    output_dir: str | None = None,
    force_refresh: bool = False,
) -> dict:
    review_config, paths_config, orchestrator = build_review_orchestrator(review_config_path)
    review_section = review_config["review"]
    yolo_config = load_config(paths_config["yolo_config"])

    image_ids = build_split_image_ids(
        image_root=yolo_config["data"]["image_root"],
        train_split=yolo_config["data"].get("train_split", 0.8),
        val_split=yolo_config["data"].get("val_split", 0.2),
        seed=yolo_config.get("seed", 42),
    )[split]
    if max_cases is not None:
        image_ids = image_ids[:max_cases]

    report_dir = Path(output_dir or review_section["output_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(review_section["cache_dir"])

    review_rows = []
    case_rows = []
    failed_cases = []
    for image_id in tqdm(image_ids, desc="Reviewing cases"):
        image_path = Path(yolo_config["data"]["image_root"]) / f"{image_id}.png"
        try:
            result = orchestrator.review_case(
                image_id=image_id,
                image_path=image_path,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
        except Exception as exc:  # noqa: BLE001
            failed_cases.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "error": str(exc),
                }
            )
            continue

        review = result["review"]
        case_packet = result["case_packet"]
        review_rows.append(prediction_row_from_labels(image_id, review["final_labels"], CLASS_NAMES))
        case_rows.append(
            {
                "image_id": image_id,
                "swin_labels": case_packet["swin"]["predicted_labels"],
                "yolo_labels": case_packet["yolo"]["predicted_labels"],
                "claude_labels": review["final_labels"],
                "supported_findings": review["supported_findings"],
                "uncertain_findings": review["uncertain_findings"],
                "localization_supported_findings": review["localization_supported_findings"],
                "review_recommendation": review["review_recommendation"],
                "confidence_band": review["confidence_band"],
                "models_disagree": case_packet["agreement_summary"]["models_disagree"],
                "report_text": result["report_text"],
            }
        )

    processed_image_ids = [row["image_id"] for row in review_rows]
    y_true_df = load_image_level_ground_truth(yolo_config["data"]["classification_csv"], processed_image_ids)
    claude_pred_df = pd.DataFrame(review_rows).set_index("image_id")
    claude_metrics, claude_confusion_rows, claude_row_summaries = evaluate_multilabel_predictions(
        image_ids=processed_image_ids,
        y_true_df=y_true_df,
        y_pred_df=claude_pred_df,
        class_names=CLASS_NAMES,
    )

    swin_pred_df = pd.DataFrame(
        [prediction_row_from_labels(row["image_id"], row["swin_labels"], CLASS_NAMES) for row in case_rows]
    ).set_index("image_id")
    yolo_pred_df = pd.DataFrame(
        [prediction_row_from_labels(row["image_id"], row["yolo_labels"], CLASS_NAMES) for row in case_rows]
    ).set_index("image_id")

    swin_metrics, _, swin_row_summaries = evaluate_multilabel_predictions(
        image_ids=processed_image_ids,
        y_true_df=y_true_df,
        y_pred_df=swin_pred_df,
        class_names=CLASS_NAMES,
    )
    yolo_metrics, _, yolo_row_summaries = evaluate_multilabel_predictions(
        image_ids=processed_image_ids,
        y_true_df=y_true_df,
        y_pred_df=yolo_pred_df,
        class_names=CLASS_NAMES,
    )

    compare_df = pd.DataFrame(
        {
            "image_id": processed_image_ids,
            "true_labels": [row["true_labels"] for row in claude_row_summaries],
            "swin_predicted_labels": [row["predicted_labels"] for row in swin_row_summaries],
            "swin_label_errors": [row["label_errors"] for row in swin_row_summaries],
            "yolo_predicted_labels": [row["predicted_labels"] for row in yolo_row_summaries],
            "yolo_label_errors": [row["label_errors"] for row in yolo_row_summaries],
            "claude_predicted_labels": [row["predicted_labels"] for row in claude_row_summaries],
            "claude_label_errors": [row["label_errors"] for row in claude_row_summaries],
        }
    )
    case_meta_df = pd.DataFrame(case_rows)[
        [
            "image_id",
            "supported_findings",
            "uncertain_findings",
            "review_recommendation",
            "confidence_band",
            "models_disagree",
        ]
    ]
    compare_df = compare_df.merge(case_meta_df, on="image_id", how="left")
    buckets = build_case_buckets(compare_df)

    summary = {
        "requested_samples": int(len(image_ids)),
        "processed_samples": int(len(processed_image_ids)),
        "failed_samples": int(len(failed_cases)),
        "swin_metrics": swin_metrics,
        "yolo_metrics": yolo_metrics,
        "claude_metrics": claude_metrics,
        "case_buckets": {name: int(len(df)) for name, df in buckets.items()},
    }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (report_dir / "claude_image_level_metrics.json").write_text(json.dumps(claude_metrics, indent=2))
    (report_dir / "claude_per_class_confusion_matrix.json").write_text(json.dumps(claude_confusion_rows, indent=2))
    pd.DataFrame(case_rows).to_json(report_dir / "report_examples.jsonl", orient="records", lines=True)
    compare_df.to_csv(report_dir / "claude_vs_baselines_case_comparison.csv", index=False)
    if failed_cases:
        pd.DataFrame(failed_cases).to_json(report_dir / "failed_cases.jsonl", orient="records", lines=True)

    report_lines = [
        "# Claude Review Case Buckets",
        "",
        f"- Requested cases: {len(image_ids)}",
        f"- Processed cases: {len(processed_image_ids)}",
        f"- Failed cases: {len(failed_cases)}",
        f"- Claude better than Swin: {len(buckets['claude_better_than_swin'])}",
        f"- Claude better than YOLO: {len(buckets['claude_better_than_yolo'])}",
        f"- Swin better than Claude: {len(buckets['swin_better_than_claude'])}",
        f"- YOLO better than Claude: {len(buckets['yolo_better_than_claude'])}",
        f"- All wrong: {len(buckets['all_wrong'])}",
        "",
        "## Claude Better Than Swin",
        "```csv",
        format_case_table(buckets["claude_better_than_swin"].head(10)),
        "```",
        "",
        "## Claude Better Than YOLO",
        "```csv",
        format_case_table(buckets["claude_better_than_yolo"].head(10)),
        "```",
        "",
        "## Swin Better Than Claude",
        "```csv",
        format_case_table(buckets["swin_better_than_claude"].head(10)),
        "```",
        "",
        "## YOLO Better Than Claude",
        "```csv",
        format_case_table(buckets["yolo_better_than_claude"].head(10)),
        "```",
        "",
        "## All Wrong",
        "```csv",
        format_case_table(buckets["all_wrong"].head(10)),
        "```",
    ]
    (report_dir / "failure_buckets.md").write_text("\n".join(report_lines))
    return summary
