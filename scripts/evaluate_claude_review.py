#!/usr/bin/env python3
"""
Evaluate Swin + YOLO + Claude review on a validation slice and compare with base methods.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import CLASS_NAMES
from src.data.detection import build_split_image_ids, load_image_level_ground_truth
from src.review import (
    ReviewOrchestrator,
    SwinInferenceEngine,
    YoloInferenceEngine,
    build_case_buckets,
    evaluate_multilabel_predictions,
)
from src.utils import load_config


def _prediction_row_from_labels(image_id: str, labels: list[str]) -> dict[str, int | str]:
    return {"image_id": image_id, **{class_name: int(class_name in labels) for class_name in CLASS_NAMES}}


def _format_case_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_csv(index=False).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Claude-backed review workflow.")
    parser.add_argument("--review-config", default="configs/claude_review.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force-refresh", action="store_true")
    args = parser.parse_args()

    review_config = load_config(args.review_config)
    paths_config = review_config["paths"]
    review_section = review_config["review"]
    yolo_config = load_config(paths_config["yolo_config"])

    image_ids = build_split_image_ids(
        image_root=yolo_config["data"]["image_root"],
        train_split=yolo_config["data"].get("train_split", 0.8),
        val_split=yolo_config["data"].get("val_split", 0.2),
        seed=yolo_config.get("seed", 42),
    )[args.split]
    if args.max_cases is not None:
        image_ids = image_ids[: args.max_cases]

    output_dir = Path(args.output_dir or review_section["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(review_section["cache_dir"])

    swin_engine = SwinInferenceEngine(
        config_path=paths_config["swin_config"],
        checkpoint_path=paths_config["swin_checkpoint"],
        threshold=float(review_section.get("swin_threshold", 0.5)),
    )
    yolo_engine = YoloInferenceEngine(
        config_path=paths_config["yolo_config"],
        weights_path=paths_config["yolo_weights"],
        conf_threshold=float(review_section.get("yolo_conf_threshold", 0.25)),
    )
    orchestrator = ReviewOrchestrator(
        review_config=review_config,
        prompt_path=paths_config["prompt_path"],
        swin_engine=swin_engine,
        yolo_engine=yolo_engine,
    )

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
                force_refresh=args.force_refresh,
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

        review_rows.append(_prediction_row_from_labels(image_id, review["final_labels"]))
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
    )

    swin_pred_df = pd.DataFrame(
        [_prediction_row_from_labels(row["image_id"], row["swin_labels"]) for row in case_rows]
    ).set_index("image_id")
    yolo_pred_df = pd.DataFrame(
        [_prediction_row_from_labels(row["image_id"], row["yolo_labels"]) for row in case_rows]
    ).set_index("image_id")

    swin_metrics, _, swin_row_summaries = evaluate_multilabel_predictions(processed_image_ids, y_true_df, swin_pred_df)
    yolo_metrics, _, yolo_row_summaries = evaluate_multilabel_predictions(processed_image_ids, y_true_df, yolo_pred_df)

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

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "claude_image_level_metrics.json").write_text(json.dumps(claude_metrics, indent=2))
    (output_dir / "claude_per_class_confusion_matrix.json").write_text(json.dumps(claude_confusion_rows, indent=2))
    pd.DataFrame(case_rows).to_json(output_dir / "report_examples.jsonl", orient="records", lines=True)
    compare_df.to_csv(output_dir / "claude_vs_baselines_case_comparison.csv", index=False)
    if failed_cases:
        pd.DataFrame(failed_cases).to_json(output_dir / "failed_cases.jsonl", orient="records", lines=True)

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
        _format_case_table(buckets["claude_better_than_swin"].head(10)),
        "```",
        "",
        "## Claude Better Than YOLO",
        "```csv",
        _format_case_table(buckets["claude_better_than_yolo"].head(10)),
        "```",
        "",
        "## Swin Better Than Claude",
        "```csv",
        _format_case_table(buckets["swin_better_than_claude"].head(10)),
        "```",
        "",
        "## YOLO Better Than Claude",
        "```csv",
        _format_case_table(buckets["yolo_better_than_claude"].head(10)),
        "```",
        "",
        "## All Wrong",
        "```csv",
        _format_case_table(buckets["all_wrong"].head(10)),
        "```",
    ]
    (output_dir / "failure_buckets.md").write_text("\n".join(report_lines))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
