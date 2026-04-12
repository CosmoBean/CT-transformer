#!/usr/bin/env python3
"""
Evaluate the Swin-based agentic triage workflow on the validation split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent import AgentAction, AgentTriagePolicy, SwinTriageService
from src.data.dataset import ChestXRayDataset
from src.utils import load_config


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _binary_ground_truth(labels: np.ndarray) -> str:
    return "abnormal" if float(labels[:-1].sum()) > 0.0 else "normal"


def _auto_prediction(action: str) -> str | None:
    if action == AgentAction.ACCEPT_NORMAL.value:
        return "normal"
    if action == AgentAction.ACCEPT_ABNORMAL.value:
        return "abnormal"
    return None


def _representative_rows(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    if "agent_action" in df.columns and (df["agent_action"] == AgentAction.FLAG_FOR_REVIEW.value).all():
        sort_cols = ["abnormal_probability_margin", "confidence_score"]
        ascending = [True, False]
    else:
        sort_cols = ["confidence_score", "heuristic_error"]
        ascending = [False, False]
    existing_cols = [col for col in sort_cols if col in df.columns]
    existing_ascending = ascending[: len(existing_cols)]
    if existing_cols:
        return df.sort_values(existing_cols, ascending=existing_ascending).head(limit)
    return df.head(limit)


def _write_case_section(handle, title: str, df: pd.DataFrame):
    handle.write(f"## {title}\n\n")
    if df.empty:
        handle.write("No cases found.\n\n")
        return

    columns = [
        "image_id",
        "ground_truth_binary",
        "heuristic_binary_prediction",
        "agent_action",
        "confidence",
        "no_finding_probability",
        "max_abnormal_probability",
        "abnormal_probability_margin",
        "top_findings",
    ]
    handle.write(_markdown_table(df[columns]))
    handle.write("\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the agentic triage workflow")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="reports/agent")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(config["data"]["data_dir"])
    csv_path = data_dir / config["data"].get("train_csv", "train.csv")

    dataset = ChestXRayDataset(
        data_dir=str(data_dir),
        csv_path=str(csv_path) if csv_path.exists() else None,
        image_size=args.image_size,
        split="val",
        mode="classification",
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    sample_lookup = {sample.image_id: sample for sample in dataset.samples}
    service = SwinTriageService(
        checkpoint_path=args.checkpoint_path,
        image_size=args.image_size,
    )
    policy = AgentTriagePolicy()

    rows: list[dict] = []
    for batch in loader:
        image_ids = list(batch["image_id"])
        image_paths = [str(sample_lookup[image_id].image_path) for image_id in image_ids]
        summaries = service.summarize_batch(
            image_ids=image_ids,
            image_paths=image_paths,
            images=batch["image"],
        )

        for image_id, labels, summary in zip(image_ids, batch["labels"].numpy(), summaries):
            decision = policy.decide(summary)
            ground_truth_binary = _binary_ground_truth(labels)
            auto_prediction = _auto_prediction(decision.action.value)
            heuristic_error = int(summary.heuristic_binary_prediction != ground_truth_binary)
            auto_error = None if auto_prediction is None else int(auto_prediction != ground_truth_binary)

            rows.append(
                {
                    "image_id": image_id,
                    "image_path": summary.image_path,
                    "ground_truth_binary": ground_truth_binary,
                    "true_labels": json.dumps(labels.tolist()),
                    "agent_action": decision.action.value,
                    "confidence": decision.confidence.value,
                    "predicted_findings": json.dumps(decision.predicted_findings),
                    "recommendation": decision.recommendation,
                    "rationale": decision.rationale,
                    "heuristic_binary_prediction": summary.heuristic_binary_prediction,
                    "auto_prediction": auto_prediction,
                    "heuristic_error": heuristic_error,
                    "auto_error": auto_error,
                    "no_finding_probability": summary.no_finding_probability,
                    "max_abnormal_probability": summary.max_abnormal_probability,
                    "abnormal_probability_margin": summary.abnormal_probability_margin,
                    "moderate_findings_count": summary.moderate_findings_count,
                    "top_findings": json.dumps(summary.top_findings),
                    "confidence_score": max(summary.no_finding_probability, summary.max_abnormal_probability),
                }
            )

    df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "agent_eval.csv"
    df.to_csv(csv_path, index=False)

    auto_df = df[df["auto_prediction"].notna()].copy()
    flagged_df = df[df["agent_action"] == AgentAction.FLAG_FOR_REVIEW.value].copy()

    cm = confusion_matrix(
        auto_df["ground_truth_binary"],
        auto_df["auto_prediction"],
        labels=["normal", "abnormal"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Abnormal"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Abnormal"])
    ax.set_title("Accepted-Case Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    action_counts = (
        df.groupby(["ground_truth_binary", "agent_action"])
        .size()
        .reset_index(name="count")
    )
    pivot = action_counts.pivot(index="ground_truth_binary", columns="agent_action", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Agent Action Breakdown by Ground Truth")
    ax.set_ylabel("Cases")
    fig.tight_layout()
    fig.savefig(output_dir / "action_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    overall_heuristic_error = float(df["heuristic_error"].mean())
    flagged_heuristic_error = float(flagged_df["heuristic_error"].mean()) if not flagged_df.empty else 0.0
    auto_error_rate = float(auto_df["auto_error"].mean()) if not auto_df.empty else 0.0
    coverage = float(len(auto_df) / len(df)) if len(df) > 0 else 0.0

    summary_payload = {
        "num_cases": int(len(df)),
        "coverage": coverage,
        "auto_error_rate": auto_error_rate,
        "overall_heuristic_binary_error_rate": overall_heuristic_error,
        "flagged_heuristic_binary_error_rate": flagged_heuristic_error,
        "review_rate": float(len(flagged_df) / len(df)) if len(df) > 0 else 0.0,
        "action_counts": {str(k): int(v) for k, v in df["agent_action"].value_counts().to_dict().items()},
        "accepted_confusion_matrix": {
            "labels": ["normal", "abnormal"],
            "matrix": cm.tolist(),
        },
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary_payload, handle, indent=2)

    with open(output_dir / "confusion_matrix_summary.md", "w") as handle:
        handle.write("# Agent Confusion Matrix Summary\n\n")
        handle.write(
            "Accepted-case metrics below are based on agent actions only. "
            "Flagged-case heuristic metrics use the model's derived normal-vs-abnormal proxy "
            "(`No finding` versus strongest abnormal class) and should be interpreted as heuristic, not as a separate trained binary model.\n\n"
        )
        handle.write(f"- Cases evaluated: `{len(df)}`\n")
        handle.write(f"- Automatic coverage: `{coverage:.3f}`\n")
        handle.write(f"- Accepted-case error rate: `{auto_error_rate:.3f}`\n")
        handle.write(f"- Overall heuristic binary error rate: `{overall_heuristic_error:.3f}`\n")
        handle.write(f"- Flagged-case heuristic binary error rate: `{flagged_heuristic_error:.3f}`\n")
        handle.write(f"- Review rate: `{summary_payload['review_rate']:.3f}`\n\n")
        handle.write("## Accepted-Case Confusion Matrix\n\n")
        handle.write(
            _markdown_table(
                pd.DataFrame(
                    cm,
                    index=["True Normal", "True Abnormal"],
                    columns=["Pred Normal", "Pred Abnormal"],
                ).reset_index(names="ground_truth")
            )
        )
        handle.write("\n\n## Action Counts\n\n")
        handle.write(
            _markdown_table(
                df["agent_action"].value_counts().rename_axis("action").reset_index(name="count")
            )
        )
        handle.write("\n")

    tn_df = auto_df[(auto_df["ground_truth_binary"] == "normal") & (auto_df["auto_prediction"] == "normal")]
    tp_df = auto_df[(auto_df["ground_truth_binary"] == "abnormal") & (auto_df["auto_prediction"] == "abnormal")]
    fp_df = auto_df[(auto_df["ground_truth_binary"] == "normal") & (auto_df["auto_prediction"] == "abnormal")]
    fn_df = auto_df[(auto_df["ground_truth_binary"] == "abnormal") & (auto_df["auto_prediction"] == "normal")]
    flagged_error_df = flagged_df[flagged_df["heuristic_error"] == 1]

    with open(output_dir / "case_review.md", "w") as handle:
        handle.write("# Agent Case Review\n\n")
        _write_case_section(handle, "True Negatives (Accepted Normal)", _representative_rows(tn_df))
        _write_case_section(handle, "True Positives (Accepted Abnormal)", _representative_rows(tp_df))
        _write_case_section(handle, "False Positives (Incorrectly Accepted Abnormal)", _representative_rows(fp_df))
        _write_case_section(handle, "False Negatives (Incorrectly Accepted Normal)", _representative_rows(fn_df))
        _write_case_section(handle, "Flagged Cases", _representative_rows(flagged_df))
        _write_case_section(handle, "Flagged Cases With Wrong Heuristic Prediction", _representative_rows(flagged_error_df))

    print(f"Agent evaluation complete. Artifacts written to {output_dir}")
    print(json.dumps(summary_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
