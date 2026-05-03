"""
Metrics and case-review helpers for the Claude review workflow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.data.dataset import CLASS_NAMES


def evaluate_multilabel_predictions(
    image_ids: list[str],
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
) -> tuple[dict, list[dict], list[dict]]:
    y_true = y_true_df[CLASS_NAMES].to_numpy(dtype=int)
    y_pred = y_pred_df[CLASS_NAMES].to_numpy(dtype=int)
    per_class_accuracy = (y_true == y_pred).mean(axis=0)
    metrics = {
        "samples": int(len(image_ids)),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "macro_accuracy": float(per_class_accuracy.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_accuracy": {
            class_name: float(acc)
            for class_name, acc in zip(CLASS_NAMES, per_class_accuracy)
        },
    }

    confusion_rows = []
    for idx, class_name in enumerate(CLASS_NAMES):
        tn, fp, fn, tp = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1]).ravel()
        confusion_rows.append(
            {
                "class_name": class_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    row_summaries = []
    for image_id, true_row, pred_row in zip(image_ids, y_true, y_pred):
        row_summaries.append(
            {
                "image_id": image_id,
                "true_labels": [CLASS_NAMES[i] for i, value in enumerate(true_row) if value == 1],
                "predicted_labels": [CLASS_NAMES[i] for i, value in enumerate(pred_row) if value == 1],
                "label_errors": int(np.not_equal(true_row, pred_row).sum()),
            }
        )

    return metrics, confusion_rows, row_summaries


def build_case_buckets(compare_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "claude_better_than_swin": compare_df[
            compare_df["claude_label_errors"] < compare_df["swin_label_errors"]
        ],
        "claude_better_than_yolo": compare_df[
            compare_df["claude_label_errors"] < compare_df["yolo_label_errors"]
        ],
        "swin_better_than_claude": compare_df[
            compare_df["swin_label_errors"] < compare_df["claude_label_errors"]
        ],
        "yolo_better_than_claude": compare_df[
            compare_df["yolo_label_errors"] < compare_df["claude_label_errors"]
        ],
        "all_wrong": compare_df[
            (compare_df["swin_label_errors"] > 0)
            & (compare_df["yolo_label_errors"] > 0)
            & (compare_df["claude_label_errors"] > 0)
        ],
    }
