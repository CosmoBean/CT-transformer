"""
Shared multilabel evaluation helpers for classifier and pipeline outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def prediction_row_from_labels(
    image_id: str,
    labels: list[str],
    class_names: list[str],
) -> dict[str, int | str]:
    return {
        "image_id": image_id,
        **{class_name: int(class_name in labels) for class_name in class_names},
    }


def format_case_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_csv(index=False).strip()


def _safe_macro_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred, average="macro"))
    except ValueError:
        return 0.0


def evaluate_multilabel_predictions(
    image_ids: list[str],
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    class_names: list[str],
) -> tuple[dict, list[dict], list[dict]]:
    y_true = y_true_df[class_names].to_numpy(dtype=int)
    y_pred = y_pred_df[class_names].to_numpy(dtype=int)
    per_class_accuracy = (y_true == y_pred).mean(axis=0)

    metrics = {
        "samples": int(len(image_ids)),
        "exact_match_accuracy": float(np.all(y_true == y_pred, axis=1).mean()),
        "macro_accuracy": float(per_class_accuracy.mean()),
        "macro_auc_roc": _safe_macro_auc_roc(y_true, y_pred),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_accuracy": {
            class_name: float(acc)
            for class_name, acc in zip(class_names, per_class_accuracy)
        },
    }

    confusion_rows = []
    for idx, class_name in enumerate(class_names):
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
                "true_labels": [class_names[i] for i, value in enumerate(true_row) if value == 1],
                "predicted_labels": [class_names[i] for i, value in enumerate(pred_row) if value == 1],
                "label_errors": int(np.not_equal(true_row, pred_row).sum()),
            }
        )

    return metrics, confusion_rows, row_summaries
