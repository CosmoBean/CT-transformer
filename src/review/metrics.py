"""
Metrics and case-review helpers for the Claude review workflow.
"""
from __future__ import annotations

import pandas as pd

from src.data.dataset import CLASS_NAMES
from src.evaluation import evaluate_multilabel_predictions as _evaluate_multilabel_predictions


def evaluate_multilabel_predictions(
    image_ids: list[str],
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
) -> tuple[dict, list[dict], list[dict]]:
    return _evaluate_multilabel_predictions(
        image_ids=image_ids,
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        class_names=CLASS_NAMES,
    )


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
