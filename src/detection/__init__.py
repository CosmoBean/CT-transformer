"""
High-level detection workflows.
"""

from .workflows import evaluate_yolo_run, infer_yolo_image, train_yolo_run

__all__ = [
    "evaluate_yolo_run",
    "infer_yolo_image",
    "train_yolo_run",
]
