"""
Library-first workflows for YOLO preparation, training, evaluation, and inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.detection import (
    build_split_image_ids,
    derive_image_level_labels_from_detections,
    load_image_level_ground_truth,
    prepare_yolo_dataset,
)
from src.data.dataset import CLASS_NAMES
from src.evaluation import evaluate_multilabel_predictions, format_case_table
from src.training.inference import predict_classifier_dataset
from src.utils import load_config, save_config


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO workflows. Install the project dependencies first."
        ) from exc
    return YOLO


def _extract_detection_metrics(metrics: Any) -> dict[str, Any]:
    box_metrics = getattr(metrics, "box", None)
    return {
        "map50": float(getattr(box_metrics, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box_metrics, "map", 0.0) or 0.0),
        "maps": [float(value) for value in getattr(box_metrics, "maps", [])] if box_metrics is not None else [],
    }


def prepare_detection_dataset_from_config(
    config: dict,
    output_dir: str | None = None,
    max_images_per_split: int | None = None,
    link_mode: str | None = None,
) -> Path:
    dataset_output_dir = Path(output_dir or config["data"]["output_dir"])
    prepare_yolo_dataset(
        image_root=config["data"]["image_root"],
        raw_annotation_path=config["data"]["raw_annotation_path"],
        image_metadata_path=config["data"]["image_metadata_path"],
        output_dir=str(dataset_output_dir),
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
        merge_iou_threshold=config["data"].get("merge_iou_threshold", 0.3),
        link_mode=link_mode or config["data"].get("link_mode", "symlink"),
        max_images_per_split=max_images_per_split,
    )
    return dataset_output_dir


def infer_yolo_image(
    config_path: str,
    weights_path: str,
    image_path: str,
    conf_threshold: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    threshold = conf_threshold if conf_threshold is not None else config["evaluation"].get("conf_threshold", 0.25)
    YOLO = _load_ultralytics()
    model = YOLO(weights_path)
    result = model.predict(
        source=image_path,
        imgsz=config["data"]["image_size"],
        conf=threshold,
        device=config.get("device", 0),
        verbose=False,
    )[0]

    names = result.names
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        for box, confidence, class_id in zip(xyxy, confs, classes):
            class_id = int(class_id)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )

    image_level = derive_image_level_labels_from_detections(
        detections,
        confidence_threshold=threshold,
    )
    return {
        "image": str(Path(image_path)),
        "weights": str(Path(weights_path)),
        "confidence_threshold": threshold,
        "detections": detections,
        "image_level_labels": image_level,
    }


def train_yolo_run(
    config_path: str,
    epochs: int | None = None,
    weights: str | None = None,
    batch_size: int | None = None,
    project_dir: str | None = None,
    run_name: str | None = None,
    dataset_output_dir: str | None = None,
    max_images_per_split: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    YOLO = _load_ultralytics()

    dataset_dir = prepare_detection_dataset_from_config(
        config=config,
        output_dir=dataset_output_dir,
        max_images_per_split=max_images_per_split,
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    resolved_weights = weights or config["model"]["weights"]
    resolved_epochs = epochs or config["training"]["num_epochs"]
    resolved_batch_size = batch_size or config["training"]["batch_size"]
    resolved_project_dir = str(Path(project_dir or config["training"]["project_dir"]).resolve())
    resolved_run_name = run_name or config["training"]["run_name"]

    model = YOLO(resolved_weights)
    model.train(
        data=str(dataset_yaml),
        imgsz=config["data"]["image_size"],
        epochs=resolved_epochs,
        batch=resolved_batch_size,
        workers=config["training"].get("num_workers", 4),
        patience=config["training"].get("patience", 10),
        project=resolved_project_dir,
        name=resolved_run_name,
        device=config.get("device", 0),
        pretrained=True,
        verbose=True,
    )

    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=config["data"]["image_size"],
        batch=resolved_batch_size,
        device=config.get("device", 0),
        conf=config["evaluation"].get("conf_threshold", 0.25),
        verbose=False,
    )
    summary = _extract_detection_metrics(metrics)

    run_dir = Path(resolved_project_dir) / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detection_metrics.json").write_text(json.dumps(summary, indent=2))
    save_config(config, str(run_dir / "resolved_config.yaml"))
    summary["run_dir"] = str(run_dir)
    return summary


def evaluate_yolo_run(
    config_path: str,
    weights_path: str,
    split: str = "val",
    conf_threshold: float | None = None,
    compare_swin_checkpoint: str | None = "experiments/agent_swin/checkpoints/best_model.pth",
    output_dir: str = "experiments/yolo_v8/reports",
    dataset_output_dir: str | None = None,
    max_images_per_split: int | None = None,
    classifier_config_path: str = "configs/default_config.yaml",
) -> dict[str, Any]:
    config = load_config(config_path)
    YOLO = _load_ultralytics()
    threshold = conf_threshold if conf_threshold is not None else config["evaluation"].get("conf_threshold", 0.25)

    dataset_dir = prepare_detection_dataset_from_config(
        config=config,
        output_dir=dataset_output_dir,
        max_images_per_split=max_images_per_split,
    )
    dataset_yaml = dataset_dir / "dataset.yaml"
    model = YOLO(weights_path)
    detection_metrics = _extract_detection_metrics(
        model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=config["data"]["image_size"],
            batch=config["training"]["batch_size"],
            device=config.get("device", 0),
            conf=threshold,
            verbose=False,
        )
    )

    image_ids = build_split_image_ids(
        image_root=config["data"]["image_root"],
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
    )[split]
    if max_images_per_split is not None:
        image_ids = image_ids[:max_images_per_split]

    comparison_rows = []
    yolo_rows = []
    for image_id in image_ids:
        prediction = infer_yolo_image(
            config_path=config_path,
            weights_path=weights_path,
            image_path=str(Path(config["data"]["image_root"]) / f"{image_id}.png"),
            conf_threshold=threshold,
        )
        image_level = prediction["image_level_labels"]
        yolo_rows.append({"image_id": image_id, **image_level})
        comparison_rows.append(
            {
                "image_id": image_id,
                "detections": prediction["detections"],
                "derived_labels": [label for label, value in image_level.items() if value == 1],
            }
        )

    yolo_pred_df = pd.DataFrame(yolo_rows).set_index("image_id")
    y_true_df = load_image_level_ground_truth(config["data"]["classification_csv"], image_ids)
    image_level_metrics, confusion_rows, row_summaries = evaluate_multilabel_predictions(
        image_ids=image_ids,
        y_true_df=y_true_df,
        y_pred_df=yolo_pred_df,
        class_names=CLASS_NAMES,
    )

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "yolo_detection_metrics.json").write_text(json.dumps(detection_metrics, indent=2))
    (report_dir / "yolo_image_level_metrics.json").write_text(json.dumps(image_level_metrics, indent=2))
    (report_dir / "yolo_per_class_confusion_matrix.json").write_text(json.dumps(confusion_rows, indent=2))
    pd.DataFrame(comparison_rows).to_json(report_dir / "yolo_predictions.jsonl", orient="records", lines=True)

    summary = {
        "detection_metrics": detection_metrics,
        "image_level_metrics": image_level_metrics,
    }

    if compare_swin_checkpoint and Path(compare_swin_checkpoint).exists():
        classifier_config = load_config(classifier_config_path)
        classifier_config["model"]["name"] = "swin_base_patch4_window7_224"
        classifier_config["data"]["mode"] = "classification"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        swin_pred_df = predict_classifier_dataset(
            config=classifier_config,
            checkpoint_path=compare_swin_checkpoint,
            split=split,
            threshold=0.5,
            device_name=device,
            batch_size=32,
        )
        _, _, swin_rows = evaluate_multilabel_predictions(
            image_ids=image_ids,
            y_true_df=y_true_df,
            y_pred_df=swin_pred_df.loc[image_ids],
            class_names=CLASS_NAMES,
        )
        compare_df = pd.DataFrame(row_summaries).rename(
            columns={
                "predicted_labels": "yolo_predicted_labels",
                "label_errors": "yolo_label_errors",
            }
        )
        compare_df["swin_predicted_labels"] = [row["predicted_labels"] for row in swin_rows]
        compare_df["swin_label_errors"] = [row["label_errors"] for row in swin_rows]

        yolo_better = compare_df[compare_df["yolo_label_errors"] < compare_df["swin_label_errors"]]
        swin_better = compare_df[compare_df["swin_label_errors"] < compare_df["yolo_label_errors"]]
        both_wrong = compare_df[
            (compare_df["yolo_label_errors"] > 0) & (compare_df["swin_label_errors"] > 0)
        ]

        compare_df.to_csv(report_dir / "yolo_vs_swin_case_comparison.csv", index=False)
        report_lines = [
            "# YOLO vs Swin Case Review",
            "",
            f"- YOLO better on {len(yolo_better)} cases",
            f"- Swin better on {len(swin_better)} cases",
            f"- Both wrong on {len(both_wrong)} cases",
            "",
            "## YOLO Better",
            "```csv",
            format_case_table(yolo_better.head(10)),
            "```",
            "",
            "## Swin Better",
            "```csv",
            format_case_table(swin_better.head(10)),
            "```",
            "",
            "## Both Wrong",
            "```csv",
            format_case_table(both_wrong.head(10)),
            "```",
            "",
        ]
        (report_dir / "yolo_vs_swin_case_review.md").write_text("\n".join(report_lines))
        summary["comparison"] = {
            "yolo_better_cases": int(len(yolo_better)),
            "swin_better_cases": int(len(swin_better)),
            "both_wrong_cases": int(len(both_wrong)),
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
