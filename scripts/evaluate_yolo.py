#!/usr/bin/env python3
"""
Evaluate YOLO detection metrics and compare derived image-level predictions to Swin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.train import create_model
from src.data.detection import (
    DETECTION_CLASS_NAMES,
    build_split_image_ids,
    derive_image_level_labels_from_detections,
    load_image_level_ground_truth,
    prepare_yolo_dataset,
)
from src.data.dataset import CLASS_NAMES, ChestXRayDataset
from src.utils import load_config


def _load_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO evaluation. Install the project dependencies first."
        ) from exc
    return YOLO


def _extract_detection_metrics(metrics) -> dict:
    box_metrics = getattr(metrics, "box", None)
    return {
        "map50": float(getattr(box_metrics, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box_metrics, "map", 0.0) or 0.0),
        "maps": [float(value) for value in getattr(box_metrics, "maps", [])] if box_metrics is not None else [],
    }


def _evaluate_image_level_predictions(
    image_ids: list[str],
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
) -> tuple[dict, list[dict], list[dict]]:
    y_true = y_true_df[CLASS_NAMES].to_numpy(dtype=int)
    y_pred = y_pred_df[CLASS_NAMES].to_numpy(dtype=int)
    per_class_accuracy = (y_true == y_pred).mean(axis=0)
    payload = {
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
        tn, fp, fn, tp = confusion_matrix(
            y_true[:, idx],
            y_pred[:, idx],
            labels=[0, 1],
        ).ravel()
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
        errors = int(np.not_equal(true_row, pred_row).sum())
        row_summaries.append(
            {
                "image_id": image_id,
                "true_labels": [CLASS_NAMES[i] for i, value in enumerate(true_row) if value == 1],
                "predicted_labels": [CLASS_NAMES[i] for i, value in enumerate(pred_row) if value == 1],
                "label_errors": errors,
            }
        )
    return payload, confusion_rows, row_summaries


def _format_case_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    return df.to_csv(index=False).strip()


def _load_swin_predictions(
    config: dict,
    checkpoint_path: Path,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
    dataset = ChestXRayDataset(
        data_dir=config["data"]["data_dir"],
        csv_path=str(Path(config["data"]["data_dir"]) / config["data"].get("train_csv", "train.csv")),
        image_size=config["data"]["image_size"],
        split=split,
        mode="classification",
        use_augmentation=False,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device))).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            for image_id, pred in zip(batch["image_id"], preds):
                rows.append(
                    {
                        "image_id": image_id,
                        **{class_name: int(value) for class_name, value in zip(CLASS_NAMES, pred)},
                    }
                )
    return pd.DataFrame(rows).set_index("image_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection and image-level metrics.")
    parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--conf-threshold", type=float, default=None)
    parser.add_argument("--compare-swin-checkpoint", default="experiments/agent_swin/checkpoints/best_model.pth")
    parser.add_argument("--output-dir", default="experiments/yolo_v8/reports")
    parser.add_argument("--dataset-output-dir", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    YOLO = _load_ultralytics()
    conf_threshold = args.conf_threshold
    if conf_threshold is None:
        conf_threshold = config["evaluation"].get("conf_threshold", 0.25)

    dataset_output_dir = args.dataset_output_dir or config["data"]["output_dir"]
    prepare_yolo_dataset(
        image_root=config["data"]["image_root"],
        raw_annotation_path=config["data"]["raw_annotation_path"],
        image_metadata_path=config["data"]["image_metadata_path"],
        output_dir=dataset_output_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
        merge_iou_threshold=config["data"].get("merge_iou_threshold", 0.3),
        link_mode=config["data"].get("link_mode", "symlink"),
        max_images_per_split=args.max_images_per_split,
    )

    dataset_yaml = Path(dataset_output_dir) / "dataset.yaml"
    model = YOLO(args.weights)
    detection_metrics = _extract_detection_metrics(
        model.val(
            data=str(dataset_yaml),
            split=args.split,
            imgsz=config["data"]["image_size"],
            batch=config["training"]["batch_size"],
            device=config.get("device", 0),
            conf=conf_threshold,
            verbose=False,
        )
    )

    image_ids = build_split_image_ids(
        image_root=config["data"]["image_root"],
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
    )[args.split]
    if args.max_images_per_split is not None:
        image_ids = image_ids[: args.max_images_per_split]
    image_paths = [str(Path(config["data"]["image_root"]) / f"{image_id}.png") for image_id in image_ids]

    yolo_rows = []
    comparison_rows = []
    for image_id, image_path in zip(image_ids, image_paths):
        result = model.predict(
            source=image_path,
            imgsz=config["data"]["image_size"],
            conf=conf_threshold,
            batch=1,
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
                if class_id < 0 or class_id >= len(DETECTION_CLASS_NAMES):
                    continue
                detections.append(
                    {
                        "class_name": names[class_id],
                        "confidence": float(confidence),
                        "bbox_xyxy": [float(value) for value in box],
                    }
                )

        image_level = derive_image_level_labels_from_detections(
            detections,
            confidence_threshold=conf_threshold,
        )
        yolo_rows.append({"image_id": image_id, **image_level})
        comparison_rows.append(
            {
                "image_id": image_id,
                "detections": detections,
                "derived_labels": [label for label, value in image_level.items() if value == 1],
            }
        )

    yolo_pred_df = pd.DataFrame(yolo_rows).set_index("image_id")
    y_true_df = load_image_level_ground_truth(config["data"]["classification_csv"], image_ids)
    image_level_metrics, confusion_rows, row_summaries = _evaluate_image_level_predictions(
        image_ids=image_ids,
        y_true_df=y_true_df,
        y_pred_df=yolo_pred_df,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "yolo_detection_metrics.json").write_text(json.dumps(detection_metrics, indent=2))
    (output_dir / "yolo_image_level_metrics.json").write_text(json.dumps(image_level_metrics, indent=2))
    (output_dir / "yolo_per_class_confusion_matrix.json").write_text(json.dumps(confusion_rows, indent=2))
    pd.DataFrame(comparison_rows).to_json(output_dir / "yolo_predictions.jsonl", orient="records", lines=True)

    summary = {
        "detection_metrics": detection_metrics,
        "image_level_metrics": image_level_metrics,
    }

    swin_checkpoint = Path(args.compare_swin_checkpoint)
    if swin_checkpoint.exists():
        classification_config = load_config("configs/default_config.yaml")
        classification_config["model"]["name"] = "swin_base_patch4_window7_224"
        classification_config["data"]["mode"] = "classification"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        swin_pred_df = _load_swin_predictions(
            config=classification_config,
            checkpoint_path=swin_checkpoint,
            split=args.split,
            device=device,
        )
        _, _, swin_rows = _evaluate_image_level_predictions(
            image_ids=image_ids,
            y_true_df=y_true_df,
            y_pred_df=swin_pred_df.loc[image_ids],
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

        compare_df.to_csv(output_dir / "yolo_vs_swin_case_comparison.csv", index=False)
        report_lines = [
            "# YOLO vs Swin Case Review",
            "",
            f"- YOLO better on {len(yolo_better)} cases",
            f"- Swin better on {len(swin_better)} cases",
            f"- Both wrong on {len(both_wrong)} cases",
            "",
            "## YOLO Better",
            "```csv",
            _format_case_table(yolo_better.head(10)),
            "```",
            "",
            "## Swin Better",
            "```csv",
            _format_case_table(swin_better.head(10)),
            "```",
            "",
            "## Both Wrong",
            "```csv",
            _format_case_table(both_wrong.head(10)),
            "```",
            "",
        ]
        (output_dir / "yolo_vs_swin_case_review.md").write_text("\n".join(report_lines))
        summary["comparison"] = {
            "yolo_better_cases": int(len(yolo_better)),
            "swin_better_cases": int(len(swin_better)),
            "both_wrong_cases": int(len(both_wrong)),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
