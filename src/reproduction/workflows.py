"""
High-level no-retrain reproduction workflows.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from sklearn.metrics import roc_auc_score

from src.data.dataset import CLASS_NAMES
from src.data.detection import load_image_level_ground_truth
from src.detection import evaluate_yolo_run
from src.evaluation import evaluate_multilabel_predictions
from src.review import evaluate_review_run, run_review_case
from src.training import predict_classifier_outputs
from src.utils import load_config


PAPER_CLASSIFIERS = [
    {
        "key": "simple_cnn",
        "model_name": "simple_cnn",
        "checkpoint_path": "model_checkpoints/paper_simple_cnn_best.pth",
    },
    {
        "key": "efficientnet_b3",
        "model_name": "efficientnet_b3",
        "checkpoint_path": "model_checkpoints/paper_efficientnet_b3_best.pth",
    },
    {
        "key": "resnet50",
        "model_name": "resnet50",
        "checkpoint_path": "model_checkpoints/paper_resnet50_best.pth",
    },
    {
        "key": "vit_base",
        "model_name": "vit_base",
        "checkpoint_path": "model_checkpoints/vit_base_100_epoch_best.pth",
    },
    {
        "key": "swin_base",
        "model_name": "swin_base_patch4_window7_224",
        "checkpoint_path": "model_checkpoints/paper_swin_base_best.pth",
    },
]


def _safe_macro_auc_roc_from_probabilities(y_true_df: pd.DataFrame, y_prob_df: pd.DataFrame) -> float:
    try:
        return float(
            roc_auc_score(
                y_true_df[CLASS_NAMES].to_numpy(dtype=int),
                y_prob_df[CLASS_NAMES].to_numpy(dtype=float),
                average="macro",
            )
        )
    except ValueError:
        return 0.0


def _load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, float | int | str | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "best_metric_name": checkpoint.get("best_metric_name"),
        "best_val_score": float(checkpoint["best_val_score"]) if checkpoint.get("best_val_score") is not None else None,
        "best_epoch": int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None,
    }


def _evaluate_classifier_checkpoint(
    config_path: str,
    model_name: str,
    checkpoint_path: str | Path,
    split: str,
    threshold: float,
    device_name: str,
    batch_size: int | None,
) -> dict:
    config = load_config(config_path)
    config["model"]["name"] = model_name
    config["data"]["mode"] = "classification"

    probability_df, prediction_df = predict_classifier_outputs(
        config=config,
        checkpoint_path=checkpoint_path,
        split=split,
        threshold=threshold,
        device_name=device_name,
        batch_size=batch_size,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    image_ids = prediction_df.index.tolist()
    y_true_df = load_image_level_ground_truth(
        Path(config["data"]["data_dir"]) / config["data"].get("train_csv", "train.csv"),
        image_ids,
    )
    label_metrics, _, _ = evaluate_multilabel_predictions(
        image_ids=image_ids,
        y_true_df=y_true_df,
        y_pred_df=prediction_df.loc[image_ids],
        class_names=CLASS_NAMES,
    )
    metadata = _load_checkpoint_metadata(checkpoint_path)

    return {
        "model_name": model_name,
        "checkpoint_path": str(Path(checkpoint_path)),
        "split": split,
        "samples": int(len(image_ids)),
        "threshold": float(threshold),
        "checkpoint_best_metric_name": metadata["best_metric_name"],
        "checkpoint_best_metric_value": metadata["best_val_score"],
        "checkpoint_best_epoch": metadata["best_epoch"],
        "exact_match_accuracy": label_metrics["exact_match_accuracy"],
        "macro_accuracy": label_metrics["macro_accuracy"],
        "macro_auc_roc": _safe_macro_auc_roc_from_probabilities(y_true_df, probability_df.loc[image_ids]),
        "macro_f1": label_metrics["macro_f1"],
    }


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in {".cache", ".git", ".gitattributes"}:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _download_snapshot(repo_id: str, repo_type: str, target_dir: Path, token: str | None, cache_root: Path) -> None:
    local_dir = cache_root / repo_type / repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_type} repo '{repo_id}' -> cache {local_dir}")
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(local_dir), token=token)
    print(f"Syncing {repo_id} -> {target_dir}")
    _copy_tree_contents(local_dir, target_dir)


def download_assets(
    repo_root: str | Path,
    dataset_repo: str | None = None,
    dataset_repo_type: str = "dataset",
    dataset_target: str = "data",
    artifacts_repo: str | None = None,
    artifacts_repo_type: str = "model",
    artifacts_target: str = ".",
    hf_token: str | None = None,
) -> None:
    root = Path(repo_root)
    cache_root = root / ".hf_downloads"
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if dataset_repo:
        _download_snapshot(dataset_repo, dataset_repo_type, root / dataset_target, token, cache_root)
    if artifacts_repo:
        _download_snapshot(artifacts_repo, artifacts_repo_type, root / artifacts_target, token, cache_root)


def compare_runs(
    split: str = "val",
    max_cases: int | None = 300,
    yolo_weights: str = "experiments/yolo_v8/full_e10local/best.pt",
    yolo_output_dir: str | None = None,
    review_output_dir: str | None = None,
    skip_yolo: bool = False,
    skip_review: bool = False,
    force_refresh: bool = False,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if not skip_yolo:
        results["yolo"] = evaluate_yolo_run(
            config_path="configs/yolo_v8_detection.yaml",
            weights_path=yolo_weights,
            split=split,
            output_dir=yolo_output_dir or f"experiments/repro_outputs/yolo_eval_{split}",
        )
    if not skip_review:
        results["review"] = evaluate_review_run(
            review_config_path="configs/claude_review.yaml",
            split=split,
            max_cases=max_cases,
            output_dir=review_output_dir or f"experiments/repro_outputs/review_eval_{max_cases or 'full'}",
            force_refresh=force_refresh,
        )
    return results


def reproduce_paper_metrics(
    output_dir: str = "experiments/paper_metrics",
    split: str = "val",
    classifier_config_path: str = "configs/default_config.yaml",
    yolo_config_path: str = "configs/yolo_v8_detection.yaml",
    classifier_keys: list[str] | None = None,
    classifier_threshold: float = 0.5,
    classifier_device: str = "auto",
    classifier_batch_size: int | None = None,
    skip_yolo: bool = False,
    yolo_weights_path: str = "model_checkpoints/paper_yolo_best.pt",
) -> dict:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = set(classifier_keys or [item["key"] for item in PAPER_CLASSIFIERS])
    classifier_rows = []
    for spec in PAPER_CLASSIFIERS:
        if spec["key"] not in selected_keys:
            continue
        classifier_rows.append(
            {
                "key": spec["key"],
                **_evaluate_classifier_checkpoint(
                    config_path=classifier_config_path,
                    model_name=spec["model_name"],
                    checkpoint_path=spec["checkpoint_path"],
                    split=split,
                    threshold=classifier_threshold,
                    device_name=classifier_device,
                    batch_size=classifier_batch_size,
                ),
            }
        )

    classifier_df = pd.DataFrame(classifier_rows)
    classifier_csv_path = report_dir / "classifier_metrics.csv"
    classifier_json_path = report_dir / "classifier_metrics.json"
    classifier_df.to_csv(classifier_csv_path, index=False)
    classifier_json_path.write_text(classifier_df.to_json(orient="records", indent=2))

    summary: dict[str, object] = {
        "split": split,
        "classifiers": classifier_rows,
    }

    if not skip_yolo:
        yolo_dir = report_dir / "yolo"
        try:
            yolo_summary = evaluate_yolo_run(
                config_path=yolo_config_path,
                weights_path=yolo_weights_path,
                split=split,
                compare_swin_checkpoint=None,
                output_dir=str(yolo_dir),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "YOLO paper-metric reproduction requires ultralytics. "
                "Install the project requirements or rerun with --skip-yolo "
                "to reproduce the classifier metrics only."
            ) from exc
        yolo_metrics = {
            "weights_path": yolo_weights_path,
            "map50": yolo_summary["detection_metrics"]["map50"],
        }
        (report_dir / "yolo_metrics.json").write_text(json.dumps(yolo_metrics, indent=2))
        summary["yolo"] = yolo_metrics

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
