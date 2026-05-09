#!/usr/bin/env python3
"""
Unified CLI for the retained CT-Transformer workflows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import prepare_vinbigdata_dataset, prepare_yolo_dataset
from src.detection import evaluate_yolo_run, infer_yolo_image, train_yolo_run
from src.reproduction import compare_runs, download_assets, reproduce_paper_metrics
from src.review import evaluate_review_run, run_review_case
from src.training import train_classifier_from_args
from src.utils import load_config, load_local_env

load_local_env(REPO_ROOT / ".env")


def _default_yolo_weights() -> str:
    review_cfg = REPO_ROOT / "configs" / "claude_review.yaml"
    text = review_cfg.read_text()
    for line in text.splitlines():
        if line.strip().startswith("yolo_weights:"):
            return line.split(":", 1)[1].strip().strip('"')
    return "experiments/yolo_v8/full_e10local/best.pt"


def _set_api_key(api_key: str | None) -> None:
    if api_key:
        os.environ["CMU_LLM_GATEWAY_API_KEY"] = api_key


def cmd_setup_data(args: argparse.Namespace) -> None:
    payload = prepare_vinbigdata_dataset(
        data_dir=args.data_dir,
        image_dataset=args.image_dataset,
        competition=args.competition,
        project_root=REPO_ROOT,
    )
    print(json.dumps(payload, indent=2))


def cmd_prepare_yolo(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = args.output_dir or config["data"]["output_dir"]
    metadata = prepare_yolo_dataset(
        image_root=config["data"]["image_root"],
        raw_annotation_path=config["data"]["raw_annotation_path"],
        image_metadata_path=config["data"]["image_metadata_path"],
        output_dir=output_dir,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.2),
        seed=config.get("seed", 42),
        merge_iou_threshold=config["data"].get("merge_iou_threshold", 0.3),
        link_mode=args.link_mode or config["data"].get("link_mode", "symlink"),
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps(metadata, indent=2))


def cmd_train_classifier(args: argparse.Namespace) -> None:
    train_classifier_from_args(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )


def cmd_train_yolo(args: argparse.Namespace) -> None:
    summary = train_yolo_run(
        config_path=args.config,
        epochs=args.epochs,
        weights=args.weights,
        batch_size=args.batch_size,
        project_dir=args.project_dir,
        run_name=args.run_name,
        dataset_output_dir=args.dataset_output_dir,
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps(summary, indent=2))


def cmd_eval_yolo(args: argparse.Namespace) -> None:
    summary = evaluate_yolo_run(
        config_path=args.config,
        weights_path=args.weights,
        split=args.split,
        conf_threshold=args.conf_threshold,
        compare_swin_checkpoint=args.compare_swin_checkpoint,
        output_dir=args.output_dir,
        dataset_output_dir=args.dataset_output_dir,
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps(summary, indent=2))


def cmd_infer_yolo(args: argparse.Namespace) -> None:
    payload = infer_yolo_image(
        config_path=args.config,
        weights_path=args.weights,
        image_path=args.image,
        conf_threshold=args.conf_threshold,
    )
    print(json.dumps(payload, indent=2))


def cmd_eval_review(args: argparse.Namespace) -> None:
    _set_api_key(args.api_key)
    summary = evaluate_review_run(
        review_config_path=args.review_config,
        split=args.split,
        max_cases=args.max_cases,
        output_dir=args.output_dir,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    _set_api_key(args.api_key)
    summary = run_review_case(
        image_path=args.image,
        review_config_path=args.review_config,
        output_dir=args.output_dir,
        image_id=args.image_id,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(summary, indent=2))


def cmd_download(args: argparse.Namespace) -> None:
    download_assets(
        repo_root=REPO_ROOT,
        dataset_repo=args.dataset_repo,
        dataset_repo_type=args.dataset_repo_type,
        dataset_target=args.dataset_target,
        artifacts_repo=args.artifacts_repo,
        artifacts_repo_type=args.artifacts_repo_type,
        artifacts_target=args.artifacts_target,
        hf_token=args.hf_token,
    )


def cmd_compare(args: argparse.Namespace) -> None:
    _set_api_key(args.api_key)
    payload = compare_runs(
        split=args.split,
        max_cases=args.max_cases,
        yolo_weights=args.yolo_weights,
        yolo_output_dir=args.yolo_output_dir,
        review_output_dir=args.review_output_dir,
        skip_yolo=args.skip_yolo,
        skip_review=args.skip_review,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(payload, indent=2))


def cmd_reproduce_paper_metrics(args: argparse.Namespace) -> None:
    payload = reproduce_paper_metrics(
        output_dir=args.output_dir,
        split=args.split,
        classifier_config_path=args.classifier_config,
        yolo_config_path=args.yolo_config,
        classifier_keys=args.classifier,
        classifier_threshold=args.classifier_threshold,
        classifier_device=args.classifier_device,
        classifier_batch_size=args.classifier_batch_size,
        skip_yolo=args.skip_yolo,
        yolo_weights_path=args.yolo_weights,
    )
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CT-Transformer CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup-data", help="Download and prepare the VinBigData dataset.")
    setup_parser.add_argument("--data-dir", default="data")
    setup_parser.add_argument("--image-dataset", default="xhlulu/vinbigdata")
    setup_parser.add_argument("--competition", default="vinbigdata-chest-xray-abnormalities-detection")
    setup_parser.set_defaults(func=cmd_setup_data)

    prep_yolo_parser = subparsers.add_parser("prepare-yolo", help="Prepare YOLO training data.")
    prep_yolo_parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    prep_yolo_parser.add_argument("--output-dir", default=None)
    prep_yolo_parser.add_argument("--max-images-per-split", type=int, default=None)
    prep_yolo_parser.add_argument("--link-mode", choices=["symlink", "copy"], default=None)
    prep_yolo_parser.set_defaults(func=cmd_prepare_yolo)

    train_parser = subparsers.add_parser("train-classifier", help="Train a classification model.")
    train_parser.add_argument("--config", default="configs/default_config.yaml")
    train_parser.add_argument("--model", default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--save-dir", default=None)
    train_parser.add_argument("--log-dir", default=None)
    train_parser.set_defaults(func=cmd_train_classifier)

    train_yolo_parser = subparsers.add_parser("train-yolo", help="Train YOLO.")
    train_yolo_parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    train_yolo_parser.add_argument("--epochs", type=int, default=None)
    train_yolo_parser.add_argument("--weights", default=None)
    train_yolo_parser.add_argument("--batch-size", type=int, default=None)
    train_yolo_parser.add_argument("--project-dir", default=None)
    train_yolo_parser.add_argument("--run-name", default=None)
    train_yolo_parser.add_argument("--dataset-output-dir", default=None)
    train_yolo_parser.add_argument("--max-images-per-split", type=int, default=None)
    train_yolo_parser.set_defaults(func=cmd_train_yolo)

    eval_yolo_parser = subparsers.add_parser("eval-yolo", help="Evaluate YOLO and compare with Swin.")
    eval_yolo_parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    eval_yolo_parser.add_argument("--weights", required=True)
    eval_yolo_parser.add_argument("--split", default="val", choices=["train", "val"])
    eval_yolo_parser.add_argument("--conf-threshold", type=float, default=None)
    eval_yolo_parser.add_argument("--compare-swin-checkpoint", default="experiments/agent_swin/checkpoints/best_model.pth")
    eval_yolo_parser.add_argument("--output-dir", default="experiments/yolo_v8/reports")
    eval_yolo_parser.add_argument("--dataset-output-dir", default=None)
    eval_yolo_parser.add_argument("--max-images-per-split", type=int, default=None)
    eval_yolo_parser.set_defaults(func=cmd_eval_yolo)

    infer_yolo_parser = subparsers.add_parser("infer-yolo", help="Run YOLO on one image.")
    infer_yolo_parser.add_argument("--config", default="configs/yolo_v8_detection.yaml")
    infer_yolo_parser.add_argument("--weights", required=True)
    infer_yolo_parser.add_argument("--image", required=True)
    infer_yolo_parser.add_argument("--conf-threshold", type=float, default=None)
    infer_yolo_parser.set_defaults(func=cmd_infer_yolo)

    eval_review_parser = subparsers.add_parser("eval-review", help="Evaluate the review workflow.")
    eval_review_parser.add_argument("--review-config", default="configs/claude_review.yaml")
    eval_review_parser.add_argument("--split", default="val", choices=["train", "val"])
    eval_review_parser.add_argument("--max-cases", type=int, default=None)
    eval_review_parser.add_argument("--output-dir", default=None)
    eval_review_parser.add_argument("--api-key", default=None)
    eval_review_parser.add_argument("--force-refresh", action="store_true")
    eval_review_parser.set_defaults(func=cmd_eval_review)

    report_parser = subparsers.add_parser("report", help="Generate one agentic report.")
    report_parser.add_argument("--image", required=True)
    report_parser.add_argument("--image-id", default=None)
    report_parser.add_argument("--review-config", default="configs/claude_review.yaml")
    report_parser.add_argument("--output-dir", default="experiments/agentic_reports")
    report_parser.add_argument("--api-key", default=None)
    report_parser.add_argument("--force-refresh", action="store_true")
    report_parser.set_defaults(func=cmd_report)

    download_parser = subparsers.add_parser("download", help="Download prepared data and artifacts from Hugging Face.")
    download_parser.add_argument("--dataset-repo", default=None)
    download_parser.add_argument("--dataset-repo-type", default="dataset", choices=["dataset", "model"])
    download_parser.add_argument("--dataset-target", default="data")
    download_parser.add_argument("--artifacts-repo", default=None)
    download_parser.add_argument("--artifacts-repo-type", default="model", choices=["dataset", "model"])
    download_parser.add_argument("--artifacts-target", default=".")
    download_parser.add_argument("--hf-token", default=None)
    download_parser.set_defaults(func=cmd_download)

    compare_parser = subparsers.add_parser("compare", help="Rerun saved comparisons without retraining.")
    compare_parser.add_argument("--split", default="val", choices=["train", "val"])
    compare_parser.add_argument("--max-cases", type=int, default=300)
    compare_parser.add_argument("--api-key", default=None)
    compare_parser.add_argument("--yolo-weights", default=_default_yolo_weights())
    compare_parser.add_argument("--yolo-output-dir", default=None)
    compare_parser.add_argument("--review-output-dir", default=None)
    compare_parser.add_argument("--skip-yolo", action="store_true")
    compare_parser.add_argument("--skip-review", action="store_true")
    compare_parser.add_argument("--force-refresh", action="store_true")
    compare_parser.set_defaults(func=cmd_compare)

    paper_parser = subparsers.add_parser("reproduce-paper-metrics", help="Reproduce classifier AUC-ROC and YOLO mAP@0.5 from saved checkpoints.")
    paper_parser.add_argument("--output-dir", default="experiments/paper_metrics")
    paper_parser.add_argument("--split", default="val", choices=["train", "val"])
    paper_parser.add_argument("--classifier-config", default="configs/default_config.yaml")
    paper_parser.add_argument("--yolo-config", default="configs/yolo_v8_detection.yaml")
    paper_parser.add_argument(
        "--classifier",
        action="append",
        choices=["simple_cnn", "efficientnet_b3", "resnet50", "vit_base", "swin_base"],
        help="Limit reproduction to one or more classifiers.",
    )
    paper_parser.add_argument("--classifier-threshold", type=float, default=0.5)
    paper_parser.add_argument("--classifier-device", default="auto")
    paper_parser.add_argument("--classifier-batch-size", type=int, default=None)
    paper_parser.add_argument("--skip-yolo", action="store_true")
    paper_parser.add_argument("--yolo-weights", default="model_checkpoints/paper_yolo_best.pt")
    paper_parser.set_defaults(func=cmd_reproduce_paper_metrics)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
