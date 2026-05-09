"""
High-level no-retrain reproduction workflows.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from src.detection import evaluate_yolo_run
from src.review import evaluate_review_run, run_review_case


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
