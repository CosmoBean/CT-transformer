#!/usr/bin/env python3
"""
Simple reproduction entrypoint for the final CT-Transformer workflow.

This script keeps the underlying training/evaluation/report scripts intact, but
provides a much simpler interface for:
- downloading prepared data from Hugging Face
- downloading checkpoints / cached artifacts from Hugging Face
- rerunning metric comparisons without retraining
- generating a single agentic report from an API key
- generating presentation comparison reports
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_DOWNLOAD_ROOT = REPO_ROOT / ".hf_downloads"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_local_env

load_local_env(REPO_ROOT / ".env")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


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


def _download_snapshot(repo_id: str, repo_type: str, target_dir: Path, token: str | None) -> None:
    local_dir = HF_DOWNLOAD_ROOT / repo_type / repo_id.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_type} repo '{repo_id}' -> cache {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        token=token,
    )
    print(f"Syncing {repo_id} -> {target_dir}")
    _copy_tree_contents(local_dir, target_dir)


def _env_with_api_key(api_key: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if api_key:
        env["CMU_LLM_GATEWAY_API_KEY"] = api_key
    return env


def _default_yolo_weights() -> str:
    review_cfg = REPO_ROOT / "configs" / "claude_review.yaml"
    text = review_cfg.read_text()
    for line in text.splitlines():
        if line.strip().startswith("yolo_weights:"):
            return line.split(":", 1)[1].strip().strip('"')
    return "experiments/yolo_v8/full_e10local/best.pt"


def cmd_download(args: argparse.Namespace) -> None:
    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if args.dataset_repo:
        _download_snapshot(
            repo_id=args.dataset_repo,
            repo_type=args.dataset_repo_type,
            target_dir=REPO_ROOT / args.dataset_target,
            token=token,
        )
    if args.artifacts_repo:
        _download_snapshot(
            repo_id=args.artifacts_repo,
            repo_type=args.artifacts_repo_type,
            target_dir=REPO_ROOT / args.artifacts_target,
            token=token,
        )
    print("Download complete.")


def cmd_compare(args: argparse.Namespace) -> None:
    env = _env_with_api_key(args.api_key)
    yolo_output = args.yolo_output_dir or f"experiments/repro_outputs/yolo_eval_{args.split}"
    review_output = args.review_output_dir or f"experiments/repro_outputs/review_eval_{args.max_cases or 'full'}"

    if not args.skip_yolo:
        _run(
            [
                sys.executable,
                "scripts/evaluate_yolo.py",
                "--weights",
                args.yolo_weights,
                "--split",
                args.split,
                "--output-dir",
                yolo_output,
            ],
            env=env,
        )

    if not args.skip_review:
        cmd = [
            sys.executable,
            "scripts/evaluate_claude_review.py",
            "--split",
            args.split,
            "--output-dir",
            review_output,
        ]
        if args.max_cases is not None:
            cmd.extend(["--max-cases", str(args.max_cases)])
        if args.force_refresh:
            cmd.append("--force-refresh")
        _run(cmd, env=env)


def cmd_report(args: argparse.Namespace) -> None:
    env = _env_with_api_key(args.api_key)
    cmd = [
        sys.executable,
        "scripts/run_agentic_report.py",
        "--image",
        args.image,
        "--output-dir",
        args.output_dir,
    ]
    if args.image_id:
        cmd.extend(["--image-id", args.image_id])
    if args.force_refresh:
        cmd.append("--force-refresh")
    _run(cmd, env=env)


def cmd_presentation(args: argparse.Namespace) -> None:
    env = _env_with_api_key(args.api_key)
    cmd = [
        sys.executable,
        "scripts/generate_presentation_comparison_reports.py",
        "--comparison-csv",
        args.comparison_csv,
        "--cache-dir",
        args.cache_dir,
        "--output-dir",
        args.output_dir,
    ]
    _run(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple reproduction workflow for CT-Transformer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download prepared dataset and/or checkpoints from Hugging Face.",
    )
    download_parser.add_argument("--dataset-repo", default=None, help="HF dataset repo containing prepared data/")
    download_parser.add_argument("--dataset-repo-type", default="dataset", choices=["dataset", "model"])
    download_parser.add_argument("--dataset-target", default="data", help="Local target directory for dataset files")
    download_parser.add_argument(
        "--artifacts-repo",
        default=None,
        help="HF repo containing experiments/, checkpoints, cached review results, etc.",
    )
    download_parser.add_argument("--artifacts-repo-type", default="model", choices=["dataset", "model"])
    download_parser.add_argument(
        "--artifacts-target",
        default=".",
        help="Local target directory for artifacts. Use '.' if the repo snapshot already contains experiments/ paths.",
    )
    download_parser.add_argument("--hf-token", default=None, help="Optional HF token for private repos.")
    download_parser.set_defaults(func=cmd_download)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Rerun stored evaluation pipelines without retraining.",
    )
    compare_parser.add_argument("--split", default="val", choices=["train", "val"])
    compare_parser.add_argument("--max-cases", type=int, default=300)
    compare_parser.add_argument("--api-key", default=None, help="API key for fresh agentic review calls if cache is missing.")
    compare_parser.add_argument("--yolo-weights", default=_default_yolo_weights())
    compare_parser.add_argument("--yolo-output-dir", default=None)
    compare_parser.add_argument("--review-output-dir", default=None)
    compare_parser.add_argument("--skip-yolo", action="store_true")
    compare_parser.add_argument("--skip-review", action="store_true")
    compare_parser.add_argument("--force-refresh", action="store_true")
    compare_parser.set_defaults(func=cmd_compare)

    report_parser = subparsers.add_parser(
        "report",
        help="Generate one agentic report for a single image.",
    )
    report_parser.add_argument("--image", required=True)
    report_parser.add_argument("--image-id", default=None)
    report_parser.add_argument("--api-key", required=True, help="Gateway API key for the report call.")
    report_parser.add_argument("--output-dir", default="experiments/agentic_reports")
    report_parser.add_argument("--force-refresh", action="store_true")
    report_parser.set_defaults(func=cmd_report)

    presentation_parser = subparsers.add_parser(
        "presentation",
        help="Generate presentation comparison PDFs from an evaluation CSV/cache bundle.",
    )
    presentation_parser.add_argument(
        "--comparison-csv",
        default="experiments/claude_review/eval_300/claude_vs_baselines_case_comparison.csv",
    )
    presentation_parser.add_argument("--cache-dir", default="experiments/claude_review/cache")
    presentation_parser.add_argument("--output-dir", default="reports/presentation_comparison_examples")
    presentation_parser.add_argument("--api-key", default=None)
    presentation_parser.set_defaults(func=cmd_presentation)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
