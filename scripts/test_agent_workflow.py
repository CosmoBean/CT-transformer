#!/usr/bin/env python3
"""
Deterministic tests for the agentic triage policy.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.agent import AgentAction, AgentTriagePolicy, CaseSummary
from src.data.dataset import CLASS_NAMES, ChestXRayDataset


def build_summary(
    image_id: str,
    *,
    no_finding: float,
    max_abnormal: float,
    margin: float,
    moderate_count: int,
    heuristic_prediction: str,
    valid_image: bool = True,
    error: str | None = None,
) -> CaseSummary:
    return CaseSummary(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.png",
        class_probabilities={},
        top_findings=["Pleural effusion"],
        top_scores={"Pleural effusion": max_abnormal, "No finding": no_finding},
        no_finding_probability=no_finding,
        max_abnormal_probability=max_abnormal,
        abnormal_probability_margin=margin,
        moderate_findings_count=moderate_count,
        heuristic_binary_prediction=heuristic_prediction,
        valid_image=valid_image,
        error=error,
    )


def test_dataset_label_reordering() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        train_dir = root / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(train_dir / "sample.png")

        shuffled_columns = ["image_id", "No finding", "Cardiomegaly", "Atelectasis"]
        remaining = [name for name in CLASS_NAMES if name not in {"No finding", "Cardiomegaly", "Atelectasis"}]
        shuffled_columns.extend(remaining)
        row = {column: 0 for column in shuffled_columns}
        row["image_id"] = "sample"
        row["No finding"] = 1
        row["Cardiomegaly"] = 1

        csv_path = root / "labels.csv"
        pd.DataFrame([row], columns=shuffled_columns).to_csv(csv_path, index=False)

        dataset = ChestXRayDataset(
            data_dir=str(root),
            csv_path=str(csv_path),
            image_size=8,
            split="train",
            train_split=1.0,
        )
        labels = dataset.samples[0].labels
        no_finding_index = CLASS_NAMES.index("No finding")
        cardiomegaly_index = CLASS_NAMES.index("Cardiomegaly")
        ok = labels[no_finding_index] == 1.0 and labels[cardiomegaly_index] == 1.0
        return ok, f"no_finding={labels[no_finding_index]} cardiomegaly={labels[cardiomegaly_index]}"


def main() -> int:
    policy = AgentTriagePolicy()

    cases = [
        (
            "clear_normal",
            build_summary(
                "clear_normal",
                no_finding=0.91,
                max_abnormal=0.08,
                margin=0.01,
                moderate_count=0,
                heuristic_prediction="normal",
            ),
            AgentAction.ACCEPT_NORMAL,
        ),
        (
            "clear_abnormal",
            build_summary(
                "clear_abnormal",
                no_finding=0.10,
                max_abnormal=0.87,
                margin=0.24,
                moderate_count=1,
                heuristic_prediction="abnormal",
            ),
            AgentAction.ACCEPT_ABNORMAL,
        ),
        (
            "borderline",
            build_summary(
                "borderline",
                no_finding=0.46,
                max_abnormal=0.51,
                margin=0.03,
                moderate_count=2,
                heuristic_prediction="abnormal",
            ),
            AgentAction.FLAG_FOR_REVIEW,
        ),
        (
            "multi_finding",
            build_summary(
                "multi_finding",
                no_finding=0.21,
                max_abnormal=0.62,
                margin=0.09,
                moderate_count=4,
                heuristic_prediction="abnormal",
            ),
            AgentAction.FLAG_FOR_REVIEW,
        ),
        (
            "invalid_image",
            build_summary(
                "invalid_image",
                no_finding=0.0,
                max_abnormal=0.0,
                margin=0.0,
                moderate_count=0,
                heuristic_prediction="unknown",
                valid_image=False,
                error="decode failed",
            ),
            AgentAction.UNABLE_TO_ASSESS,
        ),
    ]

    passed = 0
    dataset_ok, dataset_message = test_dataset_label_reordering()
    dataset_status = "PASSED" if dataset_ok else "FAILED"
    print(f"{'dataset_order':<15} {dataset_status:<6} {dataset_message}")
    if dataset_ok:
        passed += 1

    for name, summary, expected in cases:
        decision = policy.decide(summary)
        ok = decision.action == expected
        status = "PASSED" if ok else "FAILED"
        print(f"{name:<15} {status:<6} expected={expected.value} actual={decision.action.value}")
        if ok:
            passed += 1

    total = len(cases) + 1
    print(f"\nTotal: {passed}/{total} agent policy tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
