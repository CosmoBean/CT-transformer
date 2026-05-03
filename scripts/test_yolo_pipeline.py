#!/usr/bin/env python3
"""
Non-Ultralytics checks for the YOLO data preparation and image-level bridge.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES, ChestXRayDataset
from src.data.detection import (
    DETECTION_CLASS_NAMES,
    build_split_image_ids,
    derive_image_level_labels_from_detections,
    load_image_level_ground_truth,
    load_image_metadata,
    load_raw_detection_annotations,
    prepare_yolo_dataset,
)


DATA_DIR = PROJECT_ROOT / "data"


def test_raw_detection_inputs() -> None:
    raw_df = load_raw_detection_annotations(DATA_DIR / "_downloads" / "train_raw.csv")
    meta_df = load_image_metadata(DATA_DIR / "_downloads" / "vinbig_png" / "train_meta.csv")
    assert "No finding" in raw_df["class_name"].unique()
    assert raw_df["x_max"].max() > 512
    assert len(meta_df) == 15000


def test_split_consistency() -> None:
    split_ids = build_split_image_ids(
        image_root=DATA_DIR / "train",
        train_split=0.8,
        val_split=0.2,
        seed=28,
    )
    train_dataset = ChestXRayDataset(
        data_dir=str(DATA_DIR),
        csv_path=str(DATA_DIR / "train.csv"),
        image_size=512,
        split="train",
        mode="classification",
        use_augmentation=False,
        train_split=0.8,
        val_split=0.2,
        seed=28,
    )
    val_dataset = ChestXRayDataset(
        data_dir=str(DATA_DIR),
        csv_path=str(DATA_DIR / "train.csv"),
        image_size=512,
        split="val",
        mode="classification",
        use_augmentation=False,
        train_split=0.8,
        val_split=0.2,
        seed=28,
    )
    assert split_ids["train"] == [sample.image_id for sample in train_dataset.samples]
    assert split_ids["val"] == [sample.image_id for sample in val_dataset.samples]


def test_prepare_subset_dataset() -> None:
    with tempfile.TemporaryDirectory(prefix="ct_yolo_test_") as tmp_dir:
        metadata = prepare_yolo_dataset(
            image_root=DATA_DIR / "train",
            raw_annotation_path=DATA_DIR / "_downloads" / "train_raw.csv",
            image_metadata_path=DATA_DIR / "_downloads" / "vinbig_png" / "train_meta.csv",
            output_dir=Path(tmp_dir),
            train_split=0.8,
            val_split=0.2,
            seed=28,
            merge_iou_threshold=0.3,
            link_mode="symlink",
            max_images_per_split=8,
        )
        dataset_yaml = Path(metadata["dataset_yaml"])
        assert dataset_yaml.exists()
        manifest_train = (Path(tmp_dir) / "manifests" / "train.txt").read_text().strip().splitlines()
        manifest_val = (Path(tmp_dir) / "manifests" / "val.txt").read_text().strip().splitlines()
        assert len(manifest_train) == 8
        assert len(manifest_val) == 8
        label_files = list((Path(tmp_dir) / "labels" / "train").glob("*.txt"))
        assert label_files
        assert len(json.loads((Path(tmp_dir) / "metadata.json").read_text())["class_names"]) == len(DETECTION_CLASS_NAMES)


def test_image_level_bridge() -> None:
    detections = [
        {"class_name": "Pleural effusion", "confidence": 0.92},
        {"class_name": "Cardiomegaly", "confidence": 0.61},
        {"class_name": "Atelectasis", "confidence": 0.10},
    ]
    labels = derive_image_level_labels_from_detections(detections, confidence_threshold=0.25)
    assert labels["Pleural effusion"] == 1
    assert labels["Cardiomegaly"] == 1
    assert labels["Atelectasis"] == 0
    assert labels["No finding"] == 0

    labels = derive_image_level_labels_from_detections([], confidence_threshold=0.25)
    assert labels["No finding"] == 1


def test_ground_truth_lookup() -> None:
    sample_ids = build_split_image_ids(DATA_DIR / "train", 0.8, 0.2, 28)["val"][:5]
    gt = load_image_level_ground_truth(DATA_DIR / "train.csv", sample_ids)
    assert list(gt.index) == sample_ids
    assert list(gt.columns) == CLASS_NAMES


def main() -> None:
    test_raw_detection_inputs()
    test_split_consistency()
    test_prepare_subset_dataset()
    test_image_level_bridge()
    test_ground_truth_lookup()
    print("YOLO pipeline tests passed")


if __name__ == "__main__":
    main()
