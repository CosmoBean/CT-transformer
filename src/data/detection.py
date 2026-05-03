"""
Utilities for preparing and evaluating VinBigData-style detection datasets.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from .dataset import CLASS_NAMES, IMAGE_EXTENSIONS


DETECTION_CLASS_NAMES = [name for name in CLASS_NAMES if name != "No finding"]
DETECTION_CLASS_ID_TO_NAME = {idx: name for idx, name in enumerate(DETECTION_CLASS_NAMES)}
DETECTION_CLASS_NAME_TO_ID = {name: idx for idx, name in DETECTION_CLASS_ID_TO_NAME.items()}

RAW_ANNOTATION_COLUMNS = {
    "image_id",
    "class_name",
    "class_id",
    "rad_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
}
IMAGE_METADATA_COLUMNS = {"image_id", "dim0", "dim1"}


def load_raw_detection_annotations(raw_annotation_path: str | Path) -> pd.DataFrame:
    """Load the original VinBigData train.csv containing radiologist boxes."""
    raw_annotation_path = Path(raw_annotation_path)
    if not raw_annotation_path.exists():
        raise FileNotFoundError(f"Raw annotation file not found: {raw_annotation_path}")

    if zipfile.is_zipfile(raw_annotation_path):
        with zipfile.ZipFile(raw_annotation_path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV found inside {raw_annotation_path}")
            with archive.open(members[0]) as handle:
                df = pd.read_csv(handle)
    else:
        df = pd.read_csv(raw_annotation_path)

    missing = RAW_ANNOTATION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw annotation file is missing required columns: {', '.join(sorted(missing))}"
        )

    return df


def load_image_metadata(meta_csv_path: str | Path) -> pd.DataFrame:
    """Load the original image height/width metadata from the PNG export."""
    meta_csv_path = Path(meta_csv_path)
    if not meta_csv_path.exists():
        raise FileNotFoundError(f"Image metadata file not found: {meta_csv_path}")

    df = pd.read_csv(meta_csv_path)
    missing = IMAGE_METADATA_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Image metadata file is missing required columns: {', '.join(sorted(missing))}"
        )
    return df.set_index("image_id")


def build_split_image_ids(
    image_root: str | Path,
    train_split: float = 0.8,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Reproduce the existing classification train/val split from image paths."""
    del val_split  # The split is determined by train_split, matching ChestXRayDataset.
    image_root = Path(image_root)
    image_paths = sorted(
        [
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_root}")

    image_ids = [path.stem for path in image_paths]
    indices = np.arange(len(image_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_end = int(len(indices) * train_split)
    if train_end <= 0 or train_end >= len(indices):
        return {
            "train": [image_ids[idx] for idx in indices],
            "val": [],
        }

    return {
        "train": [image_ids[idx] for idx in indices[:train_end]],
        "val": [image_ids[idx] for idx in indices[train_end:]],
    }


def _intersection_area(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    intersection = _intersection_area(box_a, box_b)
    if intersection <= 0.0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def merge_overlapping_boxes(
    boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.3,
) -> list[tuple[float, float, float, float]]:
    """
    Merge overlapping boxes by connected components, then take the union box.
    """
    if not boxes:
        return []

    remaining = [tuple(map(float, box)) for box in boxes]
    merged: list[tuple[float, float, float, float]] = []

    while remaining:
        component = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            next_remaining = []
            for candidate in remaining:
                if any(
                    _intersection_area(candidate, member) > 0.0
                    or _box_iou(candidate, member) >= iou_threshold
                    for member in component
                ):
                    component.append(candidate)
                    changed = True
                else:
                    next_remaining.append(candidate)
            remaining = next_remaining

        xs_min = [box[0] for box in component]
        ys_min = [box[1] for box in component]
        xs_max = [box[2] for box in component]
        ys_max = [box[3] for box in component]
        merged.append((min(xs_min), min(ys_min), max(xs_max), max(ys_max)))

    return merged


def _materialize_image(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return
        except OSError:
            pass
        return

    if link_mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            if dst.exists() or dst.is_symlink():
                try:
                    if dst.resolve() == src.resolve():
                        return
                except OSError:
                    pass

    shutil.copy2(src, dst)


def _scale_and_clip_box(
    box: tuple[float, float, float, float],
    original_height: float,
    original_width: float,
    target_height: int,
    target_width: int,
) -> tuple[float, float, float, float] | None:
    x_min, y_min, x_max, y_max = box
    if original_width <= 0 or original_height <= 0:
        return None

    scaled_x_min = x_min * target_width / original_width
    scaled_y_min = y_min * target_height / original_height
    scaled_x_max = x_max * target_width / original_width
    scaled_y_max = y_max * target_height / original_height

    scaled_x_min = float(np.clip(scaled_x_min, 0.0, target_width))
    scaled_y_min = float(np.clip(scaled_y_min, 0.0, target_height))
    scaled_x_max = float(np.clip(scaled_x_max, 0.0, target_width))
    scaled_y_max = float(np.clip(scaled_y_max, 0.0, target_height))

    if scaled_x_max <= scaled_x_min or scaled_y_max <= scaled_y_min:
        return None
    return scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max


def _to_yolo_line(
    class_id: int,
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> str:
    x_min, y_min, x_max, y_max = box
    x_center = ((x_min + x_max) / 2.0) / image_width
    y_center = ((y_min + y_max) / 2.0) / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def derive_image_level_labels_from_detections(
    detections: list[dict[str, Any]],
    confidence_threshold: float = 0.25,
) -> dict[str, int]:
    """
    Convert a list of detection outputs into image-level 15-label predictions.
    """
    labels = {name: 0 for name in CLASS_NAMES}
    for detection in detections:
        class_name = detection.get("class_name")
        confidence = float(detection.get("confidence", 0.0))
        if class_name not in DETECTION_CLASS_NAME_TO_ID:
            continue
        if confidence < confidence_threshold:
            continue
        labels[class_name] = 1

    labels["No finding"] = 0 if any(labels[name] for name in DETECTION_CLASS_NAMES) else 1
    return labels


def load_image_level_ground_truth(
    classification_csv_path: str | Path,
    image_ids: list[str],
) -> pd.DataFrame:
    classification_csv_path = Path(classification_csv_path)
    df = pd.read_csv(classification_csv_path)
    if "image_id" not in df.columns:
        raise ValueError(f"Classification CSV missing image_id: {classification_csv_path}")
    lookup = df.set_index("image_id")[CLASS_NAMES]
    return lookup.loc[image_ids].copy()


def prepare_yolo_dataset(
    image_root: str | Path,
    raw_annotation_path: str | Path,
    image_metadata_path: str | Path,
    output_dir: str | Path,
    train_split: float = 0.8,
    val_split: float = 0.2,
    seed: int = 42,
    merge_iou_threshold: float = 0.3,
    link_mode: str = "symlink",
    max_images_per_split: int | None = None,
) -> dict[str, Any]:
    """
    Convert the raw VinBigData boxes into a YOLO directory layout.
    """
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    raw_df = load_raw_detection_annotations(raw_annotation_path)
    meta_df = load_image_metadata(image_metadata_path)
    split_ids = build_split_image_ids(
        image_root=image_root,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )
    if max_images_per_split is not None:
        split_ids = {
            split_name: image_ids[:max_images_per_split]
            for split_name, image_ids in split_ids.items()
        }

    abnormal_df = raw_df[raw_df["class_name"] != "No finding"].copy()
    abnormal_df = abnormal_df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    grouped_rows: dict[str, pd.DataFrame] = {
        image_id: group.copy()
        for image_id, group in abnormal_df.groupby("image_id", sort=False)
    }

    image_counts: dict[str, int] = {}
    box_counts: dict[str, int] = {}
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for split_name, image_ids in split_ids.items():
        image_dir = output_dir / "images" / split_name
        label_dir = output_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        manifest_lines = []
        split_box_count = 0
        for image_id in image_ids:
            src_path = image_root / f"{image_id}.png"
            if not src_path.exists():
                continue

            dst_image = image_dir / src_path.name
            _materialize_image(src_path, dst_image, link_mode=link_mode)
            manifest_lines.append(str(dst_image.resolve()))

            image_width, image_height = Image.open(src_path).size
            label_lines: list[str] = []

            if image_id in grouped_rows and image_id in meta_df.index:
                original_height = float(meta_df.at[image_id, "dim0"])
                original_width = float(meta_df.at[image_id, "dim1"])

                for class_name, class_group in grouped_rows[image_id].groupby("class_name", sort=False):
                    if class_name not in DETECTION_CLASS_NAME_TO_ID:
                        continue
                    raw_boxes = [
                        (
                            float(row.x_min),
                            float(row.y_min),
                            float(row.x_max),
                            float(row.y_max),
                        )
                        for row in class_group.itertuples()
                    ]
                    merged_boxes = merge_overlapping_boxes(
                        raw_boxes,
                        iou_threshold=merge_iou_threshold,
                    )
                    for merged_box in merged_boxes:
                        scaled_box = _scale_and_clip_box(
                            merged_box,
                            original_height=original_height,
                            original_width=original_width,
                            target_height=image_height,
                            target_width=image_width,
                        )
                        if scaled_box is None:
                            continue
                        label_lines.append(
                            _to_yolo_line(
                                class_id=DETECTION_CLASS_NAME_TO_ID[class_name],
                                box=scaled_box,
                                image_width=image_width,
                                image_height=image_height,
                            )
                        )

            split_box_count += len(label_lines)
            (label_dir / f"{image_id}.txt").write_text("\n".join(label_lines))

        image_counts[split_name] = len(manifest_lines)
        box_counts[split_name] = split_box_count
        (manifests_dir / f"{split_name}.txt").write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""))

    dataset_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": DETECTION_CLASS_ID_TO_NAME,
    }
    dataset_yaml_path = output_dir / "dataset.yaml"
    dataset_yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))

    metadata = {
        "image_root": str(image_root.resolve()),
        "raw_annotation_path": str(Path(raw_annotation_path).resolve()),
        "image_metadata_path": str(Path(image_metadata_path).resolve()),
        "dataset_yaml": str(dataset_yaml_path.resolve()),
        "train_split": train_split,
        "val_split": val_split,
        "seed": seed,
        "merge_iou_threshold": merge_iou_threshold,
        "link_mode": link_mode,
        "max_images_per_split": max_images_per_split,
        "image_counts": image_counts,
        "box_counts": box_counts,
        "class_names": DETECTION_CLASS_NAMES,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
