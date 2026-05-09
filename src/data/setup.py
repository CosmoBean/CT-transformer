"""
Dataset download and preparation helpers for VinBigData.
"""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd


TRAINING_CLASS_ORDER = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "No finding",
]

DEFAULT_IMAGE_DATASET = "xhlulu/vinbigdata"
DEFAULT_COMPETITION = "vinbigdata-chest-xray-abnormalities-detection"


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _download_if_missing(target: Path, cmd: list[str], cwd: Path) -> None:
    if target.exists():
        print(f"Already present: {target}")
        return
    _run(cmd, cwd=cwd)


def _ensure_unzipped(zip_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Already extracted: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {output_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def _normalize_annotations(raw_csv: Path, output_csv: Path) -> None:
    if zipfile.is_zipfile(raw_csv):
        with zipfile.ZipFile(raw_csv) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if not members:
                raise ValueError(f"No CSV file found inside {raw_csv}")
            with archive.open(members[0]) as handle:
                df = pd.read_csv(handle)
    else:
        df = pd.read_csv(raw_csv)

    if "image_id" not in df.columns or "class_name" not in df.columns:
        raise ValueError(f"Unexpected annotation format in {raw_csv}")

    records = []
    for image_id, group in df.groupby("image_id", sort=False):
        labels = {name: 0 for name in TRAINING_CLASS_ORDER}
        class_names = set(group["class_name"].astype(str))

        if "No finding" in class_names:
            labels["No finding"] = 1
        else:
            for class_name in class_names:
                if class_name in labels:
                    labels[class_name] = 1

        records.append({"image_id": image_id, **labels})

    pd.DataFrame.from_records(records).to_csv(output_csv, index=False)
    print(f"Wrote normalized labels: {output_csv}")


def _move_png_tree(extracted_dir: Path, data_dir: Path) -> None:
    for split in ("train", "test"):
        source = extracted_dir / split
        target = data_dir / split
        if not source.exists():
            continue
        if target.exists() and any(target.iterdir()):
            print(f"Already prepared: {target}")
            continue
        print(f"Moving {source} -> {target}")
        shutil.move(str(source), str(target))


def prepare_vinbigdata_dataset(
    data_dir: str = "data",
    image_dataset: str = DEFAULT_IMAGE_DATASET,
    competition: str = DEFAULT_COMPETITION,
    project_root: str | Path | None = None,
) -> dict[str, str]:
    repo_root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    target_data_dir = repo_root / data_dir
    downloads_dir = target_data_dir / "_downloads"
    extracted_dir = downloads_dir / "vinbig_png"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    png_zip = downloads_dir / "vinbigdata.zip"
    raw_annotations = downloads_dir / "train_raw.csv"
    prepared_annotations = target_data_dir / "train.csv"

    _download_if_missing(
        png_zip,
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            image_dataset,
            "-p",
            str(downloads_dir),
            "-o",
        ],
        cwd=repo_root,
    )

    if not raw_annotations.exists():
        _run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                "train.csv",
                "-p",
                str(downloads_dir),
            ],
            cwd=repo_root,
        )
        (downloads_dir / "train.csv").rename(raw_annotations)

    _ensure_unzipped(png_zip, extracted_dir)
    _move_png_tree(extracted_dir, target_data_dir)

    if not prepared_annotations.exists():
        _normalize_annotations(raw_annotations, prepared_annotations)
    else:
        print(f"Already present: {prepared_annotations}")

    print("Dataset ready.")
    print(f"Images: {target_data_dir / 'train'}")
    print(f"Labels: {prepared_annotations}")
    return {
        "data_dir": str(target_data_dir),
        "images_dir": str(target_data_dir / "train"),
        "labels_csv": str(prepared_annotations),
        "raw_annotations_csv": str(raw_annotations),
    }
