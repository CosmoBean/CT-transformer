"""
Label taxonomy and derived global bucket logic for the 15-label VinBigData subset.
"""
from __future__ import annotations

from typing import Any


STRONG_LOCALIZABLE = {
    "Aortic enlargement",
    "Calcification",
    "Cardiomegaly",
    "Nodule/Mass",
    "Pneumothorax",
}

DIFFUSE_OPTIONAL_LOCALIZATION = {
    "Atelectasis",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pulmonary fibrosis",
}

GLOBAL_IMAGE_LEVEL_ONLY = {
    "No finding",
}

LABEL_CATEGORY_BY_NAME = {
    **{label: "strong_localizable" for label in STRONG_LOCALIZABLE},
    **{label: "diffuse_optional_localization" for label in DIFFUSE_OPTIONAL_LOCALIZATION},
    **{label: "global_image_level_only" for label in GLOBAL_IMAGE_LEVEL_ONLY},
}

DERIVED_GLOBAL_BUCKETS: dict[str, list[str]] = {
    "No acute abnormality": ["No finding"],
    "Cardiomediastinal abnormality": ["Cardiomegaly", "Aortic enlargement", "Calcification"],
    "Pleural abnormality": ["Pleural effusion", "Pleural thickening", "Pneumothorax"],
    "Airspace or infectious-inflammatory pattern": [
        "Atelectasis",
        "Consolidation",
        "Infiltration",
        "Lung Opacity",
    ],
    "Chronic interstitial or fibrotic pattern": ["ILD", "Pulmonary fibrosis"],
    "Focal lesion or mass-like pattern": ["Nodule/Mass", "Other lesion"],
    "Possible urgent thoracic abnormality": ["Pneumothorax", "Pleural effusion", "Consolidation", "Lung Opacity"],
}


def _bucket_status_for_labels(
    bucket_labels: list[str],
    swin_predicted_labels: list[str],
    yolo_predicted_labels: list[str],
) -> tuple[list[str], list[str]]:
    swin_set = set(swin_predicted_labels)
    yolo_set = set(yolo_predicted_labels)
    supported: list[str] = []
    uncertain: list[str] = []

    for label in bucket_labels:
        if label not in swin_set and label not in yolo_set:
            continue
        category = LABEL_CATEGORY_BY_NAME.get(label, "diffuse_optional_localization")
        if category == "global_image_level_only":
            if label in swin_set and not (len(swin_set - {"No finding"}) > 0):
                supported.append(label)
            elif label in swin_set:
                uncertain.append(label)
            continue

        if category == "strong_localizable":
            if label in swin_set and label in yolo_set:
                supported.append(label)
            else:
                uncertain.append(label)
            continue

        if label in swin_set:
            supported.append(label)
        elif label in yolo_set:
            uncertain.append(label)

    return supported, uncertain


def derive_global_bucket_hints(
    swin_predicted_labels: list[str],
    yolo_predicted_labels: list[str],
) -> dict[str, Any]:
    supported_buckets: list[str] = []
    uncertain_buckets: list[str] = []
    bucket_evidence: dict[str, dict[str, list[str]]] = {}

    for bucket_name, bucket_labels in DERIVED_GLOBAL_BUCKETS.items():
        supported_labels, uncertain_labels = _bucket_status_for_labels(
            bucket_labels=bucket_labels,
            swin_predicted_labels=swin_predicted_labels,
            yolo_predicted_labels=yolo_predicted_labels,
        )
        if supported_labels or uncertain_labels:
            bucket_evidence[bucket_name] = {
                "supported_labels": supported_labels,
                "uncertain_labels": uncertain_labels,
            }
        if supported_labels:
            supported_buckets.append(bucket_name)
        elif uncertain_labels:
            uncertain_buckets.append(bucket_name)

    return {
        "allowed_bucket_names": list(DERIVED_GLOBAL_BUCKETS.keys()),
        "supported_buckets": supported_buckets,
        "uncertain_buckets": uncertain_buckets,
        "bucket_evidence": bucket_evidence,
    }
