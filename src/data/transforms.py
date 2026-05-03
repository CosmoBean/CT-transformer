"""
Image transforms for chest X-ray datasets.
"""
from __future__ import annotations

from torchvision import transforms


def build_classification_transform(
    image_size: int,
    is_train: bool,
    use_augmentation: bool = False,
) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if is_train and use_augmentation:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def build_anomaly_transform(
    image_size: int,
    is_train: bool,
    use_augmentation: bool = False,
) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if is_train and use_augmentation:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)
