from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import GoogLeNet_Weights


SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_NAMES = ["pro1", "pro2", "pro3"]
DATASET_CLASS_DIRS = {
    "original": {
        "pro1": "pro_1",
        "pro2": "pro_2",
        "pro3": "pro_3",
    },
    "original_holealign": {
        "pro1": "pro1_holealign",
        "pro2": "pro2_holealign",
        "pro3": "pro3_holealign",
    },
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_default_data_root(dataset_type: str) -> Path:
    if dataset_type == "original":
        return Path("/home/hjj747/catheter-preprocessing/data/dataset/original")
    if dataset_type == "original_holealign":
        return Path("/home/hjj747/catheter-preprocessing/data/dataset/original_holealign")
    raise ValueError(f"unsupported dataset_type: {dataset_type}")


def normalize_sample_id(path: Path) -> str:
    return path.stem.strip().lower()


def make_record_id(label_name: str, sample_id: str) -> str:
    return f"{label_name}:{sample_id}"


def collect_samples(dataset_type: str, data_root: Path | None = None) -> list[dict]:
    root = Path(data_root) if data_root is not None else get_default_data_root(dataset_type)
    class_dirs = DATASET_CLASS_DIRS[dataset_type]

    samples: list[dict] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_dirs[class_name]
        if not class_dir.exists():
            raise FileNotFoundError(f"class directory not found: {class_dir}")

        image_paths = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in image_paths:
            sample_id = normalize_sample_id(path)
            samples.append(
                {
                    "path": path,
                    "label_name": class_name,
                    "label_index": class_index,
                    "sample_id": sample_id,
                    "record_id": make_record_id(class_name, sample_id),
                }
            )

    if not samples:
        raise RuntimeError(f"no image files found under: {root}")

    return samples


def build_googlenet(num_classes: int, pretrained: bool = True):
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    if pretrained:
        model = models.googlenet(weights=weights)
    else:
        model = models.googlenet(weights=None, aux_logits=False, init_weights=False)

    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model, weights


def freeze_backbone(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def build_transform(image_size: int = 224, train: bool = False):
    transform_steps: list = [transforms.Resize((image_size, image_size))]
    if train:
        transform_steps.extend(
            [
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ]
        )

    transform_steps.extend(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def save_split_csv(rows: Iterable[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def load_split_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"split csv not found: {csv_path}")
    return pd.read_csv(csv_path)
