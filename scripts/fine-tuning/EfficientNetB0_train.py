from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from EfficientNetB0 import (
    CLASS_NAMES,
    build_efficientnet_b0,
    build_transform,
    collect_samples,
    freeze_backbone,
    load_image,
    make_record_id,
    save_split_csv,
)


@dataclass
class SampleItem:
    path: Path
    label_name: str
    label_index: int
    sample_id: str


class CatheterDataset(Dataset):
    def __init__(self, samples: list[SampleItem], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.path)
        tensor = self.transform(image)
        return tensor, sample.label_index, sample.sample_id, str(sample.path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_default_split_path(seed: int) -> Path:
    return Path("/home/hjj747/catheter-preprocessing/experiments/splits") / f"efficientnetb0_3class_seed{seed}.csv"


def generate_split_rows(samples: list[dict], seed: int, test_size: float, val_size: float) -> list[dict]:
    dataframe = pd.DataFrame(samples)
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["label_name"],
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_df["label_name"],
    )

    rows: list[dict] = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for row in split_df.itertuples(index=False):
            rows.append(
                {
                    "record_id": row.record_id,
                    "sample_id": row.sample_id,
                    "label_name": row.label_name,
                    "split": split_name,
                }
            )
    return rows


def load_or_create_splits(
    samples: list[dict],
    split_csv: Path,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, list[SampleItem]]:
    sample_lookup = {sample["record_id"]: sample for sample in samples}

    if split_csv.exists():
        split_df = pd.read_csv(split_csv)
    else:
        split_rows = generate_split_rows(samples, seed=seed, test_size=test_size, val_size=val_size)
        save_split_csv(split_rows, split_csv)
        split_df = pd.DataFrame(split_rows)

    split_samples = {"train": [], "val": [], "test": []}
    missing_ids: list[str] = []

    for row in split_df.itertuples(index=False):
        record_id = row.record_id if hasattr(row, "record_id") else make_record_id(row.label_name, row.sample_id)
        sample = sample_lookup.get(record_id)
        if sample is None:
            missing_ids.append(record_id)
            continue
        if sample["label_name"] != row.label_name:
            raise ValueError(f"label mismatch for sample_id={row.sample_id}: {sample['label_name']} != {row.label_name}")
        split_samples[row.split].append(
            SampleItem(
                path=sample["path"],
                label_name=sample["label_name"],
                label_index=sample["label_index"],
                sample_id=sample["sample_id"],
            )
        )

    if missing_ids:
        raise ValueError(f"split csv contains sample_ids not found in current dataset: {missing_ids[:10]}")

    return split_samples


def create_dataloaders(split_samples: dict[str, list[SampleItem]], image_size: int, batch_size: int, num_workers: int):
    train_dataset = CatheterDataset(split_samples["train"], transform=build_transform(image_size=image_size, train=True))
    val_dataset = CatheterDataset(split_samples["val"], transform=build_transform(image_size=image_size, train=False))
    test_dataset = CatheterDataset(split_samples["test"], transform=build_transform(image_size=image_size, train=False))

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_paths: list[str] = []
    all_sample_ids: list[str] = []

    for images, labels, sample_ids, paths in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_paths.extend(paths)
        all_sample_ids.extend(sample_ids)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": all_labels,
        "predictions": all_predictions,
        "paths": all_paths,
        "sample_ids": all_sample_ids,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels, _, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += images.size(0)

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_predictions_csv(result: dict, output_path: Path) -> None:
    rows = []
    for sample_id, path, label_idx, pred_idx in zip(
        result["sample_ids"], result["paths"], result["labels"], result["predictions"]
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "path": path,
                "label": CLASS_NAMES[label_idx],
                "prediction": CLASS_NAMES[pred_idx],
                "correct": int(label_idx == pred_idx),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 fine-tuning for catheter 3-class classification")
    parser.add_argument("--dataset-type", choices=["original", "original_holealign"], required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    output_dir = args.output_dir or (
        Path("/home/hjj747/catheter-preprocessing/experiments") / f"efficientnetb0_{args.dataset_type}"
    )
    split_csv = args.split_csv or make_default_split_path(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(dataset_type=args.dataset_type, data_root=args.data_root)
    split_samples = load_or_create_splits(
        samples=samples,
        split_csv=split_csv,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    dataloaders = create_dataloaders(
        split_samples=split_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, weights = build_efficientnet_b0(num_classes=len(CLASS_NAMES), pretrained=True)
    if args.freeze_backbone:
        freeze_backbone(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    best_val_accuracy = -1.0
    best_checkpoint_path = output_dir / "best_model.pth"
    history: list[dict] = []

    print(f"dataset_type : {args.dataset_type}")
    print(f"device       : {device}")
    print(f"split_csv    : {split_csv}")
    print(
        "split_sizes  : "
        f"train={len(split_samples['train'])}, val={len(split_samples['val'])}, test={len(split_samples['test'])}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_result = evaluate(model, dataloaders["val"], criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
        }
        history.append(row)
        print(
            f"[{epoch:02d}/{args.epochs:02d}] "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )

        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "dataset_type": args.dataset_type,
                    "image_size": args.image_size,
                    "weights_name": weights.__class__.__name__ if weights is not None else "None",
                    "split_csv": str(split_csv),
                    "freeze_backbone": args.freeze_backbone,
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = evaluate(model, dataloaders["test"], criterion, device)
    report = classification_report(
        test_result["labels"],
        test_result["predictions"],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_result["labels"], test_result["predictions"]).tolist()

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    save_predictions_csv(test_result, output_dir / "test_predictions.csv")

    summary = {
        "dataset_type": args.dataset_type,
        "data_root": str(args.data_root) if args.data_root is not None else "default",
        "split_csv": str(split_csv),
        "checkpoint_path": str(best_checkpoint_path),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone,
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_macro_f1": test_result["macro_f1"],
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print("\nTest Summary")
    print(f"accuracy : {test_result['accuracy']:.4f}")
    print(f"macro_f1 : {test_result['macro_f1']:.4f}")
    print(f"metrics   : {output_dir / 'test_metrics.json'}")
    print(f"checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
