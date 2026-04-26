from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from create_kfold_splits import create_kfold_splits, default_output_dir


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

MODEL_SCRIPTS = {
    "resnet18": "ResNet18_train.py",
    "resnet50": "ResNet50_train.py",
    "densenet121": "DenseNet121_train.py",
    "densenet201": "DenseNet201_train.py",
    "efficientnetb0": "EfficientNetB0_train.py",
    "mobilenetv2": "MobileNetv2_train.py",
    "inceptionv3": "InceptionV3_train.py",
}


def build_train_command(args, split_csv: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / MODEL_SCRIPTS[args.model]),
        "--dataset-type",
        args.dataset_type,
        "--split-csv",
        str(split_csv),
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.data_root is not None:
        command.extend(["--data-root", str(args.data_root)])
    if args.image_size is not None:
        command.extend(["--image-size", str(args.image_size)])
    if args.freeze_backbone:
        command.append("--freeze-backbone")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k-fold training for one catheter classification model")
    parser.add_argument("--model", choices=sorted(MODEL_SCRIPTS), required=True)
    parser.add_argument("--dataset-type", choices=["original", "original_holealign"], required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "experiments" / "kfold")
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_dir = args.split_dir or default_output_dir(
        dataset_type=args.dataset_type,
        folds=args.folds,
        seed=args.seed,
    )
    split_paths = create_kfold_splits(
        dataset_type=args.dataset_type,
        data_root=args.data_root,
        output_dir=split_dir,
        folds=args.folds,
        seed=args.seed,
        val_size=args.val_size,
    )

    tuning_suffix = "freeze" if args.freeze_backbone else "full"
    run_root = args.output_root / f"{args.model}_{args.dataset_type}_{args.folds}fold_{tuning_suffix}"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"split_dir : {split_dir}")
    print(f"run_root  : {run_root}")
    for fold_index, split_csv in enumerate(split_paths, start=1):
        output_dir = run_root / f"fold_{fold_index}"
        command = build_train_command(args=args, split_csv=split_csv, output_dir=output_dir)
        print("\n" + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
