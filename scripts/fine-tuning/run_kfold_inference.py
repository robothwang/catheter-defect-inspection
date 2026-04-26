from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

MODEL_SCRIPTS = {
    "resnet18": "ResNet18_inference.py",
    "resnet50": "ResNet50_inference.py",
    "densenet121": "DenseNet121_inference.py",
    "densenet201": "DenseNet201_inference.py",
    "efficientnetb0": "EfficientNetB0_inference.py",
    "mobilenetv2": "MobileNetv2_inference.py",
    "inceptionv3": "InceptionV3_inference.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference for every fold checkpoint in a k-fold experiment")
    parser.add_argument("--model", choices=sorted(MODEL_SCRIPTS), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-name", type=str, default="inference_predictions.csv")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_paths = sorted(args.run_root.glob("fold_*/best_model.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no fold checkpoints found under: {args.run_root}")

    for checkpoint_path in checkpoint_paths:
        output_csv = checkpoint_path.parent / args.output_name
        command = [
            sys.executable,
            str(SCRIPT_DIR / MODEL_SCRIPTS[args.model]),
            "--checkpoint",
            str(checkpoint_path),
            "--image-dir",
            str(args.image_dir),
            "--output-csv",
            str(output_csv),
            "--top-k",
            str(args.top_k),
            "--device",
            args.device,
        ]
        print("\n" + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
