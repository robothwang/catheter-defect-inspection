from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from EfficientNetB0 import CLASS_NAMES, build_efficientnet_b0, build_transform, load_image


SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DATASET_CLASS_DIR_GROUPS = [
    ("pro_1", "pro_2", "pro_3"),
    ("pro1_holealign", "pro2_holealign", "pro3_holealign"),
]


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_supported_images(image_dir: Path):
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def collect_image_paths(image_dir: Path) -> list[Path]:
    for class_dir_names in DATASET_CLASS_DIR_GROUPS:
        class_dirs = [image_dir / name for name in class_dir_names]
        if all(path.is_dir() for path in class_dirs):
            image_paths: list[Path] = []
            for class_dir in class_dirs:
                image_paths.extend(iter_supported_images(class_dir))
            return sorted(image_paths)

    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


@torch.no_grad()
def predict_image(model, image_path: Path, transform, device, class_names: list[str], top_k: int):
    image = load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    top_probabilities, top_indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    prediction = class_names[int(torch.argmax(probabilities).item())]
    top_rows = [
        {
            "class_name": class_names[int(index)],
            "probability": float(probability),
        }
        for probability, index in zip(top_probabilities, top_indices)
    ]
    return prediction, top_rows


def main():
    parser = argparse.ArgumentParser(description="Inference for catheter 3-class EfficientNet-B0 classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.image_path is None and args.image_dir is None:
        raise ValueError("one of --image-path or --image-dir must be provided")

    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", 224))

    model, _ = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_transform(image_size=image_size, train=False)

    if args.image_path is not None:
        prediction, top_rows = predict_image(
            model=model,
            image_path=args.image_path,
            transform=transform,
            device=device,
            class_names=class_names,
            top_k=args.top_k,
        )
        print(f"image_path  : {args.image_path}")
        print(f"checkpoint  : {args.checkpoint}")
        print(f"prediction  : {prediction}")
        print("\nClass probabilities:")
        print(pd.DataFrame(top_rows).to_string(index=False))
        return

    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        raise RuntimeError(f"no images found under: {args.image_dir}")

    rows = []
    for image_path in image_paths:
        prediction, top_rows = predict_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device,
            class_names=class_names,
            top_k=args.top_k,
        )
        row = {
            "path": str(image_path),
            "prediction": prediction,
        }
        for index, item in enumerate(top_rows, start=1):
            row[f"top{index}_class"] = item["class_name"]
            row[f"top{index}_probability"] = item["probability"]
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output_csv, index=False)
        print(f"saved: {args.output_csv}")

    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
