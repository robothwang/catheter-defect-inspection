from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image


SOURCE_ROOT = Path("/home/hjj747/catheter-preprocessing/data/dataset/original")
OUTPUT_ROOT = Path("/home/hjj747/catheter-preprocessing/data/dataset/original_rr")
CSV_PATH = OUTPUT_ROOT / "rotation_metadata.csv"
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_fill_color(image: Image.Image):
    bands = len(image.getbands())
    if image.mode in {"1", "L", "I", "F", "P"}:
        return 0
    if image.mode == "LA":
        return (0, 0)
    if image.mode == "RGBA":
        return (0, 0, 0, 0)
    if bands == 1:
        return 0
    return tuple(0 for _ in range(bands))


def rotate_image(image_path: Path, angle_deg: float) -> Image.Image:
    with Image.open(image_path) as image:
        fill_color = get_fill_color(image)
        rotated = image.rotate(
            angle=angle_deg,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=fill_color,
        )
        return rotated.copy()


def collect_image_paths(input_root: Path) -> list[Path]:
    image_paths: list[Path] = []
    for class_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(image_path)
    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Generate random-rotation versions of original catheter images")
    parser.add_argument("--input-root", type=Path, default=SOURCE_ROOT, help="original image root directory")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="rotated image output directory")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH, help="rotation metadata csv output path")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducible rotation angles")
    parser.add_argument("--angle-min", type=float, default=0.0, help="minimum rotation angle in degrees")
    parser.add_argument("--angle-max", type=float, default=360.0, help="maximum rotation angle in degrees")
    parser.add_argument("--limit", type=int, default=None, help="optional number of images to process for testing")
    args = parser.parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"input root not found: {args.input_root}")
    if args.angle_max <= args.angle_min:
        raise ValueError("angle-max must be greater than angle-min")

    rng = random.Random(args.seed)
    image_paths = collect_image_paths(args.input_root)
    if not image_paths:
        raise RuntimeError(f"no image files found under: {args.input_root}")

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total = len(image_paths)

    print("=============== random rotation generation start ===============")
    print(f"input_root : {args.input_root}")
    print(f"output_root: {args.output_root}")
    print(f"csv_path   : {args.csv_path}")
    print(f"seed       : {args.seed}")
    print(f"angle_range: [{args.angle_min}, {args.angle_max})")
    print(f"images     : {total}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(args.input_root)
        output_path = args.output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        angle_deg = rng.uniform(args.angle_min, args.angle_max)
        rotated = rotate_image(image_path, angle_deg)
        rotated.save(output_path)

        rows.append(
            {
                "filename": image_path.name,
                "sample_id": image_path.stem,
                "label_name": relative_path.parts[0],
                "original_path": str(image_path),
                "rotated_path": str(output_path),
                "rotation_deg": f"{angle_deg:.6f}",
                "seed": args.seed,
            }
        )

        print(f"[{index}/{total}] saved {relative_path} angle={angle_deg:.3f}")

    with args.csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "filename",
                "sample_id",
                "label_name",
                "original_path",
                "rotated_path",
                "rotation_deg",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("===============================================================")
    print(f"done! metadata saved: {args.csv_path}")


if __name__ == "__main__":
    main()
