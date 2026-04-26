from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ResNet18 import collect_samples, save_split_csv


def default_output_dir(dataset_type: str, folds: int, seed: int) -> Path:
    return (
        Path("/home/hjj747/catheter-preprocessing/experiments/splits/kfold")
        / f"{dataset_type}_{folds}fold_seed{seed}"
    )


def make_fold_rows(
    dataframe: pd.DataFrame,
    train_val_indices,
    test_indices,
    seed: int,
    fold_number: int,
    val_size: float,
) -> list[dict]:
    train_val_df = dataframe.iloc[train_val_indices].reset_index(drop=True)
    test_df = dataframe.iloc[test_indices].reset_index(drop=True)

    test_fraction = len(test_df) / len(dataframe)
    relative_val_size = val_size / (1.0 - test_fraction)
    if not 0.0 < relative_val_size < 1.0:
        raise ValueError(
            f"invalid --val-size={val_size}; for this fold it becomes "
            f"{relative_val_size:.4f} of the train/val pool"
        )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=seed + fold_number,
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


def summarize_fold(rows: list[dict], fold_number: int, split_csv: Path) -> list[dict]:
    dataframe = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for split_name in ["train", "val", "test"]:
        split_df = dataframe[dataframe["split"] == split_name]
        label_counts = split_df["label_name"].value_counts().to_dict()
        summary_rows.append(
            {
                "fold": fold_number,
                "split": split_name,
                "n": len(split_df),
                "pro1": label_counts.get("pro1", 0),
                "pro2": label_counts.get("pro2", 0),
                "pro3": label_counts.get("pro3", 0),
                "split_csv": str(split_csv),
            }
        )
    return summary_rows


def create_kfold_splits(
    dataset_type: str,
    data_root: Path | None,
    output_dir: Path,
    folds: int,
    seed: int,
    val_size: float,
) -> list[Path]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    if not 0.0 < val_size < 1.0:
        raise ValueError("--val-size must be between 0 and 1")

    samples = collect_samples(dataset_type=dataset_type, data_root=data_root)
    dataframe = pd.DataFrame(samples)
    min_class_count = dataframe["label_name"].value_counts().min()
    if folds > min_class_count:
        raise ValueError(f"--folds={folds} is larger than the smallest class count ({min_class_count})")

    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    split_paths: list[Path] = []
    summary_rows: list[dict] = []
    for fold_index, (train_val_indices, test_indices) in enumerate(
        splitter.split(dataframe, dataframe["label_name"]),
        start=1,
    ):
        split_csv = output_dir / f"fold_{fold_index}.csv"
        rows = make_fold_rows(
            dataframe=dataframe,
            train_val_indices=train_val_indices,
            test_indices=test_indices,
            seed=seed,
            fold_number=fold_index,
            val_size=val_size,
        )
        save_split_csv(rows, split_csv)
        split_paths.append(split_csv)
        summary_rows.extend(summarize_fold(rows=rows, fold_number=fold_index, split_csv=split_csv))

    pd.DataFrame(summary_rows).to_csv(output_dir / "fold_summary.csv", index=False)
    return split_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified k-fold split CSVs for catheter classification")
    parser.add_argument("--dataset-type", choices=["original", "original_holealign"], required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation fraction relative to the full dataset. With 5 folds, default gives 65/15/20 train/val/test.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(
        dataset_type=args.dataset_type,
        folds=args.folds,
        seed=args.seed,
    )
    split_paths = create_kfold_splits(
        dataset_type=args.dataset_type,
        data_root=args.data_root,
        output_dir=output_dir,
        folds=args.folds,
        seed=args.seed,
        val_size=args.val_size,
    )

    print(f"saved split directory: {output_dir}")
    print(f"saved fold summary   : {output_dir / 'fold_summary.csv'}")
    for split_path in split_paths:
        counts = pd.read_csv(split_path)["split"].value_counts().to_dict()
        print(
            f"{split_path.name}: "
            f"train={counts.get('train', 0)}, val={counts.get('val', 0)}, test={counts.get('test', 0)}"
        )


if __name__ == "__main__":
    main()
