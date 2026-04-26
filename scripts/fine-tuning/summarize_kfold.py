from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def discover_metric_files(root: Path) -> list[Path]:
    return sorted(root.rglob("test_metrics.json"))


def read_metric_row(metric_path: Path, root: Path) -> dict:
    with metric_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)

    fold_dir = metric_path.parent
    experiment_dir = fold_dir.parent
    return {
        "experiment": str(experiment_dir.relative_to(root)) if experiment_dir != root else experiment_dir.name,
        "fold": fold_dir.name,
        "dataset_type": metrics.get("dataset_type"),
        "split_csv": metrics.get("split_csv"),
        "checkpoint_path": metrics.get("checkpoint_path"),
        "test_loss": metrics.get("test_loss"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_macro_f1": metrics.get("test_macro_f1"),
    }


def summarize(fold_dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment, group in fold_dataframe.groupby("experiment", sort=True):
        rows.append(
            {
                "experiment": experiment,
                "n_folds": len(group),
                "accuracy_mean": group["test_accuracy"].mean(),
                "accuracy_std": group["test_accuracy"].std(ddof=1),
                "macro_f1_mean": group["test_macro_f1"].mean(),
                "macro_f1_std": group["test_macro_f1"].std(ddof=1),
                "loss_mean": group["test_loss"].mean(),
                "loss_std": group["test_loss"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize k-fold test metrics")
    parser.add_argument("--root", type=Path, default=Path("/home/hjj747/catheter-preprocessing/experiments/kfold"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    metric_files = discover_metric_files(args.root)
    if not metric_files:
        raise FileNotFoundError(f"no test_metrics.json files found under: {args.root}")

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dataframe = pd.DataFrame(read_metric_row(path, root=args.root) for path in metric_files)
    summary_dataframe = summarize(fold_dataframe)

    fold_csv = output_dir / "kfold_fold_metrics.csv"
    summary_csv = output_dir / "kfold_summary.csv"
    fold_dataframe.to_csv(fold_csv, index=False)
    summary_dataframe.to_csv(summary_csv, index=False)

    display_df = summary_dataframe.copy()
    for column in ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]:
        display_df[column] = (display_df[column] * 100).map(lambda value: f"{value:.2f}%")
    for column in ["loss_mean", "loss_std"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    print(display_df.to_string(index=False))
    print(f"\nsaved fold metrics: {fold_csv}")
    print(f"saved summary     : {summary_csv}")


if __name__ == "__main__":
    main()
