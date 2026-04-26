from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


CLASS_DIR_TO_LABEL = {
    "pro_1": "pro1",
    "pro_2": "pro2",
    "pro_3": "pro3",
    "pro1_holealign": "pro1",
    "pro2_holealign": "pro2",
    "pro3_holealign": "pro3",
}


def label_from_path(path_text: str) -> str | None:
    path = Path(path_text)
    for part in reversed(path.parts):
        if part in CLASS_DIR_TO_LABEL:
            return CLASS_DIR_TO_LABEL[part]
    return None


def read_prediction_file(csv_path: Path) -> dict:
    dataframe = pd.read_csv(csv_path)
    if "prediction" not in dataframe.columns or "path" not in dataframe.columns:
        raise ValueError(f"{csv_path} must contain path and prediction columns")

    labels = dataframe["path"].map(label_from_path)
    if labels.isna().any():
        missing_count = int(labels.isna().sum())
        raise ValueError(f"could not infer labels for {missing_count} rows in {csv_path}")

    predictions = dataframe["prediction"]
    return {
        "experiment": csv_path.parent.parent.name,
        "fold": csv_path.parent.name,
        "prediction_csv": str(csv_path),
        "n": len(dataframe),
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def summarize(fold_dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment, group in fold_dataframe.groupby("experiment", sort=True):
        rows.append(
            {
                "experiment": experiment,
                "n_folds": len(group),
                "n_per_fold": int(group["n"].iloc[0]),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=1),
                "macro_precision_mean": group["macro_precision"].mean(),
                "macro_recall_mean": group["macro_recall"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize k-fold inference CSVs")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-name", type=str, default="inference_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    prediction_files = sorted(args.root.glob(f"fold_*/{args.prediction_name}"))
    if not prediction_files:
        raise FileNotFoundError(f"no {args.prediction_name} files found under: {args.root}")

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_dataframe = pd.DataFrame(read_prediction_file(path) for path in prediction_files)
    summary_dataframe = summarize(fold_dataframe)

    fold_csv = output_dir / f"{Path(args.prediction_name).stem}_fold_metrics.csv"
    summary_csv = output_dir / f"{Path(args.prediction_name).stem}_summary.csv"
    fold_dataframe.to_csv(fold_csv, index=False)
    summary_dataframe.to_csv(summary_csv, index=False)

    display_df = summary_dataframe.copy()
    for column in [
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_precision_mean",
        "macro_recall_mean",
    ]:
        display_df[column] = (display_df[column] * 100).map(lambda value: f"{value:.2f}%")

    print(display_df.to_string(index=False))
    print(f"\nsaved fold metrics: {fold_csv}")
    print(f"saved summary     : {summary_csv}")


if __name__ == "__main__":
    main()
