import csv


ERROR_REFERENCE = 0.20


METRICS_FIELDNAMES = [
    "catheter_type",
    "filename",
    "ok",
    "error",
    "rmse_error",
    "normalized_error",
    "error_percent",
    "rmse_section",
    "rmse_pair",
    "rmse_big",
    "rmse_small",
    "rmse_all",
    "rmse_outer",
    "match_mode",
    "score",
    "ksize",
    "threshold",
    "iou_pair",
    "iou_big",
    "iou_small",
    "iou_all",
    "iou_outer",
    "scale",
    "shift_x",
    "shift_y",
    "rotation_deg",
    "fit_scale",
    "gid_angle_deg",
]


def make_metrics_row(catheter_type, filename):

    row = {field: None for field in METRICS_FIELDNAMES}
    row["catheter_type"] = catheter_type
    row["filename"] = filename
    row["ok"] = False
    row["error"] = ""
    row["match_mode"] = ""
    return row


def normalize_error_value(error_value, reference=ERROR_REFERENCE):

    if error_value is None:
        return None

    ref = max(float(reference), 1e-8)
    return float(min(max(float(error_value) / ref, 0.0), 1.0))


def write_metrics_csv(csv_path, rows):

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in METRICS_FIELDNAMES})
