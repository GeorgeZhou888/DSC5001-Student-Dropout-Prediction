from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import io
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/data.csv"
OUT = ROOT / "results/A_data/audit.json"
REFERENCE_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"

if not RAW.is_file():
    raise SystemExit("Missing data/raw/data.csv; copy the official raw file there first.")

raw_bytes = RAW.read_bytes()
header = next(csv.reader(io.StringIO(raw_bytes.decode("utf-8-sig")), delimiter=";"))
df = pd.read_csv(io.BytesIO(raw_bytes), sep=";")
if "Target" not in df.columns:
    raise SystemExit("Target column missing; check the file and delimiter.")

labels = {str(k): int(v) for k, v in df["Target"].value_counts(dropna=False).items()}
missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())
numeric = df.drop(columns="Target").select_dtypes(include="number")
nonfinite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
sha256 = hashlib.sha256(raw_bytes).hexdigest()
whitespace = {name: name.strip() for name in header if name != name.strip()}

checks = {
    "expected_shape": df.shape == (4424, 37),
    "unique_column_names": len(header) == len(set(header)),
    "expected_labels": set(labels) == {"Dropout", "Enrolled", "Graduate"},
    "no_missing_cells": missing == 0,
    "no_exact_duplicate_rows": duplicates == 0,
    "numeric_predictor_count_36": numeric.shape[1] == 36,
    "no_nonfinite_numeric_values": nonfinite == 0,
    "matches_current_official_file": sha256 == REFERENCE_SHA256,
}

report = {
    "stage": "raw_structure_only",
    "status": "PASS" if all(checks.values()) else "REVIEW_REQUIRED",
    "recorded_at_hk": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
    "source": {
        "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
        "doi": "10.24432/C5MC89",
        "license": "CC BY 4.0",
    },
    "raw_file": "data/raw/data.csv",
    "raw_bytes": len(raw_bytes),
    "raw_sha256": sha256,
    "rows": len(df),
    "columns_count": len(df.columns),
    "label_counts": labels,
    "missing_cells": missing,
    "exact_duplicate_rows": duplicates,
    "nonfinite_numeric_values": nonfinite,
    "columns": list(df.columns),
    "storage_dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
    "columns_with_outer_whitespace": whitespace,
    "checks": checks,
    "pending_review": [
        "Semantic variable types and valid code/range checks",
        "Feature availability timing and feature-set definition",
        "Processed data, fixed splits and training-only descriptive plots",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("STATUS:", report["status"])
print("SHAPE:", df.shape)
print("LABEL_COUNTS:", labels)
print("MISSING_CELLS:", missing)
print("EXACT_DUPLICATE_ROWS:", duplicates)
print("NONFINITE_NUMERIC_VALUES:", nonfinite)
print("RAW_SHA256:", sha256)
print("COLUMN_WHITESPACE:", whitespace)
print("REPORT: results/A_data/audit.json")
print("COLUMNS:")
for i, name in enumerate(df.columns, start=1):
    print(f"{i:02d}. {name!r}")

if report["status"] != "PASS":
    raise SystemExit("Review failed checks in audit.json; do not alter or delete rows automatically.")