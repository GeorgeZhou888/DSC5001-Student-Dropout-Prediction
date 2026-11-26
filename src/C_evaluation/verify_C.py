"""Independent re-check of the C stage delivery.

Reads the artefacts in results/C_final/, recomputes the headline metric from
test_predictions.csv alone, and compares it with test_metrics.csv. Nothing is
refitted and no model is loaded except to confirm the frozen bytes are intact.

    python src/C_evaluation/verify_C.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("Asia/Hong_Kong")
OUT_DIR = "results/C_final"
MANIFEST = f"{OUT_DIR}/run_manifest.json"
METRICS = f"{OUT_DIR}/test_metrics.csv"
PREDICTIONS = f"{OUT_DIR}/test_predictions.csv"
CATALOG = "results/B_validation/model_catalog.json"
SPLIT = "data/splits/split_manifest.csv"
STUDENTS = "data/processed/students.csv"
REPORT = f"{OUT_DIR}/verification_report.json"
FIGURES = [
    "01_test_macro_f1_by_model.png",
    "02_feature_set_gain.png",
    "03_confusion_matrices.png",
    "04_interpretation_random_forest.png",
    "05_interpretation_logistic_regression.png",
]
LABELS = ["Dropout", "Enrolled", "Graduate"]
TOLERANCE = 1e-12


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for path in (MANIFEST, METRICS, PREDICTIONS, CATALOG, SPLIT, STUDENTS):
        if not (ROOT / path).is_file():
            raise SystemExit(f"DELIVERY CHECK FAILED: missing {path}")

    manifest = read_json(MANIFEST)
    metrics = pd.read_csv(ROOT / METRICS)
    predictions = pd.read_csv(ROOT / PREDICTIONS)
    catalog = read_json(CATALOG)
    split = pd.read_csv(ROOT / SPLIT)
    students = pd.read_csv(ROOT / STUDENTS)

    # 1. Upstream bytes: only the two documented exception classes may appear.
    statuses = {name: entry["byte_status"] for name, entry in manifest["inputs"].items()}
    allowed = {"exact", "lf_equivalent", "self_consistent_regenerated"}
    record("input_byte_status_is_documented", set(statuses.values()) <= allowed, json.dumps(statuses))
    record("frozen_b_commit_recorded",
           bool(manifest["b_frozen_reference"]["source_snapshot_commit"]),
           str(manifest["b_frozen_reference"]["source_snapshot_commit"]))

    # 2. Model bytes still match B's catalogue.
    catalog_hashes = {record_["path"]: record_["sha256"] for record_ in catalog["models"]}
    model_ok = all(sha256(ROOT / path) == digest for path, digest in catalog_hashes.items())
    record("model_files_unchanged", model_ok, f"{len(catalog_hashes)} files")

    # 3. Prediction table shape and internal consistency.
    test_ids = set(split.loc[split["dataset_split"] == "test", "row_id"])
    pairs = predictions[["row_id", "model", "feature_set"]].drop_duplicates()
    record("prediction_rows", len(predictions) == 6195, str(len(predictions)))
    record("prediction_students", set(predictions["row_id"]) == test_ids, f"{predictions['row_id'].nunique()} ids")
    record("prediction_model_rows", len(pairs) == 7 * 885, str(len(pairs)))
    probability_columns = ["prob_dropout", "prob_enrolled", "prob_graduate"]
    sums = predictions[probability_columns].sum(axis=1)
    record("probabilities_sum_to_one", bool(((sums - 1.0).abs() <= 1e-9).all()), f"max |sum-1| = {(sums - 1).abs().max():.2e}")
    argmax = predictions[probability_columns].idxmax(axis=1).str.replace("prob_", "").str.capitalize()
    record("predicted_label_is_argmax", bool((argmax == predictions["predicted_label"]).all()),
           f"{(argmax != predictions['predicted_label']).sum()} mismatches")

    # 4. True labels come from A's table, joined by row_id.
    truth = students.set_index("row_id")["Target"]
    record("true_labels_match_a_table",
           bool((predictions["row_id"].map(truth) == predictions["true_label"]).all()))

    # 5. Recompute macro-F1 and dropout precision/recall from the CSV alone.
    worst = {"macro_f1": 0.0, "dropout_precision": 0.0, "dropout_recall": 0.0}
    for (model, feature_set), group in predictions.groupby(["model", "feature_set"], sort=False):
        row = metrics[(metrics["model"] == model) & (metrics["feature_set"] == feature_set)]
        if len(row) != 1:
            record(f"metric_row_present::{model}::{feature_set}", False, f"{len(row)} rows")
            continue
        row = row.iloc[0]
        macro_f1 = f1_score(group["true_label"], group["predicted_label"], labels=LABELS,
                            average="macro", zero_division=0)
        precision, recall, _, _ = precision_recall_fscore_support(
            group["true_label"], group["predicted_label"], labels=LABELS, zero_division=0
        )
        diffs = {
            "macro_f1": abs(macro_f1 - row["macro_f1"]),
            "dropout_precision": abs(precision[0] - row["dropout_precision"]),
            "dropout_recall": abs(recall[0] - row["dropout_recall"]),
        }
        worst = {key: max(worst[key], value) for key, value in diffs.items()}
    record("recomputed_metrics_match", all(value <= TOLERANCE for value in worst.values()),
           json.dumps({key: f"{value:.2e}" for key, value in worst.items()}))

    # 6. Baseline sanity: the majority class is Graduate by construction.
    baseline = metrics[metrics["model"] == "majority_baseline"].iloc[0]
    record("baseline_predicts_majority", bool(baseline["dropout_recall"] == 0.0 and baseline["graduate_recall"] == 1.0))
    record("baseline_accuracy_equals_graduate_share",
           abs(baseline["accuracy"] - (predictions[predictions["model"] == "majority_baseline"]["true_label"]
                                       .eq("Graduate").mean())) <= TOLERANCE)

    # 7. Figures and output hashes.
    figures_ok = True
    for name in FIGURES:
        path = ROOT / OUT_DIR / "figures" / name
        figures_ok = figures_ok and path.is_file() and path.stat().st_size > 10_000
    record("five_figures_present", figures_ok, ", ".join(FIGURES))
    recorded = manifest["outputs"]
    hash_ok = all(sha256(ROOT / path) == entry["sha256"] for path, entry in recorded.items())
    record("output_hashes_match_manifest", hash_ok, f"{len(recorded)} files")

    passed = sum(1 for check in checks if check["passed"])
    report = {
        "stage": "C_final_evaluation",
        "verified_at": datetime.now(TZ).replace(microsecond=0).isoformat(),
        "timezone": "Asia/Hong_Kong",
        "checks_passed": passed,
        "checks_total": len(checks),
        "tolerance": TOLERANCE,
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks": checks,
    }
    (ROOT / REPORT).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for check in checks:
        print(f"  [{'ok' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
    print(f"\n{passed}/{len(checks)} checks passed -> {REPORT}")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
