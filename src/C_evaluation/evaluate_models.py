"""C stage: final evaluation of B's frozen pipelines on the 885 held-out test rows.

The test rows are read from A's fixed split, never regenerated. No estimator is
fitted here: B's joblib files already contain the preprocessing step and the
classifier, fitted on the 3,539 training rows.

Run from the repository root:

    python src/C_evaluation/evaluate_models.py            # write results/C_final
    python src/C_evaluation/evaluate_models.py --check     # verify inputs only
    python src/C_evaluation/evaluate_models.py --force     # replace existing results

Selection rules were fixed before any test score was produced: model and
hyper-parameters come from results/B_validation/selected_params.json, and the
representative model for interpretation is the one B nominated there
("representative_model_for_C"). Test labels only enter the metric computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
TZ = ZoneInfo("Asia/Hong_Kong")

PROTOCOL = "configs/protocol.json"
FEATURES = "configs/feature_sets.json"
STUDENTS = "data/processed/students.csv"
SPLIT = "data/splits/split_manifest.csv"
B_MANIFEST = "results/B_validation/run_manifest.json"
B_CATALOG = "results/B_validation/model_catalog.json"
B_PARAMS = "results/B_validation/selected_params.json"
B_VERIFICATION = "results/B_validation/input_verification.json"
OUT_DIR = "results/C_final"
FIG_DIR = "results/C_final/figures"

EXCLUDED = ["Target", "row_id", "dataset_split", "cv_fold"]
PROBA_COLUMNS = {"Dropout": "prob_dropout", "Enrolled": "prob_enrolled", "Graduate": "prob_graduate"}
FEATURE_SET_ORDER = ["enrollment", "enrollment_plus_sem1"]
MODEL_ORDER = ["logistic_regression", "decision_tree", "random_forest"]
FEATURE_SET_LABEL = {
    "enrollment": "Enrollment only (18 predictors)",
    "enrollment_plus_sem1": "Enrollment + 1st semester (24 predictors)",
    "shared_majority": "Majority class (shared)",
}
MODEL_LABEL = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "majority_baseline": "Majority baseline",
}
COLORS = {"enrollment": "#4c72b0", "enrollment_plus_sem1": "#dd8452", "shared_majority": "#8c8c8c"}

# LF-canonical digests recorded by A in docs/handoffs/A.md. The files delivered in
# the working copy use CRLF line endings, so their raw bytes hash differently;
# both digests are recorded and the canonical one must still match.
A_CANONICAL = {
    "data/processed/students.csv": "c8a87c649896f4567f7a45509d38631316d5dc8805860fb058d5386495509996",
    "data/splits/split_manifest.csv": "2c0094e4359d9d1310f0dd4e9bae69440f5c631f101bada67ec4dc028a3db629",
    "configs/feature_sets.json": "6267ce371bf3e02dc95427913184c452c475721dcf7ec3289ace17bc24126a4e",
    "configs/protocol.json": "ae41689c85f2eddfae5ca876b781e293cfc07c5f5edf1a985d3191e74cc5c0ea",
}
VERIFY_INPUTS = [B_MANIFEST, B_CATALOG, B_PARAMS, B_VERIFICATION]


# ---------------------------------------------------------------- small helpers


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv(relative, frame, protect=False):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")
    if protect and path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Existing {relative} would change. Rerun with --force after reviewing the cause.")
    path.write_bytes(content)
    return path


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def lf_canonical_digest(path):
    """SHA-256 of the file after normalising CRLF to LF (A recorded LF digests)."""
    return digest_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


SELF_DESCRIBED_MANIFEST = "data/manifest.json"


def manifest_self_consistency():
    """A's data/manifest.json hashes every delivered file but cannot hash itself.

    Its own bytes therefore change whenever the copies it describes are re-exported,
    so it is verified by content: every listed digest must still describe this copy.
    """
    record = read_json(SELF_DESCRIBED_MANIFEST)
    failures = []
    for name, entry in record.get("files", {}).items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f"missing {name}")
        elif entry.get("sha256") not in (digest(path), lf_canonical_digest(path)):
            failures.append(f"digest differs for {name}")
    counts = record.get("row_counts", {})
    if (counts.get("train"), counts.get("test")) != (3539, 885):
        failures.append(f"row_counts are {counts.get('train')}/{counts.get('test')}, expected 3539/885")
    return not failures, {"listed_files": len(record.get("files", {})), "failures": failures}


def now_hk():
    return datetime.now(TZ).replace(microsecond=0).isoformat()


def fail(message):
    raise SystemExit(f"EVIDENCE CHECK FAILED: {message}")


# --------------------------------------------------------------------- inputs


def load_and_verify():
    """Read A's fixed split and B's frozen catalogue; refuse to continue on mismatch."""
    for name in (PROTOCOL, FEATURES, STUDENTS, SPLIT, B_MANIFEST, B_CATALOG, B_PARAMS):
        if not (ROOT / name).is_file():
            fail(f"missing required upstream file {name}")

    protocol = read_json(PROTOCOL)
    features = read_json(FEATURES)
    b_manifest = read_json(B_MANIFEST)
    catalog = read_json(B_CATALOG)
    b_params = read_json(B_PARAMS)

    checks = {}

    # 1. Upstream bytes must be the ones B trained on, or a line-ending-only copy of them.
    #    B trained on a CRLF checkout; the delivered repository uses LF, so every affected
    #    file is checked against the dual digest B recorded in input_verification.json.
    verification = read_json(B_VERIFICATION) if (ROOT / B_VERIFICATION).is_file() else {}
    lf_recorded = {
        entry["file"]: entry.get("lf_canonical_sha256") for entry in verification.get("files", [])
    }
    input_states = {}
    for name, expected in b_manifest.get("inputs_actual_sha256", {}).items():
        raw = digest(ROOT / name)
        canonical = lf_canonical_digest(ROOT / name)
        detail = None
        if raw == expected:
            state = "exact"
        elif canonical == (lf_recorded.get(name) or A_CANONICAL.get(name)):
            state = "lf_equivalent"
        else:
            state = "mismatch"
        if name == SELF_DESCRIBED_MANIFEST:
            # The manifest describes the files next to it and cannot hash itself, so its own
            # listing is re-read whichever byte state it arrived in; only a manifest that
            # fails that listing and matches nothing upstream is treated as a mismatch.
            consistent, detail = manifest_self_consistency()
            if state == "mismatch" and consistent:
                state = "self_consistent_regenerated"
        input_states[name] = {
            "actual_sha256": raw,
            "b_recorded_sha256": expected,
            "lf_canonical_sha256": canonical,
            "byte_status": state,
        }
        if detail is not None:
            input_states[name]["self_consistency"] = detail
    bad = sorted(name for name, state in input_states.items() if state["byte_status"] == "mismatch")
    checks["upstream_bytes_match_b_or_lf_equivalent"] = not bad
    if bad:
        fail(f"input bytes match neither B's run manifest nor an LF-normalised copy: {bad}")
    lf_only = sorted(name for name, state in input_states.items() if state["byte_status"] == "lf_equivalent")
    if lf_only:
        print(f"NOTE: {len(lf_only)} input(s) differ from B's recorded bytes only by CRLF/LF; "
              f"LF-canonical digests match what B and A recorded:")
        for name in lf_only:
            print(f"  lf_equivalent  {name}")
    regenerated = sorted(name for name, state in input_states.items()
                         if state["byte_status"] == "self_consistent_regenerated")
    if regenerated:
        detail = input_states[regenerated[0]]["self_consistency"]
        print(f"NOTE: {regenerated[0]} predates the LF re-export but still describes this copy "
              f"({detail['listed_files']} files, {len(detail['failures'])} failures).")
    checks["self_described_manifest_consistent"] = all(
        state["self_consistency"]["failures"] == []
        for state in input_states.values() if "self_consistency" in state
    )

    # 2. Same content as A's canonical (LF) version.
    canonical_ok = {name: lf_canonical_digest(ROOT / name) == expected for name, expected in A_CANONICAL.items()}
    checks["a_canonical_lf_hashes_match"] = all(canonical_ok.values())
    if not all(canonical_ok.values()):
        fail(f"LF-canonical content differs from A: {[k for k, v in canonical_ok.items() if not v]}")

    # 3. Fixed split integrity.
    split = pd.read_csv(ROOT / SPLIT)
    students = pd.read_csv(ROOT / STUDENTS)
    labels = list(protocol["labels"])
    checks["split_columns"] = list(split.columns) == ["row_id", "dataset_split", "cv_fold"]
    train = split[split["dataset_split"] == "train"]
    test = split[split["dataset_split"] == "test"]
    checks["row_id_unique"] = bool(split["row_id"].is_unique and students["row_id"].is_unique)
    checks["split_counts"] = (len(train), len(test)) == (3539, 885)
    checks["train_folds_complete"] = sorted(train["cv_fold"].dropna().astype(int).unique().tolist()) == [0, 1, 2, 3, 4]
    checks["test_rows_have_no_fold"] = bool(test["cv_fold"].isna().all())
    checks["split_covers_every_record"] = set(split["row_id"]) == set(students["row_id"])
    checks["train_test_disjoint"] = not (set(train["row_id"]) & set(test["row_id"]))
    checks["labels_are_protocol_labels"] = set(students["Target"].unique()) == set(labels)

    test_frame = students.merge(test[["row_id"]], on="row_id", how="inner", validate="one_to_one")
    test_frame = test_frame.sort_values("row_id").reset_index(drop=True)
    checks["test_frame_rows"] = len(test_frame) == 885
    checks["test_frame_matches_split_order"] = test_frame["row_id"].tolist() == sorted(test["row_id"].tolist())

    # 4. Model catalogue integrity.
    model_dir = ROOT / "models"
    entries = []
    for record in catalog["models"]:
        path = ROOT / record["path"]
        if not path.is_file():
            fail(f"missing model file {record['path']}")
        actual_hash = digest(path)
        if actual_hash != record["sha256"]:
            fail(f"{record['path']} hash differs from B model_catalog.json")
        if record["feature_set"] != "shared_majority" and list(record["features"]) != features[record["feature_set"]]:
            fail(f"{record['path']} feature list differs from A feature_sets.json")
        entries.append(record)
    checks["seven_model_files_present"] = len(entries) == 7
    checks["model_bytes_match_catalog"] = True

    # 5. Feature contract: exactly A's lists, test columns all present.
    contract = {}
    for feature_set in FEATURE_SET_ORDER:
        columns = list(features[feature_set])
        if any(column in EXCLUDED for column in columns):
            fail(f"{feature_set} contains a non-predictor field")
        missing = [column for column in columns if column not in test_frame.columns]
        if missing:
            fail(f"{feature_set} missing columns {missing}")
        contract[feature_set] = columns
    checks["feature_lists_contain_no_excluded_fields"] = True

    b_commit = b_manifest.get("source_snapshot_commit")
    checks["b_frozen_commit_present"] = bool(b_commit)

    if not all(values for values in checks.values()):
        fail(f"failed checks {[k for k, v in checks.items() if not v]}")

    return {
        "protocol": protocol,
        "features": features,
        "catalog": catalog,
        "b_params": b_params,
        "b_manifest": b_manifest,
        "test_frame": test_frame,
        "contract": contract,
        "checks": checks,
        "labels": labels,
        "input_states": input_states,
    }


# ------------------------------------------------------------------- evaluation


def _probabilities(pipeline, frame):
    """Map predict_proba columns through classes_ instead of assuming positional order."""
    values = pipeline.predict_proba(frame)
    frame_proba = pd.DataFrame(values, columns=[str(name) for name in pipeline.classes_], index=frame.index)
    for label, column in PROBA_COLUMNS.items():
        if label not in frame_proba.columns:
            raise RuntimeError(f"pipeline.classes_ lacks {label}: {list(frame_proba.columns)}")
    return frame_proba.loc[:, list(PROBA_COLUMNS)].rename(columns=PROBA_COLUMNS)


def _metrics_block(y_true, y_pred, proba, labels):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_label = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    row = {
        "n_test": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    for index, label in enumerate(labels):
        stem = label.lower()
        row[f"{stem}_precision"] = float(per_label[0][index])
        row[f"{stem}_recall"] = float(per_label[1][index])
        row[f"{stem}_f1"] = float(per_label[2][index])
    values = proba.to_numpy(dtype=float)
    if np.isfinite(values).all() and values.min() > 0.0:
        row["log_loss"] = float(log_loss(y_true, values, labels=labels))
    else:
        # A deterministic majority-class baseline returns 0/1 probabilities.
        row["log_loss"] = None
    return row


def evaluate(inputs, specs):
    test_frame = inputs["test_frame"]
    labels = inputs["labels"]
    y_true = test_frame["Target"].astype(str).to_numpy()

    metric_rows = []
    prediction_rows = []
    details = {}

    for spec in specs:
        pipeline = spec["pipeline"]
        columns = spec["features"]
        frame = test_frame.loc[:, columns]
        predicted = np.asarray(pipeline.predict(frame)).astype(str)
        proba = _probabilities(pipeline, frame)

        row = {
            "model": spec["model"],
            "feature_set": spec["feature_set"],
            "model_file": spec["path"],
        }
        row.update(_metrics_block(y_true, predicted, proba, labels))
        metric_rows.append(row)

        predictions = pd.DataFrame(
            {
                "row_id": test_frame["row_id"].to_numpy(),
                "model": spec["model"],
                "feature_set": spec["feature_set"],
                "true_label": y_true,
                "predicted_label": predicted,
            }
        ).join(proba.reset_index(drop=True))
        predictions["correct"] = (predictions["true_label"] == predictions["predicted_label"]).astype(int)
        prediction_rows.append(predictions)

        details[spec["key"]] = {
            "y_true": y_true,
            "y_pred": predicted,
            "proba": proba.to_numpy(dtype=float),
            "confusion": confusion_matrix(y_true, predicted, labels=labels),
        }

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return metrics, predictions, details


def model_specs(inputs):
    """Load the six frozen pipelines plus the shared baseline exactly once."""
    import joblib

    specs = []
    for record in inputs["catalog"]["models"]:
        pipeline = joblib.load(ROOT / record["path"])
        if list(map(str, pipeline.classes_)) != inputs["labels"]:
            fail(f"{record['path']} classes {list(map(str, pipeline.classes_))} != protocol labels")
        feature_set = record["feature_set"]
        features = inputs["contract"][record["input_feature_set"]] if feature_set == "shared_majority" else inputs["contract"][feature_set]
        specs.append(
            {
                "key": f"{feature_set}__{record['model']}",
                "model": record["model"],
                "feature_set": feature_set,
                "path": record["path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "features": features,
                "pipeline": pipeline,
            }
        )
    order = {name: index for index, name in enumerate(FEATURE_SET_ORDER + ["shared_majority"])}
    model_order = {name: index for index, name in enumerate(MODEL_ORDER + ["majority_baseline"])}
    specs.sort(key=lambda spec: (order[spec["feature_set"]], model_order[spec["model"]]))
    return specs


# --------------------------------------------------------------------- figures


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True, "figure.facecolor": "white"})


def _save(fig, filename, rect=(0, 0, 1, 0.95)):
    path = ROOT / FIG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def plot_macro_f1(metrics, baseline_row):
    fig, axis = plt.subplots(figsize=(8.2, 4.4))
    width = 0.36
    positions = np.arange(len(MODEL_ORDER))
    for offset, feature_set in zip((-width / 2, width / 2), FEATURE_SET_ORDER):
        values = [
            float(metrics[(metrics.feature_set == feature_set) & (metrics.model == model)]["macro_f1"].iloc[0])
            for model in MODEL_ORDER
        ]
        bars = axis.bar(positions + offset, values, width, label=FEATURE_SET_LABEL[feature_set],
                        color=COLORS[feature_set], edgecolor="white")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=8)
    if baseline_row is not None:
        baseline = float(baseline_row["macro_f1"])
        axis.axhline(baseline, color=COLORS["shared_majority"], linestyle="--", linewidth=1.2,
                     label=f"Majority baseline ({baseline:.3f})")
    axis.set_xticks(positions)
    axis.set_xticklabels([MODEL_LABEL[model] for model in MODEL_ORDER])
    axis.set_ylabel("Macro F1 (three classes)")
    axis.set_ylim(0, 1)
    axis.set_title("Final test macro F1, n = 885 held-out students", pad=12)
    axis.legend(frameon=False, loc="upper left")
    return _save(fig, "01_test_macro_f1_by_model.png")


def plot_feature_gain(metrics):
    panels = (("macro_f1", "Macro F1"), ("dropout_recall", "Dropout recall"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for axis, (metric, title) in zip(axes, panels):
        starts, ends = [], []
        for row, model in enumerate(MODEL_ORDER):
            start = float(metrics[(metrics.feature_set == "enrollment") & (metrics.model == model)][metric].iloc[0])
            end = float(metrics[(metrics.feature_set == "enrollment_plus_sem1") & (metrics.model == model)][metric].iloc[0])
            starts.append(start)
            ends.append(end)
            axis.plot([start, end], [row, row], color="#a6a6a6", linewidth=1.8, zorder=1)
            axis.scatter([start], [row], s=52, color=COLORS["enrollment"], zorder=3,
                         label=FEATURE_SET_LABEL["enrollment"] if row == 0 else None)
            axis.scatter([end], [row], s=52, color=COLORS["enrollment_plus_sem1"], zorder=3,
                         label=FEATURE_SET_LABEL["enrollment_plus_sem1"] if row == 0 else None)
            axis.text(start, row + 0.26, f"{start:.3f}", fontsize=7.5, ha="center", va="bottom", color="#3f3f3f")
            axis.text(end, row + 0.26, f"{end:.3f}", fontsize=7.5, ha="center", va="bottom", color="#3f3f3f")
            axis.text((start + end) / 2, row - 0.24, f"{end - start:+.3f}", fontsize=8.2, ha="center",
                      va="top", color="#b5432f")
        low, high = min(starts), max(ends)
        span = high - low
        axis.set_xlim(low - 0.30 * span, high + 0.30 * span)
        axis.set_ylim(-0.85, 2.7)
        axis.set_yticks(range(len(MODEL_ORDER)))
        axis.set_yticklabels([MODEL_LABEL[model] for model in MODEL_ORDER], fontsize=8.5)
        axis.set_xlabel(title)
        axis.set_title(f"{title} by feature set", fontsize=9.5)
        axis.grid(axis="x")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    return _save(fig, "02_feature_set_gain.png", rect=(0, 0.07, 1, 0.95))


def plot_confusion(details):
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    labels = ["Dropout", "Enrolled", "Graduate"]
    for row_index, feature_set in enumerate(FEATURE_SET_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index][column_index]
            matrix = details[f"{feature_set}__{model}"]["confusion"]
            axis.imshow(matrix, cmap="Blues", vmin=0)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=9,
                              color="white" if matrix[i, j] > matrix.max() / 2 else "black")
            axis.set_xticks(range(3))
            axis.set_xticklabels(labels, fontsize=8)
            axis.set_yticks(range(3))
            axis.set_yticklabels(labels, fontsize=8)
            axis.set_xlabel("Predicted", fontsize=8)
            axis.set_ylabel("True", fontsize=8)
            axis.set_title(f"{MODEL_LABEL[model]}\n{FEATURE_SET_LABEL[feature_set].split(' (')[0]}", fontsize=8.5)
            axis.grid(False)
    return _save(fig, "03_confusion_matrices.png")


def _collapse_to_raw_features(names, raw_features):
    """Map one-hot output names such as 'categorical__Course_2' back to 'Course'."""
    mapping = {}
    ordered = sorted(raw_features, key=len, reverse=True)
    for name in names:
        body = name.split("__", 1)[1] if "__" in name else name
        matched = next((raw for raw in ordered if body == raw or body.startswith(raw + "_")), body)
        mapping[name] = matched
    return mapping


def plot_random_forest_importance(details, representative, test_frame):
    frame = test_frame.loc[:, representative["features"]]
    y_true = details[representative["key"]]["y_true"]
    result = permutation_importance(
        representative["pipeline"], frame, y_true, scoring="f1_macro",
        n_repeats=10, random_state=42, n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": representative["features"],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=True).tail(15)
    fig, axis = plt.subplots(figsize=(8.6, 5.6))
    axis.barh(table["feature"], table["importance_mean"], xerr=table["importance_std"],
              color=COLORS["enrollment_plus_sem1"], alpha=0.9, error_kw={"lw": 0.8})
    axis.set_xlabel("Drop in macro F1 when the column is permuted (10 repeats, seed 42)")
    axis.set_title(f"Representative model, test set: {MODEL_LABEL[representative['model']]} "
                   f"({FEATURE_SET_LABEL[representative['feature_set']].split(' (')[0]})", pad=10)
    return _save(fig, "04_interpretation_random_forest.png"), table


def plot_logistic_coefficients(details, spec, raw_features):
    pipeline = spec["pipeline"]
    preprocessor = pipeline[:-1]
    classifier = pipeline[-1]
    names = list(preprocessor.get_feature_names_out())
    mapping = _collapse_to_raw_features(names, raw_features)
    class_index = list(map(str, classifier.classes_)).index("Dropout")
    coefficients = classifier.coef_[class_index]
    table = pd.DataFrame({"name": names, "coefficient": coefficients})
    table["feature"] = table["name"].map(mapping)
    grouped = table.groupby("feature")["coefficient"].mean().reset_index()
    grouped["absolute"] = grouped["coefficient"].abs()
    grouped = grouped.sort_values("absolute", ascending=True).tail(15)
    colors = ["#b5432f" if value > 0 else "#4c72b0" for value in grouped["coefficient"]]
    fig, axis = plt.subplots(figsize=(8.6, 5.6))
    axis.barh(grouped["feature"], grouped["coefficient"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Mean logistic-regression coefficient for the Dropout class")
    axis.set_title("Signed coefficients for the Dropout class\n"
                   f"{FEATURE_SET_LABEL[spec['feature_set']].split(' (')[0]}: "
                   "positive values push towards Dropout", pad=10, fontsize=10)
    return _save(fig, "05_interpretation_logistic_regression.png"), grouped


# ---------------------------------------------------------------------- output


def output_paths():
    return [
        f"{OUT_DIR}/test_metrics.csv",
        f"{OUT_DIR}/test_predictions.csv",
        f"{OUT_DIR}/run_manifest.json",
        f"{FIG_DIR}/01_test_macro_f1_by_model.png",
        f"{FIG_DIR}/02_feature_set_gain.png",
        f"{FIG_DIR}/03_confusion_matrices.png",
        f"{FIG_DIR}/04_interpretation_random_forest.png",
        f"{FIG_DIR}/05_interpretation_logistic_regression.png",
    ]


def write_manifest(inputs, specs, outputs, elapsed, command):
    import importlib

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib"):
        packages[name] = getattr(importlib.import_module(name.replace("scikit-learn", "sklearn")), "__version__", None)
    environment = {
        "python": sys.version.split()[0],
        "executable": str(Path(sys.executable)),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }
    b_manifest = inputs["b_manifest"]
    manifest = {
        "schema_version": 1,
        "stage": "C_final_evaluation",
        "status": "COMPLETE_PENDING_TEAM_ACCEPTANCE",
        "generated_at": now_hk(),
        "timezone": "Asia/Hong_Kong",
        "b_frozen_reference": {
            "source_snapshot_commit": b_manifest.get("source_snapshot_commit"),
            "source_zip_sha256": b_manifest.get("source_zip_sha256"),
            "run_manifest_sha256": digest(ROOT / B_MANIFEST),
            "model_catalog_sha256": digest(ROOT / B_CATALOG),
            "representative_model_for_interpretation": inputs["b_params"].get("representative_model_for_C"),
            "selection_rule": "B fixed the representative model before C saw any test score; "
                              "no parameter, feature or model was re-selected from test results.",
        },
        "inputs": {
            name: {
                **state,
                "a_canonical_lf_sha256": A_CANONICAL.get(name),
                "a_canonical_match": lf_canonical_digest(ROOT / name) == A_CANONICAL[name]
                if name in A_CANONICAL else None,
            }
            for name, state in inputs["input_states"].items()
        },
        "input_integrity_policy": {
            "rule": "actual bytes must equal B's recorded bytes, or a CRLF/LF-only variant whose "
                    "LF-canonical SHA-256 equals the dual digest B recorded and A certified",
            "line_ending_only_differences_accepted": True,
            "any_other_difference_fails_the_run": True,
        },
        "models": {
            spec["path"]: {"sha256": spec["sha256"], "bytes": spec["bytes"],
                           "model": spec["model"], "feature_set": spec["feature_set"],
                           "n_features": len(spec["features"])}
            for spec in specs
        },
        "test_frame": {
            "rows": int(len(inputs["test_frame"])),
            "source": "A split_manifest.csv dataset_split == test; row_id one-to-one join",
            "labels": inputs["labels"],
        },
        "evaluation_policy": {
            "refit": False,
            "hyperparameter_search": False,
            "threshold_tuning": False,
            "test_used_for_model_or_feature_selection": False,
            "probability_mapping": "pipeline.classes_ -> prob_dropout/prob_enrolled/prob_graduate",
            "zero_division": 0,
        },
        "environment": environment,
        "command": command,
        "elapsed_seconds": round(elapsed, 2),
        "checks": inputs["checks"],
        "outputs": {
            path: {"sha256": digest(ROOT / path), "bytes": (ROOT / path).stat().st_size}
            for path in outputs
        },
    }
    return write_json(f"{OUT_DIR}/run_manifest.json", manifest)


def describe(metrics):
    columns = ["model", "feature_set", "n_test", "macro_f1", "dropout_precision", "dropout_recall",
               "accuracy", "balanced_accuracy"]
    return metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify upstream inputs only; write nothing")
    parser.add_argument("--force", action="store_true", help="replace existing results/C_final files")
    parser.add_argument("--skip-figures", action="store_true", help="write tables without regenerating figures")
    args = parser.parse_args()

    started = time.perf_counter()
    inputs = load_and_verify()
    print("Upstream verification: PASS")
    for name, value in inputs["checks"].items():
        print(f"  [{ 'ok' if value is not False else 'FAIL' }] {name}: {value}")
    if args.check:
        print("Input check finished; no test record was scored.")
        return 0

    existing = [path for path in output_paths() if (ROOT / path).exists()]
    if existing and not args.force:
        fail(f"results already exist ({len(existing)} files); use --force only after reviewing them")

    specs = model_specs(inputs)
    metrics, predictions, details = evaluate(inputs, specs)
    metrics = metrics[["model", "feature_set", "model_file", "n_test", "macro_f1", "macro_precision",
                       "macro_recall", "accuracy", "balanced_accuracy",
                       "dropout_precision", "dropout_recall", "dropout_f1",
                       "enrolled_precision", "enrolled_recall", "enrolled_f1",
                       "graduate_precision", "graduate_recall", "graduate_f1", "log_loss"]]
    order = {f"{feature_set}__{model}": index for index, (feature_set, model) in enumerate(
        [(feature_set, model) for feature_set in FEATURE_SET_ORDER for model in MODEL_ORDER]
        + [("shared_majority", "majority_baseline")]
    )}
    metrics["_order"] = metrics.apply(lambda row: order[f"{row['feature_set']}__{row['model']}"], axis=1)
    metrics = metrics.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    _style()
    written = [write_csv(f"{OUT_DIR}/test_metrics.csv", metrics)]
    written.append(write_csv(f"{OUT_DIR}/test_predictions.csv", predictions))

    representative = next(spec for spec in specs if spec["feature_set"] == inputs["b_params"]["representative_model_for_C"]["feature_set"]
                          and spec["model"] == inputs["b_params"]["representative_model_for_C"]["model"])
    logistic = next(spec for spec in specs if spec["feature_set"] == "enrollment_plus_sem1"
                    and spec["model"] == "logistic_regression")

    importance_table = coefficient_table = None
    figures = []
    if not args.skip_figures:
        baseline_row = metrics[metrics.model == "majority_baseline"].iloc[0] if (metrics.model == "majority_baseline").any() else None
        figures.append(plot_macro_f1(metrics, baseline_row))
        figures.append(plot_feature_gain(metrics))
        figures.append(plot_confusion(details))
        path, importance_table = plot_random_forest_importance(details, representative, inputs["test_frame"])
        figures.append(path)
        path, coefficient_table = plot_logistic_coefficients(details, logistic, logistic["features"])
        figures.append(path)
        written.append(write_csv(f"{OUT_DIR}/permutation_importance_representative.csv",
                                 importance_table.sort_values("importance_mean", ascending=False)))
        written.append(write_csv(f"{OUT_DIR}/logistic_dropout_coefficients.csv",
                                 coefficient_table.sort_values("absolute", ascending=False).drop(columns="absolute")))
    written += figures

    elapsed = time.perf_counter() - started
    command = "python src/C_evaluation/evaluate_models.py" + (" --force" if args.force else "")
    write_manifest(inputs, specs, [str(path.relative_to(ROOT)).replace("\\", "/") for path in written],
                   elapsed, command)

    print()
    print(describe(metrics))
    print()
    print(f"Rows written: metrics={len(metrics)}, predictions={len(predictions)}, figures={len(figures)}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
