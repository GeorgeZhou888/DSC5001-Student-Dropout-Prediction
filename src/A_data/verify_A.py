"""Read-only core delivery checks; --reproduce also builds in a fresh venv.

Verification reports are saved under results/A_data; frozen inputs are never
modified. Reproduction uses temporary copies and removes them on completion.
"""

from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import venv

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PACKAGES = {"numpy", "pandas", "scikit-learn", "matplotlib", "joblib"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(root, name):
    return json.loads((root/name).read_text(encoding="utf-8"))


def save_report(name, report):
    path = ROOT / "results/A_data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check_environment(root):
    errors = []
    expected = (root/"environment/python-version.txt").read_text().strip().removeprefix("Python ")
    if platform.python_version().split(".")[:2] != expected.split(".")[:2]:
        errors.append(f"Python major/minor expected {expected}, got {platform.python_version()}")
    found = set()
    for line in (root/"environment/requirements-lock.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match:
            errors.append(f"Lock entry is not a portable exact pin: {line}")
            continue
        name, wanted = match.groups()
        found.add(normalize(name))
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Package missing: {name}")
        else:
            if actual != wanted:
                errors.append(f"{name}: expected {wanted}, got {actual}")
    if not REQUIRED_PACKAGES.issubset(found):
        errors.append("Lock does not contain the five project packages")
    return errors


def verify_delivery():
    checks = {}
    manifest = read_json(ROOT, "data/manifest.json")
    bad = []
    for relative, entry in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != entry["sha256"]:
            bad.append(relative)
    checks["manifest_hashes_match"] = not bad
    environment_errors = check_environment(ROOT)
    checks["locked_environment_matches"] = not environment_errors
    raw = pd.read_csv(ROOT/"data/raw/data.csv", sep=";", float_precision="round_trip")
    data = pd.read_csv(ROOT/"data/processed/students.csv", float_precision="round_trip")
    splits = pd.read_csv(ROOT/"data/splits/split_manifest.csv", dtype={"cv_fold": "Int64"})
    expected_ids = [f"S{i:06d}" for i in range(1, len(raw)+1)]
    checks["expected_shapes"] = raw.shape == (4424, 37) and data.shape == (4424, 38)
    checks["stable_unique_ids"] = data["row_id"].tolist() == expected_ids and data["row_id"].is_unique
    checks["split_columns_exact"] = list(splits.columns) == ["row_id", "dataset_split", "cv_fold"]
    checks["split_covers_all_ids_once"] = (splits["row_id"].is_unique and
        set(splits["row_id"]) == set(data["row_id"]) and len(splits) == len(data))
    expected = raw.rename(columns=lambda value: value.strip())
    try:
        pd.testing.assert_frame_equal(expected, data.drop(columns="row_id"), check_exact=True)
        values_preserved = True
    except AssertionError:
        values_preserved = False
    checks["raw_values_preserved"] = values_preserved
    merged = data.merge(splits, on="row_id", validate="one_to_one", sort=False)
    train = merged.loc[merged["dataset_split"] == "train"]
    test = merged.loc[merged["dataset_split"] == "test"]
    checks["train_test_sizes"] = len(train) == 3539 and len(test) == 885
    checks["train_test_disjoint"] = set(train["row_id"]).isdisjoint(test["row_id"])
    checks["test_folds_empty"] = bool(test["cv_fold"].isna().all())
    checks["training_folds_valid"] = bool(train["cv_fold"].notna().all() and
        set(train["cv_fold"].dropna().astype(int)) == set(range(5)))
    sizes = [int((train["cv_fold"] == fold).sum()) for fold in range(5)]
    checks["fold_sizes_balanced"] = sizes == [708, 708, 708, 708, 707]
    labels = ["Dropout", "Enrolled", "Graduate"]
    checks["three_labels_retained"] = set(data["Target"]) == set(labels)
    checks["stratified_outer_counts"] = (
        train["Target"].value_counts().to_dict() == {"Dropout":1137, "Enrolled":635, "Graduate":1767}
        and test["Target"].value_counts().to_dict() == {"Dropout":284, "Enrolled":159, "Graduate":442})
    balanced = True
    for label in labels:
        counts = [int(((train["cv_fold"] == fold) & (train["Target"] == label)).sum()) for fold in range(5)]
        balanced = balanced and max(counts)-min(counts) <= 1
    checks["stratified_inner_counts"] = bool(balanced)
    feature = read_json(ROOT, "configs/feature_sets.json")
    protocol = read_json(ROOT, "configs/protocol.json")
    first, second = feature["enrollment"], feature["enrollment_plus_sem1"]
    cats, nums = feature["categorical_features"], feature["numeric_features"]
    forbidden = {"Target", "row_id", "dataset_split", "cv_fold"}
    checks["feature_groups_valid"] = (len(first) == len(set(first)) == 18 and
        len(second) == len(set(second)) == 24 and set(first).issubset(second)
        and set(second).issubset(data.columns) and not forbidden.intersection(second))
    checks["types_cover_features"] = (len(cats) == 14 and len(nums) == 10
        and set(cats).isdisjoint(nums) and set(cats+nums) == set(second))
    checks["second_semester_excluded"] = not any("2nd sem" in name for name in second)
    checks["configs_ready"] = feature["status"] == protocol["status"] == "ready"
    dictionary = pd.read_csv(ROOT/"results/A_data/data_dictionary.csv")
    checks["dictionary_covers_processed_table"] = (len(dictionary) == 38 and
        dictionary["column"].is_unique and set(dictionary["column"]) == set(data.columns))
    plot_info = read_json(ROOT, "results/A_data/plot_sources.json")
    train_digest = hashlib.sha256(("\n".join(train["row_id"]) + "\n").encode()).hexdigest()
    checks["plot_provenance_training_only"] = (plot_info["scope"] == "training_only" and
        plot_info["n"] == 3539 and plot_info["train_row_ids_sha256"] == train_digest)
    checks["four_figures_exist"] = len(plot_info["figures"]) == 4 and all(
        (ROOT/"results/A_data/figures"/name).is_file() for name in plot_info["figures"])
    audit = read_json(ROOT, "results/A_data/audit.json")
    checks["audits_pass"] = (audit["status"] == "PASS" and audit["stage"] == "A_delivery"
                            and audit["field_validation"]["status"] == "PASS")
    checks = {name: bool(value) for name, value in checks.items()}
    report = {"status": "PASS" if all(checks.values()) else "FAIL",
        "checked_at_hk": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "delivery_manifest_sha256": sha(ROOT/"data/manifest.json"), "checks": checks,
        "hash_mismatches": bad, "environment_errors": environment_errors,
        "counts": {"raw":len(raw), "processed":len(data), "train":len(train), "test":len(test)},
        "fold_sizes": sizes, "no_predictive_model_evaluated": True}
    save_report("delivery_check.json", report)
    print("=== A DELIVERY CHECK ===", flush=True)
    print("STATUS:", report["status"], flush=True)
    print("CHECKS:", sum(checks.values()), "/", len(checks), flush=True)
    print("TRAIN / TEST:", len(train), "/", len(test), flush=True)
    print("CV FOLD SIZES:", sizes, flush=True)
    print("REPORT: results/A_data/delivery_check.json", flush=True)
    if report["status"] != "PASS":
        print("FAILED:", [name for name, value in checks.items() if not value], flush=True)
        raise SystemExit("Delivery verification failed; inspect delivery_check.json before proceeding.")
    return manifest


def reproduce(manifest):
    before_manifest = sha(ROOT/"data/manifest.json")
    reference = {name: sha(ROOT/name) for name in manifest["core_files_for_reproduction"]}
    report = {"status":"RUNNING", "started_at_hk":datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "delivery_manifest_sha256":before_manifest, "fresh_virtual_environment":True,
        "temporary_data_copy":True, "python_version":platform.python_version()}
    save_report("reproduction_check.json", report)
    try:
        with tempfile.TemporaryDirectory(prefix="dsc5001_A_reproduce_") as directory:
            temporary = Path(directory)
            env = temporary/"venv"
            work = temporary/"project"
            print("=== FRESH-ENVIRONMENT REPRODUCTION ===", flush=True)
            print("Creating temporary virtual environment ...", flush=True)
            venv.EnvBuilder(with_pip=True).create(env)
            interpreter = env/("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            child_env = os.environ.copy()
            child_env.pop("PYTHONPATH", None)
            child_env.pop("PYTHONHOME", None)
            child_env["VIRTUAL_ENV"] = str(env)
            child_env["PATH"] = str(interpreter.parent) + os.pathsep + child_env.get("PATH", "")
            print("Installing the exact locked dependencies ...", flush=True)
            subprocess.run([str(interpreter), "-m", "pip", "install", "--disable-pip-version-check",
                "--only-binary=:all:", "-r", str(ROOT/"environment/requirements-lock.txt")], check=True, env=child_env)
            input_files = [name for name in manifest["files"] if name.startswith(
                ("src/A_data/", "configs/", "environment/", "data/raw/"))]
            for name in input_files:
                destination = work/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/name, destination)
            print("Rebuilding from raw inputs in the temporary copy ...", flush=True)
            subprocess.run([str(interpreter), str(work/"src/A_data/prepare_data.py")], cwd=work, check=True, env=child_env)
            subprocess.run([str(interpreter), str(work/"src/A_data/verify_A.py")], cwd=work, check=True, env=child_env)
            comparisons = {name: sha(work/name) == expected for name, expected in reference.items()}
            unchanged = sha(ROOT/"data/manifest.json") == before_manifest and all(
                sha(ROOT/name) == expected for name, expected in reference.items())
            report.update({"status":"PASS" if all(comparisons.values()) and unchanged else "FAIL",
                "core_file_matches":comparisons, "original_core_unchanged":unchanged,
                "note":"PNG files are existence/hash checked in each delivery; byte-identical rendering is not required. Plot-input tables are compared."})
    except Exception as exc:
        report.update({"status":"FAIL", "error":str(exc)})
        save_report("reproduction_check.json", report)
        raise SystemExit("Fresh-environment reproduction failed; share the full output and reproduction_check.json.")
    report["finished_at_hk"] = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")
    save_report("reproduction_check.json", report)
    print("=== REPRODUCTION RESULT ===")
    print("STATUS:", report["status"])
    print("CORE FILES MATCH:", sum(report["core_file_matches"].values()), "/", len(reference))
    print("ORIGINAL CORE UNCHANGED:", report["original_core_unchanged"])
    print("REPORT: results/A_data/reproduction_check.json")
    if report["status"] != "PASS":
        raise SystemExit("Core files did not reproduce exactly; inspect the reproduction report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true", help="Rebuild using a fresh temporary venv and compare core outputs")
    args = parser.parse_args()
    manifest = verify_delivery()
    if args.reproduce:
        reproduce(manifest)


if __name__ == "__main__":
    main()
