"""Build the complete A-stage delivery. Run from the repository root.

Requires audit_raw.py, define_features.py, verify_A.py, the raw CSV and the
recorded environment. No predictive model is fitted and test features are
not used for descriptive analysis. Existing data/splits may not silently drift.
"""

from datetime import datetime
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import io
import json
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
CORE_FILES = [
    "data/processed/students.csv", "data/splits/split_manifest.csv",
    "configs/feature_sets.json", "configs/protocol.json",
    "results/A_data/data_dictionary.csv", "results/A_data/train_class_summary.csv",
    "results/A_data/train_numeric_summary.csv", "results/A_data/train_sem1_by_outcome.csv",
    "results/A_data/split_summary.csv",
]
FIGURES = [
    "01_train_class_distribution.png", "02_train_enrollment_distributions.png",
    "03_train_sem1_distributions.png", "04_train_sem1_grade_by_outcome.png",
]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(relative, value):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(relative, frame, protect=False):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")
    if protect and path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"Existing {relative} would change. Investigate the environment/protocol first.")
    path.write_bytes(content)


def run_helper(name):
    subprocess.run([sys.executable, str(ROOT / "src/A_data" / name)], cwd=ROOT, check=True)


def current_head():
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def add_identifier_to_dictionary():
    path = ROOT / "results/A_data/data_dictionary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row["column"] != "row_id"]
    identifier = {key: "" for key in fields}
    identifier.update({
        "column": "row_id", "description_cn": "按原始文件行顺序生成的稳定编号；首条学生记录为S000001",
        "semantic_type": "identifier", "source_storage_type": "generated string",
        "availability": "任何列名整理或样本划分之前生成",
        "in_enrollment": "False", "in_enrollment_plus_sem1": "False",
        "exclusion_reason": "样本标识，不是预测特征",
        "validity_rule": "S加六位序号；与原始记录顺序一一对应",
        "invalid_value_count": "0", "notes": "后续排序、连接和预测文件均保留此编号",
    })
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([identifier] + rows)


def save_figure(fig, filename):
    path = ROOT / "results/A_data/figures" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def training_summaries_and_plots(train, feature, labels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    counts = train["Target"].value_counts().reindex(labels, fill_value=0)
    class_table = pd.DataFrame({"Target": labels, "count": counts.to_numpy(),
                                "proportion": counts.to_numpy() / len(train)})
    write_csv("results/A_data/train_class_summary.csv", class_table)
    numeric = feature["numeric_features"]
    summary = train[numeric].describe().T.reset_index().rename(columns={"index": "column"})
    summary["zero_count"] = [(train[name] == 0).sum() for name in summary["column"]]
    write_csv("results/A_data/train_numeric_summary.csv", summary)
    grade = "Curricular units 1st sem (grade)"
    approved = "Curricular units 1st sem (approved)"
    outcome_rows = []
    for label in labels:
        subset = train.loc[train["Target"] == label]
        outcome_rows.append({
            "Target": label, "n": len(subset), "grade_mean": subset[grade].mean(),
            "grade_median": subset[grade].median(),
            "zero_grade_count": int((subset[grade] == 0).sum()),
            "zero_grade_proportion": float((subset[grade] == 0).mean()),
            "approved_units_mean": subset[approved].mean(),
            "approved_units_median": subset[approved].median(),
        })
    outcomes = pd.DataFrame(outcome_rows)
    write_csv("results/A_data/train_sem1_by_outcome.csv", outcomes)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, counts.to_numpy(), color=["#c65c4e", "#d4a549", "#477fa1"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + 20,
                f"{count:,} ({count/len(train):.1%})", ha="center", fontsize=10)
    ax.set(title=f"Outcome distribution | training data only (n={len(train):,})",
           ylabel="Students", ylim=(0, int(counts.max() * 1.18)))
    save_figure(fig, FIGURES[0])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = ["Age at enrollment", "Admission grade", "Previous qualification (grade)", "Application order"]
    for ax, name in zip(axes.flat, names):
        if name == "Application order":
            values = train[name].value_counts().sort_index()
            ax.bar(values.index, values.values, color="#477fa1")
        else:
            ax.hist(train[name], bins=24, color="#477fa1", edgecolor="white")
        ax.set(title=name, ylabel="Students")
    fig.suptitle(f"Enrollment variables | training data only (n={len(train):,})", y=0.99)
    save_figure(fig, FIGURES[1])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.flat, feature["enrollment_plus_sem1"][len(feature["enrollment"]):]):
        ax.hist(train[name], bins=22, color="#5b8f83", edgecolor="white")
        ax.set(title=name.replace("Curricular units 1st sem ", ""), ylabel="Students")
    fig.suptitle(f"First-semester variables | training data only (n={len(train):,})", y=0.99)
    save_figure(fig, FIGURES[2])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, name, title in [(axes[0], grade, "First-semester average grade"),
                            (axes[1], approved, "First-semester approved units")]:
        ax.boxplot([train.loc[train["Target"] == label, name].to_numpy() for label in labels])
        ax.set_xticks([1, 2, 3], labels)
        ax.set(title=title, ylabel="Original scale")
    fig.suptitle(f"Academic variables by recorded outcome | training data only (n={len(train):,})", y=0.99)
    save_figure(fig, FIGURES[3])

    grade_medians = "; ".join(f"{r.Target}: {r.grade_median:.2f}" for r in outcomes.itertuples())
    proportions = "; ".join(f"{r.Target}: {r.proportion:.1%}" for r in class_table.itertuples())
    findings = f"""# A阶段训练集描述结果

范围：仅使用固定划分中的{len(train):,}条训练记录。图形与描述统计不含测试样本。

- 训练集类别占比：{proportions}。应在模型验证中保留三类并关注宏平均F1及退学类指标。
- 第一学期平均成绩的组内中位数：{grade_medians}。这是训练数据中的描述性关联，不能解释为因果效果。
- 零成绩及零课程计数均保留；不能逐条确定零值对应的具体学业原因。
- 类别编号不表示连续数量关系；编码及缩放由B在各训练折内部拟合。

## 图与表

| 文件 | 用途 |
|---|---|
| figures/{FIGURES[0]} | 类别规模与不平衡程度 |
| figures/{FIGURES[1]} | 入学年龄、两项入学成绩及志愿顺序分布 |
| figures/{FIGURES[2]} | 第一学期六项变量的原始量纲分布 |
| figures/{FIGURES[3]} | 按最终记录结局描述第一学期成绩及通过课程数 |
| train_class_summary.csv | 类别图对应数据 |
| train_numeric_summary.csv | 两组特征中数值字段的训练集描述统计 |
| train_sem1_by_outcome.csv | 按结局分组的训练集学业描述统计 |

## 解释边界

数据来自单一机构，随机留出测试不能证明跨学校或跨时间泛化。缺少逐学生退出日期，
第一学期数据可能已反映部分学生的退出过程，故比较属于回顾性信息比较，不能宣称
已经验证实时预警或因果效果。Enrolled是独立结局，不能合并为成功；Dropout也不等同于
永久离开整个高等教育系统。18/24项特征按含义与时点预设，不依测试结果修改。
"""
    (ROOT / "results/A_data/findings.md").write_text(findings, encoding="utf-8")
    write_json("results/A_data/plot_sources.json", {
        "scope": "training_only", "n": len(train),
        "train_row_ids_sha256": hashlib.sha256(("\n".join(train["row_id"]) + "\n").encode()).hexdigest(),
        "figures": FIGURES, "numeric_columns": numeric,
        "outcome_column": "Target", "no_model_fitted": True,
    })


def write_handoff(manifest, split_summary):
    path = ROOT / "docs/handoffs/A.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = "<!-- A_AUTOGENERATED_START -->"
    end = "<!-- A_AUTOGENERATED_END -->"
    hashes = "\n".join(f"| `{name}` | `{manifest['files'][name]['sha256']}` |"
                       for name in CORE_FILES[:4])
    auto = f"""{start}
## A阶段自动交付摘要

状态：本地数据基准已生成；独立复现见results/A_data/reproduction_check.json。
本摘要不代表B已接收或组长已合并。

- 负责人：A（组长）；接收人：B；内部截止：2026-11-05 23:59，香港时间。
- 输入基准提交：`{manifest['input_baseline_commit']}`。
- 本次生成时仓库HEAD：`{manifest['repository_head_at_run']}`。
- 原始/整理记录：4424/4424；整理表38列，其中row_id为标识、Target为标签。
- 训练/测试：3539/885；seed=42；训练部分按固定cv_fold=0至4进行五折验证。
- 特征：入学18项、入学加第一学期24项；类别编码及缩放由B在训练折内拟合。
- 环境：`environment/python-version.txt`、`requirements-lock.txt`及`system-info.json`。

### 文件位置

| 路径 | 用途 |
|---|---|
| src/A_data/audit_raw.py、define_features.py、prepare_data.py、verify_A.py | 原始核验、字段定义、完整生成与独立验收 |
| data/raw/data.csv、uci_metadata.json | 保留的官方数据与字典 |
| data/processed/students.csv | 保留原量纲和所有原始字段的整理表 |
| data/splits/split_manifest.csv | 唯一训练/测试及五折划分；按row_id连接 |
| configs/feature_sets.json、protocol.json | ready状态的共同特征与实验口径 |
| data/manifest.json | 来源、清理说明、环境与文件校验值 |
| results/A_data/ | 审计、38行数据字典、训练集统计、4张图及解读 |

### 运行命令（仓库根目录）

```bash
python src/A_data/verify_A.py
python src/A_data/verify_A.py --reproduce
```

完整生成入口为`python src/A_data/prepare_data.py`，会依次调用前两项检查并统一更新全部交付。
`--reproduce`在临时全新虚拟环境及临时数据副本中重建，并比较核心文件；完成后自动清理临时目录。
B使用与记录相同的Python主次版本，从锁定依赖重建环境，读取已验收的划分，不再随机划分。
连接要求：两张CSV按row_id一对一连接；cv_fold=k表示第k次验证时留出的训练子集，测试行折号为空。
B使用feature_sets.json中的实际列表选列；row_id、Target和划分标记均不能作为输入特征。

### 核心校验值

| 文件 | SHA-256 |
|---|---|
{hashes}

### 已知局限

未知/空白编码34和99保留为类别；零成绩保留。财务及宏观时点不明字段保守排除。
第一学期比较仅为回顾性比较，缺少退出日期；不宣称实时预警、因果效果或跨机构泛化。
Enrolled保持独立类别。raw文件保留不改，整理表仅添加稳定编号及去除列名首尾空白。
详细结果见results/A_data/findings.md及data/manifest.json。
{end}
"""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in existing and end in existing:
        before, rest = existing.split(start, 1)
        _, after = rest.split(end, 1)
        result = before + auto + after
    else:
        stock = (ROOT / "docs/handoffs/A.md").exists() and hashlib.sha256(existing.encode()).hexdigest() == "6e863b281dab1ff6219bd6b0f847991370bed0068bb4cdaff06eba6a1dae95d3"
        manual = """## 人工验收记录（待真实完成后填写）

- 本阶段PR链接：待填写
- 独立复现结果与时间：待填写
- 接收人B确认及日期：待填写
- 组长验收结论：待填写
- 合并提交编号与日期：待填写
"""
        if existing and not stock:
            manual += "\n## 原有记录（保留）\n\n" + existing
        result = "# A阶段交接单\n\n" + auto + "\n" + manual
    path.write_text(result, encoding="utf-8")


def main():
    required = ["data/raw/data.csv", "data/raw/uci_metadata.json", "configs/protocol.json",
                "environment/python-version.txt", "environment/requirements-lock.txt",
                "environment/system-info.json"] + ["src/A_data/" + name for name in
                    ["audit_raw.py", "define_features.py", "prepare_data.py", "verify_A.py"]]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))
    protocol = read_json("configs/protocol.json")
    if not (protocol["random_seed"] == 42 and protocol["test_fraction"] == 0.2
            and protocol["cross_validation"]["n_splits"] == 5
            and protocol["outer_split"]["stratified"] and protocol["outer_split"]["shuffle"]
            and protocol["cross_validation"]["stratified"] and protocol["cross_validation"]["shuffle"]
            and protocol["target_column"] == "Target" and protocol["row_id_column"] == "row_id"
            and protocol["labels"] == ["Dropout", "Enrolled", "Graduate"]):
        raise SystemExit("Protocol differs from the agreed seed42/80:20/5fold/three-label baseline.")
    from verify_A import check_environment
    errors = check_environment(ROOT)
    if errors:
        raise SystemExit("Environment differs from its lock: " + "; ".join(errors))
    print("=== A STAGE: BUILD ===", flush=True)
    run_helper("audit_raw.py")
    run_helper("define_features.py")
    feature = read_json("configs/feature_sets.json")
    raw = pd.read_csv(ROOT / "data/raw/data.csv", sep=";", float_precision="round_trip")
    frame = raw.copy()
    frame.insert(0, "row_id", [f"S{i:06d}" for i in range(1, len(raw)+1)])
    frame = frame.rename(columns=lambda name: name.strip())
    write_csv("data/processed/students.csv", frame, protect=True)
    roundtrip = pd.read_csv(ROOT / "data/processed/students.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(frame, roundtrip, check_exact=True)

    positions = np.arange(len(frame))
    train_pos, test_pos = train_test_split(positions, test_size=protocol["test_fraction"],
        stratify=frame["Target"], shuffle=True, random_state=protocol["random_seed"])
    train_pos, test_pos = np.sort(train_pos), np.sort(test_pos)
    splits = pd.DataFrame({"row_id": frame["row_id"], "dataset_split": "test",
                           "cv_fold": pd.Series(pd.NA, index=frame.index, dtype="Int64")})
    splits.loc[train_pos, "dataset_split"] = "train"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, validation_pos) in enumerate(cv.split(np.zeros(len(train_pos)), frame.iloc[train_pos]["Target"])):
        splits.loc[train_pos[validation_pos], "cv_fold"] = fold
    write_csv("data/splits/split_manifest.csv", splits, protect=True)
    merged = frame.merge(splits, on="row_id", validate="one_to_one", sort=False)
    train = merged.loc[merged["dataset_split"] == "train"].copy()
    labels = protocol["labels"]
    split_rows = []
    groups = [("train", train), ("test", merged.loc[merged["dataset_split"] == "test"])]
    groups += [(f"train_fold_{i}", train.loc[train["cv_fold"] == i]) for i in range(5)]
    for name, subset in groups:
        split_rows.append({"subset": name, "n": len(subset), **{
            label: int((subset["Target"] == label).sum()) for label in labels}})
    write_csv("results/A_data/split_summary.csv", pd.DataFrame(split_rows))
    add_identifier_to_dictionary()
    training_summaries_and_plots(train, feature, labels)

    feature["status"] = "ready"
    protocol["status"] = "ready"
    protocol["row_id_policy"] = "S000001 onward in original raw CSV record order; assigned before column-name trimming"
    protocol["split_row_order"] = "Outer positions in raw record order; sort training positions before fixed StratifiedKFold"
    protocol["cv_fold_semantics"] = "Training row with fold=k is validation in iteration k; test rows have no fold"
    write_json("configs/feature_sets.json", feature)
    write_json("configs/protocol.json", protocol)
    audit = read_json("results/A_data/audit.json")
    audit.update({
        "stage": "A_delivery", "field_validation": read_json("results/A_data/field_validation.json"),
        "rows_removed": 0, "column_name_changes": feature["rename_map"],
        "processed_shape": list(frame.shape), "raw_predictor_values_preserved": True,
        "row_id_unique": bool(frame["row_id"].is_unique),
        "split_counts": {row["subset"]: row["n"] for row in split_rows},
        "feature_counts": {"enrollment": 18, "enrollment_plus_sem1": 24},
        "pending_review": ["Independent fresh-environment reproduction and B acceptance"],
        "no_model_fitted": True,
    })
    write_json("results/A_data/audit.json", audit)
    generated = ["results/A_data/audit.json", "results/A_data/field_validation.json",
                 "results/A_data/plot_sources.json", "results/A_data/findings.md"]
    generated += ["results/A_data/figures/" + name for name in FIGURES]
    frozen = sorted(set(required + CORE_FILES + generated))
    environment = read_json("environment/system-info.json")
    manifest = {
        "schema_version": 1, "status": "ready_for_independent_verification",
        "generated_at_hk": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "source": {"name": "Predict Students' Dropout and Academic Success",
            "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
            "doi": "10.24432/C5MC89", "license": "CC BY 4.0",
            "citation": "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository.",
            "acquisition": "Manual download of the UCI archive; original download time not separately recorded"},
        "input_baseline_commit": environment.get("repository_head_at_capture"),
        "repository_head_at_run": current_head(),
        "row_counts": {"raw": len(raw), "processed": len(frame), "removed": 0,
                       "train": len(train_pos), "test": len(test_pos)},
        "changes": ["Add stable row_id before any column-name normalization",
                    "Strip leading/trailing whitespace from column names only",
                    "Keep all rows, raw predictor values, zero grades and encoded unknown categories"],
        "split": {"seed": 42, "test_fraction": 0.2, "stratified": True, "cv_folds": 5,
                  "algorithm": "train_test_split; sorted training positions; StratifiedKFold",
                  "sklearn_version": metadata.version("scikit-learn"), "numpy_version": metadata.version("numpy")},
        "core_files_for_reproduction": CORE_FILES,
        "files": {name: {"sha256": digest(ROOT/name), "bytes": (ROOT/name).stat().st_size} for name in frozen},
        "not_hashed": {"data/manifest.json": "No recursive self-hash",
            "docs/handoffs/A.md": "Manual acceptance fields remain editable",
            "PROGRESS.md": "Team progress remains editable", "README.md": "Team overview remains editable",
            "results/A_data/delivery_check.json": "Written after delivery verification",
            "results/A_data/reproduction_check.json": "Written after independent reproduction"},
    }
    write_json("data/manifest.json", manifest)
    write_handoff(manifest, split_rows)
    print("=== A STAGE: GENERATED ===")
    print("PROCESSED SHAPE:", frame.shape)
    print("TRAIN / TEST:", len(train_pos), "/", len(test_pos))
    print("CV FOLD SIZES:", [int((train["cv_fold"] == i).sum()) for i in range(5)])
    print("FEATURE COUNTS: 18 / 24")
    print("FIGURES: 4 (TRAIN ONLY)")
    print("RAW ROWS REMOVED: 0")
    print("CORE OUTPUTS: data/processed/students.csv; data/splits/split_manifest.csv")
    print("MANIFEST: data/manifest.json")
    print("HANDOFF DRAFT: docs/handoffs/A.md")
    print("NEXT: python src/A_data/verify_A.py")


if __name__ == "__main__":
    main()
