"""A stage: document fields, validate source-defined codes, and define features.

Run from the repository root: python src/A_data/define_features.py
The raw CSV is read without modification. No model is fitted or data split.
"""

from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import csv
import hashlib
import io
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
METADATA_URL = "https://archive.ics.uci.edu/api/dataset?id=697"
REFERENCE_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"

ENROLLMENT = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification",
    "Previous qualification (grade)", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation", "Admission grade",
    "Displaced", "Educational special needs", "Gender",
    "Age at enrollment", "International",
]
SEMESTER_SUFFIXES = [
    "credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"
]
SEM1 = [f"Curricular units 1st sem ({name})" for name in SEMESTER_SUFFIXES]
SEM2 = [f"Curricular units 2nd sem ({name})" for name in SEMESTER_SUFFIXES]
FINANCIAL = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
MACRO = ["Unemployment rate", "Inflation rate", "GDP"]
BASE_NUMERIC = [
    "Application order", "Previous qualification (grade)",
    "Admission grade", "Age at enrollment",
]
CATEGORICAL = [name for name in ENROLLMENT if name not in BASE_NUMERIC]

MEANINGS = {
    "Marital status": "婚姻状况类别",
    "Application mode": "申请入学途径类别",
    "Application order": "申请志愿顺序编码；按有序数值使用",
    "Course": "就读专业或课程项目类别",
    "Daytime/evening attendance": "日间或夜间就读：1为日间，0为夜间",
    "Previous qualification": "既往教育经历或学历类别",
    "Previous qualification (grade)": "既往学历对应成绩，0至200分量表",
    "Nacionality": "国籍类别；保留原始拼写",
    "Mother's qualification": "母亲教育背景类别",
    "Father's qualification": "父亲教育背景类别",
    "Mother's occupation": "母亲职业类别",
    "Father's occupation": "父亲职业类别",
    "Admission grade": "入学录取成绩，0至200分量表",
    "Displaced": "是否离开原居住地就读的状态编码",
    "Educational special needs": "是否登记特殊教育需要",
    "Gender": "原始性别编码：1为男性，0为女性",
    "Age at enrollment": "入学年龄，单位为岁",
    "International": "是否国际学生",
    "Debtor": "是否处于欠费状态",
    "Tuition fees up to date": "学费是否按时缴清",
    "Scholarship holder": "是否持有奖学金或助学资助",
    "Unemployment rate": "失业率，单位为百分比",
    "Inflation rate": "通胀率，单位为百分比；负值可能有效",
    "GDP": "国内生产总值相关宏观指标；源页未充分说明量纲",
    "Target": "课程正常学制结束时记录的Dropout、Enrolled、Graduate结局",
}
SEM_MEANINGS = {
    "credited": "获认可或转入的课程单元数，不是学分总分值",
    "enrolled": "注册的课程单元数",
    "evaluations": "课程考核次数，可能超过注册课程数",
    "approved": "通过的课程单元数",
    "grade": "平均成绩，0至20分量表；保留0值",
    "without evaluations": "没有考核记录的课程单元数",
}
for term, prefix in [("1st", "第一学期"), ("2nd", "第二学期")]:
    for suffix, meaning in SEM_MEANINGS.items():
        MEANINGS[f"Curricular units {term} sem ({suffix})"] = prefix + meaning


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metadata_bytes():
    cache = ROOT / "data/raw/uci_metadata.json"
    if cache.exists():
        return cache.read_bytes()
    print("Fetching official UCI field documentation ...", flush=True)
    try:
        request = Request(METADATA_URL, headers={"User-Agent": "DSC5001-course-project/1.0"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        document = json.loads(payload)
        if document.get("status") != 200 or len(document["data"]["variables"]) != 37:
            raise ValueError("Unexpected UCI metadata structure")
    except Exception as exc:
        raise SystemExit(f"Could not obtain official metadata: {exc}. Share this error for assistance.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return payload


def main():
    raw_path = ROOT / "data/raw/data.csv"
    if not raw_path.is_file():
        raise SystemExit("Missing data/raw/data.csv. Place this script in src/A_data/.")
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if raw_sha256 != REFERENCE_SHA256:
        raise SystemExit("Raw file differs from the verified course baseline; investigate before continuing.")

    frame = pd.read_csv(io.BytesIO(raw_bytes), sep=";")
    rename_map = {name: name.strip() for name in frame.columns if name != name.strip()}
    frame = frame.rename(columns=lambda name: name.strip())
    if not frame.columns.is_unique:
        raise SystemExit("Column-name trimming produced a collision.")
    expected = ENROLLMENT + FINANCIAL + MACRO + SEM1 + SEM2 + ["Target"]
    if frame.shape != (4424, 37) or set(frame.columns) != set(expected):
        raise SystemExit("Unexpected columns or row count; investigate before continuing.")

    payload = metadata_bytes()
    metadata = json.loads(payload)["data"]["variables"]
    lookup = {row["name"].strip().casefold(): row for row in metadata}
    selected = ENROLLMENT + SEM1
    excluded = {
        name: "状态的具体采集或更新时间未逐字段确认；按保守预测时点口径排除"
        for name in FINANCIAL
    }
    excluded.update({
        name: "宏观指标的匹配年份、发布版本及入学时可用性未充分确认；保守排除"
        for name in MACRO
    })
    excluded.update({name: "第二学期信息超出本项目两组特征的范围" for name in SEM2})

    dictionary = []
    failures = {}
    for raw_name in pd.read_csv(io.BytesIO(raw_bytes), sep=";", nrows=0).columns:
        name = raw_name.strip()
        source = lookup.get(name.casefold())
        if source is None:
            raise SystemExit(f"No official documentation found for: {name}")
        series = frame[name]
        notes = []
        if name in CATEGORICAL + FINANCIAL:
            kind = "categorical"
            codes = sorted({int(v) for v in re.findall(
                r"(?<![\d.])(\d+)\s*[-–]\s+", source.get("description") or ""
            )})
            if not codes:
                raise SystemExit(f"Could not read official category codes for: {name}")
            invalid = ~series.isin(codes)
            rule = "官方类别编码集合：" + ",".join(map(str, codes))
            notes.append("按类别处理；B在每个训练折内拟合独热编码，不把代码大小当作数量关系")
        elif name == "Target":
            kind = "target"
            invalid = ~series.isin(["Dropout", "Enrolled", "Graduate"])
            rule = "Dropout / Enrolled / Graduate"
            notes.append("保持三分类；Enrolled不等同于成功；标签不能用作预测特征")
        else:
            kind = "numeric"
            values = series.to_numpy(dtype=float)
            invalid = ~np.isfinite(values)
            rule = "有限数值；源页未提供进一步的明确取值边界"
            if name == "Application order":
                invalid |= (values < 0) | (values > 9) | (values != np.floor(values))
                rule = "0至9之间的整数；按当前UCI字典保留0"
            elif name in ["Previous qualification (grade)", "Admission grade"]:
                invalid |= (values < 0) | (values > 200)
                rule = "0至200之间的有限数值"
            elif name == "Age at enrollment":
                invalid |= (values <= 0) | (values != np.floor(values))
                rule = "正整数；未设置从全体样本估计的年龄截断阈值"
            elif name in SEM1 + SEM2:
                if name.endswith("(grade)"):
                    invalid |= (values < 0) | (values > 20)
                    rule = "0至20之间的有限数值；0不自动视为缺失"
                    notes.append("仅凭此列不能区分0值的具体学业原因，保留原值")
                else:
                    invalid |= (values < 0) | (values != np.floor(values))
                    rule = "非负整数；不要求考核次数小于或等于注册课程数"

        invalid_count = int(np.asarray(invalid).sum())
        if invalid_count:
            failures[name] = invalid_count
        if raw_name != name:
            notes.append("整理表将去除列名首尾空白；原始文件保持原样")
        if name in ["Mother's qualification", "Father's qualification"]:
            notes.append("34为未知，保留为明确类别")
        if name in ["Mother's occupation", "Father's occupation"]:
            notes.append("99为空白编码，保留为明确类别；0为学生职业类别")
        if name == "Curricular units 2nd sem (without evaluations)":
            notes.append("源字典描述误写第一学期；本项目按字段名解释为第二学期并排除")
        if name in ENROLLMENT:
            availability = "入学背景；按官方总体说明归类，无逐字段时间戳"
        elif name in SEM1:
            availability = "第一学期末；无学生退出事件的具体日期"
        elif name in SEM2:
            availability = "第二学期末；不用于本项目预测输入"
        elif name == "Target":
            availability = "课程正常学制结束时记录的结局"
        else:
            availability = "具体测量或可获得时点未充分确认"
        dictionary.append({
            "raw_column": raw_name, "column": name,
            "description_cn": MEANINGS[name], "semantic_type": kind,
            "source_storage_type": source["type"], "availability": availability,
            "in_enrollment": name in ENROLLMENT,
            "in_enrollment_plus_sem1": name in selected,
            "exclusion_reason": excluded.get(name, "标签，不是特征" if name == "Target" else ""),
            "validity_rule": rule, "invalid_value_count": invalid_count,
            "notes": "；".join(notes), "source_url": SOURCE_URL,
        })

    report = {
        "stage": "field_codes_ranges_and_feature_definition",
        "status": "PASS" if not failures else "REVIEW_REQUIRED",
        "recorded_at_hk": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds"),
        "raw_sha256": raw_sha256,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "invalid_values_by_column": failures,
        "encoded_unknown_or_blank_counts": {
            name: {str(code): int((frame[name] == code).sum())}
            for name, code in [
                ("Mother's qualification", 34), ("Father's qualification", 34),
                ("Mother's occupation", 99), ("Father's occupation", 99),
            ]
        },
        "notes": [
            "No empty cells does not mean all fields have known substantive values.",
            "Source-based code/range checks do not rule out every possible data error.",
            "Feature groups follow meanings and timing, not observed model performance.",
        ],
    }
    write_json(ROOT / "results/A_data/field_validation.json", report)
    if failures:
        raise SystemExit("Field checks need review; see results/A_data/field_validation.json.")

    config = {
        "schema_version": 1, "status": "defined_pending_A_validation",
        "column_name_policy": "strip_leading_trailing_whitespace_only",
        "rename_map": rename_map,
        "enrollment": ENROLLMENT, "enrollment_plus_sem1": selected,
        "categorical_features": CATEGORICAL,
        "numeric_features": BASE_NUMERIC + SEM1,
        "excluded_fields_with_reasons": excluded,
        "non_predictor_fields": ["Target", "row_id", "dataset_split", "cv_fold"],
        "source_url": SOURCE_URL, "source_metadata_url": METADATA_URL,
        "source_metadata_file": "data/raw/uci_metadata.json",
        "source_metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "notes": [
            "Enrollment timing is a source-based project assessment, not an individual timestamp guarantee.",
            "Excluding financial and macro variables is conservative; future leakage has not been proven.",
            "Categorical encoding and learned preprocessing must be fitted within each training fold.",
            "Keep encoded unknown/blank categories and zero grades without automatic imputation.",
            "Semester comparison is retrospective; exact dropout dates are not available.",
        ],
    }
    write_json(ROOT / "configs/feature_sets.json", config)
    dictionary_path = ROOT / "results/A_data/data_dictionary.csv"
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    with dictionary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionary[0]))
        writer.writeheader()
        writer.writerows(dictionary)

    print("FIELD CHECK: PASS")
    print("DICTIONARY ROWS:", len(dictionary))
    print("ENROLLMENT FEATURES:", len(ENROLLMENT))
    print("ENROLLMENT PLUS SEM1 FEATURES:", len(selected))
    print("CATEGORICAL FEATURES:", len(CATEGORICAL))
    print("NUMERIC FEATURES (LARGER SET):", len(BASE_NUMERIC + SEM1))
    print("EXCLUDED RAW PREDICTORS:", len(excluded))
    print("INVALID VALUES:", sum(failures.values()))
    print("ENCODED UNKNOWN/BLANK:", report["encoded_unknown_or_blank_counts"])
    print("FEATURE CONFIG STATUS:", config["status"])
    print("CREATED: data/raw/uci_metadata.json")
    print("CREATED: configs/feature_sets.json")
    print("CREATED: results/A_data/data_dictionary.csv")
    print("CREATED: results/A_data/field_validation.json")


if __name__ == "__main__":
    main()
