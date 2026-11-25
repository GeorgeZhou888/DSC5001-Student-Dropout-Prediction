# 成员B：模型训练、验证与本地冻结

本交付于2026-11-04完成，只包含成员B新增/更新文件。将本包目录合并到用户上传ZIP解压后的项目根目录，保留A原始数据、配置和环境记录。

## 已完成

- 3种模型×2组信息条件，共52个候选参数配置、260次固定折训练；另有5次多数类基线验证。
- `cv_metrics.csv`：6组最终参数及共用基线的35条逐折验证记录。
- `cv_all_candidates.csv`：所有265条候选/基线逐折记录，含is_selected。
- `cv_search_results.csv`：52个候选的均值、总体标准差（ddof=0）与参数。
- `cv_summary.csv`：7组汇总。所有指标均来自固定训练集交叉验证，不是最终测试结果。
- 六个完整预处理/分类器管道及一个基线，均在3,539条训练记录上重拟合并冻结字节哈希。
- 八项针对性测试通过；交付检查16/16通过；独立重拟合并重算35个选定验证折，指标与记录逐项一致（绝对容差1e-12）。
- 885条测试记录在建立模型数据表前被丢弃，未生成测试预测、测试指标或测试图。

## 运行环境

Python 3.14.x（本次3.14.5）及A的原锁定依赖。从项目根目录：

```powershell
py -3.14 -m venv .venv-b
.\.venv-b\Scripts\python.exe -m pip install -r environment/requirements-lock.txt
```

下面的 `python` 指上述虚拟环境内的解释器，或把它替换为 `.\.venv-b\Scripts\python.exe`。

```powershell
python -m unittest discover -s src/B_models -p test_b_models.py -v
python src/B_models/verify_B.py --recheck-cv
```

`verify_B.py`会写B检查报告；`--recheck-cv`会重新训练选定参数的35个验证折模型，但不改写冻结模型、不接触测试集。

仅当需要重新训练时执行：

```powershell
python src/B_models/train_models.py --allow-lf-equivalent --source-commit c5d69b848d6f1fe696c4d82c632ede438db7fadd --source-zip-sha256 5b7ff7250f0d8d8840c0099e1f2b6f1a44a0ece59da3db92b1884182322689c3 --jobs 2 --overwrite
```

上面提交编号与ZIP摘要专指本次原始输入。更换输入后须改成真实新版本，不能继续复用。首次空白B目录运行时可去掉`--overwrite`；已有交付时默认拒绝覆盖，显式加该参数才会重建B模型和结果。完整调参本次约64秒，其他设备耗时可能不同。

## 预处理与调参

两组分别按A特征列表选列：18项入学与24项入学加第一学期。14项类别字段折内独热编码，未知类别忽略为全零；数值字段仅逻辑回归进行折内标准化，树模型保留原量纲。零值与未知编码按A规则保留，不做全局编码、缩放或自动填补。

- 逻辑回归：lbfgs、max_iter=5000，C为0.1/1/10，class_weight为None/balanced，6候选。
- 决策树：max_depth为3/6/None，min_samples_leaf为5/20，class_weight为None/balanced，12候选。
- 随机森林：200棵树、单模型n_jobs=1，max_depth为12/None，min_samples_leaf为1/5，class_weight为None/balanced，8候选。
- 所有随机种子42；交叉验证折直接来自A的cv_fold，没有再次生成。搜索并发2，原生数值线程限制1。
- 以五折macro-F1均值选参；完全同分时选固定ParameterGrid顺序中的首项。三类顺序Dropout/Enrolled/Graduate，指标零分母设为0。

## 验证结果

| 信息条件 | 模型 | 验证 macro-F1（均值±标准差） | 退学 precision | 退学 recall |
|---|---|---:|---:|---:|
| 入学 | 逻辑回归 | 0.523516 ± 0.010023 | 0.560193 | 0.525041 |
| 入学 | 决策树 | 0.455222 ± 0.034207 | 0.506261 | 0.478453 |
| 入学 | 随机森林 | 0.518996 ± 0.008463 | 0.568812 | 0.537387 |
| 入学＋第一学期 | 逻辑回归 | 0.644385 ± 0.015902 | 0.741662 | 0.643809 |
| 入学＋第一学期 | 决策树 | 0.601586 ± 0.014089 | 0.728717 | 0.674550 |
| 入学＋第一学期 | 随机森林 | 0.644883 ± 0.017185 | 0.730113 | 0.697419 |
| 共用基线 | 多数类 | 0.222013 ± 0.000187 | 0 | 0 |

入学组逻辑回归的验证均值最高；入学加第一学期组随机森林略高于逻辑回归，仅相差约0.000498，不能据此宣称显著优势。按事先在代码中固定的“六组中验证macro-F1最高者”规则，C的代表解释模型为 `enrollment_plus_sem1__random_forest.joblib`。C仍应评价全部六组和基线，不能按测试分数重新挑参数或解释对象。

调参使用了这些CV分数，因此最优CV均值不等于无偏的最终泛化效果。两组差异只是回顾性信息比较，不是因果或实时预警证据。

## 输入接收记录

使用用户指定的原始上传 ZIP。A 的26项文件中7项原字节匹配，19项仅CRLF/LF差异；这19项规范化摘要均匹配A记录，训练程序没有重写任何A文件。开启该例外的实际参数为 `--allow-lf-equivalent`，逐项证据在 `input_verification.json`。没有跳过其他内容差异检查。A 的历史验收报告不能替代此次输入版本的人工验收。

本次为Windows AMD64、Python 3.14.5；A记录为macOS ARM64、Python 3.14.6。README要求相同Python主次版本，本次满足3.14；补丁差异如实记录。19项第三方依赖版本全部匹配A锁定文件，无包版本调整。只执行新编写的B代码及测试，没有运行上传包中的A脚本。

尚未声称A已正式合并、B/C已确认或组长已验收。正式交接时须由组长确认所用ZIP快照及换行例外；如更换A基准，应按新输入重新训练并更新全部B记录。

## 交接给C

`model_catalog.json`列出每个模型的路径、SHA-256、大小、输入列、训练行数及classes。七个文件均小于20MB；最大约7.07MB。`selected_params.json`给出调参结果，`run_manifest.json`绑定数据、协议、特征、划分、环境、训练代码及输出摘要。

```python
import json
from pathlib import Path
import joblib
import pandas as pd

root = Path('.')  # 仓库根目录
catalog = json.loads((root / 'results/B_validation/model_catalog.json').read_text(encoding='utf-8'))
record = next(r for r in catalog['models']
              if r['feature_set'] == 'enrollment_plus_sem1'
              and r['model'] == 'random_forest')
pipeline = joblib.load(root / record['path'])
# C按row_id将A整理表与划分一对一连接，得到自己的test_frame后：
X = test_frame.loc[:, record['features']]
predicted_labels = pipeline.predict(X)
probabilities = pd.DataFrame(pipeline.predict_proba(X),
                             columns=pipeline.classes_, index=test_frame.index)
# probabilities['Dropout'] 为退学概率，不能按猜测的位置取列。
```

该示例未在B阶段运行测试集。只加载本交付中来源及哈希已核对的joblib文件。多数类基线使用catalog中记录的18项入学特征接口，其预测不依赖特征值，供两种信息条件共同对照。

## 提交范围

本包只改动 `src/B_models/`、`results/B_validation/`、B模型文件、`docs/handoffs/B.md`及B交付使用说明。四个局部`.gitattributes`仅保护B脚本/结果/模型/B交接单的字节，避免Git换行转换再次破坏B校验；不调整A/C文件策略。

从组长确认的输入版本创建 `member-b-models` 分支，提交本包文件并向main发PR。PR标题可用：`Complete B fixed-fold model training and frozen pipelines`。PR正文可直接使用：

> 完成三种分类器在18/24项特征上的固定五折调参、训练集重拟合与七个完整管道交付。保留35条选定逐折指标、265条完整候选/基线记录、52个候选汇总、参数、环境、输入及模型哈希。测试集未用于训练、调参或评估。八项测试、16项交付检查及35个选定折独立重算通过。输入使用指定ZIP；19项仅换行差异已验证并显式记录，待组长确认A基准及该例外；C接收与正式合并待填写。

没有自动提交PR、推送远端或填写虚假的人工验收。`PROGRESS.md`仍由组长更新。

实现参考：[scikit-learn Pipeline与组合估计器](https://scikit-learn.org/stable/modules/compose.html)、[PredefinedSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.PredefinedSplit.html)。
