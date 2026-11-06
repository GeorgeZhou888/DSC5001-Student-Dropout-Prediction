# 成员C：一次性最终评估与交付说明

本交付于2026-11-05完成，只包含成员C新增/更新文件。将本包目录合并到已含A数据与B模型的
项目根目录，保留A原始数据、配置、环境记录及B的`models/`、`results/B_validation/`。

## 已完成

- 七个冻结模型（三种分类器×两组信息条件，加共享多数类基线）在885条测试记录上的唯一一次评估。
- `test_metrics.csv`：7行×19列，含macro-F1、macro-precision/recall、accuracy、balanced accuracy、三类各自的precision/recall/F1、log_loss（树模型因精确0概率留空）及退学precision/recall。
- `test_predictions.csv`：6,195行 = 885名学生×7个模型行，长表含row_id、model、feature_set、true_label、predicted_label、三类概率与correct；概率按加载后的`pipeline.classes_`显式映射，非按列位置猜测。
- 五张图：模型对比、信息增量（18→24项哑铃图）、六张混淆矩阵、代表模型置换重要性、24变量逻辑回归退学类系数。
- 两张解释表（重要性全表、系数全表）、`run_manifest.json`（B冻结引用、逐项输入字节状态、模型与输出哈希、环境、命令、耗时）、`verification_report.json`（14项复核）。
- `findings.md`：三个研究问题的回答与局限；`docs/handoffs/C.md`：交接单。

未重新训练、未调参、未重划分、未调阈值、未按测试分数更换代表解释模型。测试集前后只用于这一次评估。

## 运行环境

Python 3.14.x（本次3.14.3）及A的原锁定依赖。从项目根目录：

```powershell
py -3.14 -m venv .venv-c
.\.venv-c\Scripts\python.exe -m pip install -r environment/requirements-lock.txt
```

下面的 `python` 指上述虚拟环境内的解释器，或把它替换为 `.\.venv-c\Scripts\python.exe`。

```powershell
python src/C_evaluation/evaluate_models.py --check
python src/C_evaluation/verify_C.py
```

`--check`只核验上游（18项）并明确不评分；`verify_C.py`只读交付并重算指标，不加载模型做预测，
不改写任何评估产物。仅当需要重跑评估时执行：

```powershell
python src/C_evaluation/evaluate_models.py --force
```

已有`results/C_final/`时默认拒绝覆盖；`--skip-figures`只重算两个CSV与run_manifest。
本次完整运行24.2秒，其他设备耗时可能不同。

## 评估口径

- 测试集来自`data/splits/split_manifest.csv`中`dataset_split == test`的885行，按row_id与
  `data/processed/students.csv`一对一连接并排序；划分数为3539/885，折0–4，测试行折号为空。
- 每个模型只读取`model_catalog.json`中记录的自己的特征列表（18项或24项）；多数类基线使用18项
  入学接口，不依赖特征值，供两组共同对照。
- 指标零分母设为0；macro平均覆盖Dropout/Enrolled/Graduate三类；基线预测恒为Graduate，其
  概率为0/1，故log_loss留空。
- 图沿用A的风格（英文标签、白底、dpi 180）。
- 置换重要性针对代表模型在测试集上计算，scoring=`f1_macro`，10次重复，seed 42。

## 主要结果

| 信息条件 | 模型 | 测试macro-F1 | 退学precision | 退学recall |
|---|---|---:|---:|---:|
| 入学 | 逻辑回归 | 0.5024 | 0.5618 | 0.5282 |
| 入学 | 决策树 | 0.4488 | 0.4739 | 0.4789 |
| 入学 | 随机森林 | 0.4907 | 0.5217 | 0.5070 |
| 入学＋第一学期 | 逻辑回归 | 0.6442 | 0.7439 | 0.6444 |
| 入学＋第一学期 | 决策树 | 0.5873 | 0.6820 | 0.6268 |
| 入学＋第一学期 | 随机森林（代表模型） | 0.6153 | 0.7099 | 0.6549 |
| 共用基线 | 多数类 | 0.2221 | 0 | 0 |

24项特征使macro-F1提升+0.125至+0.142、退学recall提升+0.116至+0.148。测试集上逻辑回归略高于
代表模型随机森林0.029，小于验证标准差，属噪声内并列，不据此改选；代表模型仍为B冻结的
`enrollment_plus_sem1__random_forest.joblib`。Enrolled类在所有模型中偏弱（F1 0.30–0.45）。

## 输入接收记录

输入字节状态取决于检出方式，两种都合法：B分支的Windows克隆重现B训练时的原始字节（7项均为
`exact`，本仓库即此情形）；LF导出的工作区仅CRLF/LF差异（`lf_equivalent`，其LF规范化摘要同时
匹配B的`input_verification.json`与A交接单记录的摘要）。`data/manifest.json`为自描述清单（不能
自哈希），其列出的26个文件摘要与当前副本一致、`row_counts`仍为3539/885。
`evaluate_models.py`只接受 exact / lf_equivalent / self_consistent_regenerated 三类状态，逐项写入
`run_manifest.json`，其余差异一律中止。A的历史验收报告不能替代对当前输入版本的人工验收。

本次为Windows AMD64、Python 3.14.3；A记录为macOS ARM64、Python 3.14.6，B为3.14.5。主次版本
3.14一致，补丁差异如实记录。19项依赖与A锁定文件一致，无包版本调整。只执行C新编写的代码，
未运行上传包中的A或B脚本，未改写任何A/B文件。

尚未声称B已确认或组长已验收。正式交接时须由组长确认所用输入快照及其字节状态。

## 提交范围

本包只改动`src/C_evaluation/`、`results/C_final/`、`docs/handoffs/C.md`及本说明文件。不调整
`configs/`、`data/`、`models/`、`results/B_validation/`、`results/A_data/`、`PROGRESS.md`。七个
joblib在评估前后哈希不变。

从组长确认的输入版本创建`member-c-evaluation`分支，提交本包文件并向main发PR。PR标题可用：
`Complete C final one-shot test evaluation with verified predictions`。PR正文可直接使用：

> 完成C阶段一次性最终评估：七个冻结模型在885条测试记录上的指标、长表预测、五张图、两张解释表
> 与run_manifest。上游18项核验与交付14项复核通过，指标可由预测CSV完全重算（差值0.00e+00）。
> 代表解释模型沿用B在测试前固定的24变量随机森林；测试上逻辑回归高0.029但属噪声内并列，未改选。
> 输入字节状态逐项记录（本仓库检出为exact，LF导出为lf_equivalent，脚本只接受这两类及自描述
> 清单重建）。未改A/B文件，PROGRESS.md仍由组长更新。

实现参考：[scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)、
[permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)。
