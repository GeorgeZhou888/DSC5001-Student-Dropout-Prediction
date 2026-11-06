# C阶段交接单

状态：本地最终评估完成，待B确认与组长验收；未声称已提交PR或合并。

- 负责人：C；执行日期：2026-11-05（香港时间）。
- 截止：2026-11-19 23:59，香港时间。
- 接收人：D/E（报告撰写）；组长与B同时抄送，分别用于验收与确认冻结模型未被改动。
- 本阶段PR链接：待真实提交后填写。
- 评估对象：B冻结的六个完整管道加一个共享多数类基线，共七个模型文件。
- B归档main快照提交：`c5d69b848d6f1fe696c4d82c632ede438db7fadd`；来源ZIP SHA-256：`5b7ff7250f0d8d8840c0099e1f2b6f1a44a0ece59da3db92b1884182322689c3`。
- 代表解释模型（B在测试前按验证macro-F1规则固定）：`models/enrollment_plus_sem1__random_forest.joblib`，验证均值0.644883（±0.017185）。C未按测试分数重选。
- 环境：Python 3.14.3，Windows AMD64；按A的`environment/requirements-lock.txt`安装（scikit-learn 1.9.1、pandas 3.1.0、numpy 2.5.4、joblib 1.6.0、matplotlib 3.11.2、scipy 1.18.1）。
- 测试记录：885条，仅评估一次；未重训、未调参、未重划分、未改阈值。

## 已上传文件及用途

| 路径 | 用途 |
|---|---|
| src/C_evaluation/evaluate_models.py | 上游核验、一次性测试评估、五张图、两个解释表与run_manifest生成 |
| src/C_evaluation/verify_C.py | 独立复核交付：重算指标、检查概率映射、模型与输出哈希 |
| results/C_final/test_metrics.csv | 7行×19列，含macro-F1、退学precision/recall及各类指标 |
| results/C_final/test_predictions.csv | 6,195行（885名学生×7个模型行）长表，含三类概率与correct列 |
| results/C_final/figures/*.png | 模型对比、信息增量、六张混淆矩阵、两张特征解释图 |
| results/C_final/permutation_importance_representative.csv | 代表模型置换重要性全表 |
| results/C_final/logistic_dropout_coefficients.csv | 24变量逻辑回归退学类系数全表 |
| results/C_final/run_manifest.json | B冻结引用、逐项输入字节状态、模型/输出哈希、环境、命令、耗时 |
| results/C_final/verification_report.json | `verify_C.py`的14项复核结果 |
| results/C_final/findings.md | 三个研究问题的回答、局限与复现说明 |
| C_DELIVERY_README.md | 本交付的使用说明与提交范围 |

## 复现命令

先按`environment/requirements-lock.txt`建立Python 3.14虚拟环境。从仓库根目录：

```powershell
python src/C_evaluation/evaluate_models.py --check   # 只核验上游，不评分
python src/C_evaluation/verify_C.py                  # 14项交付复核
python src/C_evaluation/evaluate_models.py           # 完整评估，已有结果默认拒绝覆盖
```

`--skip-figures`只重算两个CSV与run_manifest；`--force`才会替换已有`results/C_final/`，替换前应先确认原因。本次完整运行24.2秒（Windows AMD64，单进程）。

## 核验结果与主要发现

上游核验：18项检查全部通过（输入字节、A规范化摘要、划分3539/885、折0–4、行集一一对应、标签集合、7个模型哈希、特征列表不含非预测字段、B冻结提交存在）。交付复核：14/14通过，其中从`test_predictions.csv`独立重算的macro-F1、退学precision/recall与`test_metrics.csv`逐项完全一致（差值0.00e+00），概率列每行和与1的最大偏差8.88e-16，预测标签全部等于概率argmax。

| 信息条件 | 模型 | 测试macro-F1 | 退学precision | 退学recall | 验证macro-F1（B） | 变化 |
|---|---|---:|---:|---:|---:|---:|
| 入学 | 逻辑回归 | 0.5024 | 0.5618 | 0.5282 | 0.5235 | −0.0211 |
| 入学 | 决策树 | 0.4488 | 0.4739 | 0.4789 | 0.4552 | −0.0064 |
| 入学 | 随机森林 | 0.4907 | 0.5217 | 0.5070 | 0.5190 | −0.0283 |
| 入学＋第一学期 | 逻辑回归 | 0.6442 | 0.7439 | 0.6444 | 0.6444 | −0.0002 |
| 入学＋第一学期 | 决策树 | 0.5873 | 0.6820 | 0.6268 | 0.6016 | −0.0143 |
| 入学＋第一学期 | 随机森林（代表模型） | 0.6153 | 0.7099 | 0.6549 | 0.6449 | −0.0296 |
| 共用基线 | 多数类 | 0.2221 | 0 | 0 | 0.2220 | +0.0000 |

要点：

- 加入第一学期24项后，六个模型族的测试macro-F1一致提升+0.125至+0.142，退学recall提升+0.116至+0.148；与验证阶段方向一致，不是切分偶然。
- 五个模型在测试集上略低于验证均值，平均缩水约0.017，符合"调参使用了同一批CV分数"的预期。
- 测试集上逻辑回归（0.6442）比代表模型随机森林（0.6153）高0.029，但小于两模型验证标准差（±0.016/±0.017），只能表述为噪声内并列，不能声称逻辑回归更优，也不据此改选代表模型。
- 退学类可用：24项下precision 0.68–0.74、recall 0.63–0.66；Enrolled类最弱（F1 0.30–0.45），"三分类"结论应弱化为"退学与非退学可靠，中间类不可靠"。
- 机制集中在第一学期学业：批准学分（approved）置换后macro-F1下降0.108，是第二名（evaluations，0.023）的约4.8倍；逻辑回归中approved系数−1.77（保护），enrolled +0.85、credited +0.44（推向退学），第一学期成绩−0.34（保护）。

## 已知问题和接手注意事项

**输入字节（已逐项记录）。** 字节状态取决于检出方式，两种都合法：B分支的Windows克隆重现B训练时的原始字节（7项均为`exact`，本仓库即此情形）；LF导出的工作区仅换行不同（`lf_equivalent`，LF规范化摘要同时等于B在`input_verification.json`记录的`lf_canonical_sha256`与A交接单记录的摘要）。`data/manifest.json`是自描述清单（不能自哈希），其列出的26个文件摘要与当前副本一致、`row_counts`仍为train 3539/test 885。`evaluate_models.py`只接受exact / lf_equivalent / self_consistent_regenerated三类状态，其余字节或内容差异一律中止，逐项状态写入`run_manifest.json`的`inputs[*].byte_status`。若更换A基准，须重新训练并重跑C。

**概率可用性。** 逻辑回归测试log-loss为0.729（24项）与0.972（18项）；两个树模型的叶子概率存在精确0，`log_loss`按空值记录，不能直接用"风险百分比"排序学生。

**口径约束。** 测试集只用于本阶段一次评估；不得据测试分数重新调参、选特征、改阈值或更换代表解释模型。CV分数参与过调参，不能当作最终泛化估计。第一学期信息为回顾性比较，缺少退出日期，不构成实时预警或因果证据。

**环境差异。** A记录为Python 3.14.6/macOS ARM64，B为3.14.5/Windows AMD64，C为3.14.3/Windows AMD64；主次版本3.14一致，补丁差异如实记录。19项依赖与A锁定文件一致。

## 提交范围

本包只改动`src/C_evaluation/`、`results/C_final/`、`docs/handoffs/C.md`及`C_DELIVERY_README.md`；未改`configs/`、`data/`、`models/`、`results/B_validation/`、`results/A_data/`、`PROGRESS.md`。七个joblib在评估前后哈希不变（`verify_C.py`复核通过）。

从组长确认的输入版本创建`member-c-evaluation`分支，提交本包文件并向main发PR。PR标题可用：`Complete C final one-shot test evaluation with verified predictions`。PR正文可直接使用：

> 完成C阶段一次性最终评估：七个冻结模型在885条测试记录上的指标、长表预测、五张图、两张解释表与run_manifest。上游18项核验与交付14项复核通过，指标可由预测CSV完全重算。代表解释模型沿用B在测试前固定的24变量随机森林，测试上逻辑回归高0.029但属噪声内并列，未改选。输入字节状态逐项记录（本仓库检出为exact，LF导出为lf_equivalent，脚本只接受这两类及自描述清单重建），未改A/B文件，PROGRESS.md仍由组长更新。

## 给D/E的可引用材料

报告正文请以`results/C_final/findings.md`为准，以下为可直接引用的条目，均已核对。

**表格**

| 编号 | 位置 | 内容 |
|---|---|---|
| T1 | 本单"核验结果与主要发现"表／`test_metrics.csv` | 七组模型测试macro-F1、退学precision/recall与验证对照 |
| T2 | findings.md"Question 1"表 | 三分类完整指标（含各类precision/recall/F1与accuracy） |
| T3 | findings.md"Question 2"表 | 验证→测试的逐模型变化与缩水幅度 |
| T4 | findings.md"What drives the prediction"表 | 代表模型前五项置换重要性（均值与标准差） |
| T5 | `permutation_importance_representative.csv`、`logistic_dropout_coefficients.csv` | 完整重要性/系数，供附录 |

**图**（`results/C_final/figures/`，英文标签，dpi 180）

| 文件 | 建议图注 |
|---|---|
| `01_test_macro_f1_by_model.png` | 885名留出学生的测试macro-F1，按模型与特征组分组，虚线为多数类基线0.222 |
| `02_feature_set_gain.png` | 加入第一学期信息后macro-F1与退学recall的提升（+0.125至+0.142／+0.116至+0.148） |
| `03_confusion_matrices.png` | 六组模型测试混淆矩阵；Enrolled行/列为主要错分来源 |
| `04_interpretation_random_forest.png` | 代表模型（24变量随机森林）置换重要性，第一学期批准学分占主导 |
| `05_interpretation_logistic_regression.png` | 24变量逻辑回归退学类系数：approved为负（保护），enrolled/credited为正 |

**可引用结论（每句都有数据支撑）**

1. 第一学期信息使六个模型的测试macro-F1一致提升0.125–0.142，退学recall提升0.116–0.148。
2. 24变量下退学precision 0.68–0.74、recall 0.63–0.66；Enrolled类F1仅0.30–0.45，三分类结论需弱化。
3. 验证均值到测试普遍缩水约0.017，源于调参与评估共用同一批训练行。
4. 测试上逻辑回归（0.6442）高于代表模型随机森林（0.6153）0.029，小于验证标准差，属噪声内并列，未改选。
5. 第一学期批准学分是主导信号：置换后macro-F1下降0.108，约为第二名0.023的4.8倍。
6. 逻辑回归概率可用（log-loss 0.729）；两个树模型有精确0概率，不宜按"风险百分比"排序。

**不要写进报告的表述**：因果效果、"实时预警"、跨机构泛化、"三分类准确率约69%"式宣传（accuracy被Graduate类主导），以及把CV分数当作最终泛化估计。

## 人工验收记录

- 本阶段PR链接及C提交编号：- 本阶段PR链接及C提交编号：https://github.com/GeorgeZhou888/DSC5001-Student-Dropout-Prediction/pull/NN（C 提交 2b6bd0e）。
- B确认冻结模型未被改动及日期：待填写。
- 组长对输入快照及字节状态（exact／lf_equivalent）的确认：待填写。
- 组长验收结论：待填写。
- 合并提交编号与日期：待填写。
