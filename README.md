# DSC5001 学生退学与学业结果预测

Zero bias hero 五人独立课程项目。比较 18 项入学信息与加入第一学期表现的 24 项信息，预测 Dropout、Enrolled、Graduate 三类结局。

**当前状态：A 数据、B 模型、C 评估及 D/E 报告和展示材料均已收录 main。** 现有验证记录不代表已完成教师验收或 Canvas 提交。

## 最终交付

- [主报告](docs/report/report.pdf)
- [成员贡献及复现附录源码](docs/report/code_appendix.tex)
- [参考文献](docs/report/references.bib)
- [提案](docs/proposal/proposal.pdf)
- [演讲 PPT](slides/Team-Zero%20bias%20hero-PPT.pptx)
- [A0 海报](slides/Team-Zero%20bias%20hero-Poster.pptx)
- [五人双语讲稿](slides/Team-Zero%20bias%20hero-Script.docx)
- [小组录像](recording/Student-Dropout-Prediction.mp4)

附录 tex 是供主文档引用的片段。课程需另行提交 PDF 幻灯片、MP4 录像及 Appendix PDF，截止 2026-11-28 23:59；仓库存在文件不代表已上传 Canvas。

## 主要结果

| 模型 | 入学信息 macro-F1 | 加入第一学期 macro-F1 |
|---|---:|---:|
| 逻辑回归 | 0.5024 | 0.6442 |
| 决策树 | 0.4488 | 0.5873 |
| 随机森林 | 0.4907 | 0.6153 |
| 多数类基线（共享） | 0.2221 | 0.2221 |

三种模型加入第一学期信息后提升约 0.125–0.142。24 特征逻辑回归把 246 人判为 Dropout，其中 183 人判对、63 人误报，另有 101 名实际 Dropout 漏判；precision 为 74.4%，recall 为 64.4%。

逻辑回归在本次测试的 macro-F1 最高；解释模型仍是测试前按交叉验证选定的随机森林。未检验模型差异的统计显著性。单机构、单次留出的回顾性结果不证明实时预警、因果效应或干预成效。

[test_metrics.csv](results/C_final/test_metrics.csv) 有 7 行模型汇总；[test_predictions.csv](results/C_final/test_predictions.csv) 有 6,195 行（885 学生 × 7 模型）。

## 实验设计与复现

4,424 条记录，分层 80/20、seed 42，训练 3,539、测试 885。训练内五折分层交叉验证按 macro-F1 调参；预处理在训练折内部拟合。排除第二学期和时点不明确的字段。模型冻结后统一测试，不依据测试结果反向调参。

从仓库根目录运行：

```bash
python -m pip install -r environment/requirements-lock.txt
python src/A_data/verify_A.py
python -m unittest discover -s src/B_models -p test_b_models.py -v
python src/B_models/verify_B.py --recheck-cv
python src/C_evaluation/evaluate_models.py --check
python src/C_evaluation/verify_C.py
```

--check 只核对输入。完整评分命令为 python src/C_evaluation/evaluate_models.py；结果已存在时默认拒绝覆盖。日常修改报告无需强制重跑。完整重建命令见 [A交接单](docs/handoffs/A.md)、[B交接单](docs/handoffs/B.md)、[C交接单](docs/handoffs/C.md)。

A/B/C 分别记录 Python 3.14.6（macOS ARM64）、3.14.5（Windows AMD64）、3.14.3（Windows AMD64）；不是完全相同的运行环境。依赖及环境记录位于 environment/ 和各阶段运行记录。

## 成员与目录

| 成员 | 工作 | 文件位置 |
|---|---|---|
| A Ruxuan ZHOU | 数据、固定划分与探索 | src/A_data/、results/A_data/ |
| B Mingxi ZHU | 训练、调参与模型冻结 | src/B_models/、models/、results/B_validation/ |
| C Boyi WEI | 测试评估与解释 | src/C_evaluation/、results/C_final/ |
| D Mengqi YAN | 报告与参考文献 | docs/report/ |
| E Yushu WU | 复核、PPT、讲稿、录像和海报 | slides/、recording/ |

PROGRESS.md 保留历史协作记录，历史安排不代表最新交付状态。数据及实验口径见 configs/、data/manifest.json 和 data/splits/split_manifest.csv。

## 数据来源与许可

Realinho 等，Predict Students' Dropout and Academic Success，UCI Machine Learning Repository，4,424 条记录、36 个原始预测变量及一个结局标签。永久链接：https://doi.org/10.24432/C5MC89 。[数据说明](data/raw/README.md)。

数据采用 CC BY 4.0；该许可不自动适用于项目代码。当前仓库未单独声明项目代码许可证。

