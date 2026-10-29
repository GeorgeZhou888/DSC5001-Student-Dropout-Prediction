# DSC5001 学生退学与学业结果预测

本仓库用于五人小组的独立课程项目。比较“入学信息”和“入学信息＋第一学期表现”，以逻辑回归、决策树、随机森林预测 Dropout、Enrolled、Graduate 三类结局。组长负责A阶段及全流程验收。

**当前是仓库初始框架，尚未完成数据整理或实验。所有时间为香港时间（Asia/Hong_Kong）。**

## 1 工作流程和截止时间

按2026年10月30日启动安排内部截止：A完成并验收 → B训练并冻结模型 → C最终测试 → D/E整理文章与展示。截止时须提交交付物和合并请求；接手前由组长验收、合并。

| 成员 | 内部截止 | 必须上传的交付物与位置 |
|---|---|---|
| **A 数据与基准** | **2026-11-05 23:59** | 数据处理代码 `src/A_data/prepare_data.py`；原始数据 `data/raw/data.csv`；未缩放的整理数据 `data/processed/students.csv`；固定划分 `data/splits/split_manifest.csv`；特征清单 `configs/feature_sets.json`；来源与文件校验值 `data/manifest.json`；审计、数据字典和训练集描述图 `results/A_data/`；环境版本 `environment/`；交接单 `docs/handoffs/A.md`。 |
| **B 模型训练** | **2026-11-12 23:59** | 训练代码 `src/B_models/train_models.py`；五折验证表 `results/B_validation/cv_metrics.csv`；最终参数 `results/B_validation/selected_params.json`；运行记录 `results/B_validation/run_manifest.json`；拟合后的完整模型管道 `models/`；交接单 `docs/handoffs/B.md`。 |
| **C 最终评估** | **2026-11-19 23:59** | 评估代码 `src/C_evaluation/evaluate_models.py`；统一测试表 `results/C_final/test_metrics.csv`；逐样本预测 `results/C_final/test_predictions.csv`；结果图 `results/C_final/figures/`；结果解读及局限 `results/C_final/findings.md`；运行记录 `results/C_final/run_manifest.json`；交接单 `docs/handoffs/C.md`。 |
| D 文章主笔 | 2026-11-26 23:59 | 主报告、参考文献及代码附录上传 `docs/report/`；已填姓名的提案上传 `docs/proposal/`。 |
| E 写作与展示 | 2026-11-26 23:59 | 配合结果与讨论撰写，制作幻灯片和A0单页海报；文稿放 `docs/report/`，展示文件放 `docs/presentation/`。 |

11月27日由组长完成复现与提交检查，11月28日23:59前完成课程提交。上述ABC日期为组内排期，可由组长统一调整；调整须同步本表、`PROGRESS.md` 和 `configs/protocol.json`。

## 2 全组统一的实验基准

- `configs/protocol.json` 是共同实验口径；当前建议随机种子 **42**，按标签分层留出 **20%** 测试集，其余 **80%** 使用固定的五折分层交叉验证。
- A先分配稳定的 `row_id`，再形成数据和划分；`split_manifest.csv` 保存 `row_id,dataset_split,cv_fold`。训练行的折号为0至4，测试行折号留空。B/C读取该文件，**不得重新划分**。
- A提交两个特征名列表。第二组必须包含第一组，再加入第一学期学业信息；不使用第二学期字段，也不使用测量时点无法核实的状态变量。`Target`、`row_id` 和划分标记不得作为特征。
- A交付保留原始量纲的数据。类别编码、缩放、填补等需要从数据学习的处理，必须由B在每一训练折内部拟合，保存到完整模型管道；不得先用全体样本或整个80%训练集拟合后再做交叉验证。
- 六组核心实验为三种模型×两组特征，另加多数类基线。依据验证集的 **宏平均F1分数**（三类各自精确率与召回率调和平均值的等权平均）选参数；同时报告退学类精确率与召回率。保留Enrolled独立类别。
- B冻结模型、参数和运行环境后，C才统一测试。测试集不得用于特征取舍、调参或重选模型。解释图不用于反向调参；第一学期比较属于回顾性信息比较，不声称已验证实时预警或因果效果。
- A验收后冻结原始数据、划分、特征文件和口径。后续更改先开Issue说明影响，由组长确认并记录版本，B/C再统一重跑。

## 3 提交与交接规则

`main`保存已验收的共同版本。成员从**最新main**创建自己的分支：`member-a-data`、`member-b-models`、`member-c-evaluation`、`member-d-report`、`member-e-presentation`。B在A合并后开始，C在B合并后开始；不要提前从旧版本长期并行开展下游实验。

1. 上传到表中指定目录，同时填写自己的交接单，写明输入版本、运行命令、环境、输出文件及未解决问题。
2. 提交 Pull Request（合并请求，简称PR），目标为`main`；完成状态设为“待验收”。阶段截止以PR提交时间为准，验收后才算完成交接。
3. 组长核对交付物，接手成员确认能按命令复现后，**由组长合并**；`PROGRESS.md`记录接收状态及合并提交编号。
4. 组长维护进度表；其他成员不自行覆盖公共配置、划分文件或其他成员目录。不上传仅有截图、缺少代码和数据说明的结果。

“仅组长合并main”首先是团队约定；个人免费账户的私有仓库不保证能强制分支保护。若账户支持，可另行启用要求PR审核的main保护规则。

## 4 环境和目录约定

初始依赖列于 `requirements.txt`。A在独立虚拟环境中安装后，记录Python版本并导出 `environment/requirements-lock.txt`，由B核验可运行后冻结。B/C使用相同Python主次版本并执行 `python -m pip install -r environment/requirements-lock.txt`。数据预处理和模型入口脚本由对应成员实现，本初始框架未包含这些实验脚本。

| 目录或文件 | 用途 |
|---|---|
| `configs/` | 实验口径、A完成后冻结的特征清单 |
| `data/` | 原始数据、整理数据、固定划分及来源校验记录 |
| `src/A_data/`、`src/B_models/`、`src/C_evaluation/` | A/B/C各自的代码；各目录README列明交付标准 |
| `results/A_data/`、`results/B_validation/`、`results/C_final/` | 数据说明、验证结果、最终测试结果；不要混放 |
| `models/` | B交付的模型文件和完整预处理管道 |
| `docs/handoffs/` | A/B/C交接单与验收记录 |
| `docs/proposal/`、`docs/report/`、`docs/presentation/` | 提案、成稿和展示材料 |
| `PROGRESS.md` | 组长维护的唯一进度表 |

交付脚本从仓库根目录运行，使用相对路径，交接单必须给出实际可运行的命令。结果表用CSV，参数和运行记录用JSON，图用PNG。初始`.gitkeep`仅用于保留空目录。单文件超过20MB先联系组长；网页上传同样要自行排除虚拟环境、缓存和账号凭据，不能依赖`.gitignore`代为过滤。

## 5 数据来源

Predict Students' Dropout and Academic Success，4,424条学生记录、36个输入特征及1个结局标签。数据与变量字典：[UCI数据页](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)；[永久链接](https://doi.org/10.24432/C5MC89)。数据采用Creative Commons Attribution 4.0 International（署名4.0国际许可）；原始文件保留不改，遵守署名要求。数据获取和引用细节见 [`data/raw/README.md`](data/raw/README.md)。
