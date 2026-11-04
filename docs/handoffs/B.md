# B阶段交接单

状态：本地实验与模型冻结完成，待上游版本确认及团队验收；未声称已提交PR或合并。

- 负责人：B；执行日期：2026-11-04。
- 截止：2026-11-12 23:59，UTC+8。
- 接收人：C。
- 本阶段PR链接：待真实提交后填写。
- 来源ZIP：DSC5001-Student-Dropout-Prediction.zip。
- 来源ZIP SHA-256：`5b7ff7250f0d8d8840c0099e1f2b6f1a44a0ece59da3db92b1884182322689c3`。
- 归档main快照提交：`c5d69b848d6f1fe696c4d82c632ede438db7fadd`。
- A manifest记录的输入/生成基准：`986729eff3c818fff469aac46f9e523b3f6ca2a1`；与归档main快照区别保留，不冒充A验收提交。
- 输入实际字节哈希：见 `results/B_validation/run_manifest.json` 的inputs_actual_sha256。
- Python：3.14.5，Windows AMD64；19项依赖匹配A锁定版本。

## 已上传文件及用途

| 路径 | 用途 |
|---|---|
| src/B_models/train_models.py | 固定折调参、重拟合及保存模型 |
| src/B_models/verify_B.py | 核对输入、训练代码、输出及模型；可独立重算选定折 |
| src/B_models/test_b_models.py | 输入完整性、固定折、折内学习、测试集隔离、未知类别和序列化测试 |
| results/B_validation/cv_metrics.csv | 35条选定模型/基线的逐折指标 |
| results/B_validation/cv_all_candidates.csv | 265条候选/基线逐折指标 |
| results/B_validation/cv_search_results.csv、cv_summary.csv | 52候选汇总与7组最终验证汇总 |
| results/B_validation/selected_params.json | 最终参数及验证选择的C代表模型 |
| results/B_validation/run_manifest.json | 输入/输出/源代码哈希、命令、环境、种子和时刻 |
| results/B_validation/model_catalog.json | 模型到特征的对应、概率列标签、大小和哈希 |
| results/B_validation/input_verification.json | 逐项上游验收与CRLF/LF例外证据 |
| results/B_validation/environment.json、requirements-lock.txt | 本次真实环境及同A版本的依赖锁 |
| results/B_validation/delivery_check.json、cv_reproduction.csv | 16项检查及35折独立重算证据 |
| models/*.joblib | 六个完整模型管道，加一个共享多数类基线 |

## 复现命令

先按 `src/B_models/README.md` 重建Python3.14虚拟环境并安装A锁定依赖。从仓库根目录：

```powershell
python -m unittest discover -s src/B_models -p test_b_models.py -v
python src/B_models/verify_B.py --recheck-cv
```

仅需重新生成交付时：

```powershell
python src/B_models/train_models.py --allow-lf-equivalent --source-commit c5d69b848d6f1fe696c4d82c632ede438db7fadd --source-zip-sha256 5b7ff7250f0d8d8840c0099e1f2b6f1a44a0ece59da3db92b1884182322689c3 --jobs 2 --overwrite
```

已有结果默认受保护。重新训练会显式替换B模型和结果，后续重新执行verify；不改写A输入。原始完整搜索耗时约64秒。

## 核验结果与主要发现

八项针对性测试通过；16/16项交付核验通过；独立重算35个选定验证折，三个指标逐项一致（绝对容差1e-12）。六个模型与多数类基线都在3,539条固定训练记录上重拟合，所有模型序列化往返检查通过；未使用885条测试记录进行拟合或指标评估。

| 信息条件 | 模型 | 验证 macro-F1（均值±标准差） | 退学 precision | 退学 recall |
|---|---|---:|---:|---:|
| 入学 | 逻辑回归 | 0.523516 ± 0.010023 | 0.560193 | 0.525041 |
| 入学 | 决策树 | 0.455222 ± 0.034207 | 0.506261 | 0.478453 |
| 入学 | 随机森林 | 0.518996 ± 0.008463 | 0.568812 | 0.537387 |
| 入学＋第一学期 | 逻辑回归 | 0.644385 ± 0.015902 | 0.741662 | 0.643809 |
| 入学＋第一学期 | 决策树 | 0.601586 ± 0.014089 | 0.728717 | 0.674550 |
| 入学＋第一学期 | 随机森林 | 0.644883 ± 0.017185 | 0.730113 | 0.697419 |
| 共用基线 | 多数类 | 0.222013 ± 0.000187 | 0 | 0 |

按预设验证选择规则，交给C的代表解释模型为 `models/enrollment_plus_sem1__random_forest.joblib`。其与同组逻辑回归macro-F1差约0.000498，不能声称具有显著优势。

所有管道classes_实际顺序均为 `['Dropout', 'Enrolled', 'Graduate']`；C仍须以加载后的classes_显式映射。其余模型路径格式为`models/<feature_set>__<model>.joblib`，具体列名从model_catalog读取。多数类基线使用入学18项输入接口，两组共同对照。

## C读取模型示例

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

## 已知问题和接手注意事项

使用用户指定的原始上传 ZIP。A 的26项文件中7项原字节匹配，19项仅CRLF/LF差异；这19项规范化摘要均匹配A记录，训练程序没有重写任何A文件。开启该例外的实际参数为 `--allow-lf-equivalent`，逐项证据在 `input_verification.json`。没有跳过其他内容差异检查。A 的历史验收报告不能替代此次输入版本的人工验收。

本次为Windows AMD64、Python 3.14.5；A记录为macOS ARM64、Python 3.14.6。README要求相同Python主次版本，本次满足3.14；补丁差异如实记录。19项第三方依赖版本全部匹配A锁定文件，无包版本调整。只执行新编写的B代码及测试，没有运行上传包中的A脚本。

尚未声称A已正式合并、B/C已确认或组长已验收。正式交接时须由组长确认所用ZIP快照及换行例外；如更换A基准，应按新输入重新训练并更新全部B记录。

CV分数参与调参，不当作最终测试分数；第一学期信息比较仅回顾性。冻结后不要依据C测试结果重新调参、选择特征、阈值或代表解释模型。

## 人工验收记录

- 本阶段PR链接及B提交编号：待真实提交后填写。
- C接收确认及复现日期：待填写。
- 组长对A输入快照及换行例外的确认：待填写。
- 组长验收结论：待填写。
- 合并提交编号与日期：待填写。
