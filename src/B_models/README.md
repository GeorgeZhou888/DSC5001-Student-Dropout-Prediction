# B 模型训练与冻结

截止：2026-11-12 23:59，香港时间。A阶段合并后，从最新main开始。实现入口：src/B_models/train_models.py。

读取A确认的students.csv、split_manifest.csv及两个特征列表，禁止重新随机切分或重新生成交叉验证折。使用共同五折，对逻辑回归、决策树、随机森林做有限调参；六组核心配置外再训练一个多数类基线。

## 上传清单

- results/B_validation/cv_metrics.csv：每组模型、特征组、折号及验证指标；至少含macro_f1、dropout_precision、dropout_recall。
- results/B_validation/selected_params.json：每组最终参数、选择指标和平均验证分数。
- results/B_validation/run_manifest.json：A基准提交编号、协议/数据/划分/特征文件的SHA-256、Python及包版本、种子、运行命令、运行日期。
- models/：保存最终在80%训练数据上拟合的完整预处理及分类器管道。命名为enrollment__logistic_regression.joblib、enrollment__decision_tree.joblib、enrollment__random_forest.joblib，第二组前缀改为enrollment_plus_sem1；基线另存majority_baseline.joblib。
- docs/handoffs/B.md：实际执行命令、模型到特征组的对应关系、概率列对应标签及C读取模型的例子。

将模型和学习型预处理放入同一管道，确保编码与缩放在每个训练折重新拟合。按验证宏平均F1选择参数，之后仅用训练部分重新拟合。测试部分不参与拟合、调参、阈值选择或模型选择。单个模型文件若超过20MB，先联系组长决定保存方式。
