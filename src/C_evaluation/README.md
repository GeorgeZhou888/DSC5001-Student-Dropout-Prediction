# C 最终评估与解释

截止：2026-11-19 23:59，香港时间。B阶段合并并冻结后，从最新main开始。实现入口：src/C_evaluation/evaluate_models.py。

用A固定的同一测试样本评估B交付的六个模型和多数类基线。校验上游数据、划分、特征和环境版本；不重新训练或调参。

## 上传清单

- results/C_final/test_metrics.csv：模型、特征组、macro_f1、dropout_precision、dropout_recall及测试样本数，至少覆盖六组模型和基线。
- results/C_final/test_predictions.csv：长表，每行是一个模型对一个测试学生的预测；含row_id、model、feature_set、true_label、predicted_label及三类预测概率。概率顺序须按模型classes_显式映射。
- results/C_final/figures/：模型对比、两组特征信息增量、混淆矩阵、验证阶段预先选定代表模型的特征解释图。
- results/C_final/findings.md：回答三个研究问题，说明错分、信息增量和局限；不把预测关联解释为因果。
- results/C_final/run_manifest.json：B冻结提交编号、输入文件SHA-256、环境版本、运行命令和日期。
- docs/handoffs/C.md：把所有可引用表格、图片、解读和复现命令交给D/E。

最终测试后不能根据测试分数重新挑参数或特征。若发现实质实现错误，应暂停交接、记录影响并请组长确定修订版本；不得悄悄替换原结果。
