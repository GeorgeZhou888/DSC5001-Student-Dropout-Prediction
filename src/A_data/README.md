# A 数据准备与固定基准

截止：2026-11-05 23:59，香港时间。实现入口：src/A_data/prepare_data.py。

## 上传清单

| 位置 | 最低内容 |
|---|---|
| data/raw/data.csv | 官方原始数据，保持不变 |
| data/processed/students.csv | 添加稳定row_id的整理表；保留Target及原量纲，不提前全局缩放或编码 |
| data/splits/split_manifest.csv | row_id,dataset_split,cv_fold三列；dataset_split为train/test，训练折号0至4，测试折号留空 |
| configs/feature_sets.json | 两个真实列名列表；交付前把状态设为ready，验收合并的提交即冻结版本 |
| data/manifest.json | 数据来源、日期、行数、文件SHA-256、协议/特征版本，以及清理前后记录变化 |
| results/A_data/audit.json | 字段、缺失、重复、异常及标签检查结果；保留或删除记录的理由 |
| results/A_data/data_dictionary.csv | 每个字段的含义、数值/类别类型、可获得时点及是否入选 |
| results/A_data/figures/ | 仅基于训练数据的类别分布、关键背景变量与成绩分布图 |
| environment/ | Python版本、锁定依赖及系统说明 |
| docs/handoffs/A.md | 运行命令、交付列表、版本及已知问题 |

## 注意事项

先基于原始行顺序分配唯一row_id，再进行必要清理，不得因排序而重新编号。按protocol.json固定一次80/20划分，并对训练部分固定一次五折划分。学习型预处理由B在训练折内完成。

入学特征组只保留可确认在入学时已知的信息；第二组在相同入学特征上加入第一学期表现。两个列表均不得包含Target、row_id、dataset_split或cv_fold。测量时点不明的缴费/欠费等状态字段应剔除。不要仅凭看起来像重复就删除可能对应不同学生的记录。

交付前先完成环境与特征口径，把protocol.json和feature_sets.json的status都改为ready，再生成data/manifest.json中的文件校验值。以验收合并的提交为冻结版本，不在合并后另外修改状态而使校验值失效。

将清洗、划分、特征和环境版本一起交B验收；本阶段不进行模型比较或测试集误差分析。
