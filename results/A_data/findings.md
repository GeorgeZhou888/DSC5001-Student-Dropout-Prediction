# A阶段训练集描述结果

范围：仅使用固定划分中的3,539条训练记录。图形与描述统计不含测试样本。

- 训练集类别占比：Dropout: 32.1%; Enrolled: 17.9%; Graduate: 49.9%。应在模型验证中保留三类并关注宏平均F1及退学类指标。
- 第一学期平均成绩的组内中位数：Dropout: 10.83; Enrolled: 12.00; Graduate: 13.00。这是训练数据中的描述性关联，不能解释为因果效果。
- 零成绩及零课程计数均保留；不能逐条确定零值对应的具体学业原因。
- 类别编号不表示连续数量关系；编码及缩放由B在各训练折内部拟合。

## 图与表

| 文件 | 用途 |
|---|---|
| figures/01_train_class_distribution.png | 类别规模与不平衡程度 |
| figures/02_train_enrollment_distributions.png | 入学年龄、两项入学成绩及志愿顺序分布 |
| figures/03_train_sem1_distributions.png | 第一学期六项变量的原始量纲分布 |
| figures/04_train_sem1_grade_by_outcome.png | 按最终记录结局描述第一学期成绩及通过课程数 |
| train_class_summary.csv | 类别图对应数据 |
| train_numeric_summary.csv | 两组特征中数值字段的训练集描述统计 |
| train_sem1_by_outcome.csv | 按结局分组的训练集学业描述统计 |

## 解释边界

数据来自单一机构，随机留出测试不能证明跨学校或跨时间泛化。缺少逐学生退出日期，
第一学期数据可能已反映部分学生的退出过程，故比较属于回顾性信息比较，不能宣称
已经验证实时预警或因果效果。Enrolled是独立结局，不能合并为成功；Dropout也不等同于
永久离开整个高等教育系统。18/24项特征按含义与时点预设，不依测试结果修改。
