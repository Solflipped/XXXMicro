# 机器学习模型模块

本模块实现了基于肠道菌群数据的阿尔茨海默病（AD）风险预测的机器学习模型训练、评估和对比功能。

## 模块结构

```
machine_learning/
├── __init__.py                 # 模块初始化
├── model_trainer.py            # 模型训练器
├── model_evaluator.py          # 模型评估器
├── model_comparison.py         # 模型对比分析
├── run_ml_pipeline.py          # 完整流程脚本
└── README.md                   # 本文档
```

## 功能特性

### 1. 模型训练器 (MLModelTrainer)

- **支持的模型**：
  - 逻辑回归 (Logistic Regression)
  - 随机森林 (Random Forest)
  - 支持向量机 (SVM)
  - XGBoost
  - LightGBM

- **核心功能**：
  - 数据加载与预处理
  - 训练集/测试集划分（8:2分层抽样）
  - 特征标准化（Z-score标准化）
  - 超参数优化（5折分层交叉验证 + 网格搜索）
  - 模型保存与加载

### 2. 模型评估器 (ModelEvaluator)

- **评估指标**：
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率/敏感性 (Recall/Sensitivity)
  - 特异性 (Specificity)
  - F1分数 (F1-Score)
  - ROC-AUC

- **可视化功能**：
  - 混淆矩阵热力图
  - ROC曲线对比图
  - 性能指标对比图

### 3. 模型对比分析 (ModelComparison)

- **对比功能**：
  - 模型排名
  - 最优模型识别
  - 综合性能分析

- **可视化功能**：
  - 雷达图
  - 性能热力图
  - 排名条形图
  - 总结报告生成

## 使用方法

### 快速开始

```bash
# 运行完整流程
cd /root/XXXMicro/model/machine_learning
python run_ml_pipeline.py
```

### 自定义使用

```python
from machine_learning.model_trainer import MLModelTrainer
from machine_learning.model_evaluator import ModelEvaluator
from machine_learning.model_comparison import ModelComparison

# 1. 初始化训练器
trainer = MLModelTrainer(random_state=42)

# 2. 加载数据
X, y, feature_names = trainer.load_data("data.csv")

# 3. 划分数据集
X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

# 4. 特征标准化
X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)

# 5. 训练模型
models = trainer.train_all_models(X_train_scaled, y_train, use_grid_search=True)

# 6. 评估模型
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test_scaled, y_test)

# 7. 对比分析
comparison = ModelComparison(results)
comparison.generate_summary_report(output_dir="results")
```

## 超参数搜索空间

根据论文设计，各模型的超参数搜索空间如下：

### 逻辑回归
- C: [0.001, 0.01, 0.1, 1, 10, 100]
- penalty: ['l1', 'l2']
- solver: ['liblinear', 'saga']

### 随机森林
- n_estimators: [100, 200, 300, 500]
- max_depth: [5, 10, 15, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2']

### SVM
- C: [0.1, 1, 10, 100]
- kernel: ['linear', 'rbf', 'poly']
- gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
- degree: [2, 3, 4]

### XGBoost
- n_estimators: [100, 200, 300]
- max_depth: [3, 5, 7, 9]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]
- min_child_weight: [1, 3, 5]

### LightGBM
- n_estimators: [100, 200, 300]
- max_depth: [3, 5, 7, 9]
- learning_rate: [0.01, 0.05, 0.1, 0.2]
- num_leaves: [15, 31, 63, 127]
- subsample: [0.6, 0.8, 1.0]
- colsample_bytree: [0.6, 0.8, 1.0]

## 输出结果

运行完整流程后，将在输出目录生成以下文件：

```
results/machine_learning/
├── models/                              # 模型文件
│   ├── LogisticRegression.pkl
│   ├── RandomForest.pkl
│   ├── SVM.pkl
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── standard_scaler.pkl
│   └── best_params.json
│
└── evaluation/                          # 评估结果
    ├── model_comparison.csv             # 性能对比表
    ├── evaluation_results.json          # 详细评估结果
    ├── model_comparison_report.txt      # 总结报告
    ├── confusion_matrices.png           # 混淆矩阵
    ├── roc_curves_comparison.png        # ROC曲线
    ├── metrics_comparison.png           # 指标对比图
    ├── radar_chart.png                  # 雷达图
    ├── performance_heatmap.png          # 性能热力图
    └── model_ranking_roc_auc.png        # 排名图
```

## 依赖库

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

## 注意事项

1. **数据格式要求**：
   - CSV格式
   - 必须包含 `sample_id` 和 `label` 列
   - 特征列为菌群丰度数据

2. **计算资源**：
   - 网格搜索会消耗较多时间和计算资源
   - 可设置 `use_grid_search=False` 跳过超参数优化

3. **随机种子**：
   - 默认使用 `random_state=42` 确保结果可复现

4. **类别不平衡**：
   - 所有模型均设置了 `class_weight='balanced'` 处理样本不平衡

## 参考论文

本模块实现基于论文《老年痴呆肠道菌群生物标志物系统设计与实现》第3.3章节的设计：

- 3.3.1 数据划分
- 3.3.2 超参数优化
- 3.3.3 模型训练
- 3.3.4 模型评估
- 3.3.5 各模型性能对比

## 作者

刘舒琪 - 汕头大学数学与计算机学院
