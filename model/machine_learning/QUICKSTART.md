# 机器学习模型快速入门指南

## 🎯 项目概述

本模块实现了基于肠道菌群数据的阿尔茨海默病（AD）风险预测的机器学习模型，包含：
- **5种分类模型**：逻辑回归、随机森林、SVM、XGBoost、LightGBM
- **完整的训练流程**：数据划分、特征标准化、超参数优化、模型评估
- **丰富的可视化**：混淆矩阵、ROC曲线、雷达图、热力图等
- **详细的对比分析**：模型排名、性能报告、最优模型推荐

## 📦 安装依赖

```bash
cd /root/XXXMicro/model/machine_learning
pip install -r requirements.txt
```

## 🚀 快速开始

### 方式1：运行完整流程（推荐用于正式实验）

```bash
python run_ml_pipeline.py
```

**特点**：
- ✅ 进行超参数优化（5折交叉验证+网格搜索）
- ✅ 训练所有5种模型
- ✅ 生成完整的评估报告和可视化图表
- ⏱️ 运行时间：约30-60分钟（取决于数据规模）

### 方式2：快速测试（推荐用于开发调试）

```bash
python run_quick_test.py
```

**特点**：
- ✅ 使用默认参数，不进行超参数优化
- ✅ 快速验证代码功能
- ⏱️ 运行时间：约2-5分钟

### 方式3：查看使用示例

```bash
python examples.py
```

**包含5个示例**：
1. 训练单个模型
2. 训练多个模型并对比
3. 完整流程（含超参数优化）
4. 加载已训练的模型进行预测
5. 自定义超参数搜索空间

## 📊 输出结果

运行完成后，结果保存在 `/root/XXXMicro/results/machine_learning/`：

```
results/machine_learning/
├── models/                              # 训练好的模型
│   ├── LogisticRegression.pkl          # 逻辑回归模型
│   ├── RandomForest.pkl                # 随机森林模型
│   ├── SVM.pkl                         # SVM模型
│   ├── XGBoost.pkl                     # XGBoost模型
│   ├── LightGBM.pkl                    # LightGBM模型
│   ├── standard_scaler.pkl             # 标准化器
│   └── best_params.json                # 最优超参数
│
└── evaluation/                          # 评估结果
    ├── model_comparison.csv            # 性能对比表（CSV）
    ├── evaluation_results.json         # 详细评估结果（JSON）
    ├── model_comparison_report.txt     # 总结报告（TXT）
    ├── confusion_matrices.png          # 混淆矩阵图
    ├── roc_curves_comparison.png       # ROC曲线对比图
    ├── metrics_comparison.png          # 指标对比图
    ├── radar_chart.png                 # 雷达图
    ├── performance_heatmap.png         # 性能热力图
    └── model_ranking_roc_auc.png       # 排名图
```

## 📈 评估指标

模型性能通过以下6项指标评估：

| 指标 | 说明 | 计算公式 |
|-----|------|---------|
| **准确率** (Accuracy) | 正确分类的样本比例 | (TP+TN)/(TP+TN+FP+FN) |
| **精确率** (Precision) | 预测为正类中实际为正类的比例 | TP/(TP+FP) |
| **召回率** (Recall) | 实际为正类中被正确预测的比例 | TP/(TP+FN) |
| **特异性** (Specificity) | 实际为负类中被正确预测的比例 | TN/(TN+FP) |
| **F1分数** (F1-Score) | 精确率和召回率的调和平均 | 2×P×R/(P+R) |
| **ROC-AUC** | ROC曲线下面积 | 0.5-1.0 |

## 🔧 自定义使用

### 示例1：只训练特定模型

```python
from machine_learning.model_trainer import MLModelTrainer

trainer = MLModelTrainer(random_state=42)
X, y, _ = trainer.load_data("data.csv")
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)

# 只训练SVM和逻辑回归
models = trainer.train_all_models(
    X_train_scaled, y_train,
    model_names=['SVM', 'LogisticRegression'],
    use_grid_search=True
)
```

### 示例2：修改超参数搜索空间

```python
trainer = MLModelTrainer(random_state=42)

# 自定义搜索空间（更小的范围以节省时间）
trainer.param_grids['SVM'] = {
    'C': [1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

# 训练
model, params, cv_results = trainer.train_model(
    'SVM', X_train_scaled, y_train, use_grid_search=True
)
```

### 示例3：加载模型进行预测

```python
import pickle

# 加载模型
with open('results/machine_learning/models/SVM.pkl', 'rb') as f:
    model = pickle.load(f)

with open('results/machine_learning/models/standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 预测
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
y_pred_proba = model.predict_proba(X_new_scaled)[:, 1]
```

## 📚 文档说明

| 文档 | 说明 |
|-----|------|
| [README.md](README.md) | 模块功能说明 |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | 项目结构详解 |
| [QUICKSTART.md](QUICKSTART.md) | 本快速入门指南 |
| [examples.py](examples.py) | 使用示例代码 |

## ⚙️ 配置说明

### 数据路径配置

在脚本中修改数据路径：

```python
DATA_PATH = "/root/XXXMicro/Data/AD/species_abundance.csv"
```

### 输出路径配置

```python
OUTPUT_DIR = "/root/XXXMicro/results/machine_learning"
```

### 随机种子配置

```python
trainer = MLModelTrainer(random_state=42)  # 确保结果可复现
```

## 🎓 论文对应关系

本模块实现基于论文第3.3章节的设计：

| 论文章节 | 代码实现 | 文件位置 |
|---------|---------|---------|
| 3.3.1 数据划分 | `split_data()` | model_trainer.py |
| 3.3.2 超参数优化 | `train_model()` | model_trainer.py |
| 3.3.3 模型训练 | `train_all_models()` | model_trainer.py |
| 3.3.4 模型评估 | `evaluate_model()` | model_evaluator.py |
| 3.3.5 性能对比 | `ModelComparison` | model_comparison.py |

## ❓ 常见问题

### Q1: 运行时间太长怎么办？

**A**: 有以下几种方法：
1. 使用快速测试模式：`python run_quick_test.py`
2. 减少模型数量：只训练部分模型
3. 缩小超参数搜索空间
4. 减少交叉验证折数（默认5折）

### Q2: 内存不足怎么办？

**A**: 
1. 关闭超参数优化：`use_grid_search=False`
2. 减少模型数量
3. 使用更小的数据集进行测试

### Q3: 如何查看最优模型？

**A**: 查看生成的报告文件：
```bash
cat results/machine_learning/evaluation/model_comparison_report.txt
```

### Q4: 如何修改评估指标？

**A**: 在 `model_evaluator.py` 的 `evaluate_model()` 方法中添加新指标

### Q5: 数据格式要求？

**A**: 
- CSV格式
- 必须包含 `sample_id` 和 `label` 列
- 其他列为特征（菌群丰度数据）
- label: 0=对照组，1=AD患者

## 🔍 性能优化建议

1. **并行计算**：默认使用所有CPU核心（`n_jobs=-1`）
2. **缓存结果**：训练好的模型会自动保存，避免重复训练
3. **分批训练**：可以先训练部分模型，再训练其他模型
4. **使用GPU**：XGBoost和LightGBM支持GPU加速（需要额外配置）

## 📞 技术支持

如遇到问题，请检查：
1. ✅ 依赖库是否正确安装
2. ✅ 数据文件路径是否正确
3. ✅ 数据格式是否符合要求
4. ✅ Python版本（建议3.8+）

## 🎉 下一步

1. 运行快速测试验证环境：`python run_quick_test.py`
2. 查看使用示例学习用法：`python examples.py`
3. 运行完整流程获得最优模型：`python run_ml_pipeline.py`
4. 根据结果选择最优模型用于实际应用

---

**祝你使用愉快！** 🚀
