# 机器学习模型项目结构

## 完整目录结构

```
XXXMicro/
├── model/
│   ├── multi_bio_select.py              # 数据预处理和生物标志物筛选
│   └── machine_learning/                # 机器学习模型模块（新增）
│       ├── __init__.py                  # 模块初始化
│       ├── model_trainer.py             # 模型训练器
│       ├── model_evaluator.py           # 模型评估器
│       ├── model_comparison.py          # 模型对比分析
│       ├── run_ml_pipeline.py           # 完整流程脚本
│       ├── run_quick_test.py            # 快速测试脚本
│       ├── examples.py                  # 使用示例
│       ├── requirements.txt             # 依赖库列表
│       ├── README.md                    # 模块说明文档
│       └── PROJECT_STRUCTURE.md         # 本文档
│
├── Data/
│   └── AD/
│       └── species_abundance.csv        # 预处理后的数据
│
└── results/
    └── machine_learning/                # 模型训练结果
        ├── models/                      # 训练好的模型
        │   ├── LogisticRegression.pkl
        │   ├── RandomForest.pkl
        │   ├── SVM.pkl
        │   ├── XGBoost.pkl
        │   ├── LightGBM.pkl
        │   ├── standard_scaler.pkl
        │   └── best_params.json
        │
        └── evaluation/                  # 评估结果
            ├── model_comparison.csv
            ├── evaluation_results.json
            ├── model_comparison_report.txt
            ├── confusion_matrices.png
            ├── roc_curves_comparison.png
            ├── metrics_comparison.png
            ├── radar_chart.png
            ├── performance_heatmap.png
            └── model_ranking_roc_auc.png
```

## 模块关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    数据预处理模块                              │
│              (multi_bio_select.py)                          │
│  - 多层级数据拆分（目/科/属/种）                                │
│  - CLR转换                                                   │
│  - 生物标志物筛选                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ 输出预处理后的数据
                      ↓
┌─────────────────────────────────────────────────────────────┐
│                  机器学习模型模块                              │
│              (machine_learning/)                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  MLModelTrainer (model_trainer.py)                  │   │
│  │  - 数据加载与划分                                      │   │
│  │  - 特征标准化                                          │   │
│  │  - 模型训练                                            │   │
│  │  - 超参数优化（网格搜索+交叉验证）                       │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │ 训练好的模型                            │
│                    ↓                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ModelEvaluator (model_evaluator.py)                │   │
│  │  - 性能指标计算                                        │   │
│  │  - 混淆矩阵                                            │   │
│  │  - ROC曲线                                             │   │
│  │  - 可视化图表                                          │   │
│  └─────────────────┬───────────────────────────────────┘   │
│                    │ 评估结果                                │
│                    ↓                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ModelComparison (model_comparison.py)              │   │
│  │  - 模型排名                                            │   │
│  │  - 性能对比                                            │   │
│  │  - 雷达图/热力图                                        │   │
│  │  - 总结报告                                            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 工作流程

### 完整流程（run_ml_pipeline.py）

```
1. 数据加载
   ↓
2. 数据划分（8:2分层抽样）
   ↓
3. 特征标准化（Z-score）
   ↓
4. 模型训练
   ├─ 逻辑回归
   ├─ 随机森林
   ├─ SVM
   ├─ XGBoost
   └─ LightGBM
   ↓
5. 超参数优化（5折交叉验证+网格搜索）
   ↓
6. 模型评估
   ├─ 准确率
   ├─ 精确率
   ├─ 召回率
   ├─ 特异性
   ├─ F1分数
   └─ ROC-AUC
   ↓
7. 模型对比
   ├─ 性能排名
   ├─ 可视化图表
   └─ 总结报告
   ↓
8. 结果保存
```

## 使用场景

### 场景1: 快速测试（不进行超参数优化）
```bash
python run_quick_test.py
```
- 适用于快速验证代码
- 使用默认参数
- 运行时间短

### 场景2: 完整训练（含超参数优化）
```bash
python run_ml_pipeline.py
```
- 适用于正式实验
- 进行超参数优化
- 运行时间较长（取决于数据规模）

### 场景3: 自定义使用
```bash
python examples.py
```
- 查看各种使用示例
- 学习如何自定义流程

## 核心功能对应论文章节

| 功能模块 | 对应论文章节 | 说明 |
|---------|------------|------|
| 数据划分 | 3.3.1 | 8:2分层抽样 |
| 超参数优化 | 3.3.2 | 5折交叉验证+网格搜索 |
| 模型训练 | 3.3.3 | 5种分类模型 |
| 模型评估 | 3.3.4 | 6项性能指标 |
| 性能对比 | 3.3.5 | 多模型对比分析 |

## 输出文件说明

### 模型文件（.pkl）
- 使用pickle序列化保存
- 可直接加载用于预测
- 包含完整的模型参数

### 评估结果（.json/.csv）
- JSON格式：详细的评估指标
- CSV格式：性能对比表

### 可视化图表（.png/.pdf）
- PNG格式：用于展示
- PDF格式：用于论文发表

### 报告文件（.txt）
- 纯文本格式
- 包含完整的分析结果
- 易于阅读和分享

## 扩展建议

### 1. 添加新模型
在 `model_trainer.py` 中：
```python
def _create_base_model(self, model_name):
    if model_name == 'YourNewModel':
        return YourNewModel(...)
```

### 2. 自定义评估指标
在 `model_evaluator.py` 中：
```python
def evaluate_model(self, model, X_test, y_test, model_name):
    # 添加新的评估指标
    your_metric = calculate_your_metric(y_test, y_pred)
    metrics['your_metric'] = your_metric
```

### 3. 添加新的可视化
在 `model_comparison.py` 中：
```python
def plot_your_chart(self, output_dir=None):
    # 实现你的可视化逻辑
    pass
```

## 常见问题

### Q1: 如何修改数据划分比例？
A: 在调用 `split_data()` 时修改 `test_size` 参数

### Q2: 如何跳过超参数优化？
A: 设置 `use_grid_search=False`

### Q3: 如何只训练部分模型？
A: 在 `train_all_models()` 中指定 `model_names` 参数

### Q4: 如何修改超参数搜索空间？
A: 修改 `MLModelTrainer` 中的 `param_grids` 字典

### Q5: 如何处理内存不足问题？
A: 
- 减少超参数搜索空间
- 减少交叉验证折数
- 只训练部分模型

## 性能优化建议

1. **并行计算**：网格搜索默认使用 `n_jobs=-1` 利用所有CPU核心
2. **减少搜索空间**：根据经验缩小超参数范围
3. **使用快速测试模式**：开发阶段使用 `run_quick_test.py`
4. **批量处理**：一次性训练所有模型，避免重复加载数据

## 联系方式

如有问题或建议，请联系：
- 作者：刘舒琪
- 学校：汕头大学数学与计算机学院
- 专业：数据科学与大数据技术
