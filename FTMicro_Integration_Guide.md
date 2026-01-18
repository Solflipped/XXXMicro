# FTMicro 模型集成指南

## 概述
FTMicro (UFEN - Unimodal Feature Extraction Network) 现已成功集成到 train.py 训练框架中。

## 主要修改

### 1. FTMicro.py 修改
- **修复导入路径**：从 `from FT_transformer import ...` 改为 `from .FT_transformer import ...`
- **修改 forward 方法**：
  - 默认输出：只返回 logits (batch_size, 1)，用于训练
  - 可选输出：设置 `return_features=True` 时返回 (cls_features, logits)
  - 移除了 sigmoid 激活（由损失函数 BCEWithLogitsLoss 处理）
- **添加工厂方法**：`make_default()` 类方法，与其他模型保持一致

### 2. train.py 修改
- **导入模块**：添加 `from model.FTMicro import UFEN`
- **超参数**：添加 FTMicro 特有参数
  - `num_conv_layers`：卷积层数（默认 2）
  - `d_token`：token 嵌入维度（默认 192）
- **记录结构**：添加 FTMicro 的日志记录格式
- **数据加载**：FTMicro 使用单模态数据加载（与 FT_transformer 相同）
- **模型构建**：在训练循环中添加 FTMicro 模型构建逻辑

## 使用方法

### 基本用法
```python
from train import train

params = {
    'learning_rate': 1e-4,
    'batch_size': 8,
    'num_conv_layers': 2,  # 卷积层数
    'd_token': 192,         # token维度
}

train(
    disease='AD',           # 数据集
    feature='ko',           # 单模态：'ko' 或 'species'
    model_type='FTMicro',   # 模型类型
    **params
)
```

### 参数说明

#### 通用参数
- `disease`: 数据集名称（'AD', 'EW-T2D', 'Obesity' 等）
- `feature`: **仅支持单模态** - 'ko' 或 'species'
- `learning_rate` / `lr`: 学习率（默认 1e-4）
- `batch_size`: 批次大小（默认 8）

#### FTMicro 特有参数
- `num_conv_layers`: 卷积层数量（默认 2）
- `d_token`: token 嵌入维度（默认 192）
- `num_heads`: 注意力头数（模型内部默认 4，可通过修改模型代码调整）
- `dropout`: dropout 率（模型内部默认 0.1）

## 模型架构

FTMicro (UFEN) 采用以下架构：
1. **特征 Tokenization**：使用 FT-Transformer 的 FeatureTokenizer
2. **CLS Token**：添加分类 token
3. **多层卷积**：多个 1D 卷积层提取局部特征
4. **自注意力机制**：每层卷积后应用多头自注意力
5. **特征融合**：通过元素相加融合多层特征
6. **分类输出**：从 CLS token 提取全局特征进行分类

## 注意事项

1. **单模态限制**：FTMicro 目前仅支持单模态输入（'ko' 或 'species'），不支持多模态融合（如 'ko,species'）

2. **输出格式**：
   - 训练时：模型输出 logits，由 BCEWithLogitsLoss 处理
   - 推理时：可使用 `return_features=True` 获取特征向量

3. **与其他模型的对比**：
   - **vs FT_transformer**：FTMicro 在 tokenization 基础上增加了多层卷积和自注意力
   - **vs MBT/GAFT**：这些是多模态模型，FTMicro 是单模态模型

## 训练流程

train.py 使用 5 折交叉验证：
1. 固定种子 777 做 8:2 训练/测试划分
2. 训练集上做 5 折分层交叉验证
3. 每折训练独立模型，监控验证集 AUC
4. 使用早停机制（patience=20）
5. 保存每折最佳模型到 `./Checkpoints/{disease}/777/FTMicro/fold_{i}/`
6. 在测试集上评估所有折模型的平均性能

## 结果保存

结果保存在：`./results/{disease}/FTMicro.csv`

每行包含：
- 超参数：lr, batch_size, feature, num_conv_layers, d_token, seed
- 每折结果：fold, AUC, Recall, Precision, F1
- 汇总结果：fold='all', 指标格式为 mean(std)

## 测试

运行测试脚本验证集成：
```bash
python test_ftmicro.py
```

## 扩展到多模态

如果未来需要支持多模态，可以参考 MBT 模型的 Wrapper 方式：
1. 创建多个 UFEN 实例（每个模态一个）
2. 提取各模态的 cls_features
3. 使用融合模块（如注意力、拼接等）融合特征
4. 最终分类输出
