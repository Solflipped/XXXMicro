"""
使用示例：展示如何使用机器学习模块进行模型训练和评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.model_trainer import MLModelTrainer
from machine_learning.model_evaluator import ModelEvaluator
from machine_learning.model_comparison import ModelComparison


# ============================================================================
# 示例1: 基础使用 - 训练单个模型
# ============================================================================
def example_1_train_single_model():
    """示例1: 训练单个模型"""
    print("\n" + "="*80)
    print("示例1: 训练单个模型（逻辑回归）")
    print("="*80)

    # 初始化训练器
    trainer = MLModelTrainer(random_state=42)

    # 加载数据
    X, y, feature_names = trainer.load_data("/root/XXXMicro/Data/AD/species_abundance.csv")

    # 划分数据集
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    # 标准化
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)

    # 训练逻辑回归模型（不进行超参数优化）
    model, params, cv_results = trainer.train_model(
        'LogisticRegression',
        X_train_scaled,
        y_train,
        use_grid_search=False
    )

    # 评估模型
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(model, X_test_scaled, y_test, 'LogisticRegression')

    print(f"\n模型性能: ROC-AUC = {metrics['roc_auc']:.4f}")


# ============================================================================
# 示例2: 训练多个模型并对比
# ============================================================================
def example_2_train_multiple_models():
    """示例2: 训练多个模型并对比"""
    print("\n" + "="*80)
    print("示例2: 训练多个模型并对比")
    print("="*80)

    trainer = MLModelTrainer(random_state=42)

    # 加载数据
    X, y, feature_names = trainer.load_data("/root/XXXMicro/Data/AD/species_abundance.csv")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)

    # 训练多个模型（不进行超参数优化以节省时间）
    models = trainer.train_all_models(
        X_train_scaled,
        y_train,
        model_names=['LogisticRegression', 'SVM', 'RandomForest'],
        use_grid_search=False
    )

    # 评估所有模型
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test_scaled, y_test)

    # 打印对比表
    print("\n模型性能对比:")
    print(evaluator.get_comparison_table().to_string(index=False))

    # 对比分析
    comparison = ModelComparison(results)
    best_model, best_auc = comparison.get_best_model('roc_auc')
    print(f"\n最优模型: {best_model} (ROC-AUC: {best_auc:.4f})")


# ============================================================================
# 示例3: 完整流程（含超参数优化）
# ============================================================================
def example_3_full_pipeline_with_tuning():
    """示例3: 完整流程（含超参数优化）"""
    print("\n" + "="*80)
    print("示例3: 完整流程（含超参数优化）")
    print("="*80)
    print("注意: 超参数优化会消耗较长时间")

    OUTPUT_DIR = "/root/XXXMicro/results/ml_example"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = MLModelTrainer(random_state=42)

    # 数据准备
    X, y, feature_names = trainer.load_data("/root/XXXMicro/Data/AD/species_abundance.csv")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)
    trainer.scalers['standard'] = scaler

    # 训练模型（进行超参数优化）
    models = trainer.train_all_models(
        X_train_scaled,
        y_train,
        model_names=['LogisticRegression', 'SVM'],  # 只训练两个模型以节省时间
        use_grid_search=True  # 开启超参数优化
    )

    # 保存模型
    trainer.save_models(OUTPUT_DIR)

    # 评估
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test_scaled, y_test)
    evaluator.save_results(OUTPUT_DIR)

    # 生成报告
    comparison = ModelComparison(results)
    comparison.generate_summary_report(OUTPUT_DIR)

    print(f"\n结果已保存至: {OUTPUT_DIR}")


# ============================================================================
# 示例4: 加载已训练的模型进行预测
# ============================================================================
def example_4_load_and_predict():
    """示例4: 加载已训练的模型进行预测"""
    print("\n" + "="*80)
    print("示例4: 加载已训练的模型进行预测")
    print("="*80)

    MODEL_DIR = "/root/XXXMicro/results/ml_example"

    # 检查模型是否存在
    if not os.path.exists(os.path.join(MODEL_DIR, "LogisticRegression.pkl")):
        print("模型文件不存在，请先运行示例3训练模型")
        return

    # 加载模型
    import pickle
    with open(os.path.join(MODEL_DIR, "LogisticRegression.pkl"), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "standard_scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)

    print("✓ 模型加载成功")

    # 加载测试数据
    trainer = MLModelTrainer(random_state=42)
    X, y, feature_names = trainer.load_data("/root/XXXMicro/Data/AD/species_abundance.csv")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    # 标准化
    X_test_scaled = scaler.transform(X_test)

    # 预测
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n预测完成:")
    print(f"  样本数: {len(y_pred)}")
    print(f"  预测为AD的样本数: {sum(y_pred == 1)}")
    print(f"  预测为Control的样本数: {sum(y_pred == 0)}")
    print(f"  平均预测概率: {y_pred_proba.mean():.4f}")


# ============================================================================
# 示例5: 自定义超参数搜索空间
# ============================================================================
def example_5_custom_param_grid():
    """示例5: 自定义超参数搜索空间"""
    print("\n" + "="*80)
    print("示例5: 自定义超参数搜索空间")
    print("="*80)

    trainer = MLModelTrainer(random_state=42)

    # 自定义超参数搜索空间（更小的搜索空间以节省时间）
    trainer.param_grids['LogisticRegression'] = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }

    # 数据准备
    X, y, feature_names = trainer.load_data("/root/XXXMicro/Data/AD/species_abundance.csv")
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)

    # 训练
    model, params, cv_results = trainer.train_model(
        'LogisticRegression',
        X_train_scaled,
        y_train,
        use_grid_search=True
    )

    print(f"\n最优超参数: {params}")
    print(f"最优交叉验证AUC: {cv_results['best_score']:.4f}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    """运行所有示例"""
    print("="*80)
    print("机器学习模块使用示例")
    print("="*80)

    # 运行示例（可以注释掉不需要的示例）
    example_1_train_single_model()
    example_2_train_multiple_models()
    # example_3_full_pipeline_with_tuning()  # 耗时较长，默认注释
    # example_4_load_and_predict()
    # example_5_custom_param_grid()

    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)


if __name__ == "__main__":
    main()
