"""
简化版运行脚本（用于快速测试，不进行超参数优化）
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.model_trainer import MLModelTrainer
from machine_learning.model_evaluator import ModelEvaluator
from machine_learning.model_comparison import ModelComparison


def main():
    """快速测试版本（不进行超参数优化）"""

    # 配置参数
    DATA_PATH = "/root/XXXMicro/Data/AD/species_abundance.csv"
    OUTPUT_DIR = "/root/XXXMicro/results/machine_learning_quick"
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("="*80)
    print("快速测试模式（使用默认参数，不进行超参数优化）")
    print("="*80)

    # 数据加载
    trainer = MLModelTrainer(random_state=42)
    X, y, feature_names = trainer.load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)
    trainer.scalers['standard'] = scaler

    # 训练模型（不进行超参数优化）
    print("\n训练模型（使用默认参数）...")
    models = trainer.train_all_models(
        X_train_scaled, y_train,
        model_names=['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost', 'LightGBM'],
        use_grid_search=False  # 关闭超参数优化
    )

    trainer.save_models(MODEL_DIR)

    # 评估模型
    print("\n评估模型...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test_scaled, y_test)

    print("\n模型性能对比表:")
    print(evaluator.get_comparison_table().to_string(index=False))

    evaluator.save_results(EVAL_DIR)

    # 生成报告
    comparison = ModelComparison(results)
    comparison.generate_summary_report(EVAL_DIR)

    print("\n完成！结果保存在:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
