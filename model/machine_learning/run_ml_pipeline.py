"""
完整的机器学习模型训练、评估和对比流程
使用示例脚本
"""

import sys
import os
import warnings

# 屏蔽 XGBoost GPU 数据传输警告
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.model_trainer import MLModelTrainer
from machine_learning.model_evaluator import ModelEvaluator
from machine_learning.model_comparison import ModelComparison


def main():
    """主函数：完整的模型训练、评估和对比流程"""

    # ==================== 配置参数 ====================
    # 模式选择：
    # 'all_features': 使用所有属级特征（用于层级对比）
    # 'selected_biomarkers': 只使用筛选出的20个生物标志物（推荐，用于最终模型）
    MODE = 'selected_biomarkers'  # 可选: 'all_features' 或 'selected_biomarkers'

    # 数据路径
    if MODE == 'all_features':
        # 使用预处理后的完整属水平数据
        DATA_PATH = "/root/XXXMicro/Data/AD/multilevel_preprocessed1/03_genus_level/genus_processed.csv"
        OUTPUT_DIR =  "/root/XXXMicro/model/machine_learning/result"
    else:
        # 使用筛选后的20个生物标志物
        DATA_PATH = "/root/XXXMicro/Data/AD/multilevel_preprocessed1/03_genus_level/genus_processed.csv"
        OUTPUT_DIR = "/root/XXXMicro/model/machine_learning/result"
        # 生物标志物列表（从筛选结果中获取）
        BIOMARKER_FILE = "/root/XXXMicro/Data/AD/multilevel_biomarkers1/03_genus_level/final_biomarkers.csv"

    # 输出目录
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("="*80)
    print("阿尔茨海默病肠道菌群机器学习模型训练与评估系统")
    print("="*80)
    print(f"运行模式: {MODE}")
    print(f"数据路径: {DATA_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*80)

    # ==================== 步骤1: 数据加载与划分 ====================
    print("\n步骤1: 数据加载与划分")
    print("-"*80)

    # 小数据集用CPU更快
    trainer = MLModelTrainer(random_state=42, use_gpu=False)

    # 加载数据
    X, y, feature_names = trainer.load_data(DATA_PATH)

    # 如果使用筛选后的生物标志物模式，需要进一步筛选特征
    if MODE == 'selected_biomarkers' and os.path.exists(BIOMARKER_FILE):
        print(f"\n加载生物标志物列表: {BIOMARKER_FILE}")
        import pandas as pd
        biomarkers_df = pd.read_csv(BIOMARKER_FILE)
        selected_features = biomarkers_df['feature'].tolist()

        # 筛选特征
        feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
        if len(feature_indices) > 0:
            X = X[:, feature_indices]
            feature_names = [feature_names[i] for i in feature_indices]
            print(f"使用筛选后的生物标志物: {len(feature_names)} 个特征")
            print(f"特征列表: {', '.join(feature_names[:5])}...")
        else:
            print("警告: 未找到匹配的生物标志物特征，使用所有特征")
    elif MODE == 'selected_biomarkers':
        print(f"警告: 生物标志物文件不存在: {BIOMARKER_FILE}")
        print("将使用所有特征进行训练")

    # 划分训练集和测试集（8:2分层抽样）
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)

    # 特征标准化
    X_train_scaled, X_test_scaled, scaler = trainer.standardize_features(X_train, X_test)
    trainer.scalers['standard'] = scaler

    # ==================== 步骤2: 模型训练 ====================
    print("\n步骤2: 模型训练与超参数优化")
    print("-"*80)

    # 定义要训练的模型
    model_names = ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost', 'LightGBM']

    # 训练所有模型（使用网格搜索优化超参数）
    models = trainer.train_all_models(
        X_train_scaled, y_train,
        model_names=model_names,
        use_grid_search=True  # 设置为False可跳过超参数优化，加快训练速度
    )

    # 保存模型
    trainer.save_models(MODEL_DIR)

    # ==================== 步骤3: 模型评估 ====================
    print("\n步骤3: 模型评估")
    print("-"*80)

    evaluator = ModelEvaluator()

    # 评估所有模型
    results = evaluator.evaluate_all_models(models, X_test_scaled, y_test)

    # 打印对比表
    print("\n模型性能对比表:")
    print("-"*80)
    comparison_table = evaluator.get_comparison_table()
    print(comparison_table.to_string(index=False))

    # 保存评估结果
    evaluator.save_results(EVAL_DIR)

    # 绘制混淆矩阵
    print("\n绘制混淆矩阵...")
    evaluator.plot_confusion_matrices(EVAL_DIR)

    # 绘制ROC曲线
    print("\n绘制ROC曲线...")
    evaluator.plot_roc_curves(X_test_scaled, y_test, models, EVAL_DIR)

    # 绘制指标对比图
    print("\n绘制指标对比图...")
    evaluator.plot_metrics_comparison(EVAL_DIR)

    # ==================== 步骤4: 模型对比分析 ====================
    print("\n步骤4: 模型对比分析")
    print("-"*80)

    comparison = ModelComparison(results)

    # 生成总结报告
    comparison.generate_summary_report(EVAL_DIR)

    # 绘制雷达图
    print("\n绘制雷达图...")
    comparison.plot_radar_chart(EVAL_DIR)

    # 绘制热力图
    print("\n绘制热力图...")
    comparison.plot_heatmap(EVAL_DIR)

    # 绘制排名图
    print("\n绘制排名图...")
    comparison.plot_model_ranking('roc_auc', EVAL_DIR)

    # ==================== 完成 ====================
    print("\n"+"="*80)
    print("所有任务完成！")
    print("="*80)
    print(f"模型文件保存在: {MODEL_DIR}")
    print(f"评估结果保存在: {EVAL_DIR}")
    print("="*80)

    # 获取最优模型
    best_model, best_auc = comparison.get_best_model('roc_auc')
    print(f"\n最优模型: {best_model} (ROC-AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()
