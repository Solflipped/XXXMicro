"""
模型评估器
实现模型性能评估，包括准确率、精确率、召回率、特异性、F1分数、ROC-AUC等指标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import os


class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        """初始化评估器"""
        self.results = {}
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        评估单个模型

        Args:
            model: 训练好的模型
            X_test: 测试集特征
            y_test: 测试集标签
            model_name: 模型名称

        Returns:
            metrics: 评估指标字典
        """
        print(f"\n{'='*70}")
        print(f"评估模型: {model_name}")
        print(f"{'='*70}")

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)  # 敏感性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,  # 敏感性=召回率
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # 打印结果
        print(f"准确率 (Accuracy):    {accuracy:.4f}")
        print(f"精确率 (Precision):   {precision:.4f}")
        print(f"召回率 (Recall):      {recall:.4f}")
        print(f"敏感性 (Sensitivity): {recall:.4f}")
        print(f"特异性 (Specificity): {specificity:.4f}")
        print(f"F1分数 (F1-Score):    {f1:.4f}")
        print(f"ROC-AUC:              {roc_auc:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  TN={tn}, FP={fp}")
        print(f"  FN={fn}, TP={tp}")

        self.results[model_name] = metrics

        return metrics

    def evaluate_all_models(self, models, X_test, y_test):
        """
        评估所有模型

        Args:
            models: 模型字典
            X_test: 测试集特征
            y_test: 测试集标签

        Returns:
            results: 所有模型的评估结果
        """
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)

        return self.results

    def get_comparison_table(self):
        """
        生成模型性能对比表

        Returns:
            df: 对比表DataFrame
        """
        if not self.results:
            return None

        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                '模型': model_name,
                '准确率': f"{metrics['accuracy']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                '敏感性': f"{metrics['sensitivity']:.4f}",
                '特异性': f"{metrics['specificity']:.4f}",
                'F1分数': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })

        df = pd.DataFrame(comparison_data)
        return df

    def plot_confusion_matrices(self, output_dir=None):
        """
        绘制所有模型的混淆矩阵

        Args:
            output_dir: 输出目录
        """
        if not self.results:
            print("没有评估结果")
            return

        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', 'AD'],
                yticklabels=['Control', 'AD'],
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )

            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')

        # 隐藏多余的子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'confusion_matrices.pdf'), bbox_inches='tight')
            print(f"混淆矩阵已保存至: {output_dir}")

        plt.show()

    def plot_roc_curves(self, X_test, y_test, models, output_dir=None):
        """
        绘制所有模型的ROC曲线对比图

        Args:
            X_test: 测试集特征
            y_test: 测试集标签
            models: 模型字典
            output_dir: 输出目录
        """
        plt.figure(figsize=(10, 8))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, (model_name, model) in enumerate(models.items()):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            plt.plot(
                fpr, tpr,
                color=colors[idx % len(colors)],
                lw=2,
                label=f'{model_name} (AUC = {auc:.4f})'
            )

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.5000)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.pdf'), bbox_inches='tight')
            print(f"ROC曲线已保存至: {output_dir}")

        plt.show()

    def plot_metrics_comparison(self, output_dir=None):
        """
        绘制各模型性能指标对比图

        Args:
            output_dir: 输出目录
        """
        if not self.results:
            print("没有评估结果")
            return

        # 准备数据
        metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']

        data = []
        for model_name, metrics in self.results.items():
            row = [model_name] + [metrics[m] for m in metrics_names]
            data.append(row)

        df = pd.DataFrame(data, columns=['Model'] + metrics_labels)

        # 绘制条形图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, metric in enumerate(metrics_labels):
            ax = axes[idx]

            bars = ax.bar(
                df['Model'],
                df[metric],
                color=colors,
                alpha=0.8,
                edgecolor='black'
            )

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.4f}',
                    ha='center', va='bottom',
                    fontsize=9
                )

            ax.set_ylabel(metric, fontsize=11)
            ax.set_ylim([0, 1.1])
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.pdf'), bbox_inches='tight')
            print(f"指标对比图已保存至: {output_dir}")

        plt.show()

    def save_results(self, output_dir):
        """
        保存评估结果

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存对比表
        comparison_table = self.get_comparison_table()
        if comparison_table is not None:
            table_path = os.path.join(output_dir, 'model_comparison.csv')
            comparison_table.to_csv(table_path, index=False, encoding='utf-8')
            print(f"对比表已保存: {table_path}")

        # 保存详细结果
        detailed_results = {}
        for model_name, metrics in self.results.items():
            detailed_results[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity']),
                'f1_score': float(metrics['f1_score']),
                'roc_auc': float(metrics['roc_auc']),
                'confusion_matrix': {
                    'tn': metrics['tn'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn'],
                    'tp': metrics['tp']
                }
            }

        import json
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存: {results_path}")
