"""
模型对比分析
提供模型性能的综合对比和分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelComparison:
    """模型对比分析器"""

    def __init__(self, evaluation_results):
        """
        初始化对比分析器

        Args:
            evaluation_results: 评估结果字典
        """
        self.results = evaluation_results
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def rank_models(self, metric='roc_auc'):
        """
        根据指定指标对模型进行排名

        Args:
            metric: 排名依据的指标

        Returns:
            ranking: 排名结果DataFrame
        """
        ranking_data = []
        for model_name, metrics in self.results.items():
            ranking_data.append({
                'Model': model_name,
                'Score': metrics[metric]
            })

        df = pd.DataFrame(ranking_data)
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)
        df['Rank'] = df.index + 1

        return df[['Rank', 'Model', 'Score']]

    def get_best_model(self, metric='roc_auc'):
        """
        获取最优模型

        Args:
            metric: 评价指标

        Returns:
            best_model_name: 最优模型名称
            best_score: 最优分数
        """
        best_model_name = None
        best_score = -1

        for model_name, metrics in self.results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name

        return best_model_name, best_score

    def plot_radar_chart(self, output_dir=None):
        """
        绘制雷达图对比各模型性能

        Args:
            output_dir: 输出目录
        """
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']

        # 准备数据
        model_names = list(self.results.keys())
        n_metrics = len(metrics)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for idx, model_name in enumerate(model_names):
            values = [self.results[model_name][m] for m in metrics]
            values += values[:1]  # 闭合

            ax.plot(
                angles, values,
                'o-', linewidth=2,
                label=model_name,
                color=colors[idx % len(colors)]
            )
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'radar_chart.pdf'), bbox_inches='tight')
            print(f"雷达图已保存至: {output_dir}")

        plt.show()

    def plot_heatmap(self, output_dir=None):
        """
        绘制性能指标热力图

        Args:
            output_dir: 输出目录
        """
        metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']
        metrics_labels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']

        # 准备数据
        data = []
        model_names = []
        for model_name, model_metrics in self.results.items():
            model_names.append(model_name)
            data.append([model_metrics[m] for m in metrics])

        df = pd.DataFrame(data, columns=metrics_labels, index=model_names)

        # 绘制热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            df, annot=True, fmt='.4f', cmap='YlGnBu',
            cbar_kws={'label': 'Score'},
            linewidths=0.5, linecolor='gray'
        )

        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'performance_heatmap.pdf'), bbox_inches='tight')
            print(f"热力图已保存至: {output_dir}")

        plt.show()

    def generate_summary_report(self, output_dir=None):
        """
        生成模型对比总结报告

        Args:
            output_dir: 输出目录

        Returns:
            report: 报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("模型性能对比总结报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 1. 整体排名
        report_lines.append("1. 模型整体排名（按ROC-AUC）")
        report_lines.append("-" * 60)
        ranking = self.rank_models('roc_auc')
        for _, row in ranking.iterrows():
            report_lines.append(f"  {row['Rank']}. {row['Model']}: {row['Score']:.4f}")
        report_lines.append("")

        # 2. 最优模型
        best_model, best_auc = self.get_best_model('roc_auc')
        report_lines.append("2. 最优模型")
        report_lines.append("-" * 60)
        report_lines.append(f"  模型名称: {best_model}")
        report_lines.append(f"  ROC-AUC: {best_auc:.4f}")

        best_metrics = self.results[best_model]
        report_lines.append(f"  准确率: {best_metrics['accuracy']:.4f}")
        report_lines.append(f"  精确率: {best_metrics['precision']:.4f}")
        report_lines.append(f"  召回率: {best_metrics['recall']:.4f}")
        report_lines.append(f"  特异性: {best_metrics['specificity']:.4f}")
        report_lines.append(f"  F1分数: {best_metrics['f1_score']:.4f}")
        report_lines.append("")

        # 3. 各指标最优模型
        report_lines.append("3. 各指标最优模型")
        report_lines.append("-" * 60)
        metrics = {
            'accuracy': '准确率',
            'precision': '精确率',
            'recall': '召回率',
            'specificity': '特异性',
            'f1_score': 'F1分数',
            'roc_auc': 'ROC-AUC'
        }

        for metric_key, metric_name in metrics.items():
            best_model, best_score = self.get_best_model(metric_key)
            report_lines.append(f"  {metric_name}: {best_model} ({best_score:.4f})")
        report_lines.append("")

        # 4. 模型性能分析
        report_lines.append("4. 模型性能分析")
        report_lines.append("-" * 60)

        for model_name, model_metrics in self.results.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  - 准确率: {model_metrics['accuracy']:.4f}")
            report_lines.append(f"  - 精确率: {model_metrics['precision']:.4f}")
            report_lines.append(f"  - 召回率: {model_metrics['recall']:.4f}")
            report_lines.append(f"  - 特异性: {model_metrics['specificity']:.4f}")
            report_lines.append(f"  - F1分数: {model_metrics['f1_score']:.4f}")
            report_lines.append(f"  - ROC-AUC: {model_metrics['roc_auc']:.4f}")

            # 性能评价
            auc = model_metrics['roc_auc']
            if auc >= 0.9:
                performance = "优秀"
            elif auc >= 0.8:
                performance = "良好"
            elif auc >= 0.7:
                performance = "中等"
            else:
                performance = "较差"

            report_lines.append(f"  - 综合评价: {performance}")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = '\n'.join(report_lines)

        # 保存报告
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'model_comparison_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"对比报告已保存: {report_path}")

        print(report_text)

        return report_text

    def plot_model_ranking(self, metric='roc_auc', output_dir=None):
        """
        绘制模型排名条形图

        Args:
            metric: 排名依据的指标
            output_dir: 输出目录
        """
        ranking = self.rank_models(metric)

        plt.figure(figsize=(10, 6))

        colors = ['#2ca02c' if i == 0 else '#1f77b4' for i in range(len(ranking))]

        bars = plt.barh(
            ranking['Model'],
            ranking['Score'],
            color=colors,
            alpha=0.8,
            edgecolor='black'
        )

        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2.,
                f'{width:.4f}',
                ha='left', va='center',
                fontsize=10,
                fontweight='bold'
            )

        plt.xlabel(metric.upper().replace('_', '-'), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Ranking by {metric.upper().replace("_", "-")}', fontsize=14, fontweight='bold')
        plt.xlim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'model_ranking_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'model_ranking_{metric}.pdf'), bbox_inches='tight')
            print(f"排名图已保存至: {output_dir}")

        plt.show()
