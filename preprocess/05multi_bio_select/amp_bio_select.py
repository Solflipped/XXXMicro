"""
AMP 层级两段式流水线：
1) 预处理（独立保存）
2) 生物标志物筛选（独立保存）

预处理策略：删除全零特征 + 零值填充 + log 变换 + 异常值检测(保留)
筛选策略：Mann-Whitney + RF + Elastic Net + 互信息整合
最终保留特征：Top 200
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def convert_numpy_types(obj):
    """Convert numpy/pandas objects into JSON serializable native types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    return obj


class AMPPreprocessor:
    """AMP abundance preprocessor for sparse read-count matrix."""

    def __init__(
        self,
        file_path,
        outlier_method="iqr",
        root_output="/root/XXXMicro/Data/EW-T2D-5-13/amp_preprocessed",
    ):
        self.file_path = file_path
        self.outlier_method = outlier_method
        self.root_output = root_output

        self.raw_data = None
        self.processed_data = None
        self.metadata = {}

        os.makedirs(self.root_output, exist_ok=True)

    def load_raw_data(self):
        """Load the raw AMP abundance table."""
        print("=" * 70)
        print("步骤1: 加载 AMP 原始数据")
        print("=" * 70)

        try:
            self.raw_data = pd.read_csv(self.file_path)
        except Exception as exc:
            print(f"读取数据失败: {exc}")
            return False

        required_cols = {"sample_id", "label"}
        if not required_cols.issubset(set(self.raw_data.columns)):
            print("数据必须包含 'sample_id' 和 'label' 列")
            return False

        print(f"原始数据形状: {self.raw_data.shape}")
        print(f"样本数: {self.raw_data.shape[0]}")
        print(f"原始特征数: {self.raw_data.shape[1] - 2}")
        return True

    def preprocess_data(self):
        """Preprocess AMP data: remove all-zero features + zero imputation + log transform."""
        print("\n" + "=" * 70)
        print("步骤2: AMP 数据预处理")
        print("=" * 70)

        feature_cols = [c for c in self.raw_data.columns if c not in ["sample_id", "label"]]
        if not feature_cols:
            print("未找到可用特征列")
            return False

        features = self.raw_data[feature_cols].copy()
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        features = features.clip(lower=0.0)

        print("\n1. 删除全零特征:")
        zero_ratio_by_feature = (features == 0).sum(axis=0) / len(features)
        all_zero_features = zero_ratio_by_feature[zero_ratio_by_feature == 1].index.tolist()
        if len(all_zero_features) == len(feature_cols):
            print("所有特征均为全零，无法继续处理")
            return False

        if all_zero_features:
            features_filtered = features.drop(columns=all_zero_features).copy()
            print(f"  删除全零特征: {len(all_zero_features)} 个")
        else:
            features_filtered = features.copy()
            print("  未发现全零特征")

        print(f"  过滤前特征数: {len(feature_cols)}")
        print(f"  过滤后特征数: {len(features_filtered.columns)}")

        print("\n2. 零值填充:")
        min_positives = {}
        for col in features_filtered.columns:
            positive_vals = features_filtered[col][features_filtered[col] > 0]
            min_positives[col] = positive_vals.min() / 2 if len(positive_vals) > 0 else 1e-10

        features_filled = features_filtered.copy()
        for col in features_filled.columns:
            features_filled[col] = features_filled[col].apply(
                lambda x, c=col: min_positives[c] if x == 0 else x
            )
        print("  零值填充完成")

        print("\n3. log 变换:")
        features_log = np.log(features_filled)
        log_variance = features_log.var(axis=0)
        non_constant_features = log_variance[log_variance > 0].index.tolist()
        features_log = features_log[non_constant_features]
        print(f"  log 后保留非零方差特征: {len(non_constant_features)}")

        print(f"\n4. 异常值检测 ({self.outlier_method}):")
        values = features_log.values
        if self.outlier_method == "iqr":
            q1 = np.percentile(values, 25, axis=0)
            q3 = np.percentile(values, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers_mask = (values < lower) | (values > upper)
        else:
            mean_vals = np.mean(values, axis=0)
            std_vals = np.std(values, axis=0)
            lower = mean_vals - 3 * std_vals
            upper = mean_vals + 3 * std_vals
            outliers_mask = (values < lower) | (values > upper)

        outlier_count = int(outliers_mask.sum())
        samples_with_outliers = int((outliers_mask.sum(axis=1) > 0).sum())
        print(f"  总异常值数量: {outlier_count}")
        print("  处理策略: 保留异常值")

        self.processed_data = pd.concat(
            [
                self.raw_data[["sample_id", "label"]].reset_index(drop=True),
                features_log.reset_index(drop=True),
            ],
            axis=1,
        )

        raw_file = os.path.join(self.root_output, "amp_raw.csv")
        processed_file = os.path.join(self.root_output, "amp_processed.csv")

        self.raw_data.to_csv(raw_file, index=False, encoding="utf-8")
        self.processed_data.to_csv(processed_file, index=False, encoding="utf-8")

        print(f"原始 AMP 数据已保存: {raw_file}")
        print(f"预处理 AMP 数据已保存: {processed_file}")

        self.metadata = {
            "raw_shape": self.raw_data.shape,
            "processed_shape": self.processed_data.shape,
            "feature_count": len(feature_cols),
            "all_zero_removed": len(all_zero_features),
            "prevalence_passed_feature_count": len(features_filtered.columns),
            "cleaned_feature_count": len(non_constant_features),
            "outlier_method": self.outlier_method,
            "outlier_count": outlier_count,
            "samples_with_outliers": samples_with_outliers,
            "raw_file": raw_file,
            "processed_file": processed_file,
            "output_dir": self.root_output,
            "normalization": "legacy_log_impute",
        }
        return True

    def generate_report(self):
        """Generate preprocessing report and metadata."""
        print("\n" + "=" * 70)
        print("步骤3: 生成 AMP 预处理报告")
        print("=" * 70)

        report_lines = [
            "=" * 80,
            "AMP 数据预处理报告",
            "=" * 80,
            f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"原始文件: {self.file_path}",
            f"原始形状: {self.metadata.get('raw_shape')}",
            f"预处理后形状: {self.metadata.get('processed_shape')}",
            "",
            "参数:",
            f"- outlier_method: {self.metadata.get('outlier_method')}",
            f"- normalization: {self.metadata.get('normalization')}",
            "",
            "统计:",
            f"- 原始特征数: {self.metadata.get('feature_count')}",
            f"- 删除全零特征数: {self.metadata.get('all_zero_removed')}",
            f"- 预处理输入后特征数: {self.metadata.get('prevalence_passed_feature_count')}",
            f"- 清洗后特征数: {self.metadata.get('cleaned_feature_count')}",
            f"- 异常值数量: {self.metadata.get('outlier_count')}",
            f"- 含异常值样本数: {self.metadata.get('samples_with_outliers')}",
            "",
            "输出:",
            f"- 原始保存: {self.metadata.get('raw_file')}",
            f"- 预处理保存: {self.metadata.get('processed_file')}",
        ]

        report_file = os.path.join(self.root_output, "amp_preprocessing_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        metadata_file = os.path.join(self.root_output, "amp_preprocessing_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=2, ensure_ascii=False)

        print(f"预处理报告已保存: {report_file}")
        print(f"预处理元数据已保存: {metadata_file}")
        self.metadata["report_file"] = report_file
        self.metadata["metadata_file"] = metadata_file
        return True

    def run_preprocessing_pipeline(self):
        """Run the full preprocessing pipeline."""
        steps = [
            ("加载原始数据", self.load_raw_data),
            ("预处理数据", self.preprocess_data),
            ("生成预处理报告", self.generate_report),
        ]

        print("开始 AMP 预处理流水线...")
        print("=" * 80)

        for step_name, step_func in steps:
            try:
                ok = step_func()
                if not ok:
                    print(f"\n{step_name} 失败")
                    return False
                print(f"\n{step_name} 完成")
            except Exception as exc:
                print(f"\n{step_name} 出错: {exc}")
                return False

        print("\n" + "=" * 80)
        print("AMP 预处理流水线完成")
        print("=" * 80)
        print(f"预处理输出目录: {self.root_output}")
        return True


class AMPBiomarkerScreener:
    """AMP biomarker screener aligned with the stronger species pipeline."""

    def __init__(
        self,
        preprocessed_metadata,
        root_output="/root/XXXMicro/Data/EW-T2D-5-13/amp_biomarkers",
        top_k=200,
        candidate_pool_size=1000,
        eval_top_k=20,
        redundancy_threshold=0.95,
        export_filename="amp_abundance.csv",
    ):
        self.preprocessed_metadata = preprocessed_metadata
        self.root_output = root_output
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size
        self.eval_top_k = eval_top_k
        self.redundancy_threshold = redundancy_threshold
        self.export_filename = export_filename

        self.processed_data = None
        self.selected_features = []
        self.eval_features = []
        self.ranked_features = []
        self.method_summary = {}
        self.results = {}
        self.performance = None

        os.makedirs(self.root_output, exist_ok=True)

    def _load_processed_data(self):
        processed_file = self.preprocessed_metadata.get("processed_file")
        if not processed_file or not os.path.exists(processed_file):
            raise FileNotFoundError(f"预处理文件不存在: {processed_file}")
        self.processed_data = pd.read_csv(processed_file, encoding="utf-8")
        return processed_file

    @staticmethod
    def _try_fdr_correction(p_values):
        try:
            from statsmodels.stats.multitest import multipletests

            _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
            return p_adj
        except Exception:
            return p_values

    def _mann_whitney_test(self, features, labels):
        ad_features = features[labels == 1]
        control_features = features[labels == 0]

        rows = []
        for col in features.columns:
            try:
                u_stat, p_value = mannwhitneyu(
                    ad_features[col], control_features[col], alternative="two-sided"
                )

                ad_mean = ad_features[col].mean()
                control_mean = control_features[col].mean()
                ad_std = ad_features[col].std()
                control_std = control_features[col].std()
                n_ad = len(ad_features)
                n_ctrl = len(control_features)

                pooled_std = np.sqrt(
                    ((n_ad - 1) * ad_std ** 2 + (n_ctrl - 1) * control_std ** 2)
                    / max((n_ad + n_ctrl - 2), 1)
                )
                cohens_d_raw = (ad_mean - control_mean) / pooled_std if pooled_std != 0 else 0.0
                cohens_d = abs(cohens_d_raw)

                rows.append(
                    {
                        "feature": col,
                        "p_value": p_value,
                        "u_stat": u_stat,
                        "cohens_d": cohens_d,
                        "cohens_d_raw": cohens_d_raw,
                        "mean_diff": ad_mean - control_mean,
                        "ad_mean": ad_mean,
                        "control_mean": control_mean,
                        "direction": "up" if ad_mean > control_mean else "down",
                    }
                )
            except Exception:
                continue

        if not rows:
            empty_all = pd.DataFrame(
                columns=[
                    "feature",
                    "p_value",
                    "u_stat",
                    "cohens_d",
                    "cohens_d_raw",
                    "mean_diff",
                    "ad_mean",
                    "control_mean",
                    "direction",
                    "p_adj",
                ]
            )
            empty_sig = pd.DataFrame(columns=empty_all.columns)
            return empty_all, empty_sig

        all_df = pd.DataFrame(rows)
        all_df["p_adj"] = self._try_fdr_correction(all_df["p_value"].values)

        significant_df = all_df[all_df["p_adj"] < 0.05].copy()
        significant_df = significant_df.sort_values("cohens_d", ascending=False).reset_index(drop=True)

        if significant_df.empty:
            significant_df = all_df[all_df["p_value"] < 0.05].copy()
            significant_df = significant_df.sort_values("cohens_d", ascending=False).reset_index(drop=True)

        return all_df, significant_df

    def _random_forest_importance(self, features, labels, top_n=None):
        rf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        rf.fit(features, labels)
        importance_df = (
            pd.DataFrame({"feature": features.columns, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        top_n = top_n or self.candidate_pool_size
        return importance_df, importance_df.head(top_n)["feature"].tolist(), rf

    def _elastic_net_selection(self, features, labels, top_n=None):
        elastic_net = LogisticRegression(
            penalty="elasticnet",
            C=0.5,
            l1_ratio=0.3,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            max_iter=5000,
        )
        elastic_net.fit(features, labels)

        df = pd.DataFrame(
            {
                "feature": features.columns,
                "coefficient": elastic_net.coef_[0],
                "abs_coefficient": np.abs(elastic_net.coef_[0]),
            }
        )
        df = df[df["coefficient"] != 0].sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        top_n = top_n or self.candidate_pool_size
        return df, df.head(top_n)["feature"].tolist(), elastic_net

    def _mutual_information_analysis(self, features, labels, top_n=None):
        mi_scores = mutual_info_classif(features, labels, random_state=42)
        mi_df = (
            pd.DataFrame({"feature": features.columns, "mi_score": mi_scores})
            .sort_values("mi_score", ascending=False)
            .reset_index(drop=True)
        )
        top_n = top_n or self.candidate_pool_size
        return mi_df, mi_df.head(top_n)["feature"].tolist()

    @staticmethod
    def _build_rank_score_map(feature_list):
        if not feature_list:
            return {}
        if len(feature_list) == 1:
            return {feature_list[0]: 1.0}

        denom = len(feature_list) - 1
        return {feature: 1.0 - (rank / denom) for rank, feature in enumerate(feature_list)}

    def _integrate_biomarkers(self, method_features_dict):
        all_features = set()
        for features in method_features_dict.values():
            all_features.update(features[: self.candidate_pool_size])

        rank_score_maps = {
            method: self._build_rank_score_map(features[: self.candidate_pool_size])
            for method, features in method_features_dict.items()
        }

        rows = []
        for feature in all_features:
            feature_method_scores = {}
            score = 0.0
            count = 0

            for method, features in method_features_dict.items():
                if feature in features:
                    method_score = rank_score_maps[method].get(feature, 0.0)
                    feature_method_scores[method] = method_score
                    score += method_score
                    count += 1
                else:
                    feature_method_scores[method] = 0.0

            rows.append(
                {
                    "feature": feature,
                    "n_methods": count,
                    "score": score,
                    "mw_rank_score": feature_method_scores.get("mann_whitney", 0.0),
                    "rf_rank_score": feature_method_scores.get("random_forest", 0.0),
                    "elastic_net_rank_score": feature_method_scores.get("elastic_net", 0.0),
                    "mi_rank_score": feature_method_scores.get("mutual_info", 0.0),
                    "in_mann_whitney": 1 if feature in method_features_dict.get("mann_whitney", []) else 0,
                    "in_random_forest": 1 if feature in method_features_dict.get("random_forest", []) else 0,
                    "in_elastic_net": 1 if feature in method_features_dict.get("elastic_net", []) else 0,
                    "in_mutual_info": 1 if feature in method_features_dict.get("mutual_info", []) else 0,
                }
            )

        integration_df = pd.DataFrame(rows)
        integration_df = integration_df.sort_values(["n_methods", "score"], ascending=[False, False]).reset_index(drop=True)
        return integration_df

    def _select_nonredundant_features(self, ranked_df, features_df, top_k):
        selected_features = []
        skipped_features = []

        for feature in ranked_df["feature"].tolist():
            if len(selected_features) >= top_k:
                break

            if not selected_features:
                selected_features.append(feature)
                continue

            corr_values = features_df[selected_features].corrwith(features_df[feature]).abs()
            max_corr = corr_values.max() if not corr_values.empty else 0.0
            max_corr = 0.0 if pd.isna(max_corr) else float(max_corr)

            if max_corr < self.redundancy_threshold:
                selected_features.append(feature)
            else:
                skipped_features.append(
                    {
                        "feature": feature,
                        "max_abs_correlation": max_corr,
                    }
                )

        if len(selected_features) < top_k:
            for feature in ranked_df["feature"].tolist():
                if feature in selected_features:
                    continue
                selected_features.append(feature)
                if len(selected_features) >= top_k:
                    break

        return selected_features, pd.DataFrame(skipped_features)

    def _resolve_export_file(self):
        base_dir = os.path.dirname(self.preprocessed_metadata.get("output_dir", self.root_output))
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, self.export_filename)

    def select_biomarkers(self):
        print("\n" + "=" * 70)
        print(f"步骤1: AMP 生物标志物筛选 (Top {self.top_k})")
        print("=" * 70)

        processed_file = self._load_processed_data()
        print(f"使用预处理文件: {processed_file}")

        feature_cols = [c for c in self.processed_data.columns if c not in ["sample_id", "label"]]
        X = self.processed_data[feature_cols]
        y = self.processed_data["label"]

        print(f"样本数: {len(X)}")
        print(f"特征数: {len(feature_cols)}")

        print("\n1. Mann-Whitney U 检验...")
        mw_df, mw_sig_df = self._mann_whitney_test(X, y)
        print("2. 随机森林重要性...")
        rf_df, rf_top_features, _ = self._random_forest_importance(X, y, top_n=self.candidate_pool_size)
        print("3. Elastic Net 回归...")
        elastic_net_df, elastic_net_top_features, _ = self._elastic_net_selection(
            X, y, top_n=self.candidate_pool_size
        )
        print("4. 互信息分析...")
        mi_df, mi_top_features = self._mutual_information_analysis(X, y, top_n=self.candidate_pool_size)

        print("5. 多方法整合打分...")
        method_features = {
            "mann_whitney": mw_sig_df.head(self.candidate_pool_size)["feature"].tolist() if not mw_sig_df.empty else [],
            "random_forest": rf_top_features,
            "elastic_net": elastic_net_top_features,
            "mutual_info": mi_top_features,
        }
        integration_df = self._integrate_biomarkers(method_features)
        self.ranked_features = integration_df["feature"].tolist()
        self.selected_features, skipped_df = self._select_nonredundant_features(integration_df, X, self.top_k)
        self.eval_features = self.selected_features[: min(self.eval_top_k, len(self.selected_features))]

        final_df = integration_df[integration_df["feature"].isin(self.selected_features)].copy()
        final_df["rank"] = final_df["feature"].apply(lambda x: self.selected_features.index(x) + 1)
        final_df = final_df.sort_values("rank").reset_index(drop=True)

        self.method_summary = {
            "mann_whitney_total": len(mw_df),
            "mann_whitney_significant": len(mw_sig_df),
            "mann_whitney_candidates": len(method_features["mann_whitney"]),
            "random_forest_total": len(rf_df),
            "random_forest_candidates": len(rf_top_features),
            "elastic_net_total": len(elastic_net_df),
            "elastic_net_candidates": len(elastic_net_top_features),
            "mutual_info_total": len(mi_df),
            "mutual_info_candidates": len(mi_top_features),
            "integration_candidates": len(integration_df),
            "redundancy_skipped": len(skipped_df),
            "selected_top_k": len(self.selected_features),
            "eval_top_k": len(self.eval_features),
        }

        self.results = {
            "mann_whitney": mw_df,
            "mann_whitney_significant": mw_sig_df,
            "random_forest": rf_df,
            "elastic_net": elastic_net_df,
            "mutual_info": mi_df,
            "integration_all": integration_df,
            "redundancy_skipped": skipped_df,
            "final_biomarkers": final_df,
            "selected_features_topk": self.selected_features,
            "eval_features_topk": self.eval_features,
            "method_summary": self.method_summary,
        }

        print(f"导出特征数: {len(self.selected_features)}")
        print(f"评估特征数: {len(self.eval_features)}")
        print("各方法筛选数量:")
        for key, value in self.method_summary.items():
            print(f"  {key}: {value}")

        mw_df.to_csv(os.path.join(self.root_output, "amp_mann_whitney.csv"), index=False, encoding="utf-8")
        mw_sig_df.to_csv(
            os.path.join(self.root_output, "amp_mann_whitney_significant.csv"),
            index=False,
            encoding="utf-8",
        )
        rf_df.to_csv(os.path.join(self.root_output, "amp_rf_importance.csv"), index=False, encoding="utf-8")
        elastic_net_df.to_csv(
            os.path.join(self.root_output, "amp_elastic_net_coef.csv"),
            index=False,
            encoding="utf-8",
        )
        mi_df.to_csv(os.path.join(self.root_output, "amp_mutual_info.csv"), index=False, encoding="utf-8")
        integration_df.to_csv(os.path.join(self.root_output, "amp_integration_all.csv"), index=False, encoding="utf-8")
        skipped_df.to_csv(
            os.path.join(self.root_output, "amp_redundancy_skipped.csv"),
            index=False,
            encoding="utf-8",
        )
        final_df.to_csv(os.path.join(self.root_output, "amp_biomarkers.csv"), index=False, encoding="utf-8")
        with open(os.path.join(self.root_output, "amp_method_summary.json"), "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.method_summary), f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.root_output, "amp_biomarker_list.txt"), "w", encoding="utf-8") as f:
            f.write("AMP 层级生物标志物列表\n")
            f.write("=" * 50 + "\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i}. {feat}\n")

        return True

    def evaluate_performance(self):
        """Evaluate Top20 features with LOOCV logistic regression."""
        print("\n" + "=" * 70)
        print(f"步骤2: 性能评估（Top {self.eval_top_k} + 逻辑回归 + 留一法CV）")
        print("=" * 70)

        if not self.eval_features:
            print("未筛选到特征，跳过性能评估")
            return None

        X_selected = self.processed_data[self.eval_features]
        y = self.processed_data["label"]

        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import LeaveOneOut

        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        loo = LeaveOneOut()

        y_true = []
        y_pred = []
        y_prob = []

        for train_idx, test_idx in loo.split(X_selected):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            clf.fit(X_train, y_train)
            y_true.append(y_test.values[0])
            y_pred.append(clf.predict(X_test)[0])
            y_prob.append(clf.predict_proba(X_test)[0][1])

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        self.performance = {
            "feature_count": len(self.eval_features),
            "export_feature_count": len(self.selected_features),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
            "selected_features": self.eval_features,
            "export_features": self.selected_features,
        }

        print(f"准确率: {self.performance['accuracy']:.4f}")
        print(f"精确率: {self.performance['precision']:.4f}")
        print(f"召回率: {self.performance['recall']:.4f}")
        print(f"敏感性: {self.performance['sensitivity']:.4f}")
        print(f"特异性: {self.performance['specificity']:.4f}")
        print(f"F1分数: {self.performance['f1_score']:.4f}")
        print(f"ROC-AUC: {self.performance['roc_auc']:.4f}")

        with open(os.path.join(self.root_output, "amp_performance.json"), "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.performance), f, indent=2, ensure_ascii=False)

        return self.performance

    def generate_filtered_data(self):
        """Generate the final filtered AMP abundance table."""
        print("\n" + "=" * 70)
        print(f"步骤3: 生成筛选后数据（Top {self.top_k}）")
        print("=" * 70)

        if not self.selected_features:
            print("未筛选到特征，跳过 filtered 文件生成")
            return False

        filtered_data = pd.concat(
            [
                self.processed_data[["sample_id", "label"]],
                self.processed_data[self.selected_features],
            ],
            axis=1,
        )

        filtered_file = self._resolve_export_file()
        filtered_data.to_csv(filtered_file, index=False, encoding="utf-8")
        print(f"筛选后数据形状: {filtered_data.shape}")
        print(f"筛选后数据已保存: {filtered_file}")
        return True

    def generate_summary_report(self):
        """Generate the AMP screening summary report."""
        report_lines = [
            "=" * 80,
            "AMP 层级生物标志物筛选报告",
            "=" * 80,
            f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"预处理输入: {self.preprocessed_metadata.get('processed_file')}",
            f"导出特征数量: {len(self.selected_features)}",
            f"评估特征数量: {len(self.eval_features)}",
            f"去冗余阈值: |r| < {self.redundancy_threshold}",
            "",
        ]

        if self.method_summary:
            report_lines.append("各方法筛选数量:")
            for key, value in self.method_summary.items():
                report_lines.append(f"- {key}: {value}")
            report_lines.append("")

        if self.eval_features:
            report_lines.append("Top 20 标志物:")
            for idx, feat in enumerate(self.eval_features[:20], 1):
                report_lines.append(f"{idx}. {feat}")
            report_lines.append("")

        if self.performance:
            report_lines.extend(
                [
                    "性能指标:",
                    f"- Accuracy: {self.performance['accuracy']:.4f}",
                    f"- Precision: {self.performance['precision']:.4f}",
                    f"- Recall: {self.performance['recall']:.4f}",
                    f"- F1: {self.performance['f1_score']:.4f}",
                    f"- ROC-AUC: {self.performance['roc_auc']:.4f}",
                ]
            )

        report_file = os.path.join(self.root_output, "amp_screening_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"筛选报告已保存: {report_file}")
        return True

    def run_screening_pipeline(self):
        """Run the full screening pipeline."""
        print("开始 AMP 生物标志物筛选流水线...")
        print("=" * 80)

        steps = [
            ("筛选标志物", self.select_biomarkers),
            ("性能评估", self.evaluate_performance),
            ("生成筛选后数据", self.generate_filtered_data),
            ("生成筛选报告", self.generate_summary_report),
        ]

        for step_name, step_func in steps:
            try:
                result = step_func()
                if result is False:
                    print(f"\n{step_name} 失败")
                    return False
                print(f"\n{step_name} 完成")
            except Exception as exc:
                print(f"\n{step_name} 出错: {exc}")
                return False

        print("\n" + "=" * 80)
        print("AMP 筛选流水线完成")
        print("=" * 80)
        print(f"筛选输出目录: {self.root_output}")
        return True


def main():
    """Main: preprocess first, then screen biomarkers."""
    raw_data_file = "/root/XXXMicro/Data/EW-T2D-5-13/merged_amp.csv"
    preprocess_output = "/root/XXXMicro/Data/EW-T2D-5-13/amp_preprocessed"
    screening_output = "/root/XXXMicro/Data/EW-T2D-5-13/amp_biomarkers"

    if not os.path.exists(raw_data_file):
        print(f"错误: 原始数据文件不存在 - {raw_data_file}")
        return

    preprocessor = AMPPreprocessor(
        file_path=raw_data_file,
        outlier_method="iqr",
        root_output=preprocess_output,
    )

    if not preprocessor.run_preprocessing_pipeline():
        print("AMP 预处理失败")
        return

    screener = AMPBiomarkerScreener(
        preprocessed_metadata=preprocessor.metadata,
        root_output=screening_output,
        top_k=200,
        candidate_pool_size=500,
        eval_top_k=20,
        redundancy_threshold=0.95,
        export_filename="amp_abundance.csv",
    )

    if not screener.run_screening_pipeline():
        print("AMP 筛选失败")
        return

    print("\n分析完成")
    print(f"预处理结果目录: {preprocess_output}")
    print(f"筛选结果目录: {screening_output}")


if __name__ == "__main__":
    main()
