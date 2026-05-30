"""
Species 层级两段式流水线：
1) 预处理（独立保存）
2) 生物标志物筛选（独立保存）

预处理策略：提取 species + 低检出特征过滤 + 零值填充 + 异常值检测(保留) + CLR
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
    """将 numpy/pandas 类型转换为可 JSON 序列化的原生类型。"""
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


class SpeciesPreprocessor:
    """仅针对 species 层级的预处理器。"""

    def __init__(
        self,
        file_path,
        min_nonzero_ratio=0.5,
        outlier_method="iqr",
        root_output="/root/XXXMicro/Data/AD/species_preprocessed",
    ):
        self.file_path = file_path
        self.min_nonzero_ratio = min_nonzero_ratio
        self.outlier_method = outlier_method
        self.root_output = root_output

        self.raw_data = None
        self.species_data = None
        self.processed_data = None
        self.metadata = {}

        os.makedirs(self.root_output, exist_ok=True)

    def load_raw_data(self):
        """加载原始 taxonomy 数据。"""
        print("=" * 70)
        print("步骤1: 加载原始 taxonomy 数据")
        print("=" * 70)

        try:
            self.raw_data = pd.read_csv(self.file_path, sep=",")
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

    def extract_species_level(self):
        """从多层级 taxonomy 中提取 species 层级并聚合。"""
        print("\n" + "=" * 70)
        print("步骤2: 提取 species 层级")
        print("=" * 70)

        feature_cols = [c for c in self.raw_data.columns if c not in ["sample_id", "label"]]
        species_map = {}

        for feature_col in feature_cols:
            parts = str(feature_col).split("|")
            species_name = None
            for part in parts:
                if part.startswith("s__"):
                    species_name = part
                    break

            if species_name:
                if species_name not in species_map:
                    species_map[species_name] = []
                species_map[species_name].append(feature_col)

        if not species_map:
            print("未识别到 species 层级特征")
            return False

        species_series_list = []
        species_names = []
        for species_name, cols in species_map.items():
            species_series_list.append(self.raw_data[cols].sum(axis=1))
            species_names.append(species_name)

        species_df = pd.concat(species_series_list, axis=1)
        species_df.columns = species_names

        self.species_data = pd.concat(
            [
                self.raw_data[["sample_id", "label"]].reset_index(drop=True),
                species_df.reset_index(drop=True),
            ],
            axis=1,
        )

        print(f"识别到 species 数量: {len(species_names)}")
        print(f"species 层级数据形状: {self.species_data.shape}")
        return True

    def preprocess_species(self):
        """预处理 species 数据：过滤 + 零值填充 + 异常值检测 + CLR。"""
        print("\n" + "=" * 70)
        print("步骤3: species 数据预处理")
        print("=" * 70)

        feature_cols = [c for c in self.species_data.columns if c not in ["sample_id", "label"]]
        if not feature_cols:
            print("未找到 species 特征列")
            return False

        features = self.species_data[feature_cols].copy()

        print("\n1. 缺失值检查与处理:")
        # 统计零值
        zero_ratio_by_feature = (features == 0).sum() / len(features)

        # 删除全零特征（与 multi_bio_select.py 保持一致）
        all_zero_features = zero_ratio_by_feature[zero_ratio_by_feature == 1].index.tolist()
        if all_zero_features:
            features_filtered = features.drop(columns=all_zero_features)
            print(f"  删除全零特征: {len(all_zero_features)} 个")
        else:
            features_filtered = features.copy()

        print(f"  过滤前特征数: {len(feature_cols)}")
        print(f"  过滤后特征数: {len(features_filtered.columns)}")

        print("\n2. 零值填充:")
        # 计算每列的最小正值
        min_positives = {}
        for col in features_filtered.columns:
            positive_vals = features_filtered[col][features_filtered[col] > 0]
            if len(positive_vals) > 0:
                min_positives[col] = positive_vals.min() / 2
            else:
                min_positives[col] = 1e-10

        # 填充零值
        features_filled = features_filtered.copy()
        for col in features_filled.columns:
            features_filled[col] = features_filled[col].apply(
                lambda x: min_positives[col] if x == 0 else x
            )
        print("  零值填充完成")

        print(f"\n2. 异常值检测 ({self.outlier_method}):")
        if self.outlier_method == "iqr":
            Q1 = np.percentile(features_filled, 25, axis=0)
            Q3 = np.percentile(features_filled, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_mask = (features_filled < lower_bound) | (features_filled > upper_bound)
        else:  # 3sigma
            mean_vals = np.mean(features_filled, axis=0)
            std_vals = np.std(features_filled, axis=0)
            lower_bound = mean_vals - 3 * std_vals
            upper_bound = mean_vals + 3 * std_vals
            outliers_mask = (features_filled < lower_bound) | (features_filled > upper_bound)

        outlier_count = outliers_mask.sum().sum()
        sample_outlier_count = outliers_mask.sum(axis=1)
        samples_with_outliers = sample_outlier_count[sample_outlier_count > 0].count()

        print(f"  总异常值数量: {outlier_count}")
        print(f"  处理策略: 保留异常值（生物学意义）")

        print(f"\n3. CLR转换:")
        # 添加伪计数确保所有值>0
        features_positive = features_filled + 1e-10

        # 计算几何均值
        geometric_mean = np.exp(np.mean(np.log(features_positive.values), axis=1))

        # CLR转换
        features_clr = np.log(features_positive.values / geometric_mean[:, np.newaxis])
        features_clr_df = pd.DataFrame(
            features_clr,
            columns=features_filled.columns,
            index=features_filled.index
        )

        # 构建最终的预处理数据
        self.processed_data = pd.concat([
            self.species_data[['sample_id', 'label']].reset_index(drop=True),
            features_clr_df.reset_index(drop=True)
        ], axis=1)

        # 4. 保存该层级的预处理数据
        raw_file = os.path.join(self.root_output, "species_raw.csv")
        processed_file = os.path.join(self.root_output, "species_processed.csv")

        self.species_data.to_csv(raw_file, index=False, encoding="utf-8")
        self.processed_data.to_csv(processed_file, index=False, encoding="utf-8")

        print(f"原始 species 数据已保存: {raw_file}")
        print(f"预处理 species 数据已保存: {processed_file}")

        self.metadata = {
            "raw_shape": self.species_data.shape,
            "processed_shape": self.processed_data.shape,
            "feature_count": len(feature_cols),
            "cleaned_feature_count": len(features_filtered.columns),
            "min_nonzero_ratio": self.min_nonzero_ratio,
            "outlier_method": self.outlier_method,
            "outlier_count": int(outlier_count),
            "samples_with_outliers": int(samples_with_outliers),
            "raw_file": raw_file,
            "processed_file": processed_file,
            "output_dir": self.root_output,
            "level": "species",
        }
        return True

    def generate_report(self):
        """生成 species 预处理报告。"""
        print("\n" + "=" * 70)
        print("步骤4: 生成 species 预处理报告")
        print("=" * 70)

        report_lines = [
            "=" * 80,
            "Species 层级预处理报告",
            "=" * 80,
            f"处理时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"原始文件: {self.file_path}",
            f"原始形状: {self.metadata.get('raw_shape')}",
            f"预处理后形状: {self.metadata.get('processed_shape')}",
            "",
            "参数:",
            f"- min_nonzero_ratio: {self.metadata.get('min_nonzero_ratio')}",
            f"- outlier_method: {self.metadata.get('outlier_method')}",
            f"- normalization: CLR",
            "",
            "统计:",
            f"- 原始特征数: {self.metadata.get('feature_count')}",
            f"- 清洗后特征数: {self.metadata.get('cleaned_feature_count')}",
            f"- 异常值数量: {self.metadata.get('outlier_count')}",
            f"- 含异常值样本数: {self.metadata.get('samples_with_outliers')}",
            "",
            "输出:",
            f"- 原始保存: {self.metadata.get('raw_file')}",
            f"- 预处理保存: {self.metadata.get('processed_file')}",
        ]

        report_file = os.path.join(self.root_output, "species_preprocessing_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        metadata_file = os.path.join(self.root_output, "species_preprocessing_metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.metadata), f, indent=2, ensure_ascii=False)

        print(f"预处理报告已保存: {report_file}")
        print(f"预处理元数据已保存: {metadata_file}")
        self.metadata["report_file"] = report_file
        self.metadata["metadata_file"] = metadata_file
        return True

    def run_preprocessing_pipeline(self):
        """执行完整 species 预处理流水线。"""
        steps = [
            ("加载原始数据", self.load_raw_data),
            ("提取 species 层级", self.extract_species_level),
            ("预处理 species 数据", self.preprocess_species),
            ("生成预处理报告", self.generate_report),
        ]

        print("开始 species 预处理流水线...")
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
        print("species 预处理流水线完成")
        print("=" * 80)
        print(f"预处理输出目录: {self.root_output}")
        return True


class SpeciesBiomarkerScreener:
    """species 生物标志物筛选器。"""

    def __init__(
        self,
        preprocessed_metadata,
        root_output="/root/XXXMicro/Data/AD/species_biomarkers",
        top_k=200,
        candidate_pool_size=300,
        eval_top_k=20,
        redundancy_threshold=0.85,
        export_filename="species_abundance.csv",
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
                cohens_d = abs((ad_mean - control_mean) / pooled_std) if pooled_std != 0 else 0

                rows.append(
                    {
                        "feature": col,
                        "p_value": p_value,
                        "u_stat": u_stat,
                        "cohens_d": cohens_d,
                        "cohens_d_raw": (ad_mean - control_mean) / pooled_std if pooled_std != 0 else 0,
                        "log2_fc": np.log2(ad_mean / control_mean)
                        if (control_mean > 0 and ad_mean > 0)
                        else 0,
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
                    "log2_fc",
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
            C=0.1,
            l1_ratio=0.5,
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
        """将排序列表转换为统一的 0~1 排名分数。"""
        if not feature_list:
            return {}
        if len(feature_list) == 1:
            return {feature_list[0]: 1.0}

        denom = len(feature_list) - 1
        return {
            feature: 1.0 - (rank / denom)
            for rank, feature in enumerate(feature_list)
        }

    def _integrate_biomarkers(self, method_features_dict, mw_df, rf_df, elastic_net_df, mi_df):
        """按统一排名分数整合多方法结果。"""
        all_features = set()
        for features in method_features_dict.values():
            all_features.update(features[: self.candidate_pool_size])

        rank_score_maps = {
            method: self._build_rank_score_map(features[: self.candidate_pool_size])
            for method, features in method_features_dict.items()
        }

        feature_scores = {}
        feature_method_count = {}
        feature_method_scores = {}

        for feature in all_features:
            count = 0
            score = 0
            method_scores = {}

            # Mann-Whitney
            if feature in method_features_dict['mann_whitney']:
                count += 1
                method_score = rank_score_maps["mann_whitney"].get(feature, 0.0)
                score += method_score
                method_scores["mann_whitney"] = method_score
            else:
                method_scores["mann_whitney"] = 0.0

            # 随机森林
            if feature in method_features_dict['random_forest']:
                count += 1
                method_score = rank_score_maps["random_forest"].get(feature, 0.0)
                score += method_score
                method_scores["random_forest"] = method_score
            else:
                method_scores["random_forest"] = 0.0

            # Elastic Net
            if feature in method_features_dict["elastic_net"]:
                count += 1
                method_score = rank_score_maps["elastic_net"].get(feature, 0.0)
                score += method_score
                method_scores["elastic_net"] = method_score
            else:
                method_scores["elastic_net"] = 0.0

            # 互信息
            if feature in method_features_dict["mutual_info"]:
                count += 1
                method_score = rank_score_maps["mutual_info"].get(feature, 0.0)
                score += method_score
                method_scores["mutual_info"] = method_score
            else:
                method_scores["mutual_info"] = 0.0

            feature_method_count[feature] = count
            feature_scores[feature] = score
            feature_method_scores[feature] = method_scores

        integration_df = pd.DataFrame([
            {
                'feature': f,
                'n_methods': feature_method_count[f],
                'score': feature_scores[f],
                'mw_rank_score': feature_method_scores[f]["mann_whitney"],
                'rf_rank_score': feature_method_scores[f]["random_forest"],
                'elastic_net_rank_score': feature_method_scores[f]["elastic_net"],
                'mi_rank_score': feature_method_scores[f]["mutual_info"],
            }
            for f in all_features
        ])

        for method, features in method_features_dict.items():
            integration_df[f'in_{method}'] = integration_df['feature'].apply(
                lambda x: 1 if x in features else 0
            )

        integration_df = (
            integration_df.sort_values(['n_methods', 'score'], ascending=[False, False])
            .reset_index(drop=True)
        )
        return integration_df

    def _select_nonredundant_features(self, ranked_df, features_df, top_k):
        """按综合排序贪心去冗余，避免高相关特征大量重复。"""
        selected_features = []
        skipped_features = []

        for feature in ranked_df["feature"].tolist():
            if len(selected_features) >= top_k:
                break

            if not selected_features:
                selected_features.append(feature)
                continue

            corr_values = features_df[selected_features].corrwith(features_df[feature]).abs()
            max_corr = corr_values.max() if not corr_values.empty else 0
            max_corr = 0 if pd.isna(max_corr) else float(max_corr)

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
        """从预处理文件加载并执行筛选。"""
        print("\n" + "=" * 70)
        print(f"步骤1: species 生物标志物筛选 (Top {self.top_k})")
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
        rf_df, rf_top_features, _ = self._random_forest_importance(
            X, y, top_n=self.candidate_pool_size
        )
        print("3. Elastic Net 回归...")
        elastic_net_df, elastic_net_top_features, _ = self._elastic_net_selection(
            X, y, top_n=self.candidate_pool_size
        )
        print("4. 互信息分析...")
        mi_df, mi_top_features = self._mutual_information_analysis(
            X, y, top_n=self.candidate_pool_size
        )

        print("5. 多方法整合打分...")
        method_features = {
            "mann_whitney": (
                mw_sig_df.head(self.candidate_pool_size)["feature"].tolist() if not mw_sig_df.empty else []
            ),
            "random_forest": rf_top_features,
            "elastic_net": elastic_net_top_features,
            "mutual_info": mi_top_features,
        }
        integration_df = self._integrate_biomarkers(
            method_features, mw_df, rf_df, elastic_net_df, mi_df
        )
        self.ranked_features = integration_df["feature"].tolist()
        self.selected_features, skipped_df = self._select_nonredundant_features(
            integration_df, X, self.top_k
        )
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

        mw_df.to_csv(os.path.join(self.root_output, "species_mann_whitney.csv"), index=False, encoding="utf-8")
        mw_sig_df.to_csv(
            os.path.join(self.root_output, "species_mann_whitney_significant.csv"),
            index=False,
            encoding="utf-8",
        )
        rf_df.to_csv(os.path.join(self.root_output, "species_rf_importance.csv"), index=False, encoding="utf-8")
        elastic_net_df.to_csv(
            os.path.join(self.root_output, "species_elastic_net_coef.csv"),
            index=False,
            encoding="utf-8",
        )
        mi_df.to_csv(os.path.join(self.root_output, "species_mutual_info.csv"), index=False, encoding="utf-8")
        integration_df.to_csv(os.path.join(self.root_output, "species_integration_all.csv"), index=False, encoding="utf-8")
        skipped_df.to_csv(
            os.path.join(self.root_output, "species_redundancy_skipped.csv"),
            index=False,
            encoding="utf-8",
        )
        final_df.to_csv(os.path.join(self.root_output, "species_biomarkers.csv"), index=False, encoding="utf-8")
        with open(os.path.join(self.root_output, "species_method_summary.json"), "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.method_summary), f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.root_output, "species_biomarker_list.txt"), "w", encoding="utf-8") as f:
            f.write("Species 层级生物标志物列表\n")
            f.write("=" * 50 + "\n\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i}. {feat.replace('s__', '')}\n")

        return True

    def evaluate_performance(self):
        """逻辑回归 + 留一法交叉验证评估（与 multi_bio_select.py 保持一致）"""
        print("\n" + "=" * 70)
        print(f"步骤2: 性能评估（Top {self.eval_top_k} + 逻辑回归 + 留一法CV）")
        print("=" * 70)

        if not self.eval_features:
            print("未筛选到特征，跳过性能评估")
            return None

        X_selected = self.processed_data[self.eval_features]
        y = self.processed_data["label"]

        from sklearn.model_selection import LeaveOneOut
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

        clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
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

        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

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

        with open(os.path.join(self.root_output, "species_performance.json"), "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(self.performance), f, indent=2, ensure_ascii=False)

        return self.performance

    def generate_filtered_data(self):
        """生成筛选后数据。"""
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
        """输出筛选汇总报告。"""
        report_lines = [
            "=" * 80,
            "Species 层级生物标志物筛选报告",
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
                report_lines.append(f"{idx}. {feat.replace('s__', '')}")
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

        report_file = os.path.join(self.root_output, "species_screening_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"筛选报告已保存: {report_file}")
        return True

    def run_screening_pipeline(self):
        """执行完整筛选流水线。"""
        print("开始 species 生物标志物筛选流水线...")
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
        print("species 筛选流水线完成")
        print("=" * 80)
        print(f"筛选输出目录: {self.root_output}")
        return True


def main():
    """主函数：先预处理，再筛选。"""
    RAW_DATA_FILE = "/root/XXXMicro/Data/EW-T2D-5-13/merged_species.csv"
    PREPROCESS_OUTPUT = "/root/XXXMicro/Data/EW-T2D-5-13/species_preprocessed"
    SCREENING_OUTPUT = "/root/XXXMicro/Data/EW-T2D-5-13/species_biomarkers"

    if not os.path.exists(RAW_DATA_FILE):
        print(f"错误: 原始数据文件不存在 - {RAW_DATA_FILE}")
        return

    preprocessor = SpeciesPreprocessor(
        file_path=RAW_DATA_FILE,
        min_nonzero_ratio=0.5,
        outlier_method="iqr",
        root_output=PREPROCESS_OUTPUT,
    )

    if not preprocessor.run_preprocessing_pipeline():
        print("species 预处理失败")
        return

    screener = SpeciesBiomarkerScreener(
        preprocessed_metadata=preprocessor.metadata,
        root_output=SCREENING_OUTPUT,
        top_k=200,
        candidate_pool_size=500,
        eval_top_k=20,
        redundancy_threshold=0.85,
        export_filename="species_abundance.csv",
    )

    if not screener.run_screening_pipeline():
        print("species 筛选失败")
        return

    print("\n分析完成")
    print(f"预处理结果目录: {PREPROCESS_OUTPUT}")
    print(f"筛选结果目录: {SCREENING_OUTPUT}")


if __name__ == "__main__":
    main()
