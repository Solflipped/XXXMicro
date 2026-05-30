import pandas as pd
import numpy as np
import warnings
import os
import json
from collections import Counter
from scipy import stats
from scipy.stats import mannwhitneyu, pointbiserialr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# 全局配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 定义分类层级映射
TAXONOMY_CONFIG = {
    'order': {
        'prefix': 'o__', 
        'name': 'Order', 
        'folder': '01_order_level',
        'description': '目层级'
    },
    'family': {
        'prefix': 'f__', 
        'name': 'Family', 
        'folder': '02_family_level',
        'description': '科层级'
    },
    'genus': {
        'prefix': 'g__', 
        'name': 'Genus', 
        'folder': '03_genus_level',
        'description': '属层级'
    },
    'species': {
        'prefix': 's__', 
        'name': 'Species', 
        'folder': '04_species_level',
        'description': '种层级'
    }
}

# ========== JSON序列化辅助函数 ==========
def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，解决JSON序列化问题"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        return obj

class MultiLevelMicrobiomePreprocessor:
    """多层级微生物组数据预处理器"""
    
    def __init__(self, file_path, min_nonzero_ratio=0.5, outlier_method='iqr',
                 root_output="D:/project/biomarker/multilevel_preprocessed"):
        self.file_path = file_path
        self.min_nonzero_ratio = min_nonzero_ratio
        self.outlier_method = outlier_method
        self.root_output = root_output
        self.raw_data = None
        self.metadata = {}
        
        # 按层级存储预处理后的数据
        self.level_data = {}
        self.level_metadata = {}
        
        # 创建根目录
        os.makedirs(self.root_output, exist_ok=True)
    
    def load_raw_data(self):
        """加载原始数据"""
        print("="*70)
        print("步骤1: 加载原始微生物组数据")
        print("="*70)
        
        try:
            # 读取TSV文件
            self.raw_data = pd.read_csv(self.file_path, sep=',')
            print(f"  原始数据形状: {self.raw_data.shape}")
            print(f"  样本数: {self.raw_data.shape[0]}")
            print(f"  原始特征数: {self.raw_data.shape[1] - 2}")
            
            # 检查必要列
            if 'sample_id' not in self.raw_data.columns or 'label' not in self.raw_data.columns:
                print("数据必须包含'sample_id'和'label'列")
                return False
                
            # 预览前3行
            print(f"\n前3行数据预览:")
            print(self.raw_data.head(3)[['sample_id', 'label'] + list(self.raw_data.columns)[2:5]])
            
        except Exception as e:
            print(f"读取数据失败: {e}")
            return False
        
        return True
    
    def split_taxonomy_levels(self):
        """拆分不同分类层级的数据"""
        print("\n" + "="*70)
        print("步骤2: 拆分分类层级（目/科/属/种）")
        print("="*70)
        
        # 获取特征列
        feature_cols = [col for col in self.raw_data.columns 
                       if col not in ['sample_id', 'label']]
        
        # 为每个层级构建特征映射
        level_feature_maps = {}
        
        for level_key, config in TAXONOMY_CONFIG.items():
            prefix = config['prefix']
            level_feature_maps[level_key] = {}
            
            print(f"\n处理 {config['name']} 层级 ({prefix}):")
            
            # 遍历所有特征，提取对应层级信息
            for feature_col in feature_cols:
                # 拆分特征名
                parts = str(feature_col).split('|')
                level_value = None
                
                # 查找当前层级的标识
                for part in parts:
                    if part.startswith(prefix):
                        level_value = part
                        break
                
                if level_value:
                    # 将同一层级的特征归为一类
                    if level_value not in level_feature_maps[level_key]:
                        level_feature_maps[level_key][level_value] = []
                    level_feature_maps[level_key][level_value].append(feature_col)
            
            # 统计该层级的特征数
            level_feature_count = len(level_feature_maps[level_key])
            print(f"  识别到 {level_feature_count} 个 {config['name']} 层级分类单元")
            
            if level_feature_count == 0:
                print(f"未找到 {config['name']} 层级数据")
                continue
            
            # 聚合该层级的数据（求和）
            level_data_list = []
            level_names = []
            
            for level_name, cols in level_feature_maps[level_key].items():
                # 对同一层级分类单元的所有特征求和
                level_sum = self.raw_data[cols].sum(axis=1)
                level_data_list.append(level_sum)
                level_names.append(level_name)
            
            # 创建该层级的DataFrame
            level_features_df = pd.concat(level_data_list, axis=1)
            level_features_df.columns = level_names
            
            # 组合样本信息和特征
            level_full_data = pd.concat([
                self.raw_data[['sample_id', 'label']].reset_index(drop=True),
                level_features_df.reset_index(drop=True)
            ], axis=1)
            
            # 保存该层级数据
            self.level_data[level_key] = level_full_data
            
            print(f"  {config['name']} 层级数据形状: {level_full_data.shape}")
            print(f"  {config['name']} 层级特征数: {len(level_names)}")
        
        # 验证是否有有效数据
        valid_levels = [k for k, v in self.level_data.items() if v is not None and not v.empty]
        if not valid_levels:
            print("\n没有识别到任何有效层级的数据")
            return False
        
        print(f"\n成功拆分 {len(valid_levels)} 个层级的数据:")
        for level_key in valid_levels:
            config = TAXONOMY_CONFIG[level_key]
            data_shape = self.level_data[level_key].shape
            print(f"  - {config['name']}: {data_shape[0]} 样本 × {data_shape[1]-2} 特征")
        
        return True
    
    def preprocess_single_level(self, level_key):
        """预处理单个层级的数据（清洗+缺失值处理+异常值检测+CLR转换）"""
        config = TAXONOMY_CONFIG[level_key]
        level_data = self.level_data[level_key].copy()
        
        print(f"\n" + "="*60)
        print(f"步骤3: 预处理 {config['name']} 层级数据")
        print("="*60)
        
        # 创建该层级的输出目录
        level_output_dir = os.path.join(self.root_output, config['folder'])
        os.makedirs(level_output_dir, exist_ok=True)
        
        # 1. 缺失值/零值处理
        print(f"\n1. 缺失值检查与处理:")
        feature_cols = [col for col in level_data.columns 
                       if col not in ['sample_id', 'label']]
        features = level_data[feature_cols]
        
        # 统计零值
        zero_ratio_by_feature = (features == 0).sum() / len(features)
        zero_ratio_by_sample = (features == 0).sum(axis=1) / len(feature_cols)
        
        # 删除全零特征
        all_zero_features = zero_ratio_by_feature[zero_ratio_by_feature == 1].index.tolist()
        if all_zero_features:
            features_clean = features.drop(columns=all_zero_features)
            print(f"  删除全零特征: {len(all_zero_features)} 个")
        else:
            features_clean = features.copy()
        
        # 零值填充（最小正值的一半）
        min_positives = {}
        for col in features_clean.columns:
            positive_vals = features_clean[col][features_clean[col] > 0]
            if len(positive_vals) > 0:
                min_positives[col] = positive_vals.min() / 2
            else:
                min_positives[col] = 1e-10
        
        # 填充零值
        features_filled = features_clean.copy()
        for col in features_filled.columns:
            features_filled[col] = features_filled[col].apply(
                lambda x: min_positives[col] if x == 0 else x
            )
        
        # 2. 异常值检测
        print(f"\n2. 异常值检测 ({self.outlier_method}):")
        if self.outlier_method == 'iqr':
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
        
        # 3. CLR转换（中心对数比）
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
        processed_level_data = pd.concat([
            level_data[['sample_id', 'label']].reset_index(drop=True),
            features_clr_df.reset_index(drop=True)
        ], axis=1)
        
        # 4. 保存该层级的预处理数据
        raw_level_file = os.path.join(level_output_dir, f"{level_key}_raw.csv")
        processed_level_file = os.path.join(level_output_dir, f"{level_key}_processed.csv")
        
        level_data.to_csv(raw_level_file, index=False, encoding='utf-8')
        processed_level_data.to_csv(processed_level_file, index=False, encoding='utf-8')
        
        print(f"原始{config['name']}层级数据已保存: {raw_level_file}")
        print(f"预处理{config['name']}层级数据已保存: {processed_level_file}")
        
        # 保存该层级的元数据
        self.level_metadata[level_key] = {
            'config': config,
            'raw_shape': level_data.shape,
            'processed_shape': processed_level_data.shape,
            'feature_count': len(feature_cols),
            'cleaned_feature_count': len(features_clean.columns),
            'outlier_count': outlier_count,
            'samples_with_outliers': samples_with_outliers,
            'raw_file': raw_level_file,
            'processed_file': processed_level_file,
            'output_dir': level_output_dir
        }
        
        # 更新层级数据为预处理后的数据
        self.level_data[level_key] = processed_level_data
        
        return True
    
    def preprocess_all_levels(self):
        """预处理所有层级的数据"""
        for level_key in self.level_data.keys():
            if self.level_data[level_key] is not None and not self.level_data[level_key].empty:
                self.preprocess_single_level(level_key)
        
        return True
    
    def generate_preprocessing_report(self):
        """生成预处理报告"""
        print("\n" + "="*70)
        print("步骤4: 生成预处理报告")
        print("="*70)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("微生物组数据多层级预处理报告")
        report_lines.append("="*80)
        report_lines.append(f"处理日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"原始数据文件: {self.file_path}")
        report_lines.append(f"原始数据形状: {self.raw_data.shape}")
        report_lines.append("")
        
        # 各层级总结
        report_lines.append("1. 各层级数据概况")
        report_lines.append("-"*60)
        
        for level_key, metadata in self.level_metadata.items():
            config = metadata['config']
            report_lines.append(f"\n{config['name']} 层级 ({config['description']}):")
            report_lines.append(f"  - 原始特征数: {metadata['feature_count']}")
            report_lines.append(f"  - 清洗后特征数: {metadata['cleaned_feature_count']}")
            report_lines.append(f"  - 最终数据形状: {metadata['processed_shape']}")
            report_lines.append(f"  - 异常值数量: {metadata['outlier_count']}")
            report_lines.append(f"  - 输出目录: {metadata['output_dir']}")
            report_lines.append(f"  - 预处理数据文件: {metadata['processed_file']}")
        
        # 输出文件清单
        report_lines.append("\n3. 输出文件清单")
        report_lines.append("-"*60)
        report_lines.append(f"根目录: {self.root_output}")
        
        for level_key, metadata in self.level_metadata.items():
            config = metadata['config']
            report_lines.append(f"\n{config['name']} 层级文件:")
            report_lines.append(f"  - 原始数据: {metadata['raw_file']}")
            report_lines.append(f"  - 预处理数据: {metadata['processed_file']}")
        
        # 保存报告
        report_file = os.path.join(self.root_output, "preprocessing_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"预处理报告已保存: {report_file}")
        
        # 转换所有numpy类型为Python原生类型
        serializable_metadata = convert_numpy_types(self.level_metadata)
        
        # 保存元数据
        metadata_file = os.path.join(self.root_output, "preprocessing_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"预处理元数据已保存: {metadata_file}")
        
        return True
    
    def run_preprocessing_pipeline(self):
        """运行完整的预处理流程"""
        print("开始微生物组数据多层级预处理...")
        print("="*80)
        
        # 执行预处理步骤
        steps = [
            ("加载原始数据", self.load_raw_data),
            ("拆分分类层级", self.split_taxonomy_levels),
            ("预处理所有层级", self.preprocess_all_levels),
            ("生成预处理报告", self.generate_preprocessing_report)
        ]
        
        for step_name, step_func in steps:
            try:
                success = step_func()
                if not success:
                    print(f"\n{step_name} 失败")
                    return False
                print(f"\n{step_name} 完成")
            except Exception as e:
                print(f"\n{step_name} 出错: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "="*80)
        print("多层级数据预处理完成")
        print("="*80)
        print(f"预处理结果保存至: {self.root_output}")
        
        # 输出各层级文件路径（供后续筛选使用）
        print("\n各层级预处理数据文件路径:")
        for level_key, metadata in self.level_metadata.items():
            config = metadata['config']
            print(f"  {config['name']}: {metadata['processed_file']}")
        
        return True

# ===================== 生物标志物筛选模块 =====================
class MultiLevelADBiomarkerScreener:
    """多层级AD生物标志物筛选系统"""
    
    def __init__(self, preprocessed_metadata, root_output="D:/project/biomarker/multilevel_biomarkers"):
        self.preprocessed_metadata = preprocessed_metadata
        self.root_output = root_output
        self.level_results = {}
        
        os.makedirs(self.root_output, exist_ok=True)
    
    def _mann_whitney_test(self, features, labels):
        """Mann-Whitney U检验"""
        ad_features = features[labels == 1]
        control_features = features[labels == 0]
        
        results = []
        for col in features.columns:
            try:
                u_stat, p_value = mannwhitneyu(
                    ad_features[col].values, 
                    control_features[col].values,
                    alternative='two-sided'
                )
                
                # 计算效应量
                ad_mean = ad_features[col].mean()
                control_mean = control_features[col].mean()
                ad_std = ad_features[col].std()
                control_std = control_features[col].std()
                
                n_ad = len(ad_features[col])
                n_control = len(control_features[col])
                pooled_std = np.sqrt(((n_ad-1)*ad_std**2 + (n_control-1)*control_std**2) / (n_ad + n_control - 2))
                
                cohens_d = (ad_mean - control_mean) / pooled_std if pooled_std != 0 else 0
                log2_fc = np.log2(ad_mean / control_mean) if (control_mean > 0 and ad_mean > 0) else 0
                
                results.append({
                    'feature': col,
                    'p_value': p_value,
                    'u_stat': u_stat,
                    'cohens_d': abs(cohens_d),
                    'cohens_d_raw': cohens_d,
                    'log2_fc': log2_fc,
                    'ad_mean': ad_mean,
                    'control_mean': control_mean,
                    'direction': 'up' if cohens_d > 0 else 'down'
                })
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame(), pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        
        # FDR校正
        try:
            from statsmodels.stats.multitest import multipletests
            _, results_df['p_adj'], _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        except ImportError:
            results_df['p_adj'] = results_df['p_value']
        
        significant = results_df[results_df['p_adj'] < 0.05].copy()
        significant = significant.sort_values('cohens_d', ascending=False)
        
        if len(significant) == 0:
            significant = results_df[results_df['p_value'] < 0.05].copy()
        
        return results_df, significant
    
    def _random_forest_importance(self, features, labels, top_n=30):
        """随机森林特征重要性"""
        rf = RandomForestClassifier(
            n_estimators=500, max_features='sqrt', max_depth=10,
            min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=1
        )
        rf.fit(features, labels)
        
        importance_df = pd.DataFrame({
            'feature': features.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        return importance_df, top_features, rf
    
    def _lasso_selection(self, features, labels, top_n=30):
        """LASSO回归特征选择"""
        lasso = LogisticRegression(
            penalty='l1', C=0.1, solver='liblinear',
            class_weight='balanced', random_state=42, max_iter=1000
        )
        lasso.fit(features, labels)
        
        coefficients = lasso.coef_[0]
        lasso_features = pd.DataFrame({
            'feature': features.columns,
            'coefficient': coefficients,
            'abs_coefficient': abs(coefficients)
        })
        lasso_features = lasso_features[lasso_features['coefficient'] != 0].sort_values('abs_coefficient', ascending=False)
        
        top_features = lasso_features.head(top_n)['feature'].tolist()
        
        return lasso_features, top_features, lasso
    
    def _correlation_analysis(self, features, labels, top_n=30):
        """点二列相关分析"""
        correlations = []
        for col in features.columns:
            try:
                corr, p_value = pointbiserialr(labels, features[col])
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'correlation_raw': corr,
                    'p_value': p_value,
                    'direction': 'positive' if corr > 0 else 'negative'
                })
            except Exception as e:
                continue
        
        if not correlations:
            return pd.DataFrame(), []
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        try:
            from statsmodels.stats.multitest import multipletests
            _, corr_df['p_adj'], _, _ = multipletests(corr_df['p_value'], method='fdr_bh')
        except ImportError:
            corr_df['p_adj'] = corr_df['p_value']
        
        top_features = corr_df.head(top_n)['feature'].tolist()
        
        return corr_df, top_features
    
    def _integrate_biomarkers(self, method_features_dict, mw_results, rf_results, lasso_results, corr_results):
        """整合多方法结果"""
        all_features = set()
        for features in method_features_dict.values():
            all_features.update(features[:20])
        
        feature_scores = {}
        feature_method_count = {}
        
        for feature in all_features:
            count = 0
            score = 0
            
            # Mann-Whitney
            if feature in method_features_dict['mann_whitney']:
                count += 1
                match = mw_results[mw_results['feature'] == feature]
                if not match.empty:
                    score += match.iloc[0]['cohens_d'] * 2
            
            # 随机森林
            if feature in method_features_dict['random_forest']:
                count += 1
                match = rf_results[rf_results['feature'] == feature]
                if not match.empty:
                    score += match.iloc[0]['importance'] * 3
            
            # LASSO
            if feature in method_features_dict['lasso']:
                count += 1
                match = lasso_results[lasso_results['feature'] == feature]
                if not match.empty:
                    score += abs(match.iloc[0]['coefficient']) * 2.5
            
            # 相关性
            if feature in method_features_dict['correlation']:
                count += 1
                match = corr_results[corr_results['feature'] == feature]
                if not match.empty:
                    score += abs(match.iloc[0]['correlation_raw']) * 2
            
            feature_method_count[feature] = count
            feature_scores[feature] = score
        
        # 创建整合结果
        integration_df = pd.DataFrame([
            {'feature': f, 'n_methods': feature_method_count[f], 'score': feature_scores[f]}
            for f in all_features
        ])
        
        # 添加方法标记
        for method, features in method_features_dict.items():
            integration_df[f'in_{method}'] = integration_df['feature'].apply(
                lambda x: 1 if x in features else 0
            )
        
        integration_df = integration_df.sort_values(['n_methods', 'score'], ascending=[False, False])
        final_biomarkers = integration_df[integration_df['n_methods'] >= 1].head(20)
        
        return integration_df, final_biomarkers
    
    def _evaluate_performance(self, features, labels, selected_features, level_name, output_dir):
        """评估生物标志物性能"""
        if not selected_features:
            return None
        
        X_selected = features[selected_features]
        y = labels
        
        # 逻辑回归 + 留一法
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
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 绘制ROC曲线
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        if roc_auc > 0:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({level_name} Level)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'roc_curve.png'), dpi=300)
            plt.savefig(os.path.join(fig_dir, f'roc_curve.pdf'))
            plt.close()
        
        # ========== 转换numpy类型 ==========
        performance = convert_numpy_types({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'feature_count': len(selected_features),
            'selected_features': selected_features
        })
        
        return performance
    
    def _plot_visualizations(self, level_key, level_data, selected_features, output_dir):
        """绘制可视化图表"""
        config = TAXONOMY_CONFIG[level_key]
        fig_dir = os.path.join(output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 特征重要性图
        if 'random_forest' in self.level_results[level_key]:
            rf_results = self.level_results[level_key]['random_forest']['importance']
            if len(rf_results) > 0:
                top_n = min(20, len(rf_results))
                top_features = rf_results.head(top_n)
                top_features['display_name'] = top_features['feature'].apply(
                    lambda x: x.replace(config['prefix'], '')
                )
                
                plt.figure(figsize=(12, 8))
                top_features_sorted = top_features.sort_values('importance')
                
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features_sorted)))
                bars = plt.barh(range(len(top_features_sorted)), 
                               top_features_sorted['importance'],
                               color=colors, alpha=0.8, edgecolor='black')
                
                for i, (value, name) in enumerate(zip(top_features_sorted['importance'], 
                                                     top_features_sorted['display_name'])):
                    plt.text(value + 0.001, i, f'{value:.4f}', va='center', fontsize=9)
                
                plt.yticks(range(len(top_features_sorted)), top_features_sorted['display_name'])
                plt.xlabel('Feature Importance (Gini)')
                plt.title(f'Top {top_n} Feature Importance ({config["name"]} Level)')
                plt.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, 'feature_importance.png'), dpi=300)
                plt.savefig(os.path.join(fig_dir, 'feature_importance.pdf'))
                plt.close()
        
        # 热图
        if selected_features and len(selected_features) > 0:
            top_n = min(15, len(selected_features))
            top_features = selected_features[:top_n]
            
            feature_cols = [col for col in level_data.columns 
                           if col not in ['sample_id', 'label']]
            features = level_data[feature_cols][top_features].copy()
            
            # 标准化
            features_z = (features - features.mean()) / features.std()
            features_z['group'] = level_data['label'].map({0: 'Control', 1: 'AD'})
            features_z = features_z.sort_values('group')
            
            plt.figure(figsize=(15, 10))
            groups = features_z['group']
            features_z = features_z.drop('group', axis=1)
            
            sns.heatmap(features_z.T, cmap='RdBu_r', center=0,
                       yticklabels=[f.replace(config['prefix'], '') for f in features_z.columns],
                       xticklabels=False, cbar_kws={'label': 'Z-score'})
            
            from matplotlib.patches import Patch
            handles = [Patch(color='blue', label='Control'), Patch(color='red', label='AD')]
            plt.legend(handles=handles, loc='upper right')
            
            plt.title(f'Heatmap of Top {len(features_z.columns)} Biomarkers ({config["name"]} Level)')
            plt.xlabel('Samples')
            plt.ylabel('Biomarkers')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'biomarkers_heatmap.png'), dpi=300)
            plt.savefig(os.path.join(fig_dir, 'biomarkers_heatmap.pdf'))
            plt.close()
    
    def screen_single_level(self, level_key):
        """筛选单个层级的生物标志物"""
        metadata = self.preprocessed_metadata[level_key]
        config = metadata['config']
        processed_file = metadata['processed_file']
        
        print(f"\n" + "="*60)
        print(f"开始 {config['name']} 层级生物标志物筛选")
        print("="*60)
        
        # 创建输出目录
        level_output_dir = os.path.join(self.root_output, config['folder'])
        os.makedirs(level_output_dir, exist_ok=True)
        
        # 加载预处理后的数据
        level_data = pd.read_csv(processed_file, encoding='utf-8')
        
        # 分离特征和标签
        feature_cols = [col for col in level_data.columns 
                       if col not in ['sample_id', 'label']]
        features = level_data[feature_cols]
        labels = level_data['label']
        
        print(f"数据概况: {len(level_data)} 样本, {len(feature_cols)} 特征")
        print(f"AD样本数: {sum(labels == 1)}, 对照样本数: {sum(labels == 0)}")
        
        # 初始化结果存储
        self.level_results[level_key] = {}
        
        # 1. Mann-Whitney U检验
        mw_all, mw_sig = self._mann_whitney_test(features, labels)
        self.level_results[level_key]['mann_whitney'] = {
            'all_results': mw_all,
            'significant': mw_sig,
            'top_features': mw_sig.head(30)['feature'].tolist() if len(mw_sig) > 0 else []
        }
        print(f"\nMann-Whitney U检验完成，显著特征数: {len(mw_sig)}")
        
        # 2. 随机森林
        rf_imp, rf_top, rf_model = self._random_forest_importance(features, labels)
        self.level_results[level_key]['random_forest'] = {
            'importance': rf_imp,
            'top_features': rf_top,
            'model': rf_model
        }
        print(f"随机森林分析完成，Top特征数: {len(rf_top)}")
        
        # 3. LASSO
        lasso_coef, lasso_top, lasso_model = self._lasso_selection(features, labels)
        self.level_results[level_key]['lasso'] = {
            'coefficients': lasso_coef,
            'top_features': lasso_top,
            'model': lasso_model
        }
        print(f"LASSO回归完成，非零系数特征数: {len(lasso_coef)}")
        
        # 4. 相关性分析
        corr_df, corr_top = self._correlation_analysis(features, labels)
        self.level_results[level_key]['correlation'] = {
            'correlations': corr_df,
            'top_features': corr_top
        }
        print(f"相关性分析完成，Top特征数: {len(corr_top)}")
        
        # 5. 整合结果
        method_features = {
            'mann_whitney': self.level_results[level_key]['mann_whitney']['top_features'],
            'random_forest': rf_top,
            'lasso': lasso_top,
            'correlation': corr_top
        }
        
        int_all, int_final = self._integrate_biomarkers(
            method_features, mw_all, rf_imp, lasso_coef, corr_df
        )
        
        self.level_results[level_key]['integration'] = {
            'all_features': int_all,
            'final_biomarkers': int_final,
            'selected_features': int_final['feature'].tolist()
        }
        
        selected_features = int_final['feature'].tolist()
        print(f"多方法整合完成，最终筛选出 {len(selected_features)} 个生物标志物")
        
        # 6. 性能评估
        performance = self._evaluate_performance(
            features, labels, selected_features, config['name'], level_output_dir
        )
        self.level_results[level_key]['performance'] = performance
        
        # 7. 绘制可视化图表
        self._plot_visualizations(level_key, level_data, selected_features, level_output_dir)
        print(f"可视化图表生成完成")
        
        # 8. 保存结果
        # 保存统计结果
        mw_all.to_csv(os.path.join(level_output_dir, 'mann_whitney_results.csv'), index=False, encoding='utf-8')
        rf_imp.to_csv(os.path.join(level_output_dir, 'random_forest_importance.csv'), index=False, encoding='utf-8')
        lasso_coef.to_csv(os.path.join(level_output_dir, 'lasso_coefficients.csv'), index=False, encoding='utf-8')
        corr_df.to_csv(os.path.join(level_output_dir, 'correlation_results.csv'), index=False, encoding='utf-8')
        int_final.to_csv(os.path.join(level_output_dir, 'final_biomarkers.csv'), index=False, encoding='utf-8')
        
        # ========== JSON序列化 ==========
        # 转换性能结果中的numpy类型
        serializable_perf = convert_numpy_types(performance)
        
        # 保存性能结果
        with open(os.path.join(level_output_dir, 'performance_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable_perf, f, indent=2, ensure_ascii=False)
        
        # 保存标志物列表
        with open(os.path.join(level_output_dir, 'biomarker_list.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{config['name']} 层级AD生物标志物列表\n")
            f.write("="*50 + "\n\n")
            for i, feat in enumerate(selected_features, 1):
                display_name = feat.replace(config['prefix'], '')
                f.write(f"{i}. {display_name}\n")
        
        print(f"所有结果已保存至: {level_output_dir}")
        
        return True
    
    def generate_summary_report(self):
        """生成跨层级汇总报告"""
        print("\n" + "="*70)
        print("生成多层级生物标志物筛选报告")
        print("="*70)
        
        # 性能汇总
        perf_summary = []
        marker_summary = []
        
        for level_key, results in self.level_results.items():
            metadata = self.preprocessed_metadata[level_key]
            config = metadata['config']
            
            # 性能汇总
            if 'performance' in results and results['performance']:
                perf = results['performance']
                perf_summary.append({
                    'taxonomy_level': config['name'],
                    'biomarker_count': perf['feature_count'],
                    'accuracy': perf['accuracy'],
                    'sensitivity': perf['sensitivity'],
                    'specificity': perf['specificity'],
                    'f1_score': perf['f1_score'],
                    'roc_auc': perf['roc_auc']
                })
            
            # 标志物汇总
            if 'integration' in results and 'final_biomarkers' in results['integration']:
                markers = results['integration']['final_biomarkers']
                for _, row in markers.iterrows():
                    marker_summary.append({
                        'taxonomy_level': config['name'],
                        'biomarker': row['feature'].replace(config['prefix'], ''),
                        'original_name': row['feature'],
                        'n_methods': row['n_methods'],
                        'integrated_score': row['score']
                    })
        
        # 保存汇总表
        perf_df = pd.DataFrame(perf_summary)
        perf_df.to_csv(os.path.join(self.root_output, 'performance_summary.csv'), index=False, encoding='utf-8')
        
        marker_df = pd.DataFrame(marker_summary)
        marker_df.to_csv(os.path.join(self.root_output, 'biomarker_summary.csv'), index=False, encoding='utf-8')
        
        # 生成文本报告
        report_lines = []
        report_lines.append("1. 各层级筛选结果")
        report_lines.append("-"*60)
        
        for level_key, results in self.level_results.items():
            metadata = self.preprocessed_metadata[level_key]
            config = metadata['config']
            
            if 'integration' in results:
                n_markers = len(results['integration']['selected_features'])
                report_lines.append(f"\n{config['name']} 层级:")
                report_lines.append(f"  - 筛选出 {n_markers} 个生物标志物")
                
                if n_markers > 0:
                    top5 = results['integration']['selected_features'][:5]
                    top5_names = [f.replace(config['prefix'], '') for f in top5]
                    report_lines.append(f"  - Top5标志物: {', '.join(top5_names)}")
        
        # 性能总结
        report_lines.append("\n2. 模型性能总结")
        report_lines.append("-"*60)
        for _, row in perf_df.iterrows():
            report_lines.append(f"\n{row['taxonomy_level']} 层级:")
            report_lines.append(f"  - 准确率: {row['accuracy']:.3f}")
            report_lines.append(f"  - 敏感性: {row['sensitivity']:.3f}")
            report_lines.append(f"  - 特异性: {row['specificity']:.3f}")
            report_lines.append(f"  - F1分数: {row['f1_score']:.3f}")
            report_lines.append(f"  - ROC-AUC: {row['roc_auc']:.3f}")
        
        # 保存报告
        with open(os.path.join(self.root_output, 'screening_report.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"汇总报告已保存: {os.path.join(self.root_output, 'screening_report.txt')}")
        print(f"性能汇总表已保存: {os.path.join(self.root_output, 'performance_summary.csv')}")
        print(f"标志物汇总表已保存: {os.path.join(self.root_output, 'biomarker_summary.csv')}")
        
        return True
    
    def run_multilevel_screening(self):
        """运行多层级筛选"""
        print("开始AD肠道菌群多层级生物标志物筛选...")
        print("="*80)
        
        # 筛选所有层级
        for level_key in self.preprocessed_metadata.keys():
            self.screen_single_level(level_key)
        
        # 生成汇总报告
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("多层级生物标志物筛选完成！")
        print("="*80)
        print(f"筛选结果保存至: {self.root_output}")
        
        return True

# ===================== 主执行函数 =====================
def main():
    """主函数：预处理 + 多层级筛选"""
    # 第一步：数据预处理
    # 配置参数
    RAW_DATA_FILE = "/root/XXXMicro/Data/AD_new/merged_taxonomy.csv"
    PREPROCESS_OUTPUT = "/root/XXXMicro/Data/AD_new/multilevel_preprocessed1"
    SCREENING_OUTPUT = "/root/XXXMicro/Data/AD_new/multilevel_biomarkers1"
    
    # 检查原始文件
    if not os.path.exists(RAW_DATA_FILE):
        print(f"错误: 原始数据文件不存在 - {RAW_DATA_FILE}")
        return
    
    # 创建预处理器
    preprocessor = MultiLevelMicrobiomePreprocessor(
        file_path=RAW_DATA_FILE,
        min_nonzero_ratio=0.9,
        outlier_method='iqr',
        root_output=PREPROCESS_OUTPUT
    )
    
    # 运行预处理
    if not preprocessor.run_preprocessing_pipeline():
        print("预处理失败")
        return
    
    # 第二步：多层级生物标志物筛选
    # 创建筛选器
    screener = MultiLevelADBiomarkerScreener(
        preprocessed_metadata=preprocessor.level_metadata,
        root_output=SCREENING_OUTPUT
    )
    
    # 运行筛选
    if screener.run_multilevel_screening():
        print("\n分析完成")
        print(f"\n预处理结果: {PREPROCESS_OUTPUT}")
        print(f"筛选结果: {SCREENING_OUTPUT}")
    else:
        print("\n筛选过程出错")

if __name__ == "__main__":
    main()