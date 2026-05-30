"""
机器学习模型训练器
实现SVM、逻辑回归、LightGBM、随机森林、XGBoost五种模型的训练和超参数优化
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class MLModelTrainer:
    """机器学习模型训练器"""

    def __init__(self, random_state=42, use_gpu=True):
        """
        初始化训练器

        Args:
            random_state: 随机种子，确保结果可复现
            use_gpu: 是否使用GPU加速（适用于XGBoost和LightGBM）
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.models = {}
        self.scalers = {}
        self.best_params = {}
        self.cv_results = {}

        # 检测GPU可用性
        if self.use_gpu:
            self._check_gpu_availability()

        # 定义超参数搜索空间（根据论文表3-5）
        self.param_grids = {
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            },
            'RandomForest': {
                'n_estimators': [100, 300],
                'max_depth': [5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            'XGBoost': {
                'n_estimators': [100, 300],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            },
            'LightGBM': {
                'n_estimators': [100, 300],
                'max_depth': [3, 5],
                'num_leaves': [15, 31]
            }
        }

    def _check_gpu_availability(self):
        """检测GPU可用性"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ 检测到 NVIDIA GPU，将使用 GPU 加速训练")
            else:
                print("⚠ 未检测到 NVIDIA GPU，将使用 CPU 训练")
                self.use_gpu = False
        except FileNotFoundError:
            print("⚠ 未检测到 NVIDIA GPU，将使用 CPU 训练")
            self.use_gpu = False

    def load_data(self, data_path, label_col='label', sample_id_col='sample_id'):
        """
        加载预处理后的数据

        Args:
            data_path: 数据文件路径
            label_col: 标签列名
            sample_id_col: 样本ID列名

        Returns:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
        """
        print(f"加载数据: {data_path}")
        data = pd.read_csv(data_path)

        # 分离特征和标签
        feature_cols = [col for col in data.columns
                       if col not in [sample_id_col, label_col]]

        X = data[feature_cols].values
        y = data[label_col].values

        print(f"数据形状: {X.shape}")
        print(f"类别分布: AD={sum(y==1)}, Control={sum(y==0)}")

        return X, y, feature_cols

    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        划分训练集和测试集（8:2分层抽样）

        Args:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            stratify: 是否分层抽样

        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"\n数据划分 (训练集:测试集 = {1-test_size}:{test_size})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y if stratify else None,
            random_state=self.random_state
        )

        print(f"训练集: {X_train.shape}, AD={sum(y_train==1)}, Control={sum(y_train==0)}")
        print(f"测试集: {X_test.shape}, AD={sum(y_test==1)}, Control={sum(y_test==0)}")

        return X_train, X_test, y_train, y_test

    def standardize_features(self, X_train, X_test):
        """
        特征标准化（Z-score标准化）

        Args:
            X_train: 训练集特征
            X_test: 测试集特征

        Returns:
            X_train_scaled, X_test_scaled, scaler
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, scaler

    def train_model(self, model_name, X_train, y_train, use_grid_search=True, cv_folds=5):
        """
        训练单个模型（含超参数优化）

        Args:
            model_name: 模型名称
            X_train: 训练集特征
            y_train: 训练集标签
            use_grid_search: 是否使用网格搜索
            cv_folds: 交叉验证折数

        Returns:
            best_model: 最优模型
            best_params: 最优超参数
            cv_results: 交叉验证结果
        """
        print(f"\n{'='*70}")
        print(f"训练模型: {model_name}")
        print(f"{'='*70}")

        # 创建基础模型
        base_model = self._create_base_model(model_name)

        if use_grid_search and model_name in self.param_grids:
            # 使用网格搜索优化超参数
            print(f"使用{cv_folds}折交叉验证进行超参数优化...")

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grids[model_name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            print(f"最优超参数: {best_params}")
            print(f"最优交叉验证AUC: {best_score:.4f}")

            cv_results = {
                'best_score': best_score,
                'best_params': best_params,
                'cv_results': grid_search.cv_results_
            }
        else:
            # 直接训练（使用默认参数）
            print("使用默认参数训练...")
            base_model.fit(X_train, y_train)
            best_model = base_model
            best_params = base_model.get_params()
            cv_results = None

        return best_model, best_params, cv_results

    def _create_base_model(self, model_name):
        """创建基础模型"""
        if model_name == 'LogisticRegression':
            return LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            )
        elif model_name == 'RandomForest':
            return RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_name == 'SVM':
            return SVC(
                class_weight='balanced',
                probability=True,  # 启用概率预测
                random_state=self.random_state
            )
        elif model_name == 'XGBoost':
            if self.use_gpu:
                return xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    tree_method='hist',
                    device='cuda',
                    random_state=self.random_state
                )
            else:
                return xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=self.random_state,
                    n_jobs=-1
                )
        elif model_name == 'LightGBM':
            if self.use_gpu:
                return lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    class_weight='balanced',
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    random_state=self.random_state,
                    verbose=-1
                )
            else:
                return lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                )
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def train_all_models(self, X_train, y_train, model_names=None, use_grid_search=True):
        """
        训练所有模型

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            model_names: 要训练的模型列表，None表示训练所有模型
            use_grid_search: 是否使用网格搜索

        Returns:
            models: 训练好的模型字典
        """
        if model_names is None:
            model_names = ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost', 'LightGBM']

        for model_name in model_names:
            model, params, cv_results = self.train_model(
                model_name, X_train, y_train, use_grid_search
            )

            self.models[model_name] = model
            self.best_params[model_name] = params
            if cv_results:
                self.cv_results[model_name] = cv_results

        return self.models

    def save_models(self, output_dir):
        """
        保存训练好的模型和标准化器

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"模型已保存: {model_path}")

        # 保存标准化器
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(output_dir, f'{scaler_name}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"标准化器已保存: {scaler_path}")

        # 保存最优超参数
        params_path = os.path.join(output_dir, 'best_params.json')
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        print(f"最优超参数已保存: {params_path}")

    def load_models(self, model_dir):
        """
        加载训练好的模型

        Args:
            model_dir: 模型目录
        """
        model_names = ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost', 'LightGBM']

        for model_name in model_names:
            model_path = os.path.join(model_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"模型已加载: {model_path}")
