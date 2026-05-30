import torch
import os
import numpy as np
import pandas as pd
from collections import OrderedDict 
from skorch.dataset import ValidSplit
from skorch.callbacks import Callback, EarlyStopping, LRScheduler, EpochScoring
from skorch.helper import SliceDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from dataset import load_uni_features, load_multi_features
from model.KOFT import KOFT, KOFTNet
from model.UFEN import UFEN, UFENNet
from model.MSFT import MTMFTransformer
from model.XXXMicro import XXXMicro, XXXMicroNet
from utils import setup_seed, evaluate

def save_best_model(net, output_dir: str):
    """保存当前最佳验证得分的模型与优化器参数"""
    os.makedirs(output_dir, exist_ok=True)
    net.save_params(
        f_params=os.path.join(output_dir, 'model_best.pkl'),
        f_optimizer=os.path.join(output_dir, 'optim_best.pkl'),
        f_history=os.path.join(output_dir, 'history_best.json'),
    )

class SaveModel(Callback):
    """
    仅根据 valid_loss 保存当前最佳模型
    """
    def __init__(self, disease: str, model_type: str, fold: int):
        self.output_dir = f"./Checkpoints/{disease}/{model_type}/fold_{fold}"
        self.best_valid_loss = float('inf')

    def on_epoch_end(self, net, **kwargs):
        current_loss = net.history[-1, 'valid_loss']
        if current_loss < self.best_valid_loss:
            self.best_valid_loss = current_loss
            save_best_model(net, self.output_dir)


def _extract_positive_proba(net, X):
    y_prob = np.asarray(net.predict_proba(X))
    if y_prob.ndim == 1:
        return y_prob.reshape(-1)
    if y_prob.shape[1] == 1:
        return y_prob[:, 0].reshape(-1)
    return y_prob[:, 1].reshape(-1)


def valid_auc_scorer(net, X, y):
    y_true = np.asarray(y).reshape(-1)
    if np.unique(y_true).size < 2:
        return 0.5
    pos_prob = _extract_positive_proba(net, X)
    return float(roc_auc_score(y_true, pos_prob))


def train(disease, feature, model_type, cvfold, seed, **params):
    # --- 1. 初始化 & 参数设置 ---
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = os.path.join('./results', disease)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{model_type}.csv') # 结果文件

    lr = float(params.get('lr', 1e-4))
    batch_size = int(params.get('batch_size', 8))
    beta = float(params.get('beta', 0.01))
    lambda_recon = float(params.get('lambda_recon', 0.5))
    koft_max_epochs = int(params.get('max_epochs', 150))
    koft_patience = int(params.get('patience', 20))
    koft_weight_decay = float(params.get('weight_decay', 1e-4))

    # --- 2. 数据集加载 ---
    feature_list = feature.split(",")
    if model_type in {"UFEN", "KOFT"}:
        x, y, _ = load_uni_features(disease=disease, features=feature_list)
    else:
        x, y, _ = load_multi_features(disease=disease, features=feature_list)
    y = y.astype(np.float32)


    # --- 3. K-fold ---
    skf = StratifiedKFold(n_splits=cvfold, random_state=seed, shuffle=True)
    all_scores = []

    for fold, (train_id, test_id) in enumerate(skf.split(np.arange(len(y)), y)):
        print(f"\n========== Seed {seed} | Fold {fold+1}/{cvfold} ==========")
        # fold_dir = f"./Checkpoints/{disease}/UFEN/fold_{fold}"
        # os.makedirs(fold_dir, exist_ok=True)

        # --- 数据划分 ---
        if model_type in {'UFEN', 'KOFT'}:
            x_train = x['f1_input'][train_id]
            x_test  = x['f1_input'][test_id]
            inputs_dim = OrderedDict({
                "f1_input": (x_train.shape[0], x_train.shape[1])
            })
        else:
            x_train = SliceDict(**{k: v[train_id] for k, v in x.items()})
            x_test  = SliceDict(**{k: v[test_id] for k, v in x.items()})
            inputs_dim = OrderedDict({
                k: (v.shape[0], v.shape[1]) for k, v in x_train.items()
            })
        y_train, y_test = y[train_id], y[test_id]
        # --- 模型定义 以及 初始化 ---
        modelconfig = params.copy()
        for k in ['batch_size', 'lr', 'max_epochs', 'patience', 'weight_decay']:
            modelconfig.pop(k, None)
    
        if model_type == "UFEN":
            modelconfig['n_num_features'] = inputs_dim['f1_input'][1]
            model = UFEN.make_default(**modelconfig).to(device)
        elif model_type == "KOFT":
            modelconfig['n_num_features'] = inputs_dim['f1_input'][1]
            model = KOFT.make_default(**modelconfig).to(device)
        elif model_type == "MSFT":
            modelconfig['inputs_dim'] = inputs_dim
            model = MTMFTransformer(**modelconfig).to(device)
        elif model_type == "XXXMicro":
            modelconfig.pop('lambda_recon', None)
            modelconfig.pop('beta', None)
            modelconfig['inputs_dim'] = inputs_dim
            model = XXXMicro.make_default(**modelconfig).to(device)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")


        # --- 损失函数 ---
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        # 计算正样本权重：负样本数 / 正样本数
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # --- Skorch训练器 ---
        if model_type == "UFEN":
            net = UFENNet(
                model,
                beta=beta,
                max_epochs=200,
                lr=lr,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, stratified=True, random_state=seed),
                # train_split=None, # 训练全部训练集数据
                device=device,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4, # 正则化
                criterion=criterion,
                callbacks=[
                    EpochScoring(
                        scoring=valid_auc_scorer,
                        lower_is_better=False,
                        on_train=False,
                        name='valid_auc',
                    ),
                    EarlyStopping(patience=15, monitor='valid_loss', lower_is_better=True),
                    SaveModel(disease, model_type, fold),
                    # LRScheduler(policy='CosineAnnealingLR', T_max=100, eta_min=1e-6),
                ],
            )
        elif model_type == "KOFT":
            net = KOFTNet(
                model,
                max_epochs=koft_max_epochs,
                lr=lr,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, stratified=True, random_state=seed),
                device=device,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=koft_weight_decay,
                criterion=criterion,
                callbacks=[
                    EpochScoring(
                        scoring=valid_auc_scorer,
                        lower_is_better=False,
                        on_train=False,
                        name='valid_auc',
                    ),
                    EarlyStopping(patience=koft_patience, monitor='valid_loss', lower_is_better=True),
                    SaveModel(disease, model_type, fold),
                ],
            )
        elif model_type == "XXXMicro":
            net = XXXMicroNet(
                model,
                lambda_recon=lambda_recon,
                max_epochs=100,
                lr=lr,
                batch_size=batch_size,
                iterator_train__shuffle=True,   
                train_split=ValidSplit(0.2, stratified=True, random_state=seed),
                device=device,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4,
                criterion=criterion,
                callbacks=[
                    EpochScoring(
                        scoring=valid_auc_scorer,
                        lower_is_better=False,
                        on_train=False,
                        name='valid_auc',
                    ),
                    EarlyStopping(patience=20, monitor='valid_loss', lower_is_better=True),
                    SaveModel(disease, model_type, fold),
                    LRScheduler(policy='CosineAnnealingLR', T_max=100, eta_min=1e-6),
                ],
            )


        # --- 训练 ---
        net.fit(x_train, y_train)

        # --- 测试 ---
        # 加载当前种子下的最佳模型
        net.load_params(f_params=f"./Checkpoints/{disease}/{model_type}/fold_{fold}/model_best.pkl",
                        f_optimizer=f"./Checkpoints/{disease}/{model_type}/fold_{fold}/optim_best.pkl",
                        f_history=f"./Checkpoints/{disease}/{model_type}/fold_{fold}/history_best.json")

        scores, _ = evaluate(net, x_test, y_test)
        all_scores.append(scores)

        # fold_record = OrderedDict({
        #     "fold": fold + 1,
        #     **record,
        #     **scores,
        # })
        # fold_records.append(fold_record)

        print(f"[Seed {seed} | Fold {fold+1}] AUC = {scores['AUC']:.4f}")
        
    
    # --- 5.汇总结果 ---
    seed_scores = {}
    for key in all_scores[0].keys():
        seed_scores[key] = np.mean([s[key] for s in all_scores])
        
    print(f"\n [Seed {seed} 独立汇总] 平均 AUC = {seed_scores['AUC']:.4f}")

    # 将这个包含平均分的字典 (数值型) 返回给 main 函数
    return seed_scores
