import torch
import os
import numpy as np
import pandas as pd
from collections import OrderedDict 
from skorch.dataset import ValidSplit
from skorch.callbacks import Callback, EarlyStopping, LRScheduler
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from feature_selection import feature_selection_single, feature_selection_multi
from dataset import load_uni_features, load_multi_features
from model.UFEN import UFEN, UFENNet
from model.MSFT import MTMFTransformer
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
    优先看 valid_acc，如果 acc 持平，则看 valid_loss 是否更低
    """
    def __init__(self, disease: str, model_type: str, fold: int):
        self.output_dir = f"./Checkpoints/{disease}/{model_type}/fold_{fold}"
        self.best_valid_acc = -float('inf')
        self.best_valid_loss = float('inf')

    def on_epoch_end(self, net, **kwargs):
        # 提取当前 epoch 的验证集表现 (skorch 的分类器默认会计算 valid_acc)
        current_acc = net.history[-1, 'valid_acc']
        current_loss = net.history[-1, 'valid_loss']

        save_flag = False

        # 判定条件 1：如果准确率创下新高
        if current_acc > self.best_valid_acc:
            self.best_valid_acc = current_acc
            self.best_valid_loss = current_loss
            save_flag = True
            
        # 判定条件 2：如果准确率和历史最佳持平（加 1e-5 容差防精度问题），但 Loss 更低
        elif abs(current_acc - self.best_valid_acc) < 1e-5 and current_loss < self.best_valid_loss:
            self.best_valid_loss = current_loss
            save_flag = True

        # 如果满足以上任一条件，则保存模型
        if save_flag:
            save_best_model(net, self.output_dir)
   
        
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

    # --- 2. 数据集加载 ---
    feature_list = feature.split(",")
    if model_type == "UFEN":
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
        if model_type == 'UFEN':
            x_train = x['f1_input'][train_id]
            x_test  = x['f1_input'][test_id]
            inputs_dim = OrderedDict({
                "f1_input": (x_train.shape[0], x_train.shape[1])
            })
        else:
            x_train = {k: v[train_id] for k, v in x.items()}
            x_test  = {k: v[test_id] for k, v in x.items()}
            inputs_dim = OrderedDict({
                k: (v.shape[0], v.shape[1]) for k, v in x_train.items()
            })
        y_train, y_test = y[train_id], y[test_id]
        # --- 特征选择 ---
        top_k_map = {
            'species': 200,
            'ko': 800
        }
        if model_type == 'UFEN':
            top_k = top_k_map.get(feature, 200)
            x_train, x_test = feature_selection_single(
                x_train, x_test, y_train, method='anova',top_k=top_k
            )
            inputs_dim = OrderedDict({
                "f1_input": (x_train.shape[0], x_train.shape[1])
            })
        else:
            top_k_dict = {}
            for dict_key, real_feat_name in zip(x_train.keys(), feature_list):
                top_k_dict[dict_key] = top_k_map.get(real_feat_name, 200)
            x_train, x_test = feature_selection_multi(
                x_train, x_test, y_train, method='anova', top_k_dict=top_k_dict
            )
            inputs_dim = OrderedDict({
                k: (v.shape[0], v.shape[1]) for k, v in x_train.items()
            })
        # --- 模型定义 以及 初始化 ---
        modelconfig = params.copy()
        for k in ['batch_size', 'lr']:
            modelconfig.pop(k, None)
    
        if model_type == "UFEN":
            modelconfig['n_num_features'] = inputs_dim['f1_input'][1]
            model = UFEN.make_default(**modelconfig).to(device)
        elif model_type == "MSFT":
            modelconfig['inputs_dim'] = inputs_dim
            model = MTMFTransformer(**modelconfig).to(device)
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
                max_epochs=100,
                lr=lr,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, stratified=True, random_state=42),
                # train_split=None, # 训练全部训练集数据
                device=device,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4, # 正则化
                criterion=criterion,
                callbacks=[
                    EarlyStopping(patience=20, monitor='valid_loss', lower_is_better=True),
                    SaveModel(disease, model_type, fold),
                    # LRScheduler(policy='CosineAnnealingLR', T_max=100, eta_min=1e-5),
                ],
            )
        elif model_type == "MSFT":
            net = NeuralNetBinaryClassifier(
                model,
                max_epochs=100,
                lr=lr,
                batch_size=batch_size,
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, stratified=True, random_state=42),
                device=device,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4, 
                criterion=criterion,
                callbacks=[
                    EarlyStopping(patience=20, monitor='valid_loss', lower_is_better=True),
                    SaveModel(disease, model_type, fold),
                    # LRScheduler(policy='CosineAnnealingLR', T_max=100, eta_min=1e-5),
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
