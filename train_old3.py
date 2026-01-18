import torch
import os
from collections import OrderedDict as _OD
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from dataset import load_uni_features, load_multi_features
from model.FT_transformer import FTTransformer
from model.MBT import MBT
from model.MDL4Microbiome import MDL4Microbiome
from model.MSFT import MTMFTransformer, FT_Vote
import numpy as np
import pandas as pd
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split, SliceDict
from skorch.callbacks import Callback, EpochScoring, EarlyStopping
from utils import check_record, evaluate, setup_seed



def save_best_model(net, output_dir: str):
    """保存当前最佳验证得分的模型与优化器参数"""
    os.makedirs(output_dir, exist_ok=True)
    net.save_params(
        f_params=os.path.join(output_dir, 'model_best.pkl'),
        f_optimizer=os.path.join(output_dir, 'optim_best.pkl'),
        f_history=os.path.join(output_dir, 'history_best.json'),
    )


class SaveModel(Callback):
    """当监控指标出现 *_best 时，保存当前最佳模型。默认监控 valid_auc_best"""
    def __init__(self, out_dir: str, monitor: str = 'valid_auc_best'):
        self.out_dir = out_dir
        self.monitor = monitor
        os.makedirs(self.out_dir, exist_ok=True)

    def on_epoch_end(self, net, **kwargs):
        try:
            if net.history[-1, self.monitor]:
                save_best_model(net, self.out_dir)
        except KeyError:
            # 若没有该监控键，静默跳过
            pass


def _slice_sdict(sdict: SliceDict, idx):
    """根据索引切片 SliceDict（包含 f1_input / f2_input）。"""
    keys = list(sdict.keys())
    data = {k: sdict[k][idx] for k in keys}
    return SliceDict(**data)


def train(disease, feature, model_type, **params):
    """
    训练流程：
    1) 先用固定随机种子 777 做 8:2 的训练/测试划分（分层抽样，保证类比一致）
    2) 对训练集做 5 折交叉验证：每折显式给定验证集（predefined_split），暂时不使用早停；
       用 NeuralNetClassifier 训练，并用 EpochScoring('roc_auc', use_probas=True) 监控 AUC，保存每折最佳模型；
    3) 训练完成后，统计各折验证集 AUC，并在测试集上对每折模型做预测，输出平均融合后的测试 AUC。
    """

    # 固定种子与设备
    seed = 777
    setup_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =====================
    # 超参 & 记录列准备
    # =====================
    lr = float(params.get('learning_rate', params.get('lr', 1e-4)))
    batch_size = int(params.get('batch_size', 8))
    noise = 0   # 暂时不使用高斯噪声

    # FT-Transformer/MBT/MSFTTransformer/FT_Vote 相关超参
    n_blocks = int(params.get('n_blocks', 5))
    fusion_layer = params.get('fusion_layer', 3)
    n_bottlenecks = int(params.get('num_bottleneck', 4))
    num_heads = 8  # 多头注意力

    # MSFTTransformer 相关超参
    use_bottleneck = True  # 使用瓶颈
    btn_init = 'embed'     # bottleneck初始化方式
    use_cross_atn = True   # 使用交叉注意力

    # MDL4Microbiome 相关超参（两阶段训练 epoch）
    # epoch1 = int(params.get('epoch1', 30))  # individual 模型部分训练轮数
    # epoch2 = int(params.get('epoch2', 10))  # shared 模型部分训练轮数

    # 特征选择相关参数
    # fs_method = params.get('fs_method', 'none')  # 特征选择方法: 'none' / 'RandomForest+RFECV'
    # fs_n_estimators = 200
    # fs_rfecv_splits = 5
    # fs_rfecv_repeats = 1


    # 结果目录与记录文件
    results_dir = os.path.join('./results', disease)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{model_type}.csv')

    # 构造 record
    if model_type == 'MBT':
        record = {
            'lr': lr,
            'batch_size': batch_size,
            'feature': feature,
            'n_blocks': n_blocks,
            'fusion_layer': fusion_layer,
            'num_bottleneck': n_bottlenecks,
            'seed': seed,
        }
        metric_cols = ['AUC', 'Recall', 'Precision', 'F1']
        ordered_cols = ['fold','lr','batch_size','feature','n_blocks','fusion_layer','num_bottleneck','seed'] + metric_cols
    elif model_type == 'FT_transformer':
        record = {
            'lr': lr,
            'batch_size': batch_size,
            'feature': feature,
            'n_blocks': n_blocks,
            'seed': seed,
        }
        metric_cols = ['AUC', 'Recall', 'Precision', 'F1']
        ordered_cols = ['fold','lr','batch_size','feature','n_blocks','seed'] + metric_cols
    elif model_type == 'MDL4Microbiome':
        record = {
            'lr': lr,
            'batch_size': batch_size,
            'feature': feature,
            'seed': seed,
        }
        metric_cols = ['AUC', 'Recall', 'Precision', 'F1']
        ordered_cols = ['fold','lr','batch_size','feature','seed'] + metric_cols
    elif model_type == 'MSFTTransformer':
        record = {
            'lr': lr,
            'batch_size': batch_size,
            'feature': feature,
            'n_blocks': n_blocks,  # 作为层数
            'num_bottleneck': n_bottlenecks,
            'use_bottleneck': use_bottleneck,
            'btn_init': btn_init,
            'use_cross_atn': use_cross_atn,
            'seed': seed,
        }
        metric_cols = ['AUC','Recall','Precision','F1']
        ordered_cols = ['fold','lr','batch_size','feature','n_blocks','num_bottleneck','use_bottleneck','btn_init','use_cross_atn','seed'] + metric_cols
    elif model_type == 'FT_Vote':
        record = {
            'lr': lr,
            'batch_size': batch_size,
            'feature': feature,
            'n_blocks': n_blocks,
            'seed': seed,
        }
        metric_cols = ['AUC','Recall','Precision','F1']
        ordered_cols = ['fold','lr','batch_size','feature','n_blocks','seed'] + metric_cols
    else:
        raise ValueError(f"Unsupported model_type for logging schema: {model_type}")

    # 全局去重：如果已存在任意一行具有相同超参（不含 fold、指标），则直接跳过整个训练
    if not check_record(record, log_path):
        print('paras has trained. 该超参数组合已训练过')
        return None

    # 1) 训练/测试划分（使用已有的加载函数，函数内部已标准化与分层切分）
    is_multimodal = (',' in feature)
    if model_type  == "FT_transformer"  and is_multimodal:
        raise ValueError("FT_transformer 仅支持单模态（'ko' 或 'species'）")

    if model_type == "FT_transformer":
        x_train, x_test, y_train, y_test = load_uni_features(seed=seed, disease=disease, feature=feature)
        Xtr = x_train['f1_input']
        Xte = x_test['f1_input']
        print(f"[Data] FT single-modality shapes -> X_train: {Xtr.shape}, X_test: {Xte.shape}")
    elif model_type == 'MBT':
        x_train, x_test, y_train, y_test = load_multi_features(seed=seed, disease=disease, feature=feature, noise=noise)
        Xtr = x_train
        Xte = x_test
        print(f"[Data] MBT multi-modality shapes -> f1_train: {Xtr['f1_input'].shape}, f2_train: {Xtr['f2_input'].shape}; f1_test: {Xte['f1_input'].shape}, f2_test: {Xte['f2_input'].shape}")
    elif model_type == 'MDL4Microbiome':
        x_train, x_test, y_train, y_test = load_multi_features(seed=seed, disease=disease, feature=feature, noise=noise)
        Xtr = x_train
        Xte = x_test
        print(f"[Data] MDL4 multi-modality shapes -> f1_train: {Xtr['f1_input'].shape}, f2_train: {Xtr['f2_input'].shape}; f1_test: {Xte['f1_input'].shape}, f2_test: {Xte['f2_input'].shape}")
    elif model_type == 'MSFTTransformer':
        x_train, x_test, y_train, y_test = load_multi_features(seed=seed, disease=disease, feature=feature, noise=noise)
        Xtr = x_train
        Xte = x_test
        print(f"[Data] MSFTTransformer multi-modality shapes -> f1_train: {Xtr['f1_input'].shape}, f2_train: {Xtr['f2_input'].shape}; f1_test: {Xte['f1_input'].shape}, f2_test: {Xte['f2_input'].shape}")
    elif model_type == 'FT_Vote':
        x_train, x_test, y_train, y_test = load_multi_features(seed=seed, disease=disease, feature=feature, noise=noise)
        Xtr = x_train
        Xte = x_test
        print(f"[Data] FT_Vote multi-modality shapes -> f1_train: {Xtr['f1_input'].shape}, f2_train: {Xtr['f2_input'].shape}; f1_test: {Xte['f1_input'].shape}, f2_test: {Xte['f2_input'].shape}")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # y 形状调整
    y_tr_float = y_train.astype(np.float32)  # (N,1)
    y_tr_cls = y_tr_float.squeeze().astype(int)  # (N,)
    y_te = np.array(y_test).astype(int)

    # =====================
    # 5 折交叉验证（在训练集上）
    # =====================

    # StratifiedKFold: 按标签分层的 K 折切分（保证每折类别比例大致一致）
    # n_splits=5: 5 折；shuffle=True: 打乱；random_state=42: 固定随机性   宇宙的答案
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_test_aucs = []
    fold_test_recalls = []
    fold_test_precisions = []
    fold_test_f1s = []

    # 解析模态顺序
    modality_order = feature.split(',') if ',' in feature else [feature]

    # 将任意形状的 predict_proba 输出转换为正类概率向量 (N,)
    # def _to_pos_proba(y_prob: np.ndarray) -> np.ndarray:
    #     if y_prob.ndim == 1:
    #         return y_prob
    #     if y_prob.shape[1] == 1:
    #         return y_prob[:, 0]
    #     return y_prob[:, 1]

    # kf.split(X, y) -> 生成器，每次返回 (train_idx, val_idx) 的索引数组
    # 这里：
    # - 如果是多输入（SliceDict），我们用 Xtr['f1_input'] 作为“代表”来做分层切分（只要样本数一致即可）
    # - 如果是单输入（ndarray），直接用 Xtr
    # enumerate(..., start=1): 让 fold_i 从 1 开始计数（便于展示与记录）
    # isinstance(Xtr, SliceDict): 判断是否为多输入
    for fold_i, (tr_idx, val_idx) in enumerate(
        kf.split(Xtr['f1_input'] if isinstance(Xtr, SliceDict) else Xtr, y_tr_cls), start=1
    ):
        print(f"\n===== Fold {fold_i}/5 =====")

        # 准备本折的数据
        if isinstance(Xtr, SliceDict):
            # 多输入：对 SliceDict 做索引切片，保留键 f1_input/f2_input，保证两路输入与索引对齐
            X_tr_fold = _slice_sdict(Xtr, tr_idx)
            X_val_fold = _slice_sdict(Xtr, val_idx)
            print(
                f"train f1/f2: {X_tr_fold['f1_input'].shape} / {X_tr_fold['f2_input'].shape}; "
                f"val f1/f2: {X_val_fold['f1_input'].shape} / {X_val_fold['f2_input'].shape}"
            )
        else:
            # 单输入：ndarray 直接用索引数组切片
            X_tr_fold = Xtr[tr_idx]
            X_val_fold = Xtr[val_idx]
            print(f"train: {X_tr_fold.shape}; val: {X_val_fold.shape}")

        y_tr_fold = y_tr_float[tr_idx]
        y_val_fold = y_tr_float[val_idx]


        # 按模型类型构建网络
        if model_type == "FT_transformer":
            n_num_features = X_tr_fold.shape[1]
            model = FTTransformer.make_default(
                n_num_features=n_num_features,
                cat_cardinalities=None,
                n_blocks=n_blocks,
                d_out=1,
            )
            module_to_fit = model
        elif model_type == 'MBT':
            f1_dim = X_tr_fold['f1_input'].shape[1]
            f2_dim = X_tr_fold['f2_input'].shape[1]
            if modality_order[0].strip().lower() == 'species':
                n_species_features, n_ko_features = f1_dim, f2_dim
                first_species = True
            else:
                n_species_features, n_ko_features = f2_dim, f1_dim
                first_species = False
            base_mbt = MBT.make_default(
                n_species_features=n_species_features,
                n_ko_features=n_ko_features,
                num_layers=n_blocks,
                num_heads=8,
                fusion_layer=fusion_layer,
                n_bottlenecks=n_bottlenecks,
                test_with_bottlenecks=True,
            )
            class MBTWrapper(nn.Module):
                def __init__(self, mbt, first_species_flag: bool):
                    super().__init__(); self.mbt = mbt; self.first_species_flag = first_species_flag
                def forward(self, f1_input, f2_input):
                    # 将 f1_input/f2_input 映射为底层模型期望的 dict {'species':..., 'ko':...}
                    if self.first_species_flag:
                        raw_x = {'species': f1_input, 'ko': f2_input}
                    else:
                        raw_x = {'species': f2_input, 'ko': f1_input}
                    return self.mbt(raw_x)
            module_to_fit = MBTWrapper(base_mbt, first_species)
        elif model_type == 'MDL4Microbiome':
            f1_dim = X_tr_fold['f1_input'].shape[1]
            f2_dim = X_tr_fold['f2_input'].shape[1]
            if modality_order[0].strip().lower() == 'species':
                n_species_features, n_ko_features = f1_dim, f2_dim
                first_species = True
            else:
                n_species_features, n_ko_features = f2_dim, f1_dim
                first_species = False
            base_mdl = MDL4Microbiome.make_default(
                n_species_features=n_species_features,
                n_ko_features=n_ko_features,
            )
            class MDL4Wrapper(nn.Module):
                def __init__(self, mdl, first_species_flag: bool):
                    super().__init__(); self.mdl = mdl; self.first_species_flag = first_species_flag
                def forward(self, f1_input, f2_input):
                    if self.first_species_flag:
                        raw_x = {'species': f1_input, 'ko': f2_input}
                    else:
                        raw_x = {'species': f2_input, 'ko': f1_input}
                    return self.mdl(raw_x)
            module_to_fit = MDL4Wrapper(base_mdl, first_species)
        elif model_type == 'MSFTTransformer':
            f1_dim = X_tr_fold['f1_input'].shape[1]
            f2_dim = X_tr_fold['f2_input'].shape[1]
            if modality_order[0].strip().lower() == 'species':
                first_species = True
                species_dim, ko_dim = f1_dim, f2_dim
            else:
                first_species = False
                species_dim, ko_dim = f2_dim, f1_dim
            inputs_dim = _OD()
            inputs_dim['species'] = (0, species_dim)
            inputs_dim['ko'] = (1, ko_dim)
            base_msft = MTMFTransformer(
                n_layers=n_blocks,
                num_bottleneck=n_bottlenecks,
                use_bottleneck=use_bottleneck,
                btn_init=btn_init,
                use_cross_atn=use_cross_atn,
                inputs_dim=inputs_dim,
            )
            class MSFTWrapper(nn.Module):
                def __init__(self, msft_model, first_species_flag: bool):
                    super().__init__(); self.msft = msft_model; self.first_species_flag = first_species_flag
                def forward(self, f1_input, f2_input):
                    # 直接以关键字形式传递两个模态，符合 MTMFTransformer.forward(self, **features)
                    if self.first_species_flag:
                        return self.msft(species=f1_input, ko=f2_input)
                    else:
                        return self.msft(species=f2_input, ko=f1_input)
            module_to_fit = MSFTWrapper(base_msft, first_species)
        elif model_type == 'FT_Vote':
            f1_dim = X_tr_fold['f1_input'].shape[1]
            f2_dim = X_tr_fold['f2_input'].shape[1]
            if modality_order[0].strip().lower() == 'species':
                first_species = True
                species_dim, ko_dim = f1_dim, f2_dim
            else:
                first_species = False
                species_dim, ko_dim = f2_dim, f1_dim
            inputs_dim = _OD()
            inputs_dim['species'] = (0, species_dim)
            inputs_dim['ko'] = (1, ko_dim)
            # 构造用于FT_Vote的配置: 每个模态单独的FTTransformer共享n_blocks等
            ft_vote_config = {
                'n_num_features': inputs_dim,
                'n_blocks': n_blocks,
                'cat_cardinalities': None,
                'd_out': 1,
            }
            base_vote = FT_Vote(**ft_vote_config)
            class VoteWrapper(nn.Module):
                def __init__(self, vote_model, first_species_flag: bool):
                    super().__init__(); self.vote = vote_model; self.first_species_flag = first_species_flag
                def forward(self, f1_input, f2_input):
                    if self.first_species_flag:
                        return self.vote(species=f1_input, ko=f2_input)
                    else:
                        return self.vote(species=f2_input, ko=f1_input)
            module_to_fit = VoteWrapper(base_vote, first_species)
        else:
            raise ValueError("Unsupported model_type during model build")

        # 定义验证集（predefined_split）与回调（AUC 监控 + 保存最佳）
        valid_ds = Dataset(X_val_fold, y_val_fold)

        # 使用内置 roc_auc 评分器，自动选择正类概率/决策函数，避免自定义展平导致的长度不一致
        auc_cb = EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='valid_auc')
        # 早停：若 valid_auc 在 patience 轮内没有提升则停止训练
        early_stop_cb = EarlyStopping(
            monitor='valid_auc', patience=20,
            threshold=0, threshold_mode='rel', lower_is_better=False
        )
        ckpt_dir = os.path.join('./Checkpoints', disease, '777', f'{model_type}', f'fold_{fold_i}')
        save_cb = SaveModel(ckpt_dir, monitor='valid_auc_best')

        # 选择损失（BCEWithLogits）按折动态设置 pos_weight = neg/pos
        y_tr_labels = y_tr_fold.squeeze().astype(int)
        pos_ct = int((y_tr_labels == 1).sum())
        neg_ct = int((y_tr_labels == 0).sum())
        if pos_ct == 0 or neg_ct == 0:
            # 极端情况：某折出现单一类别（理论上分层抽样不会发生），回退为 1.0
            pos_weight_value = 1.0
        else:
            pos_weight_value = neg_ct / pos_ct
        pos_weight_tensor = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        print(f"[Fold {fold_i}] pos_weight=neg/pos = {neg_ct}/{pos_ct} -> {pos_weight_value:.4f}")


        net = NeuralNetClassifier(
            module_to_fit,
            max_epochs=200, # 每个fold训练epoch数
            lr=lr,
            iterator_train__shuffle=True,
            device=device,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=1e-3,
            batch_size=batch_size,
            train_split=predefined_split(valid_ds),
            criterion=criterion,
            # 回调顺序：先计算 valid_auc，再执行早停逻辑，最后保存最佳模型
            callbacks=[auc_cb, early_stop_cb, save_cb],
        )
        
        # 训练
        net.fit(X_tr_fold, y_tr_fold)

        # 加载并使用本折验证集上 AUC 最佳模型
        net.load_params(
            f_params=os.path.join(ckpt_dir, 'model_best.pkl'),
            f_optimizer=os.path.join(ckpt_dir, 'optim_best.pkl'),
            f_history=os.path.join(ckpt_dir, 'history_best.json'),
        )

        # 在测试集上做最终评估
        test_metrics = evaluate(net, Xte, y_te)
        fold_test_aucs.append(test_metrics['AUC'])
        fold_test_recalls.append(test_metrics['Recall'])
        fold_test_precisions.append(test_metrics['Precision'])
        fold_test_f1s.append(test_metrics['F1'])
        print(f"Test AUC = {test_metrics['AUC']:.4f}")

        # 写入该折结果
        try:
            existing_df = pd.read_csv(log_path)
        except Exception:
            existing_df = pd.DataFrame(columns=ordered_cols)
        existing_df = existing_df.reindex(columns=ordered_cols)

        fold_record = dict(record)
        fold_record.update({
            'fold': fold_i,
            'AUC': round(test_metrics['AUC'], 4),
            'Recall': round(test_metrics['Recall'], 4),
            'Precision': round(test_metrics['Precision'], 4),
            'F1': round(test_metrics['F1'], 4),
        })

        new_row_df = pd.DataFrame([{c: fold_record.get(c, None) for c in ordered_cols}])
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
        updated_df.to_csv(log_path, index=False)
        print(f"[Logging] Saved fold {fold_i} result to {log_path}")

    # ====== 交叉验证结束后：打印每个指标的均值与标准差（测试集） ======
    print("\n===== Test Summary (per-fold models) =====")
    print(f"AUC mean±std: {np.mean(fold_test_aucs):.4f} ± {np.std(fold_test_aucs):.4f}")
    print(f"Recall mean±std: {np.mean(fold_test_recalls):.4f} ± {np.std(fold_test_recalls):.4f}")
    print(f"Precision mean±std: {np.mean(fold_test_precisions):.4f} ± {np.std(fold_test_precisions):.4f}")
    print(f"F1 mean±std: {np.mean(fold_test_f1s):.4f} ± {np.std(fold_test_f1s):.4f}")

    # ====== 写入汇总，指标为 mean(std) ======
    try:
        existing_df = pd.read_csv(log_path)
    except Exception:
        existing_df = pd.DataFrame(columns=ordered_cols)
    existing_df = existing_df.reindex(columns=ordered_cols)

    def _fmt(mu, sd):
        return f"{mu:.4f}({sd:.4f})"

    summary_row = dict(record)
    summary_row.update({
        'fold': 'all',
        'AUC': _fmt(np.mean(fold_test_aucs), np.std(fold_test_aucs)),
        'Recall': _fmt(np.mean(fold_test_recalls), np.std(fold_test_recalls)),
        'Precision': _fmt(np.mean(fold_test_precisions), np.std(fold_test_precisions)),
        'F1': _fmt(np.mean(fold_test_f1s), np.std(fold_test_f1s)),
    })

    new_row_df = pd.DataFrame([{c: summary_row.get(c, None) for c in ordered_cols}])
    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    updated_df.to_csv(log_path, index=False)
    print(f"[Logging] Saved summary to {log_path}")
