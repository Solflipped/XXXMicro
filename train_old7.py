import torch
import os
from collections import OrderedDict 
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from skorch import NeuralNetClassifier
from skorch.dataset import Dataset, ValidSplit
from skorch.helper import predefined_split, SliceDict
from skorch.callbacks import Callback, EpochScoring, EarlyStopping
from dataset import load_uni_features, load_multi_features
from model.FT_transformer import FTTransformer
from model.FTMicro import FTMicro
from model.MBT import MBT
from model.MDL4Microbiome import MDL4Microbiome
from model.MSFT import MTMFTransformer, FT_Vote
from model.UFEN_old1 import UFEN
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
    def __init__(self, disease: str, seed: int, model_type: str):
        self.output_dir = f"./Checkpoints/{disease}/{seed}/{model_type}"

    def on_epoch_end(self, net, **kwargs):
        # 监控 skorch 自动生成的 'valid_acc_best' 或 'valid_auc_best'
        if net.history[-1, 'valid_acc_best']: 
            save_best_model(net, self.output_dir)

def train(disease, feature, model_type, seeds, **params):
    """
    训练流程：
    1) 数据加载：获取 80% 训练集和 20% 独立测试集
    2) 模型构建：
    3) 训练配置：利用 ValidSplit 自动从训练集中切分验证集
    4) Record：记录超参数与测试结果
    """
    # --- 0. 参数解析 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = float(params.get('lr', params.get('learning_rate', 1e-4)))
    batch_size = int(params.get('batch_size', 8))
    n_blocks = int(params.get('n_blocks', 4))
    n_bottlenecks = int(params.get('num_bottleneck', 4))
    modality_order = feature.split(',') if ',' in feature else [feature]  # 确定模态顺序
    # FT-Transformer/MBT/MSFTTransformer/FT_Vote 相关超参
    fusion_layer = params.get('fusion_layer', 2)
    use_cross_atn = params.get('use_cross_atn', True)   # 是否使用交叉注意力
    # MSFTTransformer 相关超参
    use_bottleneck = True  # 使用瓶颈
    btn_init = 'embed'     # bottleneck初始化方式
    # FTMicro 相关超参
    d_token = int(params.get('d_token', 128))
    fusion_depth = int(params.get('fusion_depth', 4))
    dst_embedding_length = int(params.get('dst_embedding_length', 8))
    ahl_depth = int(params.get('ahl_depth', 3))
    # UFEN 相关超参
    base_channels = int(params.get('base_channels', 96))
    expansion_factor = int(params.get('expansion_factor', 2))
    ufen_num_layers = int(params.get('num_layers', 4))
    latent_dim = int(params.get('latent_dim', 512))
    
    # 结果路径
    results_dir = os.path.join('./results', disease)
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'{model_type}.csv')

    for i, seed in enumerate(seeds):
        setup_seed(seed)
        fold = i + 1; # 记录当前是第几轮 (1-5)
        # 构造记录 record
        if model_type == 'MBT':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'n_blocks': n_blocks,
                'fusion_layer': fusion_layer,
                'num_bottleneck': n_bottlenecks,
                'use_cross_atn': use_cross_atn,
                'seed': seed,
            }
            metric_cols = ['AUC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','n_blocks','fusion_layer','num_bottleneck','use_cross_atn','seed'] + metric_cols
        elif model_type == 'FT_transformer':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'n_blocks': n_blocks,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','n_blocks','seed'] + metric_cols
        elif model_type == 'FTMicro':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'd_token': d_token,
                'fusion_depth': fusion_depth,
                'dst_embedding_length': dst_embedding_length,
                'ahl_depth': ahl_depth,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','d_token','fusion_depth','dst_embedding_length','ahl_depth','seed'] + metric_cols
        elif model_type == 'MDL4Microbiome':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','seed'] + metric_cols
        elif model_type == 'MSFTTransformer':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'n_blocks': n_blocks, 
                'num_bottleneck': n_bottlenecks,
                'use_bottleneck': use_bottleneck,
                'btn_init': btn_init,
                'use_cross_atn': use_cross_atn,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','n_blocks','num_bottleneck','use_bottleneck','btn_init','use_cross_atn','seed'] + metric_cols
        elif model_type == 'FT_Vote':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'n_blocks': n_blocks,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','n_blocks','seed'] + metric_cols
        elif model_type == 'UFEN':
            record = {
                'fold': fold,
                'lr': lr,
                'batch_size': batch_size,
                'feature': feature,
                'd_token': d_token,
                'base_channels': base_channels,
                'expansion_factor': expansion_factor,
                'num_layers': ufen_num_layers,
                'latent_dim': latent_dim,
                'seed': seed,
            }
            metric_cols = ['AUC', 'ACC', 'Recall', 'Precision', 'F1']
            ordered_cols = ['fold','lr','batch_size','feature','d_token','base_channels','expansion_factor','num_layers','latent_dim','seed'] + metric_cols
        else:
            raise ValueError(f"Unsupported model_type for logging schema: {model_type}")

        # 全局去重：如果已存在任意一行具有相同超参（不含 fold、指标），则直接跳过整个训练
        if not check_record(record, log_path):
            print('paras has trained. 该超参数组合已训练过')
            continue


        # --- 1. 数据加载 ---
        # 根据模型类型决定加载单模态还是多模态
        if model_type == "FT_transformer" or model_type == "UFEN":
            #  x_train, x_test 形状：{"f1_input": (N, D)}
            #  y_train形状为 (Ntrain,1)  y_test形状为 (Ntest,)
            x_train, x_test, y_train, y_test = load_uni_features(seed=seed, disease=disease, feature=feature)
            input_dim = x_train['f1_input'].shape[1]
            print(f"[Data] single-modality shapes -> x_train: {x_train['f1_input'].shape}, x_test: {x_test['f1_input'].shape}")
        else:
            #  x_train, x_test 形状：{"f1_input": (N, D1), "f2_input": (N, D2)}
            #  y_train形状为 (Ntrain,1)  y_test形状为 (Ntest,)
            x_train, x_test, y_train, y_test = load_multi_features(seed=seed, disease=disease, feature=feature)
            f1_dim = x_train['f1_input'].shape[1]
            f2_dim = x_train['f2_input'].shape[1]
            # 确定模态顺序
            if modality_order[0].strip().lower() == 'species':
                n_species_features, n_ko_features = f1_dim, f2_dim
            else:
                n_species_features, n_ko_features = f2_dim, f1_dim
            # 与旧版保持一致：输入维度按 f1/f2 位置定义，不做语义化
            inputs_dim = OrderedDict([
                ('f1_input', x_train['f1_input'].shape),
                ('f2_input', x_train['f2_input'].shape),
            ])
            print(f"[Data] multi-modality shapes -> f1_train: {x_train['f1_input'].shape}, f2_train: {x_train['f2_input'].shape}; f1_test: {x_test['f1_input'].shape}, f2_test: {x_test['f2_input'].shape}")


        # 标签处理：y_train 在 fit 时保持一维以支持 ValidSplit
        y_train = y_train.flatten().astype(np.float32)  #  y_train形状为 (Ntrain,1) 转换成 (Ntrain,)
        y_test = y_test.astype(np.int32)
    
        # --- 2. 模型构建 ---
        if model_type == 'FT_transformer':
            base_model = FTTransformer.make_default(
                n_num_features=input_dim,
                cat_cardinalities=None,
                n_blocks=n_blocks,
                d_out=1,
                last_layer_query_idx=[-1]
            )
            class FTWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, f1_input):
                    return self.model(x_num=f1_input)

            module_to_fit = FTWrapper(base_model)

        elif model_type == 'UFEN':
            base_model = UFEN.make_default(
                n_num_features=input_dim,
                d_token=d_token,
                base_channels=base_channels,
                expansion_factor=expansion_factor,
                num_layers=ufen_num_layers,
                latent_dim=latent_dim,
            )
            class UFENWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, f1_input):
                    return self.model(f1_input)

            module_to_fit = UFENWrapper(base_model)
        
        elif model_type == 'MDL4Microbiome':
            base_model = MDL4Microbiome.make_default(
                n_species_features=n_species_features,
                n_ko_features=n_ko_features,
            )
            class MDL4Wrapper(nn.Module):
                def __init__(self, model):
                        super().__init__(); self.model = model
                def forward(self, f1_input, f2_input):
                        return self.model(f1_input=f1_input, f2_input=f2_input)
            module_to_fit = MDL4Wrapper(base_model)
        
        elif model_type == 'MBT':
            base_model = MBT.make_default(
                n_species_features=n_species_features,
                n_ko_features=n_ko_features,
                num_layers=n_blocks,
                num_heads=8,
                fusion_layer=fusion_layer,
                n_bottlenecks=n_bottlenecks,
                test_with_bottlenecks=True,
                use_cross_atn=use_cross_atn,
            )
            class MBTWrapper(nn.Module):
                def __init__(self, model):
                        super().__init__(); self.model = model
                        self.species_first = modality_order[0].strip().lower() == 'species'
                def forward(self, f1_input, f2_input):
                        if self.species_first:
                            species_input, ko_input = f1_input, f2_input
                        else:
                            species_input, ko_input = f2_input, f1_input

                        return self.model(raw_x={
                            'species': species_input,
                            'ko': ko_input,
                        })
            module_to_fit = MBTWrapper(base_model)
        
        elif model_type == 'MSFTTransformer':
            base_model = MTMFTransformer(
                n_layers=n_blocks,
                num_bottleneck=n_bottlenecks,
                inputs_dim=inputs_dim,
                use_bottleneck=use_bottleneck,
                use_cross_atn=use_cross_atn,
                btn_init=btn_init
            )
            class MSFTWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__(); self.model = model
                def forward(self, f1_input, f2_input):
                    return self.model(f1_input=f1_input, f2_input=f2_input)
            module_to_fit = MSFTWrapper(base_model)
        
        elif model_type == 'FT_Vote':
            base_model = FT_Vote(
                n_num_features=inputs_dim,
                cat_cardinalities=None,
                n_blocks=n_blocks,
                d_out=1,
            )
            class VoteWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__(); self.model = model
                def forward(self, f1_input, f2_input):
                    return self.model(f1_input=f1_input, f2_input=f2_input)
            module_to_fit = VoteWrapper(base_model)
        
        elif model_type == 'FTMicro':
            # 使用 make_default 类方法创建 FTMicro 模型
            base_model = FTMicro.make_default(
                batch_size=batch_size,
                n_species=n_species_features,
                n_ko=n_ko_features,
                d_token=d_token,
                dst_embedding_length=dst_embedding_length,
                AHL_depth=ahl_depth,
                fusion_depth=fusion_depth,
            )

            class FTMicroWrapper(nn.Module):
                def __init__(self, model): 
                    super().__init__()
                    self.model = model
                    self.species_first = modality_order[0].strip().lower() == 'species'

                def forward(self, f1_input, f2_input): 
                    if self.species_first:
                        species_input, ko_input = f1_input, f2_input
                    else:
                        species_input, ko_input = f2_input, f1_input

                    output = self.model(species_input, ko_input)
                    
                    # 只使用主分类器的输出进行训练 (最直接的修复)
                    return output

            module_to_fit = FTMicroWrapper(base_model)
        else:
            raise ValueError("Unsupported model_type during model build")
        


        # --- 3. 训练配置 ---
        # 动态计算 pos_weight
        # pos_weight = torch.tensor(
        #     [(y_train == 0).sum() / (y_train == 1).sum()],
        #     device=device,
        #     dtype=torch.float32
        # )

        criterion = nn.BCEWithLogitsLoss().to(device)
    
        net = NeuralNetClassifier(
            module_to_fit,
            max_epochs=200,
            lr=lr,
            batch_size=batch_size,
            device=device,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=1e-4,
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, random_state=42),
            callbacks=[
                EpochScoring('roc_auc', lower_is_better=False, on_train=False, name='valid_auc'),
                EarlyStopping(monitor='valid_loss', lower_is_better=True, patience=20),
                SaveModel(disease, seed, model_type)
            ],
        )

        # --- 4. 执行训练与评估 ---
        # 标签在 fit 时需为 (N, 1) 以匹配 BCEWithLogitsLoss
        net.fit(x_train, y_train.reshape(-1, 1))
        
        
        # 加载当前种子下的最佳模型
        net.load_params(f_params=f"./Checkpoints/{disease}/{seed}/{model_type}/model_best.pkl",
                        f_optimizer=f"./Checkpoints/{disease}/{seed}/{model_type}/optim_best.pkl",
                        f_history=f"./Checkpoints/{disease}/{seed}/{model_type}/history_best.json")

        
        # 测试集评估
        test_metrics,df= evaluate(net, x_test, y_test)
        record.update(test_metrics)
        print(f"Test AUC = {test_metrics['AUC']:.4f}")

        # --- 5. 记录结果 ---
        # 保存 Record 到 CSV
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        try:
            res_df = pd.read_csv(log_path)
            res_df = pd.concat([res_df, pd.DataFrame(record, index=[0])])
        except:
            res_df = pd.DataFrame(record, index=[0])
        
        res_df.to_csv(log_path, index=False)
        

        print(f"[Logging] Saved summary to {log_path}")

    
        # --- 6. 汇总训练结果 ---
        # 只有当完成最后一轮 (fold == len(seeds)) 时才尝试汇总
        if fold == len(seeds):
            try:
                print("开始汇总当前实验的五次seed结果...")

                df = pd.read_csv(log_path)
                # 排除历史汇总
                df_no_summary = df[df['seed'] != 'all']
                # 只取最近5次实验
                df_current = df_no_summary.tail(len(seeds))

                summary = record.copy()
                summary['fold'] = 'all'
                summary['seed'] = 'all'

                for m in metric_cols:
                    vals = pd.to_numeric(df_current[m], errors='coerce')
                    summary[m] = f"{vals.mean():.4f}({vals.std():.4f})"

                df_final = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
                df_final.to_csv(log_path, index=False)

                print("五轮实验汇总完成")

            except Exception as e:
                print(f"汇总触发失败: {e}")