from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import os
from os.path import join as join
import torch
import numpy as np
import random
from sklearn.base import BaseEstimator
import pandas as pd
from typing import List, Any, Dict



def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.use_deterministic_algorithms(True)


def check_sample_order(files: List[str]) -> None:
    """
    :param files: List of feature path
    :return:
    """
    samples = []
    for file in files:
        df = pd.read_csv(file)
        print(list(df['sample_id'])[:5])
        samples.append(list(df['sample_id']))

    for sample in samples:
        if samples[0] != sample:
            assert 0, "The order of samples is inconsistent across files."

def check_record(paras: Dict, df_path: str) -> bool:
    if not os.path.exists(df_path):
        return True

    res_df = pd.read_csv(df_path)
    # 去掉 summary 行
    if 'seed' in res_df.columns:
        res_df = res_df[res_df['seed'].astype(str) != 'all']
    # 去重时不比较 fold（fold 仅表示当前批次第几轮）
    compare_keys = [k for k in paras.keys() if k != 'fold']
    # 只保留需要比较的列
    res_df = res_df[compare_keys]
    # 统一转字符串比较
    paras_str = {k: str(paras[k]) for k in compare_keys}
    
    for _, row in res_df.iterrows():
        row_dict = {k: str(row[k]) for k in compare_keys}
        if row_dict == paras_str:
            return False

    return True

# 评分指标
# def my_auc(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
#     y_proba = net.predict_proba(X)
#     return roc_auc_score(y, y_proba[:, 1])


# def my_f1(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
#     y_proba = net.predict_proba(X)
#     y_pred = np.argmax(y_proba, axis=1)
#     return f1_score(y, y_pred)


# def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
#     """统一评估接口：兼容 predict_proba 返回 (N,), (N,1) 或 (N,2)。

#     - 若为 (N,2)，按 [:,1] 取正类概率；
#     - 若为 (N,1) 或 (N,)，按该列/向量即为正类概率；
#     - 指标：AUC、Recall、Precision、F1（与当前 train.py 需求一致）。
#     """
#     try:
#         y_true = y
#         y_pred = net.predict(X)
#         y_prob = net.predict_proba(X)

#         # 取正类概率
#         if y_prob.ndim == 1:
#             pos_prob = y_prob
#         elif y_prob.shape[1] == 1:
#             pos_prob = y_prob[:, 0]
#         else:
#             pos_prob = y_prob[:, 1]

#         # 使用 zero_division=0 避免“无预测正类”时 Precision 报 UndefinedMetricWarning
#         metrics = {
#             'AUC': round(roc_auc_score(y_true, pos_prob), 4),
#             'Recall': round(recall_score(y_true, y_pred), 4),
#             'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
#             'F1': round(f1_score(y_true, y_pred), 4),  
#         }
#         return metrics

#     except Exception:
#         return {
#             'AUC': -1.0,
#             'Recall': -1.0,
#             'Precision': -1.0,
#             'F1': -1.0,
#         }


def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> (dict[str, float], pd.DataFrame):
    y_true, y_pred = y, net.predict(X)
    y_prob = net.predict_proba(X)
    #print(y_true.shape)
    #print(y_prob.shape)
    df = pd.DataFrame({
        'y_true': y_true,
        'y_prob_0': y_prob[:, 0].squeeze(),  # 第一个类别的概率
        'y_prob_1': y_prob[:, 1].squeeze()  # 第二个类别的概率
    })
    try:
        y_true, y_pred = y, net.predict(X)
        y_prob = net.predict_proba(X)
        # 记录 预测值 和 准确值
        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        metrics = {
            'AUC': round(roc_auc_score(y_true, y_prob[:, 1]), 4),
            'ACC': round(accuracy_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'F1': round(f1_score(y_true, y_pred), 4)
        }
        return metrics, df

    except:
        return {
            'AUC': -1.0,
            'ACC': -1.0,
            'Recall': -1.0,
            'Precision': -1.0,
            'F1': -1.0
        }, pd.DataFrame({})
