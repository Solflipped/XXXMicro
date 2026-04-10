from collections import Counter
import numpy as np
import pandas as pd
import traceback
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
    # torch.use_deterministic_algorithms(True)


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
def check_record(paras: dict, df_path: str) -> bool:
    """
    检查当前超参数组合是否已经完整训练过。
    逻辑：在 CSV 中寻找是否存在一行，其参数与 paras (通常包含 seed='all') 完全一致。
    返回 False：说明已存在（不用跑了）
    返回 True：说明不存在（需要跑）
    """
    if not os.path.exists(df_path):
        return True

    try:
        res_df = pd.read_csv(df_path)
    except Exception:
        return True

    if res_df.empty:
        return True

    # 初始化全为 True 的掩码
    match_mask = pd.Series(True, index=res_df.index)

    # 遍历传入的条件字典 (包含 seed='all', lr, batch_size 等)
    for k, v in paras.items():
        if k not in res_df.columns:
            return True
            
        if isinstance(v, float):
            try:
                # 处理浮点数，容许微小的精度误差，防止 1e-4 与 0.0001 不匹配
                csv_col_float = pd.to_numeric(res_df[k], errors='coerce')
                match_mask &= np.isclose(csv_col_float, v, atol=1e-8, equal_nan=False)
            except Exception:
                match_mask &= (res_df[k].astype(str) == str(v))
        else:
            # 处理字符串和整数 (比如 "all", "ko", 8)
            match_mask &= (res_df[k].astype(str) == str(v))

    # 如果有任何一行满足所有条件，说明已经完整跑过了
    if match_mask.any():
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

def evaluate(net, X, y):
    try:
        # 1. 统一转换标签为一维数组 
        y_true = np.asarray(y).reshape(-1)
        
        # 2. 获取预测结果
        # skorch 的 predict 默认返回 [N] 或 [N, 1]
        y_pred = np.asarray(net.predict(X)).reshape(-1)
        
        # skorch 的 predict_proba 应该返回 [N, 2]
        y_prob = np.asarray(net.predict_proba(X))

        # 3. 提取类别 1 的概率
        if y_prob.ndim == 2:
            if y_prob.shape[1] == 2:
                # 标准二分类：取第二列
                pos_prob = y_prob[:, 1]
            else:
                # 只有一列的情况
                pos_prob = y_prob[:, 0]
        else:
            # 已经是一维的情况
            pos_prob = y_prob.reshape(-1)

        # 4. 最终检查：确保 y_true 和 pos_prob 长度完全一致
        if len(y_true) != len(pos_prob):
            raise ValueError(f"维度不匹配: y_true({len(y_true)}) vs pos_prob({len(pos_prob)})")

        # 5. 计算指标
        metrics = {
            "AUC": float(roc_auc_score(y_true, pos_prob)),
            "ACC": float(accuracy_score(y_true, y_pred)),
            "Recall": float(recall_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred)),
        }

        # 格式化
        for k in metrics:
            metrics[k] = round(metrics[k], 4)

        df_details = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": pos_prob
        })

        return metrics, df_details

    except Exception as e:
        print(f"\n[Evaluation Error]: {e}")
        traceback.print_exc()
        
        fail_metrics = {k: -1.0 for k in ["AUC", "ACC", "Recall", "Precision", "F1"]}
        return fail_metrics, pd.DataFrame({})


# def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
#     try:
#         y_true = np.asarray(y).reshape(-1)
#         y_pred = np.asarray(net.predict(X)).reshape(-1)
#         y_prob = np.asarray(net.predict_proba(X))

#         if y_prob.ndim == 1:
#             pos_prob = y_prob.reshape(-1)
#         elif y_prob.shape[1] == 1:
#             pos_prob = y_prob[:, 0].reshape(-1)
#         else:
#             pos_prob = y_prob[:, 1].reshape(-1)

#         df = pd.DataFrame({
#             "y_true": y_true,
#             "y_pred": y_pred,
#             "y_prob_1": pos_prob,
#         })

#         metrics = {
#             "AUC": round(roc_auc_score(y_true, pos_prob), 4),
#             "ACC": round(accuracy_score(y_true, y_pred), 4),
#             "Recall": round(recall_score(y_true, y_pred), 4),
#             "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
#             "F1": round(f1_score(y_true, y_pred), 4),
#         }
#         return metrics, df

#     except Exception:
#         return {
#             "AUC": -1.0,
#             "ACC": -1.0,
#             "Recall": -1.0,
#             "Precision": -1.0,
#             "F1": -1.0,
#         }, pd.DataFrame({})


