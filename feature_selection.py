import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from skorch.helper import SliceDict


def _build_selector(method: str, k: int):
    if method == 'anova':
        return SelectKBest(score_func=f_classif, k=k)
    if method == 'mutual_info':
        return SelectKBest(score_func=mutual_info_classif, k=k)
    raise ValueError("method 必须是 'anova' 或 'mutual_info'")


def feature_selection_single(x_train, x_test, y_train, method='mutual_info', top_k=100):
    """
    单模态特征选择
    """
    # 如果原始特征数比我们想要的 top_k 还少，就不做过滤
    k = min(top_k, x_train.shape[1])
    selector = _build_selector(method, k)

    # 在训练集上拟合特征评分，并转换训练集
    x_train_selected = selector.fit_transform(x_train, y_train.flatten())
    
    # 获取选中的特征索引（方便排查问题或后续提取 Biomarker 名字）
    selected_idx = selector.get_support(indices=True)
    
    # 严格使用训练集选出的索引去转换测试集，防止数据泄露
    x_test_selected = x_test[:, selected_idx]

    # 单模态训练流程需要 ndarray，不要包装成 SliceDict。
    return x_train_selected, x_test_selected


def feature_selection_multi(x_train_dict, x_test_dict, y_train, method='mutual_info', top_k_dict=None):
    """
    多模态特征选择
    """
    # 如果没有传入字典，给一个默认的备用字典防止报错
    if top_k_dict is None:
        top_k_dict = {}

    x_train_out = {}
    x_test_out = {}
    
    for key in x_train_dict:
        # 动态获取当前模态的 top_k。
        top_k = top_k_dict.get(key, 200) 
        k = min(top_k, x_train_dict[key].shape[1])
        selector = _build_selector(method, k)
            
        x_train_out[key] = selector.fit_transform(x_train_dict[key], y_train.flatten())
        
        selected_idx = selector.get_support(indices=True)
        x_test_out[key] = x_test_dict[key][:, selected_idx]

    return SliceDict(**x_train_out), SliceDict(**x_test_out)
