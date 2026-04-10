import numpy as np
import itertools
import os
import pandas as pd
import lefse


# from skbio.stats.composition import clr
# from lefse import run_lefse
# from test_load_data import test_load_data
# from sklearn.linear_model import Lasso, LassoCV, Ridge



def calc_lda_scores(x, y):
    # Calculate LDA scores for each feature using LEfSE.
    # LEfSE expects input data in a specific format, so we transform the data accordingly
    data = pd.DataFrame(x)
    data = data.T
    data.to_csv('output.txt', sep='\t', index=True)
    

#     # Run LEfSE
    def lefse_format_input(input_file, output_file):
        command = f"python ./lefse/lefse_format_input.py {input_file} {output_file} -c 1 -o 100000"
        os.system(command)
    lefse_format_input('output.txt','output.in.txt')

    def run_lefse(input_file, output_file):
        command = f"python ./lefse/lefse_run.py {input_file} {output_file}"
        os.system(command)
    run_lefse('output.in.txt','output.txt')
    

def feature_select_lda(x, y, n_features):
    # Select top features based on LDA scores and reduce dimensionality using RandomForest.
    calc_lda_scores(x, y)
    # lda_scores = calc_lda_scores(x, y)
    data = []
    with open('output.txt', 'r') as file:
        for line in file:
            parse = line.split('\t')
            # 去掉换行符并将行添加到列表中
            data.append(float(parse[1]))
    # sorted_indices = sorted(range(len(data)), key=lambda i: data[i])
    
    # os.remove('output.txt')
    # os.remove('output.in.txt')
    # sorted_indices = sorted_indices[:n_features]
    # return sorted_indices
    # 从大到小排序
    indices = np.argsort(data)[::-1][:n_features]

    os.remove('output.txt')
    os.remove('output.in.txt')

    return indices

def calc_f3(x,y):
    labels=np.unique(y)
    indexs={}
    c_mins={}
    c_maxs={}
    for label in labels:
        index=np.where(y==label)[0]
        indexs[label]=index
        c_min=np.min(x[index])
        c_max=np.max(x[index])
        c_mins[label]=c_min
        c_maxs[label]=c_max
    label_combin=list(itertools.combinations(labels,2))
    f3=0.0

    if not label_combin:
        return f3

    for combination in label_combin:
        # sample_num=len(indexs[combination[0]])+len(indexs[combination[1]])
        # print(sample_num)
        # print(combination)
        # print(sample_num)
        c1_max,c1_min=c_maxs[combination[0]],c_mins[combination[0]]
        c2_max,c2_min=c_maxs[combination[1]],c_mins[combination[1]]
        # print(c1_max,c1_min,c2_max,c2_min)
        if c1_max<c2_min or c2_max<c1_min:
            f3+=1
        else:
            interval=(max(c1_min,c2_min),min(c1_max,c2_max))
            sample=np.hstack((x[indexs[combination[0]]],x[indexs[combination[1]]]))
            # print(sample.shape[0])
            n_overlay=0
            for k in range(sample.shape[0]):
                if sample[k]>=interval[0] and sample[k]<=interval[1]:
                    n_overlay+=1
            if sample.shape[0] > 0:
                f3+=1-n_overlay/sample.shape[0]
            else:
                f3 += 0
    f3/=len(label_combin)
    return f3


def feature_select_f3(x,y,k):
# def feature_select(x,y,k, n_feature): # cross-study validation
    n_feature=x.shape[1]
    f3s=[0.0 for i in range(n_feature)]
    for i in range(n_feature):
        if len(np.unique(x[:,i]))==1:
            f3s[i]=0
        elif len(np.unique(x[:,i]))==2:
            f3s[i]=1
        else:
            f3s[i]=calc_f3(x[:,i],y)
    # index=np.argsort(f3s)
    # index=index[-k:]
    # # return x[:,index],y

    # return index[:n_feature]

    # 从大到小排序
    indices = np.argsort(f3s)[::-1][:k]
    return indices


def feature_selection_single(x_train, x_test, y_train, method ='f3', top_k=None):
    """
    单模态特征选择
    """
    # log变换
    # x_train = np.log1p(x_train)
    # x_test  = np.log1p(x_test)

    # k = min(k, x_train.shape[1])
    k = top_k if top_k is not None else x_train.shape[1]
    k = min(k, x_train.shape[1])

    # 特征选择
    if method == 'f3':
        selected_idx = feature_select_f3(x_train, y_train.flatten(), k)
    elif method == 'lda':
        selected_idx = feature_select_lda(x_train, y_train.flatten(), k)
    else:
        raise ValueError("method must be 'f3' or 'lda'") 

    x_train = x_train[:, selected_idx]
    x_test  = x_test[:, selected_idx]
    return x_train, x_test


def feature_selection_multi(x_train, x_test, y_train, method ='f3', top_k=None):
    """
    多模态特征选择
    """
    for key in x_train:
        # log变换
        # x_train[key] = np.log1p(x_train[key])
        # x_test[key]  = np.log1p(x_test[key])

        # kk = min(k, x_train[key].shape[1])
        k = top_k if top_k is not None else x_train[key].shape[1]
        k = min(k, x_train[key].shape[1])

        # 特征选择
        if method == 'f3':
            selected_idx = feature_select_f3(x_train[key], y_train.flatten(), k)
        elif method == 'lda':
            selected_idx = feature_select_lda(x_train[key], y_train.flatten(), k)
        else:
            raise ValueError("method must be 'f3' or 'lda'")

        x_train[key] = x_train[key][:, selected_idx]
        x_test[key]  = x_test[key][:, selected_idx]

    return x_train, x_test