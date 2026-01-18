from collections import Counter
from typing import Tuple, Dict, Any, List

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDict
from utils import setup_seed, check_sample_order


class dataset(Dataset):
    def __init__(self, path: str, use_cols: List = None, normalize: bool = True):
        super().__init__()
        if use_cols is None:
            use_cols = []  #  选取加载指定列，如果为空则加载所有列
        print(f'Load dataset from {path}')

        if use_cols:
            self.data = pd.read_csv(path)[use_cols]
        else:
            self.data = pd.read_csv(path)
        # self.data.sort_values('sample_id', inplace=True)  # sort

        self.label = self.data.iloc[:, 1].values.squeeze()  # 获取标签列
        self.data = self.data.iloc[:, 2:].values            # 获取特征列

        if normalize:                                       # 对特征数据进行标准化（Z-Score 标准化）
            scaler = StandardScaler()                       # 将数据转换为均值为 0、标准差为 1 的分布
            self.data = scaler.fit_transform(self.data)

    def __len__(self):
        return self.label.shape[0]   # 返回数据集的样本数量

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# 加载单模态特征数据集
def load_uni_features(seed: int, disease: str, feature: str) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param feature: type of feature: species or ko 
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    path = f"./Data/{disease}/{feature[0]}_abundance.csv"
    data = dataset(path, use_cols=None)  # 加载数据并进行标准化（Z-Score 标准化）

    # 划分数据
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.label.astype('int'),
                                                        test_size=0.2,  #  测试集占 20%
                                                        random_state=seed,  # 设置随机种子，确保划分可复现
                                                        stratify=data.label)  # 按标签的分布进行分层划分，确保训练集和测试集的标签分布一致
    # 合并两个输入 -- skorch
    # SliceDict 是 skorch 提供的一种数据结构，类似于 Python 的字典，主要用于存储多模态数据（即多个输入）
    # f1_input 是一个键，表示单模态特征的名称，后面其他模态的输入我们可以命名为 f2_input、f3_input 等等
    x_train = SliceDict(f1_input=x_train.astype(np.float32)) # 将特征数据转换为 32 位浮点数
    x_test = SliceDict(f1_input=x_test.astype(np.float32))

    y_train, y_test = y_train, y_test
    print("Train:", Counter(y_train), "Test:", Counter(y_test))  # 输出训练集和测试集中各类别的样本数量分布
    
    # 将标签数据转换为二维数组形式，方便后续模型训练 
    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

    return x_train, x_test, y_train, y_test

# 加载多模态（两个模态：species和ko）特征数据集
def load_multi_features(seed: int, disease: str, feature: str, noise: float = 0.0) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """
    :param feature: type of feature: species or ko 
    :param seed: random seed for train and test split
    :param disease: prefix of dataset to open
    :return:
    """
    feature = feature.split(',')
    print(feature)
    if noise:
        f1_path = f"./Data/{disease}/{feature[0]}_noisy_{noise}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_noisy_{noise}_abundance.csv"
    else:
        f1_path = f"./Data/{disease}/{feature[0]}_abundance.csv"
        f2_path = f"./Data/{disease}/{feature[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])  # 检查两个模态的样本顺序是否一致

    f1_data = dataset(f1_path, use_cols=None) 
    f2_data = dataset(f2_path, use_cols=None)

    # 划分数据
    x_train_ko, x_test_ko, y_train_ko, y_test_ko = train_test_split(f1_data.data, f1_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f1_data.label)
    x_train_taxon, x_test_taxon, y_train_taxon, y_test_taxon = train_test_split(f2_data.data, f2_data.label.astype('int'),
                                                                    test_size=0.2,
                                                                    random_state=seed,
                                                                    stratify=f2_data.label)
    # 合并两个输入 -- skorch
    if (y_train_ko.all() == y_train_taxon.all()) and (y_test_ko.all() == y_test_taxon.all()):
        x_train = SliceDict(f1_input=x_train_ko.astype(np.float32), f2_input=x_train_taxon.astype(np.float32))
        x_test = SliceDict(f1_input=x_test_ko.astype(np.float32), f2_input=x_test_taxon.astype(np.float32))

        y_train, y_test = y_train_ko, y_test_ko
        print("训练集:", Counter(y_train), "测试集:", Counter(y_test))

        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)

        return x_train, x_test, y_train, y_test
    else:
        assert 0, "两个特征的标签不匹配"    # 同个样本的标签需要一致


def load_all_uni_features(disease: str, feature: str) -> tuple[np.ndarray, np.ndarray, list]:
    """
    返回单模态的全部样本（不做 train/test 划分），用于外部划分或 CV。

    Returns:
        X_all: numpy array shape (N, D)
        y_all: numpy array shape (N,)
        feature_names: list of feature column names
    """
    feature = feature.split(',')[0]
    path = f"./Data/{disease}/{feature}_abundance.csv"
    # 读取原始表以获取列名与标签
    df = pd.read_csv(path)
    labels = df.iloc[:, 1].values.astype('int')
    X = df.iloc[:, 2:].values.astype(np.float32)
    feature_names = list(df.columns[2:])
    print(f"Loaded all samples from {path}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, labels, feature_names


def load_all_multi_features(disease: str, feature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    返回多模态（两个模态）全部样本，不做划分，用于外部划分或 CV。

    Returns:
        species_X: ndarray (N, D1)
        ko_X: ndarray (N, D2)
        y_all: ndarray (N,)
        species_feature_names: list
        ko_feature_names: list
    """
    feature = feature.split(',')
    f1_path = f"./Data/{disease}/{feature[0]}_abundance.csv"
    f2_path = f"./Data/{disease}/{feature[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])

    df1 = pd.read_csv(f1_path)
    df2 = pd.read_csv(f2_path)

    y1 = df1.iloc[:, 1].values.astype('int')
    y2 = df2.iloc[:, 1].values.astype('int')
    if not np.array_equal(y1, y2):
        raise AssertionError("Labels do not match between modalities")

    species_X = df1.iloc[:, 2:].values.astype(np.float32)
    ko_X = df2.iloc[:, 2:].values.astype(np.float32)
    feature_names_sp = list(df1.columns[2:])
    feature_names_ko = list(df2.columns[2:])

    print(f"Loaded multimodal samples: species {species_X.shape}, ko {ko_X.shape}")
    return species_X, ko_X, y1, feature_names_sp, feature_names_ko
