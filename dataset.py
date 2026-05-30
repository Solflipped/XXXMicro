import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
from sklearn.preprocessing import StandardScaler
from skorch.helper import SliceDict
from utils import check_sample_order

def standardize_features(x: np.ndarray) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    return x.astype(np.float32)


class dataset(Dataset):
    def __init__(self, path: str, normalize: bool = True):
        print(f"Loading dataset from {path}")
        df = pd.read_csv(path)

        self.sample_ids = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values.astype(int)
        self.features = df.iloc[:, 2:].values.astype(np.float32)
        self.feature_names = list(df.columns[2:])

        if normalize:
            self.features = standardize_features(self.features)

    def get_data(self):
        return self.features, self.labels, self.feature_names


def load_uni_features(
    disease: str,
    features: List[str],
    normalize: bool = True,
):
    path = f"./Data/{disease}/{features[0]}_abundance.csv"
    d = dataset(path, normalize=normalize)
    x, y, feature_names = d.get_data()

    x = SliceDict(f1_input=x)
    print(
        f"Loaded all samples from {path}: {x['f1_input'].shape[0]} "
        f"samples, {x['f1_input'].shape[1]} features"
    )
    return x, y, feature_names


def load_multi_features(
    disease: str,
    features: List[str],
    normalize: bool = True,
):
    f1_path = f"./Data/{disease}/{features[0]}_abundance.csv"
    f2_path = f"./Data/{disease}/{features[1]}_abundance.csv"

    check_sample_order([f1_path, f2_path])

    d1 = dataset(f1_path, normalize=normalize)
    d2 = dataset(f2_path, normalize=normalize)

    x1, y1, n1 = d1.get_data()
    x2, y2, n2 = d2.get_data()

    assert np.array_equal(y1, y2), "Labels are inconsistent between modalities."

    print(f"Loaded all samples from {f1_path}: {x1.shape[0]} samples, {x1.shape[1]} features")
    print(f"Loaded all samples from {f2_path}: {x2.shape[0]} samples, {x2.shape[1]} features")

    x = SliceDict(
        f1_input=x1,
        f2_input=x2,
    )
    return x, y1, {"f1": n1, "f2": n2}
