"""Lightweight feature selection utilities.

This module provides a CV-safe, focused implementation for the
RandomForest + RFECV pipeline and a small wrapper `transfer_feature_selection`
that keeps backward-compatible argument order used by `train.py` in this repo.

Only the functionality needed by the training script is implemented to keep
the module robust and easy to maintain. If you need other methods later we
can add them in a similar, well-tested fashion.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import RepeatedStratifiedKFold


def _randomforest_rfecv(train_X: np.ndarray,
                         train_y: np.ndarray,
                         test_X: np.ndarray,
                         feature_list: List[str],
                         n_estimators: int = 200,
                         rfecv_n_splits: int = 5,
                         rfecv_n_repeats: int = 1,
                         random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Perform SelectFromModel (RF) followed by RFECV (RF) and return reduced arrays.

    Steps:
    - Replace NaNs with 0.0 (safe for tree-based models).
    - Run a SelectFromModel(RandomForest) to reduce feature count to ~sqrt(p) to
      speed up RFECV on very high-dimensional inputs.
    - Run RFECV on the reduced feature set and return transformed arrays
      (train/test) and the list of final selected feature names.

    This function is intended to be called on training data inside each CV fold.
    """
    # defensive copies and NaN handling
    train_X = np.nan_to_num(np.asarray(train_X), nan=0.0)
    test_X = np.nan_to_num(np.asarray(test_X), nan=0.0)

    if isinstance(feature_list, (np.ndarray, list)):
        feat_names = list(feature_list)
    else:
        # fallback, generate generic names
        feat_names = [f'f{i}' for i in range(train_X.shape[1])]

    train_df = pd.DataFrame(train_X, columns=feat_names)
    test_df = pd.DataFrame(test_X, columns=feat_names)

    # 1) preliminary RF-based filter to reduce dimensionality
    est_pre = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                     class_weight='balanced', n_jobs=-1)
    # keep a larger preselected set to allow RFECV more room to choose from
    # but never ask SelectFromModel for more features than exist
    preselect_target = max(1, int(4 * np.sqrt(train_df.shape[1])))
    max_feats = min(train_df.shape[1], preselect_target)
    selector = SelectFromModel(est_pre, max_features=max_feats)
    selector.fit(train_df, train_y)
    selected_mask = selector.get_support()
    selected_names = list(train_df.columns[selected_mask])

    # if nothing selected (rare), fall back to top-k by importance
    if len(selected_names) == 0:
        est_pre.fit(train_df, train_y)
        importances = est_pre.feature_importances_
        # fallback keep roughly 4*sqrt(p) features
        preselect_target = max(1, int(4 * np.sqrt(train_df.shape[1])))
        topk = min(train_df.shape[1], preselect_target)
        idx = np.argsort(importances)[-topk:]
        selected_names = [train_df.columns[i] for i in idx]

    X_train_reduced = train_df.loc[:, selected_names].values
    X_test_reduced = test_df.loc[:, selected_names].values

    # 2) RFECV on reduced set
    est_rfecv = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                       class_weight='balanced', n_jobs=-1)
    cv = RepeatedStratifiedKFold(n_splits=rfecv_n_splits, n_repeats=rfecv_n_repeats, random_state=random_state)
    rfecv = RFECV(estimator=est_rfecv, step=0.1, cv=cv, scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_train_reduced, train_y)

    final_mask = rfecv.support_
    final_feature_names = list(np.array(selected_names)[final_mask])
    X_train_final = rfecv.transform(X_train_reduced)
    X_test_final = rfecv.transform(X_test_reduced)

    return X_train_final, X_test_final, final_feature_names


def transfer_feature_selection(train_X: np.ndarray,
                               train_y: np.ndarray,
                               test_X: np.ndarray,
                               feature_list: List[str],
                               fs_method: str = 'all',
                               **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compatibility wrapper used by `train.py` (simplified).

    New, simplified signature intentionally omits `taxon` and `stage` since the
    lightweight implementation does not use them. Keep fs_method limited to
    'all' (no-op) or 'RandomForest+RFECV'. Additional RF/RFECV control
    parameters may be provided via kwargs:
      - n_estimators (int)
      - rfecv_n_splits (int)
      - rfecv_n_repeats (int)
      - random_state (int)

    Returns:
      (train_X_sel, test_X_sel, feature_list_sel)
    """
    if fs_method is None or fs_method.lower() in ('none', 'all'):
        # no selection
        return train_X, test_X, feature_list

    if fs_method == 'RandomForest+RFECV':
        return _randomforest_rfecv(train_X=train_X,
                                   train_y=train_y,
                                   test_X=test_X,
                                   feature_list=feature_list,
                                   n_estimators=kwargs.get('n_estimators', 200),
                                   rfecv_n_splits=kwargs.get('rfecv_n_splits', 5),
                                   rfecv_n_repeats=kwargs.get('rfecv_n_repeats', 1),
                                   random_state=kwargs.get('random_state', 0))

    # If method not supported, raise a clear error so user knows to pick another
    raise ValueError(f"fs_method '{fs_method}' not implemented in this lightweight module. Use 'RandomForest+RFECV' or 'all'/None.")


def CVout_feature_selection(X: np.ndarray, label: np.ndarray, feature_list: List[str], fs_method: str):
    """Minimal replacement for original CVout helper.

    Only supports 'all' (no-op) here. For supervised methods use
    `transfer_feature_selection` inside each CV fold.
    """
    if fs_method is None or fs_method.lower() in ('none', 'all'):
        return X, feature_list
    raise ValueError("CVout_feature_selection currently supports only 'all' (no-op). Use transfer_feature_selection inside CV folds for supervised selection.")

