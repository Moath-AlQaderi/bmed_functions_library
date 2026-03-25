# -*- coding: utf-8 -*-
"""
The below functions automates the use of sciket-learn's StratifiedGroupKFold. 
It keeps class balance like stratified k-fold, while also enforcing that all rows with the same ID stay in the same fold.
This is crucial to prevent data leakage in datasets where multiple samples from the same patient, for example, are recorded.
"""

from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np

def stratified_group_kfold_split(X, y, id_column, n_splits=5, shuffle=True, random_state=42):
    """
    Perform stratified k-fold cross-validation with grouping by ID to prevent leakage.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature table including the ID column.
    y : array-like
        Target labels.
    id_column : str
        Name of the column in X containing the group/patient/sample ID.
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, default=42
        Random seed.

    Yields
    ------
    train_idx : np.ndarray
        Indices for the training set.
    test_idx : np.ndarray
        Indices for the test set.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if id_column not in X.columns:
        raise ValueError(f"'{id_column}' not found in X columns.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    groups = X[id_column].values
    y = np.asarray(y)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for train_idx, test_idx in sgkf.split(X, y, groups=groups):
        yield train_idx, test_idx

# Example

X = pd.DataFrame({
    "patient_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "f2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]

for fold, (train_idx, test_idx) in enumerate(
    stratified_group_kfold_split(X, y, id_column="patient_id", n_splits=3),
    start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

    print(f"Fold {fold}")
    print("Train patient IDs:", sorted(X_train["patient_id"].unique()))
    print("Test patient IDs: ", sorted(X_test["patient_id"].unique()))
    print()

"""
!Note!
If you want only the feature columns in training/testing, then you must drop the ID column after splitting:
"""
X_train_features = X.iloc[train_idx].drop(columns=["patient_id"])
X_test_features = X.iloc[test_idx].drop(columns=["patient_id"])
