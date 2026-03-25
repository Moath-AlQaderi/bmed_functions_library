# -*- coding: utf-8 -*-
"""
Global MinMax scaling, where one single min and max are used across the entire dataset (not per feature) to scale features.
"""

import numpy as np

def global_minmax_scaler(X):
    """
    Scales features in X using global min and max.

    Parameters:
        X (array-like): Input data (2D or 1D)

    Returns:
        np.ndarray: Scaled data in range [0, 1]
    """
    X = np.array(X, dtype=float)

    global_min = np.min(X)
    global_max = np.max(X)

    # Avoid division by zero
    if global_max == global_min:
        return np.zeros_like(X)

    X_scaled = (X - global_min) / (global_max - global_min)

    return X_scaled

""" 
Key point:
Unlike sklearn.preprocessing.MinMaxScaler, which scales each column independently, this function:
    Uses one global min and max
    Preserves relative magnitudes across all features
"""

# Example:

X = [[1, 2, 3],
     [4, 5, 6]]

scaled_X = global_minmax_scaler(X)
print(scaled_X)