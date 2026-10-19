import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clips each column to its IQR limits (learned on train only)."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)              # float, so decimals are never cut off
        q1 = np.percentile(X, 25, axis=0)           # axis=0 -> one value per column
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        """The columns keep their names (we only change their values)."""
        return np.asarray(input_features, dtype=object)