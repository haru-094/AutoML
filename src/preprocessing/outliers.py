import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to training-set Tukey IQR bounds."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(values, 25, axis=0) - self.factor * (
            np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
        )
        self.upper_ = np.nanpercentile(values, 75, axis=0) + self.factor * (
            np.nanpercentile(values, 75, axis=0) - np.nanpercentile(values, 25, axis=0)
        )
        self.lower_ = np.nan_to_num(self.lower_, nan=-np.inf)
        self.upper_ = np.nan_to_num(self.upper_, nan=np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
