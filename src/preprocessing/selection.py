import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SafeVarianceSelector(BaseEstimator, TransformerMixin):
    """Remove constant columns, retaining one column if every input is constant."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        variances = np.nanvar(values, axis=0)
        self.support_ = np.isfinite(variances) & (variances > 0)
        if not self.support_.any() and values.shape[1]:
            self.support_[0] = True
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return values[:, self.support_]
