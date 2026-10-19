from sklearn.impute import SimpleImputer
from config import config


def make_numeric_imputer(strategy=None):
    """Imputer for numeric columns: mean, median or most_frequent."""
    strategy = strategy or config.DEFAULT_NUMERIC_IMPUTE
    if strategy not in config.IMPUTE_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {config.IMPUTE_STRATEGIES}")
    return SimpleImputer(strategy=strategy)


def make_categorical_imputer():
    """Imputer for text columns: always the most frequent value."""
    return SimpleImputer(strategy=config.DEFAULT_CATEGORICAL_IMPUTE)


def suggest_strategy(X_train):
    """Suggests median if any numeric column is strongly skewed, otherwise mean."""
    numeric = X_train.select_dtypes(include="number")
    if numeric.empty:
        return config.DEFAULT_NUMERIC_IMPUTE

    skew = numeric.skew().abs()
    if (skew > 1).any():
        return "median"
    return "mean"