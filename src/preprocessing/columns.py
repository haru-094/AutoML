from config import config


def split_columns(X, categorical_overrides=None):
    """Splits the columns into 3 groups: continuous, discrete (few values), and text."""
    numeric = X.select_dtypes(include="number").columns
    text = X.select_dtypes(exclude="number").columns.tolist()

    categorical_overrides = set(categorical_overrides or [])
    continuous = [c for c in numeric if X[c].nunique() > config.MAX_DISCRETE and c not in categorical_overrides]
    discrete = [c for c in numeric if X[c].nunique() <= config.MAX_DISCRETE or c in categorical_overrides]

    return continuous, discrete, text
