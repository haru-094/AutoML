from config import config


def split_columns(X):
    """Splits the columns into 3 groups: continuous, discrete (few values), and text."""
    numeric = X.select_dtypes(include="number").columns
    text = X.select_dtypes(exclude="number").columns.tolist()

    continuous = [c for c in numeric if X[c].nunique() > config.MAX_DISCRETE]
    discrete = [c for c in numeric if X[c].nunique() <= config.MAX_DISCRETE]

    return continuous, discrete, text