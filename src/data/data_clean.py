import pandas as pd
from config import config


def find_bad_columns(X_train):
    """Returns a dict like {"customer_id": "looks like an ID"}."""
    bad = {}
    n_rows = len(X_train)

    for col in X_train.columns:
        values = X_train[col]
        n_unique = values.nunique()

        if values.isna().mean() > config.MAX_MISSING_RATIO:
            bad[col] = "too many missing values"

        elif n_unique <= 1:
            bad[col] = "same value in every row"

        elif not pd.api.types.is_float_dtype(values) and n_unique / n_rows >= config.ID_UNIQUE_RATIO:
            bad[col] = "looks like an ID (almost every value is unique)"

        elif values.dtype == "object" and n_unique > config.MAX_CATEGORY_UNIQUE:
            bad[col] = "too many different text values"

    return bad


def drop_columns(X, bad_columns):
    """Drops the bad columns from any DataFrame (train, val, or test)."""
    return X.drop(columns=list(bad_columns), errors="ignore")