import pandas as pd
from config import config


def detect_task(y):
    """
    Looks at the target column and returns (task, reason).
    task is "classification" or "regression".
    """
    y = y.dropna()
    if y.empty:
        raise ValueError("Cannot detect the task: target contains no non-missing values.")

    if (pd.api.types.is_string_dtype(y.dtype) or pd.api.types.is_bool_dtype(y.dtype)
            or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y.dtype)):
        return "classification", "target contains text or True/False values"

    if not pd.api.types.is_numeric_dtype(y.dtype):
        return "classification", "target contains non-numeric values"
    has_decimals = (y % 1 != 0).any()
    if has_decimals:
        return "regression", "target has decimal values"

    n_unique = y.nunique()
    if n_unique <= config.MAX_CLASSES:
        return "classification", f"target has only {n_unique} unique whole numbers"

    return "regression", f"target has {n_unique} unique numeric values"
