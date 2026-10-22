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

    # Rule 1: strings, booleans and categorical targets are classification.
    if (pd.api.types.is_string_dtype(y.dtype) or pd.api.types.is_bool_dtype(y.dtype)
            or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(y.dtype)):
        return "classification", "target contains text or True/False values"

    # Rule 2: numbers with decimals (like 12.5) are regression
    if not pd.api.types.is_numeric_dtype(y.dtype):
        return "classification", "target contains non-numeric values"
    has_decimals = (y % 1 != 0).any()
    if has_decimals:
        return "regression", "target has decimal values"

    # Rule 3: whole numbers with few unique values (0/1, ratings 1-5) -> classification
    n_unique = y.nunique()
    if n_unique <= config.MAX_CLASSES:
        return "classification", f"target has only {n_unique} unique whole numbers"

    # Rule 4: everything else is regression
    return "regression", f"target has {n_unique} unique numeric values"
