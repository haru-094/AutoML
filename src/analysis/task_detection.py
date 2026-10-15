from config import config


def detect_task(y):
    """
    Looks at the target column and returns (task, reason).
    task is "classification" or "regression".
    """
    # Rule 1: text or True/False targets are always classification
    if y.dtype == "object" or y.dtype == "bool" or str(y.dtype) == "category":
        return "classification", "target contains text or True/False values"

    # Rule 2: numbers with decimals (like 12.5) are regression
    has_decimals = (y.dropna() % 1 != 0).any()
    if has_decimals:
        return "regression", "target has decimal values"

    # Rule 3: whole numbers with few unique values (0/1, ratings 1-5) -> classification
    n_unique = y.nunique()
    if n_unique <= config.MAX_CLASSES:
        return "classification", f"target has only {n_unique} unique whole numbers"

    # Rule 4: everything else is regression
    return "regression", f"target has {n_unique} unique numeric values"