from sklearn.model_selection import train_test_split
from config import config
import warnings


def split_data(df, target, stratify=False):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[target])
    if df.empty:
        raise ValueError(f"Target column '{target}' contains no non-missing values.")
    if len(df) < 3:
        raise ValueError("At least 3 rows with a non-missing target are required.")

    dropped = len(df) - len(df.dropna(subset=[target]))
    if dropped:
        warnings.warn(f"Dropped {dropped} row(s) with missing target values.", UserWarning, stacklevel=2)

    X = df.drop(columns=[target])
    y = df[target]
    strat = y if stratify else None

    try:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=strat)

        val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_share, random_state=config.RANDOM_STATE,
            stratify=y_rest if strat is not None else None)
    except ValueError as exc:
        raise ValueError(f"Could not create train/validation/test splits: {exc}") from exc

    if len(X_train) < 2 or len(X_val) < 2 or len(X_test) < 2:
        raise ValueError("Each split must contain at least 2 rows.")

    return X_train, X_val, X_test, y_train, y_val, y_test
