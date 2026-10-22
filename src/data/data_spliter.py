from sklearn.model_selection import train_test_split
from config import config
import warnings
import math


def split_data(df, target, stratify=False):
    # Stop early with a clear message if the target column doesn't exist
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Rows without a target value cannot be used for training.
    missing_targets = int(df[target].isna().sum())
    df = df.dropna(subset=[target])
    if df.empty:
        raise ValueError(f"Target column '{target}' contains no non-missing values.")
    if missing_targets:
        warnings.warn(f"Dropped {missing_targets} row(s) with missing target values.", UserWarning, stacklevel=2)
    if len(df) < 3:
        raise ValueError("At least 3 rows with a non-missing target are required for train/validation/test splits.")

    # 2. Separate features (X) from the target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Stratify = keep the same class proportions in every split
    #    (only for classification, and only if every class has enough rows)
    strat = y if stratify else None
    if strat is not None:
        counts = y.value_counts()
        # Each of the three output partitions must be able to contain every class.
        if counts.min() < 3:
            rare = counts[counts < 3].to_dict()
            raise ValueError(f"Cannot stratify train/validation/test: every class needs at least 3 rows. Rare classes: {rare}")
        n_test = math.ceil(len(y) * config.TEST_SIZE)
        n_rest = len(y) - n_test
        val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
        n_val = math.ceil(n_rest * val_share)
        n_train = len(y) - n_test - n_val
        if min(n_test, n_val) < len(counts) or n_train < len(counts):
            raise ValueError("Not enough rows to place every class in train, validation, and test. Reduce the number of classes or change split proportions.")

    # 4. First cut: take out the TEST set (20%)
    try:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=strat)

        # 5. Second cut: split the rest into TRAIN and VALIDATION
        # 20% of the original = 25% of what is left (0.2 / 0.8)
        val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)

        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_share, random_state=config.RANDOM_STATE,
            stratify=y_rest if strat is not None else None)
    except ValueError as exc:
        kind = "stratified " if strat is not None else ""
        raise ValueError(f"Could not create {kind}train/validation/test splits: {exc}") from exc

    if len(X_train) < 2 or len(X_val) < 2 or len(X_test) < 2:
        raise ValueError("Each split needs at least 2 rows to support model training and evaluation metrics. Increase the dataset size or adjust split proportions.")

    return X_train, X_val, X_test, y_train, y_val, y_test
