from sklearn.model_selection import train_test_split
from config import config


def split_data(df, target, stratify=False):
    # 1. Rows without a target value cannot be used for training
    df = df.dropna(subset=[target])

    # 2. Separate features (X) from the target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Stratify = keep the same class proportions in every split
    #    (only for classification, and only if every class has enough rows)
    strat = None
    if stratify and y.value_counts().min() >= 3:
        strat = y

    # 4. First cut: take out the TEST set (20%)
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=strat,
    )

    # 5. Second cut: split the rest into TRAIN and VALIDATION
    #    20% of the original = 25% of what is left (0.2 / 0.8)
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_share,
        random_state=config.RANDOM_STATE,
        stratify=y_rest if strat is not None else None,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test