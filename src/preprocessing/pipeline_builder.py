from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

from src.preprocessing.columns import split_columns
from src.preprocessing.imputer import make_numeric_imputer, make_categorical_imputer
from src.preprocessing.outliers import IQRClipper


def build_preprocessor(X_train, task, impute_strategy="median", scale=True, categorical_overrides=None):
    """
    Builds the preprocessing recipe. It is NOT fitted yet.
    X_train is only used to look at the column names and types.
    """
    continuous, discrete, text = split_columns(X_train, categorical_overrides)
    parts = []

    # 1. Continuous numbers: fill -> scale (optional)
    if continuous:
        steps = [("fill", make_numeric_imputer(impute_strategy)), ("outliers", IQRClipper())]
        if scale:
            scaler = RobustScaler() if task == "classification" else StandardScaler()
            steps.append(("scale", scaler))
        parts.append(("continuous", Pipeline(steps), continuous))

    # 2. Few-value numbers (codes, 0/1): fill with the most frequent value, nothing else
    if discrete:
        parts.append(("discrete", make_categorical_imputer(), discrete))

    # 3. Text: fill -> one-hot encode
    if text:
        steps = [("fill", make_categorical_imputer()),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        parts.append(("text", Pipeline(steps), text))

    columns = ColumnTransformer(parts, verbose_feature_names_out=False, sparse_threshold=0)
    # Constant transformed columns add no predictive signal and can destabilize
    # models when a dataset has very few informative features. This selector is
    # unsupervised, so preprocessing also remains usable without y values.
    return columns
