import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder

from src.preprocessing.columns import split_columns
from src.preprocessing.imputer import make_numeric_imputer, make_categorical_imputer
from src.preprocessing.outlier import OutlierHandler


def build_preprocessor(X_train, task, impute_strategy="median", clip_outliers=False, scale=True):
    """
    Builds the preprocessing recipe. It is NOT fitted yet.
    X_train is only used to look at the column names and types.
    """
    continuous, discrete, text = split_columns(X_train)
    parts = []

    # 1. Continuous numbers: fill -> clip (optional) -> scale (optional)
    if continuous:
        steps = [("fill", make_numeric_imputer(impute_strategy))]
        if clip_outliers:
            steps.append(("clip", OutlierHandler()))
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

    return ColumnTransformer(parts, verbose_feature_names_out=False, sparse_threshold=0)


def transform_to_frame(preprocessor, X):
    """Applies a FITTED preprocessor and returns a DataFrame with proper column names."""
    data = preprocessor.transform(X)
    return pd.DataFrame(data, columns=preprocessor.get_feature_names_out(), index=X.index)