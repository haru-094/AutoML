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

    if continuous:
        steps = [("fill", make_numeric_imputer(impute_strategy)), ("outliers", IQRClipper())]
        if scale:
            scaler = RobustScaler() if task == "classification" else StandardScaler()
            steps.append(("scale", scaler))
        parts.append(("continuous", Pipeline(steps), continuous))

    if discrete:
        parts.append(("discrete", make_categorical_imputer(), discrete))

    if text:
        steps = [("fill", make_categorical_imputer()),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        parts.append(("text", Pipeline(steps), text))

    columns = ColumnTransformer(parts, verbose_feature_names_out=False, sparse_threshold=0)

    return columns
