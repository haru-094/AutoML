import math
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import config
from src.modeling.model_list import get_models
from src.modeling.param_grids import PARAM_GRIDS
from src.preprocessing.pipeline_builder import build_preprocessor
from src.preprocessing.selection import SafeVarianceSelector


def tune_model(name, task, X_train, y_train, strategy, n_iter=None, categorical_overrides=None):
    """
    Searches for better settings for one model, using cross-validation on TRAIN only.
    y_train must already be numbers (for classification, the encoded classes).
    Returns the finished search, or None if the model has nothing to tune.
    """
    grid = PARAM_GRIDS.get(name, {})
    if not grid:
        return None

    if len(X_train) < 2:
        raise ValueError("Tuning requires at least 2 training rows.")
    folds = min(config.CV_FOLDS, len(X_train))
    if task == "classification":
        class_counts = pd.Series(y_train).value_counts()
        folds = min(folds, int(class_counts.min()))
        if folds < 2:
            raise ValueError("Tuning needs at least 2 training examples in every class for cross-validation.")
    if name == "KNN":
        max_fold_train = len(X_train) - math.ceil(len(X_train) / folds)
        grid = {**grid, "n_neighbors": [k for k in grid["n_neighbors"] if k <= max_fold_train]}
        if not grid["n_neighbors"]:
            grid["n_neighbors"] = [1]

    info = get_models(task)[name]
    preprocessor = build_preprocessor(X_train, task, strategy, scale=info["needs_scaling"],
                                      categorical_overrides=categorical_overrides)
    pipeline = Pipeline([("prep", preprocessor), ("selection", SafeVarianceSelector()),
                         ("model", info["model"])])

    param_space = {f"model__{key}": values for key, values in grid.items()}

    n_combinations = 1
    for values in grid.values():
        n_combinations *= len(values)
    n_iter = min(n_iter or config.TUNE_N_ITER, n_combinations)

    search = RandomizedSearchCV(
        pipeline, param_space,
        n_iter=n_iter,
        scoring="r2" if task == "regression" else "f1_macro",
        cv=folds,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search
