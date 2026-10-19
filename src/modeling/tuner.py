from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from config import config
from src.modeling.model_list import get_models
from src.modeling.param_grids import PARAM_GRIDS
from src.preprocessing.pipeline_builder import build_preprocessor


def tune_model(name, task, X_train, y_train, strategy, clip, n_iter=None):
    """
    Searches for better settings for one model, using cross-validation on TRAIN only.
    y_train must already be numbers (for classification, the encoded classes).
    Returns the finished search, or None if the model has nothing to tune.
    """
    grid = PARAM_GRIDS.get(name, {})
    if not grid:
        return None

    # The same pipeline as in the trainer: preprocessing + model
    info = get_models(task)[name]
    preprocessor = build_preprocessor(X_train, task, strategy, clip, scale=info["needs_scaling"])
    pipeline = Pipeline([("prep", preprocessor), ("model", info["model"])])

    # Inside a pipeline, settings are named step__setting: "alpha" becomes "model__alpha"
    param_space = {f"model__{key}": values for key, values in grid.items()}

    # Cannot try more combinations than exist
    n_combinations = 1
    for values in grid.values():
        n_combinations *= len(values)
    n_iter = min(n_iter or config.TUNE_N_ITER, n_combinations)

    search = RandomizedSearchCV(
        pipeline, param_space,
        n_iter=n_iter,
        scoring="r2" if task == "regression" else "f1_macro",     # same rule as the leaderboard
        cv=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
        n_jobs=1,                  # one at a time: safest on a Mac (models already use all cores)
    )
    search.fit(X_train, y_train)   # after the search, search.best_estimator_ is refitted on all of train
    return search