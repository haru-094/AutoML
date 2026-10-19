import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config import config
from src.modeling.model_list import get_models
from src.modeling.evaluator import score_regression, score_classification, score_overfitting
from src.preprocessing.pipeline_builder import build_preprocessor


def train_models(X_train, y_train, X_val, y_val, task, model_names, strategy, clip):
    """
    Trains every selected model. Returns (leaderboard, fitted_models, errors, encoder).
    """
    models = get_models(task)
    errors = {}
    fitted = {}
    rows = []

    # Classification: turn class names into numbers 0, 1, 2... (XGBoost needs this)
    encoder = None
    if task == "classification":
        encoder = LabelEncoder().fit(y_train)
        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)

    for name in model_names:
        info = models[name]

        # Skip slow models on big data instead of freezing the app
        if info.get("slow") and len(X_train) > config.SVM_MAX_ROWS:
            errors[name] = f"skipped: more than {config.SVM_MAX_ROWS} rows (too slow)"
            continue

        # One pipeline per model: preprocessing + model together
        preprocessor = build_preprocessor(X_train, task, strategy, clip, scale=info["needs_scaling"])
        pipeline = Pipeline([("prep", preprocessor), ("model", info["model"])])

        try:
            start = time.time()
            pipeline.fit(X_train, y_train)                 # fit on TRAIN only
            seconds = time.time() - start

            if task == "regression":
                val = score_regression(y_val, pipeline.predict(X_val))
                train = score_regression(y_train, pipeline.predict(X_train))
                row = {"model": name, "val_r2": val["r2"], "val_rmse": val["rmse"],
                       "val_mae": val["mae"], "train_r2": train["r2"]}
            else:
                val = score_classification(y_val, pipeline.predict(X_val))
                train = score_classification(y_train, pipeline.predict(X_train))
                row = {"model": name, "val_f1_macro": val["f1_macro"],
                       "val_balanced_acc": val["balanced_acc"], "val_accuracy": val["accuracy"],
                       "train_f1_macro": train["f1_macro"]}

            row["seconds"] = round(seconds, 1)
            rows.append(row)
            fitted[name] = pipeline

        except Exception as e:                             # one failing model must not stop the rest
            errors[name] = str(e)

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        rank_by = "val_r2" if task == "regression" else "val_f1_macro"
        leaderboard = leaderboard.sort_values(rank_by, ascending=False).reset_index(drop=True)
        leaderboard.index = leaderboard.index + 1          # rank starts at 1
        leaderboard = leaderboard.round(4)
        leaderboard = leaderboard.round(4)
        leaderboard = score_overfitting(leaderboard, task)

    return leaderboard, fitted, errors, encoder