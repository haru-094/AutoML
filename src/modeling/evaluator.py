import numpy as np
from config import config
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             f1_score, balanced_accuracy_score, accuracy_score)


def score_regression(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def score_classification(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }

def score_overfitting(leaderboard, task):
    """Adds a 'gap' column (train - validation) and a 'fit' label to the leaderboard."""
    score = "r2" if task == "regression" else "f1_macro"
    board = leaderboard.copy()

    board["gap"] = (board[f"train_{score}"] - board[f"val_{score}"]).round(4)

    def label(gap):
        if gap < config.OVERFIT_SMALL_GAP:
            return "Good fit"
        if gap < config.OVERFIT_BIG_GAP:
            return "Slight overfit"
        return "Overfit"

    board["fit"] = board["gap"].apply(label)
    return board