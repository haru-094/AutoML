import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_actual_vs_predicted(y_true, y_pred, name):
    """Scatter plot for regression. Points near the dashed line are good predictions."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="firebrick")

    low = min(np.min(y_true), np.min(y_pred))
    high = max(np.max(y_true), np.max(y_pred))
    ax.plot([low, high], [low, high], "k--", linewidth=1)    # the "perfect prediction" line

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(name)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, name):
    """Confusion matrix for classification. Rows = actual, columns = predicted."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=[str(c) for c in class_names],
        ax=ax, cmap="Reds", colorbar=False)
    ax.set_title(name)
    fig.tight_layout()
    return fig

STATUS_COLORS = {"Good fit": "green", "Slight overfit": "darkorange", "Overfit": "firebrick"}


def _score_columns(task):
    """Names of the train and validation score columns for this task."""
    if task == "regression":
        return "train_r2", "val_r2"
    return "train_f1_macro", "val_f1_macro"


def plot_model_scores(leaderboard, task):
    """One bar per model for the validation score. The best model is dark red."""
    _, val = _score_columns(task)
    board = leaderboard.sort_values(val)                  # lowest first, so the best ends on top

    colors = ["lightcoral"] * len(board)
    colors[-1] = "firebrick"

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(board) + 1.5))
    bars = ax.barh(board["model"], board[val], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)             # write the score at the end of each bar
    ax.margins(x=0.12)
    ax.set_xlabel("R²" if task == "regression" else "F1-macro")
    ax.set_title("Validation score by model (higher is better)")
    fig.tight_layout()
    return fig


def plot_overfitting(leaderboard, task):
    """Train vs validation score for each model, with the gap written next to the bars."""
    train, val = _score_columns(task)
    board = leaderboard.sort_values(val)                  # same order as the score chart

    positions = np.arange(len(board))
    height = 0.38

    fig, ax = plt.subplots(figsize=(7, 0.7 * len(board) + 1.5))
    ax.barh(positions + height / 2, board[train], height=height, color="lightgray", label="Train")
    ax.barh(positions - height / 2, board[val], height=height, color="firebrick", label="Validation")
    ax.set_yticks(positions)
    ax.set_yticklabels(board["model"])

    # Write the gap and its label to the right of each pair of bars
    for pos, (_, row) in zip(positions, board.iterrows()):
        x = max(row[train], row[val]) + 0.02
        ax.text(x, pos, f"gap {row['gap']:.2f}  {row['fit']}",
                va="center", fontsize=8, color=STATUS_COLORS[row["fit"]])

    ax.margins(x=0.35)                                    # room for the text on the right
    ax.set_xlabel("R²" if task == "regression" else "F1-macro")
    ax.set_title("Train vs validation: a big gap means overfitting")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig