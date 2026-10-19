import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("husl")
from config import config
from sklearn.feature_selection import mutual_info_classif


def get_column_types(X):
    """Splits the columns into numeric and non-numeric lists."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    return numeric_cols, categorical_cols


def overview_table(X):
    """One row per column: type, missing count, missing %, unique values."""
    table = pd.DataFrame({
        "type": X.dtypes.astype(str),
        "missing": X.isna().sum(),
        "missing_%": (X.isna().mean() * 100).round(1),
        "unique": X.nunique(),
    })
    return table


def class_balance(y):
    """Percentage of rows in each class, like 0: 62.0, 1: 38.0."""
    return (y.value_counts(normalize=True) * 100).round(1)


def draw_distribution(ax, values):
    """Histogram + smooth density curve (KDE) on the same axes."""
    ax.hist(values, bins=30, density=True, alpha=0.6)
    values.plot(kind="kde", ax=ax)
    ax.set_ylabel("Density")


def plot_target(y, task):
    """Bars for classification (class balance), distribution for regression."""
    fig, ax = plt.subplots(figsize=(6, 4))

    if task == "classification":
        y.value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of rows")
    else:
        draw_distribution(ax, y.dropna())

    ax.set_title(f"Target distribution: {y.name}")
    fig.tight_layout()
    return fig


def plot_numeric(X, col):
    """Distribution + boxplot for a numeric column (bars if it has few unique values)."""
    values = X[col].dropna()

    # Few unique values (like Pclass = 1, 2, 3) -> a bar chart reads better
    if values.nunique() <= config.MAX_DISCRETE:
        fig, ax = plt.subplots(figsize=(6, 4))
        values.value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title(f"{col} (counts)")
        fig.tight_layout()
        return fig

    # Many unique values -> distribution on the left, boxplot on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    draw_distribution(ax1, values)
    ax1.set_title(f"{col} distribution (skew = {values.skew():.2f})")
    ax2.boxplot(values, vert=False)
    ax2.set_yticks([])
    ax2.set_title(f"{col} boxplot")
    fig.tight_layout()
    return fig


def plot_categorical(X, col):
    """Bar chart of the most common categories."""
    counts = X[col].value_counts().head(config.TOP_N_CATEGORIES)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"{col} (top {config.TOP_N_CATEGORIES} categories)")
    ax.set_ylabel("Number of rows")
    fig.tight_layout()
    return fig


def plot_missing(X):
    """Bar chart of missing % per column. Returns None if nothing is missing."""
    missing = X.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing values (%)")
    ax.set_ylabel("% of rows")
    fig.tight_layout()
    return fig


def plot_correlation(X):
    """Red heatmap of correlation strength between numeric columns."""
    numeric_cols, _ = get_column_types(X)
    if len(numeric_cols) < 2:
        return None

    corr = X[numeric_cols].corr()
    size = max(6, len(numeric_cols) * 0.5)        # bigger figure for more columns

    fig, ax = plt.subplots(figsize=(size, size))
    # Shade by strength (absolute value): white = no relationship, dark red = strong
    image = ax.imshow(corr.abs(), cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    fig.colorbar(image, shrink=0.8, label="strength (0 to 1)")

    # Real numbers (with sign) inside the squares, only when there are few columns
    if len(numeric_cols) <= config.MAX_ANNOTATED_COLS:
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                value = corr.iloc[i, j]
                color = "white" if abs(value) > 0.6 else "black"   # readable on dark red
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    return fig


def plot_target_correlation(X, y):
    """Red bar chart: numeric features most correlated with the target (regression)."""
    numeric_cols, _ = get_column_types(X)
    corr = X[numeric_cols].corrwith(y).dropna()

    # Keep the strongest ones (positive or negative)
    order = corr.abs().sort_values(ascending=False).index
    corr = corr[order].head(config.TOP_N_FEATURES)

    if corr.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(corr) + 1.5))
    corr[::-1].plot(kind="barh", ax=ax, color="firebrick")   # strongest at the top
    ax.set_xlim(-1, 1)
    ax.set_title(f"Correlation with target: {y.name}")
    fig.tight_layout()
    return fig


def strong_pairs(X, threshold=0.7):
    """Table of feature pairs whose correlation is at least the threshold."""
    numeric_cols, _ = get_column_types(X)
    corr = X[numeric_cols].corr()

    rows = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):     # j > i so each pair appears once
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                rows.append({"feature_1": numeric_cols[i],
                             "feature_2": numeric_cols[j],
                             "correlation": round(r, 2)})

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("correlation", key=abs, ascending=False)
    return table

def imbalance_ratio(y):
    """How many times bigger the largest class is than the smallest."""
    counts = y.value_counts()
    return round(counts.max() / counts.min(), 1)


def plot_feature_by_class(X, y, col):
    """Shows how one feature differs between the classes."""
    is_numeric = pd.api.types.is_numeric_dtype(X[col])
    many_values = X[col].nunique() > config.MAX_DISCRETE

    fig, ax = plt.subplots(figsize=(7, 4))

    if is_numeric and many_values:
        # Numeric with many values: one boxplot per class
        classes = y.value_counts().index[:config.TOP_N_CATEGORIES]
        groups = [X.loc[y == c, col].dropna() for c in classes]
        ax.boxplot(groups)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_xlabel(y.name)
        ax.set_ylabel(col)
        ax.set_title(f"{col} by {y.name}")
    else:
        # Categories (or numbers with few values): % of each class inside each category
        top = X[col].value_counts().index[:config.TOP_N_CATEGORIES]
        keep = X[col].isin(top)
        table = pd.crosstab(X.loc[keep, col], y[keep], normalize="index") * 100
        table.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel(f"% of rows in each {y.name} class")
        ax.set_title(f"{y.name} rate for each {col}")
        ax.legend(title=y.name, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig


def plot_feature_signal(X, y):
    """Red bar chart: numeric features that tell us the most about the class."""
    numeric_cols, _ = get_column_types(X)
    if not numeric_cols:
        return None

    # The score cannot handle empty cells, so fill them with the median (for this chart only)
    X_num = X[numeric_cols].fillna(X[numeric_cols].median())
    y_codes = y.astype("category").cat.codes        # turns class names into 0, 1, 2...

    scores = mutual_info_classif(X_num, y_codes, random_state=config.RANDOM_STATE)
    scores = pd.Series(scores, index=numeric_cols).sort_values(ascending=False)
    scores = scores.head(config.TOP_N_FEATURES)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(scores) + 1.5))
    scores[::-1].plot(kind="barh", ax=ax, color="firebrick")     # strongest at the top
    ax.set_title("Feature signal for the target (mutual information)")
    fig.tight_layout()
    return fig

def make_grid(n_plots):
    """Creates a grid of empty plots. Returns the figure and a flat list of axes."""
    n_cols = config.GRID_COLUMNS
    n_rows = -(-n_plots // n_cols)                  # divide and round up
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).ravel()
    for ax in axes[n_plots:]:
        ax.axis("off")                              # hide the unused boxes
    return fig, axes


def plot_numeric_grid(X, y, task):
    """Distribution of each continuous feature (colored by class for classification)."""
    numeric_cols, _ = get_column_types(X)
    cols = [c for c in numeric_cols if X[c].nunique() > config.MAX_DISCRETE]
    cols = cols[:config.GRID_MAX_PLOTS]
    if not cols:
        return None

    data = X[cols].copy()
    hue = None
    if task == "classification":
        data["target"] = y.astype(str)              # text, so each class gets its own color
        hue = "target"

    fig, axes = make_grid(len(cols))
    for ax, col in zip(axes, cols):
        sns.histplot(data=data, x=col, hue=hue, bins=30, ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    return fig


def plot_category_grid(X, y, task):
    """Bars for each categorical or low-count feature (class percentages for classification)."""
    numeric_cols, categorical_cols = get_column_types(X)
    discrete = [c for c in numeric_cols if X[c].nunique() <= config.MAX_DISCRETE]
    cols = (categorical_cols + discrete)[:config.GRID_MAX_PLOTS]
    if not cols:
        return None

    fig, axes = make_grid(len(cols))
    for ax, col in zip(axes, cols):
        top = X[col].value_counts().index[:config.TOP_N_CATEGORIES]
        keep = X[col].isin(top)

        if task == "classification":
            table = pd.crosstab(X.loc[keep, col], y[keep], normalize="index") * 100
            table.plot(kind="bar", stacked=True, ax=ax, legend=False)
            ax.set_ylabel("% in each class")
        else:
            X.loc[keep, col].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_ylabel("Number of rows")
        ax.set_title(col)
    fig.tight_layout()
    return fig