# Settings to search for each model. The search picks random combinations from these lists.

FOREST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5],
}

BOOSTING = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.7, 0.85, 1.0],
}

XGB = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}

LGBM = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 40],
}

KNN = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
}

PARAM_GRIDS = {
    "Linear Regression": {},                       # nothing to tune
    "Ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 2, 5, 10]},
    "Random Forest": FOREST,
    "Gradient Boosting": BOOSTING,
    "XGBoost": XGB,
    "LightGBM": LGBM,
    "SVR": {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "gamma": ["scale", "auto"]},
    "SVC": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "KNN": KNN,
}