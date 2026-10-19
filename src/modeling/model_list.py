from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from config import config

RS = config.RANDOM_STATE
JOBS = config.N_JOBS

# Optional libraries: if one fails to load, we skip it instead of crashing
missing_libraries = []

try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:                      # not only ImportError: XGBoost raises its own error type
    XGBRegressor = XGBClassifier = None
    missing_libraries.append("XGBoost")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    LGBMRegressor = LGBMClassifier = None
    missing_libraries.append("LightGBM")


def get_models(task):
    """
    Returns {name: {"model": ..., "needs_scaling": True/False, "slow": True/False}}
    """
    if task == "regression":
        models = {
            "Linear Regression": {"model": LinearRegression(), "needs_scaling": True},
            "Ridge": {"model": Ridge(), "needs_scaling": True},
            "Random Forest": {"model": RandomForestRegressor(random_state=RS, n_jobs=JOBS), "needs_scaling": False},
            "Gradient Boosting": {"model": GradientBoostingRegressor(random_state=RS), "needs_scaling": False},
            "SVR": {"model": SVR(), "needs_scaling": True, "slow": True},
            "KNN": {"model": KNeighborsRegressor(n_jobs=JOBS), "needs_scaling": True},
        }
        if XGBRegressor:
            models["XGBoost"] = {"model": XGBRegressor(random_state=RS, n_jobs=JOBS), "needs_scaling": False}
        if LGBMRegressor:
            models["LightGBM"] = {"model": LGBMRegressor(random_state=RS, n_jobs=JOBS, verbose=-1), "needs_scaling": False}
        return models

    models = {
        "Logistic Regression": {"model": LogisticRegression(max_iter=1000), "needs_scaling": True},
        "Decision Tree": {"model": DecisionTreeClassifier(random_state=RS), "needs_scaling": False},
        "Random Forest": {"model": RandomForestClassifier(random_state=RS, n_jobs=JOBS), "needs_scaling": False},
        "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=RS), "needs_scaling": False},
        "SVC": {"model": SVC(), "needs_scaling": True, "slow": True},
        "KNN": {"model": KNeighborsClassifier(n_jobs=JOBS), "needs_scaling": True},
    }
    if XGBClassifier:
        models["XGBoost"] = {"model": XGBClassifier(random_state=RS, n_jobs=JOBS), "needs_scaling": False}
    if LGBMClassifier:
        models["LightGBM"] = {"model": LGBMClassifier(random_state=RS, n_jobs=JOBS, verbose=-1), "needs_scaling": False}
    return models