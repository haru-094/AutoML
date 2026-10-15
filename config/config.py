from pathlib import Path
import os

# Project root folder (the AutoML folder), so paths work from anywhere
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Reproducibility
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    CV_FOLDS = 5
    N_JOBS = -1

    # Folders
    MODEL_DIR = BASE_DIR / "models"
    EXPERIMENT_DIR = BASE_DIR / "experiments"
    SAMPLE_DIR = BASE_DIR / "dataset"
    DEFAULT_SAMPLE = "titanic"   

    # Data loading rules
    MIN_ROWS = 50

    # Cleaning rules
    ID_UNIQUE_RATIO = 0.95        # 95%+ unique values -> looks like an ID
    MAX_CATEGORY_UNIQUE = 50      # text column with more unique values -> too many categories
    MAX_MISSING_RATIO = 0.5       # more than 50% missing -> drop the column

    # Task detection rule
    MAX_CLASSES = 20              # whole-number target with 20 or fewer values -> classification

    # EDA plot settings
    TOP_N_CATEGORIES = 10         # show only the 10 most common categories
    MAX_DISCRETE = 10             # numeric column with <= 10 unique values -> bar chart
    TOP_N_FEATURES = 15           # features shown in the target-correlation chart
    MAX_ANNOTATED_COLS = 25       # write numbers inside the heatmap up to this many columns
    IMBALANCE_WARN_RATIO = 3

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)


config = Config()