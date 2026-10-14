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

    # Folders (your folder is called "dataset", not data/sample)
    MODEL_DIR = BASE_DIR / "models"
    EXPERIMENT_DIR = BASE_DIR / "experiments"
    SAMPLE_DIR = BASE_DIR / "dataset/Raw"

    # Data loading rules
    MIN_ROWS = 50
    MAX_FILE_MB = 100

    # Cleaning rules
    ID_UNIQUE_RATIO = 0.95
    MAX_CATEGORY_UNIQUE = 50
    MAX_MISSING_RATIO = 0.5

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)


config = Config()