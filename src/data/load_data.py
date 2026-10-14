import pandas as pd
from config import config


def list_samples():
    """Names of the sample datasets in data/sample (without .csv)."""
    return [file.stem for file in config.SAMPLE_DIR.glob("*.csv")]


def load_dataset(file=None, sample_name=None):
    """
    Returns (df, error).
    If it worked: (DataFrame, None)
    If it failed: (None, "message for the user")
    """
    # Case 1: nothing provided
    if file is None and sample_name is None:
        return None, "No dataset provided. Upload a file or pick a sample."

    # Case 2: try to read the data
    try:
        if file is not None:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            else:
                return None, "Unsupported file type. Use CSV or XLSX."
        else:
            df = pd.read_csv(config.SAMPLE_DIR / f"{sample_name}.csv")
    except Exception as e:
        return None, f"Could not read the file: {e}"

    # Case 3: check the data is usable
    df.columns = df.columns.str.strip()   # remove accidental spaces in column names

    if df.empty:
        return None, "The file is empty."
    if df.shape[1] < 2:
        return None, "Need at least 2 columns (features + target)."
    if len(df) < config.MIN_ROWS:
        return None, f"Need at least {config.MIN_ROWS} rows, got {len(df)}."

    return df, None