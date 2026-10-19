import numpy as np
import pandas as pd


def add_missing(name, columns, fraction=0.1):
    """Blanks out a random 10% of the cells in the chosen columns and saves a copy."""
    df = pd.read_csv(f"dataset/Raw/{name}.csv")
    for col in columns:
        rows = df.sample(frac=fraction, random_state=42).index
        df.loc[rows, col] = np.nan
    df.to_csv(f"dataset/{name}_missing.csv", index=False)
    print("saved", f"dataset/{name}_missing.csv")


add_missing("titanic", ["Age", "Fare", "Embarked"])
# add_missing("insurance", ["bmi", "sex", "region"])   # sex and region are text columns