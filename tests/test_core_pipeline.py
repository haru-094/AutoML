import io
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.analysis.task_detection import detect_task
from src.data.data_clean import find_bad_columns
from src.data.data_spliter import split_data
from src.data.load_data import load_dataset
from src.modeling.evaluator import score_classification, score_regression
from src.preprocessing.columns import split_columns
from src.preprocessing.pipeline_builder import build_preprocessor
from src.modeling.trainer import train_models
from src.modeling.tuner import tune_model


class CorePipelineTests(unittest.TestCase):
  def test_detect_task_nullable_string_and_boolean_and_empty(self):
    self.assertEqual(detect_task(pd.Series(["a", "b"], dtype="string"))[0], "classification")
    self.assertEqual(detect_task(pd.Series([True, False, pd.NA], dtype="boolean"))[0], "classification")
    with self.assertRaisesRegex(ValueError, "no non-missing"):
        detect_task(pd.Series([pd.NA, pd.NA], dtype="Int64"))


  def test_split_drops_missing_target_and_reports_it(self):
    df = pd.DataFrame({"x": range(20), "target": [0, 1] * 9 + [np.nan, np.nan]})
    with self.assertWarnsRegex(UserWarning, "Dropped 2"):
        parts = split_data(df, "target", stratify=True)
    self.assertEqual(sum(map(len, (parts[0], parts[1], parts[2]))), 18)


  def test_split_rejects_empty_or_rare_targets_clearly(self):
    with self.assertRaisesRegex(ValueError, "no non-missing"):
        split_data(pd.DataFrame({"x": [1, 2, 3], "y": [np.nan] * 3}), "y")
    df = pd.DataFrame({"x": range(20), "y": [0] * 18 + [1, 2]})
    with self.assertRaisesRegex(ValueError, "at least 3 rows"):
        split_data(df, "y", stratify=True)


  def test_duplicate_names_rejected_and_uppercase_csv_supported(self):
    csv_data = ("x,y\n" + "\n".join(f"{i},{i + 1}" for i in range(50))).encode()
    upload = io.BytesIO(csv_data)
    upload.name = "DATA.CSV"
    loaded, error = load_dataset(file=upload)
    self.assertIsNone(error)
    self.assertEqual(list(loaded.columns), ["x", "y"])

    duplicate_data = ("x, x ,y\n" + "\n".join(f"{i},{i},{i + 1}" for i in range(50))).encode()
    duplicate = io.BytesIO(duplicate_data)
    duplicate.name = "data.csv"
    loaded, error = load_dataset(file=duplicate)
    self.assertIsNone(loaded)
    self.assertIn("unique", error)


  def test_preprocessor_handles_missing_and_unseen_categories_train_only(self):
    train = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "kind": ["a", "b", None], "code": [1, 2, 1]})
    holdout = pd.DataFrame({"amount": [4.0], "kind": ["never-seen"], "code": [8]})
    prep = build_preprocessor(train, "classification", categorical_overrides=["code"])
    transformed_train = prep.fit_transform(train)
    transformed_holdout = prep.transform(holdout)
    self.assertEqual(transformed_train.shape[1], transformed_holdout.shape[1])
    self.assertFalse(np.isnan(transformed_train).any())
    self.assertFalse(np.isnan(transformed_holdout).any())
    self.assertNotIn("never-seen", " ".join(map(str, prep.named_transformers_["text"].named_steps["onehot"].categories_[0])))
    self.assertIn("code", split_columns(train, ["code"])[1])


  def test_cleaning_can_flag_float_ids_when_requested(self):
    frame = pd.DataFrame({"float_id": np.arange(100, dtype=float), "value": [1.0] * 100})
    self.assertNotIn("float_id", find_bad_columns(frame))
    self.assertIn("float_id", find_bad_columns(frame, drop_float_ids=True))


  def test_metrics_and_simple_classification_regression_models(self):
    self.assertEqual(score_classification([0, 1], [0, 1])["accuracy"], 1)
    self.assertEqual(score_regression([1.0, 2.0], [1.0, 2.0])["r2"], 1)
    self.assertTrue(LogisticRegression and LinearRegression)

  def test_training_pipeline_for_classification_and_regression(self):
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x": rng.normal(size=80), "category": ["a", "b"] * 40})
    frame.loc[0:4, "x"] = np.nan
    y_class = pd.Series(["yes", "no"] * 40)
    result = train_models(frame.iloc[:60], y_class.iloc[:60], frame.iloc[60:], y_class.iloc[60:],
                          "classification", ["Logistic Regression"], "median", False)
    self.assertFalse(result[0].empty)
    self.assertIn("Logistic Regression", result[1])

    y_reg = pd.Series(frame["x"].fillna(0) * 2 + rng.normal(scale=0.1, size=80))
    result = train_models(frame.iloc[:60], y_reg.iloc[:60], frame.iloc[60:], y_reg.iloc[60:],
                          "regression", ["Linear Regression"], "median", False)
    self.assertFalse(result[0].empty)
    self.assertIn("Linear Regression", result[1])

  def test_tuning_uses_feasible_folds_for_rare_class(self):
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = np.array([0] * 9 + [1] * 3)
    search = tune_model("Logistic Regression", "classification", X, y, "median", False, n_iter=1)
    self.assertEqual(search.cv, 3)
