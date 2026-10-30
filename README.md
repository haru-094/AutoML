# AutoML

A Streamlit app for exploring tabular data, preparing features, comparing machine-learning models, and optionally tuning and saving a model.

## Run the app

Install the project dependencies and start Streamlit:

```bash
uv sync
uv run streamlit run app.py
```

Alternatively, install the dependencies with pip:

```bash
pip install -r requirements.txt
streamlit run app.py
```

You can upload a CSV or XLSX file, or explicitly choose a sample dataset. The app starts with no sample selected.

## Workflow

The app is organized into seven tabs:

1. **Upload data** — Upload a file or choose a sample. Review the first five rows, column summary, duplicate rows, and missing values.
2. **EDA** — Choose the target and problem type. View the target distribution, missing-value chart, numeric and categorical feature distributions, target associations, and numeric feature correlations.
3. **Preprocessing** — Review automatically detected ID-like columns and choose mean or median imputation for numeric missing values. Categorical features are imputed with their most common value and one-hot encoded. There is no outlier clipping.
4. **Split data** — Review the randomized train, validation, and test row counts. Splits are reproducible; classification splits preserve class proportions when possible.
5. **Train & leaderboard** — Compare recommended models or choose models. Results are ranked by validation score and include model score and overfitting views.
6. **Tune model** — Optionally search settings for a trained model using cross-validation on the training data. The tuned model is selected only if its validation score improves.
7. **Save model** — Download the selected pipeline as `automl_model.pkl`. The tab also lets you evaluate the selected model once on the held-out test set.

Preprocessing is fitted inside each model pipeline using training data only. The test set is reserved for the final evaluation after model selection. MLflow logging is optional and off by default.

## Project layout

- `app.py` — Streamlit interface and workflow.
- `src/data/` — Dataset loading, splitting, and data checks.
- `src/preprocessing/` — Feature typing, imputation, encoding, and pipeline construction.
- `src/modeling/` — Model selection, training, scoring, and tuning.
- `src/visualization/` — EDA and model result charts.
- `src/tracking/` — Optional MLflow logging.
- `tests/` — Core pipeline tests.
