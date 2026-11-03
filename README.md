# AutoML

AutoML is a Streamlit application for building and evaluating machine-learning
pipelines for tabular data. It helps you inspect a dataset, prepare features,
compare multiple models, tune the best candidate, and download the resulting
model for later use.

## Features

- Upload CSV or XLSX datasets, or use the included sample data.
- Inspect data types, missing values, duplicates, and column summaries.
- Explore distributions, correlations, and target relationships.
- Automatically identify numeric and categorical features.
- Handle missing values, categorical encoding, scaling, and numeric outliers.
- Compare regression and classification models on a validation set.
- Tune a selected model with cross-validation.
- Evaluate the final model on a held-out test set.
- Download the complete preprocessing and model pipeline as `automl_model.pkl`.
- Optionally log training runs to MLflow.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- A CSV or XLSX dataset for custom experiments

## Quick start with uv

From the project directory, install the dependencies:

```bash
uv sync
```

Start the application:

```bash
uv run streamlit run app.py
```

Streamlit opens the application in your browser. If it does not open
automatically, visit the local URL printed in the terminal, usually
<http://localhost:8501>.

To stop the application, press `Ctrl+C` in the terminal.

## Alternative installation with pip

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Using the application

Follow the tabs from left to right:

1. **Upload data** — Upload a CSV/XLSX file or select a sample dataset. Review
   the first rows, column information, duplicates, and missing values.
2. **EDA** — Select the target column and problem type. Inspect target
   distributions, missing-value charts, feature distributions, associations,
   and correlations.
3. **Preprocessing** — Review detected ID-like columns and choose the numeric
   imputation strategy. Numeric values are imputed, clipped using training-set
   IQR bounds, and scaled when appropriate. Categorical values are imputed and
   one-hot encoded.
4. **Split data** — Create reproducible training, validation, and test splits.
   Classification splits preserve class proportions when possible.
5. **Train & leaderboard** — Select recommended models or choose models
   manually. Compare their validation scores and overfitting metrics.
6. **Tune model** — Optionally tune a trained model using cross-validation.
   The tuned model is retained only when it improves validation performance.
7. **Save model** — Evaluate the selected model once on the held-out test set
   and download the complete fitted pipeline as `automl_model.pkl`.

Preprocessing is fitted inside each model pipeline using training data only.
The test set is kept separate until final evaluation to reduce data leakage.

## Viewing MLflow experiments

MLflow logging is optional and disabled by default. To save a run, select
**Save this run to MLflow (optional)** in the application before training,
tuning, or evaluating a model.

Runs are stored locally in the repository's `experiments/` directory. They are
not sent to a remote MLflow server. The terminal only shows the Streamlit
application; experiment details are best viewed in the MLflow web interface.

Keep the Streamlit application running and open a second terminal in the
project directory:

```bash
MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui \
  --backend-store-uri ./experiments \
  --host 127.0.0.1 \
  --port 5000
```

Then open <http://127.0.0.1:5000> in your browser. Select the `automl`
experiment to view run names, parameters, metrics, nested model runs, and the
training leaderboard artifact.

Use `127.0.0.1:5000` rather than `localhost:5000`. On some macOS systems,
another local service can respond on the `localhost` address and show a blank
or forbidden page even when MLflow is running correctly.

To stop the MLflow interface, press `Ctrl+C` in the terminal running it. The
experiment files remain in `experiments/` and can be reused the next time you
start the MLflow UI.

## Included sample data

The repository includes Titanic datasets for trying the application without
uploading a file:

- `dataset/Raw/titanic.csv`
- `dataset/titanic_missing.csv`

## Project structure

```text
AutoML/
├── app.py                         # Streamlit application
├── config/                        # Application configuration
├── dataset/                       # Included sample datasets
├── src/
│   ├── analysis/                  # Task detection and analysis helpers
│   ├── data/                      # Loading, cleaning, and splitting
│   ├── modeling/                  # Models, training, scoring, and tuning
│   ├── preprocessing/             # Imputation, encoding, scaling, and clipping
│   ├── tracking/                  # Optional MLflow integration
│   └── visualization/             # EDA and model plots
├── tests/                         # Automated tests
├── pyproject.toml                 # Project metadata and dependencies
└── requirements.txt               # pip dependency list
```

## Running tests

With uv:

```bash
uv run python -m unittest tests/test_core_pipeline.py
```

With an activated pip environment:

```bash
python -m unittest tests/test_core_pipeline.py
```

## Troubleshooting

### `uv: command not found`

Install uv using the instructions at <https://docs.astral.sh/uv/getting-started/installation/>
and reopen the terminal.

### `Expected a Python module at: src/automl/__init__.py`

Use the project configuration included in this repository and run:

```bash
uv sync
uv run streamlit run app.py
```

The `pyproject.toml` file configures uv to package the existing `src` module
layout.

### Streamlit does not open a browser

Copy the `Local URL` shown in the terminal and open it manually. To choose a
different port:

```bash
uv run streamlit run app.py --server.port 8502
```
