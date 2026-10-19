import os
from config import config


def log_training_run(leaderboard, fitted, settings):
    """
    Saves one training session to MLflow:
      - 1 parent run with the settings and the leaderboard table
      - 1 child run per model with its parameters and scores
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"    # allow the local folder as storage
    import mlflow                                      # imported here so the app starts fast

    mlflow.set_tracking_uri(f"file://{os.path.abspath(config.EXPERIMENT_DIR)}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # Which leaderboard columns are numbers we want to log as metrics
    metric_columns = [c for c in leaderboard.columns
                      if c.startswith(("val_", "train_")) or c in ("gap", "seconds")]

    run_name = f"{settings['dataset']} | {settings['target']} | {settings['task']}"

    with mlflow.start_run(run_name=run_name):
        # Settings of this training session (shortened, MLflow limits the length of values)
        mlflow.log_params({key: str(value)[:500] for key, value in settings.items()})
        mlflow.log_text(leaderboard.to_csv(), "leaderboard.csv")

        # One child run per model
        for _, row in leaderboard.iterrows():
            name = row["model"]
            model = fitted[name].named_steps["model"]

            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model", name)
                mlflow.log_params({key: str(value)[:500] for key, value in model.get_params().items()})
                mlflow.set_tag("fit", row["fit"])
                for column in metric_columns:
                    mlflow.log_metric(column, float(row[column]))

def log_simple_run(run_name, params, metrics):
    """Saves one run with a few parameters and metrics (used for tuning and the final test)."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow

    mlflow.set_tracking_uri(f"file://{os.path.abspath(config.EXPERIMENT_DIR)}")
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))