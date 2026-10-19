import streamlit as st
import pandas as pd
from config import config
from src.data.load_data import load_dataset, list_samples, guess_target
from src.data.data_spliter import split_data
from src.data.data_clean import find_bad_columns, drop_columns
from src.analysis.task_detection import detect_task
from src.visualization.eda_plot import (get_column_types, overview_table, class_balance,
                                        imbalance_ratio, plot_target, plot_numeric,
                                        plot_categorical, plot_missing, plot_correlation,
                                        plot_target_correlation, plot_feature_by_class,
                                        plot_feature_signal, strong_pairs,
                                        plot_category_grid, plot_numeric_grid)
from src.preprocessing.imputer import suggest_strategy
from src.preprocessing.columns import split_columns
from src.preprocessing.pipeline_builder import build_preprocessor, transform_to_frame
from src.modeling.model_list import get_models, missing_libraries
from src.modeling.trainer import train_models
from src.visualization.model_plot import plot_actual_vs_predicted, plot_confusion_matrix, plot_model_scores, plot_overfitting
from src.tracking.mlflow_logger import log_training_run, log_simple_run

from src.modeling.tuner import tune_model
from src.modeling.evaluator import score_regression, score_classification

st.title("AutoML")

# ---------- Step 1: upload ----------
file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"], key="file_upload")

options = ["(none)"] + list_samples()

# Find the default sample, ignoring capitals and spaces
default = "(none)"
for name in options:
    if config.DEFAULT_SAMPLE and name.strip().lower() == config.DEFAULT_SAMPLE.strip().lower():
        default = name

# Set the starting value only once per browser session
if "sample_select" not in st.session_state:
    st.session_state["sample_select"] = default

sample = st.selectbox("...or use a sample dataset", options, key="sample_select")
sample_name = None if sample == "(none)" else sample

if file is not None:
    st.caption("Using your uploaded file (the sample is ignored).")

# Load the data (this creates df)
df, error = load_dataset(file=file, sample_name=sample_name)
if error:
    st.info(error)      # friendly message when nothing is uploaded
    st.stop()           # stop here, nothing below runs

st.subheader("First 5 rows")
st.dataframe(df.head(5))
st.dataframe(overview_table(df))
st.write(f"Duplicate rows: {df.duplicated().sum()}")

# ---------- Step 2: target ----------
target = st.selectbox("Select the target column (suggested automatically, change if wrong)",
                      df.columns, index=guess_target(df), key="target_select")

task, reason = detect_task(df[target])
st.write(f"Detected: **{task}** ({reason})")
task = st.radio("Problem type", ["classification", "regression"],
                index=0 if task == "classification" else 1, horizontal=True, key="task_radio")

# ---------- Step 3: split and clean ----------
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    df, target, stratify=(task == "classification"))

bad = find_bad_columns(X_train)
X_train = drop_columns(X_train, bad)
X_val = drop_columns(X_val, bad)
X_test = drop_columns(X_test, bad)

if bad:
    st.warning("Dropped columns: " + ", ".join(f"{c} ({r})" for c, r in bad.items()))

# ---------- Step 4: EDA on TRAIN ----------
st.subheader("Explore the data")
st.pyplot(plot_target(y_train, task))

numeric_cols, categorical_cols = get_column_types(X_train)

# Classification only: class balance and imbalance warning
if task == "classification":
    st.write(class_balance(y_train).astype(str) + " %")
    ratio = imbalance_ratio(y_train)
    if ratio >= config.IMBALANCE_WARN_RATIO:
        st.warning(f"Imbalanced classes: the largest class is {ratio}x the smallest. "
                   "Models will be ranked by F1/balanced accuracy, not plain accuracy.")

# Feature plots
if task == "classification":
    col = st.selectbox("Compare a feature across classes", X_train.columns, key="class_feature")
    st.pyplot(plot_feature_by_class(X_train, y_train, col))
else:
    if numeric_cols:
        col = st.selectbox("Numeric column", numeric_cols, key="numeric_feature")
        st.pyplot(plot_numeric(X_train, col))
    if categorical_cols:
        col = st.selectbox("Categorical column", categorical_cols, key="categorical_feature")
        st.pyplot(plot_categorical(X_train, col))

fig = plot_missing(X_train)
if fig:
    st.pyplot(fig)
else:
    st.success("No missing values in the training data")

# ---------- Which features matter ----------
st.subheader("Which features matter?")

if task == "regression":
    fig = plot_target_correlation(X_train, y_train)
else:
    fig = plot_feature_signal(X_train, y_train)
if fig:
    st.pyplot(fig)

pairs = strong_pairs(X_train)
if not pairs.empty:
    st.write("Strongly related feature pairs (|r| >= 0.7)")
    st.dataframe(pairs)

with st.expander("Show full correlation heatmap"):
    fig = plot_correlation(X_train)
    if fig:
        st.pyplot(fig)

with st.expander("Show all features at once"):
    fig = plot_numeric_grid(X_train, y_train, task)
    if fig:
        st.pyplot(fig)
    fig = plot_category_grid(X_train, y_train, task)
    if fig:
        st.pyplot(fig)

# ---------- Step 5: preprocessing ----------
st.subheader("Preprocessing")

continuous_cols, discrete_cols, text_cols = split_columns(X_train)
st.write(f"Continuous: {len(continuous_cols)} | Few-value: {len(discrete_cols)} | Text: {len(text_cols)}")

suggested = suggest_strategy(X_train[continuous_cols])    # only continuous columns decide this
strategy = st.selectbox("Fill missing values in continuous columns with", config.IMPUTE_STRATEGIES,
                        index=config.IMPUTE_STRATEGIES.index(suggested), key="impute_strategy")
st.caption(f"Suggested: {suggested}. Few-value and text columns are always filled with the most frequent value.")

# The default depends on the task: clip for regression, keep outliers for classification
clip = st.checkbox("Clip outliers (continuous columns only)", value=(task == "regression"),
                   key=f"clip_{task}")

preprocessor = build_preprocessor(X_train, task, strategy, clip)
preprocessor.fit(X_train)                                  # learns everything from TRAIN only

X_train_ready = transform_to_frame(preprocessor, X_train)
X_val_ready = transform_to_frame(preprocessor, X_val)

st.write(f"Columns before: {X_train.shape[1]}  ->  after: {X_train_ready.shape[1]}")
st.write(f"Missing cells after: train = {int(X_train_ready.isna().sum().sum())}, "
         f"val = {int(X_val_ready.isna().sum().sum())}")
st.write("First 5 rows of the processed training data:")
st.dataframe(X_train_ready.head(5))

# ---------- Step 6: train models ----------
st.subheader("Train models")

available = list(get_models(task).keys())
mode = st.radio("Which models do you want to train?", ["All models", "Choose models"],
                horizontal=True, key="model_mode")
if mode == "All models":
    selected = available
else:
    selected = st.multiselect("Pick models", available, default=available[:3], key="model_pick")

log_to_mlflow = st.checkbox("Log this run to MLflow", value=True, key="log_mlflow")
if st.button("Train", key="train_button"):
    if not selected:
        st.warning("Pick at least one model.")
    else:
        with st.spinner("Training models..."):
            leaderboard, fitted, errors, encoder = train_models(
                X_train, y_train, X_val, y_val, task, selected, strategy, clip)
        st.session_state["results"] = {
            "leaderboard": leaderboard, "errors": errors, "fitted": fitted,
            "encoder": encoder,
            "data_key": (sample_name, getattr(file, "name", None), target, task),
        }
        for key in ("tuning", "test_result"):
            st.session_state.pop(key, None)

        if log_to_mlflow and not leaderboard.empty:
            settings = {
                "dataset": sample_name or getattr(file, "name", "upload"),
                "target": target,
                "task": task,
                "impute_strategy": strategy,
                "clip_outliers": clip,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_features": X_train.shape[1],
                "dropped_columns": list(bad),
                "models_trained": selected,
            }
            try:
                with st.spinner("Saving to MLflow..."):
                    log_training_run(leaderboard, fitted, settings)
                st.success("Run saved to MLflow.")
            except Exception as e:
                st.warning(f"MLflow logging failed, but the training results are still shown: {e}")

# Show the results (kept in session_state so they survive other clicks)
results = st.session_state.get("results")
if results and results["data_key"] == (sample_name, getattr(file, "name", None), target, task):
    st.subheader("Leaderboard")
    if missing_libraries:
        st.info("Not available on this machine: " + ", ".join(missing_libraries) + ". The other models still work.")
    rank_by = "R²" if task == "regression" else "F1-macro"
    st.caption(f"Ranked by validation {rank_by}. The test set has not been used yet.")

    if results["leaderboard"].empty:
        st.error("No model finished training.")
    else:
        st.dataframe(results["leaderboard"])
        st.success(f"Best model: {results['leaderboard'].iloc[0]['model']}")
        best = results["leaderboard"].iloc[0]
        if best["fit"] == "Overfit":
            st.warning(f"{best['model']} has the best validation score but overfits "
                    f"(gap {best['gap']}). Tuning with regularization may help.")

        tab_scores, tab_fit = st.tabs(["Model scores", "Overfitting check"])
        with tab_scores:
            st.pyplot(plot_model_scores(results["leaderboard"], task))
        with tab_fit:
            st.pyplot(plot_overfitting(results["leaderboard"], task))
            st.caption("Gap = train score minus validation score. A big gap means the model memorized "
                    "the training data. Low scores on both sides mean the model is too simple "
                    "for this data (underfitting).")


        # Plots for the top models
        st.subheader("Top models on the validation set")
        top_names = results["leaderboard"]["model"].head(config.TOP_N_PLOTS).tolist()
        columns = st.columns(len(top_names))

        for column, name in zip(columns, top_names):
            y_pred = results["fitted"][name].predict(X_val)
            with column:
                if task == "regression":
                    st.pyplot(plot_actual_vs_predicted(y_val, y_pred, name))
                else:
                    encoder = results["encoder"]
                    st.pyplot(plot_confusion_matrix(encoder.transform(y_val), y_pred,
                                                    encoder.classes_, name))

    for name, message in results["errors"].items():
        st.warning(f"{name}: {message}")

    for name, message in results["errors"].items():
        st.warning(f"{name}: {message}")

# ---------- Step 7: tune a model ----------
results = st.session_state.get("results")
data_key = (sample_name, getattr(file, "name", None), target, task)

if results and results["data_key"] == data_key and not results["leaderboard"].empty:
    leaderboard = results["leaderboard"]
    encoder = results["encoder"]
    dataset_name = sample_name or getattr(file, "name", "upload")

    # Classification: the models work with class numbers, so convert the answers the same way
    if task == "classification":
        y_train_num = encoder.transform(y_train)
        y_val_num = encoder.transform(y_val)
        y_test_num = encoder.transform(y_test)
    else:
        y_train_num, y_val_num, y_test_num = y_train, y_val, y_test

    main = "r2" if task == "regression" else "f1_macro"
    label = "R²" if task == "regression" else "F1-macro"
    score_function = score_regression if task == "regression" else score_classification

    st.subheader("Tune a model")
    model_to_tune = st.selectbox("Model to tune", leaderboard["model"].tolist(), key="tune_model")
    n_iter = st.slider("Random combinations to try", 5, 50, config.TUNE_N_ITER, key="tune_iter")

    if st.button("Tune", key="tune_button"):
        for key in ("tuning", "test_result"):
            st.session_state.pop(key, None)

        with st.spinner(f"Tuning {model_to_tune}: {n_iter} combinations x {config.CV_FOLDS} folds..."):
            search = tune_model(model_to_tune, task, X_train, y_train_num, strategy, clip, n_iter)

        if search is None:
            st.info(f"{model_to_tune} has no settings to tune. Pick another model.")
        else:
            untuned = results["fitted"][model_to_tune]
            untuned_val = score_function(y_val_num, untuned.predict(X_val))[main]
            tuned_val = score_function(y_val_num, search.best_estimator_.predict(X_val))[main]
            use_tuned = tuned_val > untuned_val

            st.session_state["tuning"] = {
                "model": model_to_tune, "untuned_val": untuned_val, "tuned_val": tuned_val,
                "cv_score": search.best_score_, "best_params": search.best_params_,
                "use_tuned": use_tuned,
                "final": search.best_estimator_ if use_tuned else untuned,
                "data_key": data_key,
            }

            if log_to_mlflow:
                try:
                    log_simple_run(f"tuning | {model_to_tune}",
                                   {"dataset": dataset_name, "model": model_to_tune, "n_iter": n_iter,
                                    **search.best_params_},
                                   {"cv_score": search.best_score_, "val_untuned": untuned_val,
                                    "val_tuned": tuned_val})
                except Exception as e:
                    st.warning(f"MLflow logging failed: {e}")

    # Show the tuning result
    tuning = st.session_state.get("tuning")
    if tuning and tuning["data_key"] == data_key:
        st.write(f"Validation {label} for {tuning['model']}:")
        st.dataframe(pd.DataFrame({"Validation": [tuning["untuned_val"], tuning["tuned_val"]]},
                                  index=["Before tuning", "After tuning"]).round(4))
        st.write(f"Cross-validation {label} of the best combination (on the training data): "
                 f"{tuning['cv_score']:.4f}")
        st.write("Best settings found:")
        st.json({key.replace("model__", ""): str(value) for key, value in tuning["best_params"].items()})

        difference = tuning["tuned_val"] - tuning["untuned_val"]
        if tuning["use_tuned"]:
            st.success(f"Tuning improved the validation score by {difference:.4f}. The tuned model will be used. "
                       "Differences below about 0.01 can be noise on a small validation set.")
        else:
            st.info("Tuning did not beat the untuned model on validation, so the untuned model will be used.")

    # ---------- Step 8: final check on the test set ----------
    st.subheader("Final check on the test set")
    if tuning and tuning["data_key"] == data_key:
        final_pipeline = tuning["final"]
        final_name = tuning["model"] + (" (tuned)" if tuning["use_tuned"] else " (untuned)")
    else:
        final_name = leaderboard.iloc[0]["model"]
        final_pipeline = results["fitted"][final_name]

    st.write(f"Final model: **{final_name}**")
    st.caption("The test set is only honest if you look at it once. If you check the score and then change "
               "settings to improve it, the test set is no longer independent.")

    if st.button("Evaluate on the test set", key="test_button"):
        pred_val = final_pipeline.predict(X_val)
        pred_test = final_pipeline.predict(X_test)
        val_scores = score_function(y_val_num, pred_val)
        test_scores = score_function(y_test_num, pred_test)
        st.session_state["test_result"] = {
            "final_name": final_name, "data_key": data_key, "pred_test": pred_test,
            "val_scores": val_scores, "test_scores": test_scores,
        }

        if log_to_mlflow:
            try:
                log_simple_run(f"final test | {final_name}", {"dataset": dataset_name, "model": final_name},
                               {**{f"val_{k}": v for k, v in val_scores.items()},
                                **{f"test_{k}": v for k, v in test_scores.items()}})
            except Exception as e:
                st.warning(f"MLflow logging failed: {e}")

    test_result = st.session_state.get("test_result")
    if test_result and test_result["final_name"] == final_name and test_result["data_key"] == data_key:
        st.dataframe(pd.DataFrame({"Validation": test_result["val_scores"],
                                   "Test": test_result["test_scores"]}).round(4))

        drop = test_result["val_scores"][main] - test_result["test_scores"][main]
        if drop > config.OVERFIT_SMALL_GAP:
            st.warning(f"The test {label} is {drop:.3f} lower than on validation. The model may not "
                       "generalize as well as the validation score suggested.")
        else:
            st.success("The test score is close to the validation score, so the model generalizes well.")

        if task == "regression":
            st.pyplot(plot_actual_vs_predicted(y_test_num, test_result["pred_test"], final_name))
        else:
            st.pyplot(plot_confusion_matrix(y_test_num, test_result["pred_test"],
                                            encoder.classes_, final_name))