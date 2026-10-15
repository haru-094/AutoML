import streamlit as st

from config import config
from src.data.load_data import load_dataset, list_samples, guess_target
from src.data.data_spliter import split_data
from src.data.data_clean import find_bad_columns, drop_columns
from src.analysis.task_detection import detect_task
from src.visualization.eda_plot import (get_column_types, overview_table, class_balance,
                                        imbalance_ratio, plot_target, plot_numeric,
                                        plot_categorical, plot_missing, plot_correlation,
                                        plot_target_correlation, plot_feature_by_class,
                                        plot_feature_signal, strong_pairs)

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