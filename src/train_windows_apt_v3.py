"""
======================================================================
 SmartSOC Windows-APT - Model B v3.2
 Hybrid Structured + Behavioral Text Multi-Label Classifier
======================================================================

Purpose
-------
Train a multi-label MITRE ATT&CK technique classifier using:

1. Structured Windows/Sysmon/Wazuh features
2. Behavioral text represented using TF-IDF
3. One-vs-Rest Logistic Regression

Important
---------
The following fields are explicitly categorical:

    process_name
    parent_process_name
    process_extension
    parent_extension

This is intentional because pandas may load these columns using the
new "str" dtype rather than "object".

The model is evaluated using a temporal split to reduce temporal leakage.

Output
------
Model:
    models/windows_apt_v3/multilabel_logistic_v3.joblib

Structured encoder:
    models/windows_apt_v3/structured_encoder_v3.joblib

TF-IDF:
    models/windows_apt_v3/tfidf_vectorizer.joblib

Label encoder:
    models/windows_apt_v3/multilabel_encoder.joblib

Feature schema:
    models/windows_apt_v3/feature_schema_v3.json

Report:
    results/windows_apt_v3/classification_report.txt

Predictions:
    results/windows_apt_v3/predictions.csv
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import (
    csr_matrix,
    hstack,
)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import (
    MultiLabelBinarizer,
    OneHotEncoder,
)

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# ======================================================================
# PATHS
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

STRUCTURED_PATH = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "structured_features.csv"
)

TEXT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "behavior_text.csv"
)

LABEL_PATH = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "labels.csv"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v3"
    / "metadata.csv"
)


MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v3"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "windows_apt_v3"
)


MODEL_PATH = MODEL_DIR / "multilabel_logistic_v3.joblib"

STRUCTURED_ENCODER_PATH = (
    MODEL_DIR
    / "structured_encoder_v3.joblib"
)

TFIDF_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer.joblib"
)

LABEL_ENCODER_PATH = (
    MODEL_DIR
    / "multilabel_encoder.joblib"
)

SCHEMA_PATH = (
    MODEL_DIR
    / "feature_schema_v3.json"
)

REPORT_PATH = (
    RESULT_DIR
    / "classification_report.txt"
)

PREDICTION_PATH = (
    RESULT_DIR
    / "predictions.csv"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

TEMPORAL_CUTOFF = pd.Timestamp(
    "2024-12-11"
)

# Explicit categorical columns.
#
# DO NOT determine these using:
#
#     df[col].dtype == "object"
#
# because pandas may report the dtype as "str".

CATEGORICAL_COLUMNS = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]


# TF-IDF configuration

TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 20000,
    "sublinear_tf": True,
}


# Logistic Regression configuration

LOGISTIC_CONFIG = {
    "C": 1.0,
    "max_iter": 2000,
    "solver": "liblinear",
    "random_state": RANDOM_STATE,
}


# ======================================================================
# HEADER
# ======================================================================

print()
print("=" * 70)
print(" SmartSOC Windows-APT - Model B v3.2")
print(" Corrected Hybrid Structured + TF-IDF Multi-Label Training")
print("=" * 70)
print()


# ======================================================================
# CHECK FILES
# ======================================================================

print("Checking required files...")

required_files = [
    STRUCTURED_PATH,
    TEXT_PATH,
    LABEL_PATH,
    METADATA_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    print()
    print("ERROR: Required files are missing:")

    for path in missing_files:
        print(" -", path)

    raise SystemExit(1)

print("All required files found.")
print()


# ======================================================================
# LOAD DATASETS
# ======================================================================

print("Loading datasets...")

X_struct = pd.read_csv(
    STRUCTURED_PATH
)

X_text = pd.read_csv(
    TEXT_PATH
)

Y_raw = pd.read_csv(
    LABEL_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)


print("Structured rows:", len(X_struct))
print("Text rows      :", len(X_text))
print("Label rows     :", len(Y_raw))
print("Metadata rows  :", len(metadata))
print()


# ======================================================================
# BASIC VALIDATION
# ======================================================================

if not (
    len(X_struct)
    == len(X_text)
    == len(Y_raw)
    == len(metadata)
):
    raise ValueError(
        "Dataset row counts do not match."
    )


# ======================================================================
# LABEL PROCESSING
# ======================================================================

print("=" * 70)
print(" LABEL PROCESSING")
print("=" * 70)
print()


def parse_techniques(value):
    """
    Convert:

        T1059.003

    into:

        ["T1059.003"]

    and:

        T1059.003,T1087

    into:

        ["T1059.003", "T1087"]
    """

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    return parts


label_lists = (
    Y_raw["techniques"]
    .apply(parse_techniques)
)


# Remove rows with no MITRE labels.

labeled_mask = (
    label_lists.str.len() > 0
)

print(
    "Total events:",
    len(label_lists)
)

print(
    "Labeled events:",
    int(labeled_mask.sum())
)

print(
    "Unlabeled events:",
    int((~labeled_mask).sum())
)

if (~labeled_mask).sum() > 0:

    X_struct = X_struct.loc[
        labeled_mask
    ].reset_index(drop=True)

    X_text = X_text.loc[
        labeled_mask
    ].reset_index(drop=True)

    metadata = metadata.loc[
        labeled_mask
    ].reset_index(drop=True)

    label_lists = label_lists.loc[
        labeled_mask
    ].reset_index(drop=True)

    Y_raw = Y_raw.loc[
        labeled_mask
    ].reset_index(drop=True)


print()


# ======================================================================
# LABEL ENCODER
# ======================================================================

mlb = MultiLabelBinarizer()

Y = mlb.fit_transform(
    label_lists
)

techniques = list(
    mlb.classes_
)

print(
    "MITRE techniques:",
    len(techniques)
)

print(techniques)
print()


# ======================================================================
# TEMPORAL SPLIT
# ======================================================================

print("=" * 70)
print(" TEMPORAL SPLIT")
print("=" * 70)
print()


timestamps = pd.to_datetime(
    metadata["timestamp"],
    format="%b %d, %Y @ %H:%M:%S.%f",
    errors="coerce",
)


if timestamps.isna().any():

    invalid = int(
        timestamps.isna().sum()
    )

    raise ValueError(
        f"Found {invalid} invalid timestamps."
    )


print(
    "Earliest:",
    timestamps.min()
)

print(
    "Latest:",
    timestamps.max()
)

print(
    "Temporal cutoff:",
    TEMPORAL_CUTOFF
)


train_mask = (
    timestamps <= TEMPORAL_CUTOFF
)

test_mask = (
    timestamps > TEMPORAL_CUTOFF
)


train_idx = np.where(
    train_mask.values
)[0]

test_idx = np.where(
    test_mask.values
)[0]


print(
    "Training events:",
    len(train_idx)
)

print(
    "Testing events :",
    len(test_idx)
)


if len(train_idx) == 0:
    raise ValueError(
        "Training set is empty."
    )

if len(test_idx) == 0:
    raise ValueError(
        "Testing set is empty."
    )


# ======================================================================
# TEST-ONLY TECHNIQUE CHECK
# ======================================================================

train_techniques = set()

for row in label_lists.iloc[train_idx]:

    train_techniques.update(
        row
    )


test_techniques = set()

for row in label_lists.iloc[test_idx]:

    test_techniques.update(
        row
    )


test_only = sorted(
    test_techniques
    - train_techniques
)


print()

if test_only:

    print(
        "WARNING: Techniques present only in test:"
    )

    for technique in test_only:
        print(
            " -",
            technique
        )

    print()

    print(
        "These techniques cannot be learned "
        "from the training data."
    )

else:

    print(
        "No test-only techniques."
    )

print()


# ======================================================================
# STRUCTURED FEATURE PREPARATION
# ======================================================================

print("=" * 70)
print(" STRUCTURED FEATURE PREPARATION")
print("=" * 70)
print()


print(
    "Original structured feature count:",
    len(X_struct.columns)
)


# Verify categorical columns exist.

missing_categorical = [
    col
    for col in CATEGORICAL_COLUMNS
    if col not in X_struct.columns
]

if missing_categorical:

    raise ValueError(
        "Expected categorical columns are missing: "
        + str(missing_categorical)
    )


# Explicitly define categorical columns.

categorical_columns = [
    col
    for col in CATEGORICAL_COLUMNS
    if col in X_struct.columns
]


numerical_columns = [
    col
    for col in X_struct.columns
    if col not in categorical_columns
]


print(
    "Categorical:"
)

for col in categorical_columns:
    print(
        "  -",
        col
    )


print()

print(
    "Numerical:"
)

for col in numerical_columns:
    print(
        "  -",
        col
    )


print()

print(
    "Categorical feature count:",
    len(categorical_columns)
)

print(
    "Numerical feature count:",
    len(numerical_columns)
)

print()


# ======================================================================
# CLEAN CATEGORICAL DATA
# ======================================================================

X_struct = X_struct.copy()

for col in categorical_columns:

    X_struct[col] = (
        X_struct[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ======================================================================
# CATEGORICAL ENCODING
# ======================================================================

print(
    "Fitting categorical encoder on TRAINING DATA ONLY..."
)


# sklearn compatibility

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True,
        dtype=np.float32,
    )


X_cat_train = encoder.fit_transform(
    X_struct.loc[
        train_idx,
        categorical_columns,
    ]
)


X_cat_test = encoder.transform(
    X_struct.loc[
        test_idx,
        categorical_columns,
    ]
)


X_cat_train = csr_matrix(
    X_cat_train,
    dtype=np.float32,
    copy=True,
)

X_cat_test = csr_matrix(
    X_cat_test,
    dtype=np.float32,
    copy=True,
)


print(
    "Categorical train:",
    X_cat_train.shape
)

print(
    "Categorical test :",
    X_cat_test.shape
)

print()


# ======================================================================
# NUMERICAL FEATURES
# ======================================================================

print(
    "Preparing numerical features..."
)


X_num_train_df = (
    X_struct.loc[
        train_idx,
        numerical_columns,
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .fillna(0)
)


X_num_test_df = (
    X_struct.loc[
        test_idx,
        numerical_columns,
    ]
    .apply(
        pd.to_numeric,
        errors="coerce",
    )
    .fillna(0)
)


# Convert to writable numpy arrays.

X_num_train_array = np.array(
    X_num_train_df.values,
    dtype=np.float32,
    copy=True,
)

X_num_test_array = np.array(
    X_num_test_df.values,
    dtype=np.float32,
    copy=True,
)


# Replace non-finite values.

X_num_train_array[
    ~np.isfinite(X_num_train_array)
] = 0

X_num_test_array[
    ~np.isfinite(X_num_test_array)
] = 0


X_num_train = csr_matrix(
    X_num_train_array,
    dtype=np.float32,
)

X_num_test = csr_matrix(
    X_num_test_array,
    dtype=np.float32,
)


print(
    "Numerical train:",
    X_num_train.shape
)

print(
    "Numerical test :",
    X_num_test.shape
)

print()


# ======================================================================
# COMBINE STRUCTURED FEATURES
# ======================================================================

X_struct_train = hstack(
    [
        X_num_train,
        X_cat_train,
    ],
    format="csr",
    dtype=np.float32,
)


X_struct_test = hstack(
    [
        X_num_test,
        X_cat_test,
    ],
    format="csr",
    dtype=np.float32,
)


# Explicit copy prevents sparse read-only issues.

X_struct_train = csr_matrix(
    X_struct_train,
    dtype=np.float32,
    copy=True,
)

X_struct_test = csr_matrix(
    X_struct_test,
    dtype=np.float32,
    copy=True,
)


X_struct_train.sum_duplicates()
X_struct_train.sort_indices()

X_struct_test.sum_duplicates()
X_struct_test.sort_indices()


print(
    "Structured matrix:"
)

print(
    "Train:",
    X_struct_train.shape
)

print(
    "Test :",
    X_struct_test.shape
)

print()


# ======================================================================
# BEHAVIOR TEXT
# ======================================================================

print("=" * 70)
print(" BEHAVIOR TEXT / TF-IDF")
print("=" * 70)
print()


if "behavior_text" not in X_text.columns:

    raise ValueError(
        "behavior_text column not found in behavior_text.csv"
    )


train_text = (
    X_text.loc[
        train_idx,
        "behavior_text",
    ]
    .fillna("")
    .astype(str)
)


test_text = (
    X_text.loc[
        test_idx,
        "behavior_text",
    ]
    .fillna("")
    .astype(str)
)


print(
    "Training text rows:",
    len(train_text)
)

print(
    "Testing text rows :",
    len(test_text)
)

print()


# ======================================================================
# TF-IDF
# ======================================================================

print(
    "Fitting TF-IDF on TRAINING DATA ONLY..."
)


vectorizer = TfidfVectorizer(
    lowercase=TFIDF_CONFIG["lowercase"],
    ngram_range=TFIDF_CONFIG["ngram_range"],
    min_df=TFIDF_CONFIG["min_df"],
    max_features=TFIDF_CONFIG["max_features"],
    sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
    dtype=np.float32,
)


X_text_train = vectorizer.fit_transform(
    train_text
)


X_text_test = vectorizer.transform(
    test_text
)


X_text_train = csr_matrix(
    X_text_train,
    dtype=np.float32,
    copy=True,
)

X_text_test = csr_matrix(
    X_text_test,
    dtype=np.float32,
    copy=True,
)


X_text_train.sum_duplicates()
X_text_train.sort_indices()

X_text_test.sum_duplicates()
X_text_test.sort_indices()


print(
    "TF-IDF vocabulary:",
    len(vectorizer.vocabulary_)
)

print(
    "Text train:",
    X_text_train.shape
)

print(
    "Text test :",
    X_text_test.shape
)

print()


# ======================================================================
# FINAL HYBRID MATRIX
# ======================================================================

print("=" * 70)
print(" HYBRID FEATURE MATRIX")
print("=" * 70)
print()


X_train = hstack(
    [
        X_struct_train,
        X_text_train,
    ],
    format="csr",
    dtype=np.float32,
)


X_test = hstack(
    [
        X_struct_test,
        X_text_test,
    ],
    format="csr",
    dtype=np.float32,
)


X_train = csr_matrix(
    X_train,
    dtype=np.float32,
    copy=True,
)

X_test = csr_matrix(
    X_test,
    dtype=np.float32,
    copy=True,
)


X_train.sum_duplicates()
X_train.sort_indices()

X_test.sum_duplicates()
X_test.sort_indices()


print(
    "Final train matrix:",
    X_train.shape
)

print(
    "Final test matrix :",
    X_test.shape
)

print(
    "Total model features:",
    X_train.shape[1]
)

print()


# ======================================================================
# LABEL VALIDATION
# ======================================================================

print("=" * 70)
print(" LABEL VALIDATION")
print("=" * 70)
print()


Y_train = Y[train_idx]

Y_test = Y[test_idx]


print(
    "Y train:",
    Y_train.shape
)

print(
    "Y test :",
    Y_test.shape
)


missing_from_train = []

for i, technique in enumerate(
    techniques
):

    count = int(
        Y_train[:, i].sum()
    )

    if count == 0:

        missing_from_train.append(
            technique
        )


if missing_from_train:

    print()

    print(
        "WARNING: These techniques have zero "
        "training examples:"
    )

    for technique in missing_from_train:

        print(
            " -",
            technique
        )

    print()

else:

    print(
        "Every technique has training examples."
    )


print()


# ======================================================================
# TRAIN MODEL
# ======================================================================

print("=" * 70)
print(" TRAINING ONE-VS-REST LOGISTIC REGRESSION")
print("=" * 70)
print()


print(
    "This is the corrected hybrid baseline."
)

print(
    "Structured categorical features are now "
    "properly one-hot encoded."
)

print(
    "Behavioral text is represented using TF-IDF."
)

print(
    "Using n_jobs=1 for stable Windows execution."
)

print()


base_classifier = LogisticRegression(
    C=LOGISTIC_CONFIG["C"],
    max_iter=LOGISTIC_CONFIG["max_iter"],
    solver=LOGISTIC_CONFIG["solver"],
    random_state=LOGISTIC_CONFIG["random_state"],
)


model = OneVsRestClassifier(
    base_classifier,
    n_jobs=1,
)


model.fit(
    X_train,
    Y_train,
)


print()
print(
    "Training complete."
)
print()


# ======================================================================
# PREDICTIONS
# ======================================================================

print("=" * 70)
print(" GENERATING PREDICTIONS")
print("=" * 70)
print()


probabilities = model.predict_proba(
    X_test
)


probabilities = np.asarray(
    probabilities,
    dtype=np.float32,
)


# Standard baseline threshold.

THRESHOLD = 0.50


Y_pred = (
    probabilities >= THRESHOLD
).astype(np.int8)


# ----------------------------------------------------------------------
# Ensure every event receives at least one technique.
# ----------------------------------------------------------------------

empty_predictions = (
    Y_pred.sum(axis=1) == 0
)


empty_count = int(
    empty_predictions.sum()
)


if empty_count > 0:

    best_class = np.argmax(
        probabilities[empty_predictions],
        axis=1,
    )

    empty_rows = np.where(
        empty_predictions
    )[0]

    Y_pred[
        empty_rows,
        best_class,
    ] = 1


print(
    "Test events:",
    len(Y_test)
)

print(
    "Events with no prediction before fallback:",
    empty_count
)

print()


# ======================================================================
# METRICS
# ======================================================================

micro_precision = precision_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0,
)

micro_recall = recall_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0,
)

micro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="micro",
    zero_division=0,
)


macro_precision = precision_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    Y_test,
    Y_pred,
    average="macro",
    zero_division=0,
)


weighted_precision = precision_score(
    Y_test,
    Y_pred,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    Y_test,
    Y_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    Y_test,
    Y_pred,
    average="weighted",
    zero_division=0,
)


# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================

report = classification_report(
    Y_test,
    Y_pred,
    target_names=techniques,
    zero_division=0,
)


print("=" * 70)
print(" FINAL TEST RESULTS")
print("=" * 70)
print()


print(
    f"Micro Precision : {micro_precision:.4f}"
)

print(
    f"Micro Recall    : {micro_recall:.4f}"
)

print(
    f"Micro F1        : {micro_f1:.4f}"
)

print()

print(
    f"Macro Precision : {macro_precision:.4f}"
)

print(
    f"Macro Recall    : {macro_recall:.4f}"
)

print(
    f"Macro F1        : {macro_f1:.4f}"
)

print()

print(
    f"Weighted Precision : {weighted_precision:.4f}"
)

print(
    f"Weighted Recall    : {weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)

print()

print(
    "Classification Report:"
)

print()

print(
    report
)


# ======================================================================
# CONFIDENCE ANALYSIS
# ======================================================================

max_confidence = probabilities.max(
    axis=1
)

mean_confidence = float(
    max_confidence.mean()
)

median_confidence = float(
    np.median(max_confidence)
)

minimum_confidence = float(
    max_confidence.min()
)

maximum_confidence = float(
    max_confidence.max()
)


print(
    "=" * 70
)

print(
    " CONFIDENCE ANALYSIS"
)

print(
    "=" * 70
)

print()

print(
    f"Mean confidence   : {mean_confidence:.4f}"
)

print(
    f"Median confidence : {median_confidence:.4f}"
)

print(
    f"Minimum confidence: {minimum_confidence:.4f}"
)

print(
    f"Maximum confidence: {maximum_confidence:.4f}"
)

print()


# ======================================================================
# PREDICTION CSV
# ======================================================================

print(
    "Saving prediction results..."
)


prediction_rows = []


for row_number in range(
    len(test_idx)
):

    original_index = int(
        test_idx[row_number]
    )

    predicted_indices = np.where(
        Y_pred[row_number] == 1
    )[0]


    predicted_techniques = [
        techniques[i]
        for i in predicted_indices
    ]


    probability_values = [
        float(
            probabilities[
                row_number,
                i
            ]
        )
        for i in predicted_indices
    ]


    prediction_rows.append(
        {
            "original_index":
                original_index,

            "timestamp":
                metadata.iloc[
                    original_index
                ]["timestamp"],

            "agent":
                metadata.iloc[
                    original_index
                ]["agent"],

            "event_id":
                metadata.iloc[
                    original_index
                ]["event_id"],

            "actual_techniques":
                ",".join(
                    label_lists.iloc[
                        original_index
                    ]
                ),

            "predicted_techniques":
                ",".join(
                    predicted_techniques
                ),

            "predicted_probabilities":
                ",".join(
                    f"{p:.6f}"
                    for p in probability_values
                ),

            "max_confidence":
                float(
                    max_confidence[
                        row_number
                    ]
                ),
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


prediction_df.to_csv(
    PREDICTION_PATH,
    index=False,
)


print(
    "Predictions saved to:"
)

print(
    PREDICTION_PATH
)

print()


# ======================================================================
# SAVE MODEL
# ======================================================================

print(
    "Saving model artifacts..."
)


joblib.dump(
    model,
    MODEL_PATH,
)


joblib.dump(
    encoder,
    STRUCTURED_ENCODER_PATH,
)


joblib.dump(
    vectorizer,
    TFIDF_PATH,
)


joblib.dump(
    mlb,
    LABEL_ENCODER_PATH,
)


# ======================================================================
# SAVE FEATURE SCHEMA
# ======================================================================

schema = {

    "version":
        "SmartSOC-Windows-APT-v3.2",

    "model_type":
        "OneVsRest Logistic Regression",

    "classification":
        "multi-label MITRE ATT&CK technique classification",

    "temporal_cutoff":
        str(
            TEMPORAL_CUTOFF
        ),

    "structured_columns":
        list(
            X_struct.columns
        ),

    "categorical_columns":
        categorical_columns,

    "numerical_columns":
        numerical_columns,

    "categorical_encoding":
        "OneHotEncoder(handle_unknown='ignore')",

    "text_column":
        "behavior_text",

    "tfidf_config":
        {
            key:
            list(value)
            if isinstance(value, tuple)
            else value

            for key, value
            in TFIDF_CONFIG.items()
        },

    "tfidf_vocabulary_size":
        len(
            vectorizer.vocabulary_
        ),

    "structured_feature_count":
        int(
            X_struct_train.shape[1]
        ),

    "tfidf_feature_count":
        int(
            X_text_train.shape[1]
        ),

    "final_feature_count":
        int(
            X_train.shape[1]
        ),

    "techniques":
        techniques,

    "threshold":
        THRESHOLD,

    "random_state":
        RANDOM_STATE,

    "metrics":
        {
            "micro_precision":
                float(
                    micro_precision
                ),

            "micro_recall":
                float(
                    micro_recall
                ),

            "micro_f1":
                float(
                    micro_f1
                ),

            "macro_precision":
                float(
                    macro_precision
                ),

            "macro_recall":
                float(
                    macro_recall
                ),

            "macro_f1":
                float(
                    macro_f1
                ),

            "weighted_precision":
                float(
                    weighted_precision
                ),

            "weighted_recall":
                float(
                    weighted_recall
                ),

            "weighted_f1":
                float(
                    weighted_f1
                ),
        },
}


with open(
    SCHEMA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        schema,
        f,
        indent=4,
    )


# ======================================================================
# SAVE REPORT
# ======================================================================

report_text = f"""
SmartSOC Windows-APT Model B v3.2
=================================

Model:
One-vs-Rest Logistic Regression

Architecture:
Structured Windows/Sysmon/Wazuh features
+
TF-IDF behavioral text
+
Multi-label MITRE ATT&CK classification

Temporal cutoff:
{TEMPORAL_CUTOFF}

Training events:
{len(train_idx)}

Testing events:
{len(test_idx)}

Structured features:
{len(X_struct.columns)}

Numerical features:
{len(numerical_columns)}

Categorical features:
{len(categorical_columns)}

Categorical columns:
{categorical_columns}

Structured matrix:
{X_struct_train.shape}

TF-IDF vocabulary:
{len(vectorizer.vocabulary_)}

Final feature matrix:
{X_train.shape}

MITRE techniques:
{len(techniques)}

Threshold:
{THRESHOLD}

Metrics
-------

Micro Precision : {micro_precision:.6f}
Micro Recall    : {micro_recall:.6f}
Micro F1        : {micro_f1:.6f}

Macro Precision : {macro_precision:.6f}
Macro Recall    : {macro_recall:.6f}
Macro F1        : {macro_f1:.6f}

Weighted Precision : {weighted_precision:.6f}
Weighted Recall    : {weighted_recall:.6f}
Weighted F1        : {weighted_f1:.6f}

Confidence
----------

Mean:
{mean_confidence:.6f}

Median:
{median_confidence:.6f}

Minimum:
{minimum_confidence:.6f}

Maximum:
{maximum_confidence:.6f}


Classification Report
=====================

{report}
"""


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        report_text
    )


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print(" SMARTSOC WINDOWS-APT MODEL B v3.2 COMPLETE")
print("=" * 70)
print()

print(
    "Corrected categorical handling:"
)

for col in categorical_columns:

    print(
        "  ✓",
        col
    )

print()

print(
    "Final feature count:",
    X_train.shape[1]
)

print(
    "MITRE techniques:",
    len(techniques)
)

print()

print(
    "Micro F1:",
    f"{micro_f1:.4f}"
)

print(
    "Macro F1:",
    f"{macro_f1:.4f}"
)

print()

print(
    "Files created:"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Structured encoder:",
    STRUCTURED_ENCODER_PATH
)

print(
    "TF-IDF:",
    TFIDF_PATH
)

print(
    "Label encoder:",
    LABEL_ENCODER_PATH
)

print(
    "Feature schema:",
    SCHEMA_PATH
)

print(
    "Report:",
    REPORT_PATH
)

print(
    "Predictions:",
    PREDICTION_PATH
)

print()

print(
    "Next step:"
)

print(
    "Run the corrected calibration pipeline before building inference."
)

print()