import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder


# ================================================================
# SmartSOC Windows-APT Model B v3
# Corrected Hybrid Threshold Calibration
#
# Architecture:
#   Structured:
#       4 categorical + 56 numerical
#
#   Text:
#       TF-IDF
#
#   Final:
#       One-vs-Rest Logistic Regression
#
#   Output:
#       Technique-specific thresholds
# ================================================================


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "windows_apt_v3"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "windows_apt_v3"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "windows_apt_v3"
)


MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_FILE = DATA_DIR / "structured_features.csv"
TEXT_FILE = DATA_DIR / "behavior_text.csv"
LABEL_FILE = DATA_DIR / "labels.csv"
METADATA_FILE = DATA_DIR / "metadata.csv"


print()
print("=" * 75)
print(" SmartSOC Windows-APT Model B v3")
print(" Corrected Hybrid Threshold Calibration")
print("=" * 75)
print()


# ================================================================
# CHECK FILES
# ================================================================

print("Checking required files...")

required_files = [
    STRUCTURED_FILE,
    TEXT_FILE,
    LABEL_FILE,
    METADATA_FILE,
]

for file in required_files:
    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

print("All required files found.")
print()


# ================================================================
# LOAD DATA
# ================================================================

print("Loading datasets...")

structured = pd.read_csv(STRUCTURED_FILE)
text_df = pd.read_csv(TEXT_FILE)
labels_df = pd.read_csv(LABEL_FILE)
metadata = pd.read_csv(METADATA_FILE)

print("Structured rows:", len(structured))
print("Text rows      :", len(text_df))
print("Label rows     :", len(labels_df))
print("Metadata rows  :", len(metadata))
print()


# ================================================================
# BASIC VALIDATION
# ================================================================

if not (
    len(structured)
    == len(text_df)
    == len(labels_df)
    == len(metadata)
):
    raise ValueError(
        "Dataset row counts do not match."
    )


total_events = len(structured)

print("Total events:", total_events)


# ================================================================
# TIMESTAMP PARSING
# ================================================================

metadata["timestamp_parsed"] = pd.to_datetime(
    metadata["timestamp"],
    format="%b %d, %Y @ %H:%M:%S.%f",
    errors="coerce",
)

if metadata["timestamp_parsed"].isna().any():
    invalid = metadata["timestamp_parsed"].isna().sum()

    raise ValueError(
        f"Invalid timestamps detected: {invalid}"
    )


print(
    "Earliest:",
    metadata["timestamp_parsed"].min()
)

print(
    "Latest:",
    metadata["timestamp_parsed"].max()
)

print()


# ================================================================
# LABEL PROCESSING
# ================================================================

def parse_techniques(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


technique_lists = labels_df["techniques"].apply(
    parse_techniques
)


all_techniques = sorted(
    {
        technique
        for techniques in technique_lists
        for technique in techniques
    }
)


print("MITRE techniques:", len(all_techniques))
print(all_techniques)
print()


# ================================================================
# TEMPORAL SPLIT
#
# Final test:
#   <= 2024-12-11  -> calibration/model development area
#   >  2024-12-11  -> untouched final test
#
# Within the training period:
#
#   <= 2024-12-06 -> model training
#   >  2024-12-06  -> validation
# ================================================================

timestamps = metadata["timestamp_parsed"]

FINAL_CUTOFF = pd.Timestamp(
    "2024-12-11 00:00:00"
)

VALIDATION_CUTOFF = pd.Timestamp(
    "2024-12-06 00:00:00"
)


final_test_mask = timestamps > FINAL_CUTOFF

validation_mask = (
    (timestamps > VALIDATION_CUTOFF)
    & (timestamps <= FINAL_CUTOFF)
)

model_train_mask = timestamps <= VALIDATION_CUTOFF


model_train_indices = np.where(
    model_train_mask
)[0]

validation_indices = np.where(
    validation_mask
)[0]

final_test_indices = np.where(
    final_test_mask
)[0]


print("===============================================================")
print(" TEMPORAL CALIBRATION SPLIT")
print("===============================================================")
print()

print(
    "Model training cutoff:",
    VALIDATION_CUTOFF
)

print(
    "Final test cutoff:",
    FINAL_CUTOFF
)

print()

print(
    "Model training events:",
    len(model_train_indices)
)

print(
    "Validation events    :",
    len(validation_indices)
)

print(
    "Final test events    :",
    len(final_test_indices)
)

print()


if len(model_train_indices) == 0:
    raise ValueError(
        "No model-training events."
    )

if len(validation_indices) == 0:
    raise ValueError(
        "No validation events."
    )

if len(final_test_indices) == 0:
    raise ValueError(
        "No final test events."
    )


# ================================================================
# PREPARE LABEL MATRIX
# ================================================================

mlb = MultiLabelBinarizer(
    classes=all_techniques
)

Y = mlb.fit_transform(
    technique_lists
)


Y_train = Y[model_train_indices]
Y_val = Y[validation_indices]
Y_test = Y[final_test_indices]


print("Y train:", Y_train.shape)
print("Y validation:", Y_val.shape)
print("Y test:", Y_test.shape)
print()


# ================================================================
# CHECK TRAINING LABEL COVERAGE
# ================================================================

train_counts = Y_train.sum(axis=0)

missing_from_train = [
    technique
    for technique, count
    in zip(mlb.classes_, train_counts)
    if count == 0
]


if missing_from_train:

    print(
        "WARNING: techniques missing from model training:"
    )

    for technique in missing_from_train:
        print(" -", technique)

    print()

else:

    print(
        "Every MITRE technique has training examples."
    )

    print()


# ================================================================
# STRUCTURED FEATURES
# ================================================================

print("Preparing structured features...")

categorical_columns = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]


numerical_columns = [
    column
    for column in structured.columns
    if column not in categorical_columns
]


print("Categorical features:")

for column in categorical_columns:
    print(" -", column)

print()

print(
    "Numerical feature count:",
    len(numerical_columns)
)

print()


# ================================================================
# CLEAN CATEGORICAL DATA
# ================================================================

X_cat = structured[
    categorical_columns
].copy()


X_cat = X_cat.fillna("")


for column in categorical_columns:

    X_cat[column] = (
        X_cat[column]
        .astype(str)
        .replace("nan", "")
    )


# ================================================================
# FIT ONE-HOT ENCODER ONLY ON MODEL TRAINING DATA
# ================================================================

print(
    "Fitting categorical encoder on MODEL TRAINING data only..."
)


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
    dtype=np.float32,
)


X_cat_train = encoder.fit_transform(
    X_cat.iloc[model_train_indices]
)

X_cat_val = encoder.transform(
    X_cat.iloc[validation_indices]
)

X_cat_test = encoder.transform(
    X_cat.iloc[final_test_indices]
)


print(
    "Categorical train:",
    X_cat_train.shape
)

print(
    "Categorical validation:",
    X_cat_val.shape
)

print(
    "Categorical test:",
    X_cat_test.shape
)

print()


# ================================================================
# NUMERICAL FEATURES
# ================================================================

print("Preparing numerical features...")


X_num = structured[
    numerical_columns
].copy()


X_num = X_num.apply(
    pd.to_numeric,
    errors="coerce"
)


X_num = X_num.replace(
    [np.inf, -np.inf],
    np.nan
)


# IMPORTANT:
# Fit imputation statistics only on model-training data.

train_num = X_num.iloc[
    model_train_indices
]


numeric_medians = train_num.median()


train_num = train_num.fillna(
    numeric_medians
)

val_num = X_num.iloc[
    validation_indices
].fillna(
    numeric_medians
)

test_num = X_num.iloc[
    final_test_indices
].fillna(
    numeric_medians
)


train_num = train_num.astype(
    np.float32
)

val_num = val_num.astype(
    np.float32
)

test_num = test_num.astype(
    np.float32
)


X_num_train = csr_matrix(
    train_num.to_numpy()
)

X_num_val = csr_matrix(
    val_num.to_numpy()
)

X_num_test = csr_matrix(
    test_num.to_numpy()
)


print(
    "Numerical train:",
    X_num_train.shape
)

print(
    "Numerical validation:",
    X_num_val.shape
)

print(
    "Numerical test:",
    X_num_test.shape
)

print()


# ================================================================
# STRUCTURED MATRIX
# ================================================================

X_struct_train = hstack(
    [
        X_cat_train,
        X_num_train,
    ],
    format="csr",
)


X_struct_val = hstack(
    [
        X_cat_val,
        X_num_val,
    ],
    format="csr",
)


X_struct_test = hstack(
    [
        X_cat_test,
        X_num_test,
    ],
    format="csr",
)


print("Structured matrices:")

print(
    "Train:",
    X_struct_train.shape
)

print(
    "Validation:",
    X_struct_val.shape
)

print(
    "Test:",
    X_struct_test.shape
)

print()


# ================================================================
# BEHAVIOR TEXT
# ================================================================

print(
    "Preparing behavioral text..."
)


if "behavior_text" in text_df.columns:

    text_column = "behavior_text"

elif "text" in text_df.columns:

    text_column = "text"

else:

    # Fall back to first column
    text_column = text_df.columns[0]


texts = (
    text_df[text_column]
    .fillna("")
    .astype(str)
)


train_text = texts.iloc[
    model_train_indices
]

val_text = texts.iloc[
    validation_indices
]

test_text = texts.iloc[
    final_test_indices
]


print(
    "Text column:",
    text_column
)

print(
    "Training text rows:",
    len(train_text)
)

print(
    "Validation text rows:",
    len(val_text)
)

print(
    "Testing text rows:",
    len(test_text)
)

print()


# ================================================================
# TF-IDF
# ================================================================

print(
    "Fitting TF-IDF on MODEL TRAINING DATA ONLY..."
)


tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_features=15000,
    sublinear_tf=True,
    dtype=np.float32,
)


X_text_train = tfidf.fit_transform(
    train_text
)

X_text_val = tfidf.transform(
    val_text
)

X_text_test = tfidf.transform(
    test_text
)


print(
    "TF-IDF vocabulary:",
    len(tfidf.vocabulary_)
)

print(
    "Text train:",
    X_text_train.shape
)

print(
    "Text validation:",
    X_text_val.shape
)

print(
    "Text test:",
    X_text_test.shape
)

print()


# ================================================================
# FINAL HYBRID FEATURES
# ================================================================

print(
    "Combining structured + text features..."
)


X_train = hstack(
    [
        X_struct_train,
        X_text_train,
    ],
    format="csr",
)


X_val = hstack(
    [
        X_struct_val,
        X_text_val,
    ],
    format="csr",
)


X_test = hstack(
    [
        X_struct_test,
        X_text_test,
    ],
    format="csr",
)


# Explicitly make matrices writable.
# This prevents the Windows scipy/joblib
# WRITEBACKIFCOPY base is read-only issue.

X_train = csr_matrix(
    X_train,
    dtype=np.float32,
    copy=True,
)

X_val = csr_matrix(
    X_val,
    dtype=np.float32,
    copy=True,
)

X_test = csr_matrix(
    X_test,
    dtype=np.float32,
    copy=True,
)


X_train.sort_indices()
X_val.sort_indices()
X_test.sort_indices()


print(
    "Final train:",
    X_train.shape
)

print(
    "Final validation:",
    X_val.shape
)

print(
    "Final test:",
    X_test.shape
)

print()


# ================================================================
# TRAIN MODEL
# ================================================================

print(
    "Training One-vs-Rest Logistic Regression..."
)

print(
    "Using n_jobs=1 for stable Windows execution."
)

print()


base_classifier = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear",
    class_weight=None,
)


model = OneVsRestClassifier(
    base_classifier,
    n_jobs=1,
)


model.fit(
    X_train,
    Y_train
)


print(
    "Training complete."
)

print()


# ================================================================
# VALIDATION PROBABILITIES
# ================================================================

print(
    "Generating validation probabilities..."
)


val_probabilities = model.predict_proba(
    X_val
)


print(
    "Validation probability matrix:",
    val_probabilities.shape
)

print()


# ================================================================
# THRESHOLD SEARCH
# ================================================================

print(
    "Searching technique-specific thresholds..."
)

print()


candidate_thresholds = np.arange(
    0.05,
    0.96,
    0.05
)


threshold_records = []


for class_index, technique in enumerate(
    mlb.classes_
):

    y_true = Y_val[:, class_index]

    probabilities = val_probabilities[
        :, class_index
    ]


    support = int(
        y_true.sum()
    )


    # ------------------------------------------------------------
    # No validation positives
    # ------------------------------------------------------------

    if support == 0:

        threshold = 0.50

        predictions = (
            probabilities >= threshold
        )

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        threshold_records.append(
            {
                "technique": technique,
                "support": support,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "status": "NO_VALIDATION_POSITIVES",
            }
        )

        continue


    best = None


    for threshold in candidate_thresholds:

        predictions = (
            probabilities >= threshold
        )


        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )


        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )


        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )


        # Primary objective:
        # maximize F1.
        #
        # Secondary objective:
        # maximize precision when
        # F1 values are equal.

        candidate = (
            f1,
            precision,
            recall,
            threshold,
        )


        if best is None or candidate > best:
            best = candidate


    best_f1 = best[0]
    best_precision = best[1]
    best_recall = best[2]
    best_threshold = best[3]


    threshold_records.append(
        {
            "technique": technique,
            "support": support,
            "threshold": float(
                best_threshold
            ),
            "precision": float(
                best_precision
            ),
            "recall": float(
                best_recall
            ),
            "f1": float(
                best_f1
            ),
            "status": "CALIBRATED",
        }
    )


threshold_df = pd.DataFrame(
    threshold_records
)


print(
    threshold_df
    .sort_values(
        "support",
        ascending=False
    )
    .head(25)
    .to_string(index=False)
)

print()


# ================================================================
# SAVE THRESHOLDS
# ================================================================

thresholds = {
    row["technique"]:
        float(row["threshold"])
    for row in threshold_records
}


threshold_file = (
    MODEL_DIR
    / "technique_thresholds.json"
)


with open(
    threshold_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        thresholds,
        f,
        indent=4,
        sort_keys=True,
    )


threshold_csv = (
    RESULT_DIR
    / "threshold_calibration.csv"
)


threshold_df.to_csv(
    threshold_csv,
    index=False,
)


# ================================================================
# APPLY CALIBRATED THRESHOLDS
# ================================================================

def apply_thresholds(
    probabilities,
    thresholds_dict,
    classes,
):

    predictions = np.zeros_like(
        probabilities,
        dtype=np.int8,
    )


    for index, technique in enumerate(
        classes
    ):

        threshold = thresholds_dict.get(
            technique,
            0.50,
        )

        predictions[:, index] = (
            probabilities[:, index]
            >= threshold
        ).astype(np.int8)


    # ------------------------------------------------------------
    # Safety fallback:
    #
    # Every event must have at least one
    # predicted technique.
    #
    # If nothing crosses its threshold,
    # choose the highest probability technique.
    # ------------------------------------------------------------

    no_prediction = (
        predictions.sum(axis=1) == 0
    )


    fallback_count = int(
        no_prediction.sum()
    )


    if fallback_count > 0:

        best_indices = np.argmax(
            probabilities[no_prediction],
            axis=1,
        )


        rows = np.where(
            no_prediction
        )[0]


        predictions[
            rows,
            best_indices
        ] = 1


    return predictions, fallback_count


# ================================================================
# VALIDATION AFTER CALIBRATION
# ================================================================

val_predictions, val_fallbacks = (
    apply_thresholds(
        val_probabilities,
        thresholds,
        mlb.classes_,
    )
)


print(
    "==============================================================="
)

print(
    " VALIDATION AFTER THRESHOLD CALIBRATION"
)

print(
    "==============================================================="
)

print()


val_micro_precision = precision_score(
    Y_val,
    val_predictions,
    average="micro",
    zero_division=0,
)

val_micro_recall = recall_score(
    Y_val,
    val_predictions,
    average="micro",
    zero_division=0,
)

val_micro_f1 = f1_score(
    Y_val,
    val_predictions,
    average="micro",
    zero_division=0,
)

val_macro_f1 = f1_score(
    Y_val,
    val_predictions,
    average="macro",
    zero_division=0,
)


print(
    f"Micro Precision : {val_micro_precision:.4f}"
)

print(
    f"Micro Recall    : {val_micro_recall:.4f}"
)

print(
    f"Micro F1        : {val_micro_f1:.4f}"
)

print(
    f"Macro F1        : {val_macro_f1:.4f}"
)

print(
    "Fallback events :",
    val_fallbacks,
)

print()


# ================================================================
# FINAL TEST
#
# IMPORTANT:
# Thresholds were learned ONLY from validation.
#
# The final test remains untouched until this point.
# ================================================================

print(
    "Generating FINAL TEST predictions..."
)

test_probabilities = model.predict_proba(
    X_test
)


test_predictions, test_fallbacks = (
    apply_thresholds(
        test_probabilities,
        thresholds,
        mlb.classes_,
    )
)


# ================================================================
# FINAL TEST METRICS
# ================================================================

test_micro_precision = precision_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0,
)

test_micro_recall = recall_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0,
)

test_micro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0,
)

test_macro_precision = precision_score(
    Y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_macro_recall = recall_score(
    Y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_macro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="macro",
    zero_division=0,
)

test_weighted_precision = precision_score(
    Y_test,
    test_predictions,
    average="weighted",
    zero_division=0,
)

test_weighted_recall = recall_score(
    Y_test,
    test_predictions,
    average="weighted",
    zero_division=0,
)

test_weighted_f1 = f1_score(
    Y_test,
    test_predictions,
    average="weighted",
    zero_division=0,
)


print()
print("=" * 75)
print(" FINAL WINDOWS-APT MODEL B v3")
print(" CALIBRATED RESULTS")
print("=" * 75)
print()


print(
    f"Micro Precision   : {test_micro_precision:.4f}"
)

print(
    f"Micro Recall      : {test_micro_recall:.4f}"
)

print(
    f"Micro F1          : {test_micro_f1:.4f}"
)

print()


print(
    f"Macro Precision   : {test_macro_precision:.4f}"
)

print(
    f"Macro Recall      : {test_macro_recall:.4f}"
)

print(
    f"Macro F1          : {test_macro_f1:.4f}"
)

print()


print(
    f"Weighted Precision: {test_weighted_precision:.4f}"
)

print(
    f"Weighted Recall   : {test_weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {test_weighted_f1:.4f}"
)

print()


print(
    "Fallback events   :",
    test_fallbacks,
)

print()


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    Y_test,
    test_predictions,
    target_names=mlb.classes_,
    zero_division=0,
)


print("Classification Report:")
print()
print(report)


report_file = (
    RESULT_DIR
    / "calibrated_classification_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "SmartSOC Windows-APT Model B v3\n"
    )

    f.write(
        "Corrected Hybrid Calibration\n\n"
    )

    f.write(
        f"Micro Precision: "
        f"{test_micro_precision:.6f}\n"
    )

    f.write(
        f"Micro Recall: "
        f"{test_micro_recall:.6f}\n"
    )

    f.write(
        f"Micro F1: "
        f"{test_micro_f1:.6f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{test_macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{test_macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{test_macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{test_weighted_precision:.6f}\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{test_weighted_recall:.6f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{test_weighted_f1:.6f}\n"
    )

    f.write(
        f"\nFallback events: "
        f"{test_fallbacks}\n\n"
    )

    f.write(report)


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

prediction_rows = []


for row_index, original_index in enumerate(
    final_test_indices
):

    predicted_techniques = [
        mlb.classes_[class_index]
        for class_index in np.where(
            test_predictions[row_index] == 1
        )[0]
    ]


    true_techniques = [
        mlb.classes_[class_index]
        for class_index in np.where(
            Y_test[row_index] == 1
        )[0]
    ]


    max_confidence = float(
        np.max(
            test_probabilities[row_index]
        )
    )


    prediction_rows.append(
        {
            "original_index": int(
                original_index
            ),
            "timestamp": metadata.iloc[
                original_index
            ]["timestamp"],
            "agent": metadata.iloc[
                original_index
            ]["agent"],
            "event_id": metadata.iloc[
                original_index
            ]["event_id"],
            "true_techniques": ",".join(
                true_techniques
            ),
            "predicted_techniques": ",".join(
                predicted_techniques
            ),
            "confidence": max_confidence,
        }
    )


prediction_df = pd.DataFrame(
    prediction_rows
)


prediction_file = (
    RESULT_DIR
    / "calibrated_test_predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False,
)


# ================================================================
# SAVE MODEL ARTIFACTS
# ================================================================

print()
print(
    "Saving calibrated model artifacts..."
)


model_file = (
    MODEL_DIR
    / "multilabel_logistic_v3_calibrated.joblib"
)


encoder_file = (
    MODEL_DIR
    / "structured_encoder_calibrated.joblib"
)


tfidf_file = (
    MODEL_DIR
    / "tfidf_vectorizer_calibrated.joblib"
)


label_encoder_file = (
    MODEL_DIR
    / "multilabel_encoder_calibrated.joblib"
)


joblib.dump(
    model,
    model_file,
)

joblib.dump(
    encoder,
    encoder_file,
)

joblib.dump(
    tfidf,
    tfidf_file,
)

joblib.dump(
    mlb,
    label_encoder_file,
)


# ================================================================
# SAVE FEATURE SCHEMA
# ================================================================

schema = {
    "model_version": "SmartSOC Windows-APT Model B v3 calibrated",
    "categorical_columns": categorical_columns,
    "numerical_columns": numerical_columns,
    "text_column": text_column,
    "categorical_feature_count": int(
        X_struct_train.shape[1]
        - len(numerical_columns)
    ),
    "numerical_feature_count": len(
        numerical_columns
    ),
    "tfidf_feature_count": len(
        tfidf.vocabulary_
    ),
    "final_feature_count": int(
        X_train.shape[1]
    ),
    "mitre_technique_count": len(
        mlb.classes_
    ),
    "mitre_techniques": list(
        mlb.classes_
    ),
    "validation_cutoff": str(
        VALIDATION_CUTOFF
    ),
    "final_test_cutoff": str(
        FINAL_CUTOFF
    ),
    "threshold_method": (
        "validation-set F1 optimization"
    ),
    "fallback_strategy": (
        "highest probability technique"
    ),
}


schema_file = (
    MODEL_DIR
    / "feature_schema_calibrated_v3.json"
)


with open(
    schema_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        schema,
        f,
        indent=4,
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 75)
print(" SmartSOC Model B v3 Calibration COMPLETE")
print("=" * 75)
print()

print("Created:")

print(
    model_file
)

print(
    encoder_file
)

print(
    tfidf_file
)

print(
    label_encoder_file
)

print(
    threshold_file
)

print(
    schema_file
)

print(
    threshold_csv
)

print(
    report_file
)

print(
    prediction_file
)

print()

print(
    "Final Micro F1:",
    f"{test_micro_f1:.4f}"
)

print(
    "Final Macro F1:",
    f"{test_macro_f1:.4f}"
)

print()

print(
    "Next step:"
)

print(
    "Build the production Wazuh/Sysmon inference pipeline "
    "using the calibrated artifacts."
)

print()