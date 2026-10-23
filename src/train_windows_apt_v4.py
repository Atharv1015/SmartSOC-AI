"""
SmartSOC Windows-APT Model B v4
Corrected Hybrid Training Pipeline

Pipeline:
    Structured Windows/Sysmon/Wazuh features
            +
    Behavioral text -> TF-IDF
            +
    Multi-label MITRE ATT&CK classification

Important:
- Temporal split is preserved from v4 preparation.
- Categorical features are one-hot encoded using TRAINING data only.
- Numerical features are detected automatically from the structured CSV.
- Numerical features are scaled using TRAINING data only.
- TF-IDF is fitted using TRAINING text only.
- No class_weight="balanced".
- One-vs-Rest Logistic Regression is used as the baseline.
- Threshold remains 0.50 initially.
- Threshold calibration is a separate step.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "windows_apt_v4"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "windows_apt_v4"
    / "corrected"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "windows_apt_v4"
    / "corrected"
)

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

TEXT_TRAIN = DATA_DIR / "behavior_text_train.csv"
TEXT_TEST = DATA_DIR / "behavior_text_test.csv"

LABEL_TRAIN = DATA_DIR / "labels_train.csv"
LABEL_TEST = DATA_DIR / "labels_test.csv"

META_TRAIN = DATA_DIR / "metadata_train.csv"
META_TEST = DATA_DIR / "metadata_test.csv"

SCHEMA_FILE = DATA_DIR / "schema.json"

MODEL_FILE = MODEL_DIR / "multilabel_logistic_v4_corrected.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder_v4_corrected.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler_v4_corrected.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer_v4_corrected.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder_v4_corrected.joblib"
FEATURE_SCHEMA_FILE = MODEL_DIR / "feature_schema_v4_corrected.json"

REPORT_FILE = RESULT_DIR / "classification_report.txt"
PREDICTIONS_FILE = RESULT_DIR / "predictions.csv"
FEATURE_INFO_FILE = RESULT_DIR / "feature_information.json"


# ============================================================
# CONSTANTS
# ============================================================

CATEGORICAL_COLUMNS = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]

# Columns that must NEVER become model features.
EXCLUDED_COLUMNS = {
    "timestamp",
    "agent",
    "techniques",
    "label",
    "target",
    "id",
    "index",
}

# Initial threshold.
# Calibration will be performed separately.
DEFAULT_THRESHOLD = 0.50

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(f" {title}")
    print("=" * 75)


def check_required_files():
    print_header("Checking required files")

    required = [
        STRUCTURED_TRAIN,
        STRUCTURED_TEST,
        TEXT_TRAIN,
        TEXT_TEST,
        LABEL_TRAIN,
        LABEL_TEST,
        META_TRAIN,
        META_TEST,
        SCHEMA_FILE,
    ]

    missing = []

    for path in required:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[MISSING] {path}")
            missing.append(path)

    if missing:
        print()
        print("ERROR: Required files are missing.")

        for path in missing:
            print(f"  - {path}")

        raise FileNotFoundError(
            "One or more required v4 dataset files are missing."
        )

    print()
    print("All required files found.")


def load_datasets():
    print_header("Loading v4 datasets")

    X_train = pd.read_csv(STRUCTURED_TRAIN)
    X_test = pd.read_csv(STRUCTURED_TEST)

    text_train = pd.read_csv(TEXT_TRAIN)
    text_test = pd.read_csv(TEXT_TEST)

    y_train = pd.read_csv(LABEL_TRAIN)
    y_test = pd.read_csv(LABEL_TEST)

    meta_train = pd.read_csv(META_TRAIN)
    meta_test = pd.read_csv(META_TEST)

    print(f"Structured training rows : {len(X_train)}")
    print(f"Structured testing rows  : {len(X_test)}")

    print(f"Text training rows       : {len(text_train)}")
    print(f"Text testing rows        : {len(text_test)}")

    print(f"Label training rows      : {len(y_train)}")
    print(f"Label testing rows       : {len(y_test)}")

    print(f"Metadata training rows   : {len(meta_train)}")
    print(f"Metadata testing rows    : {len(meta_test)}")

    # --------------------------------------------------------
    # Alignment checks
    # --------------------------------------------------------

    if not (
        len(X_train)
        == len(text_train)
        == len(y_train)
        == len(meta_train)
    ):
        raise ValueError(
            "Training datasets are not row-aligned."
        )

    if not (
        len(X_test)
        == len(text_test)
        == len(y_test)
        == len(meta_test)
    ):
        raise ValueError(
            "Testing datasets are not row-aligned."
        )

    print()
    print("Dataset row alignment: OK")

    return (
        X_train,
        X_test,
        text_train,
        text_test,
        y_train,
        y_test,
        meta_train,
        meta_test,
    )


# ============================================================
# LABEL PREPARATION
# ============================================================

def prepare_labels(y_train, y_test):

    print_header("Preparing MITRE ATT&CK labels")

    if "techniques" not in y_train.columns:
        raise ValueError(
            "labels_train.csv does not contain 'techniques'."
        )

    if "techniques" not in y_test.columns:
        raise ValueError(
            "labels_test.csv does not contain 'techniques'."
        )

    def parse_labels(value):

        if pd.isna(value):
            return []

        value = str(value).strip()

        if not value:
            return []

        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    train_labels = y_train["techniques"].apply(parse_labels)
    test_labels = y_test["techniques"].apply(parse_labels)

    all_labels = sorted(
        set(
            technique
            for row in train_labels
            for technique in row
        )
        |
        set(
            technique
            for row in test_labels
            for technique in row
        )
    )

    print()
    print(f"MITRE techniques: {len(all_labels)}")
    print(all_labels)

    train_set = set(
        technique
        for row in train_labels
        for technique in row
    )

    test_set = set(
        technique
        for row in test_labels
        for technique in row
    )

    test_only = sorted(test_set - train_set)

    if test_only:
        print()
        print("WARNING: Test-only techniques detected:")
        print(test_only)

        raise ValueError(
            "Temporal test contains techniques absent from training. "
            "Model cannot learn these classes."
        )

    print()
    print("No test-only techniques.")

    mlb = MultiLabelBinarizer(classes=all_labels)

    Y_train = mlb.fit_transform(train_labels)
    Y_test = mlb.transform(test_labels)

    print()
    print("Y train:", Y_train.shape)
    print("Y test :", Y_test.shape)

    class_counts = Y_train.sum(axis=0)

    print()
    print("Training label distribution:")

    for technique, count in zip(
        mlb.classes_,
        class_counts,
    ):
        print(
            f"{technique:<12} {int(count)}"
        )

    zero_training = [
        technique
        for technique, count in zip(
            mlb.classes_,
            class_counts,
        )
        if count == 0
    ]

    if zero_training:
        raise ValueError(
            "Some MITRE techniques have zero training examples: "
            + str(zero_training)
        )

    return (
        mlb,
        train_labels,
        test_labels,
        Y_train,
        Y_test,
    )


# ============================================================
# STRUCTURED FEATURE DETECTION
# ============================================================

def detect_feature_columns(X_train, X_test):

    print_header("Detecting structured feature columns")

    all_columns = list(X_train.columns)

    print()
    print(f"Original structured feature count: {len(all_columns)}")

    # --------------------------------------------------------
    # Verify categorical columns
    # --------------------------------------------------------

    missing_categorical = [
        col
        for col in CATEGORICAL_COLUMNS
        if col not in X_train.columns
    ]

    if missing_categorical:
        raise ValueError(
            "Required categorical columns missing: "
            + str(missing_categorical)
        )

    print()
    print("Categorical:")

    for col in CATEGORICAL_COLUMNS:
        print(f"  - {col}")

    # --------------------------------------------------------
    # Automatically detect numerical columns
    # --------------------------------------------------------

    numerical_columns = []

    for col in all_columns:

        if col in CATEGORICAL_COLUMNS:
            continue

        if col.lower() in EXCLUDED_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(
            X_train[col]
        ):
            numerical_columns.append(col)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not numerical_columns:
        raise ValueError(
            """
NO NUMERICAL FEATURES WERE DETECTED.

This is the exact problem we are fixing in v4.

Check structured_train.csv and make sure the behavioral
numeric features are actually present.
"""
        )

    print()
    print(
        f"Numerical feature count: "
        f"{len(numerical_columns)}"
    )

    print()
    print("Numerical:")

    for col in numerical_columns:
        print(f"  - {col}")

    # --------------------------------------------------------
    # Check for unsupported object columns
    # --------------------------------------------------------

    ignored_object_columns = []

    for col in all_columns:

        if col in CATEGORICAL_COLUMNS:
            continue

        if col in numerical_columns:
            continue

        if col.lower() in EXCLUDED_COLUMNS:
            continue

        if not pd.api.types.is_numeric_dtype(
            X_train[col]
        ):
            ignored_object_columns.append(col)

    if ignored_object_columns:
        print()
        print(
            "Ignored non-numeric/non-categorical columns:"
        )

        for col in ignored_object_columns:
            print(f"  - {col}")

    return numerical_columns


# ============================================================
# STRUCTURED PREPROCESSING
# ============================================================

def prepare_structured_features(
    X_train,
    X_test,
    numerical_columns,
):

    print_header(
        "Preparing structured features"
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    print()
    print(
        "Fitting categorical encoder "
        "on TRAINING DATA ONLY..."
    )

    X_train_cat = X_train[
        CATEGORICAL_COLUMNS
    ].copy()

    X_test_cat = X_test[
        CATEGORICAL_COLUMNS
    ].copy()

    # Fill missing categorical values.
    X_train_cat = X_train_cat.fillna(
        "__MISSING__"
    ).astype(str)

    X_test_cat = X_test_cat.fillna(
        "__MISSING__"
    ).astype(str)

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        # Compatibility with older sklearn.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )

    cat_train = encoder.fit_transform(
        X_train_cat
    )

    cat_test = encoder.transform(
        X_test_cat
    )

    print(
        f"Categorical train: {cat_train.shape}"
    )

    print(
        f"Categorical test : {cat_test.shape}"
    )

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    print()
    print(
        "Preparing numerical features..."
    )

    X_train_num = X_train[
        numerical_columns
    ].copy()

    X_test_num = X_test[
        numerical_columns
    ].copy()

    # Force numeric conversion.
    for col in numerical_columns:

        X_train_num[col] = pd.to_numeric(
            X_train_num[col],
            errors="coerce",
        )

        X_test_num[col] = pd.to_numeric(
            X_test_num[col],
            errors="coerce",
        )

    # Replace infinities.
    X_train_num = X_train_num.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_test_num = X_test_num.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation based ONLY on training data.
    train_medians = X_train_num.median()

    X_train_num = X_train_num.fillna(
        train_medians
    )

    X_test_num = X_test_num.fillna(
        train_medians
    )

    # Any column that is still NaN gets zero.
    X_train_num = X_train_num.fillna(0)
    X_test_num = X_test_num.fillna(0)

    # Scale without centering so sparse matrix can be used.
    scaler = StandardScaler(
        with_mean=False,
        with_std=True,
    )

    num_train = scaler.fit_transform(
        X_train_num.astype(np.float32)
    )

    num_test = scaler.transform(
        X_test_num.astype(np.float32)
    )

    num_train = csr_matrix(
        num_train,
        dtype=np.float32,
    )

    num_test = csr_matrix(
        num_test,
        dtype=np.float32,
    )

    print(
        f"Numerical train: {num_train.shape}"
    )

    print(
        f"Numerical test : {num_test.shape}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    structured_train = hstack(
        [
            cat_train,
            num_train,
        ],
        format="csr",
        dtype=np.float32,
    )

    structured_test = hstack(
        [
            cat_test,
            num_test,
        ],
        format="csr",
        dtype=np.float32,
    )

    print()
    print("Structured matrix:")

    print(
        f"Train: {structured_train.shape}"
    )

    print(
        f"Test : {structured_test.shape}"
    )

    # --------------------------------------------------------
    # Critical sanity check
    # --------------------------------------------------------

    expected_min_numeric = len(
        numerical_columns
    )

    if structured_train.shape[1] < (
        cat_train.shape[1]
        + expected_min_numeric
    ):
        raise RuntimeError(
            "Numerical features were not correctly "
            "included in the structured matrix."
        )

    return (
        structured_train,
        structured_test,
        encoder,
        scaler,
    )


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def find_text_column(df):

    candidates = [
        "behavior_text",
        "text",
        "behavior",
        "full_log",
        "full-log",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: first column.
    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(
        "Could not determine behavior-text column. "
        f"Available columns: {list(df.columns)}"
    )


def prepare_text_features(
    text_train,
    text_test,
):

    print_header(
        "Preparing behavioral text"
    )

    train_column = find_text_column(
        text_train
    )

    test_column = find_text_column(
        text_test
    )

    print()
    print(
        f"Training text column: {train_column}"
    )

    print(
        f"Testing text column : {test_column}"
    )

    train_text = (
        text_train[train_column]
        .fillna("")
        .astype(str)
    )

    test_text = (
        text_test[test_column]
        .fillna("")
        .astype(str)
    )

    print()
    print(
        f"Training text rows: {len(train_text)}"
    )

    print(
        f"Testing text rows : {len(test_text)}"
    )

    print()
    print(
        "Fitting TF-IDF on TRAINING DATA ONLY..."
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=15000,
        dtype=np.float32,
    )

    text_train_matrix = vectorizer.fit_transform(
        train_text
    )

    text_test_matrix = vectorizer.transform(
        test_text
    )

    print(
        f"TF-IDF vocabulary: "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Text train: {text_train_matrix.shape}"
    )

    print(
        f"Text test : {text_test_matrix.shape}"
    )

    return (
        text_train_matrix,
        text_test_matrix,
        vectorizer,
    )


# ============================================================
# HYBRID MATRIX
# ============================================================

def combine_features(
    structured_train,
    structured_test,
    text_train,
    text_test,
):

    print_header(
        "Combining structured + behavioral text"
    )

    X_train = hstack(
        [
            structured_train,
            text_train,
        ],
        format="csr",
        dtype=np.float32,
    )

    X_test = hstack(
        [
            structured_test,
            text_test,
        ],
        format="csr",
        dtype=np.float32,
    )

    # Make writable CSR matrices.
    X_train = csr_matrix(
        X_train,
        copy=True,
    )

    X_test = csr_matrix(
        X_test,
        copy=True,
    )

    print()
    print(
        f"Final train matrix: {X_train.shape}"
    )

    print(
        f"Final test matrix : {X_test.shape}"
    )

    print(
        f"Total model features: "
        f"{X_train.shape[1]}"
    )

    return X_train, X_test


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    X_train,
    Y_train,
):

    print_header(
        "Training corrected One-vs-Rest Logistic Regression"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "- class_weight='balanced' is NOT used."
    )

    print(
        "- Initial threshold is 0.50."
    )

    print(
        "- Threshold calibration will happen later."
    )

    print(
        "- n_jobs=1 is used for stable Windows execution."
    )

    base_estimator = LogisticRegression(
        C=1.0,
        max_iter=1500,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model = OneVsRestClassifier(
        estimator=base_estimator,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning,
        )

        model.fit(
            X_train,
            Y_train,
        )

    print()
    print("Training complete.")

    return model


# ============================================================
# PREDICTION
# ============================================================

def generate_predictions(
    model,
    X_test,
    mlb,
):

    print_header(
        "Generating predictions"
    )

    probabilities = model.predict_proba(
        X_test
    )

    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    predictions = (
        probabilities
        >= DEFAULT_THRESHOLD
    ).astype(np.int8)

    no_prediction = (
        predictions.sum(axis=1) == 0
    )

    no_prediction_count = int(
        no_prediction.sum()
    )

    print()
    print(
        "Events with no prediction "
        f"before fallback: "
        f"{no_prediction_count}"
    )

    # --------------------------------------------------------
    # Fallback
    #
    # Every event should receive at least one
    # predicted technique for this baseline.
    # The highest probability technique is selected.
    # --------------------------------------------------------

    if no_prediction_count > 0:

        best_class = np.argmax(
            probabilities[
                no_prediction
            ],
            axis=1,
        )

        rows = np.where(
            no_prediction
        )[0]

        predictions[
            rows,
            best_class,
        ] = 1

    return (
        probabilities,
        predictions,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    Y_test,
    predictions,
    mlb,
):

    print_header(
        "Model Evaluation"
    )

    micro_precision = precision_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        Y_test,
        predictions,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        Y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        Y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        Y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        Y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

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
        f"Weighted Precision : "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        Y_test,
        predictions,
        target_names=mlb.classes_,
        zero_division=0,
    )

    print()
    print("Classification Report:")
    print()
    print(report)

    metrics = {
        "micro_precision": float(
            micro_precision
        ),
        "micro_recall": float(
            micro_recall
        ),
        "micro_f1": float(
            micro_f1
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
    }

    return report, metrics


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

def analyze_prediction_distribution(
    Y_test,
    predictions,
    probabilities,
    mlb,
):

    print_header(
        "Prediction Distribution Analysis"
    )

    print()
    print(
        "Technique prediction counts:"
    )

    actual_counts = Y_test.sum(axis=0)
    predicted_counts = predictions.sum(axis=0)

    rows = []

    for i, technique in enumerate(
        mlb.classes_
    ):

        actual = int(
            actual_counts[i]
        )

        predicted = int(
            predicted_counts[i]
        )

        rows.append(
            {
                "technique": technique,
                "actual": actual,
                "predicted": predicted,
                "difference": predicted - actual,
            }
        )

        print(
            f"{technique:<12} "
            f"actual={actual:<6} "
            f"predicted={predicted:<6} "
            f"diff={predicted - actual:+}"
        )

    distribution_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    max_confidence = probabilities.max(
        axis=1
    )

    print()
    print(
        f"Mean confidence   : "
        f"{max_confidence.mean():.4f}"
    )

    print(
        f"Median confidence : "
        f"{np.median(max_confidence):.4f}"
    )

    print(
        f"Minimum confidence: "
        f"{max_confidence.min():.4f}"
    )

    print(
        f"Maximum confidence: "
        f"{max_confidence.max():.4f}"
    )

    return distribution_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    probabilities,
    predictions,
    Y_test,
    mlb,
    meta_test,
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    print_header(
        "Saving prediction results"
    )

    output = meta_test.copy()

    # --------------------------------------------------------
    # Actual techniques
    # --------------------------------------------------------

    actual_labels = []

    for row in Y_test:

        labels = [
            mlb.classes_[i]
            for i, value in enumerate(row)
            if value == 1
        ]

        actual_labels.append(
            ",".join(labels)
        )

    # --------------------------------------------------------
    # Predicted techniques
    # --------------------------------------------------------

    predicted_labels = []

    for row in predictions:

        labels = [
            mlb.classes_[i]
            for i, value in enumerate(row)
            if value == 1
        ]

        predicted_labels.append(
            ",".join(labels)
        )

    output[
        "actual_techniques"
    ] = actual_labels

    output[
        "predicted_techniques"
    ] = predicted_labels

    # --------------------------------------------------------
    # Maximum confidence
    # --------------------------------------------------------

    output[
        "max_confidence"
    ] = probabilities.max(
        axis=1
    )

    # --------------------------------------------------------
    # Number of predicted techniques
    # --------------------------------------------------------

    output[
        "predicted_technique_count"
    ] = predictions.sum(
        axis=1
    )

    # --------------------------------------------------------
    # Individual probabilities
    # --------------------------------------------------------

    for i, technique in enumerate(
        mlb.classes_
    ):

        safe_name = (
            technique
            .replace(".", "_")
        )

        output[
            f"prob_{safe_name}"
        ] = probabilities[:, i]

        output[
            f"pred_{safe_name}"
        ] = predictions[:, i]

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )

    return output


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(
    model,
    encoder,
    scaler,
    vectorizer,
    mlb,
    numerical_columns,
    structured_feature_count,
    final_feature_count,
):

    print_header(
        "Saving model artifacts"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Categorical encoder
    # --------------------------------------------------------

    joblib.dump(
        encoder,
        ENCODER_FILE,
    )

    # --------------------------------------------------------
    # Numerical scaler
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    joblib.dump(
        vectorizer,
        TFIDF_FILE,
    )

    # --------------------------------------------------------
    # Label encoder
    # --------------------------------------------------------

    joblib.dump(
        mlb,
        LABEL_ENCODER_FILE,
    )

    # --------------------------------------------------------
    # Feature schema
    # --------------------------------------------------------

    feature_schema = {
        "model_version": "windows_apt_v4_corrected",
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": numerical_columns,
        "categorical_feature_count": int(
            len(
                encoder.get_feature_names_out(
                    CATEGORICAL_COLUMNS
                )
            )
        ),
        "numerical_feature_count": int(
            len(numerical_columns)
        ),
        "structured_feature_count": int(
            structured_feature_count
        ),
        "tfidf_feature_count": int(
            len(vectorizer.vocabulary_)
        ),
        "final_feature_count": int(
            final_feature_count
        ),
        "mitre_techniques": [
            str(x)
            for x in mlb.classes_
        ],
        "threshold": DEFAULT_THRESHOLD,
        "categorical_encoding": (
            "OneHotEncoder"
        ),
        "numerical_scaling": (
            "StandardScaler_with_mean_false"
        ),
        "text_encoding": (
            "TF-IDF_unigram_bigram"
        ),
        "classifier": (
            "OneVsRestClassifier_"
            "LogisticRegression"
        ),
        "class_weight": None,
        "random_state": RANDOM_STATE,
    }

    with open(
        FEATURE_SCHEMA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_schema,
            f,
            indent=2,
        )

    print()
    print("Artifacts created:")

    print(
        f"Model:\n{MODEL_FILE}"
    )

    print(
        f"Structured encoder:\n"
        f"{ENCODER_FILE}"
    )

    print(
        f"Numerical scaler:\n"
        f"{SCALER_FILE}"
    )

    print(
        f"TF-IDF:\n{TFIDF_FILE}"
    )

    print(
        f"Label encoder:\n"
        f"{LABEL_ENCODER_FILE}"
    )

    print(
        f"Feature schema:\n"
        f"{FEATURE_SCHEMA_FILE}"
    )

    return feature_schema


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "SmartSOC Windows-APT Model B v4 - Corrected Training"
    )

    check_required_files()

    (
        X_train_df,
        X_test_df,
        text_train_df,
        text_test_df,
        y_train_df,
        y_test_df,
        meta_train,
        meta_test,
    ) = load_datasets()

    # --------------------------------------------------------
    # Label preparation
    # --------------------------------------------------------

    (
        mlb,
        train_labels,
        test_labels,
        Y_train,
        Y_test,
    ) = prepare_labels(
        y_train_df,
        y_test_df,
    )

    # --------------------------------------------------------
    # Feature detection
    # --------------------------------------------------------

    numerical_columns = detect_feature_columns(
        X_train_df,
        X_test_df,
    )

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    (
        structured_train,
        structured_test,
        encoder,
        scaler,
    ) = prepare_structured_features(
        X_train_df,
        X_test_df,
        numerical_columns,
    )

    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    (
        text_train,
        text_test,
        vectorizer,
    ) = prepare_text_features(
        text_train_df,
        text_test_df,
    )

    # --------------------------------------------------------
    # Hybrid matrix
    # --------------------------------------------------------

    (
        X_train,
        X_test,
    ) = combine_features(
        structured_train,
        structured_test,
        text_train,
        text_test,
    )

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    print()
    print(
        "FINAL FEATURE SANITY CHECK"
    )

    print(
        f"Categorical encoded features : "
        f"{structured_train.shape[1] - len(numerical_columns)}"
    )

    print(
        f"Numerical features           : "
        f"{len(numerical_columns)}"
    )

    print(
        f"TF-IDF features              : "
        f"{text_train.shape[1]}"
    )

    print(
        f"Final features               : "
        f"{X_train.shape[1]}"
    )

    if len(numerical_columns) == 0:
        raise RuntimeError(
            "SAFETY STOP: numerical feature count is zero."
        )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        Y_train,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    (
        probabilities,
        predictions,
    ) = generate_predictions(
        model,
        X_test,
        mlb,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        report,
        metrics,
    ) = evaluate_model(
        Y_test,
        predictions,
        mlb,
    )

    # --------------------------------------------------------
    # Analyze predictions
    # --------------------------------------------------------

    distribution_df = (
        analyze_prediction_distribution(
            Y_test,
            predictions,
            probabilities,
            mlb,
        )
    )

    # --------------------------------------------------------
    # Save prediction CSV
    # --------------------------------------------------------

    save_predictions(
        probabilities,
        predictions,
        Y_test,
        mlb,
        meta_test,
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SmartSOC Windows-APT "
            "Model B v4 Corrected\n"
        )

        f.write(
            "====================================\n\n"
        )

        f.write(
            "Training configuration\n"
        )

        f.write(
            "------------------------------------\n"
        )

        f.write(
            f"Training events: "
            f"{len(X_train_df)}\n"
        )

        f.write(
            f"Testing events: "
            f"{len(X_test_df)}\n"
        )

        f.write(
            f"Categorical features: "
            f"{len(CATEGORICAL_COLUMNS)}\n"
        )

        f.write(
            f"Numerical features: "
            f"{len(numerical_columns)}\n"
        )

        f.write(
            f"TF-IDF features: "
            f"{text_train.shape[1]}\n"
        )

        f.write(
            f"Final features: "
            f"{X_train.shape[1]}\n"
        )

        f.write(
            f"MITRE techniques: "
            f"{len(mlb.classes_)}\n"
        )

        f.write(
            f"Threshold: "
            f"{DEFAULT_THRESHOLD}\n"
        )

        f.write(
            "Class weighting: None\n"
        )

        f.write(
            "\nOverall metrics\n"
        )

        f.write(
            "------------------------------------\n"
        )

        for key, value in metrics.items():

            f.write(
                f"{key}: {value:.6f}\n"
            )

        f.write(
            "\nClassification report\n"
        )

        f.write(
            "------------------------------------\n"
        )

        f.write(
            report
        )

        f.write(
            "\n\nPrediction distribution\n"
        )

        f.write(
            "------------------------------------\n"
        )

        f.write(
            distribution_df.to_string(
                index=False
            )
        )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_FILE}"
    )

    # --------------------------------------------------------
    # Save feature information
    # --------------------------------------------------------

    feature_information = {
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numerical_columns": numerical_columns,
        "categorical_count": len(
            encoder.get_feature_names_out(
                CATEGORICAL_COLUMNS
            )
        ),
        "numerical_count": len(
            numerical_columns
        ),
        "structured_count": int(
            structured_train.shape[1]
        ),
        "tfidf_count": int(
            text_train.shape[1]
        ),
        "final_count": int(
            X_train.shape[1]
        ),
        "mitre_class_count": int(
            len(mlb.classes_)
        ),
        "mitre_classes": [
            str(x)
            for x in mlb.classes_
        ],
    }

    with open(
        FEATURE_INFO_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_information,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save model artifacts
    # --------------------------------------------------------

    feature_schema = save_artifacts(
        model=model,
        encoder=encoder,
        scaler=scaler,
        vectorizer=vectorizer,
        mlb=mlb,
        numerical_columns=numerical_columns,
        structured_feature_count=structured_train.shape[1],
        final_feature_count=X_train.shape[1],
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "SmartSOC Windows-APT v4 Corrected Training Complete"
    )

    print()
    print(
        f"Training events : "
        f"{len(X_train_df)}"
    )

    print(
        f"Testing events  : "
        f"{len(X_test_df)}"
    )

    print(
        f"Categorical features : "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    print(
        f"Numerical features   : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Structured features  : "
        f"{structured_train.shape[1]}"
    )

    print(
        f"TF-IDF features      : "
        f"{text_train.shape[1]}"
    )

    print(
        f"Final features       : "
        f"{X_train.shape[1]}"
    )

    print(
        f"MITRE techniques     : "
        f"{len(mlb.classes_)}"
    )

    print()

    print(
        f"Micro F1 : "
        f"{metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a BASELINE model."
    )

    print(
        "Do NOT deploy it yet."
    )

    print(
        "Next step is threshold calibration."
    )


if __name__ == "__main__":
    main()