"""
===============================================================================
SmartSOC Windows-APT Model B v4.1
Corrected Hybrid Training Pipeline
===============================================================================

Dataset:
    data/processed/windows_apt_v4_1/

Inputs:
    structured_train.csv
    structured_test.csv
    behavior_text_train.csv
    behavior_text_test.csv
    labels_train.csv
    labels_test.csv
    metadata_train.csv
    metadata_test.csv
    schema.json

Pipeline:
    Structured categorical features
        ->
    OneHotEncoder(handle_unknown="ignore")

    Structured numerical features
        ->
    StandardScaler(with_mean=False)

    Behavior text
        ->
    TF-IDF fitted on TRAINING DATA ONLY

    All features
        ->
    Sparse hybrid matrix

    Hybrid matrix
        ->
    One-vs-Rest Logistic Regression

    Evaluation
        ->
    Predictions
        ->
    Model artifacts

IMPORTANT:
    - Test data is NEVER used to fit preprocessing.
    - Temporal train/test split is preserved.
    - No threshold calibration is performed here.
    - Calibration is the next stage.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

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

warnings.filterwarnings("ignore")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "windows_apt_v4_1"
)


# Input files
STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

TEXT_TRAIN = DATA_DIR / "behavior_text_train.csv"
TEXT_TEST = DATA_DIR / "behavior_text_test.csv"

LABEL_TRAIN = DATA_DIR / "labels_train.csv"
LABEL_TEST = DATA_DIR / "labels_test.csv"

METADATA_TRAIN = DATA_DIR / "metadata_train.csv"
METADATA_TEST = DATA_DIR / "metadata_test.csv"

SCHEMA_FILE = DATA_DIR / "schema.json"


# Output files
MODEL_FILE = MODEL_DIR / "multilabel_logistic_v4_1.joblib"
STRUCTURED_ENCODER_FILE = MODEL_DIR / "structured_encoder_v4_1.joblib"
NUMERICAL_SCALER_FILE = MODEL_DIR / "numerical_scaler_v4_1.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer_v4_1.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder_v4_1.joblib"
FEATURE_SCHEMA_FILE = MODEL_DIR / "feature_schema_v4_1.json"

REPORT_FILE = RESULT_DIR / "classification_report.txt"
PREDICTIONS_FILE = RESULT_DIR / "predictions.csv"
METRICS_FILE = RESULT_DIR / "metrics.json"


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def header(title: str):
    print()
    print("=" * 79)
    print(title)
    print("=" * 79)


def section(title: str):
    print()
    print("-" * 79)
    print(title)
    print("-" * 79)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print(f"[OK] {path}")


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def normalize_text(value) -> str:
    """
    Normalize behavior text safely.

    Handles:
        NaN
        None
        numbers
        strings
    """
    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text.lower().strip()


def find_single_column(df: pd.DataFrame, preferred_names):
    """
    Return a single appropriate column.

    Useful because labels/text files can sometimes have slightly
    different column names.
    """

    columns_lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in preferred_names:
        key = name.strip().lower()

        if key in columns_lower:
            return columns_lower[key]

    if len(df.columns) == 1:
        return df.columns[0]

    raise ValueError(
        f"Could not determine required column.\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_label_string(value) -> str:
    """
    Normalize a MITRE multilabel string.

    Example:
        ' T1105, T1059.003 '
    becomes:
        'T1059.003,T1105'
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    parts = [
        p.strip()
        for p in text.split(",")
        if p.strip()
    ]

    # Remove duplicates while preserving deterministic order.
    parts = sorted(set(parts))

    return ",".join(parts)


def split_labels(value):
    """
    Convert:
        T1105,T1059.003
    into:
        ['T1059.003', 'T1105']
    """

    if not value:
        return []

    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


# =============================================================================
# SCHEMA
# =============================================================================

def load_schema():
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return schema


def determine_feature_columns(structured_train: pd.DataFrame, schema):
    """
    Determine categorical and numerical columns.

    v4.1 expected:

        categorical = 4
        numerical   = 58
        total        = 62
    """

    # ---------------------------------------------------------
    # Preferred schema locations
    # ---------------------------------------------------------

    categorical = schema.get("categorical_columns")

    numerical = schema.get("numerical_columns")

    # Some schema versions may use nested structure.
    if categorical is None:
        categorical = (
            schema.get("features", {})
            .get("categorical_columns")
        )

    if numerical is None:
        numerical = (
            schema.get("features", {})
            .get("numerical_columns")
        )

    # ---------------------------------------------------------
    # v4.1 known categorical features
    # ---------------------------------------------------------

    if not categorical:
        categorical = [
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
        ]

    # ---------------------------------------------------------
    # Fallback numerical detection
    # ---------------------------------------------------------

    if not numerical:
        numerical = [
            c
            for c in structured_train.columns
            if c not in categorical
        ]

    categorical = [
        c for c in categorical
        if c in structured_train.columns
    ]

    numerical = [
        c for c in numerical
        if c in structured_train.columns
    ]

    return categorical, numerical


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1 - Hybrid Training"
    )

    # =========================================================================
    # CREATE OUTPUT DIRECTORIES
    # =========================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =========================================================================
    # CHECK FILES
    # =========================================================================

    section("Checking required artifacts...")

    required_files = [
        STRUCTURED_TRAIN,
        STRUCTURED_TEST,
        TEXT_TRAIN,
        TEXT_TEST,
        LABEL_TRAIN,
        LABEL_TEST,
        METADATA_TRAIN,
        METADATA_TEST,
        SCHEMA_FILE,
    ]

    for path in required_files:
        require_file(path)

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    header("Loading v4.1 dataset")

    structured_train = pd.read_csv(
        STRUCTURED_TRAIN
    )

    structured_test = pd.read_csv(
        STRUCTURED_TEST
    )

    text_train_df = pd.read_csv(
        TEXT_TRAIN
    )

    text_test_df = pd.read_csv(
        TEXT_TEST
    )

    labels_train_df = pd.read_csv(
        LABEL_TRAIN
    )

    labels_test_df = pd.read_csv(
        LABEL_TEST
    )

    metadata_train = pd.read_csv(
        METADATA_TRAIN
    )

    metadata_test = pd.read_csv(
        METADATA_TEST
    )

    schema = load_schema()

    print(
        f"Structured train: {structured_train.shape}"
    )

    print(
        f"Structured test : {structured_test.shape}"
    )

    print(
        f"Text train      : {text_train_df.shape}"
    )

    print(
        f"Text test       : {text_test_df.shape}"
    )

    print(
        f"Labels train    : {labels_train_df.shape}"
    )

    print(
        f"Labels test     : {labels_test_df.shape}"
    )

    print(
        f"Metadata train  : {metadata_train.shape}"
    )

    print(
        f"Metadata test   : {metadata_test.shape}"
    )

    # =========================================================================
    # ROW ALIGNMENT
    # =========================================================================

    header("Checking row alignment")

    train_lengths = {
        "structured": len(structured_train),
        "text": len(text_train_df),
        "labels": len(labels_train_df),
        "metadata": len(metadata_train),
    }

    test_lengths = {
        "structured": len(structured_test),
        "text": len(text_test_df),
        "labels": len(labels_test_df),
        "metadata": len(metadata_test),
    }

    print("Training:")
    for name, count in train_lengths.items():
        print(f"  {name:<12}: {count}")

    print()
    print("Testing:")
    for name, count in test_lengths.items():
        print(f"  {name:<12}: {count}")

    if len(set(train_lengths.values())) != 1:
        raise ValueError(
            "Training datasets are not row-aligned."
        )

    if len(set(test_lengths.values())) != 1:
        raise ValueError(
            "Testing datasets are not row-aligned."
        )

    print()
    print(
        f"[OK] Training aligned: {len(structured_train)}"
    )

    print(
        f"[OK] Testing aligned : {len(structured_test)}"
    )

    # =========================================================================
    # STRUCTURED SCHEMA
    # =========================================================================

    header("Inspecting structured feature schema")

    if list(structured_train.columns) != list(
        structured_test.columns
    ):
        raise ValueError(
            "Training and testing structured schemas do not match."
        )

    categorical_columns, numerical_columns = (
        determine_feature_columns(
            structured_train,
            schema
        )
    )

    print(
        f"Structured feature count: "
        f"{len(structured_train.columns)}"
    )

    print()
    print(
        f"Categorical features: {len(categorical_columns)}"
    )

    for col in categorical_columns:
        print(f"  - {col}")

    print()
    print(
        f"Numerical features: {len(numerical_columns)}"
    )

    for col in numerical_columns:
        print(f"  - {col}")

    expected_total = (
        len(categorical_columns)
        + len(numerical_columns)
    )

    if expected_total != len(structured_train.columns):
        raise ValueError(
            "Categorical + numerical feature counts do not "
            "match structured feature count."
        )

    # =========================================================================
    # CLEAN CATEGORICAL FEATURES
    # =========================================================================

    section("Preparing categorical features")

    X_cat_train = structured_train[
        categorical_columns
    ].copy()

    X_cat_test = structured_test[
        categorical_columns
    ].copy()

    for col in categorical_columns:

        X_cat_train[col] = (
            X_cat_train[col]
            .fillna("<missing>")
            .astype(str)
            .str.strip()
            .replace("", "<missing>")
        )

        X_cat_test[col] = (
            X_cat_test[col]
            .fillna("<missing>")
            .astype(str)
            .str.strip()
            .replace("", "<missing>")
        )

    print(
        f"Categorical training matrix source: "
        f"{X_cat_train.shape}"
    )

    # =========================================================================
    # ONE-HOT ENCODING
    # =========================================================================

    print()
    print(
        "Fitting OneHotEncoder on TRAINING DATA ONLY..."
    )

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float64,
        )
    except TypeError:
        # Compatibility with older scikit-learn
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float64,
        )

    X_cat_train_encoded = encoder.fit_transform(
        X_cat_train
    )

    X_cat_test_encoded = encoder.transform(
        X_cat_test
    )

    print(
        f"Categorical train: "
        f"{X_cat_train_encoded.shape}"
    )

    print(
        f"Categorical test : "
        f"{X_cat_test_encoded.shape}"
    )

    # =========================================================================
    # NUMERICAL FEATURES
    # =========================================================================

    section("Preparing numerical features")

    X_num_train = structured_train[
        numerical_columns
    ].copy()

    X_num_test = structured_test[
        numerical_columns
    ].copy()

    # Convert everything explicitly to numeric.
    for col in numerical_columns:

        X_num_train[col] = pd.to_numeric(
            X_num_train[col],
            errors="coerce"
        )

        X_num_test[col] = pd.to_numeric(
            X_num_test[col],
            errors="coerce"
        )

    # Fill missing values using TRAINING medians only.
    train_medians = X_num_train.median()

    X_num_train = X_num_train.fillna(
        train_medians
    )

    X_num_test = X_num_test.fillna(
        train_medians
    )

    # Replace infinite values.
    X_num_train = X_num_train.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_num_test = X_num_test.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_num_train = X_num_train.fillna(
        train_medians
    )

    X_num_test = X_num_test.fillna(
        train_medians
    )

    # Convert to numpy.
    X_num_train_np = X_num_train.to_numpy(
        dtype=np.float64
    )

    X_num_test_np = X_num_test.to_numpy(
        dtype=np.float64
    )

    # =========================================================================
    # SCALE NUMERICAL FEATURES
    # =========================================================================

    print(
        "Fitting numerical scaler on TRAINING DATA ONLY..."
    )

    # with_mean=False is required because the final feature
    # representation is sparse.
    scaler = StandardScaler(
        with_mean=False
    )

    X_num_train_scaled = scaler.fit_transform(
        X_num_train_np
    )

    X_num_test_scaled = scaler.transform(
        X_num_test_np
    )

    # Convert numerical matrices to sparse.
    X_num_train_sparse = csr_matrix(
        X_num_train_scaled
    )

    X_num_test_sparse = csr_matrix(
        X_num_test_scaled
    )

    print(
        f"Numerical train: "
        f"{X_num_train_sparse.shape}"
    )

    print(
        f"Numerical test : "
        f"{X_num_test_sparse.shape}"
    )

    # =========================================================================
    # STRUCTURED MATRIX
    # =========================================================================

    section("Combining structured features")

    X_struct_train = hstack(
        [
            X_cat_train_encoded,
            X_num_train_sparse,
        ],
        format="csr",
    )

    X_struct_test = hstack(
        [
            X_cat_test_encoded,
            X_num_test_sparse,
        ],
        format="csr",
    )

    print(
        f"Structured train: "
        f"{X_struct_train.shape}"
    )

    print(
        f"Structured test : "
        f"{X_struct_test.shape}"
    )

    # =========================================================================
    # BEHAVIOR TEXT
    # =========================================================================

    header("Preparing behavioral text")

    text_train_column = find_single_column(
        text_train_df,
        [
            "behavior_text",
            "text",
            "behavior",
            "command_line",
        ],
    )

    text_test_column = find_single_column(
        text_test_df,
        [
            "behavior_text",
            "text",
            "behavior",
            "command_line",
        ],
    )

    print(
        f"Training text column: {text_train_column}"
    )

    print(
        f"Testing text column : {text_test_column}"
    )

    behavior_train = (
        text_train_df[text_train_column]
        .apply(normalize_text)
        .tolist()
    )

    behavior_test = (
        text_test_df[text_test_column]
        .apply(normalize_text)
        .tolist()
    )

    empty_train = sum(
        1 for x in behavior_train
        if not x
    )

    empty_test = sum(
        1 for x in behavior_test
        if not x
    )

    print(
        f"Empty training behavior: "
        f"{empty_train} / {len(behavior_train)}"
    )

    print(
        f"Empty testing behavior : "
        f"{empty_test} / {len(behavior_test)}"
    )

    # =========================================================================
    # TF-IDF
    # =========================================================================

    print()
    print(
        "Fitting TF-IDF on TRAINING DATA ONLY..."
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20000,
        token_pattern=r"(?u)\b[\w./:\\=-]{2,}\b",
    )

    X_text_train = tfidf.fit_transform(
        behavior_train
    )

    X_text_test = tfidf.transform(
        behavior_test
    )

    print(
        f"TF-IDF vocabulary: "
        f"{len(tfidf.vocabulary_)}"
    )

    print(
        f"Text train: {X_text_train.shape}"
    )

    print(
        f"Text test : {X_text_test.shape}"
    )

    # =========================================================================
    # FINAL HYBRID MATRIX
    # =========================================================================

    header("Building hybrid feature matrix")

    X_train = hstack(
        [
            X_struct_train,
            X_text_train,
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

    # Explicitly make matrices writable and canonical.
    X_train = csr_matrix(
        X_train,
        dtype=np.float64,
        copy=True,
    )

    X_test = csr_matrix(
        X_test,
        dtype=np.float64,
        copy=True,
    )

    X_train.sum_duplicates()
    X_train.sort_indices()

    X_test.sum_duplicates()
    X_test.sort_indices()

    print(
        f"Final train matrix: {X_train.shape}"
    )

    print(
        f"Final test matrix : {X_test.shape}"
    )

    print(
        f"Total model features: {X_train.shape[1]}"
    )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and testing feature dimensions differ."
        )

    # =========================================================================
    # LABEL PREPARATION
    # =========================================================================

    header("Preparing MITRE multilabel targets")

    label_train_column = find_single_column(
        labels_train_df,
        [
            "techniques",
            "label",
            "labels",
            "mitre_techniques",
        ],
    )

    label_test_column = find_single_column(
        labels_test_df,
        [
            "techniques",
            "label",
            "labels",
            "mitre_techniques",
        ],
    )

    y_text_train = (
        labels_train_df[label_train_column]
        .apply(normalize_label_string)
        .tolist()
    )

    y_text_test = (
        labels_test_df[label_test_column]
        .apply(normalize_label_string)
        .tolist()
    )

    # =========================================================================
    # DISCOVER ALL MITRE TECHNIQUES
    # =========================================================================

    all_techniques = set()

    for value in y_text_train:
        all_techniques.update(
            split_labels(value)
        )

    for value in y_text_test:
        all_techniques.update(
            split_labels(value)
        )

    techniques = sorted(
        all_techniques
    )

    print(
        f"MITRE techniques: {len(techniques)}"
    )

    print(techniques)

    if len(techniques) != 45:
        print()
        print(
            f"[WARNING] Expected 45 techniques, "
            f"found {len(techniques)}."
        )

    # =========================================================================
    # MULTILABEL ENCODER
    # =========================================================================

    label_encoder = MultiLabelBinarizer(
        classes=techniques
    )

    y_train = label_encoder.fit_transform(
        [
            split_labels(x)
            for x in y_text_train
        ]
    )

    y_test = label_encoder.transform(
        [
            split_labels(x)
            for x in y_text_test
        ]
    )

    print(
        f"Y train: {y_train.shape}"
    )

    print(
        f"Y test : {y_test.shape}"
    )

    # =========================================================================
    # TRAINING CLASS COVERAGE
    # =========================================================================

    train_counts = y_train.sum(axis=0)

    print()
    print("Training label coverage:")

    for technique, count in zip(
        label_encoder.classes_,
        train_counts,
    ):
        print(
            f"{technique:<12} {int(count)}"
        )

    missing_from_training = [
        technique
        for technique, count in zip(
            label_encoder.classes_,
            train_counts,
        )
        if count == 0
    ]

    if missing_from_training:
        raise ValueError(
            "The following MITRE techniques have "
            "zero training examples:\n"
            + "\n".join(
                missing_from_training
            )
        )

    print()
    print(
        "[OK] Every technique has training examples."
    )

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================

    header(
        "Training One-vs-Rest Logistic Regression"
    )

    print(
        "Model: OneVsRestClassifier"
    )

    print(
        "Base estimator: LogisticRegression"
    )

    print(
        "Solver: liblinear"
    )

    print(
        "n_jobs: 1"
    )

    print(
        "class_weight: None"
    )

    print()
    print(
        "This is the v4.1 corrected hybrid baseline."
    )

    print(
        "Structured features:"
    )

    print(
        "  - 4 categorical features"
    )

    print(
        "  - 58 numerical features"
    )

    print(
        "Behavior text:"
    )

    print(
        "  - TF-IDF"
    )

    print(
        "Temporal integrity:"
    )

    print(
        "  - preprocessing fitted on training only"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # We deliberately do NOT use class_weight="balanced" here.
    #
    # Reason:
    # The v4.1 dataset contains extreme class imbalance.
    # We want this run to be a clean baseline first.
    # Class balancing can be tested later as a separate experiment.
    # -------------------------------------------------------------------------

    base_model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=2000,
        class_weight=None,
        random_state=42,
    )

    model = OneVsRestClassifier(
        estimator=base_model,
        n_jobs=1,
    )

    print()
    print(
        "Starting model.fit()..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print()
    print(
        "[OK] Training complete."
    )

    # =========================================================================
    # PREDICTIONS
    # =========================================================================

    header("Generating test predictions")

    print(
        "Generating probability scores..."
    )

    probabilities = model.predict_proba(
        X_test
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    # Standard baseline threshold.
    threshold = 0.50

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    # =========================================================================
    # FALLBACK FOR EMPTY PREDICTIONS
    # =========================================================================
    #
    # If no technique crosses 0.5, choose the highest-probability technique.
    #
    # This prevents an event from receiving zero MITRE techniques solely
    # because every score is below the threshold.
    # =========================================================================

    empty_prediction_rows = (
        y_pred.sum(axis=1) == 0
    )

    empty_count = int(
        empty_prediction_rows.sum()
    )

    print(
        f"Events with no prediction before fallback: "
        f"{empty_count}"
    )

    if empty_count > 0:

        best_indices = np.argmax(
            probabilities[
                empty_prediction_rows
            ],
            axis=1,
        )

        y_pred[
            empty_prediction_rows
        ] = 0

        rows = np.where(
            empty_prediction_rows
        )[0]

        for row, class_index in zip(
            rows,
            best_indices,
        ):
            y_pred[
                row,
                class_index
            ] = 1

    # =========================================================================
    # METRICS
    # =========================================================================

    header(
        "v4.1 Test Evaluation"
    )

    micro_precision = precision_score(
        y_test,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_test,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_test,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print(
        f"Micro Precision   : {micro_precision:.4f}"
    )

    print(
        f"Micro Recall      : {micro_recall:.4f}"
    )

    print(
        f"Micro F1          : {micro_f1:.4f}"
    )

    print()

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print()

    print(
        f"Weighted Precision: {weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall   : {weighted_recall:.4f}"
    )

    print(
        f"Weighted F1       : {weighted_f1:.4f}"
    )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print()
    print(
        "Classification Report:"
    )
    print()
    print(report)

    # =========================================================================
    # CONFIDENCE ANALYSIS
    # =========================================================================

    # Confidence associated with selected positive predictions.
    selected_confidences = []

    for row in range(
        probabilities.shape[0]
    ):

        selected = np.where(
            y_pred[row] == 1
        )[0]

        if len(selected) > 0:

            selected_confidences.extend(
                probabilities[
                    row,
                    selected
                ].tolist()
            )

    if selected_confidences:

        mean_confidence = float(
            np.mean(
                selected_confidences
            )
        )

        median_confidence = float(
            np.median(
                selected_confidences
            )
        )

        min_confidence = float(
            np.min(
                selected_confidences
            )
        )

        max_confidence = float(
            np.max(
                selected_confidences
            )
        )

    else:

        mean_confidence = 0.0
        median_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0

    print(
        f"Mean selected confidence   : "
        f"{mean_confidence:.4f}"
    )

    print(
        f"Median selected confidence : "
        f"{median_confidence:.4f}"
    )

    print(
        f"Minimum selected confidence: "
        f"{min_confidence:.4f}"
    )

    print(
        f"Maximum selected confidence: "
        f"{max_confidence:.4f}"
    )

    # =========================================================================
    # PER-TECHNIQUE METRICS
    # =========================================================================

    technique_metrics = {}

    for index, technique in enumerate(
        label_encoder.classes_
    ):

        precision = precision_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        recall = recall_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        f1 = f1_score(
            y_test[:, index],
            y_pred[:, index],
            zero_division=0,
        )

        support = int(
            y_test[:, index].sum()
        )

        technique_metrics[
            technique
        ] = {
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # =========================================================================
    # SAVE REPORT
    # =========================================================================

    section("Saving evaluation report")

    report_text = (
        "SmartSOC Windows-APT Model B v4.1\n"
        "Corrected Hybrid Baseline\n"
        "\n"
        "============================================================\n"
        "DATASET\n"
        "============================================================\n"
        f"Training events: {len(structured_train)}\n"
        f"Testing events : {len(structured_test)}\n"
        f"Structured features: {len(structured_train.columns)}\n"
        f"Categorical features: {len(categorical_columns)}\n"
        f"Numerical features: {len(numerical_columns)}\n"
        f"TF-IDF features: {len(tfidf.vocabulary_)}\n"
        f"Final model features: {X_train.shape[1]}\n"
        f"MITRE techniques: {len(label_encoder.classes_)}\n"
        "\n"
        "============================================================\n"
        "MODEL\n"
        "============================================================\n"
        "OneVsRestClassifier\n"
        "LogisticRegression\n"
        "solver=liblinear\n"
        "C=1.0\n"
        "class_weight=None\n"
        "n_jobs=1\n"
        "\n"
        "============================================================\n"
        "METRICS\n"
        "============================================================\n"
        f"Micro Precision: {micro_precision:.6f}\n"
        f"Micro Recall: {micro_recall:.6f}\n"
        f"Micro F1: {micro_f1:.6f}\n"
        f"Macro Precision: {macro_precision:.6f}\n"
        f"Macro Recall: {macro_recall:.6f}\n"
        f"Macro F1: {macro_f1:.6f}\n"
        f"Weighted Precision: {weighted_precision:.6f}\n"
        f"Weighted Recall: {weighted_recall:.6f}\n"
        f"Weighted F1: {weighted_f1:.6f}\n"
        "\n"
        "============================================================\n"
        "CONFIDENCE\n"
        "============================================================\n"
        f"Mean: {mean_confidence:.6f}\n"
        f"Median: {median_confidence:.6f}\n"
        f"Minimum: {min_confidence:.6f}\n"
        f"Maximum: {max_confidence:.6f}\n"
        "\n"
        "============================================================\n"
        "CLASSIFICATION REPORT\n"
        "============================================================\n"
        f"{report}\n"
        "\n"
        "============================================================\n"
        "PER-TECHNIQUE METRICS\n"
        "============================================================\n"
    )

    for technique in label_encoder.classes_:

        metrics = technique_metrics[
            technique
        ]

        report_text += (
            f"{technique:<12} "
            f"support={metrics['support']:<6} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}\n"
        )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report_text)

    print(
        f"[OK] {REPORT_FILE}"
    )

    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    section("Saving prediction results")

    prediction_rows = []

    for row in range(
        len(structured_test)
    ):

        predicted_techniques = [
            label_encoder.classes_[i]
            for i in np.where(
                y_pred[row] == 1
            )[0]
        ]

        prediction_rows.append(
            ",".join(
                predicted_techniques
            )
        )

    prediction_df = pd.DataFrame(
        {
            "row_index": np.arange(
                len(structured_test)
            ),
            "predicted_techniques":
                prediction_rows,
        }
    )

    # Add individual probabilities.
    for index, technique in enumerate(
        label_encoder.classes_
    ):

        prediction_df[
            f"prob_{technique}"
        ] = probabilities[
            :,
            index
        ]

    # Add original labels for analysis.
    prediction_df[
        "actual_techniques"
    ] = y_text_test

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"[OK] {PREDICTIONS_FILE}"
    )

    # =========================================================================
    # SAVE METRICS JSON
    # =========================================================================

    metrics_payload = {
        "model": (
            "SmartSOC Windows-APT Model B v4.1"
        ),
        "dataset": "windows_apt_v4_1",
        "training_events": int(
            len(structured_train)
        ),
        "testing_events": int(
            len(structured_test)
        ),
        "structured_features": int(
            len(structured_train.columns)
        ),
        "categorical_features": int(
            len(categorical_columns)
        ),
        "numerical_features": int(
            len(numerical_columns)
        ),
        "tfidf_features": int(
            len(tfidf.vocabulary_)
        ),
        "final_features": int(
            X_train.shape[1]
        ),
        "mitre_techniques": int(
            len(label_encoder.classes_)
        ),
        "threshold": threshold,
        "empty_prediction_rows_before_fallback":
            empty_count,
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
        "mean_confidence": mean_confidence,
        "median_confidence": median_confidence,
        "minimum_confidence": min_confidence,
        "maximum_confidence": max_confidence,
        "technique_metrics":
            technique_metrics,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_payload,
            f,
            indent=2,
        )

    print(
        f"[OK] {METRICS_FILE}"
    )

    # =========================================================================
    # SAVE MODEL ARTIFACTS
    # =========================================================================

    header("Saving model artifacts")

    # Model
    joblib.dump(
        model,
        MODEL_FILE,
        compress=3,
    )

    print(
        f"[OK] Model: {MODEL_FILE}"
    )

    # Structured encoder
    joblib.dump(
        encoder,
        STRUCTURED_ENCODER_FILE,
        compress=3,
    )

    print(
        f"[OK] Structured encoder: "
        f"{STRUCTURED_ENCODER_FILE}"
    )

    # Numerical scaler
    joblib.dump(
        scaler,
        NUMERICAL_SCALER_FILE,
        compress=3,
    )

    print(
        f"[OK] Numerical scaler: "
        f"{NUMERICAL_SCALER_FILE}"
    )

    # TF-IDF
    joblib.dump(
        tfidf,
        TFIDF_FILE,
        compress=3,
    )

    print(
        f"[OK] TF-IDF: {TFIDF_FILE}"
    )

    # Label encoder
    joblib.dump(
        label_encoder,
        LABEL_ENCODER_FILE,
        compress=3,
    )

    print(
        f"[OK] Label encoder: "
        f"{LABEL_ENCODER_FILE}"
    )

    # =========================================================================
    # FEATURE SCHEMA
    # =========================================================================

    categorical_encoded_count = int(
        X_cat_train_encoded.shape[1]
    )

    numerical_count = int(
        X_num_train_sparse.shape[1]
    )

    tfidf_count = int(
        X_text_train.shape[1]
    )

    final_feature_count = int(
        X_train.shape[1]
    )

    feature_schema = {
        "model": (
            "SmartSOC Windows-APT Model B v4.1"
        ),
        "dataset": "windows_apt_v4_1",
        "categorical_columns":
            categorical_columns,
        "numerical_columns":
            numerical_columns,
        "categorical_input_feature_count":
            len(categorical_columns),
        "categorical_encoded_feature_count":
            categorical_encoded_count,
        "numerical_feature_count":
            numerical_count,
        "tfidf_feature_count":
            tfidf_count,
        "final_feature_count":
            final_feature_count,
        "mitre_techniques":
            label_encoder.classes_.tolist(),
        "tfidf_configuration": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "sublinear_tf": True,
            "max_features": 20000,
        },
        "model_configuration": {
            "classifier":
                "OneVsRestClassifier",
            "base_estimator":
                "LogisticRegression",
            "solver":
                "liblinear",
            "C":
                1.0,
            "max_iter":
                2000,
            "class_weight":
                None,
            "n_jobs":
                1,
        },
        "prediction_configuration": {
            "default_threshold":
                threshold,
            "empty_prediction_fallback":
                "highest_probability",
        },
        "training_only_fitted": [
            "OneHotEncoder",
            "StandardScaler",
            "TfidfVectorizer",
            "LogisticRegression",
        ],
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

    print(
        f"[OK] Feature schema: "
        f"{FEATURE_SCHEMA_FILE}"
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    header(
        "SmartSOC Windows-APT Model B v4.1 "
        "Training Complete"
    )

    print(
        f"Training events        : "
        f"{len(structured_train)}"
    )

    print(
        f"Testing events         : "
        f"{len(structured_test)}"
    )

    print(
        f"Structured features    : "
        f"{len(structured_train.columns)}"
    )

    print(
        f"Categorical input      : "
        f"{len(categorical_columns)}"
    )

    print(
        f"Categorical encoded    : "
        f"{categorical_encoded_count}"
    )

    print(
        f"Numerical features     : "
        f"{numerical_count}"
    )

    print(
        f"TF-IDF features        : "
        f"{tfidf_count}"
    )

    print(
        f"Final model features   : "
        f"{final_feature_count}"
    )

    print(
        f"MITRE techniques       : "
        f"{len(label_encoder.classes_)}"
    )

    print()
    print(
        f"Micro F1               : "
        f"{micro_f1:.4f}"
    )

    print(
        f"Macro F1               : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1            : "
        f"{weighted_f1:.4f}"
    )

    print()
    print(
        "Artifacts:"
    )

    print(
        f"  Model          : {MODEL_FILE}"
    )

    print(
        f"  Encoder        : "
        f"{STRUCTURED_ENCODER_FILE}"
    )

    print(
        f"  Scaler         : "
        f"{NUMERICAL_SCALER_FILE}"
    )

    print(
        f"  TF-IDF         : {TFIDF_FILE}"
    )

    print(
        f"  Label encoder  : "
        f"{LABEL_ENCODER_FILE}"
    )

    print(
        f"  Schema         : "
        f"{FEATURE_SCHEMA_FILE}"
    )

    print(
        f"  Report         : "
        f"{REPORT_FILE}"
    )

    print(
        f"  Predictions    : "
        f"{PREDICTIONS_FILE}"
    )

    print(
        f"  Metrics        : "
        f"{METRICS_FILE}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is an experimental v4.1 baseline."
    )

    print(
        "Do NOT deploy it yet."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the v4.1 calibration pipeline."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()