"""
SmartSOC Windows-APT Model B v4
Threshold Calibration Pipeline

Purpose:
    1. Load the prepared v4 Windows-APT dataset.
    2. Load the trained v4 hybrid model and preprocessing artifacts.
    3. Create a chronological calibration split from the v4 training set.
    4. Generate validation probabilities.
    5. Find technique-specific thresholds.
    6. Evaluate calibrated predictions on the untouched temporal test set.
    7. Save calibrated model artifacts and thresholds.

IMPORTANT:
    The original temporal test set is NEVER used for threshold fitting.

Expected timestamp format:
    Dec 1, 2024 @ 06:24:28.026
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

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

OUTPUT_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "windows_apt_v4"
    / "calibrated"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "windows_apt_v4"
    / "calibration"
)

OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# INPUT FILES
# ============================================================================

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

TEXT_TRAIN = DATA_DIR / "behavior_text_train.csv"
TEXT_TEST = DATA_DIR / "behavior_text_test.csv"

LABEL_TRAIN = DATA_DIR / "labels_train.csv"
LABEL_TEST = DATA_DIR / "labels_test.csv"

META_TRAIN = DATA_DIR / "metadata_train.csv"
META_TEST = DATA_DIR / "metadata_test.csv"

SCHEMA_FILE = DATA_DIR / "schema.json"


# ============================================================================
# MODEL ARTIFACTS
# ============================================================================

MODEL_FILE = (
    MODEL_DIR
    / "multilabel_logistic_v4_corrected.joblib"
)

STRUCTURED_ENCODER_FILE = (
    MODEL_DIR
    / "structured_encoder_v4_corrected.joblib"
)

NUMERICAL_SCALER_FILE = (
    MODEL_DIR
    / "numerical_scaler_v4_corrected.joblib"
)

TFIDF_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer_v4_corrected.joblib"
)

LABEL_ENCODER_FILE = (
    MODEL_DIR
    / "multilabel_encoder_v4_corrected.joblib"
)

FEATURE_SCHEMA_FILE = (
    MODEL_DIR
    / "feature_schema_v4_corrected.json"
)


# ============================================================================
# OUTPUT ARTIFACTS
# ============================================================================

CALIBRATED_MODEL_FILE = (
    OUTPUT_MODEL_DIR
    / "multilabel_logistic_v4_calibrated.joblib"
)

CALIBRATED_STRUCTURED_ENCODER_FILE = (
    OUTPUT_MODEL_DIR
    / "structured_encoder_v4_calibrated.joblib"
)

CALIBRATED_SCALER_FILE = (
    OUTPUT_MODEL_DIR
    / "numerical_scaler_v4_calibrated.joblib"
)

CALIBRATED_TFIDF_FILE = (
    OUTPUT_MODEL_DIR
    / "tfidf_vectorizer_v4_calibrated.joblib"
)

CALIBRATED_LABEL_ENCODER_FILE = (
    OUTPUT_MODEL_DIR
    / "multilabel_encoder_v4_calibrated.joblib"
)

CALIBRATED_SCHEMA_FILE = (
    OUTPUT_MODEL_DIR
    / "feature_schema_v4_calibrated.json"
)

THRESHOLD_FILE = (
    OUTPUT_MODEL_DIR
    / "technique_thresholds_v4.json"
)

CALIBRATION_REPORT_FILE = (
    RESULT_DIR
    / "calibration_report.txt"
)

THRESHOLD_CSV_FILE = (
    RESULT_DIR
    / "threshold_calibration.csv"
)

TEST_PREDICTIONS_FILE = (
    RESULT_DIR
    / "calibrated_test_predictions.csv"
)


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTAMP_FORMAT = "%b %d, %Y @ %H:%M:%S.%f"

VALIDATION_FRACTION = 0.20

THRESHOLD_VALUES = np.arange(
    0.05,
    0.96,
    0.05
)

MIN_VALIDATION_SUPPORT = 5

DEFAULT_THRESHOLD = 0.50


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)
    print()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


def normalize_text(series):
    """
    Normalize behavior text while preserving the semantics used
    by the training pipeline.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\\", " ", regex=True)
        .str.replace(r"[/]", " ", regex=True)
        .str.replace(r"[:]", " ", regex=True)
        .str.replace(r"[-]", " ", regex=True)
        .str.replace(r"[_]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_timestamp(series, name):
    """
    Parse the Windows-APT timestamp using the exact known format.

    Example:
        Dec 1, 2024 @ 06:24:28.026
    """

    cleaned = (
        series
        .astype(str)
        .str.strip()
    )

    parsed = pd.to_datetime(
        cleaned,
        format=TIMESTAMP_FORMAT,
        errors="coerce"
    )

    invalid = parsed.isna().sum()

    print(
        f"{name}: "
        f"{len(parsed) - invalid} valid, "
        f"{invalid} invalid"
    )

    if invalid > 0:

        print()
        print(
            f"Sample invalid timestamps in {name}:"
        )

        bad = cleaned.loc[parsed.isna()].head(10)

        for value in bad:
            print(repr(value))

        raise ValueError(
            f"Invalid timestamps found in {name} metadata."
        )

    return parsed


def convert_labels_to_matrix(
    label_df,
    label_encoder
):
    """
    Convert comma-separated MITRE techniques into
    a multi-label binary matrix.
    """

    techniques = label_df["techniques"].fillna("")

    label_lists = []

    for value in techniques:

        if not value:
            label_lists.append([])
            continue

        items = [
            x.strip()
            for x in str(value).split(",")
            if x.strip()
        ]

        label_lists.append(items)

    classes = list(label_encoder.classes_)

    class_to_index = {
        value: index
        for index, value in enumerate(classes)
    }

    y = np.zeros(
        (
            len(label_lists),
            len(classes)
        ),
        dtype=np.int8
    )

    for row_index, labels in enumerate(label_lists):

        for label in labels:

            if label in class_to_index:

                y[
                    row_index,
                    class_to_index[label]
                ] = 1

    return y


def build_structured_matrix(
    df,
    categorical_columns,
    numerical_columns,
    encoder,
    scaler
):

    # ------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------

    categorical_df = df[
        categorical_columns
    ].copy()

    categorical_df = categorical_df.fillna("")

    categorical_matrix = encoder.transform(
        categorical_df
    )

    # ------------------------------------------------------------
    # Numerical
    # ------------------------------------------------------------

    numerical_df = df[
        numerical_columns
    ].copy()

    numerical_df = numerical_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    numerical_df = numerical_df.fillna(0)

    numerical_matrix = scaler.transform(
        numerical_df
    )

    if not sparse.issparse(numerical_matrix):
        numerical_matrix = sparse.csr_matrix(
            numerical_matrix
        )

    if not sparse.issparse(categorical_matrix):
        categorical_matrix = sparse.csr_matrix(
            categorical_matrix
        )

    structured = sparse.hstack(
        [
            categorical_matrix,
            numerical_matrix
        ],
        format="csr"
    )

    return structured


def build_hybrid_matrix(
    structured_df,
    text_series,
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns
):

    structured_matrix = build_structured_matrix(
        structured_df,
        categorical_columns,
        numerical_columns,
        encoder,
        scaler
    )

    normalized_text = normalize_text(
        text_series
    )

    text_matrix = tfidf.transform(
        normalized_text
    )

    hybrid = sparse.hstack(
        [
            structured_matrix,
            text_matrix
        ],
        format="csr"
    )

    return hybrid


def find_best_threshold(
    y_true,
    probabilities,
    technique,
    min_support=MIN_VALIDATION_SUPPORT
):

    positive = y_true.astype(int)

    support = int(
        positive.sum()
    )

    if support < min_support:

        return {
            "technique": technique,
            "support": support,
            "threshold": DEFAULT_THRESHOLD,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "status": "insufficient_support"
        }

    best = None

    for threshold in THRESHOLD_VALUES:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                positive,
                predictions,
                average="binary",
                zero_division=0
            )
        )

        candidate = {
            "technique": technique,
            "support": support,
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "status": "calibrated"
        }

        if best is None:
            best = candidate
            continue

        # Primary objective = F1.
        # Secondary objective = precision.
        # Tertiary objective = higher threshold.
        if (
            candidate["f1"] > best["f1"]
            or (
                candidate["f1"] == best["f1"]
                and candidate["precision"]
                > best["precision"]
            )
            or (
                candidate["f1"] == best["f1"]
                and candidate["precision"]
                == best["precision"]
                and candidate["threshold"]
                > best["threshold"]
            )
        ):
            best = candidate

    return best


def apply_thresholds(
    probabilities,
    techniques,
    thresholds
):

    predictions = np.zeros_like(
        probabilities,
        dtype=np.int8
    )

    for index, technique in enumerate(techniques):

        threshold = thresholds.get(
            technique,
            DEFAULT_THRESHOLD
        )

        predictions[:, index] = (
            probabilities[:, index]
            >= threshold
        ).astype(np.int8)

    # ------------------------------------------------------------
    # Fallback
    #
    # Never return an event with zero predicted techniques.
    # For such an event, select its highest-probability technique.
    # ------------------------------------------------------------

    zero_rows = (
        predictions.sum(axis=1) == 0
    )

    zero_count = int(
        zero_rows.sum()
    )

    if zero_count > 0:

        best_indices = np.argmax(
            probabilities[zero_rows],
            axis=1
        )

        rows = np.where(
            zero_rows
        )[0]

        predictions[
            rows,
            best_indices
        ] = 1

    return predictions, zero_count


def evaluate_predictions(
    y_true,
    predictions,
    title
):

    micro_p, micro_r, micro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="micro",
            zero_division=0
        )
    )

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    print_header(title)

    print(
        f"Micro Precision   : {micro_p:.4f}"
    )

    print(
        f"Micro Recall      : {micro_r:.4f}"
    )

    print(
        f"Micro F1          : {micro_f1:.4f}"
    )

    print()

    print(
        f"Macro Precision   : {macro_p:.4f}"
    )

    print(
        f"Macro Recall      : {macro_r:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print()

    print(
        f"Weighted Precision: {weighted_p:.4f}"
    )

    print(
        f"Weighted Recall   : {weighted_r:.4f}"
    )

    print(
        f"Weighted F1      : {weighted_f1:.4f}"
    )

    return {
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1)
    }


def probabilities_from_model(
    model,
    matrix
):

    probabilities = model.predict_proba(
        matrix
    )

    if isinstance(probabilities, list):

        probability_matrix = np.zeros(
            (
                matrix.shape[0],
                len(probabilities)
            ),
            dtype=np.float64
        )

        for index, value in enumerate(
            probabilities
        ):

            value = np.asarray(value)

            if value.ndim == 2:

                if value.shape[1] == 2:

                    probability_matrix[
                        :,
                        index
                    ] = value[:, 1]

                else:

                    probability_matrix[
                        :,
                        index
                    ] = value.ravel()

            else:

                probability_matrix[
                    :,
                    index
                ] = value

        return probability_matrix

    return np.asarray(
        probabilities
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "SmartSOC Windows-APT Model B v4 - Threshold Calibration"
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    # ========================================================================
    # CHECK FILES
    # ========================================================================

    print("Checking required artifacts...")

    required_files = [
        STRUCTURED_TRAIN,
        STRUCTURED_TEST,
        TEXT_TRAIN,
        TEXT_TEST,
        LABEL_TRAIN,
        LABEL_TEST,
        META_TRAIN,
        META_TEST,
        SCHEMA_FILE,
        MODEL_FILE,
        STRUCTURED_ENCODER_FILE,
        NUMERICAL_SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        FEATURE_SCHEMA_FILE
    ]

    for path in required_files:
        require_file(path)

    print()

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("Loading v4 datasets...")

    structured_train = pd.read_csv(
        STRUCTURED_TRAIN
    )

    structured_test = pd.read_csv(
        STRUCTURED_TEST
    )

    text_train = pd.read_csv(
        TEXT_TRAIN
    )

    text_test = pd.read_csv(
        TEXT_TEST
    )

    labels_train = pd.read_csv(
        LABEL_TRAIN
    )

    labels_test = pd.read_csv(
        LABEL_TEST
    )

    meta_train = pd.read_csv(
        META_TRAIN
    )

    meta_test = pd.read_csv(
        META_TEST
    )

    print(
        "Training events:",
        len(structured_train)
    )

    print(
        "Testing events :",
        len(structured_test)
    )

    print(
        "Training text rows:",
        len(text_train)
    )

    print(
        "Testing text rows :",
        len(text_test)
    )

    # ========================================================================
    # LOAD SCHEMA
    # ========================================================================

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        dataset_schema = json.load(f)

    with open(
        FEATURE_SCHEMA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feature_schema = json.load(f)

    categorical_columns = (
        feature_schema.get(
            "categorical_columns"
        )
        or dataset_schema.get(
            "categorical_columns"
        )
    )

    numerical_columns = (
        feature_schema.get(
            "numerical_columns"
        )
        or dataset_schema.get(
            "numerical_columns"
        )
    )

    if not categorical_columns:
        raise ValueError(
            "Categorical feature list missing "
            "from v4 schema."
        )

    if not numerical_columns:
        raise ValueError(
            "Numerical feature list missing "
            "from v4 schema."
        )

    print()

    print(
        "Categorical features:",
        categorical_columns
    )

    print(
        "Numerical features:",
        len(numerical_columns)
    )

    # ========================================================================
    # LOAD MODEL ARTIFACTS
    # ========================================================================

    print()
    print("Loading trained v4 artifacts...")

    model = joblib.load(
        MODEL_FILE
    )

    structured_encoder = joblib.load(
        STRUCTURED_ENCODER_FILE
    )

    numerical_scaler = joblib.load(
        NUMERICAL_SCALER_FILE
    )

    tfidf = joblib.load(
        TFIDF_FILE
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    print("Model loaded.")
    print("Structured encoder loaded.")
    print("Numerical scaler loaded.")
    print("TF-IDF loaded.")
    print("Label encoder loaded.")

    techniques = list(
        label_encoder.classes_
    )

    print()
    print(
        "MITRE techniques:",
        len(techniques)
    )

    print(
        techniques
    )

    print(
        "TF-IDF vocabulary:",
        len(tfidf.vocabulary_)
    )

    # ========================================================================
    # TIMESTAMP PARSING
    # ========================================================================

    print_header(
        "Parsing timestamps"
    )

    meta_train["timestamp"] = parse_timestamp(
        meta_train["timestamp"],
        "Training timestamps"
    )

    meta_test["timestamp"] = parse_timestamp(
        meta_test["timestamp"],
        "Testing timestamps"
    )

    print()

    print(
        "Training earliest:",
        meta_train["timestamp"].min()
    )

    print(
        "Training latest  :",
        meta_train["timestamp"].max()
    )

    print(
        "Testing earliest :",
        meta_test["timestamp"].min()
    )

    print(
        "Testing latest   :",
        meta_test["timestamp"].max()
    )

    # ========================================================================
    # ALIGN DATASET LENGTHS
    # ========================================================================

    train_lengths = {
        len(structured_train),
        len(text_train),
        len(labels_train),
        len(meta_train)
    }

    test_lengths = {
        len(structured_test),
        len(text_test),
        len(labels_test),
        len(meta_test)
    }

    if len(train_lengths) != 1:

        raise ValueError(
            "Training datasets have different row counts."
        )

    if len(test_lengths) != 1:

        raise ValueError(
            "Testing datasets have different row counts."
        )

    # ========================================================================
    # SORT TRAINING DATA CHRONOLOGICALLY
    # ========================================================================

    print_header(
        "Creating temporal calibration split"
    )

    train_order = np.argsort(
        meta_train["timestamp"].values
    )

    structured_train = (
        structured_train
        .iloc[train_order]
        .reset_index(drop=True)
    )

    text_train = (
        text_train
        .iloc[train_order]
        .reset_index(drop=True)
    )

    labels_train = (
        labels_train
        .iloc[train_order]
        .reset_index(drop=True)
    )

    meta_train = (
        meta_train
        .iloc[train_order]
        .reset_index(drop=True)
    )

    # ========================================================================
    # 80/20 TEMPORAL CALIBRATION SPLIT
    # ========================================================================

    split_index = int(
        len(meta_train)
        * (1.0 - VALIDATION_FRACTION)
    )

    if split_index <= 0:
        raise ValueError(
            "Calibration split produced empty training portion."
        )

    if split_index >= len(meta_train):
        raise ValueError(
            "Calibration split produced empty validation portion."
        )

    model_train_indices = np.arange(
        0,
        split_index
    )

    validation_indices = np.arange(
        split_index,
        len(meta_train)
    )

    print(
        "Model-training events:",
        len(model_train_indices)
    )

    print(
        "Validation events    :",
        len(validation_indices)
    )

    print(
        "Final test events     :",
        len(meta_test)
    )

    validation_cutoff = (
        meta_train.loc[
            split_index - 1,
            "timestamp"
        ]
    )

    print()
    print(
        "Validation cutoff:",
        validation_cutoff
    )

    print(
        "Validation starts:",
        meta_train.loc[
            split_index,
            "timestamp"
        ]
    )

    # ========================================================================
    # BUILD LABEL MATRICES
    # ========================================================================

    y_train_full = convert_labels_to_matrix(
        labels_train,
        label_encoder
    )

    y_test = convert_labels_to_matrix(
        labels_test,
        label_encoder
    )

    y_validation = (
        y_train_full[
            validation_indices
        ]
    )

    # ========================================================================
    # BUILD VALIDATION FEATURES
    # ========================================================================
    #
    # We use the already trained v4 preprocessing artifacts.
    # The feature representation therefore remains identical to inference.
    #

    print_header(
        "Preparing validation features"
    )

    validation_structured = (
        structured_train
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    validation_text = (
        text_train
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    validation_matrix = build_hybrid_matrix(
        validation_structured,
        validation_text.iloc[:, 0],
        structured_encoder,
        numerical_scaler,
        tfidf,
        categorical_columns,
        numerical_columns
    )

    print(
        "Validation matrix:",
        validation_matrix.shape
    )

    # ========================================================================
    # BUILD TEST FEATURES
    # ========================================================================

    print_header(
        "Preparing final temporal test features"
    )

    test_text_series = text_test.iloc[:, 0]

    test_matrix = build_hybrid_matrix(
        structured_test,
        test_text_series,
        structured_encoder,
        numerical_scaler,
        tfidf,
        categorical_columns,
        numerical_columns
    )

    print(
        "Test matrix:",
        test_matrix.shape
    )

    # ========================================================================
    # FEATURE COUNT CHECK
    # ========================================================================

    expected_feature_count = (
        feature_schema.get(
            "final_feature_count"
        )
    )

    if expected_feature_count is not None:

        if validation_matrix.shape[1] != expected_feature_count:

            raise ValueError(
                "Validation feature count mismatch.\n"
                f"Generated: {validation_matrix.shape[1]}\n"
                f"Expected : {expected_feature_count}"
            )

        if test_matrix.shape[1] != expected_feature_count:

            raise ValueError(
                "Test feature count mismatch.\n"
                f"Generated: {test_matrix.shape[1]}\n"
                f"Expected : {expected_feature_count}"
            )

    print()
    print(
        "Feature alignment verified:"
    )

    print(
        "Expected:",
        expected_feature_count
    )

    print(
        "Validation:",
        validation_matrix.shape[1]
    )

    print(
        "Test:",
        test_matrix.shape[1]
    )

    # ========================================================================
    # VALIDATION PREDICTIONS
    # ========================================================================

    print_header(
        "Generating validation probabilities"
    )

    validation_probabilities = (
        probabilities_from_model(
            model,
            validation_matrix
        )
    )

    print(
        "Validation probability matrix:",
        validation_probabilities.shape
    )

    # ========================================================================
    # THRESHOLD CALIBRATION
    # ========================================================================

    print_header(
        "Searching technique-specific thresholds"
    )

    threshold_rows = []

    threshold_dict = {}

    for class_index, technique in enumerate(
        techniques
    ):

        result = find_best_threshold(
            y_validation[:, class_index],
            validation_probabilities[
                :,
                class_index
            ],
            technique
        )

        threshold_rows.append(
            result
        )

        threshold_dict[
            technique
        ] = result["threshold"]

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df = threshold_df.sort_values(
        [
            "support",
            "f1"
        ],
        ascending=[
            False,
            False
        ]
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # SAVE THRESHOLD TABLE
    # ========================================================================

    threshold_df.to_csv(
        THRESHOLD_CSV_FILE,
        index=False
    )

    print()
    print(
        "Threshold table saved:"
    )

    print(
        THRESHOLD_CSV_FILE
    )

    # ========================================================================
    # VALIDATION CALIBRATION
    # ========================================================================

    validation_predictions, validation_fallbacks = (
        apply_thresholds(
            validation_probabilities,
            techniques,
            threshold_dict
        )
    )

    print()
    print(
        "Validation fallback events:",
        validation_fallbacks
    )

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_predictions,
        "VALIDATION AFTER THRESHOLD CALIBRATION"
    )

    # ========================================================================
    # FINAL TEST PREDICTIONS
    # ========================================================================

    print_header(
        "Generating FINAL temporal test predictions"
    )

    test_probabilities = (
        probabilities_from_model(
            model,
            test_matrix
        )
    )

    print(
        "Test probability matrix:",
        test_probabilities.shape
    )

    test_predictions, test_fallbacks = (
        apply_thresholds(
            test_probabilities,
            techniques,
            threshold_dict
        )
    )

    print()
    print(
        "Test fallback events:",
        test_fallbacks
    )

    # ========================================================================
    # TEST EVALUATION
    # ========================================================================

    test_metrics = evaluate_predictions(
        y_test,
        test_predictions,
        "FINAL WINDOWS-APT MODEL B v4"
    )

    # ========================================================================
    # CLASSIFICATION REPORT
    # ========================================================================

    report = classification_report(
        y_test,
        test_predictions,
        target_names=techniques,
        zero_division=0
    )

    print()
    print(
        "Classification Report:"
    )

    print()
    print(report)

    # ========================================================================
    # CONFIDENCE STATISTICS
    # ========================================================================

    selected_confidences = []

    for row_index in range(
        len(test_predictions)
    ):

        selected = np.where(
            test_predictions[
                row_index
            ] == 1
        )[0]

        if len(selected) == 0:

            selected = [
                int(
                    np.argmax(
                        test_probabilities[
                            row_index
                        ]
                    )
                )
            ]

        for class_index in selected:

            selected_confidences.append(
                float(
                    test_probabilities[
                        row_index,
                        class_index
                    ]
                )
            )

    if selected_confidences:

        selected_confidences = np.asarray(
            selected_confidences
        )

        print()
        print(
            "Mean selected confidence  :",
            f"{selected_confidences.mean():.4f}"
        )

        print(
            "Median selected confidence:",
            f"{np.median(selected_confidences):.4f}"
        )

        print(
            "Minimum selected confidence:",
            f"{selected_confidences.min():.4f}"
        )

        print(
            "Maximum selected confidence:",
            f"{selected_confidences.max():.4f}"
        )

    # ========================================================================
    # SAVE PREDICTIONS
    # ========================================================================

    print_header(
        "Saving calibrated test predictions"
    )

    prediction_rows = []

    for row_index in range(
        len(test_predictions)
    ):

        predicted_indices = np.where(
            test_predictions[
                row_index
            ] == 1
        )[0]

        predicted_techniques = [
            techniques[index]
            for index in predicted_indices
        ]

        prediction_rows.append(
            {
                "timestamp":
                    meta_test.iloc[
                        row_index
                    ]["timestamp"],

                "agent":
                    meta_test.iloc[
                        row_index
                    ].get(
                        "agent",
                        ""
                    ),

                "event_id":
                    meta_test.iloc[
                        row_index
                    ].get(
                        "event_id",
                        ""
                    ),

                "true_techniques":
                    labels_test.iloc[
                        row_index
                    ]["techniques"],

                "predicted_techniques":
                    ",".join(
                        predicted_techniques
                    ),

                "prediction_count":
                    len(
                        predicted_techniques
                    ),

                "max_confidence":
                    float(
                        np.max(
                            test_probabilities[
                                row_index
                            ]
                        )
                    ),

                "mean_predicted_confidence":
                    float(
                        np.mean(
                            test_probabilities[
                                row_index,
                                predicted_indices
                            ]
                        )
                    )
                    if len(predicted_indices)
                    else 0.0
            }
        )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    prediction_df.to_csv(
        TEST_PREDICTIONS_FILE,
        index=False
    )

    print(
        "Predictions saved:"
    )

    print(
        TEST_PREDICTIONS_FILE
    )

    # ========================================================================
    # SAVE THRESHOLD JSON
    # ========================================================================

    threshold_payload = {
        "model": "SmartSOC Windows-APT Model B v4",
        "calibration_method":
            "technique_specific_f1_threshold",
        "timestamp_format":
            TIMESTAMP_FORMAT,
        "default_threshold":
            DEFAULT_THRESHOLD,
        "validation_fraction":
            VALIDATION_FRACTION,
        "validation_cutoff":
            str(validation_cutoff),
        "techniques":
            techniques,
        "thresholds":
            threshold_dict,
        "validation_metrics":
            validation_metrics,
        "test_metrics":
            test_metrics
    }

    with open(
        THRESHOLD_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            threshold_payload,
            f,
            indent=4
        )

    print()
    print(
        "Threshold JSON saved:"
    )

    print(
        THRESHOLD_FILE
    )

    # ========================================================================
    # COPY / SAVE CALIBRATED ARTIFACTS
    # ========================================================================
    #
    # The model itself is unchanged. Calibration consists of the threshold
    # layer applied on top of the probability-producing model.
    #

    joblib.dump(
        model,
        CALIBRATED_MODEL_FILE
    )

    joblib.dump(
        structured_encoder,
        CALIBRATED_STRUCTURED_ENCODER_FILE
    )

    joblib.dump(
        numerical_scaler,
        CALIBRATED_SCALER_FILE
    )

    joblib.dump(
        tfidf,
        CALIBRATED_TFIDF_FILE
    )

    joblib.dump(
        label_encoder,
        CALIBRATED_LABEL_ENCODER_FILE
    )

    with open(
        CALIBRATED_SCHEMA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_schema,
            f,
            indent=4
        )

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    report_text = f"""
SmartSOC Windows-APT Model B v4
Threshold Calibration Report
===========================================

Timestamp format:
{TIMESTAMP_FORMAT}

Training events:
{len(meta_train)}

Model-training calibration portion:
{len(model_train_indices)}

Validation events:
{len(validation_indices)}

Final temporal test events:
{len(meta_test)}

Validation cutoff:
{validation_cutoff}

MITRE techniques:
{len(techniques)}

Feature count:
{test_matrix.shape[1]}

Validation fallback events:
{validation_fallbacks}

Test fallback events:
{test_fallbacks}


VALIDATION METRICS
------------------

Micro Precision : {validation_metrics["micro_precision"]:.4f}
Micro Recall    : {validation_metrics["micro_recall"]:.4f}
Micro F1        : {validation_metrics["micro_f1"]:.4f}

Macro Precision : {validation_metrics["macro_precision"]:.4f}
Macro Recall    : {validation_metrics["macro_recall"]:.4f}
Macro F1        : {validation_metrics["macro_f1"]:.4f}


FINAL TEST METRICS
------------------

Micro Precision : {test_metrics["micro_precision"]:.4f}
Micro Recall    : {test_metrics["micro_recall"]:.4f}
Micro F1        : {test_metrics["micro_f1"]:.4f}

Macro Precision : {test_metrics["macro_precision"]:.4f}
Macro Recall    : {test_metrics["macro_recall"]:.4f}
Macro F1        : {test_metrics["macro_f1"]:.4f}

Weighted Precision : {test_metrics["weighted_precision"]:.4f}
Weighted Recall    : {test_metrics["weighted_recall"]:.4f}
Weighted F1        : {test_metrics["weighted_f1"]:.4f}


CLASSIFICATION REPORT
---------------------

{report}


ARTIFACTS
---------

Calibrated model:
{CALIBRATED_MODEL_FILE}

Structured encoder:
{CALIBRATED_STRUCTURED_ENCODER_FILE}

Numerical scaler:
{CALIBRATED_SCALER_FILE}

TF-IDF:
{CALIBRATED_TFIDF_FILE}

Label encoder:
{CALIBRATED_LABEL_ENCODER_FILE}

Feature schema:
{CALIBRATED_SCHEMA_FILE}

Thresholds:
{THRESHOLD_FILE}

Threshold CSV:
{THRESHOLD_CSV_FILE}

Test predictions:
{TEST_PREDICTIONS_FILE}
"""

    with open(
        CALIBRATION_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report_text.strip()
        )

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================

    print_header(
        "SmartSOC Windows-APT Model B v4 Calibration COMPLETE"
    )

    print(
        "Created:"
    )

    print(
        CALIBRATED_MODEL_FILE
    )

    print(
        CALIBRATED_STRUCTURED_ENCODER_FILE
    )

    print(
        CALIBRATED_SCALER_FILE
    )

    print(
        CALIBRATED_TFIDF_FILE
    )

    print(
        CALIBRATED_LABEL_ENCODER_FILE
    )

    print(
        CALIBRATED_SCHEMA_FILE
    )

    print(
        THRESHOLD_FILE
    )

    print(
        THRESHOLD_CSV_FILE
    )

    print(
        CALIBRATION_REPORT_FILE
    )

    print(
        TEST_PREDICTIONS_FILE
    )

    print()
    print(
        "Final Test Micro F1:",
        f"{test_metrics['micro_f1']:.4f}"
    )

    print(
        "Final Test Macro F1:",
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "Calibration pipeline finished successfully."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()