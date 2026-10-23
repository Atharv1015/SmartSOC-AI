import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


# =============================================================================
# SmartSOC Windows-APT Model B v4.1 - Threshold Calibration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = BASE_DIR / "models" / "windows_apt_v4_1"
RESULT_DIR = BASE_DIR / "results" / "windows_apt_v4_1"
CALIBRATED_DIR = MODEL_DIR / "calibrated"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATED_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FILES
# =============================================================================

FILES = {
    "structured_train": DATA_DIR / "structured_train.csv",
    "structured_test": DATA_DIR / "structured_test.csv",
    "text_train": DATA_DIR / "behavior_text_train.csv",
    "text_test": DATA_DIR / "behavior_text_test.csv",
    "labels_train": DATA_DIR / "labels_train.csv",
    "labels_test": DATA_DIR / "labels_test.csv",
    "metadata_train": DATA_DIR / "metadata_train.csv",
    "metadata_test": DATA_DIR / "metadata_test.csv",
    "schema": DATA_DIR / "schema.json",

    "model": MODEL_DIR / "multilabel_logistic_v4_1.joblib",
    "encoder": MODEL_DIR / "structured_encoder_v4_1.joblib",
    "scaler": MODEL_DIR / "numerical_scaler_v4_1.joblib",
    "tfidf": MODEL_DIR / "tfidf_vectorizer_v4_1.joblib",
    "label_encoder": MODEL_DIR / "multilabel_encoder_v4_1.joblib",
    "feature_schema": MODEL_DIR / "feature_schema_v4_1.json",
}


# =============================================================================
# HELPERS
# =============================================================================

def header(text):
    print()
    print("=" * 79)
    print(text)
    print("=" * 79)


def subheader(text):
    print()
    print("-" * 79)
    print(text)
    print("-" * 79)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )
    print(f"[OK] {path}")


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_labels(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Remove accidental column-name artifacts
    if value.lower() in {"techniques", "nan", "none"}:
        return []

    parts = []

    for item in value.replace(";", ",").split(","):
        item = item.strip()

        if item and item.upper().startswith("T"):
            parts.append(item.upper())

    return sorted(set(parts))


def load_labels(path):
    df = pd.read_csv(path)

    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {path}")

    # Expected v4.1 label column
    if "techniques" in df.columns:
        col = "techniques"
    else:
        # Find the first non-index column
        candidates = [
            c for c in df.columns
            if not str(c).lower().startswith("unnamed")
        ]

        if not candidates:
            raise ValueError(
                f"Could not identify label column in {path}"
            )

        col = candidates[0]

    return df[col].apply(parse_labels)


def sparse_shape(matrix):
    return matrix.shape


def build_structured_matrix(
    df,
    categorical_columns,
    numerical_columns,
    encoder,
    scaler,
):
    cat = df[categorical_columns].copy()

    for col in categorical_columns:
        cat[col] = cat[col].fillna("<missing>").astype(str)

    cat_matrix = encoder.transform(cat)

    num = df[numerical_columns].copy()

    for col in numerical_columns:
        num[col] = pd.to_numeric(
            num[col],
            errors="coerce"
        ).fillna(0.0)

    num_matrix = scaler.transform(num)

    return hstack(
        [
            csr_matrix(cat_matrix),
            csr_matrix(num_matrix),
        ],
        format="csr",
    )


def build_text_matrix(text_series, tfidf):
    text = normalize_text(text_series)
    return tfidf.transform(text)


def build_hybrid_matrix(
    structured,
    text,
):
    return hstack(
        [
            structured,
            text,
        ],
        format="csr",
    )


def predict_probabilities(model, X):
    probabilities = model.predict_proba(X)

    if isinstance(probabilities, list):
        probabilities = np.column_stack(
            [
                np.asarray(p).reshape(-1)
                for p in probabilities
            ]
        )

    return np.asarray(probabilities)


def apply_thresholds(probabilities, techniques, thresholds):
    predictions = np.zeros_like(
        probabilities,
        dtype=np.int8,
    )

    for index, technique in enumerate(techniques):
        threshold = thresholds.get(
            technique,
            0.5,
        )

        predictions[:, index] = (
            probabilities[:, index] >= threshold
        ).astype(np.int8)

    # Fallback:
    # Every event gets its highest-probability technique if
    # no technique crossed its threshold.
    no_prediction = predictions.sum(axis=1) == 0

    fallback_count = int(no_prediction.sum())

    if fallback_count:
        best_indices = np.argmax(
            probabilities[no_prediction],
            axis=1,
        )

        rows = np.where(no_prediction)[0]

        predictions[
            rows,
            best_indices
        ] = 1

    return predictions, fallback_count


def metrics_dict(y_true, y_pred):
    return {
        "micro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def calibrate_thresholds(
    probabilities,
    y_true,
    techniques,
):
    """
    Search thresholds from 0.10 to 0.95.

    Objective:
        maximize per-technique F1.

    Important:
        Very rare classes are protected from aggressive
        threshold fitting. If a class has fewer than
        5 validation positives, use the default threshold.
    """

    candidates = np.round(
        np.arange(
            0.10,
            1.00,
            0.05,
        ),
        2,
    )

    thresholds = {}
    rows = []

    for index, technique in enumerate(techniques):

        y = y_true[:, index]
        p = probabilities[:, index]

        support = int(y.sum())

        if support < 5:
            best_threshold = 0.50

            pred = (
                p >= best_threshold
            ).astype(np.int8)

            precision = precision_score(
                y,
                pred,
                zero_division=0,
            )

            recall = recall_score(
                y,
                pred,
                zero_division=0,
            )

            f1 = f1_score(
                y,
                pred,
                zero_division=0,
            )

        else:

            best_threshold = 0.50
            best_f1 = -1.0
            best_precision = 0.0
            best_recall = 0.0

            for threshold in candidates:

                pred = (
                    p >= threshold
                ).astype(np.int8)

                precision = precision_score(
                    y,
                    pred,
                    zero_division=0,
                )

                recall = recall_score(
                    y,
                    pred,
                    zero_division=0,
                )

                f1 = f1_score(
                    y,
                    pred,
                    zero_division=0,
                )

                # Maximize F1.
                # If tied, prefer higher threshold
                # to reduce false positives.
                if (
                    f1 > best_f1
                    or (
                        np.isclose(f1, best_f1)
                        and threshold > best_threshold
                    )
                ):
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall

            precision = best_precision
            recall = best_recall
            f1 = best_f1

        thresholds[technique] = float(
            best_threshold
        )

        rows.append(
            {
                "technique": technique,
                "support": support,
                "threshold": float(best_threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return thresholds, pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1 - Threshold Calibration"
    )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    subheader("Checking required artifacts...")

    for path in FILES.values():
        require_file(path)

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    header("Loading v4.1 dataset")

    structured_train = pd.read_csv(
        FILES["structured_train"]
    )

    structured_test = pd.read_csv(
        FILES["structured_test"]
    )

    text_train = pd.read_csv(
        FILES["text_train"]
    )

    text_test = pd.read_csv(
        FILES["text_test"]
    )

    labels_train = load_labels(
        FILES["labels_train"]
    )

    labels_test = load_labels(
        FILES["labels_test"]
    )

    metadata_train = pd.read_csv(
        FILES["metadata_train"]
    )

    metadata_test = pd.read_csv(
        FILES["metadata_test"]
    )

    print(
        "Structured train:",
        structured_train.shape,
    )

    print(
        "Structured test :",
        structured_test.shape,
    )

    print(
        "Text train      :",
        text_train.shape,
    )

    print(
        "Text test       :",
        text_test.shape,
    )

    print(
        "Labels train    :",
        len(labels_train),
    )

    print(
        "Labels test     :",
        len(labels_test),
    )

    # -------------------------------------------------------------------------
    # Alignment
    # -------------------------------------------------------------------------

    header("Checking row alignment")

    train_lengths = [
        len(structured_train),
        len(text_train),
        len(labels_train),
        len(metadata_train),
    ]

    test_lengths = [
        len(structured_test),
        len(text_test),
        len(labels_test),
        len(metadata_test),
    ]

    if len(set(train_lengths)) != 1:
        raise ValueError(
            f"Training datasets are not aligned: {train_lengths}"
        )

    if len(set(test_lengths)) != 1:
        raise ValueError(
            f"Testing datasets are not aligned: {test_lengths}"
        )

    print(
        f"[OK] Training aligned: {train_lengths[0]}"
    )

    print(
        f"[OK] Testing aligned : {test_lengths[0]}"
    )

    # -------------------------------------------------------------------------
    # Load model artifacts
    # -------------------------------------------------------------------------

    header("Loading trained v4.1 artifacts")

    model = joblib.load(
        FILES["model"]
    )

    encoder = joblib.load(
        FILES["encoder"]
    )

    scaler = joblib.load(
        FILES["scaler"]
    )

    tfidf = joblib.load(
        FILES["tfidf"]
    )

    label_encoder = joblib.load(
        FILES["label_encoder"]
    )

    with open(
        FILES["feature_schema"],
        "r",
        encoding="utf-8",
    ) as f:
        feature_schema = json.load(f)

    print("Model loaded.")
    print("Encoder loaded.")
    print("Scaler loaded.")
    print("TF-IDF loaded.")
    print("Label encoder loaded.")
    print("Feature schema loaded.")

    # -------------------------------------------------------------------------
    # Schema
    # -------------------------------------------------------------------------

    categorical_columns = feature_schema[
        "categorical_columns"
    ]

    numerical_columns = feature_schema[
        "numerical_columns"
    ]

    techniques = list(
        label_encoder.classes_
    )

    print()
    print(
        "Categorical features:",
        len(categorical_columns),
    )

    for col in categorical_columns:
        print(" -", col)

    print(
        "Numerical features:",
        len(numerical_columns),
    )

    print(
        "MITRE techniques:",
        len(techniques),
    )

    print(
        "TF-IDF vocabulary:",
        len(tfidf.vocabulary_),
    )

    # -------------------------------------------------------------------------
    # Build training structured features
    # -------------------------------------------------------------------------

    header("Building training feature matrix")

    X_struct_train = build_structured_matrix(
        structured_train,
        categorical_columns,
        numerical_columns,
        encoder,
        scaler,
    )

    print(
        "Structured training:",
        X_struct_train.shape,
    )

    # -------------------------------------------------------------------------
    # Build training text features
    # -------------------------------------------------------------------------

    subheader(
        "Building training TF-IDF features..."
    )

    X_text_train = build_text_matrix(
        text_train.iloc[:, 0],
        tfidf,
    )

    print(
        "Text training:",
        X_text_train.shape,
    )

    # -------------------------------------------------------------------------
    # Build hybrid training matrix
    # -------------------------------------------------------------------------

    X_train = build_hybrid_matrix(
        X_struct_train,
        X_text_train,
    )

    print(
        "Hybrid training:",
        X_train.shape,
    )

    # -------------------------------------------------------------------------
    # Build test matrix
    # -------------------------------------------------------------------------

    header("Building test feature matrix")

    X_struct_test = build_structured_matrix(
        structured_test,
        categorical_columns,
        numerical_columns,
        encoder,
        scaler,
    )

    X_text_test = build_text_matrix(
        text_test.iloc[:, 0],
        tfidf,
    )

    X_test = build_hybrid_matrix(
        X_struct_test,
        X_text_test,
    )

    print(
        "Hybrid testing:",
        X_test.shape,
    )

    # -------------------------------------------------------------------------
    # Verify feature dimensions
    # -------------------------------------------------------------------------

    expected_features = feature_schema.get(
        "final_feature_count"
    )

    print()
    print(
        "Expected features:",
        expected_features,
    )

    print(
        "Training features:",
        X_train.shape[1],
    )

    print(
        "Testing features:",
        X_test.shape[1],
    )

    if (
        expected_features is not None
        and X_train.shape[1] != expected_features
    ):
        raise ValueError(
            "Training feature count does not match schema."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training/test feature dimensions do not match."
        )

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != X_train.shape[1]:
            raise ValueError(
                "Model feature count does not match "
                "generated feature count."
            )

    print(
        "[OK] Feature dimensions are consistent."
    )

    # -------------------------------------------------------------------------
    # Prepare labels
    # -------------------------------------------------------------------------

    header("Preparing multilabel targets")

    y_train = label_encoder.transform(
        labels_train
    )

    y_test = label_encoder.transform(
        labels_test
    )

    print(
        "Y train:",
        y_train.shape,
    )

    print(
        "Y test :",
        y_test.shape,
    )

    # -------------------------------------------------------------------------
    # Create chronological validation split
    # -------------------------------------------------------------------------

    header(
        "Creating chronological validation split"
    )

    meta_train = metadata_train.copy()

    if "timestamp" not in meta_train.columns:
        raise ValueError(
            "metadata_train.csv does not contain "
            "'timestamp' column."
        )

    timestamps = pd.to_datetime(
        meta_train["timestamp"],
        format="%b %d, %Y @ %H:%M:%S.%f",
        errors="coerce",
    )

    invalid = int(
        timestamps.isna().sum()
    )

    print(
        "Invalid training timestamps:",
        invalid,
    )

    if invalid:
        print(
            meta_train.loc[
                timestamps.isna(),
                "timestamp",
            ].head(10).tolist()
        )

        raise ValueError(
            "Invalid timestamps found in training metadata."
        )

    order = np.argsort(
        timestamps.values
    )

    validation_fraction = 0.20

    validation_size = max(
        1,
        int(
            len(order)
            * validation_fraction
        ),
    )

    model_train_indices = order[
        :-validation_size
    ]

    validation_indices = order[
        -validation_size:
    ]

    print(
        "Model-training events:",
        len(model_train_indices),
    )

    print(
        "Validation events:",
        len(validation_indices),
    )

    validation_cutoff = timestamps.iloc[
        validation_indices
    ].min()

    print(
        "Validation cutoff:",
        validation_cutoff,
    )

    # -------------------------------------------------------------------------
    # Validation predictions
    # -------------------------------------------------------------------------

    header(
        "Generating validation probabilities"
    )

    X_validation = X_train[
        validation_indices
    ]

    y_validation = y_train[
        validation_indices
    ]

    validation_probabilities = predict_probabilities(
        model,
        X_validation,
    )

    print(
        "Validation probability matrix:",
        validation_probabilities.shape,
    )

    # -------------------------------------------------------------------------
    # Calibrate thresholds
    # -------------------------------------------------------------------------

    header(
        "Searching technique-specific thresholds"
    )

    thresholds, threshold_df = calibrate_thresholds(
        validation_probabilities,
        y_validation,
        techniques,
    )

    print()
    print(
        threshold_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Validation evaluation after calibration
    # -------------------------------------------------------------------------

    header(
        "Validation evaluation after threshold calibration"
    )

    validation_predictions, validation_fallbacks = (
        apply_thresholds(
            validation_probabilities,
            techniques,
            thresholds,
        )
    )

    validation_metrics = metrics_dict(
        y_validation,
        validation_predictions,
    )

    for key, value in validation_metrics.items():
        print(
            f"{key:22s}: {value:.4f}"
        )

    print(
        "Fallback events:",
        validation_fallbacks,
    )

    # -------------------------------------------------------------------------
    # Final test predictions
    # -------------------------------------------------------------------------

    header(
        "Generating final test predictions"
    )

    test_probabilities = predict_probabilities(
        model,
        X_test,
    )

    print(
        "Test probability matrix:",
        test_probabilities.shape,
    )

    test_predictions, test_fallbacks = (
        apply_thresholds(
            test_probabilities,
            techniques,
            thresholds,
        )
    )

    print(
        "Fallback events:",
        test_fallbacks,
    )

    # -------------------------------------------------------------------------
    # Test metrics
    # -------------------------------------------------------------------------

    header(
        "FINAL v4.1 CALIBRATED TEST EVALUATION"
    )

    test_metrics = metrics_dict(
        y_test,
        test_predictions,
    )

    print(
        f"Micro Precision   : "
        f"{test_metrics['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall      : "
        f"{test_metrics['micro_recall']:.4f}"
    )

    print(
        f"Micro F1          : "
        f"{test_metrics['micro_f1']:.4f}"
    )

    print()

    print(
        f"Macro Precision   : "
        f"{test_metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{test_metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()

    print(
        f"Weighted Precision: "
        f"{test_metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall   : "
        f"{test_metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Save calibrated artifacts
    # -------------------------------------------------------------------------

    header(
        "Saving calibrated artifacts"
    )

    calibrated_model = (
        CALIBRATED_DIR
        / "multilabel_logistic_v4_1_calibrated.joblib"
    )

    calibrated_encoder = (
        CALIBRATED_DIR
        / "structured_encoder_v4_1_calibrated.joblib"
    )

    calibrated_scaler = (
        CALIBRATED_DIR
        / "numerical_scaler_v4_1_calibrated.joblib"
    )

    calibrated_tfidf = (
        CALIBRATED_DIR
        / "tfidf_vectorizer_v4_1_calibrated.joblib"
    )

    calibrated_label_encoder = (
        CALIBRATED_DIR
        / "multilabel_encoder_v4_1_calibrated.joblib"
    )

    calibrated_schema = (
        CALIBRATED_DIR
        / "feature_schema_v4_1_calibrated.json"
    )

    thresholds_file = (
        CALIBRATED_DIR
        / "technique_thresholds_v4_1.json"
    )

    joblib.dump(
        model,
        calibrated_model,
    )

    joblib.dump(
        encoder,
        calibrated_encoder,
    )

    joblib.dump(
        scaler,
        calibrated_scaler,
    )

    joblib.dump(
        tfidf,
        calibrated_tfidf,
    )

    joblib.dump(
        label_encoder,
        calibrated_label_encoder,
    )

    with open(
        calibrated_schema,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_schema,
            f,
            indent=2,
        )

    calibration_payload = {
        "model": "SmartSOC Windows-APT Model B v4.1",
        "calibration_method": (
            "technique_specific_f1_threshold"
        ),
        "default_threshold": 0.50,
        "validation_fraction": validation_fraction,
        "validation_cutoff": (
            str(validation_cutoff)
        ),
        "techniques": techniques,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_fallback_events": (
            validation_fallbacks
        ),
        "test_fallback_events": (
            test_fallbacks
        ),
    }

    with open(
        thresholds_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            calibration_payload,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Save threshold report
    # -------------------------------------------------------------------------

    threshold_csv = (
        RESULT_DIR
        / "threshold_calibration_v4_1.csv"
    )

    threshold_df.to_csv(
        threshold_csv,
        index=False,
    )

    metrics_file = (
        RESULT_DIR
        / "calibrated_metrics_v4_1.json"
    )

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "validation": validation_metrics,
                "test": test_metrics,
                "validation_fallback_events": (
                    validation_fallbacks
                ),
                "test_fallback_events": (
                    test_fallbacks
                ),
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    header(
        "SmartSOC Windows-APT Model B v4.1 "
        "Calibration COMPLETE"
    )

    print(
        "Calibrated model:",
        calibrated_model,
    )

    print(
        "Structured encoder:",
        calibrated_encoder,
    )

    print(
        "Numerical scaler:",
        calibrated_scaler,
    )

    print(
        "TF-IDF vectorizer:",
        calibrated_tfidf,
    )

    print(
        "Label encoder:",
        calibrated_label_encoder,
    )

    print(
        "Feature schema:",
        calibrated_schema,
    )

    print(
        "Technique thresholds:",
        thresholds_file,
    )

    print(
        "Threshold report:",
        threshold_csv,
    )

    print(
        "Metrics:",
        metrics_file,
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not deploy yet."
    )

    print(
        "Inspect the calibrated test metrics and "
        "threshold distribution before moving to inference."
    )


if __name__ == "__main__":
    main()