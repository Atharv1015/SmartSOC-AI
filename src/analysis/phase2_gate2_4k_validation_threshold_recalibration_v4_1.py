"""
SMARTSOC PHASE 2 GATE 2.4K
VALIDATION-BASED THRESHOLD RECALIBRATION

Purpose
-------
Recalibrate technique-specific thresholds using ONLY the chronological
validation portion of the training set.

IMPORTANT
---------
- The untouched test set is NOT used.
- Production model artifacts are NOT modified.
- Existing thresholds.json is NOT modified.
- A NEW proposed threshold artifact is written to the report directory.
- Validation split is chronological.
- Feature construction is aligned to the production inference pipeline.
- Compatible with current pandas versions where infer_datetime_format
  has been removed.

Project:
    C:\\Users\\Atharv\\SmartSOC-AI

Model:
    SmartSOC Windows-APT Model B v4.1
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATASET = ROOT / "data" / "processed" / "windows_apt_v4_1"

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_4k"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_TRAIN = DATASET / "structured_train.csv"
LABELS_TRAIN = DATASET / "labels_train.csv"
METADATA_TRAIN = DATASET / "metadata_train.csv"
BEHAVIOR_TRAIN = DATASET / "behavior_text_train.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
STRUCTURED_ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"


# =============================================================================
# OUTPUTS
# =============================================================================

SENSITIVITY_FILE = REPORT_DIR / "validation_threshold_sensitivity.csv"
RECOMMENDATIONS_FILE = REPORT_DIR / "validation_threshold_recommendations.csv"
PROPOSED_THRESHOLDS_FILE = REPORT_DIR / "proposed_thresholds_validation.json"
REPORT_FILE = REPORT_DIR / "phase2_gate2_4k_report.json"


# =============================================================================
# CONSTANTS
# =============================================================================

VALIDATION_FRACTION = 0.20

# Threshold candidates.
#
# We intentionally include lower values because Gate 2.4J showed that
# several classes are systematically below 0.5 despite being correctly
# ranked by the model.
#
# However, we do NOT automatically select a threshold merely because
# it maximizes F1. A precision safeguard is also evaluated.
THRESHOLD_GRID = np.round(
    np.arange(0.05, 0.951, 0.05),
    2,
)

DEFAULT_THRESHOLD = 0.50

# Minimum acceptable precision for a threshold recommendation.
PRECISION_FLOOR = 0.90

# Minimum support for a class to make a strong threshold recommendation.
MIN_STRONG_SUPPORT = 20


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def json_safe(value: Any) -> Any:
    """
    Convert numpy/pandas values into JSON-safe Python values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


# =============================================================================
# TIMESTAMP NORMALIZATION
# =============================================================================

def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parser.

    Supports examples such as:
        Dec 1, 2024 @ 06:24:28.026
        Dec 1, 2024 06:24:28.026
        ISO timestamps
        normal pandas datetime strings

    IMPORTANT:
    Do NOT use infer_datetime_format because current pandas versions
    removed that argument.
    """

    s = series.copy()

    s = (
        s.astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "NaT": np.nan,
                "None": np.nan,
            }
        )
    )

    # Elasticsearch/Kibana style timestamp.
    s = s.str.replace(
        " @ ",
        " ",
        regex=False,
    )

    # Sometimes '@' has inconsistent whitespace.
    s = s.str.replace(
        r"\s*@\s*",
        " ",
        regex=True,
    )

    # First generic parser.
    parsed = pd.to_datetime(
        s,
        errors="coerce",
    )

    # Explicit fallback for the known project timestamp format.
    missing = parsed.isna()

    if missing.any():
        parsed_fallback = pd.to_datetime(
            s[missing],
            format="%b %d, %Y %H:%M:%S.%f",
            errors="coerce",
        )

        parsed.loc[missing] = parsed_fallback

    # Additional fallback without milliseconds.
    missing = parsed.isna()

    if missing.any():
        parsed_fallback = pd.to_datetime(
            s[missing],
            format="%b %d, %Y %H:%M:%S",
            errors="coerce",
        )

        parsed.loc[missing] = parsed_fallback

    return parsed


# =============================================================================
# LABEL PARSING
# =============================================================================

def parse_label_cell(value: Any) -> List[str]:
    """
    Parse labels such as:

        T1059.001

    or:

        T1059.001,T1105

    or:

        T1059.001, T1105
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]

    return parts


def build_multilabel_matrix(
    labels: pd.Series,
    classes: List[str],
) -> np.ndarray:
    """
    Build binary multilabel matrix aligned to the production
    MultiLabelBinarizer class ordering.
    """

    class_to_index = {
        label: idx
        for idx, label in enumerate(classes)
    }

    matrix = np.zeros(
        (len(labels), len(classes)),
        dtype=np.int8,
    )

    for row_idx, value in enumerate(labels):
        parsed = parse_label_cell(value)

        for technique in parsed:
            if technique in class_to_index:
                matrix[
                    row_idx,
                    class_to_index[technique],
                ] = 1

    return matrix


# =============================================================================
# ARTIFACT VALIDATION
# =============================================================================

def load_artifacts() -> Dict[str, Any]:

    ensure_file(MODEL_FILE, "model.joblib")
    ensure_file(
        STRUCTURED_ENCODER_FILE,
        "structured_encoder.joblib",
    )
    ensure_file(
        SCALER_FILE,
        "numerical_scaler.joblib",
    )
    ensure_file(
        TFIDF_FILE,
        "tfidf_vectorizer.joblib",
    )
    ensure_file(
        LABEL_ENCODER_FILE,
        "multilabel_encoder.joblib",
    )
    ensure_file(
        SCHEMA_FILE,
        "feature_schema.json",
    )
    ensure_file(
        THRESHOLDS_FILE,
        "thresholds.json",
    )

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(STRUCTURED_ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    with open(
        THRESHOLDS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        thresholds_artifact = json.load(f)

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "schema": schema,
        "thresholds": thresholds_artifact,
    }


def validate_artifacts(
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]
    thresholds = artifacts["thresholds"]

    categorical_columns = schema.get(
        "categorical_columns",
        [],
    )

    numerical_columns = schema.get(
        "numerical_columns",
        [],
    )

    structured_columns = (
        categorical_columns
        + numerical_columns
    )

    mitre_classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    encoded_features = int(
        getattr(
            encoder,
            "get_feature_names_out",
        )(
            categorical_columns
        ).shape[0]
    )

    numerical_features = int(
        scaler.n_features_in_
    )

    tfidf_features = int(
        len(tfidf.vocabulary_)
    )

    final_features = (
        encoded_features
        + numerical_features
        + tfidf_features
    )

    model_expected = int(
        getattr(
            model.estimators_[0],
            "n_features_in_",
            final_features,
        )
    )

    threshold_map = thresholds.get(
        "thresholds",
        {},
    )

    print("Production artifact validation")
    print("-" * 78)
    print(
        f"Model type           : "
        f"{type(model).__name__}"
    )
    print(
        f"Categorical columns  : "
        f"{len(categorical_columns)}"
    )
    print(
        f"Numerical columns    : "
        f"{len(numerical_columns)}"
    )
    print(
        f"Structured columns   : "
        f"{len(structured_columns)}"
    )
    print(
        f"TF-IDF vocabulary    : "
        f"{tfidf_features}"
    )
    print(
        f"MITRE classes        : "
        f"{len(mitre_classes)}"
    )
    print(
        f"Threshold entries    : "
        f"{len(threshold_map)}"
    )

    print()
    print("Production feature dimensions")
    print("-" * 78)
    print(
        f"Categorical encoded : "
        f"{encoded_features}"
    )
    print(
        f"Numerical           : "
        f"{numerical_features}"
    )
    print(
        f"TF-IDF              : "
        f"{tfidf_features}"
    )
    print(
        f"Component total     : "
        f"{final_features}"
    )
    print(
        f"Model input         : "
        f"{model_expected}"
    )

    if final_features != model_expected:
        raise RuntimeError(
            "Production artifact feature dimension mismatch."
        )

    if len(mitre_classes) != len(model.estimators_):
        raise RuntimeError(
            "MITRE class count does not match model estimators."
        )

    if set(mitre_classes) != set(
        map(str, threshold_map.keys())
    ):
        raise RuntimeError(
            "Threshold technique IDs do not match "
            "MultiLabelBinarizer classes."
        )

    print(
        f"[PASS] Production dimensions = "
        f"{final_features}"
    )

    return {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "structured_columns": structured_columns,
        "mitre_classes": mitre_classes,
        "encoded_features": encoded_features,
        "numerical_features": numerical_features,
        "tfidf_features": tfidf_features,
        "final_features": final_features,
        "model_expected": model_expected,
        "threshold_map": {
            str(k): float(v)
            for k, v in threshold_map.items()
        },
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    ensure_file(
        STRUCTURED_TRAIN,
        "structured_train.csv",
    )

    ensure_file(
        LABELS_TRAIN,
        "labels_train.csv",
    )

    ensure_file(
        METADATA_TRAIN,
        "metadata_train.csv",
    )

    ensure_file(
        BEHAVIOR_TRAIN,
        "behavior_text_train.csv",
    )

    structured = pd.read_csv(
        STRUCTURED_TRAIN,
        low_memory=False,
    )

    labels = pd.read_csv(
        LABELS_TRAIN,
        low_memory=False,
    )

    metadata = pd.read_csv(
        METADATA_TRAIN,
        low_memory=False,
    )

    behavior = pd.read_csv(
        BEHAVIOR_TRAIN,
        low_memory=False,
    )

    print(
        f"Structured rows : {len(structured)}"
    )
    print(
        f"Label rows      : {len(labels)}"
    )
    print(
        f"Metadata rows   : {len(metadata)}"
    )
    print(
        f"Behavior rows   : {len(behavior)}"
    )

    if not (
        len(structured)
        == len(labels)
        == len(metadata)
        == len(behavior)
    ):
        raise RuntimeError(
            "Training files are not row-aligned."
        )

    return (
        structured,
        labels,
        metadata,
        behavior,
    )


# =============================================================================
# BEHAVIOR COLUMN
# =============================================================================

def resolve_behavior_column(
    behavior: pd.DataFrame,
) -> str:

    preferred = [
        "behavior_text_normalized",
        "behavior_text",
        "text",
    ]

    for column in preferred:
        if column in behavior.columns:
            return column

    raise RuntimeError(
        "No behavior text column found. "
        f"Available columns: {list(behavior.columns)}"
    )


# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================

def create_validation_split(
    structured: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    behavior: pd.DataFrame,
) -> Dict[str, Any]:

    if "timestamp" not in metadata.columns:
        raise RuntimeError(
            "Timestamp column is required in metadata_train.csv."
        )

    timestamps = normalize_timestamp_series(
        metadata["timestamp"]
    )

    invalid = int(timestamps.isna().sum())

    print(
        f"Invalid timestamps : {invalid}"
    )

    if invalid > 0:
        raise RuntimeError(
            f"{invalid} invalid timestamps found "
            "in metadata_train.csv."
        )

    order = np.argsort(
        timestamps.values
    )

    n = len(structured)

    validation_count = int(
        math.ceil(
            n * VALIDATION_FRACTION
        )
    )

    calibration_count = n - validation_count

    calibration_idx = order[
        :calibration_count
    ]

    validation_idx = order[
        calibration_count:
    ]

    calibration_times = timestamps.iloc[
        calibration_idx
    ]

    validation_times = timestamps.iloc[
        validation_idx
    ]

    calibration_latest = calibration_times.max()
    validation_earliest = validation_times.min()

    temporal_gap = (
        validation_earliest
        - calibration_latest
    )

    print(
        f"Total training rows       : {n}"
    )
    print(
        f"Calibration rows          : "
        f"{len(calibration_idx)}"
    )
    print(
        f"Validation rows           : "
        f"{len(validation_idx)}"
    )
    print(
        f"Validation fraction       : "
        f"{VALIDATION_FRACTION:.2f}"
    )
    print(
        f"Calibration latest        : "
        f"{calibration_latest}"
    )
    print(
        f"Validation earliest       : "
        f"{validation_earliest}"
    )
    print(
        f"Temporal gap              : "
        f"{temporal_gap}"
    )

    if not (
        calibration_latest
        < validation_earliest
    ):
        raise RuntimeError(
            "Chronological split is invalid."
        )

    print(
        "[PASS] Chronological validation split created"
    )

    return {
        "calibration_idx": calibration_idx,
        "validation_idx": validation_idx,
        "timestamps": timestamps,
        "calibration_latest": calibration_latest,
        "validation_earliest": validation_earliest,
        "temporal_gap": temporal_gap,
    }


# =============================================================================
# FEATURE TRANSFORMATION
# =============================================================================

def build_feature_matrix(
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    artifacts: Dict[str, Any],
    artifact_info: Dict[str, Any],
) -> sparse.csr_matrix:

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    categorical_columns = artifact_info[
        "categorical_columns"
    ]

    numerical_columns = artifact_info[
        "numerical_columns"
    ]

    behavior_column = resolve_behavior_column(
        behavior
    )

    missing_structured = [
        c
        for c in artifact_info[
            "structured_columns"
        ]
        if c not in structured.columns
    ]

    if missing_structured:
        raise RuntimeError(
            "Missing structured columns: "
            + ", ".join(missing_structured)
        )

    # -------------------------------------------------------------------------
    # CATEGORICAL
    # -------------------------------------------------------------------------

    categorical = structured[
        categorical_columns
    ].copy()

    categorical = categorical.fillna(
        ""
    )

    for column in categorical_columns:
        categorical[column] = (
            categorical[column]
            .astype(str)
        )

    X_cat = encoder.transform(
        categorical
    )

    # -------------------------------------------------------------------------
    # NUMERICAL
    # -------------------------------------------------------------------------

    numerical = structured[
        numerical_columns
    ].copy()

    for column in numerical_columns:
        numerical[column] = pd.to_numeric(
            numerical[column],
            errors="coerce",
        )

    numerical = numerical.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numerical = numerical.fillna(0.0)

    X_num = scaler.transform(
        numerical
    )

    X_num = sparse.csr_matrix(
        X_num
    )

    # -------------------------------------------------------------------------
    # TEXT
    # -------------------------------------------------------------------------

    text_series = (
        behavior[behavior_column]
        .fillna("")
        .astype(str)
    )

    X_text = tfidf.transform(
        text_series
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    X_final = sparse.hstack(
        [
            X_cat,
            X_num,
            X_text,
        ],
        format="csr",
    )

    return X_final


# =============================================================================
# PROBABILITY INFERENCE
# =============================================================================

def run_inference(
    X: sparse.csr_matrix,
    artifacts: Dict[str, Any],
    artifact_info: Dict[str, Any],
) -> np.ndarray:

    model = artifacts["model"]

    probabilities = model.predict_proba(
        X
    )

    # OneVsRest can theoretically return list-like structures
    # depending on estimator behavior. Normalize to ndarray.
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    expected_classes = len(
        artifact_info["mitre_classes"]
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            "Probability matrix is not 2-dimensional."
        )

    if probabilities.shape[1] != expected_classes:
        raise RuntimeError(
            "Probability matrix class dimension "
            f"{probabilities.shape[1]} != "
            f"{expected_classes}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Probability matrix contains "
            "non-finite values."
        )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise RuntimeError(
            "Probability matrix contains values "
            "outside [0,1]."
        )

    return probabilities


# =============================================================================
# METRICS
# =============================================================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:

    y_pred = (
        probabilities >= threshold
    ).astype(np.int8)

    tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    existing_threshold: float,
    support: int,
) -> Dict[str, Any]:

    candidate_rows = []

    for threshold in THRESHOLD_GRID:

        metrics = calculate_binary_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        candidate_rows.append(
            metrics
        )

    candidates = pd.DataFrame(
        candidate_rows
    )

    # -------------------------------------------------------------------------
    # 1. Prefer thresholds satisfying precision floor.
    # -------------------------------------------------------------------------

    precision_safe = candidates[
        candidates["precision"]
        >= PRECISION_FLOOR
    ].copy()

    if len(precision_safe) > 0:

        # Among precision-safe thresholds, maximize F1.
        #
        # Tie-break:
        #   higher recall
        #   then higher threshold
        precision_safe = (
            precision_safe
            .sort_values(
                by=[
                    "f1",
                    "recall",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
        )

        selected = (
            precision_safe.iloc[0]
        )

        selection_method = (
            "BEST_F1_WITH_PRECISION_FLOOR"
        )

    else:

        # No threshold achieves precision >= 0.90.
        #
        # In that situation, maximizing F1 alone can generate
        # dangerous false-positive behavior, especially for rare
        # classes. We therefore select the threshold with the
        # highest precision first, then F1.
        fallback = (
            candidates
            .sort_values(
                by=[
                    "precision",
                    "f1",
                    "recall",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                ],
            )
        )

        selected = fallback.iloc[0]

        selection_method = (
            "NO_PRECISION_SAFE_THRESHOLD"
        )

    # -------------------------------------------------------------------------
    # Best raw F1 threshold, for comparison only.
    # -------------------------------------------------------------------------

    best_f1 = (
        candidates
        .sort_values(
            by=[
                "f1",
                "precision",
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    existing_metrics = calculate_binary_metrics(
        y_true,
        probabilities,
        existing_threshold,
    )

    selected_threshold = float(
        selected["threshold"]
    )

    selected_metrics = calculate_binary_metrics(
        y_true,
        probabilities,
        selected_threshold,
    )

    threshold_change = (
        selected_threshold
        - existing_threshold
    )

    return {
        "existing": existing_metrics,
        "selected": selected_metrics,
        "best_f1": {
            k: json_safe(v)
            for k, v in best_f1.to_dict().items()
        },
        "selection_method": selection_method,
        "threshold_change": float(
            threshold_change
        ),
        "support": int(support),
        "precision_floor": PRECISION_FLOOR,
        "strong_support": (
            support >= MIN_STRONG_SUPPORT
        ),
        "candidate_table": candidates,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate() -> None:

    banner(
        "SMARTSOC PHASE 2 GATE 2.4K"
    )

    print(
        "VALIDATION-BASED THRESHOLD RECALIBRATION"
    )

    print()
    print(
        "IMPORTANT: The untouched test set is NOT used."
    )
    print(
        "No production threshold artifact will be modified."
    )
    print(
        "A proposed validation threshold artifact will "
        "be created separately."
    )

    # =========================================================================
    # LOAD ARTIFACTS
    # =========================================================================

    banner(
        "PRODUCTION ARTIFACT VALIDATION"
    )

    artifacts = load_artifacts()

    artifact_info = validate_artifacts(
        artifacts
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    banner(
        "LOADING TRAINING DATA"
    )

    (
        structured,
        labels_df,
        metadata,
        behavior,
    ) = load_training_data()

    # =========================================================================
    # LABEL COLUMN
    # =========================================================================

    label_column = "techniques"

    if label_column not in labels_df.columns:

        if len(labels_df.columns) == 1:
            label_column = labels_df.columns[0]
        else:
            raise RuntimeError(
                "Could not determine label column."
            )

    print(
        f"Label column     : {label_column}"
    )

    # =========================================================================
    # TIMESTAMP SOURCE
    # =========================================================================

    section(
        "TIMESTAMP SOURCE"
    )

    print(
        "Timestamp source    : metadata_train.csv"
    )
    print(
        "Timestamp column    : timestamp"
    )

    # =========================================================================
    # CHRONOLOGICAL SPLIT
    # =========================================================================

    banner(
        "CREATING CHRONOLOGICAL VALIDATION SPLIT"
    )

    split_info = create_validation_split(
        structured,
        labels_df,
        metadata,
        behavior,
    )

    validation_idx = (
        split_info["validation_idx"]
    )

    # =========================================================================
    # VALIDATION DATA
    # =========================================================================

    validation_structured = (
        structured.iloc[
            validation_idx
        ]
        .reset_index(drop=True)
    )

    validation_labels = (
        labels_df.iloc[
            validation_idx
        ]
        .reset_index(drop=True)
    )

    validation_behavior = (
        behavior.iloc[
            validation_idx
        ]
        .reset_index(drop=True)
    )

    # =========================================================================
    # FEATURE MATRIX
    # =========================================================================

    banner(
        "BUILDING VALIDATION FEATURE MATRIX"
    )

    print(
        f"Validation structured rows : "
        f"{len(validation_structured)}"
    )

    print(
        f"Validation behavior rows   : "
        f"{len(validation_behavior)}"
    )

    X_validation = build_feature_matrix(
        validation_structured,
        validation_behavior,
        artifacts,
        artifact_info,
    )

    print(
        f"Categorical matrix : "
        f"({X_validation.shape[0]}, "
        f"{artifact_info['encoded_features']})"
    )

    print(
        f"Numerical matrix   : "
        f"({X_validation.shape[0]}, "
        f"{artifact_info['numerical_features']})"
    )

    print(
        f"TF-IDF matrix      : "
        f"({X_validation.shape[0]}, "
        f"{artifact_info['tfidf_features']})"
    )

    print(
        f"Final matrix       : "
        f"{X_validation.shape}"
    )

    expected_features = (
        artifact_info["final_features"]
    )

    observed_features = (
        X_validation.shape[1]
    )

    print(
        f"Expected features  : "
        f"{expected_features}"
    )

    print(
        f"Observed features  : "
        f"{observed_features}"
    )

    if observed_features != expected_features:
        raise RuntimeError(
            "Validation feature dimension mismatch."
        )

    print(
        f"[PASS] Validation feature dimension = "
        f"{expected_features}"
    )

    # =========================================================================
    # INFERENCE
    # =========================================================================

    banner(
        "RUNNING VALIDATION INFERENCE"
    )

    probabilities = run_inference(
        X_validation,
        artifacts,
        artifact_info,
    )

    print(
        f"Probability matrix : "
        f"{probabilities.shape}"
    )

    print(
        "[PASS] Probability matrix valid"
    )

    # =========================================================================
    # TRUE LABEL MATRIX
    # =========================================================================

    y_true = build_multilabel_matrix(
        validation_labels[label_column],
        artifact_info["mitre_classes"],
    )

    if y_true.shape != probabilities.shape:

        raise RuntimeError(
            "Validation label/probability "
            "shape mismatch: "
            f"{y_true.shape} vs "
            f"{probabilities.shape}"
        )

    # =========================================================================
    # THRESHOLD RECALIBRATION
    # =========================================================================

    banner(
        "VALIDATION THRESHOLD RECALIBRATION"
    )

    print(
        f"Validation rows       : "
        f"{len(validation_labels)}"
    )

    print(
        f"MITRE techniques      : "
        f"{len(artifact_info['mitre_classes'])}"
    )

    print(
        f"Threshold candidates  : "
        f"{len(THRESHOLD_GRID)}"
    )

    print(
        f"Precision floor       : "
        f"{PRECISION_FLOOR:.2f}"
    )

    results = []
    proposed_thresholds = {}

    existing_thresholds = (
        artifact_info["threshold_map"]
    )

    classes = artifact_info[
        "mitre_classes"
    ]

    for class_idx, technique in enumerate(
        classes
    ):

        y = y_true[:, class_idx]
        p = probabilities[:, class_idx]

        support = int(
            np.sum(y)
        )

        existing_threshold = float(
            existing_thresholds.get(
                technique,
                DEFAULT_THRESHOLD,
            )
        )

        selection = choose_threshold(
            y,
            p,
            existing_threshold,
            support,
        )

        existing = selection[
            "existing"
        ]

        selected = selection[
            "selected"
        ]

        best_f1 = selection[
            "best_f1"
        ]

        # ---------------------------------------------------------------------
        # Recommendation policy
        # ---------------------------------------------------------------------
        #
        # We distinguish:
        #
        # 1. STRONG_RECOMMENDATION
        #    enough validation support + precision-safe improvement
        #
        # 2. REVIEW_REQUIRED
        #    threshold sensitivity exists but support is low or
        #    precision safeguard cannot be met
        #
        # 3. KEEP_EXISTING
        #    recalibration does not materially improve F1
        #
        f1_gain = (
            float(selected["f1"])
            - float(existing["f1"])
        )

        recall_gain = (
            float(selected["recall"])
            - float(existing["recall"])
        )

        precision_change = (
            float(selected["precision"])
            - float(existing["precision"])
        )

        if (
            selection["strong_support"]
            and selected["precision"]
            >= PRECISION_FLOOR
            and f1_gain > 0.01
        ):

            recommendation = (
                "STRONG_RECOMMENDATION"
            )

            proposed_threshold = (
                float(selected["threshold"])
            )

        elif (
            selected["precision"]
            >= PRECISION_FLOOR
            and f1_gain > 0.01
        ):

            recommendation = (
                "REVIEW_REQUIRED_LOW_SUPPORT"
            )

            proposed_threshold = (
                existing_threshold
            )

        elif f1_gain > 0.01:

            recommendation = (
                "REVIEW_REQUIRED_PRECISION"
            )

            proposed_threshold = (
                existing_threshold
            )

        else:

            recommendation = (
                "KEEP_EXISTING"
            )

            proposed_threshold = (
                existing_threshold
            )

        proposed_thresholds[
            technique
        ] = proposed_threshold

        results.append(
            {
                "technique": technique,
                "support": support,
                "existing_threshold":
                    existing_threshold,

                "existing_precision":
                    existing["precision"],

                "existing_recall":
                    existing["recall"],

                "existing_f1":
                    existing["f1"],

                "best_f1_threshold":
                    best_f1["threshold"],

                "best_f1_precision":
                    best_f1["precision"],

                "best_f1_recall":
                    best_f1["recall"],

                "best_f1":
                    best_f1["f1"],

                "selected_threshold":
                    selected["threshold"],

                "selected_precision":
                    selected["precision"],

                "selected_recall":
                    selected["recall"],

                "selected_f1":
                    selected["f1"],

                "selected_tp":
                    selected["tp"],

                "selected_fp":
                    selected["fp"],

                "selected_fn":
                    selected["fn"],

                "f1_gain":
                    f1_gain,

                "recall_gain":
                    recall_gain,

                "precision_change":
                    precision_change,

                "selection_method":
                    selection["selection_method"],

                "recommendation":
                    recommendation,

                "proposed_threshold":
                    proposed_threshold,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # =========================================================================
    # PRINT IMPORTANT CLASSES
    # =========================================================================

    banner(
        "VALIDATION THRESHOLD RECOMMENDATIONS"
    )

    important = results_df[
        (
            results_df["existing_f1"]
            < 0.80
        )
        |
        (
            results_df["f1_gain"]
            > 0.05
        )
        |
        (
            results_df["recommendation"]
            != "KEEP_EXISTING"
        )
    ].copy()

    important = important.sort_values(
        by=[
            "f1_gain",
            "support",
        ],
        ascending=[
            False,
            False,
        ],
    )

    if len(important) == 0:

        print(
            "No significant threshold "
            "recommendations."
        )

    else:

        for _, row in important.iterrows():

            print()
            print(
                f"{row['technique']}"
            )

            print(
                f"  Support              : "
                f"{int(row['support'])}"
            )

            print(
                f"  Existing threshold   : "
                f"{row['existing_threshold']:.3f}"
            )

            print(
                f"  Existing P/R/F1      : "
                f"{row['existing_precision']:.4f} / "
                f"{row['existing_recall']:.4f} / "
                f"{row['existing_f1']:.4f}"
            )

            print(
                f"  Best F1 threshold    : "
                f"{row['best_f1_threshold']:.2f}"
            )

            print(
                f"  Best F1 P/R/F1       : "
                f"{row['best_f1_precision']:.4f} / "
                f"{row['best_f1_recall']:.4f} / "
                f"{row['best_f1']:.4f}"
            )

            print(
                f"  Selected threshold   : "
                f"{row['selected_threshold']:.2f}"
            )

            print(
                f"  Selected P/R/F1      : "
                f"{row['selected_precision']:.4f} / "
                f"{row['selected_recall']:.4f} / "
                f"{row['selected_f1']:.4f}"
            )

            print(
                f"  F1 gain              : "
                f"{row['f1_gain']:+.4f}"
            )

            print(
                f"  Recommendation       : "
                f"{row['recommendation']}"
            )

    # =========================================================================
    # SAVE SENSITIVITY TABLE
    # =========================================================================

    sensitivity_rows = []

    for class_idx, technique in enumerate(
        classes
    ):

        y = y_true[:, class_idx]
        p = probabilities[:, class_idx]

        support = int(
            np.sum(y)
        )

        existing_threshold = float(
            existing_thresholds.get(
                technique,
                DEFAULT_THRESHOLD,
            )
        )

        for threshold in THRESHOLD_GRID:

            metrics = calculate_binary_metrics(
                y,
                p,
                float(threshold),
            )

            sensitivity_rows.append(
                {
                    "technique": technique,
                    "support": support,
                    **metrics,
                }
            )

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    )

    sensitivity_df.to_csv(
        SENSITIVITY_FILE,
        index=False,
    )

    results_df.to_csv(
        RECOMMENDATIONS_FILE,
        index=False,
    )

    # =========================================================================
    # PROPOSED THRESHOLD ARTIFACT
    # =========================================================================

    proposed_artifact = {
        "model": "SmartSOC Windows-APT Model B v4.1",
        "artifact_type":
            "PROPOSED_VALIDATION_THRESHOLD_ARTIFACT",
        "calibration_method":
            "chronological_validation_precision_safe_f1",
        "validation_fraction":
            VALIDATION_FRACTION,
        "precision_floor":
            PRECISION_FLOOR,
        "validation_rows":
            int(len(validation_labels)),
        "calibration_rows":
            int(
                len(
                    split_info[
                        "calibration_idx"
                    ]
                )
            ),
        "validation_cutoff":
            str(
                split_info[
                    "validation_earliest"
                ]
            ),
        "production_threshold_artifact_modified":
            False,
        "test_set_used":
            False,
        "thresholds":
            proposed_thresholds,
    }

    with open(
        PROPOSED_THRESHOLDS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(
                proposed_artifact
            ),
            f,
            indent=2,
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    banner(
        "GATE 2.4K SUMMARY"
    )

    strong_count = int(
        np.sum(
            results_df[
                "recommendation"
            ]
            == "STRONG_RECOMMENDATION"
        )
    )

    review_count = int(
        np.sum(
            results_df[
                "recommendation"
            ]
            .astype(str)
            .str.startswith(
                "REVIEW_REQUIRED"
            )
        )
    )

    keep_count = int(
        np.sum(
            results_df[
                "recommendation"
            ]
            == "KEEP_EXISTING"
        )
    )

    significant_count = int(
        np.sum(
            results_df[
                "f1_gain"
            ]
            > 0.01
        )
    )

    print(
        f"MITRE techniques             : "
        f"{len(classes)}"
    )

    print(
        f"Validation rows              : "
        f"{len(validation_labels)}"
    )

    print(
        f"Existing thresholds          : "
        f"{len(existing_thresholds)}"
    )

    print(
        f"Significant F1 improvements  : "
        f"{significant_count}"
    )

    print(
        f"Strong recommendations       : "
        f"{strong_count}"
    )

    print(
        f"Review-required              : "
        f"{review_count}"
    )

    print(
        f"Keep existing                : "
        f"{keep_count}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The proposed thresholds are "
        "validation-derived only."
    )

    print(
        "The untouched test set was NOT used."
    )

    print(
        "Production thresholds.json was NOT modified."
    )

    print(
        "Model artifacts were NOT modified."
    )

    # =========================================================================
    # GATE STATUS
    # =========================================================================

    # This gate is an ANALYSIS/RECALIBRATION gate.
    #
    # It should not claim the model is accepted merely because
    # validation F1 improves. The output is a proposal for the next
    # controlled test evaluation.
    #
    # Therefore:
    #
    #   PASS = recalibration completed successfully.
    #
    gate_status = "ANALYSIS_COMPLETE"

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4K",
        "gate_name":
            "Validation-Based Threshold Recalibration",
        "status": gate_status,
        "model":
            "SmartSOC Windows-APT Model B v4.1",

        "validation_policy": {
            "validation_fraction":
                VALIDATION_FRACTION,
            "chronological":
                True,
            "precision_floor":
                PRECISION_FLOOR,
            "threshold_candidates":
                [
                    float(x)
                    for x in THRESHOLD_GRID
                ],
        },

        "data": {
            "training_rows":
                int(len(structured)),
            "calibration_rows":
                int(
                    len(
                        split_info[
                            "calibration_idx"
                        ]
                    )
                ),
            "validation_rows":
                int(len(validation_labels)),
            "invalid_timestamps":
                0,
            "calibration_latest":
                str(
                    split_info[
                        "calibration_latest"
                    ]
                ),
            "validation_earliest":
                str(
                    split_info[
                        "validation_earliest"
                    ]
                ),
            "temporal_gap":
                str(
                    split_info[
                        "temporal_gap"
                    ]
                ),
        },

        "features": {
            "categorical_encoded":
                artifact_info[
                    "encoded_features"
                ],
            "numerical":
                artifact_info[
                    "numerical_features"
                ],
            "tfidf":
                artifact_info[
                    "tfidf_features"
                ],
            "final":
                artifact_info[
                    "final_features"
                ],
        },

        "results": {
            "mitre_classes":
                len(classes),
            "significant_f1_improvements":
                significant_count,
            "strong_recommendations":
                strong_count,
            "review_required":
                review_count,
            "keep_existing":
                keep_count,
        },

        "safety": {
            "untouched_test_used":
                False,
            "production_model_modified":
                False,
            "production_thresholds_modified":
                False,
            "dataset_modified":
                False,
        },

        "outputs": {
            "sensitivity":
                str(SENSITIVITY_FILE),
            "recommendations":
                str(RECOMMENDATIONS_FILE),
            "proposed_thresholds":
                str(PROPOSED_THRESHOLDS_FILE),
            "report":
                str(REPORT_FILE),
        },
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(report),
            f,
            indent=2,
        )

    banner(
        "GATE 2.4K COMPLETE"
    )

    print(
        f"Sensitivity      : "
        f"{SENSITIVITY_FILE}"
    )

    print(
        f"Recommendations   : "
        f"{RECOMMENDATIONS_FILE}"
    )

    print(
        f"Proposed thresholds: "
        f"{PROPOSED_THRESHOLDS_FILE}"
    )

    print(
        f"Report            : "
        f"{REPORT_FILE}"
    )

    print()
    print(
        "No model, threshold, dataset, "
        "or production artifact was modified."
    )

    print()
    print(
        f"STATUS: {gate_status}"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    evaluate()