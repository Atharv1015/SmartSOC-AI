# ==============================================================================
# SMARTSOC PHASE 2 GATE 2.4E
# UNTOUCHED TEST-SET TECHNIQUE EVALUATION
# ==============================================================================
#
# Purpose:
#   Evaluate the production SmartSOC Windows-APT Model B v4.1 against the
#   untouched structured_test.csv + behavior_text_test.csv.
#
# IMPORTANT:
#   - Does NOT retrain the model.
#   - Does NOT modify thresholds.
#   - Does NOT modify datasets.
#   - Does NOT modify production artifacts.
#   - Uses the exact production preprocessing artifacts.
#   - Uses the existing technique-specific thresholds.
#
# Outputs:
#   reports/phase2_gate2_4e/
#
#   phase2_gate2_4e_report.json
#   untouched_test_overall_metrics.json
#   per_technique_metrics.csv
#   test_prediction_summary.csv
#   false_positive_summary.csv
#   false_negative_summary.csv
#   top1_top5_results.csv
#
# ==============================================================================

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)


warnings.filterwarnings("ignore")


# ==============================================================================
# PATHS
# ==============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = ROOT / "reports" / "phase2_gate2_4e"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_TEST = DATA / "structured_test.csv"
BEHAVIOR_TEST = DATA / "behavior_text_test.csv"
LABELS_TEST = DATA / "labels_test.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
STRUCTURED_ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
NUMERICAL_SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MULTILABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
FEATURE_SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"


# ==============================================================================
# HELPERS
# ==============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    if not path.exists():
        fail(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_file(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"Missing {label}: {path}")


def ensure_numeric_matrix(matrix, name: str) -> np.ndarray:
    """
    Safely convert dense/sparse/object matrix to float64 ndarray.
    """
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except Exception as exc:
        fail(f"{name} could not be converted to numeric matrix: {exc}")

    if not np.all(np.isfinite(arr)):
        fail(f"{name} contains NaN or infinite values.")

    return arr


def parse_multilabel_value(value) -> List[str]:
    """
    Convert a labels_train/test cell such as:
        T1059.003
        T1059.003,T1087
        T1059.003, T1087
    into a clean list.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = [
        p.strip()
        for p in text.split(",")
        if p.strip()
    ]

    return parts


def labels_to_binary(
    label_series: pd.Series,
    classes: List[str],
) -> np.ndarray:
    """
    Convert the CSV multi-label representation into a binary matrix
    aligned exactly to the MultiLabelBinarizer class order.
    """
    class_to_index = {
        label: i
        for i, label in enumerate(classes)
    }

    Y = np.zeros(
        (len(label_series), len(classes)),
        dtype=np.int8,
    )

    for row_idx, value in enumerate(label_series):
        labels = parse_multilabel_value(value)

        for label in labels:
            if label in class_to_index:
                Y[row_idx, class_to_index[label]] = 1

    return Y


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ==============================================================================
# LOAD ARTIFACTS
# ==============================================================================

def load_artifacts():
    print()
    print("Loading production artifacts...")
    print()

    required = [
        (MODEL_FILE, "model"),
        (STRUCTURED_ENCODER_FILE, "structured encoder"),
        (NUMERICAL_SCALER_FILE, "numerical scaler"),
        (TFIDF_FILE, "TF-IDF vectorizer"),
        (MULTILABEL_ENCODER_FILE, "MultiLabelBinarizer"),
        (FEATURE_SCHEMA_FILE, "feature schema"),
        (THRESHOLDS_FILE, "thresholds"),
    ]

    for path, label in required:
        check_file(path, label)

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(STRUCTURED_ENCODER_FILE)
    scaler = joblib.load(NUMERICAL_SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    multilabel_encoder = joblib.load(MULTILABEL_ENCODER_FILE)
    schema = load_json(FEATURE_SCHEMA_FILE)
    thresholds_json = load_json(THRESHOLDS_FILE)

    print("Production artifact validation")
    print("-" * 78)

    print(f"Model type           : {type(model).__name__}")

    categorical_columns = schema.get("categorical_columns", [])
    numerical_columns = schema.get("numerical_columns", [])

    if not categorical_columns:
        fail("feature_schema.json does not contain categorical_columns.")

    if not numerical_columns:
        fail("feature_schema.json does not contain numerical_columns.")

    structured_columns = (
        list(categorical_columns)
        + list(numerical_columns)
    )

    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns   : {len(structured_columns)}")

    vocab_size = len(tfidf.vocabulary_)
    classes = [
        str(x)
        for x in multilabel_encoder.classes_
    ]

    print(f"TF-IDF vocabulary    : {vocab_size}")
    print(f"MITRE classes        : {len(classes)}")

    thresholds = thresholds_json.get("thresholds", {})

    if not thresholds:
        fail("No thresholds found in thresholds.json.")

    print(f"Threshold entries    : {len(thresholds)}")

    if len(classes) != 45:
        print(
            f"[WARNING] Expected 45 MITRE classes, found {len(classes)}."
        )

    if len(categorical_columns) + len(numerical_columns) != 62:
        fail(
            "Structured feature count is not 62: "
            f"{len(categorical_columns) + len(numerical_columns)}"
        )

    expected_final_features = (
        len(encoder.get_feature_names_out())
        + len(numerical_columns)
        + vocab_size
    )

    print(f"Expected final features: {expected_final_features}")

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "tfidf": tfidf,
        "multilabel_encoder": multilabel_encoder,
        "schema": schema,
        "thresholds": thresholds,
        "classes": classes,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "structured_columns": structured_columns,
        "vocab_size": vocab_size,
        "expected_final_features": expected_final_features,
    }


# ==============================================================================
# LOAD TEST DATA
# ==============================================================================

def load_test_data():
    print()
    print("Loading untouched test dataset...")
    print()

    check_file(
        STRUCTURED_TEST,
        "structured_test.csv",
    )

    check_file(
        BEHAVIOR_TEST,
        "behavior_text_test.csv",
    )

    check_file(
        LABELS_TEST,
        "labels_test.csv",
    )

    structured = pd.read_csv(
        STRUCTURED_TEST
    )

    behavior = pd.read_csv(
        BEHAVIOR_TEST
    )

    labels = pd.read_csv(
        LABELS_TEST
    )

    print(f"Structured test rows : {len(structured)}")
    print(f"Behavior test rows   : {len(behavior)}")
    print(f"Labels test rows     : {len(labels)}")

    if len(structured) != len(behavior):
        fail(
            "Structured/behavior row count mismatch: "
            f"{len(structured)} vs {len(behavior)}"
        )

    if len(structured) != len(labels):
        fail(
            "Structured/labels row count mismatch: "
            f"{len(structured)} vs {len(labels)}"
        )

    return structured, behavior, labels


# ==============================================================================
# TRANSFORM TEST DATA
# ==============================================================================

def transform_test_data(
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    artifacts: dict,
):
    schema = artifacts["schema"]
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    categorical_columns = artifacts["categorical_columns"]
    numerical_columns = artifacts["numerical_columns"]
    structured_columns = artifacts["structured_columns"]

    print()
    print("Validating test schema...")
    print("-" * 78)

    missing_structured = [
        col
        for col in structured_columns
        if col not in structured.columns
    ]

    if missing_structured:
        fail(
            "Missing structured columns:\n"
            + "\n".join(missing_structured)
        )

    unexpected = [
        col
        for col in structured.columns
        if col not in structured_columns
    ]

    if unexpected:
        print(
            "[WARNING] Unexpected structured columns:",
            unexpected,
        )

    if "behavior_text_normalized" not in behavior.columns:
        fail(
            "behavior_text_test.csv does not contain "
            "'behavior_text_normalized'."
        )

    print(
        "[PASS] All required structured columns present"
    )

    print(
        "[PASS] behavior_text_normalized present"
    )

    if len(structured) != len(behavior):
        fail("Row alignment failure.")

    print(
        f"[PASS] Row alignment = {len(structured)}"
    )

    # --------------------------------------------------------------------------
    # Structured data
    # --------------------------------------------------------------------------

    structured_work = structured[
        structured_columns
    ].copy()

    # Categorical values must be strings/object-compatible.
    categorical_data = structured_work[
        categorical_columns
    ].copy()

    for col in categorical_columns:
        categorical_data[col] = (
            categorical_data[col]
            .fillna("")
            .astype(str)
        )

    # Numerical values must be numeric and finite.
    numerical_data = structured_work[
        numerical_columns
    ].copy()

    for col in numerical_columns:
        numerical_data[col] = pd.to_numeric(
            numerical_data[col],
            errors="coerce",
        )

    if numerical_data.isna().any().any():
        bad_columns = (
            numerical_data.columns[
                numerical_data.isna().any()
            ].tolist()
        )

        fail(
            "Numerical test data contains NaN/non-numeric "
            f"values in: {bad_columns}"
        )

    if not np.isfinite(
        numerical_data.to_numpy(dtype=np.float64)
    ).all():
        fail(
            "Numerical test data contains infinite values."
        )

    # --------------------------------------------------------------------------
    # Categorical encoding
    # --------------------------------------------------------------------------

    X_cat_raw = encoder.transform(
        categorical_data
    )

    X_cat = ensure_numeric_matrix(
        X_cat_raw,
        "Categorical matrix",
    )

    print(
        f"Categorical matrix   : {X_cat.shape}"
    )

    expected_cat = len(
        encoder.get_feature_names_out()
    )

    if X_cat.shape[1] != expected_cat:
        fail(
            "Categorical encoded dimension mismatch: "
            f"expected {expected_cat}, got {X_cat.shape[1]}"
        )

    # --------------------------------------------------------------------------
    # Numerical scaling
    # --------------------------------------------------------------------------

    X_num_raw = scaler.transform(
        numerical_data
    )

    X_num = ensure_numeric_matrix(
        X_num_raw,
        "Numerical matrix",
    )

    print(
        f"Numerical matrix     : {X_num.shape}"
    )

    if X_num.shape[1] != len(numerical_columns):
        fail(
            "Numerical dimension mismatch: "
            f"expected {len(numerical_columns)}, "
            f"got {X_num.shape[1]}"
        )

    # --------------------------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------------------------

    behavior_text = (
        behavior["behavior_text_normalized"]
        .fillna("")
        .astype(str)
    )

    X_text_raw = tfidf.transform(
        behavior_text
    )

    X_text = ensure_numeric_matrix(
        X_text_raw,
        "TF-IDF matrix",
    )

    print(
        f"TF-IDF matrix        : {X_text.shape}"
    )

    if X_text.shape[1] != len(tfidf.vocabulary_):
        fail(
            "TF-IDF dimension mismatch."
        )

    # --------------------------------------------------------------------------
    # Final model input
    # --------------------------------------------------------------------------

    X = np.hstack(
        [
            X_cat,
            X_num,
            X_text,
        ]
    )

    print(
        f"Final matrix         : {X.shape}"
    )

    expected_final = artifacts[
        "expected_final_features"
    ]

    print(
        f"Expected final       : {expected_final}"
    )

    if X.shape[1] != expected_final:
        fail(
            "Final production feature dimension mismatch: "
            f"expected {expected_final}, "
            f"observed {X.shape[1]}"
        )

    print(
        f"[PASS] Final feature dimension = {X.shape[1]}"
    )

    return X


# ==============================================================================
# PREDICTION
# ==============================================================================

def predict(
    X: np.ndarray,
    artifacts: dict,
):
    model = artifacts["model"]

    print()
    print("Running production inference...")
    print()

    probabilities = model.predict_proba(X)

    if isinstance(probabilities, list):
        probabilities = np.column_stack(
            [
                np.asarray(p).reshape(-1)
                for p in probabilities
            ]
        )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    print(
        f"Probability matrix : {probabilities.shape}"
    )

    if not np.all(np.isfinite(probabilities)):
        fail(
            "Probability matrix contains NaN/inf."
        )

    if (
        np.min(probabilities) < 0
        or np.max(probabilities) > 1
    ):
        fail(
            "Probability matrix contains values outside [0,1]."
        )

    print(
        "[PASS] Probabilities finite and within [0,1]"
    )

    expected_classes = len(
        artifacts["classes"]
    )

    if probabilities.shape[1] != expected_classes:
        fail(
            "Probability/class dimension mismatch: "
            f"{probabilities.shape[1]} vs "
            f"{expected_classes}"
        )

    return probabilities


# ==============================================================================
# THRESHOLD APPLICATION
# ==============================================================================

def apply_thresholds(
    probabilities: np.ndarray,
    classes: List[str],
    thresholds: Dict[str, float],
):
    threshold_vector = np.array(
        [
            float(
                thresholds.get(
                    technique,
                    0.5,
                )
            )
            for technique in classes
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(threshold_vector)):
        fail(
            "Threshold vector contains invalid values."
        )

    predictions = (
        probabilities
        >= threshold_vector.reshape(1, -1)
    ).astype(np.int8)

    return predictions, threshold_vector


# ==============================================================================
# METRICS
# ==============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
):
    # --------------------------------------------------------------------------
    # Micro / macro / weighted
    # --------------------------------------------------------------------------

    micro_p, micro_r, micro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )

    macro_p, macro_r, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # --------------------------------------------------------------------------
    # Per-technique
    # --------------------------------------------------------------------------

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(len(classes)),
        zero_division=0,
    )

    per_rows = []

    for idx, technique in enumerate(classes):
        tp = int(
            np.sum(
                (y_true[:, idx] == 1)
                & (y_pred[:, idx] == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true[:, idx] == 0)
                & (y_pred[:, idx] == 1)
            )
        )

        fn = int(
            np.sum(
                (y_true[:, idx] == 1)
                & (y_pred[:, idx] == 0)
            )
        )

        tn = int(
            np.sum(
                (y_true[:, idx] == 0)
                & (y_pred[:, idx] == 0)
            )
        )

        threshold_trigger_rate = float(
            np.mean(y_pred[:, idx])
        )

        actual_rate = float(
            np.mean(y_true[:, idx])
        )

        per_rows.append(
            {
                "technique": technique,
                "support": int(support[idx]),
                "predicted_positive": int(
                    np.sum(y_pred[:, idx])
                ),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "threshold": None,
                "threshold_trigger_rate": threshold_trigger_rate,
                "actual_positive_rate": actual_rate,
            }
        )

    return {
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "per_technique": pd.DataFrame(
            per_rows
        ),
    }


# ==============================================================================
# TOP-1 / TOP-5
# ==============================================================================

def calculate_ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: List[str],
):
    top1_correct = 0
    top5_correct = 0

    rows = []

    for row_idx in range(len(y_true)):
        true_indices = np.flatnonzero(
            y_true[row_idx]
        )

        ranking = np.argsort(
            probabilities[row_idx]
        )[::-1]

        top1 = int(ranking[0])

        top5 = ranking[:5]

        top1_hit = (
            len(true_indices) > 0
            and top1 in true_indices
        )

        top5_hit = (
            len(true_indices) > 0
            and any(
                idx in true_indices
                for idx in top5
            )
        )

        if top1_hit:
            top1_correct += 1

        if top5_hit:
            top5_correct += 1

        top1_technique = classes[top1]

        top5_techniques = [
            classes[idx]
            for idx in top5
        ]

        rows.append(
            {
                "row": row_idx,
                "true_techniques": ",".join(
                    classes[idx]
                    for idx in true_indices
                ),
                "top1": top1_technique,
                "top1_probability": float(
                    probabilities[
                        row_idx,
                        top1,
                    ]
                ),
                "top1_hit": bool(top1_hit),
                "top5": ",".join(
                    top5_techniques
                ),
                "top5_hit": bool(top5_hit),
            }
        )

    total = len(y_true)

    top1_accuracy = safe_divide(
        top1_correct,
        total,
    )

    top5_accuracy = safe_divide(
        top5_correct,
        total,
    )

    return {
        "top1_correct": int(top1_correct),
        "top5_correct": int(top5_correct),
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "rows": pd.DataFrame(rows),
    }


# ==============================================================================
# FALSE POSITIVE / FALSE NEGATIVE SUMMARY
# ==============================================================================

def build_error_summaries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: List[str],
    thresholds: np.ndarray,
):
    fp_rows = []
    fn_rows = []

    for idx, technique in enumerate(classes):
        fp_mask = (
            (y_true[:, idx] == 0)
            & (y_pred[:, idx] == 1)
        )

        fn_mask = (
            (y_true[:, idx] == 1)
            & (y_pred[:, idx] == 0)
        )

        fp_indices = np.flatnonzero(
            fp_mask
        )

        fn_indices = np.flatnonzero(
            fn_mask
        )

        for row_idx in fp_indices:
            fp_rows.append(
                {
                    "row": int(row_idx),
                    "technique": technique,
                    "probability": float(
                        probabilities[
                            row_idx,
                            idx,
                        ]
                    ),
                    "threshold": float(
                        thresholds[idx]
                    ),
                }
            )

        for row_idx in fn_indices:
            fn_rows.append(
                {
                    "row": int(row_idx),
                    "technique": technique,
                    "probability": float(
                        probabilities[
                            row_idx,
                            idx,
                        ]
                    ),
                    "threshold": float(
                        thresholds[idx]
                    ),
                    "probability_gap_to_threshold": float(
                        thresholds[idx]
                        - probabilities[
                            row_idx,
                            idx,
                        ]
                    ),
                }
            )

    fp_df = pd.DataFrame(
        fp_rows,
        columns=[
            "row",
            "technique",
            "probability",
            "threshold",
        ],
    )

    fn_df = pd.DataFrame(
        fn_rows,
        columns=[
            "row",
            "technique",
            "probability",
            "threshold",
            "probability_gap_to_threshold",
        ],
    )

    return fp_df, fn_df


# ==============================================================================
# TEST-SET PREDICTION SUMMARY
# ==============================================================================

def build_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: List[str],
    thresholds: np.ndarray,
):
    rows = []

    for row_idx in range(len(y_true)):
        true_labels = [
            classes[i]
            for i in np.flatnonzero(
                y_true[row_idx]
            )
        ]

        triggered_labels = [
            classes[i]
            for i in np.flatnonzero(
                y_pred[row_idx]
            )
        ]

        ranking = np.argsort(
            probabilities[row_idx]
        )[::-1]

        top1_idx = int(ranking[0])

        top1 = classes[top1_idx]

        top1_probability = float(
            probabilities[
                row_idx,
                top1_idx,
            ]
        )

        top1_is_correct = (
            top1_idx
            in np.flatnonzero(
                y_true[row_idx]
            )
        )

        rows.append(
            {
                "row": row_idx,
                "true_techniques": ",".join(
                    true_labels
                ),
                "triggered_techniques": ",".join(
                    triggered_labels
                ),
                "top1": top1,
                "top1_probability": top1_probability,
                "top1_correct": bool(
                    top1_is_correct
                ),
                "trigger_count": len(
                    triggered_labels
                ),
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# PRINT TECHNIQUE SUMMARY
# ==============================================================================

def print_technique_summary(
    per_technique: pd.DataFrame,
):
    print()
    print("=" * 78)
    print("PER-TECHNIQUE TEST-SET PERFORMANCE")
    print("=" * 78)

    display_df = per_technique[
        [
            "technique",
            "support",
            "predicted_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "threshold",
        ]
    ].copy()

    display_df = display_df.sort_values(
        [
            "support",
            "f1",
        ],
        ascending=[
            False,
            True,
        ],
    )

    for _, row in display_df.iterrows():
        print(
            f"{row['technique']:10s} "
            f"support={int(row['support']):5d} "
            f"pred={int(row['predicted_positive']):5d} "
            f"FP={int(row['false_positive']):5d} "
            f"FN={int(row['false_negative']):5d} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"thr={row['threshold']:.2f}"
        )


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def evaluate():
    print("=" * 78)
    print("SMARTSOC PHASE 2 GATE 2.4E")
    print("UNTOUCHED TEST-SET TECHNIQUE EVALUATION")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # 1. Load production artifacts
    # --------------------------------------------------------------------------

    artifacts = load_artifacts()

    classes = artifacts["classes"]

    # --------------------------------------------------------------------------
    # 2. Load untouched test data
    # --------------------------------------------------------------------------

    structured, behavior, labels = (
        load_test_data()
    )

    # --------------------------------------------------------------------------
    # 3. Transform
    # --------------------------------------------------------------------------

    print()
    print("Transforming untouched test events...")
    print("-" * 78)

    X = transform_test_data(
        structured,
        behavior,
        artifacts,
    )

    # --------------------------------------------------------------------------
    # 4. Ground-truth labels
    # --------------------------------------------------------------------------

    print()
    print("Building ground-truth label matrix...")

    label_column = "techniques"

    if label_column not in labels.columns:
        fail(
            f"labels_test.csv does not contain '{label_column}'."
        )

    y_true = labels_to_binary(
        labels[label_column],
        classes,
    )

    print(
        f"Ground-truth matrix  : {y_true.shape}"
    )

    if y_true.shape != (
        len(structured),
        len(classes),
    ):
        fail(
            "Ground-truth matrix dimension mismatch."
        )

    print(
        "[PASS] Ground-truth labels aligned"
    )

    # --------------------------------------------------------------------------
    # 5. Production inference
    # --------------------------------------------------------------------------

    probabilities = predict(
        X,
        artifacts,
    )

    # --------------------------------------------------------------------------
    # 6. Existing production thresholds
    # --------------------------------------------------------------------------

    y_pred, threshold_vector = (
        apply_thresholds(
            probabilities,
            classes,
            artifacts["thresholds"],
        )
    )

    print()
    print(
        "[PASS] Existing production thresholds applied"
    )

    # --------------------------------------------------------------------------
    # 7. Overall metrics
    # --------------------------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred,
        classes,
    )

    per_technique = metrics[
        "per_technique"
    ]

    # Fill thresholds into per-technique table.
    threshold_map = {
        technique: float(
            artifacts["thresholds"].get(
                technique,
                0.5,
            )
        )
        for technique in classes
    }

    per_technique["threshold"] = (
        per_technique["technique"]
        .map(threshold_map)
    )

    # --------------------------------------------------------------------------
    # 8. Ranking metrics
    # --------------------------------------------------------------------------

    ranking = calculate_ranking_metrics(
        y_true,
        probabilities,
        classes,
    )

    ranking_df = ranking["rows"]

    # --------------------------------------------------------------------------
    # 9. Error analysis
    # --------------------------------------------------------------------------

    fp_df, fn_df = build_error_summaries(
        y_true,
        y_pred,
        probabilities,
        classes,
        threshold_vector,
    )

    prediction_summary = (
        build_prediction_summary(
            y_true,
            y_pred,
            probabilities,
            classes,
            threshold_vector,
        )
    )

    # --------------------------------------------------------------------------
    # 10. Aggregate counts
    # --------------------------------------------------------------------------

    total_true_labels = int(
        np.sum(y_true)
    )

    total_predicted_labels = int(
        np.sum(y_pred)
    )

    total_tp = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 1)
        )
    )

    total_fp = int(
        np.sum(
            (y_true == 0)
            & (y_pred == 1)
        )
    )

    total_fn = int(
        np.sum(
            (y_true == 1)
            & (y_pred == 0)
        )
    )

    exact_match_count = int(
        np.sum(
            np.all(
                y_true == y_pred,
                axis=1,
            )
        )
    )

    exact_match_accuracy = safe_divide(
        exact_match_count,
        len(y_true),
    )

    rows_with_any_prediction = int(
        np.sum(
            np.any(
                y_pred == 1,
                axis=1,
            )
        )
    )

    rows_with_any_truth = int(
        np.sum(
            np.any(
                y_true == 1,
                axis=1,
            )
        )
    )

    # --------------------------------------------------------------------------
    # 11. Print overall results
    # --------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("UNTOUCHED TEST-SET RESULTS")
    print("=" * 78)

    print()
    print(
        f"Test events             : {len(y_true)}"
    )

    print(
        f"MITRE techniques        : {len(classes)}"
    )

    print(
        f"Ground-truth labels     : {total_true_labels}"
    )

    print(
        f"Predicted labels        : {total_predicted_labels}"
    )

    print(
        f"True positives          : {total_tp}"
    )

    print(
        f"False positives         : {total_fp}"
    )

    print(
        f"False negatives         : {total_fn}"
    )

    print()

    print(
        f"Micro Precision         : "
        f"{metrics['micro']['precision']:.6f}"
    )

    print(
        f"Micro Recall            : "
        f"{metrics['micro']['recall']:.6f}"
    )

    print(
        f"Micro F1                : "
        f"{metrics['micro']['f1']:.6f}"
    )

    print()

    print(
        f"Macro Precision         : "
        f"{metrics['macro']['precision']:.6f}"
    )

    print(
        f"Macro Recall            : "
        f"{metrics['macro']['recall']:.6f}"
    )

    print(
        f"Macro F1                : "
        f"{metrics['macro']['f1']:.6f}"
    )

    print()

    print(
        f"Weighted Precision      : "
        f"{metrics['weighted']['precision']:.6f}"
    )

    print(
        f"Weighted Recall         : "
        f"{metrics['weighted']['recall']:.6f}"
    )

    print(
        f"Weighted F1             : "
        f"{metrics['weighted']['f1']:.6f}"
    )

    print()

    print(
        f"Top-1 accuracy          : "
        f"{ranking['top1_accuracy']:.6f}"
    )

    print(
        f"Top-5 accuracy          : "
        f"{ranking['top5_accuracy']:.6f}"
    )

    print(
        f"Exact-match accuracy    : "
        f"{exact_match_accuracy:.6f}"
    )

    print(
        f"Rows with predictions   : "
        f"{rows_with_any_prediction}"
    )

    print(
        f"Rows with ground truth  : "
        f"{rows_with_any_truth}"
    )

    # --------------------------------------------------------------------------
    # 12. Technique summary
    # --------------------------------------------------------------------------

    print_technique_summary(
        per_technique
    )

    # --------------------------------------------------------------------------
    # 13. Identify weak techniques
    # --------------------------------------------------------------------------

    weak = per_technique[
        (
            per_technique["support"] > 0
        )
        & (
            per_technique["f1"] < 0.50
        )
    ].copy()

    rare = per_technique[
        per_technique["support"] < 5
    ].copy()

    high_fp = per_technique[
        per_technique["false_positive"] > 0
    ].sort_values(
        "false_positive",
        ascending=False,
    )

    high_fn = per_technique[
        per_technique["false_negative"] > 0
    ].sort_values(
        "false_negative",
        ascending=False,
    )

    # --------------------------------------------------------------------------
    # 14. Print diagnostic summaries
    # --------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)

    print(
        f"Weak techniques (F1 < 0.50): "
        f"{len(weak)}"
    )

    if not weak.empty:
        for _, row in weak.iterrows():
            print(
                f"  {row['technique']:10s} "
                f"support={int(row['support'])} "
                f"F1={row['f1']:.4f} "
                f"precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f}"
            )

    print()
    print(
        f"Rare techniques (support < 5): "
        f"{len(rare)}"
    )

    if not rare.empty:
        for _, row in rare.iterrows():
            print(
                f"  {row['technique']:10s} "
                f"support={int(row['support'])}"
            )

    print()
    print(
        "Top false-positive techniques:"
    )

    for _, row in high_fp.head(10).iterrows():
        print(
            f"  {row['technique']:10s} "
            f"FP={int(row['false_positive'])} "
            f"F1={row['f1']:.4f}"
        )

    print()
    print(
        "Top false-negative techniques:"
    )

    for _, row in high_fn.head(10).iterrows():
        print(
            f"  {row['technique']:10s} "
            f"FN={int(row['false_negative'])} "
            f"F1={row['f1']:.4f}"
        )

    # --------------------------------------------------------------------------
    # 15. Save outputs
    # --------------------------------------------------------------------------

    per_technique_path = (
        REPORT_DIR
        / "per_technique_metrics.csv"
    )

    prediction_summary_path = (
        REPORT_DIR
        / "test_prediction_summary.csv"
    )

    fp_path = (
        REPORT_DIR
        / "false_positive_summary.csv"
    )

    fn_path = (
        REPORT_DIR
        / "false_negative_summary.csv"
    )

    ranking_path = (
        REPORT_DIR
        / "top1_top5_results.csv"
    )

    overall_metrics_path = (
        REPORT_DIR
        / "untouched_test_overall_metrics.json"
    )

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4e_report.json"
    )

    per_technique.to_csv(
        per_technique_path,
        index=False,
    )

    prediction_summary.to_csv(
        prediction_summary_path,
        index=False,
    )

    fp_df.to_csv(
        fp_path,
        index=False,
    )

    fn_df.to_csv(
        fn_path,
        index=False,
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
    )

    overall_metrics = {
        "phase": "Phase 2",
        "gate": "Gate 2.4E",
        "gate_name": (
            "Untouched Test-Set "
            "Technique Evaluation"
        ),
        "status": "EVALUATED",
        "model": (
            "SmartSOC Windows-APT "
            "Model B v4.1"
        ),
        "dataset": {
            "test_rows": int(len(y_true)),
            "structured_columns": int(
                len(
                    artifacts[
                        "structured_columns"
                    ]
                )
            ),
            "categorical_columns": int(
                len(
                    artifacts[
                        "categorical_columns"
                    ]
                )
            ),
            "numerical_columns": int(
                len(
                    artifacts[
                        "numerical_columns"
                    ]
                )
            ),
            "tfidf_features": int(
                artifacts["vocab_size"]
            ),
            "final_features": int(
                artifacts[
                    "expected_final_features"
                ]
            ),
            "mitre_classes": int(
                len(classes)
            ),
        },
        "label_counts": {
            "true_labels": total_true_labels,
            "predicted_labels": (
                total_predicted_labels
            ),
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
        },
        "metrics": {
            "micro": metrics["micro"],
            "macro": metrics["macro"],
            "weighted": metrics["weighted"],
            "top1_accuracy": float(
                ranking["top1_accuracy"]
            ),
            "top5_accuracy": float(
                ranking["top5_accuracy"]
            ),
            "exact_match_accuracy": float(
                exact_match_accuracy
            ),
        },
        "error_analysis": {
            "weak_techniques_f1_below_0_50": int(
                len(weak)
            ),
            "rare_techniques_support_below_5": int(
                len(rare)
            ),
            "false_positive_records": int(
                len(fp_df)
            ),
            "false_negative_records": int(
                len(fn_df)
            ),
        },
        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,
        "commands_executed": False,
    }

    with overall_metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            overall_metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------------------------
    # Gate status
    # --------------------------------------------------------------------------
    #
    # Do NOT automatically mark ACCEPTED based on arbitrary performance
    # thresholds. This gate is intended to provide the untouched test evidence
    # needed for the Phase 2 decision.
    #
    # Therefore the initial status is EVALUATED / REVIEW_REQUIRED.
    #
    # Acceptance will be decided after reviewing:
    #   - overall metrics
    #   - per-technique performance
    #   - FP/FN behavior
    #   - rare-class limitations
    #   - controlled-event findings
    #

    gate_status = (
        "REVIEW_REQUIRED"
    )

    report = {
        **overall_metrics,
        "gate_status": gate_status,
        "interpretation": {
            "note": (
                "This is the untouched test-set "
                "evaluation using the existing "
                "production model and thresholds. "
                "No model or threshold changes "
                "were made."
            ),
            "next_decision": (
                "Review per-technique performance, "
                "false positives, false negatives, "
                "and rare-class behavior before "
                "Phase 2 acceptance."
            ),
        },
        "outputs": {
            "overall_metrics": str(
                overall_metrics_path
            ),
            "per_technique_metrics": str(
                per_technique_path
            ),
            "prediction_summary": str(
                prediction_summary_path
            ),
            "false_positive_summary": str(
                fp_path
            ),
            "false_negative_summary": str(
                fn_path
            ),
            "ranking_results": str(
                ranking_path
            ),
        },
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("GATE 2.4E COMPLETE")
    print("=" * 78)

    print(
        f"Overall metrics : {overall_metrics_path}"
    )

    print(
        f"Per-technique   : {per_technique_path}"
    )

    print(
        f"Predictions     : {prediction_summary_path}"
    )

    print(
        f"False positives : {fp_path}"
    )

    print(
        f"False negatives : {fn_path}"
    )

    print(
        f"Top-1/Top-5     : {ranking_path}"
    )

    print(
        f"Report          : {report_path}"
    )

    print()
    print(
        "STATUS: REVIEW_REQUIRED"
    )

    print()
    print(
        "No model, threshold, dataset, or "
        "production artifact was modified."
    )

    print("=" * 78)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    evaluate()