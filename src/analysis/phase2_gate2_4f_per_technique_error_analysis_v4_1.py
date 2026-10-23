"""
SmartSOC Phase 2 Gate 2.4F
Per-Technique Error Analysis

Purpose
-------
Deep analysis of the untouched test-set predictions produced by
SmartSOC Windows-APT Model B v4.1.

This gate DOES NOT:
- retrain the model
- modify the model
- modify thresholds
- modify datasets
- execute malicious commands
- modify production artifacts

It analyzes:
- per-technique precision / recall / F1
- TP / FP / FN / TN
- support
- threshold
- predicted-positive counts
- missed detections
- false alarms
- thresholded vs ranking behavior
- rare-class performance
- weak-class performance
- most important FN / FP contributors

Expected production feature dimensions:
    categorical input : 4
    categorical encoded: 162
    numerical          : 58
    TF-IDF             : 6699
    final              : 6919
    MITRE classes      : 45
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

REPORT_DIR = ROOT / "reports" / "phase2_gate2_4f"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ARTIFACTS
# ============================================================================

MODEL_PATH = MODEL_DIR / "model.joblib"
ENCODER_PATH = MODEL_DIR / "structured_encoder.joblib"
SCALER_PATH = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"


# ============================================================================
# DATA
# ============================================================================

STRUCTURED_TEST = DATA / "structured_test.csv"
LABELS_TEST = DATA / "labels_test.csv"
BEHAVIOR_TEST = DATA / "behavior_text_test.csv"


# ============================================================================
# OUTPUTS
# ============================================================================

OVERALL_REPORT = REPORT_DIR / "phase2_gate2_4f_overall_report.json"
TECHNIQUE_REPORT = REPORT_DIR / "per_technique_error_analysis.csv"
FN_REPORT = REPORT_DIR / "false_negative_priority.csv"
FP_REPORT = REPORT_DIR / "false_positive_priority.csv"
RARE_REPORT = REPORT_DIR / "rare_technique_analysis.csv"
WEAK_REPORT = REPORT_DIR / "weak_technique_analysis.csv"
RANKING_REPORT = REPORT_DIR / "ranking_vs_threshold_analysis.csv"


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_FINAL_FEATURES = 6919
EXPECTED_MITRE_CLASSES = 45
EXPECTED_STRUCTURED_COLUMNS = 62
EXPECTED_CATEGORICAL_COLUMNS = 4
EXPECTED_NUMERICAL_COLUMNS = 58
EXPECTED_TFIDF_FEATURES = 6699


warnings.filterwarnings("ignore")


# ============================================================================
# HELPERS
# ============================================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(value) -> str:
    """
    Normalize a label cell into a canonical comma-separated MITRE string.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    # Normalize separators.
    text = text.replace(";", ",")
    text = text.replace(" ", "")

    parts = [x for x in text.split(",") if x]

    # Remove duplicates while preserving order.
    seen = set()
    cleaned = []

    for part in parts:
        if part not in seen:
            seen.add(part)
            cleaned.append(part)

    return ",".join(cleaned)


def multilabel_to_matrix(
    labels: pd.Series,
    classes: List[str],
) -> np.ndarray:
    """
    Convert comma-separated technique labels into binary matrix.
    """
    class_to_index = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((len(labels), len(classes)), dtype=np.int8)

    for row_idx, value in enumerate(labels):
        label_text = normalize_label(value)

        if not label_text:
            continue

        for technique in label_text.split(","):
            if technique in class_to_index:
                matrix[row_idx, class_to_index[technique]] = 1

    return matrix


def safe_dense(matrix):
    """
    Convert sparse/object matrix safely to dense numeric array.
    """
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    return np.asarray(matrix)


def validate_artifacts():
    print()
    print("=" * 78)
    print("PRODUCTION ARTIFACT VALIDATION")
    print("=" * 78)

    required = [
        MODEL_PATH,
        ENCODER_PATH,
        SCALER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        SCHEMA_PATH,
        THRESHOLD_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    schema = load_json(SCHEMA_PATH)
    thresholds = load_json(THRESHOLD_PATH)

    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]

    structured_columns = (
        categorical_columns + numerical_columns
    )

    classes = [str(x) for x in label_encoder.classes_]

    print(f"Model type           : {type(model).__name__}")
    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns   : {len(structured_columns)}")
    print(f"TF-IDF vocabulary    : {len(tfidf.vocabulary_)}")
    print(f"MITRE classes        : {len(classes)}")
    print(
        f"Threshold entries    : "
        f"{len(thresholds.get('thresholds', {}))}"
    )

    if len(categorical_columns) != EXPECTED_CATEGORICAL_COLUMNS:
        raise RuntimeError("Categorical feature count mismatch.")

    if len(numerical_columns) != EXPECTED_NUMERICAL_COLUMNS:
        raise RuntimeError("Numerical feature count mismatch.")

    if len(structured_columns) != EXPECTED_STRUCTURED_COLUMNS:
        raise RuntimeError("Structured feature count mismatch.")

    if len(tfidf.vocabulary_) != EXPECTED_TFIDF_FEATURES:
        raise RuntimeError("TF-IDF vocabulary mismatch.")

    if len(classes) != EXPECTED_MITRE_CLASSES:
        raise RuntimeError("MITRE class count mismatch.")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        thresholds,
        classes,
        categorical_columns,
        numerical_columns,
        structured_columns,
    )


def transform_test_data(
    structured_test: pd.DataFrame,
    behavior_test: pd.DataFrame,
    encoder,
    scaler,
    tfidf,
    categorical_columns: List[str],
    numerical_columns: List[str],
    structured_columns: List[str],
    behavior_column: str,
):
    """
    Reproduce the exact production transformation pipeline.
    """

    structured = structured_test.copy()

    structured = structured[structured_columns]

    categorical = structured[categorical_columns].copy()
    numerical = structured[numerical_columns].copy()

    # Production-compatible categorical transform.
    X_cat = encoder.transform(categorical)
    X_cat = safe_dense(X_cat).astype(float)

    # Production-compatible numerical transform.
    X_num = numerical.apply(
        pd.to_numeric,
        errors="coerce",
    )

    X_num = X_num.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_num = X_num.fillna(0.0)

    X_num = scaler.transform(X_num)
    X_num = safe_dense(X_num).astype(float)

    # Behavior text.
    behavior_text = (
        behavior_test[behavior_column]
        .fillna("")
        .astype(str)
    )

    X_text = tfidf.transform(behavior_text)
    X_text = safe_dense(X_text).astype(float)

    X = np.hstack(
        [
            X_cat,
            X_num,
            X_text,
        ]
    )

    print()
    print("TRANSFORMATION VALIDATION")
    print("-" * 78)

    print(f"Structured matrix    : {structured.shape}")
    print(f"Categorical matrix   : {X_cat.shape}")
    print(f"Numerical matrix     : {X_num.shape}")
    print(f"TF-IDF matrix        : {X_text.shape}")
    print(f"Final matrix         : {X.shape}")

    if X.shape[1] != EXPECTED_FINAL_FEATURES:
        raise RuntimeError(
            f"Final feature dimension mismatch: "
            f"{X.shape[1]} != {EXPECTED_FINAL_FEATURES}"
        )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Final feature matrix contains non-finite values."
        )

    print(
        f"[PASS] Final feature dimension = "
        f"{EXPECTED_FINAL_FEATURES}"
    )

    return X


def calculate_thresholds(
    thresholds: dict,
    classes: List[str],
) -> Dict[str, float]:
    """
    Return a complete technique -> threshold mapping.
    """
    raw = thresholds.get("thresholds", {})

    result = {}

    for technique in classes:
        value = raw.get(technique, 0.5)

        try:
            value = float(value)
        except Exception:
            value = 0.5

        if not 0.0 <= value <= 1.0:
            value = 0.5

        result[technique] = value

    return result


def top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: List[str],
    k: int,
) -> float:
    """
    Multi-label-friendly ranking metric:
    A sample is counted correct if ANY true technique appears
    in the top-k predicted techniques.
    """
    correct = 0

    for i in range(len(y_true)):
        true_indices = np.where(y_true[i] == 1)[0]

        if len(true_indices) == 0:
            continue

        top_indices = np.argsort(
            probabilities[i]
        )[::-1][:k]

        if any(idx in top_indices for idx in true_indices):
            correct += 1

    return correct / len(y_true)


# ============================================================================
# MAIN
# ============================================================================

def evaluate():
    print()
    print("=" * 78)
    print("SMARTSOC PHASE 2 GATE 2.4F")
    print("PER-TECHNIQUE ERROR ANALYSIS")
    print("=" * 78)

    # ----------------------------------------------------------------------
    # Load artifacts
    # ----------------------------------------------------------------------

    (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        thresholds,
        classes,
        categorical_columns,
        numerical_columns,
        structured_columns,
    ) = validate_artifacts()

    # ----------------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------------

    print()
    print("Loading untouched test data...")
    print("-" * 78)

    structured_test = pd.read_csv(STRUCTURED_TEST)
    labels_test = pd.read_csv(LABELS_TEST)
    behavior_test = pd.read_csv(BEHAVIOR_TEST)

    if len(structured_test) != len(labels_test):
        raise RuntimeError("Structured/test label row mismatch.")

    if len(structured_test) != len(behavior_test):
        raise RuntimeError("Structured/behavior row mismatch.")

    print(f"Test rows            : {len(structured_test)}")
    print(f"Structured columns   : {structured_test.shape[1]}")
    print(f"Behavior rows        : {len(behavior_test)}")

    # ----------------------------------------------------------------------
    # Identify behavior column
    # ----------------------------------------------------------------------

    behavior_candidates = [
        "behavior_text_normalized",
        "behavior_text",
    ]

    behavior_column = None

    for candidate in behavior_candidates:
        if candidate in behavior_test.columns:
            behavior_column = candidate
            break

    if behavior_column is None:
        raise RuntimeError(
            "Could not identify behavior text column."
        )

    print(f"Behavior column      : {behavior_column}")

    # ----------------------------------------------------------------------
    # Transform
    # ----------------------------------------------------------------------

    X = transform_test_data(
        structured_test=structured_test,
        behavior_test=behavior_test,
        encoder=encoder,
        scaler=scaler,
        tfidf=tfidf,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        structured_columns=structured_columns,
        behavior_column=behavior_column,
    )

    # ----------------------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------------------

    label_column = labels_test.columns[0]

    y_true = multilabel_to_matrix(
        labels_test[label_column],
        classes,
    )

    # ----------------------------------------------------------------------
    # Predictions
    # ----------------------------------------------------------------------

    print()
    print("Running production inference...")
    print("-" * 78)

    probabilities = model.predict_proba(X)

    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (
        len(structured_test),
        len(classes),
    ):
        raise RuntimeError(
            "Probability matrix dimension mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probability matrix contains non-finite values."
        )

    print(f"Probability matrix   : {probabilities.shape}")
    print("[PASS] Probability matrix valid")

    # ----------------------------------------------------------------------
    # Thresholded predictions
    # ----------------------------------------------------------------------

    threshold_map = calculate_thresholds(
        thresholds,
        classes,
    )

    threshold_array = np.array(
        [
            threshold_map[technique]
            for technique in classes
        ],
        dtype=float,
    )

    y_pred = (
        probabilities >= threshold_array.reshape(1, -1)
    ).astype(np.int8)

    # ----------------------------------------------------------------------
    # Overall metrics
    # ----------------------------------------------------------------------

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    total_true_labels = int(y_true.sum())
    total_predicted_labels = int(y_pred.sum())

    tp_total = int(
        np.logical_and(
            y_true == 1,
            y_pred == 1,
        ).sum()
    )

    fp_total = int(
        np.logical_and(
            y_true == 0,
            y_pred == 1,
        ).sum()
    )

    fn_total = int(
        np.logical_and(
            y_true == 1,
            y_pred == 0,
        ).sum()
    )

    # ----------------------------------------------------------------------
    # Per-technique analysis
    # ----------------------------------------------------------------------

    technique_rows = []

    for idx, technique in enumerate(classes):

        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        prob = probabilities[:, idx]

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        ).ravel()

        support = int(yt.sum())
        predicted_positive = int(yp.sum())

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
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        fn_rate = (
            fn / support
            if support > 0
            else 0.0
        )

        fp_rate = (
            fp / (len(yt) - support)
            if (len(yt) - support) > 0
            else 0.0
        )

        # Ranking statistics.
        ranks = []

        for row_idx in range(len(probabilities)):
            if yt[row_idx] == 1:
                order = np.argsort(
                    probabilities[row_idx]
                )[::-1]

                rank_positions = np.where(
                    order == idx
                )[0]

                if len(rank_positions):
                    ranks.append(
                        int(rank_positions[0]) + 1
                    )

        mean_rank = (
            float(np.mean(ranks))
            if ranks
            else None
        )

        median_rank = (
            float(np.median(ranks))
            if ranks
            else None
        )

        # Probability distributions for true positives / false negatives.
        tp_prob = prob[
            np.logical_and(yt == 1, yp == 1)
        ]

        fn_prob = prob[
            np.logical_and(yt == 1, yp == 0)
        ]

        fp_prob = prob[
            np.logical_and(yt == 0, yp == 1)
        ]

        technique_rows.append(
            {
                "technique": technique,
                "threshold": threshold_map[technique],
                "support": support,
                "predicted_positive": predicted_positive,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_negative_rate": fn_rate,
                "false_positive_rate": fp_rate,
                "mean_true_positive_probability": (
                    float(np.mean(tp_prob))
                    if len(tp_prob)
                    else None
                ),
                "mean_false_negative_probability": (
                    float(np.mean(fn_prob))
                    if len(fn_prob)
                    else None
                ),
                "max_false_negative_probability": (
                    float(np.max(fn_prob))
                    if len(fn_prob)
                    else None
                ),
                "mean_false_positive_probability": (
                    float(np.mean(fp_prob))
                    if len(fp_prob)
                    else None
                ),
                "mean_positive_probability": float(
                    np.mean(prob[yt == 1])
                ) if support > 0 else None,
                "mean_negative_probability": float(
                    np.mean(prob[yt == 0])
                ) if support < len(yt) else None,
                "mean_rank": mean_rank,
                "median_rank": median_rank,
                "rare_class": bool(support < 5),
                "weak_class": bool(f1 < 0.50),
            }
        )

    technique_df = pd.DataFrame(technique_rows)

    technique_df = technique_df.sort_values(
        by=["f1", "support"],
        ascending=[True, False],
    )

    technique_df.to_csv(
        TECHNIQUE_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # False-negative priority
    # ----------------------------------------------------------------------

    fn_priority = technique_df[
        [
            "technique",
            "support",
            "fn",
            "false_negative_rate",
            "recall",
            "f1",
            "threshold",
            "mean_false_negative_probability",
            "max_false_negative_probability",
        ]
    ].copy()

    fn_priority["fn_priority_score"] = (
        fn_priority["fn"]
        * (
            1.0
            - fn_priority["recall"]
        )
    )

    fn_priority = fn_priority.sort_values(
        by=[
            "fn",
            "false_negative_rate",
        ],
        ascending=False,
    )

    fn_priority.to_csv(
        FN_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # False-positive priority
    # ----------------------------------------------------------------------

    fp_priority = technique_df[
        [
            "technique",
            "support",
            "predicted_positive",
            "fp",
            "false_positive_rate",
            "precision",
            "f1",
            "threshold",
            "mean_false_positive_probability",
        ]
    ].copy()

    fp_priority["fp_priority_score"] = (
        fp_priority["fp"]
        * (
            1.0
            - fp_priority["precision"]
        )
    )

    fp_priority = fp_priority.sort_values(
        by=[
            "fp",
            "false_positive_rate",
        ],
        ascending=False,
    )

    fp_priority.to_csv(
        FP_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Rare technique analysis
    # ----------------------------------------------------------------------

    rare_df = technique_df[
        technique_df["support"] < 5
    ].copy()

    rare_df = rare_df.sort_values(
        by=["support", "f1"],
        ascending=[True, True],
    )

    rare_df.to_csv(
        RARE_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Weak technique analysis
    # ----------------------------------------------------------------------

    weak_df = technique_df[
        technique_df["f1"] < 0.50
    ].copy()

    weak_df = weak_df.sort_values(
        by=["f1", "support"],
        ascending=[True, False],
    )

    weak_df.to_csv(
        WEAK_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Ranking vs threshold analysis
    # ----------------------------------------------------------------------

    ranking_rows = []

    for idx, technique in enumerate(classes):

        yt = y_true[:, idx]

        threshold_detected = y_pred[:, idx] == 1

        top1_detected = []
        top5_detected = []

        for row_idx in range(len(y_true)):

            order = np.argsort(
                probabilities[row_idx]
            )[::-1]

            top1_detected.append(
                idx == order[0]
            )

            top5_detected.append(
                idx in order[:5]
            )

        top1_detected = np.asarray(
            top1_detected,
            dtype=bool,
        )

        top5_detected = np.asarray(
            top5_detected,
            dtype=bool,
        )

        true_mask = yt == 1

        support = int(true_mask.sum())

        threshold_recall = (
            int(
                np.logical_and(
                    true_mask,
                    threshold_detected,
                ).sum()
            )
            / support
            if support > 0
            else 0.0
        )

        top1_recall = (
            int(
                np.logical_and(
                    true_mask,
                    top1_detected,
                ).sum()
            )
            / support
            if support > 0
            else 0.0
        )

        top5_recall = (
            int(
                np.logical_and(
                    true_mask,
                    top5_detected,
                ).sum()
            )
            / support
            if support > 0
            else 0.0
        )

        ranking_rows.append(
            {
                "technique": technique,
                "support": support,
                "threshold": threshold_map[technique],
                "threshold_recall": threshold_recall,
                "top1_recall": top1_recall,
                "top5_recall": top5_recall,
                "top1_minus_threshold": (
                    top1_recall - threshold_recall
                ),
                "top5_minus_threshold": (
                    top5_recall - threshold_recall
                ),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows)

    ranking_df = ranking_df.sort_values(
        by="top1_minus_threshold",
        ascending=False,
    )

    ranking_df.to_csv(
        RANKING_REPORT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Overall ranking metrics
    # ----------------------------------------------------------------------

    top1_accuracy = top_k_accuracy(
        y_true,
        probabilities,
        classes,
        1,
    )

    top5_accuracy = top_k_accuracy(
        y_true,
        probabilities,
        classes,
        5,
    )

    # ----------------------------------------------------------------------
    # Important lists
    # ----------------------------------------------------------------------

    worst_recall = technique_df.sort_values(
        by=["recall", "support"],
        ascending=[True, False],
    ).head(10)

    worst_f1 = technique_df.sort_values(
        by=["f1", "support"],
        ascending=[True, False],
    ).head(10)

    largest_fn = technique_df.sort_values(
        by="fn",
        ascending=False,
    ).head(10)

    largest_fp = technique_df.sort_values(
        by="fp",
        ascending=False,
    ).head(10)

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print()
    print("=" * 78)
    print("GATE 2.4F OVERALL RESULTS")
    print("=" * 78)

    print(f"Test rows                 : {len(y_true)}")
    print(f"MITRE techniques          : {len(classes)}")
    print(f"True labels               : {total_true_labels}")
    print(f"Predicted labels          : {total_predicted_labels}")
    print(f"True positives            : {tp_total}")
    print(f"False positives           : {fp_total}")
    print(f"False negatives           : {fn_total}")

    print()
    print(f"Micro precision           : {micro_precision:.4f}")
    print(f"Micro recall              : {micro_recall:.4f}")
    print(f"Micro F1                  : {micro_f1:.4f}")

    print()
    print(f"Macro precision           : {macro_precision:.4f}")
    print(f"Macro recall              : {macro_recall:.4f}")
    print(f"Macro F1                  : {macro_f1:.4f}")

    print()
    print(f"Weighted precision        : {weighted_precision:.4f}")
    print(f"Weighted recall           : {weighted_recall:.4f}")
    print(f"Weighted F1              : {weighted_f1:.4f}")

    print()
    print(f"Top-1 sample accuracy     : {top1_accuracy:.4f}")
    print(f"Top-5 sample accuracy     : {top5_accuracy:.4f}")

    print()
    print("=" * 78)
    print("WEAKEST TECHNIQUES BY F1")
    print("=" * 78)

    for _, row in worst_f1.iterrows():
        print(
            f"{row['technique']:12s} "
            f"support={int(row['support']):5d} "
            f"F1={row['f1']:.4f} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"FN={int(row['fn']):5d}"
        )

    print()
    print("=" * 78)
    print("LARGEST FALSE-NEGATIVE CONTRIBUTORS")
    print("=" * 78)

    for _, row in largest_fn.iterrows():
        print(
            f"{row['technique']:12s} "
            f"support={int(row['support']):5d} "
            f"FN={int(row['fn']):5d} "
            f"recall={row['recall']:.4f} "
            f"F1={row['f1']:.4f}"
        )

    print()
    print("=" * 78)
    print("LARGEST FALSE-POSITIVE CONTRIBUTORS")
    print("=" * 78)

    for _, row in largest_fp.iterrows():
        print(
            f"{row['technique']:12s} "
            f"support={int(row['support']):5d} "
            f"FP={int(row['fp']):5d} "
            f"precision={row['precision']:.4f} "
            f"F1={row['f1']:.4f}"
        )

    print()
    print("=" * 78)
    print("LOWEST RECALL TECHNIQUES")
    print("=" * 78)

    for _, row in worst_recall.iterrows():
        print(
            f"{row['technique']:12s} "
            f"support={int(row['support']):5d} "
            f"recall={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"threshold={row['threshold']:.3f}"
        )

    print()
    print("=" * 78)
    print("RARE / WEAK CLASS SUMMARY")
    print("=" * 78)

    print(
        f"Rare techniques (support < 5) : "
        f"{len(rare_df)}"
    )

    print(
        f"Weak techniques (F1 < 0.50)   : "
        f"{len(weak_df)}"
    )

    print()
    print("=" * 78)
    print("MOST IMPORTANT RANKING VS THRESHOLD GAPS")
    print("=" * 78)

    ranking_gap = ranking_df.sort_values(
        by="top1_minus_threshold",
        ascending=False,
    ).head(10)

    for _, row in ranking_gap.iterrows():
        print(
            f"{row['technique']:12s} "
            f"support={int(row['support']):5d} "
            f"threshold-R={row['threshold_recall']:.4f} "
            f"top1-R={row['top1_recall']:.4f} "
            f"gap={row['top1_minus_threshold']:+.4f}"
        )

    # ----------------------------------------------------------------------
    # Interpretation logic
    # ----------------------------------------------------------------------

    high_fn_techniques = int(
        (technique_df["fn"] > 0).sum()
    )

    high_fp_techniques = int(
        (technique_df["fp"] > 0).sum()
    )

    rare_count = len(rare_df)
    weak_count = len(weak_df)

    # High-confidence FN count.
    high_conf_fn = int(
        (
            technique_df[
                "max_false_negative_probability"
            ].fillna(0.0)
            >= technique_df["threshold"]
        ).sum()
    )

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4F",
        "gate_name": "Per-Technique Error Analysis",
        "status": "ANALYZED",
        "model": "SmartSOC Windows-APT Model B v4.1",
        "dataset": {
            "test_rows": int(len(y_true)),
            "structured_columns": EXPECTED_STRUCTURED_COLUMNS,
            "categorical_columns": EXPECTED_CATEGORICAL_COLUMNS,
            "numerical_columns": EXPECTED_NUMERICAL_COLUMNS,
            "tfidf_features": EXPECTED_TFIDF_FEATURES,
            "final_features": EXPECTED_FINAL_FEATURES,
            "mitre_classes": len(classes),
        },
        "overall_metrics": {
            "true_labels": total_true_labels,
            "predicted_labels": total_predicted_labels,
            "true_positive": tp_total,
            "false_positive": fp_total,
            "false_negative": fn_total,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "top1_sample_accuracy": top1_accuracy,
            "top5_sample_accuracy": top5_accuracy,
        },
        "error_analysis": {
            "techniques_with_false_negatives": high_fn_techniques,
            "techniques_with_false_positives": high_fp_techniques,
            "weak_techniques_f1_below_0_50": weak_count,
            "rare_techniques_support_below_5": rare_count,
            "high_confidence_false_negative_techniques": high_conf_fn,
        },
        "important_findings": {
            "largest_false_negative_techniques": [
                str(x)
                for x in largest_fn["technique"].tolist()
            ],
            "largest_false_positive_techniques": [
                str(x)
                for x in largest_fp["technique"].tolist()
            ],
            "weakest_f1_techniques": [
                str(x)
                for x in worst_f1["technique"].tolist()
            ],
        },
        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,
        "commands_executed": False,
        "gate_status": "REVIEW_REQUIRED",
        "interpretation": {
            "note": (
                "This gate analyzes the untouched test-set predictions "
                "without modifying the model, thresholds, datasets, or "
                "production artifacts."
            ),
            "decision_basis": (
                "Per-technique recall, F1, false-negative concentration, "
                "false-positive concentration, rare-class support, and "
                "ranking-versus-threshold behavior must be reviewed before "
                "Phase 2 acceptance."
            ),
        },
        "outputs": {
            "per_technique": str(TECHNIQUE_REPORT),
            "false_negative_priority": str(FN_REPORT),
            "false_positive_priority": str(FP_REPORT),
            "rare_techniques": str(RARE_REPORT),
            "weak_techniques": str(WEAK_REPORT),
            "ranking_vs_threshold": str(RANKING_REPORT),
        },
    }

    with open(
        OVERALL_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("GATE 2.4F COMPLETE")
    print("=" * 78)

    print(f"Per-technique : {TECHNIQUE_REPORT}")
    print(f"FN priority   : {FN_REPORT}")
    print(f"FP priority   : {FP_REPORT}")
    print(f"Rare classes  : {RARE_REPORT}")
    print(f"Weak classes  : {WEAK_REPORT}")
    print(f"Ranking       : {RANKING_REPORT}")
    print(f"Report        : {OVERALL_REPORT}")

    print()
    print("No model, threshold, dataset, or production artifact was modified.")
    print()
    print("STATUS: ANALYSIS_COMPLETE")


if __name__ == "__main__":
    evaluate()