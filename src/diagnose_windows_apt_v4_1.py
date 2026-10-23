"""
SmartSOC Windows-APT Model B v4.1
False-Positive / Model Diagnostic

Purpose:
    Diagnose why v4.1 inference predicts unrelated MITRE techniques.

Checks:
    1. Test-set predictions using calibrated thresholds
    2. Per-technique TP / FP / FN / TN
    3. Precision / Recall / F1
    4. False-positive rate
    5. Prediction prevalence vs actual prevalence
    6. Techniques frequently predicted on negative examples
    7. Confidence of false positives
    8. Label co-occurrence patterns
    9. Threshold distribution
   10. Saves diagnostic CSV/JSON reports

This script DOES NOT modify the trained model.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "windows_apt_v4_1"
    / "calibrated"
)

RESULT_DIR = (
    BASE_DIR
    / "results"
    / "windows_apt_v4_1"
    / "diagnostic"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# FILE PATHS
# ============================================================================

STRUCTURED_TEST = (
    DATA_DIR / "structured_test.csv"
)

TEXT_TEST = (
    DATA_DIR / "behavior_text_test.csv"
)

LABELS_TEST = (
    DATA_DIR / "labels_test.csv"
)

METADATA_TEST = (
    DATA_DIR / "metadata_test.csv"
)

MODEL_PATH = (
    MODEL_DIR
    / "multilabel_logistic_v4_1_calibrated.joblib"
)

ENCODER_PATH = (
    MODEL_DIR
    / "structured_encoder_v4_1_calibrated.joblib"
)

SCALER_PATH = (
    MODEL_DIR
    / "numerical_scaler_v4_1_calibrated.joblib"
)

TFIDF_PATH = (
    MODEL_DIR
    / "tfidf_vectorizer_v4_1_calibrated.joblib"
)

LABEL_ENCODER_PATH = (
    MODEL_DIR
    / "multilabel_encoder_v4_1_calibrated.joblib"
)

SCHEMA_PATH = (
    MODEL_DIR
    / "feature_schema_v4_1_calibrated.json"
)

THRESHOLD_PATH = (
    MODEL_DIR
    / "technique_thresholds_v4_1.json"
)


# ============================================================================
# HELPERS
# ============================================================================

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


def parse_labels(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if text.lower() in {
        "techniques",
        "nan",
        "none",
    }:
        return []

    result = []

    for item in text.replace(
        ";",
        ",",
    ).split(","):

        item = item.strip().upper()

        if item.startswith("T"):
            result.append(item)

    return sorted(set(result))


def load_labels(path):
    df = pd.read_csv(path)

    candidates = [
        c
        for c in df.columns
        if not str(c).lower().startswith(
            "unnamed"
        )
    ]

    if "techniques" in df.columns:
        column = "techniques"

    elif candidates:
        column = candidates[0]

    else:
        raise ValueError(
            f"Could not identify label column in {path}"
        )

    return df[column].apply(
        parse_labels
    )


def build_structured(
    df,
    categorical_columns,
    numerical_columns,
    encoder,
    scaler,
):

    cat = df[
        categorical_columns
    ].copy()

    for column in categorical_columns:
        cat[column] = (
            cat[column]
            .fillna("<missing>")
            .astype(str)
        )

    cat_matrix = encoder.transform(
        cat
    )

    if not sparse.issparse(
        cat_matrix
    ):
        cat_matrix = sparse.csr_matrix(
            cat_matrix
        )

    numerical = df[
        numerical_columns
    ].copy()

    for column in numerical_columns:
        numerical[column] = pd.to_numeric(
            numerical[column],
            errors="coerce",
        ).fillna(0.0)

    numerical_matrix = scaler.transform(
        numerical
    )

    numerical_matrix = sparse.csr_matrix(
        numerical_matrix
    )

    return sparse.hstack(
        [
            cat_matrix,
            numerical_matrix,
        ],
        format="csr",
    )


def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )


def build_hybrid(
    structured,
    text,
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns,
):

    X_structured = build_structured(
        structured,
        categorical_columns,
        numerical_columns,
        encoder,
        scaler,
    )

    X_text = tfidf.transform(
        normalize_text(
            text.iloc[:, 0]
        )
    )

    return sparse.hstack(
        [
            X_structured,
            X_text,
        ],
        format="csr",
    )


def get_threshold(
    technique,
    thresholds,
):
    value = thresholds.get(
        technique,
        0.5,
    )

    try:
        return float(value)
    except Exception:
        return 0.5


# ============================================================================
# MAIN
# ============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1"
    )

    print(
        "FALSE-POSITIVE / MODEL DIAGNOSTIC"
    )

    # ------------------------------------------------------------------------
    # CHECK ARTIFACTS
    # ------------------------------------------------------------------------

    subheader(
        "Checking required artifacts..."
    )

    files = [
        STRUCTURED_TEST,
        TEXT_TEST,
        LABELS_TEST,
        METADATA_TEST,
        MODEL_PATH,
        ENCODER_PATH,
        SCALER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        SCHEMA_PATH,
        THRESHOLD_PATH,
    ]

    for path in files:
        require_file(path)

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    header(
        "Loading test dataset and artifacts"
    )

    structured_test = pd.read_csv(
        STRUCTURED_TEST
    )

    text_test = pd.read_csv(
        TEXT_TEST
    )

    labels_test = load_labels(
        LABELS_TEST
    )

    metadata_test = pd.read_csv(
        METADATA_TEST
    )

    model = joblib.load(
        MODEL_PATH
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        threshold_data = json.load(f)

    thresholds = threshold_data.get(
        "thresholds",
        {},
    )

    techniques = list(
        label_encoder.classes_
    )

    print(
        "Test rows:",
        len(structured_test),
    )

    print(
        "MITRE techniques:",
        len(techniques),
    )

    # ------------------------------------------------------------------------
    # ALIGNMENT
    # ------------------------------------------------------------------------

    header(
        "Checking alignment"
    )

    lengths = [
        len(structured_test),
        len(text_test),
        len(labels_test),
        len(metadata_test),
    ]

    print(
        "Structured:",
        lengths[0],
    )

    print(
        "Text:",
        lengths[1],
    )

    print(
        "Labels:",
        lengths[2],
    )

    print(
        "Metadata:",
        lengths[3],
    )

    if len(set(lengths)) != 1:
        raise ValueError(
            "Test datasets are not aligned."
        )

    print(
        "[OK] Test datasets aligned."
    )

    # ------------------------------------------------------------------------
    # FEATURE SCHEMA
    # ------------------------------------------------------------------------

    categorical_columns = schema[
        "categorical_columns"
    ]

    numerical_columns = schema[
        "numerical_columns"
    ]

    print()
    print(
        "Categorical features:",
        categorical_columns,
    )

    print(
        "Numerical features:",
        len(numerical_columns),
    )

    # ------------------------------------------------------------------------
    # BUILD FEATURES
    # ------------------------------------------------------------------------

    header(
        "Building test feature matrix"
    )

    X_test = build_hybrid(
        structured_test,
        text_test,
        encoder,
        scaler,
        tfidf,
        categorical_columns,
        numerical_columns,
    )

    print(
        "Generated features:",
        X_test.shape,
    )

    expected = schema.get(
        "final_feature_count"
    )

    print(
        "Expected features:",
        expected,
    )

    if expected is not None:
        if X_test.shape[1] != int(
            expected
        ):
            raise ValueError(
                "Feature dimension mismatch."
            )

    if hasattr(
        model,
        "n_features_in_"
    ):
        print(
            "Model features:",
            model.n_features_in_,
        )

        if (
            model.n_features_in_
            != X_test.shape[1]
        ):
            raise ValueError(
                "Model/test feature mismatch."
            )

    print(
        "[OK] Feature alignment verified."
    )

    # ------------------------------------------------------------------------
    # TRUE LABEL MATRIX
    # ------------------------------------------------------------------------

    header(
        "Preparing ground-truth labels"
    )

    y_true = label_encoder.transform(
        labels_test
    )

    print(
        "Y test:",
        y_true.shape,
    )

    # ------------------------------------------------------------------------
    # PROBABILITIES
    # ------------------------------------------------------------------------

    header(
        "Generating test probabilities"
    )

    probabilities = model.predict_proba(
        X_test
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(
            1,
            -1,
        )

    print(
        "Probability matrix:",
        probabilities.shape,
    )

    # ------------------------------------------------------------------------
    # APPLY CALIBRATED THRESHOLDS
    # ------------------------------------------------------------------------

    header(
        "Applying calibrated thresholds"
    )

    y_pred = np.zeros_like(
        probabilities,
        dtype=np.int8,
    )

    for index, technique in enumerate(
        techniques
    ):

        threshold = get_threshold(
            technique,
            thresholds,
        )

        y_pred[:, index] = (
            probabilities[:, index]
            >= threshold
        ).astype(np.int8)

    # ------------------------------------------------------------------------
    # FALLBACK INFORMATION
    # ------------------------------------------------------------------------

    no_prediction = (
        y_pred.sum(axis=1) == 0
    )

    fallback_count = int(
        no_prediction.sum()
    )

    print(
        "Events with no prediction:",
        fallback_count,
    )

    # IMPORTANT:
    # For this diagnostic we DO NOT apply
    # fallback predictions.
    #
    # We want to measure genuine classifier
    # behavior without artificially converting
    # every event into a positive prediction.

    # ------------------------------------------------------------------------
    # GLOBAL METRICS
    # ------------------------------------------------------------------------

    header(
        "Global diagnostic metrics"
    )

    metrics = {
        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "weighted_recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    for key, value in metrics.items():
        print(
            f"{key:<22}: {value:.4f}"
        )

    # ------------------------------------------------------------------------
    # PER TECHNIQUE ANALYSIS
    # ------------------------------------------------------------------------

    header(
        "Per-technique false-positive analysis"
    )

    rows = []

    for index, technique in enumerate(
        techniques
    ):

        actual = y_true[:, index]
        predicted = y_pred[:, index]
        probability = probabilities[
            :,
            index,
        ]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).ravel()

        support = int(
            actual.sum()
        )

        predicted_count = int(
            predicted.sum()
        )

        threshold = get_threshold(
            technique,
            thresholds,
        )

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        mean_fp_confidence = (
            float(
                probability[
                    (predicted == 1)
                    & (actual == 0)
                ].mean()
            )
            if fp > 0
            else 0.0
        )

        max_fp_confidence = (
            float(
                probability[
                    (predicted == 1)
                    & (actual == 0)
                ].max()
            )
            if fp > 0
            else 0.0
        )

        rows.append(
            {
                "technique": technique,
                "threshold": threshold,
                "support": support,
                "predicted_count": predicted_count,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(f1),
                "false_positive_rate": float(
                    false_positive_rate
                ),
                "mean_fp_confidence": (
                    mean_fp_confidence
                ),
                "max_fp_confidence": (
                    max_fp_confidence
                ),
            }
        )

    technique_df = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------------------
    # WORST FALSE POSITIVES
    # ------------------------------------------------------------------------

    subheader(
        "Techniques with highest false-positive count"
    )

    worst_fp = (
        technique_df
        .sort_values(
            [
                "FP",
                "mean_fp_confidence",
            ],
            ascending=False,
        )
        .head(15)
    )

    print(
        worst_fp[
            [
                "technique",
                "threshold",
                "support",
                "predicted_count",
                "FP",
                "precision",
                "recall",
                "f1",
                "mean_fp_confidence",
                "max_fp_confidence",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # LOWEST PRECISION
    # ------------------------------------------------------------------------

    subheader(
        "Techniques with lowest precision"
    )

    low_precision = (
        technique_df[
            technique_df["predicted_count"] > 0
        ]
        .sort_values(
            "precision",
            ascending=True,
        )
        .head(15)
    )

    print(
        low_precision[
            [
                "technique",
                "threshold",
                "support",
                "predicted_count",
                "FP",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # WORST F1
    # ------------------------------------------------------------------------

    subheader(
        "Techniques with lowest F1"
    )

    worst_f1 = (
        technique_df[
            technique_df["support"] > 0
        ]
        .sort_values(
            "f1",
            ascending=True,
        )
        .head(15)
    )

    print(
        worst_f1[
            [
                "technique",
                "threshold",
                "support",
                "predicted_count",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # OVERPREDICTION RATIO
    # ------------------------------------------------------------------------

    subheader(
        "Most over-predicted techniques"
    )

    technique_df[
        "prediction_to_actual_ratio"
    ] = (
        technique_df[
            "predicted_count"
        ]
        /
        technique_df[
            "support"
        ].replace(
            0,
            np.nan,
        )
    )

    overpredicted = (
        technique_df
        .sort_values(
            "prediction_to_actual_ratio",
            ascending=False,
        )
        .head(15)
    )

    print(
        overpredicted[
            [
                "technique",
                "support",
                "predicted_count",
                "prediction_to_actual_ratio",
                "FP",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # FALSE POSITIVE EVENT ANALYSIS
    # ------------------------------------------------------------------------

    header(
        "False-positive event analysis"
    )

    fp_event_counts = np.sum(
        (
            y_pred == 1
        )
        &
        (
            y_true == 0
        ),
        axis=1,
    )

    print(
        "Events containing at least one FP:",
        int(
            np.sum(
                fp_event_counts > 0
            )
        ),
    )

    print(
        "Total false-positive labels:",
        int(
            fp_event_counts.sum()
        ),
    )

    print(
        "Mean FP labels/event:",
        float(
            fp_event_counts.mean()
        ),
    )

    # ------------------------------------------------------------------------
    # TOP FALSE-POSITIVE EVENTS
    # ------------------------------------------------------------------------

    subheader(
        "Events with most false-positive techniques"
    )

    top_fp_indices = np.argsort(
        fp_event_counts
    )[::-1][:20]

    fp_event_rows = []

    for row_index in top_fp_indices:

        count = int(
            fp_event_counts[
                row_index
            ]
        )

        if count == 0:
            continue

        actual_labels = [
            techniques[i]
            for i in range(
                len(techniques)
            )
            if y_true[
                row_index,
                i
            ] == 1
        ]

        predicted_labels = [
            techniques[i]
            for i in range(
                len(techniques)
            )
            if y_pred[
                row_index,
                i
            ] == 1
        ]

        fp_labels = [
            techniques[i]
            for i in range(
                len(techniques)
            )
            if (
                y_pred[
                    row_index,
                    i
                ] == 1
                and
                y_true[
                    row_index,
                    i
                ] == 0
            )
        ]

        fp_event_rows.append(
            {
                "row": int(
                    row_index
                ),
                "event_id": (
                    structured_test.iloc[
                        row_index
                    ].get(
                        "event_id",
                        "",
                    )
                ),
                "actual_labels": ",".join(
                    actual_labels
                ),
                "predicted_labels": ",".join(
                    predicted_labels
                ),
                "false_positive_labels": ",".join(
                    fp_labels
                ),
                "fp_count": count,
                "behavior_text": str(
                    text_test.iloc[
                        row_index,
                        0,
                    ]
                )[:1000],
            }
        )

    fp_event_df = pd.DataFrame(
        fp_event_rows
    )

    if not fp_event_df.empty:
        print(
            fp_event_df.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # LABEL CO-OCCURRENCE
    # ------------------------------------------------------------------------

    header(
        "Actual MITRE label co-occurrence analysis"
    )

    cooccurrence = {}

    for row in labels_test:

        labels = sorted(
            set(row)
        )

        if len(labels) < 2:
            continue

        for i in range(
            len(labels)
        ):
            for j in range(
                i + 1,
                len(labels),
            ):

                pair = (
                    labels[i],
                    labels[j],
                )

                cooccurrence[
                    pair
                ] = (
                    cooccurrence.get(
                        pair,
                        0,
                    )
                    + 1
                )

    co_rows = []

    for pair, count in sorted(
        cooccurrence.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:30]:

        co_rows.append(
            {
                "technique_a": pair[0],
                "technique_b": pair[1],
                "count": count,
            }
        )

    co_df = pd.DataFrame(
        co_rows
    )

    if not co_df.empty:
        print(
            co_df.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------------
    # THRESHOLD REPORT
    # ------------------------------------------------------------------------

    header(
        "Threshold analysis"
    )

    threshold_df = technique_df[
        [
            "technique",
            "threshold",
            "support",
            "predicted_count",
            "FP",
            "precision",
            "recall",
            "f1",
        ]
    ].sort_values(
        "threshold"
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # SAVE REPORTS
    # ------------------------------------------------------------------------

    header(
        "Saving diagnostic reports"
    )

    technique_csv = (
        RESULT_DIR
        / "per_technique_diagnostics.csv"
    )

    technique_df.to_csv(
        technique_csv,
        index=False,
    )

    print(
        "[OK]",
        technique_csv,
    )

    fp_events_csv = (
        RESULT_DIR
        / "false_positive_events.csv"
    )

    fp_event_df.to_csv(
        fp_events_csv,
        index=False,
    )

    print(
        "[OK]",
        fp_events_csv,
    )

    cooccurrence_csv = (
        RESULT_DIR
        / "label_cooccurrence.csv"
    )

    co_df.to_csv(
        cooccurrence_csv,
        index=False,
    )

    print(
        "[OK]",
        cooccurrence_csv,
    )

    # ------------------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------------------

    high_fp = technique_df[
        technique_df["FP"] > 0
    ].sort_values(
        "FP",
        ascending=False,
    ).head(10)

    high_overprediction = technique_df[
        technique_df["predicted_count"] > 0
    ].sort_values(
        "prediction_to_actual_ratio",
        ascending=False,
    ).head(10)

    summary = {
        "model": (
            "SmartSOC Windows-APT Model B v4.1"
        ),
        "test_events": int(
            len(structured_test)
        ),
        "technique_count": int(
            len(techniques)
        ),
        "feature_count": int(
            X_test.shape[1]
        ),
        "fallback_events": fallback_count,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "total_false_positive_labels": int(
            fp_event_counts.sum()
        ),
        "events_with_false_positives": int(
            np.sum(
                fp_event_counts > 0
            )
        ),
        "top_false_positive_techniques": (
            high_fp[
                [
                    "technique",
                    "FP",
                    "precision",
                    "recall",
                    "f1",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),
        "top_overpredicted_techniques": (
            high_overprediction[
                [
                    "technique",
                    "support",
                    "predicted_count",
                    "prediction_to_actual_ratio",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),
    }

    summary_json = (
        RESULT_DIR
        / "diagnostic_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "[OK]",
        summary_json,
    )

    # ------------------------------------------------------------------------
    # FINAL INTERPRETATION
    # ------------------------------------------------------------------------

    header(
        "v4.1 Diagnostic Complete"
    )

    print(
        "The model was NOT modified."
    )

    print()
    print(
        "Review these three files first:"
    )

    print(
        f"1. {technique_csv}"
    )

    print(
        f"2. {fp_events_csv}"
    )

    print(
        f"3. {summary_json}"
    )

    print()
    print(
        "Important:"
    )

    print(
        "Do NOT retrain or deploy v4.2 until "
        "the dominant false-positive techniques "
        "have been identified."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()