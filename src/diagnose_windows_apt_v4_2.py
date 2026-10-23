"""
SmartSOC Windows-APT Model B v4.2
=================================

Diagnostic and threshold analysis for the v4.2 base model.

This script:
    1. Loads v4.2 model artifacts.
    2. Rebuilds the v4.2 test feature matrix.
    3. Generates probabilities.
    4. Tests global thresholds from 0.05 to 0.95.
    5. Reports per-technique precision/recall/F1.
    6. Identifies false-positive-heavy techniques.
    7. Identifies techniques that are never predicted.
    8. Saves diagnostic CSV/JSON files.

IMPORTANT:
    This script does NOT modify the v4.2 model.
    This script does NOT calibrate thresholds.
    This script does NOT overwrite any model artifact.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================================
# PATHS
# ============================================================================

BASE = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "base"
)

RESULT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_2"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# HELPERS
# ============================================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text):
    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================================
# LABEL PARSING
# ============================================================================

def parse_labels(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return sorted(
        set(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )
    )


def get_label_column(df):

    if "techniques" in df.columns:
        return "techniques"

    if "label" in df.columns:
        return "label"

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        "Could not identify MITRE label column."
    )


# ============================================================================
# TEXT
# ============================================================================

def get_text_column(df):

    if "behavior_text" in df.columns:
        return df["behavior_text"]

    if "text" in df.columns:
        return df["text"]

    return df.iloc[:, 0]


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


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_everything():

    header(
        "SmartSOC Windows-APT Model B v4.2 - Diagnostic"
    )

    files = {

        "model":
            MODEL_DIR
            / "multilabel_logistic_v4_2_base.joblib",

        "encoder":
            MODEL_DIR
            / "structured_encoder_v4_2_base.joblib",

        "scaler":
            MODEL_DIR
            / "numerical_scaler_v4_2_base.joblib",

        "tfidf":
            MODEL_DIR
            / "tfidf_vectorizer_v4_2_base.joblib",

        "labels":
            MODEL_DIR
            / "multilabel_encoder_v4_2_base.joblib",

        "schema":
            MODEL_DIR
            / "feature_schema_v4_2_base.json",

        "structured_test":
            DATA_DIR
            / "structured_test.csv",

        "text_test":
            DATA_DIR
            / "behavior_text_test.csv",

        "labels_test":
            DATA_DIR
            / "labels_test.csv",
    }

    section("Checking required artifacts")

    for path in files.values():
        require_file(path)

    print()
    print("Loading model artifacts...")

    model = joblib.load(
        files["model"]
    )

    encoder = joblib.load(
        files["encoder"]
    )

    scaler = joblib.load(
        files["scaler"]
    )

    tfidf = joblib.load(
        files["tfidf"]
    )

    mlb = joblib.load(
        files["labels"]
    )

    with open(
        files["schema"],
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    structured_test = pd.read_csv(
        files["structured_test"]
    )

    text_test = pd.read_csv(
        files["text_test"]
    )

    labels_test = pd.read_csv(
        files["labels_test"]
    )

    print("[OK] Model loaded.")
    print("[OK] Structured encoder loaded.")
    print("[OK] Numerical scaler loaded.")
    print("[OK] TF-IDF loaded.")
    print("[OK] Label encoder loaded.")
    print("[OK] Feature schema loaded.")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        structured_test,
        text_test,
        labels_test,
    )


# ============================================================================
# FEATURE CONSTRUCTION
# ============================================================================

def build_features(
    structured_test,
    text_test,
    encoder,
    scaler,
    tfidf,
    schema,
):

    header(
        "Rebuilding v4.2 Test Feature Matrix"
    )

    categorical_columns = schema[
        "categorical_columns"
    ]

    numerical_columns = schema[
        "numerical_columns"
    ]

    print(
        "Categorical features:"
    )

    for column in categorical_columns:
        print(f"  - {column}")

    print(
        f"Numerical feature count: "
        f"{len(numerical_columns)}"
    )

    structured = (
        structured_test.copy()
    )

    # ------------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------------

    for column in categorical_columns:

        structured[column] = (
            structured[column]
            .fillna("<missing>")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    # ------------------------------------------------------------------------
    # Numerical
    # ------------------------------------------------------------------------

    for column in numerical_columns:

        structured[column] = pd.to_numeric(
            structured[column],
            errors="coerce",
        ).fillna(0)

    # ------------------------------------------------------------------------
    # One-hot categorical features
    # ------------------------------------------------------------------------

    X_cat = encoder.transform(
        structured[
            categorical_columns
        ]
    )

    # ------------------------------------------------------------------------
    # Scaled numerical features
    # ------------------------------------------------------------------------

    X_num = scaler.transform(
        structured[
            numerical_columns
        ]
    )

    X_num = sparse.csr_matrix(
        X_num
    )

    # ------------------------------------------------------------------------
    # Structured matrix
    # ------------------------------------------------------------------------

    X_struct = sparse.hstack(
        [
            X_cat,
            X_num,
        ],
        format="csr",
    )

    # ------------------------------------------------------------------------
    # Behavioral text
    # ------------------------------------------------------------------------

    text = normalize_text(
        get_text_column(
            text_test
        )
    )

    empty_text = int(
        (text == "").sum()
    )

    print(
        f"Empty behavior text: "
        f"{empty_text}"
    )

    X_text = tfidf.transform(
        text
    )

    # ------------------------------------------------------------------------
    # Hybrid matrix
    # ------------------------------------------------------------------------

    X = sparse.hstack(
        [
            X_struct,
            X_text,
        ],
        format="csr",
    )

    print()
    print(
        f"Categorical encoded features : "
        f"{X_cat.shape[1]}"
    )

    print(
        f"Numerical features           : "
        f"{X_num.shape[1]}"
    )

    print(
        f"TF-IDF features              : "
        f"{X_text.shape[1]}"
    )

    print(
        f"Generated hybrid features    : "
        f"{X.shape[1]}"
    )

    expected = int(
        schema[
            "final_feature_count"
        ]
    )

    print(
        f"Expected schema features     : "
        f"{expected}"
    )

    if X.shape[1] != expected:
        raise ValueError(
            "Feature dimension mismatch.\n"
            f"Expected: {expected}\n"
            f"Generated: {X.shape[1]}"
        )

    print(
        "[OK] Feature dimensions match."
    )

    return X


# ============================================================================
# LABEL MATRIX
# ============================================================================

def build_y(
    labels_test,
    mlb,
):

    section(
        "Building test label matrix"
    )

    column = get_label_column(
        labels_test
    )

    label_lists = (
        labels_test[column]
        .apply(parse_labels)
    )

    y_test = mlb.transform(
        label_lists
    )

    print(
        f"Test events: "
        f"{y_test.shape[0]}"
    )

    print(
        f"MITRE techniques: "
        f"{y_test.shape[1]}"
    )

    print(
        f"Total positive labels: "
        f"{int(y_test.sum())}"
    )

    return y_test


# ============================================================================
# PROBABILITY GENERATION
# ============================================================================

def generate_probabilities(
    model,
    X,
):

    header(
        "Generating v4.2 Probability Matrix"
    )

    probabilities = []

    total = len(
        model.estimators_
    )

    for index, estimator in enumerate(
        model.estimators_
    ):

        if estimator is None:

            probabilities.append(
                np.zeros(
                    X.shape[0],
                    dtype=float,
                )
            )

        else:

            probability = (
                estimator
                .predict_proba(X)
            )

            if probability.shape[1] == 1:

                probabilities.append(
                    np.zeros(
                        X.shape[0],
                        dtype=float,
                    )
                )

            else:

                probabilities.append(
                    probability[:, 1]
                )

        current = index + 1

        if (
            current % 5 == 0
            or current == total
        ):

            print(
                f"Processed "
                f"{current}/{total} techniques"
            )

    probabilities = np.column_stack(
        probabilities
    )

    print()
    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Probability matrix contains "
            "NaN or infinite values."
        )

    return probabilities


# ============================================================================
# GLOBAL THRESHOLD ANALYSIS
# ============================================================================

def global_threshold_analysis(
    y_true,
    probabilities,
):

    header(
        "Global Threshold Analysis"
    )

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    rows = []

    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold":
                    threshold,

                "micro_precision":
                    precision_score(
                        y_true,
                        y_pred,
                        average="micro",
                        zero_division=0,
                    ),

                "micro_recall":
                    recall_score(
                        y_true,
                        y_pred,
                        average="micro",
                        zero_division=0,
                    ),

                "micro_f1":
                    f1_score(
                        y_true,
                        y_pred,
                        average="micro",
                        zero_division=0,
                    ),

                "macro_precision":
                    precision_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),

                "macro_recall":
                    recall_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),

                "macro_f1":
                    f1_score(
                        y_true,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    ),

                "weighted_f1":
                    f1_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    ),

                "total_predictions":
                    int(y_pred.sum()),

                "events_with_prediction":
                    int(
                        (
                            y_pred.sum(axis=1)
                            > 0
                        ).sum()
                    ),

                "events_without_prediction":
                    int(
                        (
                            y_pred.sum(axis=1)
                            == 0
                        ).sum()
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================================
# PER TECHNIQUE DIAGNOSTIC
# ============================================================================

def technique_diagnostics(
    y_true,
    probabilities,
    classes,
    threshold=0.50,
):

    header(
        f"Per-Technique Diagnostic "
        f"at Threshold {threshold:.2f}"
    )

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    rows = []

    for index, technique in enumerate(
        classes
    ):

        yt = y_true[:, index]
        yp = y_pred[:, index]
        prob = probabilities[:, index]

        tp = int(
            np.sum(
                (yt == 1)
                & (yp == 1)
            )
        )

        fp = int(
            np.sum(
                (yt == 0)
                & (yp == 1)
            )
        )

        fn = int(
            np.sum(
                (yt == 1)
                & (yp == 0)
            )
        )

        tn = int(
            np.sum(
                (yt == 0)
                & (yp == 0)
            )
        )

        support = int(
            yt.sum()
        )

        predicted = int(
            yp.sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        positive_probs = (
            prob[yt == 1]
        )

        negative_probs = (
            prob[yt == 0]
        )

        rows.append(
            {
                "technique":
                    technique,

                "support":
                    support,

                "predicted":
                    predicted,

                "TP":
                    tp,

                "FP":
                    fp,

                "FN":
                    fn,

                "TN":
                    tn,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "max_probability":
                    float(
                        np.max(prob)
                    ),

                "mean_probability":
                    float(
                        np.mean(prob)
                    ),

                "positive_mean":
                    float(
                        np.mean(
                            positive_probs
                        )
                    )
                    if len(
                        positive_probs
                    )
                    else 0.0,

                "positive_median":
                    float(
                        np.median(
                            positive_probs
                        )
                    )
                    if len(
                        positive_probs
                    )
                    else 0.0,

                "negative_mean":
                    float(
                        np.mean(
                            negative_probs
                        )
                    )
                    if len(
                        negative_probs
                    )
                    else 0.0,

                "negative_p95":
                    float(
                        np.percentile(
                            negative_probs,
                            95,
                        )
                    )
                    if len(
                        negative_probs
                    )
                    else 0.0,

                "positive_p05":
                    float(
                        np.percentile(
                            positive_probs,
                            5,
                        )
                    )
                    if len(
                        positive_probs
                    )
                    else 0.0,
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================================
# FALSE POSITIVE ANALYSIS
# ============================================================================

def false_positive_analysis(
    diagnostics,
):

    header(
        "False-Positive Analysis"
    )

    fp = (
        diagnostics
        .sort_values(
            "FP",
            ascending=False,
        )
    )

    print(
        fp[
            [
                "technique",
                "support",
                "predicted",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return fp


# ============================================================================
# ZERO-PREDICTION / ZERO-RECALL ANALYSIS
# ============================================================================

def missed_techniques(
    diagnostics,
):

    header(
        "Missed / Under-Predicted Techniques"
    )

    missed = diagnostics[
        (
            diagnostics["support"] > 0
        )
        &
        (
            diagnostics["TP"] == 0
        )
    ].copy()

    if missed.empty:

        print(
            "[OK] Every supported technique "
            "has at least one true positive."
        )

    else:

        print(
            missed[
                [
                    "technique",
                    "support",
                    "predicted",
                    "TP",
                    "FP",
                    "FN",
                    "precision",
                    "recall",
                    "f1",
                    "positive_mean",
                    "positive_median",
                ]
            ]
            .sort_values(
                "support",
                ascending=False,
            )
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    return missed


# ============================================================================
# PROBABILITY SEPARATION
# ============================================================================

def probability_separation(
    diagnostics,
):

    header(
        "Probability Separation Analysis"
    )

    data = diagnostics.copy()

    data["separation"] = (
        data["positive_mean"]
        -
        data["negative_mean"]
    )

    data["robust_separation"] = (
        data["positive_p05"]
        -
        data["negative_p95"]
    )

    print(
        data[
            [
                "technique",
                "support",
                "positive_mean",
                "positive_p05",
                "negative_mean",
                "negative_p95",
                "separation",
                "robust_separation",
            ]
        ]
        .sort_values(
            "robust_separation",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return data


# ============================================================================
# TOP PROBABILITY ANALYSIS
# ============================================================================

def top_probability_analysis(
    y_true,
    probabilities,
    classes,
):

    header(
        "Top Probability Analysis"
    )

    rows = []

    for index, technique in enumerate(
        classes
    ):

        prob = probabilities[:, index]
        truth = y_true[:, index]

        rows.append(
            {
                "technique":
                    technique,

                "support":
                    int(truth.sum()),

                "max_probability":
                    float(
                        np.max(prob)
                    ),

                "p99":
                    float(
                        np.percentile(
                            prob,
                            99,
                        )
                    ),

                "p95":
                    float(
                        np.percentile(
                            prob,
                            95,
                        )
                    ),

                "p90":
                    float(
                        np.percentile(
                            prob,
                            90,
                        )
                    ),

                "mean":
                    float(
                        np.mean(prob)
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.sort_values(
            "max_probability",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return result


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    threshold_results,
    diagnostics,
    separation,
    top_probabilities,
):

    section(
        "Saving diagnostic results"
    )

    threshold_path = (
        RESULT_DIR
        / "threshold_analysis_v4_2.csv"
    )

    diagnostic_path = (
        RESULT_DIR
        / "per_technique_diagnostics_v4_2.csv"
    )

    separation_path = (
        RESULT_DIR
        / "probability_separation_v4_2.csv"
    )

    top_probability_path = (
        RESULT_DIR
        / "top_probability_analysis_v4_2.csv"
    )

    summary_path = (
        RESULT_DIR
        / "diagnostic_summary_v4_2.json"
    )

    threshold_results.to_csv(
        threshold_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostic_path,
        index=False,
    )

    separation.to_csv(
        separation_path,
        index=False,
    )

    top_probabilities.to_csv(
        top_probability_path,
        index=False,
    )

    best_micro = (
        threshold_results
        .loc[
            threshold_results[
                "micro_f1"
            ].idxmax()
        ]
        .to_dict()
    )

    best_macro = (
        threshold_results
        .loc[
            threshold_results[
                "macro_f1"
            ].idxmax()
        ]
        .to_dict()
    )

    best_weighted = (
        threshold_results
        .loc[
            threshold_results[
                "weighted_f1"
            ].idxmax()
        ]
        .to_dict()
    )

    false_positive_total = int(
        diagnostics["FP"].sum()
    )

    true_positive_total = int(
        diagnostics["TP"].sum()
    )

    false_negative_total = int(
        diagnostics["FN"].sum()
    )

    summary = {

        "model":
            "SmartSOC Windows-APT Model B v4.2",

        "status":
            "diagnostic_only",

        "best_global_micro_f1":
            best_micro,

        "best_global_macro_f1":
            best_macro,

        "best_global_weighted_f1":
            best_weighted,

        "total_true_positives_at_0_50":
            true_positive_total,

        "total_false_positives_at_0_50":
            false_positive_total,

        "total_false_negatives_at_0_50":
            false_negative_total,

        "zero_tp_supported_techniques":
            int(
                (
                    (
                        diagnostics["support"]
                        > 0
                    )
                    &
                    (
                        diagnostics["TP"]
                        == 0
                    )
                ).sum()
            ),

        "high_fp_techniques":
            diagnostics
            .sort_values(
                "FP",
                ascending=False,
            )
            .head(10)[
                "technique"
            ]
            .tolist(),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=float,
        )

    print(
        f"[OK] {threshold_path}"
    )

    print(
        f"[OK] {diagnostic_path}"
    )

    print(
        f"[OK] {separation_path}"
    )

    print(
        f"[OK] {top_probability_path}"
    )

    print(
        f"[OK] {summary_path}"
    )

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():

    (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        structured_test,
        text_test,
        labels_test,
    ) = load_everything()

    header(
        "Model Information"
    )

    print(
        f"Model type        : "
        f"{type(model).__name__}"
    )

    print(
        f"MITRE techniques  : "
        f"{len(mlb.classes_)}"
    )

    print(
        f"TF-IDF vocabulary : "
        f"{len(tfidf.vocabulary_)}"
    )

    print(
        f"Expected features : "
        f"{schema['final_feature_count']}"
    )

    X_test = build_features(
        structured_test,
        text_test,
        encoder,
        scaler,
        tfidf,
        schema,
    )

    y_test = build_y(
        labels_test,
        mlb,
    )

    probabilities = (
        generate_probabilities(
            model,
            X_test,
        )
    )

    # ------------------------------------------------------------------------
    # Global threshold analysis
    # ------------------------------------------------------------------------

    threshold_results = (
        global_threshold_analysis(
            y_test,
            probabilities,
        )
    )

    # ------------------------------------------------------------------------
    # Standard 0.50 diagnostic
    # ------------------------------------------------------------------------

    diagnostics = (
        technique_diagnostics(
            y_test,
            probabilities,
            mlb.classes_,
            threshold=0.50,
        )
    )

    # ------------------------------------------------------------------------
    # False positives
    # ------------------------------------------------------------------------

    fp = false_positive_analysis(
        diagnostics
    )

    # ------------------------------------------------------------------------
    # Missed techniques
    # ------------------------------------------------------------------------

    missed = missed_techniques(
        diagnostics
    )

    # ------------------------------------------------------------------------
    # Probability separation
    # ------------------------------------------------------------------------

    separation = probability_separation(
        diagnostics
    )

    # ------------------------------------------------------------------------
    # Probability distribution
    # ------------------------------------------------------------------------

    top_probabilities = (
        top_probability_analysis(
            y_test,
            probabilities,
            mlb.classes_,
        )
    )

    # ------------------------------------------------------------------------
    # Best thresholds
    # ------------------------------------------------------------------------

    header(
        "Best Global Thresholds"
    )

    best_micro = (
        threshold_results
        .sort_values(
            "micro_f1",
            ascending=False,
        )
        .iloc[0]
    )

    best_macro = (
        threshold_results
        .sort_values(
            "macro_f1",
            ascending=False,
        )
        .iloc[0]
    )

    best_weighted = (
        threshold_results
        .sort_values(
            "weighted_f1",
            ascending=False,
        )
        .iloc[0]
    )

    print(
        "Best Micro F1:"
    )

    print(
        f"  threshold = "
        f"{best_micro['threshold']:.2f}"
    )

    print(
        f"  Micro F1  = "
        f"{best_micro['micro_f1']:.4f}"
    )

    print(
        f"  Precision = "
        f"{best_micro['micro_precision']:.4f}"
    )

    print(
        f"  Recall    = "
        f"{best_micro['micro_recall']:.4f}"
    )

    print()

    print(
        "Best Macro F1:"
    )

    print(
        f"  threshold = "
        f"{best_macro['threshold']:.2f}"
    )

    print(
        f"  Macro F1  = "
        f"{best_macro['macro_f1']:.4f}"
    )

    print(
        f"  Precision = "
        f"{best_macro['macro_precision']:.4f}"
    )

    print(
        f"  Recall    = "
        f"{best_macro['macro_recall']:.4f}"
    )

    print()

    print(
        "Best Weighted F1:"
    )

    print(
        f"  threshold = "
        f"{best_weighted['threshold']:.2f}"
    )

    print(
        f"  Weighted F1 = "
        f"{best_weighted['weighted_f1']:.4f}"
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    summary = save_results(
        threshold_results,
        diagnostics,
        separation,
        top_probabilities,
    )

    # ------------------------------------------------------------------------
    # Final conclusion
    # ------------------------------------------------------------------------

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "Diagnostic Complete"
    )

    print(
        f"Best Micro F1 threshold : "
        f"{best_micro['threshold']:.2f}"
    )

    print(
        f"Best Micro F1           : "
        f"{best_micro['micro_f1']:.4f}"
    )

    print(
        f"Best Macro F1 threshold : "
        f"{best_macro['threshold']:.2f}"
    )

    print(
        f"Best Macro F1           : "
        f"{best_macro['macro_f1']:.4f}"
    )

    print(
        f"Best Weighted F1        : "
        f"{best_weighted['weighted_f1']:.4f}"
    )

    print(
        f"Supported techniques "
        f"with zero TP at 0.50 : "
        f"{len(missed)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No model artifacts were modified."
    )

    print(
        "No threshold calibration was performed."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Review the diagnostic output and then "
        "perform technique-specific calibration."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()