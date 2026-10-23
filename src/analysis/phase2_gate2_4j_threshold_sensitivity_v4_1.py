"""
SmartSOC Phase 2 Gate 2.4J
Threshold Sensitivity Analysis

Purpose:
    Evaluate alternative thresholds for weak MITRE ATT&CK techniques
    using the untouched test set.

IMPORTANT:
    - Does NOT modify production model artifacts.
    - Does NOT modify thresholds.json.
    - Does NOT retrain the model.
    - Does NOT modify the dataset.
    - Uses the existing production probability outputs.
    - Test labels remain untouched; this is analysis only.

Output:
    reports/phase2_gate2_4j/
        threshold_sensitivity.csv
        threshold_recommendations.csv
        phase2_gate2_4j_report.json
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4_1"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
STRUCTURED_ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"

REPORT_DIR = ROOT / "reports" / "phase2_gate2_4j"


# ============================================================================
# TECHNIQUES REQUIRING ANALYSIS
# ============================================================================

TARGET_TECHNIQUES = [
    "T1078",
    "T1112",
    "T1565.001",
    "T1485",
    "T1531",
    "T1562.001",
    "T1070.004",
    "T1548.002",
    "T1059.001",
    "T1543.003",
]


# Thresholds to test.
# We deliberately test a broad range for diagnostic purposes.
THRESHOLDS = [
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


# ============================================================================
# HELPERS
# ============================================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_artifacts():
    banner("PRODUCTION ARTIFACT VALIDATION")

    required = [
        MODEL_FILE,
        STRUCTURED_ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLD_FILE,
        STRUCTURED_TEST,
        LABELS_TEST,
        BEHAVIOR_TEST,
    ]

    for path in required:
        validate_file(path)

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(STRUCTURED_ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        threshold_artifact = json.load(f)

    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]

    structured_columns = (
        categorical_columns +
        numerical_columns
    )

    mitre_classes = list(label_encoder.classes_)

    print(f"Model type           : {type(model).__name__}")
    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns   : {len(structured_columns)}")
    print(f"TF-IDF vocabulary    : {len(tfidf.vocabulary_)}")
    print(f"MITRE classes        : {len(mitre_classes)}")
    print(
        f"Threshold entries    : "
        f"{len(threshold_artifact.get('thresholds', {}))}"
    )

    if len(structured_columns) != 62:
        raise RuntimeError(
            f"Expected 62 structured columns, got {len(structured_columns)}"
        )

    if len(numerical_columns) != 58:
        raise RuntimeError(
            f"Expected 58 numerical columns, got {len(numerical_columns)}"
        )

    if len(mitre_classes) != 45:
        raise RuntimeError(
            f"Expected 45 MITRE classes, got {len(mitre_classes)}"
        )

    for technique in TARGET_TECHNIQUES:
        if technique not in mitre_classes:
            raise RuntimeError(
                f"Target technique missing from model: {technique}"
            )

    print("[PASS] Production artifacts validated")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        threshold_artifact,
    )


# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

def transform_test_data(
    structured,
    behavior,
    model,
    encoder,
    scaler,
    tfidf,
    schema,
):
    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]
    behavior_text_column = schema.get(
        "behavior_text_column",
        "behavior_text_normalized",
    )

    structured_columns = (
        categorical_columns +
        numerical_columns
    )

    # ------------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------------

    missing = [
        c for c in structured_columns
        if c not in structured.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing structured columns: {missing}"
        )

    if behavior_text_column not in behavior.columns:
        raise RuntimeError(
            f"Missing behavior text column: {behavior_text_column}"
        )

    structured = structured[structured_columns].copy()

    # ------------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------------

    X_cat = encoder.transform(
        structured[categorical_columns]
    )

    # ------------------------------------------------------------------------
    # Numerical
    # ------------------------------------------------------------------------

    X_num = scaler.transform(
        structured[numerical_columns]
    )

    # ------------------------------------------------------------------------
    # Behavior text
    # ------------------------------------------------------------------------

    behavior_text = (
        behavior[behavior_text_column]
        .fillna("")
        .astype(str)
    )

    X_text = tfidf.transform(behavior_text)

    # ------------------------------------------------------------------------
    # Final model matrix
    # ------------------------------------------------------------------------

    from scipy.sparse import hstack

    X = hstack(
        [
            X_cat,
            X_num,
            X_text,
        ],
        format="csr",
    )

    return X


# ============================================================================
# LABEL PARSING
# ============================================================================

def parse_multilabel_column(series):
    """
    Convert labels such as:

        T1059.003
        T1059.003,T1105
        T1070.004,T1112,T1485

    into sets.
    """

    parsed = []

    for value in series.fillna("").astype(str):

        value = value.strip()

        if not value:
            parsed.append(set())
            continue

        parts = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        parsed.append(set(parts))

    return parsed


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def evaluate():

    banner("SMARTSOC PHASE 2 GATE 2.4J")
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 78)

    (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        threshold_artifact,
    ) = load_artifacts()

    # ------------------------------------------------------------------------
    # Load untouched test set
    # ------------------------------------------------------------------------

    banner("LOADING UNTOUCHED TEST DATA")

    structured_test = pd.read_csv(STRUCTURED_TEST)
    labels_test = pd.read_csv(LABELS_TEST)
    behavior_test = pd.read_csv(BEHAVIOR_TEST)

    print(f"Test rows       : {len(structured_test)}")
    print(f"Label rows      : {len(labels_test)}")
    print(f"Behavior rows   : {len(behavior_test)}")

    if not (
        len(structured_test)
        == len(labels_test)
        == len(behavior_test)
    ):
        raise RuntimeError(
            "Structured, label, and behavior row counts do not match."
        )

    print("[PASS] Test row alignment")

    # ------------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------------

    banner("PRODUCTION TRANSFORMATION")

    X_test = transform_test_data(
        structured_test,
        behavior_test,
        model,
        encoder,
        scaler,
        tfidf,
        schema,
    )

    print(f"Final feature matrix : {X_test.shape}")

    expected_features = (
        len(schema["numerical_columns"])
        + len(schema["categorical_columns"])
        and 6919
    )

    # Safer validation directly against production artifact expectation.
    if X_test.shape[1] != 6919:
        raise RuntimeError(
            f"Expected 6919 final features, got {X_test.shape[1]}"
        )

    print("[PASS] Final feature dimension = 6919")

    # ------------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------------

    banner("RUNNING PRODUCTION INFERENCE")

    probabilities = model.predict_proba(X_test)

    # OneVsRest can return list-like structures in some versions.
    if isinstance(probabilities, list):
        probabilities = np.column_stack(
            [
                np.asarray(x).reshape(-1)
                for x in probabilities
            ]
        )

    probabilities = np.asarray(probabilities)

    print(f"Probability matrix : {probabilities.shape}")

    if probabilities.shape != (
        len(structured_test),
        len(label_encoder.classes_),
    ):
        raise RuntimeError(
            "Probability matrix dimensions do not match test rows/classes."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probability matrix contains NaN or infinite values."
        )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise RuntimeError(
            "Probability matrix contains values outside [0,1]."
        )

    print("[PASS] Probability matrix valid")

    # ------------------------------------------------------------------------
    # Build label matrix
    # ------------------------------------------------------------------------

    true_label_sets = parse_multilabel_column(
        labels_test["techniques"]
    )

    mitre_classes = list(label_encoder.classes_)

    class_index = {
        technique: idx
        for idx, technique in enumerate(mitre_classes)
    }

    y_true = np.zeros(
        (
            len(true_label_sets),
            len(mitre_classes),
        ),
        dtype=int,
    )

    for row_idx, label_set in enumerate(true_label_sets):
        for technique in label_set:
            if technique in class_index:
                y_true[
                    row_idx,
                    class_index[technique],
                ] = 1

    # ------------------------------------------------------------------------
    # Existing production thresholds
    # ------------------------------------------------------------------------

    production_thresholds = threshold_artifact.get(
        "thresholds",
        {},
    )

    # ------------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------------

    banner("THRESHOLD SENSITIVITY")

    results = []

    for technique in TARGET_TECHNIQUES:

        idx = class_index[technique]

        y = y_true[:, idx]
        p = probabilities[:, idx]

        support = int(y.sum())

        existing_threshold = float(
            production_thresholds.get(
                technique,
                0.5,
            )
        )

        print()
        print("-" * 78)
        print(f"{technique}")
        print("-" * 78)

        print(
            f"Support              : {support}"
        )

        print(
            f"Existing threshold   : "
            f"{existing_threshold:.3f}"
        )

        best_f1 = -1.0
        best_row = None

        for threshold in THRESHOLDS:

            y_pred = (
                p >= threshold
            ).astype(int)

            precision = precision_score(
                y,
                y_pred,
                zero_division=0,
            )

            recall = recall_score(
                y,
                y_pred,
                zero_division=0,
            )

            f1 = f1_score(
                y,
                y_pred,
                zero_division=0,
            )

            tp = int(
                np.sum(
                    (y == 1)
                    & (y_pred == 1)
                )
            )

            fp = int(
                np.sum(
                    (y == 0)
                    & (y_pred == 1)
                )
            )

            fn = int(
                np.sum(
                    (y == 1)
                    & (y_pred == 0)
                )
            )

            tn = int(
                np.sum(
                    (y == 0)
                    & (y_pred == 0)
                )
            )

            predicted_positive = int(
                y_pred.sum()
            )

            row = {
                "technique": technique,
                "threshold": threshold,
                "existing_threshold": existing_threshold,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "predicted_positive": predicted_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

            results.append(row)

            if f1 > best_f1:
                best_f1 = f1
                best_row = row

        # Current production threshold metrics
        current_pred = (
            p >= existing_threshold
        ).astype(int)

        current_precision = precision_score(
            y,
            current_pred,
            zero_division=0,
        )

        current_recall = recall_score(
            y,
            current_pred,
            zero_division=0,
        )

        current_f1 = f1_score(
            y,
            current_pred,
            zero_division=0,
        )

        print(
            f"Current P/R/F1      : "
            f"{current_precision:.4f} / "
            f"{current_recall:.4f} / "
            f"{current_f1:.4f}"
        )

        if best_row is not None:

            print(
                f"Best F1 threshold   : "
                f"{best_row['threshold']:.2f}"
            )

            print(
                f"Best P/R/F1         : "
                f"{best_row['precision']:.4f} / "
                f"{best_row['recall']:.4f} / "
                f"{best_row['f1']:.4f}"
            )

            print(
                f"Best TP/FP/FN       : "
                f"{best_row['tp']} / "
                f"{best_row['fp']} / "
                f"{best_row['fn']}"
            )

    # ------------------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------------------

    sensitivity_df = pd.DataFrame(results)

    # ------------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------------

    recommendations = []

    for technique in TARGET_TECHNIQUES:

        subset = sensitivity_df[
            sensitivity_df["technique"] == technique
        ].copy()

        existing_threshold = float(
            subset["existing_threshold"].iloc[0]
        )

        current_row = subset[
            np.isclose(
                subset["threshold"],
                existing_threshold,
            )
        ]

        if len(current_row) == 0:

            current_row = subset.iloc[
                [
                    (
                        subset["threshold"]
                        - existing_threshold
                    ).abs().argmin()
                ]
            ]

        else:
            current_row = current_row.iloc[0]

        best_idx = subset["f1"].idxmax()
        best_row = subset.loc[best_idx]

        # Find the highest-recall threshold while maintaining
        # at least 0.90 precision, if such a threshold exists.
        precision_safe = subset[
            subset["precision"] >= 0.90
        ]

        if len(precision_safe) > 0:

            safe_best = precision_safe.sort_values(
                [
                    "recall",
                    "f1",
                ],
                ascending=False,
            ).iloc[0]

            safe_threshold = float(
                safe_best["threshold"]
            )

        else:

            safe_threshold = None

        if (
            float(best_row["f1"])
            > float(current_row["f1"]) + 0.02
        ):
            diagnosis = "THRESHOLD_SENSITIVITY_SIGNIFICANT"

        elif (
            float(best_row["f1"])
            > float(current_row["f1"]) + 0.005
        ):
            diagnosis = "THRESHOLD_SENSITIVITY_MODERATE"

        else:
            diagnosis = "THRESHOLD_NOT_PRIMARY_ISSUE"

        recommendations.append(
            {
                "technique": technique,
                "existing_threshold": existing_threshold,
                "current_precision": float(
                    current_row["precision"]
                ),
                "current_recall": float(
                    current_row["recall"]
                ),
                "current_f1": float(
                    current_row["f1"]
                ),
                "best_f1_threshold": float(
                    best_row["threshold"]
                ),
                "best_f1_precision": float(
                    best_row["precision"]
                ),
                "best_f1_recall": float(
                    best_row["recall"]
                ),
                "best_f1": float(
                    best_row["f1"]
                ),
                "precision_ge_0_90_threshold": (
                    safe_threshold
                ),
                "diagnosis": diagnosis,
            }
        )

    recommendations_df = pd.DataFrame(
        recommendations
    )

    # ------------------------------------------------------------------------
    # Print recommendations
    # ------------------------------------------------------------------------

    banner("THRESHOLD RECOMMENDATION SUMMARY")

    print(
        recommendations_df[
            [
                "technique",
                "existing_threshold",
                "current_precision",
                "current_recall",
                "current_f1",
                "best_f1_threshold",
                "best_f1_precision",
                "best_f1_recall",
                "best_f1",
                "precision_ge_0_90_threshold",
                "diagnosis",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensitivity_path = (
        REPORT_DIR
        / "threshold_sensitivity.csv"
    )

    recommendation_path = (
        REPORT_DIR
        / "threshold_recommendations.csv"
    )

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4j_report.json"
    )

    sensitivity_df.to_csv(
        sensitivity_path,
        index=False,
    )

    recommendations_df.to_csv(
        recommendation_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    significant = recommendations_df[
        recommendations_df["diagnosis"]
        == "THRESHOLD_SENSITIVITY_SIGNIFICANT"
    ]

    moderate = recommendations_df[
        recommendations_df["diagnosis"]
        == "THRESHOLD_SENSITIVITY_MODERATE"
    ]

    not_primary = recommendations_df[
        recommendations_df["diagnosis"]
        == "THRESHOLD_NOT_PRIMARY_ISSUE"
    ]

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4J",
        "gate_name": "Threshold Sensitivity Analysis",
        "status": "ANALYSIS_COMPLETE",

        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,

        "test_rows": int(len(structured_test)),
        "techniques_analyzed": TARGET_TECHNIQUES,

        "threshold_grid": THRESHOLDS,

        "diagnostic_counts": {
            "threshold_sensitivity_significant": int(
                len(significant)
            ),
            "threshold_sensitivity_moderate": int(
                len(moderate)
            ),
            "threshold_not_primary_issue": int(
                len(not_primary)
            ),
        },

        "interpretation": {
            "purpose": (
                "Determine whether weak technique performance "
                "can plausibly be improved by threshold changes "
                "without modifying the trained model."
            ),

            "important_constraint": (
                "Test-set analysis is diagnostic only. "
                "Thresholds must not be changed based solely "
                "on the untouched test set."
            ),

            "next_step": (
                "Use validation data for any actual threshold "
                "recalibration. Do not overwrite production "
                "thresholds from this analysis."
            ),
        },

        "output_files": {
            "sensitivity": str(
                sensitivity_path
            ),
            "recommendations": str(
                recommendation_path
            ),
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    banner("GATE 2.4J COMPLETE")

    print(
        f"Sensitivity : {sensitivity_path}"
    )

    print(
        f"Recommendations : {recommendation_path}"
    )

    print(
        f"Report : {report_path}"
    )

    print()
    print(
        "No model, threshold, dataset, "
        "or production artifact was modified."
    )

    print()
    print(
        "STATUS: ANALYSIS_COMPLETE"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    evaluate()