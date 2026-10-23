"""
SmartSOC Phase 2 Gate 2.4L
Threshold Decision & Untouched-Test Simulation

IMPORTANT:
- The untouched test set is used ONLY for impact simulation.
- NO model is modified.
- NO production threshold artifact is modified.
- NO dataset is modified.
- This script compares:
    1. Current production thresholds
    2. Proposed validation thresholds
- The test set remains untouched for training/calibration purposes.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4_1"

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_4l"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"

PROPOSED_THRESHOLD_FILE = (
    ROOT
    / "reports"
    / "phase2_gate2_4k"
    / "proposed_thresholds_validation.json"
)


# ============================================================================
# HELPERS
# ============================================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def section(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]
        except Exception:
            pass

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def build_multilabel_matrix(labels_series, techniques):
    technique_to_index = {
        technique: idx
        for idx, technique in enumerate(techniques)
    }

    matrix = np.zeros(
        (len(labels_series), len(techniques)),
        dtype=np.int8,
    )

    for row_idx, value in enumerate(labels_series):
        labels = normalize_label(value)

        for label in labels:
            if label in technique_to_index:
                matrix[row_idx, technique_to_index[label]] = 1

    return matrix


def calculate_metrics(y_true, y_pred):
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
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
        "weighted_f1": float(weighted_f1),
    }


def threshold_predict(probabilities, techniques, thresholds):
    prediction = np.zeros_like(
        probabilities,
        dtype=np.int8,
    )

    for idx, technique in enumerate(techniques):
        threshold = float(
            thresholds.get(
                technique,
                0.5,
            )
        )

        prediction[:, idx] = (
            probabilities[:, idx] >= threshold
        ).astype(np.int8)

    return prediction


def sample_top1_accuracy(probabilities, y_true):
    top1 = np.argmax(
        probabilities,
        axis=1,
    )

    correct = 0

    for row_idx in range(len(y_true)):
        true_indices = np.where(
            y_true[row_idx] == 1
        )[0]

        if len(true_indices) == 0:
            continue

        if top1[row_idx] in true_indices:
            correct += 1

    valid_rows = np.sum(
        np.sum(y_true, axis=1) > 0
    )

    if valid_rows == 0:
        return 0.0

    return float(correct / valid_rows)


def sample_top5_accuracy(probabilities, y_true):
    top5 = np.argsort(
        probabilities,
        axis=1,
    )[:, -5:]

    correct = 0

    for row_idx in range(len(y_true)):
        true_indices = np.where(
            y_true[row_idx] == 1
        )[0]

        if len(true_indices) == 0:
            continue

        if any(
            idx in true_indices
            for idx in top5[row_idx]
        ):
            correct += 1

    valid_rows = np.sum(
        np.sum(y_true, axis=1) > 0
    )

    if valid_rows == 0:
        return 0.0

    return float(correct / valid_rows)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts():
    banner("SMARTSOC PHASE 2 GATE 2.4L")
    print("THRESHOLD DECISION & UNTOUCHED-TEST SIMULATION")
    print()
    print("IMPORTANT:")
    print("  - Test set is NOT used for calibration.")
    print("  - Test set is used only for impact simulation.")
    print("  - Production model will NOT be modified.")
    print("  - Production thresholds will NOT be modified.")

    section("PRODUCTION ARTIFACT VALIDATION")

    required = [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        MLB_FILE,
        SCHEMA_FILE,
        THRESHOLD_FILE,
        PROPOSED_THRESHOLD_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    mlb = joblib.load(MLB_FILE)

    schema = load_json(SCHEMA_FILE)
    current_threshold_artifact = load_json(
        THRESHOLD_FILE
    )
    proposed_threshold_artifact = load_json(
        PROPOSED_THRESHOLD_FILE
    )

    categorical_columns = schema[
        "categorical_columns"
    ]

    numerical_columns = schema[
        "numerical_columns"
    ]

    structured_columns = (
        categorical_columns
        + numerical_columns
    )

    techniques = [
        str(x)
        for x in mlb.classes_
    ]

    current_thresholds = {
        str(k): float(v)
        for k, v in current_threshold_artifact[
            "thresholds"
        ].items()
    }

    proposed_thresholds = {
        str(k): float(v)
        for k, v in proposed_threshold_artifact[
            "thresholds"
        ].items()
    }

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
        f"{len(tfidf.vocabulary_)}"
    )

    print(
        f"MITRE classes        : "
        f"{len(techniques)}"
    )

    print(
        f"Current thresholds   : "
        f"{len(current_thresholds)}"
    )

    print(
        f"Proposed thresholds  : "
        f"{len(proposed_thresholds)}"
    )

    expected_features = (
        len(
            encoder.get_feature_names_out(
                categorical_columns
            )
        )
        + len(numerical_columns)
        + len(tfidf.vocabulary_)
    )

    model_features = int(
        model.estimators_[0].coef_.shape[1]
    )

    print()
    print(
        f"Expected features    : "
        f"{expected_features}"
    )

    print(
        f"Model input features : "
        f"{model_features}"
    )

    if expected_features != model_features:
        raise RuntimeError(
            "Production feature dimension mismatch."
        )

    if len(current_thresholds) != len(techniques):
        raise RuntimeError(
            "Current threshold count does not "
            "match MITRE classes."
        )

    if len(proposed_thresholds) != len(techniques):
        raise RuntimeError(
            "Proposed threshold count does not "
            "match MITRE classes."
        )

    if set(current_thresholds) != set(techniques):
        raise RuntimeError(
            "Current threshold technique IDs "
            "do not match model classes."
        )

    if set(proposed_thresholds) != set(techniques):
        raise RuntimeError(
            "Proposed threshold technique IDs "
            "do not match model classes."
        )

    print(
        "[PASS] Production artifacts validated"
    )

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "tfidf": tfidf,
        "mlb": mlb,
        "schema": schema,
        "techniques": techniques,
        "current_thresholds": current_thresholds,
        "proposed_thresholds": proposed_thresholds,
    }


# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_test_data():
    section("LOADING UNTOUCHED TEST DATA")

    structured = pd.read_csv(
        STRUCTURED_TEST
    )

    labels = pd.read_csv(
        LABELS_TEST
    )

    behavior = pd.read_csv(
        BEHAVIOR_TEST
    )

    print(
        f"Structured rows : {len(structured)}"
    )

    print(
        f"Label rows      : {len(labels)}"
    )

    print(
        f"Behavior rows   : {len(behavior)}"
    )

    if not (
        len(structured)
        == len(labels)
        == len(behavior)
    ):
        raise RuntimeError(
            "Test row alignment failure."
        )

    print(
        "[PASS] Test row alignment"
    )

    return structured, labels, behavior


# ============================================================================
# BUILD FEATURES
# ============================================================================

def build_features(
    structured,
    behavior,
    artifacts,
):
    schema = artifacts["schema"]

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    categorical_columns = schema[
        "categorical_columns"
    ]

    numerical_columns = schema[
        "numerical_columns"
    ]

    behavior_column = schema.get(
        "behavior_text_column",
        "behavior_text_normalized",
    )

    for column in categorical_columns:
        if column not in structured.columns:
            raise RuntimeError(
                f"Missing categorical column: {column}"
            )

    for column in numerical_columns:
        if column not in structured.columns:
            raise RuntimeError(
                f"Missing numerical column: {column}"
            )

    if behavior_column not in behavior.columns:
        raise RuntimeError(
            f"Missing behavior column: "
            f"{behavior_column}"
        )

    section("PRODUCTION FEATURE TRANSFORMATION")

    categorical = structured[
        categorical_columns
    ].copy()

    numerical = structured[
        numerical_columns
    ].copy()

    behavior_text = (
        behavior[behavior_column]
        .fillna("")
        .astype(str)
    )

    categorical_matrix = encoder.transform(
        categorical
    )

    numerical_matrix = scaler.transform(
        numerical
    )

    tfidf_matrix = tfidf.transform(
        behavior_text
    )

    from scipy.sparse import hstack

    final_matrix = hstack(
        [
            categorical_matrix,
            numerical_matrix,
            tfidf_matrix,
        ],
        format="csr",
    )

    print(
        f"Categorical matrix : "
        f"{categorical_matrix.shape}"
    )

    print(
        f"Numerical matrix   : "
        f"{numerical_matrix.shape}"
    )

    print(
        f"TF-IDF matrix      : "
        f"{tfidf_matrix.shape}"
    )

    print(
        f"Final matrix       : "
        f"{final_matrix.shape}"
    )

    if final_matrix.shape[1] != 6919:
        raise RuntimeError(
            "Final production feature dimension "
            f"is {final_matrix.shape[1]}, expected 6919."
        )

    print(
        "[PASS] Final feature dimension = 6919"
    )

    return final_matrix


# ============================================================================
# RUN INFERENCE
# ============================================================================

def run_inference(
    final_matrix,
    artifacts,
):
    section("RUNNING PRODUCTION INFERENCE")

    model = artifacts["model"]

    probabilities = model.predict_proba(
        final_matrix
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    print(
        f"Probability matrix : "
        f"{probabilities.shape}"
    )

    if probabilities.shape[1] != 45:
        raise RuntimeError(
            "Probability matrix does not contain "
            "45 MITRE classes."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise RuntimeError(
            "Probability matrix contains "
            "non-finite values."
        )

    if np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        raise RuntimeError(
            "Probability values outside [0,1]."
        )

    print(
        "[PASS] Probability matrix valid"
    )

    return probabilities


# ============================================================================
# COMPARE THRESHOLDS
# ============================================================================

def compare_thresholds(artifacts):
    section("THRESHOLD COMPARISON")

    current = artifacts[
        "current_thresholds"
    ]

    proposed = artifacts[
        "proposed_thresholds"
    ]

    techniques = artifacts[
        "techniques"
    ]

    rows = []

    changed = 0

    for technique in techniques:
        old = float(
            current.get(
                technique,
                0.5,
            )
        )

        new = float(
            proposed.get(
                technique,
                0.5,
            )
        )

        delta = new - old

        if abs(delta) > 1e-12:
            changed += 1

        rows.append(
            {
                "technique": technique,
                "current_threshold": old,
                "proposed_threshold": new,
                "delta": delta,
                "changed": abs(delta) > 1e-12,
            }
        )

    df = pd.DataFrame(rows)

    print(
        f"MITRE techniques : {len(df)}"
    )

    print(
        f"Threshold changes : {changed}"
    )

    if changed == 0:
        print(
            "[INFO] Validation proposes no "
            "threshold changes."
        )
    else:
        print()
        print(
            df[df["changed"]].to_string(
                index=False
            )
        )

    return df


# ============================================================================
# TEST SIMULATION
# ============================================================================

def simulate(
    probabilities,
    y_true,
    artifacts,
    threshold_df,
):
    section(
        "UNTOUCHED-TEST THRESHOLD IMPACT SIMULATION"
    )

    techniques = artifacts[
        "techniques"
    ]

    current = artifacts[
        "current_thresholds"
    ]

    proposed = artifacts[
        "proposed_thresholds"
    ]

    y_current = threshold_predict(
        probabilities,
        techniques,
        current,
    )

    y_proposed = threshold_predict(
        probabilities,
        techniques,
        proposed,
    )

    current_metrics = calculate_metrics(
        y_true,
        y_current,
    )

    proposed_metrics = calculate_metrics(
        y_true,
        y_proposed,
    )

    current_top1 = sample_top1_accuracy(
        probabilities,
        y_true,
    )

    current_top5 = sample_top5_accuracy(
        probabilities,
        y_true,
    )

    proposed_top1 = current_top1
    proposed_top5 = current_top5

    print()
    print(
        "CURRENT PRODUCTION THRESHOLDS"
    )
    print(
        f"Micro precision : "
        f"{current_metrics['micro_precision']:.4f}"
    )
    print(
        f"Micro recall    : "
        f"{current_metrics['micro_recall']:.4f}"
    )
    print(
        f"Micro F1        : "
        f"{current_metrics['micro_f1']:.4f}"
    )
    print(
        f"Macro precision : "
        f"{current_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro recall    : "
        f"{current_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1        : "
        f"{current_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1     : "
        f"{current_metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        "PROPOSED VALIDATION THRESHOLDS"
    )
    print(
        f"Micro precision : "
        f"{proposed_metrics['micro_precision']:.4f}"
    )
    print(
        f"Micro recall    : "
        f"{proposed_metrics['micro_recall']:.4f}"
    )
    print(
        f"Micro F1        : "
        f"{proposed_metrics['micro_f1']:.4f}"
    )
    print(
        f"Macro precision : "
        f"{proposed_metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro recall    : "
        f"{proposed_metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1        : "
        f"{proposed_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1     : "
        f"{proposed_metrics['weighted_f1']:.4f}"
    )

    metric_rows = []

    for key in current_metrics:
        metric_rows.append(
            {
                "metric": key,
                "current": current_metrics[key],
                "proposed": proposed_metrics[key],
                "delta": (
                    proposed_metrics[key]
                    - current_metrics[key]
                ),
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    print()
    print(
        "METRIC DELTA"
    )
    print(
        metrics_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Per-technique analysis
    # ------------------------------------------------------------

    rows = []

    for idx, technique in enumerate(
        techniques
    ):
        yt = y_true[:, idx]

        yp_current = y_current[:, idx]
        yp_proposed = y_proposed[:, idx]

        current_p, current_r, current_f1, _ = (
            precision_recall_fscore_support(
                yt,
                yp_current,
                average="binary",
                zero_division=0,
            )
        )

        proposed_p, proposed_r, proposed_f1, _ = (
            precision_recall_fscore_support(
                yt,
                yp_proposed,
                average="binary",
                zero_division=0,
            )
        )

        current_tp = int(
            np.sum(
                (yt == 1)
                & (yp_current == 1)
            )
        )

        current_fp = int(
            np.sum(
                (yt == 0)
                & (yp_current == 1)
            )
        )

        current_fn = int(
            np.sum(
                (yt == 1)
                & (yp_current == 0)
            )
        )

        proposed_tp = int(
            np.sum(
                (yt == 1)
                & (yp_proposed == 1)
            )
        )

        proposed_fp = int(
            np.sum(
                (yt == 0)
                & (yp_proposed == 1)
            )
        )

        proposed_fn = int(
            np.sum(
                (yt == 1)
                & (yp_proposed == 0)
            )
        )

        rows.append(
            {
                "technique": technique,
                "support": int(np.sum(yt)),
                "current_threshold": float(
                    current[technique]
                ),
                "proposed_threshold": float(
                    proposed[technique]
                ),
                "threshold_delta": float(
                    proposed[technique]
                    - current[technique]
                ),
                "current_precision": float(
                    current_p
                ),
                "current_recall": float(
                    current_r
                ),
                "current_f1": float(
                    current_f1
                ),
                "proposed_precision": float(
                    proposed_p
                ),
                "proposed_recall": float(
                    proposed_r
                ),
                "proposed_f1": float(
                    proposed_f1
                ),
                "precision_delta": float(
                    proposed_p - current_p
                ),
                "recall_delta": float(
                    proposed_r - current_r
                ),
                "f1_delta": float(
                    proposed_f1 - current_f1
                ),
                "current_tp": current_tp,
                "current_fp": current_fp,
                "current_fn": current_fn,
                "proposed_tp": proposed_tp,
                "proposed_fp": proposed_fp,
                "proposed_fn": proposed_fn,
            }
        )

    per_technique = pd.DataFrame(
        rows
    )

    return (
        current_metrics,
        proposed_metrics,
        metrics_df,
        per_technique,
        current_top1,
        current_top5,
        proposed_top1,
        proposed_top5,
    )


# ============================================================================
# DECISION LOGIC
# ============================================================================

def make_decision(
    threshold_df,
    metrics_df,
    per_technique,
):
    section("THRESHOLD DECISION")

    changed = threshold_df[
        threshold_df["changed"]
    ].copy()

    if len(changed) == 0:
        decision = (
            "KEEP_EXISTING_THRESHOLDS"
        )

        reason = (
            "Chronological validation produced "
            "no threshold changes."
        )

    else:
        changed_techniques = set(
            changed["technique"]
        )

        proposed_improved = per_technique[
            per_technique["technique"].isin(
                changed_techniques
            )
            & (
                per_technique["f1_delta"]
                > 0
            )
        ]

        if len(proposed_improved) > 0:
            decision = (
                "REVIEW_PROPOSED_THRESHOLD_CHANGES"
            )

            reason = (
                "Validation proposed threshold "
                "changes with positive validation "
                "impact. Test simulation must be "
                "reviewed before production adoption."
            )

        else:
            decision = (
                "KEEP_EXISTING_THRESHOLDS"
            )

            reason = (
                "No changed threshold produced "
                "clear positive evidence sufficient "
                "for automatic production adoption."
            )

    print(
        f"Decision : {decision}"
    )

    print(
        f"Reason   : {reason}"
    )

    return decision, reason


# ============================================================================
# REPORT
# ============================================================================

def save_outputs(
    threshold_df,
    metrics_df,
    per_technique,
    current_metrics,
    proposed_metrics,
    decision,
    reason,
    artifacts,
):
    threshold_path = (
        REPORT_DIR
        / "threshold_decision_comparison.csv"
    )

    metrics_path = (
        REPORT_DIR
        / "test_metric_comparison.csv"
    )

    technique_path = (
        REPORT_DIR
        / "per_technique_threshold_impact.csv"
    )

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4l_report.json"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    per_technique.to_csv(
        technique_path,
        index=False,
    )

    improved = per_technique[
        per_technique["f1_delta"] > 0
    ]

    degraded = per_technique[
        per_technique["f1_delta"] < 0
    ]

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4L",
        "gate_name": (
            "Threshold Decision & "
            "Untouched-Test Simulation"
        ),
        "model": (
            "SmartSOC Windows-APT "
            "Model B v4.1"
        ),
        "status": "ANALYSIS_COMPLETE",
        "policy": {
            "test_used_for_calibration": False,
            "test_used_for_impact_simulation": True,
            "production_model_modified": False,
            "production_thresholds_modified": False,
            "dataset_modified": False,
        },
        "dimensions": {
            "test_rows": 11707,
            "mitre_classes": len(
                artifacts["techniques"]
            ),
            "final_features": 6919,
        },
        "thresholds": {
            "current_count": len(
                artifacts[
                    "current_thresholds"
                ]
            ),
            "proposed_count": len(
                artifacts[
                    "proposed_thresholds"
                ]
            ),
            "changed_count": int(
                threshold_df[
                    "changed"
                ].sum()
            ),
        },
        "test_metrics": {
            "current": current_metrics,
            "proposed": proposed_metrics,
        },
        "impact": {
            "techniques_improved_f1": int(
                len(improved)
            ),
            "techniques_degraded_f1": int(
                len(degraded)
            ),
        },
        "decision": {
            "decision": decision,
            "reason": reason,
        },
        "outputs": {
            "threshold_comparison": str(
                threshold_path
            ),
            "metric_comparison": str(
                metrics_path
            ),
            "per_technique": str(
                technique_path
            ),
            "report": str(
                report_path
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

    return (
        threshold_path,
        metrics_path,
        technique_path,
        report_path,
    )


# ============================================================================
# MAIN
# ============================================================================

def evaluate():
    artifacts = load_artifacts()

    structured, labels, behavior = (
        load_test_data()
    )

    final_matrix = build_features(
        structured,
        behavior,
        artifacts,
    )

    probabilities = run_inference(
        final_matrix,
        artifacts,
    )

    techniques = artifacts[
        "techniques"
    ]

    labels_column = labels.columns[0]

    y_true = build_multilabel_matrix(
        labels[labels_column],
        techniques,
    )

    print()
    print(
        f"True-label matrix : "
        f"{y_true.shape}"
    )

    threshold_df = compare_thresholds(
        artifacts
    )

    (
        current_metrics,
        proposed_metrics,
        metrics_df,
        per_technique,
        current_top1,
        current_top5,
        proposed_top1,
        proposed_top5,
    ) = simulate(
        probabilities,
        y_true,
        artifacts,
        threshold_df,
    )

    decision, reason = make_decision(
        threshold_df,
        metrics_df,
        per_technique,
    )

    (
        threshold_path,
        metrics_path,
        technique_path,
        report_path,
    ) = save_outputs(
        threshold_df,
        metrics_df,
        per_technique,
        current_metrics,
        proposed_metrics,
        decision,
        reason,
        artifacts,
    )

    banner(
        "GATE 2.4L COMPLETE"
    )

    print(
        f"Threshold comparison : "
        f"{threshold_path}"
    )

    print(
        f"Metric comparison    : "
        f"{metrics_path}"
    )

    print(
        f"Per-technique impact : "
        f"{technique_path}"
    )

    print(
        f"Report               : "
        f"{report_path}"
    )

    print()
    print(
        "No model, threshold, dataset, "
        "or production artifact was modified."
    )

    print()
    print(
        f"DECISION: {decision}"
    )


if __name__ == "__main__":
    evaluate()