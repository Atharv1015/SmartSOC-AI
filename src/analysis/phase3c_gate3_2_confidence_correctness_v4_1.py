import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.exceptions import InconsistentVersionWarning


# ============================================================
# SMARTSOC PHASE 3C
# GATE 3C.2 — CONFIDENCE VS CORRECTNESS ANALYSIS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

PRODUCTION_DIR = (
    PROJECT_ROOT / "models" / "windows_apt_v4_1" / "production"
)

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "phase3c_gate3_2"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PRODUCTION_DIR / "model.joblib"
ENCODER_PATH = PRODUCTION_DIR / "structured_encoder.joblib"
SCALER_PATH = PRODUCTION_DIR / "numerical_scaler.joblib"
TFIDF_PATH = PRODUCTION_DIR / "tfidf_vectorizer.joblib"
MLB_PATH = PRODUCTION_DIR / "multilabel_encoder.joblib"
SCHEMA_PATH = PRODUCTION_DIR / "feature_schema.json"
THRESHOLD_PATH = PRODUCTION_DIR / "thresholds.json"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)


def fail(message):
    print(f"[FAIL] {message}")
    raise RuntimeError(message)


print("=" * 72)
print("SMARTSOC PHASE 3C")
print("GATE 3C.2 — CONFIDENCE VS CORRECTNESS ANALYSIS")
print("=" * 72)


# ============================================================
# 1. LOAD PRODUCTION ARTIFACTS
# ============================================================

print("\nPRODUCTION ARTIFACTS")

required = {
    "model.joblib": MODEL_PATH,
    "structured_encoder.joblib": ENCODER_PATH,
    "numerical_scaler.joblib": SCALER_PATH,
    "tfidf_vectorizer.joblib": TFIDF_PATH,
    "multilabel_encoder.joblib": MLB_PATH,
    "feature_schema.json": SCHEMA_PATH,
    "thresholds.json": THRESHOLD_PATH,
}

for name, path in required.items():
    if not path.exists():
        fail(f"Missing artifact: {path}")
    print(f"[PASS] {name}")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
tfidf = joblib.load(TFIDF_PATH)
mlb = joblib.load(MLB_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    threshold_config = json.load(f)

print("[PASS] Production artifacts loaded")


# ============================================================
# 2. SCHEMA
# ============================================================

categorical_columns = schema["categorical_columns"]
numerical_columns = schema["numerical_columns"]
techniques = schema["mitre_techniques"]

expected_final = int(schema["final_feature_count"])
expected_tfidf = int(schema["tfidf_feature_count"])

print("\nFEATURE SCHEMA")
print(f"  Categorical : {len(categorical_columns)}")
print(f"  Numerical   : {len(numerical_columns)}")
print(f"  TF-IDF      : {expected_tfidf}")
print(f"  Final       : {expected_final}")
print(f"  MITRE       : {len(techniques)}")

if len(categorical_columns) != 4:
    fail("Expected 4 categorical columns")

if len(numerical_columns) != 58:
    fail("Expected 58 numerical columns")

if expected_final != 6919:
    fail("Expected final feature count of 6919")

if len(techniques) != 45:
    fail("Expected 45 MITRE techniques")

print("[PASS] Schema validation")


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nTEST DATA")

structured = pd.read_csv(STRUCTURED_TEST)
labels = pd.read_csv(LABELS_TEST)
behavior = pd.read_csv(BEHAVIOR_TEST)

print(f"  Structured : {len(structured)}")
print(f"  Labels     : {len(labels)}")
print(f"  Behavior   : {len(behavior)}")

if len(structured) != len(labels):
    fail("Structured/labels mismatch")

if len(structured) != len(behavior):
    fail("Structured/behavior mismatch")

if "techniques" not in labels.columns:
    fail("Missing techniques column")

behavior_column = "behavior_text_normalized"

if behavior_column not in behavior.columns:
    fail("Missing behavior_text_normalized")

print("[PASS] Test alignment")


# ============================================================
# 4. BUILD FEATURE MATRIX
# ============================================================

print("\nFEATURE TRANSFORMATION")

X_cat = encoder.transform(
    structured[categorical_columns].copy()
)

X_num = scaler.transform(
    structured[numerical_columns].to_numpy(dtype=np.float64)
)

behavior_text = (
    behavior[behavior_column]
    .fillna("")
    .astype(str)
    .tolist()
)

X_text = tfidf.transform(behavior_text)

X_final = hstack(
    [
        csr_matrix(X_cat),
        csr_matrix(X_num),
        X_text,
    ],
    format="csr",
)

print(f"  Categorical encoded : {X_cat.shape}")
print(f"  Numerical            : {X_num.shape}")
print(f"  TF-IDF               : {X_text.shape}")
print(f"  Final                : {X_final.shape}")

if X_final.shape[1] != expected_final:
    fail("Final feature dimension mismatch")

print("[PASS] Feature transformation")


# ============================================================
# 5. INFERENCE
# ============================================================

print("\nMODEL INFERENCE")

probabilities = np.asarray(
    model.predict_proba(X_final),
    dtype=np.float64,
)

print(f"  Probability matrix: {probabilities.shape}")

if probabilities.shape != (
    len(structured),
    len(techniques),
):
    fail("Probability matrix shape mismatch")

if not np.isfinite(probabilities).all():
    fail("Non-finite probabilities")

print("[PASS] Inference")


# ============================================================
# 6. LOAD THRESHOLDS
# ============================================================

thresholds = threshold_config.get("thresholds", {})

if len(thresholds) != len(techniques):
    fail("Threshold count mismatch")

threshold_array = np.array(
    [float(thresholds[t]) for t in techniques],
    dtype=np.float64,
)

print(f"  Thresholds: {len(threshold_array)}")
print("[PASS] Threshold configuration")


# ============================================================
# 7. TRUE LABEL MATRIX
# ============================================================

print("\nTRUE LABEL MATRIX")

true_matrix = np.zeros(
    (len(labels), len(techniques)),
    dtype=np.int8,
)

technique_to_index = {
    technique: i
    for i, technique in enumerate(techniques)
}

unknown_labels = []

for row_idx, value in enumerate(
    labels["techniques"].fillna("")
):

    raw = str(value).strip()

    if not raw:
        continue

    normalized = (
        raw.replace(";", ",")
        .replace("|", ",")
    )

    tokens = [
        token.strip()
        for token in normalized.split(",")
        if token.strip()
    ]

    for token in tokens:
        if token in technique_to_index:
            true_matrix[
                row_idx,
                technique_to_index[token]
            ] = 1
        else:
            unknown_labels.append(token)

print(
    f"  True label assignments: "
    f"{int(true_matrix.sum())}"
)

print(
    f"  Unknown labels: "
    f"{len(unknown_labels)}"
)

if unknown_labels:
    fail(
        "Unknown MITRE labels detected: "
        + str(sorted(set(unknown_labels))[:20])
    )

print("[PASS] True labels")


# ============================================================
# 8. TOP-1 / TOP-2
# ============================================================

top1_index = np.argmax(
    probabilities,
    axis=1,
)

sorted_probs = np.sort(
    probabilities,
    axis=1,
)

top1_probability = sorted_probs[:, -1]
top2_probability = sorted_probs[:, -2]

top1_margin = (
    top1_probability -
    top2_probability
)

top1_technique = [
    techniques[i]
    for i in top1_index
]

top1_threshold = threshold_array[top1_index]

top1_detected = (
    top1_probability >= top1_threshold
)


# ============================================================
# 9. TOP-1 CORRECTNESS
# ============================================================

top1_correct = np.array(
    [
        bool(true_matrix[i, top1_index[i]])
        for i in range(len(structured))
    ],
    dtype=bool,
)

top1_accuracy = float(
    top1_correct.mean()
)

print("\nTOP-1 PERFORMANCE")
print(f"  Accuracy: {top1_accuracy:.6f}")
print(
    f"  Correct: "
    f"{int(top1_correct.sum())}"
)
print(
    f"  Incorrect: "
    f"{int((~top1_correct).sum())}"
)


# ============================================================
# 10. OPERATIONAL CONFIDENCE BANDS
# ============================================================

bands = [
    ("LOW", 0.00, 0.50),
    ("MEDIUM", 0.50, 0.80),
    ("HIGH", 0.80, 0.95),
    ("VERY_HIGH", 0.95, 1.0000001),
]

band_rows = []

for name, lower, upper in bands:

    mask = (
        (top1_probability >= lower)
        & (top1_probability < upper)
    )

    count = int(mask.sum())

    if count:
        accuracy = float(
            top1_correct[mask].mean()
        )

        mean_probability = float(
            top1_probability[mask].mean()
        )

        mean_margin = float(
            top1_margin[mask].mean()
        )
    else:
        accuracy = None
        mean_probability = None
        mean_margin = None

    band_rows.append({
        "confidence_band": name,
        "lower_bound": lower,
        "upper_bound": upper,
        "events": count,
        "fraction": float(
            count / len(top1_probability)
        ),
        "correct": (
            int(top1_correct[mask].sum())
            if count else 0
        ),
        "incorrect": (
            int((~top1_correct[mask]).sum())
            if count else 0
        ),
        "accuracy": accuracy,
        "mean_probability": mean_probability,
        "mean_top1_top2_margin": mean_margin,
    })

band_df = pd.DataFrame(band_rows)


# ============================================================
# 11. MARGIN BANDS
# ============================================================

margin_bands = [
    ("VERY_AMBIGUOUS", 0.00, 0.05),
    ("AMBIGUOUS", 0.05, 0.10),
    ("MODERATE", 0.10, 0.25),
    ("CLEAR", 0.25, 0.50),
    ("VERY_CLEAR", 0.50, np.inf),
]

margin_rows = []

for name, lower, upper in margin_bands:

    mask = (
        (top1_margin >= lower)
        & (top1_margin < upper)
    )

    count = int(mask.sum())

    margin_rows.append({
        "margin_band": name,
        "lower_bound": lower,
        "upper_bound": upper,
        "events": count,
        "fraction": float(
            count / len(top1_margin)
        ),
        "accuracy": (
            float(top1_correct[mask].mean())
            if count else None
        ),
        "mean_top1_probability": (
            float(top1_probability[mask].mean())
            if count else None
        ),
    })

margin_df = pd.DataFrame(margin_rows)


# ============================================================
# 12. DETECTED VS NON-DETECTED
# ============================================================

detected_mask = top1_detected
non_detected_mask = ~top1_detected

detection_comparison = {
    "detected_top1": {
        "events": int(detected_mask.sum()),
        "accuracy": (
            float(top1_correct[detected_mask].mean())
            if detected_mask.any()
            else None
        ),
        "mean_probability": (
            float(
                top1_probability[detected_mask].mean()
            )
            if detected_mask.any()
            else None
        ),
        "mean_margin": (
            float(
                top1_margin[detected_mask].mean()
            )
            if detected_mask.any()
            else None
        ),
    },

    "non_detected_top1": {
        "events": int(non_detected_mask.sum()),
        "accuracy": (
            float(
                top1_correct[non_detected_mask].mean()
            )
            if non_detected_mask.any()
            else None
        ),
        "mean_probability": (
            float(
                top1_probability[non_detected_mask].mean()
            )
            if non_detected_mask.any()
            else None
        ),
        "mean_margin": (
            float(
                top1_margin[non_detected_mask].mean()
            )
            if non_detected_mask.any()
            else None
        ),
    },
}


# ============================================================
# 13. HIGH-CONFIDENCE ERRORS
# ============================================================

high_confidence_thresholds = [
    ("HIGH_0.80", 0.80),
    ("VERY_HIGH_0.95", 0.95),
    ("EXTREME_0.99", 0.99),
]

high_confidence_rows = []

for name, cutoff in high_confidence_thresholds:

    mask = top1_probability >= cutoff

    incorrect = mask & (~top1_correct)

    high_confidence_rows.append({
        "confidence_level": name,
        "cutoff": cutoff,
        "events": int(mask.sum()),
        "correct": int(top1_correct[mask].sum()),
        "incorrect": int(incorrect.sum()),
        "accuracy": (
            float(top1_correct[mask].mean())
            if mask.any()
            else None
        ),
        "error_rate": (
            float(incorrect.sum() / mask.sum())
            if mask.any()
            else None
        ),
    })

high_confidence_df = pd.DataFrame(
    high_confidence_rows
)


# ============================================================
# 14. FALSE HIGH-CONFIDENCE PREDICTIONS
# ============================================================

false_mask = (
    (~top1_correct)
    & (top1_probability >= 0.80)
)

false_indices = np.where(false_mask)[0]

false_indices = false_indices[
    np.argsort(
        -top1_probability[false_indices]
    )
]

false_high_confidence_events = []

for idx in false_indices:

    true_techniques = [
        techniques[j]
        for j in range(len(techniques))
        if true_matrix[idx, j]
    ]

    false_high_confidence_events.append({
        "row": int(idx),
        "predicted": top1_technique[idx],
        "probability": float(
            top1_probability[idx]
        ),
        "second_probability": float(
            top2_probability[idx]
        ),
        "margin": float(
            top1_margin[idx]
        ),
        "threshold": float(
            top1_threshold[idx]
        ),
        "detected": bool(
            top1_detected[idx]
        ),
        "true_techniques": "|".join(
            true_techniques
        ),
    })


false_high_df = pd.DataFrame(
    false_high_confidence_events
)


# ============================================================
# 15. TECHNIQUE-LEVEL CONFIDENCE VS CORRECTNESS
# ============================================================

technique_rows = []

for j, technique in enumerate(techniques):

    predicted = (
        probabilities[:, j]
        >= threshold_array[j]
    )

    true = (
        true_matrix[:, j].astype(bool)
    )

    tp = int(
        (predicted & true).sum()
    )

    fp = int(
        (predicted & ~true).sum()
    )

    fn = int(
        (~predicted & true).sum()
    )

    support = int(true.sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    detected_probs = probabilities[
        predicted,
        j
    ]

    true_probs = probabilities[
        true,
        j
    ]

    false_positive_probs = probabilities[
        predicted & ~true,
        j
    ]

    technique_rows.append({
        "technique": technique,
        "threshold": float(
            threshold_array[j]
        ),
        "support": support,
        "detections": int(predicted.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_detected_score": (
            float(detected_probs.mean())
            if len(detected_probs)
            else None
        ),
        "mean_true_score": (
            float(true_probs.mean())
            if len(true_probs)
            else None
        ),
        "mean_false_positive_score": (
            float(false_positive_probs.mean())
            if len(false_positive_probs)
            else None
        ),
        "max_score": float(
            probabilities[:, j].max()
        ),
    })

technique_df = pd.DataFrame(
    technique_rows
)


# ============================================================
# 16. AMBIGUOUS EVENTS
# ============================================================

ambiguous_mask = (
    top1_margin < 0.10
)

ambiguous_correct = (
    top1_correct[ambiguous_mask]
)

ambiguous_summary = {
    "events": int(ambiguous_mask.sum()),
    "fraction": float(
        ambiguous_mask.mean()
    ),
    "correct": int(
        ambiguous_correct.sum()
    ),
    "incorrect": int(
        (~ambiguous_correct).sum()
    ),
    "accuracy": (
        float(ambiguous_correct.mean())
        if ambiguous_mask.any()
        else None
    ),
    "mean_top1_probability": (
        float(
            top1_probability[
                ambiguous_mask
            ].mean()
        )
        if ambiguous_mask.any()
        else None
    ),
}


# ============================================================
# 17. FALLBACK EVENTS
# ============================================================

all_detection_matrix = (
    probabilities >= threshold_array.reshape(
        1, -1
    )
)

fallback_mask = (
    all_detection_matrix.sum(axis=1) == 0
)

fallback_correct = (
    top1_correct[fallback_mask]
)

fallback_summary = {
    "events": int(fallback_mask.sum()),
    "fraction": float(
        fallback_mask.mean()
    ),
    "top1_accuracy": (
        float(fallback_correct.mean())
        if fallback_mask.any()
        else None
    ),
    "mean_top1_probability": (
        float(
            top1_probability[
                fallback_mask
            ].mean()
        )
        if fallback_mask.any()
        else None
    ),
    "mean_margin": (
        float(
            top1_margin[
                fallback_mask
            ].mean()
        )
        if fallback_mask.any()
        else None
    ),
}


# ============================================================
# 18. OPERATIONAL BAND RECOMMENDATION
# ============================================================

# These are ANALYTICAL recommendations only.
# They do NOT modify production policy.

very_high_accuracy = float(
    band_df.loc[
        band_df["confidence_band"] == "VERY_HIGH",
        "accuracy"
    ].iloc[0]
)

high_accuracy = float(
    band_df.loc[
        band_df["confidence_band"] == "HIGH",
        "accuracy"
    ].iloc[0]
)

medium_accuracy = float(
    band_df.loc[
        band_df["confidence_band"] == "MEDIUM",
        "accuracy"
    ].iloc[0]
)

low_accuracy = float(
    band_df.loc[
        band_df["confidence_band"] == "LOW",
        "accuracy"
    ].iloc[0]
)

recommendation = {
    "very_high": {
        "range": ">=0.95",
        "observed_accuracy": very_high_accuracy,
        "analytical_use":
            "candidate for highest-confidence handling, "
            "subject to margin, technique, risk, and analyst policy",
    },
    "high": {
        "range": "0.80-0.95",
        "observed_accuracy": high_accuracy,
        "analytical_use":
            "high-confidence AI reasoning candidate",
    },
    "medium": {
        "range": "0.50-0.80",
        "observed_accuracy": medium_accuracy,
        "analytical_use":
            "additional evidence and analyst review recommended",
    },
    "low": {
        "range": "<0.50",
        "observed_accuracy": low_accuracy,
        "analytical_use":
            "do not treat confidence alone as sufficient for response",
    },
    "important_note":
        "These bands are analytical recommendations only. "
        "They do not alter production thresholds or response policy.",
}


# ============================================================
# 19. QUALITY CHECKS
# ============================================================

checks = {
    "production_feature_dimension_valid":
        X_final.shape[1] == 6919,

    "probability_matrix_valid":
        probabilities.shape == (
            len(structured),
            45,
        ),

    "probabilities_finite":
        bool(np.isfinite(probabilities).all()),

    "true_label_alignment_valid":
        true_matrix.shape == (
            len(structured),
            45,
        ),

    "top1_correctness_completed":
        len(top1_correct) == len(structured),

    "confidence_band_analysis_completed":
        len(band_df) == 4,

    "margin_analysis_completed":
        len(margin_df) == 5,

    "technique_analysis_completed":
        len(technique_df) == 45,

    "high_confidence_error_analysis_completed":
        True,
}

decision = (
    "PASS"
    if all(checks.values())
    else "REVIEW_REQUIRED"
)


# ============================================================
# 20. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("PHASE 3C GATE 3.2 SUMMARY")
print("=" * 72)

print(
    f"Test events                 : "
    f"{len(structured)}"
)

print(
    f"Top-1 accuracy              : "
    f"{top1_accuracy:.6f}"
)

print("\nCONFIDENCE BAND RESULTS")

for _, row in band_df.iterrows():

    print(
        f"  {row['confidence_band']:10s} "
        f"events={int(row['events']):5d} "
        f"accuracy={row['accuracy']:.6f} "
        f"mean_score={row['mean_probability']:.6f}"
    )

print("\nMARGIN RESULTS")

for _, row in margin_df.iterrows():

    print(
        f"  {row['margin_band']:16s} "
        f"events={int(row['events']):5d} "
        f"accuracy={row['accuracy']:.6f}"
    )

print("\nDETECTED VS NON-DETECTED")

for key, value in detection_comparison.items():

    print(
        f"  {key:18s} "
        f"events={value['events']} "
        f"accuracy={value['accuracy']} "
        f"mean_score={value['mean_probability']}"
    )

print("\nHIGH-CONFIDENCE ERROR ANALYSIS")

for _, row in high_confidence_df.iterrows():

    print(
        f"  {row['confidence_level']:14s} "
        f"events={int(row['events']):5d} "
        f"incorrect={int(row['incorrect']):4d} "
        f"accuracy={row['accuracy']:.6f}"
    )

print("\nAMBIGUITY")

print(
    f"  Margin < 0.10 events : "
    f"{ambiguous_summary['events']}"
)

print(
    f"  Accuracy             : "
    f"{ambiguous_summary['accuracy']}"
)

print("\nFALLBACK")

print(
    f"  Events               : "
    f"{fallback_summary['events']}"
)

print(
    f"  Fraction             : "
    f"{fallback_summary['fraction']:.6f}"
)

print(
    f"  Top-1 accuracy       : "
    f"{fallback_summary['top1_accuracy']}"
)

print("\nQUALITY CHECKS")

for name, passed in checks.items():
    print(
        f"  [{'PASS' if passed else 'FAIL'}] "
        f"{name}"
    )

print(f"\nDECISION: {decision}")


# ============================================================
# 21. SAVE CSV REPORTS
# ============================================================

band_path = (
    REPORT_DIR /
    "confidence_band_analysis.csv"
)

margin_path = (
    REPORT_DIR /
    "confidence_margin_analysis.csv"
)

high_conf_path = (
    REPORT_DIR /
    "high_confidence_reliability.csv"
)

technique_path = (
    REPORT_DIR /
    "technique_confidence_analysis.csv"
)

false_high_path = (
    REPORT_DIR /
    "false_high_confidence_predictions.csv"
)

band_df.to_csv(
    band_path,
    index=False
)

margin_df.to_csv(
    margin_path,
    index=False
)

high_confidence_df.to_csv(
    high_conf_path,
    index=False
)

technique_df.to_csv(
    technique_path,
    index=False
)

false_high_df.to_csv(
    false_high_path,
    index=False
)


# ============================================================
# 22. JSON REPORT
# ============================================================

report = {
    "gate": "3C.2",
    "title": "Confidence vs Correctness Analysis",
    "decision": decision,

    "test_events": int(len(structured)),

    "top1_accuracy": top1_accuracy,

    "top1_score_statistics": {
        "mean": float(top1_probability.mean()),
        "median": float(np.median(top1_probability)),
        "minimum": float(top1_probability.min()),
        "maximum": float(top1_probability.max()),
    },

    "top1_top2_margin_statistics": {
        "mean": float(top1_margin.mean()),
        "median": float(np.median(top1_margin)),
        "minimum": float(top1_margin.min()),
        "maximum": float(top1_margin.max()),
    },

    "confidence_bands":
        band_df.to_dict(orient="records"),

    "margin_bands":
        margin_df.to_dict(orient="records"),

    "detected_vs_non_detected":
        detection_comparison,

    "high_confidence_reliability":
        high_confidence_df.to_dict(
            orient="records"
        ),

    "false_high_confidence_count":
        int(len(false_high_confidence_events)),

    "ambiguous_events":
        ambiguous_summary,

    "fallback":
        fallback_summary,

    "operational_confidence_recommendation":
        recommendation,

    "quality_checks":
        checks,

    "methodology_notes": [
        "The untouched test set was used for read-only evaluation.",
        "Production thresholds were loaded without modification.",
        "No model retraining was performed.",
        "No threshold recalibration was performed.",
        "No production artifacts were modified.",
        "Logistic regression outputs are treated as model scores, "
        "not automatically calibrated probabilities.",
        "Confidence and correctness are evaluated separately from "
        "causal explanation.",
        "Operational confidence bands are analytical recommendations "
        "only and are not production response policy."
    ],

    "output_files": {
        "confidence_bands":
            str(band_path),
        "margin_analysis":
            str(margin_path),
        "high_confidence_reliability":
            str(high_conf_path),
        "technique_analysis":
            str(technique_path),
        "false_high_confidence":
            str(false_high_path),
    },
}

report_path = (
    REPORT_DIR /
    "phase3c_gate3_2_confidence_correctness_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )

print("\nReport:")
print(report_path)

print("\nNO PRODUCTION ARTIFACTS WERE MODIFIED.")
print("=" * 72)