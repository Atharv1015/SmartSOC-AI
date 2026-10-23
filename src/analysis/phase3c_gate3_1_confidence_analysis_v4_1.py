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
# GATE 3C.1 — CONFIDENCE & UNCERTAINTY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

PRODUCTION_DIR = PROJECT_ROOT / "models" / "windows_apt_v4_1" / "production"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "windows_apt_v4_1"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase3c_gate3_1"

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


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return None
    except Exception:
        return None


print("=" * 72)
print("SMARTSOC PHASE 3C")
print("GATE 3C.1 — CONFIDENCE & UNCERTAINTY ANALYSIS")
print("=" * 72)


# ============================================================
# 1. PRODUCTION ARTIFACT VALIDATION
# ============================================================

print("\nPRODUCTION ARTIFACTS")

required_artifacts = {
    "model.joblib": MODEL_PATH,
    "structured_encoder.joblib": ENCODER_PATH,
    "numerical_scaler.joblib": SCALER_PATH,
    "tfidf_vectorizer.joblib": TFIDF_PATH,
    "multilabel_encoder.joblib": MLB_PATH,
    "feature_schema.json": SCHEMA_PATH,
    "thresholds.json": THRESHOLD_PATH,
}

for name, path in required_artifacts.items():
    if path.exists():
        print(f"[PASS] {name}")
    else:
        fail(f"Missing production artifact: {path}")


print("\nLOADING PRODUCTION ARTIFACTS")

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
# 2. SCHEMA VALIDATION
# ============================================================

categorical_columns = schema["categorical_columns"]
numerical_columns = schema["numerical_columns"]
mitre_techniques = schema["mitre_techniques"]

expected_structured = int(schema["final_feature_count"] - schema["tfidf_feature_count"])
expected_final = int(schema["final_feature_count"])
expected_tfidf = int(schema["tfidf_feature_count"])

print("\nFEATURE SCHEMA")
print(f"  Categorical : {len(categorical_columns)}")
print(f"  Numerical   : {len(numerical_columns)}")
print(f"  Techniques  : {len(mitre_techniques)}")
print(f"  Final       : {expected_final}")

if len(categorical_columns) != 4:
    fail("Expected 4 categorical columns")

if len(numerical_columns) != 58:
    fail("Expected 58 numerical columns")

if len(mitre_techniques) != 45:
    fail("Expected 45 MITRE techniques")


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print("\nINPUT DATA")

structured = pd.read_csv(STRUCTURED_TEST)
labels = pd.read_csv(LABELS_TEST)
behavior = pd.read_csv(BEHAVIOR_TEST)

print(f"  Structured rows: {len(structured)}")
print(f"  Labels rows    : {len(labels)}")
print(f"  Behavior rows  : {len(behavior)}")

if len(structured) != len(labels):
    fail("Structured/label row mismatch")

if len(structured) != len(behavior):
    fail("Structured/behavior row mismatch")

behavior_column = "behavior_text_normalized"

if behavior_column not in behavior.columns:
    fail(f"Missing behavior column: {behavior_column}")

print("[PASS] Input alignment")


# ============================================================
# 4. BUILD PRODUCTION FEATURE MATRIX
# ============================================================

print("\nBUILDING PRODUCTION FEATURE MATRIX")

X_cat_raw = structured[categorical_columns].copy()

X_cat = encoder.transform(X_cat_raw)

X_num_raw = structured[numerical_columns].to_numpy(dtype=np.float64)
X_num = scaler.transform(X_num_raw)

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
    fail(
        f"Final feature dimension mismatch: "
        f"{X_final.shape[1]} != {expected_final}"
    )

if X_text.shape[1] != expected_tfidf:
    fail(
        f"TF-IDF dimension mismatch: "
        f"{X_text.shape[1]} != {expected_tfidf}"
    )

print("[PASS] Production feature dimensions")


# ============================================================
# 5. MODEL INFERENCE
# ============================================================

print("\nMODEL INFERENCE")

probabilities = model.predict_proba(X_final)

probabilities = np.asarray(probabilities, dtype=np.float64)

print(f"  Probability matrix: {probabilities.shape}")

if probabilities.shape != (len(structured), len(mitre_techniques)):
    fail(
        "Probability matrix dimension mismatch: "
        f"{probabilities.shape}"
    )

if not np.isfinite(probabilities).all():
    fail("Non-finite probabilities detected")

print("[PASS] Probability matrix")


# ============================================================
# 6. THRESHOLD VALIDATION
# ============================================================

print("\nTHRESHOLD CONFIGURATION")

thresholds = threshold_config.get("thresholds", {})

if not thresholds:
    fail("No thresholds found")

threshold_values = {}

for technique in mitre_techniques:
    if technique not in thresholds:
        fail(f"Missing threshold for {technique}")

    value = safe_float(thresholds[technique])

    if value is None:
        fail(f"Invalid threshold for {technique}")

    if not 0.0 <= value <= 1.0:
        fail(f"Threshold outside [0,1] for {technique}")

    threshold_values[technique] = value

print(f"  Thresholds loaded: {len(threshold_values)}")
print("[PASS] Threshold configuration")


# ============================================================
# 7. BASIC CONFIDENCE STATISTICS
# ============================================================

print("\nCALCULATING CONFIDENCE STATISTICS")

top1_index = np.argmax(probabilities, axis=1)
top1_probability = probabilities[
    np.arange(len(probabilities)),
    top1_index
]

sorted_probabilities = np.sort(probabilities, axis=1)

top2_probability = sorted_probabilities[:, -2]

top1_margin = top1_probability - top2_probability

threshold_array = np.array(
    [threshold_values[t] for t in mitre_techniques],
    dtype=np.float64,
)

top1_threshold = threshold_array[top1_index]

top1_threshold_distance = top1_probability - top1_threshold


# ============================================================
# 8. MULTI-LABEL DETECTION MATRIX
# ============================================================

detection_matrix = probabilities >= threshold_array.reshape(1, -1)

detected_count_per_event = detection_matrix.sum(axis=1)

fallback_events = detected_count_per_event == 0

fallback_count = int(fallback_events.sum())

print(f"  Events: {len(probabilities)}")
print(f"  Events with >=1 detection: {int((~fallback_events).sum())}")
print(f"  Fallback events: {fallback_count}")

fallback_rate = fallback_count / len(probabilities)


# ============================================================
# 9. CONFIDENCE BANDS
# ============================================================

def confidence_band(p):
    if p < 0.50:
        return "low"
    elif p < 0.80:
        return "medium"
    elif p < 0.95:
        return "high"
    else:
        return "very_high"


confidence_bands = np.array(
    [confidence_band(float(p)) for p in top1_probability]
)

band_order = [
    "low",
    "medium",
    "high",
    "very_high",
]

band_summary = {}

for band in band_order:
    mask = confidence_bands == band

    band_summary[band] = {
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "mean_top1_probability": (
            float(top1_probability[mask].mean())
            if mask.any()
            else None
        ),
        "mean_top1_margin": (
            float(top1_margin[mask].mean())
            if mask.any()
            else None
        ),
    }


# ============================================================
# 10. TOP-1 / TOP-2 MARGIN BANDS
# ============================================================

margin_bins = [
    ("<0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    (">=0.50", 0.50, np.inf),
]

margin_summary = {}

for name, low, high in margin_bins:
    mask = (top1_margin >= low) & (top1_margin < high)

    margin_summary[name] = {
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "mean_top1_probability": (
            float(top1_probability[mask].mean())
            if mask.any()
            else None
        ),
    }


# ============================================================
# 11. THRESHOLD-DISTANCE BANDS
# ============================================================

distance_bins = [
    ("below_threshold", -np.inf, 0.0),
    ("0_to_0.05_above", 0.0, 0.05),
    ("0.05_to_0.20_above", 0.05, 0.20),
    ("0.20_to_0.50_above", 0.20, 0.50),
    ("0.50_plus_above", 0.50, np.inf),
]

threshold_distance_summary = {}

for name, low, high in distance_bins:
    mask = (
        (top1_threshold_distance >= low)
        & (top1_threshold_distance < high)
    )

    threshold_distance_summary[name] = {
        "count": int(mask.sum()),
        "fraction": float(mask.mean()),
        "mean_top1_probability": (
            float(top1_probability[mask].mean())
            if mask.any()
            else None
        ),
    }


# ============================================================
# 12. TRUE-LABEL ALIGNMENT
# ============================================================

print("\nTRUE LABEL ANALYSIS")

true_label_matrix = np.zeros(
    (len(labels), len(mitre_techniques)),
    dtype=np.int8
)

if "techniques" not in labels.columns:
    fail("labels_test.csv missing 'techniques' column")

unknown_true_labels = 0
true_label_count = 0

for i, value in enumerate(labels["techniques"].fillna("")):
    if not str(value).strip():
        continue

    raw = str(value)

    # Support common delimiters.
    normalized = (
        raw.replace(";", ",")
        .replace("|", ",")
        .replace(" ", ",")
    )

    tokens = [
        token.strip()
        for token in normalized.split(",")
        if token.strip()
    ]

    for token in tokens:
        if token in mitre_techniques:
            true_label_matrix[i, mitre_techniques.index(token)] = 1
            true_label_count += 1
        else:
            unknown_true_labels += 1

print(f"  True labels: {true_label_count}")
print(f"  Unknown labels: {unknown_true_labels}")


# ============================================================
# 13. TOP-1 CORRECTNESS
# ============================================================

has_true_label = true_label_matrix.sum(axis=1) > 0

top1_correct = np.zeros(len(probabilities), dtype=bool)

for i in range(len(probabilities)):
    if has_true_label[i]:
        top1_correct[i] = bool(
            true_label_matrix[i, top1_index[i]] == 1
        )

top1_correct_count = int(top1_correct.sum())

top1_evaluable_count = int(has_true_label.sum())

top1_accuracy = (
    top1_correct_count / top1_evaluable_count
    if top1_evaluable_count
    else None
)

print(f"  Evaluable events: {top1_evaluable_count}")
print(f"  Top-1 correct: {top1_correct_count}")
print(f"  Top-1 accuracy: {top1_accuracy}")


# ============================================================
# 14. CONFIDENCE VS CORRECTNESS
# ============================================================

confidence_correctness = {}

for band in band_order:
    mask = (confidence_bands == band) & has_true_label

    if mask.any():
        correct = top1_correct[mask]

        confidence_correctness[band] = {
            "count": int(mask.sum()),
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "mean_probability": float(
                top1_probability[mask].mean()
            ),
        }
    else:
        confidence_correctness[band] = {
            "count": 0,
            "correct": 0,
            "accuracy": None,
            "mean_probability": None,
        }


# ============================================================
# 15. HIGH-CONFIDENCE ERROR CHECK
# ============================================================

high_confidence_masks = {
    ">=0.80": top1_probability >= 0.80,
    ">=0.95": top1_probability >= 0.95,
    ">=0.99": top1_probability >= 0.99,
}

high_confidence_analysis = {}

for name, mask in high_confidence_masks.items():
    evaluable = mask & has_true_label

    if evaluable.any():
        correct = top1_correct[evaluable]

        high_confidence_analysis[name] = {
            "events": int(mask.sum()),
            "evaluable_events": int(evaluable.sum()),
            "correct": int(correct.sum()),
            "incorrect": int((~correct).sum()),
            "accuracy": float(correct.mean()),
            "mean_probability": float(
                top1_probability[evaluable].mean()
            ),
        }
    else:
        high_confidence_analysis[name] = {
            "events": int(mask.sum()),
            "evaluable_events": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": None,
            "mean_probability": None,
        }


# ============================================================
# 16. PER-TECHNIQUE CONFIDENCE
# ============================================================

per_technique = []

for j, technique in enumerate(mitre_techniques):

    probs = probabilities[:, j]
    detected = detection_matrix[:, j]
    true = true_label_matrix[:, j].astype(bool)

    row = {
        "technique": technique,
        "threshold": float(threshold_array[j]),
        "support": int(true.sum()),
        "detected_count": int(detected.sum()),
        "mean_probability": float(probs.mean()),
        "median_probability": float(np.median(probs)),
        "max_probability": float(probs.max()),
        "p95_probability": float(np.percentile(probs, 95)),
        "mean_detected_probability": (
            float(probs[detected].mean())
            if detected.any()
            else None
        ),
        "mean_true_probability": (
            float(probs[true].mean())
            if true.any()
            else None
        ),
        "mean_false_probability": (
            float(probs[~true].mean())
            if (~true).any()
            else None
        ),
    }

    if detected.any():
        row["detection_rate"] = float(detected.mean())
    else:
        row["detection_rate"] = 0.0

    per_technique.append(row)

per_technique_df = pd.DataFrame(per_technique)


# ============================================================
# 17. MOST UNCERTAIN EVENTS
# ============================================================

uncertain_order = np.argsort(top1_margin)

uncertain_events = []

for idx in uncertain_order[:20]:
    uncertain_events.append({
        "row": int(idx),
        "top1_technique": mitre_techniques[int(top1_index[idx])],
        "top1_probability": float(top1_probability[idx]),
        "top2_probability": float(top2_probability[idx]),
        "top1_margin": float(top1_margin[idx]),
        "top1_threshold": float(top1_threshold[idx]),
        "threshold_distance": float(
            top1_threshold_distance[idx]
        ),
        "detected": bool(detection_matrix[idx].any()),
    })


# ============================================================
# 18. HIGHEST-CONFIDENCE EVENTS
# ============================================================

highest_order = np.argsort(-top1_probability)

highest_confidence_events = []

for idx in highest_order[:20]:
    highest_confidence_events.append({
        "row": int(idx),
        "top1_technique": mitre_techniques[int(top1_index[idx])],
        "top1_probability": float(top1_probability[idx]),
        "top2_probability": float(top2_probability[idx]),
        "top1_margin": float(top1_margin[idx]),
        "top1_threshold": float(top1_threshold[idx]),
        "threshold_distance": float(
            top1_threshold_distance[idx]
        ),
        "detected": bool(detection_matrix[idx].any()),
        "top1_correct": (
            bool(top1_correct[idx])
            if has_true_label[idx]
            else None
        ),
    })


# ============================================================
# 19. FALSE HIGH-CONFIDENCE EVENTS
# ============================================================

false_high_confidence = []

false_mask = (
    has_true_label
    & (~top1_correct)
    & (top1_probability >= 0.80)
)

false_indices = np.where(false_mask)[0]

false_indices = false_indices[
    np.argsort(-top1_probability[false_indices])
]

for idx in false_indices[:20]:
    true_techniques = [
        mitre_techniques[j]
        for j in range(len(mitre_techniques))
        if true_label_matrix[idx, j] == 1
    ]

    false_high_confidence.append({
        "row": int(idx),
        "predicted_technique": mitre_techniques[int(top1_index[idx])],
        "predicted_probability": float(top1_probability[idx]),
        "top2_probability": float(top2_probability[idx]),
        "margin": float(top1_margin[idx]),
        "true_techniques": true_techniques,
    })


# ============================================================
# 20. DECISION LOGIC
# ============================================================

checks = {
    "production_feature_dimension_valid":
        X_final.shape[1] == expected_final,

    "probability_matrix_valid":
        probabilities.shape == (
            len(structured),
            len(mitre_techniques)
        ),

    "probabilities_finite":
        bool(np.isfinite(probabilities).all()),

    "threshold_count_valid":
        len(threshold_values) == len(mitre_techniques),

    "confidence_analysis_completed":
        len(band_summary) == 4,

    "margin_analysis_completed":
        len(margin_summary) == 5,

    "per_technique_analysis_completed":
        len(per_technique_df) == len(mitre_techniques),
}

all_checks_pass = all(checks.values())

# This gate is an analytical gate, not a requirement for
# perfect calibration. High-confidence errors are documented,
# not automatically treated as a model failure.
if all_checks_pass:
    decision = "PASS"
else:
    decision = "REVIEW_REQUIRED"


# ============================================================
# 21. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("PHASE 3C GATE 3.1 SUMMARY")
print("=" * 72)

print(f"Test events                 : {len(probabilities)}")
print(f"MITRE techniques            : {len(mitre_techniques)}")
print(f"Fallback events             : {fallback_count}")
print(f"Fallback rate               : {fallback_rate:.6f}")

print("\nTOP-1 CONFIDENCE")
print(f"  Mean                      : {top1_probability.mean():.6f}")
print(f"  Median                    : {np.median(top1_probability):.6f}")
print(f"  Minimum                   : {top1_probability.min():.6f}")
print(f"  Maximum                   : {top1_probability.max():.6f}")

print("\nTOP-1 / TOP-2 MARGIN")
print(f"  Mean                      : {top1_margin.mean():.6f}")
print(f"  Median                    : {np.median(top1_margin):.6f}")
print(f"  Minimum                   : {top1_margin.min():.6f}")
print(f"  Maximum                   : {top1_margin.max():.6f}")

print("\nCONFIDENCE BANDS")

for band in band_order:
    item = band_summary[band]

    print(
        f"  {band:10s}: "
        f"{item['count']:5d} "
        f"({item['fraction']:.4f}) "
        f"mean={item['mean_top1_probability']}"
    )

print("\nTOP-1 CORRECTNESS")

if top1_accuracy is not None:
    print(f"  Evaluable events            : {top1_evaluable_count}")
    print(f"  Top-1 correct               : {top1_correct_count}")
    print(f"  Top-1 accuracy              : {top1_accuracy:.6f}")
else:
    print("  No evaluable true labels")

print("\nHIGH-CONFIDENCE RELIABILITY")

for name, item in high_confidence_analysis.items():
    print(
        f"  {name:6s}: "
        f"events={item['events']} "
        f"evaluable={item['evaluable_events']} "
        f"incorrect={item['incorrect']} "
        f"accuracy={item['accuracy']}"
    )

print("\nQUALITY CHECKS")

for name, passed in checks.items():
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {name}"
    )

print(f"\nDECISION: {decision}")


# ============================================================
# 22. SAVE REPORTS
# ============================================================

per_technique_path = (
    REPORT_DIR / "per_technique_confidence.csv"
)

per_technique_df.to_csv(
    per_technique_path,
    index=False
)

uncertain_path = (
    REPORT_DIR / "most_uncertain_events.csv"
)

pd.DataFrame(uncertain_events).to_csv(
    uncertain_path,
    index=False
)

highest_path = (
    REPORT_DIR / "highest_confidence_events.csv"
)

pd.DataFrame(highest_confidence_events).to_csv(
    highest_path,
    index=False
)

false_high_path = (
    REPORT_DIR / "false_high_confidence_events.csv"
)

pd.DataFrame(false_high_confidence).to_csv(
    false_high_path,
    index=False
)


report = {
    "gate": "3C.1",
    "title": "Confidence & Uncertainty Analysis",
    "decision": decision,

    "production": {
        "structured_features": int(expected_structured),
        "tfidf_features": int(expected_tfidf),
        "final_features": int(expected_final),
        "mitre_techniques": int(len(mitre_techniques)),
        "threshold_count": int(len(threshold_values)),
    },

    "test": {
        "rows": int(len(probabilities)),
        "true_labels": int(true_label_count),
        "unknown_true_labels": int(unknown_true_labels),
    },

    "fallback": {
        "events": int(fallback_count),
        "rate": float(fallback_rate),
    },

    "top1_confidence": {
        "mean": float(top1_probability.mean()),
        "median": float(np.median(top1_probability)),
        "minimum": float(top1_probability.min()),
        "maximum": float(top1_probability.max()),
    },

    "top1_top2_margin": {
        "mean": float(top1_margin.mean()),
        "median": float(np.median(top1_margin)),
        "minimum": float(top1_margin.min()),
        "maximum": float(top1_margin.max()),
    },

    "confidence_bands": band_summary,

    "margin_bands": margin_summary,

    "threshold_distance_bands":
        threshold_distance_summary,

    "top1_correctness": {
        "evaluable_events": int(top1_evaluable_count),
        "correct_events": int(top1_correct_count),
        "accuracy": (
            float(top1_accuracy)
            if top1_accuracy is not None
            else None
        ),
    },

    "confidence_vs_correctness":
        confidence_correctness,

    "high_confidence_analysis":
        high_confidence_analysis,

    "fallback_note":
        "Fallback events are events with no technique probability "
        "meeting its production threshold. The production prediction "
        "configuration may use highest_probability fallback.",

    "uncertainty_note":
        "Probability values from logistic regression are model scores "
        "and are not automatically calibrated probabilities. "
        "This gate measures confidence behavior and reliability "
        "without modifying thresholds or production artifacts.",

    "quality_checks": checks,

    "output_files": {
        "per_technique":
            str(per_technique_path),
        "most_uncertain":
            str(uncertain_path),
        "highest_confidence":
            str(highest_path),
        "false_high_confidence":
            str(false_high_path),
    },
}

report_path = (
    REPORT_DIR /
    "phase3c_gate3_1_confidence_analysis_report.json"
)

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\nReport:")
print(report_path)

print("\nNO PRODUCTION ARTIFACTS WERE MODIFIED.")
print("=" * 72)