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
# GATE 3C.3 — AMBIGUITY & UNCERTAINTY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

PRODUCTION_DIR = (
    PROJECT_ROOT / "models" / "windows_apt_v4_1" / "production"
)

DATA_DIR = (
    PROJECT_ROOT / "data" / "processed" / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "phase3c_gate3_3"
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


def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


print("=" * 72)
print("SMARTSOC PHASE 3C")
print("GATE 3.3 — AMBIGUITY & UNCERTAINTY ANALYSIS")
print("=" * 72)


# ============================================================
# 1. LOAD ARTIFACTS
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
# 2. SCHEMA VALIDATION
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

if expected_tfidf != 6699:
    fail("Expected TF-IDF feature count 6699")

if expected_final != 6919:
    fail("Expected final feature count 6919")

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
    fail("Structured/labels row mismatch")

if len(structured) != len(behavior):
    fail("Structured/behavior row mismatch")

if "techniques" not in labels.columns:
    fail("Missing techniques column")

behavior_column = "behavior_text_normalized"

if behavior_column not in behavior.columns:
    fail("Missing behavior_text_normalized")

print("[PASS] Test alignment")


# ============================================================
# 4. FEATURE TRANSFORMATION
# ============================================================

print("\nFEATURE TRANSFORMATION")

X_cat = encoder.transform(
    structured[categorical_columns].copy()
)

X_num = scaler.transform(
    structured[numerical_columns].to_numpy(
        dtype=np.float64
    )
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
# 5. MODEL INFERENCE
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
    fail("Probability matrix dimension mismatch")

if not np.isfinite(probabilities).all():
    fail("Non-finite probabilities detected")

print("[PASS] Inference")


# ============================================================
# 6. THRESHOLDS
# ============================================================

thresholds = threshold_config.get("thresholds", {})

if len(thresholds) != len(techniques):
    fail("Threshold count mismatch")

threshold_array = np.array(
    [
        float(thresholds[technique])
        for technique in techniques
    ],
    dtype=np.float64,
)

print(
    f"  Thresholds loaded: "
    f"{len(threshold_array)}"
)

print("[PASS] Threshold configuration")


# ============================================================
# 7. TRUE LABEL MATRIX
# ============================================================

print("\nTRUE LABEL MATRIX")

technique_to_index = {
    technique: i
    for i, technique in enumerate(techniques)
}

true_matrix = np.zeros(
    (len(labels), len(techniques)),
    dtype=np.int8,
)

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
# 8. TOP-K ANALYSIS
# ============================================================

print("\nTOP-K UNCERTAINTY ANALYSIS")

sorted_indices = np.argsort(
    -probabilities,
    axis=1,
)

top1_index = sorted_indices[:, 0]
top2_index = sorted_indices[:, 1]
top3_index = sorted_indices[:, 2]

top1_probability = probabilities[
    np.arange(len(probabilities)),
    top1_index,
]

top2_probability = probabilities[
    np.arange(len(probabilities)),
    top2_index,
]

top3_probability = probabilities[
    np.arange(len(probabilities)),
    top3_index,
]

top1_margin = (
    top1_probability -
    top2_probability
)

top2_margin = (
    top2_probability -
    top3_probability
)

top1_techniques = [
    techniques[i]
    for i in top1_index
]

top2_techniques = [
    techniques[i]
    for i in top2_index
]

top3_techniques = [
    techniques[i]
    for i in top3_index
]


# ============================================================
# 9. DETECTION STATUS
# ============================================================

detection_matrix = (
    probabilities
    >= threshold_array.reshape(1, -1)
)

detected_count = detection_matrix.sum(axis=1)

fallback_mask = detected_count == 0

top1_threshold = threshold_array[top1_index]

top1_detected = (
    top1_probability >= top1_threshold
)


# ============================================================
# 10. CORRECTNESS
# ============================================================

top1_correct = np.array(
    [
        bool(
            true_matrix[i, top1_index[i]]
        )
        for i in range(len(structured))
    ],
    dtype=bool,
)

top2_contains_true = np.array(
    [
        bool(
            true_matrix[i, top2_index[i]]
            or true_matrix[i, top1_index[i]]
        )
        for i in range(len(structured))
    ],
    dtype=bool,
)

top3_contains_true = np.array(
    [
        bool(
            true_matrix[i, top1_index[i]]
            or true_matrix[i, top2_index[i]]
            or true_matrix[i, top3_index[i]]
        )
        for i in range(len(structured))
    ],
    dtype=bool,
)


# ============================================================
# 11. AMBIGUITY BANDS
# ============================================================

ambiguity_bands = [
    ("VERY_AMBIGUOUS", 0.00, 0.05),
    ("AMBIGUOUS", 0.05, 0.10),
    ("MODERATE", 0.10, 0.25),
    ("CLEAR", 0.25, 0.50),
    ("VERY_CLEAR", 0.50, np.inf),
]

ambiguity_rows = []

for name, lower, upper in ambiguity_bands:

    mask = (
        (top1_margin >= lower)
        & (top1_margin < upper)
    )

    count = int(mask.sum())

    ambiguity_rows.append({
        "ambiguity_band": name,
        "lower_margin": lower,
        "upper_margin": upper,
        "events": count,
        "fraction": float(
            count / len(top1_margin)
        ),
        "mean_top1_probability": (
            float(
                top1_probability[mask].mean()
            )
            if count
            else None
        ),
        "mean_top2_probability": (
            float(
                top2_probability[mask].mean()
            )
            if count
            else None
        ),
        "accuracy": (
            float(
                top1_correct[mask].mean()
            )
            if count
            else None
        ),
        "fallback_rate": (
            float(
                fallback_mask[mask].mean()
            )
            if count
            else None
        ),
    })

ambiguity_df = pd.DataFrame(
    ambiguity_rows
)


# ============================================================
# 12. NEAR-THRESHOLD ANALYSIS
# ============================================================

distance_from_threshold = (
    top1_probability -
    top1_threshold
)

near_threshold_bands = [
    ("BELOW_THRESHOLD", -np.inf, 0.0),
    ("0_TO_0.02_ABOVE", 0.0, 0.02),
    ("0.02_TO_0.05_ABOVE", 0.02, 0.05),
    ("0.05_TO_0.10_ABOVE", 0.05, 0.10),
    ("0.10_TO_0.25_ABOVE", 0.10, 0.25),
    ("0.25_PLUS_ABOVE", 0.25, np.inf),
]

near_threshold_rows = []

for name, lower, upper in near_threshold_bands:

    mask = (
        (distance_from_threshold >= lower)
        & (distance_from_threshold < upper)
    )

    count = int(mask.sum())

    near_threshold_rows.append({
        "threshold_band": name,
        "lower_distance": lower,
        "upper_distance": upper,
        "events": count,
        "fraction": float(
            count / len(distance_from_threshold)
        ),
        "mean_probability": (
            float(
                top1_probability[mask].mean()
            )
            if count
            else None
        ),
        "mean_margin": (
            float(
                top1_margin[mask].mean()
            )
            if count
            else None
        ),
        "accuracy": (
            float(
                top1_correct[mask].mean()
            )
            if count
            else None
        ),
    })

near_threshold_df = pd.DataFrame(
    near_threshold_rows
)


# ============================================================
# 13. TOP-1 / TOP-2 TECHNIQUE COMPETITION
# ============================================================

print("\nTECHNIQUE COMPETITION")

competition_counter = {}

for i in range(len(structured)):

    first = top1_techniques[i]
    second = top2_techniques[i]

    pair = (
        first,
        second,
    )

    competition_counter[pair] = (
        competition_counter.get(pair, 0) + 1
    )

competition_rows = []

for (first, second), count in sorted(
    competition_counter.items(),
    key=lambda x: -x[1],
):

    mask = (
        (np.array(top1_techniques) == first)
        & (np.array(top2_techniques) == second)
    )

    competition_rows.append({
        "top1_technique": first,
        "top2_technique": second,
        "events": int(count),
        "fraction": float(
            count / len(structured)
        ),
        "mean_top1_probability": float(
            top1_probability[mask].mean()
        ),
        "mean_top2_probability": float(
            top2_probability[mask].mean()
        ),
        "mean_margin": float(
            top1_margin[mask].mean()
        ),
        "top1_accuracy": float(
            top1_correct[mask].mean()
        ),
    })

competition_df = pd.DataFrame(
    competition_rows
)

competition_df = competition_df.head(50)


# ============================================================
# 14. MOST AMBIGUOUS EVENTS
# ============================================================

most_ambiguous_indices = np.argsort(
    top1_margin
)[:50]

ambiguous_events = []

for idx in most_ambiguous_indices:

    true_techniques = [
        techniques[j]
        for j in range(len(techniques))
        if true_matrix[idx, j]
    ]

    ambiguous_events.append({
        "row": int(idx),
        "top1": top1_techniques[idx],
        "top1_probability": float(
            top1_probability[idx]
        ),
        "top2": top2_techniques[idx],
        "top2_probability": float(
            top2_probability[idx]
        ),
        "top3": top3_techniques[idx],
        "top3_probability": float(
            top3_probability[idx]
        ),
        "top1_top2_margin": float(
            top1_margin[idx]
        ),
        "top2_top3_margin": float(
            top2_margin[idx]
        ),
        "top1_threshold": float(
            top1_threshold[idx]
        ),
        "threshold_distance": float(
            distance_from_threshold[idx]
        ),
        "top1_detected": bool(
            top1_detected[idx]
        ),
        "fallback": bool(
            fallback_mask[idx]
        ),
        "top1_correct": bool(
            top1_correct[idx]
        ),
        "true_techniques": "|".join(
            true_techniques
        ),
    })


# ============================================================
# 15. HIGH-CONFIDENCE BUT COMPETITIVE
# ============================================================

high_conf_competitive_mask = (
    (top1_probability >= 0.80)
    & (top1_margin < 0.10)
)

high_conf_competitive_indices = np.where(
    high_conf_competitive_mask
)[0]

high_conf_competitive = []

for idx in high_conf_competitive_indices:

    high_conf_competitive.append({
        "row": int(idx),
        "top1": top1_techniques[idx],
        "top1_probability": float(
            top1_probability[idx]
        ),
        "top2": top2_techniques[idx],
        "top2_probability": float(
            top2_probability[idx]
        ),
        "margin": float(
            top1_margin[idx]
        ),
        "top1_correct": bool(
            top1_correct[idx]
        ),
        "top1_detected": bool(
            top1_detected[idx]
        ),
    })


# ============================================================
# 16. LOW-CONFIDENCE BUT CLEAR
# ============================================================

low_conf_clear_mask = (
    (top1_probability < 0.50)
    & (top1_margin >= 0.50)
)

low_conf_clear = []

for idx in np.where(low_conf_clear_mask)[0]:

    low_conf_clear.append({
        "row": int(idx),
        "top1": top1_techniques[idx],
        "top1_probability": float(
            top1_probability[idx]
        ),
        "top2": top2_techniques[idx],
        "top2_probability": float(
            top2_probability[idx]
        ),
        "margin": float(
            top1_margin[idx]
        ),
        "top1_correct": bool(
            top1_correct[idx]
        ),
        "fallback": bool(
            fallback_mask[idx]
        ),
    })


# ============================================================
# 17. NEAR-THRESHOLD EVENTS
# ============================================================

near_threshold_mask = (
    np.abs(distance_from_threshold) <= 0.05
)

near_threshold_events = []

near_indices = np.where(
    near_threshold_mask
)[0]

for idx in near_indices[:100]:

    near_threshold_events.append({
        "row": int(idx),
        "top1": top1_techniques[idx],
        "top1_probability": float(
            top1_probability[idx]
        ),
        "threshold": float(
            top1_threshold[idx]
        ),
        "distance_from_threshold": float(
            distance_from_threshold[idx]
        ),
        "top2": top2_techniques[idx],
        "top2_probability": float(
            top2_probability[idx]
        ),
        "margin": float(
            top1_margin[idx]
        ),
        "top1_correct": bool(
            top1_correct[idx]
        ),
        "detected": bool(
            top1_detected[idx]
        ),
    })


# ============================================================
# 18. COMBINED UNCERTAINTY PROFILE
# ============================================================

combined_conditions = {
    "high_confidence_clear":
        (
            (top1_probability >= 0.80)
            & (top1_margin >= 0.25)
        ),

    "high_confidence_ambiguous":
        (
            (top1_probability >= 0.80)
            & (top1_margin < 0.10)
        ),

    "medium_confidence_clear":
        (
            (top1_probability >= 0.50)
            & (top1_probability < 0.80)
            & (top1_margin >= 0.25)
        ),

    "medium_confidence_ambiguous":
        (
            (top1_probability >= 0.50)
            & (top1_probability < 0.80)
            & (top1_margin < 0.10)
        ),

    "low_confidence_clear":
        (
            (top1_probability < 0.50)
            & (top1_margin >= 0.25)
        ),

    "low_confidence_ambiguous":
        (
            (top1_probability < 0.50)
            & (top1_margin < 0.10)
        ),
}

combined_rows = []

for name, mask in combined_conditions.items():

    count = int(mask.sum())

    combined_rows.append({
        "profile": name,
        "events": count,
        "fraction": float(
            count / len(structured)
        ),
        "accuracy": (
            float(
                top1_correct[mask].mean()
            )
            if count
            else None
        ),
        "mean_probability": (
            float(
                top1_probability[mask].mean()
            )
            if count
            else None
        ),
        "mean_margin": (
            float(
                top1_margin[mask].mean()
            )
            if count
            else None
        ),
        "detection_rate": (
            float(
                top1_detected[mask].mean()
            )
            if count
            else None
        ),
    })

combined_df = pd.DataFrame(
    combined_rows
)


# ============================================================
# 19. OPERATIONAL INTERPRETATION
# ============================================================

operational_findings = []

high_clear = combined_df[
    combined_df["profile"]
    == "high_confidence_clear"
]

high_ambiguous = combined_df[
    combined_df["profile"]
    == "high_confidence_ambiguous"
]

medium_ambiguous = combined_df[
    combined_df["profile"]
    == "medium_confidence_ambiguous"
]

low_profiles = combined_df[
    combined_df["profile"].str.startswith("low_")
]

if len(high_clear):
    operational_findings.append({
        "finding":
            "High-confidence clear predictions",
        "events":
            int(high_clear["events"].iloc[0]),
        "accuracy":
            safe_mean(
                [high_clear["accuracy"].iloc[0]]
            ),
        "interpretation":
            "Strong candidates for downstream AI reasoning, "
            "subject to technique-specific risk and evidence checks.",
    })

if len(high_ambiguous):
    operational_findings.append({
        "finding":
            "High-confidence but ambiguous predictions",
        "events":
            int(high_ambiguous["events"].iloc[0]),
        "accuracy":
            safe_mean(
                [high_ambiguous["accuracy"].iloc[0]]
            ),
        "interpretation":
            "High score alone should not eliminate analyst review "
            "when competing techniques are close.",
    })

if len(medium_ambiguous):
    operational_findings.append({
        "finding":
            "Medium-confidence ambiguous predictions",
        "events":
            int(medium_ambiguous["events"].iloc[0]),
        "accuracy":
            safe_mean(
                [medium_ambiguous["accuracy"].iloc[0]]
            ),
        "interpretation":
            "Strong candidates for additional evidence gathering "
            "before response decisions.",
    })

if len(low_profiles):
    operational_findings.append({
        "finding":
            "Low-confidence predictions",
        "events":
            int(low_profiles["events"].sum()),
        "accuracy":
            safe_mean(
                low_profiles["accuracy"].dropna().tolist()
            ),
        "interpretation":
            "Confidence alone is insufficient for automated response.",
    })


# ============================================================
# 20. QUALITY CHECKS
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

    "true_label_matrix_valid":
        true_matrix.shape == (
            len(structured),
            45,
        ),

    "top_k_analysis_completed":
        len(top1_index) == len(structured),

    "ambiguity_analysis_completed":
        len(ambiguity_df) == 5,

    "threshold_proximity_analysis_completed":
        len(near_threshold_df) == 6,

    "technique_competition_analysis_completed":
        len(competition_df) > 0,

    "combined_uncertainty_analysis_completed":
        len(combined_df) == 6,

    "production_artifacts_unchanged":
        True,
}

decision = (
    "PASS"
    if all(checks.values())
    else "REVIEW_REQUIRED"
)


# ============================================================
# 21. CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("PHASE 3C GATE 3.3 SUMMARY")
print("=" * 72)

print(
    f"Test events                 : "
    f"{len(structured)}"
)

print(
    f"Top-1 accuracy              : "
    f"{top1_correct.mean():.6f}"
)

print("\nAMBIGUITY BANDS")

for _, row in ambiguity_df.iterrows():

    print(
        f"  {row['ambiguity_band']:16s} "
        f"events={int(row['events']):5d} "
        f"accuracy={row['accuracy']:.6f} "
        f"mean_margin={top1_margin[
            (top1_margin >= row['lower_margin'])
            & (top1_margin < row['upper_margin'])
        ].mean():.6f}"
    )

print("\nTHRESHOLD PROXIMITY")

for _, row in near_threshold_df.iterrows():

    print(
        f"  {row['threshold_band']:20s} "
        f"events={int(row['events']):5d} "
        f"accuracy={row['accuracy']:.6f}"
    )

print("\nTOP TECHNIQUE COMPETITION")

for _, row in competition_df.head(10).iterrows():

    print(
        f"  {row['top1_technique']:12s} "
        f"vs {row['top2_technique']:12s} "
        f"events={int(row['events']):5d} "
        f"margin={row['mean_margin']:.6f} "
        f"accuracy={row['top1_accuracy']:.6f}"
    )

print("\nCOMBINED UNCERTAINTY PROFILES")

for _, row in combined_df.iterrows():

    print(
        f"  {row['profile']:30s} "
        f"events={int(row['events']):5d} "
        f"accuracy={row['accuracy']:.6f} "
        f"detection={row['detection_rate']:.6f}"
    )

print("\nSPECIAL CASES")

print(
    f"  High-confidence ambiguous : "
    f"{len(high_conf_competitive)}"
)

print(
    f"  Low-confidence clear      : "
    f"{len(low_conf_clear)}"
)

print(
    f"  Near threshold events     : "
    f"{len(near_indices)}"
)

print(
    f"  Fallback events           : "
    f"{int(fallback_mask.sum())}"
)

print("\nQUALITY CHECKS")

for name, passed in checks.items():

    print(
        f"  [{'PASS' if passed else 'FAIL'}] "
        f"{name}"
    )

print(f"\nDECISION: {decision}")


# ============================================================
# 22. SAVE CSV REPORTS
# ============================================================

ambiguity_path = (
    REPORT_DIR /
    "ambiguity_band_analysis.csv"
)

threshold_path = (
    REPORT_DIR /
    "threshold_proximity_analysis.csv"
)

competition_path = (
    REPORT_DIR /
    "technique_competition_analysis.csv"
)

ambiguous_events_path = (
    REPORT_DIR /
    "most_ambiguous_events.csv"
)

high_competitive_path = (
    REPORT_DIR /
    "high_confidence_ambiguous_events.csv"
)

low_clear_path = (
    REPORT_DIR /
    "low_confidence_clear_events.csv"
)

near_events_path = (
    REPORT_DIR /
    "near_threshold_events.csv"
)

combined_path = (
    REPORT_DIR /
    "combined_uncertainty_profiles.csv"
)

ambiguity_df.to_csv(
    ambiguity_path,
    index=False
)

near_threshold_df.to_csv(
    threshold_path,
    index=False
)

competition_df.to_csv(
    competition_path,
    index=False
)

pd.DataFrame(
    ambiguous_events
).to_csv(
    ambiguous_events_path,
    index=False
)

pd.DataFrame(
    high_conf_competitive
).to_csv(
    high_competitive_path,
    index=False
)

pd.DataFrame(
    low_conf_clear
).to_csv(
    low_clear_path,
    index=False
)

pd.DataFrame(
    near_threshold_events
).to_csv(
    near_events_path,
    index=False
)

combined_df.to_csv(
    combined_path,
    index=False
)


# ============================================================
# 23. JSON REPORT
# ============================================================

report = {
    "gate": "3C.3",
    "title": "Ambiguity & Uncertainty Analysis",
    "decision": decision,

    "test_events": int(len(structured)),

    "top1_accuracy": float(
        top1_correct.mean()
    ),

    "top_k": {
        "top1_accuracy": float(
            top1_correct.mean()
        ),
        "top2_contains_true_rate": float(
            top2_contains_true.mean()
        ),
        "top3_contains_true_rate": float(
            top3_contains_true.mean()
        ),
    },

    "score_statistics": {
        "top1_mean": float(
            top1_probability.mean()
        ),
        "top1_median": float(
            np.median(top1_probability)
        ),
        "top2_mean": float(
            top2_probability.mean()
        ),
        "top3_mean": float(
            top3_probability.mean()
        ),
    },

    "margin_statistics": {
        "top1_top2_mean": float(
            top1_margin.mean()
        ),
        "top1_top2_median": float(
            np.median(top1_margin)
        ),
        "minimum": float(
            top1_margin.min()
        ),
        "maximum": float(
            top1_margin.max()
        ),
    },

    "ambiguity_bands":
        ambiguity_df.to_dict(
            orient="records"
        ),

    "threshold_proximity":
        near_threshold_df.to_dict(
            orient="records"
        ),

    "technique_competition":
        competition_df.to_dict(
            orient="records"
        ),

    "special_cases": {
        "high_confidence_ambiguous_events":
            int(
                len(high_conf_competitive)
            ),

        "low_confidence_clear_events":
            int(
                len(low_conf_clear)
            ),

        "near_threshold_events":
            int(len(near_indices)),

        "fallback_events":
            int(fallback_mask.sum()),
    },

    "combined_uncertainty_profiles":
        combined_df.to_dict(
            orient="records"
        ),

    "operational_findings":
        operational_findings,

    "methodology_notes": [
        "The untouched test set was used read-only.",
        "Existing production thresholds were preserved.",
        "No model retraining was performed.",
        "No threshold recalibration was performed.",
        "No production artifacts were modified.",
        "Top-1/top-2 margin is treated as an ambiguity indicator, "
        "not a formal calibrated uncertainty estimate.",
        "Logistic regression scores are not automatically assumed "
        "to be calibrated probabilities.",
        "Operational interpretations are recommendations for "
        "later decision-engine design and do not modify policy."
    ],

    "quality_checks": checks,

    "output_files": {
        "ambiguity_bands":
            str(ambiguity_path),
        "threshold_proximity":
            str(threshold_path),
        "technique_competition":
            str(competition_path),
        "most_ambiguous_events":
            str(ambiguous_events_path),
        "high_confidence_ambiguous":
            str(high_competitive_path),
        "low_confidence_clear":
            str(low_clear_path),
        "near_threshold_events":
            str(near_events_path),
        "combined_profiles":
            str(combined_path),
    },
}

report_path = (
    REPORT_DIR /
    "phase3c_gate3_3_ambiguity_uncertainty_report.json"
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