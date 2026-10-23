import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# SMARTSOC PHASE 2 GATE 2.4D
# REAL TRAINING-EVENT REPLAY VALIDATION
#
# PURPOSE
# -------
# Test the production inference pipeline using REAL rows from
# the training dataset rather than synthetic feature averages.
#
# This determines whether the production artifacts can correctly
# recognize actual training examples belonging to the target
# MITRE techniques.
#
# IMPORTANT
# ---------
# Diagnostic only.
#
# No:
#   - model retraining
#   - threshold modification
#   - dataset modification
#   - artifact modification
#   - malicious command execution
#
# ============================================================


ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_4d"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TARGET TECHNIQUES
# ============================================================

TARGET_TECHNIQUES = [
    "T1027.004",
    "T1059.001",
    "T1059.003",
    "T1105",
    "T1112",
    "T1543.003",
    "T1070.004",
]


# Number of real training examples per technique.
#
# Keep this small because this is a diagnostic replay, not a
# full training evaluation.
#
# The examples are deterministic so the experiment is
# reproducible.
# ============================================================

SAMPLES_PER_TECHNIQUE = 10


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print()
print("=" * 78)
print("SMARTSOC PHASE 2 GATE 2.4D")
print("REAL TRAINING-EVENT REPLAY VALIDATION")
print("=" * 78)
print()

print("Loading production artifacts...")

MODEL = joblib.load(
    MODEL_DIR / "model.joblib"
)

ENCODER = joblib.load(
    MODEL_DIR / "structured_encoder.joblib"
)

SCALER = joblib.load(
    MODEL_DIR / "numerical_scaler.joblib"
)

TFIDF = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

LABEL_ENCODER = joblib.load(
    MODEL_DIR / "multilabel_encoder.joblib"
)

with open(
    MODEL_DIR / "feature_schema.json",
    "r",
    encoding="utf-8"
) as f:
    SCHEMA = json.load(f)

with open(
    MODEL_DIR / "thresholds.json",
    "r",
    encoding="utf-8"
) as f:
    THRESHOLD_DATA = json.load(f)

THRESHOLDS = THRESHOLD_DATA[
    "thresholds"
]


# ============================================================
# SCHEMA
# ============================================================

CATEGORICAL_COLUMNS = list(
    SCHEMA["categorical_columns"]
)

NUMERICAL_COLUMNS = list(
    SCHEMA["numerical_columns"]
)

STRUCTURED_COLUMNS = (
    CATEGORICAL_COLUMNS
    + NUMERICAL_COLUMNS
)

TECHNIQUES = [
    str(x)
    for x in LABEL_ENCODER.classes_
]


print()
print("Production artifact validation")
print("-" * 78)

print(
    f"Model type           : "
    f"{type(MODEL).__name__}"
)

print(
    f"Categorical columns  : "
    f"{len(CATEGORICAL_COLUMNS)}"
)

print(
    f"Numerical columns    : "
    f"{len(NUMERICAL_COLUMNS)}"
)

print(
    f"Structured columns   : "
    f"{len(STRUCTURED_COLUMNS)}"
)

print(
    f"TF-IDF vocabulary    : "
    f"{len(TFIDF.vocabulary_)}"
)

print(
    f"MITRE classes        : "
    f"{len(TECHNIQUES)}"
)

print(
    f"Threshold entries    : "
    f"{len(THRESHOLDS)}"
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print()
print("Loading training dataset...")

STRUCTURED_TRAIN = pd.read_csv(
    DATA_DIR / "structured_train.csv"
)

LABELS_TRAIN = pd.read_csv(
    DATA_DIR / "labels_train.csv"
)

BEHAVIOR_TRAIN = pd.read_csv(
    DATA_DIR / "behavior_text_train.csv"
)


# ============================================================
# VALIDATE ROW COUNTS
# ============================================================

if not (
    len(STRUCTURED_TRAIN)
    == len(LABELS_TRAIN)
    == len(BEHAVIOR_TRAIN)
):

    raise RuntimeError(
        "Training dataset row alignment failure."
    )


print(
    f"Training rows        : "
    f"{len(STRUCTURED_TRAIN)}"
)

print(
    f"Behavior rows        : "
    f"{len(BEHAVIOR_TRAIN)}"
)


# ============================================================
# LABEL PARSER
# ============================================================

def parse_labels(
    value
):

    if pd.isna(value):

        return set()

    return {
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    }


# ============================================================
# FIND TRAINING ROWS FOR TECHNIQUE
# ============================================================

def technique_mask(
    technique
):

    return LABELS_TRAIN[
        "techniques"
    ].apply(
        lambda value:
        technique
        in parse_labels(value)
    )


# ============================================================
# SAMPLE REAL TRAINING ROWS
#
# Deterministic selection:
# take the first N matching rows.
#
# We intentionally use the REAL structured row and the REAL
# behavior_text_normalized row.
# ============================================================

selected_rows = []

for technique in TARGET_TECHNIQUES:

    mask = technique_mask(
        technique
    )

    indices = (
        LABELS_TRAIN.index[
            mask
        ]
        .tolist()
    )

    if not indices:

        print()
        print(
            f"[WARNING] No training rows "
            f"for {technique}"
        )

        continue

    chosen = indices[
        :SAMPLES_PER_TECHNIQUE
    ]

    for index in chosen:

        selected_rows.append({

            "source_index":
                int(index),

            "target_technique":
                technique,

        })


print()
print(
    f"Real training replay cases: "
    f"{len(selected_rows)}"
)


# ============================================================
# BUILD INPUT MATRICES
# ============================================================

structured_rows = []

behavior_texts = []

metadata_rows = []


for item in selected_rows:

    index = item[
        "source_index"
    ]

    technique = item[
        "target_technique"
    ]

    structured_row = (
        STRUCTURED_TRAIN
        .iloc[index]
        .copy()
    )

    # Exact production structured schema.
    structured_row = (
        structured_row[
            STRUCTURED_COLUMNS
        ]
    )

    structured_rows.append(
        structured_row
    )

    behavior_value = (
        BEHAVIOR_TRAIN
        .iloc[index][
            "behavior_text_normalized"
        ]
    )

    if pd.isna(
        behavior_value
    ):

        behavior_value = ""

    behavior_texts.append(
        str(behavior_value)
    )

    metadata_rows.append({

        "source_index":
            index,

        "target_technique":
            technique,

        "original_labels":
            str(
                LABELS_TRAIN
                .iloc[index][
                    "techniques"
                ]
            ),

        "behavior_text":
            str(
                behavior_value
            ),
    })


structured = pd.DataFrame(
    structured_rows
)

metadata = pd.DataFrame(
    metadata_rows
)


# ============================================================
# TRANSFORM — SAME PRODUCTION PIPELINE
# ============================================================

print()
print("Transforming real training events...")
print("-" * 78)


# ------------------------------------------------------------
# Categorical
# ------------------------------------------------------------

categorical_input = (
    structured[
        CATEGORICAL_COLUMNS
    ]
    .fillna("")
    .astype(str)
)

X_cat = ENCODER.transform(
    categorical_input
)


# ------------------------------------------------------------
# Numerical
# ------------------------------------------------------------

numerical_input = (
    structured[
        NUMERICAL_COLUMNS
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

X_num = SCALER.transform(
    numerical_input
)


# ------------------------------------------------------------
# Behavior text
# ------------------------------------------------------------

X_text = TFIDF.transform(
    pd.Series(
        behavior_texts
    )
    .fillna("")
    .astype(str)
)


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

from scipy.sparse import hstack

X = hstack(
    [
        X_cat,
        X_num,
        X_text
    ],
    format="csr"
)


print(
    f"Structured matrix    : "
    f"{structured.shape}"
)

print(
    f"Categorical matrix   : "
    f"{X_cat.shape}"
)

print(
    f"Numerical matrix     : "
    f"{X_num.shape}"
)

print(
    f"TF-IDF matrix        : "
    f"{X_text.shape}"
)

print(
    f"Final matrix         : "
    f"{X.shape}"
)


# ============================================================
# FEATURE DIMENSION CHECK
# ============================================================

expected_features = (
    X_cat.shape[1]
    + X_num.shape[1]
    + X_text.shape[1]
)

print()
print(
    f"Expected final features: "
    f"{expected_features}"
)

print(
    f"Observed final features: "
    f"{X.shape[1]}"
)

if X.shape[1] != 6919:

    raise RuntimeError(
        "Production feature dimension "
        "is not 6919."
    )

if X.shape[1] != expected_features:

    raise RuntimeError(
        "Internal feature dimension mismatch."
    )

print(
    "[PASS] Production feature dimension = 6919"
)


# ============================================================
# PREDICTION
# ============================================================

probabilities = MODEL.predict_proba(
    X
)

probabilities = np.asarray(
    probabilities
)

print()
print(
    f"Probability matrix: "
    f"{probabilities.shape}"
)

if probabilities.shape != (
    len(selected_rows),
    len(TECHNIQUES)
):

    raise RuntimeError(
        "Probability matrix dimensions "
        "do not match expected dimensions."
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
        "Probability values outside [0,1]."
    )

print(
    "[PASS] Probability matrix valid"
)


# ============================================================
# RESULTS
# ============================================================

results = []


for row_number, item in enumerate(
    selected_rows
):

    target = item[
        "target_technique"
    ]

    source_index = item[
        "source_index"
    ]

    scores = {}

    for class_index, technique in enumerate(
        TECHNIQUES
    ):

        scores[
            technique
        ] = float(
            probabilities[
                row_number,
                class_index
            ]
        )

    target_probability = scores.get(
        target,
        0.0
    )

    threshold = float(
        THRESHOLDS.get(
            target,
            0.5
        )
    )

    top_technique = max(
        scores,
        key=scores.get
    )

    top_probability = scores[
        top_technique
    ]

    rank = (
        1
        +
        sum(
            value
            > target_probability
            for value in scores.values()
        )
    )

    triggered = [

        technique

        for technique, probability
        in scores.items()

        if probability
        >= float(
            THRESHOLDS.get(
                technique,
                0.5
            )
        )
    ]

    expected_detected = (
        target
        in triggered
    )

    original_labels = (
        str(
            LABELS_TRAIN
            .iloc[source_index][
                "techniques"
            ]
        )
    )

    behavior_text = (
        BEHAVIOR_TRAIN
        .iloc[source_index][
            "behavior_text_normalized"
        ]
    )

    if pd.isna(
        behavior_text
    ):

        behavior_text = ""

    results.append({

        "source_index":
            source_index,

        "target_technique":
            target,

        "original_labels":
            original_labels,

        "target_probability":
            target_probability,

        "threshold":
            threshold,

        "expected_detected":
            expected_detected,

        "top_technique":
            top_technique,

        "top_probability":
            top_probability,

        "rank":
            rank,

        "triggered_techniques":
            ",".join(
                triggered
            ),

        "behavior_text":
            str(
                behavior_text
            ),
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 78)
print(
    "REAL TRAINING-EVENT REPLAY RESULTS"
)
print("=" * 78)


for technique in TARGET_TECHNIQUES:

    subset = results_df[
        results_df[
            "target_technique"
        ]
        == technique
    ]

    if subset.empty:
        continue

    detected = int(
        subset[
            "expected_detected"
        ].sum()
    )

    total = len(
        subset
    )

    rate = (
        detected / total
        if total
        else 0.0
    )

    print()
    print(
        f"{technique}"
    )

    print(
        f"  Real training events : "
        f"{total}"
    )

    print(
        f"  Detected             : "
        f"{detected}"
    )

    print(
        f"  Detection rate       : "
        f"{rate:.4f}"
    )

    print(
        f"  Mean target P        : "
        f"{subset['target_probability'].mean():.4f}"
    )

    print(
        f"  Median target P      : "
        f"{subset['target_probability'].median():.4f}"
    )

    print(
        f"  Mean rank            : "
        f"{subset['rank'].mean():.2f}"
    )

    print()
    print(
        "  Individual samples:"
    )

    for _, row in subset.iterrows():

        status = (
            "PASS"
            if row[
                "expected_detected"
            ]
            else
            "MISS"
        )

        print(
            f"    row={int(row['source_index'])} "
            f"{status} "
            f"P={row['target_probability']:.4f} "
            f"rank={int(row['rank'])} "
            f"top={row['top_technique']} "
            f"topP={row['top_probability']:.4f}"
        )


# ============================================================
# OVERALL METRICS
# ============================================================

total_cases = len(
    results_df
)

total_detected = int(
    results_df[
        "expected_detected"
    ].sum()
)

overall_detection_rate = (
    total_detected
    / total_cases
    if total_cases
    else 0.0
)


# ============================================================
# TOP-1 ACCURACY
# ============================================================

top1_correct = int(
    (
        results_df[
            "top_technique"
        ]
        ==
        results_df[
            "target_technique"
        ]
    ).sum()
)

top1_accuracy = (
    top1_correct
    / total_cases
    if total_cases
    else 0.0
)


# ============================================================
# TOP-5 ACCURACY
# ============================================================

top5_hits = 0

for row_number in range(
    len(results_df)
):

    target = results_df.iloc[
        row_number
    ][
        "target_technique"
    ]

    scores = probabilities[
        row_number
    ]

    top_indices = np.argsort(
        scores
    )[::-1][:5]

    top5_techniques = {

        TECHNIQUES[index]

        for index in top_indices
    }

    if target in top5_techniques:

        top5_hits += 1


top5_accuracy = (
    top5_hits
    / total_cases
    if total_cases
    else 0.0
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print(
    "GATE 2.4D SUMMARY"
)
print("=" * 78)

print(
    f"Total real training events : "
    f"{total_cases}"
)

print(
    f"Expected techniques detected: "
    f"{total_detected}"
)

print(
    f"Detection rate             : "
    f"{overall_detection_rate:.4f}"
)

print(
    f"Top-1 accuracy             : "
    f"{top1_accuracy:.4f}"
)

print(
    f"Top-5 accuracy             : "
    f"{top5_accuracy:.4f}"
)


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

if overall_detection_rate >= 0.90:

    diagnostic_status = (
        "PRODUCTION_PIPELINE_RECOGNIZES_REAL_TRAINING_EVENTS"
    )

elif overall_detection_rate >= 0.70:

    diagnostic_status = (
        "PARTIAL_RECOGNITION_OF_REAL_TRAINING_EVENTS"
    )

else:

    diagnostic_status = (
        "PRODUCTION_PIPELINE_REQUIRES_DEEPER_REVIEW"
    )


print()
print(
    f"Diagnostic status: "
    f"{diagnostic_status}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    REPORT_DIR
    / "real_training_event_replay_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# TECHNIQUE SUMMARY
# ============================================================

technique_summary_rows = []


for technique in TARGET_TECHNIQUES:

    subset = results_df[
        results_df[
            "target_technique"
        ]
        == technique
    ]

    if subset.empty:
        continue

    technique_summary_rows.append({

        "technique":
            technique,

        "training_rows_available":
            int(
                technique_mask(
                    technique
                ).sum()
            ),

        "replayed_rows":
            len(subset),

        "detected":
            int(
                subset[
                    "expected_detected"
                ].sum()
            ),

        "detection_rate":
            float(
                subset[
                    "expected_detected"
                ].mean()
            ),

        "mean_probability":
            float(
                subset[
                    "target_probability"
                ].mean()
            ),

        "median_probability":
            float(
                subset[
                    "target_probability"
                ].median()
            ),

        "minimum_probability":
            float(
                subset[
                    "target_probability"
                ].min()
            ),

        "maximum_probability":
            float(
                subset[
                    "target_probability"
                ].max()
            ),

        "mean_rank":
            float(
                subset[
                    "rank"
                ].mean()
            ),
    })


technique_summary_df = pd.DataFrame(
    technique_summary_rows
)


summary_path = (
    REPORT_DIR
    / "real_training_event_replay_technique_summary.csv"
)

technique_summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.4D",

    "gate_name":
        "Real Training-Event Replay Validation",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "purpose":
        "Validate production inference using actual training rows rather than synthetic feature averages.",

    "production_artifacts_modified":
        False,

    "model_retrained":
        False,

    "thresholds_modified":
        False,

    "datasets_modified":
        False,

    "commands_executed":
        False,

    "dimensions": {

        "structured":
            int(
                structured.shape[1]
            ),

        "categorical_input":
            len(
                CATEGORICAL_COLUMNS
            ),

        "categorical_encoded":
            int(
                X_cat.shape[1]
            ),

        "numerical":
            int(
                X_num.shape[1]
            ),

        "tfidf":
            int(
                X_text.shape[1]
            ),

        "final":
            int(
                X.shape[1]
            ),

        "mitre_classes":
            len(
                TECHNIQUES
            ),
    },

    "replayed_events":
        total_cases,

    "overall_detection_rate":
        overall_detection_rate,

    "top1_accuracy":
        top1_accuracy,

    "top5_accuracy":
        top5_accuracy,

    "diagnostic_status":
        diagnostic_status,

    "target_techniques":
        TARGET_TECHNIQUES,

    "results_file":
        str(
            results_path
        ),

    "summary_file":
        str(
            summary_path
        ),
}


report_path = (
    REPORT_DIR
    / "phase2_gate2_4d_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print(
    "GATE 2.4D COMPLETE"
)
print("=" * 78)

print(
    f"Results : {results_path}"
)

print(
    f"Summary : {summary_path}"
)

print(
    f"Report  : {report_path}"
)

print()
print(
    "No model, threshold, dataset, "
    "or production artifact was modified."
)

print("=" * 78)