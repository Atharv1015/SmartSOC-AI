import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# SMARTSOC PHASE 2 GATE 2.4C
# TRAINING-ALIGNED MALICIOUS-EVENT VALIDATION
#
# DIAGNOSTIC ONLY
#
# This script:
#   - does NOT retrain the model
#   - does NOT modify model artifacts
#   - does NOT modify thresholds
#   - does NOT modify datasets
#   - does NOT execute malicious commands
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
    / "phase2_gate2_4c"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print()
print("=" * 78)
print("SMARTSOC PHASE 2 GATE 2.4C")
print("TRAINING-ALIGNED MALICIOUS-EVENT VALIDATION")
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

THRESHOLDS = THRESHOLD_DATA["thresholds"]


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
# LABEL PARSER
# ============================================================

def parse_labels(value):

    if pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    }


# ============================================================
# BEHAVIOR FEATURES
# ============================================================

BEHAVIOR_FEATURES = [
    "uses_powershell",
    "uses_cmd",
    "uses_net",
    "uses_whoami",
    "uses_schtasks",
    "uses_reg",
    "uses_rundll32",
    "uses_wmic",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "uses_sc",
    "uses_mshta",
    "uses_regsvr32",
    "uses_cscript",
    "uses_wscript",
    "uses_wevtutil",
    "uses_vssadmin",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "has_target_filename",
    "has_image_loaded",
    "has_target_object",
    "administrative_powershell_context",
    "strong_behavior_signal",
]


# ============================================================
# TARGET TECHNIQUES
#
# We intentionally exclude extremely rare techniques such as
# T1562.001 with only one training example.
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


# ============================================================
# TRAINING ROWS
# ============================================================

def get_training_rows(
    technique
):

    mask = LABELS_TRAIN[
        "techniques"
    ].apply(
        lambda value:
        technique in parse_labels(value)
    )

    return STRUCTURED_TRAIN.loc[
        mask
    ].copy()


# ============================================================
# TRAINING PROFILE
# ============================================================

def calculate_training_profile(
    technique
):

    rows = get_training_rows(
        technique
    )

    if rows.empty:
        return None

    profile = {}

    for feature in BEHAVIOR_FEATURES:

        if feature not in rows.columns:

            profile[feature] = 0.0
            continue

        values = pd.to_numeric(
            rows[feature],
            errors="coerce"
        ).fillna(0)

        profile[feature] = float(
            values.mean()
        )

    return {
        "technique": technique,
        "rows": int(len(rows)),
        "profile": profile,
    }


# ============================================================
# BUILD ALIGNED BINARY PROFILE
#
# For binary behavior features:
#
# mean >= 0.50 -> 1
# mean <  0.50 -> 0
# ============================================================

def build_aligned_profile(
    training_profile
):

    aligned = {}

    for feature in BEHAVIOR_FEATURES:

        mean_value = training_profile[
            "profile"
        ].get(
            feature,
            0.0
        )

        aligned[feature] = (
            1
            if mean_value >= 0.50
            else 0
        )

    return aligned


# ============================================================
# TRAINING CATEGORICAL DEFAULTS
# ============================================================

def get_training_categorical_defaults():

    defaults = {}

    for column in CATEGORICAL_COLUMNS:

        if column not in STRUCTURED_TRAIN.columns:

            defaults[column] = ""
            continue

        values = (
            STRUCTURED_TRAIN[column]
            .fillna("")
            .astype(str)
        )

        mode = values.mode()

        if len(mode) > 0:

            defaults[column] = str(
                mode.iloc[0]
            )

        else:

            defaults[column] = ""

    return defaults


# ============================================================
# TRAINING NUMERICAL DEFAULTS
# ============================================================

def get_training_numerical_defaults():

    defaults = {}

    for column in NUMERICAL_COLUMNS:

        if column not in STRUCTURED_TRAIN.columns:

            defaults[column] = 0.0
            continue

        values = pd.to_numeric(
            STRUCTURED_TRAIN[column],
            errors="coerce"
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if values.empty:

            defaults[column] = 0.0

        else:

            median = float(
                values.median()
            )

            minimum = float(
                values.min()
            )

            maximum = float(
                values.max()
            )

            defaults[column] = float(
                np.clip(
                    median,
                    minimum,
                    maximum
                )
            )

    return defaults


CATEGORICAL_DEFAULTS = (
    get_training_categorical_defaults()
)

NUMERICAL_DEFAULTS = (
    get_training_numerical_defaults()
)


# ============================================================
# CREATE CONTROLLED EVENTS
# ============================================================

def create_controlled_events():

    events = []

    descriptions = {

        "T1027.004":
            "Training-aligned encoded-command profile",

        "T1059.001":
            "Training-aligned PowerShell profile",

        "T1059.003":
            "Training-aligned command-shell profile",

        "T1105":
            "Training-aligned payload-transfer profile",

        "T1112":
            "Training-aligned registry-modification profile",

        "T1543.003":
            "Training-aligned Windows-service profile",

        "T1070.004":
            "Training-aligned file-deletion profile",
    }

    for technique in TARGET_TECHNIQUES:

        training_profile = (
            calculate_training_profile(
                technique
            )
        )

        if training_profile is None:
            continue

        aligned = build_aligned_profile(
            training_profile
        )

        event = {}

        # ----------------------------------------------------
        # Categorical values
        # ----------------------------------------------------

        for column in CATEGORICAL_COLUMNS:

            event[column] = (
                CATEGORICAL_DEFAULTS[
                    column
                ]
            )

        # ----------------------------------------------------
        # Numerical values
        # ----------------------------------------------------

        for column in NUMERICAL_COLUMNS:

            event[column] = (
                NUMERICAL_DEFAULTS[
                    column
                ]
            )

        # ----------------------------------------------------
        # Behavioral values
        # ----------------------------------------------------

        for feature in BEHAVIOR_FEATURES:

            if feature in event:

                event[feature] = aligned[
                    feature
                ]

        # ----------------------------------------------------
        # Event ID
        # ----------------------------------------------------

        if "event_id" in event:

            event["event_id"] = 999999

        # ----------------------------------------------------
        # Keep system_level inside observed training domain
        # ----------------------------------------------------

        if "system_level" in event:

            observed = pd.to_numeric(
                STRUCTURED_TRAIN[
                    "system_level"
                ],
                errors="coerce"
            ).dropna()

            if not observed.empty:

                event[
                    "system_level"
                ] = float(
                    observed.median()
                )

        # ----------------------------------------------------
        # Command length
        # ----------------------------------------------------

        if "command_length" in event:

            observed = pd.to_numeric(
                STRUCTURED_TRAIN[
                    "command_length"
                ],
                errors="coerce"
            ).dropna()

            if not observed.empty:

                event[
                    "command_length"
                ] = float(
                    observed.median()
                )

        # ----------------------------------------------------
        # Parent command length
        # ----------------------------------------------------

        if "parent_command_length" in event:

            observed = pd.to_numeric(
                STRUCTURED_TRAIN[
                    "parent_command_length"
                ],
                errors="coerce"
            ).dropna()

            if not observed.empty:

                event[
                    "parent_command_length"
                ] = float(
                    observed.median()
                )

        # ----------------------------------------------------
        # Neutral text.
        #
        # This test intentionally isolates the structured
        # training-aligned feature representation.
        # ----------------------------------------------------

        behavior_text = ""

        events.append({

            "case_id":
                f"ALIGNED_{technique}",

            "technique":
                technique,

            "description":
                descriptions.get(
                    technique,
                    "Training-aligned profile"
                ),

            "training_rows":
                training_profile["rows"],

            "structured":
                event,

            "behavior_text":
                behavior_text,
        })

    return events


# ============================================================
# TRANSFORM EVENTS
# ============================================================

def transform_events(
    events
):

    structured_rows = []

    behavior_texts = []

    for item in events:

        structured_rows.append(
            item["structured"]
        )

        behavior_texts.append(
            item["behavior_text"]
        )

    structured = pd.DataFrame(
        structured_rows
    )

    # Ensure schema columns exist.
    for column in STRUCTURED_COLUMNS:

        if column not in structured.columns:

            if column in CATEGORICAL_COLUMNS:

                structured[column] = ""

            else:

                structured[column] = 0.0

    # Exact schema order.
    structured = structured[
        STRUCTURED_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    X_text = TFIDF.transform(
        pd.Series(
            behavior_texts
        )
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    from scipy.sparse import hstack

    X = hstack(
        [
            X_cat,
            X_num,
            X_text
        ],
        format="csr"
    )

    return (
        structured,
        X,
        X_cat,
        X_num,
        X_text,
    )


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return 0.0


# ============================================================
# PREDICT
# ============================================================

def predict_events(
    X
):

    probabilities = MODEL.predict_proba(
        X
    )

    if isinstance(
        probabilities,
        list
    ):

        probability_matrix = np.column_stack(
            [
                np.asarray(
                    item
                ).reshape(-1)
                for item in probabilities
            ]
        )

    else:

        probability_matrix = np.asarray(
            probabilities
        )

    return probability_matrix


# ============================================================
# PROFILE REPORT
# ============================================================

def build_profile_table():

    rows = []

    for technique in TARGET_TECHNIQUES:

        profile = (
            calculate_training_profile(
                technique
            )
        )

        if profile is None:
            continue

        aligned = build_aligned_profile(
            profile
        )

        for feature in BEHAVIOR_FEATURES:

            rows.append({

                "technique":
                    technique,

                "training_rows":
                    profile["rows"],

                "feature":
                    feature,

                "training_mean":
                    profile[
                        "profile"
                    ][feature],

                "aligned_value":
                    aligned[
                        feature
                    ],
            })

    return pd.DataFrame(rows)


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    events = create_controlled_events()

    print()
    print(
        f"Training-aligned cases: "
        f"{len(events)}"
    )

    (
        structured,
        X,
        X_cat,
        X_num,
        X_text,
    ) = transform_events(
        events
    )

    print(
        f"Structured matrix: "
        f"{structured.shape}"
    )

    print(
        f"Categorical encoded matrix: "
        f"{X_cat.shape}"
    )

    print(
        f"Numerical matrix: "
        f"{X_num.shape}"
    )

    print(
        f"TF-IDF matrix: "
        f"{X_text.shape}"
    )

    print(
        f"Final matrix: "
        f"{X.shape}"
    )

    # ========================================================
    # IMPORTANT FIX
    #
    # The raw categorical input has 4 columns, but after
    # OneHotEncoder it becomes 162 model features.
    #
    # Therefore final expected dimension is:
    #
    #   162 + 58 + 6699 = 6919
    # ========================================================

    categorical_encoded_features = (
        X_cat.shape[1]
    )

    numerical_features = (
        X_num.shape[1]
    )

    tfidf_features = (
        X_text.shape[1]
    )

    expected_features = (
        categorical_encoded_features
        + numerical_features
        + tfidf_features
    )

    print()
    print(
        "Feature dimension validation"
    )
    print("-" * 78)

    print(
        f"Raw categorical features     : "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    print(
        f"Encoded categorical features : "
        f"{categorical_encoded_features}"
    )

    print(
        f"Numerical features            : "
        f"{numerical_features}"
    )

    print(
        f"TF-IDF features               : "
        f"{tfidf_features}"
    )

    print(
        f"Expected final features      : "
        f"{expected_features}"
    )

    print(
        f"Observed final features      : "
        f"{X.shape[1]}"
    )

    if X.shape[1] != expected_features:

        raise RuntimeError(
            "Final feature dimension mismatch."
        )

    if X.shape[1] != 6919:

        raise RuntimeError(
            "Production model expects 6919 "
            "features but transformed input "
            "does not contain 6919 features."
        )

    print(
        "[PASS] Final feature dimension = 6919"
    )

    # ========================================================
    # Probability validation
    # ========================================================

    probability_matrix = predict_events(
        X
    )

    print()
    print(
        f"Probability matrix: "
        f"{probability_matrix.shape}"
    )

    if not np.isfinite(
        probability_matrix
    ).all():

        raise RuntimeError(
            "Probability matrix contains "
            "non-finite values."
        )

    if (
        probability_matrix.min() < 0
        or probability_matrix.max() > 1
    ):

        raise RuntimeError(
            "Probability matrix contains "
            "values outside [0,1]."
        )

    print(
        "[PASS] Probabilities finite "
        "and within [0,1]"
    )

    # ========================================================
    # Build results
    # ========================================================

    result_rows = []

    for index, event in enumerate(
        events
    ):

        expected = event[
            "technique"
        ]

        probabilities = (
            probability_matrix[index]
        )

        scores = {}

        for class_index, technique in enumerate(
            TECHNIQUES
        ):

            if class_index < len(
                probabilities
            ):

                scores[
                    technique
                ] = safe_float(
                    probabilities[
                        class_index
                    ]
                )

        top_technique = max(
            scores,
            key=scores.get
        )

        top_probability = scores[
            top_technique
        ]

        expected_probability = scores.get(
            expected,
            0.0
        )

        expected_threshold = float(
            THRESHOLDS.get(
                expected,
                0.5
            )
        )

        triggered = []

        for technique, probability in (
            scores.items()
        ):

            threshold = float(
                THRESHOLDS.get(
                    technique,
                    0.5
                )
            )

            if probability >= threshold:

                triggered.append(
                    technique
                )

        expected_detected = (
            expected
            in triggered
        )

        rank = (
            1
            +
            sum(
                probability
                >
                expected_probability
                for probability
                in scores.values()
            )
        )

        result_rows.append({

            "case_id":
                event["case_id"],

            "description":
                event["description"],

            "expected_technique":
                expected,

            "training_rows":
                event["training_rows"],

            "expected_probability":
                expected_probability,

            "expected_threshold":
                expected_threshold,

            "expected_detected":
                expected_detected,

            "top_technique":
                top_technique,

            "top_probability":
                top_probability,

            "rank_of_expected":
                rank,

            "triggered_techniques":
                ",".join(
                    triggered
                ),
        })

    results_df = pd.DataFrame(
        result_rows
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "TRAINING-ALIGNED MALICIOUS RESULTS"
    )
    print("=" * 78)

    for _, row in results_df.iterrows():

        status = (
            "PASS"
            if row[
                "expected_detected"
            ]
            else
            "MISS"
        )

        print()
        print(
            f"{row['case_id']} | "
            f"{status}"
        )

        print(
            f"  Description : "
            f"{row['description']}"
        )

        print(
            f"  Expected    : "
            f"{row['expected_technique']}"
        )

        print(
            f"  Train rows  : "
            f"{int(row['training_rows'])}"
        )

        print(
            f"  Expected P  : "
            f"{row['expected_probability']:.4f}"
        )

        print(
            f"  Threshold   : "
            f"{row['expected_threshold']:.4f}"
        )

        print(
            f"  Rank        : "
            f"{int(row['rank_of_expected'])}"
        )

        print(
            f"  Top         : "
            f"{row['top_technique']}"
        )

        print(
            f"  Top P       : "
            f"{row['top_probability']:.4f}"
        )

        print(
            f"  Triggered   : "
            f"{row['triggered_techniques']}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(
        results_df
    )

    passed = int(
        results_df[
            "expected_detected"
        ].sum()
    )

    missed = (
        total - passed
    )

    detection_rate = (
        passed / total
        if total > 0
        else 0.0
    )

    print()
    print("=" * 78)
    print(
        "GATE 2.4C SUMMARY"
    )
    print("=" * 78)

    print(
        f"Total aligned cases : "
        f"{total}"
    )

    print(
        f"Passed              : "
        f"{passed}"
    )

    print(
        f"Missed              : "
        f"{missed}"
    )

    print(
        f"Detection rate      : "
        f"{detection_rate:.4f}"
    )

    # ========================================================
    # Diagnostic interpretation
    # ========================================================

    if detection_rate >= 0.80:

        diagnostic_status = (
            "MODEL_CAN_RECOGNIZE_TRAINING_ALIGNED_PATTERNS"
        )

    elif detection_rate >= 0.50:

        diagnostic_status = (
            "PARTIAL_TRAINING_ALIGNED_RECOGNITION"
        )

    else:

        diagnostic_status = (
            "MODEL_REQUIRES_DEEPER_REVIEW"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_path = (
        REPORT_DIR
        / "training_aligned_malicious_results.csv"
    )

    results_df.to_csv(
        result_path,
        index=False
    )

    profile_df = (
        build_profile_table()
    )

    profile_path = (
        REPORT_DIR
        / "training_aligned_feature_profiles.csv"
    )

    profile_df.to_csv(
        profile_path,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "phase":
            "Phase 2",

        "gate":
            "Gate 2.4C",

        "gate_name":
            "Training-Aligned Malicious-Event Validation",

        "status":
            "DIAGNOSTIC_COMPLETE",

        "diagnostic_interpretation":
            diagnostic_status,

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

        "feature_dimensions": {

            "raw_categorical":
                len(CATEGORICAL_COLUMNS),

            "encoded_categorical":
                categorical_encoded_features,

            "numerical":
                numerical_features,

            "tfidf":
                tfidf_features,

            "final":
                X.shape[1],

            "expected":
                expected_features,
        },

        "total_cases":
            total,

        "passed":
            passed,

        "missed":
            missed,

        "detection_rate":
            detection_rate,

        "target_techniques":
            TARGET_TECHNIQUES,

        "result_file":
            str(result_path),

        "profile_file":
            str(profile_path),
    }

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4c_report.json"
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

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 78)
    print(
        "GATE 2.4C COMPLETE"
    )
    print("=" * 78)

    print(
        f"Results : {result_path}"
    )

    print(
        f"Profiles: {profile_path}"
    )

    print(
        f"Report  : {report_path}"
    )

    print()
    print(
        f"DIAGNOSTIC STATUS: "
        f"{diagnostic_status}"
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    evaluate()