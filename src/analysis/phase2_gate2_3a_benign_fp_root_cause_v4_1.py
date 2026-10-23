import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.3A
# BENIGN FALSE-POSITIVE ROOT-CAUSE ANALYSIS
#
# PURPOSE
# -------
# Investigate why controlled benign telemetry generated threshold-crossing
# MITRE ATT&CK predictions during Phase 2 Gate 2.3.
#
# IMPORTANT
# ---------
# This script is READ-ONLY with respect to:
#   - production model
#   - thresholds
#   - training dataset
#   - test dataset
#
# It does NOT:
#   - retrain the model
#   - modify thresholds
#   - modify datasets
#   - overwrite production artifacts
#   - execute Windows commands
#
# It creates analysis reports only.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

GATE23_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3a"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase2_gate2_3a_root_cause_report.json"
)

CASE_RESULTS_FILE = (
    REPORT_DIR
    / "benign_fp_case_analysis.csv"
)

NEIGHBOR_RESULTS_FILE = (
    REPORT_DIR
    / "benign_fp_training_neighbors.csv"
)

LABEL_ANALYSIS_FILE = (
    REPORT_DIR
    / "benign_fp_label_analysis.csv"
)

FEATURE_ANALYSIS_FILE = (
    REPORT_DIR
    / "benign_fp_feature_analysis.csv"
)


# =============================================================================
# ARTIFACTS
# =============================================================================

SCHEMA_FILE = (
    MODEL_DIR
    / "feature_schema.json"
)

ENCODER_FILE = (
    MODEL_DIR
    / "structured_encoder.joblib"
)

SCALER_FILE = (
    MODEL_DIR
    / "numerical_scaler.joblib"
)

TFIDF_FILE = (
    MODEL_DIR
    / "tfidf_vectorizer.joblib"
)

MODEL_FILE = (
    MODEL_DIR
    / "model.joblib"
)

MULTILABEL_ENCODER_FILE = (
    MODEL_DIR
    / "multilabel_encoder.joblib"
)

THRESHOLD_FILE = (
    MODEL_DIR
    / "thresholds.json"
)

BENIGN_RESULTS_FILE = (
    GATE23_DIR
    / "controlled_benign_results.csv"
)


# =============================================================================
# EXPECTED DIMENSIONS
# =============================================================================

EXPECTED_STRUCTURED = 62
EXPECTED_CATEGORICAL = 4
EXPECTED_NUMERICAL = 58
EXPECTED_CATEGORICAL_ENCODED = 162
EXPECTED_TFIDF = 6699
EXPECTED_FINAL = 6919
EXPECTED_MITRE = 45


# =============================================================================
# WARNINGS
# =============================================================================

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn",
)


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    print()
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def require_file(path):
    if not path.exists():
        fail(
            f"Required file does not exist: {path}"
        )

    print(
        f"[OK] {path}"
    )


def to_dense_array(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy()

    if isinstance(matrix, pd.Series):
        matrix = matrix.to_numpy()

    return np.asarray(
        matrix,
        dtype=np.float64
    )


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)

    text = (
        text
        .lower()
        .strip()
    )

    return " ".join(
        text.split()
    )


def parse_label_string(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3A")
print("BENIGN FALSE-POSITIVE ROOT-CAUSE ANALYSIS")
print("=" * 80)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 1. CHECK INPUTS
# =============================================================================

print()
print("-" * 80)
print("Checking required files")
print("-" * 80)


required_files = [
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    TFIDF_FILE,
    MODEL_FILE,
    MULTILABEL_ENCODER_FILE,
    THRESHOLD_FILE,
    BENIGN_RESULTS_FILE,

    DATA_DIR / "structured_train.csv",
    DATA_DIR / "labels_train.csv",
    DATA_DIR / "behavior_text_train.csv",
]


for path in required_files:
    require_file(path)


# =============================================================================
# 2. LOAD PRODUCTION ARTIFACTS
# =============================================================================

print()
print("-" * 80)
print("Loading production artifacts")
print("-" * 80)


with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as f:
    schema = json.load(f)


categorical_columns = [
    str(x)
    for x in schema[
        "categorical_columns"
    ]
]

numerical_columns = [
    str(x)
    for x in schema[
        "numerical_columns"
    ]
]

structured_columns = (
    categorical_columns
    + numerical_columns
)


schema_mitre = [
    str(x)
    for x in schema[
        "mitre_techniques"
    ]
]


encoder = joblib.load(
    ENCODER_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

tfidf = joblib.load(
    TFIDF_FILE
)

model = joblib.load(
    MODEL_FILE
)

multilabel_encoder = joblib.load(
    MULTILABEL_ENCODER_FILE
)


mitre_classes = [
    str(x)
    for x in multilabel_encoder.classes_
]


with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_data = json.load(f)


if (
    isinstance(threshold_data, dict)
    and "thresholds" in threshold_data
):
    thresholds = threshold_data[
        "thresholds"
    ]
else:
    thresholds = threshold_data


thresholds = {
    str(k): float(v)
    for k, v in thresholds.items()
}


# =============================================================================
# 3. VALIDATE ARTIFACTS
# =============================================================================

print()
print("-" * 80)
print("Validating artifact dimensions")
print("-" * 80)


if len(categorical_columns) != EXPECTED_CATEGORICAL:
    fail(
        "Unexpected categorical column count."
    )


if len(numerical_columns) != EXPECTED_NUMERICAL:
    fail(
        "Unexpected numerical column count."
    )


if len(structured_columns) != EXPECTED_STRUCTURED:
    fail(
        "Unexpected structured feature count."
    )


if len(mitre_classes) != EXPECTED_MITRE:
    fail(
        "Unexpected MITRE class count."
    )


encoder_dimension = sum(
    len(category)
    for category in encoder.categories_
)


if encoder_dimension != EXPECTED_CATEGORICAL_ENCODED:
    fail(
        "Unexpected categorical encoded dimension."
    )


if scaler.n_features_in_ != EXPECTED_NUMERICAL:
    fail(
        "Unexpected numerical scaler dimension."
    )


if len(tfidf.vocabulary_) != EXPECTED_TFIDF:
    fail(
        "Unexpected TF-IDF vocabulary dimension."
    )


if set(mitre_classes) != set(schema_mitre):
    fail(
        "MITRE classes differ between schema "
        "and multilabel encoder."
    )


for technique in mitre_classes:

    if technique not in thresholds:
        fail(
            f"Missing threshold for {technique}"
        )


print(
    "[PASS] Production artifact dimensions."
)


# =============================================================================
# 4. LOAD TRAINING DATA
# =============================================================================

print()
print("-" * 80)
print("Loading training data")
print("-" * 80)


structured_train = pd.read_csv(
    DATA_DIR / "structured_train.csv"
)

labels_train = pd.read_csv(
    DATA_DIR / "labels_train.csv"
)

behavior_train = pd.read_csv(
    DATA_DIR / "behavior_text_train.csv"
)


print(
    f"Structured train : "
    f"{structured_train.shape}"
)

print(
    f"Labels train     : "
    f"{labels_train.shape}"
)

print(
    f"Behavior train   : "
    f"{behavior_train.shape}"
)


if len(structured_train) != len(labels_train):
    fail(
        "Training structured and label row "
        "counts do not match."
    )


if len(structured_train) != len(behavior_train):
    fail(
        "Training structured and behavior row "
        "counts do not match."
    )


# =============================================================================
# 5. VALIDATE TRAINING SCHEMA
# =============================================================================

print()
print("-" * 80)
print("Validating training schema")
print("-" * 80)


missing_structured = [
    col
    for col in structured_columns
    if col not in structured_train.columns
]


if missing_structured:
    fail(
        "Training data is missing structured "
        f"columns: {missing_structured}"
    )


if (
    "behavior_text_normalized"
    not in behavior_train.columns
):
    fail(
        "behavior_text_normalized is missing "
        "from behavior_train.csv."
    )


if (
    "techniques"
    not in labels_train.columns
):
    fail(
        "techniques column is missing "
        "from labels_train.csv."
    )


print(
    "[PASS] Training schema valid."
)


# =============================================================================
# 6. LOAD GATE 2.3 RESULTS
# =============================================================================

print()
print("-" * 80)
print("Loading Gate 2.3 benign results")
print("-" * 80)


benign_results = pd.read_csv(
    BENIGN_RESULTS_FILE
)


if benign_results.empty:
    fail(
        "Gate 2.3 result file is empty."
    )


required_result_columns = [
    "case_id",
    "description",
    "expected",
    "top_technique",
    "top_probability",
    "top_threshold",
    "triggered_technique_count",
    "triggered_techniques",
    "false_positive",
    "status",
]


missing_result_columns = [
    col
    for col in required_result_columns
    if col not in benign_results.columns
]


if missing_result_columns:
    fail(
        "Gate 2.3 results missing columns: "
        f"{missing_result_columns}"
    )


fp_results = benign_results[
    benign_results[
        "false_positive"
    ].astype(bool)
].copy()


print(
    f"Gate 2.3 cases      : "
    f"{len(benign_results)}"
)

print(
    f"False-positive cases: "
    f"{len(fp_results)}"
)


if fp_results.empty:
    fail(
        "No false-positive cases found. "
        "Root-cause analysis is unnecessary."
    )


# =============================================================================
# 7. RECONSTRUCT BENIGN CASES
# =============================================================================
#
# These definitions intentionally match Gate 2.3.
# =============================================================================

print()
print("-" * 80)
print("Reconstructing controlled benign cases")
print("-" * 80)


BENIGN_CASES = [

    {
        "case_id":
            "BENIGN_001",

        "process_name":
            "powershell.exe",

        "parent_process_name":
            "explorer.exe",

        "command_line":
            "powershell.exe Get-Host",

        "behavior_text":
            "powershell.exe get host",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_002",

        "process_name":
            "cmd.exe",

        "parent_process_name":
            "explorer.exe",

        "command_line":
            "cmd.exe /c dir",

        "behavior_text":
            "cmd.exe dir",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_003",

        "process_name":
            "ipconfig.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "ipconfig.exe /all",

        "behavior_text":
            "ipconfig.exe all",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_004",

        "process_name":
            "whoami.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "whoami.exe",

        "behavior_text":
            "whoami.exe",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_005",

        "process_name":
            "tasklist.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "tasklist.exe",

        "behavior_text":
            "tasklist.exe",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_006",

        "process_name":
            "netstat.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "netstat.exe -ano",

        "behavior_text":
            "netstat.exe ano",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_007",

        "process_name":
            "net.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "net.exe user",

        "behavior_text":
            "net.exe user",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_008",

        "process_name":
            "sc.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "sc.exe query",

        "behavior_text":
            "sc.exe query",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_009",

        "process_name":
            "schtasks.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "schtasks.exe /query",

        "behavior_text":
            "schtasks.exe query",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_010",

        "process_name":
            "reg.exe",

        "parent_process_name":
            "cmd.exe",

        "command_line":
            "reg.exe query HKCU\\Software",

        "behavior_text":
            "reg.exe query hkcu software",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_011",

        "process_name":
            "wscript.exe",

        "parent_process_name":
            "explorer.exe",

        "command_line":
            "wscript.exe C:\\Users\\Public\\maintenance.vbs",

        "behavior_text":
            "wscript.exe maintenance.vbs",

        "administrative":
            False,
    },

    {
        "case_id":
            "BENIGN_012",

        "process_name":
            "powershell.exe",

        "parent_process_name":
            "explorer.exe",

        "command_line":
            "powershell.exe Get-Service",

        "behavior_text":
            "powershell.exe get service",

        "administrative":
            True,
    },
]


case_lookup = {
    case["case_id"]: case
    for case in BENIGN_CASES
}


# =============================================================================
# 8. BUILD CONTROLLED STRUCTURED REPRESENTATIONS
# =============================================================================

def build_case(case):

    row = {}

    # -------------------------------------------------------------------------
    # Default numerical values
    # -------------------------------------------------------------------------

    for column in numerical_columns:
        row[column] = 0.0


    process_name = (
        case["process_name"]
        .lower()
    )

    parent_name = (
        case["parent_process_name"]
        .lower()
    )

    command_line = (
        case["command_line"]
    )

    command_lower = (
        command_line.lower()
    )


    # -------------------------------------------------------------------------
    # Categorical features
    # -------------------------------------------------------------------------

    process_extension = ""

    parent_extension = ""

    if "." in process_name:
        process_extension = (
            process_name
            .rsplit(".", 1)[1]
        )

    if "." in parent_name:
        parent_extension = (
            parent_name
            .rsplit(".", 1)[1]
        )


    row[
        "process_name"
    ] = process_name

    row[
        "parent_process_name"
    ] = parent_name

    row[
        "process_extension"
    ] = process_extension

    row[
        "parent_extension"
    ] = parent_extension


    # -------------------------------------------------------------------------
    # Event metadata
    # -------------------------------------------------------------------------

    row[
        "event_id"
    ] = 1.0

    row[
        "system_level"
    ] = 0.0

    row[
        "is_sysmon"
    ] = 1.0

    row[
        "is_security_channel"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Process metadata
    # -------------------------------------------------------------------------

    row[
        "has_process_image"
    ] = 1.0

    row[
        "has_parent_image"
    ] = 1.0

    row[
        "has_user"
    ] = 1.0

    row[
        "has_parent_user"
    ] = 1.0

    row[
        "is_system_user"
    ] = 0.0

    row[
        "is_admin_user"
    ] = (
        1.0
        if case["administrative"]
        else 0.0
    )

    row[
        "integrity_high"
    ] = (
        1.0
        if case["administrative"]
        else 0.0
    )

    row[
        "integrity_medium"
    ] = (
        0.0
        if case["administrative"]
        else 1.0
    )


    # -------------------------------------------------------------------------
    # Command metadata
    # -------------------------------------------------------------------------

    row[
        "command_length"
    ] = float(
        len(command_line)
    )

    row[
        "parent_command_length"
    ] = float(
        len(parent_name)
    )

    row[
        "has_command_line"
    ] = 1.0

    row[
        "has_parent_command"
    ] = 1.0


    # -------------------------------------------------------------------------
    # Tool indicators
    # -------------------------------------------------------------------------

    row[
        "uses_powershell"
    ] = float(
        "powershell"
        in command_lower
    )

    row[
        "uses_cmd"
    ] = float(
        "cmd.exe"
        in command_lower
    )

    row[
        "uses_net"
    ] = float(
        "net.exe"
        in command_lower
    )

    row[
        "uses_whoami"
    ] = float(
        "whoami"
        in command_lower
    )

    row[
        "uses_schtasks"
    ] = float(
        "schtasks"
        in command_lower
    )

    row[
        "uses_reg"
    ] = float(
        "reg.exe"
        in command_lower
    )

    row[
        "uses_rundll32"
    ] = 0.0

    row[
        "uses_wmic"
    ] = 0.0

    row[
        "uses_certutil"
    ] = 0.0

    row[
        "uses_bitsadmin"
    ] = 0.0

    row[
        "uses_ipconfig"
    ] = float(
        "ipconfig"
        in command_lower
    )

    row[
        "uses_netstat"
    ] = float(
        "netstat"
        in command_lower
    )

    row[
        "uses_tasklist"
    ] = float(
        "tasklist"
        in command_lower
    )

    row[
        "uses_sc"
    ] = float(
        "sc.exe"
        in command_lower
    )

    row[
        "uses_mshta"
    ] = 0.0

    row[
        "uses_regsvr32"
    ] = 0.0

    row[
        "uses_cscript"
    ] = 0.0

    row[
        "uses_wscript"
    ] = float(
        "wscript"
        in command_lower
    )

    row[
        "uses_wevtutil"
    ] = 0.0

    row[
        "uses_vssadmin"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Suspicious behavior indicators
    # -------------------------------------------------------------------------

    row[
        "has_encoded_command"
    ] = 0.0

    row[
        "has_download_behavior"
    ] = 0.0

    row[
        "has_remote_execution_hint"
    ] = 0.0

    row[
        "has_registry_operation"
    ] = 0.0

    row[
        "has_file_deletion"
    ] = 0.0

    row[
        "has_obfuscation_hint"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Process relationship
    # -------------------------------------------------------------------------

    row[
        "same_process_parent"
    ] = float(
        process_name
        == parent_name
    )

    row[
        "suspicious_parent_wazuh"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Hash/network/target indicators
    # -------------------------------------------------------------------------

    row[
        "has_hash"
    ] = 0.0

    row[
        "has_sha256"
    ] = 0.0

    row[
        "has_md5"
    ] = 0.0

    row[
        "has_source_ip"
    ] = 0.0

    row[
        "has_destination_ip"
    ] = 0.0

    row[
        "has_source_port"
    ] = 0.0

    row[
        "has_destination_port"
    ] = 0.0

    row[
        "has_protocol"
    ] = 0.0

    row[
        "is_network_event"
    ] = 0.0

    row[
        "has_target_filename"
    ] = 0.0

    row[
        "has_image_loaded"
    ] = 0.0

    row[
        "has_target_object"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Context
    # -------------------------------------------------------------------------

    row[
        "administrative_powershell_context"
    ] = float(
        case["administrative"]
        and "powershell"
        in command_lower
    )

    row[
        "strong_behavior_signal"
    ] = 0.0


    # -------------------------------------------------------------------------
    # Ordered structured row
    # -------------------------------------------------------------------------

    return {
        column: row[column]
        for column in structured_columns
    }


controlled_rows = []

controlled_text = []

controlled_ids = []


for case in BENIGN_CASES:

    controlled_rows.append(
        build_case(case)
    )

    controlled_text.append(
        case["behavior_text"]
    )

    controlled_ids.append(
        case["case_id"]
    )


controlled_structured = pd.DataFrame(
    controlled_rows,
    columns=structured_columns
)


controlled_behavior = pd.DataFrame(
    {
        "behavior_text_normalized":
            controlled_text
    }
)


# =============================================================================
# 9. TRANSFORM CONTROLLED DATA
# =============================================================================

print()
print("-" * 80)
print("Transforming controlled benign data")
print("-" * 80)


categorical_matrix = encoder.transform(
    controlled_structured[
        categorical_columns
    ]
    .fillna("")
    .astype(str)
)


numerical_input = (
    controlled_structured[
        numerical_columns
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


if numerical_input.isna().any().any():
    fail(
        "Controlled numerical input contains NaN."
    )


numerical_matrix = scaler.transform(
    numerical_input
)


tfidf_matrix = tfidf.transform(
    controlled_behavior[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


categorical_dense = to_dense_array(
    categorical_matrix
)

numerical_dense = to_dense_array(
    numerical_matrix
)

tfidf_dense = to_dense_array(
    tfidf_matrix
)


controlled_matrix = np.hstack(
    [
        categorical_dense,
        numerical_dense,
        tfidf_dense,
    ]
)


if controlled_matrix.shape != (
    len(BENIGN_CASES),
    EXPECTED_FINAL
):

    fail(
        "Controlled model input dimension "
        f"mismatch: {controlled_matrix.shape}"
    )


print(
    f"[PASS] Controlled model matrix: "
    f"{controlled_matrix.shape}"
)


# =============================================================================
# 10. CONTROLLED MODEL PROBABILITIES
# =============================================================================

print()
print("-" * 80)
print("Calculating controlled model probabilities")
print("-" * 80)


controlled_probabilities = model.predict_proba(
    controlled_matrix
)

controlled_probabilities = to_dense_array(
    controlled_probabilities
)


if controlled_probabilities.shape != (
    len(BENIGN_CASES),
    EXPECTED_MITRE
):

    fail(
        "Controlled probability matrix dimension "
        f"mismatch: {controlled_probabilities.shape}"
    )


# =============================================================================
# 11. CREATE TRAINING REPRESENTATION
# =============================================================================

print()
print("-" * 80)
print("Creating training representation index")
print("-" * 80)


train_structured = (
    structured_train[
        structured_columns
    ]
    .copy()
)


train_behavior = (
    behavior_train[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
)


train_labels = (
    labels_train[
        "techniques"
    ]
    .fillna("")
    .astype(str)
)


# -----------------------------------------------------------------------------
# Exact representation strings
# -----------------------------------------------------------------------------

def structured_signature(row):

    parts = []

    for column in structured_columns:

        value = row[column]

        if pd.isna(value):
            value = ""

        parts.append(
            f"{column}={value}"
        )

    return "|".join(parts)


print(
    "Building structured signatures..."
)


train_structured_signatures = (
    train_structured.apply(
        structured_signature,
        axis=1
    )
)


controlled_structured_signatures = (
    controlled_structured.apply(
        structured_signature,
        axis=1
    )
)


# =============================================================================
# 12. EXACT STRUCTURED MATCH ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing exact structured matches")
print("-" * 80)


signature_to_train_indices = {}


for index, signature in enumerate(
    train_structured_signatures
):

    signature_to_train_indices.setdefault(
        signature,
        []
    ).append(index)


exact_match_records = []


for case_index, case_id in enumerate(
    controlled_ids
):

    signature = (
        controlled_structured_signatures.iloc[
            case_index
        ]
    )

    matching_indices = (
        signature_to_train_indices.get(
            signature,
            []
        )
    )


    label_counter = {}


    for train_index in matching_indices:

        label = (
            train_labels.iloc[
                train_index
            ]
        )

        label_counter[label] = (
            label_counter.get(label, 0)
            + 1
        )


    dominant_label = ""

    dominant_count = 0


    if label_counter:

        dominant_label, dominant_count = (
            max(
                label_counter.items(),
                key=lambda x: x[1]
            )
        )


    exact_match_records.append(
        {
            "case_id":
                case_id,

            "exact_structured_train_rows":
                len(matching_indices),

            "unique_labels_among_exact_matches":
                len(label_counter),

            "dominant_training_label":
                dominant_label,

            "dominant_training_label_count":
                dominant_count,

            "training_label_distribution":
                json.dumps(
                    label_counter,
                    sort_keys=True
                ),
        }
    )


exact_match_df = pd.DataFrame(
    exact_match_records
)


# =============================================================================
# 13. FULL EXACT MATCH ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing exact structured + behavior matches")
print("-" * 80)


full_signature_to_train_indices = {}


for index in range(
    len(train_structured)
):

    signature = (
        train_structured_signatures.iloc[
            index
        ]
        + "||BEHAVIOR||"
        + normalize_text(
            train_behavior.iloc[index]
        )
    )

    full_signature_to_train_indices.setdefault(
        signature,
        []
    ).append(index)


full_match_records = []


for case_index, case_id in enumerate(
    controlled_ids
):

    full_signature = (
        controlled_structured_signatures.iloc[
            case_index
        ]
        + "||BEHAVIOR||"
        + normalize_text(
            controlled_text[
                case_index
            ]
        )
    )


    matching_indices = (
        full_signature_to_train_indices.get(
            full_signature,
            []
        )
    )


    label_counter = {}


    for train_index in matching_indices:

        label = (
            train_labels.iloc[
                train_index
            ]
        )

        label_counter[label] = (
            label_counter.get(label, 0)
            + 1
        )


    full_match_records.append(
        {
            "case_id":
                case_id,

            "exact_full_train_rows":
                len(matching_indices),

            "unique_labels_among_exact_full_matches":
                len(label_counter),

            "full_match_label_distribution":
                json.dumps(
                    label_counter,
                    sort_keys=True
                ),
        }
    )


full_match_df = pd.DataFrame(
    full_match_records
)


# =============================================================================
# 14. TF-IDF SIMILARITY ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing TF-IDF lexical similarity")
print("-" * 80)


train_tfidf = tfidf.transform(
    train_behavior.tolist()
)


# Normalize sparse matrices manually.
train_tfidf_norms = np.sqrt(
    train_tfidf.multiply(
        train_tfidf
    ).sum(
        axis=1
    )
).A1


controlled_tfidf_norms = np.sqrt(
    tfidf_matrix.multiply(
        tfidf_matrix
    ).sum(
        axis=1
    )
).A1


tfidf_neighbor_records = []


TOP_K_TFIDF = 10


for case_index, case_id in enumerate(
    controlled_ids
):

    query = tfidf_matrix[
        case_index
    ]


    query_norm = (
        controlled_tfidf_norms[
            case_index
        ]
    )


    if query_norm == 0:

        similarities = np.zeros(
            len(train_behavior)
        )

    else:

        dot_products = (
            train_tfidf
            .dot(
                query.T
            )
            .toarray()
            .ravel()
        )

        similarities = (
            dot_products
            / (
                train_tfidf_norms
                * query_norm
                + 1e-12
            )
        )


    top_indices = np.argsort(
        similarities
    )[::-1][
        :TOP_K_TFIDF
    ]


    for rank, train_index in enumerate(
        top_indices,
        start=1
    ):

        similarity = float(
            similarities[
                train_index
            ]
        )

        tfidf_neighbor_records.append(
            {
                "case_id":
                    case_id,

                "rank":
                    rank,

                "train_row_index":
                    int(train_index),

                "tfidf_cosine_similarity":
                    similarity,

                "training_label":
                    train_labels.iloc[
                        train_index
                    ],

                "training_behavior_text":
                    train_behavior.iloc[
                        train_index
                    ],

                "training_process_name":
                    train_structured.iloc[
                        train_index
                    ][
                        "process_name"
                    ],

                "training_parent_process_name":
                    train_structured.iloc[
                        train_index
                    ][
                        "parent_process_name"
                    ],

                "training_command_length":
                    safe_float(
                        train_structured.iloc[
                            train_index
                        ][
                            "command_length"
                        ]
                    ),
            }
        )


tfidf_neighbor_df = pd.DataFrame(
    tfidf_neighbor_records
)


# =============================================================================
# 15. STRUCTURED NUMERICAL/CATEGORICAL DISTANCE
# =============================================================================

print()
print("-" * 80)
print("Analyzing structured-feature similarity")
print("-" * 80)


train_categorical = encoder.transform(
    train_structured[
        categorical_columns
    ]
    .fillna("")
    .astype(str)
)


train_numerical = scaler.transform(
    train_structured[
        numerical_columns
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


train_categorical_dense = to_dense_array(
    train_categorical
)

train_numerical_dense = to_dense_array(
    train_numerical
)


# -----------------------------------------------------------------------------
# Use a bounded feature distance:
#
#   categorical mismatch count
#   +
#   mean absolute scaled numerical difference
#
# This is diagnostic only.
# It is NOT the model's training objective.
# -----------------------------------------------------------------------------

structured_neighbor_records = []


TOP_K_STRUCTURED = 10


for case_index, case_id in enumerate(
    controlled_ids
):

    controlled_cat = (
        categorical_dense[
            case_index
        ]
    )

    controlled_num = (
        numerical_dense[
            case_index
        ]
    )


    cat_mismatch = np.sum(
        train_categorical_dense
        != controlled_cat,
        axis=1
    )


    num_distance = np.mean(
        np.abs(
            train_numerical_dense
            - controlled_num
        ),
        axis=1
    )


    # Primary ordering:
    # categorical mismatch first,
    # then numerical distance.
    order = np.lexsort(
        (
            num_distance,
            cat_mismatch
        )
    )


    top_indices = order[
        :TOP_K_STRUCTURED
    ]


    for rank, train_index in enumerate(
        top_indices,
        start=1
    ):

        structured_neighbor_records.append(
            {
                "case_id":
                    case_id,

                "rank":
                    rank,

                "train_row_index":
                    int(train_index),

                "categorical_mismatch_count":
                    int(
                        cat_mismatch[
                            train_index
                        ]
                    ),

                "mean_scaled_numerical_distance":
                    float(
                        num_distance[
                            train_index
                        ]
                    ),

                "training_label":
                    train_labels.iloc[
                        train_index
                    ],

                "training_process_name":
                    train_structured.iloc[
                        train_index
                    ][
                        "process_name"
                    ],

                "training_parent_process_name":
                    train_structured.iloc[
                        train_index
                    ][
                        "parent_process_name"
                    ],

                "training_behavior_text":
                    train_behavior.iloc[
                        train_index
                    ],
            }
        )


structured_neighbor_df = pd.DataFrame(
    structured_neighbor_records
)


# =============================================================================
# 16. LABEL DISTRIBUTION ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing training labels")
print("-" * 80)


label_counts = {}


for value in train_labels:

    labels = parse_label_string(
        value
    )

    for label in labels:

        label_counts[label] = (
            label_counts.get(label, 0)
            + 1
        )


label_analysis_records = []


for case_index, case_id in enumerate(
    controlled_ids
):

    case_probabilities = (
        controlled_probabilities[
            case_index
        ]
    )


    ranking = np.argsort(
        case_probabilities
    )[::-1]


    top_techniques = [
        mitre_classes[
            int(index)
        ]
        for index in ranking[:5]
    ]


    candidate_techniques = set(
        top_techniques
    )


    # Also include all threshold-triggered techniques.
    for class_index, technique in enumerate(
        mitre_classes
    ):

        probability = float(
            case_probabilities[
                class_index
            ]
        )

        if probability >= thresholds[
            technique
        ]:

            candidate_techniques.add(
                technique
            )


    for technique in sorted(
        candidate_techniques
    ):

        label_analysis_records.append(
            {
                "case_id":
                    case_id,

                "technique":
                    technique,

                "training_row_count":
                    int(
                        label_counts.get(
                            technique,
                            0
                        )
                    ),

                "training_row_percentage":
                    (
                        100.0
                        * label_counts.get(
                            technique,
                            0
                        )
                        / len(
                            train_labels
                        )
                    ),

                "controlled_probability":
                    float(
                        case_probabilities[
                            mitre_classes.index(
                                technique
                            )
                        ]
                    ),

                "production_threshold":
                    float(
                        thresholds[
                            technique
                        ]
                    ),

                "threshold_triggered":
                    bool(
                        case_probabilities[
                            mitre_classes.index(
                                technique
                            )
                        ]
                        >= thresholds[
                            technique
                        ]
                    ),
            }
        )


label_analysis_df = pd.DataFrame(
    label_analysis_records
)


# =============================================================================
# 17. MODEL COEFFICIENT ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing model coefficients")
print("-" * 80)


# OneVsRestClassifier contains one LogisticRegression estimator per class.
#
# We inspect the largest positive coefficients for the predicted techniques.
# This helps determine whether lexical terms such as "cmd", "net", "query",
# etc. are strongly associated with a technique.
# =============================================================================


feature_names = []

try:

    categorical_feature_names = (
        encoder.get_feature_names_out(
            categorical_columns
        )
    )

    feature_names.extend(
        [
            str(x)
            for x in categorical_feature_names
        ]
    )

except Exception:

    categorical_feature_names = [
        f"categorical_feature_{i}"
        for i in range(
            EXPECTED_CATEGORICAL_ENCODED
        )
    ]

    feature_names.extend(
        categorical_feature_names
    )


feature_names.extend(
    numerical_columns
)


try:

    tfidf_feature_names = (
        tfidf.get_feature_names_out()
    )

    feature_names.extend(
        [
            str(x)
            for x in tfidf_feature_names
        ]
    )

except Exception:

    tfidf_feature_names = [
        f"tfidf_feature_{i}"
        for i in range(
            EXPECTED_TFIDF
        )
    ]

    feature_names.extend(
        tfidf_feature_names
    )


if len(feature_names) != EXPECTED_FINAL:
    fail(
        "Model feature-name count does not "
        f"equal {EXPECTED_FINAL}."
    )


coefficient_records = []


# The model may expose estimators_ in OneVsRestClassifier.
if not hasattr(
    model,
    "estimators_"
):

    fail(
        "Production model does not expose "
        "estimators_."
    )


for case_index, case_id in enumerate(
    controlled_ids
):

    probabilities = (
        controlled_probabilities[
            case_index
        ]
    )


    ranking = np.argsort(
        probabilities
    )[::-1
    ]


    # Analyze top three predicted classes.
    top_indices = ranking[:3]


    for class_index in top_indices:

        class_index = int(
            class_index
        )

        technique = (
            mitre_classes[
                class_index
            ]
        )

        probability = float(
            probabilities[
                class_index
            ]
        )

        estimator = (
            model.estimators_[
                class_index
            ]
        )


        if not hasattr(
            estimator,
            "coef_"
        ):
            continue


        coefficients = (
            estimator.coef_.ravel()
        )


        if len(coefficients) != len(
            feature_names
        ):
            continue


        # Top positive coefficients.
        positive_indices = np.argsort(
            coefficients
        )[::-1][:15]


        for rank, feature_index in enumerate(
            positive_indices,
            start=1
        ):

            coefficient = float(
                coefficients[
                    feature_index
                ]
            )


            if coefficient <= 0:
                continue


            coefficient_records.append(
                {
                    "case_id":
                        case_id,

                    "technique":
                        technique,

                    "controlled_probability":
                        probability,

                    "feature_rank":
                        rank,

                    "feature":
                        feature_names[
                            feature_index
                        ],

                    "coefficient":
                        coefficient,

                    "feature_group":
                        (
                            "categorical"
                            if feature_index
                            < EXPECTED_CATEGORICAL_ENCODED
                            else
                            (
                                "numerical"
                                if feature_index
                                < (
                                    EXPECTED_CATEGORICAL_ENCODED
                                    + EXPECTED_NUMERICAL
                                )
                                else
                                "tfidf"
                            )
                        ),
                }
            )


coefficient_df = pd.DataFrame(
    coefficient_records
)


# =============================================================================
# 18. PER-CASE ROOT-CAUSE SUMMARY
# =============================================================================

print()
print("-" * 80)
print("Building per-case root-cause summary")
print("-" * 80)


case_summary_records = []


for case_index, case_id in enumerate(
    controlled_ids
):

    case = case_lookup[
        case_id
    ]


    probabilities = (
        controlled_probabilities[
            case_index
        ]
    )


    ranking = np.argsort(
        probabilities
    )[::-1
    ]


    top_index = int(
        ranking[0]
    )

    top_technique = (
        mitre_classes[
            top_index
        ]
    )

    top_probability = float(
        probabilities[
            top_index
        ]
    )

    top_threshold = float(
        thresholds[
            top_technique
        ]
    )


    triggered = []


    for class_index, technique in enumerate(
        mitre_classes
    ):

        probability = float(
            probabilities[
                class_index
            ]
        )

        threshold = float(
            thresholds[
                technique
            ]
        )

        if probability >= threshold:

            triggered.append(
                technique
            )


    exact_structured = (
        exact_match_df[
            exact_match_df[
                "case_id"
            ]
            == case_id
        ]
    )


    exact_full = (
        full_match_df[
            full_match_df[
                "case_id"
            ]
            == case_id
        ]
    )


    tfidf_neighbors = (
        tfidf_neighbor_df[
            tfidf_neighbor_df[
                "case_id"
            ]
            == case_id
        ]
    )


    structured_neighbors = (
        structured_neighbor_df[
            structured_neighbor_df[
                "case_id"
            ]
            == case_id
        ]
    )


    top_tfidf_similarity = 0.0

    if not tfidf_neighbors.empty:

        top_tfidf_similarity = float(
            tfidf_neighbors[
                "tfidf_cosine_similarity"
            ].max()
        )


    top_structured_mismatch = -1

    top_structured_distance = -1.0

    if not structured_neighbors.empty:

        best_structured = (
            structured_neighbors.iloc[0]
        )

        top_structured_mismatch = int(
            best_structured[
                "categorical_mismatch_count"
            ]
        )

        top_structured_distance = float(
            best_structured[
                "mean_scaled_numerical_distance"
            ]
        )


    case_summary_records.append(
        {
            "case_id":
                case_id,

            "description":
                BENIGN_CASES[
                    case_index
                ].get(
                    "description",
                    case_id
                ),

            "process_name":
                case["process_name"],

            "command_line":
                case["command_line"],

            "behavior_text":
                case["behavior_text"],

            "top_technique":
                top_technique,

            "top_probability":
                top_probability,

            "top_threshold":
                top_threshold,

            "triggered_technique_count":
                len(triggered),

            "triggered_techniques":
                ",".join(triggered),

            "exact_structured_train_rows":
                int(
                    exact_structured.iloc[0][
                        "exact_structured_train_rows"
                    ]
                ),

            "unique_labels_exact_structured":
                int(
                    exact_structured.iloc[0][
                        "unique_labels_among_exact_matches"
                    ]
                ),

            "exact_structured_label_distribution":
                exact_structured.iloc[0][
                    "training_label_distribution"
                ],

            "exact_full_train_rows":
                int(
                    exact_full.iloc[0][
                        "exact_full_train_rows"
                    ]
                ),

            "unique_labels_exact_full":
                int(
                    exact_full.iloc[0][
                        "unique_labels_among_exact_full_matches"
                    ]
                ),

            "exact_full_label_distribution":
                exact_full.iloc[0][
                    "full_match_label_distribution"
                ],

            "top_tfidf_training_similarity":
                top_tfidf_similarity,

            "top_structured_categorical_mismatch":
                top_structured_mismatch,

            "top_structured_numeric_distance":
                top_structured_distance,

            "likely_issue":
                (
                    "EXACT_TRAINING_MATCH"
                    if (
                        int(
                            exact_full.iloc[0][
                                "exact_full_train_rows"
                            ]
                        ) > 0
                    )
                    else
                    (
                        "STRUCTURED_SIMILARITY"
                        if (
                            top_structured_mismatch
                            <= 1
                            and top_structured_distance
                            <= 1.0
                        )
                        else
                        (
                            "TEXT_SIMILARITY"
                            if (
                                top_tfidf_similarity
                                >= 0.50
                            )
                            else
                            "MODEL_OR_DATASET_REQUIRES_REVIEW"
                        )
                    )
                ),
        }
    )


case_summary_df = pd.DataFrame(
    case_summary_records
)


# =============================================================================
# 19. GLOBAL DIAGNOSTIC SUMMARY
# =============================================================================

print()
print("-" * 80)
print("Calculating global diagnostic indicators")
print("-" * 80)


exact_full_match_cases = int(
    (
        case_summary_df[
            "exact_full_train_rows"
        ]
        > 0
    ).sum()
)


exact_structured_match_cases = int(
    (
        case_summary_df[
            "exact_structured_train_rows"
        ]
        > 0
    ).sum()
)


high_text_similarity_cases = int(
    (
        case_summary_df[
            "top_tfidf_training_similarity"
        ]
        >= 0.50
    ).sum()
)


ambiguous_exact_structured_cases = int(
    (
        case_summary_df[
            "unique_labels_exact_structured"
        ]
        > 1
    ).sum()
)


ambiguous_exact_full_cases = int(
    (
        case_summary_df[
            "unique_labels_exact_full"
        ]
        > 1
    ).sum()
)


likely_issue_counts = (
    case_summary_df[
        "likely_issue"
    ]
    .value_counts()
    .to_dict()
)


# =============================================================================
# 20. PRINT IMPORTANT RESULTS
# =============================================================================

print()
print("=" * 80)
print("ROOT-CAUSE ANALYSIS RESULTS")
print("=" * 80)


print()
print(
    f"Benign cases analyzed       : "
    f"{len(case_summary_df)}"
)

print(
    f"Exact full training matches : "
    f"{exact_full_match_cases}"
)

print(
    f"Exact structured matches    : "
    f"{exact_structured_match_cases}"
)

print(
    f"High TF-IDF similarity cases: "
    f"{high_text_similarity_cases}"
)

print(
    f"Ambiguous structured labels : "
    f"{ambiguous_exact_structured_cases}"
)

print(
    f"Ambiguous full labels       : "
    f"{ambiguous_exact_full_cases}"
)


print()
print("-" * 80)
print("Per-case findings")
print("-" * 80)


for _, row in case_summary_df.iterrows():

    print()
    print(
        f"{row['case_id']}"
    )

    print(
        f"  Process        : "
        f"{row['process_name']}"
    )

    print(
        f"  Command        : "
        f"{row['command_line']}"
    )

    print(
        f"  Top prediction : "
        f"{row['top_technique']}"
    )

    print(
        f"  Probability    : "
        f"{row['top_probability']:.6f}"
    )

    print(
        f"  Threshold      : "
        f"{row['top_threshold']:.6f}"
    )

    print(
        f"  Exact full rows: "
        f"{row['exact_full_train_rows']}"
    )

    print(
        f"  Exact struct.  : "
        f"{row['exact_structured_train_rows']}"
    )

    print(
        f"  TF-IDF best    : "
        f"{row['top_tfidf_training_similarity']:.6f}"
    )

    print(
        f"  Likely issue   : "
        f"{row['likely_issue']}"
    )


# =============================================================================
# 21. SAVE CSV REPORTS
# =============================================================================

print()
print("-" * 80)
print("Saving diagnostic reports")
print("-" * 80)


case_summary_df.to_csv(
    CASE_RESULTS_FILE,
    index=False
)


tfidf_neighbor_df.to_csv(
    NEIGHBOR_RESULTS_FILE,
    index=False
)


label_analysis_df.to_csv(
    LABEL_ANALYSIS_FILE,
    index=False
)


coefficient_df.to_csv(
    FEATURE_ANALYSIS_FILE,
    index=False
)


# =============================================================================
# 22. CREATE JSON REPORT
# =============================================================================

report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3A",

    "gate_name":
        "Benign False-Positive Root-Cause Analysis",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "purpose":
        "Identify whether controlled benign false positives "
        "are associated with exact training matches, "
        "structured-feature similarity, lexical similarity, "
        "label ambiguity, or model coefficient behavior.",

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

        "structured_features":
            EXPECTED_STRUCTURED,

        "categorical_features":
            EXPECTED_CATEGORICAL,

        "categorical_encoded_features":
            EXPECTED_CATEGORICAL_ENCODED,

        "numerical_features":
            EXPECTED_NUMERICAL,

        "tfidf_features":
            EXPECTED_TFIDF,

        "final_features":
            EXPECTED_FINAL,

        "mitre_classes":
            EXPECTED_MITRE,
    },

    "dataset": {

        "training_rows":
            int(
                len(structured_train)
            ),

        "training_structured_columns":
            int(
                len(structured_columns)
            ),

        "training_behavior_rows":
            int(
                len(train_behavior)
            ),

    },

    "controlled_benign": {

        "total_cases":
            int(
                len(BENIGN_CASES)
            ),

        "false_positive_cases":
            int(
                len(fp_results)
            ),

    },

    "diagnostic_summary": {

        "exact_full_match_cases":
            exact_full_match_cases,

        "exact_structured_match_cases":
            exact_structured_match_cases,

        "high_tfidf_similarity_cases":
            high_text_similarity_cases,

        "ambiguous_exact_structured_cases":
            ambiguous_exact_structured_cases,

        "ambiguous_exact_full_cases":
            ambiguous_exact_full_cases,

        "likely_issue_counts":
            {
                str(k): int(v)
                for k, v
                in likely_issue_counts.items()
            },
    },

    "important_note":
        "This report is diagnostic only. "
        "No production artifact or dataset was modified.",

    "files": {

        "case_analysis":
            str(
                CASE_RESULTS_FILE
            ),

        "training_neighbors":
            str(
                NEIGHBOR_RESULTS_FILE
            ),

        "label_analysis":
            str(
                LABEL_ANALYSIS_FILE
            ),

        "feature_analysis":
            str(
                FEATURE_ANALYSIS_FILE
            ),

    },
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# =============================================================================
# 23. FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print(
    "PHASE 2 GATE 2.3A STATUS: "
    "DIAGNOSTIC_COMPLETE"
)
print("=" * 80)

print()
print(
    f"JSON report : {REPORT_FILE}"
)

print(
    f"Case report : {CASE_RESULTS_FILE}"
)

print(
    f"Neighbors   : {NEIGHBOR_RESULTS_FILE}"
)

print(
    f"Labels      : {LABEL_ANALYSIS_FILE}"
)

print(
    f"Features    : {FEATURE_ANALYSIS_FILE}"
)

print()
print(
    "Production model artifacts were NOT modified."
)

print(
    "Thresholds were NOT modified."
)

print(
    "Training/test datasets were NOT modified."
)

print(
    "No Windows commands were executed."
)