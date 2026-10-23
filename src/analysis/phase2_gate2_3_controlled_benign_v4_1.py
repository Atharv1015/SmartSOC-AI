import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.3
# CONTROLLED BENIGN EVENT VALIDATION
#
# CORRECTED VERSION
#
# IMPORTANT:
#   system_level is an ML feature from the processed dataset.
#   Wazuh rule level is NOT the same thing as system_level.
#
#   The processed v4.1 training data overwhelmingly contains:
#       system_level = 4
#
#   Therefore controlled benign Sysmon/Wazuh-style events use:
#       system_level = 4
#
#   This script is READ-ONLY:
#       - no model modification
#       - no threshold modification
#       - no dataset modification
#       - no Windows command execution
#       - no retraining
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

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
    / "phase2_gate2_3"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase2_gate2_3_controlled_benign_report.json"
)

RESULT_FILE = (
    REPORT_DIR
    / "controlled_benign_results.csv"
)


# =============================================================================
# ARTIFACT PATHS
# =============================================================================

SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"


# =============================================================================
# DATA PATHS
# =============================================================================

TRAIN_STRUCTURED_FILE = (
    DATA_DIR / "structured_train.csv"
)

TRAIN_LABEL_FILE = (
    DATA_DIR / "labels_train.csv"
)


# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_STRUCTURED_COLUMNS = 62

EXPECTED_NUMERICAL_COLUMNS = 58

EXPECTED_CATEGORICAL_COLUMNS = 4

EXPECTED_MITRE_CLASSES = 45


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):

    print()
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def section(title):

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def normalize_text(value):

    if value is None:
        return ""

    return str(value).strip()


def binary(value):

    return int(bool(value))


def has_any(text, terms):

    text = normalize_text(text).lower()

    return int(
        any(
            term.lower() in text
            for term in terms
        )
    )


# =============================================================================
# CONTROLLED BENIGN EVENTS
# =============================================================================
#
# These are intentionally benign administrative / diagnostic activities.
#
# IMPORTANT:
# system_level = 4 represents the observed ML feature semantics.
#
# A separate conceptual Wazuh rule level is NOT placed into system_level.
# =============================================================================

BENIGN_CASES = [

    {
        "case_id": "BENIGN_001",
        "description":
            "Benign PowerShell process enumeration",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "Get-Process",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_002",
        "description":
            "Benign PowerShell service enumeration",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "Get-Service",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_003",
        "description":
            "Benign PowerShell hostname/system information",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "$env:COMPUTERNAME",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_004",
        "description":
            "Benign IP configuration query",
        "process_name":
            "cmd.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "ipconfig /all",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_005",
        "description":
            "Benign identity query",
        "process_name":
            "cmd.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "whoami",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_006",
        "description":
            "Benign task enumeration",
        "process_name":
            "cmd.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "tasklist",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_007",
        "description":
            "Benign network connection enumeration",
        "process_name":
            "cmd.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "netstat -ano",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_008",
        "description":
            "Benign PowerShell service check",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "Get-Service | Where-Object {$_.Status -eq 'Running'}",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_009",
        "description":
            "Benign PowerShell filesystem enumeration",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "Get-ChildItem C:\\Windows",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_010",
        "description":
            "Benign local account enumeration",
        "process_name":
            "cmd.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "net user",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_011",
        "description":
            "Benign Windows service host observation",
        "process_name":
            "services.exe",
        "parent_process_name":
            "wininit.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "",
        "parent_command_line":
            "",
        "user":
            "SYSTEM",
        "parent_user":
            "SYSTEM",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },

    {
        "case_id": "BENIGN_012",
        "description":
            "Benign PowerShell system directory observation",
        "process_name":
            "powershell.exe",
        "parent_process_name":
            "explorer.exe",
        "process_extension":
            "exe",
        "parent_extension":
            "exe",
        "command_line":
            "Get-ChildItem C:\\Windows\\System32",
        "parent_command_line":
            "",
        "user":
            "Atharv",
        "parent_user":
            "Atharv",
        "system_level":
            4,
        "is_sysmon":
            1,
        "is_security_channel":
            0,
    },
]


# =============================================================================
# STRUCTURED FEATURE BUILDER
# =============================================================================

def build_structured_row(case):

    process_name = normalize_text(
        case["process_name"]
    ).lower()

    parent_process_name = normalize_text(
        case["parent_process_name"]
    ).lower()

    process_extension = normalize_text(
        case["process_extension"]
    ).lower()

    parent_extension = normalize_text(
        case["parent_extension"]
    ).lower()

    command_line = normalize_text(
        case["command_line"]
    )

    parent_command_line = normalize_text(
        case["parent_command_line"]
    )

    user = normalize_text(
        case["user"]
    )

    parent_user = normalize_text(
        case["parent_user"]
    )

    command_lower = command_line.lower()

    parent_command_lower = (
        parent_command_line.lower()
    )

    combined_text = (
        command_lower
        + " "
        + parent_command_lower
    )

    row = {}

    # -------------------------------------------------------------------------
    # Identity / event metadata
    # -------------------------------------------------------------------------

    row["event_id"] = 1

    row["system_level"] = float(
        case["system_level"]
    )

    row["is_sysmon"] = int(
        case["is_sysmon"]
    )

    row["is_security_channel"] = int(
        case["is_security_channel"]
    )

    # -------------------------------------------------------------------------
    # Process fields
    # -------------------------------------------------------------------------

    row["process_name"] = process_name

    row["parent_process_name"] = (
        parent_process_name
    )

    row["process_extension"] = (
        process_extension
    )

    row["parent_extension"] = (
        parent_extension
    )

    row["has_process_image"] = int(
        bool(process_name)
    )

    row["has_parent_image"] = int(
        bool(parent_process_name)
    )

    row["has_user"] = int(
        bool(user)
    )

    row["has_parent_user"] = int(
        bool(parent_user)
    )

    row["is_system_user"] = int(
        user.lower() == "system"
    )

    row["is_admin_user"] = 0

    row["integrity_high"] = 0

    row["integrity_medium"] = 1

    # -------------------------------------------------------------------------
    # Command lengths
    # -------------------------------------------------------------------------

    row["command_length"] = len(
        command_line
    )

    row["parent_command_length"] = len(
        parent_command_line
    )

    row["has_command_line"] = int(
        bool(command_line)
    )

    row["has_parent_command"] = int(
        bool(parent_command_line)
    )

    # -------------------------------------------------------------------------
    # Utility indicators
    # -------------------------------------------------------------------------

    row["uses_powershell"] = int(
        "powershell" in process_name
        or "powershell" in command_lower
    )

    row["uses_cmd"] = int(
        process_name == "cmd.exe"
        or "cmd.exe" in command_lower
    )

    row["uses_net"] = has_any(
        combined_text,
        ["net ", "net.exe"]
    )

    row["uses_whoami"] = has_any(
        combined_text,
        ["whoami"]
    )

    row["uses_schtasks"] = has_any(
        combined_text,
        ["schtasks"]
    )

    row["uses_reg"] = has_any(
        combined_text,
        ["reg ", "reg.exe"]
    )

    row["uses_rundll32"] = has_any(
        combined_text,
        ["rundll32"]
    )

    row["uses_wmic"] = has_any(
        combined_text,
        ["wmic"]
    )

    row["uses_certutil"] = has_any(
        combined_text,
        ["certutil"]
    )

    row["uses_bitsadmin"] = has_any(
        combined_text,
        ["bitsadmin"]
    )

    row["uses_ipconfig"] = has_any(
        combined_text,
        ["ipconfig"]
    )

    row["uses_netstat"] = has_any(
        combined_text,
        ["netstat"]
    )

    row["uses_tasklist"] = has_any(
        combined_text,
        ["tasklist"]
    )

    row["uses_sc"] = has_any(
        combined_text,
        ["sc ", "sc.exe"]
    )

    row["uses_mshta"] = has_any(
        combined_text,
        ["mshta"]
    )

    row["uses_regsvr32"] = has_any(
        combined_text,
        ["regsvr32"]
    )

    row["uses_cscript"] = has_any(
        combined_text,
        ["cscript"]
    )

    row["uses_wscript"] = has_any(
        combined_text,
        ["wscript"]
    )

    row["uses_wevtutil"] = has_any(
        combined_text,
        ["wevtutil"]
    )

    row["uses_vssadmin"] = has_any(
        combined_text,
        ["vssadmin"]
    )

    # -------------------------------------------------------------------------
    # Suspicious behavior indicators
    # -------------------------------------------------------------------------

    row["has_encoded_command"] = int(
        "-enc" in command_lower
        or "-encodedcommand" in command_lower
    )

    row["has_download_behavior"] = int(
        any(
            x in command_lower
            for x in [
                "invoke-webrequest",
                "iwr ",
                "wget ",
                "curl ",
                "downloadstring",
                "bitsadmin",
                "start-bitstransfer",
            ]
        )
    )

    row["has_remote_execution_hint"] = int(
        any(
            x in command_lower
            for x in [
                "psexec",
                "winrm",
                "invoke-command",
                "enter-pssession",
                "wmic /node",
            ]
        )
    )

    row["has_registry_operation"] = int(
        any(
            x in command_lower
            for x in [
                "reg add",
                "reg delete",
                "set-itemproperty",
                "new-itemproperty",
                "remove-itemproperty",
            ]
        )
    )

    row["has_file_deletion"] = int(
        any(
            x in command_lower
            for x in [
                "remove-item",
                "del ",
                "erase ",
                "rmdir ",
                "rd ",
            ]
        )
    )

    row["has_obfuscation_hint"] = int(
        any(
            x in command_lower
            for x in [
                "frombase64string",
                "invoke-expression",
                "iex ",
                "base64",
                "char(",
            ]
        )
    )

    row["same_process_parent"] = int(
        process_name != ""
        and process_name == parent_process_name
    )

    row["suspicious_parent_wazuh"] = int(
        "wazuh" in parent_process_name
    )

    # -------------------------------------------------------------------------
    # Hash/network/file fields
    # -------------------------------------------------------------------------

    row["has_hash"] = 0
    row["has_sha256"] = 0
    row["has_md5"] = 0
    row["has_source_ip"] = 0
    row["has_destination_ip"] = 0
    row["has_source_port"] = 0
    row["has_destination_port"] = 0
    row["has_protocol"] = 0
    row["is_network_event"] = 0
    row["has_target_filename"] = 0
    row["has_image_loaded"] = 0
    row["has_target_object"] = 0

    # -------------------------------------------------------------------------
    # Administrative context
    # -------------------------------------------------------------------------

    row["administrative_powershell_context"] = int(
        row["uses_powershell"]
        and (
            user.lower() == "administrator"
            or user.lower() == "system"
            or "admin" in user.lower()
        )
    )

    # -------------------------------------------------------------------------
    # Strong behavior signal
    #
    # Benign cases intentionally do not contain strong attack behaviors.
    # -------------------------------------------------------------------------

    row["strong_behavior_signal"] = int(
        any(
            row[x] == 1
            for x in [
                "has_encoded_command",
                "has_download_behavior",
                "has_remote_execution_hint",
                "has_registry_operation",
                "has_file_deletion",
                "has_obfuscation_hint",
            ]
        )
    )

    return row


# =============================================================================
# BEHAVIOR TEXT BUILDER
# =============================================================================

def build_behavior_text(case, row):

    pieces = []

    process_name = normalize_text(
        case["process_name"]
    )

    parent_name = normalize_text(
        case["parent_process_name"]
    )

    command_line = normalize_text(
        case["command_line"]
    )

    if process_name:
        pieces.append(
            process_name
        )

    if parent_name:
        pieces.append(
            parent_name
        )

    if command_line:
        pieces.append(
            command_line
        )

    text = " ".join(
        pieces
    )

    text = " ".join(
        text.lower().split()
    )

    return text


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3")
print("CONTROLLED BENIGN EVENT VALIDATION")
print("CORRECTED SYSTEM_LEVEL SEMANTICS")
print("=" * 80)


# =============================================================================
# 1. CHECK ARTIFACTS
# =============================================================================

section("Checking production artifacts")

for path in [
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    TFIDF_FILE,
    MODEL_FILE,
    MLB_FILE,
    THRESHOLD_FILE,
]:

    if not path.exists():
        fail(
            f"Missing required artifact: {path}"
        )

    print(f"[OK] {path}")


# =============================================================================
# 2. LOAD SCHEMA
# =============================================================================

section("Loading production schema")

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

behavior_text_column = (
    schema.get(
        "behavior_text_column",
        "behavior_text_normalized"
    )
)


print(
    f"Categorical columns : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical columns   : "
    f"{len(numerical_columns)}"
)

print(
    f"Structured features : "
    f"{len(structured_columns)}"
)

print(
    f"Behavior text       : "
    f"{behavior_text_column}"
)


if len(categorical_columns) != EXPECTED_CATEGORICAL_COLUMNS:

    fail(
        "Unexpected categorical feature count."
    )


if len(numerical_columns) != EXPECTED_NUMERICAL_COLUMNS:

    fail(
        "Unexpected numerical feature count."
    )


if len(structured_columns) != EXPECTED_STRUCTURED_COLUMNS:

    fail(
        "Unexpected structured feature count."
    )


if "system_level" not in numerical_columns:

    fail(
        "system_level is not present in numerical_columns."
    )


# =============================================================================
# 3. LOAD ARTIFACTS
# =============================================================================

section(
    "Loading production preprocessing and model artifacts"
)

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

mlb = joblib.load(
    MLB_FILE
)


with open(
    THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    threshold_config = json.load(f)


if "thresholds" not in threshold_config:

    fail(
        "Threshold artifact does not contain a 'thresholds' object."
    )


thresholds = threshold_config["thresholds"]

# =============================================================================
# 4. VALIDATE ARTIFACT DIMENSIONS
# =============================================================================

section("Validating artifact dimensions")

encoder_features = len(
    encoder.get_feature_names_out()
)

scaler_features = len(
    getattr(
        scaler,
        "scale_",
        []
    )
)

tfidf_features = len(
    getattr(
        tfidf,
        "vocabulary_",
        {}
    )
)

mitre_classes = [
    str(x)
    for x in mlb.classes_
]


print(
    f"Encoder = "
    f"{encoder_features} features."
)

print(
    f"Scaler = "
    f"{scaler_features} features."
)

print(
    f"TF-IDF = "
    f"{tfidf_features} features."
)

print(
    f"MITRE classes = "
    f"{len(mitre_classes)}."
)


if encoder_features != 162:

    fail(
        "Unexpected encoder dimension."
    )

if scaler_features != 58:

    fail(
        "Unexpected scaler dimension."
    )

if tfidf_features != 6699:

    fail(
        "Unexpected TF-IDF dimension."
    )

if len(mitre_classes) != EXPECTED_MITRE_CLASSES:

    fail(
        "Unexpected MITRE class count."
    )


print(
    "[PASS] Production artifact dimensions valid."
)


# =============================================================================
# 5. VALIDATE THRESHOLDS
# =============================================================================

section("Loading production thresholds")

if not isinstance(
    thresholds,
    dict
):

    fail(
        "Threshold artifact is not a JSON object."
    )


missing_thresholds = [
    technique
    for technique in mitre_classes
    if technique not in thresholds
]


if missing_thresholds:

    fail(
        "Missing thresholds for: "
        + ", ".join(
            missing_thresholds
        )
    )


print(
    f"[PASS] "
    f"{len(thresholds)} production thresholds loaded."
)


# =============================================================================
# 6. TRAINING DATA SEMANTIC CHECK
# =============================================================================

section(
    "Validating system_level against training data"
)

if not TRAIN_STRUCTURED_FILE.exists():

    fail(
        f"Missing training structured dataset: "
        f"{TRAIN_STRUCTURED_FILE}"
    )


train_structured = pd.read_csv(
    TRAIN_STRUCTURED_FILE
)


if "system_level" not in train_structured.columns:

    fail(
        "Training data does not contain system_level."
    )


training_system_levels = (
    pd.to_numeric(
        train_structured[
            "system_level"
        ],
        errors="coerce"
    )
)


print(
    "Training system_level distribution:"
)

print(
    training_system_levels
    .value_counts()
    .sort_index()
)


if 4.0 not in set(
    training_system_levels
    .dropna()
    .tolist()
):

    fail(
        "system_level=4 is not present in training data. "
        "Controlled-test semantic assumption must be revisited."
    )


if 12.0 in set(
    training_system_levels
    .dropna()
    .tolist()
):

    print(
        "[WARNING] system_level=12 exists in training."
    )

else:

    print(
        "[PASS] system_level=12 is absent from training."
    )

    print(
        "[PASS] Controlled events use system_level=4."
    )


# =============================================================================
# 7. PREPARE CONTROLLED CASES
# =============================================================================

section(
    "Preparing corrected controlled benign test cases"
)

print(
    f"Controlled benign cases: "
    f"{len(BENIGN_CASES)}"
)


structured_rows = []

behavior_rows = []


for case in BENIGN_CASES:

    row = build_structured_row(
        case
    )

    row["case_id"] = case[
        "case_id"
    ]

    structured_rows.append(
        row
    )

    behavior_rows.append(
        build_behavior_text(
            case,
            row
        )
    )


structured_df = pd.DataFrame(
    structured_rows
)


behavior_df = pd.DataFrame(
    {
        behavior_text_column:
            behavior_rows
    }
)


# Put columns into exact schema order.

structured_df = structured_df[
    [
        "case_id"
    ]
    + structured_columns
]


# =============================================================================
# 8. VALIDATE STRUCTURED FEATURES
# =============================================================================

section(
    "Validating corrected controlled feature schema"
)

missing_columns = [
    col
    for col in structured_columns
    if col not in structured_df.columns
]


if missing_columns:

    fail(
        "Missing controlled structured columns: "
        + ", ".join(
            missing_columns
        )
    )


unexpected_columns = [
    col
    for col in structured_df.columns
    if col not in (
        ["case_id"]
        + structured_columns
    )
]


if unexpected_columns:

    fail(
        "Unexpected controlled structured columns: "
        + ", ".join(
            unexpected_columns
        )
    )


if len(structured_df) != len(
    behavior_df
):

    fail(
        "Structured/behavior row count mismatch."
    )


if structured_df[
    "system_level"
].isna().any():

    fail(
        "Controlled system_level contains NaN."
    )


print(
    f"Structured test matrix: "
    f"{structured_df[structured_columns].shape}"
)

print(
    f"Behavior text matrix: "
    f"{behavior_df.shape}"
)

print(
    "[PASS] Controlled feature schema valid."
)


print()
print(
    "Controlled system_level values:"
)

print(
    structured_df[
        [
            "case_id",
            "system_level"
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 9. PRODUCTION PREPROCESSING
# =============================================================================

section(
    "Running production preprocessing"
)

X_structured = structured_df[
    structured_columns
].copy()


X_categorical = X_structured[
    categorical_columns
]


X_numerical = X_structured[
    numerical_columns
]


X_cat = encoder.transform(
    X_categorical
)

X_num = scaler.transform(
    X_numerical
)

X_text = tfidf.transform(
    behavior_df[
        behavior_text_column
    ]
)


# =============================================================================
# 10. MATRIX DIMENSIONS
# =============================================================================

print()
print(
    "Validating controlled feature dimensions"
)

print(
    f"Categorical encoded : "
    f"{X_cat.shape}"
)

print(
    f"Numerical scaled    : "
    f"{X_num.shape}"
)

print(
    f"TF-IDF              : "
    f"{X_text.shape}"
)


if X_cat.shape != (
    len(BENIGN_CASES),
    162
):

    fail(
        "Invalid categorical encoded shape."
    )


if X_num.shape != (
    len(BENIGN_CASES),
    58
):

    fail(
        "Invalid numerical shape."
    )


if X_text.shape != (
    len(BENIGN_CASES),
    6699
):

    fail(
        "Invalid TF-IDF shape."
    )


from scipy.sparse import hstack


X_final = hstack(
    [
        X_cat,
        X_num,
        X_text,
    ]
).tocsr()


print(
    f"[PASS] Final controlled matrix: "
    f"{X_final.shape}"
)


if X_final.shape != (
    len(BENIGN_CASES),
    6919
):

    fail(
        "Invalid final feature matrix shape."
    )


# =============================================================================
# 11. NUMERICAL STABILITY
# =============================================================================

section(
    "Checking numerical stability"
)


X_num_array = np.asarray(
    X_num,
    dtype=float
)


if not np.isfinite(
    X_num_array
).all():

    fail(
        "Non-finite numerical values detected."
    )


X_cat_array = (
    X_cat.toarray()
    if hasattr(
        X_cat,
        "toarray"
    )
    else np.asarray(
        X_cat
    )
)


if not np.isfinite(
    np.asarray(
        X_cat_array,
        dtype=float
    )
).all():

    fail(
        "Non-finite categorical values detected."
    )


X_text_array = (
    X_text.toarray()
)


if not np.isfinite(
    np.asarray(
        X_text_array,
        dtype=float
    )
).all():

    fail(
        "Non-finite TF-IDF values detected."
    )


print(
    "[PASS] All controlled feature matrices are finite."
)


# =============================================================================
# 12. MODEL INFERENCE
# =============================================================================

section(
    "Running production model"
)

probabilities = model.predict_proba(
    X_final
)


probabilities = np.asarray(
    probabilities,
    dtype=float
)


print(
    f"Probability matrix: "
    f"{probabilities.shape}"
)


if probabilities.shape != (
    len(BENIGN_CASES),
    len(mitre_classes)
):

    fail(
        "Invalid probability matrix shape."
    )


if not np.isfinite(
    probabilities
).all():

    fail(
        "Non-finite probabilities detected."
    )


if (
    probabilities.min() < 0
    or probabilities.max() > 1
):

    fail(
        "Probabilities outside [0,1]."
    )


print(
    "[PASS] Probability matrix valid."
)


# =============================================================================
# 13. ANALYZE PREDICTIONS
# =============================================================================

section(
    "Analyzing corrected benign predictions"
)


result_rows = []


false_positive_cases = 0

passed_cases = 0

triggered_technique_count = 0


for row_index, case in enumerate(
    BENIGN_CASES
):

    probs = probabilities[
        row_index
    ]


    sorted_indices = np.argsort(
        probs
    )[
        ::-1
    ]


    top_index = int(
        sorted_indices[0]
    )

    top_technique = (
        mitre_classes[
            top_index
        ]
    )

    top_probability = float(
        probs[
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
            probs[
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
                (
                    technique,
                    probability,
                    threshold,
                )
            )


    is_false_positive = (
        len(triggered) > 0
    )


    if is_false_positive:

        false_positive_cases += 1

    else:

        passed_cases += 1


    triggered_technique_count += len(
        triggered
    )


    triggered_text = ";".join(
        [
            (
                f"{technique}:"
                f"{probability:.6f}:"
                f"{threshold:.6f}"
            )
            for technique, probability, threshold
            in triggered
        ]
    )


    result_rows.append(
        {
            "case_id":
                case["case_id"],

            "description":
                case["description"],

            "system_level":
                case["system_level"],

            "top_prediction":
                top_technique,

            "top_probability":
                top_probability,

            "top_threshold":
                top_threshold,

            "triggered":
                is_false_positive,

            "triggered_count":
                len(triggered),

            "triggered_techniques":
                triggered_text,

            "command_line":
                case["command_line"],

            "process_name":
                case["process_name"],

            "parent_process_name":
                case["parent_process_name"],

            "behavior_text":
                behavior_rows[
                    row_index
                ],
        }
    )


    status = (
        "FALSE_POSITIVE"
        if is_false_positive
        else "PASS"
    )


    print()
    print(
        f"{case['case_id']} | "
        f"{status} | "
        f"Top={top_technique} | "
        f"Probability={top_probability:.4f} | "
        f"Threshold={top_threshold:.4f}"
    )


    if triggered:

        print(
            "    Triggered: "
            + ";".join(
                [
                    (
                        f"{technique}:"
                        f"{probability:.6f}:"
                        f"{threshold:.6f}"
                    )
                    for technique, probability, threshold
                    in triggered
                ]
            )
        )


# =============================================================================
# 14. SUMMARY
# =============================================================================

section(
    "Corrected controlled benign validation summary"
)


total_cases = len(
    BENIGN_CASES
)


false_positive_rate = (
    false_positive_cases
    / total_cases
)


print(
    f"Total benign cases       : "
    f"{total_cases}"
)

print(
    f"Passed benign cases      : "
    f"{passed_cases}"
)

print(
    f"False-positive cases     : "
    f"{false_positive_cases}"
)

print(
    f"Triggered techniques     : "
    f"{triggered_technique_count}"
)

print(
    f"False-positive rate      : "
    f"{false_positive_rate:.4f}"
)


# =============================================================================
# 15. SAVE RESULTS
# =============================================================================

section(
    "Saving diagnostic reports"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results_df = pd.DataFrame(
    result_rows
)


results_df.to_csv(
    RESULT_FILE,
    index=False
)


# =============================================================================
# 16. GATE STATUS
# =============================================================================
#
# IMPORTANT:
# This is intentionally conservative.
#
# The gate is accepted only when:
#   - all controlled benign cases trigger nothing
#
# Any remaining FP keeps the gate at REVIEW_REQUIRED.
#
# We do NOT alter thresholds here.
# =============================================================================

if false_positive_cases == 0:

    gate_status = "ACCEPTED"

else:

    gate_status = "REVIEW_REQUIRED"


report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3",

    "gate_name":
        "Controlled Benign Event Validation",

    "status":
        gate_status,

    "validation_version":
        "corrected_system_level_semantics",

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

    "controlled_cases":
        total_cases,

    "passed_cases":
        passed_cases,

    "false_positive_cases":
        false_positive_cases,

    "false_positive_rate":
        false_positive_rate,

    "triggered_technique_count":
        triggered_technique_count,

    "system_level_semantics": {

        "controlled_value":
            4,

        "previous_invalid_value":
            12,

        "reason":
            (
                "The processed v4.1 dataset overwhelmingly "
                "uses system_level=4. system_level=12 is "
                "absent from training and therefore was "
                "treated as an out-of-distribution controlled "
                "test value."
            ),

        "training_distribution":
            {
                str(k):
                    int(v)
                for k, v
                in train_structured[
                    "system_level"
                ]
                .value_counts()
                .sort_index()
                .items()
            },
    },

    "dimensions": {

        "structured":
            len(structured_columns),

        "categorical":
            len(categorical_columns),

        "numerical":
            len(numerical_columns),

        "categorical_encoded":
            encoder_features,

        "tfidf":
            tfidf_features,

        "final":
            X_final.shape[1],

        "mitre_classes":
            len(mitre_classes),
    },

    "files": {

        "results":
            str(RESULT_FILE),

        "report":
            str(REPORT_FILE),
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
# FINAL OUTPUT
# =============================================================================

print()
print("=" * 80)
print(
    f"PHASE 2 GATE 2.3 STATUS: "
    f"{gate_status}"
)
print("=" * 80)

print()
print(
    "Controlled system_level was corrected:"
)

print(
    "    Previous: 12  [OUT-OF-DISTRIBUTION]"
)

print(
    "    Current : 4   [OBSERVED TRAINING VALUE]"
)

print()
print(
    f"False-positive rate: "
    f"{false_positive_rate:.4f}"
)

print()
print(
    f"Report : {REPORT_FILE}"
)

print(
    f"Results: {RESULT_FILE}"
)

print()
print(
    "Production model artifacts were NOT modified."
)

print(
    "Thresholds were NOT modified."
)

print(
    "Training data was NOT modified."
)

print(
    "No Windows commands were executed."
)