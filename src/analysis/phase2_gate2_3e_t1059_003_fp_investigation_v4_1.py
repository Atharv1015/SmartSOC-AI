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
# T1059.003 FALSE-POSITIVE INVESTIGATION
#
# READ-ONLY DIAGNOSTIC
#
# Purpose:
#   Investigate the five remaining benign false positives that trigger
#   T1059.003 (Windows Command Shell).
#
# This script does NOT:
#   - retrain the model
#   - modify model artifacts
#   - modify thresholds
#   - modify datasets
#   - execute Windows commands
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
    / "phase2_gate2_3_t1059_003_investigation"
)

REPORT_FILE = (
    REPORT_DIR
    / "t1059_003_fp_investigation_report.json"
)

CASE_FILE = (
    REPORT_DIR
    / "t1059_003_fp_case_analysis.csv"
)

NEIGHBOR_FILE = (
    REPORT_DIR
    / "t1059_003_fp_training_neighbors.csv"
)

COEFFICIENT_FILE = (
    REPORT_DIR
    / "t1059_003_fp_coefficients.csv"
)

TRAIN_STRUCTURED = (
    DATA_DIR
    / "structured_train.csv"
)

TRAIN_BEHAVIOR = (
    DATA_DIR
    / "behavior_text_train.csv"
)

TRAIN_LABELS = (
    DATA_DIR
    / "labels_train.csv"
)

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

MLB_FILE = (
    MODEL_DIR
    / "multilabel_encoder.joblib"
)

THRESHOLD_FILE = (
    MODEL_DIR
    / "thresholds.json"
)


TARGET_TECHNIQUE = "T1059.003"


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    print()
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def clean_text(value):
    if value is None:
        return ""

    return " ".join(
        str(value)
        .lower()
        .strip()
        .split()
    )


# =============================================================================
# CONTROLLED BENIGN CASES
#
# These MUST match the corrected Gate 2.3 cases.
# =============================================================================

BENIGN_CASES = [

    {
        "case_id": "BENIGN_001",
        "description": "Benign PowerShell process enumeration",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "Get-Process",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_002",
        "description": "Benign PowerShell service enumeration",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "Get-Service",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_003",
        "description": "Benign PowerShell hostname/system information",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "$env:COMPUTERNAME",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_004",
        "description": "Benign IP configuration query",
        "process_name": "cmd.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "ipconfig /all",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_005",
        "description": "Benign identity query",
        "process_name": "cmd.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "whoami",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_006",
        "description": "Benign task enumeration",
        "process_name": "cmd.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "tasklist",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_007",
        "description": "Benign network connection enumeration",
        "process_name": "cmd.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "netstat -ano",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_008",
        "description": "Benign PowerShell service check",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line":
            "Get-Service | Where-Object {$_.Status -eq 'Running'}",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_009",
        "description": "Benign PowerShell filesystem enumeration",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "Get-ChildItem C:\\Windows",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_010",
        "description": "Benign local account enumeration",
        "process_name": "cmd.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "net user",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_011",
        "description": "Benign Windows service host observation",
        "process_name": "services.exe",
        "parent_process_name": "wininit.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "",
        "parent_command_line": "",
        "user": "SYSTEM",
        "parent_user": "SYSTEM",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },

    {
        "case_id": "BENIGN_012",
        "description": "Benign PowerShell system directory observation",
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "process_extension": "exe",
        "parent_extension": "exe",
        "command_line": "Get-ChildItem C:\\Windows\\System32",
        "parent_command_line": "",
        "user": "Atharv",
        "parent_user": "Atharv",
        "system_level": 4,
        "is_sysmon": 1,
        "is_security_channel": 0,
    },
]


# =============================================================================
# FEATURE BUILDER
# =============================================================================

def build_structured_row(case):

    process_name = clean_text(
        case["process_name"]
    )

    parent_process_name = clean_text(
        case["parent_process_name"]
    )

    process_extension = clean_text(
        case["process_extension"]
    )

    parent_extension = clean_text(
        case["parent_extension"]
    )

    command_line = str(
        case["command_line"]
    ).strip()

    parent_command_line = str(
        case["parent_command_line"]
    ).strip()

    command_lower = command_line.lower()

    parent_command_lower = (
        parent_command_line.lower()
    )

    combined = (
        command_lower
        + " "
        + parent_command_lower
    )

    row = {}

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

    row["process_name"] = process_name
    row["parent_process_name"] = parent_process_name
    row["process_extension"] = process_extension
    row["parent_extension"] = parent_extension

    row["has_process_image"] = int(
        bool(process_name)
    )

    row["has_parent_image"] = int(
        bool(parent_process_name)
    )

    row["has_user"] = int(
        bool(case["user"])
    )

    row["has_parent_user"] = int(
        bool(case["parent_user"])
    )

    row["is_system_user"] = int(
        clean_text(case["user"]) == "system"
    )

    row["is_admin_user"] = 0
    row["integrity_high"] = 0
    row["integrity_medium"] = 1

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

    row["uses_powershell"] = int(
        "powershell" in process_name
        or "powershell" in command_lower
    )

    row["uses_cmd"] = int(
        process_name == "cmd.exe"
        or "cmd.exe" in command_lower
    )

    def any_term(terms):
        return int(
            any(
                term in combined
                for term in terms
            )
        )

    row["uses_net"] = any_term(
        ["net ", "net.exe"]
    )

    row["uses_whoami"] = any_term(
        ["whoami"]
    )

    row["uses_schtasks"] = any_term(
        ["schtasks"]
    )

    row["uses_reg"] = any_term(
        ["reg ", "reg.exe"]
    )

    row["uses_rundll32"] = any_term(
        ["rundll32"]
    )

    row["uses_wmic"] = any_term(
        ["wmic"]
    )

    row["uses_certutil"] = any_term(
        ["certutil"]
    )

    row["uses_bitsadmin"] = any_term(
        ["bitsadmin"]
    )

    row["uses_ipconfig"] = any_term(
        ["ipconfig"]
    )

    row["uses_netstat"] = any_term(
        ["netstat"]
    )

    row["uses_tasklist"] = any_term(
        ["tasklist"]
    )

    row["uses_sc"] = any_term(
        ["sc ", "sc.exe"]
    )

    row["uses_mshta"] = any_term(
        ["mshta"]
    )

    row["uses_regsvr32"] = any_term(
        ["regsvr32"]
    )

    row["uses_cscript"] = any_term(
        ["cscript"]
    )

    row["uses_wscript"] = any_term(
        ["wscript"]
    )

    row["uses_wevtutil"] = any_term(
        ["wevtutil"]
    )

    row["uses_vssadmin"] = any_term(
        ["vssadmin"]
    )

    row["has_encoded_command"] = int(
        "-enc" in command_lower
        or "-encodedcommand" in command_lower
    )

    row["has_download_behavior"] = int(
        any(
            term in command_lower
            for term in [
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
            term in command_lower
            for term in [
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
            term in command_lower
            for term in [
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
            term in command_lower
            for term in [
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
            term in command_lower
            for term in [
                "frombase64string",
                "invoke-expression",
                "iex ",
                "base64",
                "char(",
            ]
        )
    )

    row["same_process_parent"] = int(
        bool(process_name)
        and process_name == parent_process_name
    )

    row["suspicious_parent_wazuh"] = int(
        "wazuh" in parent_process_name
    )

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

    row["administrative_powershell_context"] = int(
        row["uses_powershell"]
        and (
            "administrator"
            in clean_text(case["user"])
            or clean_text(case["user"]) == "system"
            or "admin"
            in clean_text(case["user"])
        )
    )

    row["strong_behavior_signal"] = int(
        any(
            row[key] == 1
            for key in [
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


def build_behavior_text(case):

    pieces = []

    if case["process_name"]:
        pieces.append(
            case["process_name"]
        )

    if case["parent_process_name"]:
        pieces.append(
            case["parent_process_name"]
        )

    if case["command_line"]:
        pieces.append(
            case["command_line"]
        )

    return clean_text(
        " ".join(pieces)
    )


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3")
print("T1059.003 FALSE-POSITIVE INVESTIGATION")
print("=" * 80)


# =============================================================================
# CHECK FILES
# =============================================================================

section("Checking required files")

required_files = [
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    TFIDF_FILE,
    MODEL_FILE,
    MLB_FILE,
    THRESHOLD_FILE,
    TRAIN_STRUCTURED,
    TRAIN_BEHAVIOR,
    TRAIN_LABELS,
]

for path in required_files:

    if not path.exists():

        fail(
            f"Missing file: {path}"
        )

    print(
        f"[OK] {path}"
    )


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

section("Loading production artifacts")

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


thresholds = threshold_config[
    "thresholds"
]


mitre_classes = [
    str(x)
    for x in mlb.classes_
]


if TARGET_TECHNIQUE not in mitre_classes:

    fail(
        f"{TARGET_TECHNIQUE} not found in model classes."
    )


target_index = mitre_classes.index(
    TARGET_TECHNIQUE
)

target_threshold = float(
    thresholds[
        TARGET_TECHNIQUE
    ]
)


print(
    f"Target technique : "
    f"{TARGET_TECHNIQUE}"
)

print(
    f"Target index     : "
    f"{target_index}"
)

print(
    f"Target threshold : "
    f"{target_threshold}"
)


# =============================================================================
# LOAD TRAINING DATA
# =============================================================================

section("Loading training data")

train_structured = pd.read_csv(
    TRAIN_STRUCTURED
)

train_behavior = pd.read_csv(
    TRAIN_BEHAVIOR
)

train_labels = pd.read_csv(
    TRAIN_LABELS
)


if len(train_structured) != len(
    train_behavior
):

    fail(
        "Training structured/behavior row mismatch."
    )


if len(train_structured) != len(
    train_labels
):

    fail(
        "Training structured/label row mismatch."
    )


print(
    f"Training rows: "
    f"{len(train_structured)}"
)


# =============================================================================
# TRAINING TARGET DISTRIBUTION
# =============================================================================

section(
    f"Training label analysis for {TARGET_TECHNIQUE}"
)


target_mask = (
    train_labels[
        "techniques"
    ]
    .fillna("")
    .astype(str)
    .apply(
        lambda x:
            TARGET_TECHNIQUE in [
                item.strip()
                for item in x.split(",")
                if item.strip()
            ]
    )
)


target_rows = train_structured[
    target_mask
].copy()


print(
    f"Training rows containing "
    f"{TARGET_TECHNIQUE}: "
    f"{int(target_mask.sum())}"
)


if len(target_rows) == 0:

    fail(
        "No training examples found for target technique."
    )


print()
print(
    "T1059.003 training process distribution:"
)

if "process_name" in target_rows.columns:

    print(
        target_rows[
            "process_name"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .value_counts()
        .head(20)
        .to_string()
    )


print()
print(
    "T1059.003 training uses_cmd distribution:"
)

if "uses_cmd" in target_rows.columns:

    print(
        target_rows[
            "uses_cmd"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


print()
print(
    "T1059.003 training system_level distribution:"
)

if "system_level" in target_rows.columns:

    print(
        target_rows[
            "system_level"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


# =============================================================================
# BUILD CONTROLLED DATA
# =============================================================================

section("Building corrected benign cases")

structured_rows = []

behavior_rows = []

for case in BENIGN_CASES:

    row = build_structured_row(
        case
    )

    structured_rows.append(
        row
    )

    behavior_rows.append(
        build_behavior_text(
            case
        )
    )


controlled_structured = pd.DataFrame(
    structured_rows
)

controlled_behavior = pd.DataFrame(
    {
        behavior_text_column:
            behavior_rows
    }
)


controlled_structured = (
    controlled_structured[
        structured_columns
    ]
)


# =============================================================================
# PREPROCESS CONTROLLED CASES
# =============================================================================

section("Running production preprocessing")

X_cat = encoder.transform(
    controlled_structured[
        categorical_columns
    ]
)

X_num = scaler.transform(
    controlled_structured[
        numerical_columns
    ]
)

X_text = tfidf.transform(
    controlled_behavior[
        behavior_text_column
    ]
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
    f"Final matrix: "
    f"{X_final.shape}"
)


# =============================================================================
# PREDICTIONS
# =============================================================================

section(
    "Finding T1059.003 false-positive cases"
)

probabilities = np.asarray(
    model.predict_proba(
        X_final
    ),
    dtype=float
)


fp_indices = []

for i, case in enumerate(
    BENIGN_CASES
):

    probability = float(
        probabilities[
            i,
            target_index
        ]
    )

    triggered = (
        probability
        >= target_threshold
    )

    print(
        f"{case['case_id']} | "
        f"T1059.003={probability:.6f} | "
        f"threshold={target_threshold:.6f} | "
        f"triggered={triggered}"
    )

    if triggered:

        fp_indices.append(
            i
        )


print()
print(
    f"T1059.003 false-positive cases: "
    f"{len(fp_indices)}"
)


# =============================================================================
# FEATURE CONTRIBUTIONS
# =============================================================================

section(
    "Calculating T1059.003 feature contributions"
)


# OneVsRestClassifier contains one binary LogisticRegression estimator
# for each encoded class.

estimator = model.estimators_[
    target_index
]


coef = np.asarray(
    estimator.coef_
).ravel()


intercept = float(
    np.asarray(
        estimator.intercept_
    ).ravel()[0]
)


print(
    f"Coefficient count: "
    f"{len(coef)}"
)


# Get final feature names.

cat_names = list(
    encoder.get_feature_names_out(
        categorical_columns
    )
)

num_names = list(
    numerical_columns
)

text_names = [
    f"tfidf:{term}"
    for term in tfidf.get_feature_names_out()
]


feature_names = (
    cat_names
    + num_names
    + text_names
)


if len(feature_names) != len(
    coef
):

    fail(
        "Feature-name count does not match model coefficients."
    )


# =============================================================================
# PER-CASE CONTRIBUTIONS
# =============================================================================

case_rows = []

coefficient_rows = []

neighbor_rows = []


for i in fp_indices:

    case = BENIGN_CASES[i]

    probability = float(
        probabilities[
            i,
            target_index
        ]
    )

    # Sparse row.

    x_row = X_final.getrow(
        i
    )

    feature_indices = (
        x_row.indices
    )

    feature_values = (
        x_row.data
    )

    contributions = []

    for feature_index, value in zip(
        feature_indices,
        feature_values
    ):

        contribution = float(
            value
            * coef[
                feature_index
            ]
        )

        contributions.append(
            (
                feature_index,
                feature_names[
                    feature_index
                ],
                float(value),
                contribution,
            )
        )


    contributions.sort(
        key=lambda x:
            abs(x[3]),
        reverse=True
    )


    positive = [
        item
        for item in contributions
        if item[3] > 0
    ]

    negative = [
        item
        for item in contributions
        if item[3] < 0
    ]


    print()
    print(
        f"{case['case_id']}"
    )

    print(
        f"  Command     : "
        f"{case['command_line']}"
    )

    print(
        f"  Process     : "
        f"{case['process_name']}"
    )

    print(
        f"  Probability : "
        f"{probability:.6f}"
    )

    print(
        "  TOP POSITIVE CONTRIBUTIONS:"
    )


    for item in positive[:10]:

        print(
            f"    + {item[1]} | "
            f"feature_value={item[2]:.6f} | "
            f"contribution={item[3]:.6f}"
        )


    print(
        "  TOP NEGATIVE CONTRIBUTIONS:"
    )


    for item in negative[:5]:

        print(
            f"    - {item[1]} | "
            f"feature_value={item[2]:.6f} | "
            f"contribution={item[3]:.6f}"
        )


    top_positive = (
        positive[:10]
    )


    case_rows.append(
        {
            "case_id":
                case["case_id"],

            "command_line":
                case["command_line"],

            "process_name":
                case["process_name"],

            "parent_process_name":
                case["parent_process_name"],

            "system_level":
                case["system_level"],

            "uses_cmd":
                build_structured_row(
                    case
                )[
                    "uses_cmd"
                ],

            "uses_powershell":
                build_structured_row(
                    case
                )[
                    "uses_powershell"
                ],

            "probability":
                probability,

            "threshold":
                target_threshold,

            "intercept":
                intercept,

            "top_positive_feature":
                (
                    top_positive[0][1]
                    if top_positive
                    else ""
                ),

            "top_positive_contribution":
                (
                    top_positive[0][3]
                    if top_positive
                    else 0.0
                ),

            "top_5_positive_features":
                ";".join(
                    [
                        item[1]
                        for item
                        in top_positive[:5]
                    ]
                ),
        }
    )


    for rank, item in enumerate(
        positive[:20],
        start=1
    ):

        coefficient_rows.append(
            {
                "case_id":
                    case["case_id"],

                "rank":
                    rank,

                "feature":
                    item[1],

                "feature_value":
                    item[2],

                "coefficient":
                    float(
                        coef[
                            item[0]
                        ]
                    ),

                "contribution":
                    item[3],
            }
        )


# =============================================================================
# TRAINING NEIGHBOR ANALYSIS
# =============================================================================

section(
    "Finding nearest training examples"
)


# To compare the controlled benign events with training behavior,
# use TF-IDF cosine similarity.
#
# This is diagnostic only.

train_behavior_text = (
    train_behavior[
        behavior_text_column
    ]
    .fillna("")
    .astype(str)
)

train_text_matrix = tfidf.transform(
    train_behavior_text
)

for i in fp_indices:

    controlled_vector = X_text[
        i
    ]


    similarities = (
        train_text_matrix
        @ controlled_vector.T
    )


    similarities = np.asarray(
        similarities.toarray()
        if hasattr(
            similarities,
            "toarray"
        )
        else similarities
    ).ravel()


    top_indices = np.argsort(
        similarities
    )[
        ::-1
    ][:10]


    for rank, train_index in enumerate(
        top_indices,
        start=1
    ):

        label = str(
            train_labels.iloc[
                train_index
            ]["techniques"]
        )

        process = str(
            train_structured.iloc[
                train_index
            ].get(
                "process_name",
                ""
            )
        )

        command_length = (
            train_structured.iloc[
                train_index
            ].get(
                "command_length",
                np.nan
            )
        )

        uses_cmd = (
            train_structured.iloc[
                train_index
            ].get(
                "uses_cmd",
                np.nan
            )
        )

        neighbor_rows.append(
            {
                "benign_case":
                    BENIGN_CASES[
                        i
                    ]["case_id"],

                "rank":
                    rank,

                "train_index":
                    int(train_index),

                "tfidf_cosine_similarity":
                    float(
                        similarities[
                            train_index
                        ]
                    ),

                "train_label":
                    label,

                "train_process_name":
                    process,

                "train_command_length":
                    command_length,

                "train_uses_cmd":
                    uses_cmd,

                "train_behavior_text":
                    str(
                        train_behavior.iloc[
                            train_index
                        ][
                            behavior_text_column
                        ]
                    ),
            }
        )


# =============================================================================
# GLOBAL COEFFICIENT ANALYSIS
# =============================================================================

section(
    "Inspecting global T1059.003 coefficients"
)


coefficient_df = pd.DataFrame(
    {
        "feature":
            feature_names,

        "coefficient":
            coef,
    }
)


coefficient_df[
    "abs_coefficient"
] = coefficient_df[
    "coefficient"
].abs()


top_positive_coefficients = (
    coefficient_df[
        coefficient_df[
            "coefficient"
        ] > 0
    ]
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(50)
)


print(
    "Top positive T1059.003 coefficients:"
)

print(
    top_positive_coefficients[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


coefficient_df.to_csv(
    COEFFICIENT_FILE,
    index=False
)


# =============================================================================
# SAVE REPORTS
# =============================================================================

section(
    "Saving investigation reports"
)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


pd.DataFrame(
    case_rows
).to_csv(
    CASE_FILE,
    index=False
)


pd.DataFrame(
    coefficient_rows
).to_csv(
    COEFFICIENT_FILE,
    index=False
)


pd.DataFrame(
    neighbor_rows
).to_csv(
    NEIGHBOR_FILE,
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3",

    "investigation":
        "T1059.003 false-positive root-cause analysis",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "target_technique":
        TARGET_TECHNIQUE,

    "target_threshold":
        target_threshold,

    "controlled_cases":
        len(BENIGN_CASES),

    "t1059_003_false_positive_cases":
        len(fp_indices),

    "false_positive_case_ids":
        [
            BENIGN_CASES[i]["case_id"]
            for i in fp_indices
        ],

    "training_target_rows":
        int(target_mask.sum()),

    "training_t1059_003_process_distribution":
        (
            target_rows[
                "process_name"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
            if "process_name"
            in target_rows.columns
            else {}
        ),

    "training_t1059_003_uses_cmd_distribution":
        (
            {
                str(k):
                    int(v)
                for k, v
                in target_rows[
                    "uses_cmd"
                ]
                .value_counts()
                .sort_index()
                .items()
            }
            if "uses_cmd"
            in target_rows.columns
            else {}
        ),

    "training_t1059_003_system_level_distribution":
        (
            {
                str(k):
                    int(v)
                for k, v
                in target_rows[
                    "system_level"
                ]
                .value_counts()
                .sort_index()
                .items()
            }
            if "system_level"
            in target_rows.columns
            else {}
        ),

    "model":

        {
            "type":
                type(model).__name__,

            "feature_count":
                len(coef),

            "intercept":
                intercept,
        },

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

    "files":
        {
            "case_analysis":
                str(CASE_FILE),

            "training_neighbors":
                str(NEIGHBOR_FILE),

            "coefficients":
                str(COEFFICIENT_FILE),

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


print()
print("=" * 80)
print("T1059.003 FP INVESTIGATION COMPLETE")
print("=" * 80)

print(
    f"False-positive cases: "
    f"{len(fp_indices)}"
)

print(
    f"Report: "
    f"{REPORT_FILE}"
)

print(
    f"Case analysis: "
    f"{CASE_FILE}"
)

print(
    f"Training neighbors: "
    f"{NEIGHBOR_FILE}"
)

print(
    f"Coefficients: "
    f"{COEFFICIENT_FILE}"
)

print()
print(
    "NO production artifacts were modified."
)

print(
    "NO thresholds were modified."
)

print(
    "NO training data was modified."
)

print(
    "NO commands were executed."
)