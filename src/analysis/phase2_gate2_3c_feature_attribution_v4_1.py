import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.3C
# FEATURE ATTRIBUTION / FALSE-POSITIVE ROOT-CAUSE ANALYSIS
#
# READ-ONLY
#
# This script does NOT:
#   - retrain the model
#   - modify the model
#   - modify thresholds
#   - modify datasets
#   - execute Windows commands
#
# It analyzes:
#   1. The 12 controlled benign cases
#   2. Model predictions
#   3. Structured feature values
#   4. Logistic Regression coefficients
#   5. TF-IDF term contributions
#   6. Categorical feature contributions
#   7. Numerical feature contributions
#
# Purpose:
#   Determine exactly why benign administrative activity is being classified
#   as malicious MITRE ATT&CK activity.
# =============================================================================


warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

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
    / "phase2_gate2_3c"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase2_gate2_3c_feature_attribution_report.json"
)

CASE_FILE = (
    REPORT_DIR
    / "benign_case_attribution.csv"
)

CONTRIBUTION_FILE = (
    REPORT_DIR
    / "benign_feature_contributions.csv"
)

TFIDF_FILE = (
    REPORT_DIR
    / "benign_tfidf_contributions.csv"
)

COEFFICIENT_FILE = (
    REPORT_DIR
    / "model_coefficient_analysis.csv"
)

FEATURE_SUMMARY_FILE = (
    REPORT_DIR
    / "feature_false_positive_summary.csv"
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

TFIDF_FILE_ARTIFACT = (
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

THRESHOLDS_FILE = (
    MODEL_DIR
    / "thresholds.json"
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
        fail(f"Required file missing: {path}")
    print(f"[OK] {path}")


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def build_case(
    case_id,
    event_id,
    system_level,
    process_name,
    parent_process_name,
    command_line,
    user="administrator",
    parent_user="administrator",
    integrity="High",
    is_sysmon=1,
    is_security_channel=0,
    behavior_text=None,
):
    """
    Construct a controlled benign event.

    These cases intentionally represent legitimate administrative activity.
    No commands are executed.
    """

    command_line = "" if command_line is None else command_line
    behavior_text = (
        behavior_text
        if behavior_text is not None
        else command_line
    )

    return {
        "case_id": case_id,
        "event_id": event_id,
        "system_level": system_level,
        "is_sysmon": is_sysmon,
        "is_security_channel": is_security_channel,
        "process_name": process_name,
        "parent_process_name": parent_process_name,
        "process_extension": (
            Path(process_name).suffix.lower().lstrip(".")
            if process_name
            else ""
        ),
        "parent_extension": (
            Path(parent_process_name).suffix.lower().lstrip(".")
            if parent_process_name
            else ""
        ),
        "has_process_image": int(bool(process_name)),
        "has_parent_image": int(bool(parent_process_name)),
        "has_user": int(bool(user)),
        "has_parent_user": int(bool(parent_user)),
        "is_system_user": int(
            normalize(user) in ["system", "local system"]
        ),
        "is_admin_user": int(
            normalize(user) in [
                "administrator",
                "admin",
            ]
        ),
        "integrity_high": int(
            normalize(integrity) == "high"
        ),
        "integrity_medium": int(
            normalize(integrity) == "medium"
        ),
        "command_length": len(command_line),
        "parent_command_length": 0,
        "has_command_line": int(bool(command_line)),
        "has_parent_command": 0,
        "uses_powershell": int(
            "powershell" in normalize(process_name)
            or "powershell" in normalize(command_line)
            or "pwsh" in normalize(command_line)
        ),
        "uses_cmd": int(
            normalize(process_name) in [
                "cmd.exe",
                "cmd",
            ]
            or "cmd.exe" in normalize(command_line)
        ),
        "uses_net": int(
            "net.exe" in normalize(command_line)
            or normalize(process_name) == "net.exe"
            or normalize(process_name) == "net"
        ),
        "uses_whoami": int(
            "whoami" in normalize(command_line)
        ),
        "uses_schtasks": int(
            "schtasks" in normalize(command_line)
        ),
        "uses_reg": int(
            "reg.exe" in normalize(command_line)
            or normalize(process_name) == "reg.exe"
        ),
        "uses_rundll32": int(
            "rundll32" in normalize(command_line)
        ),
        "uses_wmic": int(
            "wmic" in normalize(command_line)
        ),
        "uses_certutil": int(
            "certutil" in normalize(command_line)
        ),
        "uses_bitsadmin": int(
            "bitsadmin" in normalize(command_line)
        ),
        "uses_ipconfig": int(
            "ipconfig" in normalize(command_line)
        ),
        "uses_netstat": int(
            "netstat" in normalize(command_line)
        ),
        "uses_tasklist": int(
            "tasklist" in normalize(command_line)
        ),
        "uses_sc": int(
            normalize(process_name) in [
                "sc.exe",
                "sc",
            ]
            or "sc.exe" in normalize(command_line)
        ),
        "uses_mshta": int(
            "mshta" in normalize(command_line)
        ),
        "uses_regsvr32": int(
            "regsvr32" in normalize(command_line)
        ),
        "uses_cscript": int(
            "cscript" in normalize(command_line)
        ),
        "uses_wscript": int(
            "wscript" in normalize(command_line)
        ),
        "uses_wevtutil": int(
            "wevtutil" in normalize(command_line)
        ),
        "uses_vssadmin": int(
            "vssadmin" in normalize(command_line)
        ),
        "has_encoded_command": int(
            "-enc" in normalize(command_line)
            or "encodedcommand" in normalize(command_line)
        ),
        "has_download_behavior": int(
            any(
                x in normalize(command_line)
                for x in [
                    "invoke-webrequest",
                    "iwr ",
                    "downloadstring",
                    "curl ",
                    "wget ",
                    "bitsadmin",
                ]
            )
        ),
        "has_remote_execution_hint": int(
            any(
                x in normalize(command_line)
                for x in [
                    "psexec",
                    "winrm",
                    "invoke-command",
                    "enter-pssession",
                    "wmic /node",
                ]
            )
        ),
        "has_registry_operation": int(
            "reg " in normalize(command_line)
            or "reg.exe" in normalize(command_line)
            or "registry" in normalize(command_line)
        ),
        "has_file_deletion": int(
            any(
                x in normalize(command_line)
                for x in [
                    "remove-item",
                    "del ",
                    "erase ",
                    "rm ",
                    "rmdir",
                ]
            )
        ),
        "has_obfuscation_hint": int(
            any(
                x in normalize(command_line)
                for x in [
                    "base64",
                    "frombase64string",
                    "invoke-expression",
                    "iex ",
                ]
            )
        ),
        "same_process_parent": int(
            normalize(process_name)
            == normalize(parent_process_name)
            and bool(process_name)
        ),
        "suspicious_parent_wazuh": int(
            "wazuh" in normalize(parent_process_name)
        ),
        "has_hash": 0,
        "has_sha256": 0,
        "has_md5": 0,
        "has_source_ip": 0,
        "has_destination_ip": 0,
        "has_source_port": 0,
        "has_destination_port": 0,
        "has_protocol": 0,
        "is_network_event": 0,
        "has_target_filename": 0,
        "has_image_loaded": 0,
        "has_target_object": 0,
        "administrative_powershell_context": int(
            (
                "powershell"
                in normalize(process_name)
            )
            and (
                normalize(integrity)
                == "high"
            )
        ),
        "strong_behavior_signal": int(
            any(
                [
                    "-enc" in normalize(command_line),
                    "downloadstring" in normalize(command_line),
                    "invoke-command" in normalize(command_line),
                    "remove-item" in normalize(command_line),
                    "frombase64string" in normalize(command_line),
                ]
            )
        ),
        "behavior_text_normalized": behavior_text,
    }


# =============================================================================
# CONTROLLED BENIGN CASES
# =============================================================================
#
# Same conceptual set used in Gate 2.3.
# They are intentionally benign examples.
# =============================================================================

BENIGN_CASES = [

    build_case(
        "BENIGN_001",
        1,
        12,
        "powershell.exe",
        "explorer.exe",
        "Get-Process",
        behavior_text=(
            "powershell.exe get-process"
        ),
    ),

    build_case(
        "BENIGN_002",
        2,
        12,
        "powershell.exe",
        "explorer.exe",
        "Get-Service",
        behavior_text=(
            "powershell.exe get-service"
        ),
    ),

    build_case(
        "BENIGN_003",
        3,
        12,
        "powershell.exe",
        "explorer.exe",
        "Get-ComputerInfo",
        behavior_text=(
            "powershell.exe get-computerinfo"
        ),
    ),

    build_case(
        "BENIGN_004",
        4,
        12,
        "cmd.exe",
        "explorer.exe",
        "ipconfig /all",
        behavior_text=(
            "cmd.exe ipconfig /all"
        ),
    ),

    build_case(
        "BENIGN_005",
        5,
        12,
        "cmd.exe",
        "explorer.exe",
        "whoami",
        behavior_text=(
            "cmd.exe whoami"
        ),
    ),

    build_case(
        "BENIGN_006",
        6,
        12,
        "cmd.exe",
        "explorer.exe",
        "tasklist",
        behavior_text=(
            "cmd.exe tasklist"
        ),
    ),

    build_case(
        "BENIGN_007",
        7,
        12,
        "cmd.exe",
        "explorer.exe",
        "netstat -ano",
        behavior_text=(
            "cmd.exe netstat -ano"
        ),
    ),

    build_case(
        "BENIGN_008",
        8,
        12,
        "powershell.exe",
        "services.exe",
        "Get-Service",
        behavior_text=(
            "powershell.exe get-service"
        ),
    ),

    build_case(
        "BENIGN_009",
        9,
        12,
        "powershell.exe",
        "explorer.exe",
        "Get-ChildItem C:\\Windows",
        behavior_text=(
            "powershell.exe get-childitem c windows"
        ),
    ),

    build_case(
        "BENIGN_010",
        10,
        12,
        "cmd.exe",
        "explorer.exe",
        "net user",
        behavior_text=(
            "cmd.exe net user"
        ),
    ),

    build_case(
        "BENIGN_011",
        11,
        12,
        "services.exe",
        "wininit.exe",
        "",
        behavior_text=(
            "services.exe"
        ),
    ),

    build_case(
        "BENIGN_012",
        12,
        12,
        "powershell.exe",
        "explorer.exe",
        "Get-WinEvent -LogName System -MaxEvents 10",
        behavior_text=(
            "powershell.exe get-winevent logname system"
        ),
    ),
]


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3C")
print("FEATURE ATTRIBUTION / FALSE-POSITIVE ROOT-CAUSE ANALYSIS")
print("=" * 80)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 1. CHECK ARTIFACTS
# =============================================================================

print()
print("-" * 80)
print("Checking production artifacts")
print("-" * 80)


for path in [
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    TFIDF_FILE_ARTIFACT,
    MODEL_FILE,
    MLB_FILE,
    THRESHOLDS_FILE,
]:
    require_file(path)


# =============================================================================
# 2. LOAD ARTIFACTS
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


encoder = joblib.load(
    ENCODER_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

tfidf = joblib.load(
    TFIDF_FILE_ARTIFACT
)

model = joblib.load(
    MODEL_FILE
)

mlb = joblib.load(
    MLB_FILE
)

with open(
    THRESHOLDS_FILE,
    "r",
    encoding="utf-8"
) as f:
    thresholds = json.load(f)


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


mitre_classes = [
    str(x)
    for x in mlb.classes_
]


print(
    f"Categorical features : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical features   : "
    f"{len(numerical_columns)}"
)

print(
    f"Structured features  : "
    f"{len(structured_columns)}"
)

print(
    f"TF-IDF features      : "
    f"{len(tfidf.vocabulary_)}"
)

print(
    f"MITRE classes         : "
    f"{len(mitre_classes)}"
)


# =============================================================================
# 3. VALIDATE CASE SCHEMA
# =============================================================================

print()
print("-" * 80)
print("Validating controlled benign cases")
print("-" * 80)


cases_df = pd.DataFrame(
    BENIGN_CASES
)


missing_case_columns = [
    col
    for col in structured_columns
    if col not in cases_df.columns
]


if missing_case_columns:

    fail(
        "Controlled benign cases are missing "
        f"structured columns: {missing_case_columns}"
    )


print(
    f"Controlled cases: "
    f"{len(cases_df)}"
)


# =============================================================================
# 4. BUILD PRODUCTION FEATURE MATRIX
# =============================================================================

print()
print("-" * 80)
print("Building production feature matrix")
print("-" * 80)


X_structured = (
    cases_df[
        structured_columns
    ]
    .copy()
)


X_behavior = (
    cases_df[
        ["behavior_text_normalized"]
    ]
    .copy()
)


# -------------------------------------------------------------------------
# categorical
# -------------------------------------------------------------------------

X_cat = encoder.transform(
    X_structured[
        categorical_columns
    ]
)


# -------------------------------------------------------------------------
# numerical
# -------------------------------------------------------------------------

X_num_raw = (
    X_structured[
        numerical_columns
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0.0)
)


X_num = scaler.transform(
    X_num_raw
)


# -------------------------------------------------------------------------
# TF-IDF
# -------------------------------------------------------------------------

X_text = tfidf.transform(
    X_behavior[
        "behavior_text_normalized"
    ].astype(str)
)


# -------------------------------------------------------------------------
# final matrix
# -------------------------------------------------------------------------

from scipy.sparse import hstack

X_final = hstack(
    [
        X_cat,
        X_num,
        X_text,
    ]
).tocsr()


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

print(
    f"Final matrix        : "
    f"{X_final.shape}"
)


expected_features = (
    len(
        encoder.get_feature_names_out()
    )
    + len(numerical_columns)
    + len(tfidf.vocabulary_)
)


if X_final.shape[1] != expected_features:

    fail(
        "Final feature dimension mismatch."
    )


# =============================================================================
# 5. MODEL PREDICTIONS
# =============================================================================

print()
print("-" * 80)
print("Running model inference")
print("-" * 80)


probabilities = model.predict_proba(
    X_final
)


if isinstance(
    probabilities,
    list
):

    probability_matrix = np.column_stack(
        [
            np.asarray(x).reshape(-1)
            for x in probabilities
        ]
    )

else:

    probability_matrix = np.asarray(
        probabilities
    )


if probability_matrix.shape[1] != len(
    mitre_classes
):

    fail(
        "Probability matrix does not match "
        "MITRE class count."
    )


print(
    f"Probability matrix: "
    f"{probability_matrix.shape}"
)


# =============================================================================
# 6. FEATURE NAMES
# =============================================================================

categorical_feature_names = list(
    encoder.get_feature_names_out(
        categorical_columns
    )
)

numeric_feature_names = list(
    numerical_columns
)

tfidf_feature_names = list(
    tfidf.get_feature_names_out()
)


all_feature_names = (
    categorical_feature_names
    + numeric_feature_names
    + tfidf_feature_names
)


if len(all_feature_names) != X_final.shape[1]:

    fail(
        "Feature-name count does not match "
        "model input dimension."
    )


# =============================================================================
# 7. MODEL COEFFICIENT ACCESS
# =============================================================================

print()
print("-" * 80)
print("Inspecting Logistic Regression coefficients")
print("-" * 80)


if not hasattr(
    model,
    "estimators_"
):

    fail(
        "Production model does not expose "
        "OneVsRest estimators_."
    )


estimators = model.estimators_


if len(estimators) != len(
    mitre_classes
):

    fail(
        "Estimator count does not match "
        "MITRE class count."
    )


# =============================================================================
# 8. PREDICTION + CONTRIBUTION ANALYSIS
# =============================================================================

case_records = []
contribution_records = []
tfidf_records = []
coefficient_records = []


# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def threshold_for(technique):

    value = thresholds.get(
        technique,
        0.5
    )

    try:
        return float(value)
    except Exception:
        return 0.5


# =============================================================================
# LOOP THROUGH BENIGN CASES
# =============================================================================

for row_index, case in cases_df.iterrows():

    case_id = case[
        "case_id"
    ]


    probs = probability_matrix[
        row_index
    ]


    # -------------------------------------------------------------------------
    # ranked predictions
    # -------------------------------------------------------------------------

    ranking = np.argsort(
        probs
    )[::-1]


    top_index = int(
        ranking[0]
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

    top_threshold = (
        threshold_for(
            top_technique
        )
    )


    triggered = []

    for idx in ranking:

        technique = (
            mitre_classes[
                int(idx)
            ]
        )

        probability = float(
            probs[
                int(idx)
            ]
        )

        threshold = (
            threshold_for(
                technique
            )
        )

        if probability >= threshold:

            triggered.append(
                {
                    "technique":
                        technique,

                    "probability":
                        probability,

                    "threshold":
                        threshold,
                }
            )


    # -------------------------------------------------------------------------
    # top predicted class coefficient
    # -------------------------------------------------------------------------

    estimator = estimators[
        top_index
    ]


    coefficients = np.asarray(
        estimator.coef_
    ).reshape(-1)


    # -------------------------------------------------------------------------
    # dense final row
    # -------------------------------------------------------------------------

    feature_vector = (
        X_final[
            row_index
        ]
        .toarray()
        .reshape(-1)
    )


    # -------------------------------------------------------------------------
    # logistic linear contribution
    # -------------------------------------------------------------------------

    contributions = (
        feature_vector
        * coefficients
    )


    contribution_order = np.argsort(
        np.abs(
            contributions
        )
    )[::-1]


    # -------------------------------------------------------------------------
    # collect top contributions
    # -------------------------------------------------------------------------

    positive_contributions = []
    negative_contributions = []


    for feature_index in (
        contribution_order
    ):

        value = float(
            feature_vector[
                feature_index
            ]
        )

        coefficient = float(
            coefficients[
                feature_index
            ]
        )

        contribution = float(
            contributions[
                feature_index
            ]
        )


        if (
            abs(value)
            < 1e-12
        ):

            continue


        feature_name = (
            all_feature_names[
                feature_index
            ]
        )


        record = {

            "case_id":
                case_id,

            "predicted_technique":
                top_technique,

            "feature":
                feature_name,

            "feature_index":
                int(feature_index),

            "feature_value":
                value,

            "coefficient":
                coefficient,

            "linear_contribution":
                contribution,

            "direction":
                (
                    "POSITIVE"
                    if contribution > 0
                    else "NEGATIVE"
                ),

            "feature_group":
                (
                    "categorical"
                    if feature_index
                    < len(categorical_feature_names)
                    else
                    "numerical"
                    if feature_index
                    < (
                        len(categorical_feature_names)
                        + len(numeric_feature_names)
                    )
                    else
                    "tfidf"
                ),
        }


        contribution_records.append(
            record
        )


        if contribution > 0:

            positive_contributions.append(
                record
            )

        elif contribution < 0:

            negative_contributions.append(
                record
            )


    # -------------------------------------------------------------------------
    # top positive features
    # -------------------------------------------------------------------------

    positive_contributions = sorted(
        positive_contributions,
        key=lambda x:
            x[
                "linear_contribution"
            ],
        reverse=True
    )


    negative_contributions = sorted(
        negative_contributions,
        key=lambda x:
            x[
                "linear_contribution"
            ]
    )


    top_positive = (
        positive_contributions[:10]
    )

    top_negative = (
        negative_contributions[:10]
    )


    # -------------------------------------------------------------------------
    # TF-IDF-only contributions
    # -------------------------------------------------------------------------

    tfidf_start = (
        len(categorical_feature_names)
        + len(numeric_feature_names)
    )


    tfidf_values = (
        X_text[
            row_index
        ]
        .toarray()
        .reshape(-1)
    )


    tfidf_coefficients = (
        coefficients[
            tfidf_start:
        ]
    )


    tfidf_contributions = (
        tfidf_values
        * tfidf_coefficients
    )


    tfidf_order = np.argsort(
        np.abs(
            tfidf_contributions
        )
    )[::-1]


    for tfidf_index in tfidf_order:

        value = float(
            tfidf_values[
                tfidf_index
            ]
        )

        contribution = float(
            tfidf_contributions[
                tfidf_index
            ]
        )


        if abs(value) < 1e-12:
            continue


        term = (
            tfidf_feature_names[
                tfidf_index
            ]
        )


        tfidf_records.append(
            {
                "case_id":
                    case_id,

                "predicted_technique":
                    top_technique,

                "term":
                    term,

                "tfidf_value":
                    value,

                "coefficient":
                    float(
                        tfidf_coefficients[
                            tfidf_index
                        ]
                    ),

                "contribution":
                    contribution,

                "direction":
                    (
                        "POSITIVE"
                        if contribution > 0
                        else "NEGATIVE"
                    ),
            }
        )


    # -------------------------------------------------------------------------
    # case-level record
    # -------------------------------------------------------------------------

    case_records.append(
        {
            "case_id":
                case_id,

            "process_name":
                case[
                    "process_name"
                ],

            "parent_process_name":
                case[
                    "parent_process_name"
                ],

            "command_line":
                case.get(
                    "behavior_text_normalized",
                    ""
                ),

            "top_technique":
                top_technique,

            "top_probability":
                top_probability,

            "top_threshold":
                top_threshold,

            "top_triggered":
                int(
                    top_probability
                    >= top_threshold
                ),

            "triggered_technique_count":
                len(triggered),

            "triggered_techniques":
                ";".join(
                    x["technique"]
                    for x in triggered
                ),

            "top_positive_features":
                "; ".join(
                    f"{x['feature']}="
                    f"{x['linear_contribution']:.6f}"
                    for x
                    in top_positive
                ),

            "top_negative_features":
                "; ".join(
                    f"{x['feature']}="
                    f"{x['linear_contribution']:.6f}"
                    for x
                    in top_negative
                ),

            "top_positive_tfidf":
                "; ".join(
                    f"{x['term']}="
                    f"{x['contribution']:.6f}"
                    for x
                    in sorted(
                        [
                            r
                            for r
                            in tfidf_records
                            if r[
                                "case_id"
                            ] == case_id
                        ],
                        key=lambda x:
                            x[
                                "contribution"
                            ],
                        reverse=True
                    )[:10]
                ),
        }
    )


# =============================================================================
# 9. GLOBAL COEFFICIENT ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing model coefficient strength")
print("-" * 80)


for class_index, technique in enumerate(
    mitre_classes
):

    estimator = estimators[
        class_index
    ]


    coefficients = np.asarray(
        estimator.coef_
    ).reshape(-1)


    top_positive_indices = np.argsort(
        coefficients
    )[::-1][:20]


    top_negative_indices = np.argsort(
        coefficients
    )[:20]


    for feature_index in (
        top_positive_indices
    ):

        coefficient_records.append(
            {
                "technique":
                    technique,

                "direction":
                    "POSITIVE",

                "feature":
                    all_feature_names[
                        feature_index
                    ],

                "feature_index":
                    int(feature_index),

                "coefficient":
                    float(
                        coefficients[
                            feature_index
                        ]
                    ),
            }
        )


    for feature_index in (
        top_negative_indices
    ):

        coefficient_records.append(
            {
                "technique":
                    technique,

                "direction":
                    "NEGATIVE",

                "feature":
                    all_feature_names[
                        feature_index
                    ],

                "feature_index":
                    int(feature_index),

                "coefficient":
                    float(
                        coefficients[
                            feature_index
                        ]
                    ),
            }
        )


# =============================================================================
# 10. FALSE-POSITIVE FEATURE FREQUENCY
# =============================================================================

print()
print("-" * 80)
print("Calculating recurring false-positive features")
print("-" * 80)


contribution_df = pd.DataFrame(
    contribution_records
)


if len(
    contribution_df
) > 0:

    positive_df = contribution_df[
        contribution_df[
            "direction"
        ] == "POSITIVE"
    ]


    feature_summary = (
        positive_df
        .groupby(
            [
                "feature",
                "feature_group",
            ],
            as_index=False
        )
        .agg(
            positive_case_count=(
                "case_id",
                "nunique"
            ),

            total_positive_contribution=(
                "linear_contribution",
                "sum"
            ),

            mean_positive_contribution=(
                "linear_contribution",
                "mean"
            ),

            max_positive_contribution=(
                "linear_contribution",
                "max"
            ),
        )
        .sort_values(
            [
                "positive_case_count",
                "total_positive_contribution",
            ],
            ascending=False
        )
    )

else:

    feature_summary = pd.DataFrame()


# =============================================================================
# 11. PRINT CASE RESULTS
# =============================================================================

print()
print("=" * 80)
print("BENIGN FEATURE ATTRIBUTION RESULTS")
print("=" * 80)


for row in case_records:

    print()
    print(
        f"{row['case_id']}"
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
        f"  Triggered      : "
        f"{bool(row['top_triggered'])}"
    )

    print(
        "  Positive features:"
    )

    positive_text = (
        row[
            "top_positive_features"
        ]
    )

    if positive_text:

        for item in positive_text.split(
            "; "
        )[:5]:

            print(
                f"    + {item}"
            )

    else:

        print(
            "    None"
        )


    print(
        "  Positive TF-IDF terms:"
    )

    tfidf_text = (
        row[
            "top_positive_tfidf"
        ]
    )

    if tfidf_text:

        for item in tfidf_text.split(
            "; "
        )[:5]:

            print(
                f"    + {item}"
            )

    else:

        print(
            "    None"
        )


# =============================================================================
# 12. GLOBAL FEATURE SUMMARY
# =============================================================================

print()
print("-" * 80)
print("MOST RECURRING POSITIVE FEATURES")
print("-" * 80)


if len(
    feature_summary
) > 0:

    for _, row in (
        feature_summary
        .head(30)
        .iterrows()
    ):

        print(
            f"{row['feature']:<45} "
            f"cases={int(row['positive_case_count']):<3} "
            f"total={row['total_positive_contribution']:.6f}"
        )

else:

    print(
        "No positive feature contributions found."
    )


# =============================================================================
# 13. SAVE REPORTS
# =============================================================================

print()
print("-" * 80)
print("Saving diagnostic reports")
print("-" * 80)


case_df = pd.DataFrame(
    case_records
)

case_df.to_csv(
    CASE_FILE,
    index=False
)


contribution_df.to_csv(
    CONTRIBUTION_FILE,
    index=False
)


tfidf_df = pd.DataFrame(
    tfidf_records
)

tfidf_df.to_csv(
    TFIDF_FILE,
    index=False
)


coefficient_df = pd.DataFrame(
    coefficient_records
)

coefficient_df.to_csv(
    COEFFICIENT_FILE,
    index=False
)


feature_summary.to_csv(
    FEATURE_SUMMARY_FILE,
    index=False
)


# =============================================================================
# 14. BUILD JSON SUMMARY
# =============================================================================

false_positive_cases = [
    row
    for row
    in case_records
    if row[
        "top_triggered"
    ] == 1
]


recurring_features = []

if len(
    feature_summary
) > 0:

    for _, row in (
        feature_summary
        .head(30)
        .iterrows()
    ):

        recurring_features.append(
            {
                "feature":
                    row["feature"],

                "feature_group":
                    row[
                        "feature_group"
                    ],

                "positive_case_count":
                    int(
                        row[
                            "positive_case_count"
                        ]
                    ),

                "total_positive_contribution":
                    float(
                        row[
                            "total_positive_contribution"
                        ]
                    ),

                "mean_positive_contribution":
                    float(
                        row[
                            "mean_positive_contribution"
                        ]
                    ),

                "max_positive_contribution":
                    float(
                        row[
                            "max_positive_contribution"
                        ]
                    ),
            }
        )


report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3C",

    "gate_name":
        "Feature Attribution / False-Positive Root-Cause Analysis",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "read_only":
        True,

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

    "model": {

        "type":
            type(model).__name__,

        "mitre_classes":
            len(mitre_classes),

        "categorical_features":
            len(categorical_feature_names),

        "numerical_features":
            len(numeric_feature_names),

        "tfidf_features":
            len(tfidf_feature_names),

        "final_features":
            len(all_feature_names),
    },

    "controlled_benign": {

        "total_cases":
            len(case_records),

        "triggered_cases":
            len(false_positive_cases),

        "trigger_rate":
            (
                len(false_positive_cases)
                / len(case_records)
                if case_records
                else 0.0
            ),
    },

    "recurring_positive_features":
        recurring_features,

    "case_results":
        case_records,

    "interpretation_guidance": {

        "positive_structured_feature":
            "A positive contribution means the feature "
            "pushes the selected Logistic Regression "
            "classifier toward the predicted technique.",

        "positive_tfidf_term":
            "A positive TF-IDF contribution means the "
            "observed behavior-text term pushes the "
            "selected classifier toward the predicted technique.",

        "repeated_feature":
            "A feature contributing positively across "
            "many benign cases is a candidate root cause, "
            "but should not be removed without checking "
            "its attack-side importance.",

        "threshold_warning":
            "This gate does not recommend changing thresholds. "
            "Threshold changes should be evaluated only after "
            "feature and dataset semantics are understood.",
    },

    "reports": {

        "case_attribution":
            str(CASE_FILE),

        "feature_contributions":
            str(CONTRIBUTION_FILE),

        "tfidf_contributions":
            str(TFIDF_FILE),

        "model_coefficients":
            str(COEFFICIENT_FILE),

        "feature_summary":
            str(FEATURE_SUMMARY_FILE),
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
# FINAL
# =============================================================================

print()
print("=" * 80)
print(
    "PHASE 2 GATE 2.3C STATUS: "
    "DIAGNOSTIC_COMPLETE"
)
print("=" * 80)

print()
print(
    f"JSON report       : {REPORT_FILE}"
)

print(
    f"Case attribution  : {CASE_FILE}"
)

print(
    f"Contributions     : {CONTRIBUTION_FILE}"
)

print(
    f"TF-IDF analysis   : {TFIDF_FILE}"
)

print(
    f"Coefficients      : {COEFFICIENT_FILE}"
)

print(
    f"Feature summary   : {FEATURE_SUMMARY_FILE}"
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