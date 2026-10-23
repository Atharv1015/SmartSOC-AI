import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

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
    / "phase2_gate2_4b"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

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
    encoding="utf-8",
) as f:
    SCHEMA = json.load(f)

with open(
    MODEL_DIR / "thresholds.json",
    "r",
    encoding="utf-8",
) as f:
    THRESHOLDS = json.load(f)["thresholds"]


CATEGORICAL_COLUMNS = (
    SCHEMA["categorical_columns"]
)

NUMERICAL_COLUMNS = (
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


# ============================================================
# LOAD DATA
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
# IMPORTANT FEATURE COLUMNS
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
# CONTROLLED CASES
# ============================================================

CONTROLLED_CASES = [

    {
        "case_id": "MALICIOUS_001",
        "expected": [
            "T1027.004",
            "T1059.001",
        ],
        "description":
            "Encoded PowerShell execution",
        "features": {
            "uses_powershell": 1,
            "uses_cmd": 0,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 1,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 1,
            "has_target_filename": 0,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 1,
            "strong_behavior_signal": 1,
        },
    },

    {
        "case_id": "MALICIOUS_002",
        "expected": [
            "T1059.003",
        ],
        "description":
            "Command shell discovery",
        "features": {
            "uses_powershell": 0,
            "uses_cmd": 1,
            "uses_net": 1,
            "uses_whoami": 1,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 0,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 0,
            "strong_behavior_signal": 0,
        },
    },

    {
        "case_id": "MALICIOUS_003",
        "expected": [
            "T1105",
        ],
        "description":
            "PowerShell payload download",
        "features": {
            "uses_powershell": 1,
            "uses_cmd": 0,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 1,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 1,
            "strong_behavior_signal": 1,
        },
    },

    {
        "case_id": "MALICIOUS_004",
        "expected": [
            "T1059.001",
            "T1112",
        ],
        "description":
            "Registry Run-key modification",
        "features": {
            "uses_powershell": 1,
            "uses_cmd": 0,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 1,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 1,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 1,
            "administrative_powershell_context": 1,
            "strong_behavior_signal": 1,
        },
    },

    {
        "case_id": "MALICIOUS_005",
        "expected": [
            "T1543.003",
        ],
        "description":
            "Windows service creation",
        "features": {
            "uses_powershell": 0,
            "uses_cmd": 1,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 1,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 1,
            "administrative_powershell_context": 0,
            "strong_behavior_signal": 0,
        },
    },

    {
        "case_id": "MALICIOUS_006",
        "expected": [
            "T1059.001",
            "T1070.004",
        ],
        "description":
            "PowerShell file deletion",
        "features": {
            "uses_powershell": 1,
            "uses_cmd": 0,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 1,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 1,
            "strong_behavior_signal": 1,
        },
    },

    {
        "case_id": "MALICIOUS_007",
        "expected": [
            "T1562.001",
        ],
        "description":
            "Security-tool disabling",
        "features": {
            "uses_powershell": 1,
            "uses_cmd": 0,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 0,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 1,
            "strong_behavior_signal": 0,
        },
    },

    {
        "case_id": "MALICIOUS_008",
        "expected": [
            "T1059.003",
            "T1070.004",
        ],
        "description":
            "Windows event log clearing",
        "features": {
            "uses_powershell": 0,
            "uses_cmd": 1,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 1,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 0,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 0,
            "strong_behavior_signal": 0,
        },
    },

    {
        "case_id": "MALICIOUS_009",
        "expected": [
            "T1059.003",
        ],
        "description":
            "Certutil download",
        "features": {
            "uses_powershell": 0,
            "uses_cmd": 1,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 1,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 1,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 0,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 0,
            "strong_behavior_signal": 1,
        },
    },

    {
        "case_id": "MALICIOUS_010",
        "expected": [
            "T1059.003",
            "T1485",
        ],
        "description":
            "Destructive deletion",
        "features": {
            "uses_powershell": 0,
            "uses_cmd": 1,
            "uses_net": 0,
            "uses_whoami": 0,
            "uses_schtasks": 0,
            "uses_reg": 0,
            "uses_rundll32": 0,
            "uses_wmic": 0,
            "uses_certutil": 0,
            "uses_bitsadmin": 0,
            "uses_ipconfig": 0,
            "uses_netstat": 0,
            "uses_tasklist": 0,
            "uses_sc": 0,
            "uses_mshta": 0,
            "uses_regsvr32": 0,
            "uses_cscript": 0,
            "uses_wscript": 0,
            "uses_wevtutil": 0,
            "uses_vssadmin": 0,
            "has_encoded_command": 0,
            "has_download_behavior": 0,
            "has_remote_execution_hint": 0,
            "has_registry_operation": 0,
            "has_file_deletion": 1,
            "has_obfuscation_hint": 0,
            "has_target_filename": 1,
            "has_image_loaded": 0,
            "has_target_object": 0,
            "administrative_powershell_context": 0,
            "strong_behavior_signal": 1,
        },
    },
]


# ============================================================
# LABEL PARSER
# ============================================================

def parse_labels(value):

    if pd.isna(value):
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


# ============================================================
# TRAINING PROFILE
# ============================================================

def get_training_profile(
    technique
):

    mask = LABELS_TRAIN[
        "techniques"
    ].apply(
        lambda x:
        technique in parse_labels(x)
    )

    rows = STRUCTURED_TRAIN.loc[
        mask
    ]

    if len(rows) == 0:
        return None

    profile = {
        "technique": technique,
        "rows": int(len(rows)),
    }

    for column in BEHAVIOR_FEATURES:

        if column not in rows.columns:
            continue

        profile[column] = float(
            pd.to_numeric(
                rows[column],
                errors="coerce",
            )
            .fillna(0)
            .mean()
        )

    return profile


# ============================================================
# COMPARE CONTROLLED VS TRAINING
# ============================================================

def compare_case(
    case
):

    rows = []

    controlled = case[
        "features"
    ]

    for technique in case[
        "expected"
    ]:

        profile = get_training_profile(
            technique
        )

        if profile is None:

            rows.append({
                "case_id":
                    case["case_id"],
                "description":
                    case["description"],
                "technique":
                    technique,
                "training_rows":
                    0,
                "feature":
                    "NO_TRAINING_EXAMPLES",
                "controlled_value":
                    np.nan,
                "training_mean":
                    np.nan,
                "absolute_difference":
                    np.nan,
                "status":
                    "NO_TRAINING_SUPPORT",
            })

            continue

        for feature in BEHAVIOR_FEATURES:

            controlled_value = float(
                controlled.get(
                    feature,
                    0,
                )
            )

            training_mean = float(
                profile.get(
                    feature,
                    0,
                )
            )

            difference = abs(
                controlled_value
                - training_mean
            )

            if (
                controlled_value
                != training_mean
            ):

                status = (
                    "MISMATCH"
                )

            else:

                status = (
                    "MATCH"
                )

            rows.append({

                "case_id":
                    case["case_id"],

                "description":
                    case["description"],

                "technique":
                    technique,

                "training_rows":
                    profile["rows"],

                "feature":
                    feature,

                "controlled_value":
                    controlled_value,

                "training_mean":
                    training_mean,

                "absolute_difference":
                    difference,

                "status":
                    status,
            })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("SMARTSOC PHASE 2 GATE 2.4B")
    print("MALICIOUS TRAINING-VS-CONTROLLED FEATURE PARITY")
    print("=" * 78)

    all_rows = []

    for case in CONTROLLED_CASES:

        rows = compare_case(
            case
        )

        all_rows.extend(
            rows
        )

    result_df = pd.DataFrame(
        all_rows
    )

    result_path = (
        REPORT_DIR
        / "malicious_feature_parity.csv"
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for case in CONTROLLED_CASES:

        case_rows = result_df[
            result_df["case_id"]
            == case["case_id"]
        ]

        for technique in case[
            "expected"
        ]:

            subset = case_rows[
                case_rows["technique"]
                == technique
            ]

            if subset.empty:
                continue

            mismatches = subset[
                subset["status"]
                == "MISMATCH"
            ]

            summary_rows.append({

                "case_id":
                    case["case_id"],

                "description":
                    case["description"],

                "technique":
                    technique,

                "training_rows":
                    int(
                        subset[
                            "training_rows"
                        ].iloc[0]
                    ),

                "features_compared":
                    int(len(subset)),

                "mismatched_features":
                    int(len(mismatches)),

                "mismatch_rate":
                    float(
                        len(mismatches)
                        / len(subset)
                    ),

                "status":
                    (
                        "NO_TRAINING_SUPPORT"
                        if subset[
                            "training_rows"
                        ].iloc[0] == 0
                        else
                        "HIGH_MISMATCH"
                        if len(mismatches)
                        >= len(subset) * 0.5
                        else
                        "PARTIAL_MISMATCH"
                        if len(mismatches) > 0
                        else
                        "GOOD_PARITY"
                    ),
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        REPORT_DIR
        / "malicious_feature_parity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print(
        "FEATURE PARITY SUMMARY"
    )
    print("=" * 78)

    for _, row in summary_df.iterrows():

        print(
            f"{row['case_id']:<15}"
            f"{row['technique']:<12}"
            f"train={int(row['training_rows']):<6}"
            f"mismatch="
            f"{int(row['mismatched_features']):<3}/"
            f"{int(row['features_compared']):<3}"
            f"rate="
            f"{row['mismatch_rate']:.2%}"
            f"  {row['status']}"
        )

    # --------------------------------------------------------
    # Important mismatches
    # --------------------------------------------------------

    print()
    print(
        "IMPORTANT FEATURE MISMATCHES"
    )
    print("=" * 78)

    important = result_df[
        (
            result_df["status"]
            == "MISMATCH"
        )
        &
        (
            result_df["absolute_difference"]
            >= 0.25
        )
    ].copy()

    important = important.sort_values(
        [
            "case_id",
            "technique",
            "absolute_difference",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    for _, row in important.iterrows():

        print(
            f"{row['case_id']:<15}"
            f"{row['technique']:<12}"
            f"{row['feature']:<35}"
            f"controlled="
            f"{row['controlled_value']:.3f}"
            f"  train_mean="
            f"{row['training_mean']:.3f}"
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "phase":
            "Phase 2",

        "gate":
            "Gate 2.4B",

        "gate_name":
            "Malicious Training-vs-Controlled Feature Parity",

        "status":
            "DIAGNOSTIC_COMPLETE",

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
            len(CONTROLLED_CASES),

        "features_compared":
            len(BEHAVIOR_FEATURES),

        "result_file":
            str(result_path),

        "summary_file":
            str(summary_path),
    }

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4b_report.json"
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
        )

    print()
    print("=" * 78)
    print("GATE 2.4B COMPLETE")
    print("=" * 78)

    print(
        f"Results : {result_path}"
    )

    print(
        f"Summary : {summary_path}"
    )

    print(
        f"Report  : {report_path}"
    )

    print()
    print(
        "STATUS: DIAGNOSTIC_COMPLETE"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()