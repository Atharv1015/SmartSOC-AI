"""
SmartSOC Windows-APT Model B v4.1
Aligned Production Inference

Purpose:
    Reproduce the v4.1 training-time feature generation as closely as
    possible for normalized Wazuh events without modifying production
    model artifacts.

Production feature layout:
    categorical encoded : 162
    numerical           : 58
    TF-IDF              : 6699
    final               : 6919
"""

from pathlib import Path
from typing import Any, Dict, List

import json
import re
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

# Fallback for the actual project directory naming observed in your output.
if not MODEL_DIR.exists():
    MODEL_DIR = (
        PROJECT_ROOT
        / "models"
        / "windows"
        / "apt"
        / "v4"
        / "1"
        / "production"
    )

# If the above does not exist, use the known project-relative location.
if not MODEL_DIR.exists():
    MODEL_DIR = Path(
        r".\models\windows_apt_v4_1\production"
    )


MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
MANIFEST_FILE = MODEL_DIR / "model_manifest.json"


# ============================================================================
# EXPECTED PRODUCTION DIMENSIONS
# ============================================================================

EXPECTED_CATEGORICAL_INPUT = 4
EXPECTED_CATEGORICAL_ENCODED = 162
EXPECTED_NUMERICAL = 58
EXPECTED_TFIDF = 6699
EXPECTED_FEATURES = 6919
EXPECTED_TECHNIQUES = 45


CATEGORICAL_FEATURES = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]


NUMERICAL_FEATURES = [
    "event_id",
    "system_level",
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
    "command_length",
    "parent_command_length",
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
    "administrative_powershell_context",
    "strong_behavior_signal",
]


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def header(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text: str) -> None:
    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required production artifact does not exist:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================================
# SAFE CONVERSION
# ============================================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def presence(value: Any) -> int:
    return int(bool(safe_string(value)))


def contains(value: Any, pattern: str) -> int:
    text = safe_string(value)

    if not text:
        return 0

    return int(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


# ============================================================================
# FILE NAME / EXTENSION
# ============================================================================

def filename(path: Any) -> str:
    value = safe_string(path)

    if not value:
        return ""

    value = value.replace("/", "\\")

    return value.split("\\")[-1].lower()


def extension(name: Any) -> str:
    value = safe_string(name)

    if "." not in value:
        return ""

    return value.rsplit(".", 1)[-1].lower()


# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean(value: Any) -> str:
    return safe_string(value)


def sanitize_text(value: Any) -> str:
    """
    Match the supplied v3 preparation normalization.

    Order is important:
        1. remove hashes
        2. replace IPs
        3. replace IPv6-ish values
        4. collapse Windows/UNC paths to filenames
        5. replace URLs
        6. replace long numbers
        7. normalize separators
        8. remove unsupported characters
        9. collapse whitespace
    """

    value = clean(value).lower()

    if not value:
        return ""

    # Hashes
    value = re.sub(
        r"\b(?:md5|sha1|sha256|imphash)=[a-f0-9]{8,}\b",
        " <HASH> ",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\b[a-f0-9]{32,64}\b",
        " <HASH> ",
        value,
    )

    # IPv4
    value = re.sub(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        " <IP> ",
        value,
    )

    # IPv6-ish
    value = re.sub(
        r"\b[0-9a-f]{0,4}(?::[0-9a-f]{1,4}){2,7}\b",
        " <IP6> ",
        value,
        flags=re.IGNORECASE,
    )

    # Windows paths
    value = re.sub(
        r"[a-z]:\\(?:[^\\\s]+\\)*([^\\\s]+)",
        r" \1 ",
        value,
        flags=re.IGNORECASE,
    )

    # UNC paths
    value = re.sub(
        r"\\\\[^\\\s]+\\(?:[^\\\s]+\\)*([^\\\s]+)",
        r" \1 ",
        value,
    )

    # URLs
    value = re.sub(
        r"https?://[^\s]+",
        " <URL> ",
        value,
    )

    # Long numbers
    value = re.sub(
        r"\b\d{4,}\b",
        " <NUM> ",
        value,
    )

    # Separators
    value = re.sub(
        r"[\\/:,;|]+",
        " ",
        value,
    )

    # Keep command characters used by training
    value = re.sub(
        r"[^a-z0-9_.<>\-+=' ]+",
        " ",
        value,
    )

    # Whitespace
    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


# ============================================================================
# WAZUH EVENT ACCESS
# ============================================================================

def recursive_flatten(
    value: Any,
    prefix: str = "",
) -> Dict[str, Any]:

    output: Dict[str, Any] = {}

    if isinstance(value, dict):

        for key, child in value.items():

            new_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.update(
                recursive_flatten(
                    child,
                    new_prefix,
                )
            )

    elif isinstance(value, list):

        for index, child in enumerate(value):

            new_prefix = (
                f"{prefix}.{index}"
                if prefix
                else str(index)
            )

            output.update(
                recursive_flatten(
                    child,
                    new_prefix,
                )
            )

    else:
        output[prefix] = value

    return output


def get_field(
    flattened: Dict[str, Any],
    *names: str,
    default: Any = "",
) -> Any:

    normalized = {
        str(key).lower(): value
        for key, value in flattened.items()
    }

    # Exact name first
    for name in names:

        key = str(name).lower()

        if key in normalized:
            return normalized[key]

    # Suffix match
    for name in names:

        suffix = str(name).lower()

        for key, value in normalized.items():

            if key.endswith(suffix):
                return value

    return default


# ============================================================================
# NORMALIZE WAZUH EVENT
# ============================================================================

def normalize_wazuh_event(
    event: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(event, dict):
        raise TypeError(
            "Wazuh event must be a dictionary."
        )

    flat = recursive_flatten(event)

    normalized: Dict[str, Any] = {}

    normalized["timestamp"] = get_field(
        flat,
        "timestamp",
        "_source.timestamp",
        default="",
    )

    normalized["event_id"] = get_field(
        flat,
        "event_id",
        "eventid",
        "eventID",
        "_source.data.win.system.eventID",
        default="0",
    )

    normalized["rule_id"] = get_field(
        flat,
        "rule_id",
        "rule.id",
        "_source.rule.id",
        default="",
    )

    normalized["rule_level"] = get_field(
        flat,
        "rule_level",
        "rule.level",
        "_source.rule.level",
        "_source.rule.level",
        default="0",
    )

    normalized["provider"] = get_field(
        flat,
        "provider",
        "providerName",
        "_source.data.win.system.providerName",
        default="",
    )

    normalized["channel"] = get_field(
        flat,
        "channel",
        "_source.data.win.system.channel",
        default="",
    )

    normalized["system_level"] = get_field(
        flat,
        "system_level",
        "level",
        "_source.data.win.system.level",
        default="0",
    )

    normalized["image"] = get_field(
        flat,
        "image",
        "_source.data.win.eventdata.image",
        default="",
    )

    normalized["parent_image"] = get_field(
        flat,
        "parent_image",
        "parentImage",
        "_source.data.win.eventdata.parentImage",
        default="",
    )

    normalized["command_line"] = get_field(
        flat,
        "command_line",
        "commandLine",
        "_source.data.win.eventdata.commandLine",
        default="",
    )

    normalized["parent_command_line"] = get_field(
        flat,
        "parent_command_line",
        "parentCommandLine",
        "_source.data.win.eventdata.parentCommandLine",
        default="",
    )

    normalized["user"] = get_field(
        flat,
        "user",
        "_source.data.win.eventdata.user",
        default="",
    )

    normalized["parent_user"] = get_field(
        flat,
        "parent_user",
        "parentUser",
        "_source.data.win.eventdata.parentUser",
        default="",
    )

    normalized["integrity_level"] = get_field(
        flat,
        "integrity_level",
        "integrityLevel",
        "_source.data.win.eventdata.integrityLevel",
        default="",
    )

    normalized["hashes"] = get_field(
        flat,
        "hashes",
        "_source.data.win.eventdata.hashes",
        default="",
    )

    normalized["source_ip"] = get_field(
        flat,
        "source_ip",
        "sourceIp",
        "_source.data.win.eventdata.sourceIp",
        default="",
    )

    normalized["destination_ip"] = get_field(
        flat,
        "destination_ip",
        "destinationIp",
        "_source.data.win.eventdata.destinationIp",
        default="",
    )

    normalized["source_port"] = get_field(
        flat,
        "source_port",
        "sourcePort",
        "_source.data.win.eventdata.sourcePort",
        default="",
    )

    normalized["destination_port"] = get_field(
        flat,
        "destination_port",
        "destinationPort",
        "_source.data.win.eventdata.destinationPort",
        default="",
    )

    normalized["protocol"] = get_field(
        flat,
        "protocol",
        "_source.data.win.eventdata.protocol",
        default="",
    )

    normalized["target_filename"] = get_field(
        flat,
        "target_filename",
        "targetFilename",
        "_source.data.win.eventdata.targetFilename",
        default="",
    )

    normalized["image_loaded"] = get_field(
        flat,
        "image_loaded",
        "imageLoaded",
        "_source.data.win.eventdata.imageLoaded",
        default="",
    )

    normalized["target_object"] = get_field(
        flat,
        "target_object",
        "targetObject",
        "_source.data.win.eventdata.targetObject",
        default="",
    )

    normalized["original_file_name"] = get_field(
        flat,
        "original_file_name",
        "originalFileName",
        "_source.data.win.eventdata.originalFileName",
        default="",
    )

    return normalized


# ============================================================================
# DERIVE EXACT STRUCTURED FEATURES
# ============================================================================

def derive_features(
    event: Dict[str, Any],
) -> Dict[str, Any]:

    image = safe_string(
        event.get("image", "")
    )

    parent_image = safe_string(
        event.get("parent_image", "")
    )

    command_line = safe_string(
        event.get("command_line", "")
    )

    parent_command = safe_string(
        event.get("parent_command_line", "")
    )

    user = safe_string(
        event.get("user", "")
    ).lower()

    parent_user = safe_string(
        event.get("parent_user", "")
    ).lower()

    integrity = safe_string(
        event.get("integrity_level", "")
    ).lower()

    provider = safe_string(
        event.get("provider", "")
    ).lower()

    channel = safe_string(
        event.get("channel", "")
    ).lower()

    hashes = safe_string(
        event.get("hashes", "")
    ).lower()

    source_ip = safe_string(
        event.get("source_ip", "")
    )

    destination_ip = safe_string(
        event.get("destination_ip", "")
    )

    source_port = safe_string(
        event.get("source_port", "")
    )

    destination_port = safe_string(
        event.get("destination_port", "")
    )

    protocol = safe_string(
        event.get("protocol", "")
    )

    target_filename = safe_string(
        event.get("target_filename", "")
    )

    image_loaded = safe_string(
        event.get("image_loaded", "")
    )

    target_object = safe_string(
        event.get("target_object", "")
    )

    process_name = filename(image)
    parent_process_name = filename(parent_image)

    process_ext = extension(process_name)
    parent_ext = extension(parent_process_name)

    cmd = command_line.lower()
    parent_cmd = parent_command.lower()

    combined_cmd = (
        cmd
        + " "
        + parent_cmd
    )

    features: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # Event
    # ------------------------------------------------------------------------

    features["event_id"] = safe_float(
        event.get("event_id", 0)
    )

    features["system_level"] = safe_float(
        event.get("system_level", 0)
    )

    features["is_sysmon"] = contains(
        provider,
        r"sysmon",
    )

    features["is_security_channel"] = contains(
        channel,
        r"security",
    )

    # ------------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------------

    features["process_name"] = process_name
    features["parent_process_name"] = parent_process_name
    features["process_extension"] = process_ext
    features["parent_extension"] = parent_ext

    features["has_process_image"] = presence(image)
    features["has_parent_image"] = presence(parent_image)

    # ------------------------------------------------------------------------
    # User / privilege
    # ------------------------------------------------------------------------

    features["has_user"] = presence(user)
    features["has_parent_user"] = presence(parent_user)

    features["is_system_user"] = contains(
        user,
        r"nt authority\\system|^system$",
    )

    features["is_admin_user"] = contains(
        user,
        r"administrator|admin",
    )

    features["integrity_high"] = contains(
        integrity,
        r"high|system",
    )

    features["integrity_medium"] = contains(
        integrity,
        r"medium",
    )

    # ------------------------------------------------------------------------
    # Command statistics
    # ------------------------------------------------------------------------

    features["command_length"] = len(cmd)
    features["parent_command_length"] = len(parent_cmd)

    features["has_command_line"] = presence(cmd)
    features["has_parent_command"] = presence(parent_cmd)

    # ------------------------------------------------------------------------
    # Tool indicators
    # ------------------------------------------------------------------------

    tool_patterns = {
        "uses_powershell":
            r"\bpowershell(?:\.exe)?\b|\bpwsh(?:\.exe)?\b",

        "uses_cmd":
            r"\bcmd(?:\.exe)?\b",

        "uses_net":
            r"\bnet(?:\.exe)?\b|\bnet1(?:\.exe)?\b",

        "uses_whoami":
            r"\bwhoami(?:\.exe)?\b",

        "uses_schtasks":
            r"\bschtasks(?:\.exe)?\b",

        "uses_reg":
            r"\breg(?:\.exe)?\b",

        "uses_rundll32":
            r"\brundll32(?:\.exe)?\b",

        "uses_wmic":
            r"\bwmic(?:\.exe)?\b",

        "uses_certutil":
            r"\bcertutil(?:\.exe)?\b",

        "uses_bitsadmin":
            r"\bbitsadmin(?:\.exe)?\b",

        "uses_ipconfig":
            r"\bipconfig(?:\.exe)?\b",

        "uses_netstat":
            r"\bnetstat(?:\.exe)?\b",

        "uses_tasklist":
            r"\btasklist(?:\.exe)?\b",

        "uses_sc":
            r"\bsc(?:\.exe)?\b",

        "uses_mshta":
            r"\bmshta(?:\.exe)?\b",

        "uses_regsvr32":
            r"\bregsvr32(?:\.exe)?\b",

        "uses_cscript":
            r"\bcscript(?:\.exe)?\b",

        "uses_wscript":
            r"\bwscript(?:\.exe)?\b",

        "uses_wevtutil":
            r"\bwevtutil(?:\.exe)?\b",

        "uses_vssadmin":
            r"\bvssadmin(?:\.exe)?\b",
    }

    for feature_name, pattern in tool_patterns.items():

        features[feature_name] = contains(
            combined_cmd,
            pattern,
        )

    # ------------------------------------------------------------------------
    # Behavioral indicators
    # ------------------------------------------------------------------------

    features["has_encoded_command"] = contains(
        combined_cmd,
        r"-enc(?:odedcommand)?\b",
    )

    features["has_download_behavior"] = contains(
        combined_cmd,
        r"invoke-webrequest|invoke-restmethod|download|curl|wget|bitsadmin",
    )

    features["has_remote_execution_hint"] = contains(
        combined_cmd,
        r"psexec|winrm|wmic|invoke-command|enter-pssession",
    )

    features["has_registry_operation"] = contains(
        combined_cmd,
        r"\breg(?:\.exe)?\b|hkey_|currentversion\\run",
    )

    features["has_file_deletion"] = contains(
        combined_cmd,
        r"\bdel\b|\berase\b|remove-item|rmdir|\brd\b",
    )

    features["has_obfuscation_hint"] = contains(
        combined_cmd,
        r"base64|frombase64string|encodedcommand",
    )

    # ------------------------------------------------------------------------
    # Process relationship
    # ------------------------------------------------------------------------

    features["same_process_parent"] = int(
        process_name == parent_process_name
        and process_name != ""
    )

    features["suspicious_parent_wazuh"] = contains(
        parent_process_name,
        r"wazuh-agent|ossec-agent",
    )

    # ------------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------------

    features["has_hash"] = presence(hashes)

    features["has_sha256"] = contains(
        hashes,
        r"sha256=",
    )

    features["has_md5"] = contains(
        hashes,
        r"md5=",
    )

    # ------------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------------

    features["has_source_ip"] = presence(source_ip)
    features["has_destination_ip"] = presence(destination_ip)
    features["has_source_port"] = presence(source_port)
    features["has_destination_port"] = presence(destination_port)
    features["has_protocol"] = presence(protocol)

    features["is_network_event"] = int(
        (
            features["has_source_ip"]
            + features["has_destination_ip"]
            + features["has_destination_port"]
        )
        > 0
    )

    # ------------------------------------------------------------------------
    # Other telemetry
    # ------------------------------------------------------------------------

    features["has_target_filename"] = presence(
        target_filename
    )

    features["has_image_loaded"] = presence(
        image_loaded
    )

    features["has_target_object"] = presence(
        target_object
    )

    # ------------------------------------------------------------------------
    # Administrative PowerShell context
    #
    # This follows the supplied preparation logic.
    # ------------------------------------------------------------------------

    admin_commands = [
        "get-service",
        "get-process",
        "get-childitem",
        "get-item",
        "get-content",
        "get-command",
        "get-help",
        "get-computerinfo",
        "get-wmiobject",
        "get-ciminstance",
        "get-eventlog",
        "get-hotfix",
        "get-netadapter",
        "get-netipconfiguration",
        "get-nettcpconnection",
    ]

    administrative_context = int(
        features["uses_powershell"] == 1
        and any(
            command in combined_cmd
            for command in admin_commands
        )
    )

    features["administrative_powershell_context"] = (
        administrative_context
    )

    # ------------------------------------------------------------------------
    # Strong behavioral signal
    #
    # IMPORTANT:
    # The supplied v3 preparation pipeline defines this as:
    # any one of the attack_signal_columns being present.
    # ------------------------------------------------------------------------

    attack_signal_columns = [
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "uses_certutil",
        "uses_bitsadmin",
        "uses_mshta",
        "uses_rundll32",
        "uses_regsvr32",
        "uses_vssadmin",
    ]

    features["strong_behavior_signal"] = int(
        any(
            features.get(
                column,
                0,
            )
            == 1
            for column in attack_signal_columns
        )
    )

    # ------------------------------------------------------------------------
    # Final schema check
    # ------------------------------------------------------------------------

    missing = [
        feature
        for feature in NUMERICAL_FEATURES
        if feature not in features
    ]

    if missing:
        raise ValueError(
            "Missing numerical features: "
            + ", ".join(missing)
        )

    return features


# ============================================================================
# BEHAVIOR TEXT
# ============================================================================

def build_behavior_text(
    event: Dict[str, Any],
    features: Dict[str, Any],
) -> str:

    text_columns = [
        event.get("command_line", ""),
        event.get("parent_command_line", ""),
        event.get("image", ""),
        event.get("parent_image", ""),
        event.get("original_file_name", ""),
        event.get("target_filename", ""),
        event.get("image_loaded", ""),
        event.get("target_object", ""),
    ]

    behavior_text = ""

    for value in text_columns:

        sanitized = sanitize_text(value)

        if sanitized:

            behavior_text += (
                " "
                + sanitized
            )

    behavior_text = re.sub(
        r"\s+",
        " ",
        behavior_text,
    ).strip()

    return behavior_text


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts() -> Dict[str, Any]:

    header("Checking v4.1 production artifacts")

    required = [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
        MANIFEST_FILE,
    ]

    for path in required:
        require_file(path)

    section("Loading production artifacts")

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    with open(
        THRESHOLDS_FILE,
        "r",
        encoding="utf-8",
    ) as handle:
        thresholds = json.load(handle)

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    print()
    print(
        "Model type:",
        type(model).__name__,
    )

    print(
        "TF-IDF vocabulary:",
        len(tfidf.vocabulary_),
    )

    print(
        "Scaler features:",
        getattr(
            scaler,
            "n_features_in_",
            None,
        ),
    )

    print(
        "Expected features:",
        getattr(
            model,
            "n_features_in_",
            None,
        ),
    )

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "schema": schema,
        "thresholds": thresholds,
        "manifest": manifest,
    }


# ============================================================================
# PRODUCTION SCHEMA VALIDATION
# ============================================================================

def validate_artifacts(
    artifacts: Dict[str, Any],
) -> None:

    header("Validating v4.1 production schema")

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]
    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]

    schema_cat = schema.get(
        "categorical_columns",
        [],
    )

    schema_num = schema.get(
        "numerical_columns",
        [],
    )

    schema_cat_encoded = schema.get(
        "categorical_encoded_feature_count"
    )

    schema_num_count = schema.get(
        "numerical_feature_count"
    )

    schema_tfidf = schema.get(
        "tfidf_feature_count"
    )

    schema_final = schema.get(
        "final_feature_count"
    )

    print(
        "Categorical input :",
        len(CATEGORICAL_FEATURES),
    )

    print(
        "Categorical encoded:",
        len(encoder.get_feature_names_out()),
    )

    print(
        "Numerical          :",
        getattr(
            scaler,
            "n_features_in_",
            None,
        ),
    )

    print(
        "TF-IDF             :",
        len(tfidf.vocabulary_),
    )

    print(
        "Model features     :",
        getattr(
            model,
            "n_features_in_",
            None,
        ),
    )

    print(
        "MITRE techniques   :",
        len(label_encoder.classes_),
    )

    checks = [
        (
            "categorical input",
            len(CATEGORICAL_FEATURES),
            EXPECTED_CATEGORICAL_INPUT,
        ),
        (
            "categorical encoded",
            len(encoder.get_feature_names_out()),
            EXPECTED_CATEGORICAL_ENCODED,
        ),
        (
            "numerical",
            getattr(
                scaler,
                "n_features_in_",
                None,
            ),
            EXPECTED_NUMERICAL,
        ),
        (
            "TF-IDF",
            len(tfidf.vocabulary_),
            EXPECTED_TFIDF,
        ),
        (
            "model features",
            getattr(
                model,
                "n_features_in_",
                None,
            ),
            EXPECTED_FEATURES,
        ),
        (
            "techniques",
            len(label_encoder.classes_),
            EXPECTED_TECHNIQUES,
        ),
        (
            "schema categorical",
            len(schema_cat),
            EXPECTED_CATEGORICAL_INPUT,
        ),
        (
            "schema numerical",
            len(schema_num),
            EXPECTED_NUMERICAL,
        ),
        (
            "schema categorical encoded",
            schema_cat_encoded,
            EXPECTED_CATEGORICAL_ENCODED,
        ),
        (
            "schema numerical count",
            schema_num_count,
            EXPECTED_NUMERICAL,
        ),
        (
            "schema TF-IDF",
            schema_tfidf,
            EXPECTED_TFIDF,
        ),
        (
            "schema final",
            schema_final,
            EXPECTED_FEATURES,
        ),
    ]

    failed = []

    for name, actual, expected in checks:

        if actual != expected:
            failed.append(
                f"{name}: expected {expected}, got {actual}"
            )

    if schema_num != NUMERICAL_FEATURES:
        failed.append(
            "Schema numerical feature ordering does not match "
            "the production v4.1 schema."
        )

    if schema_cat != CATEGORICAL_FEATURES:
        failed.append(
            "Schema categorical feature ordering does not match "
            "the production v4.1 schema."
        )

    if failed:

        for item in failed:
            print("[FAIL]", item)

        raise RuntimeError(
            "Production artifact validation failed."
        )

    print()
    print(
        "[PASS] Production artifact dimensions are valid."
    )


# ============================================================================
# FEATURE MATRIX
# ============================================================================

def build_feature_matrix(
    event: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    section(
        "Building exact v4.1-aligned features"
    )

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    features = derive_features(event)

    # ------------------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------------------

    categorical_df = pd.DataFrame(
        [[
            features["process_name"],
            features["parent_process_name"],
            features["process_extension"],
            features["parent_extension"],
        ]],
        columns=CATEGORICAL_FEATURES,
    )

    categorical_encoded = encoder.transform(
        categorical_df
    )

    # ------------------------------------------------------------------------
    # Numerical
    #
    # CRITICAL:
    # Use exact production schema order.
    # ------------------------------------------------------------------------

    numerical_values = []

    for feature in NUMERICAL_FEATURES:

        numerical_values.append(
            safe_float(
                features.get(
                    feature,
                    0.0,
                )
            )
        )

    numerical_array = np.asarray(
        numerical_values,
        dtype=np.float64,
    ).reshape(
        1,
        -1,
    )

    if numerical_array.shape[1] != EXPECTED_NUMERICAL:

        raise ValueError(
            "Numerical feature count is not 58."
        )

    # The scaler was fitted without feature names.
    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            UserWarning,
        )

        numerical_scaled = scaler.transform(
            numerical_array
        )

    # ------------------------------------------------------------------------
    # Behavioral text
    # ------------------------------------------------------------------------

    behavior_text = build_behavior_text(
        event,
        features,
    )

    tfidf_features = tfidf.transform(
        [behavior_text]
    )

    # ------------------------------------------------------------------------
    # Hybrid
    # ------------------------------------------------------------------------

    categorical_sparse = csr_matrix(
        categorical_encoded,
        dtype=np.float64,
    )

    numerical_sparse = csr_matrix(
        numerical_scaled,
        dtype=np.float64,
    )

    hybrid = hstack(
        [
            categorical_sparse,
            numerical_sparse,
            tfidf_features,
        ],
        format="csr",
    )

    hybrid = csr_matrix(
        hybrid,
        dtype=np.float64,
        copy=True,
    )

    hybrid.sum_duplicates()
    hybrid.sort_indices()

    # ------------------------------------------------------------------------
    # Dimension checks
    # ------------------------------------------------------------------------

    print()
    print("Feature dimensions:")

    print(
        "  categorical encoded:",
        categorical_encoded.shape,
    )

    print(
        "  numerical raw      :",
        numerical_array.shape,
    )

    print(
        "  numerical scaled   :",
        numerical_scaled.shape,
    )

    print(
        "  TF-IDF             :",
        tfidf_features.shape,
    )

    print(
        "  final              :",
        hybrid.shape,
    )

    if categorical_encoded.shape[1] != EXPECTED_CATEGORICAL_ENCODED:

        raise ValueError(
            "Categorical encoder produced "
            f"{categorical_encoded.shape[1]} instead of "
            f"{EXPECTED_CATEGORICAL_ENCODED}."
        )

    if numerical_scaled.shape[1] != EXPECTED_NUMERICAL:

        raise ValueError(
            "Numerical scaler produced "
            f"{numerical_scaled.shape[1]} instead of "
            f"{EXPECTED_NUMERICAL}."
        )

    if tfidf_features.shape[1] != EXPECTED_TFIDF:

        raise ValueError(
            "TF-IDF produced "
            f"{tfidf_features.shape[1]} instead of "
            f"{EXPECTED_TFIDF}."
        )

    if hybrid.shape[1] != EXPECTED_FEATURES:

        raise ValueError(
            "Final feature count must be "
            f"{EXPECTED_FEATURES}, got {hybrid.shape[1]}."
        )

    return {
        "features": features,
        "categorical_df": categorical_df,
        "numerical_df": pd.DataFrame(
            [features],
            columns=NUMERICAL_FEATURES,
        ),
        "numerical_array": numerical_array,
        "numerical_scaled": numerical_scaled,
        "behavior_text": behavior_text,
        "tfidf_features": tfidf_features,
        "matrix": hybrid,
    }


# ============================================================================
# THRESHOLD LOOKUP
# ============================================================================

def get_threshold(
    thresholds: Dict[str, Any],
    technique: str,
) -> float:

    if not isinstance(thresholds, dict):
        return 0.5

    # Common threshold structures
    for container_name in [
        "thresholds",
        "technique_thresholds",
        "per_technique",
    ]:

        container = thresholds.get(
            container_name
        )

        if isinstance(container, dict):

            value = container.get(
                technique
            )

            if value is not None:
                return safe_float(
                    value,
                    0.5,
                )

    # Direct dictionary
    value = thresholds.get(
        technique
    )

    if value is not None:
        return safe_float(
            value,
            0.5,
        )

    return safe_float(
        thresholds.get(
            "default_threshold",
            0.5,
        ),
        0.5,
    )


# ============================================================================
# PREDICTION
# ============================================================================

def predict(
    feature_result: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    thresholds = artifacts["thresholds"]

    matrix = feature_result["matrix"]

    probabilities = model.predict_proba(
        matrix
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(
            1,
            -1,
        )

    if probabilities.shape[1] != EXPECTED_TECHNIQUES:

        raise ValueError(
            "Model probability output has "
            f"{probabilities.shape[1]} classes instead of "
            f"{EXPECTED_TECHNIQUES}."
        )

    techniques = [
        str(x)
        for x in label_encoder.classes_
    ]

    all_scores = []

    for index, technique in enumerate(techniques):

        confidence = float(
            probabilities[0, index]
        )

        threshold = get_threshold(
            thresholds,
            technique,
        )

        detected = (
            confidence >= threshold
        )

        all_scores.append(
            {
                "technique": technique,
                "confidence": confidence,
                "threshold": threshold,
                "detected": detected,
            }
        )

    all_scores.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    detected = [
        item
        for item in all_scores
        if item["detected"]
    ]

    return {
        "detected": detected,
        "all_scores": all_scores,
        "probabilities": probabilities,
    }


# ============================================================================
# RESULT JSON
# ============================================================================

def build_result_json(
    event: Dict[str, Any],
    feature_result: Dict[str, Any],
    prediction_result: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    features = feature_result["features"]

    manifest = artifacts["manifest"]

    detected = prediction_result[
        "detected"
    ]

    all_scores = prediction_result[
        "all_scores"
    ]

    timestamp = (
        event.get("timestamp")
        or datetime.now(
            timezone.utc
        ).isoformat()
    )

    return {
        "model_version": manifest.get(
            "version",
            "v4.1",
        ),

        "timestamp": timestamp,

        "event_id": str(
            event.get(
                "event_id",
                "",
            )
        ),

        "rule_id": str(
            event.get(
                "rule_id",
                "",
            )
        ),

        "rule_level": safe_float(
            event.get(
                "rule_level",
                0,
            )
        ),

        "process_name": features[
            "process_name"
        ],

        "parent_process_name": features[
            "parent_process_name"
        ],

        "behavior_text": feature_result[
            "behavior_text"
        ],

        "predictions": [
            {
                "technique": item[
                    "technique"
                ],
                "confidence": round(
                    item[
                        "confidence"
                    ],
                    6,
                ),
                "threshold": round(
                    item[
                        "threshold"
                    ],
                    6,
                ),
            }
            for item in detected
        ],

        "top_probability_scores": [
            {
                "technique": item[
                    "technique"
                ],
                "confidence": round(
                    item[
                        "confidence"
                    ],
                    6,
                ),
                "threshold": round(
                    item[
                        "threshold"
                    ],
                    6,
                ),
                "detected": item[
                    "detected"
                ],
            }
            for item in all_scores[:10]
        ],

        "prediction_count": len(
            detected
        ),

        "feature_count": int(
            feature_result[
                "matrix"
            ].shape[1]
        ),

        "mitre_technique_count": len(
            artifacts[
                "label_encoder"
            ].classes_
        ),
    }


# ============================================================================
# DEMO EVENT
# ============================================================================

def build_demo_event() -> Dict[str, Any]:

    """
    Controlled benign administrative PowerShell event.

    This is intentionally:
        powershell.exe Get-Service

    It should exercise the administrative PowerShell feature without
    introducing encoded commands, downloads, remote execution, deletion,
    or obfuscation.
    """

    return {
        "_source": {
            "timestamp":
                "2026-08-12T10:00:00.000+0000",

            "data": {
                "win": {
                    "system": {
                        "eventID": "1",
                        "level": "12",
                        "providerName":
                            "Microsoft-Windows-Sysmon",
                        "channel":
                            "Microsoft-Windows-Sysmon/Operational",
                    },

                    "eventdata": {
                        "image":
                            r"C:\Windows\System32"
                            r"\WindowsPowerShell\v1.0"
                            r"\powershell.exe",

                        "parentImage":
                            r"C:\Windows\explorer.exe",

                        "commandLine":
                            "powershell.exe Get-Service",

                        "parentCommandLine":
                            "explorer.exe",

                        "user":
                            "administrator",

                        "parentUser":
                            "administrator",

                        "integrityLevel":
                            "High",

                        "hashes":
                            "",

                        "sourceIp":
                            "",

                        "destinationIp":
                            "",

                        "sourcePort":
                            "",

                        "destinationPort":
                            "",

                        "protocol":
                            "",

                        "targetFilename":
                            "",

                        "imageLoaded":
                            "",

                        "targetObject":
                            "",
                    },
                }
            },

            "rule": {
                "id": "100001",
                "level": "12",
            },
        }
    }


# ============================================================================
# PRINT FEATURE SUMMARY
# ============================================================================

def print_feature_summary(
    features: Dict[str, Any],
) -> None:

    print()
    print("Important derived features:")

    important = [
        "event_id",
        "system_level",
        "is_sysmon",
        "is_security_channel",
        "process_name",
        "parent_process_name",
        "uses_powershell",
        "uses_cmd",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "administrative_powershell_context",
        "strong_behavior_signal",
    ]

    for name in important:

        if name in features:

            print(
                f"  {name:<35}: "
                f"{features[name]}"
            )


# ============================================================================
# FINAL ALIGNMENT CHECK
# ============================================================================

def final_alignment_check(
    feature_result: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> None:

    header(
        "Final production-alignment checks"
    )

    matrix = feature_result[
        "matrix"
    ]

    encoder = artifacts[
        "encoder"
    ]

    scaler = artifacts[
        "scaler"
    ]

    tfidf = artifacts[
        "tfidf"
    ]

    label_encoder = artifacts[
        "label_encoder"
    ]

    checks = [
        (
            "Categorical features",
            EXPECTED_CATEGORICAL_ENCODED,
            len(
                encoder.get_feature_names_out()
            ),
        ),

        (
            "Numerical features",
            EXPECTED_NUMERICAL,
            getattr(
                scaler,
                "n_features_in_",
                None,
            ),
        ),

        (
            "TF-IDF features",
            EXPECTED_TFIDF,
            len(
                tfidf.vocabulary_
            ),
        ),

        (
            "Final features",
            EXPECTED_FEATURES,
            matrix.shape[1],
        ),

        (
            "MITRE techniques",
            EXPECTED_TECHNIQUES,
            len(
                label_encoder.classes_
            ),
        ),
    ]

    all_passed = True

    for name, expected, actual in checks:

        passed = (
            expected == actual
        )

        print(
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name:<25} "
            f"expected={expected} "
            f"actual={actual}"
        )

        if not passed:
            all_passed = False

    if not all_passed:

        raise RuntimeError(
            "One or more production-alignment checks failed."
        )

    print()
    print(
        "[PASS] v4.1 aligned inference completed."
    )

    print(
        "[PASS] Production artifacts were not modified."
    )

    print(
        "[PASS] Final model input dimension = 6919."
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    header(
        "SmartSOC Windows-APT Model B v4.1"
    )

    print(
        "ALIGNED PRODUCTION INFERENCE"
    )

    print()
    print(
        "Production model remains unchanged."
    )

    artifacts = load_artifacts()

    validate_artifacts(
        artifacts
    )

    # ========================================================================
    # CONTROLLED TEST
    # ========================================================================

    header(
        "CONTROLLED TEST 1: "
        "BENIGN ADMINISTRATIVE POWERSHELL"
    )

    demo_raw = build_demo_event()

    section(
        "Building normalized Wazuh event"
    )

    event = normalize_wazuh_event(
        demo_raw
    )

    print(
        "[OK] Wazuh event normalized."
    )

    feature_result = build_feature_matrix(
        event,
        artifacts,
    )

    print_feature_summary(
        feature_result[
            "features"
        ]
    )

    # ========================================================================
    # PREDICTION
    # ========================================================================

    section(
        "Running v4.1 production model"
    )

    prediction_result = predict(
        feature_result,
        artifacts,
    )

    detected = prediction_result[
        "detected"
    ]

    all_scores = prediction_result[
        "all_scores"
    ]

    print()

    print(
        "Detected MITRE techniques:"
    )

    if detected:

        for item in detected:

            print(
                f"{item['technique']:<12} "
                f"confidence="
                f"{item['confidence']:.4f} "
                f"threshold="
                f"{item['threshold']:.4f}"
            )

    else:

        print(
            "  None"
        )

    print()

    print(
        "Top probability scores:"
    )

    for item in all_scores[:10]:

        marker = (
            "*"
            if item["detected"]
            else " "
        )

        print(
            f"{marker} "
            f"{item['technique']:<12} "
            f"{item['confidence']:.4f} "
            f"threshold="
            f"{item['threshold']:.4f}"
        )

    # ========================================================================
    # RESULT JSON
    # ========================================================================

    result_json = build_result_json(
        event,
        feature_result,
        prediction_result,
        artifacts,
    )

    header(
        "Inference result"
    )

    print(
        json.dumps(
            result_json,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ========================================================================
    # FINAL CHECKS
    # ========================================================================

    final_alignment_check(
        feature_result,
        artifacts,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()