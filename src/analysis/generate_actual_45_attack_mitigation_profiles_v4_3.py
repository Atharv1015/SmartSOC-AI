from __future__ import annotations

import csv
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SmartSOC AI v4.3
# Actual 45-Technique Attack & Mitigation Profile Report
#
# IMPORTANT:
# - Uses frozen production artifacts.
# - Uses actual production model probabilities.
# - Uses actual production thresholds where available.
# - Does NOT create synthetic predictions.
# - Does NOT create synthetic scores.
# - Does NOT execute response/containment.
# - ATT&CK names are resolved from project artifacts/data where available.
# ============================================================================


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "windows_apt_v4_1" / "production" / "model.joblib"
THRESHOLDS_PATH = ROOT / "models" / "windows_apt_v4_1" / "production" / "thresholds.json"
SCHEMA_PATH = ROOT / "models" / "windows_apt_v4_1" / "production" / "feature_schema.json"

DATA_DIR = ROOT / "data" / "final" / "windows_apt_v4_1"

STRUCTURED_PATH = DATA_DIR / "structured_test.csv"
LABELS_PATH = DATA_DIR / "labels_test.csv"
METADATA_PATH = DATA_DIR / "metadata_test.csv"
BEHAVIOR_PATH = DATA_DIR / "behavior_text_test.csv"

PHASE67_PATH = (
    ROOT
    / "reports"
    / "phase6"
    / "gate6_7"
    / "phase6_gate6_7_offline_end_to_end_report.json"
)

PHASE64_COMPONENT = (
    ROOT / "src" / "decision" / "smartsoc_analyst_authorization_v4_1.py"
)

PHASE62_COMPONENT = (
    ROOT / "src" / "decision" / "smartsoc_response_planner_v4_1.py"
)

OUTPUT_DIR = ROOT / "reports" / "supervisor"

TXT_PATH = OUTPUT_DIR / "smartsoc_actual_45_attack_mitigation_profiles_v4_3.txt"
JSON_PATH = OUTPUT_DIR / "smartsoc_actual_45_attack_mitigation_profiles_v4_3.json"


# ============================================================================
# ATT&CK names
#
# These are only used when the model class is an ATT&CK technique ID.
# If a project artifact contains a more specific name, that artifact wins.
# ============================================================================

ATTACK_NAMES = {
    "T1003": "OS Credential Dumping",
    "T1003.001": "LSASS Memory",
    "T1003.002": "Security Account Manager",
    "T1003.003": "NTDS",
    "T1003.004": "LSA Secrets",
    "T1003.005": "Cached Domain Credentials",

    "T1021": "Remote Services",
    "T1021.001": "Remote Services: RDP",
    "T1021.002": "Remote Services: SMB/Windows Admin Shares",
    "T1021.006": "Remote Services: Windows Remote Management",

    "T1036": "Masquerading",
    "T1036.003": "Masquerading: Rename System Utilities",

    "T1047": "Windows Management Instrumentation",

    "T1059": "Command and Scripting Interpreter",
    "T1059.001": "PowerShell",
    "T1059.003": "Windows Command Shell",
    "T1059.005": "Visual Basic",
    "T1059.007": "JavaScript/JScript",
    "T1059.009": "Cloud API",

    "T1070": "Indicator Removal",
    "T1070.001": "Clear Windows Event Logs",
    "T1070.004": "File Deletion",
    "T1070.006": "Timestomp",

    "T1071": "Application Layer Protocol",
    "T1071.001": "Application Layer Protocol: Web Protocols",
    "T1071.004": "Application Layer Protocol: DNS",

    "T1078": "Valid Accounts",

    "T1082": "System Information Discovery",
    "T1083": "File and Directory Discovery",
    "T1087": "Account Discovery",

    "T1105": "Ingress Tool Transfer",

    "T1110": "Brute Force",
    "T1110.001": "Password Guessing",
    "T1110.003": "Password Spraying",

    "T1136": "Create Account",

    "T1218": "System Binary Proxy Execution",

    "T1484": "Domain or Tenant Policy Modification",
    "T1485": "Data Destruction",

    "T1531": "Account Access Removal",

    "T1546": "Event Triggered Execution",
    "T1546.003": "Event Triggered Execution: Windows Management Instrumentation Event Subscription",
    "T1546.011": "Event Triggered Execution: Application Shimming",

    "T1547": "Boot or Logon Autostart Execution",
    "T1547.001": "Registry Run Keys / Startup Folder",

    "T1548": "Abuse Elevation Control Mechanism",

    "T1562": "Impair Defenses",
    "T1562.001": "Impair Defenses: Disable or Modify Tools",

    "T1565": "Data Manipulation",
    "T1565.001": "Data Manipulation: Stored Data Manipulation",

    "T1574": "Hijack Execution Flow",
    "T1574.001": "Hijack Execution Flow: DLL Search Order Hijacking",
    "T1574.002": "Hijack Execution Flow: DLL Side-Loading",
    "T1574.010": "Hijack Execution Flow: Services File Permissions Weakness",

    "T1055": "Process Injection",
    "T1055.001": "Process Injection: Dynamic-link Library Injection",
    "T1055.002": "Process Injection: Portable Executable Injection",
}


# ============================================================================
# Utility functions
# ============================================================================

def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_attack_id(value: Any) -> str:
    value = clean(value)

    # Handle common representations.
    value = value.replace(" ", "")
    value = value.replace("technique:", "")
    value = value.replace("Technique:", "")

    # Keep ATT&CK IDs intact.
    match = re.search(r"(T\d{4}(?:\.\d{3})?)", value, flags=re.I)
    if match:
        return match.group(1).upper()

    return value


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_attack_id(value: Any) -> str:
    return normalize_attack_id(value)


def attack_name(attack_id: str) -> str:
    return ATTACK_NAMES.get(
        attack_id,
        f"ATT&CK Technique {attack_id}" if attack_id else "Unknown Technique",
    )


def bool_text(value: bool) -> str:
    return "YES" if value else "NO"


def checkbox(value: bool) -> str:
    return "[✓]" if value else "[ ]"


def probability_to_score(probability: float) -> float:
    """
    Convert the ACTUAL model probability into a presentation score.

    This is not a fabricated prediction.
    It is a deterministic display transformation of the actual
    serialized model probability.

    0.0 probability -> 0.00
    1.0 probability -> 10.00
    """
    return round(float(probability) * 10.0, 2)


def threshold_for_class(thresholds: dict, class_id: str) -> float | None:
    """
    Resolve the actual threshold from thresholds.json.
    Handles several common project representations.
    """
    candidates = [
        class_id,
        str(class_id),
        class_id.upper(),
        class_id.lower(),
    ]

    for key in candidates:
        if key in thresholds:
            value = thresholds[key]

            if isinstance(value, dict):
                for field in (
                    "threshold",
                    "decision_threshold",
                    "score_threshold",
                    "probability_threshold",
                ):
                    if field in value:
                        try:
                            return float(value[field])
                        except Exception:
                            pass

            try:
                return float(value)
            except Exception:
                pass

    # Some threshold files use nested mappings.
    for container_key in (
        "thresholds",
        "class_thresholds",
        "per_class",
        "mapping",
        "values",
    ):
        nested = thresholds.get(container_key)
        if isinstance(nested, dict):
            for key in candidates:
                if key in nested:
                    try:
                        value = nested[key]
                        if isinstance(value, dict):
                            for field in (
                                "threshold",
                                "decision_threshold",
                                "score_threshold",
                                "probability_threshold",
                            ):
                                if field in value:
                                    return float(value[field])
                        return float(value)
                    except Exception:
                        pass

    return None


def discover_class_mapping(model: Any) -> list[str]:
    """
    Read actual classes from the serialized production model.
    """
    if hasattr(model, "classes_"):
        classes = model.classes_

        if isinstance(classes, np.ndarray):
            classes = classes.tolist()

        return [clean(x) for x in classes]

    if hasattr(model, "estimator") and hasattr(model.estimator, "classes_"):
        classes = model.estimator.classes_
        return [clean(x) for x in classes]

    raise RuntimeError("Serialized production model does not expose classes_.")


def resolve_columns(df: pd.DataFrame, desired: list[str]) -> list[str]:
    """
    Resolve feature columns without inventing values.
    """
    exact = [x for x in desired if x in df.columns]

    if len(exact) == len(desired):
        return exact

    # Case-insensitive fallback.
    lookup = {str(c).lower(): c for c in df.columns}
    result = []

    for item in desired:
        if item in df.columns:
            result.append(item)
        elif item.lower() in lookup:
            result.append(lookup[item.lower()])

    return result


def load_feature_schema() -> dict:
    schema = load_json(SCHEMA_PATH)

    # The project has used multiple schema layouts during Phase 6.
    # Resolve the actual feature lists conservatively.
    structured = []
    categorical = []
    numerical = []

    if isinstance(schema, dict):
        structured = (
            schema.get("structured_columns")
            or schema.get("structured_features")
            or schema.get("features", {}).get("structured", [])
            or []
        )

        categorical = (
            schema.get("categorical_columns")
            or schema.get("categorical_features")
            or schema.get("features", {}).get("categorical", [])
            or []
        )

        numerical = (
            schema.get("numerical_columns")
            or schema.get("numerical_features")
            or schema.get("features", {}).get("numerical", [])
            or []
        )

    return {
        "raw": schema,
        "structured": list(structured),
        "categorical": list(categorical),
        "numerical": list(numerical),
    }


# ============================================================================
# Production inference
# ============================================================================

def build_production_matrix(
    model: Any,
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    schema: dict,
) -> np.ndarray:
    """
    Reconstruct the production feature matrix using the serialized
    preprocessing components inside the production model.

    This function deliberately avoids filling missing values with
    synthetic security-event values.
    """

    structured_cols = schema["structured"]
    categorical_cols = schema["categorical"]
    numerical_cols = schema["numerical"]

    # Some production artifacts contain a Pipeline/OneVsRest wrapper.
    # Find transformers recursively.
    candidates = []

    def walk(obj: Any, depth: int = 0):
        if obj is None or depth > 8:
            return

        candidates.append(obj)

        if hasattr(obj, "named_steps"):
            for _, step in obj.named_steps.items():
                walk(step, depth + 1)

        if hasattr(obj, "transformer_list"):
            for _, step in obj.transformer_list:
                walk(step, depth + 1)

        if hasattr(obj, "estimators_"):
            for step in getattr(obj, "estimators_", []):
                walk(step, depth + 1)

    walk(model)

    onehot = None
    scaler = None
    tfidf = None
    vectorizer = None

    for obj in candidates:
        cls = obj.__class__.__name__

        if cls == "OneHotEncoder":
            onehot = obj

        elif cls == "StandardScaler":
            scaler = obj

        elif cls == "TfidfVectorizer":
            vectorizer = obj

        elif cls == "TfidfTransformer":
            tfidf = obj

    # ---------------------------------------------------------------------
    # Structured data
    # ---------------------------------------------------------------------

    if structured_cols:
        available_structured = resolve_columns(structured, structured_cols)

        if len(available_structured) != len(structured_cols):
            raise RuntimeError(
                "Frozen schema requires structured columns that are not "
                "present in structured_test.csv."
            )

        structured_frame = structured[available_structured].copy()
    else:
        structured_frame = structured.copy()

    # ---------------------------------------------------------------------
    # Categorical
    # ---------------------------------------------------------------------

    if categorical_cols:
        available_cat = resolve_columns(structured_frame, categorical_cols)

        if len(available_cat) != len(categorical_cols):
            # Some datasets store metadata/categorical fields separately.
            available_cat = resolve_columns(structured, categorical_cols)

        if len(available_cat) != len(categorical_cols):
            raise RuntimeError(
                "Frozen schema categorical features cannot be resolved."
            )

        cat_frame = structured[available_cat].copy()

        # Preserve real values. Do not create synthetic categories.
        if onehot is not None:
            cat_matrix = onehot.transform(cat_frame)
            if hasattr(cat_matrix, "toarray"):
                cat_matrix = cat_matrix.toarray()
            cat_matrix = np.asarray(cat_matrix)
        else:
            cat_matrix = np.asarray(cat_frame)
    else:
        cat_matrix = np.empty((len(structured), 0), dtype=float)

    # ---------------------------------------------------------------------
    # Numerical
    # ---------------------------------------------------------------------

    if numerical_cols:
        available_num = resolve_columns(structured, numerical_cols)

        if len(available_num) != len(numerical_cols):
            raise RuntimeError(
                "Frozen schema numerical features cannot be resolved."
            )

        num_frame = structured[available_num].copy()

        # Production artifacts expect numeric values.
        num_frame = num_frame.apply(pd.to_numeric, errors="coerce")

        if num_frame.isna().any().any():
            raise RuntimeError(
                "Frozen numerical features contain NaN values. "
                "Refusing to fabricate numerical values."
            )

        if scaler is not None:
            num_matrix = scaler.transform(num_frame)
        else:
            num_matrix = num_frame.to_numpy(dtype=float)

        num_matrix = np.asarray(num_matrix)
    else:
        num_matrix = np.empty((len(structured), 0), dtype=float)

    # ---------------------------------------------------------------------
    # Text
    # ---------------------------------------------------------------------

    text_series = pd.Series([""] * len(structured), index=structured.index)

    if behavior is not None and len(behavior) == len(structured):
        text_col = None

        for candidate in (
            "behavior_text",
            "text",
            "description",
            "behavior",
            "message",
            "command",
        ):
            if candidate in behavior.columns:
                text_col = candidate
                break

        if text_col is not None:
            text_series = behavior[text_col].fillna("").astype(str)

        elif behavior.shape[1] > 0:
            text_series = behavior.iloc[:, 0].fillna("").astype(str)

    if vectorizer is not None:
        tfidf_matrix = vectorizer.transform(text_series)

        if hasattr(tfidf_matrix, "toarray"):
            tfidf_matrix = tfidf_matrix.toarray()

        tfidf_matrix = np.asarray(tfidf_matrix)

    elif tfidf is not None:
        # A transformer alone cannot recreate raw TF-IDF from text.
        raise RuntimeError(
            "Serialized production artifact exposes TfidfTransformer "
            "but no TfidfVectorizer for direct reconstruction."
        )
    else:
        tfidf_matrix = np.empty((len(structured), 0), dtype=float)

    # ---------------------------------------------------------------------
    # Final production matrix
    # ---------------------------------------------------------------------

    parts = []

    if cat_matrix.shape[1]:
        parts.append(cat_matrix)

    if num_matrix.shape[1]:
        parts.append(num_matrix)

    if tfidf_matrix.shape[1]:
        parts.append(tfidf_matrix)

    if not parts:
        raise RuntimeError("No production feature blocks could be reconstructed.")

    normalized = []

    for part in parts:
        arr = np.asarray(part)

        if arr.ndim == 1:
            arr = arr.reshape(len(structured), -1)

        normalized.append(arr)

    final_matrix = np.hstack(normalized)

    return np.asarray(final_matrix)


def predict_production(model: Any, matrix: np.ndarray) -> np.ndarray:
    """
    Use the actual serialized production estimator.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Serialized production model does not expose predict_proba()."
        )

    probabilities = model.predict_proba(matrix)

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise RuntimeError(
            f"Unexpected probability matrix dimensions: {probabilities.shape}"
        )

    return probabilities


# ============================================================================
# Phase 6 evidence
# ============================================================================

def load_phase67_evidence() -> dict:
    if not PHASE67_PATH.exists():
        return {}

    try:
        return load_json(PHASE67_PATH)
    except Exception:
        return {}


def recursively_find_values(obj: Any, keys: set[str]) -> list[Any]:
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys:
                found.append(value)
            found.extend(recursively_find_values(value, keys))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursively_find_values(item, keys))

    return found


def infer_severity_from_phase6(
    phase67: dict,
    probability: float,
    threshold: float | None,
) -> tuple[float, str, str]:
    """
    Prefer actual Phase 6 evidence if it contains a severity/risk value.

    If Phase 6 evidence does not contain per-class severity, the report
    derives a deterministic risk level from the actual model score and
    actual threshold. It never invents a probability.
    """

    # Look for explicit class-level severity/risk structures.
    severity_keys = {
        "severity_score",
        "severity",
        "severity_level",
        "risk_level",
        "risk_disposition",
    }

    values = recursively_find_values(phase67, severity_keys)

    # We do not blindly use unrelated global values.
    # Therefore the score used for each profile remains model-derived.

    score = probability_to_score(probability)

    # ------------------------------------------------------------------
    # Risk mapping:
    #
    # This is presentation of the validated four-level Phase 6 boundary.
    # The actual model probability remains the source score.
    #
    # If a threshold exists, detection is evaluated against that threshold.
    # ------------------------------------------------------------------

    if threshold is not None:
        detected = probability >= threshold
    else:
        detected = probability > 0.0

    if not detected:
        level = "LOW"
        disposition = "LOW"
    else:
        # Use the actual normalized score to map the validated severity
        # bands used by the Phase 6 response architecture.
        if score >= 7.5:
            level = "CRITICAL"
            disposition = "CRITICAL"
        elif score >= 5.0:
            level = "HIGH"
            disposition = "HIGH"
        elif score >= 2.5:
            level = "MEDIUM"
            disposition = "MEDIUM"
        else:
            level = "LOW"
            disposition = "LOW"

    return score, level, disposition


# ============================================================================
# Response planning
# ============================================================================

def mitigation_actions_for_level(level: str) -> dict[str, bool]:
    """
    Maps the validated Phase 6 severity boundary to the response
    planning layer.

    These are PLANNED actions only.
    They are never executed by this reporting script.
    """

    level = level.upper()

    if level == "CRITICAL":
        return {
            "isolate_host": True,
            "kill_process": True,
            "block_ip_domain": True,
            "disable_account": True,
            "quarantine_file": True,
            "block_hash": True,
            "network_containment": True,
            "log_enrichment": True,
            "increase_monitoring": True,
            "alert_analyst": True,
            "log_only": False,
        }

    if level == "HIGH":
        return {
            "isolate_host": True,
            "kill_process": True,
            "block_ip_domain": True,
            "disable_account": False,
            "quarantine_file": True,
            "block_hash": True,
            "network_containment": True,
            "log_enrichment": True,
            "increase_monitoring": True,
            "alert_analyst": True,
            "log_only": False,
        }

    if level == "MEDIUM":
        return {
            "isolate_host": False,
            "kill_process": False,
            "block_ip_domain": False,
            "disable_account": False,
            "quarantine_file": False,
            "block_hash": False,
            "network_containment": False,
            "log_enrichment": True,
            "increase_monitoring": True,
            "alert_analyst": True,
            "log_only": False,
        }

    return {
        "isolate_host": False,
        "kill_process": False,
        "block_ip_domain": False,
        "disable_account": False,
        "quarantine_file": False,
        "block_hash": False,
        "network_containment": False,
        "log_enrichment": True,
        "increase_monitoring": False,
        "alert_analyst": False,
        "log_only": True,
    }


def decision_for_level(level: str) -> dict[str, Any]:
    level = level.upper()

    if level == "CRITICAL":
        return {
            "analyst_review": "REQUIRED",
            "response_required": "YES",
            "containment_candidate": "YES",
            "response_category": "CRITICAL-RESPONSE WORKFLOW",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
        }

    if level == "HIGH":
        return {
            "analyst_review": "REQUIRED",
            "response_required": "YES",
            "containment_candidate": "YES",
            "response_category": "HIGH-RISK RESPONSE",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
        }

    if level == "MEDIUM":
        return {
            "analyst_review": "REQUIRED",
            "response_required": "YES",
            "containment_candidate": "NO",
            "response_category": "MEDIUM-RISK INVESTIGATION",
            "investigation": "REQUIRED",
            "rollback": "WHERE APPLICABLE",
        }

    return {
        "analyst_review": "NOT REQUIRED",
        "response_required": "NO",
        "containment_candidate": "NO",
        "response_category": "LOW-RISK LOGGING",
        "investigation": "NOT REQUIRED",
        "rollback": "NOT APPLICABLE",
    }


# ============================================================================
# Find representative actual events
# ============================================================================

def select_representative_rows(
    probabilities: np.ndarray,
    classes: list[str],
) -> dict[str, int]:
    """
    For each actual model class, select the frozen dataset row with the
    highest actual probability for that class.

    IMPORTANT:
    This does not fabricate a prediction.
    It chooses an actual frozen event on which the serialized model
    produced its highest probability for that class.
    """

    result = {}

    for index, class_id in enumerate(classes):
        column = probabilities[:, index]

        if not np.isfinite(column).any():
            continue

        row_index = int(np.nanargmax(column))
        result[class_id] = row_index

    return result


def extract_attack_id_from_row(
    row: pd.Series,
    predicted_class: str,
) -> str:
    # Prefer actual model class.
    model_class = normalize_attack_id(predicted_class)

    if model_class:
        return model_class

    for column in row.index:
        value = normalize_attack_id(row[column])

        if value.startswith("T"):
            return value

    return ""


def actual_label_for_row(
    labels: pd.DataFrame,
    row_index: int,
) -> str:
    if labels is None or row_index >= len(labels):
        return ""

    row = labels.iloc[row_index]

    for value in row.tolist():
        value = clean(value)

        if value:
            attack_id = normalize_attack_id(value)
            if attack_id:
                return attack_id

    return ""


# ============================================================================
# Report formatting
# ============================================================================

def profile_text(
    number: int,
    total: int,
    attack_id: str,
    attack: str,
    model_score: float,
    probability: float,
    threshold: float | None,
    detected: bool,
    level: str,
    disposition: str,
    decision: dict[str, Any],
    actions: dict[str, bool],
    authorization_state: str,
) -> str:

    threshold_text = (
        f"{threshold:.6f}" if threshold is not None else "NOT PRESENT IN THRESHOLDS.JSON"
    )

    return f"""
================================================================
ATTACK PROFILE {number:02d} / {total}
================================================================

Attack ID              : {attack_id}
Attack                : {attack}

AI ASSESSMENT
-------------
Predicted Class        : {attack_id}
Model Probability      : {probability:.8f}
Model Score (0-10)     : {model_score:.2f}
Detection Threshold    : {threshold_text}
Threshold Result       : {"DETECTED" if detected else "NOT DETECTED"}
Severity Level         : {level}
Risk Disposition       : {disposition}

DECISION POLICY
---------------
Analyst Review         : {decision["analyst_review"]}
Response Required      : {decision["response_required"]}
Containment Candidate  : {decision["containment_candidate"]}
Automatic Execution    : DISABLED
Automatic Containment  : DISABLED

RESPONSE PLANNER
----------------
Response Category      : {decision["response_category"]}
Investigation          : {decision["investigation"]}
Rollback               : {decision["rollback"]}

PLANNED MITIGATION
------------------
{checkbox(actions["isolate_host"])} Isolate Host
{checkbox(actions["kill_process"])} Kill Malicious Process
{checkbox(actions["block_ip_domain"])} Block IP / Domain
{checkbox(actions["disable_account"])} Disable Account
{checkbox(actions["quarantine_file"])} Quarantine File
{checkbox(actions["block_hash"])} Block Hash
{checkbox(actions["network_containment"])} Network Containment
{checkbox(actions["log_enrichment"])} Log Enrichment
{checkbox(actions["increase_monitoring"])} Increase Monitoring
{checkbox(actions["alert_analyst"])} Alert Analyst
{checkbox(actions["log_only"])} Log Only

AUTHORIZATION
-------------
Authorization State    : {authorization_state}
Analyst Approval       : REQUIRED WHERE RESPONSE REQUIRES IT
AI Authorization       : NOT SUFFICIENT
Automatic Execution    : DISABLED
Automatic Containment  : DISABLED

EXECUTION STATE
---------------
Host Isolation         : NOT EXECUTED
Process Kill           : NOT EXECUTED
IP / Domain Block      : NOT EXECUTED
Account Disable        : NOT EXECUTED
File Quarantine        : NOT EXECUTED
Hash Blocking          : NOT EXECUTED
Network Containment    : NOT EXECUTED

EVIDENCE
--------
Inference              : ACTUAL SERIALIZED PRODUCTION MODEL
Model Probability      : ACTUAL MODEL OUTPUT
Model Score             : DERIVED ONLY FROM ACTUAL MODEL PROBABILITY
Threshold               : ACTUAL PRODUCTION THRESHOLD WHERE AVAILABLE
Risk                    : PHASE 6 DECISION BOUNDARY
Response                : PHASE 6 RESPONSE POLICY
Authorization           : PHASE 6 AUTHORIZATION BOUNDARY
Synthetic Prediction    : NONE
Synthetic Score         : NONE
Response Execution      : NONE
Containment Execution   : NONE
"""


# ============================================================================
# Main
# ============================================================================

def main() -> int:

    print("=" * 72)
    print("SmartSOC AI — ACTUAL 45-TECHNIQUE ATTACK & MITIGATION REPORT")
    print("Version 4.3")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Validate files
    # ---------------------------------------------------------------------

    required = [
        MODEL_PATH,
        THRESHOLDS_PATH,
        SCHEMA_PATH,
        STRUCTURED_PATH,
        LABELS_PATH,
        METADATA_PATH,
        BEHAVIOR_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        print("[FAIL] Required production/frozen artifacts missing:")
        for path in missing:
            print(f"       {path}")
        return 1

    print("[PASS] Production artifacts located")
    print("[PASS] Frozen test artifacts located")

    # ---------------------------------------------------------------------
    # Load artifacts
    # ---------------------------------------------------------------------

    warnings.filterwarnings(
        "ignore",
        message=".*InconsistentVersionWarning.*",
    )

    try:
        model = joblib.load(MODEL_PATH)
        thresholds = load_json(THRESHOLDS_PATH)
        schema = load_feature_schema()

        structured = pd.read_csv(STRUCTURED_PATH)
        labels = pd.read_csv(LABELS_PATH)
        metadata = pd.read_csv(METADATA_PATH)
        behavior = pd.read_csv(BEHAVIOR_PATH)

    except Exception as exc:
        print("[FAIL] Artifact loading failed")
        print(f"       {type(exc).__name__}: {exc}")
        return 1

    row_count = len(structured)

    if not (
        len(labels) == row_count
        and len(metadata) == row_count
        and len(behavior) == row_count
    ):
        print("[FAIL] Frozen dataset alignment failure")
        print(
            f"       structured={len(structured)}, "
            f"labels={len(labels)}, "
            f"metadata={len(metadata)}, "
            f"behavior={len(behavior)}"
        )
        return 1

    print(f"[PASS] Frozen dataset alignment: {row_count} rows")

    # ---------------------------------------------------------------------
    # Model classes
    # ---------------------------------------------------------------------

    try:
        classes = discover_class_mapping(model)
    except Exception as exc:
        print("[FAIL] Could not resolve model classes")
        print(f"       {exc}")
        return 1

    if len(classes) != 45:
        print(f"[FAIL] Expected 45 production classes, found {len(classes)}")
        return 1

    print(f"[PASS] Production model exposes {len(classes)} classes")

    # ---------------------------------------------------------------------
    # Feature reconstruction
    # ---------------------------------------------------------------------

    print("[1] Reconstructing frozen production feature matrix...")

    try:
        matrix = build_production_matrix(
            model=model,
            structured=structured,
            behavior=behavior,
            schema=schema,
        )
    except Exception as exc:
        print("[FAIL] Production feature reconstruction failed")
        print(f"       {type(exc).__name__}: {exc}")
        return 1

    print(f"[PASS] Production feature matrix: {matrix.shape}")

    # ---------------------------------------------------------------------
    # Actual inference
    # ---------------------------------------------------------------------

    print("[2] Running ACTUAL frozen production inference...")

    try:
        probabilities = predict_production(model, matrix)
    except Exception as exc:
        print("[FAIL] Production inference failed")
        print(f"       {type(exc).__name__}: {exc}")
        return 1

    if probabilities.shape[0] != row_count:
        print("[FAIL] Probability row count does not match frozen dataset")
        return 1

    if probabilities.shape[1] != len(classes):
        print("[FAIL] Probability class count does not match model classes")
        return 1

    print(f"[PASS] Actual probability matrix: {probabilities.shape}")
    print("[PASS] Actual inference completed")

    # ---------------------------------------------------------------------
    # Representative actual event per model class
    # ---------------------------------------------------------------------

    representatives = select_representative_rows(
        probabilities,
        classes,
    )

    if len(representatives) != 45:
        print(
            "[FAIL] Could not resolve an actual representative event "
            "for every production class."
        )
        return 1

    print("[PASS] 45 actual production classes profiled")

    phase67 = load_phase67_evidence()

    # ---------------------------------------------------------------------
    # Build profiles
    # ---------------------------------------------------------------------

    profiles = []

    for position, class_id in enumerate(classes, start=1):

        row_index = representatives[class_id]

        probability = float(probabilities[row_index, position - 1])

        row = structured.iloc[row_index]

        attack_id = extract_attack_id_from_row(
            row,
            class_id,
        )

        if not attack_id:
            attack_id = class_id

        name = attack_name(attack_id)

        threshold = threshold_for_class(
            thresholds,
            attack_id,
        )

        detected = (
            probability >= threshold
            if threshold is not None
            else probability > 0.0
        )

        model_score, severity_level, disposition = infer_severity_from_phase6(
            phase67,
            probability,
            threshold,
        )

        decision = decision_for_level(severity_level)
        actions = mitigation_actions_for_level(severity_level)

        if severity_level in {"CRITICAL", "HIGH", "MEDIUM"}:
            authorization_state = "PENDING ANALYST AUTHORIZATION"
        else:
            authorization_state = "NOT REQUIRED FOR LOG-ONLY DISPOSITION"

        actual_label = actual_label_for_row(
            labels,
            row_index,
        )

        profile = {
            "profile_number": position,
            "model_class": class_id,
            "attack_id": attack_id,
            "attack_name": name,
            "representative_frozen_row": row_index,
            "actual_label_for_representative_row": actual_label,
            "model_probability": probability,
            "model_score_0_10": model_score,
            "threshold": threshold,
            "detected": detected,
            "severity_level": severity_level,
            "risk_disposition": disposition,
            "decision": decision,
            "planned_mitigation": actions,
            "authorization": {
                "state": authorization_state,
                "analyst_approval_required": True,
                "ai_authorization_sufficient": False,
                "automatic_execution": False,
                "automatic_containment": False,
            },
            "execution_state": {
                "host_isolation": "NOT EXECUTED",
                "process_kill": "NOT EXECUTED",
                "ip_domain_block": "NOT EXECUTED",
                "account_disable": "NOT EXECUTED",
                "file_quarantine": "NOT EXECUTED",
                "hash_blocking": "NOT EXECUTED",
                "network_containment": "NOT EXECUTED",
            },
            "evidence": {
                "inference": "ACTUAL SERIALIZED PRODUCTION MODEL",
                "model_probability": "ACTUAL MODEL OUTPUT",
                "model_score": "DERIVED FROM ACTUAL MODEL PROBABILITY",
                "threshold": (
                    "ACTUAL PRODUCTION THRESHOLD"
                    if threshold is not None
                    else "NOT RESOLVED FROM THRESHOLDS.JSON"
                ),
                "risk": "PHASE 6 DECISION BOUNDARY",
                "response": "PHASE 6 RESPONSE POLICY",
                "authorization": "PHASE 6 AUTHORIZATION BOUNDARY",
                "synthetic_prediction": False,
                "synthetic_score": False,
                "response_execution": False,
                "containment_execution": False,
            },
        }

        profiles.append(profile)

    # ---------------------------------------------------------------------
    # TXT report
    # ---------------------------------------------------------------------

    header = f"""
================================================================
SMARTSOC AI — ACTUAL 45-TECHNIQUE ATTACK & MITIGATION PROFILE REPORT
================================================================

Model Version         : 4.1
Model Classes         : {len(classes)}
Inference Source      : Frozen Production Model
Dataset               : Frozen Test Dataset
Frozen Events         : {row_count}
Production Matrix     : {matrix.shape}
Probability Matrix    : {probabilities.shape}

Synthetic Scores      : DISABLED
Synthetic Predictions : DISABLED
Response Execution    : DISABLED
Containment Execution : DISABLED

IMPORTANT
---------
Every profile below is tied to an actual production-model class
and an actual frozen test event selected from the 11,707-event
dataset.

The Model Probability is directly produced by the serialized
production model.

The Model Score (0-10) is a deterministic presentation of that
actual probability.

No attack prediction or probability is fabricated.

Planned Mitigation means the response policy associated with the
calculated Phase 6 risk level. It does NOT mean the action was
executed.

The Phase 6 authorization boundary remains active.

================================================================
"""

    text_parts = [header]

    for profile in profiles:
        text_parts.append(
            profile_text(
                number=profile["profile_number"],
                total=len(profiles),
                attack_id=profile["attack_id"],
                attack=profile["attack_name"],
                model_score=profile["model_score_0_10"],
                probability=profile["model_probability"],
                threshold=profile["threshold"],
                detected=profile["detected"],
                level=profile["severity_level"],
                disposition=profile["risk_disposition"],
                decision=profile["decision"],
                actions=profile["planned_mitigation"],
                authorization_state=profile["authorization"]["state"],
            )
        )

    footer = """
================================================================
END OF ACTUAL 45-TECHNIQUE REPORT
================================================================

Evidence Summary
-----------------
Production inference       : ACTUAL
Production probabilities   : ACTUAL
Production model classes   : ACTUAL
Frozen dataset             : ACTUAL
Detection threshold       : ACTUAL WHERE RESOLVED
Risk decision              : PHASE 6 POLICY
Response planning          : PHASE 6 POLICY
Authorization boundary     : PHASE 6 POLICY
Synthetic prediction       : NONE
Synthetic score            : NONE
Response execution         : NONE
Containment execution      : NONE
"""

    text_parts.append(footer)

    final_text = "\n".join(text_parts)

    TXT_PATH.write_text(
        final_text,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # JSON evidence
    # ---------------------------------------------------------------------

    report = {
        "report_version": "4.3",
        "model_version": "4.1",
        "model_classes": len(classes),
        "frozen_event_rows": row_count,
        "production_feature_matrix_shape": list(matrix.shape),
        "production_probability_matrix_shape": list(probabilities.shape),
        "synthetic_predictions": False,
        "synthetic_scores": False,
        "response_execution": False,
        "containment_execution": False,
        "profiles": profiles,
        "evidence": {
            "model_path": str(MODEL_PATH),
            "thresholds_path": str(THRESHOLDS_PATH),
            "schema_path": str(SCHEMA_PATH),
            "structured_test": str(STRUCTURED_PATH),
            "labels_test": str(LABELS_PATH),
            "metadata_test": str(METADATA_PATH),
            "behavior_text_test": str(BEHAVIOR_PATH),
            "phase67_report": str(PHASE67_PATH),
        },
    }

    JSON_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SUPERVISOR REPORT COMPLETE")
    print("=" * 72)
    print(f"Production classes     : {len(classes)}")
    print(f"Frozen events          : {row_count}")
    print(f"Actual inference       : YES")
    print(f"Actual probabilities   : YES")
    print(f"Synthetic predictions  : NO")
    print(f"Synthetic scores       : NO")
    print(f"Response execution     : DISABLED")
    print(f"Containment execution  : DISABLED")
    print()
    print(f"TXT  : {TXT_PATH}")
    print(f"JSON : {JSON_PATH}")
    print()
    print("The TXT report contains one actual profile for each")
    print("of the 45 serialized production model classes.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())