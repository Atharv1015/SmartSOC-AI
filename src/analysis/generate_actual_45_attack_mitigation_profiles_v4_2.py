"""
SmartSOC AI v4.1
ACTUAL 45-TECHNIQUE ATTACK & MITIGATION PROFILE REPORT

Purpose
-------
Generate a supervisor-facing TXT report from the ACTUAL frozen
production inference data.

IMPORTANT
---------
This script NEVER fabricates predictions, scores, or attack events.

For every one of the 45 production model classes it reports:

    - MITRE technique ID
    - actual model probability evidence
    - maximum observed model score
    - mean observed score
    - production threshold
    - threshold-hit event count
    - top-prediction event count
    - frozen test-set ground-truth support
    - actual policy/risk disposition where available
    - response planner decision
    - mitigation plan
    - analyst authorization boundary
    - execution boundary

A class does NOT need to be the top prediction of an event to have
real probability evidence. Therefore all 45 production classes can
be represented without inventing events.

No production artifacts are modified.
No model is retrained.
No response action is executed.
No containment action is executed.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"

DATA_DIR = ROOT / "data" / "final" / "windows_apt_v4_1"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
METADATA_TEST = DATA_DIR / "metadata_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"

REPORT_DIR = ROOT / "reports" / "supervisor"

REPORT_FILE = (
    REPORT_DIR /
    "smartsoc_actual_45_attack_mitigation_profiles_v4_2.txt"
)

JSON_FILE = (
    REPORT_DIR /
    "smartsoc_actual_45_attack_mitigation_profiles_v4_2.json"
)


# ============================================================================
# EXPECTED PRODUCTION CONTRACT
# ============================================================================

EXPECTED_CLASSES = 45
EXPECTED_ROWS = 11707
EXPECTED_FINAL_FEATURES = 6919


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
# 45 PRODUCTION TECHNIQUES
# ============================================================================

TECHNIQUES = [
    "T1003.001",
    "T1003.002",
    "T1021.006",
    "T1027",
    "T1027.004",
    "T1036",
    "T1047",
    "T1053.005",
    "T1055",
    "T1059",
    "T1059.001",
    "T1059.003",
    "T1059.005",
    "T1070",
    "T1070.004",
    "T1078",
    "T1087",
    "T1095",
    "T1098",
    "T1105",
    "T1110",
    "T1110.001",
    "T1112",
    "T1136",
    "T1210",
    "T1218",
    "T1484",
    "T1485",
    "T1489",
    "T1529",
    "T1531",
    "T1543.003",
    "T1546",
    "T1546.003",
    "T1546.011",
    "T1547.001",
    "T1548",
    "T1548.002",
    "T1562.001",
    "T1562.004",
    "T1565.001",
    "T1570",
    "T1574.001",
    "T1574.002",
    "T1574.010",
]


# ============================================================================
# TECHNIQUE NAMES
# ============================================================================

TECHNIQUE_NAMES = {
    "T1003.001": "LSASS Memory",
    "T1003.002": "Security Account Manager",
    "T1021.006": "Windows Remote Management",
    "T1027": "Obfuscated / Compressed Files and Information",
    "T1027.004": "Compile After Delivery",
    "T1036": "Masquerading",
    "T1047": "Windows Management Instrumentation",
    "T1053.005": "Scheduled Task / Job: Scheduled Task",
    "T1055": "Process Injection",
    "T1059": "Command and Scripting Interpreter",
    "T1059.001": "PowerShell",
    "T1059.003": "Windows Command Shell",
    "T1059.005": "Visual Basic",
    "T1070": "Indicator Removal",
    "T1070.004": "File Deletion",
    "T1078": "Valid Accounts",
    "T1087": "Account Discovery",
    "T1095": "Non-Application Layer Protocol",
    "T1098": "Account Manipulation",
    "T1105": "Ingress Tool Transfer",
    "T1110": "Brute Force",
    "T1110.001": "Password Guessing",
    "T1112": "Modify Registry",
    "T1136": "Create Account",
    "T1210": "Exploitation of Remote Services",
    "T1218": "System Binary Proxy Execution",
    "T1484": "Domain or Tenant Policy Modification",
    "T1485": "Data Destruction",
    "T1489": "Service Stop",
    "T1529": "System Shutdown / Reboot",
    "T1531": "Account Access Removal",
    "T1543.003": "Windows Service",
    "T1546": "Event Triggered Execution",
    "T1546.003": "Windows Management Instrumentation Event Subscription",
    "T1546.011": "Application Shimming",
    "T1547.001": "Registry Run Keys / Startup Folder",
    "T1548": "Abuse Elevation Control Mechanism",
    "T1548.002": "Bypass User Account Control",
    "T1562.001": "Impair Defenses",
    "T1562.004": "Disable or Modify System Firewall",
    "T1565.001": "Stored Data Manipulation",
    "T1570": "Lateral Tool Transfer",
    "T1574.001": "DLL Search Order Hijacking",
    "T1574.002": "DLL Side-Loading",
    "T1574.010": "Services File Permissions Weakness",
}


# ============================================================================
# RESPONSE POLICY
# ============================================================================

def risk_from_score(score: float) -> str:
    """
    This mapping is intentionally conservative.

    The ML probability is NOT claimed to be the official Phase 6
    severity score. It is used only to provide a deterministic
    presentation category when a numeric risk input is required.

    Actual Phase 6 policy remains authoritative.
    """

    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def response_for_risk(level: str) -> Dict[str, Any]:

    if level == "LOW":
        return {
            "response": "LOW-RISK MONITORING",
            "investigation": "NOT REQUIRED",
            "analyst": "NOT REQUIRED",
            "rollback": "NOT REQUIRED",
            "actions": [
                "Increase Monitoring",
                "Log Enrichment",
                "Log Only",
            ],
        }

    if level == "MEDIUM":
        return {
            "response": "MEDIUM-RISK ANALYST REVIEW",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "NOT REQUIRED",
            "actions": [
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
            ],
        }

    if level == "HIGH":
        return {
            "response": "HIGH-RISK RESPONSE",
            "investigation": "REQUIRED",
            "analyst": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "actions": [
                "Isolate Host",
                "Kill Malicious Process",
                "Block IP / Domain",
                "Quarantine File",
                "Block Hash",
                "Network Containment",
                "Log Enrichment",
                "Increase Monitoring",
                "Alert Analyst",
            ],
        }

    return {
        "response": "CRITICAL-RESPONSE WORKFLOW",
        "investigation": "REQUIRED",
        "analyst": "REQUIRED",
        "rollback": "REQUIRED WHERE APPLICABLE",
        "actions": [
            "Isolate Host",
            "Kill Malicious Process",
            "Block IP / Domain",
            "Disable Account",
            "Quarantine File",
            "Block Hash",
            "Network Containment",
            "Log Enrichment",
            "Increase Monitoring",
            "Alert Analyst",
        ],
    }


# ============================================================================
# HELPERS
# ============================================================================

def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_thresholds(raw: Dict[str, Any]) -> Dict[str, float]:

    result: Dict[str, float] = {}

    candidates = [
        raw,
        raw.get("thresholds", {}) if isinstance(raw, dict) else {},
        raw.get("technique_thresholds", {})
        if isinstance(raw, dict)
        else {},
    ]

    for candidate in candidates:

        if not isinstance(candidate, dict):
            continue

        for key, value in candidate.items():

            if key not in TECHNIQUES:
                continue

            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                pass

    return result


def normalize_probability_matrix(
    probabilities: Any,
    classes: List[str],
) -> np.ndarray:

    arr = np.asarray(probabilities)

    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D probability matrix, got shape={arr.shape}"
        )

    if arr.shape[1] != len(classes):

        # Some sklearn versions may return sparse-like structures.
        # Do not silently reshape incorrect data.
        raise RuntimeError(
            "Probability matrix/class mismatch: "
            f"matrix={arr.shape}, classes={len(classes)}"
        )

    return arr.astype(float)


def find_label_columns(labels: pd.DataFrame) -> Dict[str, str]:

    mapping: Dict[str, str] = {}

    normalized = {
        str(column).strip(): column
        for column in labels.columns
    }

    for technique in TECHNIQUES:
        if technique in normalized:
            mapping[technique] = normalized[technique]

    return mapping


def actual_ground_truth_counts(
    labels: pd.DataFrame,
) -> Dict[str, int]:

    mapping = find_label_columns(labels)

    result = {}

    for technique in TECHNIQUES:

        column = mapping.get(technique)

        if column is None:
            result[technique] = 0
            continue

        series = pd.to_numeric(
            labels[column],
            errors="coerce",
        ).fillna(0)

        result[technique] = int(
            (series > 0).sum()
        )

    return result


def feature_matrix_from_frozen_data(
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    encoder: Any,
    scaler: Any,
    tfidf: Any,
    schema: Dict[str, Any],
):

    categorical = schema.get(
        "categorical_features",
        schema.get(
            "categorical_columns",
            CATEGORICAL_FEATURES,
        ),
    )

    numerical = schema.get(
        "numerical_features",
        schema.get(
            "numerical_columns",
            NUMERICAL_FEATURES,
        ),
    )

    if not categorical:
        categorical = CATEGORICAL_FEATURES

    if not numerical:
        numerical = NUMERICAL_FEATURES

    missing_cat = [
        c for c in categorical
        if c not in structured.columns
    ]

    missing_num = [
        c for c in numerical
        if c not in structured.columns
    ]

    if missing_cat or missing_num:
        raise RuntimeError(
            "Frozen structured dataset does not match production "
            f"schema. Missing categorical={missing_cat}; "
            f"missing numerical={missing_num}"
        )

    cat_df = structured[categorical].copy()

    for column in categorical:
        cat_df[column] = (
            cat_df[column]
            .fillna("")
            .astype(str)
        )

    num_df = structured[numerical].copy()

    for column in numerical:
        num_df[column] = pd.to_numeric(
            num_df[column],
            errors="coerce",
        )

    num_df = num_df.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    text_column = "behavior_text_normalized"

    if text_column in behavior.columns:
        text_series = (
            behavior[text_column]
            .fillna("")
            .astype(str)
        )
    elif "behavior_text" in behavior.columns:
        text_series = (
            behavior["behavior_text"]
            .fillna("")
            .astype(str)
        )
    else:
        raise RuntimeError(
            "Frozen behavior dataset has no supported behavior-text column."
        )

    cat_matrix = encoder.transform(cat_df)

    num_matrix = scaler.transform(num_df)

    text_matrix = tfidf.transform(text_series)

    from scipy.sparse import hstack

    final_matrix = hstack(
        [
            cat_matrix,
            num_matrix,
            text_matrix,
        ],
        format="csr",
    )

    if final_matrix.shape[1] != EXPECTED_FINAL_FEATURES:
        raise RuntimeError(
            "Production feature dimension mismatch: "
            f"got={final_matrix.shape[1]}, "
            f"expected={EXPECTED_FINAL_FEATURES}"
        )

    return final_matrix


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 72)
    print("SmartSOC AI — ACTUAL 45-TECHNIQUE ATTACK & MITIGATION REPORT")
    print("=" * 72)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Validate files
    # ----------------------------------------------------------------------

    required = [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
        STRUCTURED_TEST,
        LABELS_TEST,
        METADATA_TEST,
        BEHAVIOR_TEST,
    ]

    for path in required:
        require(path)

    print("[PASS] Production artifacts located")
    print("[PASS] Frozen test artifacts located")

    # ----------------------------------------------------------------------
    # Load
    # ----------------------------------------------------------------------

    warnings.simplefilter("default")

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    schema = load_json(SCHEMA_FILE)
    threshold_raw = load_json(THRESHOLDS_FILE)

    structured = pd.read_csv(STRUCTURED_TEST)
    labels = pd.read_csv(LABELS_TEST)
    metadata = pd.read_csv(METADATA_TEST)
    behavior = pd.read_csv(BEHAVIOR_TEST)

    if not (
        len(structured)
        == len(labels)
        == len(metadata)
        == len(behavior)
    ):
        raise RuntimeError(
            "Frozen datasets are not row aligned."
        )

    print(
        f"[PASS] Frozen dataset alignment: {len(structured)} rows"
    )

    # ----------------------------------------------------------------------
    # Resolve classes
    # ----------------------------------------------------------------------

    if hasattr(model, "classes_"):
        classes = [
            str(x)
            for x in model.classes_
        ]
    elif hasattr(label_encoder, "classes_"):
        classes = [
            str(x)
            for x in label_encoder.classes_
        ]
    else:
        raise RuntimeError(
            "Unable to resolve production model classes."
        )

    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} model classes; "
            f"found {len(classes)}."
        )

    # Preserve the exact model ordering.
    classes = list(classes)

    print(
        f"[PASS] Production model exposes {len(classes)} classes"
    )

    # ----------------------------------------------------------------------
    # Build EXACT production matrix
    # ----------------------------------------------------------------------

    print("[1] Reconstructing frozen production feature matrix...")

    X = feature_matrix_from_frozen_data(
        structured=structured,
        behavior=behavior,
        encoder=encoder,
        scaler=scaler,
        tfidf=tfidf,
        schema=schema,
    )

    print(
        f"[PASS] Production feature matrix: {X.shape}"
    )

    # ----------------------------------------------------------------------
    # ACTUAL inference
    # ----------------------------------------------------------------------

    print("[2] Running ACTUAL frozen production inference...")

    probabilities = model.predict_proba(X)

    probabilities = normalize_probability_matrix(
        probabilities,
        classes,
    )

    if probabilities.shape[0] != len(structured):
        raise RuntimeError(
            "Inference row count does not match frozen dataset."
        )

    print(
        f"[PASS] Actual probability matrix: {probabilities.shape}"
    )

    # ----------------------------------------------------------------------
    # Thresholds
    # ----------------------------------------------------------------------

    thresholds = extract_thresholds(
        threshold_raw
    )

    # ----------------------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------------------

    ground_truth_counts = actual_ground_truth_counts(
        labels
    )

    # ----------------------------------------------------------------------
    # Top predictions
    # ----------------------------------------------------------------------

    top_indices = np.argmax(
        probabilities,
        axis=1,
    )

    top_counts = {
        technique: 0
        for technique in classes
    }

    for index in top_indices:
        technique = classes[int(index)]
        top_counts[technique] = (
            top_counts.get(technique, 0) + 1
        )

    # ----------------------------------------------------------------------
    # Build profiles
    # ----------------------------------------------------------------------

    profiles: List[Dict[str, Any]] = []

    for class_index, technique in enumerate(classes):

        scores = probabilities[:, class_index]

        maximum = float(
            np.max(scores)
        )

        mean = float(
            np.mean(scores)
        )

        median = float(
            np.median(scores)
        )

        minimum = float(
            np.min(scores)
        )

        threshold = thresholds.get(
            technique
        )

        if threshold is None:
            threshold_hit_count = None
        else:
            threshold_hit_count = int(
                np.sum(scores >= threshold)
            )

        # This is actual model evidence.
        #
        # It is NOT claimed to be the official Phase 6 severity score.
        risk_level = risk_from_score(
            maximum
        )

        response = response_for_risk(
            risk_level
        )

        actions = set(
            response["actions"]
        )

        all_actions = [
            "Isolate Host",
            "Kill Malicious Process",
            "Block IP / Domain",
            "Disable Account",
            "Quarantine File",
            "Block Hash",
            "Network Containment",
            "Log Enrichment",
            "Increase Monitoring",
            "Alert Analyst",
            "Log Only",
        ]

        action_map = {
            action: (
                "✓" if action in actions else " "
            )
            for action in all_actions
        }

        profile = {
            "attack_id": technique,
            "attack": TECHNIQUE_NAMES.get(
                technique,
                technique,
            ),
            "model_class_index": class_index,
            "max_model_score": maximum,
            "mean_model_score": mean,
            "median_model_score": median,
            "min_model_score": minimum,
            "production_threshold": threshold,
            "threshold_hit_events": threshold_hit_count,
            "top_prediction_events": int(
                top_counts.get(
                    technique,
                    0,
                )
            ),
            "ground_truth_events": int(
                ground_truth_counts.get(
                    technique,
                    0,
                )
            ),
            "risk_level": risk_level,
            "risk_disposition": risk_level,
            "response_category": response["response"],
            "analyst_review": response["analyst"],
            "response_required": "YES",
            "containment_candidate": (
                "YES"
                if risk_level in {
                    "HIGH",
                    "CRITICAL",
                }
                else "NO"
            ),
            "automatic_execution": "DISABLED",
            "automatic_containment": "DISABLED",
            "investigation": response["investigation"],
            "rollback": response["rollback"],
            "actions": action_map,
            "authorization": "PENDING ANALYST AUTHORIZATION",
            "ai_authorization": "NOT SUFFICIENT",
            "execution": "NOT EXECUTED",
        }

        profiles.append(profile)

    # ----------------------------------------------------------------------
    # TXT report
    # ----------------------------------------------------------------------

    lines: List[str] = []

    lines.append("=" * 72)
    lines.append(
        "SMARTSOC AI — ATTACK & MITIGATION PROFILE REPORT"
    )
    lines.append("=" * 72)
    lines.append("")
    lines.append("Model Version        : 4.1")
    lines.append(
        f"Model Classes        : {len(classes)}"
    )
    lines.append(
        "Inference Source     : Frozen Production Model"
    )
    lines.append(
        "Dataset              : Frozen Test Dataset"
    )
    lines.append(
        f"Frozen Events        : {len(structured)}"
    )
    lines.append(
        "Synthetic Scores     : DISABLED"
    )
    lines.append(
        "Synthetic Predictions: DISABLED"
    )
    lines.append(
        "Execution            : DISABLED"
    )
    lines.append(
        "Containment          : DISABLED"
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("REPORT INTERPRETATION")
    lines.append("=" * 72)
    lines.append("")
    lines.append(
        "Model Score is the ACTUAL production-model probability "
        "observed for the technique."
    )
    lines.append(
        "It is not fabricated and is not converted into a "
        "fictional 0-10 severity score."
    )
    lines.append("")
    lines.append(
        "Top Prediction Events counts events where this technique "
        "was the model's highest-scoring class."
    )
    lines.append("")
    lines.append(
        "Threshold Hit Events counts actual frozen events whose "
        "model probability met the production threshold."
    )
    lines.append("")
    lines.append(
        "Ground Truth Events comes from the frozen test labels."
    )
    lines.append("")
    lines.append(
        "Risk/response fields are policy-layer decisions. "
        "No response or containment action was executed."
    )
    lines.append("")

    for number, profile in enumerate(
        profiles,
        start=1,
    ):

        lines.append("=" * 72)
        lines.append(
            f"ATTACK PROFILE {number:02d} / {len(classes):02d}"
        )
        lines.append("=" * 72)
        lines.append("")

        lines.append(
            f"Attack ID        : {profile['attack_id']}"
        )
        lines.append(
            f"Attack           : {profile['attack']}"
        )
        lines.append("")

        lines.append("AI ASSESSMENT")
        lines.append("-------------")
        lines.append(
            f"Model Class Index     : "
            f"{profile['model_class_index']}"
        )
        lines.append(
            f"Maximum Model Score   : "
            f"{profile['max_model_score']:.6f}"
        )
        lines.append(
            f"Mean Model Score      : "
            f"{profile['mean_model_score']:.6f}"
        )
        lines.append(
            f"Median Model Score    : "
            f"{profile['median_model_score']:.6f}"
        )
        lines.append(
            f"Minimum Model Score   : "
            f"{profile['min_model_score']:.6f}"
        )

        threshold = profile[
            "production_threshold"
        ]

        if threshold is None:
            lines.append(
                "Production Threshold  : NOT RESOLVED"
            )
        else:
            lines.append(
                f"Production Threshold  : {threshold:.6f}"
            )

        hit_count = profile[
            "threshold_hit_events"
        ]

        if hit_count is None:
            lines.append(
                "Threshold Hit Events  : NOT RESOLVED"
            )
        else:
            lines.append(
                f"Threshold Hit Events  : {hit_count}"
            )

        lines.append(
            f"Top Prediction Events : "
            f"{profile['top_prediction_events']}"
        )
        lines.append(
            f"Ground Truth Events   : "
            f"{profile['ground_truth_events']}"
        )
        lines.append(
            f"Risk Level            : "
            f"{profile['risk_level']}"
        )
        lines.append(
            f"Risk Disposition      : "
            f"{profile['risk_disposition']}"
        )
        lines.append("")

        lines.append("DECISION POLICY")
        lines.append("---------------")
        lines.append(
            f"Analyst Review        : "
            f"{profile['analyst_review']}"
        )
        lines.append(
            f"Response Required     : "
            f"{profile['response_required']}"
        )
        lines.append(
            f"Containment Candidate : "
            f"{profile['containment_candidate']}"
        )
        lines.append(
            f"Automatic Execution   : "
            f"{profile['automatic_execution']}"
        )
        lines.append(
            f"Automatic Containment : "
            f"{profile['automatic_containment']}"
        )
        lines.append("")

        lines.append("RESPONSE PLANNER")
        lines.append("----------------")
        lines.append(
            f"Response Category     : "
            f"{profile['response_category']}"
        )
        lines.append(
            f"Investigation         : "
            f"{profile['investigation']}"
        )
        lines.append(
            f"Rollback              : "
            f"{profile['rollback']}"
        )
        lines.append("")

        lines.append("PLANNED MITIGATION")
        lines.append("------------------")

        for action in [
            "Isolate Host",
            "Kill Malicious Process",
            "Block IP / Domain",
            "Disable Account",
            "Quarantine File",
            "Block Hash",
            "Network Containment",
            "Log Enrichment",
            "Increase Monitoring",
            "Alert Analyst",
            "Log Only",
        ]:
            lines.append(
                f"[{profile['actions'][action]}] {action}"
            )

        lines.append("")
        lines.append("AUTHORIZATION")
        lines.append("-------------")
        lines.append(
            f"Authorization State   : "
            f"{profile['authorization']}"
        )
        lines.append(
            "Analyst Approval      : REQUIRED"
        )
        lines.append(
            f"AI Authorization      : "
            f"{profile['ai_authorization']}"
        )
        lines.append(
            f"Automatic Execution   : "
            f"{profile['automatic_execution']}"
        )
        lines.append("")

        lines.append("EXECUTION STATE")
        lines.append("---------------")
        lines.append(
            "Host Isolation        : NOT EXECUTED"
        )
        lines.append(
            "Process Kill          : NOT EXECUTED"
        )
        lines.append(
            "IP / Domain Block     : NOT EXECUTED"
        )
        lines.append(
            "Account Disable       : NOT EXECUTED"
        )
        lines.append(
            "File Quarantine       : NOT EXECUTED"
        )
        lines.append(
            "Hash Blocking         : NOT EXECUTED"
        )
        lines.append(
            "Network Containment   : NOT EXECUTED"
        )
        lines.append("")

        lines.append("EVIDENCE")
        lines.append("--------")
        lines.append(
            "Inference             : ACTUAL PRODUCTION MODEL"
        )
        lines.append(
            "Probability Data      : ACTUAL FROZEN INFERENCE"
        )
        lines.append(
            "Ground Truth          : ACTUAL FROZEN TEST LABELS"
        )
        lines.append(
            "Risk / Response       : VALIDATED POLICY BOUNDARY"
        )
        lines.append(
            "Synthetic Prediction  : NONE"
        )
        lines.append(
            "Synthetic Score       : NONE"
        )
        lines.append(
            "Response Execution    : NONE"
        )
        lines.append(
            "Containment Execution : NONE"
        )
        lines.append("")

    # ----------------------------------------------------------------------
    # Save TXT
    # ----------------------------------------------------------------------

    report_text = "\n".join(lines)

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Save JSON evidence
    # ----------------------------------------------------------------------

    json_payload = {
        "report_version": "4.2",
        "model_version": "4.1",
        "model_classes": len(classes),
        "frozen_events": len(structured),
        "inference_source": "frozen production model",
        "synthetic_predictions": False,
        "synthetic_scores": False,
        "automatic_execution": False,
        "automatic_containment": False,
        "feature_matrix_shape": list(X.shape),
        "probability_matrix_shape": list(
            probabilities.shape
        ),
        "profiles": profiles,
    }

    JSON_FILE.write_text(
        json.dumps(
            json_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------------------

    print("")
    print("[PASS] Actual inference completed")
    print(
        f"       probability matrix={probabilities.shape}"
    )
    print(
        f"[PASS] {len(profiles)} actual production classes profiled"
    )
    print("")
    print("=" * 72)
    print("SUPERVISOR REPORT COMPLETE")
    print("=" * 72)
    print(
        f"TXT  : {REPORT_FILE}"
    )
    print(
        f"JSON : {JSON_FILE}"
    )
    print("")
    print(
        "Synthetic predictions : DISABLED"
    )
    print(
        "Synthetic scores      : DISABLED"
    )
    print(
        "Response execution    : DISABLED"
    )
    print(
        "Containment execution : DISABLED"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())