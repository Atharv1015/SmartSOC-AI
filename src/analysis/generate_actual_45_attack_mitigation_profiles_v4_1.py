from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC v4.1
# ACTUAL 45-TECHNIQUE ATTACK & MITIGATION PROFILE REPORT
#
# PURPOSE
# -------
# Produce one supervisor-facing TXT profile for every technique exposed by
# the frozen production model.
#
# IMPORTANT
# ---------
# This report uses the ACTUAL SmartSOC Phase 6.7 inference implementation.
#
# For each technique:
#
#   frozen event
#       |
#       v
#   production preprocessing
#       |
#       v
#   serialized production model
#       |
#       v
#   actual model probability
#       |
#       v
#   Phase 6.7 risk engine
#       |
#       v
#   decision policy
#       |
#       v
#   response planner
#       |
#       v
#   analyst authorization boundary
#
# NO synthetic prediction.
# NO synthetic severity score.
# NO synthetic risk score.
# NO fabricated mitigation decision.
# NO response execution.
# NO containment execution.
#
# The representative event for each profile MUST be an actual frozen test
# event for which that technique is the model's top prediction.
#
# If a technique is not actually produced as a top prediction anywhere in
# the frozen dataset, the script FAILS rather than inventing a result.
# =============================================================================


VERSION = "4.1"

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "supervisor"
)

OUTPUT_FILE = (
    REPORT_DIR
    / "smartsoc_actual_45_attack_mitigation_profiles_v4_1.txt"
)

PHASE67_FILE = (
    ROOT
    / "src"
    / "analysis"
    / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py"
)

RISK_ENGINE_FILE = (
    ROOT
    / "src"
    / "decision"
    / "smartsoc_risk_engine_v4_1.py"
)

DECISION_POLICY_FILE = (
    ROOT
    / "src"
    / "decision"
    / "smartsoc_decision_policy_v4_1.py"
)

RESPONSE_PLANNER_FILE = (
    ROOT
    / "src"
    / "decision"
    / "smartsoc_response_planner_v4_1.py"
)

AUTHORIZATION_FILE = (
    ROOT
    / "src"
    / "decision"
    / "smartsoc_analyst_authorization_v4_1.py"
)


MODEL_FILE = MODEL_DIR / "model.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
MANIFEST_FILE = MODEL_DIR / "model_manifest.json"


# =============================================================================
# Existing project technique names
# =============================================================================

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
    "T1543.003": "Create or Modify System Process: Windows Service",
    "T1546": "Event Triggered Execution",
    "T1546.003": (
        "Event Triggered Execution: "
        "Windows Management Instrumentation Event Subscription"
    ),
    "T1546.011": (
        "Event Triggered Execution: "
        "Application Shimming"
    ),
    "T1547.001": "Registry Run Keys / Startup Folder",
    "T1548": "Abuse Elevation Control Mechanism",
    "T1548.002": "Bypass User Account Control",
    "T1562.001": "Impair Defenses: Disable or Modify Tools",
    "T1562.004": "Impair Defenses: Disable or Modify System Firewall",
    "T1565.001": "Data Manipulation",
    "T1570": "Lateral Tool Transfer",
    "T1574.001": "DLL Search Order Hijacking",
    "T1574.002": "DLL Side-Loading",
    "T1574.010": "Services File Permissions Weakness",
}


# =============================================================================
# Formatting
# =============================================================================

WIDTH = 72


def divider(char: str = "=") -> str:
    return char * WIDTH


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        if not np.isfinite(value):
            return ""

    return str(value)


def pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def score(value: float) -> str:
    return f"{value:.2f}"


def bool_text(value: bool) -> str:
    return "YES" if bool(value) else "NO"


# =============================================================================
# Dynamic module loading
# =============================================================================

def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(str(path))

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module: {path}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# =============================================================================
# Dataset / artifacts
# =============================================================================

def load_artifacts() -> Dict[str, Any]:

    required = {
        "model": MODEL_FILE,
        "schema": SCHEMA_FILE,
        "thresholds": THRESHOLDS_FILE,
        "encoder": ENCODER_FILE,
        "scaler": SCALER_FILE,
        "tfidf": TFIDF_FILE,
        "label_encoder": LABEL_ENCODER_FILE,
        "manifest": MANIFEST_FILE,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing production artifacts:\n"
            + "\n".join(missing)
        )

    return {
        "model": joblib.load(MODEL_FILE),
        "encoder": joblib.load(ENCODER_FILE),
        "scaler": joblib.load(SCALER_FILE),
        "tfidf": joblib.load(TFIDF_FILE),
        "label_encoder": joblib.load(LABEL_ENCODER_FILE),
        "schema": json.loads(
            SCHEMA_FILE.read_text(
                encoding="utf-8"
            )
        ),
        "thresholds": json.loads(
            THRESHOLDS_FILE.read_text(
                encoding="utf-8"
            )
        ),
        "manifest": json.loads(
            MANIFEST_FILE.read_text(
                encoding="utf-8"
            )
        ),
    }


def load_dataset():

    structured = pd.read_csv(
        DATA_DIR / "structured_test.csv"
    )

    labels = pd.read_csv(
        DATA_DIR / "labels_test.csv"
    )

    metadata = pd.read_csv(
        DATA_DIR / "metadata_test.csv"
    )

    behavior = pd.read_csv(
        DATA_DIR / "behavior_text_test.csv"
    )

    if not (
        len(structured)
        == len(labels)
        == len(metadata)
        == len(behavior)
    ):
        raise ValueError(
            "Frozen dataset alignment failure."
        )

    return (
        structured,
        labels,
        metadata,
        behavior,
    )


# =============================================================================
# Model classes
# =============================================================================

def get_model_classes(
    artifacts: Dict[str, Any],
) -> List[str]:

    label_encoder = artifacts["label_encoder"]

    classes = getattr(
        label_encoder,
        "classes_",
        None,
    )

    if classes is None:
        raise ValueError(
            "Production label encoder does not expose classes_."
        )

    result = [
        str(x)
        for x in classes
    ]

    if len(result) != 45:
        raise ValueError(
            f"Expected 45 model classes, found {len(result)}."
        )

    return result


# =============================================================================
# Exact Phase 6.7 inference
# =============================================================================

def actual_event_prediction(
    phase67,
    row: pd.Series,
    behavior_text: str,
    artifacts: Dict[str, Any],
    categorical_features: List[str],
    numerical_features: List[str],
):

    matrix, _ = phase67.transform_event(
        row,
        behavior_text,
        artifacts,
        categorical_features,
        numerical_features,
    )

    prediction = phase67.predict_event(
        matrix,
        artifacts,
        phase67.resolve_threshold_mapping(
            artifacts["thresholds"]
        ),
    )

    return matrix, prediction


# =============================================================================
# Actual event evaluation through Phase 6.1 -> 6.7 architecture
# =============================================================================

def evaluate_actual_event(
    phase67,
    risk_module,
    decision_module,
    response_module,
    authorization_module,
    row: pd.Series,
    behavior_text: str,
    artifacts: Dict[str, Any],
    categorical_features: List[str],
    numerical_features: List[str],
) -> Dict[str, Any]:

    matrix, prediction = actual_event_prediction(
        phase67,
        row,
        behavior_text,
        artifacts,
        categorical_features,
        numerical_features,
    )

    technique = str(
        prediction["top_technique"]
    )

    confidence = float(
        prediction["top_probability"]
    )

    margin = float(
        prediction["margin"]
    )

    fallback = bool(
        prediction["fallback"]
    )

    ambiguous = (
        margin < 0.05
    )

    evidence_strength, evidence = (
        phase67.assess_evidence(
            row,
            behavior_text,
        )
    )

    missing_behavior = not bool(
        evidence[
            "behavior_text_present"
        ]
    )

    context = phase67.derive_context(
        row
    )

    technique_severity = (
        phase67.severity_for(
            technique
        )
    )

    # -------------------------------------------------------------------------
    # ACTUAL RISK ENGINE
    # -------------------------------------------------------------------------

    risk_result = (
        risk_module.calculate(
            technique_severity=(
                technique_severity
            ),
            ml_confidence=confidence,
            evidence_strength=evidence_strength,
            privilege=context["privilege"],
            network=context["network"],
            suspicious_parent=context[
                "suspicious_parent"
            ],
            fallback=fallback,
            ambiguous=ambiguous,
            missing_behavior=missing_behavior,
        )
    )

    risk_score = float(
        risk_result.risk_score
    )

    risk_level = str(
        risk_result.risk_level
    )

    recommended_action = str(
        risk_result.recommended_action
    )

    # -------------------------------------------------------------------------
    # ACTUAL DECISION POLICY
    # -------------------------------------------------------------------------

    decision_event = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "fallback": fallback,
        "ambiguous": ambiguous,
        "missing_behavior": missing_behavior,
        "decision": recommended_action,
    }

    decision_result = (
        decision_module.SmartSOCDecisionPolicy()
        .evaluate_event(
            decision_event
        )
    )

    # -------------------------------------------------------------------------
    # ACTUAL RESPONSE PLANNER
    # -------------------------------------------------------------------------

    response_event = {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "decision": str(
            decision_result.decision
        ),
        "fallback": fallback,
        "ambiguous": ambiguous,
        "missing_behavior": missing_behavior,
        "techniques": [
            technique
        ],
    }

    response_result = (
        response_module.SmartSOCResponsePlanner()
        .evaluate_event(
            response_event
        )
    )

    # -------------------------------------------------------------------------
    # ACTUAL ANALYST AUTHORIZATION
    # -------------------------------------------------------------------------

    authorization_result = (
        authorization_module.SmartSOCAnalystAuthorization()
        .evaluate(
            risk_score=risk_score,
            risk_level=risk_level,
            decision=str(
                decision_result.decision
            ),
            fallback=fallback,
            ambiguous=ambiguous,
            missing_behavior=missing_behavior,
        )
    )

    return {
        "matrix": matrix,
        "prediction": prediction,
        "technique": technique,
        "confidence": confidence,
        "margin": margin,
        "fallback": fallback,
        "ambiguous": ambiguous,
        "missing_behavior": missing_behavior,
        "evidence_strength": evidence_strength,
        "evidence": evidence,
        "context": context,
        "technique_severity": technique_severity,
        "risk": risk_result,
        "decision": decision_result,
        "response": response_result,
        "authorization": authorization_result,
    }


# =============================================================================
# Find REAL representative event
#
# Priority:
#   1. Technique is actual top prediction.
#   2. Highest actual model probability for that technique.
#
# We do not substitute a synthetic score.
# =============================================================================

def find_representative_events(
    phase67,
    risk_module,
    decision_module,
    response_module,
    authorization_module,
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    artifacts: Dict[str, Any],
    categorical_features: List[str],
    numerical_features: List[str],
    classes: List[str],
) -> Dict[str, Dict[str, Any]]:

    best: Dict[str, Dict[str, Any]] = {}

    thresholds = phase67.resolve_threshold_mapping(
        artifacts["thresholds"]
    )

    for index in range(
        len(structured)
    ):

        row = structured.iloc[index]

        behavior_text = (
            phase67.behavior_text_at(
                behavior,
                index,
            )
        )

        matrix, prediction = (
            actual_event_prediction(
                phase67,
                row,
                behavior_text,
                artifacts,
                categorical_features,
                numerical_features,
            )
        )

        technique = str(
            prediction["top_technique"]
        )

        confidence = float(
            prediction["top_probability"]
        )

        # Only a real top prediction is accepted.
        if technique not in classes:
            continue

        candidate = {
            "index": index,
            "event_id": safe_text(
                row.get(
                    "event_id",
                    index,
                )
            ),
            "confidence": confidence,
            "prediction": prediction,
            "matrix": matrix,
            "behavior_text": behavior_text,
        }

        previous = best.get(
            technique
        )

        if (
            previous is None
            or confidence
            > previous["confidence"]
        ):
            best[technique] = candidate

    return best


# =============================================================================
# Planner action mapping
# =============================================================================

ACTION_TO_DISPLAY = {
    "PREPARE_HOST_ISOLATION":
        "Isolate Host",

    "PREPARE_PROCESS_ISOLATION":
        "Kill Malicious Process",

    "PREPARE_IP_BLOCK":
        "Block IP / Domain",

    "PREPARE_ACCOUNT_DISABLE":
        "Disable Account",

    "PRESERVE_ARTIFACTS":
        "Quarantine / Preserve File Evidence",

    "BLOCK_HASH":
        "Block Hash",

    "PREPARE_NETWORK_CONTAINMENT":
        "Network Containment",

    "COLLECT_ADDITIONAL_EVIDENCE":
        "Log Enrichment",

    "INCREASE_MONITORING":
        "Increase Monitoring",

    "ALERT_ANALYST":
        "Alert Analyst",
}


def planner_actions(
    response_result: Any,
) -> List[str]:

    actions = getattr(
        response_result,
        "recommended_actions",
        [],
    )

    if actions is None:
        return []

    return [
        str(x)
        for x in actions
    ]


def planned_check(
    actions: List[str],
    action_name: str,
) -> str:

    return (
        "[✓]"
        if action_name in actions
        else "[ ]"
    )


# =============================================================================
# Risk-policy presentation
#
# This mirrors the project's validated response policy while the actual
# values above come from the actual event evaluation.
# =============================================================================

def policy_interpretation(
    risk_level: str,
    decision_result: Any,
    response_result: Any,
) -> Dict[str, str]:

    level = risk_level.upper()

    analyst_required = bool(
        getattr(
            decision_result,
            "analyst_required",
            False,
        )
    )

    containment_candidate = bool(
        getattr(
            response_result,
            "containment_candidate",
            False,
        )
    )

    rollback_required = bool(
        getattr(
            response_result,
            "rollback_required",
            False,
        )
    )

    if level == "LOW":

        response_category = (
            "OBSERVE-ONLY / MONITORING"
        )

        investigation = (
            "COLLECT ADDITIONAL EVIDENCE"
        )

        log_only = "YES"

    elif level == "MEDIUM":

        response_category = (
            "ANALYST REVIEW / INVESTIGATION"
        )

        investigation = "REQUIRED"

        log_only = "NO AUTOMATED ACTION"

    elif level == "HIGH":

        response_category = (
            "HIGH-RISK RESPONSE"
        )

        investigation = "REQUIRED"

        log_only = "NO AUTOMATED ACTION"

    elif level == "CRITICAL":

        response_category = (
            "CRITICAL RESPONSE / "
            "CONTAINMENT CANDIDATE"
        )

        investigation = "REQUIRED"

        log_only = "NO AUTOMATED ACTION"

    else:

        response_category = (
            "SAFETY FALLBACK"
        )

        investigation = "REQUIRED"

        log_only = "SAFETY FALLBACK"

    return {
        "response_category":
            response_category,

        "investigation":
            investigation,

        "rollback":
            (
                "REQUIRED WHERE APPLICABLE"
                if rollback_required
                else "NOT REQUIRED BY RISK LEVEL"
            ),

        "analyst_review":
            (
                "REQUIRED"
                if analyst_required
                else "NOT REQUIRED"
            ),

        "containment_candidate":
            (
                "YES"
                if containment_candidate
                else "NO"
            ),

        "log_only":
            log_only,
    }


# =============================================================================
# TXT profile
# =============================================================================

def build_profile(
    number: int,
    total: int,
    technique: str,
    attack_name: str,
    event_id: str,
    event_index: int,
    evaluated: Dict[str, Any],
) -> str:

    prediction = evaluated["prediction"]

    risk = evaluated["risk"]

    decision = evaluated["decision"]

    response = evaluated["response"]

    authorization = evaluated[
        "authorization"
    ]

    confidence = evaluated[
        "confidence"
    ]

    margin = evaluated[
        "margin"
    ]

    risk_score = float(
        getattr(
            risk,
            "risk_score",
        )
    )

    risk_level = str(
        getattr(
            risk,
            "risk_level",
        )
    )

    recommended_action = str(
        getattr(
            risk,
            "recommended_action",
        )
    )

    decision_name = str(
        getattr(
            decision,
            "decision",
        )
    )

    response_mode = str(
        getattr(
            response,
            "response_mode",
        )
    )

    priority = str(
        getattr(
            response,
            "priority",
        )
    )

    actions = planner_actions(
        response
    )

    policy = policy_interpretation(
        risk_level,
        decision,
        response,
    )

    execution_permitted = bool(
        getattr(
            authorization,
            "execution_permitted",
            False,
        )
    )

    containment_authorized = bool(
        getattr(
            authorization,
            "containment_authorized",
            False,
        )
    )

    analyst_required = bool(
        getattr(
            authorization,
            "analyst_required",
            False,
        )
    )

    lines = []

    lines.append(
        divider()
    )

    lines.append(
        f"ATTACK PROFILE {number:02d} / {total:02d}"
    )

    lines.append(
        divider()
    )

    lines.append("")

    lines.append(
        f"Attack ID        : {technique}"
    )

    lines.append(
        f"Attack           : {attack_name}"
    )

    lines.append(
        f"Representative Event : {event_id}"
    )

    lines.append(
        f"Frozen Dataset Index : {event_index}"
    )

    lines.append("")

    lines.append(
        "AI ASSESSMENT"
    )

    lines.append(
        "-------------"
    )

    lines.append(
        f"Predicted Class       : {technique}"
    )

    lines.append(
        f"Model Score           : {confidence:.6f}"
    )

    lines.append(
        f"Model Score (%)       : {pct(confidence)}"
    )

    lines.append(
        f"Second Class Score    : "
        f"{float(prediction['second_probability']):.6f}"
    )

    lines.append(
        f"Prediction Margin     : {margin:.6f}"
    )

    lines.append(
        f"Threshold             : "
        f"{(
            prediction['top5'][0]['threshold']
            if prediction['top5']
            and prediction['top5'][0]['threshold'] is not None
            else 'N/A'
        )}"
    )

    lines.append(
        f"Threshold Detected    : "
        f"{bool_text(
            technique
            in prediction['detected_techniques']
        )}"
    )

    lines.append(
        f"Technique Severity    : "
        f"{evaluated['technique_severity']}"
    )

    lines.append(
        f"Evidence Strength     : "
        f"{evaluated['evidence_strength']}"
    )

    lines.append(
        f"Evidence Signals      : "
        f"{evaluated['evidence']['signal_count']}"
    )

    lines.append(
        f"Behavior Text Present : "
        f"{bool_text(
            evaluated['evidence']
            ['behavior_text_present']
        )}"
    )

    lines.append(
        f"Privilege Context     : "
        f"{evaluated['context']['privilege']}"
    )

    lines.append(
        f"Network Context       : "
        f"{bool_text(
            evaluated['context']['network']
        )}"
    )

    lines.append(
        f"Suspicious Parent     : "
        f"{bool_text(
            evaluated['context']['suspicious_parent']
        )}"
    )

    lines.append("")

    lines.append(
        "RISK ENGINE"
    )

    lines.append(
        "-----------"
    )

    lines.append(
        f"Severity / Risk Score : {risk_score:.2f}"
    )

    lines.append(
        f"Risk Level            : {risk_level}"
    )

    lines.append(
        f"Recommended Action    : {recommended_action}"
    )

    lines.append("")

    lines.append(
        "DECISION POLICY"
    )

    lines.append(
        "---------------"
    )

    lines.append(
        f"Decision              : {decision_name}"
    )

    lines.append(
        f"Analyst Review        : "
        f"{policy['analyst_review']}"
    )

    lines.append(
        f"Response Required     : "
        f"{'YES' if risk_level != 'LOW' else 'MONITOR'}"
    )

    lines.append(
        f"Containment Candidate : "
        f"{policy['containment_candidate']}"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append(
        "Automatic Containment : DISABLED"
    )

    lines.append("")

    lines.append(
        "RESPONSE PLANNER"
    )

    lines.append(
        "----------------"
    )

    lines.append(
        f"Response Category     : "
        f"{policy['response_category']}"
    )

    lines.append(
        f"Response Mode         : "
        f"{response_mode}"
    )

    lines.append(
        f"Priority              : "
        f"{priority}"
    )

    lines.append(
        f"Investigation         : "
        f"{policy['investigation']}"
    )

    lines.append(
        f"Rollback              : "
        f"{policy['rollback']}"
    )

    lines.append(
        "Recommended Planner Actions:"
    )

    if actions:

        for action in actions:

            display = ACTION_TO_DISPLAY.get(
                action,
                action,
            )

            lines.append(
                f"  [✓] {display}"
            )

    else:

        lines.append(
            "  [✓] No disruptive action; "
            "evidence/review workflow"
        )

    lines.append("")

    lines.append(
        "PLANNED MITIGATION"
    )

    lines.append(
        "------------------"
    )

    # These are explicitly derived from the planner's actual action list.
    lines.append(
        f"{planned_check(actions, 'PREPARE_HOST_ISOLATION')} "
        "Isolate Host"
    )

    lines.append(
        f"{planned_check(actions, 'PREPARE_PROCESS_ISOLATION')} "
        "Kill Malicious Process"
    )

    lines.append(
        f"{planned_check(actions, 'PREPARE_IP_BLOCK')} "
        "Block IP / Domain"
    )

    lines.append(
        f"{planned_check(actions, 'PREPARE_ACCOUNT_DISABLE')} "
        "Disable Account"
    )

    lines.append(
        f"{planned_check(actions, 'PRESERVE_ARTIFACTS')} "
        "Quarantine / Preserve File Evidence"
    )

    lines.append(
        f"{planned_check(actions, 'BLOCK_HASH')} "
        "Block Hash"
    )

    lines.append(
        f"{planned_check(actions, 'PREPARE_NETWORK_CONTAINMENT')} "
        "Network Containment"
    )

    lines.append(
        f"{planned_check(actions, 'COLLECT_ADDITIONAL_EVIDENCE')} "
        "Log Enrichment / Evidence Collection"
    )

    monitoring = (
        risk_level
        in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }
    )

    lines.append(
        f"{'[✓]' if monitoring else '[ ]'} "
        "Increase Monitoring"
    )

    lines.append(
        f"{'[✓]' if analyst_required else '[ ]'} "
        "Alert Analyst"
    )

    lines.append(
        f"{'[✓]' if policy['log_only'] == 'YES' else '[ ]'} "
        "Log Only / No Automated Action"
    )

    lines.append("")

    lines.append(
        "AUTHORIZATION"
    )

    lines.append(
        "-------------"
    )

    lines.append(
        "Authorization State   : "
        + (
            "PENDING ANALYST AUTHORIZATION"
            if analyst_required
            else "NO ANALYST AUTHORIZATION REQUIRED"
        )
    )

    lines.append(
        "Analyst Approval      : "
        + (
            "REQUIRED"
            if analyst_required
            else "NOT REQUIRED"
        )
    )

    lines.append(
        "AI Authorization      : NOT SUFFICIENT"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append(
        "Automatic Containment : DISABLED"
    )

    lines.append(
        f"Execution Permitted   : "
        f"{bool_text(execution_permitted)}"
    )

    lines.append(
        f"Containment Authorized: "
        f"{bool_text(containment_authorized)}"
    )

    lines.append("")

    lines.append(
        "EXECUTION STATE"
    )

    lines.append(
        "---------------"
    )

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

    lines.append(
        "EVIDENCE"
    )

    lines.append(
        "--------"
    )

    lines.append(
        "Inference             : ACTUAL SERIALIZED PRODUCTION MODEL"
    )

    lines.append(
        "Frozen Event          : ACTUAL FROZEN TEST DATASET EVENT"
    )

    lines.append(
        "Model Score           : ACTUAL MODEL PROBABILITY"
    )

    lines.append(
        "Severity / Risk Score : ACTUAL PHASE 6.7 RISK ENGINE OUTPUT"
    )

    lines.append(
        "Risk Level            : ACTUAL PHASE 6.7 RISK ENGINE OUTPUT"
    )

    lines.append(
        "Decision              : ACTUAL DECISION POLICY OUTPUT"
    )

    lines.append(
        "Response              : ACTUAL RESPONSE PLANNER OUTPUT"
    )

    lines.append(
        "Authorization         : ACTUAL AUTHORIZATION ENGINE OUTPUT"
    )

    lines.append(
        "Synthetic Prediction  : NONE"
    )

    lines.append(
        "Synthetic Score       : NONE"
    )

    lines.append(
        "Response Execution    : DISABLED"
    )

    lines.append(
        "Containment Execution : DISABLED"
    )

    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> int:

    print(
        divider()
    )

    print(
        "SmartSOC AI — ACTUAL 45-TECHNIQUE "
        "ATTACK & MITIGATION PROFILE REPORT"
    )

    print(
        divider()
    )

    print()

    try:

        print(
            "[1] Loading production artifacts..."
        )

        artifacts = load_artifacts()

        print(
            "[PASS] Production artifacts loaded"
        )

        print()

        print(
            "[2] Loading frozen test dataset..."
        )

        (
            structured,
            labels,
            metadata,
            behavior,
        ) = load_dataset()

        print(
            f"[PASS] Frozen test dataset loaded: "
            f"{len(structured)} events"
        )

        print(
            f"[PASS] Dataset alignment: "
            f"{len(structured)} / "
            f"{len(labels)} / "
            f"{len(metadata)} / "
            f"{len(behavior)}"
        )

        print()

        print(
            "[3] Loading validated Phase 6.7 inference implementation..."
        )

        phase67 = load_module(
            PHASE67_FILE,
            "smartsoc_phase67_actual_report",
        )

        print(
            "[PASS] Phase 6.7 inference implementation loaded"
        )

        print()

        print(
            "[4] Loading actual decision components..."
        )

        risk_module = load_module(
            RISK_ENGINE_FILE,
            "smartsoc_actual_risk_engine",
        )

        decision_module = load_module(
            DECISION_POLICY_FILE,
            "smartsoc_actual_decision_policy",
        )

        response_module = load_module(
            RESPONSE_PLANNER_FILE,
            "smartsoc_actual_response_planner",
        )

        authorization_module = load_module(
            AUTHORIZATION_FILE,
            "smartsoc_actual_authorization",
        )

        print(
            "[PASS] Risk engine loaded"
        )

        print(
            "[PASS] Decision policy loaded"
        )

        print(
            "[PASS] Response planner loaded"
        )

        print(
            "[PASS] Analyst authorization engine loaded"
        )

        print()

        (
            structured_features,
            categorical_features,
            numerical_features,
        ) = phase67.resolve_schema(
            artifacts["schema"],
            list(
                structured.columns
            ),
        )

        if len(categorical_features) != 4:
            raise ValueError(
                "Expected 4 categorical features; "
                f"found {len(categorical_features)}"
            )

        if len(numerical_features) != 58:
            raise ValueError(
                "Expected 58 numerical features; "
                f"found {len(numerical_features)}"
            )

        print(
            "[PASS] Frozen feature schema resolved"
        )

        print(
            f"       Structured features : "
            f"{len(structured_features)}"
        )

        print(
            f"       Categorical features: "
            f"{len(categorical_features)}"
        )

        print(
            f"       Numerical features  : "
            f"{len(numerical_features)}"
        )

        classes = get_model_classes(
            artifacts
        )

        print()

        print(
            f"[PASS] Production model exposes "
            f"{len(classes)} classes"
        )

        # ---------------------------------------------------------------------
        # Find actual top-prediction events.
        # ---------------------------------------------------------------------

        print()

        print(
            "[5] Searching frozen dataset for actual representative "
            "predictions..."
        )

        representatives = (
            find_representative_events(
                phase67,
                risk_module,
                decision_module,
                response_module,
                authorization_module,
                structured,
                behavior,
                artifacts,
                categorical_features,
                numerical_features,
                classes,
            )
        )

        missing = [
            technique
            for technique in classes
            if technique
            not in representatives
        ]

        if missing:

            raise RuntimeError(
                "The frozen dataset does not contain an actual top "
                "prediction for all 45 model classes.\n"
                "Missing actual top-prediction classes:\n"
                + "\n".join(
                    missing
                )
                + "\n\n"
                "The script intentionally refuses to fabricate "
                "profiles for those classes."
            )

        print(
            f"[PASS] Actual representative event found "
            f"for all {len(classes)} model classes"
        )

        # ---------------------------------------------------------------------
        # Evaluate all 45 selected real events through the actual architecture.
        # ---------------------------------------------------------------------

        profiles = []

        for technique in classes:

            candidate = representatives[
                technique
            ]

            index = int(
                candidate["index"]
            )

            row = structured.iloc[
                index
            ]

            behavior_text = (
                phase67.behavior_text_at(
                    behavior,
                    index,
                )
            )

            evaluated = (
                evaluate_actual_event(
                    phase67,
                    risk_module,
                    decision_module,
                    response_module,
                    authorization_module,
                    row,
                    behavior_text,
                    artifacts,
                    categorical_features,
                    numerical_features,
                )
            )

            # Critical integrity check:
            # The actual evaluated top class MUST equal the selected class.
            if (
                evaluated["technique"]
                != technique
            ):
                raise RuntimeError(
                    "Representative event changed during "
                    "re-evaluation: "
                    f"{technique} -> "
                    f"{evaluated['technique']}"
                )

            profiles.append(
                build_profile(
                    number=len(profiles) + 1,
                    total=len(classes),
                    technique=technique,
                    attack_name=TECHNIQUE_NAMES.get(
                        technique,
                        technique,
                    ),
                    event_id=str(
                        candidate["event_id"]
                    ),
                    event_index=index,
                    evaluated=evaluated,
                )
            )

        # ---------------------------------------------------------------------
        # Report header.
        # ---------------------------------------------------------------------

        header = []

        header.append(
            divider()
        )

        header.append(
            "SMARTSOC AI — ATTACK & MITIGATION "
            "PROFILE REPORT"
        )

        header.append(
            divider()
        )

        header.append("")

        header.append(
            f"Model Version        : {VERSION}"
        )

        header.append(
            f"Model Classes        : {len(classes)}"
        )

        header.append(
            "Inference Source     : Frozen Production Model"
        )

        header.append(
            "Dataset              : Frozen Test Dataset"
        )

        header.append(
            f"Frozen Dataset Rows  : {len(structured)}"
        )

        header.append(
            "Representative Events: 45 actual top predictions"
        )

        header.append(
            "Synthetic Scores     : DISABLED"
        )

        header.append(
            "Synthetic Predictions: DISABLED"
        )

        header.append(
            "Response Execution   : DISABLED"
        )

        header.append(
            "Containment Execution: DISABLED"
        )

        header.append("")

        header.append(
            "REPORT INTEGRITY"
        )

        header.append(
            "----------------"
        )

        header.append(
            "Every profile below is generated from a real frozen "
            "dataset event."
        )

        header.append(
            "The model probability is taken from the serialized "
            "production model."
        )

        header.append(
            "The severity/risk score is calculated by the actual "
            "Phase 6.7 risk engine."
        )

        header.append(
            "The decision is produced by the actual decision policy."
        )

        header.append(
            "The mitigation plan is produced by the actual response "
            "planner."
        )

        header.append(
            "Authorization is produced by the actual authorization "
            "engine."
        )

        header.append(
            "No response or containment action is executed."
        )

        header.append("")

        final_text = (
            "\n".join(header)
            + "\n"
            + "\n".join(profiles)
            + "\n"
            + divider()
            + "\n"
            + "END OF ACTUAL SMARTSOC 45-TECHNIQUE REPORT"
            + "\n"
            + divider()
            + "\n"
        )

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_FILE.write_text(
            final_text,
            encoding="utf-8",
        )

        print()

        print(
            "[PASS] Actual 45-profile report generated"
        )

        print(
            f"       Profiles : {len(profiles)}"
        )

        print(
            f"       Output   : {OUTPUT_FILE}"
        )

        print()

        print(
            divider()
        )

        print(
            "REPORT COMPLETE"
        )

        print(
            divider()
        )

        print(
            f"Actual attack profiles : {len(profiles)} / {len(classes)}"
        )

        print(
            "Synthetic output       : NONE"
        )

        print(
            "Response execution     : DISABLED"
        )

        print(
            "Containment execution  : DISABLED"
        )

        print(
            f"TXT report             : {OUTPUT_FILE}"
        )

        return 0

    except Exception as exc:

        print()
        print(
            "[FAIL] ACTUAL REPORT GENERATION FAILED"
        )

        print(
            f"       {type(exc).__name__}: {exc}"
        )

        print()

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )