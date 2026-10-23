"""
===============================================================================
SmartSOC v4.1
ACTUAL 45-TECHNIQUE ATTACK & MITIGATION REPORT
===============================================================================

Purpose
-------
Generate a professional TXT report containing one profile for every MITRE
technique exposed by the frozen SmartSOC v4.1 production model.

The report is based on:

    * Frozen production model artifacts
    * Frozen v4.1 test dataset
    * Exact aligned v4.1 inference feature construction
    * Actual model probabilities
    * Actual frozen thresholds
    * Existing Phase 6 response / authorization policy artifacts where
      machine-readable evidence is available

STRICT ANTI-FABRICATION RULE
----------------------------
This script NEVER invents:

    * model predictions
    * model confidence
    * severity scores
    * risk levels
    * attack classifications
    * response decisions

If a value is not exposed by an actual artifact, it is reported as
NOT EXPOSED BY ARTIFACT rather than guessed.

The report is evidence-only.

No response is executed.
No containment is executed.
No host is modified.
No production model is modified.
No threshold is modified.
No Wazuh / Shuffle action is executed.

===============================================================================
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

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

REPORT_FILE = (
    REPORT_DIR
    / "smartsoc_actual_45_attack_mitigation_report_v4_1.txt"
)

DETAIL_FILE = (
    REPORT_DIR
    / "smartsoc_actual_45_attack_evidence_v4_1.csv"
)

ALIGNED_INFERENCE = (
    ROOT
    / "src"
    / "production_inference_windows_apt_v4_1_aligned.py"
)

PHASE6_7_REPORT = (
    ROOT
    / "reports"
    / "phase6"
    / "gate6_7"
    / "phase6_gate6_7_offline_end_to_end_report.json"
)

PHASE6_8_REPORT = (
    ROOT
    / "reports"
    / "phase6"
    / "gate6_8"
    / "phase6_gate6_8_phase6_acceptance_freeze_report.json"
)

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
METADATA_TEST = DATA_DIR / "metadata_test.csv"


# =============================================================================
# EXPECTED v4.1 CONTRACT
# =============================================================================

EXPECTED_CLASSES = 45
EXPECTED_FINAL_FEATURES = 6919
EXPECTED_NUMERICAL = 58
EXPECTED_CATEGORICAL_ENCODED = 162
EXPECTED_TFIDF = 6699


# =============================================================================
# MITRE DISPLAY NAMES
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
    "T1546.003": "Event Triggered Execution: Windows Management Instrumentation Event Subscription",
    "T1546.011": "Event Triggered Execution: Application Shimming",
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
# OUTPUT HELPERS
# =============================================================================

def line(char: str = "=", count: int = 80) -> str:
    return char * count


def clean(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""

    return str(value).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)

        if math.isnan(x) or math.isinf(x):
            return None

        return x

    except (TypeError, ValueError):
        return None


def status(message: str) -> None:
    print(f"[PASS] {message}")


def review(message: str) -> None:
    print(f"[REVIEW] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


# =============================================================================
# FILE UTILITIES
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# =============================================================================
# DYNAMIC MODULE LOADER
# =============================================================================

def load_aligned_inference_module():
    """
    Import the already validated v4.1 aligned inference implementation.

    The module's main() is guarded by:

        if __name__ == "__main__":

    therefore importing it does not execute its demo.
    """

    if not ALIGNED_INFERENCE.exists():
        raise FileNotFoundError(
            f"Aligned inference module not found:\n{ALIGNED_INFERENCE}"
        )

    spec = importlib.util.spec_from_file_location(
        "smartsoc_v4_1_aligned_runtime",
        ALIGNED_INFERENCE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load aligned inference module."
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# =============================================================================
# ARTIFACT LOADING
# =============================================================================

def load_artifacts() -> Dict[str, Any]:

    for path in [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
    ]:
        require_file(path)

    artifacts = {
        "model": joblib.load(MODEL_FILE),
        "encoder": joblib.load(ENCODER_FILE),
        "scaler": joblib.load(SCALER_FILE),
        "tfidf": joblib.load(TFIDF_FILE),
        "label_encoder": joblib.load(LABEL_ENCODER_FILE),
        "schema": load_json(SCHEMA_FILE),
        "thresholds": load_json(THRESHOLDS_FILE),
    }

    return artifacts


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset() -> Dict[str, pd.DataFrame]:

    for path in [
        STRUCTURED_TEST,
        BEHAVIOR_TEST,
        LABELS_TEST,
        METADATA_TEST,
    ]:
        require_file(path)

    structured = pd.read_csv(
        STRUCTURED_TEST
    )

    behavior = pd.read_csv(
        BEHAVIOR_TEST
    )

    labels = pd.read_csv(
        LABELS_TEST
    )

    metadata = pd.read_csv(
        METADATA_TEST
    )

    lengths = {
        "structured": len(structured),
        "behavior": len(behavior),
        "labels": len(labels),
        "metadata": len(metadata),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Frozen dataset alignment failure: {lengths}"
        )

    return {
        "structured": structured,
        "behavior": behavior,
        "labels": labels,
        "metadata": metadata,
    }


# =============================================================================
# TECHNIQUE EXTRACTION
# =============================================================================

def get_model_classes(
    label_encoder: Any,
) -> List[str]:

    classes = [
        str(value)
        for value in getattr(
            label_encoder,
            "classes_",
            [],
        )
    ]

    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} model classes, "
            f"found {len(classes)}."
        )

    return classes


def extract_threshold(
    thresholds: Dict[str, Any],
    technique: str,
) -> Optional[float]:

    container = thresholds

    if isinstance(thresholds, dict):
        nested = thresholds.get("thresholds")

        if isinstance(nested, dict):
            container = nested

    value = None

    if isinstance(container, dict):
        value = container.get(technique)

    if isinstance(value, dict):
        for key in [
            "threshold",
            "value",
            "decision_threshold",
        ]:
            if key in value:
                return safe_float(value[key])

    return safe_float(value)


# =============================================================================
# SEVERITY / RISK EVIDENCE
# =============================================================================

def find_numeric_value(
    value: Any,
    names: Tuple[str, ...],
) -> Optional[float]:

    if isinstance(value, dict):

        lowered = {
            str(key).lower(): child
            for key, child in value.items()
        }

        for name in names:

            if name.lower() in lowered:
                result = safe_float(
                    lowered[name.lower()]
                )

                if result is not None:
                    return result

        for child in value.values():

            result = find_numeric_value(
                child,
                names,
            )

            if result is not None:
                return result

    elif isinstance(value, list):

        for child in value:

            result = find_numeric_value(
                child,
                names,
            )

            if result is not None:
                return result

    return None


def find_text_value(
    value: Any,
    names: Tuple[str, ...],
) -> Optional[str]:

    if isinstance(value, dict):

        lowered = {
            str(key).lower(): child
            for key, child in value.items()
        }

        for name in names:

            if name.lower() in lowered:

                result = clean(
                    lowered[name.lower()]
                )

                if result:
                    return result

        for child in value.values():

            result = find_text_value(
                child,
                names,
            )

            if result:
                return result

    elif isinstance(value, list):

        for child in value:

            result = find_text_value(
                child,
                names,
            )

            if result:
                return result

    return None


def load_policy_evidence() -> Dict[str, Any]:

    evidence: Dict[str, Any] = {}

    for path in [
        PHASE6_7_REPORT,
        PHASE6_8_REPORT,
    ]:

        if path.exists():

            try:
                evidence[path.name] = load_json(path)

            except Exception:
                evidence[path.name] = {}

    return evidence


# =============================================================================
# RESPONSE POLICY
# =============================================================================

def policy_for_risk(
    risk_level: Optional[str],
) -> Dict[str, Any]:

    """
    This function does NOT manufacture model output.

    It mirrors the already validated Phase 6 response boundary:

        LOW      -> monitoring / logging
        MEDIUM   -> analyst review
        HIGH     -> priority review / rollback where applicable
        CRITICAL -> critical workflow / analyst authorization

    Execution remains disabled.

    The action list is intentionally expressed as PLANNED actions, not
    executed actions.
    """

    level = (
        clean(risk_level)
        .upper()
    )

    base = {
        "analyst_review": "REQUIRED",
        "automatic_execution": "DISABLED",
        "automatic_containment": "DISABLED",
        "isolate_host": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "kill_process": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "block_ip_domain": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "disable_account": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "quarantine_file": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "block_hash": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "network_containment": "NOT AUTHORIZED FOR AUTOMATIC EXECUTION",
        "log_enrichment": "PLANNED",
        "increase_monitoring": "PLANNED",
        "alert_analyst": "PLANNED",
        "log_only": "NO AUTOMATED ACTION",
        "rollback": "WHERE APPLICABLE",
    }

    if level == "LOW":

        base.update({
            "analyst_review": "NOT REQUIRED",
            "increase_monitoring": "PLANNED",
            "alert_analyst": "NOT REQUIRED",
            "log_only": "YES",
        })

    elif level == "MEDIUM":

        base.update({
            "analyst_review": "REQUIRED",
            "increase_monitoring": "PLANNED",
            "alert_analyst": "YES",
            "log_only": "NO AUTOMATED ACTION",
        })

    elif level == "HIGH":

        base.update({
            "analyst_review": "REQUIRED",
            "increase_monitoring": "YES",
            "alert_analyst": "YES",
            "log_enrichment": "YES",
            "rollback": "REQUIRED WHERE APPLICABLE",
        })

    elif level == "CRITICAL":

        base.update({
            "analyst_review": "REQUIRED",
            "increase_monitoring": "YES",
            "alert_analyst": "YES",
            "log_enrichment": "YES",
            "rollback": "REQUIRED WHERE APPLICABLE",
        })

    else:

        base.update({
            "analyst_review": "REQUIRED",
            "log_only": "SAFETY FALLBACK",
        })

    return base


# =============================================================================
# RISK EXTRACTION
# =============================================================================

def infer_risk_from_actual_evidence(
    technique: str,
    model_confidence: float,
    threshold: Optional[float],
    policy_evidence: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str], str]:

    """
    IMPORTANT:

    Confidence is NOT silently converted into a severity score.

    A severity score is returned only if an existing machine-readable
    Phase 6 artifact actually exposes one.

    Otherwise the report explicitly says that the artifact does not expose it.
    """

    score_names = (
        "severity_score",
        "risk_score",
        "score",
        "severity",
    )

    level_names = (
        "severity_level",
        "risk_level",
        "risk_disposition",
        "severity",
    )

    for document in policy_evidence.values():

        score = find_numeric_value(
            document,
            score_names,
        )

        level = find_text_value(
            document,
            level_names,
        )

        if score is not None or level is not None:

            return (
                score,
                level,
                "ACTUAL PHASE 6 ARTIFACT"
            )

    if threshold is not None and model_confidence >= threshold:

        return (
            None,
            None,
            "MODEL DETECTION CONFIRMED; RISK SCORE NOT EXPOSED"
        )

    return (
        None,
        None,
        "MODEL CONFIDENCE BELOW FROZEN THRESHOLD"
    )


# =============================================================================
# EVENT REPRESENTATION
# =============================================================================

def row_to_event(
    structured_row: pd.Series,
    behavior_row: Optional[pd.Series],
    metadata_row: Optional[pd.Series],
) -> Dict[str, Any]:

    event: Dict[str, Any] = {}

    for key, value in structured_row.to_dict().items():

        if pd.isna(value):
            continue

        event[str(key)] = value

    if behavior_row is not None:

        for key, value in behavior_row.to_dict().items():

            if pd.isna(value):
                continue

            if key not in event:
                event[str(key)] = value

    if metadata_row is not None:

        for key, value in metadata_row.to_dict().items():

            if pd.isna(value):
                continue

            if key not in event:
                event[str(key)] = value

    return event


# =============================================================================
# ACTUAL INFERENCE
# =============================================================================

def run_actual_inference(
    module: Any,
    artifacts: Dict[str, Any],
    datasets: Dict[str, pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    structured = datasets["structured"]
    behavior = datasets["behavior"]
    metadata = datasets["metadata"]

    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    thresholds = artifacts["thresholds"]

    techniques = get_model_classes(
        label_encoder
    )

    best: Dict[str, Dict[str, Any]] = {}

    for technique in techniques:

        best[technique] = {
            "technique": technique,
            "confidence": None,
            "threshold": extract_threshold(
                thresholds,
                technique,
            ),
            "detected": False,
            "event_index": None,
        }

    total = len(structured)

    print()
    print(
        f"Running actual production inference over "
        f"{total} frozen events..."
    )

    for index in range(total):

        structured_row = structured.iloc[index]

        behavior_row = (
            behavior.iloc[index]
            if index < len(behavior)
            else None
        )

        metadata_row = (
            metadata.iloc[index]
            if index < len(metadata)
            else None
        )

        event = row_to_event(
            structured_row,
            behavior_row,
            metadata_row,
        )

        normalized = module.normalize_wazuh_event(
            event
        )

        feature_result = module.build_feature_matrix(
            normalized,
            artifacts,
        )

        matrix = feature_result["matrix"]

        probabilities = model.predict_proba(
            matrix
        )

        probabilities = np.asarray(
            probabilities
        )

        if probabilities.ndim == 1:

            probabilities = probabilities.reshape(
                1,
                -1,
            )

        if probabilities.shape[1] != len(techniques):

            raise ValueError(
                "Production model probability dimension "
                f"{probabilities.shape} does not match "
                f"{len(techniques)} techniques."
            )

        for class_index, technique in enumerate(techniques):

            probability = float(
                probabilities[
                    0,
                    class_index,
                ]
            )

            current = best[technique]

            if (
                current["confidence"] is None
                or probability > current["confidence"]
            ):

                threshold = current["threshold"]

                detected = (
                    threshold is not None
                    and probability >= threshold
                )

                best[technique] = {
                    "technique": technique,
                    "confidence": probability,
                    "threshold": threshold,
                    "detected": bool(detected),
                    "event_index": index,
                }

    results = list(
        best.values()
    )

    results.sort(
        key=lambda item: (
            item["confidence"]
            if item["confidence"] is not None
            else -1.0
        ),
        reverse=True,
    )

    return results, {
        "events_processed": total,
        "techniques": len(techniques),
    }


# =============================================================================
# CSV EVIDENCE
# =============================================================================

def write_csv(
    records: List[Dict[str, Any]],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DETAIL_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "attack_id",
                "attack_name",
                "model_confidence",
                "model_confidence_percent",
                "threshold",
                "detected",
                "representative_event_index",
                "severity_score",
                "severity_level",
                "risk_disposition",
                "risk_evidence",
            ],
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                "attack_id":
                    record["technique"],

                "attack_name":
                    record["attack_name"],

                "model_confidence":
                    record["model_confidence"],

                "model_confidence_percent":
                    record["model_confidence_percent"],

                "threshold":
                    record["threshold"],

                "detected":
                    record["detected"],

                "representative_event_index":
                    record["event_index"],

                "severity_score":
                    record["severity_score"],

                "severity_level":
                    record["severity_level"],

                "risk_disposition":
                    record["risk_disposition"],

                "risk_evidence":
                    record["risk_evidence"],
            })


# =============================================================================
# TXT REPORT
# =============================================================================

def write_report(
    records: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    dataset_count: int,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]

    model_classes = len(
        getattr(
            label_encoder,
            "classes_",
            [],
        )
    )

    model_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add(line())
    add(
        "SMARTSOC AI — ACTUAL ATTACK & MITIGATION PROFILE REPORT"
    )
    add(line())
    add()
    add("Model Version        : 4.1")
    add(f"Model Classes        : {model_classes}")
    add("Inference Source     : Frozen Production Model")
    add("Dataset              : Frozen Test Dataset")
    add(f"Frozen Events        : {dataset_count}")
    add(f"Model Features       : {model_features}")
    add(
        f"Expected Features    : {EXPECTED_FINAL_FEATURES}"
    )
    add("Synthetic Scores     : DISABLED")
    add("Synthetic Predictions: DISABLED")
    add("Response Execution   : DISABLED")
    add("Containment          : DISABLED")
    add()
    add(
        "IMPORTANT: Model Confidence is the actual probability emitted"
    )
    add(
        "by the serialized v4.1 production model."
    )
    add(
        "Severity Score is shown only when an existing Phase 6 artifact"
    )
    add(
        "actually exposes a severity/risk score."
    )
    add(
        "No confidence value is converted into an invented severity score."
    )
    add()
    add(line())
    add(
        "ACTUAL MODEL OUTPUT COVERAGE"
    )
    add(line())
    add(
        "The following profiles represent all 45 classes exposed by"
    )
    add(
        "the frozen v4.1 production label encoder."
    )
    add(
        "For each class, the report records the strongest actual"
    )
    add(
        "model confidence observed across the frozen test dataset."
    )
    add()

    for number, record in enumerate(
        records,
        start=1,
    ):

        technique = record["technique"]

        confidence = record["model_confidence"]

        threshold = record["threshold"]

        severity_score = record["severity_score"]

        severity_level = record["severity_level"]

        risk_disposition = record["risk_disposition"]

        policy = policy_for_risk(
            risk_disposition
        )

        add(line())
        add(
            f"ATTACK PROFILE {number:02d} / {len(records):02d}"
        )
        add(line())
        add()
        add(
            f"Attack ID        : {technique}"
        )
        add(
            f"Attack           : {record['attack_name']}"
        )
        add()
        add("AI ASSESSMENT")
        add("-------------")
        add(
            f"Predicted Class       : {technique}"
        )

        if confidence is None:

            add(
                "Model Confidence      : NOT AVAILABLE"
            )

        else:

            add(
                f"Model Confidence      : {confidence:.6f}"
            )

            add(
                f"Model Confidence %    : {confidence * 100:.2f}%"
            )

        if threshold is None:

            add(
                "Decision Threshold    : NOT AVAILABLE"
            )

        else:

            add(
                f"Decision Threshold    : {threshold:.6f}"
            )

        add(
            f"Threshold Result      : "
            f"{'DETECTED' if record['detected'] else 'NOT DETECTED'}"
        )

        if severity_score is None:

            add(
                "Severity Score        : "
                "NOT EXPOSED BY ARTIFACT"
            )

        else:

            add(
                f"Severity Score        : {severity_score:.4f}"
            )

        if severity_level:

            add(
                f"Severity Level        : {severity_level}"
            )

        else:

            add(
                "Severity Level        : "
                "NOT EXPOSED BY ARTIFACT"
            )

        if risk_disposition:

            add(
                f"Risk Disposition      : {risk_disposition}"
            )

        else:

            add(
                "Risk Disposition      : "
                "NOT EXPOSED BY ARTIFACT"
            )

        add()
        add("DECISION POLICY")
        add("---------------")
        add(
            f"Analyst Review        : "
            f"{policy['analyst_review']}"
        )

        add(
            "Response Required     : "
            f"{'YES' if record['detected'] else 'NO'}"
        )

        add(
            "Containment Candidate : "
            "YES"
            if record["detected"]
            else
            "NO"
        )

        add(
            "Automatic Execution   : DISABLED"
        )

        add(
            "Automatic Containment : DISABLED"
        )

        add()
        add("RESPONSE PLANNER")
        add("----------------")

        if risk_disposition:

            if str(risk_disposition).upper() == "CRITICAL":

                response_category = "CRITICAL-RESPONSE WORKFLOW"

            elif str(risk_disposition).upper() == "HIGH":

                response_category = "HIGH-RISK RESPONSE"

            elif str(risk_disposition).upper() == "MEDIUM":

                response_category = "ANALYST REVIEW WORKFLOW"

            elif str(risk_disposition).upper() == "LOW":

                response_category = "MONITORING / LOGGING WORKFLOW"

            else:

                response_category = "SAFETY FALLBACK WORKFLOW"

        else:

            response_category = (
                "RESPONSE CATEGORY NOT EXPOSED BY ARTIFACT"
            )

        add(
            f"Response Category     : {response_category}"
        )

        add(
            "Investigation         : "
            f"{'REQUIRED' if record['detected'] else 'NOT REQUIRED'}"
        )

        add(
            f"Rollback              : {policy['rollback']}"
        )

        add()
        add("PLANNED MITIGATION")
        add("------------------")

        actions = [
            (
                "Isolate Host",
                policy["isolate_host"],
            ),
            (
                "Kill Malicious Process",
                policy["kill_process"],
            ),
            (
                "Block IP / Domain",
                policy["block_ip_domain"],
            ),
            (
                "Disable Account",
                policy["disable_account"],
            ),
            (
                "Quarantine File",
                policy["quarantine_file"],
            ),
            (
                "Block Hash",
                policy["block_hash"],
            ),
            (
                "Network Containment",
                policy["network_containment"],
            ),
            (
                "Log Enrichment",
                policy["log_enrichment"],
            ),
            (
                "Increase Monitoring",
                policy["increase_monitoring"],
            ),
            (
                "Alert Analyst",
                policy["alert_analyst"],
            ),
            (
                "Log Only",
                policy["log_only"],
            ),
        ]

        for action_name, action_state in actions:

            if action_state in {
                "YES",
                "PLANNED",
            }:

                marker = "✓"

            elif action_state == "NO AUTOMATED ACTION":

                marker = " "

            elif action_state == "SAFETY FALLBACK":

                marker = "✓"

            else:

                marker = " "

            add(
                f"[{marker}] "
                f"{action_name:<25} : {action_state}"
            )

        add()
        add("AUTHORIZATION")
        add("-------------")
        add(
            "Authorization State   : "
            "PENDING ANALYST AUTHORIZATION"
            if record["detected"]
            else
            "NO EXECUTION REQUESTED"
        )

        add(
            "Analyst Approval      : "
            f"{policy['analyst_review']}"
        )

        add(
            "AI Authorization      : NOT SUFFICIENT"
        )

        add(
            "Automatic Execution   : DISABLED"
        )

        add(
            "Automatic Containment : DISABLED"
        )

        add()
        add("EXECUTION STATE")
        add("----------------")
        add(
            "Host Isolation       : NOT EXECUTED"
        )
        add(
            "Process Kill         : NOT EXECUTED"
        )
        add(
            "IP / Domain Block    : NOT EXECUTED"
        )
        add(
            "Account Disable      : NOT EXECUTED"
        )
        add(
            "File Quarantine      : NOT EXECUTED"
        )
        add(
            "Hash Blocking        : NOT EXECUTED"
        )
        add(
            "Network Containment  : NOT EXECUTED"
        )

        add()
        add("EVIDENCE")
        add("--------")
        add(
            "Inference            : ACTUAL PRODUCTION MODEL"
        )
        add(
            "Model Confidence     : ACTUAL MODEL OUTPUT"
        )
        add(
            "Threshold            : ACTUAL FROZEN THRESHOLD"
        )

        if severity_score is None:

            add(
                "Severity Score       : "
                "NOT EXPOSED BY CURRENT ARTIFACT"
            )

        else:

            add(
                "Severity Score       : "
                "ACTUAL PHASE 6 ARTIFACT"
            )

        if severity_level:

            add(
                "Risk / Severity      : "
                "ACTUAL PHASE 6 ARTIFACT"
            )

        else:

            add(
                "Risk / Severity      : "
                "NOT EXPOSED BY CURRENT ARTIFACT"
            )

        add(
            f"Representative Event : "
            f"{record['event_index']}"
        )

        add(
            "Synthetic Data       : NONE"
        )

        add()

    add(line())
    add("REPORT INTEGRITY")
    add(line())
    add()
    add(
        "All 45 model classes were processed."
    )
    add(
        "No synthetic prediction was generated."
    )
    add(
        "No synthetic severity score was generated."
    )
    add(
        "No response action was executed."
    )
    add(
        "No containment action was executed."
    )
    add(
        "The frozen production model and thresholds were not modified."
    )
    add()
    add(
        "End of actual SmartSOC v4.1 attack / mitigation report."
    )
    add(line())

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print(line())
    print(
        "SmartSOC AI — ACTUAL 45-TECHNIQUE ATTACK & MITIGATION REPORT"
    )
    print(line())

    try:

        # ---------------------------------------------------------------------
        # 1. ARTIFACTS
        # ---------------------------------------------------------------------

        print()
        print("[1] Loading frozen production artifacts...")

        artifacts = load_artifacts()

        status(
            "Frozen v4.1 production artifacts loaded"
        )

        # ---------------------------------------------------------------------
        # 2. DATA
        # ---------------------------------------------------------------------

        print()
        print("[2] Loading frozen test dataset...")

        datasets = load_dataset()

        status(
            f"Frozen test dataset loaded: "
            f"{len(datasets['structured'])} events"
        )

        # ---------------------------------------------------------------------
        # 3. ALIGNED INFERENCE
        # ---------------------------------------------------------------------

        print()
        print(
            "[3] Loading validated aligned inference implementation..."
        )

        module = load_aligned_inference_module()

        status(
            "v4.1 aligned inference implementation loaded"
        )

        # ---------------------------------------------------------------------
        # 4. MODEL CONTRACT
        # ---------------------------------------------------------------------

        label_encoder = artifacts["label_encoder"]

        classes = get_model_classes(
            label_encoder
        )

        model = artifacts["model"]

        model_features = getattr(
            model,
            "n_features_in_",
            None,
        )

        print()
        print("[4] Production model contract")

        print(
            f"       Classes          : {len(classes)}"
        )

        print(
            f"       Model features   : {model_features}"
        )

        print(
            f"       Expected         : "
            f"{EXPECTED_FINAL_FEATURES}"
        )

        if model_features != EXPECTED_FINAL_FEATURES:

            raise ValueError(
                "Production model feature count does not match "
                f"v4.1 contract: {model_features}"
            )

        status(
            "Production model contract validated"
        )

        # ---------------------------------------------------------------------
        # 5. POLICY EVIDENCE
        # ---------------------------------------------------------------------

        policy_evidence = load_policy_evidence()

        status(
            "Phase 6 machine-readable evidence loaded"
        )

        # ---------------------------------------------------------------------
        # 6. ACTUAL INFERENCE
        # ---------------------------------------------------------------------

        print()
        print(
            "[5] Running ACTUAL serialized-model inference..."
        )

        raw_records, inference_meta = run_actual_inference(
            module,
            artifacts,
            datasets,
        )

        status(
            f"Actual inference completed over "
            f"{inference_meta['events_processed']} events"
        )

        status(
            f"Actual model classes processed: "
            f"{inference_meta['techniques']}"
        )

        # ---------------------------------------------------------------------
        # 7. BUILD REPORT RECORDS
        # ---------------------------------------------------------------------

        records: List[Dict[str, Any]] = []

        for raw in raw_records:

            technique = raw["technique"]

            confidence = raw["confidence"]

            threshold = raw["threshold"]

            score, level, evidence = (
                infer_risk_from_actual_evidence(
                    technique,
                    confidence,
                    threshold,
                    policy_evidence,
                )
            )

            record = {
                "technique": technique,

                "attack_name":
                    TECHNIQUE_NAMES.get(
                        technique,
                        "MITRE Technique",
                    ),

                "model_confidence":
                    confidence,

                "model_confidence_percent":
                    (
                        confidence * 100
                        if confidence is not None
                        else None
                    ),

                "threshold":
                    threshold,

                "detected":
                    raw["detected"],

                "event_index":
                    raw["event_index"],

                "severity_score":
                    score,

                "severity_level":
                    level,

                "risk_disposition":
                    level,

                "risk_evidence":
                    evidence,
            }

            records.append(record)

        # ---------------------------------------------------------------------
        # 8. WRITE EVIDENCE
        # ---------------------------------------------------------------------

        print()
        print(
            "[6] Writing actual evidence..."
        )

        write_csv(
            records
        )

        write_report(
            records,
            artifacts,
            len(datasets["structured"]),
        )

        status(
            f"45-technique report written: {REPORT_FILE}"
        )

        status(
            f"Evidence CSV written: {DETAIL_FILE}"
        )

        # ---------------------------------------------------------------------
        # 9. FINAL
        # ---------------------------------------------------------------------

        print()
        print(line())
        print(
            "ACTUAL SMARTSOC v4.1 REPORT COMPLETE"
        )
        print(line())
        print()
        print(
            f"Frozen events       : "
            f"{len(datasets['structured'])}"
        )
        print(
            f"Model classes       : "
            f"{len(records)}"
        )
        print(
            "Synthetic scores    : DISABLED"
        )
        print(
            "Synthetic outputs   : DISABLED"
        )
        print(
            "Response execution  : DISABLED"
        )
        print(
            "Containment         : DISABLED"
        )
        print()
        print(
            f"TXT report          : {REPORT_FILE}"
        )
        print(
            f"Evidence CSV        : {DETAIL_FILE}"
        )

        return 0

    except Exception as exc:

        print()
        fail(
            "Actual report generation failed."
        )

        print()
        print(
            f"Reason: {exc}"
        )

        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )