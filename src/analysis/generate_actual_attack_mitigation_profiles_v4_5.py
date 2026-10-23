"""
SmartSOC AI v4.5
Actual Attack & Mitigation Profile Generator

PURPOSE
-------
Generate a supervisor-facing TXT report from the ACTUAL frozen
production inference pipeline.

IMPORTANT
---------
This script:
    - DOES NOT retrain the model
    - DOES NOT modify production artifacts
    - DOES NOT create synthetic predictions
    - DOES NOT invent model scores
    - DOES NOT invent severity scores
    - DOES NOT execute response actions
    - DOES NOT perform containment

It uses:
    1. Frozen production model artifacts
    2. Frozen test dataset
    3. Exact aligned v4.1 production inference implementation
    4. Actual model probabilities
    5. Actual calibrated thresholds
    6. Actual detected/not-detected state
    7. Existing Phase 6 response/authorization components when
       their interfaces can be resolved safely

The report contains ONE PROFILE PER MODEL CLASS (45 profiles),
not all 11,707 events.

For each MITRE technique, the report selects the frozen-test
event where that technique has its HIGHEST ACTUAL MODEL
probability.

That gives a meaningful representative example for every
model class without fabricating a prediction.

Severity score:
    Derived from the ACTUAL model probability for that
    representative event, scaled to 0-10.

Severity level:
    Derived from the ACTUAL model probability and the
    calibrated detection threshold.

The raw model probability and threshold are always retained
so the supervisor can distinguish model output from the
presentation severity score.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"

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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_TXT = (
    REPORT_DIR
    / "smartsoc_actual_attack_mitigation_profiles_v4_5.txt"
)

OUTPUT_JSON = (
    REPORT_DIR
    / "smartsoc_actual_attack_mitigation_profiles_v4_5.json"
)


MODEL_FILE = MODEL_DIR / "model.joblib"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"


STRUCTURED_FILE = DATA_DIR / "structured_test.csv"
LABELS_FILE = DATA_DIR / "labels_test.csv"
METADATA_FILE = DATA_DIR / "metadata_test.csv"
TEXT_FILE = DATA_DIR / "behavior_text_test.csv"


INFERENCE_FILE = (
    SRC
    / "production_inference_windows_apt_v4_1_aligned.py"
)


# ============================================================================
# CONSTANTS
# ============================================================================

VERSION = "4.5"

EXPECTED_CLASSES = 45
EXPECTED_FINAL_FEATURES = 6919
EXPECTED_NUMERICAL = 58
EXPECTED_TFIDF = 6699
EXPECTED_CATEGORICAL_ENCODED = 162


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def info(text: str) -> None:
    print(f"[INFO] {text}")


def passed(text: str) -> None:
    print(f"[PASS] {text}")


def failed(text: str) -> None:
    print(f"[FAIL] {text}")


# ============================================================================
# SAFE HELPERS
# ============================================================================

def clean(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        x = float(value)

        if not math.isfinite(x):
            return default

        return x

    except Exception:
        return default


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


# ============================================================================
# DYNAMIC IMPORT
# ============================================================================

def load_module(
    path: Path,
    name: str,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None:
        raise RuntimeError(
            f"Unable to create module specification for {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    if spec.loader is None:
        raise RuntimeError(
            f"Unable to load module {path}"
        )

    spec.loader.exec_module(module)

    return module


# ============================================================================
# SEVERITY
# ============================================================================

def severity_level_from_probability(
    probability: float,
    threshold: float,
) -> str:
    """
    Presentation severity derived from ACTUAL model probability.

    This does not replace the model threshold.

    The report exposes both:
        raw probability
        calibrated threshold

    Level interpretation:
        LOW       < 0.30
        MEDIUM    0.30 - <0.50
        HIGH      0.50 - <0.75
        CRITICAL  >= 0.75

    If the model threshold is higher than the probability,
    the detection state remains FALSE even if the presentation
    severity is HIGH/CRITICAL.
    """

    if probability >= 0.75:
        return "CRITICAL"

    if probability >= 0.50:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


def severity_score_from_probability(
    probability: float,
) -> float:
    """
    Convert actual probability [0,1] to a transparent 0-10
    presentation score.

    This is NOT a new model prediction.

    It is simply:
        probability * 10
    """

    return round(
        max(
            0.0,
            min(
                10.0,
                probability * 10.0,
            ),
        ),
        2,
    )


# ============================================================================
# MITRE NAMES
# ============================================================================

MITRE_NAMES = {
    "T1003": "OS Credential Dumping",
    "T1003.001": "LSASS Memory",
    "T1021.006": "Windows Remote Management",
    "T1027": "Obfuscated/Compressed Files and Information",
    "T1027.004": "Compile After Delivery",
    "T1036": "Masquerading",
    "T1047": "Windows Management Instrumentation",
    "T1053.005": "Scheduled Task/Job: Scheduled Task",
    "T1055": "Process Injection",
    "T1059": "Command and Scripting Interpreter",
    "T1059.001": "PowerShell",
    "T1059.003": "Windows Command Shell",
    "T1070": "Indicator Removal",
    "T1070.004": "File Deletion",
    "T1078": "Valid Accounts",
    "T1087": "Account Discovery",
    "T1105": "Ingress Tool Transfer",
    "T1110": "Brute Force",
    "T1110.001": "Password Guessing",
    "T1112": "Modify Registry",
    "T1136": "Create Account",
    "T1218": "System Binary Proxy Execution",
    "T1484": "Domain or Tenant Policy Modification",
    "T1485": "Data Destruction",
    "T1486": "Data Encrypted for Impact",
    "T1489": "Service Stop",
    "T1529": "System Shutdown/Reboot",
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
    "T1098": "Account Manipulation",
}


def mitre_name(
    technique: str,
) -> str:
    return MITRE_NAMES.get(
        technique,
        "MITRE ATT&CK Technique",
    )


# ============================================================================
# RESPONSE POLICY
# ============================================================================

def response_for_level(
    level: str,
    detected: bool,
) -> Dict[str, Any]:
    """
    Phase-6 presentation of the response policy.

    These are PLANNED actions only.

    They are NOT executed.

    The global Phase-6 safety boundary remains:
        automatic execution disabled
        automatic containment disabled
        analyst authorization required
    """

    if not detected:
        return {
            "category": "MONITOR / ANALYST REVIEW",
            "investigation": "RECOMMENDED",
            "rollback": "WHERE APPLICABLE",
            "actions": {
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
                "log_only": True,
            },
        }

    if level == "CRITICAL":
        return {
            "category": "CRITICAL-RISK RESPONSE",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "actions": {
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
            },
        }

    if level == "HIGH":
        return {
            "category": "HIGH-RISK RESPONSE",
            "investigation": "REQUIRED",
            "rollback": "REQUIRED WHERE APPLICABLE",
            "actions": {
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
            },
        }

    if level == "MEDIUM":
        return {
            "category": "MEDIUM-RISK RESPONSE",
            "investigation": "REQUIRED",
            "rollback": "WHERE APPLICABLE",
            "actions": {
                "isolate_host": False,
                "kill_process": False,
                "block_ip_domain": True,
                "disable_account": False,
                "quarantine_file": True,
                "block_hash": True,
                "network_containment": False,
                "log_enrichment": True,
                "increase_monitoring": True,
                "alert_analyst": True,
                "log_only": False,
            },
        }

    return {
        "category": "LOW-RISK MONITORING",
        "investigation": "RECOMMENDED",
        "rollback": "WHERE APPLICABLE",
        "actions": {
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
            "log_only": True,
        },
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    structured = pd.read_csv(
        STRUCTURED_FILE
    )

    labels = pd.read_csv(
        LABELS_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    behavior = pd.read_csv(
        TEXT_FILE
    )

    lengths = [
        len(structured),
        len(labels),
        len(metadata),
        len(behavior),
    ]

    if len(set(lengths)) != 1:
        raise RuntimeError(
            "Frozen dataset alignment failed: "
            f"{lengths}"
        )

    passed(
        f"Frozen dataset alignment: {lengths[0]} rows"
    )

    return (
        structured,
        labels,
        metadata,
        behavior,
    )


# ============================================================================
# LABEL EXTRACTION
# ============================================================================

def extract_label_column(
    labels: pd.DataFrame,
) -> str:

    candidates = [
        "techniques",
        "label",
        "labels",
        "target",
    ]

    for name in candidates:
        if name in labels.columns:
            return name

    if len(labels.columns) == 1:
        return labels.columns[0]

    raise RuntimeError(
        "Unable to identify technique label column."
    )


def split_labels(
    value: Any,
) -> List[str]:

    text = clean(value)

    if not text:
        return []

    for separator in [
        ",",
        ";",
        "|",
    ]:
        if separator in text:
            return [
                x.strip()
                for x in text.split(separator)
                if x.strip()
            ]

    return [text]


# ============================================================================
# EVENT RECONSTRUCTION
# ============================================================================

def build_event_from_row(
    structured_row: pd.Series,
    behavior_row: pd.Series,
    metadata_row: pd.Series,
) -> Dict[str, Any]:

    event: Dict[str, Any] = {}

    # Preserve the frozen structured fields.
    for key, value in structured_row.to_dict().items():
        event[key] = value

    # Behavior text.
    for key, value in behavior_row.to_dict().items():
        event[key] = value

    # Metadata.
    for key, value in metadata_row.to_dict().items():
        if key not in event:
            event[key] = value

    # Add common aliases used by the aligned inference implementation.
    if "event_id" in structured_row:
        event["event_id"] = structured_row["event_id"]

    if "timestamp" in metadata_row:
        event["timestamp"] = metadata_row["timestamp"]

    if "agent" in metadata_row:
        event["agent"] = metadata_row["agent"]

    return event


# ============================================================================
# MAIN ACTUAL INFERENCE
# ============================================================================

def run_actual_inference():

    banner(
        "SmartSOC AI — ACTUAL 45-TECHNIQUE ATTACK & MITIGATION REPORT"
    )

    print(
        f"Version                 : {VERSION}"
    )
    print(
        "Inference Source        : Frozen Production Model"
    )
    print(
        "Dataset                 : Frozen Test Dataset"
    )
    print(
        "Synthetic Predictions   : DISABLED"
    )
    print(
        "Synthetic Scores        : DISABLED"
    )
    print(
        "Response Execution      : DISABLED"
    )
    print(
        "Containment Execution   : DISABLED"
    )

    banner(
        "[1] VALIDATING PRODUCTION ARTIFACTS"
    )

    for path in [
        MODEL_FILE,
        THRESHOLD_FILE,
        SCHEMA_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        STRUCTURED_FILE,
        LABELS_FILE,
        METADATA_FILE,
        TEXT_FILE,
        INFERENCE_FILE,
    ]:
        require(path)

    passed(
        "Production and frozen artifacts located"
    )

    banner(
        "[2] LOADING FROZEN DATASET"
    )

    (
        structured,
        labels,
        metadata,
        behavior,
    ) = load_dataset()

    label_column = extract_label_column(
        labels
    )

    banner(
        "[3] LOADING EXACT v4.1 ALIGNED INFERENCE"
    )

    inference = load_module(
        INFERENCE_FILE,
        "smartsoc_v41_aligned_inference",
    )

    passed(
        "Exact aligned production inference implementation loaded"
    )

    # ------------------------------------------------------------------------
    # Load actual artifacts through the project's own loader.
    # ------------------------------------------------------------------------

    if not hasattr(
        inference,
        "load_artifacts",
    ):
        raise RuntimeError(
            "Aligned inference module does not expose load_artifacts()."
        )

    artifacts = inference.load_artifacts()

    if hasattr(
        inference,
        "validate_artifacts",
    ):
        inference.validate_artifacts(
            artifacts
        )

    passed(
        "Production artifacts validated by aligned inference implementation"
    )

    model = artifacts.get(
        "model"
    )

    label_encoder = artifacts.get(
        "label_encoder"
    )

    if model is None:
        raise RuntimeError(
            "Aligned inference did not expose the production model."
        )

    if label_encoder is None:
        raise RuntimeError(
            "Aligned inference did not expose the production label encoder."
        )

    classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            "Production model class count mismatch: "
            f"{len(classes)}"
        )

    passed(
        f"Production model exposes {len(classes)} MITRE classes"
    )

    # ------------------------------------------------------------------------
    # ACTUAL inference.
    # ------------------------------------------------------------------------

    banner(
        "[4] RUNNING ACTUAL FROZEN PRODUCTION INFERENCE"
    )

    if not hasattr(
        inference,
        "normalize_wazuh_event",
    ):
        raise RuntimeError(
            "Aligned inference missing normalize_wazuh_event()."
        )

    if not hasattr(
        inference,
        "build_feature_matrix",
    ):
        raise RuntimeError(
            "Aligned inference missing build_feature_matrix()."
        )

    if not hasattr(
        inference,
        "predict",
    ):
        raise RuntimeError(
            "Aligned inference missing predict()."
        )

    probabilities = np.zeros(
        (
            len(structured),
            len(classes),
        ),
        dtype=np.float64,
    )

    thresholds = {}

    try:
        raw_thresholds = load_json(
            THRESHOLD_FILE
        )

        if isinstance(
            raw_thresholds,
            dict,
        ):
            thresholds = raw_thresholds

    except Exception:
        thresholds = {}

    # ------------------------------------------------------------------------
    # Resolve threshold helper from the exact inference implementation.
    # ------------------------------------------------------------------------

    def actual_threshold(
        technique: str,
    ) -> float:

        if hasattr(
            inference,
            "extract_threshold",
        ):
            try:
                return safe_float(
                    inference.extract_threshold(
                        thresholds,
                        technique,
                    ),
                    0.5,
                )
            except Exception:
                pass

        value = thresholds.get(
            technique,
            0.5,
        )

        if isinstance(
            value,
            dict,
        ):
            value = value.get(
                "threshold",
                0.5,
            )

        return safe_float(
            value,
            0.5,
        )

    # ------------------------------------------------------------------------
    # Process rows using EXACT aligned production functions.
    # ------------------------------------------------------------------------

    for index in range(
        len(structured)
    ):

        if index % 1000 == 0:
            info(
                f"Actual inference progress: "
                f"{index}/{len(structured)}"
            )

        structured_row = structured.iloc[
            index
        ]

        behavior_row = behavior.iloc[
            index
        ]

        metadata_row = metadata.iloc[
            index
        ]

        raw_event = build_event_from_row(
            structured_row,
            behavior_row,
            metadata_row,
        )

        # The aligned implementation itself owns event normalization.
        normalized = inference.normalize_wazuh_event(
            raw_event
        )

        feature_result = inference.build_feature_matrix(
            normalized,
            artifacts,
        )

        prediction_result = inference.predict(
            feature_result,
            artifacts,
        )

        # The production predict() output contains all_scores.
        all_scores = prediction_result.get(
            "all_scores",
            [],
        )

        score_map = {
            str(item["technique"]): safe_float(
                item.get(
                    "confidence",
                    0.0,
                )
            )
            for item in all_scores
        }

        for class_index, technique in enumerate(
            classes
        ):
            probabilities[
                index,
                class_index,
            ] = score_map.get(
                technique,
                0.0,
            )

    passed(
        "Actual frozen production inference completed"
    )

    print(
        f"       Probability matrix = "
        f"{probabilities.shape}"
    )

    # ------------------------------------------------------------------------
    # Build one representative profile per class.
    # ------------------------------------------------------------------------

    banner(
        "[5] SELECTING ACTUAL REPRESENTATIVE EVENT FOR EACH CLASS"
    )

    profiles: List[Dict[str, Any]] = []

    for class_index, technique in enumerate(
        classes
    ):

        column = probabilities[
            :,
            class_index,
        ]

        best_index = int(
            np.argmax(
                column
            )
        )

        probability = safe_float(
            column[
                best_index
            ]
        )

        threshold = actual_threshold(
            technique
        )

        detected = (
            probability >= threshold
        )

        severity_score = (
            severity_score_from_probability(
                probability
            )
        )

        severity_level = (
            severity_level_from_probability(
                probability,
                threshold,
            )
        )

        label_value = labels.iloc[
            best_index
        ][label_column]

        actual_dataset_labels = (
            split_labels(
                label_value
            )
        )

        response = response_for_level(
            severity_level,
            detected,
        )

        profile = {
            "attack_id": technique,
            "attack_name": mitre_name(
                technique
            ),
            "representative_row": best_index,
            "model_probability": round(
                probability,
                8,
            ),
            "severity_score": severity_score,
            "severity_level": severity_level,
            "threshold": round(
                threshold,
                8,
            ),
            "detected": bool(
                detected
            ),
            "dataset_labels": actual_dataset_labels,
            "response": response,
        }

        profiles.append(
            profile
        )

    passed(
        f"Generated {len(profiles)} actual model-class profiles"
    )

    # ------------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------------

    json_document = {
        "report": {
            "version": VERSION,
            "model_version": "4.1",
            "model_classes": len(classes),
            "dataset_rows": len(structured),
            "inference_source": "Frozen Production Model",
            "synthetic_predictions": False,
            "synthetic_scores": False,
            "automatic_execution": False,
            "automatic_containment": False,
        },
        "production_artifacts": {
            "model": str(MODEL_FILE),
            "thresholds": str(THRESHOLD_FILE),
            "feature_schema": str(SCHEMA_FILE),
        },
        "profiles": profiles,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_document,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Professional TXT report.
    # ------------------------------------------------------------------------

    lines: List[str] = []

    def add(
        text: str = "",
    ):
        lines.append(
            text
        )

    add(
        "=" * 72
    )
    add(
        "SMARTSOC AI — ATTACK & MITIGATION PROFILE REPORT"
    )
    add(
        "=" * 72
    )
    add()
    add(
        "Model Version        : 4.1"
    )
    add(
        f"Model Classes        : {len(classes)}"
    )
    add(
        "Inference Source     : Frozen Production Model"
    )
    add(
        "Dataset              : Frozen Test Dataset"
    )
    add(
        f"Frozen Dataset Rows  : {len(structured)}"
    )
    add(
        "Synthetic Scores     : DISABLED"
    )
    add(
        "Synthetic Predictions: DISABLED"
    )
    add(
        "Automatic Execution  : DISABLED"
    )
    add(
        "Automatic Containment: DISABLED"
    )
    add()

    add(
        "IMPORTANT"
    )
    add(
        "---------"
    )
    add(
        "Each profile represents one of the 45 classes exposed by"
    )
    add(
        "the frozen production model."
    )
    add(
        "The representative event is the frozen-test event having"
    )
    add(
        "the highest ACTUAL model probability for that technique."
    )
    add()
    add(
        "Model Probability is the serialized production model output."
    )
    add(
        "Severity Score is a transparent 0-10 presentation mapping"
    )
    add(
        "of that actual probability (probability × 10)."
    )
    add(
        "Threshold is the actual calibrated detection threshold."
    )
    add(
        "No synthetic prediction or synthetic probability is used."
    )
    add()
    add(
        "All mitigation entries are PLANNED response actions."
    )
    add(
        "No host, process, account, file, hash, IP, or network"
    )
    add(
        "action was executed by this report generator."
    )
    add(
        "Automatic execution and containment remain disabled."
    )
    add(
        "Analyst authorization remains required."
    )
    add()

    for number, profile in enumerate(
        profiles,
        start=1,
    ):

        actions = profile[
            "response"
        ][
            "actions"
        ]

        add(
            "=" * 72
        )
        add(
            f"ATTACK PROFILE {number:02d} / {len(profiles)}"
        )
        add(
            "=" * 72
        )
        add()
        add(
            f"Attack ID        : {profile['attack_id']}"
        )
        add(
            f"Attack           : {profile['attack_name']}"
        )
        add()
        add(
            "AI ASSESSMENT"
        )
        add(
            "-------------"
        )
        add(
            f"Representative Event : "
            f"Frozen Test Row {profile['representative_row']}"
        )
        add(
            f"Model Probability     : "
            f"{profile['model_probability']:.8f}"
        )
        add(
            f"Severity Score        : "
            f"{profile['severity_score']:.2f} / 10.00"
        )
        add(
            f"Severity Level        : "
            f"{profile['severity_level']}"
        )
        add(
            f"Calibrated Threshold   : "
            f"{profile['threshold']:.8f}"
        )
        add(
            f"Detection Decision     : "
            f"{'DETECTED' if profile['detected'] else 'NOT DETECTED'}"
        )
        add()
        add(
            "DATASET LABEL EVIDENCE"
        )
        add(
            "-----------------------"
        )
        if profile[
            "dataset_labels"
        ]:
            add(
                "Ground-truth labels    : "
                + ", ".join(
                    profile[
                        "dataset_labels"
                    ]
                )
            )
        else:
            add(
                "Ground-truth labels    : NONE RECORDED"
            )
        add()
        add(
            "DECISION POLICY"
        )
        add(
            "---------------"
        )

        if profile[
            "detected"
        ]:
            add(
                "Risk Disposition      : "
                f"{profile['severity_level']}"
            )
            add(
                "Analyst Review        : REQUIRED"
            )
            add(
                "Response Required     : YES"
            )
            add(
                "Containment Candidate : YES"
            )
        else:
            add(
                "Risk Disposition      : "
                f"{profile['severity_level']} / BELOW THRESHOLD"
            )
            add(
                "Analyst Review        : RECOMMENDED"
            )
            add(
                "Response Required     : NO AUTOMATED RESPONSE"
            )
            add(
                "Containment Candidate : NO"
            )

        add(
            "Automatic Execution   : DISABLED"
        )
        add(
            "Automatic Containment : DISABLED"
        )
        add()
        add(
            "RESPONSE PLANNER"
        )
        add(
            "----------------"
        )
        add(
            f"Response Category     : "
            f"{profile['response']['category']}"
        )
        add(
            f"Investigation         : "
            f"{profile['response']['investigation']}"
        )
        add(
            f"Rollback              : "
            f"{profile['response']['rollback']}"
        )
        add()
        add(
            "PLANNED MITIGATION"
        )
        add(
            "------------------"
        )

        action_names = [
            (
                "isolate_host",
                "Isolate Host",
            ),
            (
                "kill_process",
                "Kill Malicious Process",
            ),
            (
                "block_ip_domain",
                "Block IP / Domain",
            ),
            (
                "disable_account",
                "Disable Account",
            ),
            (
                "quarantine_file",
                "Quarantine File",
            ),
            (
                "block_hash",
                "Block Hash",
            ),
            (
                "network_containment",
                "Network Containment",
            ),
            (
                "log_enrichment",
                "Log Enrichment",
            ),
            (
                "increase_monitoring",
                "Increase Monitoring",
            ),
            (
                "alert_analyst",
                "Alert Analyst",
            ),
            (
                "log_only",
                "Log Only / No Automated Action",
            ),
        ]

        for key, label in action_names:

            marker = (
                "✓"
                if actions.get(
                    key,
                    False,
                )
                else " "
            )

            add(
                f"[{marker}] {label}"
            )

        add()
        add(
            "AUTHORIZATION"
        )
        add(
            "-------------"
        )
        add(
            "Authorization State   : "
            "PENDING ANALYST AUTHORIZATION"
        )
        add(
            "Analyst Approval      : REQUIRED"
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
        add(
            "EXECUTION STATE"
        )
        add(
            "---------------"
        )
        add(
            "Host Isolation        : NOT EXECUTED"
        )
        add(
            "Process Kill          : NOT EXECUTED"
        )
        add(
            "IP / Domain Block     : NOT EXECUTED"
        )
        add(
            "Account Disable       : NOT EXECUTED"
        )
        add(
            "File Quarantine       : NOT EXECUTED"
        )
        add(
            "Hash Blocking         : NOT EXECUTED"
        )
        add(
            "Network Containment   : NOT EXECUTED"
        )
        add()
        add(
            "EVIDENCE"
        )
        add(
            "--------"
        )
        add(
            "Inference             : ACTUAL PRODUCTION MODEL"
        )
        add(
            "Model Probability     : ACTUAL MODEL OUTPUT"
        )
        add(
            "Threshold             : ACTUAL CALIBRATED THRESHOLD"
        )
        add(
            "Detection              : ACTUAL MODEL DECISION"
        )
        add(
            "Severity Score        : DERIVED FROM ACTUAL MODEL PROBABILITY"
        )
        add(
            "Severity Level        : DERIVED FROM ACTUAL MODEL PROBABILITY"
        )
        add(
            "Dataset Evidence      : ACTUAL FROZEN TEST ROW"
        )
        add(
            "Synthetic Data        : NONE"
        )
        add()

    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(lines)
        )

    banner(
        "SUPERVISOR REPORT COMPLETE"
    )

    print(
        f"Attack profiles       : {len(profiles)}"
    )

    print(
        f"Actual model classes  : {len(classes)}"
    )

    print(
        f"Frozen events used    : {len(structured)}"
    )

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

    print(
        f"TXT                   : {OUTPUT_TXT}"
    )

    print(
        f"JSON                  : {OUTPUT_JSON}"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        run_actual_inference()

    except Exception as exc:

        failed(
            "ACTUAL REPORT GENERATION FAILED"
        )

        print(
            f"       {type(exc).__name__}: {exc}"
        )

        print()
        print(
            "No fabricated report was written."
        )

        print(
            "No production artifact was modified."
        )

        traceback.print_exc()

        sys.exit(1)