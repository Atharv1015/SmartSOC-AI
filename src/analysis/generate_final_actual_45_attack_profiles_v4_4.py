"""
SmartSOC AI v4.4
FINAL ACTUAL 45-TECHNIQUE ATTACK & MITIGATION PROFILE REPORT

Purpose
-------
Generate a supervisor-ready report from the REAL frozen v4.1
production artifacts and frozen test dataset.

This script:

1. Loads the serialized v4.1 production model.
2. Loads the exact production preprocessing artifacts.
3. Loads the frozen 11,707-event test dataset.
4. Reconstructs the exact 6,919-dimensional production feature matrix.
5. Runs ACTUAL production inference.
6. Extracts the ACTUAL probability for every one of the 45 MITRE classes.
7. Resolves the ACTUAL production threshold for every class.
8. Determines ACTUAL DETECTED / NOT DETECTED state.
9. Selects one representative frozen event per MITRE class using the
   highest actual model probability for that class.
10. Reports the actual frozen ground-truth label when available.
11. Maps MITRE IDs to technique names.
12. Maps the validated Phase 6 safety boundary to response/authorization
    state without claiming that execution occurred.

IMPORTANT
---------
No synthetic predictions.
No synthetic scores.
No fabricated severity values.
No fabricated attack events.
No response execution.
No containment execution.
Production artifacts are read-only.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "supervisor"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
MANIFEST_FILE = MODEL_DIR / "model_manifest.json"

STRUCTURED_FILE = DATA_DIR / "structured_test.csv"
TEXT_FILE = DATA_DIR / "behavior_text_test.csv"
LABELS_FILE = DATA_DIR / "labels_test.csv"
METADATA_FILE = DATA_DIR / "metadata_test.csv"

TXT_FILE = (
    REPORT_DIR
    / "smartsoc_final_actual_45_attack_profiles_v4_4.txt"
)

JSON_FILE = (
    REPORT_DIR
    / "smartsoc_final_actual_45_attack_profiles_v4_4.json"
)


# ============================================================================
# EXPECTED PRODUCTION DIMENSIONS
# ============================================================================

EXPECTED_CATEGORICAL = 4
EXPECTED_NUMERICAL = 58
EXPECTED_TFIDF = 6699
EXPECTED_FINAL = 6919
EXPECTED_CLASSES = 45


# ============================================================================
# MITRE ATT&CK TECHNIQUE NAMES
#
# These are names for the model's class IDs.
# The model itself supplies the IDs/probabilities.
# ============================================================================

MITRE_NAMES = {
    "T1003.001": "OS Credential Dumping: LSASS Memory",
    "T1003.002": "OS Credential Dumping: Security Account Manager",
    "T1021.006": "Remote Services: Windows Remote Management",
    "T1027": "Obfuscated Files or Information",
    "T1027.004": "Obfuscated Files or Information: Compile After Delivery",
    "T1036": "Masquerading",
    "T1047": "Windows Management Instrumentation",
    "T1053.005": "Scheduled Task/Job: Scheduled Task",
    "T1055": "Process Injection",
    "T1059": "Command and Scripting Interpreter",
    "T1059.001": "Command and Scripting Interpreter: PowerShell",
    "T1059.003": "Command and Scripting Interpreter: Windows Command Shell",
    "T1059.005": "Command and Scripting Interpreter: Visual Basic",
    "T1070": "Indicator Removal",
    "T1070.004": "Indicator Removal: File Deletion",
    "T1078": "Valid Accounts",
    "T1087": "Account Discovery",
    "T1095": "Non-Application Layer Protocol",
    "T1098": "Account Manipulation",
    "T1105": "Ingress Tool Transfer",
    "T1110": "Brute Force",
    "T1110.001": "Brute Force: Password Guessing",
    "T1112": "Modify Registry",
    "T1136": "Create Account",
    "T1210": "Exploitation of Remote Services",
    "T1218": "System Binary Proxy Execution",
    "T1484": "Domain or Tenant Policy Modification",
    "T1485": "Data Destruction",
    "T1489": "Service Stop",
    "T1529": "System Shutdown/Reboot",
    "T1531": "Account Access Removal",
    "T1543.003": "Create or Modify System Process: Windows Service",
    "T1546": "Event Triggered Execution",
    "T1546.003": "Event Triggered Execution: Windows Management Instrumentation Event Subscription",
    "T1546.011": "Event Triggered Execution: Application Shimming",
    "T1547.001": "Boot or Logon Autostart Execution: Registry Run Keys / Startup Folder",
    "T1548": "Abuse Elevation Control Mechanism",
    "T1548.002": "Abuse Elevation Control Mechanism: Bypass User Account Control",
    "T1562.001": "Impair Defenses: Disable or Modify Tools",
    "T1562.004": "Impair Defenses: Disable or Modify System Firewall",
    "T1565.001": "Data Manipulation: Stored Data Manipulation",
    "T1570": "Lateral Tool Transfer",
    "T1574.001": "Hijack Execution Flow: DLL Search Order Hijacking",
    "T1574.002": "Hijack Execution Flow: DLL Side-Loading",
    "T1574.010": "Hijack Execution Flow: Services File Permissions Weakness",
}


# ============================================================================
# HELPERS
# ============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


def safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def numeric_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:

    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    bad = converted.isna()

    if bad.any():
        bad_values = (
            series.loc[bad]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        # Empty strings can legitimately represent absent numeric signals.
        non_empty_bad = [
            x
            for x in bad_values
            if x.strip().lower()
            not in {
                "",
                "nan",
                "none",
                "null",
                "<na>",
            }
        ]

        if non_empty_bad:
            raise ValueError(
                f"Non-numeric values found in numerical feature "
                f"'{column_name}': {non_empty_bad[:10]}"
            )

        converted = converted.fillna(0.0)

    return converted.astype(float)


def normalize_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


# ============================================================================
# THRESHOLD EXTRACTION
# ============================================================================

def find_threshold(
    thresholds: Any,
    technique: str,
) -> float:

    """
    Resolve the actual threshold from thresholds.json.

    Supports the observed v4.1/v4.2-style structures without inventing
    a fallback threshold.
    """

    def recursive(obj: Any) -> Any:

        if isinstance(obj, dict):

            if technique in obj:
                return obj[technique]

            for key in (
                "thresholds",
                "technique_thresholds",
                "mapping",
                "values",
                "data",
            ):
                if key in obj:
                    result = recursive(obj[key])

                    if result is not None:
                        return result

            for value in obj.values():

                result = recursive(value)

                if result is not None:
                    return result

        elif isinstance(obj, list):

            for item in obj:

                result = recursive(item)

                if result is not None:
                    return result

        return None

    raw = recursive(thresholds)

    if raw is None:
        raise KeyError(
            f"No production threshold found for {technique}"
        )

    if isinstance(raw, dict):

        for key in (
            "threshold",
            "value",
            "selected_threshold",
            "decision_threshold",
        ):
            if key in raw:
                raw = raw[key]
                break

    value = float(raw)

    if not math.isfinite(value):
        raise ValueError(
            f"Invalid production threshold for {technique}: {raw}"
        )

    return value


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts() -> Dict[str, Any]:

    print("[1] Loading production artifacts...")

    for path in (
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
        MANIFEST_FILE,
    ):
        require_file(path)

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

    with SCHEMA_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        schema = json.load(handle)

    with THRESHOLDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        thresholds = json.load(handle)

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as handle:
        manifest = json.load(handle)

    print("[PASS] Production artifacts loaded")
    print(
        f"       Model : {type(model).__name__}"
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
# SCHEMA
# ============================================================================

def resolve_schema(
    schema: Dict[str, Any],
) -> tuple[List[str], List[str]]:

    categorical = (
        schema.get("categorical_columns")
        or schema.get("categorical_features")
        or schema.get("categorical")
        or []
    )

    numerical = (
        schema.get("numerical_columns")
        or schema.get("numerical_features")
        or schema.get("numerical")
        or []
    )

    categorical = [
        str(x)
        for x in categorical
    ]

    numerical = [
        str(x)
        for x in numerical
    ]

    if len(categorical) != EXPECTED_CATEGORICAL:
        raise ValueError(
            f"Expected {EXPECTED_CATEGORICAL} categorical features, "
            f"got {len(categorical)}"
        )

    if len(numerical) != EXPECTED_NUMERICAL:
        raise ValueError(
            f"Expected {EXPECTED_NUMERICAL} numerical features, "
            f"got {len(numerical)}"
        )

    return categorical, numerical


# ============================================================================
# DATASET
# ============================================================================

def load_dataset() -> Dict[str, pd.DataFrame]:

    print()
    print("[2] Loading frozen test dataset...")

    for path in (
        STRUCTURED_FILE,
        TEXT_FILE,
        LABELS_FILE,
        METADATA_FILE,
    ):
        require_file(path)

    structured = pd.read_csv(
        STRUCTURED_FILE
    )

    text = pd.read_csv(
        TEXT_FILE
    )

    labels = pd.read_csv(
        LABELS_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    lengths = {
        "structured": len(structured),
        "text": len(text),
        "labels": len(labels),
        "metadata": len(metadata),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Frozen dataset alignment failure: {lengths}"
        )

    print(
        "[PASS] Frozen dataset alignment: "
        f"{len(structured)} rows"
    )

    return {
        "structured": structured,
        "text": text,
        "labels": labels,
        "metadata": metadata,
    }


# ============================================================================
# LABEL EXTRACTION
# ============================================================================

def extract_label_strings(
    labels: pd.DataFrame,
) -> List[str]:

    if labels.shape[1] == 0:
        return [""] * len(labels)

    column = labels.columns[0]

    return [
        safe_string(x)
        for x in labels[column]
    ]


def labels_for_row(
    label_string: str,
) -> List[str]:

    if not label_string:
        return []

    result = []

    for value in label_string.split(","):

        value = value.strip()

        if value:
            result.append(value)

    return result


# ============================================================================
# FEATURE MATRIX
# ============================================================================

def build_feature_matrix(
    dataset: Dict[str, pd.DataFrame],
    artifacts: Dict[str, Any],
    categorical_columns: List[str],
    numerical_columns: List[str],
):

    structured = dataset["structured"].copy()
    text = dataset["text"].copy()

    # ------------------------------------------------------------
    # Categorical
    # ------------------------------------------------------------

    missing_cat = [
        c
        for c in categorical_columns
        if c not in structured.columns
    ]

    if missing_cat:
        raise KeyError(
            f"Missing categorical columns: {missing_cat}"
        )

    categorical_df = structured[
        categorical_columns
    ].copy()

    for column in categorical_columns:

        categorical_df[column] = (
            categorical_df[column]
            .map(safe_string)
        )

    categorical_matrix = (
        artifacts["encoder"]
        .transform(categorical_df)
    )

    if categorical_matrix.shape[1] != EXPECTED_CATEGORICAL:
        # Do not confuse 4 input columns with 162 encoded dimensions.
        if categorical_matrix.shape[1] != 162:
            raise ValueError(
                "Unexpected encoded categorical dimension: "
                f"{categorical_matrix.shape}"
            )

    # ------------------------------------------------------------
    # Numerical
    # ------------------------------------------------------------

    missing_num = [
        c
        for c in numerical_columns
        if c not in structured.columns
    ]

    if missing_num:
        raise KeyError(
            f"Missing numerical columns: {missing_num}"
        )

    numerical_df = structured[
        numerical_columns
    ].copy()

    for column in numerical_columns:

        numerical_df[column] = numeric_series(
            numerical_df[column],
            column,
        )

    numerical_array = numerical_df.to_numpy(
        dtype=float
    )

    if numerical_array.shape[1] != EXPECTED_NUMERICAL:
        raise ValueError(
            "Unexpected numerical dimension: "
            f"{numerical_array.shape}"
        )

    numerical_scaled = (
        artifacts["scaler"]
        .transform(numerical_array)
    )

    if numerical_scaled.shape[1] != EXPECTED_NUMERICAL:
        raise ValueError(
            "Unexpected scaled numerical dimension: "
            f"{numerical_scaled.shape}"
        )

    # ------------------------------------------------------------
    # Behavior text
    # ------------------------------------------------------------

    if "behavior_text" in text.columns:

        behavior = (
            text["behavior_text"]
            .fillna("")
            .astype(str)
        )

    else:

        behavior = (
            text.iloc[:, 0]
            .fillna("")
            .astype(str)
        )

    tfidf_matrix = (
        artifacts["tfidf"]
        .transform(behavior)
    )

    if tfidf_matrix.shape[1] != EXPECTED_TFIDF:
        raise ValueError(
            "Unexpected TF-IDF dimension: "
            f"{tfidf_matrix.shape}"
        )

    # ------------------------------------------------------------
    # Exact production hybrid matrix
    # ------------------------------------------------------------

    final_matrix = hstack(
        [
            categorical_matrix,
            numerical_scaled,
            tfidf_matrix,
        ],
        format="csr",
    )

    if final_matrix.shape[1] != EXPECTED_FINAL:
        raise ValueError(
            "Unexpected final feature dimension: "
            f"{final_matrix.shape}; "
            f"expected {EXPECTED_FINAL}"
        )

    return final_matrix


# ============================================================================
# INFERENCE
# ============================================================================

def run_actual_inference(
    matrix,
    artifacts: Dict[str, Any],
) -> np.ndarray:

    print()
    print("[3] Running ACTUAL frozen production inference...")

    model = artifacts["model"]

    probabilities = model.predict_proba(
        matrix
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"Unexpected probability dimensions: "
            f"{probabilities.shape}"
        )

    if probabilities.shape[1] != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"got {probabilities.shape[1]}"
        )

    print(
        "[PASS] Actual probability matrix: "
        f"{probabilities.shape}"
    )

    return probabilities


# ============================================================================
# PROFILE GENERATION
# ============================================================================

def build_profiles(
    probabilities: np.ndarray,
    dataset: Dict[str, pd.DataFrame],
    artifacts: Dict[str, Any],
) -> List[Dict[str, Any]]:

    label_encoder = artifacts["label_encoder"]

    techniques = [
        str(x)
        for x in label_encoder.classes_
    ]

    if len(techniques) != EXPECTED_CLASSES:
        raise ValueError(
            f"Production model exposes {len(techniques)} classes, "
            f"expected {EXPECTED_CLASSES}"
        )

    label_strings = extract_label_strings(
        dataset["labels"]
    )

    profiles = []

    for class_index, technique in enumerate(techniques):

        scores = probabilities[
            :,
            class_index
        ]

        scores = np.asarray(
            scores,
            dtype=float
        )

        representative_index = int(
            np.nanargmax(scores)
        )

        actual_score = normalize_probability(
            scores[representative_index]
        )

        threshold = find_threshold(
            artifacts["thresholds"],
            technique,
        )

        detected = (
            actual_score >= threshold
        )

        ground_truth_labels = labels_for_row(
            label_strings[representative_index]
        )

        ground_truth_contains_class = (
            technique in ground_truth_labels
        )

        # --------------------------------------------------------
        # IMPORTANT:
        # Severity is NOT invented from probability.
        #
        # The actual model output is probability.
        # Phase 6 risk/severity is a separate policy layer.
        # --------------------------------------------------------

        profile = {
            "class_index": class_index + 1,
            "technique": technique,
            "attack_name": MITRE_NAMES.get(
                technique,
                "MITRE ATT&CK technique",
            ),
            "model_probability": actual_score,
            "model_probability_percent": (
                actual_score * 100.0
            ),
            "production_threshold": threshold,
            "detected_at_threshold": bool(detected),
            "representative_event_index": (
                representative_index + 1
            ),
            "frozen_ground_truth_labels": (
                ground_truth_labels
            ),
            "ground_truth_contains_class": (
                bool(ground_truth_contains_class)
            ),
            "severity_score": None,
            "severity_level": (
                "NOT A MODEL OUTPUT"
            ),
            "risk_disposition": (
                "RESOLVED BY PHASE 6 POLICY "
                "ONLY WHEN EVENT DECISION IS AVAILABLE"
            ),
            "analyst_review": "REQUIRED FOR EXECUTION",
            "response_required": (
                "YES"
                if detected
                else "NO AUTOMATED RESPONSE"
            ),
            "containment_candidate": (
                "CANDIDATE ONLY"
                if detected
                else "NO"
            ),
            "automatic_response": "DISABLED",
            "automatic_containment": "DISABLED",
            "authorization_state": (
                "PENDING ANALYST AUTHORIZATION"
                if detected
                else "NO EXECUTION PATH"
            ),
            "ai_authorization_sufficient": "NO",
            "execution_state": "NOT EXECUTED",
            "containment_state": "NOT EXECUTED",
        }

        profiles.append(
            profile
        )

    return profiles


# ============================================================================
# RESPONSE POLICY
# ============================================================================

def mitigation_for_profile(
    profile: Dict[str, Any],
) -> Dict[str, str]:

    detected = profile[
        "detected_at_threshold"
    ]

    if not detected:

        return {
            "isolate_host": "NOT SELECTED",
            "kill_process": "NOT SELECTED",
            "block_ip_domain": "NOT SELECTED",
            "disable_account": "NOT SELECTED",
            "quarantine_file": "NOT SELECTED",
            "block_hash": "NOT SELECTED",
            "network_containment": "NOT SELECTED",
            "log_enrichment": "NOT SELECTED",
            "increase_monitoring": "NOT SELECTED",
            "alert_analyst": "YES",
            "log_only": "YES",
        }

    # This is deliberately expressed as a PLANNED RESPONSE
    # behind the Phase 6 authorization boundary.
    #
    # It is NOT execution.
    return {
        "isolate_host": "CANDIDATE",
        "kill_process": "CANDIDATE",
        "block_ip_domain": "CANDIDATE",
        "disable_account": "CANDIDATE",
        "quarantine_file": "CANDIDATE",
        "block_hash": "CANDIDATE",
        "network_containment": "CANDIDATE",
        "log_enrichment": "YES",
        "increase_monitoring": "YES",
        "alert_analyst": "YES",
        "log_only": "NO",
    }


# ============================================================================
# TEXT REPORT
# ============================================================================

def check(value: str) -> str:

    if value in {
        "YES",
        "CANDIDATE",
    }:
        return "[✓]"

    if value in {
        "NO",
        "NOT SELECTED",
    }:
        return "[ ]"

    return f"[{value}]"


def format_profile(
    profile: Dict[str, Any],
) -> str:

    mitigation = mitigation_for_profile(
        profile
    )

    score = profile[
        "model_probability"
    ]

    threshold = profile[
        "production_threshold"
    ]

    gt = profile[
        "frozen_ground_truth_labels"
    ]

    gt_text = (
        ", ".join(gt)
        if gt
        else "NONE"
    )

    detected = (
        "DETECTED"
        if profile[
            "detected_at_threshold"
        ]
        else "NOT DETECTED"
    )

    lines = []

    lines.append(
        "=" * 64
    )

    lines.append(
        f"ATTACK PROFILE "
        f"{profile['class_index']:02d} / {EXPECTED_CLASSES}"
    )

    lines.append(
        "=" * 64
    )

    lines.append(
        f"Attack ID        : {profile['technique']}"
    )

    lines.append(
        f"Attack           : {profile['attack_name']}"
    )

    lines.append("")

    lines.append(
        "AI ASSESSMENT"
    )

    lines.append(
        "-------------"
    )

    lines.append(
        f"Predicted Class       : {profile['technique']}"
    )

    lines.append(
        f"Model Probability     : {score:.6f}"
    )

    lines.append(
        f"Model Score (%)       : {score * 100:.2f}%"
    )

    lines.append(
        f"Production Threshold  : {threshold:.6f}"
    )

    lines.append(
        f"Detection Result      : {detected}"
    )

    lines.append(
        "Severity Score        : "
        "NOT A DIRECT MODEL OUTPUT"
    )

    lines.append(
        "Severity Level        : "
        "NOT DERIVED FROM PROBABILITY"
    )

    lines.append(
        f"Representative Event  : "
        f"{profile['representative_event_index']}"
    )

    lines.append("")

    lines.append(
        "FROZEN DATASET EVIDENCE"
    )

    lines.append(
        "-----------------------"
    )

    lines.append(
        f"Ground Truth Labels    : {gt_text}"
    )

    lines.append(
        "Ground Truth Match     : "
        + (
            "YES"
            if profile[
                "ground_truth_contains_class"
            ]
            else "NO"
        )
    )

    lines.append("")

    lines.append(
        "DECISION POLICY"
    )

    lines.append(
        "---------------"
    )

    lines.append(
        f"Analyst Review         : "
        f"{profile['analyst_review']}"
    )

    lines.append(
        f"Response Required      : "
        f"{profile['response_required']}"
    )

    lines.append(
        f"Containment Candidate  : "
        f"{profile['containment_candidate']}"
    )

    lines.append(
        f"Automatic Response     : "
        f"{profile['automatic_response']}"
    )

    lines.append(
        f"Automatic Containment  : "
        f"{profile['automatic_containment']}"
    )

    lines.append("")

    lines.append(
        "RESPONSE PLANNER"
    )

    lines.append(
        "----------------"
    )

    if profile[
        "detected_at_threshold"
    ]:

        lines.append(
            "Response Category      : "
            "RESPONSE PLANNING REQUIRED"
        )

        lines.append(
            "Investigation          : REQUIRED"
        )

        lines.append(
            "Rollback               : "
            "REQUIRED WHERE APPLICABLE"
        )

    else:

        lines.append(
            "Response Category      : "
            "NO AUTOMATED RESPONSE"
        )

        lines.append(
            "Investigation          : "
            "NOT AUTOMATICALLY INITIATED"
        )

        lines.append(
            "Rollback               : "
            "NOT APPLICABLE"
        )

    lines.append("")

    lines.append(
        "PLANNED MITIGATION"
    )

    lines.append(
        "------------------"
    )

    lines.append(
        f"{check(mitigation['isolate_host'])} "
        "Isolate Host"
    )

    lines.append(
        f"{check(mitigation['kill_process'])} "
        "Kill Malicious Process"
    )

    lines.append(
        f"{check(mitigation['block_ip_domain'])} "
        "Block IP / Domain"
    )

    lines.append(
        f"{check(mitigation['disable_account'])} "
        "Disable Account"
    )

    lines.append(
        f"{check(mitigation['quarantine_file'])} "
        "Quarantine File"
    )

    lines.append(
        f"{check(mitigation['block_hash'])} "
        "Block Hash"
    )

    lines.append(
        f"{check(mitigation['network_containment'])} "
        "Network Containment"
    )

    lines.append(
        f"{check(mitigation['log_enrichment'])} "
        "Log Enrichment"
    )

    lines.append(
        f"{check(mitigation['increase_monitoring'])} "
        "Increase Monitoring"
    )

    lines.append(
        f"{check(mitigation['alert_analyst'])} "
        "Alert Analyst"
    )

    lines.append(
        f"{check(mitigation['log_only'])} "
        "Log Only"
    )

    lines.append("")

    lines.append(
        "AUTHORIZATION"
    )

    lines.append(
        "-------------"
    )

    lines.append(
        f"Authorization State   : "
        f"{profile['authorization_state']}"
    )

    lines.append(
        f"Analyst Approval      : "
        f"{profile['analyst_review']}"
    )

    lines.append(
        f"AI Authorization      : "
        f"{profile['ai_authorization_sufficient']}"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append(
        "Automatic Containment : DISABLED"
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
        "Inference             : "
        "ACTUAL SERIALIZED PRODUCTION MODEL"
    )

    lines.append(
        "Model Score           : "
        "ACTUAL PRODUCTION PROBABILITY"
    )

    lines.append(
        "Threshold             : "
        "ACTUAL PRODUCTION THRESHOLD"
    )

    lines.append(
        "Detection             : "
        "ACTUAL SCORE vs THRESHOLD COMPARISON"
    )

    lines.append(
        "Ground Truth          : "
        "ACTUAL FROZEN TEST DATASET LABEL"
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

    return "\n".join(lines)


# ============================================================================
# JSON REPORT
# ============================================================================

def write_json(
    profiles: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    row_count: int,
) -> None:

    manifest = artifacts[
        "manifest"
    ]

    output = {
        "report": {
            "name": (
                "SmartSOC AI — Actual 45-Technique "
                "Attack & Mitigation Profile Report"
            ),
            "version": "4.4",
            "model_version": manifest.get(
                "version",
                "4.1",
            ),
            "model_classes": EXPECTED_CLASSES,
            "frozen_event_rows": row_count,
            "synthetic_predictions": False,
            "synthetic_scores": False,
            "automatic_execution": False,
            "automatic_containment": False,
            "production_artifacts_modified": False,
        },
        "profiles": profiles,
    }

    with JSON_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            output,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print(
        "=" * 72
    )

    print(
        "SmartSOC AI — ACTUAL 45-TECHNIQUE "
        "ATTACK & MITIGATION REPORT"
    )

    print(
        "Version 4.4"
    )

    print(
        "=" * 72
    )

    warnings.filterwarnings(
        "ignore",
        message=".*feature names.*",
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
    )

    try:

        artifacts = load_artifacts()

        dataset = load_dataset()

        categorical_columns, numerical_columns = (
            resolve_schema(
                artifacts["schema"]
            )
        )

        print()
        print(
            "[PASS] Frozen feature schema resolved"
        )

        print(
            f"       Categorical input : "
            f"{len(categorical_columns)}"
        )

        print(
            f"       Numerical         : "
            f"{len(numerical_columns)}"
        )

        print(
            f"       Expected TF-IDF   : "
            f"{EXPECTED_TFIDF}"
        )

        print(
            f"       Final features    : "
            f"{EXPECTED_FINAL}"
        )

        matrix = build_feature_matrix(
            dataset,
            artifacts,
            categorical_columns,
            numerical_columns,
        )

        print()
        print(
            "[PASS] Production feature matrix: "
            f"{matrix.shape}"
        )

        probabilities = run_actual_inference(
            matrix,
            artifacts,
        )

        profiles = build_profiles(
            probabilities,
            dataset,
            artifacts,
        )

        if len(profiles) != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Expected {EXPECTED_CLASSES} profiles, "
                f"generated {len(profiles)}"
            )

        print()
        print(
            f"[PASS] {len(profiles)} actual production "
            "classes profiled"
        )

        # ------------------------------------------------------------
        # TXT
        # ------------------------------------------------------------

        report_lines = []

        report_lines.append(
            "=" * 72
        )

        report_lines.append(
            "SMARTSOC AI — ACTUAL 45-TECHNIQUE "
            "ATTACK & MITIGATION PROFILE REPORT"
        )

        report_lines.append(
            "=" * 72
        )

        report_lines.append("")

        report_lines.append(
            "Model Version        : 4.1"
        )

        report_lines.append(
            "Model Classes        : 45"
        )

        report_lines.append(
            "Inference Source     : Frozen Production Model"
        )

        report_lines.append(
            "Dataset              : Frozen Test Dataset"
        )

        report_lines.append(
            "Frozen Event Rows    : 11707"
        )

        report_lines.append(
            "Synthetic Scores     : DISABLED"
        )

        report_lines.append(
            "Synthetic Predictions: DISABLED"
        )

        report_lines.append(
            "Execution            : DISABLED"
        )

        report_lines.append(
            "Containment          : DISABLED"
        )

        report_lines.append("")

        report_lines.append(
            "IMPORTANT MODEL OUTPUT NOTE"
        )

        report_lines.append(
            "----------------------------"
        )

        report_lines.append(
            "The actual model output is a probability "
            "for each MITRE class."
        )

        report_lines.append(
            "Severity Score is NOT fabricated from "
            "the probability."
        )

        report_lines.append(
            "Detection is determined by the actual "
            "production threshold."
        )

        report_lines.append(
            "Response planning remains behind the "
            "validated Phase 6 authorization boundary."
        )

        report_lines.append("")

        for profile in profiles:

            report_lines.append(
                format_profile(
                    profile
                )
            )

            report_lines.append("")

        report_lines.append(
            "=" * 72
        )

        report_lines.append(
            "END OF ACTUAL PRODUCTION MODEL REPORT"
        )

        report_lines.append(
            "=" * 72
        )

        with TXT_FILE.open(
            "w",
            encoding="utf-8",
        ) as handle:

            handle.write(
                "\n".join(
                    report_lines
                )
            )

        write_json(
            profiles,
            artifacts,
            len(dataset["structured"]),
        )

        print()
        print(
            "========================================================================"
        )

        print(
            "SUPERVISOR REPORT COMPLETE"
        )

        print(
            "========================================================================"
        )

        print(
            f"Actual frozen events : "
            f"{len(dataset['structured'])}"
        )

        print(
            "Actual probability matrix : "
            f"{probabilities.shape}"
        )

        print(
            "Actual MITRE classes : 45"
        )

        print(
            "Synthetic predictions : DISABLED"
        )

        print(
            "Synthetic scores : DISABLED"
        )

        print(
            "Response execution : DISABLED"
        )

        print(
            "Containment execution : DISABLED"
        )

        print(
            f"TXT  : {TXT_FILE}"
        )

        print(
            f"JSON : {JSON_FILE}"
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

        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )