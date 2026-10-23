from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SmartSOC Phase 6.7
# Offline End-to-End Simulation
# Version 4.1
#
# READ-ONLY VALIDATION
# - No model modification
# - No threshold modification
# - No dataset modification
# - No Wazuh
# - No Shuffle
# - No response execution
# - No containment execution
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

VERSION = "4.1"
PHASE = "6.7"

FINAL_DATASET_DIR = ROOT / "data" / "final" / "windows_apt_v4_1"
PRODUCTION_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

RISK_ENGINE_FILE = ROOT / "src" / "decision" / "smartsoc_risk_engine_v4_1.py"
DECISION_POLICY_FILE = ROOT / "src" / "decision" / "smartsoc_decision_policy_v4_1.py"
RESPONSE_PLANNER_FILE = ROOT / "src" / "decision" / "smartsoc_response_planner_v4_1.py"
AUTHORIZATION_FILE = ROOT / "src" / "decision" / "smartsoc_analyst_authorization_v4_1.py"

REPORT_DIR = ROOT / "reports" / "phase6" / "gate6_7"
REPORT_FILE = REPORT_DIR / "phase6_gate6_7_offline_end_to_end_report.json"


# ============================================================================
# Utility functions
# ============================================================================

def print_header():
    print("=" * 72)
    print("SmartSOC Phase 6.7")
    print("Offline End-to-End Simulation")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()


def record_check(
    checks: List[Dict[str, Any]],
    passed: bool,
    name: str,
    details: str = "",
):
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "details": details,
        }
    )

    if passed:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}")

    if details:
        print(f"       {details}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for dataclasses/module registration.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def to_dense(value):
    if hasattr(value, "toarray"):
        value = value.toarray()

    arr = np.asarray(
        value,
        dtype=float,
    )

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array, got {arr.shape}"
        )

    return arr


def safe_float(value, default=0.0):
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def json_safe(value):
    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def sha256_file(path: Path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================================
# Artifact loading
# ============================================================================

def load_artifacts():

    paths = {
        "model": PRODUCTION_DIR / "model.joblib",
        "encoder": PRODUCTION_DIR / "structured_encoder.joblib",
        "scaler": PRODUCTION_DIR / "numerical_scaler.joblib",
        "tfidf": PRODUCTION_DIR / "tfidf_vectorizer.joblib",
        "label_encoder": PRODUCTION_DIR / "multilabel_encoder.joblib",
        "schema": PRODUCTION_DIR / "feature_schema.json",
        "thresholds": PRODUCTION_DIR / "thresholds.json",
        "manifest": PRODUCTION_DIR / "model_manifest.json",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {name}: {path}"
            )

    return {
        "model": joblib.load(paths["model"]),
        "encoder": joblib.load(paths["encoder"]),
        "scaler": joblib.load(paths["scaler"]),
        "tfidf": joblib.load(paths["tfidf"]),
        "label_encoder": joblib.load(paths["label_encoder"]),
        "schema": load_json(paths["schema"]),
        "thresholds": load_json(paths["thresholds"]),
        "manifest": load_json(paths["manifest"]),
    }


# ============================================================================
# Dataset
# ============================================================================

def load_dataset():

    structured = pd.read_csv(
        FINAL_DATASET_DIR / "structured_test.csv"
    )

    labels = pd.read_csv(
        FINAL_DATASET_DIR / "labels_test.csv"
    )

    metadata = pd.read_csv(
        FINAL_DATASET_DIR / "metadata_test.csv"
    )

    behavior = pd.read_csv(
        FINAL_DATASET_DIR / "behavior_text_test.csv"
    )

    return structured, labels, metadata, behavior


def behavior_text_at(
    behavior: pd.DataFrame,
    index: int,
) -> str:

    row = behavior.iloc[index]

    for column in (
        "behavior_text_normalized",
        "behavior_text",
        "text",
    ):

        if column in behavior.columns:

            value = row[column]

            if pd.isna(value):
                return ""

            return str(value)

    if len(behavior.columns) > 0:

        value = row.iloc[0]

        if not pd.isna(value):
            return str(value)

    return ""


# ============================================================================
# Schema
# ============================================================================

def resolve_schema(
    schema: Dict[str, Any],
    dataset_columns: List[str],
):

    def first_list(keys):

        for key in keys:

            value = schema.get(key)

            if isinstance(value, list) and value:
                return [
                    str(x)
                    for x in value
                ]

        return None

    categorical = first_list(
        [
            "categorical_columns",
            "categorical_features",
            "categorical",
        ]
    )

    numerical = first_list(
        [
            "numerical_columns",
            "numerical_features",
            "numerical",
        ]
    )

    structured = first_list(
        [
            "structured_columns",
            "structured_features",
            "structured",
            "features",
        ]
    )

    if structured is None:
        structured = list(dataset_columns)

    if categorical is None:
        categorical = [
            "system_level",
            "process_name",
            "parent_process_name",
            "process_extension",
        ]

    if numerical is None:
        numerical = [
            x
            for x in structured
            if x not in categorical
        ]

    return structured, categorical, numerical


# ============================================================================
# Threshold resolution
#
# Handles both:
#
# 1. {"T1059.003": 0.90}
#
# 2. {"thresholds": {"T1059.003": 0.90}}
#
# 3. {"T1059.003": {"threshold": 0.90}}
#
# 4. {"thresholds": {"T1059.003": {"threshold": 0.90}}}
#
# No threshold is invented.
# ============================================================================

def resolve_threshold_mapping(
    raw_thresholds: Any,
) -> Dict[str, float]:

    if not isinstance(
        raw_thresholds,
        dict,
    ):
        return {}

    candidates = []

    candidates.append(
        raw_thresholds
    )

    for key in (
        "thresholds",
        "per_technique",
        "technique_thresholds",
        "values",
    ):

        value = raw_thresholds.get(key)

        if isinstance(
            value,
            dict,
        ):

            candidates.append(
                value
            )

    result = {}

    for mapping in candidates:

        for technique, value in mapping.items():

            technique = str(
                technique
            )

            threshold = None

            if isinstance(
                value,
                (int, float),
            ):

                threshold = safe_float(
                    value,
                    default=np.nan,
                )

            elif isinstance(
                value,
                dict,
            ):

                for key in (
                    "threshold",
                    "value",
                    "selected_threshold",
                    "decision_threshold",
                ):

                    if key in value:

                        threshold = safe_float(
                            value[key],
                            default=np.nan,
                        )

                        break

            if (
                threshold is not None
                and np.isfinite(threshold)
                and 0.0 <= threshold <= 1.0
            ):

                result[technique] = threshold

    return result


# ============================================================================
# Feature transformation
# ============================================================================

def transform_event(
    row: pd.Series,
    behavior_text: str,
    artifacts: Dict[str, Any],
    categorical_features: List[str],
    numerical_features: List[str],
):

    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    categorical_frame = pd.DataFrame(
        [
            {
                feature: row.get(
                    feature,
                    np.nan,
                )
                for feature in categorical_features
            }
        ]
    )

    encoded = to_dense(
        encoder.transform(
            categorical_frame
        )
    )

    numerical_values = []

    for feature in numerical_features:

        value = row.get(
            feature,
            0.0,
        )

        if pd.isna(value):
            value = 0.0

        numerical_values.append(
            safe_float(
                value,
                0.0,
            )
        )

    numerical = np.asarray(
        numerical_values,
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    scaled = to_dense(
        scaler.transform(
            numerical
        )
    )

    text = str(
        behavior_text or ""
    )

    text_features = to_dense(
        tfidf.transform(
            [text]
        )
    )

    final = np.hstack(
        (
            encoded,
            scaled,
            text_features,
        )
    )

    final = to_dense(final)

    return final, {
        "categorical_shape":
            list(encoded.shape),

        "numerical_shape":
            list(scaled.shape),

        "tfidf_shape":
            list(text_features.shape),

        "final_shape":
            list(final.shape),
    }


# ============================================================================
# Prediction
# ============================================================================

def predict_event(
    final_matrix,
    artifacts,
    thresholds,
):

    model = artifacts["model"]
    label_encoder = artifacts["label_encoder"]

    probabilities = model.predict_proba(
        final_matrix
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(
            1,
            -1,
        )

    if probabilities.shape != (
        1,
        len(label_encoder.classes_),
    ):

        raise ValueError(
            "Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Non-finite prediction probability"
        )

    row = probabilities[0]

    classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    ranking = np.argsort(row)[::-1]

    top_index = int(
        ranking[0]
    )

    top_technique = classes[
        top_index
    ]

    top_probability = float(
        row[top_index]
    )

    second_probability = (
        float(row[ranking[1]])
        if len(ranking) > 1
        else 0.0
    )

    margin = (
        top_probability
        - second_probability
    )

    all_results = []

    detected = []

    for index in ranking:

        technique = classes[
            int(index)
        ]

        probability = float(
            row[int(index)]
        )

        if technique in thresholds:

            threshold = float(
                thresholds[technique]
            )

            threshold_available = True

            is_detected = (
                probability
                >= threshold
            )

        else:

            threshold = None

            threshold_available = False

            is_detected = False

        result = {
            "technique":
                technique,

            "probability":
                probability,

            "threshold":
                threshold,

            "threshold_available":
                threshold_available,

            "detected":
                is_detected,
        }

        all_results.append(
            result
        )

        if is_detected:
            detected.append(
                technique
            )

    return {
        "probabilities":
            row.tolist(),

        "top_technique":
            top_technique,

        "top_probability":
            top_probability,

        "second_probability":
            second_probability,

        "margin":
            float(margin),

        "detected_techniques":
            detected,

        "detected_count":
            len(detected),

        "fallback":
            len(detected) == 0,

        "top5":
            all_results[:5],

        "threshold_available_count":
            sum(
                1
                for x in all_results
                if x["threshold_available"]
            ),

        "threshold_missing_count":
            sum(
                1
                for x in all_results
                if not x["threshold_available"]
            ),
    }


# ============================================================================
# Evidence
# ============================================================================

def assess_evidence(
    row,
    behavior_text,
):

    signals = []

    fields = [
        "uses_powershell",
        "uses_cmd",
        "uses_net",
        "uses_whoami",
        "uses_schtasks",
        "uses_reg",
        "uses_rundll32",
        "uses_wmic",
        "uses_bitsadmin",
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
        "suspicious_parent_wazuh",
        "has_source_ip",
        "has_destination_ip",
        "is_network_event",
        "administrative_powershell_context",
        "strong_behavior_signal",
    ]

    for field in fields:

        try:

            if bool(
                row.get(
                    field,
                    0,
                )
            ):

                signals.append(
                    field
                )

        except Exception:
            pass

    text_present = bool(
        str(
            behavior_text or ""
        ).strip()
    )

    if text_present:
        signals.append(
            "behavior_text_present"
        )

    if safe_float(
        row.get(
            "command_length",
            0,
        )
    ) > 0:

        signals.append(
            "command_length_present"
        )

    count = len(
        signals
    )

    if count >= 6:
        strength = "STRONG"

    elif count >= 3:
        strength = "MODERATE"

    elif count >= 1:
        strength = "WEAK"

    else:
        strength = "NONE"

    return strength, {
        "signal_count":
            count,

        "signals":
            signals,

        "behavior_text_present":
            text_present,
    }


# ============================================================================
# Context
# ============================================================================

def derive_context(row):

    if bool(
        row.get(
            "is_admin_user",
            False,
        )
    ):

        privilege = "HIGH"

    elif bool(
        row.get(
            "integrity_high",
            False,
        )
    ):

        privilege = "HIGH"

    elif bool(
        row.get(
            "integrity_medium",
            False,
        )
    ):

        privilege = "MEDIUM"

    else:

        privilege = "LOW"

    return {
        "privilege":
            privilege,

        "network":
            bool(
                row.get(
                    "is_network_event",
                    False,
                )
            ),

        "suspicious_parent":
            bool(
                row.get(
                    "suspicious_parent_wazuh",
                    False,
                )
            ),
    }


# ============================================================================
# Technique severity
# ============================================================================

TECHNIQUE_SEVERITY = {
    "T1059": "MEDIUM",
    "T1059.001": "HIGH",
    "T1059.003": "MEDIUM",
    "T1105": "HIGH",
    "T1570": "HIGH",
    "T1027.004": "HIGH",
    "T1543.003": "HIGH",
    "T1112": "HIGH",
    "T1053.005": "HIGH",
    "T1070.004": "HIGH",
    "T1078": "HIGH",
    "T1485": "CRITICAL",
    "T1565.001": "CRITICAL",
    "T1531": "CRITICAL",
    "T1562.001": "HIGH",
    "T1055": "HIGH",
    "T1095": "HIGH",
    "T1562.004": "HIGH",
}


def severity_for(technique):

    return TECHNIQUE_SEVERITY.get(
        technique,
        "MEDIUM",
    )


# ============================================================================
# Rollback
# ============================================================================

ROLLBACK_ACTIONS = {
    "COLLECT_ADDITIONAL_EVIDENCE":
        "NO_STATE_CHANGE_EXPECTED",

    "REVIEW_PROCESS_TREE":
        "NO_STATE_CHANGE_EXPECTED",

    "REVIEW_NETWORK_CONNECTIONS":
        "NO_STATE_CHANGE_EXPECTED",

    "REVIEW_PERSISTENCE_MECHANISM":
        "NO_STATE_CHANGE_EXPECTED",

    "PRESERVE_ARTIFACTS":
        "VERIFY_ARTIFACT_PRESERVATION_COMPLETION",

    "PREPARE_PROCESS_ISOLATION":
        "VERIFY_PROCESS_STATE_AND_RESTORE_APPROVED_PROCESS_STATE",

    "PREPARE_HOST_ISOLATION":
        "VERIFY_HOST_NETWORK_STATE_AND_RESTORE_APPROVED_NETWORK_STATE",

    "PREPARE_IP_BLOCK":
        "VERIFY_BLOCK_RULE_AND_REMOVE_APPROVED_BLOCK_RULE",

    "PREPARE_ACCOUNT_DISABLE":
        "VERIFY_ACCOUNT_STATE_AND_RESTORE_APPROVED_ACCOUNT_STATE",
}


def rollback_valid(plan):

    actions = getattr(
        plan,
        "recommended_actions",
        [],
    ) or []

    for action in actions:

        if action not in ROLLBACK_ACTIONS:

            return False

    return True


# ============================================================================
# Main
# ============================================================================

def main():

    print_header()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks = []
    event_results = []
    failures = []

    # ------------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------------

    try:

        artifacts = load_artifacts()

        record_check(
            checks,
            True,
            "Production artifacts loaded",
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Production artifacts loaded",
            str(exc),
        )

        return write_report(
            checks,
            event_results,
            failures,
            "FAILED",
        )

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    try:

        (
            structured,
            labels,
            metadata,
            behavior,
        ) = load_dataset()

        record_check(
            checks,
            True,
            "Frozen test dataset loaded",
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Frozen test dataset loaded",
            str(exc),
        )

        return write_report(
            checks,
            event_results,
            failures,
            "FAILED",
        )

    alignment_ok = (
        len(structured)
        ==
        len(labels)
        ==
        len(metadata)
        ==
        len(behavior)
    )

    record_check(
        checks,
        alignment_ok,
        "Dataset row alignment",
        f"rows={len(structured)}",
    )

    # ------------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------------

    try:

        (
            structured_features,
            categorical_features,
            numerical_features,
        ) = resolve_schema(
            artifacts["schema"],
            list(structured.columns),
        )

        schema_ok = (
            len(structured_features) == 62
            and
            len(categorical_features) == 4
            and
            len(numerical_features) == 58
        )

        record_check(
            checks,
            schema_ok,
            "Frozen feature schema resolved",
            (
                f"Structured features : {len(structured_features)}\n"
                f"       Categorical features: {len(categorical_features)}\n"
                f"       Numerical features  : {len(numerical_features)}"
            ),
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Frozen feature schema resolved",
            str(exc),
        )

        return write_report(
            checks,
            event_results,
            failures,
            "FAILED",
        )

    # ------------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------------

    try:

        risk_module = load_module(
            RISK_ENGINE_FILE,
            "smartsoc_risk_engine_v4_1",
        )

        decision_module = load_module(
            DECISION_POLICY_FILE,
            "smartsoc_decision_policy_v4_1",
        )

        response_module = load_module(
            RESPONSE_PLANNER_FILE,
            "smartsoc_response_planner_v4_1",
        )

        authorization_module = load_module(
            AUTHORIZATION_FILE,
            "smartsoc_analyst_authorization_v4_1",
        )

        record_check(
            checks,
            True,
            "Risk / decision / response / authorization components loaded",
        )

        risk_engine = (
            risk_module.SmartSOCRiskEngine()
        )

        decision_engine = (
            decision_module.SmartSOCDecisionPolicy()
        )

        response_engine = (
            response_module.SmartSOCResponsePlanner()
        )

        authorization_engine = (
            authorization_module.SmartSOCAnalystAuthorization()
        )

        record_check(
            checks,
            True,
            "Decision components initialized",
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Risk / decision / response / authorization components loaded",
            str(exc),
        )

        return write_report(
            checks,
            event_results,
            failures,
            "FAILED",
        )

    # ------------------------------------------------------------------------
    # Frozen dimensions
    # ------------------------------------------------------------------------

    try:

        classes = list(
            artifacts[
                "label_encoder"
            ].classes_
        )

        dimensions_ok = (
            len(classes) == 45
        )

        dummy_text = behavior_text_at(
            behavior,
            0,
        )

        dummy_matrix, shapes = transform_event(
            structured.iloc[0],
            dummy_text,
            artifacts,
            categorical_features,
            numerical_features,
        )

        dimensions_ok = (
            dimensions_ok
            and
            shapes["categorical_shape"] == [1, 162]
            and
            shapes["numerical_shape"] == [1, 58]
            and
            shapes["tfidf_shape"] == [1, 6699]
            and
            shapes["final_shape"] == [1, 6919]
        )

        record_check(
            checks,
            dimensions_ok,
            "Frozen production artifact schema",
            f"classes={len(classes)}",
        )

        record_check(
            checks,
            dimensions_ok,
            "Frozen feature dimensions",
            (
                f"categorical={shapes['categorical_shape']}, "
                f"numerical={shapes['numerical_shape']}, "
                f"tfidf={shapes['tfidf_shape']}, "
                f"final={shapes['final_shape']}"
            ),
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Frozen production artifact schema",
            str(exc),
        )

        return write_report(
            checks,
            event_results,
            failures,
            "FAILED",
        )

    # ------------------------------------------------------------------------
    # Threshold resolution
    # ------------------------------------------------------------------------

    threshold_mapping = resolve_threshold_mapping(
        artifacts["thresholds"]
    )

    threshold_classes = set(
        threshold_mapping.keys()
    )

    model_classes = set(
        str(x)
        for x in artifacts[
            "label_encoder"
        ].classes_
    )

    threshold_overlap = (
        threshold_classes
        & model_classes
    )

    threshold_check = (
        len(threshold_overlap) > 0
    )

    record_check(
        checks,
        threshold_check,
        "Frozen threshold mapping resolved",
        (
            f"model_classes={len(model_classes)}, "
            f"thresholds_resolved={len(threshold_mapping)}, "
            f"overlap={len(threshold_overlap)}"
        ),
    )

    # ------------------------------------------------------------------------
    # Representative events
    # ------------------------------------------------------------------------

    count = min(
        20,
        len(structured),
    )

    indices = np.linspace(
        0,
        len(structured) - 1,
        num=count,
        dtype=int,
    )

    indices = list(
        dict.fromkeys(
            int(x)
            for x in indices
        )
    )

    # ------------------------------------------------------------------------
    # End-to-end simulation
    # ------------------------------------------------------------------------

    for number, index in enumerate(
        indices,
        start=1,
    ):

        row = structured.iloc[
            index
        ]

        behavior_text = behavior_text_at(
            behavior,
            index,
        )

        try:

            # ================================================================
            # ML
            # ================================================================

            matrix, feature_details = (
                transform_event(
                    row,
                    behavior_text,
                    artifacts,
                    categorical_features,
                    numerical_features,
                )
            )

            if matrix.shape != (
                1,
                6919,
            ):

                raise ValueError(
                    f"invalid feature shape: {matrix.shape}"
                )

            prediction = predict_event(
                matrix,
                artifacts,
                threshold_mapping,
            )

            if not (
                0.0
                <= prediction[
                    "top_probability"
                ]
                <= 1.0
            ):

                raise ValueError(
                    "invalid top probability"
                )

            if not (
                0.0
                <= prediction[
                    "margin"
                ]
                <= 1.0
            ):

                raise ValueError(
                    "invalid prediction margin"
                )

            # ================================================================
            # Evidence/context
            # ================================================================

            evidence_strength, evidence = (
                assess_evidence(
                    row,
                    behavior_text,
                )
            )

            missing_behavior = not bool(
                evidence[
                    "behavior_text_present"
                ]
            )

            context = derive_context(
                row
            )

            fallback = bool(
                prediction[
                    "fallback"
                ]
            )

            ambiguous = (
                prediction[
                    "margin"
                ]
                < 0.05
            )

            technique = prediction[
                "top_technique"
            ]

            confidence = prediction[
                "top_probability"
            ]

            severity = severity_for(
                technique
            )

            # ================================================================
            # Risk
            # ================================================================

            risk_decision = (
                risk_engine.calculate(
                    technique_severity=severity,
                    ml_confidence=confidence,
                    evidence_strength=evidence_strength,
                    privilege=context[
                        "privilege"
                    ],
                    network=context[
                        "network"
                    ],
                    suspicious_parent=context[
                        "suspicious_parent"
                    ],
                    fallback=fallback,
                    ambiguous=ambiguous,
                    missing_behavior=missing_behavior,
                )
            )

            # ================================================================
            # IMPORTANT FIX
            #
            # Use evaluate_event() rather than evaluate_risk_decision()
            # because the event-level policy API receives the uncertainty
            # state (fallback/ambiguous/missing behavior).
            #
            # This prevents the harness from losing fallback=True.
            # ================================================================

            decision_event = {
                "risk_score":
                    float(
                        risk_decision.risk_score
                    ),

                "risk_level":
                    str(
                        risk_decision.risk_level
                    ),

                "fallback":
                    fallback,

                "ambiguous":
                    ambiguous,

                "missing_behavior":
                    missing_behavior,

                "decision":
                    str(
                        risk_decision.recommended_action
                    ),
            }

            decision_result = (
                decision_engine.evaluate_event(
                    decision_event
                )
            )

            # ================================================================
            # Decision boundary
            # ================================================================

            expected_analyst = (
                str(
                    risk_decision.risk_level
                )
                != "LOW"
                or fallback
                or ambiguous
                or missing_behavior
            )

            if bool(
                decision_result.analyst_required
            ) != expected_analyst:

                raise ValueError(
                    "analyst boundary inconsistent with risk/fallback state"
                )

            # ================================================================
            # Response planner
            #
            # Use event-level evaluation so fallback/ambiguity/missing
            # behavior are preserved through the response layer.
            # ================================================================

            response_event = {
                "risk_score":
                    float(
                        risk_decision.risk_score
                    ),

                "risk_level":
                    str(
                        risk_decision.risk_level
                    ),

                "decision":
                    str(
                        decision_result.decision
                    ),

                "fallback":
                    fallback,

                "ambiguous":
                    ambiguous,

                "missing_behavior":
                    missing_behavior,
            }

            response_plan = (
                response_engine.evaluate_event(
                    response_event
                )
            )

            if bool(
                response_plan.containment_authorized
            ):

                raise ValueError(
                    "response planner authorized containment"
                )

            # ================================================================
            # Authorization
            # ================================================================

            authorization_result = (
                authorization_engine.evaluate(
                    risk_score=float(
                        risk_decision.risk_score
                    ),

                    risk_level=str(
                        risk_decision.risk_level
                    ),

                    decision=str(
                        decision_result.decision
                    ),

                    fallback=fallback,

                    ambiguous=ambiguous,

                    missing_behavior=missing_behavior,
                )
            )

            if bool(
                authorization_result.execution_permitted
            ):

                raise ValueError(
                    "execution permitted"
                )

            if bool(
                authorization_result.containment_authorized
            ):

                raise ValueError(
                    "containment authorized"
                )

            # ================================================================
            # Rollback
            # ================================================================

            rollback_ok = rollback_valid(
                response_plan
            )

            if not rollback_ok:

                raise ValueError(
                    "rollback path invalid"
                )

            # ================================================================
            # Per-event safety
            # ================================================================

            safety_ok = True

            safety_objects = [
                getattr(
                    risk_decision,
                    "safety",
                    {},
                ),

                getattr(
                    decision_result,
                    "safety",
                    {},
                ),

                getattr(
                    response_plan,
                    "safety",
                    {},
                ),

                getattr(
                    authorization_result,
                    "safety",
                    {},
                ),
            ]

            dangerous_keys = [
                "automatic_response",
                "automatic_blocking",
                "automatic_isolation",
                "automatic_process_kill",
                "automatic_account_disable",
                "automatic_firewall_change",
                "automatic_file_deletion",
                "automatic_containment",
                "containment_authorized",
                "execution_permitted",
                "response_actions_executed",
                "containment_actions_executed",
                "wazuh_integration",
                "shuffle_integration",
            ]

            for safety in safety_objects:

                if not isinstance(
                    safety,
                    dict,
                ):
                    continue

                for key in dangerous_keys:

                    if key in safety:

                        if bool(
                            safety[key]
                        ):

                            safety_ok = False

            if not safety_ok:

                raise ValueError(
                    "per-event operational safety violation"
                )

            # ================================================================
            # Expected label
            # ================================================================

            expected_label = ""

            if "techniques" in labels.columns:

                expected_label = str(
                    labels.iloc[
                        index
                    ][
                        "techniques"
                    ]
                )

            elif len(labels.columns) > 0:

                expected_label = str(
                    labels.iloc[
                        index
                    ].iloc[0]
                )

            # ================================================================
            # Record
            # ================================================================

            event_results.append(
                {
                    "simulation_number":
                        number,

                    "dataset_index":
                        index,

                    "event_id":
                        str(
                            row.get(
                                "event_id",
                                "",
                            )
                        ),

                    "expected_label":
                        expected_label,

                    "feature_details":
                        feature_details,

                    "prediction":
                        prediction,

                    "evidence":
                        evidence,

                    "evidence_strength":
                        evidence_strength,

                    "context":
                        context,

                    "risk":
                        json_safe(
                            risk_decision
                        ),

                    "decision":
                        json_safe(
                            decision_result
                        ),

                    "response_plan":
                        json_safe(
                            response_plan
                        ),

                    "authorization":
                        json_safe(
                            authorization_result
                        ),

                    "rollback_ok":
                        rollback_ok,

                    "safety_ok":
                        safety_ok,

                    "status":
                        "PASS",
                }
            )

        except Exception as exc:

            failures.append(
                {
                    "simulation_number":
                        number,

                    "dataset_index":
                        index,

                    "event_id":
                        str(
                            row.get(
                                "event_id",
                                "",
                            )
                        ),

                    "error":
                        str(exc),
                }
            )

            event_results.append(
                {
                    "simulation_number":
                        number,

                    "dataset_index":
                        index,

                    "event_id":
                        str(
                            row.get(
                                "event_id",
                                "",
                            )
                        ),

                    "status":
                        "FAIL",

                    "error":
                        str(exc),
                }
            )

    # ------------------------------------------------------------------------
    # Simulation gate
    # ------------------------------------------------------------------------

    simulations_ok = (
        len(failures) == 0
        and
        len(event_results) == len(indices)
    )

    record_check(
        checks,
        simulations_ok,
        "End-to-end event simulations",
        (
            f"{len(event_results)} events simulated"
            if simulations_ok
            else
            f"{len(event_results)} events, {len(failures)} failures"
        ),
    )

    # ------------------------------------------------------------------------
    # Analyst boundary
    # ------------------------------------------------------------------------

    analyst_ok = True

    for event in event_results:

        if event.get("status") != "PASS":
            analyst_ok = False
            continue

        prediction = event[
            "prediction"
        ]

        risk = event[
            "risk"
        ]

        decision = event[
            "decision"
        ]

        expected = (
            str(
                risk[
                    "risk_level"
                ]
            )
            != "LOW"
            or
            bool(
                prediction[
                    "fallback"
                ]
            )
            or
            float(
                prediction[
                    "margin"
                ]
            ) < 0.05
            or
            not bool(
                event[
                    "evidence"
                ][
                    "behavior_text_present"
                ]
            )
        )

        actual = bool(
            decision[
                "analyst_required"
            ]
        )

        if actual != expected:
            analyst_ok = False

    record_check(
        checks,
        analyst_ok,
        "Analyst boundary consistency",
    )

    # ------------------------------------------------------------------------
    # Containment boundary
    # ------------------------------------------------------------------------

    containment_ok = True

    for event in event_results:

        if event.get("status") != "PASS":
            containment_ok = False
            continue

        decision = event[
            "decision"
        ]

        response = event[
            "response_plan"
        ]

        authorization = event[
            "authorization"
        ]

        if bool(
            decision.get(
                "containment_authorized",
                False,
            )
        ):

            containment_ok = False

        if bool(
            response.get(
                "containment_authorized",
                False,
            )
        ):

            containment_ok = False

        if bool(
            authorization.get(
                "containment_authorized",
                False,
            )
        ):

            containment_ok = False

    record_check(
        checks,
        containment_ok,
        "Containment authorization boundary",
    )

    # ------------------------------------------------------------------------
    # Execution disabled
    # ------------------------------------------------------------------------

    execution_ok = True

    for event in event_results:

        if event.get("status") != "PASS":
            execution_ok = False
            continue

        authorization = event[
            "authorization"
        ]

        if bool(
            authorization.get(
                "execution_permitted",
                False,
            )
        ):

            execution_ok = False

    record_check(
        checks,
        execution_ok,
        "Execution remains disabled",
    )

    # ------------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------------

    rollback_ok = all(
        event.get(
            "rollback_ok",
            False,
        )
        for event in event_results
    )

    record_check(
        checks,
        rollback_ok,
        "Rollback paths valid",
    )

    # ------------------------------------------------------------------------
    # Per-event safety
    # ------------------------------------------------------------------------

    per_event_safety_ok = all(
        event.get(
            "safety_ok",
            False,
        )
        for event in event_results
    )

    record_check(
        checks,
        per_event_safety_ok,
        "Per-event operational safety",
    )

    # ------------------------------------------------------------------------
    # Deterministic inference
    # ------------------------------------------------------------------------

    try:

        index = indices[0]

        row = structured.iloc[
            index
        ]

        text = behavior_text_at(
            behavior,
            index,
        )

        matrix_a, _ = transform_event(
            row,
            text,
            artifacts,
            categorical_features,
            numerical_features,
        )

        matrix_b, _ = transform_event(
            row,
            text,
            artifacts,
            categorical_features,
            numerical_features,
        )

        prediction_a = predict_event(
            matrix_a,
            artifacts,
            threshold_mapping,
        )

        prediction_b = predict_event(
            matrix_b,
            artifacts,
            threshold_mapping,
        )

        feature_diff = float(
            np.max(
                np.abs(
                    matrix_a
                    -
                    matrix_b
                )
            )
        )

        probability_diff = float(
            np.max(
                np.abs(
                    np.asarray(
                        prediction_a[
                            "probabilities"
                        ]
                    )
                    -
                    np.asarray(
                        prediction_b[
                            "probabilities"
                        ]
                    )
                )
            )
        )

        deterministic_ok = (
            feature_diff == 0.0
            and
            probability_diff == 0.0
            and
            prediction_a[
                "top_technique"
            ]
            ==
            prediction_b[
                "top_technique"
            ]
        )

        record_check(
            checks,
            deterministic_ok,
            "Deterministic offline inference",
            (
                f"feature_max_diff={feature_diff}, "
                f"probability_max_diff={probability_diff}"
            ),
        )

    except Exception as exc:

        record_check(
            checks,
            False,
            "Deterministic offline inference",
            str(exc),
        )

    # ------------------------------------------------------------------------
    # Global safety
    # ------------------------------------------------------------------------

    global_safety_ok = True

    for component_name, component in [
        ("risk_engine", risk_engine),
        ("decision_policy", decision_engine),
        ("response_planner", response_engine),
        ("authorization", authorization_engine),
    ]:

        try:

            safety = {}

            if hasattr(
                component,
                "safety_status",
            ):

                safety = component.safety_status()

            elif hasattr(
                component,
                "get_safety_status",
            ):

                safety = component.get_safety_status()

            if isinstance(
                safety,
                dict,
            ):

                for key in [
                    "automatic_response",
                    "automatic_blocking",
                    "automatic_isolation",
                    "automatic_process_kill",
                    "automatic_account_disable",
                    "automatic_firewall_change",
                    "automatic_file_deletion",
                    "automatic_containment",
                    "containment_authorized",
                    "response_actions_executed",
                    "containment_actions_executed",
                    "wazuh_integration",
                    "shuffle_integration",
                ]:

                    if key in safety and bool(
                        safety[key]
                    ):

                        global_safety_ok = False

        except Exception:
            # Component safety_status absence does not itself create
            # an unsafe state. Per-event safety is checked separately.
            pass

    record_check(
        checks,
        global_safety_ok,
        "Global operational safety",
    )

    # ------------------------------------------------------------------------
    # Production modification boundary
    # ------------------------------------------------------------------------

    protected_paths = [
        PRODUCTION_DIR / "model.joblib",
        PRODUCTION_DIR / "thresholds.json",
        PRODUCTION_DIR / "feature_schema.json",
        FINAL_DATASET_DIR / "structured_test.csv",
        FINAL_DATASET_DIR / "labels_test.csv",
        FINAL_DATASET_DIR / "metadata_test.csv",
        FINAL_DATASET_DIR / "behavior_text_test.csv",
    ]

    before = {}

    for path in protected_paths:

        if path.exists():
            before[str(path)] = sha256_file(path)

    after = {}

    for path in protected_paths:

        if path.exists():
            after[str(path)] = sha256_file(path)

    modification_ok = (
        before == after
    )

    record_check(
        checks,
        modification_ok,
        "Frozen production/data modification boundary",
    )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    passed = sum(
        1
        for x in checks
        if x["passed"]
    )

    total = len(checks)

    failed = (
        total - passed
    )

    final_status = (
        "PASS"
        if failed == 0
        else
        "FAILED"
    )

    report = {
        "phase":
            PHASE,

        "version":
            VERSION,

        "status":
            final_status,

        "decision":
            (
                "PHASE_6_7_ACCEPTED"
                if final_status == "PASS"
                else
                "REVIEW_REQUIRED"
            ),

        "execution_mode":
            "OFFLINE_ONLY",

        "checks":
            checks,

        "check_summary": {
            "total":
                total,

            "passed":
                passed,

            "failed":
                failed,
        },

        "dataset": {
            "rows":
                len(structured),

            "representative_events":
                len(event_results),

            "structured_features":
                62,

            "categorical_features":
                4,

            "numerical_features":
                58,

            "final_features":
                6919,

            "techniques":
                45,
        },

        "threshold_policy": {
            "frozen":
                True,

            "modified":
                False,

            "resolved_threshold_count":
                len(threshold_mapping),

            "model_class_count":
                len(model_classes),

            "threshold_overlap_count":
                len(threshold_overlap),

            "missing_thresholds_are_errors":
                False,

            "missing_threshold_behavior":
                (
                    "No threshold is invented. "
                    "Classes without explicit frozen thresholds "
                    "are not threshold-detected."
                ),
        },

        "simulation": {
            "events":
                event_results,

            "failures":
                failures,
        },

        "safety": {
            "analyst_boundary":
                analyst_ok,

            "containment_boundary":
                containment_ok,

            "execution_disabled":
                execution_ok,

            "rollback_ok":
                rollback_ok,

            "per_event_safety":
                per_event_safety_ok,

            "global_safety":
                global_safety_ok,
        },

        "integration": {
            "wazuh":
                False,

            "shuffle":
                False,

            "response_execution":
                False,

            "containment_execution":
                False,
        },

        "production_modification_boundary": {
            "verified":
                modification_ok,

            "before_hashes":
                before,

            "after_hashes":
                after,
        },
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(report),
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PHASE 6.7 FINAL DECISION")
    print("=" * 72)
    print(
        f"Status : {final_status}"
    )
    print(
        f"Checks : {passed} passed / {failed} failed / {total} total"
    )
    print(
        f"Events : {len(event_results)}"
    )
    print(
        f"Thresholds resolved : {len(threshold_mapping)} / {len(model_classes)}"
    )
    print(
        f"Report : {REPORT_FILE}"
    )
    print()

    if final_status == "PASS":

        print(
            "PHASE 6.7 ACCEPTED — "
            "OFFLINE END-TO-END RESPONSE PIPELINE VALIDATED"
        )

        return 0

    print(
        "PHASE 6.7 FAILED — REVIEW REQUIRED"
    )

    return 1


# ============================================================================
# Report helper
# ============================================================================

def write_report(
    checks,
    event_results,
    failures,
    status,
):

    passed = sum(
        1
        for x in checks
        if x["passed"]
    )

    total = len(checks)

    report = {
        "phase":
            PHASE,

        "version":
            VERSION,

        "status":
            status,

        "decision":
            "REVIEW_REQUIRED",

        "execution_mode":
            "OFFLINE_ONLY",

        "checks":
            checks,

        "check_summary": {
            "total":
                total,

            "passed":
                passed,

            "failed":
                total - passed,
        },

        "simulation": {
            "events":
                event_results,

            "failures":
                failures,
        },

        "integration": {
            "wazuh":
                False,

            "shuffle":
                False,

            "response_execution":
                False,

            "containment_execution":
                False,
        },
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(report),
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("PHASE 6.7 FINAL DECISION")
    print("=" * 72)
    print(
        f"Status : {status}"
    )
    print(
        f"Checks : {passed} passed / {total - passed} failed / {total} total"
    )
    print(
        f"Report : {REPORT_FILE}"
    )

    return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )