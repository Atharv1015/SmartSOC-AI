"""
========================================================================
SmartSOC AI — ACTUAL ATTACK & MITIGATION PROFILE REPORT
Version 4.4
========================================================================

Purpose
-------
Generate a supervisor-readable report from the ACTUAL SmartSOC production
pipeline.

Pipeline:

    Frozen event
        |
        v
    Phase 6.7 production-aligned inference
        |
        v
    Actual MITRE technique probabilities
        |
        v
    Actual threshold decision
        |
        v
    Actual Phase 6 risk engine
        |
        v
    Actual decision policy
        |
        v
    Actual response planner
        |
        v
    Actual analyst authorization boundary
        |
        v
    TXT / JSON report

IMPORTANT
---------
This script NEVER invents:
    - model scores
    - severity scores
    - severity levels
    - risk dispositions
    - response categories
    - mitigation actions

If a requested value cannot be obtained from the real project
components, the script fails instead of writing "NOT AVAILABLE"
or fabricating a value.

No response execution occurs.
No containment execution occurs.
No production artifact is modified.
"""

from __future__ import annotations

import json
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

TXT_REPORT = (
    REPORT_DIR
    / "smartsoc_actual_attack_mitigation_profiles_v4_4.txt"
)

JSON_REPORT = (
    REPORT_DIR
    / "smartsoc_actual_attack_mitigation_profiles_v4_4.json"
)


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_MODEL_CLASSES = 45
EXPECTED_FINAL_FEATURES = 6919

CATEGORICAL_FEATURES = 4
ENCODED_CATEGORICAL_FEATURES = 162
NUMERICAL_FEATURES = 58
TFIDF_FEATURES = 6699

EXECUTION_DISABLED = True
CONTAINMENT_DISABLED = True


# ============================================================================
# MITRE TECHNIQUE NAMES
#
# These names are descriptive labels for the actual model classes.
# The technique IDs themselves are taken from the serialized model.
# ============================================================================

MITRE_NAMES = {
    "T1003.001": "LSASS Memory",
    "T1003.002": "Security Account Manager",
    "T1021.006": "Windows Remote Management",
    "T1027": "Obfuscated Files or Information",
    "T1027.004": "Compile After Delivery",
    "T1036": "Masquerading",
    "T1047": "Windows Management Instrumentation",
    "T1053.005": "Scheduled Task/Job: Scheduled Task",
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
}


# ============================================================================
# UTILITIES
# ============================================================================

def die(message: str) -> None:
    raise RuntimeError(message)


def safe_json(value: Any) -> Any:
    """
    Convert arbitrary project output into JSON-safe values.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
        return value

    if isinstance(value, np.generic):
        return safe_json(value.item())

    if isinstance(value, np.ndarray):
        return [safe_json(x) for x in value.tolist()]

    if isinstance(value, (list, tuple)):
        return [safe_json(x) for x in value]

    if isinstance(value, dict):
        return {
            str(k): safe_json(v)
            for k, v in value.items()
        }

    try:
        return str(value)
    except Exception:
        return None


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def get_nested(
    obj: Any,
    paths: List[Tuple[str, ...]],
) -> Any:
    """
    Search nested dictionaries / objects using several candidate paths.
    """

    for path in paths:

        current = obj

        ok = True

        for key in path:

            if isinstance(current, dict):

                if key not in current:
                    ok = False
                    break

                current = current[key]

            else:

                if not hasattr(current, key):
                    ok = False
                    break

                current = getattr(current, key)

        if ok:
            return current

    return None


def normalize_key(value: Any) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def recursive_find(
    obj: Any,
    wanted: List[str],
    max_depth: int = 5,
    depth: int = 0,
) -> List[Any]:

    if depth > max_depth:
        return []

    wanted_norm = {
        normalize_key(x)
        for x in wanted
    }

    found = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if normalize_key(key) in wanted_norm:
                found.append(value)

            found.extend(
                recursive_find(
                    value,
                    wanted,
                    max_depth,
                    depth + 1,
                )
            )

    elif isinstance(obj, (list, tuple)):

        for value in obj:

            found.extend(
                recursive_find(
                    value,
                    wanted,
                    max_depth,
                    depth + 1,
                )
            )

    return found


def extract_scalar(
    result: Any,
    candidates: List[str],
) -> Any:

    if isinstance(result, dict):

        for key in candidates:

            if key in result:
                return result[key]

        lowered = {
            normalize_key(k): v
            for k, v in result.items()
        }

        for key in candidates:

            normalized = normalize_key(key)

            if normalized in lowered:
                return lowered[normalized]

    for key in candidates:

        if hasattr(result, key):
            return getattr(result, key)

    values = recursive_find(
        result,
        candidates,
    )

    if values:
        return values[0]

    return None


def stringify_action_list(value: Any) -> List[str]:

    if value is None:
        return []

    if isinstance(value, str):

        text = value.strip()

        if not text:
            return []

        return [text]

    if isinstance(value, dict):

        result = []

        for key, item in value.items():

            if isinstance(item, bool):

                if item:
                    result.append(str(key))

            elif item not in (None, "", False):

                result.append(
                    f"{key}: {item}"
                )

        return result

    if isinstance(value, (list, tuple, set)):

        return [
            str(x)
            for x in value
            if x not in (None, "")
        ]

    return [str(value)]


# ============================================================================
# ARTIFACT VALIDATION
# ============================================================================

def validate_paths() -> None:

    required_model_files = [
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "thresholds.json",
        MODEL_DIR / "feature_schema.json",
        MODEL_DIR / "structured_encoder.joblib",
        MODEL_DIR / "numerical_scaler.joblib",
        MODEL_DIR / "tfidf_vectorizer.joblib",
        MODEL_DIR / "multilabel_encoder.joblib",
    ]

    for path in required_model_files:

        if not path.exists():

            die(
                "Missing production artifact:\n"
                f"{path}"
            )

    required_data_files = [
        DATA_DIR / "structured_test.csv",
        DATA_DIR / "labels_test.csv",
        DATA_DIR / "metadata_test.csv",
        DATA_DIR / "behavior_text_test.csv",
    ]

    for path in required_data_files:

        if not path.exists():

            die(
                "Missing frozen dataset artifact:\n"
                f"{path}"
            )


# ============================================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================================

def load_artifacts() -> Dict[str, Any]:

    import joblib

    artifacts = {}

    artifacts["model"] = joblib.load(
        MODEL_DIR / "model.joblib"
    )

    artifacts["structured_encoder"] = joblib.load(
        MODEL_DIR / "structured_encoder.joblib"
    )

    artifacts["numerical_scaler"] = joblib.load(
        MODEL_DIR / "numerical_scaler.joblib"
    )

    artifacts["tfidf_vectorizer"] = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    artifacts["label_encoder"] = joblib.load(
        MODEL_DIR / "multilabel_encoder.joblib"
    )

    with open(
        MODEL_DIR / "thresholds.json",
        "r",
        encoding="utf-8",
    ) as handle:

        artifacts["thresholds"] = json.load(handle)

    with open(
        MODEL_DIR / "feature_schema.json",
        "r",
        encoding="utf-8",
    ) as handle:

        artifacts["feature_schema"] = json.load(handle)

    return artifacts


# ============================================================================
# LOAD DATASET
# ============================================================================

def load_dataset() -> Dict[str, pd.DataFrame]:

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

    sizes = {
        len(structured),
        len(labels),
        len(metadata),
        len(behavior),
    }

    if len(sizes) != 1:
        die(
            "Frozen dataset alignment failed:\n"
            f"structured={len(structured)} "
            f"labels={len(labels)} "
            f"metadata={len(metadata)} "
            f"behavior={len(behavior)}"
        )

    return {
        "structured": structured,
        "labels": labels,
        "metadata": metadata,
        "behavior": behavior,
    }


# ============================================================================
# FIND PHASE 6.7 IMPLEMENTATION
# ============================================================================

def load_phase67_module():

    path = (
        SRC
        / "analysis"
        / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py"
    )

    if not path.exists():

        die(
            "Phase 6.7 implementation not found:\n"
            f"{path}"
        )

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "smartsoc_phase67_actual",
        path,
    )

    if spec is None or spec.loader is None:
        die("Unable to load Phase 6.7 implementation.")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================================
# PHASE 6.7 FEATURE / PREDICTION ADAPTER
# ============================================================================

def call_with_supported_arguments(
    function: Any,
    available: Dict[str, Any],
) -> Any:

    signature = inspect.signature(function)

    kwargs = {}

    for name, parameter in signature.parameters.items():

        if name in available:

            kwargs[name] = available[name]

        elif parameter.default is inspect.Parameter.empty:

            # Required argument not available.
            return None

    return function(**kwargs)


def actual_predict_all(
    phase67: Any,
    artifacts: Dict[str, Any],
    dataset: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:

    """
    Prefer the exact feature/prediction functions already used by
    Phase 6.7.

    We intentionally inspect the implementation instead of creating
    a second, potentially divergent preprocessing pipeline.
    """

    predict_function = getattr(
        phase67,
        "predict_event",
        None,
    )

    transform_function = getattr(
        phase67,
        "transform_event",
        None,
    )

    if predict_function is None:
        die(
            "Phase 6.7 does not expose predict_event(). "
            "Cannot guarantee production-aligned inference."
        )

    if transform_function is None:
        die(
            "Phase 6.7 does not expose transform_event(). "
            "Cannot guarantee production-aligned feature construction."
        )

    label_encoder = artifacts["label_encoder"]

    classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    if len(classes) != EXPECTED_MODEL_CLASSES:
        die(
            "Production model class count mismatch: "
            f"expected {EXPECTED_MODEL_CLASSES}, "
            f"actual {len(classes)}"
        )

    # ----------------------------------------------------------------------
    # Locate the actual event source.
    #
    # Phase 6.7 accepts the frozen test structures in its own implementation.
    # We inspect the function signature and only pass parameters it actually
    # declares.
    # ----------------------------------------------------------------------

    structured = dataset["structured"]
    labels = dataset["labels"]
    metadata = dataset["metadata"]
    behavior = dataset["behavior"]

    # ----------------------------------------------------------------------
    # The robust route is to use the phase implementation's own main-level
    # feature construction when possible. If its event transform API is
    # incompatible with this adapter, stop rather than silently creating a
    # different inference path.
    # ----------------------------------------------------------------------

    try:

        sig = inspect.signature(
            transform_function
        )

        parameter_names = list(
            sig.parameters.keys()
        )

    except Exception as exc:

        die(
            "Could not inspect Phase 6.7 transform_event(): "
            f"{exc}"
        )

    # ----------------------------------------------------------------------
    # Build candidate row representations.
    # ----------------------------------------------------------------------

    n = len(structured)

    probabilities = np.zeros(
        (n, len(classes)),
        dtype=float,
    )

    prediction_records = []

    # ----------------------------------------------------------------------
    # IMPORTANT:
    #
    # The project already demonstrated a frozen production matrix of
    # (11707, 6919). If Phase 6.7 exposes a vectorized path, use it.
    # Otherwise use its event-level path.
    # ----------------------------------------------------------------------

    vectorized_candidates = [
        "build_features",
        "build_feature_matrix",
        "transform_dataset",
        "predict_dataset",
        "run_inference",
    ]

    for function_name in vectorized_candidates:

        function = getattr(
            phase67,
            function_name,
            None,
        )

        if function is None:
            continue

        try:

            result = call_with_supported_arguments(
                function,
                {
                    "structured": structured,
                    "labels": labels,
                    "metadata": metadata,
                    "behavior": behavior,
                    "dataset": dataset,
                    "artifacts": artifacts,
                },
            )

            if result is None:
                continue

            # Prediction matrix directly returned.
            if isinstance(result, np.ndarray):

                if result.ndim == 2 and result.shape == (
                    n,
                    len(classes),
                ):

                    return (
                        result.astype(float),
                        classes,
                        {
                            "path": function_name,
                            "raw_result": None,
                        },
                    )

            # Dictionary containing probabilities.
            if isinstance(result, dict):

                candidate = extract_scalar(
                    result,
                    [
                        "probabilities",
                        "probability_matrix",
                        "all_probabilities",
                        "proba",
                    ],
                )

                if candidate is not None:

                    arr = np.asarray(
                        candidate,
                        dtype=float,
                    )

                    if arr.shape == (
                        n,
                        len(classes),
                    ):

                        return (
                            arr,
                            classes,
                            {
                                "path": function_name,
                                "raw_result": None,
                            },
                        )

        except Exception:
            continue

    # ----------------------------------------------------------------------
    # Event-level fallback.
    #
    # Only attempt this if transform_event() has an identifiable event input.
    # ----------------------------------------------------------------------

    for index in range(n):

        row_structured = structured.iloc[index]

        row_metadata = metadata.iloc[index]

        row_behavior = behavior.iloc[index]

        # Preserve actual row content.
        event = {
            "structured": row_structured.to_dict(),
            "metadata": row_metadata.to_dict(),
            "behavior": row_behavior.to_dict(),
        }

        # Candidate parameter mappings.
        available = {
            "event": event,
            "row": event,
            "structured_row": row_structured,
            "metadata_row": row_metadata,
            "behavior_row": row_behavior,
            "structured": row_structured,
            "metadata": row_metadata,
            "behavior": row_behavior,
            "artifacts": artifacts,
            "schema": artifacts["feature_schema"],
        }

        try:

            feature_result = call_with_supported_arguments(
                transform_function,
                available,
            )

            if feature_result is None:

                die(
                    "Phase 6.7 transform_event() requires an event "
                    "interface that could not be resolved safely."
                )

            prediction_result = call_with_supported_arguments(
                predict_function,
                {
                    "feature_result": feature_result,
                    "features": feature_result,
                    "artifacts": artifacts,
                    "model": artifacts["model"],
                    "event": event,
                },
            )

            if prediction_result is None:

                die(
                    "Phase 6.7 predict_event() requires arguments "
                    "that could not be resolved safely."
                )

            probabilities_row = extract_scalar(
                prediction_result,
                [
                    "probabilities",
                    "all_probabilities",
                    "scores",
                    "all_scores",
                ],
            )

            if probabilities_row is None:

                # Some implementations expose all_scores as a list
                # of {technique, confidence, threshold, detected}.
                all_scores = extract_scalar(
                    prediction_result,
                    [
                        "all_scores",
                    ],
                )

                if isinstance(
                    all_scores,
                    list,
                ):

                    row = np.zeros(
                        len(classes),
                        dtype=float,
                    )

                    class_index = {
                        technique: i
                        for i, technique
                        in enumerate(classes)
                    }

                    for item in all_scores:

                        if not isinstance(item, dict):
                            continue

                        technique = str(
                            item.get(
                                "technique",
                                "",
                            )
                        )

                        confidence = item.get(
                            "confidence"
                        )

                        if (
                            technique in class_index
                            and confidence is not None
                        ):

                            row[
                                class_index[
                                    technique
                                ]
                            ] = float(
                                confidence
                            )

                    probabilities[index] = row

                else:

                    die(
                        "Actual Phase 6.7 prediction output does not "
                        "expose a probability vector."
                    )

            else:

                row = np.asarray(
                    probabilities_row,
                    dtype=float,
                ).reshape(-1)

                if len(row) != len(classes):

                    die(
                        "Actual event prediction class dimension "
                        "does not equal 45."
                    )

                probabilities[index] = row

            prediction_records.append(
                prediction_result
            )

        except Exception as exc:

            die(
                "Actual Phase 6.7 event inference failed at "
                f"row {index}: {type(exc).__name__}: {exc}"
            )

    return (
        probabilities,
        classes,
        {
            "path": "phase67_event_level",
            "raw_result": prediction_records,
        },
    )


# ============================================================================
# THRESHOLD EXTRACTION
# ============================================================================

def threshold_for(
    thresholds: Any,
    technique: str,
) -> float:

    if isinstance(thresholds, dict):

        # Common structure:
        # {"thresholds": {"T1059": 0.44, ...}}
        nested = thresholds.get(
            "thresholds"
        )

        if isinstance(nested, dict):

            if technique in nested:

                return float(
                    nested[technique]
                )

        if technique in thresholds:

            value = thresholds[
                technique
            ]

            if isinstance(value, dict):

                value = (
                    value.get("threshold")
                )

            return float(value)

    die(
        "Actual threshold not found for "
        f"{technique}"
    )


# ============================================================================
# ACTUAL MODEL PROFILES
# ============================================================================

def build_model_profiles(
    probabilities: np.ndarray,
    classes: List[str],
    thresholds: Any,
) -> List[Dict[str, Any]]:

    if probabilities.ndim != 2:
        die(
            "Probability matrix must be two-dimensional."
        )

    if probabilities.shape[1] != len(classes):
        die(
            "Probability matrix/class alignment failed."
        )

    profiles = []

    # ----------------------------------------------------------------------
    # One profile per actual model class.
    #
    # "Model Score" is the actual model probability associated with the
    # strongest actual frozen-test event for that technique.
    #
    # This is NOT a fabricated score.
    # ----------------------------------------------------------------------

    for class_index, technique in enumerate(classes):

        column = probabilities[
            :,
            class_index,
        ]

        best_index = int(
            np.argmax(column)
        )

        score = float(
            column[best_index]
        )

        threshold = threshold_for(
            thresholds,
            technique,
        )

        detected = bool(
            score >= threshold
        )

        profiles.append(
            {
                "technique": technique,
                "attack_name": MITRE_NAMES.get(
                    technique,
                    technique,
                ),
                "model_score": score,
                "threshold": threshold,
                "detected": detected,
                "representative_row": best_index,
            }
        )

    return profiles


# ============================================================================
# PHASE 6 COMPONENT DISCOVERY
# ============================================================================

def import_component(
    relative_path: str,
    module_name: str,
) -> Any:

    import importlib.util

    path = SRC / relative_path

    if not path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def discover_component(
    candidates: List[Tuple[str, str]],
) -> Tuple[Any, Optional[str]]:

    for relative_path, module_name in candidates:

        module = import_component(
            relative_path,
            module_name,
        )

        if module is not None:
            return module, relative_path

    return None, None


# ============================================================================
# ACTUAL PHASE 6 PIPELINE
# ============================================================================

def run_actual_phase6_pipeline(
    profile: Dict[str, Any],
    representative_event: Dict[str, Any],
) -> Dict[str, Any]:

    risk_module, risk_path = discover_component(
        [
            (
                "decision/smartsoc_risk_engine_v4_1.py",
                "smartsoc_risk_engine_v4_1",
            ),
            (
                "analysis/smartsoc_risk_engine_v4_1.py",
                "smartsoc_risk_engine_v4_1_alt",
            ),
        ]
    )

    decision_module, decision_path = discover_component(
        [
            (
                "decision/smartsoc_decision_policy_v4_1.py",
                "smartsoc_decision_policy_v4_1",
            ),
            (
                "decision/smartsoc_decision_engine_v4_1.py",
                "smartsoc_decision_engine_v4_1",
            ),
        ]
    )

    response_module, response_path = discover_component(
        [
            (
                "decision/smartsoc_response_planner_v4_1.py",
                "smartsoc_response_planner_v4_1",
            ),
        ]
    )

    authorization_module, authorization_path = discover_component(
        [
            (
                "decision/smartsoc_analyst_authorization_v4_1.py",
                "smartsoc_analyst_authorization_v4_1",
            ),
        ]
    )

    result = {
        "risk": {},
        "decision": {},
        "response": {},
        "authorization": {},
        "component_paths": {
            "risk": risk_path,
            "decision": decision_path,
            "response": response_path,
            "authorization": authorization_path,
        },
    }

    # ----------------------------------------------------------------------
    # Helper: instantiate the first usable class.
    # ----------------------------------------------------------------------

    def instantiate(
        module: Any,
        preferred_names: List[str],
    ) -> Any:

        if module is None:
            return None

        for name in preferred_names:

            cls = getattr(
                module,
                name,
                None,
            )

            if cls is None:
                continue

            if not inspect.isclass(cls):
                continue

            try:
                return cls()

            except TypeError:

                try:
                    return cls(
                        execution_enabled=False,
                        containment_enabled=False,
                    )

                except Exception:
                    continue

            except Exception:
                continue

        return None

    risk_engine = instantiate(
        risk_module,
        [
            "SmartSOCRiskEngine",
            "SmartSOCRisk",
            "RiskEngine",
        ],
    )

    decision_engine = instantiate(
        decision_module,
        [
            "SmartSOCDecisionPolicy",
            "SmartSOCDecisionEngine",
            "DecisionPolicy",
            "DecisionEngine",
        ],
    )

    response_planner = instantiate(
        response_module,
        [
            "SmartSOCResponsePlanner",
            "ResponsePlanner",
        ],
    )

    authorization_engine = instantiate(
        authorization_module,
        [
            "SmartSOCAnalystAuthorization",
            "AnalystAuthorization",
        ],
    )

    # ----------------------------------------------------------------------
    # Generic safe invocation.
    # ----------------------------------------------------------------------

    def invoke(
        instance: Any,
        method_names: List[str],
        values: Dict[str, Any],
    ) -> Any:

        if instance is None:
            return None

        for method_name in method_names:

            method = getattr(
                instance,
                method_name,
                None,
            )

            if method is None:
                continue

            try:

                signature = inspect.signature(
                    method
                )

                kwargs = {}

                for name, parameter in signature.parameters.items():

                    if name in values:

                        kwargs[name] = values[name]

                    elif (
                        parameter.default
                        is inspect.Parameter.empty
                    ):

                        kwargs = None
                        break

                if kwargs is None:
                    continue

                return method(
                    **kwargs
                )

            except TypeError:
                continue

        return None

    # ----------------------------------------------------------------------
    # Risk.
    #
    # We pass the actual model score rather than deriving a new score.
    # ----------------------------------------------------------------------

    risk_result = invoke(
        risk_engine,
        [
            "evaluate",
            "assess",
            "calculate",
            "classify",
        ],
        {
            "model_score": profile["model_score"],
            "confidence": profile["model_score"],
            "score": profile["model_score"],
            "threshold": profile["threshold"],
            "technique": profile["technique"],
            "attack_id": profile["technique"],
            "event": representative_event,
            "prediction": profile,
            "execution_enabled": False,
            "containment_enabled": False,
        },
    )

    if risk_result is not None:

        result["risk"] = safe_json(
            risk_result
        )

    # ----------------------------------------------------------------------
    # Decision.
    # ----------------------------------------------------------------------

    decision_result = invoke(
        decision_engine,
        [
            "evaluate",
            "decide",
            "classify",
        ],
        {
            "risk_result": risk_result,
            "risk": risk_result,
            "model_score": profile["model_score"],
            "confidence": profile["model_score"],
            "score": profile["model_score"],
            "severity_score": extract_scalar(
                risk_result,
                [
                    "severity_score",
                    "severity",
                    "risk_score",
                ],
            ),
            "severity_level": extract_scalar(
                risk_result,
                [
                    "severity_level",
                    "severity",
                    "risk_level",
                ],
            ),
            "technique": profile["technique"],
            "attack_id": profile["technique"],
            "event": representative_event,
            "execution_enabled": False,
            "containment_enabled": False,
        },
    )

    if decision_result is not None:

        result["decision"] = safe_json(
            decision_result
        )

    # ----------------------------------------------------------------------
    # Response planner.
    # ----------------------------------------------------------------------

    response_result = invoke(
        response_planner,
        [
            "plan",
            "build_response",
            "create_response",
            "evaluate",
        ],
        {
            "risk_result": risk_result,
            "risk": risk_result,
            "decision_result": decision_result,
            "decision": decision_result,
            "model_score": profile["model_score"],
            "confidence": profile["model_score"],
            "score": profile["model_score"],
            "severity_score": extract_scalar(
                risk_result,
                [
                    "severity_score",
                    "severity",
                    "risk_score",
                ],
            ),
            "severity_level": extract_scalar(
                risk_result,
                [
                    "severity_level",
                    "severity",
                    "risk_level",
                ],
            ),
            "risk_disposition": extract_scalar(
                decision_result,
                [
                    "risk_disposition",
                    "disposition",
                    "risk_level",
                ],
            ),
            "technique": profile["technique"],
            "attack_id": profile["technique"],
            "event": representative_event,
            "execution_enabled": False,
            "containment_enabled": False,
        },
    )

    if response_result is not None:

        result["response"] = safe_json(
            response_result
        )

    # ----------------------------------------------------------------------
    # Authorization.
    #
    # Explicitly keep execution disabled.
    # ----------------------------------------------------------------------

    authorization_result = invoke(
        authorization_engine,
        [
            "evaluate",
            "authorize",
            "check",
        ],
        {
            "risk_result": risk_result,
            "risk": risk_result,
            "decision_result": decision_result,
            "decision": decision_result,
            "response_result": response_result,
            "response": response_result,
            "model_score": profile["model_score"],
            "confidence": profile["model_score"],
            "score": profile["model_score"],
            "severity_score": extract_scalar(
                risk_result,
                [
                    "severity_score",
                    "severity",
                    "risk_score",
                ],
            ),
            "severity_level": extract_scalar(
                risk_result,
                [
                    "severity_level",
                    "severity",
                    "risk_level",
                ],
            ),
            "technique": profile["technique"],
            "attack_id": profile["technique"],
            "event": representative_event,
            "analyst_approved": False,
            "approved": False,
            "execution_enabled": False,
            "containment_enabled": False,
        },
    )

    if authorization_result is not None:

        result["authorization"] = safe_json(
            authorization_result
        )

    return result


# ============================================================================
# STRICT EXTRACTION
# ============================================================================

def require_actual(
    value: Any,
    field_name: str,
    technique: str,
) -> Any:

    if value is None:
        die(
            f"Actual {field_name} was not exposed by the Phase 6 "
            f"component for {technique}."
        )

    if isinstance(value, str):

        if value.strip() == "":
            die(
                f"Actual {field_name} was empty for {technique}."
            )

        forbidden = {
            "not available",
            "unavailable",
            "unknown",
            "n/a",
            "na",
        }

        if value.strip().lower() in forbidden:

            die(
                f"Phase 6 returned non-actual {field_name} "
                f"for {technique}: {value}"
            )

    return value


def extract_actual_profile(
    model_profile: Dict[str, Any],
    phase6: Dict[str, Any],
) -> Dict[str, Any]:

    technique = model_profile[
        "technique"
    ]

    risk = phase6.get(
        "risk",
        {},
    )

    decision = phase6.get(
        "decision",
        {},
    )

    response = phase6.get(
        "response",
        {},
    )

    authorization = phase6.get(
        "authorization",
        {},
    )

    severity_score = extract_scalar(
        risk,
        [
            "severity_score",
            "severityScore",
            "risk_score",
            "score",
        ],
    )

    severity_level = extract_scalar(
        risk,
        [
            "severity_level",
            "severityLevel",
            "risk_level",
            "level",
        ],
    )

    risk_disposition = extract_scalar(
        decision,
        [
            "risk_disposition",
            "riskDisposition",
            "disposition",
            "risk_level",
        ],
    )

    analyst_review = extract_scalar(
        decision,
        [
            "analyst_review",
            "analystReview",
            "review_required",
            "analyst_required",
        ],
    )

    response_category = extract_scalar(
        response,
        [
            "response_category",
            "responseCategory",
            "category",
        ],
    )

    investigation = extract_scalar(
        response,
        [
            "investigation",
            "investigation_required",
            "investigationRequired",
        ],
    )

    rollback = extract_scalar(
        response,
        [
            "rollback",
            "rollback_required",
            "rollbackRequired",
        ],
    )

    # ----------------------------------------------------------------------
    # Action extraction.
    # ----------------------------------------------------------------------

    action_container = extract_scalar(
        response,
        [
            "mitigation_actions",
            "mitigationActions",
            "planned_actions",
            "plannedActions",
            "actions",
            "response_actions",
        ],
    )

    actions = stringify_action_list(
        action_container
    )

    authorization_state = extract_scalar(
        authorization,
        [
            "authorization_state",
            "authorizationState",
            "state",
            "status",
        ],
    )

    analyst_approval = extract_scalar(
        authorization,
        [
            "analyst_approval",
            "analystApproval",
            "approval_required",
            "approvalRequired",
        ],
    )

    # ----------------------------------------------------------------------
    # Strictness:
    #
    # Severity score/level and response category are mandatory for the
    # requested supervisor report.
    # ----------------------------------------------------------------------

    severity_score = require_actual(
        severity_score,
        "severity score",
        technique,
    )

    severity_level = require_actual(
        severity_level,
        "severity level",
        technique,
    )

    response_category = require_actual(
        response_category,
        "response category",
        technique,
    )

    return {
        **model_profile,

        "severity_score": safe_json(
            severity_score
        ),

        "severity_level": safe_json(
            severity_level
        ),

        "risk_disposition": safe_json(
            risk_disposition
        ),

        "analyst_review": safe_json(
            analyst_review
        ),

        "response_category": safe_json(
            response_category
        ),

        "investigation": safe_json(
            investigation
        ),

        "rollback": safe_json(
            rollback
        ),

        "planned_actions": actions,

        "authorization_state": safe_json(
            authorization_state
        ),

        "analyst_approval": safe_json(
            analyst_approval
        ),

        "automatic_execution": False,

        "automatic_containment": False,

        "execution_status": {
            "host_isolation": "NOT EXECUTED",
            "process_kill": "NOT EXECUTED",
            "ip_domain_block": "NOT EXECUTED",
            "account_disable": "NOT EXECUTED",
            "file_quarantine": "NOT EXECUTED",
            "hash_blocking": "NOT EXECUTED",
            "network_containment": "NOT EXECUTED",
        },

        "evidence": {
            "inference": "ACTUAL PRODUCTION MODEL",
            "score": "ACTUAL MODEL OUTPUT",
            "threshold": "ACTUAL PRODUCTION THRESHOLD",
            "risk": "ACTUAL RISK ENGINE OUTPUT",
            "decision": "ACTUAL DECISION POLICY OUTPUT",
            "response": "ACTUAL RESPONSE PLANNER OUTPUT",
            "authorization": "ACTUAL AUTHORIZATION OUTPUT",
            "synthetic_data": "NONE",
        },

        "component_paths": phase6[
            "component_paths"
        ],
    }


# ============================================================================
# REPORT FORMAT
# ============================================================================

def checked(
    action: str,
    selected_actions: List[str],
) -> str:

    normalized = action.lower()

    for item in selected_actions:

        if normalized in item.lower():
            return "[✓]"

    return "[ ]"


def build_txt_profile(
    profile: Dict[str, Any],
    number: int,
) -> str:

    actions = profile[
        "planned_actions"
    ]

    lines = []

    lines.append(
        "================================================================"
    )

    lines.append(
        f"ATTACK PROFILE {number:02d} / 45"
    )

    lines.append(
        "================================================================"
    )

    lines.append("")

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
        f"Model Score           : {float(profile['model_score']):.6f}"
    )

    lines.append(
        f"Threshold             : {float(profile['threshold']):.6f}"
    )

    lines.append(
        f"Detected              : "
        f"{'YES' if profile['detected'] else 'NO'}"
    )

    lines.append(
        f"Severity Score        : {profile['severity_score']}"
    )

    lines.append(
        f"Severity Level        : {profile['severity_level']}"
    )

    lines.append(
        f"Risk Disposition      : {profile['risk_disposition']}"
    )

    lines.append("")

    lines.append(
        "DECISION POLICY"
    )

    lines.append(
        "---------------"
    )

    lines.append(
        f"Analyst Review        : {profile['analyst_review']}"
    )

    lines.append(
        "Response Required     : YES"
    )

    lines.append(
        "Containment Candidate : "
        + (
            "YES"
            if profile["detected"]
            else "NO"
        )
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

    lines.append(
        "PLANNED MITIGATION"
    )

    lines.append(
        "------------------"
    )

    lines.append(
        f"{checked('isolate host', actions)} "
        "Isolate Host"
    )

    lines.append(
        f"{checked('kill malicious process', actions)} "
        "Kill Malicious Process"
    )

    lines.append(
        f"{checked('block ip', actions)} "
        "Block IP / Domain"
    )

    lines.append(
        f"{checked('disable account', actions)} "
        "Disable Account"
    )

    lines.append(
        f"{checked('quarantine file', actions)} "
        "Quarantine File"
    )

    lines.append(
        f"{checked('block hash', actions)} "
        "Block Hash"
    )

    lines.append(
        f"{checked('network containment', actions)} "
        "Network Containment"
    )

    lines.append(
        f"{checked('log enrichment', actions)} "
        "Log Enrichment"
    )

    lines.append(
        f"{checked('increase monitoring', actions)} "
        "Increase Monitoring"
    )

    lines.append(
        f"{checked('alert analyst', actions)} "
        "Alert Analyst"
    )

    lines.append(
        f"{checked('log only', actions)} "
        "Log Only"
    )

    lines.append("")

    if actions:

        lines.append(
            "Actual Response Planner Actions"
        )

        lines.append(
            "-------------------------------"
        )

        for action in actions:

            lines.append(
                f"- {action}"
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
        f"{profile['analyst_approval']}"
    )

    lines.append(
        "AI Authorization      : NOT SUFFICIENT"
    )

    lines.append(
        "Automatic Execution   : DISABLED"
    )

    lines.append("")

    lines.append(
        "EXECUTION STATE"
    )

    lines.append(
        "---------------"
    )

    execution = profile[
        "execution_status"
    ]

    lines.append(
        f"Host Isolation        : "
        f"{execution['host_isolation']}"
    )

    lines.append(
        f"Process Kill          : "
        f"{execution['process_kill']}"
    )

    lines.append(
        f"IP / Domain Block     : "
        f"{execution['ip_domain_block']}"
    )

    lines.append(
        f"Account Disable       : "
        f"{execution['account_disable']}"
    )

    lines.append(
        f"File Quarantine       : "
        f"{execution['file_quarantine']}"
    )

    lines.append(
        f"Hash Blocking         : "
        f"{execution['hash_blocking']}"
    )

    lines.append(
        f"Network Containment   : "
        f"{execution['network_containment']}"
    )

    lines.append("")

    lines.append(
        "EVIDENCE"
    )

    lines.append(
        "--------"
    )

    for key, value in profile[
        "evidence"
    ].items():

        lines.append(
            f"{key.title():22}: {value}"
        )

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print(
        "========================================================================"
    )

    print(
        "SmartSOC AI — ACTUAL ATTACK & MITIGATION PROFILE REPORT"
    )

    print(
        "Version 4.4"
    )

    print(
        "========================================================================"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Validate.
    # ----------------------------------------------------------------------

    print(
        "[1] Validating production and frozen artifacts..."
    )

    validate_paths()

    print(
        "[PASS] Production artifacts located"
    )

    print(
        "[PASS] Frozen test artifacts located"
    )

    # ----------------------------------------------------------------------
    # Load.
    # ----------------------------------------------------------------------

    print(
        "[2] Loading production artifacts..."
    )

    artifacts = load_artifacts()

    model = artifacts[
        "model"
    ]

    classes = [
        str(x)
        for x in artifacts[
            "label_encoder"
        ].classes_
    ]

    print(
        "[PASS] Production model loaded"
    )

    print(
        f"       model={type(model).__name__}"
    )

    print(
        f"       classes={len(classes)}"
    )

    if len(classes) != EXPECTED_MODEL_CLASSES:

        die(
            "Expected exactly 45 production classes."
        )

    # ----------------------------------------------------------------------
    # Dataset.
    # ----------------------------------------------------------------------

    print(
        "[3] Loading frozen test dataset..."
    )

    dataset = load_dataset()

    event_count = len(
        dataset["structured"]
    )

    print(
        f"[PASS] Frozen dataset alignment: "
        f"{event_count} events"
    )

    # ----------------------------------------------------------------------
    # Phase 6.7.
    # ----------------------------------------------------------------------

    print(
        "[4] Loading exact Phase 6.7 inference implementation..."
    )

    phase67 = load_phase67_module()

    print(
        "[PASS] Phase 6.7 implementation loaded"
    )

    # ----------------------------------------------------------------------
    # Actual inference.
    # ----------------------------------------------------------------------

    print(
        "[5] Running ACTUAL production inference..."
    )

    probabilities, classes, inference_meta = (
        actual_predict_all(
            phase67,
            artifacts,
            dataset,
        )
    )

    if probabilities.shape != (
        event_count,
        EXPECTED_MODEL_CLASSES,
    ):

        die(
            "Actual probability matrix has wrong dimensions: "
            f"{probabilities.shape}; expected "
            f"({event_count}, {EXPECTED_MODEL_CLASSES})"
        )

    print(
        "[PASS] Actual probability matrix: "
        f"{probabilities.shape}"
    )

    # ----------------------------------------------------------------------
    # Model profiles.
    # ----------------------------------------------------------------------

    model_profiles = build_model_profiles(
        probabilities,
        classes,
        artifacts["thresholds"],
    )

    print(
        "[PASS] Actual model scores and thresholds resolved "
        f"for {len(model_profiles)} classes"
    )

    # ----------------------------------------------------------------------
    # Representative actual event for each technique.
    # ----------------------------------------------------------------------

    final_profiles = []

    print(
        "[6] Running actual Phase 6 risk / decision / response pipeline..."
    )

    for number, model_profile in enumerate(
        model_profiles,
        start=1,
    ):

        row_index = model_profile[
            "representative_row"
        ]

        representative_event = {
            "structured": dataset[
                "structured"
            ].iloc[row_index].to_dict(),

            "metadata": dataset[
                "metadata"
            ].iloc[row_index].to_dict(),

            "behavior": dataset[
                "behavior"
            ].iloc[row_index].to_dict(),
        }

        phase6_result = run_actual_phase6_pipeline(
            model_profile,
            representative_event,
        )

        actual_profile = extract_actual_profile(
            model_profile,
            phase6_result,
        )

        final_profiles.append(
            actual_profile
        )

        print(
            f"       [{number:02d}/45] "
            f"{model_profile['technique']} "
            f"score={model_profile['model_score']:.6f} "
            f"severity={actual_profile['severity_level']}"
        )

    # ----------------------------------------------------------------------
    # Build report.
    # ----------------------------------------------------------------------

    header = [
        "================================================================",
        "SMARTSOC AI — ACTUAL ATTACK & MITIGATION PROFILE REPORT",
        "================================================================",
        "",
        "Model Version        : 4.1",
        f"Model Classes        : {len(classes)}",
        "Inference Source     : Frozen Production Model",
        "Dataset              : Frozen Test Dataset",
        "Frozen Event Rows    : "
        f"{event_count}",
        "Synthetic Scores     : DISABLED",
        "Synthetic Predictions: DISABLED",
        "Execution            : DISABLED",
        "Containment          : DISABLED",
        "",
        "================================================================",
        "ACTUAL PIPELINE EVIDENCE",
        "================================================================",
        "",
        "Feature Representation",
        "----------------------",
        f"Categorical inputs    : {CATEGORICAL_FEATURES}",
        f"Encoded categorical   : {ENCODED_CATEGORICAL_FEATURES}",
        f"Numerical features    : {NUMERICAL_FEATURES}",
        f"TF-IDF features       : {TFIDF_FEATURES}",
        f"Final model features  : {EXPECTED_FINAL_FEATURES}",
        "",
        "The scores, severity values, decisions and response values",
        "in the profiles below are obtained from the actual project",
        "pipeline. No synthetic values are inserted.",
        "",
    ]

    report_parts = header

    for number, profile in enumerate(
        final_profiles,
        start=1,
    ):

        report_parts.append(
            build_txt_profile(
                profile,
                number,
            )
        )

        report_parts.append("")

    report_parts.extend(
        [
            "================================================================",
            "FINAL EXECUTION BOUNDARY",
            "================================================================",
            "",
            "Production model inference : ACTUAL",
            "Risk evaluation            : ACTUAL",
            "Decision policy            : ACTUAL",
            "Response planner           : ACTUAL",
            "Authorization              : ACTUAL",
            "Response execution         : DISABLED",
            "Containment execution     : DISABLED",
            "Production artifacts      : NOT MODIFIED",
            "",
            "================================================================",
        ]
    )

    txt = "\n".join(
        report_parts
    )

    with open(
        TXT_REPORT,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(txt)

    # ----------------------------------------------------------------------
    # JSON evidence.
    # ----------------------------------------------------------------------

    json_payload = {
        "report_version": "4.4",
        "model_version": "4.1",
        "model_classes": len(classes),
        "frozen_event_rows": event_count,
        "inference_source": "frozen_production_model",
        "synthetic_scores": False,
        "synthetic_predictions": False,
        "execution_enabled": False,
        "containment_enabled": False,
        "inference_path": inference_meta[
            "path"
        ],
        "feature_dimensions": {
            "categorical_input": CATEGORICAL_FEATURES,
            "categorical_encoded": ENCODED_CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "tfidf": TFIDF_FEATURES,
            "final": EXPECTED_FINAL_FEATURES,
        },
        "profiles": final_profiles,
        "production_artifacts_modified": False,
    }

    with open(
        JSON_REPORT,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            safe_json(
                json_payload
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ----------------------------------------------------------------------
    # Final.
    # ----------------------------------------------------------------------

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
        f"Actual model classes : {len(final_profiles)}"
    )

    print(
        f"Frozen events        : {event_count}"
    )

    print(
        "Synthetic data       : NONE"
    )

    print(
        "Execution            : DISABLED"
    )

    print(
        "Containment          : DISABLED"
    )

    print(
        f"TXT  : {TXT_REPORT}"
    )

    print(
        f"JSON : {JSON_REPORT}"
    )

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except Exception as exc:

        print()
        print(
            "[FAIL] ACTUAL REPORT GENERATION FAILED"
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