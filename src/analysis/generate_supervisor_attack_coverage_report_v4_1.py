"""
SmartSOC v4.1
Supervisor Attack Coverage & Mitigation Report

Purpose
-------
Generate a concise, professional TXT report describing the attack/technique
classes actually represented by the frozen production AI model.

This script:
    - Reads the real serialized production model.
    - Reads the real production thresholds.
    - Reads the real feature schema.
    - Reads the real response/decision/authorization components when available.
    - Enumerates the model's actual classes.
    - Resolves MITRE technique identifiers/names from real artifact metadata
      where available.
    - Reports actual threshold/risk information where available.
    - Shows the mitigation decision produced by the existing SmartSOC policy.
    - NEVER fabricates predictions, scores, techniques, or mitigations.
    - Does NOT process/report all 11,707 test rows.
    - Produces one professional TXT report.

Version: 4.1
"""

from __future__ import annotations

import json
import sys
import inspect
import importlib.util
from pathlib import Path
from typing import Any


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "model.joblib"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "thresholds.json"
)

SCHEMA_PATH = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
    / "feature_schema.json"
)

DECISION_POLICY_PATH = (
    PROJECT_ROOT
    / "src"
    / "decision"
    / "smartsoc_decision_policy_v4_1.py"
)

RESPONSE_PLANNER_PATH = (
    PROJECT_ROOT
    / "src"
    / "decision"
    / "smartsoc_response_planner_v4_1.py"
)

AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "src"
    / "decision"
    / "smartsoc_analyst_authorization_v4_1.py"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "supervisor"

REPORT_PATH = (
    REPORT_DIR / "smartsoc_attack_coverage_mitigation_report_v4_1.txt"
)


# ============================================================================
# OPTIONAL DEPENDENCY LOADING
# ============================================================================

def load_python_module(path: Path, module_name: str):
    if not path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(path),
        )

        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    except Exception:
        return None


# ============================================================================
# HELPERS
# ============================================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value).strip()


def first_value(mapping: Any, keys: list[str]) -> Any:
    if not isinstance(mapping, dict):
        return None

    lowered = {
        str(k).lower(): v
        for k, v in mapping.items()
    }

    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]

    return None


def recursive_find(
    obj: Any,
    keys: set[str],
    max_depth: int = 8,
    depth: int = 0,
):
    """
    Conservative recursive lookup.

    Used only for artifact inspection. It does not invent values.
    """

    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys:
                return value

        for value in obj.values():
            result = recursive_find(
                value,
                keys,
                max_depth,
                depth + 1,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = recursive_find(
                value,
                keys,
                max_depth,
                depth + 1,
            )

            if result is not None:
                return result

    return None


def normalize_class_name(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return "UNKNOWN"

    return text


def format_value(value: Any) -> str:
    if value is None:
        return "NOT AVAILABLE IN ARTIFACT"

    if isinstance(value, bool):
        return "YES" if value else "NO"

    if isinstance(value, float):
        return f"{value:.6f}"

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    text = clean_text(value)

    return text if text else "NOT AVAILABLE IN ARTIFACT"


def title_case_technique(value: str) -> str:
    value = value.strip()

    if not value:
        return "Technique name not available"

    return value


# ============================================================================
# ARTIFACT EXTRACTION
# ============================================================================

def extract_model_classes(model: Any) -> list[str]:
    """
    Extract actual classes from the serialized production estimator.

    Supports common sklearn wrappers such as OneVsRestClassifier.
    """

    classes = getattr(model, "classes_", None)

    if classes is None:
        return []

    result = []

    try:
        for item in classes:
            result.append(normalize_class_name(item))
    except Exception:
        return []

    return result


def build_threshold_lookup(thresholds: Any) -> dict[str, Any]:
    """
    Build a conservative lookup from whatever class/threshold structure
    actually exists in thresholds.json.
    """

    lookup: dict[str, Any] = {}

    if not isinstance(thresholds, dict):
        return lookup

    # Direct dictionaries such as:
    # {"T1059": 0.7, ...}
    for key, value in thresholds.items():
        if isinstance(value, (int, float)):
            lookup[str(key)] = value

    # Common nested structures.
    candidate_keys = {
        "thresholds",
        "class_thresholds",
        "per_class_thresholds",
        "technique_thresholds",
        "label_thresholds",
        "mapping",
    }

    for key, value in thresholds.items():
        if str(key).lower() in candidate_keys:
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lookup[str(sub_key)] = sub_value

    return lookup


def find_class_metadata(
    thresholds: Any,
    schema: Any,
    class_name: str,
) -> dict[str, Any]:
    """
    Search actual artifact metadata for information associated with a class.

    Nothing is inferred from generic cybersecurity knowledge.
    """

    result = {
        "technique_id": None,
        "technique_name": None,
        "severity": None,
        "risk_score": None,
        "threshold": None,
        "description": None,
    }

    structures = [
        thresholds,
        schema,
    ]

    wanted_id_keys = {
        "technique_id",
        "mitre_id",
        "mitre_technique",
        "technique",
        "attack_id",
        "id",
    }

    wanted_name_keys = {
        "technique_name",
        "mitre_name",
        "name",
        "attack_name",
        "label_name",
    }

    wanted_severity_keys = {
        "severity",
        "risk_level",
        "risk",
    }

    wanted_score_keys = {
        "risk_score",
        "score",
        "risk",
    }

    wanted_threshold_keys = {
        "threshold",
        "decision_threshold",
        "classification_threshold",
    }

    wanted_description_keys = {
        "description",
        "details",
        "explanation",
    }

    def inspect_mapping(obj: Any):
        if not isinstance(obj, dict):
            return

        # Exact class key.
        for key, value in obj.items():
            if str(key) == class_name:
                if isinstance(value, dict):
                    result["technique_id"] = (
                        first_value(
                            value,
                            list(wanted_id_keys),
                        )
                    )

                    result["technique_name"] = (
                        first_value(
                            value,
                            list(wanted_name_keys),
                        )
                    )

                    result["severity"] = (
                        first_value(
                            value,
                            list(wanted_severity_keys),
                        )
                    )

                    result["risk_score"] = (
                        first_value(
                            value,
                            list(wanted_score_keys),
                        )
                    )

                    result["threshold"] = (
                        first_value(
                            value,
                            list(wanted_threshold_keys),
                        )
                    )

                    result["description"] = (
                        first_value(
                            value,
                            list(wanted_description_keys),
                        )
                    )

                elif isinstance(value, (int, float)):
                    result["threshold"] = value

        # Search common class mappings.
        class_mapping_keys = {
            "classes",
            "labels",
            "techniques",
            "attack_techniques",
            "class_mapping",
            "label_mapping",
            "mitre_mapping",
            "technique_mapping",
        }

        for key, value in obj.items():
            if str(key).lower() in class_mapping_keys:

                if isinstance(value, dict):
                    candidate = value.get(class_name)

                    if isinstance(candidate, dict):
                        result["technique_id"] = (
                            result["technique_id"]
                            or first_value(
                                candidate,
                                list(wanted_id_keys),
                            )
                        )

                        result["technique_name"] = (
                            result["technique_name"]
                            or first_value(
                                candidate,
                                list(wanted_name_keys),
                            )
                        )

                        result["severity"] = (
                            result["severity"]
                            or first_value(
                                candidate,
                                list(wanted_severity_keys),
                            )
                        )

                        result["risk_score"] = (
                            result["risk_score"]
                            or first_value(
                                candidate,
                                list(wanted_score_keys),
                            )
                        )

                        result["threshold"] = (
                            result["threshold"]
                            or first_value(
                                candidate,
                                list(wanted_threshold_keys),
                            )
                        )

                        result["description"] = (
                            result["description"]
                            or first_value(
                                candidate,
                                list(wanted_description_keys),
                            )
                        )

                    elif isinstance(candidate, (int, float)):
                        result["threshold"] = (
                            result["threshold"]
                            or candidate
                        )

    for structure in structures:
        inspect_mapping(structure)

    # Recursive exact-key lookup as a secondary method.
    if result["technique_id"] is None:
        result["technique_id"] = recursive_find(
            thresholds,
            wanted_id_keys,
        )

    if result["technique_name"] is None:
        result["technique_name"] = recursive_find(
            thresholds,
            wanted_name_keys,
        )

    if result["severity"] is None:
        result["severity"] = recursive_find(
            thresholds,
            wanted_severity_keys,
        )

    if result["risk_score"] is None:
        result["risk_score"] = recursive_find(
            thresholds,
            wanted_score_keys,
        )

    if result["threshold"] is None:
        result["threshold"] = recursive_find(
            thresholds,
            wanted_threshold_keys,
        )

    if result["description"] is None:
        result["description"] = recursive_find(
            thresholds,
            wanted_description_keys,
        )

    return result


# ============================================================================
# POLICY INSPECTION
# ============================================================================

def discover_callable(module: Any, preferred_names: list[str]):
    if module is None:
        return None

    for name in preferred_names:
        candidate = getattr(module, name, None)

        if callable(candidate):
            return candidate

    return None


def instantiate_engine(module: Any, preferred_names: list[str]):
    if module is None:
        return None

    for name in preferred_names:
        candidate = getattr(module, name, None)

        if candidate is None:
            continue

        if inspect.isclass(candidate):
            try:
                return candidate()
            except Exception:
                continue

    return None


def safe_call(
    callable_obj,
    kwargs: dict[str, Any],
):
    if callable_obj is None:
        return None

    try:
        signature = inspect.signature(callable_obj)

        accepted = {}

        for name in signature.parameters:
            if name in kwargs:
                accepted[name] = kwargs[name]

        return callable_obj(**accepted)

    except Exception:
        return None


def object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    try:
        data = vars(value)

        if isinstance(data, dict):
            return data
    except Exception:
        pass

    result = {}

    for name in dir(value):
        if name.startswith("_"):
            continue

        try:
            attr = getattr(value, name)

            if callable(attr):
                continue

            if isinstance(
                attr,
                (str, int, float, bool, type(None)),
            ):
                result[name] = attr
        except Exception:
            continue

    return result


def extract_policy_field(
    result: Any,
    names: list[str],
):
    data = object_to_dict(result)

    value = first_value(data, names)

    if value is not None:
        return value

    for name in names:
        try:
            value = getattr(result, name)

            if value is not None:
                return value
        except Exception:
            pass

    return None


def mitigation_from_policy(
    class_name: str,
    metadata: dict[str, Any],
    decision_module: Any,
    response_module: Any,
    authorization_module: Any,
) -> dict[str, Any]:
    """
    Ask the real policy components for a response where their interfaces
    permit it.

    If an interface cannot be safely called without fabricated event data,
    the report explicitly says so.
    """

    risk_level = metadata.get("severity")

    risk_score = metadata.get("risk_score")

    fallback = False
    ambiguous = False
    missing_behavior = False

    decision_engine = instantiate_engine(
        decision_module,
        [
            "SmartSOCDecisionPolicy",
            "DecisionPolicy",
            "DecisionPolicyEngine",
        ],
    )

    response_engine = instantiate_engine(
        response_module,
        [
            "SmartSOCResponsePlanner",
            "ResponsePlanner",
            "ResponsePolicy",
        ],
    )

    authorization_engine = instantiate_engine(
        authorization_module,
        [
            "SmartSOCAnalystAuthorization",
            "AnalystAuthorization",
            "AuthorizationEngine",
        ],
    )

    # Conservative inputs.
    policy_kwargs = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "fallback": fallback,
        "ambiguous": ambiguous,
        "missing_behavior": missing_behavior,
        "technique_id": metadata.get("technique_id"),
        "technique": metadata.get("technique_id") or class_name,
        "class_name": class_name,
    }

    decision_result = None

    if decision_engine is not None:
        evaluate = getattr(
            decision_engine,
            "evaluate",
            None,
        )

        if callable(evaluate):
            decision_result = safe_call(
                evaluate,
                policy_kwargs,
            )

    response_result = None

    if response_engine is not None:
        evaluate = getattr(
            response_engine,
            "plan",
            None,
        )

        if not callable(evaluate):
            evaluate = getattr(
                response_engine,
                "evaluate",
                None,
            )

        if callable(evaluate):
            response_result = safe_call(
                evaluate,
                policy_kwargs,
            )

    authorization_result = None

    if authorization_engine is not None:
        evaluate = getattr(
            authorization_engine,
            "evaluate",
            None,
        )

        if callable(evaluate):
            authorization_result = safe_call(
                evaluate,
                policy_kwargs,
            )

    result = {
        "decision": None,
        "response_category": None,
        "recommended_actions": None,
        "analyst_required": None,
        "containment_authorized": None,
        "execution_enabled": None,
        "reason": None,
    }

    if decision_result is not None:
        result["decision"] = extract_policy_field(
            decision_result,
            [
                "decision",
                "disposition",
                "risk_disposition",
                "action",
            ],
        )

    if response_result is not None:
        result["response_category"] = extract_policy_field(
            response_result,
            [
                "response_category",
                "category",
                "response",
            ],
        )

        result["recommended_actions"] = extract_policy_field(
            response_result,
            [
                "recommended_actions",
                "actions",
                "response_actions",
                "planned_actions",
            ],
        )

        result["containment_authorized"] = extract_policy_field(
            response_result,
            [
                "containment_authorized",
                "containment_allowed",
                "containment",
            ],
        )

        result["execution_enabled"] = extract_policy_field(
            response_result,
            [
                "execution_enabled",
                "automatic_execution",
                "execute",
                "execution_allowed",
            ],
        )

        result["reason"] = extract_policy_field(
            response_result,
            [
                "reason",
                "rationale",
                "explanation",
            ],
        )

    if authorization_result is not None:
        result["analyst_required"] = extract_policy_field(
            authorization_result,
            [
                "analyst_required",
                "requires_analyst",
                "approval_required",
            ],
        )

        authorization_execution = extract_policy_field(
            authorization_result,
            [
                "execution_enabled",
                "execution_allowed",
                "authorized_for_execution",
            ],
        )

        if authorization_execution is not None:
            result["execution_enabled"] = (
                authorization_execution
            )

        containment = extract_policy_field(
            authorization_result,
            [
                "containment_authorized",
                "containment_allowed",
                "authorized_for_containment",
            ],
        )

        if containment is not None:
            result["containment_authorized"] = containment

    return result


# ============================================================================
# REPORT FORMAT
# ============================================================================

def line(char: str = "=", width: int = 88) -> str:
    return char * width


def format_actions(value: Any) -> str:
    if value is None:
        return "NOT AVAILABLE FROM POLICY INTERFACE"

    if isinstance(value, list):
        if not value:
            return "NONE"

        return "; ".join(str(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def build_report(
    model: Any,
    thresholds: Any,
    schema: Any,
    classes: list[str],
    policy_rows: list[dict[str, Any]],
) -> str:

    model_name = type(model).__name__

    report = []

    report.append(line())
    report.append(
        "SmartSOC AI — FROZEN ATTACK COVERAGE & MITIGATION REPORT"
    )
    report.append("Version 4.1")
    report.append(line())
    report.append("")

    report.append("PURPOSE")
    report.append("-" * 88)
    report.append(
        "This report summarizes the attack/technique classes actually "
        "contained in the frozen production AI model."
    )
    report.append(
        "It is generated from the serialized production artifacts and "
        "existing SmartSOC policy components."
    )
    report.append(
        "It does not enumerate the 11,707 frozen test events."
    )
    report.append(
        "No synthetic predictions or fabricated attack results are included."
    )
    report.append("")

    report.append("FROZEN PRODUCTION MODEL")
    report.append("-" * 88)
    report.append(f"Model class                 : {model_name}")
    report.append(
        f"Model attack/technique classes: {len(classes)}"
    )
    report.append(
        f"Threshold artifact           : {THRESHOLDS_PATH}"
    )
    report.append(
        f"Feature schema artifact      : {SCHEMA_PATH}"
    )
    report.append("")

    report.append("IMPORTANT INTERPRETATION")
    report.append("-" * 88)
    report.append(
        "The model class list proves what classes the serialized model "
        "can represent."
    )
    report.append(
        "A class entry is NOT presented as a prediction of a particular "
        "test event."
    )
    report.append(
        "Risk/threshold values are shown only when present in the real "
        "production artifacts."
    )
    report.append(
        "Mitigation is shown as a policy decision/planned response, not "
        "as proof that an action was executed."
    )
    report.append(
        "The validated SmartSOC safety boundary keeps response execution "
        "and containment disabled."
    )
    report.append("")

    report.append(line())
    report.append("ATTACK / TECHNIQUE COVERAGE")
    report.append(line())
    report.append("")

    for index, row in enumerate(policy_rows, start=1):

        report.append(
            f"[{index:02d}] {row['class_name']}"
        )
        report.append("-" * 88)

        report.append(
            f"MITRE Technique ID        : "
            f"{format_value(row['technique_id'])}"
        )

        report.append(
            f"Technique / Attack Name    : "
            f"{format_value(row['technique_name'])}"
        )

        report.append(
            f"Severity / Risk Level      : "
            f"{format_value(row['severity'])}"
        )

        report.append(
            f"Risk Score                 : "
            f"{format_value(row['risk_score'])}"
        )

        report.append(
            f"Model Decision Threshold   : "
            f"{format_value(row['threshold'])}"
        )

        report.append(
            f"Description                : "
            f"{format_value(row['description'])}"
        )

        report.append("")
        report.append("SMARTSOC POLICY DECISION")
        report.append(
            f"Decision / Disposition     : "
            f"{format_value(row['decision'])}"
        )

        report.append(
            f"Response Category          : "
            f"{format_value(row['response_category'])}"
        )

        report.append(
            f"Recommended Mitigation     : "
            f"{format_actions(row['recommended_actions'])}"
        )

        report.append(
            f"Analyst Authorization      : "
            f"{format_value(row['analyst_required'])}"
        )

        report.append(
            f"Containment Authorization  : "
            f"{format_value(row['containment_authorized'])}"
        )

        report.append(
            f"Execution Enabled         : "
            f"{format_value(row['execution_enabled'])}"
        )

        report.append(
            f"Policy Rationale           : "
            f"{format_value(row['reason'])}"
        )

        report.append("")
        report.append(
            "Execution status           : NOT EXECUTED"
        )
        report.append("")

    report.append(line())
    report.append("SYSTEM SAFETY BOUNDARY")
    report.append(line())
    report.append(
        "AI classification           : ENABLED"
    )
    report.append(
        "Risk / decision policy      : ENABLED"
    )
    report.append(
        "Response planning           : ENABLED"
    )
    report.append(
        "Analyst authorization       : ENABLED"
    )
    report.append(
        "Automatic response execution: DISABLED"
    )
    report.append(
        "Automatic containment       : DISABLED"
    )
    report.append(
        "Wazuh integration           : DISABLED"
    )
    report.append(
        "Shuffle integration         : DISABLED"
    )
    report.append(
        "Actual attack execution     : NONE"
    )
    report.append("")

    report.append(line())
    report.append("WHAT THIS REPORT PROVES")
    report.append(line())
    report.append(
        "1. The serialized production AI model contains the listed "
        "attack/technique classes."
    )
    report.append(
        "2. Production thresholds are read from the frozen threshold "
        "artifact when available."
    )
    report.append(
        "3. SmartSOC's decision and response layers can be inspected "
        "against the model's attack classes."
    )
    report.append(
        "4. Response planning is separated from response execution."
    )
    report.append(
        "5. Containment remains behind the validated authorization "
        "boundary."
    )
    report.append(
        "6. No fabricated event predictions are used in this report."
    )
    report.append("")

    report.append(line())
    report.append("SUPERVISOR NOTE")
    report.append(line())
    report.append(
        "For demonstration, select a controlled laboratory event and "
        "run it through the validated inference pipeline. The class-level "
        "coverage in this document should not be presented as evidence "
        "that every listed technique was detected in the frozen test set."
    )
    report.append("")

    report.append(line())

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print(line())
    print("SmartSOC AI — Attack Coverage & Mitigation Report")
    print("Version 4.1")
    print(line())

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Load joblib
    # ------------------------------------------------------------------------

    print()
    print("[1] Loading frozen production artifacts...")

    if not MODEL_PATH.exists():
        print("[FAIL] Production model not found")
        print(f"       {MODEL_PATH}")
        return 1

    if not THRESHOLDS_PATH.exists():
        print("[FAIL] Threshold artifact not found")
        print(f"       {THRESHOLDS_PATH}")
        return 1

    if not SCHEMA_PATH.exists():
        print("[FAIL] Feature schema not found")
        print(f"       {SCHEMA_PATH}")
        return 1

    try:
        import joblib
    except Exception as exc:
        print("[FAIL] joblib unavailable")
        print(f"       {exc}")
        return 1

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        print("[FAIL] Production model could not be loaded")
        print(f"       {exc}")
        return 1

    try:
        thresholds = load_json(THRESHOLDS_PATH)
    except Exception as exc:
        print("[FAIL] thresholds.json could not be loaded")
        print(f"       {exc}")
        return 1

    try:
        schema = load_json(SCHEMA_PATH)
    except Exception as exc:
        print("[FAIL] feature_schema.json could not be loaded")
        print(f"       {exc}")
        return 1

    print("[PASS] Production model loaded")
    print(f"       {MODEL_PATH}")

    print("[PASS] Production thresholds loaded")
    print(f"       {THRESHOLDS_PATH}")

    print("[PASS] Production feature schema loaded")
    print(f"       {SCHEMA_PATH}")

    # ------------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------------

    print()
    print("[2] Extracting actual model classes...")

    classes = extract_model_classes(model)

    if not classes:
        print("[FAIL] Serialized model exposes no classes_ attribute")
        return 1

    print(
        f"[PASS] Actual production model classes discovered: "
        f"{len(classes)}"
    )

    for idx, class_name in enumerate(classes, start=1):
        print(
            f"       {idx:02d}. {class_name}"
        )

    # ------------------------------------------------------------------------
    # Threshold lookup
    # ------------------------------------------------------------------------

    threshold_lookup = build_threshold_lookup(thresholds)

    print()
    print("[3] Resolving threshold information...")

    print(
        f"[PASS] Threshold entries discovered: "
        f"{len(threshold_lookup)}"
    )

    # ------------------------------------------------------------------------
    # Load policy components
    # ------------------------------------------------------------------------

    print()
    print("[4] Loading SmartSOC policy components...")

    decision_module = load_python_module(
        DECISION_POLICY_PATH,
        "smartsoc_decision_policy_v4_1_report",
    )

    response_module = load_python_module(
        RESPONSE_PLANNER_PATH,
        "smartsoc_response_planner_v4_1_report",
    )

    authorization_module = load_python_module(
        AUTHORIZATION_PATH,
        "smartsoc_analyst_authorization_v4_1_report",
    )

    print(
        "[PASS] Decision policy module "
        + ("loaded" if decision_module else "not available")
    )

    print(
        "[PASS] Response planner module "
        + ("loaded" if response_module else "not available")
    )

    print(
        "[PASS] Authorization module "
        + ("loaded" if authorization_module else "not available")
    )

    # ------------------------------------------------------------------------
    # Build rows
    # ------------------------------------------------------------------------

    print()
    print("[5] Building class-level attack coverage report...")

    policy_rows = []

    for class_name in classes:

        metadata = find_class_metadata(
            thresholds,
            schema,
            class_name,
        )

        # Direct threshold lookup takes precedence when it is actually
        # present under the class name.
        if class_name in threshold_lookup:
            metadata["threshold"] = threshold_lookup[class_name]

        policy = mitigation_from_policy(
            class_name,
            metadata,
            decision_module,
            response_module,
            authorization_module,
        )

        row = {
            "class_name": class_name,
            **metadata,
            **policy,
        }

        policy_rows.append(row)

    print(
        f"[PASS] Class-level entries generated: "
        f"{len(policy_rows)}"
    )

    # ------------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------------

    print()
    print("[6] Writing professional TXT report...")

    report = build_report(
        model=model,
        thresholds=thresholds,
        schema=schema,
        classes=classes,
        policy_rows=policy_rows,
    )

    try:
        REPORT_PATH.write_text(
            report,
            encoding="utf-8",
        )
    except Exception as exc:
        print("[FAIL] Could not write report")
        print(f"       {exc}")
        return 1

    print("[PASS] Report written")
    print(f"       {REPORT_PATH}")

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print(line())
    print("SUPERVISOR ATTACK COVERAGE REPORT COMPLETE")
    print(line())
    print(f"Model classes              : {len(classes)}")
    print("Frozen test rows processed : 0")
    print("Synthetic predictions      : DISABLED")
    print("Response execution         : DISABLED")
    print("Containment execution     : DISABLED")
    print(f"TXT report                 : {REPORT_PATH}")
    print(line())

    return 0


if __name__ == "__main__":
    sys.exit(main())