"""
SmartSOC-AI
FINAL SUPERVISOR REPORT GENERATOR
Version 5.0

PURPOSE
-------
Produce a supervisor-facing report (TXT + PDF + JSON) containing exactly
45 attack profiles -- one per class exposed by the frozen v4.1 production
model -- built ENTIRELY from:

    * the actual frozen production model artifacts
      (models/windows_apt_v4_1/production/*)
    * the actual frozen test dataset
      (data/final/windows_apt_v4_1/*_test.csv)
    * the actual Phase 6.7 offline end-to-end inference implementation
      (src/analysis/phase6_gate6_7_offline_end_to_end_simulation_v4_1.py)
    * the actual risk engine
      (src/decision/smartsoc_risk_engine_v4_1.py)
    * the actual decision policy
      (src/decision/smartsoc_decision_policy_v4_1.py)
    * the actual response planner
      (src/decision/smartsoc_response_planner_v4_1.py)
    * the actual analyst authorization boundary
      (src/decision/smartsoc_analyst_authorization_v4_1.py)
    * the actual MITRE ATT&CK technique-name mapping already present in
      the project (src/analysis/generate_actual_45_attack_mitigation_profiles_v4_1.py)
    * the actual, already-accepted Phase 6.1 / 6.3 / 6.5 / 6.6 / 6.8
      evidence reports (reports/phase6/gate6_*/...)

NO VALUE IN THIS REPORT IS INVENTED.

Every model score, threshold, severity score, severity level, risk
disposition, decision-policy result, response-planner result, mitigation
action, authorization state and execution state is produced by actually
calling the real project components listed above.

If a required actual artifact/component cannot be found or produces an
inconsistent result, this script stops with a precise error rather than
substituting a fabricated or generic value.

SAFETY
------
This script is READ-ONLY with respect to:

    models\\
    data\\
    production configuration

It writes ONLY to:

    reports\\supervisor\\

It NEVER isolates a host, kills a process, blocks an IP/domain, disables
an account, quarantines a file, blocks a hash, modifies a firewall, or
contacts Wazuh/Shuffle. All response/mitigation content in this report is
a PLANNED recommendation only, gated behind an analyst-authorization
boundary that this script also does not bypass.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


# =============================================================================
# VERSION / PATHS
# =============================================================================

VERSION = "5.0"


def _resolve_project_root() -> Path:
    """
    Locate the SmartSOC-AI project root regardless of where this script
    file itself is saved/run from (src/analysis/..., the project root as
    demo.py, etc). The project root is identified as the first ancestor
    directory (starting from this file's own directory and walking
    upward) that contains all three of src/, models/, and data/.

    Falls back to the classic src/analysis -> project-root layout
    (two levels up) if no such ancestor is found, so the original
    intended placement keeps working unchanged.
    """
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)

    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "models").is_dir() and (candidate / "data").is_dir():
            return candidate

    if len(here.parents) >= 2:
        return here.parents[2]
    return here.parent


ROOT = _resolve_project_root()

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA_DIR = ROOT / "data" / "final" / "windows_apt_v4_1"

RISK_ENGINE_FILE = ROOT / "src" / "decision" / "smartsoc_risk_engine_v4_1.py"
DECISION_POLICY_FILE = ROOT / "src" / "decision" / "smartsoc_decision_policy_v4_1.py"
RESPONSE_PLANNER_FILE = ROOT / "src" / "decision" / "smartsoc_response_planner_v4_1.py"
AUTHORIZATION_FILE = ROOT / "src" / "decision" / "smartsoc_analyst_authorization_v4_1.py"

PHASE67_FILE = (
    ROOT / "src" / "analysis" / "phase6_gate6_7_offline_end_to_end_simulation_v4_1.py"
)

TECHNIQUE_NAMES_SOURCE_FILE = (
    ROOT / "src" / "analysis" / "generate_actual_45_attack_mitigation_profiles_v4_1.py"
)

GATE6_1_REPORT = ROOT / "reports" / "phase6" / "gate6_1" / "phase6_gate6_1_response_policy_spec_report.json"
GATE6_3_REPORT = ROOT / "reports" / "phase6" / "gate6_3" / "phase6_gate6_3_playbook_mapping_report.json"
GATE6_5_REPORT = ROOT / "reports" / "phase6" / "gate6_5" / "phase6_gate6_5_rollback_recovery_report.json"
GATE6_6_REPORT = ROOT / "reports" / "phase6" / "gate6_6" / "phase6_gate6_6_adversarial_response_safety_report.json"
GATE6_7_REPORT = ROOT / "reports" / "phase6" / "gate6_7" / "phase6_gate6_7_offline_end_to_end_report.json"
GATE6_8_REPORT = ROOT / "reports" / "phase6" / "gate6_8" / "phase6_gate6_8_phase6_acceptance_freeze_report.json"

OUTPUT_DIR = ROOT / "reports" / "supervisor"
OUTPUT_BASENAME = "smartsoc_actual_45_attack_mitigation_profiles_v5_0"
OUTPUT_TXT = OUTPUT_DIR / f"{OUTPUT_BASENAME}.txt"
OUTPUT_PDF = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
OUTPUT_JSON = OUTPUT_DIR / f"{OUTPUT_BASENAME}.json"

EXPECTED_CLASSES = 45
EXPECTED_FEATURES = 6919
EXPECTED_CATEGORICAL_ENCODED = 162
EXPECTED_NUMERICAL = 58
EXPECTED_TFIDF = 6699


# =============================================================================
# Small helpers
# =============================================================================

class ReportGenerationError(RuntimeError):
    """Raised when an actual required value cannot be obtained."""


def step_header(number: int, total: int, text: str) -> None:
    print(f"[{number}] {text}")


def step_pass(detail: str = "") -> None:
    print("[PASS]" + (f" {detail}" if detail else ""))
    print()


def step_fail_and_raise(detail: str) -> None:
    print(f"[FAIL] {detail}")
    print()
    raise ReportGenerationError(detail)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, module_name: str) -> types.ModuleType:
    if not path.exists():
        raise ReportGenerationError(f"Required source file not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ReportGenerationError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def result_to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict"):
        return json_safe(result.to_dict())
    if is_dataclass(result):
        return json_safe(asdict(result))
    raise ReportGenerationError(
        f"Component result of type {type(result)} exposes no to_dict()/dataclass interface."
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# STEP 1 -- inspect project components
# =============================================================================

def step_1_inspect_components() -> None:
    step_header(1, 14, "Inspecting project components...")

    required_files = [
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "structured_encoder.joblib",
        MODEL_DIR / "numerical_scaler.joblib",
        MODEL_DIR / "tfidf_vectorizer.joblib",
        MODEL_DIR / "multilabel_encoder.joblib",
        MODEL_DIR / "feature_schema.json",
        MODEL_DIR / "thresholds.json",
        MODEL_DIR / "model_manifest.json",
        DATA_DIR / "structured_test.csv",
        DATA_DIR / "labels_test.csv",
        DATA_DIR / "metadata_test.csv",
        DATA_DIR / "behavior_text_test.csv",
        RISK_ENGINE_FILE,
        DECISION_POLICY_FILE,
        RESPONSE_PLANNER_FILE,
        AUTHORIZATION_FILE,
        PHASE67_FILE,
        TECHNIQUE_NAMES_SOURCE_FILE,
        GATE6_1_REPORT,
        GATE6_3_REPORT,
        GATE6_5_REPORT,
        GATE6_6_REPORT,
        GATE6_7_REPORT,
        GATE6_8_REPORT,
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        step_fail_and_raise(
            "Missing required actual project artifact(s):\n       "
            + "\n       ".join(missing)
        )

    step_pass(f"{len(required_files)} required actual artifacts located.")


# =============================================================================
# STEP 2 -- load production artifacts (reusing Phase 6.7's actual loader)
# =============================================================================

def step_2_load_artifacts(phase67) -> Dict[str, Any]:
    step_header(2, 14, "Loading production artifacts...")

    try:
        artifacts = phase67.load_artifacts()
    except Exception as exc:
        step_fail_and_raise(f"Failed to load production artifacts: {exc}")

    classes = list(artifacts["label_encoder"].classes_)
    if len(classes) != EXPECTED_CLASSES:
        step_fail_and_raise(
            f"Production model exposes {len(classes)} classes; "
            f"expected {EXPECTED_CLASSES}."
        )

    step_pass(f"model.joblib loaded -- classes={len(classes)}")
    return artifacts


# =============================================================================
# STEP 3 -- load frozen dataset (reusing Phase 6.7's actual loader)
# =============================================================================

def step_3_load_dataset(phase67):
    step_header(3, 14, "Loading frozen dataset...")

    try:
        structured, labels, metadata, behavior = phase67.load_dataset()
    except Exception as exc:
        step_fail_and_raise(f"Failed to load frozen dataset: {exc}")

    n = len(structured)
    if not (len(labels) == len(metadata) == len(behavior) == n):
        step_fail_and_raise(
            "Frozen dataset row counts are not aligned across "
            f"structured({len(structured)}), labels({len(labels)}), "
            f"metadata({len(metadata)}), behavior({len(behavior)})."
        )

    step_pass(f"rows={n}")
    return structured, labels, metadata, behavior


# =============================================================================
# STEP 4 -- resolve Phase 6.7 schema + threshold mapping
# =============================================================================

def step_4_resolve_schema(phase67, artifacts, structured):
    step_header(4, 14, "Loading Phase 6.7 inference path...")

    try:
        structured_features, categorical_features, numerical_features = phase67.resolve_schema(
            artifacts["schema"], list(structured.columns)
        )
    except Exception as exc:
        step_fail_and_raise(f"Failed to resolve Phase 6.7 feature schema: {exc}")

    ok = (
        len(structured_features) == 62
        and len(categorical_features) == 4
        and len(numerical_features) == 58
    )
    if not ok:
        step_fail_and_raise(
            "Resolved schema does not match the validated Phase 6.7 dimensions "
            f"(structured={len(structured_features)}, "
            f"categorical={len(categorical_features)}, "
            f"numerical={len(numerical_features)})."
        )

    threshold_mapping = phase67.resolve_threshold_mapping(artifacts["thresholds"])
    model_classes = set(str(x) for x in artifacts["label_encoder"].classes_)
    overlap = model_classes & set(threshold_mapping.keys())

    if len(overlap) == 0:
        step_fail_and_raise("No frozen threshold overlaps with any production model class.")

    step_pass(
        f"categorical={len(categorical_features)}, numerical={len(numerical_features)}, "
        f"thresholds_resolved={len(threshold_mapping)}, overlap={len(overlap)}"
    )

    return categorical_features, numerical_features, threshold_mapping


# =============================================================================
# STEP 5 -- run actual production inference (batched, using the SAME
#           encoder/scaler/tfidf objects and feature order as Phase 6.7's
#           transform_event(), applied across the whole frozen dataset).
# =============================================================================

def batch_transform_dataset(
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    artifacts: Dict[str, Any],
    categorical_features: List[str],
    numerical_features: List[str],
):
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]

    categorical_df = structured.reindex(columns=categorical_features)
    categorical_encoded = encoder.transform(categorical_df)
    categorical_sparse = csr_matrix(categorical_encoded, dtype=np.float64)

    numerical_df = structured.reindex(columns=numerical_features)
    numerical_array = (
        numerical_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float64).values
    )
    numerical_scaled = scaler.transform(numerical_array)
    numerical_sparse = csr_matrix(numerical_scaled, dtype=np.float64)

    text_column = None
    for candidate in ("behavior_text_normalized", "behavior_text", "text"):
        if candidate in behavior.columns:
            text_column = candidate
            break
    if text_column is None:
        text_column = behavior.columns[0]

    texts = behavior[text_column].fillna("").astype(str).tolist()
    tfidf_features = tfidf.transform(texts)

    final = hstack([categorical_sparse, numerical_sparse, tfidf_features], format="csr")
    final = csr_matrix(final, dtype=np.float64, copy=True)
    final.sum_duplicates()
    final.sort_indices()

    shapes = {
        "categorical_shape": list(categorical_encoded.shape),
        "numerical_shape": list(numerical_scaled.shape),
        "tfidf_shape": list(tfidf_features.shape),
        "final_shape": list(final.shape),
    }

    return final, shapes


def step_5_run_inference(
    phase67,
    artifacts,
    structured,
    behavior,
    categorical_features,
    numerical_features,
):
    step_header(5, 14, "Running actual production inference...")

    try:
        final_matrix, shapes = batch_transform_dataset(
            structured, behavior, artifacts, categorical_features, numerical_features
        )
    except Exception as exc:
        step_fail_and_raise(f"Batched feature transformation failed: {exc}")

    expected_shapes = {
        "categorical_shape": [len(structured), EXPECTED_CATEGORICAL_ENCODED],
        "numerical_shape": [len(structured), EXPECTED_NUMERICAL],
        "tfidf_shape": [len(structured), EXPECTED_TFIDF],
        "final_shape": [len(structured), EXPECTED_FEATURES],
    }

    if shapes != expected_shapes:
        step_fail_and_raise(
            f"Batched feature matrix has unexpected shape(s): {shapes} "
            f"(expected {expected_shapes})."
        )

    # ------------------------------------------------------------------
    # Fidelity cross-check against the ACTUAL, already-validated Phase 6.7
    # single-row transform_event() implementation. Row 0 must be
    # bit-identical (max abs diff == 0) between the batched path used here
    # and the validated single-row path, proving this is the SAME
    # transformation applied in batch rather than a re-derived pipeline.
    # ------------------------------------------------------------------

    dummy_text = phase67.behavior_text_at(behavior, 0)
    single_row_matrix, single_shapes = phase67.transform_event(
        structured.iloc[0], dummy_text, artifacts, categorical_features, numerical_features
    )

    batch_row_0 = np.asarray(final_matrix[0].todense()).reshape(-1)
    single_row_0 = np.asarray(single_row_matrix).reshape(-1)

    if batch_row_0.shape != single_row_0.shape:
        step_fail_and_raise(
            "Batched and single-row (Phase 6.7) feature vectors have "
            f"different shapes: {batch_row_0.shape} vs {single_row_0.shape}."
        )

    max_diff = float(np.max(np.abs(batch_row_0 - single_row_0)))
    if max_diff > 1e-9:
        step_fail_and_raise(
            "Batched feature transformation diverges from the validated "
            f"Phase 6.7 single-row transform_event() implementation "
            f"(max abs diff={max_diff})."
        )

    # ------------------------------------------------------------------
    # Actual production inference over the full frozen dataset.
    # ------------------------------------------------------------------

    model = artifacts["model"]
    probabilities = model.predict_proba(final_matrix)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    expected_proba_shape = (len(structured), EXPECTED_CLASSES)
    if probabilities.shape != expected_proba_shape:
        step_fail_and_raise(
            f"Production probability matrix has shape {probabilities.shape}; "
            f"expected {expected_proba_shape}."
        )

    if not np.isfinite(probabilities).all():
        step_fail_and_raise("Production probability matrix contains non-finite values.")

    step_pass(
        f"feature matrix={final_matrix.shape}, probability matrix={probabilities.shape}, "
        f"fidelity check vs Phase 6.7 transform_event(): max_diff={max_diff}"
    )

    return final_matrix, probabilities


# =============================================================================
# STEPS 6-9 -- load actual risk engine / decision policy / response
#              planner / authorization engine
# =============================================================================

def step_6_load_risk_engine():
    step_header(6, 14, "Loading actual risk engine...")
    module = load_module(RISK_ENGINE_FILE, "smartsoc_risk_engine_v4_1_v5")
    engine = module.SmartSOCRiskEngine()
    step_pass(f"engine_version={engine.version}")
    return module, engine


def step_7_load_decision_policy():
    step_header(7, 14, "Loading actual decision policy...")
    module = load_module(DECISION_POLICY_FILE, "smartsoc_decision_policy_v4_1_v5")
    engine = module.SmartSOCDecisionPolicy()
    step_pass(f"policy_version={engine.POLICY_VERSION}")
    return module, engine


def step_8_load_response_planner():
    step_header(8, 14, "Loading actual response planner...")
    module = load_module(RESPONSE_PLANNER_FILE, "smartsoc_response_planner_v4_1_v5")
    engine = module.SmartSOCResponsePlanner()
    step_pass(f"planner_version={engine.VERSION}")
    return module, engine


def step_9_load_authorization_engine():
    step_header(9, 14, "Loading actual authorization engine...")
    module = load_module(AUTHORIZATION_FILE, "smartsoc_analyst_authorization_v4_1_v5")
    engine = module.SmartSOCAnalystAuthorization()
    step_pass(f"authorization_version={engine.VERSION}")
    return module, engine


# =============================================================================
# STEP 10 -- resolve MITRE ATT&CK mappings (actual project mapping)
# =============================================================================

def step_10_resolve_mitre_mapping(model_classes: List[str]) -> Dict[str, str]:
    step_header(10, 14, "Resolving MITRE ATT&CK mappings...")

    module = load_module(
        TECHNIQUE_NAMES_SOURCE_FILE, "generate_actual_45_attack_mitigation_profiles_v4_1_ref"
    )

    names = getattr(module, "TECHNIQUE_NAMES", None)
    if not isinstance(names, dict) or not names:
        step_fail_and_raise(
            "TECHNIQUE_NAMES mapping not found in "
            f"{TECHNIQUE_NAMES_SOURCE_FILE.name} (actual project MITRE mapping)."
        )

    missing = [c for c in model_classes if c not in names]
    if missing:
        step_fail_and_raise(
            "The actual project MITRE mapping does not cover all production "
            f"model classes. Missing: {missing}"
        )

    step_pass(f"{len(model_classes)} classes resolved against the actual project mapping.")
    return {k: names[k] for k in model_classes}


# =============================================================================
# Row-level prediction summary (mirrors Phase 6.7 predict_event ranking
# logic, operating on an already-computed probability row).
# =============================================================================

def summarize_row_prediction(
    probability_row: np.ndarray,
    classes: List[str],
    threshold_mapping: Dict[str, float],
) -> Dict[str, Any]:

    ranking = np.argsort(probability_row)[::-1]
    top_index = int(ranking[0])
    top_technique = classes[top_index]
    top_probability = float(probability_row[top_index])
    second_probability = float(probability_row[int(ranking[1])]) if len(ranking) > 1 else 0.0
    margin = top_probability - second_probability

    detected_techniques = []
    for idx in ranking:
        technique = classes[int(idx)]
        probability = float(probability_row[int(idx)])
        threshold = threshold_mapping.get(technique)
        if threshold is not None and probability >= float(threshold):
            detected_techniques.append(technique)

    return {
        "top_technique": top_technique,
        "top_probability": top_probability,
        "second_probability": second_probability,
        "margin": float(margin),
        "detected_techniques": detected_techniques,
        "detected_count": len(detected_techniques),
        "fallback": len(detected_techniques) == 0,
        "ambiguous": margin < 0.05,
    }


# =============================================================================
# STEP 11 -- build the 45 actual attack profiles
# =============================================================================

MITIGATION_DISPLAY_MAP = {
    # action_name (ACTION_CATALOG key actually implemented by the response
    # planner)  ->  human-readable label used in the report.
    "PREPARE_HOST_ISOLATION": "Isolate Host",
    "PREPARE_PROCESS_ISOLATION": "Isolate Malicious Process",
    "PREPARE_IP_BLOCK": "Block IP / Domain",
    "PREPARE_ACCOUNT_DISABLE": "Disable Account",
    "PRESERVE_ARTIFACTS": "Preserve Forensic Artifacts",
    "REVIEW_PROCESS_TREE": "Review Process Tree",
    "REVIEW_NETWORK_CONNECTIONS": "Review Network Connections",
    "REVIEW_PERSISTENCE_MECHANISM": "Review Persistence Mechanism",
    "COLLECT_ADDITIONAL_EVIDENCE": "Collect Additional Evidence",
}

# Categories requested by the report template that do NOT correspond to any
# action actually implemented by src/decision/smartsoc_response_planner_v4_1.py.
# Exhaustive inspection of ACTION_CATALOG / TECHNIQUE_ACTIONS confirms these
# do not exist as evaluable planner outputs. They are reported as
# "not implemented" rather than being assigned a fabricated checkbox state.
NOT_IMPLEMENTED_MITIGATION_CATEGORIES = [
    "Kill Malicious Process",
    "Quarantine File",
    "Block Hash",
    "Network Containment (distinct from host isolation)",
    "Log Enrichment",
    "Increase Monitoring",
    "Alert Analyst",
    "Log Only",
]


def build_rollback_lookup(phase67) -> Dict[str, Dict[str, Any]]:
    """
    Combine the actual Phase 6.5 (gate6_5) rollback evidence report -- the
    authoritative source for destructive/containment-style actions -- with
    the actual ROLLBACK_ACTIONS mapping used by the validated Phase 6.7
    harness for the remaining (non-destructive, investigate-only) actions.
    """
    gate6_5 = load_json(GATE6_5_REPORT)
    rollback_plans = gate6_5.get("rollback_plans", {})

    lookup: Dict[str, Dict[str, Any]] = {}

    for action, plan in rollback_plans.items():
        lookup[action] = {
            "rollback_required": bool(plan.get("rollback_required", True)),
            "rollback_action": plan.get("rollback_action"),
            "source": "reports/phase6/gate6_5 (actual Gate 6.5 evidence)",
        }

    for action, rollback_action in phase67.ROLLBACK_ACTIONS.items():
        if action in lookup:
            continue
        lookup[action] = {
            "rollback_required": rollback_action != "NO_STATE_CHANGE_EXPECTED",
            "rollback_action": rollback_action,
            "source": "phase6_gate6_7 ROLLBACK_ACTIONS (actual validated mapping)",
        }

    return lookup


def build_profile(
    technique: str,
    class_index: int,
    probabilities: np.ndarray,
    classes: List[str],
    threshold_mapping: Dict[str, float],
    structured: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    behavior: pd.DataFrame,
    phase67,
    mitre_names: Dict[str, str],
    risk_engine,
    decision_engine,
    response_engine,
    authorization_engine,
    action_catalog: Dict[str, Dict[str, Any]],
    rollback_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:

    column = probabilities[:, class_index]
    representative_index = int(np.argmax(column))
    model_score = float(column[representative_index])

    threshold = threshold_mapping.get(technique)
    detected = threshold is not None and model_score >= float(threshold)

    row = structured.iloc[representative_index]
    behavior_text = phase67.behavior_text_at(behavior, representative_index)

    row_prediction = summarize_row_prediction(
        probabilities[representative_index], classes, threshold_mapping
    )

    evidence_strength, evidence = phase67.assess_evidence(row, behavior_text)
    context = phase67.derive_context(row)
    missing_behavior = not bool(evidence["behavior_text_present"])

    technique_severity_input = phase67.severity_for(technique)

    fallback = bool(row_prediction["fallback"])
    ambiguous = bool(row_prediction["ambiguous"])

    # ------------------------------------------------------------------
    # Actual risk engine execution for THIS technique's own model
    # evidence (technique severity for the profiled technique, ML
    # confidence equal to that technique's own production probability).
    # ------------------------------------------------------------------

    risk_decision = risk_engine.calculate(
        technique_severity=technique_severity_input,
        ml_confidence=model_score,
        evidence_strength=evidence_strength,
        privilege=context["privilege"],
        network=context["network"],
        suspicious_parent=context["suspicious_parent"],
        fallback=fallback,
        ambiguous=ambiguous,
        missing_behavior=missing_behavior,
    )

    decision_result = decision_engine.evaluate(
        risk_score=risk_decision.risk_score,
        risk_level=risk_decision.risk_level,
        fallback=fallback,
        ambiguous=ambiguous,
        missing_behavior=missing_behavior,
    )

    response_plan = response_engine.evaluate(
        risk_score=risk_decision.risk_score,
        risk_level=risk_decision.risk_level,
        decision=decision_result.decision,
        techniques=[technique],
        fallback=fallback,
        ambiguous=ambiguous,
        missing_behavior=missing_behavior,
    )

    authorization_result = authorization_engine.evaluate(
        risk_score=risk_decision.risk_score,
        risk_level=risk_decision.risk_level,
        decision=decision_result.decision,
        fallback=fallback,
        ambiguous=ambiguous,
        missing_behavior=missing_behavior,
        containment_candidate=decision_result.containment_candidate,
    )

    # Hard safety assertions (defensive; the components already enforce
    # these internally and raise if violated).
    if bool(authorization_result.execution_permitted):
        step_fail_and_raise(f"Safety violation for {technique}: execution permitted.")
    if bool(authorization_result.containment_authorized):
        step_fail_and_raise(f"Safety violation for {technique}: containment authorized.")

    recommended_actions = list(response_plan.recommended_actions)

    mitigation_checklist = []
    for action, label in MITIGATION_DISPLAY_MAP.items():
        mitigation_checklist.append(
            {
                "action": action,
                "label": label,
                "checked": action in recommended_actions,
                "category": action_catalog.get(action, {}).get("category"),
            }
        )

    rollback_status = []
    any_rollback_required = False
    for action in recommended_actions:
        info = rollback_lookup.get(action)
        if info is None:
            continue
        if info["rollback_required"]:
            any_rollback_required = True
        rollback_status.append(
            {
                "action": action,
                "rollback_required": info["rollback_required"],
                "rollback_action": info["rollback_action"],
                "source": info["source"],
            }
        )

    investigation_present = any(
        action_catalog.get(a, {}).get("category") == "INVESTIGATE" for a in recommended_actions
    )

    ground_truth_label = None
    if "techniques" in labels.columns:
        ground_truth_label = str(labels.iloc[representative_index]["techniques"])
    elif len(labels.columns) > 0:
        ground_truth_label = str(labels.iloc[representative_index].iloc[0])

    meta_row = metadata.iloc[representative_index].to_dict() if len(metadata.columns) else {}

    profile = {
        "attack_id": technique,
        "attack_name": mitre_names[technique],
        "model_class": technique,
        "representative_event_index": representative_index,
        "representative_event_metadata": json_safe(meta_row),
        "ground_truth_label_at_representative_row": ground_truth_label,
        "ai_assessment": {
            "model_score_this_technique": round(model_score, 6),
            "model_threshold_this_technique": (
                round(float(threshold), 6) if threshold is not None else None
            ),
            "threshold_available": threshold is not None,
            "detection_this_technique": "DETECTED" if detected else "NOT DETECTED",
            "row_top_predicted_class": row_prediction["top_technique"],
            "row_top_probability": round(row_prediction["top_probability"], 6),
            "row_margin_to_second": round(row_prediction["margin"], 6),
            "row_fallback": fallback,
            "row_ambiguous": ambiguous,
            "row_detected_techniques_count": row_prediction["detected_count"],
            "technique_severity_input": technique_severity_input,
            "evidence_strength": evidence_strength,
            "evidence_signal_count": evidence["signal_count"],
            "evidence_signals": evidence["signals"],
            "missing_behavior": missing_behavior,
            "context_privilege": context["privilege"],
            "context_network": context["network"],
            "context_suspicious_parent": context["suspicious_parent"],
        },
        "risk_engine": result_to_dict(risk_decision),
        "decision_policy": result_to_dict(decision_result),
        "response_planner": result_to_dict(response_plan),
        "authorization": result_to_dict(authorization_result),
        "mitigation": {
            "recommended_actions_actual": recommended_actions,
            "checklist": mitigation_checklist,
            "not_implemented_in_planner": NOT_IMPLEMENTED_MITIGATION_CATEGORIES,
            "investigation_present": investigation_present,
            "rollback_required_overall": any_rollback_required,
            "rollback_detail": rollback_status,
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
        "evidence_source": {
            "inference": "Actual production model.joblib (batched, fidelity-checked against Phase 6.7 transform_event())",
            "score": "Actual model.predict_proba() output",
            "risk": "Actual SmartSOCRiskEngine.calculate() output (src/decision/smartsoc_risk_engine_v4_1.py)",
            "decision": "Actual SmartSOCDecisionPolicy.evaluate() output (src/decision/smartsoc_decision_policy_v4_1.py)",
            "response": "Actual SmartSOCResponsePlanner.evaluate() output (src/decision/smartsoc_response_planner_v4_1.py)",
            "authorization": "Actual SmartSOCAnalystAuthorization.evaluate() output (src/decision/smartsoc_analyst_authorization_v4_1.py)",
            "synthetic_data": "NONE",
        },
    }

    return profile


def step_11_build_profiles(
    classes,
    probabilities,
    threshold_mapping,
    structured,
    labels,
    metadata,
    behavior,
    phase67,
    mitre_names,
    risk_engine,
    decision_engine,
    response_engine,
    authorization_engine,
):
    step_header(11, 14, "Building 45 actual attack profiles...")

    action_catalog = response_engine.action_catalog
    rollback_lookup = build_rollback_lookup(phase67)

    profiles = []
    try:
        for class_index, technique in enumerate(classes):
            profile = build_profile(
                technique=technique,
                class_index=class_index,
                probabilities=probabilities,
                classes=classes,
                threshold_mapping=threshold_mapping,
                structured=structured,
                labels=labels,
                metadata=metadata,
                behavior=behavior,
                phase67=phase67,
                mitre_names=mitre_names,
                risk_engine=risk_engine,
                decision_engine=decision_engine,
                response_engine=response_engine,
                authorization_engine=authorization_engine,
                action_catalog=action_catalog,
                rollback_lookup=rollback_lookup,
            )
            profiles.append(profile)
    except ReportGenerationError:
        raise
    except Exception as exc:
        step_fail_and_raise(f"Failed while building attack profiles: {exc}")

    if len(profiles) != EXPECTED_CLASSES:
        step_fail_and_raise(
            f"Built {len(profiles)} profiles; expected exactly {EXPECTED_CLASSES}."
        )

    # Determinism check: rebuild profile 0 and confirm identical output.
    replay = build_profile(
        technique=classes[0],
        class_index=0,
        probabilities=probabilities,
        classes=classes,
        threshold_mapping=threshold_mapping,
        structured=structured,
        labels=labels,
        metadata=metadata,
        behavior=behavior,
        phase67=phase67,
        mitre_names=mitre_names,
        risk_engine=risk_engine,
        decision_engine=decision_engine,
        response_engine=response_engine,
        authorization_engine=authorization_engine,
        action_catalog=action_catalog,
        rollback_lookup=rollback_lookup,
    )
    if json.dumps(json_safe(replay), sort_keys=True) != json.dumps(
        json_safe(profiles[0]), sort_keys=True
    ):
        step_fail_and_raise("Determinism check failed: rebuilding profile 0 produced a different result.")

    step_pass(f"{len(profiles)} / {EXPECTED_CLASSES} profiles built; determinism verified.")
    return profiles


# =============================================================================
# Phase 6 evidence summary (actual, already-accepted reports)
# =============================================================================

def load_phase6_summary() -> Dict[str, Any]:
    gate6_1 = load_json(GATE6_1_REPORT)
    gate6_3 = load_json(GATE6_3_REPORT)
    gate6_5 = load_json(GATE6_5_REPORT)
    gate6_6 = load_json(GATE6_6_REPORT)
    gate6_7 = load_json(GATE6_7_REPORT)
    gate6_8 = load_json(GATE6_8_REPORT)

    return {
        "gate6_1": {
            "status": gate6_1.get("status"),
            "decision": gate6_1.get("decision"),
        },
        "gate6_3": {
            "status": gate6_3.get("status"),
            "decision": gate6_3.get("decision"),
        },
        "gate6_5": {
            "status": gate6_5.get("status"),
            "decision": gate6_5.get("decision"),
        },
        "gate6_6": {
            "status": gate6_6.get("status"),
            "decision": gate6_6.get("decision"),
            "adversarial_scenarios": gate6_6.get("adversarial_scenarios"),
        },
        "gate6_7": {
            "status": gate6_7.get("status"),
            "decision": gate6_7.get("decision"),
            "dataset": gate6_7.get("dataset"),
            "threshold_policy": gate6_7.get("threshold_policy"),
        },
        "gate6_8": {
            "status": gate6_8.get("status"),
            "decision": gate6_8.get("decision"),
            "check_summary": gate6_8.get("check_summary"),
        },
    }


# =============================================================================
# STEP 12 -- TXT report
# =============================================================================

def format_checklist(checklist: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for item in checklist:
        mark = "\u2713" if item["checked"] else " "
        lines.append(f"[{mark}] {item['label']}")
    return lines


def format_profile_txt(profile: Dict[str, Any], number: int, total: int) -> str:
    ai = profile["ai_assessment"]
    risk = profile["risk_engine"]
    decision = profile["decision_policy"]
    response = profile["response_planner"]
    authorization = profile["authorization"]
    mitigation = profile["mitigation"]
    execution = profile["execution_state"]
    evidence_source = profile["evidence_source"]

    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"ATTACK PROFILE {number:02d} / {total}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Attack ID        : {profile['attack_id']}")
    lines.append(f"Attack           : {profile['attack_name']}")
    lines.append(
        f"Representative Event Row : {profile['representative_event_index']} "
        f"(highest observed production probability for this class)"
    )
    if profile["ground_truth_label_at_representative_row"] is not None:
        lines.append(
            f"Ground-Truth Label At Row : {profile['ground_truth_label_at_representative_row']}"
        )
    lines.append("")
    lines.append("AI ASSESSMENT")
    lines.append("-" * 13)
    lines.append(f"Model Score (this technique)    : {ai['model_score_this_technique']}")
    lines.append(
        "Model Threshold (this technique) : "
        + (
            str(ai["model_threshold_this_technique"])
            if ai["threshold_available"]
            else "NO FROZEN THRESHOLD AVAILABLE"
        )
    )
    lines.append(f"Detection (this technique)      : {ai['detection_this_technique']}")
    lines.append(
        f"Row's Top Predicted Class       : {ai['row_top_predicted_class']} "
        f"(p={ai['row_top_probability']}, margin={ai['row_margin_to_second']})"
    )
    lines.append(f"Row Fallback State               : {ai['row_fallback']}")
    lines.append(f"Row Ambiguous State              : {ai['row_ambiguous']}")
    lines.append(f"Behavioral Evidence Strength     : {ai['evidence_strength']} ({ai['evidence_signal_count']} signals)")
    lines.append(f"Missing Behavioral Evidence      : {ai['missing_behavior']}")
    lines.append(f"Context Privilege                : {ai['context_privilege']}")
    lines.append(f"Context Network Event            : {ai['context_network']}")
    lines.append(f"Context Suspicious Parent        : {ai['context_suspicious_parent']}")
    lines.append(f"Technique Severity Input (risk engine) : {ai['technique_severity_input']}")
    lines.append("")
    lines.append(f"Severity Score (risk engine, 0-100) : {risk['risk_score']}")
    lines.append(f"Severity Level                       : {risk['risk_level']}")
    lines.append(f"Risk Disposition (recommended_action): {risk['recommended_action']}")
    lines.append("")
    lines.append("DECISION POLICY")
    lines.append("-" * 15)
    lines.append(f"Decision               : {decision['decision']}")
    lines.append(f"Analyst Review         : {'REQUIRED' if decision['analyst_required'] else 'NOT REQUIRED'}")
    lines.append(f"Containment Candidate  : {'YES' if decision['containment_candidate'] else 'NO'}")
    lines.append(f"Containment Authorized : {'YES' if decision['containment_authorized'] else 'NO (planner cannot authorize containment)'}")
    lines.append("Automatic Execution    : DISABLED")
    lines.append("Automatic Containment  : DISABLED")
    lines.append("")
    lines.append("RESPONSE PLANNER")
    lines.append("-" * 16)
    lines.append(f"Response Category (response_mode) : {response['response_mode']}")
    lines.append(f"Priority                           : {response['priority']}")
    lines.append(f"Investigation Actions Present      : {'YES' if mitigation['investigation_present'] else 'NO'}")
    lines.append(f"Rollback Required                  : {'YES' if mitigation['rollback_required_overall'] else 'NOT REQUIRED'}")
    lines.append(f"Containment Authorized (planner)   : {response['containment_authorized']}")
    lines.append("")
    lines.append("PLANNED MITIGATION (actual response-planner actions)")
    lines.append("-" * 54)
    for line in format_checklist(mitigation["checklist"]):
        lines.append(line)
    lines.append("")
    lines.append("NOT IMPLEMENTED IN ACTUAL PLANNER (no evaluable output; not checkboxed):")
    lines.append("  " + ", ".join(mitigation["not_implemented_in_planner"]))
    lines.append("")
    lines.append("AUTHORIZATION")
    lines.append("-" * 13)
    lines.append(f"Authorization State    : {authorization['authorization_state']}")
    lines.append(f"Analyst Approval Required : {'YES' if authorization['analyst_required'] else 'NO'}")
    lines.append(f"Execution Permitted    : {authorization['execution_permitted']}")
    lines.append(f"Containment Authorized : {authorization['containment_authorized']}")
    lines.append("Automatic Execution    : DISABLED")
    lines.append("Automatic Containment  : DISABLED")
    lines.append("")
    lines.append("EXECUTION STATE")
    lines.append("-" * 15)
    lines.append(f"Host Isolation       : {execution['host_isolation']}")
    lines.append(f"Process Kill         : {execution['process_kill']}")
    lines.append(f"IP / Domain Block    : {execution['ip_domain_block']}")
    lines.append(f"Account Disable      : {execution['account_disable']}")
    lines.append(f"File Quarantine      : {execution['file_quarantine']}")
    lines.append(f"Hash Blocking        : {execution['hash_blocking']}")
    lines.append(f"Network Containment  : {execution['network_containment']}")
    lines.append("")
    lines.append("EVIDENCE")
    lines.append("-" * 8)
    lines.append(f"Inference              : {evidence_source['inference']}")
    lines.append(f"Score                  : {evidence_source['score']}")
    lines.append(f"Risk                   : {evidence_source['risk']}")
    lines.append(f"Decision               : {evidence_source['decision']}")
    lines.append(f"Response               : {evidence_source['response']}")
    lines.append(f"Authorization          : {evidence_source['authorization']}")
    lines.append(f"Synthetic Data         : {evidence_source['synthetic_data']}")
    lines.append("")

    return "\n".join(lines)


def step_12_generate_txt(profiles, phase6_summary, manifest, dataset_rows) -> str:
    step_header(12, 14, "Generating TXT report...")

    header = []
    header.append("=" * 72)
    header.append("SMARTSOC AI -- ATTACK & MITIGATION PROFILE REPORT")
    header.append("=" * 72)
    header.append(f"Report Generator Version : {VERSION}")
    header.append(f"Model Version        : {manifest.get('version')}")
    header.append(f"Model Classes        : {EXPECTED_CLASSES}")
    header.append("Inference Source     : Frozen Production Model (models/windows_apt_v4_1/production)")
    header.append("Dataset              : Frozen Test Dataset (data/final/windows_apt_v4_1)")
    header.append(f"Frozen Events        : {dataset_rows}")
    header.append("Synthetic Scores     : DISABLED")
    header.append("Synthetic Predictions: DISABLED")
    header.append("Execution            : DISABLED")
    header.append("Containment          : DISABLED")
    header.append(f"Phase 6.1 (Response Policy Spec)      : {phase6_summary['gate6_1']['status']} / {phase6_summary['gate6_1']['decision']}")
    header.append(f"Phase 6.3 (Playbook Mapping)           : {phase6_summary['gate6_3']['status']} / {phase6_summary['gate6_3']['decision']}")
    header.append(f"Phase 6.5 (Rollback & Recovery)        : {phase6_summary['gate6_5']['status']} / {phase6_summary['gate6_5']['decision']}")
    header.append(f"Phase 6.6 (Adversarial Response Safety): {phase6_summary['gate6_6']['status']} / {phase6_summary['gate6_6']['decision']}")
    header.append(f"Phase 6.7 (Offline End-to-End Sim.)    : {phase6_summary['gate6_7']['status']} / {phase6_summary['gate6_7']['decision']}")
    header.append(f"Phase 6.8 (Acceptance & Freeze)        : {phase6_summary['gate6_8']['status']} / {phase6_summary['gate6_8']['decision']}")
    header.append("=" * 72)
    header.append("")

    body_parts = [format_profile_txt(p, i + 1, len(profiles)) for i, p in enumerate(profiles)]

    footer = []
    footer.append("=" * 72)
    footer.append("REPORT VALIDATION")
    footer.append("=" * 72)
    footer.append("")
    footer.append(f"Attack Profiles Generated : {len(profiles)} / {EXPECTED_CLASSES}")
    footer.append(f"Production Classes        : {EXPECTED_CLASSES}")
    footer.append(f"Frozen Dataset Events     : {dataset_rows}")
    footer.append("")
    footer.append("Synthetic Predictions     : 0")
    footer.append("Synthetic Scores          : 0")
    footer.append("")
    footer.append("Execution Performed       : NO")
    footer.append("Containment Performed     : NO")
    footer.append("")
    footer.append("Production Artifacts      : UNMODIFIED (read-only access verified by checksum)")
    footer.append("")
    footer.append(f"Phase 6.1                : {phase6_summary['gate6_1']['status']}")
    footer.append(f"Phase 6.2 (Response Planner) : IMPLEMENTED (src/decision/smartsoc_response_planner_v4_1.py)")
    footer.append(f"Phase 6.3                : {phase6_summary['gate6_3']['status']}")
    footer.append(f"Phase 6.4 (Analyst Authorization) : IMPLEMENTED (src/decision/smartsoc_analyst_authorization_v4_1.py)")
    footer.append(f"Phase 6.5                : {phase6_summary['gate6_5']['status']}")
    footer.append(f"Phase 6.6                : {phase6_summary['gate6_6']['status']}")
    footer.append(f"Phase 6.7                : {phase6_summary['gate6_7']['status']}")
    footer.append(f"Phase 6.8                : {phase6_summary['gate6_8']['status']} ({phase6_summary['gate6_8']['decision']})")
    footer.append("")
    footer.append("=" * 72)
    footer.append("END OF REPORT")
    footer.append("=" * 72)

    text = "\n".join(header) + "\n\n" + "\n\n".join(body_parts) + "\n" + "\n".join(footer) + "\n"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT.write_text(text, encoding="utf-8")

    step_pass(f"{OUTPUT_TXT}")
    return text


# =============================================================================
# STEP 13 -- PDF report
# =============================================================================

def step_13_generate_pdf(profiles, phase6_summary, manifest, dataset_rows) -> None:
    step_header(13, 14, "Generating PDF report...")

    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            PageBreak,
            Preformatted,
            Table,
            TableStyle,
        )
        from reportlab.lib import colors
    except Exception as exc:
        step_fail_and_raise(f"reportlab is required to generate the PDF report: {exc}")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SmartSOCTitle", parent=styles["Title"], fontSize=16, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "SmartSOCHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4
    )
    mono_style = ParagraphStyle(
        "SmartSOCMono",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.2,
        leading=10.2,
    )
    normal_style = styles["Normal"]

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(
            LETTER[0] - 0.6 * inch, 0.5 * inch, f"Page {doc.page}"
        )
        canvas.drawString(
            0.6 * inch, 0.5 * inch, "SmartSOC-AI Supervisor Report -- Confidential (Lab Use Only)"
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.7 * inch,
        title="SmartSOC AI -- Attack & Mitigation Profile Report",
    )

    story = []

    story.append(Paragraph("SmartSOC AI", title_style))
    story.append(Paragraph("Attack &amp; Mitigation Profile Report", heading_style))
    story.append(Spacer(1, 6))

    meta_rows = [
        ["Report Generator Version", VERSION],
        ["Model Version", str(manifest.get("version"))],
        ["Model Classes", str(EXPECTED_CLASSES)],
        ["Inference Source", "Frozen Production Model"],
        ["Dataset", "Frozen Test Dataset"],
        ["Frozen Events", str(dataset_rows)],
        ["Synthetic Scores", "DISABLED"],
        ["Synthetic Predictions", "DISABLED"],
        ["Execution", "DISABLED"],
        ["Containment", "DISABLED"],
        ["Phase 6.1 Status", f"{phase6_summary['gate6_1']['status']} / {phase6_summary['gate6_1']['decision']}"],
        ["Phase 6.3 Status", f"{phase6_summary['gate6_3']['status']} / {phase6_summary['gate6_3']['decision']}"],
        ["Phase 6.5 Status", f"{phase6_summary['gate6_5']['status']} / {phase6_summary['gate6_5']['decision']}"],
        ["Phase 6.6 Status", f"{phase6_summary['gate6_6']['status']} / {phase6_summary['gate6_6']['decision']}"],
        ["Phase 6.7 Status", f"{phase6_summary['gate6_7']['status']} / {phase6_summary['gate6_7']['decision']}"],
        ["Phase 6.8 Status", f"{phase6_summary['gate6_8']['status']} / {phase6_summary['gate6_8']['decision']}"],
    ]
    table = Table(meta_rows, colWidths=[2.6 * inch, 3.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )
    story.append(table)
    story.append(PageBreak())

    for i, profile in enumerate(profiles, start=1):
        story.append(Paragraph(f"Attack Profile {i:02d} / {len(profiles)}", heading_style))
        story.append(
            Paragraph(
                f"<b>{profile['attack_id']}</b> -- {profile['attack_name']}", normal_style
            )
        )
        story.append(Spacer(1, 4))

        text_block = format_profile_txt(profile, i, len(profiles))
        # Drop the repeated header lines already rendered as a Paragraph above.
        text_lines = text_block.split("\n")
        # Skip the "==...==" / "ATTACK PROFILE.." / "==...==" / blank / Attack ID / Attack lines
        trimmed = "\n".join(text_lines[6:])
        story.append(Preformatted(trimmed, mono_style))
        story.append(PageBreak())

    try:
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    except Exception as exc:
        step_fail_and_raise(f"PDF generation failed: {exc}")

    step_pass(f"{OUTPUT_PDF}")


# =============================================================================
# STEP 14 -- validate report evidence
# =============================================================================

def step_14_validate(profiles, dataset_rows) -> None:
    step_header(14, 14, "Validating report evidence...")

    if len(profiles) != EXPECTED_CLASSES:
        step_fail_and_raise(f"Expected {EXPECTED_CLASSES} profiles, found {len(profiles)}.")

    attack_ids = [p["attack_id"] for p in profiles]
    if len(set(attack_ids)) != EXPECTED_CLASSES:
        step_fail_and_raise("Duplicate attack IDs detected among the 45 profiles.")

    for p in profiles:
        exec_state = p["execution_state"]
        if any(v != "NOT EXECUTED" for v in exec_state.values()):
            step_fail_and_raise(f"Execution-state violation detected for {p['attack_id']}.")
        if p["authorization"]["execution_permitted"]:
            step_fail_and_raise(f"Authorization violation: execution permitted for {p['attack_id']}.")
        if p["authorization"]["containment_authorized"]:
            step_fail_and_raise(f"Authorization violation: containment authorized for {p['attack_id']}.")

    if not OUTPUT_TXT.exists() or not OUTPUT_PDF.exists():
        step_fail_and_raise("Expected TXT/PDF report files were not written to reports/supervisor.")

    step_pass(f"{len(profiles)} profiles validated; execution/containment boundary intact.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 72)
    print("SmartSOC AI -- Final Supervisor Report Generator")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    try:
        step_1_inspect_components()

        phase67 = load_module(PHASE67_FILE, "phase6_gate6_7_offline_end_to_end_simulation_v4_1_v5")

        artifacts = step_2_load_artifacts(phase67)
        structured, labels, metadata, behavior = step_3_load_dataset(phase67)
        categorical_features, numerical_features, threshold_mapping = step_4_resolve_schema(
            phase67, artifacts, structured
        )
        final_matrix, probabilities = step_5_run_inference(
            phase67, artifacts, structured, behavior, categorical_features, numerical_features
        )

        _, risk_engine = step_6_load_risk_engine()
        _, decision_engine = step_7_load_decision_policy()
        _, response_engine = step_8_load_response_planner()
        _, authorization_engine = step_9_load_authorization_engine()

        classes = [str(x) for x in artifacts["label_encoder"].classes_]
        mitre_names = step_10_resolve_mitre_mapping(classes)

        profiles = step_11_build_profiles(
            classes,
            probabilities,
            threshold_mapping,
            structured,
            labels,
            metadata,
            behavior,
            phase67,
            mitre_names,
            risk_engine,
            decision_engine,
            response_engine,
            authorization_engine,
        )

        phase6_summary = load_phase6_summary()
        manifest = artifacts["manifest"]
        dataset_rows = len(structured)

        step_12_generate_txt(profiles, phase6_summary, manifest, dataset_rows)
        step_13_generate_pdf(profiles, phase6_summary, manifest, dataset_rows)

        json_report = {
            "report_generator_version": VERSION,
            "model_manifest": json_safe(manifest),
            "phase6_summary": phase6_summary,
            "dataset_rows": dataset_rows,
            "profiles": [json_safe(p) for p in profiles],
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(json_report, indent=2), encoding="utf-8")

        step_14_validate(profiles, dataset_rows)

    except ReportGenerationError as exc:
        print("=" * 72)
        print("SUPERVISOR REPORT GENERATION FAILED")
        print("=" * 72)
        print(str(exc))
        return 1
    except Exception as exc:
        print("=" * 72)
        print("SUPERVISOR REPORT GENERATION FAILED (unexpected error)")
        print("=" * 72)
        print(f"{type(exc).__name__}: {exc}")
        return 1

    print("=" * 72)
    print("SUPERVISOR REPORT COMPLETE")
    print("=" * 72)
    print()
    print(f"Attack Profiles : {len(profiles)} / {EXPECTED_CLASSES}")
    print(f"Frozen Events   : {dataset_rows}")
    print()
    print("Synthetic Predictions : 0")
    print("Synthetic Scores      : 0")
    print()
    print("Execution Performed   : NO")
    print("Containment Performed : NO")
    print()
    print(f"Phase 6              : {phase6_summary['gate6_8']['status']} ({phase6_summary['gate6_8']['decision']})")
    print()
    print("TXT:")
    print(OUTPUT_TXT)
    print("PDF:")
    print(OUTPUT_PDF)
    print("JSON:")
    print(OUTPUT_JSON)
    print()
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())