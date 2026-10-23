from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 5.4
# Uncertainty and Safety Policy Validation
# Version 4.1
#
# Purpose:
#   Validate that uncertainty conditions such as fallback,
#   ambiguity, missing behavioral evidence, and high-risk uncertainty
#   cannot bypass the analyst-approval safety boundary.
#
# Safety:
#   This test is completely side-effect free.
#
#   It does NOT:
#       - modify the ML model
#       - modify the dataset
#       - modify thresholds
#       - execute response actions
#       - execute containment
#       - connect Wazuh
#       - connect Shuffle
# =====================================================================


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENGINE_FILE = (
    PROJECT_ROOT
    / "src"
    / "decision"
    / "smartsoc_decision_policy_v4_1.py"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase5"
    / "gate5_4"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase5_gate5_4_uncertainty_safety_validation_report.json"
)


# =====================================================================
# Dynamic import
# =====================================================================

def load_policy_engine():
    import importlib.util

    module_name = "smartsoc_decision_policy_v4_1"

    if not ENGINE_FILE.exists():
        raise FileNotFoundError(
            f"Decision policy engine not found: {ENGINE_FILE}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        ENGINE_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load policy engine: {ENGINE_FILE}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# =====================================================================
# Uncertainty scenarios
# =====================================================================

UNCERTAINTY_SCENARIOS: List[Dict[str, Any]] = [

    {
        "id": "U01",
        "name": "Low-confidence low-risk fallback",
        "risk_score": 20.0,
        "risk_level": "LOW",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U02",
        "name": "Low-confidence medium-risk fallback",
        "risk_score": 40.0,
        "risk_level": "MEDIUM",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U03",
        "name": "Low-confidence high-risk fallback",
        "risk_score": 65.0,
        "risk_level": "HIGH",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U04",
        "name": "Low-confidence critical fallback",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U05",
        "name": "Ambiguous low-risk event",
        "risk_score": 20.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U06",
        "name": "Ambiguous medium-risk event",
        "risk_score": 40.0,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U07",
        "name": "Ambiguous high-risk event",
        "risk_score": 65.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U08",
        "name": "Ambiguous critical event",
        "risk_score": 95.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U09",
        "name": "Critical fallback with missing behavior",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U10",
        "name": "Critical ambiguity with missing behavior",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U11",
        "name": "High-risk fallback with missing behavior",
        "risk_score": 70.0,
        "risk_level": "HIGH",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U12",
        "name": "High-risk ambiguity with missing behavior",
        "risk_score": 70.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U13",
        "name": "Maximum-risk fallback",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U14",
        "name": "Maximum-risk ambiguity",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "U15",
        "name": "Critical fallback and ambiguity",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

]


# =====================================================================
# Normal confidence/risk scenarios
# =====================================================================

NORMAL_SCENARIOS: List[Dict[str, Any]] = [

    {
        "id": "N01",
        "name": "Normal low risk",
        "risk_score": 10.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_candidate": False,
    },

    {
        "id": "N02",
        "name": "Normal medium risk",
        "risk_score": 40.0,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "N03",
        "name": "Normal high risk",
        "risk_score": 65.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "N04",
        "name": "Normal critical risk",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

]


# =====================================================================
# Scenario validation
# =====================================================================

def validate_scenarios(
    policy,
    scenarios: List[Dict[str, Any]],
) -> Dict[str, Any]:

    results = []

    for scenario in scenarios:

        result = policy.evaluate(
            risk_score=scenario["risk_score"],
            risk_level=scenario["risk_level"],
            fallback=scenario["fallback"],
            ambiguous=scenario["ambiguous"],
            missing_behavior=scenario["missing_behavior"],
        )

        checks = {

            "decision_correct": (
                result.decision
                == scenario["expected_decision"]
            ),

            "containment_candidate_correct": (
                result.containment_candidate
                == scenario["expected_candidate"]
            ),

            "containment_never_authorized": (
                result.containment_authorized is False
            ),

            "automatic_response_disabled": (
                result.safety[
                    "automatic_response"
                ] is False
            ),

            "automatic_blocking_disabled": (
                result.safety[
                    "automatic_blocking"
                ] is False
            ),

            "automatic_isolation_disabled": (
                result.safety[
                    "automatic_isolation"
                ] is False
            ),

            "automatic_process_kill_disabled": (
                result.safety[
                    "automatic_process_kill"
                ] is False
            ),

            "response_not_executed": (
                result.safety[
                    "response_actions_executed"
                ] is False
            ),

            "containment_not_executed": (
                result.safety[
                    "containment_actions_executed"
                ] is False
            ),

            "wazuh_disabled": (
                result.safety[
                    "wazuh_integration"
                ] is False
            ),

            "shuffle_disabled": (
                result.safety[
                    "shuffle_integration"
                ] is False
            ),
        }

        passed = all(checks.values())

        results.append(
            {
                "id": scenario["id"],
                "name": scenario["name"],

                "input": {
                    "risk_score":
                        scenario["risk_score"],

                    "risk_level":
                        scenario["risk_level"],

                    "fallback":
                        scenario["fallback"],

                    "ambiguous":
                        scenario["ambiguous"],

                    "missing_behavior":
                        scenario["missing_behavior"],
                },

                "expected": {
                    "decision":
                        scenario["expected_decision"],

                    "containment_candidate":
                        scenario["expected_candidate"],
                },

                "actual":
                    result.to_dict(),

                "checks":
                    checks,

                "passed":
                    passed,
            }
        )

    return {
        "total": len(results),

        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),

        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),

        "results": results,
    }


# =====================================================================
# Fallback safety escalation test
# =====================================================================

def test_fallback_safety(policy) -> Dict[str, Any]:

    results = []

    scores = [
        0.0,
        25.0,
        50.0,
        75.0,
        90.0,
        100.0,
    ]

    for score in scores:

        result = policy.evaluate(
            risk_score=score,
            risk_level="CRITICAL",
            fallback=True,
        )

        checks = {

            "fallback_forces_analyst_review":
                result.decision
                == "ANALYST_REVIEW",

            "fallback_not_containment_candidate":
                result.containment_candidate
                is False,

            "fallback_not_authorized":
                result.containment_authorized
                is False,

            "automatic_response_disabled":
                result.safety[
                    "automatic_response"
                ] is False,

            "automatic_isolation_disabled":
                result.safety[
                    "automatic_isolation"
                ] is False,

            "process_kill_disabled":
                result.safety[
                    "automatic_process_kill"
                ] is False,

        }

        results.append(
            {
                "risk_score": score,
                "actual": result.to_dict(),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    return {
        "total": len(results),
        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),
        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),
        "results": results,
    }


# =====================================================================
# Ambiguity safety escalation test
# =====================================================================

def test_ambiguity_safety(policy) -> Dict[str, Any]:

    results = []

    scores = [
        0.0,
        25.0,
        50.0,
        75.0,
        90.0,
        100.0,
    ]

    for score in scores:

        result = policy.evaluate(
            risk_score=score,
            risk_level="CRITICAL",
            ambiguous=True,
        )

        checks = {

            "ambiguity_forces_analyst_review":
                result.decision
                == "ANALYST_REVIEW",

            "ambiguity_not_containment_candidate":
                result.containment_candidate
                is False,

            "ambiguity_not_authorized":
                result.containment_authorized
                is False,

            "automatic_response_disabled":
                result.safety[
                    "automatic_response"
                ] is False,

            "automatic_isolation_disabled":
                result.safety[
                    "automatic_isolation"
                ] is False,

            "process_kill_disabled":
                result.safety[
                    "automatic_process_kill"
                ] is False,

        }

        results.append(
            {
                "risk_score": score,
                "actual": result.to_dict(),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    return {
        "total": len(results),
        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),
        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),
        "results": results,
    }


# =====================================================================
# Missing behavior safety test
# =====================================================================

def test_missing_behavior(policy) -> Dict[str, Any]:

    scenarios = [
        ("LOW", 15.0),
        ("MEDIUM", 40.0),
        ("HIGH", 65.0),
        ("CRITICAL", 90.0),
    ]

    results = []

    for level, score in scenarios:

        result = policy.evaluate(
            risk_score=score,
            risk_level=level,
            missing_behavior=True,
        )

        checks = {

            "missing_behavior_recorded":
                result.missing_behavior is True,

            "no_automatic_response":
                result.safety[
                    "automatic_response"
                ] is False,

            "no_automatic_isolation":
                result.safety[
                    "automatic_isolation"
                ] is False,

            "no_process_kill":
                result.safety[
                    "automatic_process_kill"
                ] is False,

            "containment_not_authorized":
                result.containment_authorized
                is False,

        }

        results.append(
            {
                "risk_level": level,
                "risk_score": score,
                "actual": result.to_dict(),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    return {
        "total": len(results),
        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),
        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),
        "results": results,
    }


# =====================================================================
# Determinism under uncertainty
# =====================================================================

def test_uncertainty_determinism(policy) -> Dict[str, Any]:

    test_inputs = [
        {
            "risk_score": 95.0,
            "risk_level": "CRITICAL",
            "fallback": True,
            "ambiguous": False,
            "missing_behavior": False,
        },
        {
            "risk_score": 95.0,
            "risk_level": "CRITICAL",
            "fallback": False,
            "ambiguous": True,
            "missing_behavior": False,
        },
        {
            "risk_score": 95.0,
            "risk_level": "CRITICAL",
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": True,
        },
    ]

    results = []

    for index, inputs in enumerate(test_inputs, start=1):

        first = policy.evaluate(**inputs)

        second = policy.evaluate(**inputs)

        results.append(
            {
                "case": index,
                "inputs": inputs,
                "passed": first == second,
                "first": first.to_dict(),
                "second": second.to_dict(),
            }
        )

    return {
        "total": len(results),
        "passed": sum(
            1
            for result in results
            if result["passed"]
        ),
        "failed": sum(
            1
            for result in results
            if not result["passed"]
        ),
        "results": results,
    }


# =====================================================================
# Global side-effect inspection
# =====================================================================

def test_global_safety(policy) -> Dict[str, Any]:

    safety = policy.safety_status()

    checks = {

        "automatic_response":
            safety["automatic_response"] is False,

        "automatic_blocking":
            safety["automatic_blocking"] is False,

        "automatic_isolation":
            safety["automatic_isolation"] is False,

        "automatic_process_kill":
            safety["automatic_process_kill"] is False,

        "response_actions_executed":
            safety["response_actions_executed"] is False,

        "containment_actions_executed":
            safety["containment_actions_executed"] is False,

        "wazuh_integration":
            safety["wazuh_integration"] is False,

        "shuffle_integration":
            safety["shuffle_integration"] is False,

        "containment_requires_analyst":
            safety["containment_requires_analyst"] is True,

        "fallback_cannot_directly_contain":
            safety[
                "fallback_can_directly_contain"
            ] is False,

        "ambiguous_cannot_directly_contain":
            safety[
                "ambiguous_can_directly_contain"
            ] is False,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "safety": safety,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> int:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SmartSOC Phase 5.4")
    print("Uncertainty and Safety Policy Validation")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Policy engine: {ENGINE_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print()

    # -----------------------------------------------------------------
    # Load policy
    # -----------------------------------------------------------------

    try:

        module = load_policy_engine()

        policy = (
            module.SmartSOCDecisionPolicy()
        )

        print(
            "[PASS] Decision policy engine loading"
        )

    except Exception as exc:

        print(
            "[FAIL] Decision policy engine loading"
        )

        print(
            f"       {type(exc).__name__}: {exc}"
        )

        return 1

    print()

    # -----------------------------------------------------------------
    # Uncertainty scenarios
    # -----------------------------------------------------------------

    uncertainty = validate_scenarios(
        policy,
        UNCERTAINTY_SCENARIOS,
    )

    for result in uncertainty["results"]:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{result['id']} — "
            f"{result['name']}"
        )

        if not result["passed"]:

            failed_checks = [
                key
                for key, value
                in result["checks"].items()
                if not value
            ]

            print(
                "       Failed checks: "
                + ", ".join(failed_checks)
            )

    print()

    # -----------------------------------------------------------------
    # Normal scenarios
    # -----------------------------------------------------------------

    normal = validate_scenarios(
        policy,
        NORMAL_SCENARIOS,
    )

    for result in normal["results"]:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{result['id']} — "
            f"{result['name']}"
        )

    print()

    # -----------------------------------------------------------------
    # Fallback safety
    # -----------------------------------------------------------------

    fallback_safety = test_fallback_safety(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if fallback_safety["failed"] == 0
            else "FAIL"
        )
        + "] Fallback safety escalation"
    )

    # -----------------------------------------------------------------
    # Ambiguity safety
    # -----------------------------------------------------------------

    ambiguity_safety = test_ambiguity_safety(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if ambiguity_safety["failed"] == 0
            else "FAIL"
        )
        + "] Ambiguity safety escalation"
    )

    # -----------------------------------------------------------------
    # Missing behavior
    # -----------------------------------------------------------------

    missing_behavior = test_missing_behavior(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if missing_behavior["failed"] == 0
            else "FAIL"
        )
        + "] Missing behavioral evidence safety"
    )

    # -----------------------------------------------------------------
    # Determinism
    # -----------------------------------------------------------------

    determinism = test_uncertainty_determinism(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if determinism["failed"] == 0
            else "FAIL"
        )
        + "] Uncertainty determinism"
    )

    # -----------------------------------------------------------------
    # Global safety
    # -----------------------------------------------------------------

    global_safety = test_global_safety(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if global_safety["passed"]
            else "FAIL"
        )
        + "] Global operational safety"
    )

    # -----------------------------------------------------------------
    # Overall result
    # -----------------------------------------------------------------

    overall_passed = (

        uncertainty["failed"] == 0

        and normal["failed"] == 0

        and fallback_safety["failed"] == 0

        and ambiguity_safety["failed"] == 0

        and missing_behavior["failed"] == 0

        and determinism["failed"] == 0

        and global_safety["passed"]
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    report = {

        "phase": "5.4",

        "phase_name":
            "Uncertainty and Safety Policy Validation",

        "version": VERSION,

        "status": (
            "PASS"
            if overall_passed
            else "FAIL"
        ),

        "uncertainty_scenarios":
            uncertainty,

        "normal_scenarios":
            normal,

        "fallback_safety":
            fallback_safety,

        "ambiguity_safety":
            ambiguity_safety,

        "missing_behavior_safety":
            missing_behavior,

        "uncertainty_determinism":
            determinism,

        "global_safety":
            global_safety,

        "production_safety": {

            "model_modified": False,

            "dataset_modified": False,

            "thresholds_modified": False,

            "wazuh_integration": False,

            "shuffle_integration": False,

            "response_actions_executed": False,

            "containment_actions_executed": False,
        },

        "decision": (
            "ACCEPTED"
            if overall_passed
            else "REVIEW_REQUIRED"
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------

    print()

    print("=" * 72)
    print("PHASE 5.4 FINAL DECISION")
    print("=" * 72)

    print(
        "Status : "
        + (
            "PASS"
            if overall_passed
            else "FAIL"
        )
    )

    print(
        "Uncertainty scenarios: "
        f"{uncertainty['total']} total / "
        f"{uncertainty['failed']} failed"
    )

    print(
        "Normal scenarios: "
        f"{normal['total']} total / "
        f"{normal['failed']} failed"
    )

    print(
        "Fallback safety: "
        f"{fallback_safety['total']} total / "
        f"{fallback_safety['failed']} failed"
    )

    print(
        "Ambiguity safety: "
        f"{ambiguity_safety['total']} total / "
        f"{ambiguity_safety['failed']} failed"
    )

    print(
        "Missing behavior safety: "
        f"{missing_behavior['total']} total / "
        f"{missing_behavior['failed']} failed"
    )

    print(
        "Uncertainty determinism: "
        f"{determinism['total']} total / "
        f"{determinism['failed']} failed"
    )

    print(
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 5.4 ACCEPTED — "
            "UNCERTAINTY AND SAFETY POLICY VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 5.4 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())