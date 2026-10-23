from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 5.3
# Decision Policy Scenario Validation
# Version 4.1
#
# Purpose:
#   Validate the Phase 5.2 Decision Policy Engine against realistic
#   SOC decision scenarios.
#
# Safety:
#   - No model modification
#   - No dataset modification
#   - No threshold modification
#   - No Wazuh integration
#   - No Shuffle integration
#   - No response execution
#   - No containment execution
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
    / "gate5_3"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase5_gate5_3_decision_policy_scenarios_report.json"
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
            f"Unable to create import specification for: "
            f"{ENGINE_FILE}"
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


# =====================================================================
# Scenario definitions
# =====================================================================

SCENARIOS: List[Dict[str, Any]] = [

    {
        "id": "S01",
        "name": "Benign low-risk event",
        "risk_score": 10.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_analyst": False,
        "expected_containment_candidate": False,
    },

    {
        "id": "S02",
        "name": "Low-risk event with missing behavior",
        "risk_score": 20.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "MONITOR",
        "expected_analyst": False,
        "expected_containment_candidate": False,
    },

    {
        "id": "S03",
        "name": "Suspicious PowerShell medium risk",
        "risk_score": 40.0,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S04",
        "name": "Encoded PowerShell high risk",
        "risk_score": 60.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S05",
        "name": "Download behavior high risk",
        "risk_score": 70.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S06",
        "name": "Critical persistence candidate",
        "risk_score": 80.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_analyst": True,
        "expected_containment_candidate": True,
    },

    {
        "id": "S07",
        "name": "Critical destructive behavior",
        "risk_score": 95.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_analyst": True,
        "expected_containment_candidate": True,
    },

    {
        "id": "S08",
        "name": "Fallback despite critical risk",
        "risk_score": 95.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S09",
        "name": "Ambiguous critical event",
        "risk_score": 95.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S10",
        "name": "Fallback high-risk event",
        "risk_score": 65.0,
        "risk_level": "HIGH",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S11",
        "name": "Ambiguous high-risk event",
        "risk_score": 65.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S12",
        "name": "Medium risk missing behavior",
        "risk_score": 35.0,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S13",
        "name": "Critical missing behavior",
        "risk_score": 85.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_analyst": True,
        "expected_containment_candidate": True,
    },

    {
        "id": "S14",
        "name": "Lowest valid risk score",
        "risk_score": 0.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_analyst": False,
        "expected_containment_candidate": False,
    },

    {
        "id": "S15",
        "name": "Highest valid risk score",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_analyst": True,
        "expected_containment_candidate": True,
    },

    {
        "id": "S16",
        "name": "Low risk ambiguous",
        "risk_score": 20.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S17",
        "name": "Low risk fallback",
        "risk_score": 20.0,
        "risk_level": "LOW",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S18",
        "name": "High risk with missing behavior",
        "risk_score": 74.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S19",
        "name": "Critical fallback plus missing behavior",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },

    {
        "id": "S20",
        "name": "Critical ambiguity plus missing behavior",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_analyst": True,
        "expected_containment_candidate": False,
    },
]


# =====================================================================
# Scenario validation
# =====================================================================

def run_scenario_tests(policy) -> Dict[str, Any]:

    results: List[Dict[str, Any]] = []

    for scenario in SCENARIOS:

        result = policy.evaluate(
            risk_score=scenario["risk_score"],
            risk_level=scenario["risk_level"],
            fallback=scenario["fallback"],
            ambiguous=scenario["ambiguous"],
            missing_behavior=scenario["missing_behavior"],
        )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # DecisionPolicyResult does NOT expose automatic_response as
        # a direct attribute.
        #
        # Safety flags are stored in:
        #
        #     result.safety["automatic_response"]
        #
        # Therefore all safety checks below correctly use result.safety.
        # -------------------------------------------------------------

        checks = {

            "decision": (
                result.decision
                == scenario["expected_decision"]
            ),

            "analyst_required": (
                result.analyst_required
                == scenario["expected_analyst"]
            ),

            "containment_candidate": (
                result.containment_candidate
                == scenario[
                    "expected_containment_candidate"
                ]
            ),

            "containment_not_authorized": (
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

            "response_actions_not_executed": (
                result.safety[
                    "response_actions_executed"
                ] is False
            ),

            "containment_actions_not_executed": (
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
                    "risk_score": scenario[
                        "risk_score"
                    ],
                    "risk_level": scenario[
                        "risk_level"
                    ],
                    "fallback": scenario[
                        "fallback"
                    ],
                    "ambiguous": scenario[
                        "ambiguous"
                    ],
                    "missing_behavior": scenario[
                        "missing_behavior"
                    ],
                },

                "expected": {
                    "decision": scenario[
                        "expected_decision"
                    ],
                    "analyst_required": scenario[
                        "expected_analyst"
                    ],
                    "containment_candidate": scenario[
                        "expected_containment_candidate"
                    ],
                },

                "actual": result.to_dict(),

                "checks": checks,

                "passed": passed,
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
# Determinism validation
# =====================================================================

def test_determinism(policy) -> Dict[str, Any]:

    inputs = {
        "risk_score": 88.6,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
    }

    first = policy.evaluate(**inputs)

    second = policy.evaluate(**inputs)

    return {
        "passed": first == second,
        "first": first.to_dict(),
        "second": second.to_dict(),
    }


# =====================================================================
# Monotonicity validation
# =====================================================================

def test_monotonicity(policy) -> Dict[str, Any]:

    ordered_levels = [
        ("LOW", 10.0),
        ("MEDIUM", 35.0),
        ("HIGH", 60.0),
        ("CRITICAL", 90.0),
    ]

    decision_order = {
        "MONITOR": 0,
        "ANALYST_REVIEW": 1,
        "PRIORITY_ANALYST_REVIEW": 2,
        "CONTAINMENT_CANDIDATE": 3,
    }

    decisions: List[str] = []

    for level, score in ordered_levels:

        result = policy.evaluate(
            risk_score=score,
            risk_level=level,
        )

        decisions.append(
            result.decision
        )

    numeric = [
        decision_order[decision]
        for decision in decisions
    ]

    passed = (
        numeric == sorted(numeric)
    )

    return {
        "passed": passed,

        "levels": [
            level
            for level, _ in ordered_levels
        ],

        "decisions": decisions,

        "numeric_order": numeric,
    }


# =====================================================================
# Global safety validation
# =====================================================================

def test_global_safety(policy) -> Dict[str, Any]:

    safety = policy.safety_status()

    checks: Dict[str, bool] = {}

    required_false = [
        "automatic_response",
        "automatic_blocking",
        "automatic_isolation",
        "automatic_process_kill",
        "response_actions_executed",
        "containment_actions_executed",
        "wazuh_integration",
        "shuffle_integration",
    ]

    for key in required_false:

        checks[key] = (
            safety[key] is False
        )

    checks["containment_requires_analyst"] = (
        safety[
            "containment_requires_analyst"
        ]
        is True
    )

    checks["fallback_cannot_directly_contain"] = (
        safety[
            "fallback_can_directly_contain"
        ]
        is False
    )

    checks["ambiguous_cannot_directly_contain"] = (
        safety[
            "ambiguous_can_directly_contain"
        ]
        is False
    )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "safety": safety,
    }


# =====================================================================
# Invalid input validation
# =====================================================================

def test_invalid_inputs(policy) -> Dict[str, Any]:

    invalid_level_rejected = False

    try:

        policy.evaluate(
            risk_score=50,
            risk_level="INVALID",
        )

    except ValueError:

        invalid_level_rejected = True

    invalid_score_low_rejected = False

    try:

        policy.evaluate(
            risk_score=-1,
            risk_level="LOW",
        )

    except ValueError:

        invalid_score_low_rejected = True

    invalid_score_high_rejected = False

    try:

        policy.evaluate(
            risk_score=101,
            risk_level="CRITICAL",
        )

    except ValueError:

        invalid_score_high_rejected = True

    checks = {
        "invalid_level_rejected":
            invalid_level_rejected,

        "negative_score_rejected":
            invalid_score_low_rejected,

        "score_above_100_rejected":
            invalid_score_high_rejected,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


# =====================================================================
# Additional containment safety validation
# =====================================================================

def test_containment_safety(policy) -> Dict[str, Any]:

    scenarios = [

        {
            "name": "Critical normal",
            "risk_score": 95,
            "risk_level": "CRITICAL",
            "fallback": False,
            "ambiguous": False,
            "expected_candidate": True,
        },

        {
            "name": "Critical fallback",
            "risk_score": 95,
            "risk_level": "CRITICAL",
            "fallback": True,
            "ambiguous": False,
            "expected_candidate": False,
        },

        {
            "name": "Critical ambiguous",
            "risk_score": 95,
            "risk_level": "CRITICAL",
            "fallback": False,
            "ambiguous": True,
            "expected_candidate": False,
        },

    ]

    results = []

    for scenario in scenarios:

        result = policy.evaluate(
            risk_score=scenario["risk_score"],
            risk_level=scenario["risk_level"],
            fallback=scenario["fallback"],
            ambiguous=scenario["ambiguous"],
        )

        checks = {
            "expected_candidate":
                result.containment_candidate
                == scenario["expected_candidate"],

            "never_authorized":
                result.containment_authorized is False,

            "analyst_required":
                result.analyst_required is True,

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
                "name": scenario["name"],
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
# Main
# =====================================================================

def main() -> int:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SmartSOC Phase 5.3")
    print("Decision Policy Scenario Validation")
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
    # Load policy engine
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
    # Scenario tests
    # -----------------------------------------------------------------

    scenarios = run_scenario_tests(
        policy
    )

    for result in scenarios["results"]:

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

            print(
                "       Expected: "
                f"{result['expected']['decision']}"
            )

            print(
                "       Actual:   "
                f"{result['actual']['decision']}"
            )

            failed_checks = [
                key
                for key, value
                in result["checks"].items()
                if not value
            ]

            print(
                "       Failed checks: "
                f"{', '.join(failed_checks)}"
            )

    print()

    # -----------------------------------------------------------------
    # Determinism
    # -----------------------------------------------------------------

    determinism = test_determinism(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if determinism["passed"]
            else "FAIL"
        )
        + "] Deterministic repeated evaluation"
    )

    # -----------------------------------------------------------------
    # Monotonicity
    # -----------------------------------------------------------------

    monotonicity = test_monotonicity(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if monotonicity["passed"]
            else "FAIL"
        )
        + "] Risk disposition monotonicity"
    )

    # -----------------------------------------------------------------
    # Global safety
    # -----------------------------------------------------------------

    safety = test_global_safety(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if safety["passed"]
            else "FAIL"
        )
        + "] Global operational safety"
    )

    # -----------------------------------------------------------------
    # Invalid input rejection
    # -----------------------------------------------------------------

    invalid_inputs = test_invalid_inputs(
        policy
    )

    print(
        "["
        + (
            "PASS"
            if invalid_inputs["passed"]
            else "FAIL"
        )
        + "] Invalid input rejection"
    )

    # -----------------------------------------------------------------
    # Containment safety
    # -----------------------------------------------------------------

    containment_safety = (
        test_containment_safety(policy)
    )

    print(
        "["
        + (
            "PASS"
            if containment_safety["passed"]
            == containment_safety["total"]
            else "FAIL"
        )
        + "] Containment safety boundary"
    )

    # -----------------------------------------------------------------
    # Overall result
    # -----------------------------------------------------------------

    overall_passed = (

        scenarios["failed"] == 0

        and determinism["passed"]

        and monotonicity["passed"]

        and safety["passed"]

        and invalid_inputs["passed"]

        and (
            containment_safety["failed"] == 0
        )
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    report = {

        "phase": "5.3",

        "phase_name":
            "Decision Policy Scenario Validation",

        "version": VERSION,

        "status": (
            "PASS"
            if overall_passed
            else "FAIL"
        ),

        "scenario_tests": scenarios,

        "determinism": determinism,

        "monotonicity": monotonicity,

        "safety": safety,

        "invalid_inputs": invalid_inputs,

        "containment_safety":
            containment_safety,

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
    # Final console output
    # -----------------------------------------------------------------

    print()

    print("=" * 72)
    print("PHASE 5.3 FINAL DECISION")
    print("=" * 72)

    print(
        "Status   : "
        + (
            "PASS"
            if overall_passed
            else "FAIL"
        )
    )

    print(
        "Scenarios: "
        f"{scenarios['total']} total / "
        f"{scenarios['failed']} failed"
    )

    print(
        "Containment safety: "
        f"{containment_safety['total']} total / "
        f"{containment_safety['failed']} failed"
    )

    print(
        f"Report   : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 5.3 ACCEPTED — "
            "DECISION POLICY SCENARIOS VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 5.3 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())