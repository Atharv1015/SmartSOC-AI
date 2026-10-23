from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 5.5
# Adversarial Decision Policy Validation
# Version 4.1
#
# Purpose:
#   Attempt to break the Phase 5.2 Decision Policy using malformed,
#   contradictory, extreme, and uncertainty-heavy inputs.
#
# Safety:
#   This script is completely side-effect free.
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
    / "gate5_5"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase5_gate5_5_adversarial_decision_policy_report.json"
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
# Helper
# =====================================================================

def check_safety(result) -> Dict[str, bool]:

    return {
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


# =====================================================================
# Adversarial scenarios
# =====================================================================

ADVERSARIAL_SCENARIOS: List[Dict[str, Any]] = [

    # -----------------------------------------------------------------
    # Extreme critical cases
    # -----------------------------------------------------------------

    {
        "id": "A01",
        "name": "Maximum risk critical",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

    {
        "id": "A02",
        "name": "Maximum risk critical fallback",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A03",
        "name": "Maximum risk critical ambiguous",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A04",
        "name": "Maximum risk critical fallback ambiguous",
        "risk_score": 100.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    # -----------------------------------------------------------------
    # Zero / minimum risk cases
    # -----------------------------------------------------------------

    {
        "id": "A05",
        "name": "Zero risk low",
        "risk_score": 0.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_candidate": False,
    },

    {
        "id": "A06",
        "name": "Zero risk fallback",
        "risk_score": 0.0,
        "risk_level": "LOW",
        "fallback": True,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A07",
        "name": "Zero risk ambiguous",
        "risk_score": 0.0,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": True,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    # -----------------------------------------------------------------
    # Contradictory / unusual combinations
    # -----------------------------------------------------------------

    {
        "id": "A08",
        "name": "Low risk with all uncertainty flags",
        "risk_score": 15.0,
        "risk_level": "LOW",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A09",
        "name": "Medium risk with all uncertainty flags",
        "risk_score": 45.0,
        "risk_level": "MEDIUM",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A10",
        "name": "High risk with all uncertainty flags",
        "risk_score": 70.0,
        "risk_level": "HIGH",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A11",
        "name": "Critical risk with all uncertainty flags",
        "risk_score": 95.0,
        "risk_level": "CRITICAL",
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    # -----------------------------------------------------------------
    # Boundary scores
    # -----------------------------------------------------------------

    {
        "id": "A12",
        "name": "Just below medium boundary",
        "risk_score": 24.999999,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_candidate": False,
    },

    {
        "id": "A13",
        "name": "Medium boundary",
        "risk_score": 25.0,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A14",
        "name": "Just below high boundary",
        "risk_score": 49.999999,
        "risk_level": "MEDIUM",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A15",
        "name": "High boundary",
        "risk_score": 50.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A16",
        "name": "Just below critical boundary",
        "risk_score": 74.999999,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A17",
        "name": "Critical boundary",
        "risk_score": 75.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

    # -----------------------------------------------------------------
    # Type coercion / normalization
    # -----------------------------------------------------------------

    {
        "id": "A18",
        "name": "Lowercase critical risk level",
        "risk_score": 90.0,
        "risk_level": "critical",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

    {
        "id": "A19",
        "name": "Whitespace around risk level",
        "risk_score": 60.0,
        "risk_level": "  HIGH  ",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_candidate": False,
    },

    # -----------------------------------------------------------------
    # Missing behavior combinations
    # -----------------------------------------------------------------

    {
        "id": "A20",
        "name": "Critical missing behavior only",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

    {
        "id": "A21",
        "name": "High missing behavior only",
        "risk_score": 70.0,
        "risk_level": "HIGH",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": True,
        "expected_decision": "PRIORITY_ANALYST_REVIEW",
        "expected_candidate": False,
    },

    # -----------------------------------------------------------------
    # Boolean-like values
    # -----------------------------------------------------------------

    {
        "id": "A22",
        "name": "Integer truthy fallback",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": 1,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A23",
        "name": "Integer truthy ambiguity",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": 1,
        "missing_behavior": False,
        "expected_decision": "ANALYST_REVIEW",
        "expected_candidate": False,
    },

    {
        "id": "A24",
        "name": "Integer truthy missing behavior",
        "risk_score": 90.0,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": 1,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

    # -----------------------------------------------------------------
    # Very small / very large floating precision values
    # -----------------------------------------------------------------

    {
        "id": "A25",
        "name": "Very small low risk",
        "risk_score": 0.000001,
        "risk_level": "LOW",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "MONITOR",
        "expected_candidate": False,
    },

    {
        "id": "A26",
        "name": "High precision critical",
        "risk_score": 99.999999,
        "risk_level": "CRITICAL",
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "expected_decision": "CONTAINMENT_CANDIDATE",
        "expected_candidate": True,
    },

]


# =====================================================================
# Adversarial execution
# =====================================================================

def run_adversarial_tests(policy) -> Dict[str, Any]:

    results = []

    for scenario in ADVERSARIAL_SCENARIOS:

        result = policy.evaluate(
            risk_score=scenario["risk_score"],
            risk_level=scenario["risk_level"],
            fallback=scenario["fallback"],
            ambiguous=scenario["ambiguous"],
            missing_behavior=scenario["missing_behavior"],
        )

        safety = check_safety(result)

        checks = {

            "decision_correct": (
                result.decision
                == scenario["expected_decision"]
            ),

            "candidate_correct": (
                result.containment_candidate
                == scenario["expected_candidate"]
            ),

            **safety,
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
# Invalid / malicious input rejection
# =====================================================================

def test_invalid_inputs(policy) -> Dict[str, Any]:

    tests = []

    def expect_rejection(
        name: str,
        callback,
    ) -> None:

        rejected = False
        error_type = None

        try:
            callback()

        except Exception as exc:

            rejected = True
            error_type = type(exc).__name__

        tests.append(
            {
                "name": name,
                "rejected": rejected,
                "error_type": error_type,
            }
        )

    # Invalid risk level
    expect_rejection(
        "invalid risk level",
        lambda: policy.evaluate(
            risk_score=50,
            risk_level="UNKNOWN",
        ),
    )

    # Negative score
    expect_rejection(
        "negative risk score",
        lambda: policy.evaluate(
            risk_score=-0.000001,
            risk_level="LOW",
        ),
    )

    # Score greater than 100
    expect_rejection(
        "risk score above 100",
        lambda: policy.evaluate(
            risk_score=100.000001,
            risk_level="CRITICAL",
        ),
    )

    # NaN
    expect_rejection(
        "NaN risk score",
        lambda: policy.evaluate(
            risk_score=float("nan"),
            risk_level="CRITICAL",
        ),
    )

    # Positive infinity
    expect_rejection(
        "positive infinity risk score",
        lambda: policy.evaluate(
            risk_score=float("inf"),
            risk_level="CRITICAL",
        ),
    )

    # Negative infinity
    expect_rejection(
        "negative infinity risk score",
        lambda: policy.evaluate(
            risk_score=float("-inf"),
            risk_level="LOW",
        ),
    )

    passed = all(
        test["rejected"]
        for test in tests
    )

    return {
        "total": len(tests),
        "passed": sum(
            1
            for test in tests
            if test["rejected"]
        ),
        "failed": sum(
            1
            for test in tests
            if not test["rejected"]
        ),
        "results": tests,
        "passed": passed,
    }


# =====================================================================
# Determinism under adversarial inputs
# =====================================================================

def test_determinism(policy) -> Dict[str, Any]:

    cases = [
        {
            "risk_score": 100.0,
            "risk_level": "CRITICAL",
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": True,
        },
        {
            "risk_score": 75.0,
            "risk_level": "CRITICAL",
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },
        {
            "risk_score": 0.0,
            "risk_level": "LOW",
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": True,
        },
    ]

    results = []

    for index, inputs in enumerate(
        cases,
        start=1,
    ):

        first = policy.evaluate(
            **inputs
        )

        second = policy.evaluate(
            **inputs
        )

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
# Global safety inspection
# =====================================================================

def test_global_safety(policy) -> Dict[str, Any]:

    safety = policy.safety_status()

    checks = {

        "automatic_response_disabled":
            safety[
                "automatic_response"
            ] is False,

        "automatic_blocking_disabled":
            safety[
                "automatic_blocking"
            ] is False,

        "automatic_isolation_disabled":
            safety[
                "automatic_isolation"
            ] is False,

        "automatic_process_kill_disabled":
            safety[
                "automatic_process_kill"
            ] is False,

        "response_actions_not_executed":
            safety[
                "response_actions_executed"
            ] is False,

        "containment_actions_not_executed":
            safety[
                "containment_actions_executed"
            ] is False,

        "wazuh_integration_disabled":
            safety[
                "wazuh_integration"
            ] is False,

        "shuffle_integration_disabled":
            safety[
                "shuffle_integration"
            ] is False,

        "containment_requires_analyst":
            safety[
                "containment_requires_analyst"
            ] is True,

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
    print("SmartSOC Phase 5.5")
    print("Adversarial Decision Policy Validation")
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
    # Load engine
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
    # Adversarial scenarios
    # -----------------------------------------------------------------

    adversarial = run_adversarial_tests(
        policy
    )

    for result in adversarial["results"]:

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

            print(
                "       Expected: "
                f"{result['expected']['decision']}"
            )

            print(
                "       Actual: "
                f"{result['actual']['decision']}"
            )

    print()

    # -----------------------------------------------------------------
    # Invalid inputs
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
        + "] Invalid/adversarial input rejection"
    )

    if not invalid_inputs["passed"]:

        for result in invalid_inputs["results"]:

            if not result["rejected"]:

                print(
                    "       NOT REJECTED: "
                    + result["name"]
                )

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
            if determinism["failed"] == 0
            else "FAIL"
        )
        + "] Adversarial determinism"
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
    # Overall
    # -----------------------------------------------------------------

    overall_passed = (

        adversarial["failed"] == 0

        and invalid_inputs["failed"] == 0

        and determinism["failed"] == 0

        and safety["passed"]
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    report = {

        "phase": "5.5",

        "phase_name":
            "Adversarial Decision Policy Validation",

        "version": VERSION,

        "status": (
            "PASS"
            if overall_passed
            else "FAIL"
        ),

        "adversarial_scenarios":
            adversarial,

        "invalid_input_rejection":
            invalid_inputs,

        "determinism":
            determinism,

        "global_safety":
            safety,

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
    print("PHASE 5.5 FINAL DECISION")
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
        "Adversarial scenarios: "
        f"{adversarial['total']} total / "
        f"{adversarial['failed']} failed"
    )

    print(
        "Invalid inputs: "
        f"{invalid_inputs['total']} total / "
        f"{invalid_inputs['failed']} failed"
    )

    print(
        "Determinism: "
        f"{determinism['total']} total / "
        f"{determinism['failed']} failed"
    )

    print(
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 5.5 ACCEPTED — "
            "ADVERSARIAL DECISION POLICY VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 5.5 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())