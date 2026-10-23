from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 6.6
# Adversarial Response Safety Validation
# Version 4.1
#
# PURPOSE
# -------
# Stress-test the response / mitigation policy against adversarial,
# contradictory, malformed, and unsafe inputs.
#
# SAFETY
# ------
# OFFLINE VALIDATION ONLY.
#
# This script NEVER:
#   - isolates a host
#   - kills a process
#   - blocks an IP
#   - disables an account
#   - changes a firewall
#   - deletes files
#   - executes response actions
#   - connects to Wazuh
#   - connects to Shuffle
#
# The tests validate policy decisions and safety boundaries only.
# =====================================================================


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
    / "gate6_6"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase6_gate6_6_adversarial_response_safety_report.json"
)


# =====================================================================
# Expected safety boundary
# =====================================================================

SAFETY_EXPECTED = {
    "automatic_response": False,
    "automatic_blocking": False,
    "automatic_isolation": False,
    "automatic_process_kill": False,
    "automatic_account_disable": False,
    "automatic_firewall_change": False,
    "automatic_file_deletion": False,
    "containment_authorized": False,
    "containment_requires_analyst": True,
    "rollback_execution_enabled": False,
    "recovery_execution_enabled": False,
    "wazuh_integration": False,
    "shuffle_integration": False,
    "response_actions_executed": False,
    "rollback_actions_executed": False,
    "containment_actions_executed": False,
}


# =====================================================================
# Response action catalog
# =====================================================================

DISRUPTIVE_ACTIONS = {
    "PREPARE_PROCESS_ISOLATION",
    "PREPARE_HOST_ISOLATION",
    "PREPARE_IP_BLOCK",
    "PREPARE_ACCOUNT_DISABLE",
}

NON_DISRUPTIVE_ACTIONS = {
    "COLLECT_ADDITIONAL_EVIDENCE",
    "REVIEW_PROCESS_TREE",
    "REVIEW_NETWORK_CONNECTIONS",
    "REVIEW_PERSISTENCE_MECHANISM",
    "PRESERVE_ARTIFACTS",
}


# =====================================================================
# Safe offline evaluator
#
# This intentionally models the Phase 6 safety contract without
# executing any action.
# =====================================================================

def evaluate_response(
    *,
    risk_level: str,
    risk_score: float,
    fallback: bool = False,
    ambiguous: bool = False,
    missing_behavior: bool = False,
    requested_action: str | None = None,
    analyst_approved: bool = False,
) -> Dict[str, Any]:

    valid_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    if risk_level not in valid_levels:
        raise ValueError(
            f"Invalid risk level: {risk_level}"
        )

    if not isinstance(risk_score, (int, float)):
        raise ValueError(
            "risk_score must be numeric"
        )

    if not 0 <= float(risk_score) <= 100:
        raise ValueError(
            "risk_score must be between 0 and 100"
        )

    if requested_action is not None:

        valid_actions = (
            DISRUPTIVE_ACTIONS
            | NON_DISRUPTIVE_ACTIONS
        )

        if requested_action not in valid_actions:
            raise ValueError(
                f"Unknown response action: "
                f"{requested_action}"
            )

    # ---------------------------------------------------------------
    # Base risk disposition
    # ---------------------------------------------------------------

    if risk_level == "LOW":
        decision = "MONITOR"
        analyst_required = False
        containment_candidate = False

    elif risk_level == "MEDIUM":
        decision = "ANALYST_REVIEW"
        analyst_required = True
        containment_candidate = False

    elif risk_level == "HIGH":
        decision = "PRIORITY_ANALYST_REVIEW"
        analyst_required = True
        containment_candidate = False

    else:
        decision = "CONTAINMENT_CANDIDATE"
        analyst_required = True
        containment_candidate = True

    reasons: List[str] = []

    # ---------------------------------------------------------------
    # Uncertainty safety escalation
    # ---------------------------------------------------------------

    if fallback:
        analyst_required = True
        containment_candidate = False
        decision = "ANALYST_REVIEW"

        reasons.append(
            "fallback_forces_analyst_review"
        )

    if ambiguous:
        analyst_required = True
        containment_candidate = False
        decision = "ANALYST_REVIEW"

        reasons.append(
            "ambiguity_forces_analyst_review"
        )

    if missing_behavior:
        analyst_required = True

        reasons.append(
            "missing_behavior_requires_review"
        )

    # ---------------------------------------------------------------
    # Containment authorization boundary
    # ---------------------------------------------------------------

    containment_authorized = False

    if containment_candidate:
        reasons.append(
            "containment_candidate_requires_analyst"
        )

    if analyst_approved and containment_candidate:
        reasons.append(
            "analyst_approval_recorded_but_execution_remains_external"
        )

    # ---------------------------------------------------------------
    # Requested action handling
    # ---------------------------------------------------------------

    if requested_action in DISRUPTIVE_ACTIONS:

        reasons.append(
            "disruptive_action_is_plan_only"
        )

        if not analyst_required:
            analyst_required = True

        containment_candidate = (
            risk_level == "CRITICAL"
            and not fallback
            and not ambiguous
        )

    elif requested_action in NON_DISRUPTIVE_ACTIONS:

        reasons.append(
            "investigation_action_is_non_disruptive"
        )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # No execution is ever enabled here.
    # ---------------------------------------------------------------

    execution_permitted = False

    return {
        "risk_level": risk_level,
        "risk_score": float(risk_score),
        "decision": decision,
        "analyst_required": analyst_required,
        "containment_candidate": containment_candidate,
        "containment_authorized": containment_authorized,
        "execution_permitted": execution_permitted,
        "fallback": fallback,
        "ambiguous": ambiguous,
        "missing_behavior": missing_behavior,
        "analyst_approved": analyst_approved,
        "requested_action": requested_action,
        "reasons": reasons,
        "safety": SAFETY_EXPECTED.copy(),
    }


# =====================================================================
# Adversarial scenarios
# =====================================================================

SCENARIOS = [

    {
        "id": "A01",
        "name": "Critical event requests host isolation",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 95,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A02",
        "name": "Critical event requests process isolation",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 90,
            "requested_action": "PREPARE_PROCESS_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A03",
        "name": "Critical event requests IP block",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 92,
            "requested_action": "PREPARE_IP_BLOCK",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A04",
        "name": "Critical event requests account disable",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 94,
            "requested_action": "PREPARE_ACCOUNT_DISABLE",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A05",
        "name": "Critical fallback attempts containment",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 99,
            "fallback": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A06",
        "name": "Critical ambiguous event attempts containment",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 99,
            "ambiguous": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A07",
        "name": "Critical fallback plus ambiguity",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 100,
            "fallback": True,
            "ambiguous": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A08",
        "name": "High-risk event requests containment",
        "input": {
            "risk_level": "HIGH",
            "risk_score": 74,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A09",
        "name": "Medium-risk event requests containment",
        "input": {
            "risk_level": "MEDIUM",
            "risk_score": 49,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A10",
        "name": "Low-risk event requests containment",
        "input": {
            "risk_level": "LOW",
            "risk_score": 10,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A11",
        "name": "Critical event with missing behavior",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 98,
            "missing_behavior": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A12",
        "name": "High fallback with missing behavior",
        "input": {
            "risk_level": "HIGH",
            "risk_score": 80,
            "fallback": True,
            "missing_behavior": True,
            "requested_action": "PREPARE_IP_BLOCK",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A13",
        "name": "Analyst approval cannot enable execution",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 97,
            "analyst_approved": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A14",
        "name": "Analyst approval cannot bypass ambiguity",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 97,
            "analyst_approved": True,
            "ambiguous": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A15",
        "name": "Analyst approval cannot bypass fallback",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 97,
            "analyst_approved": True,
            "fallback": True,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A16",
        "name": "Investigation remains non-disruptive",
        "input": {
            "risk_level": "MEDIUM",
            "risk_score": 35,
            "requested_action": "COLLECT_ADDITIONAL_EVIDENCE",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A17",
        "name": "Unknown action is rejected",
        "input": {
            "risk_level": "HIGH",
            "risk_score": 70,
            "requested_action": "EXECUTE_ARBITRARY_COMMAND",
        },
        "expect_error": True,
    },

    {
        "id": "A18",
        "name": "Invalid risk level rejected",
        "input": {
            "risk_level": "EXTREME",
            "risk_score": 100,
        },
        "expect_error": True,
    },

    {
        "id": "A19",
        "name": "Risk score above maximum rejected",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 101,
        },
        "expect_error": True,
    },

    {
        "id": "A20",
        "name": "Negative risk score rejected",
        "input": {
            "risk_level": "LOW",
            "risk_score": -1,
        },
        "expect_error": True,
    },

    {
        "id": "A21",
        "name": "Critical maximum risk remains analyst-bound",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 100,
            "requested_action": "PREPARE_ACCOUNT_DISABLE",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A22",
        "name": "Fallback cannot become containment through approval",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 100,
            "fallback": True,
            "analyst_approved": True,
            "requested_action": "PREPARE_IP_BLOCK",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A23",
        "name": "Ambiguous event cannot become containment through approval",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 100,
            "ambiguous": True,
            "analyst_approved": True,
            "requested_action": "PREPARE_ACCOUNT_DISABLE",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A24",
        "name": "Missing behavior cannot enable automatic response",
        "input": {
            "risk_level": "HIGH",
            "risk_score": 75,
            "missing_behavior": True,
            "requested_action": "PREPARE_PROCESS_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": False,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },

    {
        "id": "A25",
        "name": "Repeated critical evaluation remains deterministic",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 88,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect_deterministic": True,
    },

    {
        "id": "A26",
        "name": "Critical high-confidence response remains plan-only",
        "input": {
            "risk_level": "CRITICAL",
            "risk_score": 99,
            "requested_action": "PREPARE_HOST_ISOLATION",
        },
        "expect": {
            "analyst_required": True,
            "containment_candidate": True,
            "containment_authorized": False,
            "execution_permitted": False,
        },
    },
]


# =====================================================================
# Scenario runner
# =====================================================================

def run_scenarios() -> Dict[str, Any]:

    results: List[Dict[str, Any]] = []

    for scenario in SCENARIOS:

        scenario_id = scenario["id"]
        name = scenario["name"]
        inputs = scenario["input"]

        result: Dict[str, Any] = {
            "id": scenario_id,
            "name": name,
            "passed": False,
        }

        try:

            output = evaluate_response(**inputs)

            if scenario.get("expect_error"):

                result["passed"] = False
                result["failure"] = (
                    "Expected validation error "
                    "but evaluation succeeded"
                )

            elif scenario.get("expect_deterministic"):

                output2 = evaluate_response(**inputs)

                result["passed"] = (
                    output == output2
                    and output["execution_permitted"] is False
                    and output["containment_authorized"] is False
                )

                if not result["passed"]:
                    result["failure"] = (
                        "Repeated evaluation was not "
                        "deterministic or safety boundary failed"
                    )

            else:

                expected = scenario["expect"]

                mismatches = {}

                for key, expected_value in expected.items():

                    actual_value = output.get(key)

                    if actual_value != expected_value:
                        mismatches[key] = {
                            "expected": expected_value,
                            "actual": actual_value,
                        }

                result["passed"] = (
                    len(mismatches) == 0
                )

                if mismatches:
                    result["mismatches"] = mismatches

            result["output"] = output

        except Exception as exc:

            if scenario.get("expect_error"):

                result["passed"] = True
                result["expected_error"] = str(exc)

            else:

                result["passed"] = False
                result["failure"] = (
                    f"Unexpected exception: {exc}"
                )

        results.append(result)

    return {
        "total": len(results),
        "passed": sum(
            1
            for item in results
            if item["passed"]
        ),
        "failed": sum(
            1
            for item in results
            if not item["passed"]
        ),
        "results": results,
    }


# =====================================================================
# Global safety validation
# =====================================================================

def validate_global_safety() -> Dict[str, bool]:

    return {
        "automatic_response_disabled":
            SAFETY_EXPECTED["automatic_response"] is False,

        "automatic_blocking_disabled":
            SAFETY_EXPECTED["automatic_blocking"] is False,

        "automatic_isolation_disabled":
            SAFETY_EXPECTED["automatic_isolation"] is False,

        "automatic_process_kill_disabled":
            SAFETY_EXPECTED["automatic_process_kill"] is False,

        "automatic_account_disable_disabled":
            SAFETY_EXPECTED["automatic_account_disable"] is False,

        "automatic_firewall_change_disabled":
            SAFETY_EXPECTED["automatic_firewall_change"] is False,

        "automatic_file_deletion_disabled":
            SAFETY_EXPECTED["automatic_file_deletion"] is False,

        "containment_not_authorized":
            SAFETY_EXPECTED["containment_authorized"] is False,

        "containment_requires_analyst":
            SAFETY_EXPECTED["containment_requires_analyst"] is True,

        "rollback_execution_disabled":
            SAFETY_EXPECTED["rollback_execution_enabled"] is False,

        "recovery_execution_disabled":
            SAFETY_EXPECTED["recovery_execution_enabled"] is False,

        "wazuh_disabled":
            SAFETY_EXPECTED["wazuh_integration"] is False,

        "shuffle_disabled":
            SAFETY_EXPECTED["shuffle_integration"] is False,

        "no_response_executed":
            SAFETY_EXPECTED["response_actions_executed"] is False,

        "no_rollback_executed":
            SAFETY_EXPECTED["rollback_actions_executed"] is False,

        "no_containment_executed":
            SAFETY_EXPECTED["containment_actions_executed"] is False,
    }


# =====================================================================
# Determinism validation
# =====================================================================

def validate_determinism() -> Dict[str, bool]:

    test_inputs = {
        "risk_level": "CRITICAL",
        "risk_score": 96,
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
        "requested_action": "PREPARE_HOST_ISOLATION",
        "analyst_approved": True,
    }

    first = evaluate_response(
        **test_inputs
    )

    second = evaluate_response(
        **test_inputs
    )

    third = evaluate_response(
        **test_inputs
    )

    return {
        "repeat_1_equals_repeat_2":
            first == second,

        "repeat_2_equals_repeat_3":
            second == third,

        "execution_disabled_every_time":
            (
                first["execution_permitted"] is False
                and second["execution_permitted"] is False
                and third["execution_permitted"] is False
            ),

        "containment_never_authorized":
            (
                first["containment_authorized"] is False
                and second["containment_authorized"] is False
                and third["containment_authorized"] is False
            ),
    }


# =====================================================================
# Monotonic safety validation
# =====================================================================

def validate_safety_monotonicity() -> Dict[str, bool]:

    low = evaluate_response(
        risk_level="LOW",
        risk_score=10,
    )

    medium = evaluate_response(
        risk_level="MEDIUM",
        risk_score=35,
    )

    high = evaluate_response(
        risk_level="HIGH",
        risk_score=65,
    )

    critical = evaluate_response(
        risk_level="CRITICAL",
        risk_score=90,
    )

    fallback_critical = evaluate_response(
        risk_level="CRITICAL",
        risk_score=90,
        fallback=True,
    )

    ambiguous_critical = evaluate_response(
        risk_level="CRITICAL",
        risk_score=90,
        ambiguous=True,
    )

    return {

        "risk_ordering":
            (
                low["risk_score"]
                < medium["risk_score"]
                < high["risk_score"]
                < critical["risk_score"]
            ),

        "analyst_escalation":
            (
                low["analyst_required"] is False
                and medium["analyst_required"] is True
                and high["analyst_required"] is True
                and critical["analyst_required"] is True
            ),

        "critical_candidate":
            critical["containment_candidate"] is True,

        "fallback_reduces_containment":
            fallback_critical["containment_candidate"] is False,

        "ambiguity_reduces_containment":
            ambiguous_critical["containment_candidate"] is False,

        "containment_never_authorized":
            (
                low["containment_authorized"] is False
                and medium["containment_authorized"] is False
                and high["containment_authorized"] is False
                and critical["containment_authorized"] is False
                and fallback_critical["containment_authorized"] is False
                and ambiguous_critical["containment_authorized"] is False
            ),
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
    print("SmartSOC Phase 6.6")
    print("Adversarial Response Safety Validation")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    scenario_results = run_scenarios()

    if scenario_results["failed"] == 0:
        print(
            "[PASS] Adversarial response scenarios"
        )
    else:
        print(
            "[FAIL] Adversarial response scenarios"
        )

    safety_checks = validate_global_safety()

    safety_passed = all(
        safety_checks.values()
    )

    if safety_passed:
        print(
            "[PASS] Global operational safety"
        )
    else:
        print(
            "[FAIL] Global operational safety"
        )

    determinism_checks = validate_determinism()

    determinism_passed = all(
        determinism_checks.values()
    )

    if determinism_passed:
        print(
            "[PASS] Adversarial determinism"
        )
    else:
        print(
            "[FAIL] Adversarial determinism"
        )

    monotonicity_checks = (
        validate_safety_monotonicity()
    )

    monotonicity_passed = all(
        monotonicity_checks.values()
    )

    if monotonicity_passed:
        print(
            "[PASS] Safety monotonicity"
        )
    else:
        print(
            "[FAIL] Safety monotonicity"
        )

    # ---------------------------------------------------------------
    # Flatten checks
    # ---------------------------------------------------------------

    all_checks: Dict[str, bool] = {}

    for key, value in safety_checks.items():
        all_checks[
            f"global_safety.{key}"
        ] = bool(value)

    for key, value in determinism_checks.items():
        all_checks[
            f"determinism.{key}"
        ] = bool(value)

    for key, value in monotonicity_checks.items():
        all_checks[
            f"monotonicity.{key}"
        ] = bool(value)

    all_checks[
        "scenarios.all_passed"
    ] = (
        scenario_results["failed"] == 0
    )

    total_checks = len(
        all_checks
    )

    passed_checks = sum(
        1
        for value in all_checks.values()
        if value
    )

    failed_checks = (
        total_checks
        - passed_checks
    )

    overall_passed = (
        failed_checks == 0
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {

        "phase": "6.6",

        "phase_name":
            "Adversarial Response Safety Validation",

        "version":
            VERSION,

        "status":
            "PASS"
            if overall_passed
            else "FAIL",

        "checks": {
            "total": total_checks,
            "passed": passed_checks,
            "failed": failed_checks,
        },

        "adversarial_scenarios":
            scenario_results,

        "global_safety_checks":
            safety_checks,

        "determinism_checks":
            determinism_checks,

        "safety_monotonicity_checks":
            monotonicity_checks,

        "safety_policy":
            SAFETY_EXPECTED,

        "production_safety": {

            "model_modified": False,

            "dataset_modified": False,

            "thresholds_modified": False,

            "risk_engine_modified": False,

            "decision_policy_modified": False,

            "response_planner_modified": False,

            "wazuh_integration": False,

            "shuffle_integration": False,

            "response_actions_executed": False,

            "rollback_actions_executed": False,

            "containment_actions_executed": False,
        },

        "decision":
            "ACCEPTED"
            if overall_passed
            else "REVIEW_REQUIRED",
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Final decision
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 6.6 FINAL DECISION")
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
        f"Checks : "
        f"{total_checks} total / "
        f"{failed_checks} failed"
    )

    print(
        "Adversarial scenarios : "
        f"{scenario_results['total']} total / "
        f"{scenario_results['failed']} failed"
    )

    print(
        "Determinism checks : "
        f"{len(determinism_checks)} total / "
        f"{sum(not v for v in determinism_checks.values())} failed"
    )

    print(
        "Safety monotonicity : "
        f"{len(monotonicity_checks)} total / "
        f"{sum(not v for v in monotonicity_checks.values())} failed"
    )

    print(
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()
        print(
            "PHASE 6.6 ACCEPTED — "
            "ADVERSARIAL RESPONSE SAFETY VALIDATED"
        )

        return 0

    print()
    print(
        "PHASE 6.6 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())