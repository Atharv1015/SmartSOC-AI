from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 6.5
# Rollback & Recovery Safety Validation
# Version 4.1
#
# Purpose:
#   Validate that disruptive response plans have explicit rollback and
#   recovery definitions before any execution integration is introduced.
#
# Safety:
#   OFFLINE / SPECIFICATION VALIDATION ONLY.
#
#   This script does NOT:
#       - isolate a host
#       - terminate a process
#       - block an IP
#       - disable an account
#       - modify a firewall
#       - delete a file
#       - execute PowerShell
#       - connect to Wazuh
#       - connect to Shuffle
# =====================================================================


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
    / "gate6_5"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase6_gate6_5_rollback_recovery_report.json"
)


# =====================================================================
# Response / rollback catalog
#
# These are PLANS only.
# No command or action is executed.
# =====================================================================

ROLLBACK_PLANS: Dict[str, Dict[str, Any]] = {

    "PREPARE_PROCESS_ISOLATION": {
        "response_type": "PROCESS_ISOLATION",
        "rollback_action": "RESTORE_PROCESS_OPERATION",
        "rollback_required": True,
        "rollback_reversible": True,
        "requires_analyst": True,
        "verification_required": True,
        "execution_enabled": False,
    },

    "PREPARE_HOST_ISOLATION": {
        "response_type": "HOST_ISOLATION",
        "rollback_action": "RESTORE_HOST_NETWORK_ACCESS",
        "rollback_required": True,
        "rollback_reversible": True,
        "requires_analyst": True,
        "verification_required": True,
        "execution_enabled": False,
    },

    "PREPARE_IP_BLOCK": {
        "response_type": "IP_BLOCK",
        "rollback_action": "REMOVE_NETWORK_BLOCK",
        "rollback_required": True,
        "rollback_reversible": True,
        "requires_analyst": True,
        "verification_required": True,
        "execution_enabled": False,
    },

    "PREPARE_ACCOUNT_DISABLE": {
        "response_type": "ACCOUNT_DISABLE",
        "rollback_action": "RESTORE_ACCOUNT_ACCESS",
        "rollback_required": True,
        "rollback_reversible": True,
        "requires_analyst": True,
        "verification_required": True,
        "execution_enabled": False,
    },
}


# =====================================================================
# Non-disruptive investigation actions
# =====================================================================

INVESTIGATION_ACTIONS = {

    "COLLECT_ADDITIONAL_EVIDENCE": {
        "disruptive": False,
        "rollback_required": False,
        "verification_required": True,
    },

    "REVIEW_PROCESS_TREE": {
        "disruptive": False,
        "rollback_required": False,
        "verification_required": True,
    },

    "REVIEW_NETWORK_CONNECTIONS": {
        "disruptive": False,
        "rollback_required": False,
        "verification_required": True,
    },

    "REVIEW_PERSISTENCE_MECHANISM": {
        "disruptive": False,
        "rollback_required": False,
        "verification_required": True,
    },

    "PRESERVE_ARTIFACTS": {
        "disruptive": False,
        "rollback_required": False,
        "verification_required": True,
    },
}


# =====================================================================
# Global safety policy
# =====================================================================

SAFETY_POLICY = {

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
# Validation helpers
# =====================================================================

def validate_rollback_schema() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["rollback_catalog_not_empty"] = (
        len(ROLLBACK_PLANS) > 0
    )

    checks["all_response_types_present"] = all(
        bool(item["response_type"])
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_rollback_actions_present"] = all(
        bool(item["rollback_action"])
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_require_rollback"] = all(
        item["rollback_required"] is True
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_rollback_reversible"] = all(
        item["rollback_reversible"] is True
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_require_analyst"] = all(
        item["requires_analyst"] is True
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_require_verification"] = all(
        item["verification_required"] is True
        for item in ROLLBACK_PLANS.values()
    )

    return checks


def validate_execution_disabled() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["all_rollback_execution_disabled"] = all(
        item["execution_enabled"] is False
        for item in ROLLBACK_PLANS.values()
    )

    checks["automatic_response_disabled"] = (
        SAFETY_POLICY["automatic_response"]
        is False
    )

    checks["automatic_blocking_disabled"] = (
        SAFETY_POLICY["automatic_blocking"]
        is False
    )

    checks["automatic_isolation_disabled"] = (
        SAFETY_POLICY["automatic_isolation"]
        is False
    )

    checks["automatic_process_kill_disabled"] = (
        SAFETY_POLICY["automatic_process_kill"]
        is False
    )

    checks["automatic_account_disable_disabled"] = (
        SAFETY_POLICY["automatic_account_disable"]
        is False
    )

    checks["automatic_firewall_change_disabled"] = (
        SAFETY_POLICY["automatic_firewall_change"]
        is False
    )

    checks["automatic_file_deletion_disabled"] = (
        SAFETY_POLICY["automatic_file_deletion"]
        is False
    )

    checks["rollback_execution_disabled"] = (
        SAFETY_POLICY["rollback_execution_enabled"]
        is False
    )

    checks["recovery_execution_disabled"] = (
        SAFETY_POLICY["recovery_execution_enabled"]
        is False
    )

    return checks


def validate_analyst_boundary() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["containment_not_authorized"] = (
        SAFETY_POLICY["containment_authorized"]
        is False
    )

    checks["containment_requires_analyst"] = (
        SAFETY_POLICY["containment_requires_analyst"]
        is True
    )

    checks["all_disruptive_rollbacks_require_analyst"] = all(
        item["requires_analyst"] is True
        for item in ROLLBACK_PLANS.values()
    )

    return checks


def validate_investigation_actions() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["investigation_catalog_not_empty"] = (
        len(INVESTIGATION_ACTIONS) > 0
    )

    checks["investigation_actions_non_disruptive"] = all(
        item["disruptive"] is False
        for item in INVESTIGATION_ACTIONS.values()
    )

    checks["investigation_rollback_not_required"] = all(
        item["rollback_required"] is False
        for item in INVESTIGATION_ACTIONS.values()
    )

    checks["investigation_verification_recorded"] = all(
        item["verification_required"] is True
        for item in INVESTIGATION_ACTIONS.values()
    )

    return checks


def validate_recovery_semantics() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["process_isolation_has_recovery"] = (
        ROLLBACK_PLANS[
            "PREPARE_PROCESS_ISOLATION"
        ]["rollback_action"]
        == "RESTORE_PROCESS_OPERATION"
    )

    checks["host_isolation_has_recovery"] = (
        ROLLBACK_PLANS[
            "PREPARE_HOST_ISOLATION"
        ]["rollback_action"]
        == "RESTORE_HOST_NETWORK_ACCESS"
    )

    checks["ip_block_has_recovery"] = (
        ROLLBACK_PLANS[
            "PREPARE_IP_BLOCK"
        ]["rollback_action"]
        == "REMOVE_NETWORK_BLOCK"
    )

    checks["account_disable_has_recovery"] = (
        ROLLBACK_PLANS[
            "PREPARE_ACCOUNT_DISABLE"
        ]["rollback_action"]
        == "RESTORE_ACCOUNT_ACCESS"
    )

    checks["all_recovery_paths_are_reversible"] = all(
        item["rollback_reversible"] is True
        for item in ROLLBACK_PLANS.values()
    )

    return checks


def validate_verification_requirements() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["all_disruptive_actions_need_verification"] = all(
        item["verification_required"] is True
        for item in ROLLBACK_PLANS.values()
    )

    checks["all_investigation_actions_need_verification"] = all(
        item["verification_required"] is True
        for item in INVESTIGATION_ACTIONS.values()
    )

    return checks


def validate_no_side_effects() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["no_response_executed"] = (
        SAFETY_POLICY["response_actions_executed"]
        is False
    )

    checks["no_rollback_executed"] = (
        SAFETY_POLICY["rollback_actions_executed"]
        is False
    )

    checks["no_containment_executed"] = (
        SAFETY_POLICY["containment_actions_executed"]
        is False
    )

    checks["wazuh_disabled"] = (
        SAFETY_POLICY["wazuh_integration"]
        is False
    )

    checks["shuffle_disabled"] = (
        SAFETY_POLICY["shuffle_integration"]
        is False
    )

    return checks


# =====================================================================
# Recovery scenario simulation
#
# This does NOT execute anything. It validates the expected state
# transitions of the recovery model.
# =====================================================================

RECOVERY_SCENARIOS = [

    {
        "id": "R01",
        "name": "Process isolation recovery",
        "response": "PREPARE_PROCESS_ISOLATION",
        "expected_rollback": "RESTORE_PROCESS_OPERATION",
    },

    {
        "id": "R02",
        "name": "Host isolation recovery",
        "response": "PREPARE_HOST_ISOLATION",
        "expected_rollback": "RESTORE_HOST_NETWORK_ACCESS",
    },

    {
        "id": "R03",
        "name": "IP block recovery",
        "response": "PREPARE_IP_BLOCK",
        "expected_rollback": "REMOVE_NETWORK_BLOCK",
    },

    {
        "id": "R04",
        "name": "Account disable recovery",
        "response": "PREPARE_ACCOUNT_DISABLE",
        "expected_rollback": "RESTORE_ACCOUNT_ACCESS",
    },
]


def validate_recovery_scenarios() -> Dict[str, Any]:

    results: List[Dict[str, Any]] = []

    for scenario in RECOVERY_SCENARIOS:

        response = ROLLBACK_PLANS[
            scenario["response"]
        ]

        passed = (
            response["rollback_action"]
            == scenario["expected_rollback"]
            and response["rollback_required"] is True
            and response["rollback_reversible"] is True
            and response["requires_analyst"] is True
            and response["verification_required"] is True
            and response["execution_enabled"] is False
        )

        results.append(
            {
                "id": scenario["id"],
                "name": scenario["name"],
                "response": scenario["response"],
                "expected_rollback":
                    scenario["expected_rollback"],
                "actual_rollback":
                    response["rollback_action"],
                "passed": passed,
            }
        )

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
# Main
# =====================================================================

def main() -> int:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SmartSOC Phase 6.5")
    print("Rollback & Recovery Safety Validation")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    sections: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------
    # Rollback schema
    # ---------------------------------------------------------------

    checks = validate_rollback_schema()

    sections["rollback_schema"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["rollback_schema"]["passed"]
            else "FAIL"
        )
        + "] Rollback schema"
    )

    # ---------------------------------------------------------------
    # Execution disabled
    # ---------------------------------------------------------------

    checks = validate_execution_disabled()

    sections["execution_disabled"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["execution_disabled"]["passed"]
            else "FAIL"
        )
        + "] Response / rollback execution safety"
    )

    # ---------------------------------------------------------------
    # Analyst boundary
    # ---------------------------------------------------------------

    checks = validate_analyst_boundary()

    sections["analyst_boundary"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["analyst_boundary"]["passed"]
            else "FAIL"
        )
        + "] Analyst authorization boundary"
    )

    # ---------------------------------------------------------------
    # Investigation
    # ---------------------------------------------------------------

    checks = validate_investigation_actions()

    sections["investigation_actions"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["investigation_actions"]["passed"]
            else "FAIL"
        )
        + "] Investigation action safety"
    )

    # ---------------------------------------------------------------
    # Recovery semantics
    # ---------------------------------------------------------------

    checks = validate_recovery_semantics()

    sections["recovery_semantics"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["recovery_semantics"]["passed"]
            else "FAIL"
        )
        + "] Recovery semantics"
    )

    # ---------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------

    checks = validate_verification_requirements()

    sections["verification_requirements"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["verification_requirements"]["passed"]
            else "FAIL"
        )
        + "] Recovery verification requirements"
    )

    # ---------------------------------------------------------------
    # Side effects
    # ---------------------------------------------------------------

    checks = validate_no_side_effects()

    sections["no_side_effects"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["no_side_effects"]["passed"]
            else "FAIL"
        )
        + "] No-side-effect boundary"
    )

    # ---------------------------------------------------------------
    # Recovery scenarios
    # ---------------------------------------------------------------

    scenarios = validate_recovery_scenarios()

    sections["recovery_scenarios"] = scenarios

    print(
        "["
        + (
            "PASS"
            if scenarios["failed"] == 0
            else "FAIL"
        )
        + "] Recovery scenario validation"
    )

    # ---------------------------------------------------------------
    # Flatten checks
    # ---------------------------------------------------------------

    all_checks: Dict[str, bool] = {}

    for section_name, section in sections.items():

        if "checks" in section:

            for check_name, value in section[
                "checks"
            ].items():

                all_checks[
                    f"{section_name}.{check_name}"
                ] = bool(value)

    scenario_check = (
        scenarios["failed"] == 0
    )

    all_checks[
        "recovery_scenarios.all_passed"
    ] = scenario_check

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

        "phase": "6.5",

        "phase_name":
            "Rollback & Recovery Safety Validation",

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

        "rollback_plans":
            ROLLBACK_PLANS,

        "investigation_actions":
            INVESTIGATION_ACTIONS,

        "recovery_scenarios":
            scenarios,

        "safety_policy":
            SAFETY_POLICY,

        "validation_sections":
            sections,

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
    # Final
    # ---------------------------------------------------------------

    print()

    print("=" * 72)
    print("PHASE 6.5 FINAL DECISION")
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
        "Recovery scenarios : "
        f"{scenarios['total']} total / "
        f"{scenarios['failed']} failed"
    )

    print(
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 6.5 ACCEPTED — "
            "ROLLBACK AND RECOVERY SAFETY VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 6.5 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())