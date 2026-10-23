from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 6.1
# Response / Mitigation Policy Specification
# Version 4.1
#
# Purpose:
#   Validate the formal response and mitigation policy before any
#   Wazuh / Shuffle execution integration is introduced.
#
# Safety:
#   This gate is specification-only.
#
#   It does NOT:
#       - execute response actions
#       - isolate hosts
#       - block IP addresses
#       - terminate processes
#       - disable accounts
#       - modify firewall rules
#       - modify Wazuh
#       - modify Shuffle
# =====================================================================


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
    / "gate6_1"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase6_gate6_1_response_policy_spec_report.json"
)


# =====================================================================
# Formal response policy
# =====================================================================

RISK_POLICY = {
    "LOW": {
        "decision": "MONITOR",
        "response_mode": "OBSERVE_ONLY",
        "analyst_required": False,
        "containment_candidate": False,
        "containment_authorized": False,
        "execution_allowed": False,
        "rollback_required": False,
    },

    "MEDIUM": {
        "decision": "ANALYST_REVIEW",
        "response_mode": "INVESTIGATE",
        "analyst_required": True,
        "containment_candidate": False,
        "containment_authorized": False,
        "execution_allowed": False,
        "rollback_required": False,
    },

    "HIGH": {
        "decision": "PRIORITY_ANALYST_REVIEW",
        "response_mode": "PREPARE_RESPONSE",
        "analyst_required": True,
        "containment_candidate": False,
        "containment_authorized": False,
        "execution_allowed": False,
        "rollback_required": True,
    },

    "CRITICAL": {
        "decision": "CONTAINMENT_CANDIDATE",
        "response_mode": "PREPARE_CONTAINMENT",
        "analyst_required": True,
        "containment_candidate": True,
        "containment_authorized": False,
        "execution_allowed": False,
        "rollback_required": True,
    },
}


# =====================================================================
# Approved response categories
# =====================================================================

RESPONSE_CATEGORIES = {
    "OBSERVE_ONLY": {
        "description":
            "Continue monitoring and preserve available telemetry.",

        "destructive": False,

        "requires_analyst":
            False,

        "may_execute":
            False,
    },

    "INVESTIGATE": {
        "description":
            "Collect and review additional evidence before response.",

        "destructive": False,

        "requires_analyst":
            True,

        "may_execute":
            False,
    },

    "PREPARE_RESPONSE": {
        "description":
            "Prepare an analyst-reviewed response plan without executing it.",

        "destructive": False,

        "requires_analyst":
            True,

        "may_execute":
            False,
    },

    "PREPARE_CONTAINMENT": {
        "description":
            "Prepare containment options for analyst authorization.",

        "destructive": False,

        "requires_analyst":
            True,

        "may_execute":
            False,
    },
}


# =====================================================================
# Response action catalog
#
# These are PLAN TYPES only.
# They are NOT executable actions in Phase 6.1.
# =====================================================================

RESPONSE_ACTIONS = {

    "COLLECT_ADDITIONAL_EVIDENCE": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_PROCESS_TREE": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_NETWORK_CONNECTIONS": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_PERSISTENCE_MECHANISM": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_PROCESS_ISOLATION": {
        "category": "PREPARE_RESPONSE",
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_HOST_ISOLATION": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_IP_BLOCK": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_ACCOUNT_DISABLE": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PRESERVE_ARTIFACTS": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },
}


# =====================================================================
# Technique-oriented mitigation policy
#
# These are recommendations, not execution commands.
# =====================================================================

TECHNIQUE_POLICY = {

    "T1059": {
        "name": "Command and Scripting Interpreter",
        "recommended_actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1059.001": {
        "name": "PowerShell",
        "recommended_actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PRESERVE_ARTIFACTS",
        ],
    },

    "T1059.003": {
        "name": "Windows Command Shell",
        "recommended_actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1105": {
        "name": "Ingress Tool Transfer",
        "recommended_actions": [
            "REVIEW_NETWORK_CONNECTIONS",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1570": {
        "name": "Lateral Tool Transfer",
        "recommended_actions": [
            "REVIEW_NETWORK_CONNECTIONS",
            "REVIEW_PROCESS_TREE",
            "PRESERVE_ARTIFACTS",
        ],
    },

    "T1027.004": {
        "name": "Compile After Delivery",
        "recommended_actions": [
            "PRESERVE_ARTIFACTS",
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1543.003": {
        "name": "Windows Service",
        "recommended_actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1112": {
        "name": "Modify Registry",
        "recommended_actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1053.005": {
        "name": "Scheduled Task/Job",
        "recommended_actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1070.004": {
        "name": "File Deletion",
        "recommended_actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1078": {
        "name": "Valid Accounts",
        "recommended_actions": [
            "COLLECT_ADDITIONAL_EVIDENCE",
            "REVIEW_PROCESS_TREE",
            "REVIEW_NETWORK_CONNECTIONS",
        ],
    },

    "T1485": {
        "name": "Data Destruction",
        "recommended_actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
    },

    "T1565.001": {
        "name": "Stored Data Manipulation",
        "recommended_actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
    },

    "T1531": {
        "name": "Account Access Removal",
        "recommended_actions": [
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_ACCOUNT_DISABLE",
        ],
    },

    "T1562.001": {
        "name": "Impair Defenses",
        "recommended_actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
    },

}


# =====================================================================
# Uncertainty safety policy
# =====================================================================

UNCERTAINTY_POLICY = {

    "fallback": {
        "forces_analyst_review": True,
        "containment_authorized": False,
        "automatic_response": False,
    },

    "ambiguous": {
        "forces_analyst_review": True,
        "containment_authorized": False,
        "automatic_response": False,
    },

    "missing_behavior": {
        "forces_evidence_collection": True,
        "automatic_response": False,
    },
}


# =====================================================================
# Operational safety boundary
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

    "fallback_can_contain": False,

    "ambiguous_can_contain": False,

    "missing_behavior_can_contain_automatically": False,

    "wazuh_integration": False,

    "shuffle_integration": False,

    "response_actions_executed": False,

    "containment_actions_executed": False,
}


# =====================================================================
# Validation helpers
# =====================================================================

def validate_risk_policy() -> Dict[str, Any]:

    required_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    actual_levels = set(
        RISK_POLICY.keys()
    )

    checks = {

        "all_risk_levels_present":
            actual_levels == required_levels,

        "low_monitor":
            RISK_POLICY["LOW"]["decision"]
            == "MONITOR",

        "medium_analyst":
            RISK_POLICY["MEDIUM"]["decision"]
            == "ANALYST_REVIEW",

        "high_priority_review":
            RISK_POLICY["HIGH"]["decision"]
            == "PRIORITY_ANALYST_REVIEW",

        "critical_candidate":
            RISK_POLICY["CRITICAL"]["decision"]
            == "CONTAINMENT_CANDIDATE",

        "low_no_analyst":
            RISK_POLICY["LOW"]["analyst_required"]
            is False,

        "medium_requires_analyst":
            RISK_POLICY["MEDIUM"]["analyst_required"]
            is True,

        "high_requires_analyst":
            RISK_POLICY["HIGH"]["analyst_required"]
            is True,

        "critical_requires_analyst":
            RISK_POLICY["CRITICAL"]["analyst_required"]
            is True,

        "critical_not_authorized":
            RISK_POLICY["CRITICAL"]["containment_authorized"]
            is False,

        "no_risk_level_allows_execution":
            all(
                item["execution_allowed"] is False
                for item in RISK_POLICY.values()
            ),
    }

    return checks


def validate_response_categories() -> Dict[str, Any]:

    required_categories = {
        "OBSERVE_ONLY",
        "INVESTIGATE",
        "PREPARE_RESPONSE",
        "PREPARE_CONTAINMENT",
    }

    checks = {

        "required_categories_present":
            set(RESPONSE_CATEGORIES.keys())
            == required_categories,

        "observe_only_not_executable":
            RESPONSE_CATEGORIES[
                "OBSERVE_ONLY"
            ]["may_execute"] is False,

        "investigate_requires_analyst":
            RESPONSE_CATEGORIES[
                "INVESTIGATE"
            ]["requires_analyst"] is True,

        "prepare_response_requires_analyst":
            RESPONSE_CATEGORIES[
                "PREPARE_RESPONSE"
            ]["requires_analyst"] is True,

        "prepare_containment_requires_analyst":
            RESPONSE_CATEGORIES[
                "PREPARE_CONTAINMENT"
            ]["requires_analyst"] is True,

        "no_category_is_automatically_executable":
            all(
                item["may_execute"] is False
                for item in RESPONSE_CATEGORIES.values()
            ),
    }

    return checks


def validate_action_catalog() -> Dict[str, Any]:

    checks = {}

    checks[
        "action_catalog_non_empty"
    ] = len(RESPONSE_ACTIONS) > 0

    checks[
        "all_actions_require_approval"
    ] = all(
        item["requires_approval"] is True
        for item in RESPONSE_ACTIONS.values()
    )

    checks[
        "no_action_is_automatic"
    ] = all(
        item["automatic"] is False
        for item in RESPONSE_ACTIONS.values()
    )

    checks[
        "containment_actions_are_preparatory"
    ] = all(
        (
            not name.startswith("PREPARE_")
            or item["requires_approval"] is True
        )
        for name, item in RESPONSE_ACTIONS.items()
    )

    checks[
        "all_actions_have_categories"
    ] = all(
        item["category"]
        in RESPONSE_CATEGORIES
        for item in RESPONSE_ACTIONS.values()
    )

    return checks


def validate_technique_policy() -> Dict[str, Any]:

    checks = {}

    checks[
        "technique_policy_non_empty"
    ] = len(TECHNIQUE_POLICY) > 0

    checks[
        "all_techniques_have_names"
    ] = all(
        bool(item["name"])
        for item in TECHNIQUE_POLICY.values()
    )

    checks[
        "all_techniques_have_recommendations"
    ] = all(
        len(item["recommended_actions"]) > 0
        for item in TECHNIQUE_POLICY.values()
    )

    checks[
        "all_recommended_actions_exist"
    ] = all(
        action in RESPONSE_ACTIONS
        for item in TECHNIQUE_POLICY.values()
        for action in item["recommended_actions"]
    )

    return checks


def validate_uncertainty_policy() -> Dict[str, Any]:

    checks = {

        "fallback_forces_review":
            UNCERTAINTY_POLICY[
                "fallback"
            ]["forces_analyst_review"] is True,

        "fallback_cannot_contain":
            UNCERTAINTY_POLICY[
                "fallback"
            ]["containment_authorized"] is False,

        "fallback_no_automatic_response":
            UNCERTAINTY_POLICY[
                "fallback"
            ]["automatic_response"] is False,

        "ambiguity_forces_review":
            UNCERTAINTY_POLICY[
                "ambiguous"
            ]["forces_analyst_review"] is True,

        "ambiguity_cannot_contain":
            UNCERTAINTY_POLICY[
                "ambiguous"
            ]["containment_authorized"] is False,

        "ambiguity_no_automatic_response":
            UNCERTAINTY_POLICY[
                "ambiguous"
            ]["automatic_response"] is False,

        "missing_behavior_requires_evidence":
            UNCERTAINTY_POLICY[
                "missing_behavior"
            ]["forces_evidence_collection"] is True,

        "missing_behavior_no_automatic_response":
            UNCERTAINTY_POLICY[
                "missing_behavior"
            ]["automatic_response"] is False,
    }

    return checks


def validate_safety_policy() -> Dict[str, Any]:

    checks = {

        "automatic_response_disabled":
            SAFETY_POLICY[
                "automatic_response"
            ] is False,

        "automatic_blocking_disabled":
            SAFETY_POLICY[
                "automatic_blocking"
            ] is False,

        "automatic_isolation_disabled":
            SAFETY_POLICY[
                "automatic_isolation"
            ] is False,

        "automatic_process_kill_disabled":
            SAFETY_POLICY[
                "automatic_process_kill"
            ] is False,

        "automatic_account_disable_disabled":
            SAFETY_POLICY[
                "automatic_account_disable"
            ] is False,

        "automatic_firewall_change_disabled":
            SAFETY_POLICY[
                "automatic_firewall_change"
            ] is False,

        "automatic_file_deletion_disabled":
            SAFETY_POLICY[
                "automatic_file_deletion"
            ] is False,

        "containment_not_authorized":
            SAFETY_POLICY[
                "containment_authorized"
            ] is False,

        "containment_requires_analyst":
            SAFETY_POLICY[
                "containment_requires_analyst"
            ] is True,

        "fallback_cannot_contain":
            SAFETY_POLICY[
                "fallback_can_contain"
            ] is False,

        "ambiguous_cannot_contain":
            SAFETY_POLICY[
                "ambiguous_can_contain"
            ] is False,

        "missing_behavior_no_auto_containment":
            SAFETY_POLICY[
                "missing_behavior_can_contain_automatically"
            ] is False,

        "wazuh_disabled":
            SAFETY_POLICY[
                "wazuh_integration"
            ] is False,

        "shuffle_disabled":
            SAFETY_POLICY[
                "shuffle_integration"
            ] is False,

        "no_response_executed":
            SAFETY_POLICY[
                "response_actions_executed"
            ] is False,

        "no_containment_executed":
            SAFETY_POLICY[
                "containment_actions_executed"
            ] is False,
    }

    return checks


def validate_monotonicity() -> Dict[str, Any]:

    order = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ]

    decisions = [
        RISK_POLICY[level]["decision"]
        for level in order
    ]

    checks = {

        "four_distinct_risk_levels":
            len(decisions) == 4,

        "low_is_least_escalated":
            decisions[0] == "MONITOR",

        "medium_escalates":
            decisions[1] == "ANALYST_REVIEW",

        "high_escalates":
            decisions[2] == "PRIORITY_ANALYST_REVIEW",

        "critical_escalates":
            decisions[3] == "CONTAINMENT_CANDIDATE",

        "critical_still_requires_analyst":
            RISK_POLICY[
                "CRITICAL"
            ]["analyst_required"] is True,
    }

    return checks


def validate_no_execution_metadata() -> Dict[str, Any]:

    suspicious_execution_terms = {
        "EXECUTE",
        "RUN",
        "KILL",
        "ISOLATE",
        "BLOCK",
        "DISABLE",
        "DELETE",
    }

    # The catalog may contain PREPARE_* names, but these are plans,
    # not executable operations.
    checks = {

        "all_actions_are_declared_as_nonautomatic":
            all(
                action["automatic"] is False
                for action in RESPONSE_ACTIONS.values()
            ),

        "all_actions_require_approval":
            all(
                action["requires_approval"] is True
                for action in RESPONSE_ACTIONS.values()
            ),

        "safety_boundary_has_no_execution":
            SAFETY_POLICY[
                "response_actions_executed"
            ] is False
            and SAFETY_POLICY[
                "containment_actions_executed"
            ] is False,

        "integration_disabled":
            SAFETY_POLICY[
                "wazuh_integration"
            ] is False
            and SAFETY_POLICY[
                "shuffle_integration"
            ] is False,

        "no_automatic_containment":
            SAFETY_POLICY[
                "containment_authorized"
            ] is False,
    }

    return checks


# =====================================================================
# Main
# =====================================================================

def main() -> int:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("SmartSOC Phase 6.1")
    print("Response / Mitigation Policy Specification")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    all_checks: Dict[str, bool] = {}
    sections: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------
    # Risk policy
    # ---------------------------------------------------------------

    checks = validate_risk_policy()

    sections["risk_policy"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[f"risk_policy.{key}"] = value

    # ---------------------------------------------------------------
    # Response categories
    # ---------------------------------------------------------------

    checks = validate_response_categories()

    sections["response_categories"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"response_categories.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Action catalog
    # ---------------------------------------------------------------

    checks = validate_action_catalog()

    sections["action_catalog"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"action_catalog.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Technique policy
    # ---------------------------------------------------------------

    checks = validate_technique_policy()

    sections["technique_policy"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"technique_policy.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Uncertainty
    # ---------------------------------------------------------------

    checks = validate_uncertainty_policy()

    sections["uncertainty_policy"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"uncertainty_policy.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Safety
    # ---------------------------------------------------------------

    checks = validate_safety_policy()

    sections["safety_policy"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"safety_policy.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Monotonicity
    # ---------------------------------------------------------------

    checks = validate_monotonicity()

    sections["monotonicity"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"monotonicity.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Execution boundary
    # ---------------------------------------------------------------

    checks = validate_no_execution_metadata()

    sections["execution_boundary"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    for key, value in checks.items():
        all_checks[
            f"execution_boundary.{key}"
        ] = value

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    labels = [
        (
            "Risk policy specification",
            sections["risk_policy"],
        ),
        (
            "Response category specification",
            sections["response_categories"],
        ),
        (
            "Response action catalog",
            sections["action_catalog"],
        ),
        (
            "Technique mitigation policy",
            sections["technique_policy"],
        ),
        (
            "Uncertainty policy",
            sections["uncertainty_policy"],
        ),
        (
            "Operational safety policy",
            sections["safety_policy"],
        ),
        (
            "Risk disposition monotonicity",
            sections["monotonicity"],
        ),
        (
            "Execution boundary",
            sections["execution_boundary"],
        ),
    ]

    for label, section in labels:

        print(
            "["
            + (
                "PASS"
                if section["passed"]
                else "FAIL"
            )
            + "] "
            + label
        )

    print()

    # ---------------------------------------------------------------
    # Count checks
    # ---------------------------------------------------------------

    total_checks = len(all_checks)

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

        "phase": "6.1",

        "phase_name":
            "Response / Mitigation Policy Specification",

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

        "risk_policy":
            RISK_POLICY,

        "response_categories":
            RESPONSE_CATEGORIES,

        "response_actions":
            RESPONSE_ACTIONS,

        "technique_policy":
            TECHNIQUE_POLICY,

        "uncertainty_policy":
            UNCERTAINTY_POLICY,

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

            "wazuh_integration": False,

            "shuffle_integration": False,

            "response_actions_executed": False,

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
    # Final output
    # ---------------------------------------------------------------

    print("=" * 72)
    print("PHASE 6.1 FINAL DECISION")
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
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 6.1 ACCEPTED — "
            "RESPONSE / MITIGATION POLICY SPECIFIED"
        )

        return 0

    print()

    print(
        "PHASE 6.1 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())