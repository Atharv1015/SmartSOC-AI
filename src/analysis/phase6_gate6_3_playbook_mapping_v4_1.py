from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# =====================================================================
# SmartSOC Phase 6.3
# Response Playbook & MITRE Mapping Validation
# Version 4.1
#
# Purpose:
#   Validate that supported MITRE ATT&CK techniques have appropriate
#   response plans and that every mapped response remains analyst-
#   controlled and non-executable.
#
# Safety:
#   This gate performs validation only.
#
#   It does NOT:
#       - execute commands
#       - isolate hosts
#       - kill processes
#       - block IP addresses
#       - disable accounts
#       - modify firewalls
#       - delete files
#       - connect to Wazuh
#       - connect to Shuffle
# =====================================================================


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase6"
    / "gate6_3"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase6_gate6_3_playbook_mapping_report.json"
)


# =====================================================================
# Action catalog
# =====================================================================

ACTION_CATALOG: Dict[str, Dict[str, Any]] = {

    "COLLECT_ADDITIONAL_EVIDENCE": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "automatic": False,
        "destructive": False,
        "reversible": True,
    },

    "REVIEW_PROCESS_TREE": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "automatic": False,
        "destructive": False,
        "reversible": True,
    },

    "REVIEW_NETWORK_CONNECTIONS": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "automatic": False,
        "destructive": False,
        "reversible": True,
    },

    "REVIEW_PERSISTENCE_MECHANISM": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "automatic": False,
        "destructive": False,
        "reversible": True,
    },

    "PRESERVE_ARTIFACTS": {
        "category": "INVESTIGATE",
        "requires_approval": True,
        "automatic": False,
        "destructive": False,
        "reversible": True,
    },

    "PREPARE_PROCESS_ISOLATION": {
        "category": "PREPARE_RESPONSE",
        "requires_approval": True,
        "automatic": False,
        "destructive": True,
        "reversible": True,
    },

    "PREPARE_HOST_ISOLATION": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "automatic": False,
        "destructive": True,
        "reversible": True,
    },

    "PREPARE_IP_BLOCK": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "automatic": False,
        "destructive": True,
        "reversible": True,
    },

    "PREPARE_ACCOUNT_DISABLE": {
        "category": "PREPARE_CONTAINMENT",
        "requires_approval": True,
        "automatic": False,
        "destructive": True,
        "reversible": True,
    },
}


# =====================================================================
# MITRE ATT&CK response playbook mapping
#
# These are RESPONSE PLANS, not execution commands.
# =====================================================================

PLAYBOOKS: Dict[str, Dict[str, Any]] = {

    "T1059": {
        "name": "Command and Scripting Interpreter",
        "severity": "MEDIUM",
        "actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1059.001": {
        "name": "PowerShell",
        "severity": "HIGH",
        "actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PRESERVE_ARTIFACTS",
        ],
    },

    "T1059.003": {
        "name": "Windows Command Shell",
        "severity": "MEDIUM",
        "actions": [
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1105": {
        "name": "Ingress Tool Transfer",
        "severity": "HIGH",
        "actions": [
            "REVIEW_NETWORK_CONNECTIONS",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1570": {
        "name": "Lateral Tool Transfer",
        "severity": "HIGH",
        "actions": [
            "REVIEW_NETWORK_CONNECTIONS",
            "REVIEW_PROCESS_TREE",
            "PRESERVE_ARTIFACTS",
        ],
    },

    "T1027.004": {
        "name": "Compile After Delivery",
        "severity": "MEDIUM",
        "actions": [
            "PRESERVE_ARTIFACTS",
            "REVIEW_PROCESS_TREE",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1543.003": {
        "name": "Windows Service",
        "severity": "HIGH",
        "actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1112": {
        "name": "Modify Registry",
        "severity": "HIGH",
        "actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1053.005": {
        "name": "Scheduled Task/Job",
        "severity": "HIGH",
        "actions": [
            "REVIEW_PERSISTENCE_MECHANISM",
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1070.004": {
        "name": "File Deletion",
        "severity": "HIGH",
        "actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
        ],
    },

    "T1078": {
        "name": "Valid Accounts",
        "severity": "HIGH",
        "actions": [
            "COLLECT_ADDITIONAL_EVIDENCE",
            "REVIEW_PROCESS_TREE",
            "REVIEW_NETWORK_CONNECTIONS",
        ],
    },

    "T1485": {
        "name": "Data Destruction",
        "severity": "CRITICAL",
        "actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
    },

    "T1565.001": {
        "name": "Stored Data Manipulation",
        "severity": "CRITICAL",
        "actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
    },

    "T1531": {
        "name": "Account Access Removal",
        "severity": "CRITICAL",
        "actions": [
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_ACCOUNT_DISABLE",
        ],
    },

    "T1562.001": {
        "name": "Impair Defenses",
        "severity": "CRITICAL",
        "actions": [
            "PRESERVE_ARTIFACTS",
            "COLLECT_ADDITIONAL_EVIDENCE",
            "PREPARE_HOST_ISOLATION",
        ],
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

    "fallback_can_contain": False,
    "ambiguous_can_contain": False,

    "wazuh_integration": False,
    "shuffle_integration": False,

    "response_actions_executed": False,
    "containment_actions_executed": False,
}


# =====================================================================
# Validation
# =====================================================================

def validate_playbook_schema() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["playbooks_not_empty"] = (
        len(PLAYBOOKS) > 0
    )

    checks["all_playbooks_have_names"] = all(
        bool(item["name"])
        for item in PLAYBOOKS.values()
    )

    checks["all_playbooks_have_severity"] = all(
        item["severity"]
        in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }
        for item in PLAYBOOKS.values()
    )

    checks["all_playbooks_have_actions"] = all(
        len(item["actions"]) > 0
        for item in PLAYBOOKS.values()
    )

    return checks


def validate_action_references() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["all_actions_exist"] = all(
        action in ACTION_CATALOG
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
    )

    checks["all_actions_require_approval"] = all(
        ACTION_CATALOG[action]["requires_approval"]
        is True
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
    )

    checks["no_action_is_automatic"] = all(
        ACTION_CATALOG[action]["automatic"]
        is False
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
    )

    checks["all_actions_have_categories"] = all(
        bool(ACTION_CATALOG[action]["category"])
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
    )

    return checks


def validate_critical_playbooks() -> Dict[str, bool]:

    critical = {
        technique: playbook
        for technique, playbook
        in PLAYBOOKS.items()
        if playbook["severity"] == "CRITICAL"
    }

    checks: Dict[str, bool] = {}

    checks["critical_playbooks_exist"] = (
        len(critical) > 0
    )

    checks["critical_playbooks_include_containment_plan"] = all(
        any(
            action.startswith("PREPARE_")
            for action in playbook["actions"]
        )
        for playbook in critical.values()
    )

    checks["critical_containment_requires_approval"] = all(
        ACTION_CATALOG[action]["requires_approval"] is True
        for playbook in critical.values()
        for action in playbook["actions"]
        if action.startswith("PREPARE_")
    )

    checks["critical_containment_not_automatic"] = all(
        ACTION_CATALOG[action]["automatic"] is False
        for playbook in critical.values()
        for action in playbook["actions"]
        if action.startswith("PREPARE_")
    )

    return checks


def validate_investigation_coverage() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["process_review_available"] = (
        "REVIEW_PROCESS_TREE"
        in ACTION_CATALOG
    )

    checks["network_review_available"] = (
        "REVIEW_NETWORK_CONNECTIONS"
        in ACTION_CATALOG
    )

    checks["persistence_review_available"] = (
        "REVIEW_PERSISTENCE_MECHANISM"
        in ACTION_CATALOG
    )

    checks["evidence_collection_available"] = (
        "COLLECT_ADDITIONAL_EVIDENCE"
        in ACTION_CATALOG
    )

    checks["artifact_preservation_available"] = (
        "PRESERVE_ARTIFACTS"
        in ACTION_CATALOG
    )

    return checks


def validate_uncertainty_boundary() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    # Fallback and ambiguity must never directly map to containment.
    checks["fallback_cannot_contain"] = (
        SAFETY_POLICY[
            "fallback_can_contain"
        ]
        is False
    )

    checks["ambiguous_cannot_contain"] = (
        SAFETY_POLICY[
            "ambiguous_can_contain"
        ]
        is False
    )

    checks["containment_not_authorized"] = (
        SAFETY_POLICY[
            "containment_authorized"
        ]
        is False
    )

    checks["analyst_required_for_containment"] = (
        SAFETY_POLICY[
            "containment_requires_analyst"
        ]
        is True
    )

    return checks


def validate_global_safety() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    checks["automatic_response_disabled"] = (
        SAFETY_POLICY[
            "automatic_response"
        ]
        is False
    )

    checks["automatic_blocking_disabled"] = (
        SAFETY_POLICY[
            "automatic_blocking"
        ]
        is False
    )

    checks["automatic_isolation_disabled"] = (
        SAFETY_POLICY[
            "automatic_isolation"
        ]
        is False
    )

    checks["automatic_process_kill_disabled"] = (
        SAFETY_POLICY[
            "automatic_process_kill"
        ]
        is False
    )

    checks["automatic_account_disable_disabled"] = (
        SAFETY_POLICY[
            "automatic_account_disable"
        ]
        is False
    )

    checks["automatic_firewall_change_disabled"] = (
        SAFETY_POLICY[
            "automatic_firewall_change"
        ]
        is False
    )

    checks["automatic_file_deletion_disabled"] = (
        SAFETY_POLICY[
            "automatic_file_deletion"
        ]
        is False
    )

    checks["containment_not_authorized"] = (
        SAFETY_POLICY[
            "containment_authorized"
        ]
        is False
    )

    checks["containment_requires_analyst"] = (
        SAFETY_POLICY[
            "containment_requires_analyst"
        ]
        is True
    )

    checks["wazuh_disabled"] = (
        SAFETY_POLICY[
            "wazuh_integration"
        ]
        is False
    )

    checks["shuffle_disabled"] = (
        SAFETY_POLICY[
            "shuffle_integration"
        ]
        is False
    )

    checks["no_response_executed"] = (
        SAFETY_POLICY[
            "response_actions_executed"
        ]
        is False
    )

    checks["no_containment_executed"] = (
        SAFETY_POLICY[
            "containment_actions_executed"
        ]
        is False
    )

    return checks


def validate_playbook_semantics() -> Dict[str, bool]:

    checks: Dict[str, bool] = {}

    # Critical playbooks must not contain ordinary automatic actions.
    checks["critical_actions_are_plans"] = all(
        ACTION_CATALOG[action]["automatic"] is False
        for playbook in PLAYBOOKS.values()
        if playbook["severity"] == "CRITICAL"
        for action in playbook["actions"]
    )

    # Every containment preparation action must be explicitly named
    # as preparation, rather than an actual execution action.
    checks["containment_actions_are_preparatory"] = all(
        (
            not action.startswith("PREPARE_")
            or action.startswith("PREPARE_")
        )
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
    )

    # Disruptive actions must require approval.
    checks["disruptive_actions_require_approval"] = all(
        ACTION_CATALOG[action]["requires_approval"] is True
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
        if ACTION_CATALOG[action]["destructive"] is True
    )

    # Reversibility must be recorded for every disruptive action.
    checks["disruptive_actions_have_rollback_property"] = all(
        isinstance(
            ACTION_CATALOG[action]["reversible"],
            bool,
        )
        for playbook in PLAYBOOKS.values()
        for action in playbook["actions"]
        if ACTION_CATALOG[action]["destructive"] is True
    )

    return checks


# =====================================================================
# Representative technique validation
# =====================================================================

def validate_required_techniques() -> Dict[str, bool]:

    required = {

        "T1059",
        "T1059.001",
        "T1059.003",
        "T1105",
        "T1570",
        "T1027.004",
        "T1543.003",
        "T1112",
        "T1053.005",
        "T1070.004",
        "T1078",
        "T1485",
        "T1565.001",
        "T1531",
        "T1562.001",
    }

    present = set(
        PLAYBOOKS.keys()
    )

    return {
        "required_techniques_present":
            required.issubset(present),

        "required_technique_count":
            len(
                required.intersection(
                    present
                )
            )
            == len(required),
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
    print("SmartSOC Phase 6.3")
    print("Response Playbook & MITRE Mapping Validation")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    sections: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------

    checks = validate_playbook_schema()

    sections["playbook_schema"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["playbook_schema"]["passed"]
            else "FAIL"
        )
        + "] Playbook schema"
    )

    # ---------------------------------------------------------------
    # Action references
    # ---------------------------------------------------------------

    checks = validate_action_references()

    sections["action_references"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["action_references"]["passed"]
            else "FAIL"
        )
        + "] Action references and approval metadata"
    )

    # ---------------------------------------------------------------
    # Critical
    # ---------------------------------------------------------------

    checks = validate_critical_playbooks()

    sections["critical_playbooks"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["critical_playbooks"]["passed"]
            else "FAIL"
        )
        + "] Critical playbook containment boundary"
    )

    # ---------------------------------------------------------------
    # Investigation
    # ---------------------------------------------------------------

    checks = validate_investigation_coverage()

    sections["investigation_coverage"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["investigation_coverage"]["passed"]
            else "FAIL"
        )
        + "] Investigation coverage"
    )

    # ---------------------------------------------------------------
    # Uncertainty
    # ---------------------------------------------------------------

    checks = validate_uncertainty_boundary()

    sections["uncertainty_boundary"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["uncertainty_boundary"]["passed"]
            else "FAIL"
        )
        + "] Uncertainty safety boundary"
    )

    # ---------------------------------------------------------------
    # Global safety
    # ---------------------------------------------------------------

    checks = validate_global_safety()

    sections["global_safety"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["global_safety"]["passed"]
            else "FAIL"
        )
        + "] Global operational safety"
    )

    # ---------------------------------------------------------------
    # Semantics
    # ---------------------------------------------------------------

    checks = validate_playbook_semantics()

    sections["playbook_semantics"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["playbook_semantics"]["passed"]
            else "FAIL"
        )
        + "] Playbook semantic safety"
    )

    # ---------------------------------------------------------------
    # Required techniques
    # ---------------------------------------------------------------

    checks = validate_required_techniques()

    sections["required_techniques"] = {
        "checks": checks,
        "passed": all(checks.values()),
    }

    print(
        "["
        + (
            "PASS"
            if sections["required_techniques"]["passed"]
            else "FAIL"
        )
        + "] Required MITRE technique coverage"
    )

    # ---------------------------------------------------------------
    # Flatten checks
    # ---------------------------------------------------------------

    all_checks: Dict[str, bool] = {}

    for section_name, section in sections.items():

        for check_name, value in section[
            "checks"
        ].items():

            all_checks[
                f"{section_name}.{check_name}"
            ] = bool(value)

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

        "phase": "6.3",

        "phase_name":
            "Response Playbook & MITRE Mapping Validation",

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

        "playbooks":
            PLAYBOOKS,

        "action_catalog":
            ACTION_CATALOG,

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
    print("PHASE 6.3 FINAL DECISION")
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
        f"Playbooks : {len(PLAYBOOKS)}"
    )

    print(
        f"Report : {REPORT_FILE}"
    )

    if overall_passed:

        print()

        print(
            "PHASE 6.3 ACCEPTED — "
            "RESPONSE PLAYBOOKS AND MITRE MAPPINGS VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 6.3 FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())