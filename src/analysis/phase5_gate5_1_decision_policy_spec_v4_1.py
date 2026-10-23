from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


VERSION = "4.1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "phase5" / "gate5_1"
REPORT_FILE = REPORT_DIR / "phase5_gate5_1_decision_policy_spec_report.json"

DECISION_POLICY = {
    "version": VERSION,
    "purpose": (
        "Convert the deterministic Phase 3D RiskDecision into a formal "
        "SOC disposition without executing response or containment actions."
    ),

    "risk_level_mapping": {
        "LOW": {
            "decision": "MONITOR",
            "analyst_required": False,
            "containment_authorized": False,
        },
        "MEDIUM": {
            "decision": "ANALYST_REVIEW",
            "analyst_required": True,
            "containment_authorized": False,
        },
        "HIGH": {
            "decision": "PRIORITY_ANALYST_REVIEW",
            "analyst_required": True,
            "containment_authorized": False,
        },
        "CRITICAL": {
            "decision": "CONTAINMENT_CANDIDATE",
            "analyst_required": True,
            "containment_authorized": False,
        },
    },

    "safety_rules": {
        "automatic_response": False,
        "automatic_blocking": False,
        "automatic_isolation": False,
        "automatic_process_kill": False,
        "containment_requires_analyst": True,
        "fallback_can_directly_contain": False,
        "ambiguous_can_directly_contain": False,
    },

    "uncertainty_policy": {
        "fallback": {
            "required_disposition": "ANALYST_REVIEW",
            "reason": (
                "Fallback classification is insufficient for direct "
                "containment authorization."
            ),
        },
        "ambiguous": {
            "required_disposition": "ANALYST_REVIEW",
            "reason": (
                "Ambiguous classification requires analyst assessment "
                "before escalation or containment."
            ),
        },
        "missing_behavior": {
            "requires_caution": True,
            "reason": (
                "Missing behavioral evidence reduces decision confidence "
                "and must not independently authorize containment."
            ),
        },
    },

    "decision_precedence": [
        "SAFETY_OVERRIDE",
        "UNCERTAINTY_OVERRIDE",
        "RISK_LEVEL_MAPPING",
    ],

    "side_effects": {
        "response_actions_executed": False,
        "containment_actions_executed": False,
        "wazuh_integration": False,
        "shuffle_integration": False,
    },
}


VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

VALID_DECISIONS = {
    "MONITOR",
    "ANALYST_REVIEW",
    "PRIORITY_ANALYST_REVIEW",
    "CONTAINMENT_CANDIDATE",
}


def evaluate_policy(
    *,
    risk_score: float,
    risk_level: str,
    fallback: bool = False,
    ambiguous: bool = False,
    missing_behavior: bool = False,
) -> Dict[str, Any]:
    """
    Apply the Phase 5.1 decision policy.

    This function is intentionally side-effect free.
    It only returns a decision object.
    """

    risk_level = str(risk_level).strip().upper()

    if risk_level not in VALID_RISK_LEVELS:
        raise ValueError(f"Invalid risk level: {risk_level}")

    risk_score = float(risk_score)

    if not 0.0 <= risk_score <= 100.0:
        raise ValueError(f"Risk score outside [0,100]: {risk_score}")

    reasons: List[str] = []

    # ------------------------------------------------------------
    # Safety override
    # ------------------------------------------------------------
    # Phase 5 never authorizes automatic containment.
    containment_authorized = False
    automatic_response = False

    reasons.append(
        "Decision policy is side-effect free; no response action is executed."
    )

    # ------------------------------------------------------------
    # Uncertainty override
    # ------------------------------------------------------------
    if fallback:
        decision = "ANALYST_REVIEW"
        reasons.append(
            "Fallback classification requires analyst review."
        )
        uncertainty_override = True

    elif ambiguous:
        decision = "ANALYST_REVIEW"
        reasons.append(
            "Ambiguous classification requires analyst review."
        )
        uncertainty_override = True

    else:
        mapping = DECISION_POLICY["risk_level_mapping"][risk_level]
        decision = mapping["decision"]
        uncertainty_override = False

        reasons.append(
            f"Risk level {risk_level} maps to {decision}."
        )

    # ------------------------------------------------------------
    # Missing behavioral evidence
    # ------------------------------------------------------------
    if missing_behavior:
        reasons.append(
            "Behavioral evidence is incomplete; analyst caution is required."
        )

    # ------------------------------------------------------------
    # Analyst requirement
    # ------------------------------------------------------------
    analyst_required = decision != "MONITOR"

    if decision == "CONTAINMENT_CANDIDATE":
        analyst_required = True
        containment_authorized = False
        reasons.append(
            "Containment is only a candidate disposition and requires "
            "explicit analyst approval."
        )

    return {
        "policy_version": VERSION,
        "risk_score": round(risk_score, 6),
        "risk_level": risk_level,
        "decision": decision,
        "analyst_required": analyst_required,
        "containment_authorized": containment_authorized,
        "automatic_response": automatic_response,
        "fallback": bool(fallback),
        "ambiguous": bool(ambiguous),
        "missing_behavior": bool(missing_behavior),
        "uncertainty_override": uncertainty_override,
        "reasons": reasons,
        "side_effects": {
            "response_actions_executed": False,
            "containment_actions_executed": False,
            "wazuh_integration": False,
            "shuffle_integration": False,
        },
    }


def run_specification_tests() -> Dict[str, Any]:
    tests = []

    def check(
        name: str,
        condition: bool,
        detail: str = "",
    ) -> None:
        tests.append(
            {
                "name": name,
                "passed": bool(condition),
                "detail": detail,
            }
        )

    # ------------------------------------------------------------
    # 1. Policy schema
    # ------------------------------------------------------------
    check(
        "policy_version",
        DECISION_POLICY["version"] == VERSION,
    )

    check(
        "risk_levels_complete",
        set(DECISION_POLICY["risk_level_mapping"]) == VALID_RISK_LEVELS,
    )

    # ------------------------------------------------------------
    # 2. Risk-level mapping
    # ------------------------------------------------------------
    expected_mapping = {
        "LOW": "MONITOR",
        "MEDIUM": "ANALYST_REVIEW",
        "HIGH": "PRIORITY_ANALYST_REVIEW",
        "CRITICAL": "CONTAINMENT_CANDIDATE",
    }

    actual_mapping = {
        level: value["decision"]
        for level, value in DECISION_POLICY["risk_level_mapping"].items()
    }

    check(
        "risk_level_mapping",
        actual_mapping == expected_mapping,
        json.dumps(actual_mapping, sort_keys=True),
    )

    # ------------------------------------------------------------
    # 3. Low-risk behavior
    # ------------------------------------------------------------
    low = evaluate_policy(
        risk_score=15,
        risk_level="LOW",
    )

    check(
        "low_maps_to_monitor",
        low["decision"] == "MONITOR",
    )

    check(
        "low_does_not_require_analyst",
        low["analyst_required"] is False,
    )

    # ------------------------------------------------------------
    # 4. Medium-risk behavior
    # ------------------------------------------------------------
    medium = evaluate_policy(
        risk_score=40,
        risk_level="MEDIUM",
    )

    check(
        "medium_maps_to_analyst_review",
        medium["decision"] == "ANALYST_REVIEW",
    )

    check(
        "medium_requires_analyst",
        medium["analyst_required"] is True,
    )

    # ------------------------------------------------------------
    # 5. High-risk behavior
    # ------------------------------------------------------------
    high = evaluate_policy(
        risk_score=65,
        risk_level="HIGH",
    )

    check(
        "high_maps_to_priority_review",
        high["decision"] == "PRIORITY_ANALYST_REVIEW",
    )

    check(
        "high_requires_analyst",
        high["analyst_required"] is True,
    )

    # ------------------------------------------------------------
    # 6. Critical behavior
    # ------------------------------------------------------------
    critical = evaluate_policy(
        risk_score=90,
        risk_level="CRITICAL",
    )

    check(
        "critical_maps_to_containment_candidate",
        critical["decision"] == "CONTAINMENT_CANDIDATE",
    )

    check(
        "critical_requires_analyst",
        critical["analyst_required"] is True,
    )

    check(
        "critical_does_not_authorize_containment",
        critical["containment_authorized"] is False,
    )

    # ------------------------------------------------------------
    # 7. Fallback safety
    # ------------------------------------------------------------
    fallback = evaluate_policy(
        risk_score=90,
        risk_level="CRITICAL",
        fallback=True,
    )

    check(
        "fallback_forces_analyst_review",
        fallback["decision"] == "ANALYST_REVIEW",
    )

    check(
        "fallback_cannot_contain",
        fallback["containment_authorized"] is False,
    )

    # ------------------------------------------------------------
    # 8. Ambiguity safety
    # ------------------------------------------------------------
    ambiguous = evaluate_policy(
        risk_score=90,
        risk_level="CRITICAL",
        ambiguous=True,
    )

    check(
        "ambiguous_forces_analyst_review",
        ambiguous["decision"] == "ANALYST_REVIEW",
    )

    check(
        "ambiguous_cannot_contain",
        ambiguous["containment_authorized"] is False,
    )

    # ------------------------------------------------------------
    # 9. Missing behavior
    # ------------------------------------------------------------
    missing = evaluate_policy(
        risk_score=75,
        risk_level="CRITICAL",
        missing_behavior=True,
    )

    check(
        "missing_behavior_is_recorded",
        missing["missing_behavior"] is True,
    )

    check(
        "missing_behavior_no_automatic_response",
        missing["automatic_response"] is False,
    )

    # ------------------------------------------------------------
    # 10. Side-effect safety
    # ------------------------------------------------------------
    for result in [low, medium, high, critical, fallback, ambiguous, missing]:
        check(
            "no_response_side_effect",
            result["side_effects"]["response_actions_executed"] is False,
        )

        check(
            "no_containment_side_effect",
            result["side_effects"]["containment_actions_executed"] is False,
        )

        check(
            "no_wazuh_integration",
            result["side_effects"]["wazuh_integration"] is False,
        )

        check(
            "no_shuffle_integration",
            result["side_effects"]["shuffle_integration"] is False,
        )

    # ------------------------------------------------------------
    # 11. Determinism
    # ------------------------------------------------------------
    first = evaluate_policy(
        risk_score=88.6,
        risk_level="CRITICAL",
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    second = evaluate_policy(
        risk_score=88.6,
        risk_level="CRITICAL",
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    check(
        "deterministic_output",
        first == second,
    )

    # ------------------------------------------------------------
    # 12. Monotonic risk disposition
    # ------------------------------------------------------------
    order = {
        "MONITOR": 0,
        "ANALYST_REVIEW": 1,
        "PRIORITY_ANALYST_REVIEW": 2,
        "CONTAINMENT_CANDIDATE": 3,
    }

    sequence = [
        evaluate_policy(risk_score=10, risk_level="LOW")["decision"],
        evaluate_policy(risk_score=35, risk_level="MEDIUM")["decision"],
        evaluate_policy(risk_score=60, risk_level="HIGH")["decision"],
        evaluate_policy(risk_score=90, risk_level="CRITICAL")["decision"],
    ]

    numeric_sequence = [order[x] for x in sequence]

    check(
        "risk_disposition_monotonic",
        numeric_sequence == sorted(numeric_sequence),
        json.dumps(sequence),
    )

    passed = sum(1 for t in tests if t["passed"])
    failed = len(tests) - passed

    return {
        "tests": tests,
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SmartSOC Phase 5.1")
    print("Decision Policy Specification")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Report:       {REPORT_FILE}")
    print()

    test_results = run_specification_tests()

    for test in test_results["tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"[{status}] {test['name']}")

    report = {
        "phase": "5.1",
        "phase_name": "Decision Policy Specification",
        "version": VERSION,
        "status": test_results["status"],
        "policy": DECISION_POLICY,
        "test_results": test_results,
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
            if test_results["status"] == "PASS"
            else "REVIEW_REQUIRED"
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PHASE 5.1 FINAL DECISION")
    print("=" * 72)
    print(f"Status : {test_results['status']}")
    print(f"Tests  : {test_results['total']} total / {test_results['failed']} failed")
    print(f"Report : {REPORT_FILE}")

    if test_results["status"] == "PASS":
        print()
        print("PHASE 5.1 ACCEPTED")
        return 0

    print()
    print("PHASE 5.1 FAILED — DO NOT PROCEED")
    return 1


if __name__ == "__main__":
    sys.exit(main())