from __future__ import annotations

import sys

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


# =====================================================================
# SmartSOC Phase 6.2
# Deterministic Response Planner
# Version 4.1
#
# Purpose:
#   Convert the validated SmartSOC decision/risk result into a
#   structured response PLAN.
#
# IMPORTANT:
#   This component is planning-only.
#
#   It does NOT:
#       - execute commands
#       - kill processes
#       - isolate hosts
#       - block IP addresses
#       - disable accounts
#       - modify firewalls
#       - delete files
#       - connect to Wazuh
#       - connect to Shuffle
#
#   All containment remains analyst-authorized only.
# =====================================================================


VERSION = "4.1"


# =====================================================================
# Response action metadata
# =====================================================================

ACTION_CATALOG: Dict[str, Dict[str, Any]] = {

    "COLLECT_ADDITIONAL_EVIDENCE": {
        "category": "INVESTIGATE",
        "description": (
            "Collect additional telemetry and contextual evidence "
            "before taking disruptive action."
        ),
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_PROCESS_TREE": {
        "category": "INVESTIGATE",
        "description": (
            "Review process ancestry, child processes, command lines, "
            "users, integrity levels, and execution context."
        ),
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_NETWORK_CONNECTIONS": {
        "category": "INVESTIGATE",
        "description": (
            "Review source/destination addresses, ports, protocols, "
            "and suspicious network relationships."
        ),
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "REVIEW_PERSISTENCE_MECHANISM": {
        "category": "INVESTIGATE",
        "description": (
            "Review services, scheduled tasks, registry persistence, "
            "and other persistence mechanisms."
        ),
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "PRESERVE_ARTIFACTS": {
        "category": "INVESTIGATE",
        "description": (
            "Preserve relevant files, hashes, command lines, logs, "
            "and other forensic artifacts."
        ),
        "requires_approval": True,
        "destructive": False,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_PROCESS_ISOLATION": {
        "category": "PREPARE_RESPONSE",
        "description": (
            "Prepare a process-isolation option for analyst approval. "
            "No process action is executed by this planner."
        ),
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_HOST_ISOLATION": {
        "category": "PREPARE_CONTAINMENT",
        "description": (
            "Prepare a host-isolation option for analyst approval. "
            "No network isolation is executed by this planner."
        ),
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_IP_BLOCK": {
        "category": "PREPARE_CONTAINMENT",
        "description": (
            "Prepare an IP-blocking option for analyst approval. "
            "No firewall or network block is executed by this planner."
        ),
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },

    "PREPARE_ACCOUNT_DISABLE": {
        "category": "PREPARE_CONTAINMENT",
        "description": (
            "Prepare an account-disable option for analyst approval. "
            "No account modification is executed by this planner."
        ),
        "requires_approval": True,
        "destructive": True,
        "reversible": True,
        "automatic": False,
    },
}


# =====================================================================
# Technique-specific planning hints
# =====================================================================

TECHNIQUE_ACTIONS: Dict[str, List[str]] = {

    "T1059": [
        "REVIEW_PROCESS_TREE",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1059.001": [
        "REVIEW_PROCESS_TREE",
        "COLLECT_ADDITIONAL_EVIDENCE",
        "PRESERVE_ARTIFACTS",
    ],

    "T1059.003": [
        "REVIEW_PROCESS_TREE",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1105": [
        "REVIEW_NETWORK_CONNECTIONS",
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1570": [
        "REVIEW_NETWORK_CONNECTIONS",
        "REVIEW_PROCESS_TREE",
        "PRESERVE_ARTIFACTS",
    ],

    "T1027.004": [
        "PRESERVE_ARTIFACTS",
        "REVIEW_PROCESS_TREE",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1543.003": [
        "REVIEW_PERSISTENCE_MECHANISM",
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1112": [
        "REVIEW_PERSISTENCE_MECHANISM",
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1053.005": [
        "REVIEW_PERSISTENCE_MECHANISM",
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1070.004": [
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
    ],

    "T1078": [
        "COLLECT_ADDITIONAL_EVIDENCE",
        "REVIEW_PROCESS_TREE",
        "REVIEW_NETWORK_CONNECTIONS",
    ],

    "T1485": [
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
        "PREPARE_HOST_ISOLATION",
    ],

    "T1565.001": [
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
        "PREPARE_HOST_ISOLATION",
    ],

    "T1531": [
        "COLLECT_ADDITIONAL_EVIDENCE",
        "PREPARE_ACCOUNT_DISABLE",
    ],

    "T1562.001": [
        "PRESERVE_ARTIFACTS",
        "COLLECT_ADDITIONAL_EVIDENCE",
        "PREPARE_HOST_ISOLATION",
    ],
}


# =====================================================================
# Safety constants
# =====================================================================

SAFETY_POLICY: Dict[str, bool] = {

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
# Result dataclass
# =====================================================================

@dataclass(frozen=True)
class ResponsePlan:

    planner_version: str

    risk_score: float

    risk_level: str

    decision: str

    response_mode: str

    priority: str

    recommended_actions: List[str]

    analyst_required: bool

    containment_candidate: bool

    containment_authorized: bool

    rollback_required: bool

    fallback: bool

    ambiguous: bool

    missing_behavior: bool

    reasons: List[str]

    safety: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# Planner
# =====================================================================

class SmartSOCResponsePlanner:
    """
    Deterministic, side-effect-free SmartSOC response planner.
    """

    VERSION = VERSION

    VALID_RISK_LEVELS = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    VALID_DECISIONS = {
        "MONITOR",
        "ANALYST_REVIEW",
        "PRIORITY_ANALYST_REVIEW",
        "CONTAINMENT_CANDIDATE",
    }

    def __init__(self) -> None:

        self.action_catalog = ACTION_CATALOG
        self.technique_actions = TECHNIQUE_ACTIONS

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_risk_level(
        risk_level: str,
    ) -> str:

        if not isinstance(
            risk_level,
            str,
        ):
            raise TypeError(
                "risk_level must be a string"
            )

        normalized = risk_level.strip().upper()

        if normalized not in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }:
            raise ValueError(
                f"Invalid risk level: {risk_level}"
            )

        return normalized

    @staticmethod
    def _normalize_score(
        risk_score: float,
    ) -> float:

        if isinstance(
            risk_score,
            bool,
        ):
            raise TypeError(
                "risk_score must be numeric"
            )

        try:
            score = float(risk_score)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                "risk_score must be numeric"
            ) from exc

        if score != score:
            raise ValueError(
                "risk_score cannot be NaN"
            )

        if score == float("inf"):
            raise ValueError(
                "risk_score cannot be +inf"
            )

        if score == float("-inf"):
            raise ValueError(
                "risk_score cannot be -inf"
            )

        if score < 0.0 or score > 100.0:
            raise ValueError(
                "risk_score must be between 0 and 100"
            )

        return score

    @staticmethod
    def _normalize_bool(
        value: Any,
        name: str,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            int,
        ) and value in (0, 1):
            return bool(value)

        raise TypeError(
            f"{name} must be boolean"
        )

    # -----------------------------------------------------------------
    # Priority
    # -----------------------------------------------------------------

    @staticmethod
    def _priority_for_risk(
        risk_level: str,
    ) -> str:

        mapping = {

            "LOW":
                "LOW",

            "MEDIUM":
                "MEDIUM",

            "HIGH":
                "HIGH",

            "CRITICAL":
                "CRITICAL",
        }

        return mapping[risk_level]

    # -----------------------------------------------------------------
    # Base response mode
    # -----------------------------------------------------------------

    @staticmethod
    def _response_mode_for_risk(
        risk_level: str,
    ) -> str:

        mapping = {

            "LOW":
                "OBSERVE_ONLY",

            "MEDIUM":
                "INVESTIGATE",

            "HIGH":
                "PREPARE_RESPONSE",

            "CRITICAL":
                "PREPARE_CONTAINMENT",
        }

        return mapping[risk_level]

    # -----------------------------------------------------------------
    # Base decision
    # -----------------------------------------------------------------

    @staticmethod
    def _decision_for_risk(
        risk_level: str,
    ) -> str:

        mapping = {

            "LOW":
                "MONITOR",

            "MEDIUM":
                "ANALYST_REVIEW",

            "HIGH":
                "PRIORITY_ANALYST_REVIEW",

            "CRITICAL":
                "CONTAINMENT_CANDIDATE",
        }

        return mapping[risk_level]

    # -----------------------------------------------------------------
    # Technique actions
    # -----------------------------------------------------------------

    def _technique_recommendations(
        self,
        techniques: Optional[List[str]],
    ) -> List[str]:

        if techniques is None:
            return []

        if not isinstance(
            techniques,
            list,
        ):
            raise TypeError(
                "techniques must be a list"
            )

        recommendations: List[str] = []

        for technique in techniques:

            if not isinstance(
                technique,
                str,
            ):
                raise TypeError(
                    "technique identifiers must be strings"
                )

            normalized = technique.strip()

            actions = self.technique_actions.get(
                normalized,
                [],
            )

            for action in actions:

                if action not in recommendations:
                    recommendations.append(
                        action
                    )

        return recommendations

    # -----------------------------------------------------------------
    # Safety action filtering
    # -----------------------------------------------------------------

    @staticmethod
    def _filter_actions_for_safety(
        actions: List[str],
        risk_level: str,
        fallback: bool,
        ambiguous: bool,
        missing_behavior: bool,
    ) -> List[str]:

        filtered: List[str] = []

        for action in actions:

            if action not in ACTION_CATALOG:
                continue

            metadata = ACTION_CATALOG[action]

            # No automatic execution is ever allowed.
            if metadata["automatic"] is True:
                continue

            # Fallback and ambiguity may only produce analyst review
            # and evidence-oriented recommendations.
            if fallback or ambiguous:

                if metadata["category"] not in {
                    "INVESTIGATE",
                }:
                    continue

            # Missing behavioral evidence requires evidence collection
            # before any disruptive plan.
            if missing_behavior:

                if action in {
                    "PREPARE_PROCESS_ISOLATION",
                    "PREPARE_HOST_ISOLATION",
                    "PREPARE_IP_BLOCK",
                    "PREPARE_ACCOUNT_DISABLE",
                }:
                    continue

            # Containment preparation is only valid at CRITICAL.
            if action.startswith(
                "PREPARE_"
            ):

                if risk_level != "CRITICAL":
                    continue

            filtered.append(action)

        return filtered

    # -----------------------------------------------------------------
    # Main evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        *,
        risk_score: float,
        risk_level: str,
        decision: Optional[str] = None,
        techniques: Optional[List[str]] = None,
        fallback: bool = False,
        ambiguous: bool = False,
        missing_behavior: bool = False,
    ) -> ResponsePlan:

        score = self._normalize_score(
            risk_score
        )

        level = self._normalize_risk_level(
            risk_level
        )

        fallback_value = self._normalize_bool(
            fallback,
            "fallback",
        )

        ambiguous_value = self._normalize_bool(
            ambiguous,
            "ambiguous",
        )

        missing_behavior_value = self._normalize_bool(
            missing_behavior,
            "missing_behavior",
        )

        expected_decision = (
            self._decision_for_risk(level)
        )

        if decision is None:

            final_decision = (
                expected_decision
            )

        else:

            if not isinstance(
                decision,
                str,
            ):
                raise TypeError(
                    "decision must be a string"
                )

            final_decision = (
                decision.strip().upper()
            )

            if final_decision not in self.VALID_DECISIONS:
                raise ValueError(
                    f"Invalid decision: {decision}"
                )

        # -------------------------------------------------------------
        # Start with risk-based recommendations
        # -------------------------------------------------------------

        actions = self._technique_recommendations(
            techniques
        )

        reasons: List[str] = []

        # Always provide at least one meaningful recommendation.
        if not actions:

            if level == "LOW":

                actions = [
                    "COLLECT_ADDITIONAL_EVIDENCE"
                ]

            elif level == "MEDIUM":

                actions = [
                    "COLLECT_ADDITIONAL_EVIDENCE",
                    "REVIEW_PROCESS_TREE",
                ]

            elif level == "HIGH":

                actions = [
                    "COLLECT_ADDITIONAL_EVIDENCE",
                    "REVIEW_PROCESS_TREE",
                    "PRESERVE_ARTIFACTS",
                ]

            else:

                actions = [
                    "COLLECT_ADDITIONAL_EVIDENCE",
                    "PRESERVE_ARTIFACTS",
                ]

        # -------------------------------------------------------------
        # Critical risk
        # -------------------------------------------------------------

        containment_candidate = (
            level == "CRITICAL"
        )

        analyst_required = (
            level != "LOW"
        )

        rollback_required = (
            level in {
                "HIGH",
                "CRITICAL",
            }
        )

        response_mode = (
            self._response_mode_for_risk(level)
        )

        # -------------------------------------------------------------
        # Uncertainty safety escalation
        # -------------------------------------------------------------

        if fallback_value:

            analyst_required = True

            reasons.append(
                "Fallback inference requires analyst review."
            )

            response_mode = (
                "INVESTIGATE"
            )

            final_decision = (
                "ANALYST_REVIEW"
            )

            containment_candidate = False

        if ambiguous_value:

            analyst_required = True

            reasons.append(
                "Ambiguous detection requires analyst review."
            )

            response_mode = (
                "INVESTIGATE"
            )

            final_decision = (
                "ANALYST_REVIEW"
            )

            containment_candidate = False

        if missing_behavior_value:

            reasons.append(
                "Behavioral evidence is missing; "
                "additional evidence should be collected."
            )

        # -------------------------------------------------------------
        # Critical containment candidate
        # -------------------------------------------------------------

        if (
            level == "CRITICAL"
            and not fallback_value
            and not ambiguous_value
        ):

            containment_candidate = True

            analyst_required = True

            reasons.append(
                "Critical risk permits preparation of a "
                "containment candidate."
            )

            reasons.append(
                "Containment remains analyst-authorized only."
            )

            if (
                "PREPARE_HOST_ISOLATION"
                not in actions
            ):

                actions.append(
                    "PREPARE_HOST_ISOLATION"
                )

        # -------------------------------------------------------------
        # HIGH response preparation
        # -------------------------------------------------------------

        if level == "HIGH":

            analyst_required = True

            reasons.append(
                "High risk requires priority analyst review "
                "before disruptive response."
            )

        # -------------------------------------------------------------
        # MEDIUM investigation
        # -------------------------------------------------------------

        if level == "MEDIUM":

            analyst_required = True

            reasons.append(
                "Medium risk requires analyst investigation."
            )

        # -------------------------------------------------------------
        # LOW monitoring
        # -------------------------------------------------------------

        if level == "LOW":

            analyst_required = False

            reasons.append(
                "Low risk remains in monitor-only disposition."
            )

        # -------------------------------------------------------------
        # Missing behavior removes disruptive actions
        # -------------------------------------------------------------

        actions = self._filter_actions_for_safety(
            actions=actions,
            risk_level=level,
            fallback=fallback_value,
            ambiguous=ambiguous_value,
            missing_behavior=missing_behavior_value,
        )

        # -------------------------------------------------------------
        # Safety rule:
        # containment is never authorized by planner.
        # -------------------------------------------------------------

        containment_authorized = False

        # -------------------------------------------------------------
        # Safety rule:
        # fallback/ambiguity cannot be containment candidates.
        # -------------------------------------------------------------

        if fallback_value or ambiguous_value:

            containment_candidate = False

        # -------------------------------------------------------------
        # If no actions remain, use evidence collection.
        # -------------------------------------------------------------

        if not actions:

            actions = [
                "COLLECT_ADDITIONAL_EVIDENCE"
            ]

        # -------------------------------------------------------------
        # Deduplicate while preserving order.
        # -------------------------------------------------------------

        actions = list(
            dict.fromkeys(actions)
        )

        # -------------------------------------------------------------
        # Final deterministic safety record.
        # -------------------------------------------------------------

        safety = dict(
            SAFETY_POLICY
        )

        # -------------------------------------------------------------
        # Final consistency checks.
        # -------------------------------------------------------------

        if containment_authorized:

            raise RuntimeError(
                "Safety violation: containment cannot be authorized."
            )

        if any(
            ACTION_CATALOG[action]["automatic"]
            for action in actions
        ):

            raise RuntimeError(
                "Safety violation: automatic action detected."
            )

        if (
            fallback_value
            and containment_candidate
        ):

            raise RuntimeError(
                "Safety violation: fallback containment candidate."
            )

        if (
            ambiguous_value
            and containment_candidate
        ):

            raise RuntimeError(
                "Safety violation: ambiguous containment candidate."
            )

        return ResponsePlan(

            planner_version=self.VERSION,

            risk_score=score,

            risk_level=level,

            decision=final_decision,

            response_mode=response_mode,

            priority=self._priority_for_risk(level),

            recommended_actions=actions,

            analyst_required=analyst_required,

            containment_candidate=containment_candidate,

            containment_authorized=False,

            rollback_required=rollback_required,

            fallback=fallback_value,

            ambiguous=ambiguous_value,

            missing_behavior=missing_behavior_value,

            reasons=reasons,

            safety=safety,
        )

    # -----------------------------------------------------------------
    # Evaluate from a decision-policy result
    # -----------------------------------------------------------------

    def evaluate_decision(
        self,
        decision_result: Any,
        *,
        techniques: Optional[List[str]] = None,
    ) -> ResponsePlan:

        required = [
            "risk_score",
            "risk_level",
            "decision",
            "fallback",
            "ambiguous",
            "missing_behavior",
        ]

        for field in required:

            if not hasattr(
                decision_result,
                field,
            ):

                raise AttributeError(
                    "Decision result missing field: "
                    + field
                )

        return self.evaluate(

            risk_score=(
                decision_result.risk_score
            ),

            risk_level=(
                decision_result.risk_level
            ),

            decision=(
                decision_result.decision
            ),

            techniques=techniques,

            fallback=(
                decision_result.fallback
            ),

            ambiguous=(
                decision_result.ambiguous
            ),

            missing_behavior=(
                decision_result.missing_behavior
            ),
        )

    # -----------------------------------------------------------------
    # Evaluate event dictionary
    # -----------------------------------------------------------------

    def evaluate_event(
        self,
        event: Dict[str, Any],
    ) -> ResponsePlan:

        if not isinstance(
            event,
            dict,
        ):

            raise TypeError(
                "event must be a dictionary"
            )

        return self.evaluate(

            risk_score=event[
                "risk_score"
            ],

            risk_level=event[
                "risk_level"
            ],

            decision=event.get(
                "decision"
            ),

            techniques=event.get(
                "techniques"
            ),

            fallback=event.get(
                "fallback",
                False,
            ),

            ambiguous=event.get(
                "ambiguous",
                False,
            ),

            missing_behavior=event.get(
                "missing_behavior",
                False,
            ),
        )

    # -----------------------------------------------------------------
    # Safety status
    # -----------------------------------------------------------------

    def safety_status(self) -> Dict[str, bool]:

        return dict(
            SAFETY_POLICY
        )


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> Dict[str, bool]:

    planner = SmartSOCResponsePlanner()

    checks: Dict[str, bool] = {}

    # ---------------------------------------------------------------
    # LOW
    # ---------------------------------------------------------------

    low = planner.evaluate(
        risk_score=10,
        risk_level="LOW",
        techniques=["T1059.003"],
    )

    checks[
        "low_monitor"
    ] = (
        low.decision == "MONITOR"
    )

    checks[
        "low_not_analyst_required"
    ] = (
        low.analyst_required is False
    )

    checks[
        "low_no_containment"
    ] = (
        low.containment_candidate is False
    )

    # ---------------------------------------------------------------
    # MEDIUM
    # ---------------------------------------------------------------

    medium = planner.evaluate(
        risk_score=40,
        risk_level="MEDIUM",
        techniques=["T1112"],
    )

    checks[
        "medium_analyst_review"
    ] = (
        medium.decision == "ANALYST_REVIEW"
    )

    checks[
        "medium_requires_analyst"
    ] = (
        medium.analyst_required is True
    )

    checks[
        "medium_no_containment"
    ] = (
        medium.containment_candidate is False
    )

    # ---------------------------------------------------------------
    # HIGH
    # ---------------------------------------------------------------

    high = planner.evaluate(
        risk_score=65,
        risk_level="HIGH",
        techniques=["T1543.003"],
    )

    checks[
        "high_priority_review"
    ] = (
        high.decision
        == "PRIORITY_ANALYST_REVIEW"
    )

    checks[
        "high_requires_analyst"
    ] = (
        high.analyst_required is True
    )

    checks[
        "high_rollback_required"
    ] = (
        high.rollback_required is True
    )

    checks[
        "high_no_containment"
    ] = (
        high.containment_candidate is False
    )

    # ---------------------------------------------------------------
    # CRITICAL
    # ---------------------------------------------------------------

    critical = planner.evaluate(
        risk_score=90,
        risk_level="CRITICAL",
        techniques=["T1485"],
    )

    checks[
        "critical_candidate"
    ] = (
        critical.containment_candidate is True
    )

    checks[
        "critical_requires_analyst"
    ] = (
        critical.analyst_required is True
    )

    checks[
        "critical_not_authorized"
    ] = (
        critical.containment_authorized is False
    )

    checks[
        "critical_rollback_required"
    ] = (
        critical.rollback_required is True
    )

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    fallback = planner.evaluate(
        risk_score=95,
        risk_level="CRITICAL",
        techniques=["T1485"],
        fallback=True,
    )

    checks[
        "fallback_forces_review"
    ] = (
        fallback.decision
        == "ANALYST_REVIEW"
    )

    checks[
        "fallback_no_containment"
    ] = (
        fallback.containment_candidate is False
    )

    # ---------------------------------------------------------------
    # Ambiguous
    # ---------------------------------------------------------------

    ambiguous = planner.evaluate(
        risk_score=95,
        risk_level="CRITICAL",
        techniques=["T1485"],
        ambiguous=True,
    )

    checks[
        "ambiguous_forces_review"
    ] = (
        ambiguous.decision
        == "ANALYST_REVIEW"
    )

    checks[
        "ambiguous_no_containment"
    ] = (
        ambiguous.containment_candidate is False
    )

    # ---------------------------------------------------------------
    # Missing behavior
    # ---------------------------------------------------------------

    missing = planner.evaluate(
        risk_score=90,
        risk_level="CRITICAL",
        techniques=["T1485"],
        missing_behavior=True,
    )

    checks[
        "missing_behavior_recorded"
    ] = (
        missing.missing_behavior is True
    )

    checks[
        "missing_behavior_no_host_isolation_plan"
    ] = (
        "PREPARE_HOST_ISOLATION"
        not in missing.recommended_actions
    )

    # ---------------------------------------------------------------
    # Safety
    # ---------------------------------------------------------------

    safety = planner.safety_status()

    checks[
        "automatic_response_disabled"
    ] = (
        safety["automatic_response"]
        is False
    )

    checks[
        "automatic_blocking_disabled"
    ] = (
        safety["automatic_blocking"]
        is False
    )

    checks[
        "automatic_isolation_disabled"
    ] = (
        safety["automatic_isolation"]
        is False
    )

    checks[
        "automatic_process_kill_disabled"
    ] = (
        safety["automatic_process_kill"]
        is False
    )

    checks[
        "containment_not_authorized"
    ] = (
        safety["containment_authorized"]
        is False
    )

    checks[
        "containment_requires_analyst"
    ] = (
        safety["containment_requires_analyst"]
        is True
    )

    checks[
        "wazuh_disabled"
    ] = (
        safety["wazuh_integration"]
        is False
    )

    checks[
        "shuffle_disabled"
    ] = (
        safety["shuffle_integration"]
        is False
    )

    checks[
        "no_response_executed"
    ] = (
        safety["response_actions_executed"]
        is False
    )

    checks[
        "no_containment_executed"
    ] = (
        safety["containment_actions_executed"]
        is False
    )

    # ---------------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------------

    first = planner.evaluate(
        risk_score=88.5,
        risk_level="CRITICAL",
        techniques=[
            "T1485",
            "T1105",
        ],
    )

    second = planner.evaluate(
        risk_score=88.5,
        risk_level="CRITICAL",
        techniques=[
            "T1485",
            "T1105",
        ],
    )

    checks[
        "deterministic_output"
    ] = (
        first == second
    )

    # ---------------------------------------------------------------
    # Invalid inputs
    # ---------------------------------------------------------------

    invalid_tests = [

        lambda: planner.evaluate(
            risk_score=-1,
            risk_level="LOW",
        ),

        lambda: planner.evaluate(
            risk_score=101,
            risk_level="CRITICAL",
        ),

        lambda: planner.evaluate(
            risk_score=float("nan"),
            risk_level="LOW",
        ),

        lambda: planner.evaluate(
            risk_score=10,
            risk_level="INVALID",
        ),

    ]

    invalid_rejected = 0

    for test in invalid_tests:

        try:
            test()

        except Exception:
            invalid_rejected += 1

    checks[
        "invalid_inputs_rejected"
    ] = (
        invalid_rejected
        == len(invalid_tests)
    )

    return checks


# =====================================================================
# CLI
# =====================================================================

def main() -> int:

    print("=" * 72)
    print("SmartSOC Phase 6.2")
    print("Deterministic Response Planner")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    try:

        checks = self_test()

    except Exception as exc:

        print(
            "[FAIL] Response planner self-test"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    for name, passed in checks.items():

        print(
            "["
            + (
                "PASS"
                if passed
                else "FAIL"
            )
            + "] "
            + name
        )

    total = len(checks)

    passed = sum(
        1
        for value in checks.values()
        if value
    )

    failed = (
        total - passed
    )

    print()

    print("=" * 72)
    print("PHASE 6.2 ENGINE SELF-TEST")
    print("=" * 72)

    print(
        f"Status : "
        f"{'PASS' if failed == 0 else 'FAIL'}"
    )

    print(
        f"Tests  : "
        f"{total} total / "
        f"{failed} failed"
    )

    if failed == 0:

        print()

        print(
            "PHASE 6.2 ENGINE ACCEPTED — "
            "DETERMINISTIC RESPONSE PLANNER VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 6.2 ENGINE FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())