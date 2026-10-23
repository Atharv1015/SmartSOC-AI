from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict


# =====================================================================
# SmartSOC Phase 6.4
# Analyst Authorization Boundary
# Version 4.1
#
# Purpose:
#   Enforce a hard separation between:
#
#       AI decision / response recommendation
#                         ↓
#                 Analyst authorization
#                         ↓
#                  Future execution
#
# Safety:
#   This component NEVER executes a response.
#
#   It does NOT:
#       - kill processes
#       - isolate hosts
#       - block IPs
#       - disable accounts
#       - modify firewalls
#       - delete files
#       - call Wazuh
#       - call Shuffle
#
#   It only records whether a proposed response plan has received
#   explicit analyst authorization.
# =====================================================================


VERSION = "4.1"


# =====================================================================
# Safety policy
# =====================================================================

SAFETY_POLICY: Dict[str, bool] = {

    # Automatic execution is forbidden.
    "automatic_response": False,
    "automatic_blocking": False,
    "automatic_isolation": False,
    "automatic_process_kill": False,
    "automatic_account_disable": False,
    "automatic_firewall_change": False,
    "automatic_file_deletion": False,

    # AI cannot authorize containment.
    "ai_can_authorize_containment": False,

    # Analyst authorization is mandatory for containment.
    "analyst_required_for_containment": True,

    # Uncertain detections cannot bypass the analyst boundary.
    "fallback_can_bypass_analyst": False,
    "ambiguous_can_bypass_analyst": False,
    "missing_behavior_can_bypass_analyst": False,

    # No integrations in this phase.
    "wazuh_integration": False,
    "shuffle_integration": False,

    # This component executes nothing.
    "response_actions_executed": False,
    "containment_actions_executed": False,
}


# =====================================================================
# Valid states
# =====================================================================

VALID_DECISIONS = {
    "MONITOR",
    "ANALYST_REVIEW",
    "PRIORITY_ANALYST_REVIEW",
    "CONTAINMENT_CANDIDATE",
}


VALID_AUTHORIZATION_STATES = {
    "NOT_REQUIRED",
    "PENDING",
    "APPROVED",
    "REJECTED",
}


# =====================================================================
# Result
# =====================================================================

@dataclass(frozen=True)
class AuthorizationResult:

    authorization_version: str

    risk_score: float

    risk_level: str

    decision: str

    fallback: bool

    ambiguous: bool

    missing_behavior: bool

    containment_candidate: bool

    analyst_required: bool

    authorization_state: str

    execution_permitted: bool

    containment_authorized: bool

    reasons: tuple[str, ...]

    safety: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


# =====================================================================
# Authorization engine
# =====================================================================

class SmartSOCAnalystAuthorization:

    VERSION = VERSION

    def __init__(self) -> None:

        self.safety = dict(
            SAFETY_POLICY
        )

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------

    @staticmethod
    def _score(
        value: Any,
    ) -> float:

        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                "risk_score must be numeric"
            )

        try:

            score = float(value)

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

        if score in (
            float("inf"),
            float("-inf"),
        ):

            raise ValueError(
                "risk_score must be finite"
            )

        if not 0.0 <= score <= 100.0:

            raise ValueError(
                "risk_score must be between 0 and 100"
            )

        return score

    @staticmethod
    def _risk_level(
        value: Any,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                "risk_level must be a string"
            )

        level = value.strip().upper()

        if level not in {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }:

            raise ValueError(
                f"Invalid risk level: {value}"
            )

        return level

    @staticmethod
    def _decision(
        value: Any,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                "decision must be a string"
            )

        decision = value.strip().upper()

        if decision not in VALID_DECISIONS:

            raise ValueError(
                f"Invalid decision: {value}"
            )

        return decision

    @staticmethod
    def _boolean(
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

    @staticmethod
    def _authorization_state(
        value: Any,
    ) -> str:

        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                "authorization_state must be a string"
            )

        state = value.strip().upper()

        if state not in VALID_AUTHORIZATION_STATES:

            raise ValueError(
                f"Invalid authorization state: {value}"
            )

        return state

    # -----------------------------------------------------------------
    # Determine whether analyst review is mandatory
    # -----------------------------------------------------------------

    @staticmethod
    def _requires_analyst(
        *,
        risk_level: str,
        decision: str,
        fallback: bool,
        ambiguous: bool,
        missing_behavior: bool,
        containment_candidate: bool,
    ) -> bool:

        if (
            fallback
            or ambiguous
            or missing_behavior
        ):

            return True

        if containment_candidate:

            return True

        if risk_level in {
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }:

            return True

        if decision in {
            "ANALYST_REVIEW",
            "PRIORITY_ANALYST_REVIEW",
            "CONTAINMENT_CANDIDATE",
        }:

            return True

        return False

    # -----------------------------------------------------------------
    # Main authorization decision
    # -----------------------------------------------------------------

    def evaluate(
        self,
        *,
        risk_score: float,
        risk_level: str,
        decision: str,
        fallback: bool = False,
        ambiguous: bool = False,
        missing_behavior: bool = False,
        containment_candidate: bool = False,
        analyst_authorization: str = "NOT_REQUIRED",
    ) -> AuthorizationResult:

        score = self._score(
            risk_score
        )

        level = self._risk_level(
            risk_level
        )

        normalized_decision = self._decision(
            decision
        )

        fallback_value = self._boolean(
            fallback,
            "fallback",
        )

        ambiguous_value = self._boolean(
            ambiguous,
            "ambiguous",
        )

        missing_behavior_value = self._boolean(
            missing_behavior,
            "missing_behavior",
        )

        candidate_value = self._boolean(
            containment_candidate,
            "containment_candidate",
        )

        authorization = self._authorization_state(
            analyst_authorization
        )

        # -------------------------------------------------------------
        # Hard safety boundary:
        # AI cannot authorize containment.
        # -------------------------------------------------------------

        if candidate_value:

            analyst_required = True

        else:

            analyst_required = self._requires_analyst(
                risk_level=level,
                decision=normalized_decision,
                fallback=fallback_value,
                ambiguous=ambiguous_value,
                missing_behavior=missing_behavior_value,
                containment_candidate=candidate_value,
            )

        reasons = []

        # -------------------------------------------------------------
        # Fallback
        # -------------------------------------------------------------

        if fallback_value:

            reasons.append(
                "Fallback detection requires analyst review."
            )

        # -------------------------------------------------------------
        # Ambiguity
        # -------------------------------------------------------------

        if ambiguous_value:

            reasons.append(
                "Ambiguous detection requires analyst review."
            )

        # -------------------------------------------------------------
        # Missing behavior
        # -------------------------------------------------------------

        if missing_behavior_value:

            reasons.append(
                "Missing behavioral evidence requires analyst review."
            )

        # -------------------------------------------------------------
        # Containment
        # -------------------------------------------------------------

        if candidate_value:

            reasons.append(
                "Containment is a candidate only."
            )

            reasons.append(
                "Explicit analyst authorization is required."
            )

        # -------------------------------------------------------------
        # Determine authorization state
        # -------------------------------------------------------------

        if not analyst_required:

            final_state = (
                "NOT_REQUIRED"
            )

            execution_permitted = False

        else:

            if authorization == "APPROVED":

                # -----------------------------------------------------
                # CRITICAL SAFETY RULE:
                #
                # Even APPROVED does not make this component execute
                # anything. It only records authorization.
                # -----------------------------------------------------

                final_state = "APPROVED"

                execution_permitted = False

                reasons.append(
                    "Analyst approval recorded; "
                    "execution remains outside this component."
                )

            elif authorization == "REJECTED":

                final_state = "REJECTED"

                execution_permitted = False

                reasons.append(
                    "Analyst rejected the proposed response."
                )

            else:

                final_state = "PENDING"

                execution_permitted = False

                reasons.append(
                    "Analyst authorization is pending."
                )

        # -------------------------------------------------------------
        # Hard containment boundary
        # -------------------------------------------------------------

        containment_authorized = False

        # Even an APPROVED analyst state does not turn this component
        # into an execution engine.
        if (
            candidate_value
            and final_state == "APPROVED"
        ):

            reasons.append(
                "Containment approval is recorded, "
                "but execution authorization is not granted "
                "to this planner."
            )

        # -------------------------------------------------------------
        # Final safety assertions
        # -------------------------------------------------------------

        if execution_permitted:

            raise RuntimeError(
                "Safety violation: execution was permitted."
            )

        if containment_authorized:

            raise RuntimeError(
                "Safety violation: containment was authorized "
                "by the authorization boundary."
            )

        if self.safety[
            "automatic_response"
        ]:

            raise RuntimeError(
                "Safety violation: automatic response enabled."
            )

        return AuthorizationResult(

            authorization_version=self.VERSION,

            risk_score=score,

            risk_level=level,

            decision=normalized_decision,

            fallback=fallback_value,

            ambiguous=ambiguous_value,

            missing_behavior=missing_behavior_value,

            containment_candidate=candidate_value,

            analyst_required=analyst_required,

            authorization_state=final_state,

            execution_permitted=False,

            containment_authorized=False,

            reasons=tuple(reasons),

            safety=dict(
                self.safety
            ),
        )

    # -----------------------------------------------------------------
    # Decision-policy result adapter
    # -----------------------------------------------------------------

    def evaluate_decision(
        self,
        decision_result: Any,
        *,
        analyst_authorization: str = "NOT_REQUIRED",
    ) -> AuthorizationResult:

        required = [
            "risk_score",
            "risk_level",
            "decision",
            "fallback",
            "ambiguous",
            "missing_behavior",
            "containment_candidate",
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

            fallback=(
                decision_result.fallback
            ),

            ambiguous=(
                decision_result.ambiguous
            ),

            missing_behavior=(
                decision_result.missing_behavior
            ),

            containment_candidate=(
                decision_result.containment_candidate
            ),

            analyst_authorization=(
                analyst_authorization
            ),
        )

    # -----------------------------------------------------------------
    # Event adapter
    # -----------------------------------------------------------------

    def evaluate_event(
        self,
        event: Dict[str, Any],
    ) -> AuthorizationResult:

        if not isinstance(
            event,
            dict,
        ):

            raise TypeError(
                "event must be a dictionary"
            )

        required = [
            "risk_score",
            "risk_level",
            "decision",
        ]

        for field in required:

            if field not in event:

                raise KeyError(
                    f"Missing event field: {field}"
                )

        return self.evaluate(

            risk_score=event[
                "risk_score"
            ],

            risk_level=event[
                "risk_level"
            ],

            decision=event[
                "decision"
            ],

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

            containment_candidate=event.get(
                "containment_candidate",
                False,
            ),

            analyst_authorization=event.get(
                "analyst_authorization",
                "NOT_REQUIRED",
            ),
        )

    # -----------------------------------------------------------------
    # Safety status
    # -----------------------------------------------------------------

    def safety_status(self) -> Dict[str, bool]:

        return dict(
            self.safety
        )


# =====================================================================
# Self-test
# =====================================================================

def self_test() -> Dict[str, bool]:

    engine = (
        SmartSOCAnalystAuthorization()
    )

    checks: Dict[str, bool] = {}

    # ---------------------------------------------------------------
    # LOW
    # ---------------------------------------------------------------

    low = engine.evaluate(

        risk_score=10,

        risk_level="LOW",

        decision="MONITOR",
    )

    checks[
        "low_authorization_not_required"
    ] = (
        low.authorization_state
        == "NOT_REQUIRED"
    )

    checks[
        "low_execution_disabled"
    ] = (
        low.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # MEDIUM
    # ---------------------------------------------------------------

    medium = engine.evaluate(

        risk_score=40,

        risk_level="MEDIUM",

        decision="ANALYST_REVIEW",
    )

    checks[
        "medium_pending"
    ] = (
        medium.authorization_state
        == "PENDING"
    )

    checks[
        "medium_requires_analyst"
    ] = (
        medium.analyst_required is True
    )

    checks[
        "medium_execution_disabled"
    ] = (
        medium.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # HIGH
    # ---------------------------------------------------------------

    high = engine.evaluate(

        risk_score=65,

        risk_level="HIGH",

        decision="PRIORITY_ANALYST_REVIEW",
    )

    checks[
        "high_pending"
    ] = (
        high.authorization_state
        == "PENDING"
    )

    checks[
        "high_requires_analyst"
    ] = (
        high.analyst_required is True
    )

    checks[
        "high_execution_disabled"
    ] = (
        high.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # CRITICAL
    # ---------------------------------------------------------------

    critical = engine.evaluate(

        risk_score=90,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        containment_candidate=True,
    )

    checks[
        "critical_pending"
    ] = (
        critical.authorization_state
        == "PENDING"
    )

    checks[
        "critical_requires_analyst"
    ] = (
        critical.analyst_required is True
    )

    checks[
        "critical_candidate"
    ] = (
        critical.containment_candidate is True
    )

    checks[
        "critical_not_authorized"
    ] = (
        critical.containment_authorized is False
    )

    checks[
        "critical_execution_disabled"
    ] = (
        critical.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # Analyst approval
    # ---------------------------------------------------------------

    approved = engine.evaluate(

        risk_score=90,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        containment_candidate=True,

        analyst_authorization="APPROVED",
    )

    checks[
        "approved_state_recorded"
    ] = (
        approved.authorization_state
        == "APPROVED"
    )

    checks[
        "approval_does_not_enable_execution"
    ] = (
        approved.execution_permitted is False
    )

    checks[
        "approval_does_not_authorize_containment"
    ] = (
        approved.containment_authorized is False
    )

    # ---------------------------------------------------------------
    # Analyst rejection
    # ---------------------------------------------------------------

    rejected = engine.evaluate(

        risk_score=90,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        containment_candidate=True,

        analyst_authorization="REJECTED",
    )

    checks[
        "rejection_recorded"
    ] = (
        rejected.authorization_state
        == "REJECTED"
    )

    checks[
        "rejection_blocks_execution"
    ] = (
        rejected.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------

    fallback = engine.evaluate(

        risk_score=95,

        risk_level="CRITICAL",

        decision="ANALYST_REVIEW",

        fallback=True,

        containment_candidate=False,
    )

    checks[
        "fallback_requires_analyst"
    ] = (
        fallback.analyst_required is True
    )

    checks[
        "fallback_pending"
    ] = (
        fallback.authorization_state
        == "PENDING"
    )

    checks[
        "fallback_no_execution"
    ] = (
        fallback.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # Ambiguous
    # ---------------------------------------------------------------

    ambiguous = engine.evaluate(

        risk_score=95,

        risk_level="CRITICAL",

        decision="ANALYST_REVIEW",

        ambiguous=True,

        containment_candidate=False,
    )

    checks[
        "ambiguous_requires_analyst"
    ] = (
        ambiguous.analyst_required is True
    )

    checks[
        "ambiguous_no_execution"
    ] = (
        ambiguous.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # Missing behavior
    # ---------------------------------------------------------------

    missing = engine.evaluate(

        risk_score=90,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        missing_behavior=True,

        containment_candidate=True,
    )

    checks[
        "missing_behavior_requires_analyst"
    ] = (
        missing.analyst_required is True
    )

    checks[
        "missing_behavior_no_execution"
    ] = (
        missing.execution_permitted is False
    )

    # ---------------------------------------------------------------
    # AI cannot self-authorize
    # ---------------------------------------------------------------

    ai_attempt = engine.evaluate(

        risk_score=100,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        containment_candidate=True,

        analyst_authorization="APPROVED",
    )

    checks[
        "ai_approval_still_not_execution"
    ] = (
        ai_attempt.execution_permitted is False
    )

    checks[
        "ai_cannot_authorize_containment"
    ] = (
        ai_attempt.containment_authorized is False
    )

    # ---------------------------------------------------------------
    # Safety
    # ---------------------------------------------------------------

    safety = engine.safety_status()

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
        "automatic_account_disable_disabled"
    ] = (
        safety["automatic_account_disable"]
        is False
    )

    checks[
        "automatic_firewall_change_disabled"
    ] = (
        safety["automatic_firewall_change"]
        is False
    )

    checks[
        "automatic_file_deletion_disabled"
    ] = (
        safety["automatic_file_deletion"]
        is False
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
        "response_actions_not_executed"
    ] = (
        safety["response_actions_executed"]
        is False
    )

    checks[
        "containment_actions_not_executed"
    ] = (
        safety["containment_actions_executed"]
        is False
    )

    # ---------------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------------

    first = engine.evaluate(

        risk_score=88.5,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        fallback=False,

        ambiguous=False,

        missing_behavior=False,

        containment_candidate=True,

        analyst_authorization="PENDING",
    )

    second = engine.evaluate(

        risk_score=88.5,

        risk_level="CRITICAL",

        decision="CONTAINMENT_CANDIDATE",

        fallback=False,

        ambiguous=False,

        missing_behavior=False,

        containment_candidate=True,

        analyst_authorization="PENDING",
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

        lambda: engine.evaluate(
            risk_score=-1,
            risk_level="LOW",
            decision="MONITOR",
        ),

        lambda: engine.evaluate(
            risk_score=101,
            risk_level="CRITICAL",
            decision="CONTAINMENT_CANDIDATE",
        ),

        lambda: engine.evaluate(
            risk_score=float("nan"),
            risk_level="LOW",
            decision="MONITOR",
        ),

        lambda: engine.evaluate(
            risk_score=10,
            risk_level="INVALID",
            decision="MONITOR",
        ),

        lambda: engine.evaluate(
            risk_score=10,
            risk_level="LOW",
            decision="INVALID",
        ),

        lambda: engine.evaluate(
            risk_score=10,
            risk_level="LOW",
            decision="MONITOR",
            analyst_authorization="INVALID",
        ),
    ]

    rejected_count = 0

    for test in invalid_tests:

        try:
            test()

        except Exception:
            rejected_count += 1

    checks[
        "invalid_inputs_rejected"
    ] = (
        rejected_count
        == len(invalid_tests)
    )

    return checks


# =====================================================================
# Main
# =====================================================================

def main() -> int:

    print("=" * 72)
    print("SmartSOC Phase 6.4")
    print("Analyst Authorization Boundary")
    print(f"Version {VERSION}")
    print("=" * 72)
    print()

    try:

        checks = self_test()

    except Exception as exc:

        print(
            "[FAIL] Analyst authorization self-test"
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
    print("PHASE 6.4 ENGINE SELF-TEST")
    print("=" * 72)

    print(
        "Status : "
        + (
            "PASS"
            if failed == 0
            else "FAIL"
        )
    )

    print(
        f"Tests  : "
        f"{total} total / "
        f"{failed} failed"
    )

    if failed == 0:

        print()

        print(
            "PHASE 6.4 ENGINE ACCEPTED — "
            "ANALYST AUTHORIZATION BOUNDARY VALIDATED"
        )

        return 0

    print()

    print(
        "PHASE 6.4 ENGINE FAILED — "
        "DO NOT PROCEED"
    )

    return 1


if __name__ == "__main__":
    sys.exit(main())