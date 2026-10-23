from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


ENGINE_VERSION = "4.1"


@dataclass(frozen=True)
class DecisionPolicyResult:
    """
    Immutable result produced by the SmartSOC Decision Policy.

    This policy only determines the SOC disposition.
    It does not execute any response or containment action.
    """

    policy_version: str
    risk_score: float
    risk_level: str
    decision: str

    analyst_required: bool
    containment_candidate: bool
    containment_authorized: bool

    fallback: bool
    ambiguous: bool
    missing_behavior: bool
    uncertainty_override: bool

    reasons: List[str]

    safety: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SmartSOCDecisionPolicy:
    """
    Phase 5.2 deterministic SOC Decision Policy.

    Input:
        Phase 3D RiskDecision or equivalent risk information.

    Output:
        Deterministic SOC disposition.

    Important:
        This class performs NO operational side effects.
    """

    POLICY_VERSION = ENGINE_VERSION

    VALID_RISK_LEVELS = {
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    }

    DECISION_ORDER = {
        "MONITOR": 0,
        "ANALYST_REVIEW": 1,
        "PRIORITY_ANALYST_REVIEW": 2,
        "CONTAINMENT_CANDIDATE": 3,
    }

    RISK_MAPPING = {
        "LOW": "MONITOR",
        "MEDIUM": "ANALYST_REVIEW",
        "HIGH": "PRIORITY_ANALYST_REVIEW",
        "CRITICAL": "CONTAINMENT_CANDIDATE",
    }

    def __init__(self) -> None:
        self.safety_policy = {
            "automatic_response": False,
            "automatic_blocking": False,
            "automatic_isolation": False,
            "automatic_process_kill": False,
            "containment_requires_analyst": True,
            "fallback_can_directly_contain": False,
            "ambiguous_can_directly_contain": False,
            "response_actions_executed": False,
            "containment_actions_executed": False,
            "wazuh_integration": False,
            "shuffle_integration": False,
        }

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_risk_level(risk_level: Any) -> str:
        level = str(risk_level).strip().upper()

        if level not in SmartSOCDecisionPolicy.VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid risk level: {risk_level!r}. "
                f"Expected one of "
                f"{sorted(SmartSOCDecisionPolicy.VALID_RISK_LEVELS)}"
            )

        return level

    @staticmethod
    def _normalize_score(risk_score: Any) -> float:
        try:
            score = float(risk_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid risk score: {risk_score!r}"
            ) from exc

        if not 0.0 <= score <= 100.0:
            raise ValueError(
                f"Risk score must be within [0,100], got {score}"
            )

        return round(score, 6)

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        return bool(value)

    # ------------------------------------------------------------------
    # Core policy
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        risk_score: float,
        risk_level: str,
        fallback: bool = False,
        ambiguous: bool = False,
        missing_behavior: bool = False,
    ) -> DecisionPolicyResult:
        """
        Evaluate the SOC decision policy.

        Precedence:

        1. Safety boundary
        2. Fallback / ambiguity uncertainty override
        3. Risk-level mapping

        No external action is performed.
        """

        score = self._normalize_score(risk_score)
        level = self._normalize_risk_level(risk_level)

        fallback = self._normalize_bool(fallback)
        ambiguous = self._normalize_bool(ambiguous)
        missing_behavior = self._normalize_bool(missing_behavior)

        reasons: List[str] = []

        # --------------------------------------------------------------
        # Mandatory safety boundary
        # --------------------------------------------------------------

        reasons.append(
            "Decision policy is side-effect free; "
            "no operational response is executed."
        )

        # --------------------------------------------------------------
        # Uncertainty override
        # --------------------------------------------------------------

        uncertainty_override = False

        if fallback:
            decision = "ANALYST_REVIEW"
            uncertainty_override = True

            reasons.append(
                "Fallback classification requires analyst review."
            )

        elif ambiguous:
            decision = "ANALYST_REVIEW"
            uncertainty_override = True

            reasons.append(
                "Ambiguous classification requires analyst review."
            )

        else:
            decision = self.RISK_MAPPING[level]

            reasons.append(
                f"Risk level {level} maps to {decision}."
            )

        # --------------------------------------------------------------
        # Missing behavioral evidence
        # --------------------------------------------------------------

        if missing_behavior:
            reasons.append(
                "Behavioral evidence is incomplete; "
                "analyst caution is required."
            )

        # --------------------------------------------------------------
        # Analyst boundary
        # --------------------------------------------------------------

        if decision == "MONITOR":
            analyst_required = False

        else:
            analyst_required = True

        # --------------------------------------------------------------
        # Containment boundary
        # --------------------------------------------------------------

        containment_candidate = (
            decision == "CONTAINMENT_CANDIDATE"
        )

        # Critical NEVER directly authorizes containment.
        containment_authorized = False

        if containment_candidate:
            analyst_required = True

            reasons.append(
                "Containment is only a candidate disposition "
                "and requires explicit analyst approval."
            )

        # --------------------------------------------------------------
        # Additional safety rule
        # --------------------------------------------------------------

        if fallback or ambiguous:
            containment_authorized = False

            if containment_candidate:
                decision = "ANALYST_REVIEW"
                containment_candidate = False

                reasons.append(
                    "Uncertainty override prevents direct containment "
                    "candidate escalation."
                )

        # --------------------------------------------------------------
        # Final immutable result
        # --------------------------------------------------------------

        return DecisionPolicyResult(
            policy_version=self.POLICY_VERSION,
            risk_score=score,
            risk_level=level,
            decision=decision,
            analyst_required=analyst_required,
            containment_candidate=containment_candidate,
            containment_authorized=containment_authorized,
            fallback=fallback,
            ambiguous=ambiguous,
            missing_behavior=missing_behavior,
            uncertainty_override=uncertainty_override,
            reasons=reasons,
            safety=dict(self.safety_policy),
        )

    # ------------------------------------------------------------------
    # RiskDecision adapter
    # ------------------------------------------------------------------

    def evaluate_risk_decision(
        self,
        risk_decision: Any,
        *,
        fallback: bool = False,
        ambiguous: bool = False,
        missing_behavior: bool = False,
    ) -> DecisionPolicyResult:
        """
        Consume the existing Phase 3D RiskDecision object.

        Expected attributes:

            risk_score
            risk_level

        The adapter deliberately does not recalculate risk.
        """

        if not hasattr(risk_decision, "risk_score"):
            raise TypeError(
                "RiskDecision object does not contain risk_score."
            )

        if not hasattr(risk_decision, "risk_level"):
            raise TypeError(
                "RiskDecision object does not contain risk_level."
            )

        return self.evaluate(
            risk_score=risk_decision.risk_score,
            risk_level=risk_decision.risk_level,
            fallback=fallback,
            ambiguous=ambiguous,
            missing_behavior=missing_behavior,
        )

    # ------------------------------------------------------------------
    # Dictionary adapter
    # ------------------------------------------------------------------

    def evaluate_event(
        self,
        event: Dict[str, Any],
    ) -> DecisionPolicyResult:
        """
        Evaluate a dictionary representation of a risk event.
        """

        if not isinstance(event, dict):
            raise TypeError("event must be a dictionary")

        if "risk_score" not in event:
            raise KeyError("event requires 'risk_score'")

        if "risk_level" not in event:
            raise KeyError("event requires 'risk_level'")

        return self.evaluate(
            risk_score=event["risk_score"],
            risk_level=event["risk_level"],
            fallback=event.get("fallback", False),
            ambiguous=event.get("ambiguous", False),
            missing_behavior=event.get("missing_behavior", False),
        )

    # ------------------------------------------------------------------
    # Safety inspection
    # ------------------------------------------------------------------

    def safety_status(self) -> Dict[str, bool]:
        return dict(self.safety_policy)


def self_test() -> Dict[str, Any]:
    """
    Built-in Phase 5.2 engine self-test.
    """

    policy = SmartSOCDecisionPolicy()

    tests: List[Dict[str, Any]] = []

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

    # --------------------------------------------------------------
    # LOW
    # --------------------------------------------------------------

    low = policy.evaluate(
        risk_score=15,
        risk_level="LOW",
    )

    check(
        "low_monitor",
        low.decision == "MONITOR",
    )

    check(
        "low_no_analyst",
        low.analyst_required is False,
    )

    # --------------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------------

    medium = policy.evaluate(
        risk_score=40,
        risk_level="MEDIUM",
    )

    check(
        "medium_analyst_review",
        medium.decision == "ANALYST_REVIEW",
    )

    check(
        "medium_requires_analyst",
        medium.analyst_required is True,
    )

    # --------------------------------------------------------------
    # HIGH
    # --------------------------------------------------------------

    high = policy.evaluate(
        risk_score=65,
        risk_level="HIGH",
    )

    check(
        "high_priority_review",
        high.decision == "PRIORITY_ANALYST_REVIEW",
    )

    check(
        "high_requires_analyst",
        high.analyst_required is True,
    )

    # --------------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------------

    critical = policy.evaluate(
        risk_score=90,
        risk_level="CRITICAL",
    )

    check(
        "critical_containment_candidate",
        critical.decision == "CONTAINMENT_CANDIDATE",
    )

    check(
        "critical_requires_analyst",
        critical.analyst_required is True,
    )

    check(
        "critical_not_authorized",
        critical.containment_authorized is False,
    )

    # --------------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------------

    fallback = policy.evaluate(
        risk_score=95,
        risk_level="CRITICAL",
        fallback=True,
    )

    check(
        "fallback_analyst_review",
        fallback.decision == "ANALYST_REVIEW",
    )

    check(
        "fallback_no_containment",
        fallback.containment_authorized is False,
    )

    # --------------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------------

    ambiguous = policy.evaluate(
        risk_score=95,
        risk_level="CRITICAL",
        ambiguous=True,
    )

    check(
        "ambiguous_analyst_review",
        ambiguous.decision == "ANALYST_REVIEW",
    )

    check(
        "ambiguous_no_containment",
        ambiguous.containment_authorized is False,
    )

    # --------------------------------------------------------------
    # MISSING BEHAVIOR
    # --------------------------------------------------------------

    missing = policy.evaluate(
        risk_score=80,
        risk_level="CRITICAL",
        missing_behavior=True,
    )

    check(
        "missing_behavior_recorded",
        missing.missing_behavior is True,
    )

    check(
        "missing_behavior_no_auto_response",
        missing.safety["automatic_response"] is False,
    )

    # --------------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------------

    safety = policy.safety_status()

    for key in [
        "automatic_response",
        "automatic_blocking",
        "automatic_isolation",
        "automatic_process_kill",
        "response_actions_executed",
        "containment_actions_executed",
        "wazuh_integration",
        "shuffle_integration",
    ]:
        check(
            f"safety_{key}_disabled",
            safety[key] is False,
        )

    check(
        "containment_requires_analyst",
        safety["containment_requires_analyst"] is True,
    )

    check(
        "fallback_cannot_directly_contain",
        safety["fallback_can_directly_contain"] is False,
    )

    check(
        "ambiguous_cannot_directly_contain",
        safety["ambiguous_can_directly_contain"] is False,
    )

    # --------------------------------------------------------------
    # Determinism
    # --------------------------------------------------------------

    result_a = policy.evaluate(
        risk_score=88.6,
        risk_level="CRITICAL",
    )

    result_b = policy.evaluate(
        risk_score=88.6,
        risk_level="CRITICAL",
    )

    check(
        "deterministic_result",
        result_a == result_b,
    )

    # --------------------------------------------------------------
    # Dictionary adapter
    # --------------------------------------------------------------

    event_result = policy.evaluate_event(
        {
            "risk_score": 55,
            "risk_level": "HIGH",
        }
    )

    check(
        "dictionary_adapter",
        event_result.decision == "PRIORITY_ANALYST_REVIEW",
    )

    # --------------------------------------------------------------
    # Invalid input rejection
    # --------------------------------------------------------------

    invalid_level_rejected = False

    try:
        policy.evaluate(
            risk_score=50,
            risk_level="INVALID",
        )
    except ValueError:
        invalid_level_rejected = True

    check(
        "invalid_risk_level_rejected",
        invalid_level_rejected,
    )

    invalid_score_rejected = False

    try:
        policy.evaluate(
            risk_score=101,
            risk_level="CRITICAL",
        )
    except ValueError:
        invalid_score_rejected = True

    check(
        "invalid_risk_score_rejected",
        invalid_score_rejected,
    )

    passed = sum(
        1 for test in tests
        if test["passed"]
    )

    failed = len(tests) - passed

    return {
        "engine_version": ENGINE_VERSION,
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "tests": tests,
    }


if __name__ == "__main__":
    result = self_test()

    print("=" * 72)
    print("SmartSOC Phase 5.2")
    print("Decision Policy Engine Self-Test")
    print(f"Version {ENGINE_VERSION}")
    print("=" * 72)
    print()

    for test in result["tests"]:
        status = "PASS" if test["passed"] else "FAIL"
        print(f"[{status}] {test['name']}")

    print()
    print("=" * 72)
    print("PHASE 5.2 ENGINE SELF-TEST")
    print("=" * 72)
    print(
        f"Status : {result['status']}"
    )
    print(
        f"Tests  : {result['total']} total / "
        f"{result['failed']} failed"
    )

    raise SystemExit(
        0 if result["status"] == "PASS" else 1
    )