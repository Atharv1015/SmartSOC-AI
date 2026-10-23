"""
SMARTSOC
PHASE 3D.2 — DETERMINISTIC SOC RISK ENGINE

Version:
    v4.1

Purpose:
    Convert ML classification, confidence, evidence, and
    event context into a deterministic 0-100 SOC risk score.

Important:
    This module DOES NOT:
        - execute commands
        - block IPs
        - isolate hosts
        - kill processes
        - modify Wazuh
        - call Shuffle
        - perform autonomous containment

    It only produces a risk/decision object.

Architecture:

    Wazuh/Sysmon Event
            |
            v
    ML Classification
            |
            +---- confidence
            |
            +---- evidence
            |
            +---- context
            |
            v
    SmartSOC Risk Engine
            |
            v
    risk_score 0-100
    risk_level
    recommended_action
    reasons
    component scores
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import json


# ============================================================
# ENGINE VERSION
# ============================================================

ENGINE_VERSION = "4.1"


# ============================================================
# SCORE LIMITS
# ============================================================

MAX_TECHNIQUE_SEVERITY = 30
MAX_ML_CONFIDENCE = 20
MAX_BEHAVIORAL_EVIDENCE = 25
MAX_CONTEXT = 15
MAX_UNCERTAINTY_PENALTY = 10

POSITIVE_SCORE_MAX = (
    MAX_TECHNIQUE_SEVERITY
    + MAX_ML_CONFIDENCE
    + MAX_BEHAVIORAL_EVIDENCE
    + MAX_CONTEXT
)

MAX_FINAL_SCORE = 100


# ============================================================
# RISK LEVELS
# ============================================================

RISK_LEVELS = {
    "LOW": {
        "min": 0,
        "max": 24,
    },
    "MEDIUM": {
        "min": 25,
        "max": 49,
    },
    "HIGH": {
        "min": 50,
        "max": 74,
    },
    "CRITICAL": {
        "min": 75,
        "max": 100,
    },
}


# ============================================================
# ACTIONS
# ============================================================

RECOMMENDED_ACTIONS = {
    "LOW": "MONITOR",
    "MEDIUM": "ANALYST_REVIEW",
    "HIGH": "PRIORITY_ANALYST_REVIEW",
    "CRITICAL": "CONTAINMENT_CANDIDATE",
}


# ============================================================
# TECHNIQUE SEVERITY
# ============================================================

TECHNIQUE_SEVERITY_VALUES = {
    "LOW": 5,
    "MEDIUM": 12,
    "HIGH": 22,
    "CRITICAL": 30,
}


# ============================================================
# BEHAVIORAL EVIDENCE
# ============================================================

BEHAVIORAL_EVIDENCE_VALUES = {
    "NONE": 0,
    "WEAK": 6,
    "MODERATE": 14,
    "STRONG": 21,
    "VERY_STRONG": 25,
}


# ============================================================
# PRIVILEGE / CONTEXT
# ============================================================

PRIVILEGE_VALUES = {
    "LOW": 0,
    "MEDIUM": 3,
    "HIGH": 6,
    "SYSTEM": 7,
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RiskComponents:
    """
    Individual risk-engine components.
    """

    technique_severity: float
    ml_confidence: float
    behavioral_evidence: float
    context: float
    uncertainty_penalty: float


@dataclass
class RiskDecision:
    """
    Complete SmartSOC risk-engine result.
    """

    engine_version: str
    risk_score: float
    risk_level: str
    recommended_action: str
    components: RiskComponents
    reasons: List[str]
    safety: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result into JSON-compatible dictionary.
        """

        result = asdict(self)

        result["components"] = asdict(
            self.components
        )

        return result

    def to_json(self) -> str:
        """
        Serialize result to formatted JSON.
        """

        return json.dumps(
            self.to_dict(),
            indent=2
        )


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _normalize_level(
    value: str,
    allowed: Dict[str, Any],
    default: str
) -> str:

    if value is None:
        return default

    value = str(
        value
    ).strip().upper()

    if value not in allowed:
        return default

    return value


def _normalize_bool(
    value: Any
) -> bool:

    if isinstance(
        value,
        bool
    ):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        str
    ):

        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
            "on",
        }

    return bool(value)


def _normalize_confidence(
    value: Any
) -> float:

    try:

        confidence = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        confidence = 0.0

    return max(
        0.0,
        min(
            1.0,
            confidence
        )
    )


# ============================================================
# COMPONENT CALCULATORS
# ============================================================

def calculate_technique_severity(
    severity: str
) -> float:

    severity = _normalize_level(
        severity,
        TECHNIQUE_SEVERITY_VALUES,
        "LOW"
    )

    return float(
        TECHNIQUE_SEVERITY_VALUES[
            severity
        ]
    )


def calculate_ml_confidence(
    confidence: Any
) -> float:

    confidence = _normalize_confidence(
        confidence
    )

    return (
        confidence
        *
        MAX_ML_CONFIDENCE
    )


def calculate_behavioral_evidence(
    evidence_strength: str
) -> float:

    evidence_strength = _normalize_level(
        evidence_strength,
        BEHAVIORAL_EVIDENCE_VALUES,
        "NONE"
    )

    return float(
        BEHAVIORAL_EVIDENCE_VALUES[
            evidence_strength
        ]
    )


def calculate_context(
    privilege: str,
    network: bool,
    suspicious_parent: bool
) -> float:

    privilege = _normalize_level(
        privilege,
        PRIVILEGE_VALUES,
        "LOW"
    )

    score = float(
        PRIVILEGE_VALUES[
            privilege
        ]
    )

    if _normalize_bool(
        network
    ):
        score += 4

    if _normalize_bool(
        suspicious_parent
    ):
        score += 4

    return min(
        score,
        float(MAX_CONTEXT)
    )


def calculate_uncertainty_penalty(
    fallback: bool,
    ambiguous: bool,
    missing_behavior: bool
) -> float:

    penalty = 0.0

    if _normalize_bool(
        fallback
    ):
        penalty += 5

    if _normalize_bool(
        ambiguous
    ):
        penalty += 3

    if _normalize_bool(
        missing_behavior
    ):
        penalty += 2

    return min(
        penalty,
        float(MAX_UNCERTAINTY_PENALTY)
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(
    score: float
) -> str:

    score = max(
        0.0,
        min(
            float(MAX_FINAL_SCORE),
            float(score)
        )
    )

    if score < 25:
        return "LOW"

    if score < 50:
        return "MEDIUM"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


# ============================================================
# REASON GENERATION
# ============================================================

def generate_reasons(
    *,
    technique_score: float,
    confidence_score: float,
    evidence_score: float,
    context_score: float,
    uncertainty_score: float,
    privilege: str,
    network: bool,
    suspicious_parent: bool,
    fallback: bool,
    ambiguous: bool,
    missing_behavior: bool
) -> List[str]:

    reasons = []

    if technique_score >= 22:

        reasons.append(
            "High or critical technique severity"
        )

    elif technique_score >= 12:

        reasons.append(
            "Medium technique severity"
        )

    if confidence_score >= 16:

        reasons.append(
            "High model confidence contribution"
        )

    elif confidence_score >= 10:

        reasons.append(
            "Moderate model confidence contribution"
        )

    if evidence_score >= 21:

        reasons.append(
            "Strong behavioral evidence"
        )

    elif evidence_score >= 14:

        reasons.append(
            "Moderate behavioral evidence"
        )

    elif evidence_score == 0:

        reasons.append(
            "No behavioral evidence available"
        )

    if network:

        reasons.append(
            "Network behavior present"
        )

    if suspicious_parent:

        reasons.append(
            "Suspicious parent-process context"
        )

    if str(
        privilege
    ).upper() in {
        "HIGH",
        "SYSTEM",
    }:

        reasons.append(
            "Elevated privilege context"
        )

    if fallback:

        reasons.append(
            "Prediction entered fallback state"
        )

    if ambiguous:

        reasons.append(
            "Prediction is ambiguous"
        )

    if missing_behavior:

        reasons.append(
            "Behavioral context is missing"
        )

    if uncertainty_score > 0:

        reasons.append(
            "Uncertainty penalty applied"
        )

    if not reasons:

        reasons.append(
            "Limited suspicious evidence"
        )

    return reasons


# ============================================================
# MAIN ENGINE
# ============================================================

class SmartSOCRiskEngine:
    """
    Deterministic SmartSOC risk engine.

    No external side effects.
    """

    def __init__(
        self,
        version: str = ENGINE_VERSION
    ):

        self.version = version

    # --------------------------------------------------------
    # calculate
    # --------------------------------------------------------

    def calculate(
        self,
        *,
        technique_severity: str,
        ml_confidence: float,
        evidence_strength: str,
        privilege: str = "LOW",
        network: bool = False,
        suspicious_parent: bool = False,
        fallback: bool = False,
        ambiguous: bool = False,
        missing_behavior: bool = False,
    ) -> RiskDecision:

        technique_score = (
            calculate_technique_severity(
                technique_severity
            )
        )

        confidence_score = (
            calculate_ml_confidence(
                ml_confidence
            )
        )

        evidence_score = (
            calculate_behavioral_evidence(
                evidence_strength
            )
        )

        context_component = (
            calculate_context(
                privilege,
                network,
                suspicious_parent
            )
        )

        uncertainty_component = (
            calculate_uncertainty_penalty(
                fallback,
                ambiguous,
                missing_behavior
            )
        )

        raw_score = (
            technique_score
            + confidence_score
            + evidence_score
            + context_component
            - uncertainty_component
        )

        final_score = max(
            0.0,
            min(
                float(MAX_FINAL_SCORE),
                raw_score
            )
        )

        level = calculate_risk_level(
            final_score
        )

        action = (
            RECOMMENDED_ACTIONS[
                level
            ]
        )

        reasons = generate_reasons(
            technique_score=technique_score,
            confidence_score=confidence_score,
            evidence_score=evidence_score,
            context_score=context_component,
            uncertainty_score=uncertainty_component,
            privilege=privilege,
            network=_normalize_bool(network),
            suspicious_parent=_normalize_bool(
                suspicious_parent
            ),
            fallback=_normalize_bool(
                fallback
            ),
            ambiguous=_normalize_bool(
                ambiguous
            ),
            missing_behavior=_normalize_bool(
                missing_behavior
            ),
        )

        return RiskDecision(

            engine_version=self.version,

            risk_score=round(
                final_score,
                2
            ),

            risk_level=level,

            recommended_action=action,

            components=RiskComponents(

                technique_severity=round(
                    technique_score,
                    2
                ),

                ml_confidence=round(
                    confidence_score,
                    2
                ),

                behavioral_evidence=round(
                    evidence_score,
                    2
                ),

                context=round(
                    context_component,
                    2
                ),

                uncertainty_penalty=round(
                    uncertainty_component,
                    2
                ),
            ),

            reasons=reasons,

            safety={
                "automatic_response": False,
                "automatic_blocking": False,
                "automatic_isolation": False,
                "automatic_process_kill": False,
                "requires_analyst_for_containment": True,
                "fallback_can_directly_contain": False,
            },
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def evaluate_event(
    event: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience interface for future Wazuh/Shuffle adapters.

    Expected input:

    {
        "technique_severity": "HIGH",
        "ml_confidence": 0.92,
        "evidence_strength": "STRONG",
        "privilege": "HIGH",
        "network": true,
        "suspicious_parent": true,
        "fallback": false,
        "ambiguous": false,
        "missing_behavior": false
    }
    """

    engine = SmartSOCRiskEngine()

    result = engine.calculate(
        technique_severity=event.get(
            "technique_severity",
            "LOW"
        ),

        ml_confidence=event.get(
            "ml_confidence",
            0.0
        ),

        evidence_strength=event.get(
            "evidence_strength",
            "NONE"
        ),

        privilege=event.get(
            "privilege",
            "LOW"
        ),

        network=event.get(
            "network",
            False
        ),

        suspicious_parent=event.get(
            "suspicious_parent",
            False
        ),

        fallback=event.get(
            "fallback",
            False
        ),

        ambiguous=event.get(
            "ambiguous",
            False
        ),

        missing_behavior=event.get(
            "missing_behavior",
            False
        ),
    )

    return result.to_dict()


# ============================================================
# MODULE SELF-TEST
# ============================================================

def run_self_test() -> Dict[str, Any]:
    """
    Built-in deterministic smoke test.

    This does not touch the filesystem, Wazuh,
    Shuffle, or production artifacts.
    """

    engine = SmartSOCRiskEngine()

    tests = []

    # --------------------------------------------------------
    # 1. Benign administrative activity
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="LOW",
        ml_confidence=0.45,
        evidence_strength="WEAK",
        privilege="MEDIUM",
        network=False,
        suspicious_parent=False,
        fallback=True,
        ambiguous=True,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "benign_administrative_cmd",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected_max":
                "MEDIUM",

            "passed":
                result.risk_level
                in {
                    "LOW",
                    "MEDIUM",
                },
        }
    )

    # --------------------------------------------------------
    # 2. Suspicious PowerShell
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="HIGH",
        ml_confidence=0.90,
        evidence_strength="STRONG",
        privilege="HIGH",
        network=True,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "suspicious_powershell",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected_min":
                "HIGH",

            "passed":
                result.risk_level
                in {
                    "HIGH",
                    "CRITICAL",
                },
        }
    )

    # --------------------------------------------------------
    # 3. Encoded PowerShell download
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="CRITICAL",
        ml_confidence=0.99,
        evidence_strength="VERY_STRONG",
        privilege="HIGH",
        network=True,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "encoded_powershell_download",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected":
                "CRITICAL",

            "passed":
                result.risk_level
                ==
                "CRITICAL",
        }
    )

    # --------------------------------------------------------
    # 4. Missing behavior
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.60,
        evidence_strength="WEAK",
        privilege="MEDIUM",
        network=False,
        suspicious_parent=False,
        fallback=True,
        ambiguous=True,
        missing_behavior=True,
    )

    tests.append(
        {
            "name":
                "missing_behavior_context",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected_max":
                "MEDIUM",

            "passed":
                result.risk_level
                in {
                    "LOW",
                    "MEDIUM",
                },
        }
    )

    # --------------------------------------------------------
    # 5. Destructive behavior
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="CRITICAL",
        ml_confidence=0.97,
        evidence_strength="VERY_STRONG",
        privilege="SYSTEM",
        network=False,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "destructive_behavior",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected":
                "CRITICAL",

            "passed":
                result.risk_level
                ==
                "CRITICAL",
        }
    )

    # --------------------------------------------------------
    # 6. Low-confidence fallback
    # --------------------------------------------------------

    result = engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.35,
        evidence_strength="WEAK",
        privilege="LOW",
        network=False,
        suspicious_parent=False,
        fallback=True,
        ambiguous=True,
        missing_behavior=True,
    )

    tests.append(
        {
            "name":
                "low_confidence_fallback",

            "score":
                result.risk_score,

            "level":
                result.risk_level,

            "expected_max":
                "LOW",

            "passed":
                result.risk_level
                ==
                "LOW",
        }
    )

    # --------------------------------------------------------
    # Monotonicity
    # --------------------------------------------------------

    base = engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.50,
        evidence_strength="MODERATE",
        privilege="MEDIUM",
        network=False,
        suspicious_parent=False,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    higher_confidence = engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.80,
        evidence_strength="MODERATE",
        privilege="MEDIUM",
        network=False,
        suspicious_parent=False,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "confidence_monotonicity",

            "score":
                higher_confidence.risk_score,

            "baseline":
                base.risk_score,

            "passed":
                higher_confidence.risk_score
                >=
                base.risk_score,
        }
    )

    stronger_evidence = engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.50,
        evidence_strength="VERY_STRONG",
        privilege="MEDIUM",
        network=False,
        suspicious_parent=False,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "evidence_monotonicity",

            "score":
                stronger_evidence.risk_score,

            "baseline":
                base.risk_score,

            "passed":
                stronger_evidence.risk_score
                >=
                base.risk_score,
        }
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    safety_result = engine.calculate(
        technique_severity="CRITICAL",
        ml_confidence=1.0,
        evidence_strength="VERY_STRONG",
        privilege="SYSTEM",
        network=True,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    )

    tests.append(
        {
            "name":
                "automatic_response_safety",

            "action":
                safety_result.recommended_action,

            "passed":
                safety_result.safety[
                    "automatic_response"
                ]
                is False,
        }
    )

    tests.append(
        {
            "name":
                "containment_analyst_requirement",

            "passed":
                safety_result.safety[
                    "requires_analyst_for_containment"
                ]
                is True,
        }
    )

    # --------------------------------------------------------
    # Score bounds
    # --------------------------------------------------------

    all_scores_valid = all(
        0
        <= float(
            test.get(
                "score",
                0
            )
        )
        <= 100
        for test in tests
        if "score" in test
    )

    tests.append(
        {
            "name":
                "score_bounds",

            "passed":
                all_scores_valid,
        }
    )

    passed = sum(
        bool(
            test["passed"]
        )
        for test in tests
    )

    total = len(
        tests
    )

    return {
        "engine_version":
            ENGINE_VERSION,

        "total_tests":
            total,

        "passed_tests":
            passed,

        "failed_tests":
            total - passed,

        "decision":
            "PASS"
            if passed == total
            else "REVIEW_REQUIRED",

        "tests":
            tests,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "SMARTSOC PHASE 3D.2"
    )
    print(
        "DETERMINISTIC SOC RISK ENGINE"
    )
    print("=" * 72)

    print(
        f"\nEngine version: "
        f"{ENGINE_VERSION}"
    )

    print(
        "\nSCORING"
    )

    print(
        f"  Technique severity : "
        f"{MAX_TECHNIQUE_SEVERITY}"
    )

    print(
        f"  ML confidence      : "
        f"{MAX_ML_CONFIDENCE}"
    )

    print(
        f"  Behavioral evidence: "
        f"{MAX_BEHAVIORAL_EVIDENCE}"
    )

    print(
        f"  Context            : "
        f"{MAX_CONTEXT}"
    )

    print(
        f"  Uncertainty penalty: "
        f"{MAX_UNCERTAINTY_PENALTY}"
    )

    print(
        f"  Positive max       : "
        f"{POSITIVE_SCORE_MAX}"
    )

    print(
        f"  Final score range  : "
        f"0–{MAX_FINAL_SCORE}"
    )

    print(
        "\nRISK LEVELS"
    )

    for level, bounds in (
        RISK_LEVELS.items()
    ):

        print(
            f"  {level:8s} "
            f"{bounds['min']}–{bounds['max']}"
        )

    print(
        "\nACTIONS"
    )

    for level, action in (
        RECOMMENDED_ACTIONS.items()
    ):

        print(
            f"  {level:8s} → {action}"
        )

    print(
        "\nRUNNING SELF-TEST"
    )

    test_report = run_self_test()

    for test in test_report[
        "tests"
    ]:

        if test["passed"]:

            print(
                f"[PASS] "
                f"{test['name']}"
            )

        else:

            print(
                f"[FAIL] "
                f"{test['name']}"
            )

            print(
                f"       {test}"
            )

    print(
        "\n" + "=" * 72
    )

    print(
        "PHASE 3D.2 SELF-TEST RESULT"
    )

    print(
        "=" * 72
    )

    print(
        f"Decision       : "
        f"{test_report['decision']}"
    )

    print(
        f"Tests passed   : "
        f"{test_report['passed_tests']}"
        f"/"
        f"{test_report['total_tests']}"
    )

    print(
        f"Tests failed   : "
        f"{test_report['failed_tests']}"
    )

    print(
        "Automatic response: DISABLED"
    )

    print(
        "Wazuh modification: NONE"
    )

    print(
        "Shuffle execution : NONE"
    )

    print(
        "Containment       : ANALYST REQUIRED"
    )

    print(
        "=" * 72
    )

    if test_report[
        "decision"
    ] != "PASS":

        raise RuntimeError(
            "Phase 3D.2 self-test failed."
        )