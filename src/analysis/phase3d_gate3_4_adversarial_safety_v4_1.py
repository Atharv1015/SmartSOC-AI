"""
SMARTSOC PHASE 3D
GATE 3D.4 — ADVERSARIAL SAFETY VALIDATION

Stress-tests the deterministic SmartSOC Risk Engine against:
- extreme inputs
- malformed inputs
- invalid confidence values
- unknown categorical values
- contradictory evidence
- fallback conditions
- maximum uncertainty
- missing behavior
- score/action boundary conditions

Safety objective:
The risk engine must never authorize autonomous response,
direct containment, blocking, isolation, or process termination.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

DECISION_DIR = (
    PROJECT_ROOT
    / "src"
    / "decision"
)

ENGINE_PATH = (
    DECISION_DIR
    / "smartsoc_risk_engine_v4_1.py"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase3d"
    / "gate3_4"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    REPORT_DIR
    / "adversarial_safety_results.json"
)

FINAL_REPORT_PATH = (
    REPORT_DIR
    / "phase3d_gate3_4_report.json"
)


# ============================================================
# IMPORT ENGINE
# ============================================================

if not ENGINE_PATH.exists():

    raise FileNotFoundError(
        f"Risk engine not found: {ENGINE_PATH}"
    )

sys.path.insert(
    0,
    str(DECISION_DIR)
)

from smartsoc_risk_engine_v4_1 import (
    SmartSOCRiskEngine,
)


# ============================================================
# CONSTANTS
# ============================================================

VALID_ACTIONS = {
    "MONITOR",
    "ANALYST_REVIEW",
    "PRIORITY_ANALYST_REVIEW",
    "CONTAINMENT_CANDIDATE",
}

VALID_LEVELS = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
}


# ============================================================
# QUALITY CHECKS
# ============================================================

quality_checks = {}


def check(
    name: str,
    condition: bool,
    message: str = "",
):

    quality_checks[name] = bool(condition)

    if condition:

        print(
            f"[PASS] {name}"
        )

    else:

        print(
            f"[FAIL] {name}"
        )

        if message:

            print(
                f"       {message}"
            )


# ============================================================
# HELPERS
# ============================================================

def is_valid_score(
    value,
) -> bool:

    try:

        value = float(value)

        return (
            math.isfinite(value)
            and
            0.0 <= value <= 100.0
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def safety_is_intact(
    result,
) -> bool:

    safety = result.safety

    return (
        safety[
            "automatic_response"
        ]
        is False
        and
        safety[
            "automatic_blocking"
        ]
        is False
        and
        safety[
            "automatic_isolation"
        ]
        is False
        and
        safety[
            "automatic_process_kill"
        ]
        is False
        and
        safety[
            "requires_analyst_for_containment"
        ]
        is True
        and
        safety[
            "fallback_can_directly_contain"
        ]
        is False
    )


def component(
    result,
    name: str,
):
    """
    Safely retrieve a RiskComponents field.

    RiskComponents is a dataclass/object in the
    SmartSOC risk engine, not a dictionary.
    """

    return getattr(
        result.components,
        name,
    )


# ============================================================
# START
# ============================================================

print("=" * 72)
print("SMARTSOC PHASE 3D")
print("GATE 3D.4 — ADVERSARIAL SAFETY VALIDATION")
print("=" * 72)


# ============================================================
# ENGINE INITIALIZATION
# ============================================================

print("\nENGINE INITIALIZATION")

engine = SmartSOCRiskEngine()

check(
    "Risk engine exists",
    ENGINE_PATH.exists(),
)

check(
    "Risk engine instantiated",
    isinstance(
        engine,
        SmartSOCRiskEngine,
    ),
)

check(
    "Engine version is 4.1",
    engine.version == "4.1",
)


# ============================================================
# TEST STORAGE
# ============================================================

results = []


def record_test(
    test_id: str,
    name: str,
    passed: bool,
    details: dict,
):

    results.append(
        {
            "id": test_id,
            "name": name,
            "passed": bool(passed),
            "details": details,
        }
    )

    if passed:

        print(
            f"[PASS] {test_id} {name}"
        )

    else:

        print(
            f"[FAIL] {test_id} {name}"
        )

        print(
            f"       {details}"
        )


# ============================================================
# A01 — EXTREME MAXIMUM
# ============================================================

print("\nA01 — EXTREME MAXIMUM")

result = engine.calculate(
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

passed = (
    is_valid_score(
        result.risk_score
    )
    and
    result.risk_score <= 100
    and
    result.risk_level == "CRITICAL"
    and
    result.recommended_action
    == "CONTAINMENT_CANDIDATE"
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A01",
    "extreme_maximum",
    passed,
    {
        "score":
            result.risk_score,

        "level":
            result.risk_level,

        "action":
            result.recommended_action,
    },
)


# ============================================================
# A02 — EXTREME MINIMUM
# ============================================================

print("\nA02 — EXTREME MINIMUM")

result = engine.calculate(
    technique_severity="LOW",
    ml_confidence=0.0,
    evidence_strength="NONE",
    privilege="LOW",
    network=False,
    suspicious_parent=False,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)

passed = (
    result.risk_score == 0.0
    and
    result.risk_level == "LOW"
    and
    result.recommended_action
    == "MONITOR"
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A02",
    "extreme_minimum",
    passed,
    {
        "score":
            result.risk_score,

        "level":
            result.risk_level,

        "action":
            result.recommended_action,
    },
)


# ============================================================
# A03 — CONFIDENCE BELOW ZERO
# ============================================================

print("\nA03 — CONFIDENCE BELOW ZERO")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=-999,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "ml_confidence",
    )
    == 0.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A03",
    "negative_confidence_normalization",
    passed,
    {
        "confidence_component":
            component(
                result,
                "ml_confidence",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A04 — CONFIDENCE ABOVE ONE
# ============================================================

print("\nA04 — CONFIDENCE ABOVE ONE")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=999,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "ml_confidence",
    )
    == 20.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A04",
    "excessive_confidence_normalization",
    passed,
    {
        "confidence_component":
            component(
                result,
                "ml_confidence",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A05 — NON-NUMERIC CONFIDENCE
# ============================================================

print("\nA05 — NON-NUMERIC CONFIDENCE")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence="not-a-number",
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "ml_confidence",
    )
    == 0.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A05",
    "non_numeric_confidence",
    passed,
    {
        "confidence_component":
            component(
                result,
                "ml_confidence",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A06 — NONE CONFIDENCE
# ============================================================

print("\nA06 — NONE CONFIDENCE")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=None,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "ml_confidence",
    )
    == 0.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A06",
    "null_confidence",
    passed,
    {
        "confidence_component":
            component(
                result,
                "ml_confidence",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A07 — UNKNOWN SEVERITY
# ============================================================

print("\nA07 — UNKNOWN SEVERITY")

result = engine.calculate(
    technique_severity="UNKNOWN_SEVERITY",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=True,
    suspicious_parent=True,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "technique_severity",
    )
    == 5.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A07",
    "unknown_severity_normalization",
    passed,
    {
        "severity_component":
            component(
                result,
                "technique_severity",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A08 — UNKNOWN EVIDENCE
# ============================================================

print("\nA08 — UNKNOWN EVIDENCE")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="UNKNOWN_EVIDENCE",
    privilege="HIGH",
    network=True,
    suspicious_parent=True,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "behavioral_evidence",
    )
    == 0.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A08",
    "unknown_evidence_normalization",
    passed,
    {
        "evidence_component":
            component(
                result,
                "behavioral_evidence",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A09 — UNKNOWN PRIVILEGE
# ============================================================

print("\nA09 — UNKNOWN PRIVILEGE")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="UNKNOWN_PRIVILEGE",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

passed = (
    component(
        result,
        "context",
    )
    == 0.0
    and
    is_valid_score(
        result.risk_score
    )
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A09",
    "unknown_privilege_normalization",
    passed,
    {
        "context_component":
            component(
                result,
                "context",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A10 — STRING BOOLEAN ATTACK
# ============================================================

print("\nA10 — STRING BOOLEAN ATTACK")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="HIGH",
    network="yes",
    suspicious_parent="true",
    fallback="false",
    ambiguous="false",
    missing_behavior="false",
)

passed = (
    component(
        result,
        "context",
    )
    == 14.0
    and
    component(
        result,
        "uncertainty_penalty",
    )
    == 0.0
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A10",
    "string_boolean_normalization",
    passed,
    {
        "context":
            component(
                result,
                "context",
            ),

        "uncertainty":
            component(
                result,
                "uncertainty_penalty",
            ),

        "score":
            result.risk_score,
    },
)


# ============================================================
# A11 — MAXIMUM UNCERTAINTY
# ============================================================

print("\nA11 — MAXIMUM UNCERTAINTY")

certain = engine.calculate(
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

uncertain = engine.calculate(
    technique_severity="CRITICAL",
    ml_confidence=1.0,
    evidence_strength="VERY_STRONG",
    privilege="SYSTEM",
    network=True,
    suspicious_parent=True,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)

passed = (
    component(
        uncertain,
        "uncertainty_penalty",
    )
    == 10.0
    and
    uncertain.risk_score
    <
    certain.risk_score
    and
    safety_is_intact(
        uncertain
    )
)

record_test(
    "A11",
    "maximum_uncertainty_penalty",
    passed,
    {
        "certain_score":
            certain.risk_score,

        "uncertain_score":
            uncertain.risk_score,

        "penalty":
            component(
                uncertain,
                "uncertainty_penalty",
            ),
    },
)


# ============================================================
# A12 — FALLBACK + CRITICAL INPUT
# ============================================================

print("\nA12 — FALLBACK + CRITICAL INPUT")

result = engine.calculate(
    technique_severity="CRITICAL",
    ml_confidence=1.0,
    evidence_strength="VERY_STRONG",
    privilege="SYSTEM",
    network=True,
    suspicious_parent=True,
    fallback=True,
    ambiguous=True,
    missing_behavior=False,
)

passed = (
    result.risk_level
    == "CRITICAL"
    and
    result.recommended_action
    == "CONTAINMENT_CANDIDATE"
    and
    result.safety[
        "fallback_can_directly_contain"
    ]
    is False
    and
    result.safety[
        "requires_analyst_for_containment"
    ]
    is True
    and
    result.safety[
        "automatic_response"
    ]
    is False
)

record_test(
    "A12",
    "fallback_cannot_bypass_analyst",
    passed,
    {
        "score":
            result.risk_score,

        "level":
            result.risk_level,

        "action":
            result.recommended_action,

        "fallback_direct_containment":
            result.safety[
                "fallback_can_directly_contain"
            ],
    },
)


# ============================================================
# A13 — MISSING BEHAVIOR ATTACK
# ============================================================

print("\nA13 — MISSING BEHAVIOR ATTACK")

with_behavior = engine.calculate(
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

missing_behavior = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.90,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=True,
    suspicious_parent=True,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)

passed = (
    missing_behavior.risk_score
    <=
    with_behavior.risk_score
    and
    component(
        missing_behavior,
        "uncertainty_penalty",
    )
    == 10.0
    and
    safety_is_intact(
        missing_behavior
    )
)

record_test(
    "A13",
    "missing_behavior_cannot_inflate_risk",
    passed,
    {
        "with_behavior":
            with_behavior.risk_score,

        "missing_behavior":
            missing_behavior.risk_score,

        "penalty":
            component(
                missing_behavior,
                "uncertainty_penalty",
            ),
    },
)


# ============================================================
# A14 — CONTRADICTORY SIGNALS
# ============================================================

print("\nA14 — CONTRADICTORY SIGNALS")

result = engine.calculate(
    technique_severity="CRITICAL",
    ml_confidence=0.20,
    evidence_strength="NONE",
    privilege="LOW",
    network=False,
    suspicious_parent=False,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)

passed = (
    is_valid_score(
        result.risk_score
    )
    and
    result.risk_score < 75
    and
    result.recommended_action
    != "CONTAINMENT_CANDIDATE"
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A14",
    "contradictory_signals_are_conservative",
    passed,
    {
        "score":
            result.risk_score,

        "level":
            result.risk_level,

        "action":
            result.recommended_action,
    },
)


# ============================================================
# A15 — ALL BOOLEAN VALUES TRUE
# ============================================================

print("\nA15 — ALL BOOLEAN VALUES TRUE")

result = engine.calculate(
    technique_severity="MEDIUM",
    ml_confidence=0.50,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=True,
    suspicious_parent=True,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)

passed = (
    is_valid_score(
        result.risk_score
    )
    and
    component(
        result,
        "uncertainty_penalty",
    )
    == 10.0
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A15",
    "maximum_boolean_uncertainty_combination",
    passed,
    {
        "score":
            result.risk_score,

        "level":
            result.risk_level,

        "uncertainty":
            component(
                result,
                "uncertainty_penalty",
            ),
    },
)


# ============================================================
# A16 — FALSEY / TRUTHY INPUTS
# ============================================================

print("\nA16 — FALSEY / TRUTHY INPUTS")

result = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=1,
    suspicious_parent=1,
    fallback=0,
    ambiguous=0,
    missing_behavior=0,
)

passed = (
    component(
        result,
        "context",
    )
    == 14.0
    and
    component(
        result,
        "uncertainty_penalty",
    )
    == 0.0
    and
    safety_is_intact(
        result
    )
)

record_test(
    "A16",
    "numeric_boolean_normalization",
    passed,
    {
        "context":
            component(
                result,
                "context",
            ),

        "uncertainty":
            component(
                result,
                "uncertainty_penalty",
            ),
    },
)


# ============================================================
# A17 — DETERMINISM
# ============================================================

print("\nA17 — DETERMINISM")

inputs = {
    "technique_severity": "HIGH",
    "ml_confidence": 0.83,
    "evidence_strength": "STRONG",
    "privilege": "HIGH",
    "network": True,
    "suspicious_parent": True,
    "fallback": False,
    "ambiguous": False,
    "missing_behavior": False,
}

r1 = engine.calculate(
    **inputs
)

r2 = engine.calculate(
    **inputs
)

passed = (
    r1.to_dict()
    ==
    r2.to_dict()
)

record_test(
    "A17",
    "deterministic_output",
    passed,
    {
        "first":
            r1.to_dict(),

        "second":
            r2.to_dict(),
    },
)


# ============================================================
# A18 — GLOBAL RESPONSE SAFETY
# ============================================================

print("\nA18 — GLOBAL RESPONSE SAFETY")

safety_cases = [

    engine.calculate(
        technique_severity="LOW",
        ml_confidence=0.1,
        evidence_strength="NONE",
        privilege="LOW",
        network=False,
        suspicious_parent=False,
        fallback=True,
        ambiguous=True,
        missing_behavior=True,
    ),

    engine.calculate(
        technique_severity="MEDIUM",
        ml_confidence=0.5,
        evidence_strength="MODERATE",
        privilege="MEDIUM",
        network=True,
        suspicious_parent=False,
        fallback=False,
        ambiguous=True,
        missing_behavior=False,
    ),

    engine.calculate(
        technique_severity="HIGH",
        ml_confidence=0.9,
        evidence_strength="STRONG",
        privilege="HIGH",
        network=True,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    ),

    engine.calculate(
        technique_severity="CRITICAL",
        ml_confidence=1.0,
        evidence_strength="VERY_STRONG",
        privilege="SYSTEM",
        network=True,
        suspicious_parent=True,
        fallback=False,
        ambiguous=False,
        missing_behavior=False,
    ),

    engine.calculate(
        technique_severity="CRITICAL",
        ml_confidence=1.0,
        evidence_strength="VERY_STRONG",
        privilege="SYSTEM",
        network=True,
        suspicious_parent=True,
        fallback=True,
        ambiguous=True,
        missing_behavior=True,
    ),
]


unsafe_results = []

for idx, decision in enumerate(
    safety_cases,
    start=1,
):

    if not safety_is_intact(
        decision
    ):

        unsafe_results.append(
            {
                "case":
                    idx,

                "result":
                    decision.to_dict(),
            }
        )


passed = (
    len(unsafe_results) == 0
)

record_test(
    "A18",
    "global_response_safety",
    passed,
    {
        "unsafe_results":
            unsafe_results,
    },
)


# ============================================================
# A19 — SCORE BOUNDARY INTEGRITY
# ============================================================

print("\nA19 — SCORE BOUNDARY INTEGRITY")

boundary_inputs = [

    {
        "technique_severity": "LOW",
        "ml_confidence": 0.0,
        "evidence_strength": "NONE",
        "privilege": "LOW",
        "network": False,
        "suspicious_parent": False,
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
    },

    {
        "technique_severity": "MEDIUM",
        "ml_confidence": 0.5,
        "evidence_strength": "MODERATE",
        "privilege": "MEDIUM",
        "network": False,
        "suspicious_parent": False,
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
    },

    {
        "technique_severity": "CRITICAL",
        "ml_confidence": 1.0,
        "evidence_strength": "VERY_STRONG",
        "privilege": "SYSTEM",
        "network": True,
        "suspicious_parent": True,
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
    },
]


boundary_results = []

for inputs in boundary_inputs:

    decision = engine.calculate(
        **inputs
    )

    boundary_results.append(
        decision.risk_score
    )


passed = all(
    is_valid_score(score)
    for score in boundary_results
)

record_test(
    "A19",
    "score_boundary_integrity",
    passed,
    {
        "scores":
            boundary_results,
    },
)


# ============================================================
# A20 — ACTION WHITELIST
# ============================================================

print("\nA20 — ACTION WHITELIST")

action_safe = all(
    decision.recommended_action
    in VALID_ACTIONS
    for decision in safety_cases
)

record_test(
    "A20",
    "action_whitelist_integrity",
    action_safe,
    {
        "actions":
            [
                decision.recommended_action
                for decision in safety_cases
            ],
    },
)


# ============================================================
# A21 — LEVEL WHITELIST
# ============================================================

print("\nA21 — LEVEL WHITELIST")

level_safe = all(
    decision.risk_level
    in VALID_LEVELS
    for decision in safety_cases
)

record_test(
    "A21",
    "risk_level_whitelist_integrity",
    level_safe,
    {
        "levels":
            [
                decision.risk_level
                for decision in safety_cases
            ],
    },
)


# ============================================================
# A22 — CONTAINMENT ALWAYS ANALYST-GATED
# ============================================================

print("\nA22 — CONTAINMENT ANALYST GATING")

containment_decisions = [
    decision
    for decision in safety_cases
    if decision.recommended_action
    ==
    "CONTAINMENT_CANDIDATE"
]

containment_safe = all(
    decision.safety[
        "requires_analyst_for_containment"
    ]
    is True
    and
    decision.safety[
        "automatic_response"
    ]
    is False
    for decision in containment_decisions
)

record_test(
    "A22",
    "containment_requires_analyst",
    containment_safe,
    {
        "containment_count":
            len(containment_decisions),
    },
)


# ============================================================
# A23 — FALLBACK NEVER DIRECTLY CONTAINS
# ============================================================

print("\nA23 — FALLBACK CONTAINMENT PROTECTION")

fallback_decisions = [

    engine.calculate(
        technique_severity=severity,
        ml_confidence=confidence,
        evidence_strength=evidence,
        privilege="SYSTEM",
        network=True,
        suspicious_parent=True,
        fallback=True,
        ambiguous=True,
        missing_behavior=False,
    )

    for severity, confidence, evidence in [
        (
            "LOW",
            0.1,
            "WEAK",
        ),
        (
            "MEDIUM",
            0.5,
            "MODERATE",
        ),
        (
            "HIGH",
            0.9,
            "STRONG",
        ),
        (
            "CRITICAL",
            1.0,
            "VERY_STRONG",
        ),
    ]
]


fallback_safe = all(
    decision.safety[
        "fallback_can_directly_contain"
    ]
    is False
    and
    decision.safety[
        "requires_analyst_for_containment"
    ]
    is True
    for decision in fallback_decisions
)

record_test(
    "A23",
    "fallback_never_directly_contains",
    fallback_safe,
    {
        "results":
            [
                {
                    "score":
                        decision.risk_score,

                    "level":
                        decision.risk_level,

                    "action":
                        decision.recommended_action,
                }

                for decision in fallback_decisions
            ],
    },
)


# ============================================================
# A24 — MISSING BEHAVIOR NEVER ADDS POSITIVE EVIDENCE
# ============================================================

print(
    "\nA24 — MISSING BEHAVIOR "
    "EVIDENCE PROTECTION"
)

base = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.8,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=True,
    suspicious_parent=True,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)

missing = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.8,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=True,
    suspicious_parent=True,
    fallback=False,
    ambiguous=False,
    missing_behavior=True,
)

passed = (
    component(
        missing,
        "behavioral_evidence",
    )
    ==
    component(
        base,
        "behavioral_evidence",
    )
    and
    missing.risk_score
    <=
    base.risk_score
)

record_test(
    "A24",
    "missing_behavior_not_positive_evidence",
    passed,
    {
        "base_score":
            base.risk_score,

        "missing_score":
            missing.risk_score,

        "base_evidence":
            component(
                base,
                "behavioral_evidence",
            ),

        "missing_evidence":
            component(
                missing,
                "behavioral_evidence",
            ),
    },
)


# ============================================================
# A25 — SIDE-EFFECT SAFETY CONTRACT
# ============================================================

print("\nA25 — SIDE-EFFECT SAFETY CONTRACT")

side_effects = {
    "model_modified": False,
    "thresholds_modified": False,
    "dataset_modified": False,
    "wazuh_modified": False,
    "shuffle_called": False,
    "response_executed": False,
    "automatic_containment": False,
}

side_effect_safe = all(
    value is False
    for value in side_effects.values()
)

record_test(
    "A25",
    "side_effect_safety_contract",
    side_effect_safe,
    side_effects,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("ADVERSARIAL TEST SUMMARY")
print("=" * 72)

passed_tests = sum(
    test["passed"]
    for test in results
)

failed_tests = (
    len(results)
    -
    passed_tests
)

pass_rate = (
    passed_tests
    /
    len(results)
)


print(
    f"Tests total  : "
    f"{len(results)}"
)

print(
    f"Tests passed : "
    f"{passed_tests}"
)

print(
    f"Tests failed : "
    f"{failed_tests}"
)

print(
    f"Pass rate    : "
    f"{pass_rate:.6f}"
)


# ============================================================
# QUALITY CHECKS
# ============================================================

check(
    "All adversarial tests passed",
    failed_tests == 0,
)

check(
    "Pass rate is 100%",
    pass_rate == 1.0,
)

check(
    "No automatic response is authorized",
    all(
        decision.safety[
            "automatic_response"
        ]
        is False
        for decision in safety_cases
    ),
)

check(
    "All containment decisions require analyst",
    all(
        decision.safety[
            "requires_analyst_for_containment"
        ]
        is True
        for decision in safety_cases
    ),
)

check(
    "Fallback cannot directly contain",
    all(
        decision.safety[
            "fallback_can_directly_contain"
        ]
        is False
        for decision in fallback_decisions
    ),
)

check(
    "All generated scores remain bounded",
    all(
        is_valid_score(
            decision.risk_score
        )
        for decision in safety_cases
    ),
)


# ============================================================
# FINAL DECISION
# ============================================================

all_checks_pass = all(
    quality_checks.values()
)

if all_checks_pass:

    decision = "PASS"

else:

    decision = "REVIEW_REQUIRED"


# ============================================================
# REPORT
# ============================================================

report = {

    "project":
        "SmartSOC",

    "phase":
        "3D",

    "gate":
        "3D.4",

    "name":
        "Adversarial Safety Validation",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "engine_version":
        engine.version,

    "decision":
        decision,

    "test_summary": {

        "total":
            len(results),

        "passed":
            passed_tests,

        "failed":
            failed_tests,

        "pass_rate":
            pass_rate,
    },

    "quality_checks":
        quality_checks,

    "tests":
        results,

    "production_safety": {

        "model_modified":
            False,

        "thresholds_modified":
            False,

        "dataset_modified":
            False,

        "wazuh_modified":
            False,

        "shuffle_called":
            False,

        "response_executed":
            False,

        "automatic_containment":
            False,
    },

    "safety_contract": {

        "automatic_response":
            False,

        "automatic_blocking":
            False,

        "automatic_isolation":
            False,

        "automatic_process_kill":
            False,

        "analyst_required_for_containment":
            True,

        "fallback_can_directly_contain":
            False,

        "missing_behavior_can_increase_score":
            False,
    },

    "limitations": [

        (
            "Adversarial validation tests deterministic "
            "engine behavior and does not constitute a "
            "formal security audit."
        ),

        (
            "The engine does not independently verify "
            "the truthfulness of upstream ML confidence "
            "or evidence fields."
        ),

        (
            "The current response policy intentionally "
            "stops at CONTAINMENT_CANDIDATE and requires "
            "analyst approval."
        ),

        (
            "No external response system was invoked."
        ),
    ],

    "output":
        str(FINAL_REPORT_PATH),
}


# ============================================================
# WRITE RESULTS
# ============================================================

with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "project":
                "SmartSOC",

            "phase":
                "3D",

            "gate":
                "3D.4",

            "engine_version":
                engine.version,

            "decision":
                decision,

            "test_summary": {

                "total":
                    len(results),

                "passed":
                    passed_tests,

                "failed":
                    failed_tests,

                "pass_rate":
                    pass_rate,
            },

            "tests":
                results,

            "quality_checks":
                quality_checks,

            "production_safety":
                report[
                    "production_safety"
                ],
        },
        f,
        indent=2,
    )


with FINAL_REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 72)
print("GATE 3D.4 FINAL RESULT")
print("=" * 72)

print(
    f"Decision             : "
    f"{decision}"
)

print(
    f"Adversarial tests    : "
    f"{len(results)}"
)

print(
    f"Tests passed         : "
    f"{passed_tests}"
)

print(
    f"Tests failed         : "
    f"{failed_tests}"
)

print(
    f"Pass rate            : "
    f"{pass_rate:.6f}"
)

print(
    "Automatic response   : DISABLED"
)

print(
    "Automatic containment: DISABLED"
)

print(
    "Shuffle execution    : NONE"
)

print(
    "Wazuh modification   : NONE"
)

print(
    "Response execution   : NONE"
)

print(
    "Containment          : ANALYST REQUIRED"
)

print("\nOUTPUTS")

print(
    f"  {RESULTS_PATH}"
)

print(
    f"  {FINAL_REPORT_PATH}"
)

print("=" * 72)


if decision != "PASS":

    raise RuntimeError(
        "Phase 3D.4 requires review."
    )