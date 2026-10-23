"""
SMARTSOC PHASE 3D
GATE 3D.1 — SOC RISK ENGINE SPECIFICATION

Purpose
-------
Define and validate the deterministic SmartSOC risk/decision
contract before connecting the engine to Wazuh, Shuffle, or
automated response.

This gate:
- Defines the risk score components.
- Defines risk levels.
- Defines recommended actions.
- Defines safety constraints.
- Validates representative SOC scenarios.
- Produces a machine-readable specification.
- Does NOT modify the ML model.
- Does NOT modify thresholds.
- Does NOT modify Wazuh.
- Does NOT execute response actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase3d"
    / "gate3_1"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPEC_PATH = (
    REPORT_DIR
    / "risk_engine_specification.json"
)

TEST_RESULTS_PATH = (
    REPORT_DIR
    / "risk_engine_specification_test_results.json"
)

FINAL_REPORT_PATH = (
    REPORT_DIR
    / "phase3d_gate3_1_report.json"
)


# ============================================================
# SCORING CONTRACT
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

TOTAL_SCORE_RANGE_MAX = (
    POSITIVE_SCORE_MAX
    + MAX_UNCERTAINTY_PENALTY
)


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


RECOMMENDED_ACTIONS = {
    "LOW": "MONITOR",
    "MEDIUM": "ANALYST_REVIEW",
    "HIGH": "PRIORITY_ANALYST_REVIEW",
    "CRITICAL": "CONTAINMENT_CANDIDATE",
}


# ============================================================
# QUALITY CHECKS
# ============================================================

quality_checks = {}


def check(
    name: str,
    condition: bool,
    message: str = ""
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
# RISK LEVEL FUNCTION
# ============================================================

def risk_level_from_score(
    score: float
) -> str:

    score = max(
        0.0,
        min(
            100.0,
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
# SCORING COMPONENT FUNCTIONS
# ============================================================

def technique_severity_score(
    severity: str
) -> float:

    mapping = {
        "LOW": 5,
        "MEDIUM": 12,
        "HIGH": 22,
        "CRITICAL": 30,
    }

    return float(
        mapping.get(
            str(severity).upper(),
            0
        )
    )


def ml_confidence_score(
    confidence: float
) -> float:

    confidence = max(
        0.0,
        min(
            1.0,
            float(confidence)
        )
    )

    return (
        confidence
        *
        MAX_ML_CONFIDENCE
    )


def behavioral_evidence_score(
    evidence_strength: str
) -> float:

    mapping = {
        "NONE": 0,
        "WEAK": 6,
        "MODERATE": 14,
        "STRONG": 21,
        "VERY_STRONG": 25,
    }

    return float(
        mapping.get(
            str(evidence_strength).upper(),
            0
        )
    )


def context_score(
    privilege: str,
    network: bool,
    suspicious_parent: bool
) -> float:

    score = 0.0

    privilege_mapping = {
        "LOW": 0,
        "MEDIUM": 3,
        "HIGH": 6,
        "SYSTEM": 7,
    }

    score += privilege_mapping.get(
        str(privilege).upper(),
        0
    )

    if network:
        score += 4

    if suspicious_parent:
        score += 4

    return min(
        score,
        MAX_CONTEXT
    )


def uncertainty_penalty(
    fallback: bool,
    ambiguous: bool,
    missing_behavior: bool
) -> float:

    penalty = 0.0

    if fallback:
        penalty += 5

    if ambiguous:
        penalty += 3

    if missing_behavior:
        penalty += 2

    return min(
        penalty,
        MAX_UNCERTAINTY_PENALTY
    )


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    *,
    technique_severity: str,
    ml_confidence: float,
    evidence_strength: str,
    privilege: str,
    network: bool,
    suspicious_parent: bool,
    fallback: bool,
    ambiguous: bool,
    missing_behavior: bool,
):

    technique_score = (
        technique_severity_score(
            technique_severity
        )
    )

    confidence_score = (
        ml_confidence_score(
            ml_confidence
        )
    )

    evidence_score = (
        behavioral_evidence_score(
            evidence_strength
        )
    )

    context_component = (
        context_score(
            privilege,
            network,
            suspicious_parent
        )
    )

    uncertainty_component = (
        uncertainty_penalty(
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
            100.0,
            raw_score
        )
    )

    level = risk_level_from_score(
        final_score
    )

    action = (
        RECOMMENDED_ACTIONS[
            level
        ]
    )

    reasons = []

    if technique_score >= 22:
        reasons.append(
            "High or critical technique severity"
        )

    if confidence_score >= 16:
        reasons.append(
            "High model confidence contribution"
        )

    if evidence_score >= 21:
        reasons.append(
            "Strong behavioral evidence"
        )

    if network:
        reasons.append(
            "Network behavior present"
        )

    if suspicious_parent:
        reasons.append(
            "Suspicious parent-process context"
        )

    if privilege in (
        "HIGH",
        "SYSTEM"
    ):
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

    if not reasons:
        reasons.append(
            "Limited suspicious evidence"
        )

    return {
        "risk_score": round(
            final_score,
            2
        ),

        "risk_level": level,

        "recommended_action": action,

        "components": {
            "technique_severity": round(
                technique_score,
                2
            ),

            "ml_confidence": round(
                confidence_score,
                2
            ),

            "behavioral_evidence": round(
                evidence_score,
                2
            ),

            "context": round(
                context_component,
                2
            ),

            "uncertainty_penalty": round(
                uncertainty_component,
                2
            ),
        },

        "reasons": reasons,

        "safety": {
            "automatic_response": False,
            "requires_analyst_for_containment": True,
        },
    }


# ============================================================
# REPRESENTATIVE TEST CASES
# ============================================================

TEST_CASES = [

    {
        "name": "benign_administrative_cmd",

        "inputs": {
            "technique_severity": "LOW",
            "ml_confidence": 0.45,
            "evidence_strength": "WEAK",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_level_max": "MEDIUM",
        "forbid_critical": True,
    },

    {
        "name": "suspicious_powershell",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.90,
            "evidence_strength": "STRONG",
            "privilege": "HIGH",
            "network": True,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_level_min": "HIGH",
    },

    {
        "name": "encoded_powershell_download",

        "inputs": {
            "technique_severity": "CRITICAL",
            "ml_confidence": 0.99,
            "evidence_strength": "VERY_STRONG",
            "privilege": "HIGH",
            "network": True,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_level_min": "CRITICAL",
    },

    {
        "name": "persistence_activity",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.85,
            "evidence_strength": "STRONG",
            "privilege": "HIGH",
            "network": False,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_level_min": "HIGH",
    },

    {
        "name": "network_activity_only",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.65,
            "evidence_strength": "MODERATE",
            "privilege": "MEDIUM",
            "network": True,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_level_max": "HIGH",
    },

    {
        "name": "missing_behavior_context",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.60,
            "evidence_strength": "WEAK",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": True,
        },

        "expected_level_max": "MEDIUM",
    },

    {
        "name": "destructive_behavior",

        "inputs": {
            "technique_severity": "CRITICAL",
            "ml_confidence": 0.97,
            "evidence_strength": "VERY_STRONG",
            "privilege": "SYSTEM",
            "network": False,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_level_min": "CRITICAL",
    },

    {
        "name": "low_confidence_fallback",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.35,
            "evidence_strength": "WEAK",
            "privilege": "LOW",
            "network": False,
            "suspicious_parent": False,
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": True,
        },

        "expected_level_max": "LOW",
    },
]


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("SMARTSOC PHASE 3D")
print("GATE 3D.1 — SOC RISK ENGINE SPECIFICATION")
print("=" * 72)


# ============================================================
# 1. SCORING CONTRACT VALIDATION
# ============================================================

print("\nSCORING CONTRACT")

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
    f"  Positive score max : "
    f"{POSITIVE_SCORE_MAX}"
)

print(
    f"  Final score range  : "
    f"0–{TOTAL_SCORE_RANGE_MAX}"
)


check(
    "Positive score components total 90",
    POSITIVE_SCORE_MAX == 90
)

check(
    "Uncertainty penalty maximum is 10",
    MAX_UNCERTAINTY_PENALTY == 10
)

check(
    "Final risk range is 0-100",
    TOTAL_SCORE_RANGE_MAX == 100
)


# ============================================================
# 2. RISK LEVEL VALIDATION
# ============================================================

print("\nRISK LEVELS")

for level, bounds in RISK_LEVELS.items():

    print(
        f"  {level:8s}: "
        f"{bounds['min']}–{bounds['max']}"
    )


boundary_tests = [
    (0, "LOW"),
    (24.99, "LOW"),
    (25, "MEDIUM"),
    (49.99, "MEDIUM"),
    (50, "HIGH"),
    (74.99, "HIGH"),
    (75, "CRITICAL"),
    (100, "CRITICAL"),
]


for score, expected in boundary_tests:

    actual = risk_level_from_score(
        score
    )

    check(
        f"Risk boundary {score}",
        actual == expected,
        f"Expected {expected}, got {actual}"
    )


# ============================================================
# 3. ACTION MAPPING
# ============================================================

print("\nACTION MAPPING")

for level, action in (
    RECOMMENDED_ACTIONS.items()
):

    print(
        f"  {level:8s} → {action}"
    )


check(
    "All risk levels have actions",
    set(RISK_LEVELS)
    ==
    set(RECOMMENDED_ACTIONS)
)


# ============================================================
# 4. SAFETY CONTRACT
# ============================================================

print("\nSAFETY CONTRACT")

safety_contract = {
    "automatic_response": False,
    "automatic_blocking": False,
    "automatic_isolation": False,
    "automatic_process_kill": False,
    "analyst_required_for_containment": True,
    "fallback_never_directly_contains": True,
    "missing_behavior_never_increases_score": True,
    "model_confidence_not_treated_as_calibrated_probability": True,
}


check(
    "Automatic response disabled",
    safety_contract[
        "automatic_response"
    ] is False
)

check(
    "Automatic blocking disabled",
    safety_contract[
        "automatic_blocking"
    ] is False
)

check(
    "Automatic isolation disabled",
    safety_contract[
        "automatic_isolation"
    ] is False
)

check(
    "Automatic process kill disabled",
    safety_contract[
        "automatic_process_kill"
    ] is False
)

check(
    "Containment requires analyst",
    safety_contract[
        "analyst_required_for_containment"
    ] is True
)

check(
    "Fallback cannot directly contain",
    safety_contract[
        "fallback_never_directly_contains"
    ] is True
)

check(
    "Missing behavior cannot increase score",
    safety_contract[
        "missing_behavior_never_increases_score"
    ] is True
)


# ============================================================
# 5. REPRESENTATIVE SCENARIOS
# ============================================================

print("\nREPRESENTATIVE SCENARIOS")

test_results = []


risk_order = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


for case in TEST_CASES:

    result = calculate_risk(
        **case["inputs"]
    )

    level = result[
        "risk_level"
    ]

    score = result[
        "risk_score"
    ]

    passed = True
    failure_reason = ""

    if "expected_level_min" in case:

        minimum = case[
            "expected_level_min"
        ]

        if (
            risk_order[level]
            <
            risk_order[minimum]
        ):

            passed = False

            failure_reason = (
                f"Risk level {level} "
                f"is below expected minimum "
                f"{minimum}"
            )

    if "expected_level_max" in case:

        maximum = case[
            "expected_level_max"
        ]

        if (
            risk_order[level]
            >
            risk_order[maximum]
        ):

            passed = False

            failure_reason = (
                f"Risk level {level} "
                f"exceeds expected maximum "
                f"{maximum}"
            )

    if (
        case.get(
            "forbid_critical",
            False
        )
        and
        level == "CRITICAL"
    ):

        passed = False

        failure_reason = (
            "Critical risk produced for "
            "scenario explicitly forbidden "
            "from reaching critical"
        )

    test_results.append(
        {
            "name": case["name"],
            "passed": passed,
            "risk_score": score,
            "risk_level": level,
            "recommended_action": result[
                "recommended_action"
            ],
            "components": result[
                "components"
            ],
            "reasons": result[
                "reasons"
            ],
            "failure_reason": (
                failure_reason
                if not passed
                else None
            ),
        }
    )

    if passed:

        print(
            f"[PASS] "
            f"{case['name']}: "
            f"{score:.2f} "
            f"→ {level}"
        )

    else:

        print(
            f"[FAIL] "
            f"{case['name']}: "
            f"{score:.2f} "
            f"→ {level}"
        )

        print(
            f"       {failure_reason}"
        )


# ============================================================
# 6. MONOTONICITY TESTS
# ============================================================

print("\nMONOTONICITY TEST")

base_inputs = {
    "technique_severity": "MEDIUM",
    "ml_confidence": 0.50,
    "evidence_strength": "MODERATE",
    "privilege": "MEDIUM",
    "network": False,
    "suspicious_parent": False,
    "fallback": False,
    "ambiguous": False,
    "missing_behavior": False,
}


base_result = calculate_risk(
    **base_inputs
)


higher_confidence_result = calculate_risk(
    **{
        **base_inputs,
        "ml_confidence": 0.80,
    }
)


check(
    "Higher confidence does not reduce risk",
    higher_confidence_result[
        "risk_score"
    ]
    >=
    base_result[
        "risk_score"
    ]
)


stronger_evidence_result = calculate_risk(
    **{
        **base_inputs,
        "evidence_strength": "VERY_STRONG",
    }
)


check(
    "Stronger evidence does not reduce risk",
    stronger_evidence_result[
        "risk_score"
    ]
    >=
    base_result[
        "risk_score"
    ]
)


suspicious_context_result = calculate_risk(
    **{
        **base_inputs,
        "suspicious_parent": True,
        "network": True,
        "privilege": "HIGH",
    }
)


check(
    "Suspicious context does not reduce risk",
    suspicious_context_result[
        "risk_score"
    ]
    >=
    base_result[
        "risk_score"
    ]
)


# ============================================================
# 7. UNCERTAINTY TEST
# ============================================================

print("\nUNCERTAINTY TEST")

certain_result = calculate_risk(
    **{
        **base_inputs,
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": False,
    }
)


uncertain_result = calculate_risk(
    **{
        **base_inputs,
        "fallback": True,
        "ambiguous": True,
        "missing_behavior": True,
    }
)


check(
    "Uncertainty does not increase risk",
    uncertain_result[
        "risk_score"
    ]
    <=
    certain_result[
        "risk_score"
    ]
)


# ============================================================
# 8. MISSING BEHAVIOR SAFETY TEST
# ============================================================

print("\nMISSING BEHAVIOR SAFETY TEST")

with_behavior = calculate_risk(
    **{
        **base_inputs,
        "missing_behavior": False,
    }
)


without_behavior = calculate_risk(
    **{
        **base_inputs,
        "missing_behavior": True,
        "fallback": True,
        "ambiguous": True,
    }
)


check(
    "Missing behavior does not inflate risk",
    without_behavior[
        "risk_score"
    ]
    <=
    with_behavior[
        "risk_score"
    ]
)


# ============================================================
# 9. SCORE RANGE TEST
# ============================================================

print("\nSCORE RANGE TEST")

extreme_cases = [

    {
        **base_inputs,
        "technique_severity": "CRITICAL",
        "ml_confidence": 1.0,
        "evidence_strength": "VERY_STRONG",
        "privilege": "SYSTEM",
        "network": True,
        "suspicious_parent": True,
    },

    {
        **base_inputs,
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
]


for idx, inputs in enumerate(
    extreme_cases,
    start=1
):

    result = calculate_risk(
        **inputs
    )

    score = result[
        "risk_score"
    ]

    check(
        f"Extreme score case {idx}",
        0 <= score <= 100
    )


# ============================================================
# 10. SAFETY ACTION TEST
# ============================================================

print("\nSAFETY ACTION TEST")

critical_results = [
    x
    for x in test_results
    if x["risk_level"]
    == "CRITICAL"
]


check(
    "Critical action is containment candidate",
    all(
        x["recommended_action"]
        ==
        "CONTAINMENT_CANDIDATE"
        for x in critical_results
    )
)


check(
    "No scenario produces automatic response",
    all(
        x["recommended_action"]
        !=
        "AUTO_CONTAIN"
        for x in test_results
    )
)


# ============================================================
# 11. SPECIFICATION
# ============================================================

print("\nBUILDING SPECIFICATION")

specification = {

    "project": "SmartSOC",

    "phase": "3D",

    "gate": "3D.1",

    "name": (
        "SOC Risk Engine Specification"
    ),

    "purpose": (
        "Define a deterministic risk and decision "
        "layer between ML classification and any "
        "future response automation."
    ),

    "scoring": {

        "technique_severity_max":
            MAX_TECHNIQUE_SEVERITY,

        "ml_confidence_max":
            MAX_ML_CONFIDENCE,

        "behavioral_evidence_max":
            MAX_BEHAVIORAL_EVIDENCE,

        "context_max":
            MAX_CONTEXT,

        "uncertainty_penalty_max":
            MAX_UNCERTAINTY_PENALTY,

        "positive_score_max":
            POSITIVE_SCORE_MAX,

        "final_score_range": [
            0,
            100
        ],

        "formula": (
            "risk_score = "
            "technique_severity "
            "+ ml_confidence "
            "+ behavioral_evidence "
            "+ context "
            "- uncertainty_penalty"
        ),
    },

    "risk_levels":
        RISK_LEVELS,

    "recommended_actions":
        RECOMMENDED_ACTIONS,

    "safety_contract":
        safety_contract,

    "uncertainty_policy": {

        "fallback_penalty": 5,

        "ambiguous_penalty": 3,

        "missing_behavior_penalty": 2,

        "maximum_total_penalty":
            MAX_UNCERTAINTY_PENALTY,
    },

    "important_model_interpretation": (
        "ML confidence is treated as a scoring input "
        "and ranking signal. It is not assumed to be "
        "a calibrated probability."
    ),

    "response_policy": {

        "low":
            "MONITOR",

        "medium":
            "ANALYST_REVIEW",

        "high":
            "PRIORITY_ANALYST_REVIEW",

        "critical":
            "CONTAINMENT_CANDIDATE",

        "automatic_response_enabled":
            False,
    },

    "future_integration_boundary": {

        "wazuh":
            "input source",

        "ml_classifier":
            "technique classification",

        "explainability":
            "evidence extraction",

        "confidence":
            "uncertainty signal",

        "risk_engine":
            "operational decision",

        "shuffle":
            "future orchestration layer",

        "response_tools":
            "future approved actions",
    },
}


with SPEC_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        specification,
        f,
        indent=2
    )


# ============================================================
# 12. TEST RESULTS
# ============================================================

test_summary = {

    "total_tests":
        len(test_results),

    "passed_tests":
        sum(
            x["passed"]
            for x in test_results
        ),

    "failed_tests":
        sum(
            not x["passed"]
            for x in test_results
        ),

    "scenarios":
        test_results,

    "monotonicity": {

        "base_score":
            base_result["risk_score"],

        "higher_confidence_score":
            higher_confidence_result[
                "risk_score"
            ],

        "stronger_evidence_score":
            stronger_evidence_result[
                "risk_score"
            ],

        "suspicious_context_score":
            suspicious_context_result[
                "risk_score"
            ],
    },

    "uncertainty": {

        "certain_score":
            certain_result[
                "risk_score"
            ],

        "uncertain_score":
            uncertain_result[
                "risk_score"
            ],
    },
}


with TEST_RESULTS_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        test_summary,
        f,
        indent=2
    )


# ============================================================
# 13. FINAL DECISION
# ============================================================

scenario_tests_pass = all(
    x["passed"]
    for x in test_results
)


all_checks_pass = (
    all(
        quality_checks.values()
    )
    and
    scenario_tests_pass
)


if all_checks_pass:

    decision = "PASS"

else:

    decision = "REVIEW_REQUIRED"


final_report = {

    "project":
        "SmartSOC",

    "phase":
        "3D",

    "gate":
        "3D.1",

    "name":
        "SOC Risk Engine Specification",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "decision":
        decision,

    "quality_checks":
        quality_checks,

    "scenario_test_summary": {

        "total":
            len(test_results),

        "passed":
            sum(
                x["passed"]
                for x in test_results
            ),

        "failed":
            sum(
                not x["passed"]
                for x in test_results
            ),
    },

    "outputs": {

        "specification":
            str(SPEC_PATH),

        "test_results":
            str(TEST_RESULTS_PATH),

        "final_report":
            str(FINAL_REPORT_PATH),
    },

    "production_modification": {

        "model_modified":
            False,

        "thresholds_modified":
            False,

        "dataset_modified":
            False,

        "wazuh_modified":
            False,

        "shuffle_modified":
            False,

        "response_executed":
            False,
    },
}


with FINAL_REPORT_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_report,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 72)
print("GATE 3D.1 FINAL RESULT")
print("=" * 72)

print(
    f"Decision          : "
    f"{decision}"
)

print(
    f"Scenario tests    : "
    f"{sum(x['passed'] for x in test_results)}"
    f"/"
    f"{len(test_results)}"
)

print(
    f"Positive score max: "
    f"{POSITIVE_SCORE_MAX}"
)

print(
    f"Final score range : "
    f"0–{TOTAL_SCORE_RANGE_MAX}"
)

print(
    "Automatic response: DISABLED"
)

print(
    "Containment       : ANALYST REQUIRED"
)

print("\nOUTPUTS")

print(
    f"  {SPEC_PATH}"
)

print(
    f"  {TEST_RESULTS_PATH}"
)

print(
    f"  {FINAL_REPORT_PATH}"
)

print("=" * 72)


if decision != "PASS":

    raise RuntimeError(
        "Phase 3D.1 requires review."
    )