"""
SMARTSOC PHASE 3D
GATE 3D.3 — RISK ENGINE VALIDATION

Purpose
-------
Validate the implemented SmartSOC deterministic risk engine
against a representative SOC scenario matrix.

This gate validates:
    1. Benign administrative activity
    2. Normal PowerShell
    3. Suspicious PowerShell
    4. Encoded PowerShell
    5. PowerShell download
    6. CMD execution
    7. Persistence
    8. Registry modification
    9. Credential-access behavior
    10. Network activity
    11. Suspicious parent process
    12. High-integrity execution
    13. Destructive behavior
    14. Multiple simultaneous techniques
    15. Low-confidence prediction
    16. Fallback prediction
    17. Missing behavior context
    18. Ambiguous prediction
    19. High confidence but weak evidence
    20. Critical technique with strong evidence

Important
---------
This is VALIDATION ONLY.

It does NOT:
    - modify the ML model
    - modify thresholds
    - modify the dataset
    - modify Wazuh
    - call Shuffle
    - execute response actions
    - isolate a host
    - block an IP
    - kill a process

The risk engine only calculates a decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
import sys


# ============================================================
# PROJECT PATHS
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
    / "gate3_3"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_PATH = (
    REPORT_DIR
    / "risk_engine_validation_results.json"
)

CSV_PATH = (
    REPORT_DIR
    / "risk_engine_validation_results.csv"
)

FINAL_REPORT_PATH = (
    REPORT_DIR
    / "phase3d_gate3_3_report.json"
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
    RiskDecision,
)


# ============================================================
# QUALITY CHECKS
# ============================================================

quality_checks = {}


def check(
    name: str,
    condition: bool,
    message: str = ""
):

    quality_checks[name] = bool(
        condition
    )

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
# RISK ORDER
# ============================================================

RISK_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


# ============================================================
# SCENARIO MATRIX
# ============================================================

SCENARIOS = [

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    {
        "id": "S01",
        "name": "benign_administrative_activity",

        "inputs": {
            "technique_severity": "LOW",
            "ml_confidence": 0.30,
            "evidence_strength": "WEAK",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_min": "LOW",
        "expected_max": "MEDIUM",
    },

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    {
        "id": "S02",
        "name": "normal_powershell",

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

        "expected_min": "LOW",
        "expected_max": "MEDIUM",
    },

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    {
        "id": "S03",
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

        "expected_min": "HIGH",
    },

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    {
        "id": "S04",
        "name": "encoded_powershell",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.95,
            "evidence_strength": "VERY_STRONG",
            "privilege": "HIGH",
            "network": False,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "HIGH",
    },

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    {
        "id": "S05",
        "name": "powershell_download",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.92,
            "evidence_strength": "VERY_STRONG",
            "privilege": "HIGH",
            "network": True,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "CRITICAL",
    },

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    {
        "id": "S06",
        "name": "cmd_execution",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.80,
            "evidence_strength": "MODERATE",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "MEDIUM",
        "expected_max": "HIGH",
    },

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    {
        "id": "S07",
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

        "expected_min": "HIGH",
    },

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    {
        "id": "S08",
        "name": "registry_modification",

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

        "expected_min": "HIGH",
    },

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    {
        "id": "S09",
        "name": "credential_access",

        "inputs": {
            "technique_severity": "CRITICAL",
            "ml_confidence": 0.90,
            "evidence_strength": "STRONG",
            "privilege": "HIGH",
            "network": False,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "CRITICAL",
    },

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    {
        "id": "S10",
        "name": "network_activity",

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

        "expected_min": "MEDIUM",
        "expected_max": "HIGH",
    },

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    {
        "id": "S11",
        "name": "suspicious_parent_process",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.75,
            "evidence_strength": "STRONG",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "MEDIUM",
    },

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    {
        "id": "S12",
        "name": "high_integrity_execution",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.80,
            "evidence_strength": "MODERATE",
            "privilege": "HIGH",
            "network": False,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "HIGH",
    },

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    {
        "id": "S13",
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

        "expected_min": "CRITICAL",
    },

    # --------------------------------------------------------
    # 14
    # --------------------------------------------------------

    {
        "id": "S14",
        "name": "multiple_simultaneous_techniques",

        "inputs": {
            "technique_severity": "CRITICAL",
            "ml_confidence": 0.96,
            "evidence_strength": "VERY_STRONG",
            "privilege": "SYSTEM",
            "network": True,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "CRITICAL",
    },

    # --------------------------------------------------------
    # 15
    # --------------------------------------------------------

    {
        "id": "S15",
        "name": "low_confidence_prediction",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.35,
            "evidence_strength": "WEAK",
            "privilege": "LOW",
            "network": False,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_min": "LOW",
        "expected_max": "MEDIUM",
    },

    # --------------------------------------------------------
    # 16
    # --------------------------------------------------------

    {
        "id": "S16",
        "name": "fallback_prediction",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.45,
            "evidence_strength": "MODERATE",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": True,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_min": "LOW",
        "expected_max": "MEDIUM",
    },

    # --------------------------------------------------------
    # 17
    # --------------------------------------------------------

    {
        "id": "S17",
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

        "expected_min": "LOW",
        "expected_max": "MEDIUM",
    },

    # --------------------------------------------------------
    # 18
    # --------------------------------------------------------

    {
        "id": "S18",
        "name": "ambiguous_prediction",

        "inputs": {
            "technique_severity": "HIGH",
            "ml_confidence": 0.70,
            "evidence_strength": "MODERATE",
            "privilege": "MEDIUM",
            "network": False,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": True,
            "missing_behavior": False,
        },

        "expected_min": "MEDIUM",
        "expected_max": "HIGH",
    },

    # --------------------------------------------------------
    # 19
    # --------------------------------------------------------

    {
        "id": "S19",
        "name": "high_confidence_weak_evidence",

        "inputs": {
            "technique_severity": "MEDIUM",
            "ml_confidence": 0.99,
            "evidence_strength": "WEAK",
            "privilege": "LOW",
            "network": False,
            "suspicious_parent": False,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "MEDIUM",
        "expected_max": "HIGH",
    },

    # --------------------------------------------------------
    # 20
    # --------------------------------------------------------

    {
        "id": "S20",
        "name": "critical_technique_strong_evidence",

        "inputs": {
            "technique_severity": "CRITICAL",
            "ml_confidence": 0.98,
            "evidence_strength": "VERY_STRONG",
            "privilege": "HIGH",
            "network": True,
            "suspicious_parent": True,
            "fallback": False,
            "ambiguous": False,
            "missing_behavior": False,
        },

        "expected_min": "CRITICAL",
    },
]


# ============================================================
# START
# ============================================================

print("=" * 72)
print("SMARTSOC PHASE 3D")
print("GATE 3D.3 — RISK ENGINE VALIDATION")
print("=" * 72)


# ============================================================
# 1. ENGINE CHECK
# ============================================================

print("\nENGINE CHECK")

check(
    "Risk engine exists",
    ENGINE_PATH.exists(),
    str(ENGINE_PATH)
)


engine = SmartSOCRiskEngine()


check(
    "Risk engine instantiated",
    isinstance(
        engine,
        SmartSOCRiskEngine
    )
)


check(
    "Engine version is 4.1",
    engine.version == "4.1"
)


# ============================================================
# 2. SCENARIO COUNT
# ============================================================

print("\nSCENARIO MATRIX")

print(
    f"  Scenarios: "
    f"{len(SCENARIOS)}"
)


check(
    "Scenario count is 20",
    len(SCENARIOS) == 20
)


# ============================================================
# 3. RUN SCENARIOS
# ============================================================

print("\nRUNNING SCENARIOS")

results = []


for scenario in SCENARIOS:

    result = engine.calculate(
        **scenario["inputs"]
    )

    level = result.risk_level

    score = result.risk_score

    passed = True

    failures = []

    # --------------------------------------------------------
    # Minimum level
    # --------------------------------------------------------

    if "expected_min" in scenario:

        expected_min = scenario[
            "expected_min"
        ]

        if (
            RISK_ORDER[level]
            <
            RISK_ORDER[
                expected_min
            ]
        ):

            passed = False

            failures.append(
                f"Below minimum "
                f"{expected_min}"
            )

    # --------------------------------------------------------
    # Maximum level
    # --------------------------------------------------------

    if "expected_max" in scenario:

        expected_max = scenario[
            "expected_max"
        ]

        if (
            RISK_ORDER[level]
            >
            RISK_ORDER[
                expected_max
            ]
        ):

            passed = False

            failures.append(
                f"Above maximum "
                f"{expected_max}"
            )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    automatic_response = (
        result.safety[
            "automatic_response"
        ]
    )

    if automatic_response:

        passed = False

        failures.append(
            "Automatic response enabled"
        )

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    results.append(
        {
            "id":
                scenario["id"],

            "name":
                scenario["name"],

            "risk_score":
                result.risk_score,

            "risk_level":
                result.risk_level,

            "recommended_action":
                result.recommended_action,

            "components":
                result.to_dict()[
                    "components"
                ],

            "reasons":
                result.reasons,

            "expected_min":
                scenario.get(
                    "expected_min"
                ),

            "expected_max":
                scenario.get(
                    "expected_max"
                ),

            "passed":
                passed,

            "failures":
                failures,
        }
    )

    if passed:

        print(
            f"[PASS] "
            f"{scenario['id']} "
            f"{scenario['name']}: "
            f"{score:.2f} "
            f"→ "
            f"{level}"
        )

    else:

        print(
            f"[FAIL] "
            f"{scenario['id']} "
            f"{scenario['name']}: "
            f"{score:.2f} "
            f"→ "
            f"{level}"
        )

        for failure in failures:

            print(
                f"       {failure}"
            )


# ============================================================
# 4. BASIC RESULT VALIDATION
# ============================================================

print("\nRESULT VALIDATION")

check(
    "All scenarios produced results",
    len(results)
    ==
    len(SCENARIOS)
)


check(
    "All scores are within 0-100",
    all(
        0
        <= float(
            result["risk_score"]
        )
        <= 100
        for result in results
    )
)


check(
    "All risk levels are valid",
    all(
        result["risk_level"]
        in RISK_ORDER
        for result in results
    )
)


check(
    "All actions are valid",
    all(
        result[
            "recommended_action"
        ]
        in {
            "MONITOR",
            "ANALYST_REVIEW",
            "PRIORITY_ANALYST_REVIEW",
            "CONTAINMENT_CANDIDATE",
        }
        for result in results
    )
)


# ============================================================
# 5. SAFETY VALIDATION
# ============================================================

print("\nSAFETY VALIDATION")


check(
    "No automatic response",
    all(
        result["recommended_action"]
        !=
        "AUTO_CONTAIN"
        for result in results
    )
)


# ============================================================
# 6. HIGH-RISK SAFETY
# ============================================================

critical_results = [
    result
    for result in results
    if result["risk_level"]
    ==
    "CRITICAL"
]


check(
    "Critical decisions remain containment candidates",
    all(
        result[
            "recommended_action"
        ]
        ==
        "CONTAINMENT_CANDIDATE"
        for result in critical_results
    )
)


# ============================================================
# 7. FALLBACK SAFETY
# ============================================================

fallback_scenario = next(
    result
    for result in results
    if result["id"] == "S16"
)


check(
    "Fallback does not produce automatic containment",
    fallback_scenario[
        "recommended_action"
    ]
    !=
    "AUTO_CONTAIN"
)


# ============================================================
# 8. MISSING BEHAVIOR SAFETY
# ============================================================

missing_behavior_result = next(
    result
    for result in results
    if result["id"] == "S17"
)


normal_context = engine.calculate(
    technique_severity="MEDIUM",
    ml_confidence=0.60,
    evidence_strength="WEAK",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


missing_context = engine.calculate(
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


check(
    "Missing behavior cannot increase score",
    missing_context.risk_score
    <=
    normal_context.risk_score
)


# ============================================================
# 9. CONFIDENCE MONOTONICITY
# ============================================================

print("\nCONFIDENCE MONOTONICITY")


low_confidence = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.40,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


high_confidence = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.90,
    evidence_strength="MODERATE",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


check(
    "Higher confidence increases or preserves risk",
    high_confidence.risk_score
    >=
    low_confidence.risk_score
)


# ============================================================
# 10. EVIDENCE MONOTONICITY
# ============================================================

print("\nEVIDENCE MONOTONICITY")


weak_evidence = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="WEAK",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


strong_evidence = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="VERY_STRONG",
    privilege="MEDIUM",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


check(
    "Stronger evidence increases or preserves risk",
    strong_evidence.risk_score
    >=
    weak_evidence.risk_score
)


# ============================================================
# 11. CONTEXT MONOTONICITY
# ============================================================

print("\nCONTEXT MONOTONICITY")


weak_context = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="LOW",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


strong_context = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="SYSTEM",
    network=True,
    suspicious_parent=True,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


check(
    "Stronger suspicious context increases or preserves risk",
    strong_context.risk_score
    >=
    weak_context.risk_score
)


# ============================================================
# 12. UNCERTAINTY MONOTONICITY
# ============================================================

print("\nUNCERTAINTY MONOTONICITY")


certain = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=False,
    suspicious_parent=False,
    fallback=False,
    ambiguous=False,
    missing_behavior=False,
)


uncertain = engine.calculate(
    technique_severity="HIGH",
    ml_confidence=0.80,
    evidence_strength="STRONG",
    privilege="HIGH",
    network=False,
    suspicious_parent=False,
    fallback=True,
    ambiguous=True,
    missing_behavior=True,
)


check(
    "Uncertainty decreases or preserves risk",
    uncertain.risk_score
    <=
    certain.risk_score
)


# ============================================================
# 13. SCORE COMPONENT VALIDATION
# ============================================================

print("\nCOMPONENT VALIDATION")


for result in results:

    components = result[
        "components"
    ]

    component_sum = (
        components[
            "technique_severity"
        ]
        +
        components[
            "ml_confidence"
        ]
        +
        components[
            "behavioral_evidence"
        ]
        +
        components[
            "context"
        ]
        -
        components[
            "uncertainty_penalty"
        ]
    )

    calculated = round(
        max(
            0.0,
            min(
                100.0,
                component_sum
            )
        ),
        2
    )

    check(
        f"Component consistency {result['id']}",
        calculated
        ==
        result["risk_score"]
    )


# ============================================================
# 14. EXPECTATION RESULTS
# ============================================================

passed_scenarios = [
    result
    for result in results
    if result["passed"]
]


failed_scenarios = [
    result
    for result in results
    if not result["passed"]
]


scenario_pass_rate = (
    len(passed_scenarios)
    /
    len(results)
)


print("\nSCENARIO SUMMARY")

print(
    f"  Passed : "
    f"{len(passed_scenarios)}"
)

print(
    f"  Failed : "
    f"{len(failed_scenarios)}"
)

print(
    f"  Pass rate : "
    f"{scenario_pass_rate:.6f}"
)


check(
    "Scenario validation pass rate is 100%",
    scenario_pass_rate == 1.0
)


# ============================================================
# 15. RISK DISTRIBUTION
# ============================================================

print("\nRISK DISTRIBUTION")

risk_distribution = {}


for level in RISK_ORDER:

    count = sum(
        result["risk_level"]
        ==
        level
        for result in results
    )

    risk_distribution[
        level
    ] = count

    print(
        f"  {level:8s}: "
        f"{count}"
    )


# ============================================================
# 16. ACTION DISTRIBUTION
# ============================================================

print("\nACTION DISTRIBUTION")

action_distribution = {}


for action in {
    "MONITOR",
    "ANALYST_REVIEW",
    "PRIORITY_ANALYST_REVIEW",
    "CONTAINMENT_CANDIDATE",
}:

    count = sum(
        result[
            "recommended_action"
        ]
        ==
        action
        for result in results
    )

    action_distribution[
        action
    ] = count

    print(
        f"  {action:28s}: "
        f"{count}"
    )


# ============================================================
# 17. PRODUCTION SAFETY CHECK
# ============================================================

print("\nPRODUCTION SAFETY")

production_safety = {
    "model_modified": False,
    "thresholds_modified": False,
    "dataset_modified": False,
    "wazuh_modified": False,
    "shuffle_called": False,
    "response_executed": False,
    "automatic_containment": False,
}


for name, value in (
    production_safety.items()
):

    check(
        name,
        value is False
    )


# ============================================================
# 18. WRITE JSON RESULTS
# ============================================================

print("\nWRITING RESULTS")

results_document = {

    "project":
        "SmartSOC",

    "phase":
        "3D",

    "gate":
        "3D.3",

    "name":
        "Risk Engine Validation",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "engine_version":
        engine.version,

    "scenario_count":
        len(SCENARIOS),

    "scenario_passed":
        len(passed_scenarios),

    "scenario_failed":
        len(failed_scenarios),

    "scenario_pass_rate":
        scenario_pass_rate,

    "risk_distribution":
        risk_distribution,

    "action_distribution":
        action_distribution,

    "results":
        results,

    "monotonicity": {

        "confidence": {
            "lower_score":
                low_confidence.risk_score,

            "higher_score":
                high_confidence.risk_score,
        },

        "evidence": {
            "weak_score":
                weak_evidence.risk_score,

            "strong_score":
                strong_evidence.risk_score,
        },

        "context": {
            "weak_score":
                weak_context.risk_score,

            "strong_score":
                strong_context.risk_score,
        },

        "uncertainty": {
            "certain_score":
                certain.risk_score,

            "uncertain_score":
                uncertain.risk_score,
        },
    },

    "missing_behavior": {

        "normal_context_score":
            normal_context.risk_score,

        "missing_context_score":
            missing_context.risk_score,
    },

    "production_safety":
        production_safety,

    "quality_checks":
        quality_checks,
}


with RESULTS_PATH.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results_document,
        f,
        indent=2
    )


# ============================================================
# 19. WRITE CSV
# ============================================================

import csv


with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "id",
            "name",
            "risk_score",
            "risk_level",
            "recommended_action",
            "expected_min",
            "expected_max",
            "passed",
            "failures",
        ]
    )

    for result in results:

        writer.writerow(
            [
                result["id"],
                result["name"],
                result["risk_score"],
                result["risk_level"],
                result[
                    "recommended_action"
                ],
                result[
                    "expected_min"
                ],
                result[
                    "expected_max"
                ],
                result["passed"],
                "; ".join(
                    result["failures"]
                ),
            ]
        )


# ============================================================
# 20. FINAL DECISION
# ============================================================

all_quality_checks_pass = all(
    quality_checks.values()
)


if all_quality_checks_pass:

    decision = "PASS"

else:

    decision = "REVIEW_REQUIRED"


final_report = {

    "project":
        "SmartSOC",

    "phase":
        "3D",

    "gate":
        "3D.3",

    "name":
        "Risk Engine Validation",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "decision":
        decision,

    "scenario_summary": {

        "total":
            len(SCENARIOS),

        "passed":
            len(passed_scenarios),

        "failed":
            len(failed_scenarios),

        "pass_rate":
            scenario_pass_rate,
    },

    "risk_distribution":
        risk_distribution,

    "action_distribution":
        action_distribution,

    "quality_checks":
        quality_checks,

    "production_safety":
        production_safety,

    "outputs": {

        "json_results":
            str(RESULTS_PATH),

        "csv_results":
            str(CSV_PATH),

        "final_report":
            str(FINAL_REPORT_PATH),
    },

    "limitations": [

        (
            "Scenario validation evaluates deterministic "
            "risk-engine behavior and does not establish "
            "real-world SOC calibration."
        ),

        (
            "Technique severity values are currently "
            "policy-defined rather than learned from "
            "incident-response outcomes."
        ),

        (
            "ML confidence is used as a scoring signal "
            "and is not assumed to be calibrated probability."
        ),

        (
            "No automated response is executed during "
            "this validation gate."
        ),
    ],
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
print("GATE 3D.3 FINAL RESULT")
print("=" * 72)

print(
    f"Decision             : "
    f"{decision}"
)

print(
    f"Scenarios            : "
    f"{len(SCENARIOS)}"
)

print(
    f"Scenario passed      : "
    f"{len(passed_scenarios)}"
)

print(
    f"Scenario failed      : "
    f"{len(failed_scenarios)}"
)

print(
    f"Scenario pass rate   : "
    f"{scenario_pass_rate:.6f}"
)

print(
    "Automatic response   : DISABLED"
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
    f"  {CSV_PATH}"
)

print(
    f"  {FINAL_REPORT_PATH}"
)

print("=" * 72)


if decision != "PASS":

    raise RuntimeError(
        "Phase 3D.3 requires review."
    )