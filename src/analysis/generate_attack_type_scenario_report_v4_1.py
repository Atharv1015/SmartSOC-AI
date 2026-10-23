"""
SmartSOC v4.1
Attack-Type Scenario / AI Decision Report

Purpose
-------
Generate a concise, professional TXT report containing one representative
scenario for every attack type / MITRE technique actually represented by
the frozen production model and frozen test dataset.

IMPORTANT
---------
This script is evidence-oriented.

It does NOT:
- fabricate predictions
- invent confidence scores
- invent attack classes
- invent mitigation actions
- modify production artifacts
- execute response actions
- execute containment
- contact Wazuh
- contact Shuffle

When actual production inference cannot be reproduced safely, the value is
reported as NOT AVAILABLE rather than being guessed.

The report is intended for supervisor / capstone presentation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "supervisor"
)

OUTPUT_TXT = (
    REPORT_DIR
    / "smartsoc_attack_type_scenario_report_v4_1.txt"
)

OUTPUT_JSON = (
    REPORT_DIR
    / "smartsoc_attack_type_scenario_report_v4_1.json"
)


# ---------------------------------------------------------------------------
# ARTIFACT PATHS
# ---------------------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLD_PATH = MODEL_DIR / "thresholds.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

STRUCTURED_PATH = DATA_DIR / "structured_test.csv"
LABELS_PATH = DATA_DIR / "labels_test.csv"
METADATA_PATH = DATA_DIR / "metadata_test.csv"
BEHAVIOR_PATH = DATA_DIR / "behavior_text_test.csv"


# ---------------------------------------------------------------------------
# SAFE IMPORTS
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        if isinstance(value, bool):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
    }:
        return ""

    return text


def display_value(value: Any, fallback: str = "NOT AVAILABLE") -> str:
    text = clean_text(value)
    return text if text else fallback


def normalize_key(value: Any) -> str:
    return clean_text(value).strip().lower()


# ---------------------------------------------------------------------------
# CONSOLE OUTPUT
# ---------------------------------------------------------------------------

def banner() -> None:
    print("=" * 76)
    print("SmartSOC v4.1")
    print("Attack-Type AI Prediction & Mitigation Scenario Report")
    print("=" * 76)
    print()


def info(message: str) -> None:
    print(f"[INFO] {message}")


def passed(message: str) -> None:
    print(f"[PASS] {message}")


def review(message: str) -> None:
    print(f"[REVIEW] {message}")


def failed(message: str) -> None:
    print(f"[FAIL] {message}")


# ---------------------------------------------------------------------------
# CSV LOADING
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
        errors="replace",
    ) as handle:

        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {path}")

        return [
            {
                str(key): clean_text(value)
                for key, value in row.items()
            }
            for row in reader
        ]


def load_frozen_dataset() -> dict[str, list[dict[str, str]]]:
    structured = read_csv_rows(STRUCTURED_PATH)
    labels = read_csv_rows(LABELS_PATH)
    metadata = read_csv_rows(METADATA_PATH)
    behavior = read_csv_rows(BEHAVIOR_PATH)

    lengths = {
        "structured": len(structured),
        "labels": len(labels),
        "metadata": len(metadata),
        "behavior": len(behavior),
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            "Frozen dataset row alignment failure: "
            + repr(lengths)
        )

    passed(
        f"Frozen dataset loaded: {len(structured)} aligned events"
    )

    return {
        "structured": structured,
        "labels": labels,
        "metadata": metadata,
        "behavior": behavior,
    }


# ---------------------------------------------------------------------------
# MODEL INSPECTION
# ---------------------------------------------------------------------------

def load_model() -> Any:
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError(
            "joblib is required to load the production model"
        ) from exc

    if not MODEL_PATH.exists():
        raise FileNotFoundError(str(MODEL_PATH))

    model = joblib.load(MODEL_PATH)

    passed(
        f"Production model loaded: {MODEL_PATH}"
    )

    return model


def get_model_classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)

    if classes is None:
        raise RuntimeError(
            "Production model does not expose classes_. "
            "Actual attack-type coverage cannot be determined safely."
        )

    result = []

    for value in classes:
        text = clean_text(value)

        if text:
            result.append(text)

    if not result:
        raise RuntimeError(
            "Production model exposes no usable classes."
        )

    return result


def inspect_model(model: Any) -> dict[str, Any]:
    classes = get_model_classes(model)

    information = {
        "model_class": type(model).__name__,
        "class_count": len(classes),
        "classes": classes,
    }

    passed(
        f"Frozen production model classes: {len(classes)}"
    )

    return information


# ---------------------------------------------------------------------------
# THRESHOLD INSPECTION
# ---------------------------------------------------------------------------

def load_thresholds() -> dict[str, Any]:
    if not THRESHOLD_PATH.exists():
        review(
            "Threshold file unavailable. "
            "Severity/risk mapping will not be fabricated."
        )
        return {}

    try:
        thresholds = load_json(THRESHOLD_PATH)

        if not isinstance(thresholds, dict):
            review("Threshold artifact is not a JSON object.")
            return {}

        passed("Production thresholds loaded")

        return thresholds

    except Exception as exc:
        review(
            f"Threshold artifact could not be loaded: {exc}"
        )
        return {}


def find_threshold(
    thresholds: dict[str, Any],
    attack_class: str,
) -> float | None:

    target = normalize_key(attack_class)

    # Direct lookup.
    for key, value in thresholds.items():
        if normalize_key(key) == target:
            numeric = safe_float(value)

            if numeric is not None:
                return numeric

    # Common nested layouts.
    for container_key in (
        "thresholds",
        "class_thresholds",
        "decision_thresholds",
        "per_class",
        "classes",
    ):

        container = thresholds.get(container_key)

        if not isinstance(container, dict):
            continue

        for key, value in container.items():

            if normalize_key(key) != target:
                continue

            if isinstance(value, dict):
                for candidate_key in (
                    "threshold",
                    "score",
                    "value",
                ):
                    numeric = safe_float(
                        value.get(candidate_key)
                    )

                    if numeric is not None:
                        return numeric

            numeric = safe_float(value)

            if numeric is not None:
                return numeric

    return None


# ---------------------------------------------------------------------------
# ATTACK LABEL EXTRACTION
# ---------------------------------------------------------------------------

def find_label_column(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None

    candidates = [
        "label",
        "attack",
        "attack_type",
        "attack_type_id",
        "technique",
        "technique_id",
        "mitre_technique",
        "mitre_id",
        "class",
        "target",
        "y",
    ]

    existing = {
        normalize_key(column): column
        for column in rows[0].keys()
    }

    for candidate in candidates:
        if normalize_key(candidate) in existing:
            return existing[normalize_key(candidate)]

    # Fallback: choose a column containing label-like terminology.
    for column in rows[0].keys():
        key = normalize_key(column)

        if any(
            token in key
            for token in (
                "label",
                "attack",
                "technique",
                "class",
                "target",
            )
        ):
            return column

    return None


def extract_frozen_attack_types(
    labels: list[dict[str, str]],
) -> list[str]:

    column = find_label_column(labels)

    if column is None:
        review(
            "Could not safely identify the frozen label column."
        )
        return []

    values = []

    for row in labels:
        value = clean_text(row.get(column))

        if value:
            values.append(value)

    unique = sorted(
        set(values),
        key=lambda x: normalize_key(x),
    )

    passed(
        f"Frozen dataset attack/label classes: {len(unique)}"
    )

    return unique


# ---------------------------------------------------------------------------
# MODEL CLASS COVERAGE
# ---------------------------------------------------------------------------

def build_attack_coverage(
    model_classes: list[str],
    frozen_classes: list[str],
) -> list[dict[str, Any]]:

    frozen_lookup = {
        normalize_key(value): value
        for value in frozen_classes
    }

    coverage = []

    for index, model_class in enumerate(model_classes):

        normalized = normalize_key(model_class)

        coverage.append(
            {
                "model_class_index": index,
                "model_attack_type": model_class,
                "frozen_dataset_match": (
                    frozen_lookup.get(normalized)
                ),
            }
        )

    return coverage


# ---------------------------------------------------------------------------
# REPRESENTATIVE EVENT SELECTION
# ---------------------------------------------------------------------------

def build_label_index(
    labels: list[dict[str, str]],
) -> dict[str, list[int]]:

    column = find_label_column(labels)

    if column is None:
        return {}

    index: dict[str, list[int]] = defaultdict(list)

    for row_number, row in enumerate(labels):

        value = clean_text(row.get(column))

        if not value:
            continue

        index[normalize_key(value)].append(row_number)

    return dict(index)


def representative_event_for_class(
    attack_class: str,
    label_index: dict[str, list[int]],
) -> int | None:

    candidates = label_index.get(
        normalize_key(attack_class),
        [],
    )

    if not candidates:
        return None

    # Deterministic representative.
    return candidates[0]


# ---------------------------------------------------------------------------
# OPTIONAL MITRE METADATA
# ---------------------------------------------------------------------------

def infer_mitre_id_from_text(value: str) -> str | None:
    """
    Extract a technique-like identifier when it is explicitly present
    in the frozen artifact text.

    No external MITRE knowledge is injected here.
    """

    import re

    match = re.search(
        r"\bT\d{4}(?:\.\d{3})?\b",
        value.upper(),
    )

    if match:
        return match.group(0)

    return None


def find_mitre_identifier(
    attack_class: str,
    rows: list[dict[str, str]],
) -> str | None:

    candidates = [
        "technique_id",
        "mitre_id",
        "mitre_technique",
        "technique",
        "attack_type",
        "label",
        "class",
    ]

    for row in rows[:1000]:

        for column in row.keys():

            key = normalize_key(column)

            if not any(
                token in key
                for token in (
                    "technique",
                    "mitre",
                    "attack",
                    "label",
                    "class",
                )
            ):
                continue

            value = clean_text(row.get(column))

            if not value:
                continue

            if normalize_key(value) != normalize_key(attack_class):
                continue

            identifier = infer_mitre_id_from_text(value)

            if identifier:
                return identifier

    identifier = infer_mitre_id_from_text(attack_class)

    return identifier


# ---------------------------------------------------------------------------
# RESPONSE POLICY
# ---------------------------------------------------------------------------

def safe_response_policy(
    risk_level: str,
    fallback: bool = False,
    ambiguous: bool = False,
    missing_behavior: bool = False,
) -> dict[str, str]:

    """
    Mirrors the validated SmartSOC response safety boundary.

    This function describes policy outcome only.

    It does NOT execute anything.
    """

    normalized = normalize_key(risk_level)

    if normalized == "critical":
        response = "Critical response candidate"
        analyst = "REQUIRED"
        containment = "NOT AUTHORIZED AUTOMATICALLY"

    elif normalized == "high":
        response = "High-priority review / rollback candidate"
        analyst = "REQUIRED"
        containment = "NOT AUTHORIZED AUTOMATICALLY"

    elif normalized == "medium":
        response = "Analyst review"
        analyst = "REQUIRED"
        containment = "NOT AUTHORIZED AUTOMATICALLY"

    elif normalized == "low":
        response = "Monitor / record"
        analyst = "NOT REQUIRED FOR BASIC TRIAGE"
        containment = "NOT AUTHORIZED"

    else:
        response = "NOT AVAILABLE"
        analyst = "NOT AVAILABLE"
        containment = "NOT AUTHORIZED"

    if fallback:
        response += " + fallback safety escalation"

    if ambiguous:
        response += " + ambiguity review"

    if missing_behavior:
        response += " + missing behavioral evidence review"

    return {
        "response": response,
        "analyst_authorization": analyst,
        "containment": containment,
        "execution": "DISABLED",
    }


# ---------------------------------------------------------------------------
# RISK MAPPING
# ---------------------------------------------------------------------------

def risk_level_from_score(score: float | None) -> str:
    if score is None:
        return "NOT AVAILABLE"

    # These boundaries correspond to the validated policy structure.
    # The score itself must come from actual model output / artifact.
    if score >= 0.90:
        return "CRITICAL"

    if score >= 0.70:
        return "HIGH"

    if score >= 0.40:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------------
# PRODUCTION INFERENCE ADAPTER
# ---------------------------------------------------------------------------

def try_production_inference(
    model: Any,
    event_row: dict[str, str],
) -> dict[str, Any]:

    """
    Attempt only conservative direct model inference.

    We deliberately do NOT invent a preprocessing pipeline here.

    If the serialized production model does not accept the frozen event
    representation directly, the report records NOT AVAILABLE.

    This protects the report from presenting a prediction generated by an
    incompatible preprocessing path.
    """

    try:
        import pandas as pd
        import numpy as np

        frame = pd.DataFrame([event_row])

        # Direct predict_proba attempt.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(frame)

            if probabilities is None:
                return {
                    "available": False,
                    "reason": "predict_proba returned no result",
                }

            array = np.asarray(probabilities)

            if array.ndim != 2 or array.shape[0] != 1:
                return {
                    "available": False,
                    "reason": (
                        "Unexpected probability matrix shape: "
                        f"{array.shape}"
                    ),
                }

            classes = get_model_classes(model)

            if array.shape[1] != len(classes):
                return {
                    "available": False,
                    "reason": (
                        "Probability/class dimension mismatch"
                    ),
                }

            probabilities_1d = array[0]

            best_index = int(
                np.argmax(probabilities_1d)
            )

            best_probability = float(
                probabilities_1d[best_index]
            )

            predicted_class = str(
                classes[best_index]
            )

            return {
                "available": True,
                "predicted_class": predicted_class,
                "probability": best_probability,
                "class_index": best_index,
            }

        return {
            "available": False,
            "reason": (
                "Serialized production model does not expose "
                "predict_proba"
            ),
        }

    except Exception as exc:

        return {
            "available": False,
            "reason": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


# ---------------------------------------------------------------------------
# SCENARIO GENERATION
# ---------------------------------------------------------------------------

def generate_scenarios(
    model: Any,
    model_classes: list[str],
    thresholds: dict[str, Any],
    dataset: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    labels = dataset["labels"]
    structured = dataset["structured"]
    metadata = dataset["metadata"]
    behavior = dataset["behavior"]

    label_index = build_label_index(labels)

    scenarios = []

    direct_inference_available = True
    inference_failures = []

    for model_index, attack_class in enumerate(model_classes):

        representative_index = representative_event_for_class(
            attack_class,
            label_index,
        )

        scenario: dict[str, Any] = {
            "scenario_number": len(scenarios) + 1,
            "model_class_index": model_index,
            "attack_type": attack_class,
            "mitre_technique_id": None,
            "representative_event_row": None,
            "actual_ai_prediction": None,
            "actual_ai_confidence": None,
            "threshold": find_threshold(
                thresholds,
                attack_class,
            ),
            "severity_score": None,
            "severity_level": "NOT AVAILABLE",
            "mitigation_decision": "NOT AVAILABLE",
            "analyst_authorization": "NOT AVAILABLE",
            "containment": "NOT AUTHORIZED",
            "execution": "DISABLED",
            "evidence_status": "NOT AVAILABLE",
            "notes": [],
        }

        if representative_index is None:

            scenario["notes"].append(
                "No matching representative event found "
                "in the frozen label dataset."
            )

            scenario["evidence_status"] = (
                "MODEL CLASS PRESENT / NO REPRESENTATIVE FROZEN EVENT"
            )

            scenarios.append(scenario)
            continue

        scenario["representative_event_row"] = (
            representative_index + 1
        )

        scenario["mitre_technique_id"] = (
            find_mitre_identifier(
                attack_class,
                metadata,
            )
            or find_mitre_identifier(
                attack_class,
                labels,
            )
        )

        event_row = {}

        event_row.update(structured[representative_index])

        # Behavior and metadata are included only as fields available in
        # the frozen artifacts. No transformation is fabricated here.
        for key, value in metadata[representative_index].items():
            if key not in event_row:
                event_row[key] = value

        for key, value in behavior[representative_index].items():
            if key not in event_row:
                event_row[key] = value

        inference = try_production_inference(
            model,
            event_row,
        )

        if inference.get("available"):

            predicted_class = inference.get(
                "predicted_class"
            )

            probability = safe_float(
                inference.get("probability")
            )

            scenario["actual_ai_prediction"] = (
                predicted_class
            )

            scenario["actual_ai_confidence"] = (
                probability
            )

            # The probability is the actual model output.
            scenario["severity_score"] = probability

            scenario["severity_level"] = (
                risk_level_from_score(probability)
            )

            policy = safe_response_policy(
                scenario["severity_level"]
            )

            scenario["mitigation_decision"] = (
                policy["response"]
            )

            scenario["analyst_authorization"] = (
                policy["analyst_authorization"]
            )

            scenario["containment"] = (
                policy["containment"]
            )

            scenario["execution"] = (
                policy["execution"]
            )

            scenario["evidence_status"] = (
                "ACTUAL PRODUCTION MODEL OUTPUT"
            )

        else:

            direct_inference_available = False

            reason = inference.get(
                "reason",
                "unknown inference failure",
            )

            inference_failures.append(
                {
                    "attack_type": attack_class,
                    "reason": reason,
                }
            )

            scenario["notes"].append(
                "Direct inference through the serialized model "
                "interface was unavailable."
            )

            scenario["notes"].append(
                f"Reason: {reason}"
            )

            scenario["notes"].append(
                "No synthetic prediction was generated."
            )

            scenario["evidence_status"] = (
                "FROZEN CLASS / ACTUAL PREDICTION NOT REPRODUCED"
            )

        scenarios.append(scenario)

    status = {
        "direct_inference_available": direct_inference_available,
        "inference_failures": inference_failures,
        "scenarios_generated": len(scenarios),
    }

    return scenarios, status


# ---------------------------------------------------------------------------
# PROFESSIONAL TXT REPORT
# ---------------------------------------------------------------------------

def format_probability(value: Any) -> str:
    numeric = safe_float(value)

    if numeric is None:
        return "NOT AVAILABLE"

    return f"{numeric:.6f} ({numeric * 100:.2f}%)"


def format_score(value: Any) -> str:
    numeric = safe_float(value)

    if numeric is None:
        return "NOT AVAILABLE"

    return f"{numeric:.6f}"


def scenario_to_text(
    scenario: dict[str, Any],
) -> str:

    lines = []

    lines.append("-" * 76)
    lines.append(
        f"SCENARIO {scenario['scenario_number']:02d}"
    )
    lines.append("-" * 76)

    lines.append(
        f"Attack Type / Model Class : "
        f"{display_value(scenario['attack_type'])}"
    )

    lines.append(
        f"MITRE Technique ID       : "
        f"{display_value(scenario['mitre_technique_id'])}"
    )

    lines.append(
        f"Model Class Index        : "
        f"{display_value(scenario['model_class_index'])}"
    )

    lines.append(
        f"Representative Event     : "
        f"{display_value(scenario['representative_event_row'])}"
    )

    lines.append("")

    lines.append("AI RESULT")

    lines.append(
        f"Predicted Attack Type     : "
        f"{display_value(scenario['actual_ai_prediction'])}"
    )

    lines.append(
        f"AI Confidence             : "
        f"{format_probability(scenario['actual_ai_confidence'])}"
    )

    lines.append(
        f"Severity / Risk Score     : "
        f"{format_score(scenario['severity_score'])}"
    )

    lines.append(
        f"Risk Level                : "
        f"{display_value(scenario['severity_level'])}"
    )

    lines.append(
        f"Production Threshold      : "
        f"{format_score(scenario['threshold'])}"
    )

    lines.append("")

    lines.append("DECISION / MITIGATION")

    lines.append(
        f"AI Decision               : "
        f"{display_value(scenario['mitigation_decision'])}"
    )

    lines.append(
        f"Analyst Authorization     : "
        f"{display_value(scenario['analyst_authorization'])}"
    )

    lines.append(
        f"Containment               : "
        f"{display_value(scenario['containment'])}"
    )

    lines.append(
        f"Response Execution        : "
        f"{display_value(scenario['execution'])}"
    )

    lines.append("")

    lines.append(
        f"Evidence Status           : "
        f"{display_value(scenario['evidence_status'])}"
    )

    notes = scenario.get("notes") or []

    if notes:

        lines.append("")

        lines.append("Evidence Notes")

        for note in notes:
            lines.append(f"  - {note}")

    return "\n".join(lines)


def write_txt_report(
    model_info: dict[str, Any],
    dataset_size: int,
    scenarios: list[dict[str, Any]],
    inference_status: dict[str, Any],
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    actual_count = sum(
        1
        for scenario in scenarios
        if scenario["evidence_status"]
        == "ACTUAL PRODUCTION MODEL OUTPUT"
    )

    not_reproduced = len(scenarios) - actual_count

    lines = []

    lines.append("=" * 76)
    lines.append("SmartSOC AI")
    lines.append("FROZEN MODEL ATTACK-TYPE SCENARIO REPORT")
    lines.append("Version 4.1")
    lines.append("=" * 76)

    lines.append("")

    lines.append(
        "Purpose: Supervisor / Capstone Evidence"
    )

    lines.append(
        "Evidence policy: Actual frozen artifacts only"
    )

    lines.append(
        "Synthetic predictions: DISABLED"
    )

    lines.append(
        "Response execution: DISABLED"
    )

    lines.append(
        "Containment execution: DISABLED"
    )

    lines.append(
        "Wazuh / Shuffle execution: DISABLED"
    )

    lines.append("")

    lines.append("=" * 76)
    lines.append("FROZEN PRODUCTION MODEL")
    lines.append("=" * 76)

    lines.append(
        f"Model class               : "
        f"{model_info['model_class']}"
    )

    lines.append(
        f"Model attack classes      : "
        f"{model_info['class_count']}"
    )

    lines.append(
        f"Frozen test events        : "
        f"{dataset_size}"
    )

    lines.append(
        f"Scenario entries          : "
        f"{len(scenarios)}"
    )

    lines.append(
        f"Actual production outputs : "
        f"{actual_count}"
    )

    lines.append(
        f"Not reproduced            : "
        f"{not_reproduced}"
    )

    lines.append("")

    lines.append("=" * 76)
    lines.append("IMPORTANT INTERPRETATION")
    lines.append("=" * 76)

    lines.append(
        "Each scenario represents one attack class exposed by the "
        "serialized production model."
    )

    lines.append(
        "Where a compatible production inference result was obtained, "
        "the prediction and confidence shown are actual model output."
    )

    lines.append(
        "Where production inference could not be reproduced through the "
        "available interface, the report explicitly marks the result "
        "as NOT AVAILABLE."
    )

    lines.append(
        "No fabricated confidence, severity, prediction, or mitigation "
        "result is inserted."
    )

    lines.append("")

    for scenario in scenarios:
        lines.append(
            scenario_to_text(scenario)
        )
        lines.append("")

    lines.append("=" * 76)
    lines.append("SYSTEM SAFETY BOUNDARY")
    lines.append("=" * 76)

    lines.append(
        "AI detection/classification : VALIDATED"
    )

    lines.append(
        "Risk/decision policy        : VALIDATED"
    )

    lines.append(
        "Response planning           : VALIDATED"
    )

    lines.append(
        "Analyst authorization       : VALIDATED"
    )

    lines.append(
        "Rollback/recovery           : VALIDATED"
    )

    lines.append(
        "Adversarial response safety : VALIDATED"
    )

    lines.append(
        "Offline end-to-end pipeline : VALIDATED"
    )

    lines.append(
        "Automatic response execution: DISABLED"
    )

    lines.append(
        "Automatic containment       : DISABLED"
    )

    lines.append(
        "Wazuh / Shuffle integration: DISABLED"
    )

    lines.append("")

    lines.append("=" * 76)
    lines.append("END OF REPORT")
    lines.append("=" * 76)

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# JSON REPORT
# ---------------------------------------------------------------------------

def write_json_report(
    model_info: dict[str, Any],
    dataset_size: int,
    scenarios: list[dict[str, Any]],
    inference_status: dict[str, Any],
) -> None:

    actual_count = sum(
        1
        for scenario in scenarios
        if scenario["evidence_status"]
        == "ACTUAL PRODUCTION MODEL OUTPUT"
    )

    report = {
        "report": {
            "name": (
                "SmartSOC Frozen Model Attack-Type "
                "Scenario Report"
            ),
            "version": "4.1",
            "evidence_policy": (
                "actual_frozen_artifacts_only"
            ),
            "synthetic_predictions": False,
            "response_execution": False,
            "containment_execution": False,
        },
        "production_model": model_info,
        "frozen_dataset_events": dataset_size,
        "scenario_count": len(scenarios),
        "actual_production_outputs": actual_count,
        "inference_status": inference_status,
        "scenarios": scenarios,
        "operational_safety": {
            "automatic_response": False,
            "automatic_containment": False,
            "wazuh_enabled": False,
            "shuffle_enabled": False,
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:

    banner()

    print("[1] Loading frozen production artifacts...")

    try:
        model = load_model()

        # Schema is verified for existence and readability.
        if SCHEMA_PATH.exists():
            load_json(SCHEMA_PATH)
            passed("Production feature schema loaded")
        else:
            review(
                "Production feature schema not found."
            )

        thresholds = load_thresholds()

        model_info = inspect_model(model)

    except Exception as exc:

        failed(
            f"Production artifact loading failed: {exc}"
        )

        return 1

    print()
    print("[2] Loading frozen test dataset...")

    try:
        dataset = load_frozen_dataset()

    except Exception as exc:

        failed(
            f"Frozen dataset loading failed: {exc}"
        )

        return 1

    frozen_classes = extract_frozen_attack_types(
        dataset["labels"]
    )

    coverage = build_attack_coverage(
        model_info["classes"],
        frozen_classes,
    )

    print()
    print("[3] Building attack-type coverage...")

    model_only = sum(
        1
        for item in coverage
        if item["frozen_dataset_match"] is None
    )

    matched = len(coverage) - model_only

    passed(
        f"Model classes represented in frozen labels: "
        f"{matched}/{len(coverage)}"
    )

    if model_only:
        review(
            f"{model_only} model classes have no exact frozen-label "
            "match; they will not receive fabricated event output."
        )

    print()
    print("[4] Generating one representative scenario per model class...")

    scenarios, inference_status = generate_scenarios(
        model=model,
        model_classes=model_info["classes"],
        thresholds=thresholds,
        dataset=dataset,
    )

    actual_count = sum(
        1
        for scenario in scenarios
        if scenario["evidence_status"]
        == "ACTUAL PRODUCTION MODEL OUTPUT"
    )

    if actual_count:
        passed(
            f"Actual production inference outputs: "
            f"{actual_count}/{len(scenarios)}"
        )
    else:
        review(
            "No compatible direct production inference result "
            "was reproduced. No synthetic output was generated."
        )

    print()
    print("[5] Writing professional supervisor report...")

    write_txt_report(
        model_info=model_info,
        dataset_size=len(dataset["labels"]),
        scenarios=scenarios,
        inference_status=inference_status,
    )

    write_json_report(
        model_info=model_info,
        dataset_size=len(dataset["labels"]),
        scenarios=scenarios,
        inference_status=inference_status,
    )

    passed(
        f"TXT report: {OUTPUT_TXT}"
    )

    passed(
        f"JSON evidence: {OUTPUT_JSON}"
    )

    print()
    print("=" * 76)
    print("ATTACK-TYPE SCENARIO REPORT COMPLETE")
    print("=" * 76)

    print(
        f"Frozen events       : {len(dataset['labels'])}"
    )

    print(
        f"Model attack classes: {len(model_info['classes'])}"
    )

    print(
        f"Scenario entries    : {len(scenarios)}"
    )

    print(
        f"Actual AI outputs   : {actual_count}"
    )

    print(
        f"Synthetic output    : DISABLED"
    )

    print(
        f"Response execution  : DISABLED"
    )

    print(
        f"Containment         : DISABLED"
    )

    print()
    print(
        "The TXT report contains one professional scenario per "
        "frozen model attack class."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())