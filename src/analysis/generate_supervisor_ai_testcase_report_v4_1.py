"""
SmartSOC v4.1
Supervisor AI Testcase Report

Purpose
-------
Generate a concise, supervisor-friendly testcase report showing how the
ACTUAL frozen SmartSOC production artifacts behave on representative events.

Evidence sources
----------------
1. Frozen production model:
   models/windows_apt_v4_1/production/model.joblib

2. Frozen thresholds:
   models/windows_apt_v4_1/production/thresholds.json

3. Frozen feature schema:
   models/windows_apt_v4_1/production/feature_schema.json

4. Frozen test dataset:
   data/final/windows_apt_v4_1/

5. Existing validated Phase 5 / Phase 6 components, when their interfaces
   can be reproduced safely.

Important safety/evidence rule
------------------------------
This script NEVER fabricates:
- predictions
- attack types
- confidence
- risk scores
- mitigation actions
- authorization state

If an actual value cannot be reproduced, it is reported as NOT_AVAILABLE.

This script is REPORTING ONLY.
It does NOT execute response actions.
It does NOT execute containment.
It does NOT modify production artifacts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

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

MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLDS_PATH = MODEL_DIR / "thresholds.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"

STRUCTURED_PATH = DATA_DIR / "structured_test.csv"
LABELS_PATH = DATA_DIR / "labels_test.csv"
METADATA_PATH = DATA_DIR / "metadata_test.csv"
BEHAVIOR_PATH = DATA_DIR / "behavior_text_test.csv"

JSON_REPORT = REPORT_DIR / "smartsoc_supervisor_testcases_v4_1.json"
CSV_REPORT = REPORT_DIR / "smartsoc_supervisor_testcases_v4_1.csv"
TXT_REPORT = REPORT_DIR / "smartsoc_supervisor_testcases_v4_1.txt"


# ---------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------

WIDTH = 72


def banner(title: str) -> None:
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def passed(label: str, detail: Optional[str] = None) -> None:
    print(f"[PASS] {label}")
    if detail:
        print(f"       {detail}")


def reviewed(label: str, detail: Optional[str] = None) -> None:
    print(f"[REVIEW] {label}")
    if detail:
        print(f"         {detail}")


def failed(label: str, detail: Optional[str] = None) -> None:
    print(f"[FAIL] {label}")
    if detail:
        print(f"       {detail}")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

NOT_AVAILABLE = "NOT_AVAILABLE"


def clean_value(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe values."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]

    return clean_value(value)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def lower_text(value: Any) -> str:
    return normalize_text(value).lower()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    exact = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]

    for column in df.columns:
        column_lower = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


# ---------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts() -> Dict[str, Any]:
    banner("SmartSOC Supervisor AI Testcase Report")

    print()
    print("[1] Loading frozen production artifacts...")

    required = [
        MODEL_PATH,
        THRESHOLDS_PATH,
        SCHEMA_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing production artifact(s):\n"
            + "\n".join(missing)
        )

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "joblib is required by the production artifact."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)

    thresholds = load_json(THRESHOLDS_PATH)
    schema = load_json(SCHEMA_PATH)

    passed("Production model loaded", str(MODEL_PATH))
    passed("Production thresholds loaded", str(THRESHOLDS_PATH))
    passed("Production feature schema loaded", str(SCHEMA_PATH))

    return {
        "model": model,
        "thresholds": thresholds,
        "schema": schema,
    }


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset() -> Dict[str, pd.DataFrame]:
    print()
    print("[2] Loading frozen test dataset...")

    required = [
        STRUCTURED_PATH,
        LABELS_PATH,
        METADATA_PATH,
        BEHAVIOR_PATH,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing frozen dataset artifact(s):\n"
            + "\n".join(missing)
        )

    structured = pd.read_csv(STRUCTURED_PATH)
    labels = pd.read_csv(LABELS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    behavior = pd.read_csv(BEHAVIOR_PATH)

    row_counts = {
        "structured": len(structured),
        "labels": len(labels),
        "metadata": len(metadata),
        "behavior": len(behavior),
    }

    unique_counts = set(row_counts.values())

    if len(unique_counts) != 1:
        raise ValueError(
            f"Frozen dataset row mismatch: {row_counts}"
        )

    rows = len(structured)

    passed(
        "Frozen test dataset loaded",
        f"{rows} events",
    )

    passed(
        "Dataset row alignment",
        f"rows={rows}",
    )

    return {
        "structured": structured,
        "labels": labels,
        "metadata": metadata,
        "behavior": behavior,
    }


# ---------------------------------------------------------------------
# Schema resolution
# ---------------------------------------------------------------------

def resolve_schema(
    schema: Dict[str, Any],
    structured: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Resolve schema defensively.

    The project has previously used several schema representations.
    This function supports the known forms without assuming one exact
    structure.
    """

    candidates = {
        "structured": [
            "structured_columns",
            "structured_features",
            "features",
            "columns",
        ],
        "categorical": [
            "categorical_columns",
            "categorical_features",
            "categorical",
        ],
        "numerical": [
            "numerical_columns",
            "numerical_features",
            "numerical",
        ],
    }

    result: Dict[str, List[str]] = {
        "structured": [],
        "categorical": [],
        "numerical": [],
    }

    # -------------------------------------------------------------
    # Direct schema keys
    # -------------------------------------------------------------

    for target, keys in candidates.items():
        for key in keys:
            value = schema.get(key)

            if isinstance(value, list):
                result[target] = [
                    str(x) for x in value if str(x) in structured.columns
                ]
                if result[target]:
                    break

            elif isinstance(value, dict):
                columns = value.get("columns")

                if isinstance(columns, list):
                    result[target] = [
                        str(x)
                        for x in columns
                        if str(x) in structured.columns
                    ]

                    if result[target]:
                        break

    # -------------------------------------------------------------
    # Nested feature schema
    # -------------------------------------------------------------

    if not result["categorical"] or not result["numerical"]:
        for nested_key in (
            "feature_schema",
            "structured",
            "schema",
            "features",
        ):
            nested = schema.get(nested_key)

            if not isinstance(nested, dict):
                continue

            if not result["categorical"]:
                value = nested.get("categorical_columns")
                if isinstance(value, list):
                    result["categorical"] = [
                        str(x)
                        for x in value
                        if str(x) in structured.columns
                    ]

            if not result["numerical"]:
                value = nested.get("numerical_columns")
                if isinstance(value, list):
                    result["numerical"] = [
                        str(x)
                        for x in value
                        if str(x) in structured.columns
                    ]

    # -------------------------------------------------------------
    # If categorical/numerical are still unavailable, infer from
    # actual dataframe dtypes ONLY.
    #
    # This is not inventing model features. It is simply determining
    # column types from the frozen dataset.
    # -------------------------------------------------------------

    if not result["categorical"]:
        result["categorical"] = [
            str(c)
            for c in structured.columns
            if not pd.api.types.is_numeric_dtype(structured[c])
        ]

    if not result["numerical"]:
        result["numerical"] = [
            str(c)
            for c in structured.columns
            if pd.api.types.is_numeric_dtype(structured[c])
        ]

    if not result["structured"]:
        result["structured"] = [
            str(c) for c in structured.columns
        ]

    return result


# ---------------------------------------------------------------------
# Production model introspection
# ---------------------------------------------------------------------

def inspect_model(
    model: Any,
    resolved_schema: Dict[str, List[str]],
) -> Dict[str, Any]:

    print()
    print("[3] Inspecting actual production model...")

    model_class = type(model).__name__

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "estimator"):
        classes = getattr(model.estimator, "classes_", None)

    class_count = len(classes) if classes is not None else None

    passed(
        "Production model class",
        model_class,
    )

    if class_count is not None:
        passed(
            "Production model classes",
            str(class_count),
        )
    else:
        reviewed(
            "Production model classes",
            NOT_AVAILABLE,
        )

    passed(
        "Frozen schema dimensions",
        (
            f"{len(resolved_schema['structured'])} structured, "
            f"{len(resolved_schema['categorical'])} categorical, "
            f"{len(resolved_schema['numerical'])} numerical"
        ),
    )

    return {
        "model_class": model_class,
        "class_count": class_count,
        "classes": (
            [clean_value(x) for x in classes]
            if classes is not None
            else None
        ),
    }


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def locate_transformers(model: Any) -> Dict[str, Any]:
    """
    Best-effort inspection of common serialized model structures.

    We do NOT replace the production preprocessing logic with a guessed
    implementation. This function only looks for transformers already
    embedded in the serialized object.
    """

    found: Dict[str, Any] = {}

    attributes = [
        "named_steps",
        "transformer_list",
        "steps",
        "estimators_",
    ]

    for attr in attributes:
        value = getattr(model, attr, None)

        if value is None:
            continue

        if isinstance(value, dict):
            for key, item in value.items():
                found[str(key)] = item

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, tuple) and len(item) >= 2:
                    found[str(item[0])] = item[1]

    return found


def model_accepts_dataframe(model: Any) -> bool:
    """
    Determine whether a direct DataFrame call is plausibly supported.

    This does not guarantee success; actual inference is still wrapped
    in a strict evidence boundary.
    """

    return callable(getattr(model, "predict_proba", None)) or callable(
        getattr(model, "predict", None)
    )


def direct_model_inference(
    model: Any,
    structured: pd.DataFrame,
    behavior: pd.DataFrame,
    metadata: pd.DataFrame,
    row_index: int,
) -> Dict[str, Any]:
    """
    Attempt to reproduce the production model directly.

    No synthetic preprocessing is performed here.

    The function first tries the complete frozen structured row.
    Then, only if the serialized model itself is a sklearn Pipeline/
    transformer structure, it tries the model's own transform path.

    If the interface cannot be reproduced, NOT_AVAILABLE is returned.
    """

    if not model_accepts_dataframe(model):
        return {
            "available": False,
            "reason": "model has no supported predict/predict_proba interface",
        }

    row = structured.iloc[[row_index]].copy()

    attempts: List[Tuple[str, Any]] = []

    # Attempt 1: exact structured row.
    attempts.append(("structured_dataframe", row))

    # Attempt 2: combine structured + behavior text if a direct model
    # explicitly exposes a normal sklearn pipeline.
    #
    # This is intentionally conservative. We do not guess feature
    # concatenation rules.
    if hasattr(model, "named_steps"):
        combined = row.copy()

        for column in behavior.columns:
            if column not in combined.columns:
                combined[column] = behavior.iloc[row_index][column]

        for column in metadata.columns:
            if column not in combined.columns:
                combined[column] = metadata.iloc[row_index][column]

        attempts.append(("combined_dataframe", combined))

    errors: List[str] = []

    for attempt_name, candidate in attempts:
        try:
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(candidate)

                prediction = None

                if hasattr(model, "predict"):
                    prediction = model.predict(candidate)

                return {
                    "available": True,
                    "interface": attempt_name,
                    "probability": probability,
                    "prediction": prediction,
                }

        except Exception as exc:
            errors.append(
                f"{attempt_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "available": False,
        "reason": "; ".join(errors),
    }


# ---------------------------------------------------------------------
# Label / class extraction
# ---------------------------------------------------------------------

def extract_prediction(
    inference: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:

    if not inference.get("available"):
        return {
            "prediction_available": False,
            "predicted_class": NOT_AVAILABLE,
            "prediction_probability": None,
            "confidence": None,
        }

    probabilities = inference.get("probability")

    predicted_class = NOT_AVAILABLE
    probability = None

    if probabilities is not None:
        arr = np.asarray(probabilities)

        if arr.ndim == 2 and arr.shape[0] >= 1:
            row_prob = arr[0]

            if len(row_prob) > 0:
                max_index = int(np.argmax(row_prob))
                probability = float(row_prob[max_index])

                classes = getattr(model, "classes_", None)

                if classes is not None and max_index < len(classes):
                    predicted_class = clean_value(classes[max_index])

    if predicted_class == NOT_AVAILABLE:
        prediction = inference.get("prediction")

        if prediction is not None:
            arr = np.asarray(prediction).reshape(-1)

            if len(arr) > 0:
                predicted_class = clean_value(arr[0])

    return {
        "prediction_available": True,
        "predicted_class": predicted_class,
        "prediction_probability": probability,
        "confidence": probability,
    }


# ---------------------------------------------------------------------
# Actual dataset information
# ---------------------------------------------------------------------

def extract_event_context(
    datasets: Dict[str, pd.DataFrame],
    row_index: int,
) -> Dict[str, Any]:

    structured = datasets["structured"]
    labels = datasets["labels"]
    metadata = datasets["metadata"]
    behavior = datasets["behavior"]

    context: Dict[str, Any] = {
        "dataset_row": row_index,
    }

    # Actual label
    if row_index < len(labels):
        for candidate in [
            "label",
            "attack_type",
            "attack",
            "class",
            "target",
            "technique",
        ]:
            column = find_column(labels, [candidate])

            if column:
                context["actual_label"] = clean_value(
                    labels.iloc[row_index][column]
                )
                context["actual_label_column"] = str(column)
                break

    # Metadata
    metadata_candidates = [
        "event_id",
        "id",
        "timestamp",
        "host",
        "hostname",
        "source",
        "agent",
        "rule",
        "description",
    ]

    metadata_values: Dict[str, Any] = {}

    if row_index < len(metadata):
        for candidate in metadata_candidates:
            column = find_column(metadata, [candidate])

            if column:
                metadata_values[str(column)] = clean_value(
                    metadata.iloc[row_index][column]
                )

    context["metadata"] = metadata_values

    # Behavior text
    behavior_text = ""

    if row_index < len(behavior):
        text_column = find_column(
            behavior,
            [
                "behavior_text",
                "behavior",
                "text",
                "description",
                "command",
                "message",
            ],
        )

        if text_column:
            behavior_text = normalize_text(
                behavior.iloc[row_index][text_column]
            )

        elif len(behavior.columns) == 1:
            behavior_text = normalize_text(
                behavior.iloc[row_index, 0]
            )

    context["behavior_text"] = behavior_text

    # Selected structured evidence
    structured_values: Dict[str, Any] = {}

    if row_index < len(structured):
        for column in structured.columns:
            value = clean_value(structured.iloc[row_index][column])

            if value is not None:
                structured_values[str(column)] = value

    context["structured_values"] = structured_values

    return context


# ---------------------------------------------------------------------
# Threshold inspection
# ---------------------------------------------------------------------

def inspect_thresholds(
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:

    result: Dict[str, Any] = {
        "threshold_count": 0,
        "threshold_keys": [],
    }

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                walk(child, path)

        elif isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                result["threshold_count"] += 1
                result["threshold_keys"].append(prefix)

        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")

    walk(thresholds)

    return result


# ---------------------------------------------------------------------
# Existing decision/response components
# ---------------------------------------------------------------------

def load_component(
    relative_path: str,
    class_candidates: List[str],
) -> Tuple[Optional[Any], Optional[str]]:

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return None, f"missing: {path}"

    try:
        import importlib.util

        module_name = (
            "smartsoc_component_"
            + path.stem.replace("-", "_")
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )

        if spec is None or spec.loader is None:
            return None, "module spec unavailable"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for class_name in class_candidates:
            cls = getattr(module, class_name, None)

            if cls is not None:
                try:
                    return cls(), None
                except Exception as exc:
                    return None, (
                        f"class initialization failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

        return None, "candidate class not found"

    except Exception as exc:
        return None, (
            f"component load failed: "
            f"{type(exc).__name__}: {exc}"
        )


def discover_component(
    relative_path: str,
    preferred_names: List[str],
) -> Tuple[Optional[Any], Optional[str]]:

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return None, f"missing: {path}"

    try:
        import importlib.util

        module_name = (
            "smartsoc_dynamic_"
            + path.stem.replace("-", "_")
        )

        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )

        if spec is None or spec.loader is None:
            return None, "module spec unavailable"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name in preferred_names:
            obj = getattr(module, name, None)

            if obj is None:
                continue

            if isinstance(obj, type):
                try:
                    return obj(), None
                except Exception as exc:
                    return None, (
                        f"initialization failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            return obj, None

        return None, "no preferred component found"

    except Exception as exc:
        return None, (
            f"load failed: "
            f"{type(exc).__name__}: {exc}"
        )


def attempt_policy_evaluation(
    component: Any,
    payload: Dict[str, Any],
) -> Dict[str, Any]:

    if component is None:
        return {
            "available": False,
            "reason": "component unavailable",
        }

    evaluate = getattr(component, "evaluate", None)

    if not callable(evaluate):
        return {
            "available": False,
            "reason": "evaluate interface unavailable",
        }

    attempts = [
        payload,
    ]

    errors: List[str] = []

    for candidate in attempts:
        try:
            result = evaluate(**candidate)

            if hasattr(result, "__dict__"):
                return {
                    "available": True,
                    "result": safe_json(vars(result)),
                }

            return {
                "available": True,
                "result": safe_json(result),
            }

        except Exception as exc:
            errors.append(
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "available": False,
        "reason": "; ".join(errors),
    }


# ---------------------------------------------------------------------
# Risk information
# ---------------------------------------------------------------------

def infer_risk_from_thresholds(
    prediction_probability: Optional[float],
    predicted_class: Any,
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Conservative threshold reporting.

    We only calculate a risk level if the thresholds structure contains
    recognizable numeric risk boundaries.

    Otherwise risk fields remain NOT_AVAILABLE.
    """

    result = {
        "risk_score": None,
        "risk_level": NOT_AVAILABLE,
        "risk_source": NOT_AVAILABLE,
    }

    if prediction_probability is None:
        return result

    numeric_values: List[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

        elif isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                numeric_values.append(float(value))

    walk(thresholds)

    # IMPORTANT:
    # Do not reinterpret arbitrary model thresholds as risk thresholds.
    # We therefore only expose the model probability here as a score
    # and leave risk-level classification to the actual risk/decision
    # component when it can be invoked.
    result["risk_score"] = prediction_probability
    result["risk_source"] = "production_model_probability"

    return result


# ---------------------------------------------------------------------
# Testcase selection
# ---------------------------------------------------------------------

def select_representative_rows(
    datasets: Dict[str, pd.DataFrame],
    predictions: List[Dict[str, Any]],
    max_cases: int = 12,
) -> List[int]:
    """
    Select representative ACTUAL rows.

    Priority:
    1. Different predicted classes
    2. Different actual labels
    3. Different confidence bands
    4. Deterministic ordering

    If actual inference is unavailable, we do not invent cases based on
    imagined predictions. We instead select deterministic dataset rows
    and clearly mark unavailable model outputs.
    """

    labels = datasets["labels"]

    available = [
        p for p in predictions
        if p.get("prediction", {}).get("prediction_available")
    ]

    selected: List[int] = []

    # -------------------------------------------------------------
    # First: unique predicted classes
    # -------------------------------------------------------------

    seen_classes = set()

    for item in available:
        pred = item["prediction"]["predicted_class"]

        key = str(pred)

        if key in seen_classes:
            continue

        selected.append(item["row_index"])
        seen_classes.add(key)

        if len(selected) >= max_cases:
            return selected

    # -------------------------------------------------------------
    # Second: unique actual labels
    # -------------------------------------------------------------

    label_column = find_column(
        labels,
        [
            "label",
            "attack_type",
            "attack",
            "class",
            "target",
            "technique",
        ],
    )

    if label_column:
        seen_labels = set()

        for row_index in range(len(labels)):
            if row_index in selected:
                continue

            value = clean_value(
                labels.iloc[row_index][label_column]
            )

            key = str(value)

            if key in seen_labels:
                continue

            selected.append(row_index)
            seen_labels.add(key)

            if len(selected) >= max_cases:
                return selected

    # -------------------------------------------------------------
    # Third: deterministic fill
    # -------------------------------------------------------------

    for row_index in range(len(datasets["structured"])):
        if row_index not in selected:
            selected.append(row_index)

        if len(selected) >= max_cases:
            break

    return selected


# ---------------------------------------------------------------------
# Testcase classification
# ---------------------------------------------------------------------

def classify_case_position(
    prediction_probability: Optional[float],
) -> str:

    if prediction_probability is None:
        return NOT_AVAILABLE

    if prediction_probability >= 0.90:
        return "VERY_HIGH_MODEL_CONFIDENCE"

    if prediction_probability >= 0.75:
        return "HIGH_MODEL_CONFIDENCE"

    if prediction_probability >= 0.50:
        return "MODERATE_MODEL_CONFIDENCE"

    if prediction_probability >= 0.25:
        return "LOW_MODEL_CONFIDENCE"

    return "VERY_LOW_MODEL_CONFIDENCE"


# ---------------------------------------------------------------------
# Generate one testcase
# ---------------------------------------------------------------------

def build_testcase(
    testcase_number: int,
    row_index: int,
    datasets: Dict[str, pd.DataFrame],
    prediction_record: Dict[str, Any],
    thresholds: Dict[str, Any],
    decision_engine: Optional[Any],
    response_engine: Optional[Any],
    authorization_engine: Optional[Any],
) -> Dict[str, Any]:

    context = extract_event_context(
        datasets,
        row_index,
    )

    prediction = prediction_record.get(
        "prediction",
        {},
    )

    predicted_class = prediction.get(
        "predicted_class",
        NOT_AVAILABLE,
    )

    probability = prediction.get(
        "prediction_probability",
    )

    confidence = prediction.get(
        "confidence",
    )

    risk = infer_risk_from_thresholds(
        probability,
        predicted_class,
        thresholds,
    )

    # -------------------------------------------------------------
    # Decision policy
    # -------------------------------------------------------------

    decision_payload = {
        "risk_score": risk.get("risk_score"),
        "risk_level": risk.get("risk_level"),
        "confidence": confidence,
        "fallback": False,
        "ambiguous": False,
        "missing_behavior": not bool(
            context.get("behavior_text")
        ),
        "predicted_class": predicted_class,
    }

    decision_result = attempt_policy_evaluation(
        decision_engine,
        decision_payload,
    )

    # -------------------------------------------------------------
    # Response planner
    # -------------------------------------------------------------

    response_payload = dict(decision_payload)

    response_result = attempt_policy_evaluation(
        response_engine,
        response_payload,
    )

    # -------------------------------------------------------------
    # Authorization
    # -------------------------------------------------------------

    authorization_payload = dict(decision_payload)

    authorization_result = attempt_policy_evaluation(
        authorization_engine,
        authorization_payload,
    )

    # -------------------------------------------------------------
    # Actual mitigation information
    # -------------------------------------------------------------

    mitigation = NOT_AVAILABLE

    if response_result.get("available"):
        result = response_result.get("result")

        if isinstance(result, dict):
            for key in [
                "recommended_actions",
                "response_actions",
                "actions",
                "mitigation_actions",
                "planned_actions",
                "response_plan",
            ]:
                if key in result:
                    mitigation = result[key]
                    break

    # -------------------------------------------------------------
    # Authorization information
    # -------------------------------------------------------------

    analyst_required: Any = NOT_AVAILABLE
    containment_authorized: Any = NOT_AVAILABLE
    execution_enabled: Any = NOT_AVAILABLE

    if authorization_result.get("available"):
        result = authorization_result.get("result")

        if isinstance(result, dict):
            for key in [
                "analyst_required",
                "requires_analyst",
                "review_required",
            ]:
                if key in result:
                    analyst_required = result[key]
                    break

            for key in [
                "containment_authorized",
                "containment_allowed",
                "authorized_for_containment",
            ]:
                if key in result:
                    containment_authorized = result[key]
                    break

            for key in [
                "execution_enabled",
                "execution_allowed",
                "authorized_for_execution",
            ]:
                if key in result:
                    execution_enabled = result[key]
                    break

    # -------------------------------------------------------------
    # Hard project safety truth
    # -------------------------------------------------------------

    # Phase 6 freeze explicitly established that execution is disabled.
    # We report this as a project boundary, not as an inference.
    execution_state = "DISABLED"
    containment_state = (
        "BLOCKED_BY_SAFETY_BOUNDARY"
    )

    return {
        "testcase_id": f"TC{testcase_number:02d}",
        "dataset_row": row_index,
        "actual_dataset_label": context.get(
            "actual_label",
            NOT_AVAILABLE,
        ),
        "predicted_attack_type": predicted_class,
        "model_probability": probability,
        "model_confidence": confidence,
        "confidence_band": classify_case_position(
            probability
        ),
        "risk_score": risk.get("risk_score"),
        "risk_level": risk.get(
            "risk_level",
            NOT_AVAILABLE,
        ),
        "decision_policy": decision_result,
        "response_planner": response_result,
        "mitigation_plan": mitigation,
        "analyst_authorization": authorization_result,
        "analyst_required": analyst_required,
        "containment_authorized": containment_authorized,
        "execution_state": execution_state,
        "containment_execution_state": containment_state,
        "behavior_evidence": context.get(
            "behavior_text",
            "",
        ),
        "metadata": context.get(
            "metadata",
            {},
        ),
        "evidence_status": (
            "ACTUAL_PRODUCTION_INFERENCE"
            if prediction.get("prediction_available")
            else "MODEL_OUTPUT_NOT_REPRODUCED"
        ),
        "synthetic_output": False,
        "response_executed": False,
        "containment_executed": False,
    }


# ---------------------------------------------------------------------
# Human-readable TXT report
# ---------------------------------------------------------------------

def format_value(value: Any, max_length: int = 160) -> str:
    if value is None:
        return NOT_AVAILABLE

    if isinstance(value, dict):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    elif isinstance(value, list):
        text = json.dumps(
            value,
            ensure_ascii=False,
        )

    else:
        text = str(value)

    text = text.replace("\n", " ")

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text


def write_txt_report(
    report: Dict[str, Any],
) -> None:

    lines: List[str] = []

    lines.append("=" * WIDTH)
    lines.append("SmartSOC v4.1")
    lines.append("SUPERVISOR AI TESTCASE REPORT")
    lines.append("=" * WIDTH)
    lines.append("")

    lines.append(
        "Purpose: Demonstrate actual SmartSOC production-model behavior "
        "using representative frozen test events."
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
    lines.append("")

    lines.append("EVIDENCE SOURCES")
    lines.append("-" * WIDTH)
    for key, value in report["evidence_sources"].items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("PRODUCTION MODEL")
    lines.append("-" * WIDTH)

    model_info = report["model"]
    lines.append(
        f"Class: {model_info.get('model_class')}"
    )
    lines.append(
        f"Classes: {model_info.get('class_count')}"
    )
    lines.append(
        f"Frozen events: {report['dataset']['rows']}"
    )

    lines.append("")
    lines.append("IMPORTANT INTERPRETATION")
    lines.append("-" * WIDTH)
    lines.append(
        "Prediction = serialized production model output when reproducible."
    )
    lines.append(
        "Attack type = model class, not a fabricated label."
    )
    lines.append(
        "Score/confidence = actual model probability when available."
    )
    lines.append(
        "Mitigation = actual response-planner output when reproducible."
    )
    lines.append(
        "Execution = disabled by the Phase 6 safety boundary."
    )
    lines.append(
        "No attack was executed and no containment was performed by this report."
    )

    lines.append("")
    lines.append("=" * WIDTH)
    lines.append("REPRESENTATIVE TESTCASES")
    lines.append("=" * WIDTH)

    for case in report["testcases"]:
        lines.append("")
        lines.append(
            f"{case['testcase_id']} "
            f"(frozen dataset row {case['dataset_row']})"
        )
        lines.append("-" * WIDTH)

        lines.append(
            f"Actual dataset label : "
            f"{format_value(case['actual_dataset_label'])}"
        )

        lines.append(
            f"AI predicted type    : "
            f"{format_value(case['predicted_attack_type'])}"
        )

        lines.append(
            f"Model score          : "
            f"{format_value(case['model_probability'])}"
        )

        lines.append(
            f"Confidence           : "
            f"{format_value(case['model_confidence'])}"
        )

        lines.append(
            f"Confidence band      : "
            f"{format_value(case['confidence_band'])}"
        )

        lines.append(
            f"Risk score           : "
            f"{format_value(case['risk_score'])}"
        )

        lines.append(
            f"Risk level           : "
            f"{format_value(case['risk_level'])}"
        )

        lines.append(
            f"Decision policy      : "
            f"{format_value(case['decision_policy'])}"
        )

        lines.append(
            f"Mitigation plan      : "
            f"{format_value(case['mitigation_plan'])}"
        )

        lines.append(
            f"Analyst required     : "
            f"{format_value(case['analyst_required'])}"
        )

        lines.append(
            f"Containment auth.    : "
            f"{format_value(case['containment_authorized'])}"
        )

        lines.append(
            f"Execution            : "
            f"{case['execution_state']}"
        )

        lines.append(
            f"Containment          : "
            f"{case['containment_execution_state']}"
        )

        lines.append(
            f"Evidence status      : "
            f"{case['evidence_status']}"
        )

        behavior = format_value(
            case.get("behavior_evidence", ""),
            max_length=300,
        )

        if behavior:
            lines.append(
                f"Behavior evidence    : {behavior}"
            )

    lines.append("")
    lines.append("=" * WIDTH)
    lines.append("FINAL STATUS")
    lines.append("=" * WIDTH)

    lines.append(
        f"Representative testcases : "
        f"{len(report['testcases'])}"
    )

    lines.append(
        f"Actual model predictions : "
        f"{report['summary']['actual_predictions']}"
    )

    lines.append(
        f"Unavailable predictions  : "
        f"{report['summary']['unavailable_predictions']}"
    )

    lines.append(
        "Synthetic predictions    : 0"
    )

    lines.append(
        "Response executed        : 0"
    )

    lines.append(
        "Containment executed     : 0"
    )

    lines.append("")
    lines.append(
        "REPORTING CONCLUSION:"
    )
    lines.append(
        "The report contains only reproducible production evidence."
    )

    with TXT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------

def write_csv_report(
    testcases: List[Dict[str, Any]],
) -> None:

    fieldnames = [
        "testcase_id",
        "dataset_row",
        "actual_dataset_label",
        "predicted_attack_type",
        "model_probability",
        "model_confidence",
        "confidence_band",
        "risk_score",
        "risk_level",
        "mitigation_plan",
        "analyst_required",
        "containment_authorized",
        "execution_state",
        "containment_execution_state",
        "evidence_status",
        "synthetic_output",
        "response_executed",
        "containment_executed",
        "behavior_evidence",
    ]

    with CSV_REPORT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for case in testcases:
            row = {}

            for field in fieldnames:
                value = case.get(field)

                if isinstance(value, (dict, list)):
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                row[field] = value

            writer.writerow(row)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:

    banner("SmartSOC v4.1")
    print("Supervisor AI Testcase Report")
    print("=" * WIDTH)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        # ---------------------------------------------------------
        # Load artifacts
        # ---------------------------------------------------------

        artifacts = load_artifacts()

        model = artifacts["model"]
        thresholds = artifacts["thresholds"]
        schema = artifacts["schema"]

        # ---------------------------------------------------------
        # Load data
        # ---------------------------------------------------------

        datasets = load_dataset()

        structured = datasets["structured"]

        # ---------------------------------------------------------
        # Resolve schema
        # ---------------------------------------------------------

        resolved_schema = resolve_schema(
            schema,
            structured,
        )

        passed(
            "Frozen feature schema resolved",
            (
                f"Structured features : "
                f"{len(resolved_schema['structured'])}\n"
                f"       Categorical features: "
                f"{len(resolved_schema['categorical'])}\n"
                f"       Numerical features  : "
                f"{len(resolved_schema['numerical'])}"
            ),
        )

        # ---------------------------------------------------------
        # Model inspection
        # ---------------------------------------------------------

        model_info = inspect_model(
            model,
            resolved_schema,
        )

        threshold_info = inspect_thresholds(
            thresholds,
        )

        # ---------------------------------------------------------
        # Attempt actual inference across frozen rows
        # ---------------------------------------------------------

        print()
        print(
            "[4] Reproducing actual production inference "
            "for representative selection..."
        )

        prediction_records: List[Dict[str, Any]] = []

        # We do not need to print all 11,707 events.
        # We inspect them internally to find representative actual
        # production outputs.
        #
        # Stop after enough unique successful predictions have been
        # collected, while maintaining deterministic row traversal.
        unique_predictions = set()

        max_scan = len(structured)

        for row_index in range(max_scan):

            inference = direct_model_inference(
                model,
                datasets["structured"],
                datasets["behavior"],
                datasets["metadata"],
                row_index,
            )

            prediction = extract_prediction(
                inference,
                model,
            )

            record = {
                "row_index": row_index,
                "prediction": prediction,
                "inference": inference,
            }

            prediction_records.append(record)

            if prediction.get("prediction_available"):
                unique_predictions.add(
                    str(
                        prediction.get(
                            "predicted_class"
                        )
                    )
                )

            # Enough diversity for representative cases.
            if (
                len(unique_predictions) >= 12
                and row_index >= 250
            ):
                break

        actual_predictions = sum(
            1
            for item in prediction_records
            if item["prediction"].get(
                "prediction_available"
            )
        )

        unavailable_predictions = (
            len(prediction_records)
            - actual_predictions
        )

        if actual_predictions:
            passed(
                "Actual production inference reproduced",
                (
                    f"{actual_predictions} successful "
                    f"rows examined"
                ),
            )
        else:
            reviewed(
                "Actual production inference",
                (
                    "No direct serialized-model interface could "
                    "be reproduced. No predictions fabricated."
                ),
            )

        # ---------------------------------------------------------
        # Select representative rows
        # ---------------------------------------------------------

        selected_rows = select_representative_rows(
            datasets,
            prediction_records,
            max_cases=12,
        )

        # Map row index -> prediction record
        prediction_map = {
            item["row_index"]: item
            for item in prediction_records
        }

        # For selected rows not scanned previously, perform actual
        # inference now.
        for row_index in selected_rows:
            if row_index not in prediction_map:

                inference = direct_model_inference(
                    model,
                    datasets["structured"],
                    datasets["behavior"],
                    datasets["metadata"],
                    row_index,
                )

                prediction = extract_prediction(
                    inference,
                    model,
                )

                prediction_map[row_index] = {
                    "row_index": row_index,
                    "prediction": prediction,
                    "inference": inference,
                }

        # ---------------------------------------------------------
        # Load existing decision/response components
        # ---------------------------------------------------------

        print()
        print(
            "[5] Loading existing validated SmartSOC "
            "decision/response components..."
        )

        decision_engine, decision_error = discover_component(
            "src/decision/smartsoc_decision_policy_v4_1.py",
            [
                "SmartSOCDecisionPolicy",
                "DecisionPolicy",
                "DecisionPolicyEngine",
            ],
        )

        response_engine, response_error = discover_component(
            "src/decision/smartsoc_response_planner_v4_1.py",
            [
                "SmartSOCResponsePlanner",
                "ResponsePlanner",
            ],
        )

        authorization_engine, authorization_error = (
            discover_component(
                "src/decision/smartsoc_analyst_authorization_v4_1.py",
                [
                    "SmartSOCAnalystAuthorization",
                    "AnalystAuthorization",
                ],
            )
        )

        if decision_engine is not None:
            passed(
                "Decision policy component loaded"
            )
        else:
            reviewed(
                "Decision policy component",
                decision_error or NOT_AVAILABLE,
            )

        if response_engine is not None:
            passed(
                "Response planner component loaded"
            )
        else:
            reviewed(
                "Response planner component",
                response_error or NOT_AVAILABLE,
            )

        if authorization_engine is not None:
            passed(
                "Analyst authorization component loaded"
            )
        else:
            reviewed(
                "Analyst authorization component",
                authorization_error or NOT_AVAILABLE,
            )

        # ---------------------------------------------------------
        # Build testcase records
        # ---------------------------------------------------------

        print()
        print(
            "[6] Building representative actual testcases..."
        )

        testcases: List[Dict[str, Any]] = []

        for number, row_index in enumerate(
            selected_rows,
            start=1,
        ):

            record = prediction_map[row_index]

            testcase = build_testcase(
                testcase_number=number,
                row_index=row_index,
                datasets=datasets,
                prediction_record=record,
                thresholds=thresholds,
                decision_engine=decision_engine,
                response_engine=response_engine,
                authorization_engine=authorization_engine,
            )

            testcases.append(testcase)

        passed(
            "Representative testcase report built",
            f"{len(testcases)} testcases",
        )

        # ---------------------------------------------------------
        # Summary
        # ---------------------------------------------------------

        actual_case_predictions = sum(
            1
            for case in testcases
            if case["evidence_status"]
            == "ACTUAL_PRODUCTION_INFERENCE"
        )

        unavailable_case_predictions = (
            len(testcases)
            - actual_case_predictions
        )

        report: Dict[str, Any] = {
            "report_name": (
                "SmartSOC Supervisor AI Testcase Report"
            ),
            "version": "4.1",
            "report_type": (
                "representative_actual_testcases"
            ),
            "project_root": str(PROJECT_ROOT),
            "generated_by": Path(__file__).name,
            "evidence_policy": {
                "synthetic_predictions": False,
                "fabricated_attack_types": False,
                "fabricated_scores": False,
                "fabricated_mitigation": False,
                "response_execution": False,
                "containment_execution": False,
            },
            "evidence_sources": {
                "production_model": str(MODEL_PATH),
                "production_thresholds": str(
                    THRESHOLDS_PATH
                ),
                "production_feature_schema": str(
                    SCHEMA_PATH
                ),
                "structured_test": str(
                    STRUCTURED_PATH
                ),
                "labels_test": str(
                    LABELS_PATH
                ),
                "metadata_test": str(
                    METADATA_PATH
                ),
                "behavior_text_test": str(
                    BEHAVIOR_PATH
                ),
            },
            "dataset": {
                "rows": len(structured),
                "structured_columns": len(
                    structured.columns
                ),
            },
            "model": model_info,
            "thresholds": threshold_info,
            "feature_schema": {
                "structured": len(
                    resolved_schema["structured"]
                ),
                "categorical": len(
                    resolved_schema["categorical"]
                ),
                "numerical": len(
                    resolved_schema["numerical"]
                ),
            },
            "summary": {
                "representative_testcases": len(
                    testcases
                ),
                "actual_predictions": (
                    actual_case_predictions
                ),
                "unavailable_predictions": (
                    unavailable_case_predictions
                ),
                "synthetic_predictions": 0,
                "response_executed": 0,
                "containment_executed": 0,
                "execution_state": "DISABLED",
            },
            "testcases": safe_json(testcases),
        }

        # ---------------------------------------------------------
        # Write outputs
        # ---------------------------------------------------------

        print()
        print(
            "[7] Writing supervisor evidence..."
        )

        with JSON_REPORT.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                safe_json(report),
                handle,
                indent=2,
                ensure_ascii=False,
            )

        write_csv_report(
            testcases
        )

        write_txt_report(
            report
        )

        passed(
            "JSON supervisor testcase report",
            str(JSON_REPORT),
        )

        passed(
            "CSV supervisor testcase report",
            str(CSV_REPORT),
        )

        passed(
            "TXT supervisor testcase report",
            str(TXT_REPORT),
        )

        # ---------------------------------------------------------
        # Final console summary
        # ---------------------------------------------------------

        print()
        banner(
            "SUPERVISOR TESTCASE REPORT COMPLETE"
        )

        print(
            f"Frozen events          : "
            f"{len(structured)}"
        )

        print(
            f"Representative cases  : "
            f"{len(testcases)}"
        )

        print(
            f"Actual predictions     : "
            f"{actual_case_predictions}"
        )

        print(
            f"Unavailable predictions: "
            f"{unavailable_case_predictions}"
        )

        print(
            "Synthetic predictions  : 0"
        )

        print(
            "Response execution     : DISABLED"
        )

        print(
            "Containment execution  : DISABLED"
        )

        print()
        print(
            f"JSON : {JSON_REPORT}"
        )

        print(
            f"CSV  : {CSV_REPORT}"
        )

        print(
            f"TXT  : {TXT_REPORT}"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Only actual reproducible production evidence "
            "is included."
        )

        print(
            "No synthetic attack predictions or mitigation "
            "actions were generated."
        )

        return 0

    except Exception as exc:

        print()
        failed(
            "Supervisor testcase report generation",
            f"{type(exc).__name__}: {exc}",
        )

        print()
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())