"""
SmartSOC AI Supervisor Evidence Report
Version 4.1

Purpose
-------
Generate supervisor-facing evidence from the ACTUAL SmartSOC v4.1
production inference implementation and frozen test artifacts.

IMPORTANT
---------
This script NEVER fabricates predictions.

It:
    1. Loads the frozen production artifacts.
    2. Loads the frozen 11,707-event test dataset.
    3. Imports the existing aligned v4.1 production inference module.
    4. Uses that module's real feature-building and prediction functions.
    5. Records actual predictions returned by the serialized model.
    6. Records threshold decisions from the production threshold artifact.
    7. Produces event-level and aggregate evidence.
    8. Explicitly records that response/containment execution is disabled.

If the existing inference interface cannot be used safely, the script
FAILS CLOSED and reports the reason.

It does NOT:
    - create synthetic events
    - create synthetic predictions
    - alter model artifacts
    - alter thresholds
    - execute response actions
    - execute containment
    - contact Wazuh
    - contact Shuffle
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

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

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLDS_FILE = MODEL_DIR / "thresholds.json"
MANIFEST_FILE = MODEL_DIR / "model_manifest.json"

STRUCTURED_FILE = DATA_DIR / "structured_test.csv"
TEXT_FILE = DATA_DIR / "behavior_text_test.csv"
LABELS_FILE = DATA_DIR / "labels_test.csv"
METADATA_FILE = DATA_DIR / "metadata_test.csv"

ALIGNED_INFERENCE_FILE = (
    PROJECT_ROOT
    / "src"
    / "production_inference_windows_apt_v4_1_aligned.py"
)

JSON_REPORT = (
    REPORT_DIR
    / "smartsoc_supervisor_ai_status_v4_1.json"
)

CSV_REPORT = (
    REPORT_DIR
    / "smartsoc_supervisor_event_outputs_v4_1.csv"
)

TXT_REPORT = (
    REPORT_DIR
    / "smartsoc_supervisor_ai_status_v4_1.txt"
)


# ============================================================================
# EXPECTED PRODUCTION CONTRACT
# ============================================================================

EXPECTED_CATEGORICAL_INPUT = 4
EXPECTED_CATEGORICAL_ENCODED = 162
EXPECTED_NUMERICAL = 58
EXPECTED_TFIDF = 6699
EXPECTED_FINAL = 6919
EXPECTED_TECHNIQUES = 45


# ============================================================================
# DISPLAY
# ============================================================================

def header(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


def section(text: str) -> None:
    print()
    print("-" * 72)
    print(text)
    print("-" * 72)


def passed(text: str) -> None:
    print(f"[PASS] {text}")


def review(text: str) -> None:
    print(f"[REVIEW] {text}")


def failed(text: str) -> None:
    print(f"[FAIL] {text}")


# ============================================================================
# SAFE HELPERS
# ============================================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_json_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, np.ndarray):
        return [
            safe_json_value(x)
            for x in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(k): safe_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_json_value(x)
            for x in value
        ]

    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )


# ============================================================================
# MODULE LOADING
# ============================================================================

def load_aligned_module():
    """
    Load the already-existing aligned production inference implementation.

    We intentionally do not duplicate its feature logic here.
    """

    require(ALIGNED_INFERENCE_FILE)

    spec = importlib.util.spec_from_file_location(
        "smartsoc_v41_aligned_inference",
        str(ALIGNED_INFERENCE_FILE),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to construct import specification for "
            "aligned production inference module."
        )

    module = importlib.util.module_from_spec(spec)

    sys.modules[
        "smartsoc_v41_aligned_inference"
    ] = module

    spec.loader.exec_module(module)

    return module


# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_artifacts() -> Dict[str, Any]:

    for path in [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
    ]:
        require(path)

    artifacts = {
        "model": joblib.load(MODEL_FILE),
        "encoder": joblib.load(ENCODER_FILE),
        "scaler": joblib.load(SCALER_FILE),
        "tfidf": joblib.load(TFIDF_FILE),
        "label_encoder": joblib.load(
            LABEL_ENCODER_FILE
        ),
        "schema": json.loads(
            SCHEMA_FILE.read_text(
                encoding="utf-8"
            )
        ),
        "thresholds": json.loads(
            THRESHOLDS_FILE.read_text(
                encoding="utf-8"
            ),
        ),
    }

    if MANIFEST_FILE.exists():
        artifacts["manifest"] = json.loads(
            MANIFEST_FILE.read_text(
                encoding="utf-8"
            )
        )
    else:
        artifacts["manifest"] = {}

    return artifacts


# ============================================================================
# DATASET
# ============================================================================

def load_dataset() -> Dict[str, pd.DataFrame]:

    for path in [
        STRUCTURED_FILE,
        TEXT_FILE,
        LABELS_FILE,
        METADATA_FILE,
    ]:
        require(path)

    structured = pd.read_csv(
        STRUCTURED_FILE
    )

    text = pd.read_csv(
        TEXT_FILE
    )

    labels = pd.read_csv(
        LABELS_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    lengths = {
        "structured": len(structured),
        "text": len(text),
        "labels": len(labels),
        "metadata": len(metadata),
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Frozen dataset row alignment failed: {lengths}"
        )

    return {
        "structured": structured,
        "text": text,
        "labels": labels,
        "metadata": metadata,
    }


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_production_contract(
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:

    model = artifacts["model"]
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    tfidf = artifacts["tfidf"]
    label_encoder = artifacts["label_encoder"]
    schema = artifacts["schema"]

    schema_categorical = schema.get(
        "categorical_features",
        schema.get(
            "categorical_columns",
            [],
        ),
    )

    schema_numerical = schema.get(
        "numerical_features",
        schema.get(
            "numerical_columns",
            [],
        ),
    )

    encoded_count = len(
        encoder.get_feature_names_out()
    )

    scaler_count = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    tfidf_count = len(
        tfidf.get_feature_names_out()
    )

    model_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_count is None:
        estimators = getattr(
            model,
            "estimators_",
            [],
        )

        if estimators:
            model_count = getattr(
                estimators[0],
                "n_features_in_",
                None,
            )

    technique_count = len(
        label_encoder.classes_
    )

    contract = {
        "categorical_input": len(
            schema_categorical
        ),
        "categorical_encoded": encoded_count,
        "numerical": scaler_count,
        "tfidf": tfidf_count,
        "final_model_features": model_count,
        "mitre_techniques": technique_count,
        "expected": {
            "categorical_input":
                EXPECTED_CATEGORICAL_INPUT,
            "categorical_encoded":
                EXPECTED_CATEGORICAL_ENCODED,
            "numerical":
                EXPECTED_NUMERICAL,
            "tfidf":
                EXPECTED_TFIDF,
            "final_model_features":
                EXPECTED_FINAL,
            "mitre_techniques":
                EXPECTED_TECHNIQUES,
        },
    }

    checks = [
        (
            "categorical_input",
            contract["categorical_input"],
            EXPECTED_CATEGORICAL_INPUT,
        ),
        (
            "categorical_encoded",
            contract["categorical_encoded"],
            EXPECTED_CATEGORICAL_ENCODED,
        ),
        (
            "numerical",
            contract["numerical"],
            EXPECTED_NUMERICAL,
        ),
        (
            "tfidf",
            contract["tfidf"],
            EXPECTED_TFIDF,
        ),
        (
            "final_model_features",
            contract["final_model_features"],
            EXPECTED_FINAL,
        ),
        (
            "mitre_techniques",
            contract["mitre_techniques"],
            EXPECTED_TECHNIQUES,
        ),
    ]

    failures = []

    for name, actual, expected in checks:

        if actual != expected:
            failures.append(
                {
                    "component": name,
                    "expected": expected,
                    "actual": actual,
                }
            )

    if failures:
        raise RuntimeError(
            "Production artifact contract mismatch:\n"
            + json.dumps(
                failures,
                indent=2,
            )
        )

    return contract


# ============================================================================
# ALIGNED INFERENCE ADAPTER
# ============================================================================

def find_function(module, name: str):
    function = getattr(
        module,
        name,
        None,
    )

    if function is None:
        raise RuntimeError(
            f"Aligned inference module does not expose "
            f"required function: {name}"
        )

    if not callable(function):
        raise RuntimeError(
            f"Aligned inference attribute is not callable: {name}"
        )

    return function


def normalize_result(
    prediction_result: Any,
) -> Dict[str, Any]:

    if not isinstance(
        prediction_result,
        dict,
    ):
        raise RuntimeError(
            "Aligned inference prediction result "
            "is not a dictionary."
        )

    return prediction_result


def extract_prediction_summary(
    prediction_result: Dict[str, Any],
) -> Dict[str, Any]:

    detected = prediction_result.get(
        "detected",
        [],
    )

    all_scores = prediction_result.get(
        "all_scores",
        [],
    )

    if detected is None:
        detected = []

    if all_scores is None:
        all_scores = []

    normalized_detected = []

    for item in detected:

        if not isinstance(item, dict):
            continue

        normalized_detected.append(
            {
                "technique": safe_string(
                    item.get("technique")
                ),
                "confidence": safe_json_value(
                    item.get("confidence")
                ),
                "threshold": safe_json_value(
                    item.get("threshold")
                ),
            }
        )

    normalized_scores = []

    for item in all_scores[:10]:

        if not isinstance(item, dict):
            continue

        normalized_scores.append(
            {
                "technique": safe_string(
                    item.get("technique")
                ),
                "confidence": safe_json_value(
                    item.get("confidence")
                ),
                "threshold": safe_json_value(
                    item.get("threshold")
                ),
                "detected": bool(
                    item.get("detected", False)
                ),
            }
        )

    return {
        "detected": normalized_detected,
        "top_scores": normalized_scores,
        "prediction_count": len(
            normalized_detected
        ),
    }


# ============================================================================
# FROZEN ROW -> RAW EVENT ADAPTER
# ============================================================================

def build_event_from_row(
    structured_row: pd.Series,
    text_row: pd.Series,
    metadata_row: pd.Series,
) -> Dict[str, Any]:

    """
    Build a flattened event containing the actual frozen row values.

    This does NOT invent missing security telemetry.

    Values that are absent from the frozen structured dataset remain absent.
    """

    event = {}

    for key, value in structured_row.to_dict().items():
        event[str(key)] = (
            None
            if pd.isna(value)
            else value
        )

    for key, value in text_row.to_dict().items():

        if (
            str(key)
            not in event
        ):
            event[str(key)] = (
                None
                if pd.isna(value)
                else value
            )

    for key, value in metadata_row.to_dict().items():

        if (
            str(key)
            not in event
        ):
            event[str(key)] = (
                None
                if pd.isna(value)
                else value
            )

    return event


# ============================================================================
# SINGLE EVENT INFERENCE
# ============================================================================

def infer_one(
    module,
    artifacts,
    structured_row,
    text_row,
    metadata_row,
) -> Dict[str, Any]:

    normalize = find_function(
        module,
        "normalize_wazuh_event",
    )

    build_features = find_function(
        module,
        "build_feature_matrix",
    )

    predict = find_function(
        module,
        "predict",
    )

    raw_event = build_event_from_row(
        structured_row,
        text_row,
        metadata_row,
    )

    normalized = normalize(
        raw_event
    )

    feature_result = build_features(
        normalized,
        artifacts,
    )

    prediction_result = predict(
        feature_result,
        artifacts,
    )

    summary = extract_prediction_summary(
        normalize_result(
            prediction_result
        )
    )

    return {
        "raw_event_fields": len(raw_event),
        "normalized": safe_json_value(
            normalized
        ),
        "feature_result": safe_json_value(
            {
                key: value
                for key, value in feature_result.items()
                if key not in {
                    "matrix",
                    "numerical_array",
                    "numerical_scaled",
                    "categorical_encoded",
                    "tfidf_features",
                    "numerical_sparse",
                }
            }
        ),
        "prediction": summary,
    }


# ============================================================================
# EVENT ID / LABEL HELPERS
# ============================================================================

def first_value(
    *values: Any,
) -> str:

    for value in values:

        text = safe_string(value)

        if text:
            return text

    return ""


def get_event_identifier(
    structured_row: pd.Series,
    metadata_row: pd.Series,
    index: int,
) -> str:

    candidates = [
        structured_row.get(
            "event_id",
            "",
        ),
        metadata_row.get(
            "event_id",
            "",
        ),
        metadata_row.get(
            "_id",
            "",
        ),
        metadata_row.get(
            "id",
            "",
        ),
    ]

    result = first_value(
        *candidates
    )

    if result:
        return result

    return f"frozen_row_{index}"


def get_actual_label(
    labels_row: pd.Series,
) -> str:

    for name in [
        "techniques",
        "technique",
        "label",
        "labels",
    ]:

        if name in labels_row.index:
            return safe_string(
                labels_row[name]
            )

    return ""


def get_behavior_text(
    text_row: pd.Series,
) -> str:

    for name in [
        "behavior_text_normalized",
        "behavior_text",
        "text",
    ]:

        if name in text_row.index:
            return safe_string(
                text_row[name]
            )

    values = []

    for value in text_row.tolist():

        text = safe_string(value)

        if text:
            values.append(text)

    return " ".join(values)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def run_report(
    module,
    artifacts,
    dataset,
) -> Dict[str, Any]:

    structured = dataset["structured"]
    text = dataset["text"]
    labels = dataset["labels"]
    metadata = dataset["metadata"]

    total = len(structured)

    event_records: List[Dict[str, Any]] = []

    failures: List[Dict[str, Any]] = []

    technique_counter = Counter()

    prediction_count = 0

    confidence_values = []

    actual_label_counter = Counter()

    for index in range(total):

        structured_row = structured.iloc[index]
        text_row = text.iloc[index]
        labels_row = labels.iloc[index]
        metadata_row = metadata.iloc[index]

        event_id = get_event_identifier(
            structured_row,
            metadata_row,
            index,
        )

        actual_label = get_actual_label(
            labels_row
        )

        behavior_text = get_behavior_text(
            text_row
        )

        try:

            result = infer_one(
                module,
                artifacts,
                structured_row,
                text_row,
                metadata_row,
            )

            prediction = result[
                "prediction"
            ]

            detected = prediction[
                "detected"
            ]

            for item in detected:

                technique = item[
                    "technique"
                ]

                if technique:
                    technique_counter[
                        technique
                    ] += 1

                confidence = item[
                    "confidence"
                ]

                if isinstance(
                    confidence,
                    (int, float),
                ):
                    confidence_values.append(
                        float(confidence)
                    )

            if detected:
                prediction_count += 1

            if actual_label:
                actual_label_counter[
                    actual_label
                ] += 1

            record = {
                "row_index": index,
                "event_id": event_id,
                "actual_label": actual_label,
                "behavior_text": behavior_text,
                "process_name":
                    safe_string(
                        structured_row.get(
                            "process_name",
                            "",
                        )
                    ),
                "parent_process_name":
                    safe_string(
                        structured_row.get(
                            "parent_process_name",
                            "",
                        )
                    ),
                "prediction_count":
                    prediction[
                        "prediction_count"
                    ],
                "predictions":
                    detected,
                "top_probability_scores":
                    prediction[
                        "top_scores"
                    ],
                "normalized_event":
                    result[
                        "normalized"
                    ],
                "feature_evidence":
                    result[
                        "feature_result"
                    ],
                "inference_status":
                    "ACTUAL_PRODUCTION_INFERENCE",
            }

            event_records.append(
                record
            )

        except Exception as exc:

            failures.append(
                {
                    "row_index": index,
                    "event_id": event_id,
                    "error_type":
                        type(exc).__name__,
                    "error":
                        str(exc),
                }
            )

            event_records.append(
                {
                    "row_index": index,
                    "event_id": event_id,
                    "actual_label": actual_label,
                    "behavior_text": behavior_text,
                    "prediction_count": None,
                    "predictions": [],
                    "top_probability_scores": [],
                    "inference_status": "FAILED",
                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "report_type":
            "SmartSOC Supervisor AI Status",
        "report_version":
            "4.1",
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "evidence_policy": {
            "synthetic_predictions":
                False,
            "synthetic_events":
                False,
            "prediction_source":
                "serialized production model",
            "feature_source":
                "existing aligned v4.1 production inference module",
            "execution_enabled":
                False,
            "containment_enabled":
                False,
            "wazuh_integration_enabled":
                False,
            "shuffle_integration_enabled":
                False,
        },
        "production_contract":
            validate_production_contract(
                artifacts
            ),
        "dataset": {
            "events":
                total,
            "structured_rows":
                len(structured),
            "text_rows":
                len(text),
            "label_rows":
                len(labels),
            "metadata_rows":
                len(metadata),
        },
        "inference": {
            "events_attempted":
                total,
            "events_successful":
                total - len(failures),
            "events_failed":
                len(failures),
            "events_with_prediction":
                prediction_count,
            "detected_techniques":
                dict(
                    technique_counter
                ),
            "actual_label_frequency":
                dict(
                    actual_label_counter
                ),
            "confidence": {
                "count":
                    len(confidence_values),
                "mean":
                    (
                        float(
                            np.mean(
                                confidence_values
                            )
                        )
                        if confidence_values
                        else None
                    ),
                "median":
                    (
                        float(
                            np.median(
                                confidence_values
                            )
                        )
                        if confidence_values
                        else None
                    ),
                "minimum":
                    (
                        float(
                            np.min(
                                confidence_values
                            )
                        )
                        if confidence_values
                        else None
                    ),
                "maximum":
                    (
                        float(
                            np.max(
                                confidence_values
                            )
                        )
                        if confidence_values
                        else None
                    ),
            },
        },
        "failures":
            failures,
        "event_records":
            event_records,
        "artifact_hashes": {
            path.name:
                sha256(path)
            for path in [
                MODEL_FILE,
                ENCODER_FILE,
                SCALER_FILE,
                TFIDF_FILE,
                LABEL_ENCODER_FILE,
                SCHEMA_FILE,
                THRESHOLDS_FILE,
            ]
        },
    }

    return report


# ============================================================================
# CSV
# ============================================================================

def write_csv(
    records: List[Dict[str, Any]],
) -> None:

    fields = [
        "row_index",
        "event_id",
        "actual_label",
        "process_name",
        "parent_process_name",
        "behavior_text",
        "prediction_count",
        "predictions",
        "top_probability_scores",
        "inference_status",
        "error",
    ]

    with CSV_REPORT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:

            writer.writerow(
                {
                    field:
                        json.dumps(
                            record.get(
                                field,
                                "",
                            ),
                            ensure_ascii=False,
                        )
                        if field
                        in {
                            "predictions",
                            "top_probability_scores",
                        }
                        else record.get(
                            field,
                            "",
                        )
                    for field in fields
                }
            )


# ============================================================================
# TXT
# ============================================================================

def write_txt(
    report: Dict[str, Any],
) -> None:

    contract = report[
        "production_contract"
    ]

    inference = report[
        "inference"
    ]

    failures = report[
        "failures"
    ]

    lines = []

    lines.append(
        "SmartSOC AI Supervisor Status Report"
    )

    lines.append(
        "Version 4.1"
    )

    lines.append("")

    lines.append(
        "IMPORTANT: This report contains "
        "actual production-model inference only."
    )

    lines.append(
        "Synthetic predictions are disabled."
    )

    lines.append("")

    lines.append(
        "PRODUCTION MODEL"
    )

    lines.append(
        f"Model: {MODEL_FILE}"
    )

    lines.append(
        f"Model input features: "
        f"{contract['final_model_features']}"
    )

    lines.append(
        f"MITRE techniques: "
        f"{contract['mitre_techniques']}"
    )

    lines.append("")

    lines.append(
        "FEATURE PIPELINE"
    )

    lines.append(
        f"Categorical input: "
        f"{contract['categorical_input']}"
    )

    lines.append(
        f"Categorical encoded: "
        f"{contract['categorical_encoded']}"
    )

    lines.append(
        f"Numerical: "
        f"{contract['numerical']}"
    )

    lines.append(
        f"TF-IDF: "
        f"{contract['tfidf']}"
    )

    lines.append(
        f"Final vector: "
        f"{contract['final_model_features']}"
    )

    lines.append("")

    lines.append(
        "FROZEN DATASET"
    )

    lines.append(
        f"Events: "
        f"{report['dataset']['events']}"
    )

    lines.append("")

    lines.append(
        "ACTUAL INFERENCE"
    )

    lines.append(
        f"Events attempted: "
        f"{inference['events_attempted']}"
    )

    lines.append(
        f"Successful: "
        f"{inference['events_successful']}"
    )

    lines.append(
        f"Failed: "
        f"{inference['events_failed']}"
    )

    lines.append(
        f"Events with >=1 prediction: "
        f"{inference['events_with_prediction']}"
    )

    lines.append("")

    lines.append(
        "DETECTED TECHNIQUES"
    )

    for technique, count in sorted(
        inference[
            "detected_techniques"
        ].items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        lines.append(
            f"{technique}: {count}"
        )

    confidence = inference[
        "confidence"
    ]

    lines.append("")

    lines.append(
        "CONFIDENCE"
    )

    lines.append(
        f"Count: {confidence['count']}"
    )

    lines.append(
        f"Mean: {confidence['mean']}"
    )

    lines.append(
        f"Median: {confidence['median']}"
    )

    lines.append(
        f"Minimum: {confidence['minimum']}"
    )

    lines.append(
        f"Maximum: {confidence['maximum']}"
    )

    lines.append("")

    lines.append(
        "OPERATIONAL SAFETY"
    )

    lines.append(
        "Response execution: DISABLED"
    )

    lines.append(
        "Containment execution: DISABLED"
    )

    lines.append(
        "Wazuh integration: DISABLED"
    )

    lines.append(
        "Shuffle integration: DISABLED"
    )

    lines.append("")

    if failures:

        lines.append(
            "FAILURES"
        )

        for failure in failures[:20]:

            lines.append(
                json.dumps(
                    failure,
                    ensure_ascii=False,
                )
            )

    else:

        lines.append(
            "FAILURES: NONE"
        )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    if failures:

        lines.append(
            "ACTUAL INFERENCE WAS NOT COMPLETELY "
            "REPRODUCED FOR ALL EVENTS."
        )

        lines.append(
            "Do not claim complete 11,707-event "
            "production inference from this report."
        )

    else:

        lines.append(
            "All frozen events were processed "
            "through the existing aligned "
            "production inference implementation."
        )

        lines.append(
            "Predictions in this report are "
            "serialized-model outputs."
        )

    TXT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    header(
        "SmartSOC AI Supervisor Status Report"
    )

    print(
        "Version 4.1"
    )

    try:

        section(
            "1. Loading aligned production inference"
        )

        module = load_aligned_module()

        passed(
            "Existing aligned production inference module loaded"
        )

        section(
            "2. Loading production artifacts"
        )

        artifacts = load_artifacts()

        passed(
            "Production artifacts loaded"
        )

        section(
            "3. Validating frozen production contract"
        )

        contract = validate_production_contract(
            artifacts
        )

        passed(
            "Production feature contract validated"
        )

        print(
            f"       categorical input  = "
            f"{contract['categorical_input']}"
        )

        print(
            f"       categorical encoded = "
            f"{contract['categorical_encoded']}"
        )

        print(
            f"       numerical           = "
            f"{contract['numerical']}"
        )

        print(
            f"       TF-IDF              = "
            f"{contract['tfidf']}"
        )

        print(
            f"       final               = "
            f"{contract['final_model_features']}"
        )

        print(
            f"       MITRE techniques    = "
            f"{contract['mitre_techniques']}"
        )

        section(
            "4. Loading frozen test dataset"
        )

        dataset = load_dataset()

        passed(
            f"Frozen dataset loaded: "
            f"{len(dataset['structured'])} events"
        )

        section(
            "5. Running ACTUAL production inference"
        )

        print(
            "Synthetic predictions: DISABLED"
        )

        print(
            "Response execution: DISABLED"
        )

        print(
            "Containment execution: DISABLED"
        )

        report = run_report(
            module,
            artifacts,
            dataset,
        )

        successful = report[
            "inference"
        ][
            "events_successful"
        ]

        failed_count = report[
            "inference"
        ][
            "events_failed"
        ]

        if failed_count == 0:

            passed(
                f"Actual production inference completed: "
                f"{successful} events"
            )

        else:

            review(
                f"Actual production inference completed "
                f"for {successful} events; "
                f"{failed_count} failed"
            )

        section(
            "6. Writing supervisor evidence"
        )

        JSON_REPORT.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        write_csv(
            report[
                "event_records"
            ]
        )

        write_txt(
            report
        )

        passed(
            f"JSON report: {JSON_REPORT}"
        )

        passed(
            f"CSV report: {CSV_REPORT}"
        )

        passed(
            f"TXT report: {TXT_REPORT}"
        )

        header(
            "SUPERVISOR REPORT COMPLETE"
        )

        print(
            f"Frozen events      : "
            f"{len(dataset['structured'])}"
        )

        print(
            f"Actual successful  : "
            f"{successful}"
        )

        print(
            f"Actual failures    : "
            f"{failed_count}"
        )

        print(
            "Synthetic output   : DISABLED"
        )

        print(
            "Response execution : DISABLED"
        )

        print(
            "Containment        : DISABLED"
        )

        if failed_count == 0:

            print()
            print(
                "ACTUAL PRODUCTION INFERENCE "
                "EVIDENCE GENERATED."
            )

            return 0

        print()
        print(
            "REPORT GENERATED WITH REVIEW STATUS."
        )

        print(
            "No fabricated predictions were inserted."
        )

        return 2

    except Exception as exc:

        failed(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "FAIL-CLOSED:"
        )

        print(
            "No synthetic predictions were generated."
        )

        print(
            "No model artifacts were modified."
        )

        print(
            "No response actions were executed."
        )

        print()
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )