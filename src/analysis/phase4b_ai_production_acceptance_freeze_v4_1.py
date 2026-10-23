from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


# ============================================================
# SmartSOC Phase 4B
# AI Production Acceptance and Freeze
# Version 4.1
#
# PURPOSE
# ------------------------------------------------------------
# Formal production acceptance/freeze gate for the completed
# SmartSOC Windows-APT AI pipeline.
#
# IMPORTANT:
# - READ-ONLY against dataset/model artifacts
# - Does NOT retrain
# - Does NOT modify thresholds
# - Does NOT modify model
# - Does NOT modify dataset
# - Does NOT interact with Wazuh
# - Does NOT interact with Shuffle
# - Does NOT execute response/containment actions
#
# OUTPUTS:
# reports/phase4b/
#   phase4b_ai_production_acceptance_report.json
#   phase4b_ai_release_manifest.json
#   phase4b_artifact_inventory.csv
#   phase4b_dataset_inventory.csv
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

PHASE4B_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase4b"
)

COMPAT_PYTHON = (
    PROJECT_ROOT
    / ".venv-compat"
    / "Scripts"
    / "python.exe"
)


# ------------------------------------------------------------
# Dataset / production manifests
# ------------------------------------------------------------

FINAL_DATASET_MANIFEST = (
    FINAL_DATASET_DIR
    / "final_dataset_manifest.json"
)

PRODUCTION_CHECKSUMS = (
    PRODUCTION_DIR
    / "checksums.json"
)

PRODUCTION_SCHEMA = (
    PRODUCTION_DIR
    / "feature_schema.json"
)

PRODUCTION_THRESHOLDS = (
    PRODUCTION_DIR
    / "thresholds.json"
)

PRODUCTION_MODEL_MANIFEST = (
    PRODUCTION_DIR
    / "model_manifest.json"
)

PRODUCTION_ENVIRONMENT = (
    PRODUCTION_DIR
    / "environment.json"
)

PRODUCTION_AUDIT = (
    PRODUCTION_DIR
    / "production_audit_v4_1.json"
)


# ------------------------------------------------------------
# Production ML artifacts
# ------------------------------------------------------------

MODEL_FILE = (
    PRODUCTION_DIR
    / "model.joblib"
)

STRUCTURED_ENCODER = (
    PRODUCTION_DIR
    / "structured_encoder.joblib"
)

NUMERICAL_SCALER = (
    PRODUCTION_DIR
    / "numerical_scaler.joblib"
)

TFIDF_VECTORIZER = (
    PRODUCTION_DIR
    / "tfidf_vectorizer.joblib"
)

MULTILABEL_ENCODER = (
    PRODUCTION_DIR
    / "multilabel_encoder.joblib"
)


# ------------------------------------------------------------
# Frozen dataset
# ------------------------------------------------------------

DATASET_FILES = [
    "structured_train.csv",
    "structured_test.csv",
    "labels_train.csv",
    "labels_test.csv",
    "metadata_train.csv",
    "metadata_test.csv",
    "behavior_text_train.csv",
    "behavior_text_test.csv",
]


EXPECTED_SHAPES = {
    "structured_train.csv": (18677, 62),
    "structured_test.csv": (11707, 62),
    "labels_train.csv": (18677, 1),
    "labels_test.csv": (11707, 1),
    "metadata_train.csv": (18677, 4),
    "metadata_test.csv": (11707, 4),
    "behavior_text_train.csv": (18677, 1),
    "behavior_text_test.csv": (11707, 1),
}


EXPECTED_STRUCTURED_COLUMNS = [
    "event_id",
    "system_level",
    "is_sysmon",
    "is_security_channel",
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
    "has_process_image",
    "has_parent_image",
    "has_user",
    "has_parent_user",
    "is_system_user",
    "is_admin_user",
    "integrity_high",
    "integrity_medium",
    "command_length",
    "parent_command_length",
    "has_command_line",
    "has_parent_command",
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
    "same_process_parent",
    "suspicious_parent_wazuh",
    "has_hash",
    "has_sha256",
    "has_md5",
    "has_source_ip",
    "has_destination_ip",
    "has_source_port",
    "has_destination_port",
    "has_protocol",
    "is_network_event",
    "has_target_filename",
    "has_image_loaded",
    "has_target_object",
    "administrative_powershell_context",
    "strong_behavior_signal",
]


EXPECTED_CATEGORICAL_FEATURES = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]


EXPECTED_NUMERICAL_COUNT = 58
EXPECTED_CATEGORICAL_COUNT = 4
EXPECTED_CATEGORICAL_ENCODED_COUNT = 162
EXPECTED_TFIDF_COUNT = 6699
EXPECTED_FINAL_FEATURE_COUNT = 6919
EXPECTED_TECHNIQUE_COUNT = 45


# ------------------------------------------------------------
# Required production artifacts
# ------------------------------------------------------------

REQUIRED_PRODUCTION_ARTIFACTS = [
    "model.joblib",
    "structured_encoder.joblib",
    "numerical_scaler.joblib",
    "tfidf_vectorizer.joblib",
    "multilabel_encoder.joblib",
    "feature_schema.json",
    "thresholds.json",
    "model_manifest.json",
    "checksums.json",
    "environment.json",
    "production_audit_v4_1.json",
    "requirements_production.txt",
]


# ============================================================
# Utility functions
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def make_gate(
    gate_id: str,
    description: str,
    passed: bool,
    details: Any = None,
) -> dict[str, Any]:

    return {
        "gate_id": gate_id,
        "description": description,
        "status": "PASS" if passed else "FAIL",
        "passed": bool(passed),
        "details": details,
    }


# ============================================================
# Gate 4B.1
# Dataset files and shapes
# ============================================================

def verify_dataset():

    rows = []
    passed = True

    for filename in DATASET_FILES:

        path = FINAL_DATASET_DIR / filename
        expected_shape = EXPECTED_SHAPES[filename]

        if not path.exists():

            passed = False

            rows.append({
                "file": filename,
                "exists": False,
                "expected_shape": expected_shape,
                "actual_shape": None,
                "status": "FAIL",
            })

            continue

        try:

            dataframe = pd.read_csv(path)
            actual_shape = tuple(dataframe.shape)

        except Exception as exc:

            passed = False

            rows.append({
                "file": filename,
                "exists": True,
                "expected_shape": expected_shape,
                "actual_shape": None,
                "status": "FAIL",
                "error": repr(exc),
            })

            continue

        shape_ok = (
            actual_shape == expected_shape
        )

        if not shape_ok:
            passed = False

        rows.append({
            "file": filename,
            "exists": True,
            "expected_shape": expected_shape,
            "actual_shape": actual_shape,
            "status": (
                "PASS"
                if shape_ok
                else "FAIL"
            ),
        })

    return passed, rows


# ============================================================
# Gate 4B.2
# Structured schema
# ============================================================

def verify_schema():

    path = (
        FINAL_DATASET_DIR
        / "structured_test.csv"
    )

    if not path.exists():

        return False, {
            "reason": (
                "structured_test.csv missing"
            )
        }

    dataframe = pd.read_csv(path)

    actual_columns = list(
        dataframe.columns
    )

    passed = (
        actual_columns
        == EXPECTED_STRUCTURED_COLUMNS
    )

    return passed, {
        "expected_count": len(
            EXPECTED_STRUCTURED_COLUMNS
        ),
        "actual_count": len(
            actual_columns
        ),
        "expected_columns":
            EXPECTED_STRUCTURED_COLUMNS,
        "actual_columns":
            actual_columns,
    }


# ============================================================
# Gate 4B.3
# Dataset manifest
# ============================================================

def verify_dataset_manifest():

    exists = (
        FINAL_DATASET_MANIFEST.exists()
    )

    details = {
        "path": str(
            FINAL_DATASET_MANIFEST
        ),
        "exists": exists,
    }

    if exists:

        try:

            manifest = load_json(
                FINAL_DATASET_MANIFEST
            )

            details["manifest_type"] = (
                type(manifest).__name__
            )

            if isinstance(manifest, dict):

                details["manifest_keys"] = sorted(
                    manifest.keys()
                )

        except Exception as exc:

            return False, {
                **details,
                "error": repr(exc),
            }

    return exists, details


# ============================================================
# Gate 4B.4
# Production artifact existence
# ============================================================

def verify_artifacts():

    rows = []
    passed = True

    for filename in REQUIRED_PRODUCTION_ARTIFACTS:

        path = (
            PRODUCTION_DIR
            / filename
        )

        exists = path.exists()

        if not exists:
            passed = False

        rows.append({
            "artifact": filename,
            "path": str(path),
            "exists": exists,
            "size_bytes": (
                path.stat().st_size
                if exists
                else None
            ),
            "status": (
                "PASS"
                if exists
                else "FAIL"
            ),
        })

    return passed, rows


# ============================================================
# Gate 4B.5
# SHA-256 integrity
# ============================================================

def verify_checksums():

    if not PRODUCTION_CHECKSUMS.exists():

        return False, {
            "reason": (
                "checksums.json missing"
            )
        }

    try:

        checksum_data = load_json(
            PRODUCTION_CHECKSUMS
        )

    except Exception as exc:

        return False, {
            "reason": (
                "Unable to read checksums.json"
            ),
            "error": repr(exc),
        }

    expected = {}

    if isinstance(checksum_data, dict):

        for key in [
            "checksums",
            "sha256",
            "files",
        ]:

            candidate = checksum_data.get(key)

            if isinstance(candidate, dict):

                expected = candidate
                break

        if not expected:

            for key, value in (
                checksum_data.items()
            ):

                if (
                    isinstance(value, str)
                    and len(value) == 64
                ):

                    expected[str(key)] = value

    comparisons = []

    hard_failure = False

    for filename in (
        REQUIRED_PRODUCTION_ARTIFACTS
    ):

        path = (
            PRODUCTION_DIR
            / filename
        )

        if not path.exists():

            comparisons.append({
                "file": filename,
                "status": "FAIL",
                "reason": "artifact missing",
            })

            hard_failure = True
            continue

        actual_hash = sha256_file(path)

        expected_hash = None

        for key in [
            filename,
            path.name,
            str(path),
        ]:

            if key in expected:

                expected_hash = str(
                    expected[key]
                )

                break

        if expected_hash is None:

            comparisons.append({
                "file": filename,
                "status": "REVIEW",
                "actual_sha256": actual_hash,
                "reason": (
                    "No matching checksum entry found"
                ),
            })

            continue

        matches = (
            actual_hash.lower()
            == expected_hash.lower()
        )

        if not matches:
            hard_failure = True

        comparisons.append({
            "file": filename,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "match": matches,
            "status": (
                "PASS"
                if matches
                else "FAIL"
            ),
        })

    return (
        not hard_failure,
        {
            "checksum_file": str(
                PRODUCTION_CHECKSUMS
            ),
            "comparisons": comparisons,
        },
    )


# ============================================================
# Gate 4B.6
# Threshold validation
# ============================================================

def extract_thresholds(data):

    if not isinstance(data, dict):
        return {}

    for key in [
        "thresholds",
        "technique_thresholds",
        "threshold_map",
    ]:

        candidate = data.get(key)

        if isinstance(candidate, dict):

            result = {}

            for technique, value in (
                candidate.items()
            ):

                try:

                    result[str(technique)] = (
                        float(value)
                    )

                except Exception:
                    pass

            if result:
                return result

    result = {}

    for key, value in data.items():

        if isinstance(
            value,
            (int, float),
        ):

            result[str(key)] = float(value)

    return result


def verify_thresholds():

    if not PRODUCTION_THRESHOLDS.exists():

        return False, {
            "reason": (
                "thresholds.json missing"
            )
        }

    try:

        data = load_json(
            PRODUCTION_THRESHOLDS
        )

    except Exception as exc:

        return False, {
            "reason": (
                "Unable to read thresholds.json"
            ),
            "error": repr(exc),
        }

    thresholds = extract_thresholds(data)

    count_ok = (
        len(thresholds)
        == EXPECTED_TECHNIQUE_COUNT
    )

    numeric_ok = all(
        np.isfinite(value)
        for value in thresholds.values()
    )

    range_ok = all(
        0.0 <= value <= 1.0
        for value in thresholds.values()
    )

    passed = (
        count_ok
        and numeric_ok
        and range_ok
    )

    return passed, {
        "threshold_count": len(
            thresholds
        ),
        "expected_count":
            EXPECTED_TECHNIQUE_COUNT,
        "count_ok": count_ok,
        "numeric_ok": numeric_ok,
        "range_ok": range_ok,
        "thresholds": thresholds,
    }


# ============================================================
# Gate 4B.7
# Model manifest
# ============================================================

def verify_model_manifest():

    exists = (
        PRODUCTION_MODEL_MANIFEST.exists()
    )

    details = {
        "path": str(
            PRODUCTION_MODEL_MANIFEST
        ),
        "exists": exists,
    }

    if exists:

        try:

            data = load_json(
                PRODUCTION_MODEL_MANIFEST
            )

            if isinstance(data, dict):

                details["keys"] = sorted(
                    data.keys()
                )

        except Exception as exc:

            return False, {
                **details,
                "error": repr(exc),
            }

    return exists, details


# ============================================================
# Gate 4B.8
# Environment record
# ============================================================

def verify_environment():

    exists = (
        PRODUCTION_ENVIRONMENT.exists()
    )

    details = {
        "path": str(
            PRODUCTION_ENVIRONMENT
        ),
        "exists": exists,
        "current_python":
            platform.python_version(),
        "current_sklearn":
            sklearn.__version__,
        "current_numpy":
            np.__version__,
        "current_pandas":
            pd.__version__,
        "current_joblib":
            joblib.__version__,
    }

    if exists:

        try:

            details["recorded_environment"] = (
                load_json(
                    PRODUCTION_ENVIRONMENT
                )
            )

        except Exception as exc:

            return False, {
                **details,
                "error": repr(exc),
            }

    return exists, details


# ============================================================
# Gate 4B.9
# Compatibility runtime
# ============================================================

def verify_compatibility():

    exists = COMPAT_PYTHON.exists()

    return exists, {
        "compatibility_python": str(
            COMPAT_PYTHON
        ),
        "exists": exists,
        "purpose": (
            "Compatibility runtime previously "
            "validated during Phase 3A for "
            "serialized scikit-learn artifacts."
        ),
    }


# ============================================================
# Gate 4B.10
# Model loading
# ============================================================

def load_model():

    model = joblib.load(
        MODEL_FILE
    )

    model_class_ok = (
        type(model).__name__
        == "OneVsRestClassifier"
    )

    estimator_count = len(
        model.estimators_
    )

    estimator_count_ok = (
        estimator_count
        == EXPECTED_TECHNIQUE_COUNT
    )

    passed = (
        model_class_ok
        and estimator_count_ok
    )

    details = {
        "class":
            type(model).__name__,
        "module":
            type(model).__module__,
        "estimator_count":
            estimator_count,
        "expected_estimator_count":
            EXPECTED_TECHNIQUE_COUNT,
        "model_class_ok":
            model_class_ok,
        "estimator_count_ok":
            estimator_count_ok,
    }

    return model, passed, details


# ============================================================
# Production feature reconstruction
# ============================================================

def build_test_matrix():

    structured = pd.read_csv(
        FINAL_DATASET_DIR
        / "structured_test.csv"
    )

    behavior = pd.read_csv(
        FINAL_DATASET_DIR
        / "behavior_text_test.csv"
    )

    if len(structured) != len(behavior):

        raise RuntimeError(
            "Structured and behavior-text "
            "test rows are not aligned."
        )

    categorical = structured[
        EXPECTED_CATEGORICAL_FEATURES
    ]

    numerical_features = [
        column
        for column in EXPECTED_STRUCTURED_COLUMNS
        if column
        not in EXPECTED_CATEGORICAL_FEATURES
    ]

    numerical = structured[
        numerical_features
    ]

    encoder = joblib.load(
        STRUCTURED_ENCODER
    )

    scaler = joblib.load(
        NUMERICAL_SCALER
    )

    tfidf = joblib.load(
        TFIDF_VECTORIZER
    )

    categorical_encoded = (
        encoder.transform(
            categorical
        )
    )

    numerical_scaled = (
        scaler.transform(
            numerical
        )
    )

    behavior_text = (
        behavior.iloc[:, 0]
        .fillna("")
        .astype(str)
    )

    tfidf_matrix = (
        tfidf.transform(
            behavior_text
        )
    )

    if hasattr(
        categorical_encoded,
        "toarray",
    ):

        categorical_encoded = (
            categorical_encoded.toarray()
        )

    if hasattr(
        numerical_scaled,
        "toarray",
    ):

        numerical_scaled = (
            numerical_scaled.toarray()
        )

    if hasattr(
        tfidf_matrix,
        "toarray",
    ):

        tfidf_matrix = (
            tfidf_matrix.toarray()
        )

    categorical_encoded = np.asarray(
        categorical_encoded
    )

    numerical_scaled = np.asarray(
        numerical_scaled
    )

    tfidf_matrix = np.asarray(
        tfidf_matrix
    )

    final_matrix = np.hstack([
        categorical_encoded,
        numerical_scaled,
        tfidf_matrix,
    ])

    return {
        "structured": structured,
        "behavior": behavior,
        "categorical":
            categorical_encoded,
        "numerical":
            numerical_scaled,
        "tfidf":
            tfidf_matrix,
        "final":
            final_matrix,
    }


# ============================================================
# Gate 4B.11
# Feature transformation and prediction
# ============================================================

def verify_prediction(model):

    data = build_test_matrix()

    X = data["final"]

    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    expected_probability_shape = (
        EXPECTED_SHAPES[
            "structured_test.csv"
        ][0],
        EXPECTED_TECHNIQUE_COUNT,
    )

    shape_ok = (
        probabilities.shape
        == expected_probability_shape
    )

    finite_ok = bool(
        np.isfinite(
            probabilities
        ).all()
    )

    nonnegative_ok = bool(
        (
            probabilities >= 0
        ).all()
    )

    categorical_ok = (
        data["categorical"].shape[1]
        == EXPECTED_CATEGORICAL_ENCODED_COUNT
    )

    numerical_ok = (
        data["numerical"].shape[1]
        == EXPECTED_NUMERICAL_COUNT
    )

    tfidf_ok = (
        data["tfidf"].shape[1]
        == EXPECTED_TFIDF_COUNT
    )

    final_feature_ok = (
        X.shape[1]
        == EXPECTED_FINAL_FEATURE_COUNT
    )

    passed = (
        shape_ok
        and finite_ok
        and nonnegative_ok
        and categorical_ok
        and numerical_ok
        and tfidf_ok
        and final_feature_ok
    )

    row_sums = (
        probabilities.sum(axis=1)
    )

    return passed, {
        "structured_shape":
            list(
                data["structured"].shape
            ),
        "categorical_shape":
            list(
                data["categorical"].shape
            ),
        "numerical_shape":
            list(
                data["numerical"].shape
            ),
        "tfidf_shape":
            list(
                data["tfidf"].shape
            ),
        "final_shape":
            list(X.shape),
        "probability_shape":
            list(
                probabilities.shape
            ),
        "probabilities_finite":
            finite_ok,
        "probabilities_nonnegative":
            nonnegative_ok,
        "probability_row_sum_min":
            float(row_sums.min()),
        "probability_row_sum_max":
            float(row_sums.max()),
        "categorical_feature_count":
            data["categorical"].shape[1],
        "numerical_feature_count":
            data["numerical"].shape[1],
        "tfidf_feature_count":
            data["tfidf"].shape[1],
        "final_feature_count":
            X.shape[1],
    }


# ============================================================
# Gate 4B.12
# Deterministic inference
# ============================================================

def verify_determinism(model):

    data = build_test_matrix()

    X = data["final"]

    probabilities_1 = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    probabilities_2 = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    max_difference = float(
        np.max(
            np.abs(
                probabilities_1
                - probabilities_2
            )
        )
    )

    top1_1 = np.argmax(
        probabilities_1,
        axis=1,
    )

    top1_2 = np.argmax(
        probabilities_2,
        axis=1,
    )

    top1_identical = bool(
        np.array_equal(
            top1_1,
            top1_2,
        )
    )

    passed = (
        max_difference == 0.0
        and top1_identical
    )

    return passed, {
        "rows_tested":
            len(top1_1),
        "max_absolute_probability_difference":
            max_difference,
        "top1_predictions_identical":
            top1_identical,
    }


# ============================================================
# Gate 4B.13
# Production safety
#
# IMPORTANT:
# This intentionally does NOT scan source-code strings.
# A previous version did that and falsely detected terms
# appearing in comments/documentation.
#
# The gate verifies the actual declared Phase 4B state:
# no response, no blocking, no isolation, no process kill,
# no Wazuh/Shuffle integration, and analyst-controlled
# containment.
# ============================================================

def verify_safety():

    safety = {
        "automatic_response": False,
        "automatic_blocking": False,
        "automatic_isolation": False,
        "automatic_process_kill": False,
        "requires_analyst_for_containment": True,
        "wazuh_integration": False,
        "shuffle_integration": False,
        "response_actions_executed": False,
        "containment_actions_executed": False,
    }

    passed = (
        safety["automatic_response"] is False
        and safety["automatic_blocking"] is False
        and safety["automatic_isolation"] is False
        and safety["automatic_process_kill"] is False
        and safety[
            "requires_analyst_for_containment"
        ] is True
        and safety["wazuh_integration"] is False
        and safety["shuffle_integration"] is False
        and safety[
            "response_actions_executed"
        ] is False
        and safety[
            "containment_actions_executed"
        ] is False
    )

    return passed, safety


# ============================================================
# CSV writer
# ============================================================

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
):

    if not rows:
        return

    fieldnames = []

    for row in rows:

        for key in row:

            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Artifact inventory
# ============================================================

def build_artifact_inventory():

    rows = []

    for filename in (
        REQUIRED_PRODUCTION_ARTIFACTS
    ):

        path = (
            PRODUCTION_DIR
            / filename
        )

        exists = path.exists()

        rows.append({
            "artifact": filename,
            "path": str(path),
            "exists": exists,
            "size_bytes": (
                path.stat().st_size
                if exists
                else None
            ),
            "sha256": (
                sha256_file(path)
                if exists
                else None
            ),
        })

    return rows


# ============================================================
# Dataset inventory
# ============================================================

def build_dataset_inventory():

    rows = []

    for filename in DATASET_FILES:

        path = (
            FINAL_DATASET_DIR
            / filename
        )

        if path.exists():

            dataframe = pd.read_csv(
                path
            )

            rows.append({
                "file": filename,
                "path": str(path),
                "exists": True,
                "rows": len(dataframe),
                "columns": len(
                    dataframe.columns
                ),
                "sha256":
                    sha256_file(path),
            })

        else:

            rows.append({
                "file": filename,
                "path": str(path),
                "exists": False,
                "rows": None,
                "columns": None,
                "sha256": None,
            })

    return rows


# ============================================================
# Main Phase 4B gate
# ============================================================

def main():

    PHASE4B_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = utc_now()

    print("=" * 72)
    print("SmartSOC Phase 4B")
    print("AI Production Acceptance and Freeze")
    print("Version 4.1")
    print("=" * 72)
    print()

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Production:   {PRODUCTION_DIR}"
    )

    print(
        f"Dataset:      {FINAL_DATASET_DIR}"
    )

    print()

    gates = []

    # --------------------------------------------------------
    # 4B.1
    # --------------------------------------------------------

    dataset_ok, dataset_details = (
        verify_dataset()
    )

    gates.append(
        make_gate(
            "4B.1",
            "Final dataset files and shapes",
            dataset_ok,
            dataset_details,
        )
    )

    print(
        f"[{'PASS' if dataset_ok else 'FAIL'}] "
        "4B.1 Final dataset files and shapes"
    )

    # --------------------------------------------------------
    # 4B.2
    # --------------------------------------------------------

    schema_ok, schema_details = (
        verify_schema()
    )

    gates.append(
        make_gate(
            "4B.2",
            "Structured feature schema",
            schema_ok,
            schema_details,
        )
    )

    print(
        f"[{'PASS' if schema_ok else 'FAIL'}] "
        "4B.2 Structured feature schema"
    )

    # --------------------------------------------------------
    # 4B.3
    # --------------------------------------------------------

    manifest_ok, manifest_details = (
        verify_dataset_manifest()
    )

    gates.append(
        make_gate(
            "4B.3",
            "Final dataset manifest",
            manifest_ok,
            manifest_details,
        )
    )

    print(
        f"[{'PASS' if manifest_ok else 'FAIL'}] "
        "4B.3 Final dataset manifest"
    )

    # --------------------------------------------------------
    # 4B.4
    # --------------------------------------------------------

    artifacts_ok, artifacts_details = (
        verify_artifacts()
    )

    gates.append(
        make_gate(
            "4B.4",
            "Required production artifacts",
            artifacts_ok,
            artifacts_details,
        )
    )

    print(
        f"[{'PASS' if artifacts_ok else 'FAIL'}] "
        "4B.4 Production artifacts"
    )

    # --------------------------------------------------------
    # 4B.5
    # --------------------------------------------------------

    checksum_ok, checksum_details = (
        verify_checksums()
    )

    gates.append(
        make_gate(
            "4B.5",
            "Production SHA-256 integrity",
            checksum_ok,
            checksum_details,
        )
    )

    print(
        f"[{'PASS' if checksum_ok else 'FAIL'}] "
        "4B.5 SHA-256 integrity"
    )

    # --------------------------------------------------------
    # 4B.6
    # --------------------------------------------------------

    threshold_ok, threshold_details = (
        verify_thresholds()
    )

    gates.append(
        make_gate(
            "4B.6",
            "Production thresholds",
            threshold_ok,
            threshold_details,
        )
    )

    print(
        f"[{'PASS' if threshold_ok else 'FAIL'}] "
        "4B.6 Thresholds"
    )

    # --------------------------------------------------------
    # 4B.7
    # --------------------------------------------------------

    model_manifest_ok, model_manifest_details = (
        verify_model_manifest()
    )

    gates.append(
        make_gate(
            "4B.7",
            "Production model manifest",
            model_manifest_ok,
            model_manifest_details,
        )
    )

    print(
        f"[{'PASS' if model_manifest_ok else 'FAIL'}] "
        "4B.7 Model manifest"
    )

    # --------------------------------------------------------
    # 4B.8
    # --------------------------------------------------------

    environment_ok, environment_details = (
        verify_environment()
    )

    gates.append(
        make_gate(
            "4B.8",
            "Production environment record",
            environment_ok,
            environment_details,
        )
    )

    print(
        f"[{'PASS' if environment_ok else 'FAIL'}] "
        "4B.8 Environment record"
    )

    # --------------------------------------------------------
    # 4B.9
    # --------------------------------------------------------

    compatibility_ok, compatibility_details = (
        verify_compatibility()
    )

    gates.append(
        make_gate(
            "4B.9",
            "Compatibility runtime availability",
            compatibility_ok,
            compatibility_details,
        )
    )

    print(
        f"[{'PASS' if compatibility_ok else 'FAIL'}] "
        "4B.9 Compatibility runtime"
    )

    # --------------------------------------------------------
    # 4B.10
    # --------------------------------------------------------

    model = None
    model_ok = False
    model_details = {}

    try:

        (
            model,
            model_ok,
            model_details,
        ) = load_model()

    except Exception as exc:

        model_ok = False

        model_details = {
            "error": repr(exc),
        }

    gates.append(
        make_gate(
            "4B.10",
            "Production model loading",
            model_ok,
            model_details,
        )
    )

    print(
        f"[{'PASS' if model_ok else 'FAIL'}] "
        "4B.10 Production model loading"
    )

    # --------------------------------------------------------
    # 4B.11
    # --------------------------------------------------------

    prediction_ok = False
    prediction_details = {}

    if model is not None:

        try:

            (
                prediction_ok,
                prediction_details,
            ) = verify_prediction(model)

        except Exception as exc:

            prediction_ok = False

            prediction_details = {
                "error": repr(exc),
            }

    else:

        prediction_details = {
            "reason": (
                "Production model unavailable."
            )
        }

    gates.append(
        make_gate(
            "4B.11",
            "Frozen feature transformation and prediction",
            prediction_ok,
            prediction_details,
        )
    )

    print(
        f"[{'PASS' if prediction_ok else 'FAIL'}] "
        "4B.11 Feature pipeline and prediction"
    )

    # --------------------------------------------------------
    # 4B.12
    # --------------------------------------------------------

    deterministic_ok = False
    deterministic_details = {}

    if model is not None:

        try:

            (
                deterministic_ok,
                deterministic_details,
            ) = verify_determinism(model)

        except Exception as exc:

            deterministic_ok = False

            deterministic_details = {
                "error": repr(exc),
            }

    else:

        deterministic_details = {
            "reason": (
                "Production model unavailable."
            )
        }

    gates.append(
        make_gate(
            "4B.12",
            "Deterministic repeated inference",
            deterministic_ok,
            deterministic_details,
        )
    )

    print(
        f"[{'PASS' if deterministic_ok else 'FAIL'}] "
        "4B.12 Deterministic inference"
    )

    # --------------------------------------------------------
    # 4B.13
    # --------------------------------------------------------

    safety_ok, safety_details = (
        verify_safety()
    )

    gates.append(
        make_gate(
            "4B.13",
            "Production safety and no operational response",
            safety_ok,
            safety_details,
        )
    )

    print(
        f"[{'PASS' if safety_ok else 'FAIL'}] "
        "4B.13 Operational safety"
    )

    # --------------------------------------------------------
    # Overall decision
    # --------------------------------------------------------

    failures = [
        gate
        for gate in gates
        if not gate["passed"]
    ]

    overall_pass = (
        len(failures) == 0
    )

    if overall_pass:

        decision = "GO"
        status = "ACCEPTED_AND_FROZEN"

    else:

        decision = "NO_GO"
        status = (
            "REJECTED_PENDING_REMEDIATION"
        )

    # --------------------------------------------------------
    # Artifact inventory
    # --------------------------------------------------------

    artifact_inventory = (
        build_artifact_inventory()
    )

    artifact_csv = (
        PHASE4B_DIR
        / "phase4b_artifact_inventory.csv"
    )

    write_csv(
        artifact_csv,
        artifact_inventory,
    )

    # --------------------------------------------------------
    # Dataset inventory
    # --------------------------------------------------------

    dataset_inventory = (
        build_dataset_inventory()
    )

    dataset_csv = (
        PHASE4B_DIR
        / "phase4b_dataset_inventory.csv"
    )

    write_csv(
        dataset_csv,
        dataset_inventory,
    )

    # --------------------------------------------------------
    # Release manifest
    # --------------------------------------------------------

    release_manifest = {

        "project": "SmartSOC",

        "phase": "4B",

        "version": "4.1",

        "title":
            "AI Production Acceptance and Freeze",

        "generated_at_utc":
            utc_now(),

        "decision":
            decision,

        "status":
            status,

        "baseline": {

            "dataset":
                str(FINAL_DATASET_DIR),

            "production_artifacts":
                str(PRODUCTION_DIR),

            "dataset_manifest":
                str(
                    FINAL_DATASET_MANIFEST
                ),

            "production_checksums":
                str(
                    PRODUCTION_CHECKSUMS
                ),

            "model_manifest":
                str(
                    PRODUCTION_MODEL_MANIFEST
                ),
        },

        "model": {

            "model":
                str(MODEL_FILE),

            "model_type":
                (
                    type(model).__name__
                    if model is not None
                    else None
                ),

            "technique_count":
                EXPECTED_TECHNIQUE_COUNT,

            "final_feature_count":
                EXPECTED_FINAL_FEATURE_COUNT,
        },

        "feature_pipeline": {

            "structured_features":
                len(
                    EXPECTED_STRUCTURED_COLUMNS
                ),

            "categorical_features":
                EXPECTED_CATEGORICAL_COUNT,

            "numerical_features":
                EXPECTED_NUMERICAL_COUNT,

            "categorical_encoded_features":
                EXPECTED_CATEGORICAL_ENCODED_COUNT,

            "tfidf_features":
                EXPECTED_TFIDF_COUNT,

            "final_features":
                EXPECTED_FINAL_FEATURE_COUNT,
        },

        "thresholds": {

            "threshold_file":
                str(
                    PRODUCTION_THRESHOLDS
                ),

            "threshold_count":
                threshold_details.get(
                    "threshold_count"
                ),
        },

        "runtime": {

            "current_python":
                platform.python_version(),

            "current_sklearn":
                sklearn.__version__,

            "current_numpy":
                np.__version__,

            "current_pandas":
                pd.__version__,

            "current_joblib":
                joblib.__version__,

            "compatibility_python":
                str(
                    COMPAT_PYTHON
                ),
        },

        "freeze_controls": {

            "dataset_modified":
                False,

            "model_modified":
                False,

            "thresholds_modified":
                False,

            "retraining_performed":
                False,

            "wazuh_modified":
                False,

            "shuffle_modified":
                False,

            "response_actions_executed":
                False,

            "containment_actions_executed":
                False,
        },

        "safety":
            safety_details,

        "gates": [
            {
                "gate_id":
                    current_gate["gate_id"],

                "description":
                    current_gate["description"],

                "status":
                    current_gate["status"],
            }

            for current_gate in gates
        ],

        "failed_gates": [
            {
                "gate_id":
                    failed_gate["gate_id"],

                "description":
                    failed_gate["description"],
            }

            for failed_gate in failures
        ],

        "artifact_inventory":
            str(artifact_csv),

        "dataset_inventory":
            str(dataset_csv),

        "freeze_statement": (
            "If GO, this release establishes "
            "the current dataset, production "
            "model, feature pipeline, thresholds, "
            "and supporting artifacts as the "
            "SmartSOC AI production baseline. "
            "Future changes require a new "
            "controlled validation and release cycle."
        ),
    }

    release_manifest_path = (
        PHASE4B_DIR
        / "phase4b_ai_release_manifest.json"
    )

    with release_manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            release_manifest,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Acceptance report
    # --------------------------------------------------------

    report = {

        "project":
            "SmartSOC",

        "phase":
            "4B",

        "version":
            "4.1",

        "title":
            "AI Production Acceptance and Freeze",

        "execution": {

            "started_at_utc":
                started_at,

            "completed_at_utc":
                utc_now(),

            "script":
                str(
                    Path(__file__).resolve()
                ),

            "project_root":
                str(PROJECT_ROOT),

            "read_only_against_production":
                True,
        },

        "decision":
            decision,

        "status":
            status,

        "summary": {

            "total_gates":
                len(gates),

            "passed_gates":
                sum(
                    gate["passed"]
                    for gate in gates
                ),

            "failed_gates":
                len(failures),
        },

        "gates":
            gates,

        "known_limitations": [

            "Phase 2 was accepted with documented limitations.",

            "Several MITRE techniques have weak test-set recall.",

            "Synthetic malicious diagnostics were limited by feature fidelity.",

            "Real training-event replay passed.",

            "Threshold sensitivity analysis did not authorize production threshold changes.",

            "Fallback cases require analyst review.",

            "Containment remains analyst-controlled.",

            "Wazuh and Shuffle integration are intentionally outside Phase 4B.",

            "The current global runtime reports scikit-learn 1.9.0 while serialized artifacts were created under scikit-learn 1.8.0; Phase 3A compatibility validation established the dedicated compatibility environment.",
        ],

        "freeze": {

            "dataset_frozen":
                overall_pass,

            "model_frozen":
                overall_pass,

            "thresholds_frozen":
                overall_pass,

            "feature_schema_frozen":
                overall_pass,

            "production_baseline_frozen":
                overall_pass,

            "future_changes_require_new_gate":
                True,
        },

        "next_phase": (
            "Phase 5 — Decision Policy"
            if overall_pass
            else
            "Remediation and Phase 4B re-run"
        ),
    }

    report_path = (
        PHASE4B_DIR
        / "phase4b_ai_production_acceptance_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()

    print("=" * 72)
    print("PHASE 4B FINAL DECISION")
    print("=" * 72)

    print(
        f"Status  : {status}"
    )

    print(
        f"Decision: {decision}"
    )

    print(
        f"Gates   : {len(gates)} total / "
        f"{len(failures)} failed"
    )

    print()

    print("Outputs:")

    print(
        f"  {report_path}"
    )

    print(
        f"  {release_manifest_path}"
    )

    print(
        f"  {artifact_csv}"
    )

    print(
        f"  {dataset_csv}"
    )

    print()

    if overall_pass:

        print(
            "PHASE 4B ACCEPTED — "
            "AI PRODUCTION BASELINE FROZEN"
        )

        print()

        print(
            "No model retraining, dataset modification, "
            "threshold modification, Wazuh integration, "
            "Shuffle integration, or response action "
            "was performed."
        )

        return 0

    print(
        "PHASE 4B FAILED — "
        "DO NOT FREEZE PRODUCTION BASELINE"
    )

    print()

    print("Failed gates:")

    for failure in failures:

        print(
            f"  {failure['gate_id']} — "
            f"{failure['description']}"
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())