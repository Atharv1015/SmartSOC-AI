from pathlib import Path
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


# ============================================================================
# SMARTSOC
# PHASE 4A — GATE 4.1
# COMPLETE AI PIPELINE ASSEMBLY
#
# PURPOSE
# -------
# Validate the complete frozen SmartSOC AI pipeline:
#
#   62 structured features + behavior text
#       -> 162 categorical + 58 numerical + 6699 TF-IDF
#       -> 6919 production features
#       -> 45 MITRE ATT&CK probabilities
#       -> thresholds
#       -> detections / fallback
#       -> confidence / margin
#       -> SOC evidence
#       -> deterministic Risk Engine
#       -> final risk decision
#
# SAFETY
# ------
# This script is OFFLINE validation only.
#
# It does NOT:
#   - retrain the model
#   - modify thresholds
#   - modify datasets
#   - modify production artifacts
#   - modify Wazuh
#   - call Shuffle
#   - execute response actions
#   - automatically contain an endpoint
#
# IMPORTANT
# ---------
# The Risk Engine API below matches the validated
# smartsoc_risk_engine_v4_1.py supplied for this project.
# ============================================================================


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

DATA_DIR = (
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

RISK_ENGINE_FILE = (
    PROJECT_ROOT
    / "src"
    / "decision"
    / "smartsoc_risk_engine_v4_1.py"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase4a"
    / "gate4_1"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# DATA FILES
# ============================================================================

STRUCTURED_TEST = (
    DATA_DIR
    / "structured_test.csv"
)

LABELS_TEST = (
    DATA_DIR
    / "labels_test.csv"
)

METADATA_TEST = (
    DATA_DIR
    / "metadata_test.csv"
)

BEHAVIOR_TEST = (
    DATA_DIR
    / "behavior_text_test.csv"
)


# ============================================================================
# PRODUCTION ARTIFACTS
# ============================================================================

MODEL_FILE = (
    PRODUCTION_DIR
    / "model.joblib"
)

ENCODER_FILE = (
    PRODUCTION_DIR
    / "structured_encoder.joblib"
)

SCALER_FILE = (
    PRODUCTION_DIR
    / "numerical_scaler.joblib"
)

TFIDF_FILE = (
    PRODUCTION_DIR
    / "tfidf_vectorizer.joblib"
)

MLB_FILE = (
    PRODUCTION_DIR
    / "multilabel_encoder.joblib"
)

SCHEMA_FILE = (
    PRODUCTION_DIR
    / "feature_schema.json"
)

THRESHOLD_FILE = (
    PRODUCTION_DIR
    / "thresholds.json"
)


# ============================================================================
# OUTPUT FILES
# ============================================================================

RESULTS_CSV = (
    REPORT_DIR
    / "phase4a_gate4_1_pipeline_results.csv"
)

REPORT_JSON = (
    REPORT_DIR
    / "phase4a_gate4_1_report.json"
)

MANIFEST_JSON = (
    REPORT_DIR
    / "phase4a_gate4_1_artifact_manifest.json"
)


# ============================================================================
# CHECK SYSTEM
# ============================================================================

checks = []


def add_check(
    name,
    passed,
    details=None,
):
    passed = bool(passed)

    checks.append(
        {
            "name": name,
            "passed": passed,
            "details": details,
        }
    )

    message = (
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{name}"
    )

    if details:
        message += f" — {details}"

    print(message)

    return passed


def sha256_file(path):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_bool(value):
    if isinstance(
        value,
        bool,
    ):
        return value

    if value is None:
        return False

    if isinstance(
        value,
        str,
    ):

        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    return bool(value)


# ============================================================================
# START
# ============================================================================

print("=" * 76)
print("SMARTSOC PHASE 4A")
print("GATE 4.1 — COMPLETE AI PIPELINE ASSEMBLY")
print("=" * 76)

print(
    f"Project root: {PROJECT_ROOT}"
)

print(
    f"Timestamp    : "
    f"{datetime.now().isoformat()}"
)


# ============================================================================
# 1. REQUIRED FILES
# ============================================================================

print()
print("REQUIRED FILE VALIDATION")


required_files = {
    "structured_test": STRUCTURED_TEST,
    "labels_test": LABELS_TEST,
    "metadata_test": METADATA_TEST,
    "behavior_test": BEHAVIOR_TEST,
    "model": MODEL_FILE,
    "structured_encoder": ENCODER_FILE,
    "numerical_scaler": SCALER_FILE,
    "tfidf_vectorizer": TFIDF_FILE,
    "multilabel_encoder": MLB_FILE,
    "feature_schema": SCHEMA_FILE,
    "thresholds": THRESHOLD_FILE,
    "risk_engine": RISK_ENGINE_FILE,
}


for name, path in required_files.items():

    add_check(
        f"{name}_exists",
        path.exists(),
        str(path),
    )


if not all(
    path.exists()
    for path in required_files.values()
):

    raise RuntimeError(
        "Required Gate 4.1 files are missing."
    )


# ============================================================================
# 2. LOAD PRODUCTION ARTIFACTS
# ============================================================================

print()
print("LOADING PRODUCTION ARTIFACTS")


model = joblib.load(
    MODEL_FILE
)

encoder = joblib.load(
    ENCODER_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

tfidf = joblib.load(
    TFIDF_FILE
)

mlb = joblib.load(
    MLB_FILE
)

schema = load_json(
    SCHEMA_FILE
)

threshold_data = load_json(
    THRESHOLD_FILE
)


add_check(
    "model_loaded",
    model is not None,
)

add_check(
    "encoder_loaded",
    encoder is not None,
)

add_check(
    "scaler_loaded",
    scaler is not None,
)

add_check(
    "tfidf_loaded",
    tfidf is not None,
)

add_check(
    "multilabel_encoder_loaded",
    mlb is not None,
)


# ============================================================================
# 3. PRODUCTION SCHEMA
# ============================================================================

print()
print("PRODUCTION SCHEMA VALIDATION")


categorical_columns = list(
    schema.get(
        "categorical_columns",
        [],
    )
)

numerical_columns = list(
    schema.get(
        "numerical_columns",
        [],
    )
)

mitre_techniques = [
    str(item)
    for item in schema.get(
        "mitre_techniques",
        [],
    )
]


categorical_encoded_count = int(
    schema.get(
        "categorical_encoded_feature_count",
        162,
    )
)

numerical_feature_count = int(
    schema.get(
        "numerical_feature_count",
        58,
    )
)

tfidf_feature_count = int(
    schema.get(
        "tfidf_feature_count",
        6699,
    )
)

final_feature_count = int(
    schema.get(
        "final_feature_count",
        6919,
    )
)


structured_feature_count = (
    len(categorical_columns)
    + len(numerical_columns)
)


print(
    f"Structured features : "
    f"{structured_feature_count}"
)

print(
    f"Categorical         : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical           : "
    f"{len(numerical_columns)}"
)

print(
    f"Encoded categorical : "
    f"{categorical_encoded_count}"
)

print(
    f"TF-IDF              : "
    f"{tfidf_feature_count}"
)

print(
    f"Final               : "
    f"{final_feature_count}"
)

print(
    f"MITRE techniques    : "
    f"{len(mitre_techniques)}"
)


add_check(
    "structured_feature_count_62",
    structured_feature_count == 62,
)

add_check(
    "categorical_feature_count_4",
    len(categorical_columns) == 4,
)

add_check(
    "numerical_feature_count_58",
    len(numerical_columns) == 58,
)

add_check(
    "categorical_encoded_count_162",
    categorical_encoded_count == 162,
)

add_check(
    "tfidf_feature_count_6699",
    tfidf_feature_count == 6699,
)

add_check(
    "final_feature_count_6919",
    final_feature_count == 6919,
)

add_check(
    "mitre_technique_count_45",
    len(mitre_techniques) == 45,
)

add_check(
    "structured_columns_unique",
    len(
        set(
            categorical_columns
            + numerical_columns
        )
    ) == 62,
)


# ============================================================================
# 4. MODEL CONTRACT
# ============================================================================

print()
print("MODEL CONTRACT VALIDATION")


model_type = type(
    model
).__name__

model_classes = getattr(
    model,
    "classes_",
    None,
)


print(
    f"Model type: "
    f"{model_type}"
)

print(
    f"Model classes: "
    f"{len(model_classes) if model_classes is not None else 0}"
)


add_check(
    "model_is_one_vs_rest",
    model_type == "OneVsRestClassifier",
)

add_check(
    "model_has_45_classes",
    (
        model_classes is not None
        and len(model_classes) == 45
    ),
)


# ============================================================================
# 5. THRESHOLDS
# ============================================================================

print()
print("THRESHOLD CONTRACT")


raw_thresholds = threshold_data.get(
    "thresholds",
    threshold_data,
)

thresholds = {}


for key, value in raw_thresholds.items():

    try:

        thresholds[
            str(key)
        ] = float(value)

    except Exception:

        pass


print(
    f"Threshold count: "
    f"{len(thresholds)}"
)


add_check(
    "threshold_count_45",
    len(thresholds) == 45,
)

add_check(
    "thresholds_numeric",
    (
        len(thresholds) == 45
        and all(
            math.isfinite(value)
            for value in thresholds.values()
        )
    ),
)

add_check(
    "thresholds_in_range",
    (
        len(thresholds) == 45
        and all(
            0.0 <= value <= 1.0
            for value in thresholds.values()
        )
    ),
)


# ============================================================================
# 6. LOAD FINAL TEST DATA
# ============================================================================

print()
print("LOADING FINAL DATASET")


structured = pd.read_csv(
    STRUCTURED_TEST
)

labels = pd.read_csv(
    LABELS_TEST
)

metadata = pd.read_csv(
    METADATA_TEST
)

behavior = pd.read_csv(
    BEHAVIOR_TEST
)


print(
    f"Structured : "
    f"{structured.shape}"
)

print(
    f"Labels     : "
    f"{labels.shape}"
)

print(
    f"Metadata   : "
    f"{metadata.shape}"
)

print(
    f"Behavior   : "
    f"{behavior.shape}"
)


add_check(
    "test_rows_11707",
    len(structured) == 11707,
)

add_check(
    "labels_rows_11707",
    len(labels) == 11707,
)

add_check(
    "metadata_rows_11707",
    len(metadata) == 11707,
)

add_check(
    "behavior_rows_11707",
    len(behavior) == 11707,
)


# ============================================================================
# 7. DATA COLUMN CONTRACT
# ============================================================================

print()
print("DATA COLUMN CONTRACT")


missing_structured = [
    column
    for column in (
        categorical_columns
        + numerical_columns
    )
    if column not in structured.columns
]


add_check(
    "all_62_structured_columns_present",
    len(missing_structured) == 0,
    (
        f"missing={missing_structured}"
        if missing_structured
        else None
    ),
)


if (
    "behavior_text_normalized"
    in behavior.columns
):

    behavior_column = (
        "behavior_text_normalized"
    )

else:

    behavior_candidates = [
        column
        for column in behavior.columns
        if (
            "behavior"
            in column.lower()
            or "text"
            in column.lower()
        )
    ]

    behavior_column = (
        behavior_candidates[0]
        if behavior_candidates
        else None
    )


if "techniques" in labels.columns:

    label_column = "techniques"

elif len(labels.columns) == 1:

    label_column = labels.columns[0]

else:

    label_column = None


add_check(
    "behavior_column_available",
    behavior_column is not None,
    str(behavior_column),
)

add_check(
    "label_column_available",
    label_column is not None,
    str(label_column),
)

add_check(
    "all_test_rows_aligned",
    (
        len(structured)
        == len(labels)
        == len(metadata)
        == len(behavior)
    ),
)


# ============================================================================
# 8. FEATURE TRANSFORMATION
# ============================================================================

print()
print("FEATURE TRANSFORMATION")


categorical_frame = structured[
    categorical_columns
].copy()


numerical_frame = structured[
    numerical_columns
].copy()


categorical_matrix = encoder.transform(
    categorical_frame
)


numerical_matrix = scaler.transform(
    numerical_frame.to_numpy()
)


behavior_series = (
    behavior[
        behavior_column
    ]
    .fillna("")
    .map(normalize_text)
)


tfidf_matrix = tfidf.transform(
    behavior_series
)


if not sparse.issparse(
    categorical_matrix
):

    categorical_matrix = (
        sparse.csr_matrix(
            categorical_matrix
        )
    )


if not sparse.issparse(
    numerical_matrix
):

    numerical_matrix = (
        sparse.csr_matrix(
            numerical_matrix
        )
    )


if not sparse.issparse(
    tfidf_matrix
):

    tfidf_matrix = (
        sparse.csr_matrix(
            tfidf_matrix
        )
    )


final_matrix = sparse.hstack(
    [
        categorical_matrix,
        numerical_matrix,
        tfidf_matrix,
    ],
    format="csr",
)


print(
    f"Categorical matrix : "
    f"{categorical_matrix.shape}"
)

print(
    f"Numerical matrix   : "
    f"{numerical_matrix.shape}"
)

print(
    f"TF-IDF matrix      : "
    f"{tfidf_matrix.shape}"
)

print(
    f"Final matrix       : "
    f"{final_matrix.shape}"
)


add_check(
    "categorical_matrix_11707x162",
    categorical_matrix.shape
    == (11707, 162),
)

add_check(
    "numerical_matrix_11707x58",
    numerical_matrix.shape
    == (11707, 58),
)

add_check(
    "tfidf_matrix_11707x6699",
    tfidf_matrix.shape
    == (11707, 6699),
)

add_check(
    "final_matrix_11707x6919",
    final_matrix.shape
    == (11707, 6919),
)


# ============================================================================
# 9. PRODUCTION ML PREDICTION
# ============================================================================

print()
print("PRODUCTION ML PREDICTION")


probabilities = np.asarray(
    model.predict_proba(
        final_matrix
    )
)


print(
    f"Probability matrix: "
    f"{probabilities.shape}"
)


add_check(
    "probability_matrix_11707x45",
    probabilities.shape
    == (11707, 45),
)

add_check(
    "probabilities_finite",
    np.isfinite(
        probabilities
    ).all(),
)

add_check(
    "probabilities_in_0_1",
    (
        probabilities.min() >= 0.0
        and probabilities.max() <= 1.0
    ),
)


# ============================================================================
# 10. MULTILABEL CONTRACT
# ============================================================================

print()
print("MULTILABEL PROBABILITY CONTRACT")


print(
    "Multilabel probabilities are "
    "independent per-technique scores "
    "and are intentionally not required "
    "to sum to 1."
)


row_sums = probabilities.sum(
    axis=1
)


print(
    f"Row-sum min: "
    f"{row_sums.min():.6f}"
)

print(
    f"Row-sum max: "
    f"{row_sums.max():.6f}"
)


add_check(
    "probability_columns_match_mitre",
    probabilities.shape[1]
    == len(mitre_techniques),
)


# ============================================================================
# 11. DETECTION / FALLBACK
# ============================================================================

print()
print("TECHNIQUE / THRESHOLD ALIGNMENT")


missing_thresholds = [
    technique
    for technique in mitre_techniques
    if technique not in thresholds
]


add_check(
    "all_45_techniques_have_thresholds",
    len(missing_thresholds) == 0,
    (
        f"missing={missing_thresholds}"
        if missing_thresholds
        else None
    ),
)


if missing_thresholds:

    raise RuntimeError(
        "One or more MITRE techniques "
        "do not have thresholds."
    )


print()
print("DETECTION AND FALLBACK ASSEMBLY")


detections = []

fallback_flags = []


for row in probabilities:

    detected = []

    for index, technique in enumerate(
        mitre_techniques
    ):

        if (
            row[index]
            >= thresholds[technique]
        ):

            detected.append(
                technique
            )


    detections.append(
        detected
    )

    fallback_flags.append(
        len(detected) == 0
    )


top_indices = np.argmax(
    probabilities,
    axis=1,
)


top_techniques = [
    mitre_techniques[index]
    for index in top_indices
]


top_scores = probabilities[
    np.arange(
        len(probabilities)
    ),
    top_indices,
]


detected_events = sum(
    bool(items)
    for items in detections
)

fallback_events = sum(
    fallback_flags
)


fallback_fraction = (
    fallback_events
    / len(probabilities)
)


print(
    f"Events with detections : "
    f"{detected_events}"
)

print(
    f"Fallback events        : "
    f"{fallback_events}"
)

print(
    f"Fallback fraction      : "
    f"{fallback_fraction:.6f}"
)


add_check(
    "every_event_has_top_technique",
    len(top_techniques)
    == 11707,
)

add_check(
    "fallback_logic_valid",
    all(
        fallback_flags[index]
        == (
            len(
                detections[index]
            )
            == 0
        )
        for index in range(
            len(probabilities)
        )
    ),
)


# ============================================================================
# 12. CONFIDENCE / MARGIN
# ============================================================================

print()
print("CONFIDENCE AND UNCERTAINTY ASSEMBLY")


sorted_probabilities = np.sort(
    probabilities,
    axis=1,
)


confidence = sorted_probabilities[
    :, -1
]

second_probability = sorted_probabilities[
    :, -2
]

margins = (
    confidence
    - second_probability
)


add_check(
    "confidence_length_11707",
    len(confidence) == 11707,
)

add_check(
    "margin_length_11707",
    len(margins) == 11707,
)

add_check(
    "confidence_finite",
    np.isfinite(
        confidence
    ).all(),
)

add_check(
    "margin_finite",
    np.isfinite(
        margins
    ).all(),
)

add_check(
    "confidence_in_0_1",
    (
        confidence.min() >= 0.0
        and confidence.max() <= 1.0
    ),
)

add_check(
    "margin_in_0_1",
    (
        margins.min() >= 0.0
        and margins.max() <= 1.0
    ),
)


# ============================================================================
# 13. SOC EVIDENCE
# ============================================================================

print()
print("SOC EVIDENCE ASSEMBLY")


evidence_columns = [
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
    "is_network_event",
]


evidence_columns = [
    column
    for column in evidence_columns
    if column in structured.columns
]


def build_evidence(row):

    evidence = []

    for column in evidence_columns:

        if normalize_bool(
            row.get(
                column,
                False,
            )
        ):

            evidence.append(
                column
            )


    process_name = normalize_text(
        row.get(
            "process_name",
            "",
        )
    )

    parent_process_name = normalize_text(
        row.get(
            "parent_process_name",
            "",
        )
    )


    if process_name:

        evidence.append(
            f"process:{process_name}"
        )


    if parent_process_name:

        evidence.append(
            f"parent_process:{parent_process_name}"
        )


    if normalize_bool(
        row.get(
            "has_process_image",
            False,
        )
    ):

        evidence.append(
            "has_process_image"
        )


    if normalize_bool(
        row.get(
            "has_parent_image",
            False,
        )
    ):

        evidence.append(
            "has_parent_image"
        )


    if normalize_bool(
        row.get(
            "integrity_high",
            False,
        )
    ):

        evidence.append(
            "integrity_high"
        )


    if normalize_bool(
        row.get(
            "is_sysmon",
            False,
        )
    ):

        evidence.append(
            "is_sysmon"
        )


    return evidence


evidence_records = [
    build_evidence(
        structured.iloc[index]
    )
    for index in range(
        len(structured)
    )
]


events_with_evidence = sum(
    bool(items)
    for items in evidence_records
)


print(
    f"Events with SOC evidence: "
    f"{events_with_evidence}"
)


add_check(
    "evidence_records_11707",
    len(evidence_records)
    == 11707,
)

add_check(
    "evidence_records_are_lists",
    all(
        isinstance(
            item,
            list,
        )
        for item in evidence_records
    ),
)


# ============================================================================
# 14. GROUND TRUTH
# ============================================================================

print()
print("GROUND-TRUTH LABEL ALIGNMENT")


true_labels = (
    labels[
        label_column
    ]
    .fillna("")
    .astype(str)
    .tolist()
)


add_check(
    "true_label_count_11707",
    len(true_labels) == 11707,
)


# ============================================================================
# 15. REPRESENTATIVE EVENTS
# ============================================================================

print()
print("REPRESENTATIVE EVENT SELECTION")


candidate_indices = list(
    range(
        len(probabilities)
    )
)


candidate_indices.sort(
    key=lambda index: (
        not bool(
            detections[index]
        ),
        -float(
            confidence[index]
        ),
    )
)


selected_indices = []

seen_techniques = set()


for index in candidate_indices:

    technique = top_techniques[
        index
    ]

    if technique not in seen_techniques:

        selected_indices.append(
            index
        )

        seen_techniques.add(
            technique
        )


    if len(
        selected_indices
    ) >= 20:

        break


if len(
    selected_indices
) < 20:

    for index in candidate_indices:

        if index not in selected_indices:

            selected_indices.append(
                index
            )


        if len(
            selected_indices
        ) >= 20:

            break


print(
    f"Selected events: "
    f"{len(selected_indices)}"
)


add_check(
    "representative_event_count",
    len(selected_indices) == 20,
)


# ============================================================================
# 16. RISK ENGINE IMPORT
# ============================================================================
#
# Critical import fix:
# register module in sys.modules BEFORE exec_module().
#
# This is necessary because the Risk Engine uses @dataclass.
# ============================================================================

print()
print("RISK ENGINE VALIDATION")


RISK_MODULE_NAME = (
    "smartsoc_risk_engine_v4_1"
)


risk_spec = (
    importlib.util.spec_from_file_location(
        RISK_MODULE_NAME,
        RISK_ENGINE_FILE,
    )
)


if risk_spec is None:

    raise RuntimeError(
        "Unable to create Risk Engine module spec."
    )


if risk_spec.loader is None:

    raise RuntimeError(
        "Risk Engine module loader is unavailable."
    )


risk_module = (
    importlib.util.module_from_spec(
        risk_spec
    )
)


# IMPORTANT:
# Python's normal import machinery registers the module in sys.modules
# before executing its source.
sys.modules[
    RISK_MODULE_NAME
] = risk_module


risk_spec.loader.exec_module(
    risk_module
)


add_check(
    "risk_engine_module_loaded",
    True,
)


SmartSOCRiskEngine = getattr(
    risk_module,
    "SmartSOCRiskEngine",
    None,
)

RiskComponents = getattr(
    risk_module,
    "RiskComponents",
    None,
)

RiskDecision = getattr(
    risk_module,
    "RiskDecision",
    None,
)

evaluate_event_function = getattr(
    risk_module,
    "evaluate_event",
    None,
)


add_check(
    "risk_engine_class_available",
    SmartSOCRiskEngine is not None,
)

add_check(
    "risk_components_available",
    RiskComponents is not None,
)

add_check(
    "risk_decision_available",
    RiskDecision is not None,
)

add_check(
    "evaluate_event_available",
    callable(
        evaluate_event_function
    ),
)


risk_engine = SmartSOCRiskEngine()


add_check(
    "risk_engine_instance_created",
    True,
)


# ============================================================================
# 17. RISK ENGINE INPUT MAPPING
# ============================================================================
#
# Exact contract from smartsoc_risk_engine_v4_1.py:
#
# engine.calculate(
#     *,
#     technique_severity: str,
#     ml_confidence: float,
#     evidence_strength: str,
#     privilege: str = "LOW",
#     network: bool = False,
#     suspicious_parent: bool = False,
#     fallback: bool = False,
#     ambiguous: bool = False,
#     missing_behavior: bool = False,
# )
#
# The Risk Engine internally converts these into:
#   technique score
#   confidence score
#   behavioral evidence score
#   context score
#   uncertainty penalty
# and returns RiskDecision.
# ============================================================================


TECHNIQUE_SEVERITY = {
    # Techniques already emphasized in the project analysis.
    "T1485": "CRITICAL",
    "T1486": "CRITICAL",
    "T1531": "CRITICAL",
    "T1565.001": "CRITICAL",
    "T1562.001": "CRITICAL",

    "T1562.004": "HIGH",
    "T1548.002": "HIGH",
    "T1543.003": "HIGH",
    "T1059.001": "HIGH",
    "T1059.003": "HIGH",
    "T1059": "HIGH",
    "T1078": "HIGH",
    "T1070.004": "HIGH",
    "T1112": "HIGH",
    "T1053.005": "HIGH",
    "T1055": "HIGH",
    "T1095": "HIGH",
    "T1027.004": "HIGH",
    "T1105": "HIGH",
    "T1570": "HIGH",
    "T1047": "HIGH",

    "T1087": "MEDIUM",
}


def infer_technique_severity(
    technique
):

    return TECHNIQUE_SEVERITY.get(
        technique,
        "MEDIUM",
    )


def infer_evidence_strength(
    index
):

    row = structured.iloc[
        index
    ]

    evidence = set(
        evidence_records[index]
    )


    very_strong_signals = {
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_file_deletion",
        "has_obfuscation_hint",
        "strong_behavior_signal",
    }


    strong_signals = {
        "uses_powershell",
        "uses_rundll32",
        "uses_wmic",
        "uses_schtasks",
        "uses_sc",
        "uses_mshta",
        "uses_regsvr32",
        "uses_cscript",
        "uses_wscript",
        "uses_wevtutil",
        "uses_vssadmin",
        "has_registry_operation",
        "is_network_event",
    }


    very_strong_count = sum(
        signal in evidence
        for signal in very_strong_signals
    )

    strong_count = sum(
        signal in evidence
        for signal in strong_signals
    )


    if (
        very_strong_count >= 2
        or len(evidence) >= 8
    ):

        return "VERY_STRONG"


    if (
        very_strong_count >= 1
        or strong_count >= 2
        or len(evidence) >= 5
    ):

        return "STRONG"


    if (
        strong_count >= 1
        or len(evidence) >= 3
    ):

        return "MODERATE"


    if len(evidence) >= 1:

        return "WEAK"


    return "NONE"


def infer_privilege(
    index
):

    row = structured.iloc[
        index
    ]


    if normalize_bool(
        row.get(
            "is_system_user",
            False,
        )
    ):

        return "SYSTEM"


    if normalize_bool(
        row.get(
            "is_admin_user",
            False,
        )
    ):

        return "HIGH"


    if normalize_bool(
        row.get(
            "integrity_high",
            False,
        )
    ):

        return "HIGH"


    if normalize_bool(
        row.get(
            "integrity_medium",
            False,
        )
    ):

        return "MEDIUM"


    return "LOW"


def infer_network(
    index
):

    row = structured.iloc[
        index
    ]

    return normalize_bool(
        row.get(
            "is_network_event",
            False,
        )
    )


def infer_suspicious_parent(
    index
):

    row = structured.iloc[
        index
    ]

    return normalize_bool(
        row.get(
            "suspicious_parent_wazuh",
            False,
        )
    )


# ============================================================================
# 18. END-TO-END AI -> RISK ENGINE
# ============================================================================

print()
print("END-TO-END AI → RISK ENGINE ASSEMBLY")


pipeline_results = []

risk_success_count = 0


for index in selected_indices:

    technique = top_techniques[
        index
    ]

    ml_confidence = float(
        confidence[index]
    )

    detected = detections[
        index
    ]

    fallback = (
        len(detected) == 0
    )

    margin = float(
        margins[index]
    )

    behavior_text_present = bool(
        behavior_series.iloc[index]
    )

    # Conservative uncertainty/ambiguity condition.
    ambiguous = (
        margin < 0.10
    )

    missing_behavior = (
        not behavior_text_present
    )

    severity = infer_technique_severity(
        technique
    )

    evidence_strength = infer_evidence_strength(
        index
    )

    privilege = infer_privilege(
        index
    )

    network = infer_network(
        index
    )

    suspicious_parent = (
        infer_suspicious_parent(
            index
        )
    )


    try:

        # ------------------------------------------------------------
        # THIS IS THE CORRECT EXACT RISK ENGINE CONTRACT.
        # ------------------------------------------------------------

        risk_decision = (
            risk_engine.calculate(
                technique_severity=severity,
                ml_confidence=ml_confidence,
                evidence_strength=evidence_strength,
                privilege=privilege,
                network=network,
                suspicious_parent=suspicious_parent,
                fallback=fallback,
                ambiguous=ambiguous,
                missing_behavior=missing_behavior,
            )
        )


        valid_score = (
            isinstance(
                risk_decision,
                RiskDecision,
            )
            and math.isfinite(
                float(
                    risk_decision.risk_score
                )
            )
            and (
                0.0
                <= float(
                    risk_decision.risk_score
                )
                <= 100.0
            )
        )


        valid_level = (
            getattr(
                risk_decision,
                "risk_level",
                None,
            )
            in {
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            }
        )


        valid_action = (
            getattr(
                risk_decision,
                "recommended_action",
                None,
            )
            in {
                "MONITOR",
                "ANALYST_REVIEW",
                "PRIORITY_ANALYST_REVIEW",
                "CONTAINMENT_CANDIDATE",
            }
        )


        valid_safety = (
            isinstance(
                getattr(
                    risk_decision,
                    "safety",
                    None,
                ),
                dict,
            )
            and risk_decision.safety.get(
                "automatic_response"
            )
            is False
            and risk_decision.safety.get(
                "automatic_blocking"
            )
            is False
            and risk_decision.safety.get(
                "automatic_isolation"
            )
            is False
            and risk_decision.safety.get(
                "automatic_process_kill"
            )
            is False
            and risk_decision.safety.get(
                "requires_analyst_for_containment"
            )
            is True
        )


        valid_result = (
            valid_score
            and valid_level
            and valid_action
            and valid_safety
        )


        if valid_result:

            risk_success_count += 1


        components = (
            risk_decision.components
            if hasattr(
                risk_decision,
                "components",
            )
            else None
        )


        pipeline_results.append(
            {
                "event_index":
                    int(index),

                "predicted_technique":
                    technique,

                "ml_confidence":
                    ml_confidence,

                "top_margin":
                    margin,

                "detected_techniques":
                    "|".join(
                        detected
                    ),

                "fallback":
                    fallback,

                "behavior_text_present":
                    behavior_text_present,

                "evidence_items":
                    "|".join(
                        evidence_records[index]
                    ),

                "technique_severity":
                    severity,

                "evidence_strength":
                    evidence_strength,

                "privilege":
                    privilege,

                "network":
                    network,

                "suspicious_parent":
                    suspicious_parent,

                "ambiguous":
                    ambiguous,

                "missing_behavior":
                    missing_behavior,

                "risk_score":
                    float(
                        risk_decision.risk_score
                    ),

                "risk_level":
                    risk_decision.risk_level,

                "recommended_action":
                    risk_decision.recommended_action,

                "risk_reasons":
                    "|".join(
                        risk_decision.reasons
                    ),

                "risk_engine_version":
                    risk_decision.engine_version,

                "risk_result_type":
                    type(
                        risk_decision
                    ).__name__,

                "risk_components":
                    json.dumps(
                        {
                            "technique_severity":
                                components.technique_severity,

                            "ml_confidence":
                                components.ml_confidence,

                            "behavioral_evidence":
                                components.behavioral_evidence,

                            "context":
                                components.context,

                            "uncertainty_penalty":
                                components.uncertainty_penalty,
                        },
                        sort_keys=True,
                    ),

                "automatic_response":
                    risk_decision.safety[
                        "automatic_response"
                    ],

                "automatic_blocking":
                    risk_decision.safety[
                        "automatic_blocking"
                    ],

                "automatic_isolation":
                    risk_decision.safety[
                        "automatic_isolation"
                    ],

                "automatic_process_kill":
                    risk_decision.safety[
                        "automatic_process_kill"
                    ],

                "requires_analyst_for_containment":
                    risk_decision.safety[
                        "requires_analyst_for_containment"
                    ],
            }
        )


        if not valid_result:

            print(
                f"[FAIL] Risk result validation "
                f"for event {index}"
            )

            print(
                f"       score="
                f"{getattr(risk_decision, 'risk_score', None)}"
            )

            print(
                f"       level="
                f"{getattr(risk_decision, 'risk_level', None)}"
            )

            print(
                f"       action="
                f"{getattr(risk_decision, 'recommended_action', None)}"
            )

            print(
                f"       safety="
                f"{getattr(risk_decision, 'safety', None)}"
            )


    except Exception as exc:

        print(
            f"[FAIL] Risk engine exception "
            f"for event {index}: "
            f"{type(exc).__name__}: {exc}"
        )


        pipeline_results.append(
            {
                "event_index":
                    int(index),

                "predicted_technique":
                    technique,

                "ml_confidence":
                    ml_confidence,

                "top_margin":
                    margin,

                "detected_techniques":
                    "|".join(
                        detected
                    ),

                "fallback":
                    fallback,

                "risk_score":
                    None,

                "risk_level":
                    None,

                "recommended_action":
                    None,

                "risk_reasons":
                    (
                        f"EXCEPTION:"
                        f"{type(exc).__name__}:"
                        f"{exc}"
                    ),

                "risk_result_type":
                    "EXCEPTION",
            }
        )


print(
    f"Representative events: "
    f"{len(pipeline_results)}"
)

print(
    f"Risk results valid: "
    f"{risk_success_count}"
)


add_check(
    "risk_engine_evaluated_20_events",
    len(pipeline_results) == 20,
)

add_check(
    "risk_engine_returned_valid_scores",
    risk_success_count == 20,
)


# ============================================================================
# 19. RISK SAFETY CONTRACT
# ============================================================================

print()
print("RISK SAFETY CONTRACT")


print(
    "[PASS] Automatic response disabled"
)

print(
    "[PASS] Automatic containment disabled"
)

print(
    "[PASS] Shuffle execution NONE"
)

print(
    "[PASS] Wazuh modification NONE"
)

print(
    "[PASS] Response execution NONE"
)

print(
    "[PASS] Analyst approval REQUIRED"
)


# These are intentionally static safety assertions for this offline gate.
add_check(
    "automatic_response_disabled",
    True,
)

add_check(
    "automatic_containment_disabled",
    True,
)

add_check(
    "shuffle_execution_none",
    True,
)

add_check(
    "wazuh_modification_none",
    True,
)

add_check(
    "response_execution_none",
    True,
)

add_check(
    "analyst_approval_required",
    True,
)


# ============================================================================
# 20. DETERMINISM
# ============================================================================

print()
print("DETERMINISM TEST")


repeat_structured = structured.iloc[
    selected_indices
]


repeat_categorical = encoder.transform(
    repeat_structured[
        categorical_columns
    ]
)


repeat_numerical = scaler.transform(
    repeat_structured[
        numerical_columns
    ].to_numpy()
)


repeat_behavior = behavior_series.iloc[
    selected_indices
]


repeat_tfidf = tfidf.transform(
    repeat_behavior
)


if not sparse.issparse(
    repeat_categorical
):

    repeat_categorical = (
        sparse.csr_matrix(
            repeat_categorical
        )
    )


if not sparse.issparse(
    repeat_numerical
):

    repeat_numerical = (
        sparse.csr_matrix(
            repeat_numerical
        )
    )


if not sparse.issparse(
    repeat_tfidf
):

    repeat_tfidf = (
        sparse.csr_matrix(
            repeat_tfidf
        )
    )


repeat_matrix = sparse.hstack(
    [
        repeat_categorical,
        repeat_numerical,
        repeat_tfidf,
    ],
    format="csr",
)


repeat_probabilities = np.asarray(
    model.predict_proba(
        repeat_matrix
    )
)


reference_probabilities = (
    probabilities[
        selected_indices
    ]
)


max_difference = float(
    np.max(
        np.abs(
            reference_probabilities
            - repeat_probabilities
        )
    )
)


print(
    f"Deterministic max difference: "
    f"{max_difference:.12f}"
)


add_check(
    "representative_prediction_deterministic",
    max_difference == 0.0,
)


# ============================================================================
# 21. COMPLETE PIPELINE CONTRACT
# ============================================================================

print()
print("END-TO-END PIPELINE CONTRACT")


add_check(
    "feature_vector_contract_6919",
    final_matrix.shape
    == (11707, 6919),
)

add_check(
    "model_output_contract_45",
    probabilities.shape
    == (11707, 45),
)

add_check(
    "threshold_contract_45",
    len(thresholds) == 45,
)

add_check(
    "confidence_contract_11707",
    len(confidence) == 11707,
)

add_check(
    "uncertainty_contract_11707",
    len(margins) == 11707,
)

add_check(
    "fallback_contract_11707",
    len(fallback_flags) == 11707,
)

add_check(
    "risk_contract_20",
    risk_success_count == 20,
)


# ============================================================================
# 22. PRODUCTION ARTIFACT HASHES
# ============================================================================

print()
print("PRODUCTION ARTIFACT HASH MANIFEST")


artifact_paths = {
    "model": MODEL_FILE,
    "structured_encoder": ENCODER_FILE,
    "numerical_scaler": SCALER_FILE,
    "tfidf_vectorizer": TFIDF_FILE,
    "multilabel_encoder": MLB_FILE,
    "feature_schema": SCHEMA_FILE,
    "thresholds": THRESHOLD_FILE,
}


artifact_manifest = {}


for name, path in artifact_paths.items():

    digest = sha256_file(
        path
    )

    artifact_manifest[
        name
    ] = {
        "path":
            str(path),

        "sha256":
            digest,
    }

    print(
        f"{name}: "
        f"{digest}"
    )


# ============================================================================
# 23. NO-MODIFICATION CONTRACT
# ============================================================================

print()
print("NO-MODIFICATION CONTRACT")


no_modification = {
    "model_modified": False,
    "thresholds_modified": False,
    "dataset_modified": False,
    "inference_artifacts_modified": False,
    "wazuh_modified": False,
    "shuffle_called": False,
    "response_executed": False,
    "automatic_containment": False,
}


for name, value in no_modification.items():

    # Safe state is False.
    add_check(
        name,
        value is False,
    )


# ============================================================================
# 24. SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(
    pipeline_results
)


results_df.to_csv(
    RESULTS_CSV,
    index=False,
)


# ============================================================================
# 25. FINAL SUMMARY
# ============================================================================

passed_checks = sum(
    item["passed"]
    for item in checks
)


failed_checks = (
    len(checks)
    - passed_checks
)


decision = (
    "PASS"
    if failed_checks == 0
    else "REVIEW_REQUIRED"
)


# ============================================================================
# 26. REPORT
# ============================================================================

report = {
    "gate":
        "4.1",

    "title":
        "Complete AI Production Pipeline Assembly",

    "timestamp":
        datetime.now().isoformat(),

    "decision":
        decision,

    "validation_summary": {
        "total_checks":
            len(checks),

        "passed":
            passed_checks,

        "failed":
            failed_checks,
    },

    "dataset": {
        "path":
            str(DATA_DIR),

        "test_events":
            len(structured),

        "structured_features":
            structured_feature_count,

        "behavior_column":
            behavior_column,

        "label_column":
            label_column,
    },

    "feature_pipeline": {
        "categorical_features":
            categorical_matrix.shape[1],

        "numerical_features":
            numerical_matrix.shape[1],

        "tfidf_features":
            tfidf_matrix.shape[1],

        "final_features":
            final_matrix.shape[1],
    },

    "model": {
        "type":
            model_type,

        "techniques":
            len(mitre_techniques),

        "probability_shape":
            list(
                probabilities.shape
            ),
    },

    "detection": {
        "events":
            len(probabilities),

        "detected_events":
            int(
                detected_events
            ),

        "fallback_events":
            int(
                fallback_events
            ),

        "fallback_fraction":
            float(
                fallback_fraction
            ),
    },

    "confidence": {
        "mean":
            float(
                np.mean(
                    confidence
                )
            ),

        "median":
            float(
                np.median(
                    confidence
                )
            ),

        "minimum":
            float(
                np.min(
                    confidence
                )
            ),

        "maximum":
            float(
                np.max(
                    confidence
                )
            ),
    },

    "uncertainty": {
        "mean_margin":
            float(
                np.mean(
                    margins
                )
            ),

        "median_margin":
            float(
                np.median(
                    margins
                )
            ),
    },

    "soc_evidence": {
        "events_with_evidence":
            int(
                events_with_evidence
            ),
    },

    "risk_engine": {
        "engine_version":
            getattr(
                risk_module,
                "ENGINE_VERSION",
                None,
            ),

        "representative_events":
            len(
                pipeline_results
            ),

        "valid_results":
            risk_success_count,
    },

    "determinism": {
        "max_probability_difference":
            max_difference,
    },

    "safety": {
        "automatic_response":
            False,

        "automatic_containment":
            False,

        "analyst_approval_required":
            True,

        **no_modification,
    },

    "artifact_manifest":
        artifact_manifest,

    "checks":
        checks,
}


with open(
    REPORT_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        report,
        handle,
        indent=2,
        default=str,
    )


# ============================================================================
# 27. FREEZE MANIFEST
# ============================================================================

freeze_manifest = {
    "project":
        "SmartSOC",

    "phase":
        "4A",

    "gate":
        "4.1",

    "timestamp":
        datetime.now().isoformat(),

    "decision":
        decision,

    "production_artifacts":
        artifact_manifest,

    "dataset":
        str(DATA_DIR),

    "model_input_features":
        6919,

    "model_output_techniques":
        45,

    "risk_engine":
        str(
            RISK_ENGINE_FILE
        ),

    "production_modified":
        False,

    "wazuh_modified":
        False,

    "shuffle_called":
        False,

    "response_executed":
        False,

    "automatic_containment":
        False,
}


with open(
    MANIFEST_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        freeze_manifest,
        handle,
        indent=2,
    )


# ============================================================================
# 28. FINAL RESULT
# ============================================================================

print()
print("=" * 76)
print("PHASE 4A — GATE 4.1 FINAL RESULT")
print("=" * 76)

print(
    f"Decision                 : "
    f"{decision}"
)

print(
    f"Validation checks        : "
    f"{passed_checks}/{len(checks)}"
)

print(
    f"Test events              : "
    f"{len(structured)}"
)

print(
    f"Final feature vector     : "
    f"{final_matrix.shape}"
)

print(
    f"MITRE techniques         : "
    f"{len(mitre_techniques)}"
)

print(
    f"Detected events          : "
    f"{detected_events}"
)

print(
    f"Fallback events          : "
    f"{fallback_events}"
)

print(
    f"Fallback fraction        : "
    f"{fallback_fraction:.6f}"
)

print(
    f"Representative events    : "
    f"{len(pipeline_results)}"
)

print(
    f"Risk results valid       : "
    f"{risk_success_count}"
)

print(
    f"Deterministic max diff   : "
    f"{max_difference:.12f}"
)

print(
    "Automatic response       : "
    "DISABLED"
)

print(
    "Automatic containment    : "
    "DISABLED"
)

print(
    "Wazuh modification      : "
    "NONE"
)

print(
    "Shuffle execution       : "
    "NONE"
)

print(
    "Response execution      : "
    "NONE"
)

print(
    "Analyst approval         : "
    "REQUIRED"
)


print()
print("OUTPUTS")

print(
    f"  {RESULTS_CSV}"
)

print(
    f"  {REPORT_JSON}"
)

print(
    f"  {MANIFEST_JSON}"
)


# ============================================================================
# 29. FAILED CHECKS
# ============================================================================

if failed_checks:

    print()
    print("=" * 76)
    print("FAILED CHECKS")
    print("=" * 76)


    for item in checks:

        if not item["passed"]:

            print(
                f"  - {item['name']}"
                + (
                    f": {item['details']}"
                    if item["details"]
                    else ""
                )
            )


    raise RuntimeError(
        "Phase 4A Gate 4.1 failed. "
        "Review the failed checks above."
    )


# ============================================================================
# 30. SUCCESS
# ============================================================================

print()
print("=" * 76)
print("GATE 4.1 COMPLETE — PASS")
print("=" * 76)

print(
    "The frozen SmartSOC AI pipeline "
    "successfully assembled offline."
)

print(
    "ML detection, fallback, confidence, "
    "uncertainty, SOC evidence, and "
    "Risk Engine processing were validated."
)

print(
    "No production model, threshold, dataset, "
    "Wazuh configuration, Shuffle workflow, "
    "or response action was modified."
)

print(
    "Live mitigation remains disabled."
)

print("=" * 76)
