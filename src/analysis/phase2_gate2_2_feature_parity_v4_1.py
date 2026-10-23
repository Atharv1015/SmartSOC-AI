import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.2
# FEATURE-PARITY VALIDATION
# =============================================================================


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4_1"

MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

REPORT_DIR = ROOT / "reports" / "phase2_gate2_2"

STRUCTURED_TEST_FILE = DATA_DIR / "structured_test.csv"
BEHAVIOR_TEST_FILE = DATA_DIR / "behavior_text_test.csv"

SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
MULTILABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"

REPORT_FILE = (
    REPORT_DIR / "phase2_gate2_2_feature_parity_report.json"
)

SAMPLE_FILE = (
    REPORT_DIR / "feature_parity_inference_samples.csv"
)


# -----------------------------------------------------------------------------
# EXPECTED CONSTANTS
# -----------------------------------------------------------------------------

EXPECTED_STRUCTURED_FEATURES = 62
EXPECTED_CATEGORICAL_FEATURES = 4
EXPECTED_NUMERICAL_FEATURES = 58
EXPECTED_CATEGORICAL_ENCODED_FEATURES = 162
EXPECTED_TFIDF_FEATURES = 6699
EXPECTED_FINAL_FEATURES = 6919
EXPECTED_MITRE_CLASSES = 45


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn",
)


def fail(message):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check_file(path):
    if not path.exists():
        fail(f"Required file not found: {path}")

    print(f"[OK] {path}")


def shape_check(actual, expected, name):
    if actual != expected:
        fail(
            f"{name} mismatch. "
            f"Expected={expected}, actual={actual}"
        )

    print(
        f"[PASS] {name} "
        f"expected={expected} actual={actual}"
    )


def to_numeric_array(matrix):
    """
    Safely convert dense/sparse/object matrices to a numeric NumPy array.
    """

    # Sparse matrix
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    # Pandas object
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy()

    elif isinstance(matrix, pd.Series):
        matrix = matrix.to_numpy()

    # Force numeric dtype
    return np.asarray(matrix, dtype=np.float64)


def finite_check(matrix, name):
    """
    Validate that a matrix contains only finite numeric values.
    """

    try:
        arr = to_numeric_array(matrix)

    except Exception as exc:
        fail(
            f"{name} could not be converted to a numeric "
            f"matrix for finite-value validation: {exc}"
        )

    if not np.all(np.isfinite(arr)):
        fail(
            f"{name} contains NaN or infinite values."
        )

    print(f"[PASS] {name} finite.")


# =============================================================================
# START
# =============================================================================

print("=" * 78)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.2")
print("FEATURE-PARITY VALIDATION")
print("=" * 78)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 1. REQUIRED FILES
# =============================================================================

print()
print("-" * 78)
print("Checking required files")
print("-" * 78)

required_files = [
    STRUCTURED_TEST_FILE,
    BEHAVIOR_TEST_FILE,
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    TFIDF_FILE,
    MODEL_FILE,
    MULTILABEL_ENCODER_FILE,
]

for path in required_files:
    check_file(path)


# =============================================================================
# 2. LOAD FEATURE SCHEMA
# =============================================================================

print()
print("-" * 78)
print("Loading feature schema")
print("-" * 78)

with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as f:
    schema = json.load(f)


categorical_columns = [
    str(x)
    for x in schema.get(
        "categorical_columns",
        []
    )
]

numerical_columns = [
    str(x)
    for x in schema.get(
        "numerical_columns",
        []
    )
]

schema_classes = [
    str(x)
    for x in schema.get(
        "mitre_techniques",
        []
    )
]


expected_cat_input = int(
    schema.get(
        "categorical_input_feature_count",
        len(categorical_columns)
    )
)

expected_cat_encoded = int(
    schema.get(
        "categorical_encoded_feature_count",
        EXPECTED_CATEGORICAL_ENCODED_FEATURES
    )
)

expected_numerical = int(
    schema.get(
        "numerical_feature_count",
        len(numerical_columns)
    )
)

expected_tfidf = int(
    schema.get(
        "tfidf_feature_count",
        EXPECTED_TFIDF_FEATURES
    )
)

expected_final = int(
    schema.get(
        "final_feature_count",
        EXPECTED_FINAL_FEATURES
    )
)

expected_mitre = len(schema_classes)


print(
    f"Categorical columns : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical columns   : "
    f"{len(numerical_columns)}"
)

print(
    f"Structured columns  : "
    f"{len(categorical_columns) + len(numerical_columns)}"
)

print(
    f"Expected cat input  : "
    f"{expected_cat_input}"
)

print(
    f"Expected cat encoded: "
    f"{expected_cat_encoded}"
)

print(
    f"Expected numerical  : "
    f"{expected_numerical}"
)

print(
    f"Expected TF-IDF     : "
    f"{expected_tfidf}"
)

print(
    f"Expected final      : "
    f"{expected_final}"
)

print(
    f"MITRE techniques    : "
    f"{expected_mitre}"
)


# =============================================================================
# 3. SCHEMA CONSISTENCY
# =============================================================================

print()
print("-" * 78)
print("Validating schema consistency")
print("-" * 78)

if len(categorical_columns) != expected_cat_input:
    fail(
        "Categorical input count mismatch in schema. "
        f"Declared={expected_cat_input}, "
        f"actual={len(categorical_columns)}"
    )

if len(numerical_columns) != expected_numerical:
    fail(
        "Numerical feature count mismatch in schema. "
        f"Declared={expected_numerical}, "
        f"actual={len(numerical_columns)}"
    )

structured_count = (
    len(categorical_columns)
    + len(numerical_columns)
)

if structured_count != EXPECTED_STRUCTURED_FEATURES:
    fail(
        "Structured feature count mismatch. "
        f"Expected={EXPECTED_STRUCTURED_FEATURES}, "
        f"actual={structured_count}"
    )

if (
    expected_cat_encoded
    + expected_numerical
    + expected_tfidf
    != expected_final
):
    fail(
        "Final feature arithmetic is inconsistent."
    )

print(
    "[PASS] Schema dimensions are internally consistent."
)

print(
    f"[PASS] {len(categorical_columns)} categorical + "
    f"{len(numerical_columns)} numerical = "
    f"{structured_count} structured features."
)

print(
    f"[PASS] {expected_cat_encoded} categorical + "
    f"{expected_numerical} numerical + "
    f"{expected_tfidf} TF-IDF = "
    f"{expected_final} final features."
)


# =============================================================================
# 4. LOAD HELD-OUT TEST DATA
# =============================================================================

print()
print("-" * 78)
print("Loading held-out test data")
print("-" * 78)

structured_test = pd.read_csv(
    STRUCTURED_TEST_FILE
)

behavior_test = pd.read_csv(
    BEHAVIOR_TEST_FILE
)

print(
    f"Structured test shape : "
    f"{structured_test.shape}"
)

print(
    f"Behavior test shape   : "
    f"{behavior_test.shape}"
)


# =============================================================================
# 5. STRUCTURED SCHEMA VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating held-out structured schema")
print("-" * 78)

actual_structured_columns = list(
    structured_test.columns
)

expected_structured_columns = (
    categorical_columns
    + numerical_columns
)

missing_structured = [
    col
    for col in expected_structured_columns
    if col not in actual_structured_columns
]

unexpected_structured = [
    col
    for col in actual_structured_columns
    if col not in expected_structured_columns
]

if missing_structured:
    fail(
        "Missing structured feature columns: "
        f"{missing_structured}"
    )

print(
    f"[PASS] All "
    f"{len(expected_structured_columns)} "
    "expected structured columns are present."
)

if unexpected_structured:
    fail(
        "Unexpected structured feature columns: "
        f"{unexpected_structured}"
    )

print(
    "[PASS] No unexpected structured feature columns."
)


# =============================================================================
# 6. BEHAVIOR TEXT VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating behavior-text schema")
print("-" * 78)

if (
    "behavior_text_normalized"
    not in behavior_test.columns
):
    fail(
        "behavior_text_normalized column is missing."
    )

print(
    "[PASS] behavior_text_normalized present."
)


# =============================================================================
# 7. ROW ALIGNMENT
# =============================================================================

print()
print("-" * 78)
print("Checking row alignment")
print("-" * 78)

structured_rows = len(
    structured_test
)

behavior_rows = len(
    behavior_test
)

if structured_rows != behavior_rows:
    fail(
        "Structured and behavior row counts differ. "
        f"Structured={structured_rows}, "
        f"Behavior={behavior_rows}"
    )

print(
    f"[PASS] Structured and behavior rows aligned: "
    f"{structured_rows}"
)


# =============================================================================
# 8. LOAD PRODUCTION ARTIFACTS
# =============================================================================

print()
print("-" * 78)
print("Loading production preprocessing artifacts")
print("-" * 78)

structured_encoder = joblib.load(
    ENCODER_FILE
)

numerical_scaler = joblib.load(
    SCALER_FILE
)

tfidf_vectorizer = joblib.load(
    TFIDF_FILE
)

model = joblib.load(
    MODEL_FILE
)

print(
    f"Encoder type : "
    f"{type(structured_encoder).__name__}"
)

print(
    f"Scaler type  : "
    f"{type(numerical_scaler).__name__}"
)

print(
    f"TF-IDF type  : "
    f"{type(tfidf_vectorizer).__name__}"
)

print(
    f"Model type   : "
    f"{type(model).__name__}"
)


# =============================================================================
# 9. PRODUCTION ARTIFACT DIMENSIONS
# =============================================================================

print()
print("-" * 78)
print("Validating production artifact dimensions")
print("-" * 78)

if not hasattr(
    structured_encoder,
    "categories_"
):
    fail(
        "Structured encoder does not expose categories_."
    )

if not hasattr(
    numerical_scaler,
    "n_features_in_"
):
    fail(
        "Numerical scaler does not expose "
        "n_features_in_."
    )

if not hasattr(
    tfidf_vectorizer,
    "vocabulary_"
):
    fail(
        "TF-IDF vectorizer does not expose "
        "vocabulary_."
    )

if not hasattr(
    model,
    "predict_proba"
):
    fail(
        "Production model does not expose "
        "predict_proba()."
    )


encoder_output_features = sum(
    len(categories)
    for categories
    in structured_encoder.categories_
)

shape_check(
    encoder_output_features,
    expected_cat_encoded,
    "Categorical encoded features"
)

shape_check(
    numerical_scaler.n_features_in_,
    expected_numerical,
    "Numerical features"
)

shape_check(
    len(tfidf_vectorizer.vocabulary_),
    expected_tfidf,
    "TF-IDF vocabulary"
)


# =============================================================================
# 10. BUILD CATEGORICAL MATRIX
# =============================================================================

print()
print("-" * 78)
print("Building categorical feature matrix")
print("-" * 78)

categorical_input = (
    structured_test[
        categorical_columns
    ]
    .fillna("")
    .astype(str)
)

print(
    f"Categorical input shape: "
    f"{categorical_input.shape}"
)

if (
    categorical_input.shape[1]
    != expected_cat_input
):
    fail(
        "Categorical input feature count mismatch."
    )

categorical_encoded = (
    structured_encoder.transform(
        categorical_input
    )
)

print(
    f"Categorical encoded shape: "
    f"{categorical_encoded.shape}"
)

shape_check(
    categorical_encoded.shape[1],
    expected_cat_encoded,
    "Categorical encoded"
)


# =============================================================================
# 11. BUILD NUMERICAL MATRIX
# =============================================================================

print()
print("-" * 78)
print("Building numerical feature matrix")
print("-" * 78)

numerical_raw = (
    structured_test[
        numerical_columns
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)

if numerical_raw.isna().any().any():

    bad_columns = (
        numerical_raw.columns[
            numerical_raw.isna().any()
        ]
        .tolist()
    )

    fail(
        "Numerical feature matrix contains "
        "NaN/non-numeric values in: "
        f"{bad_columns}"
    )

print(
    f"Numerical raw shape: "
    f"{numerical_raw.shape}"
)

if (
    numerical_raw.shape[1]
    != expected_numerical
):
    fail(
        "Numerical input feature count mismatch."
    )

numerical_scaled = (
    numerical_scaler.transform(
        numerical_raw
    )
)

print(
    f"Numerical scaled shape: "
    f"{numerical_scaled.shape}"
)

shape_check(
    numerical_scaled.shape[1],
    expected_numerical,
    "Numerical"
)


# =============================================================================
# 12. BUILD TF-IDF MATRIX
# =============================================================================

print()
print("-" * 78)
print("Building TF-IDF feature matrix")
print("-" * 78)

behavior_text = (
    behavior_test[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
    .tolist()
)

tfidf_matrix = (
    tfidf_vectorizer.transform(
        behavior_text
    )
)

print(
    f"TF-IDF shape: "
    f"{tfidf_matrix.shape}"
)

shape_check(
    tfidf_matrix.shape[1],
    expected_tfidf,
    "TF-IDF"
)


# =============================================================================
# 13. FEATURE DIMENSION VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating feature dimensions")
print("-" * 78)

shape_check(
    categorical_encoded.shape[1],
    expected_cat_encoded,
    "Categorical encoded"
)

shape_check(
    numerical_scaled.shape[1],
    expected_numerical,
    "Numerical"
)

shape_check(
    tfidf_matrix.shape[1],
    expected_tfidf,
    "TF-IDF"
)

calculated_final_features = (
    categorical_encoded.shape[1]
    + numerical_scaled.shape[1]
    + tfidf_matrix.shape[1]
)

shape_check(
    calculated_final_features,
    expected_final,
    "Final"
)


# =============================================================================
# 14. NUMERICAL STABILITY
# =============================================================================

print()
print("-" * 78)
print("Checking numerical stability")
print("-" * 78)

# IMPORTANT:
# Convert OneHotEncoder output safely to float before np.isfinite().
finite_check(
    categorical_encoded,
    "Categorical features"
)

finite_check(
    numerical_scaled,
    "Numerical features"
)

finite_check(
    tfidf_matrix,
    "TF-IDF features"
)


# =============================================================================
# 15. CONSTRUCT FINAL MODEL MATRIX
# =============================================================================

print()
print("-" * 78)
print("Constructing final model matrix")
print("-" * 78)

categorical_array = to_numeric_array(
    categorical_encoded
)

numerical_array = to_numeric_array(
    numerical_scaled
)

tfidf_array = to_numeric_array(
    tfidf_matrix
)

final_matrix = np.hstack(
    [
        categorical_array,
        numerical_array,
        tfidf_array,
    ]
)

print(
    f"Final model matrix shape: "
    f"{final_matrix.shape}"
)

if (
    final_matrix.shape[1]
    != expected_final
):
    fail(
        "Final model matrix dimension mismatch. "
        f"Expected={expected_final}, "
        f"actual={final_matrix.shape[1]}"
    )

print(
    f"[PASS] Model input dimension = "
    f"{final_matrix.shape[1]}"
)


# =============================================================================
# 16. FINAL MATRIX STABILITY
# =============================================================================

finite_check(
    final_matrix,
    "Final model matrix"
)


# =============================================================================
# 17. RUN PRODUCTION MODEL
# =============================================================================

print()
print("-" * 78)
print("Running production model on held-out test data")
print("-" * 78)

probabilities = model.predict_proba(
    final_matrix
)

probabilities = np.asarray(
    probabilities,
    dtype=np.float64
)

print(
    f"Probability matrix shape: "
    f"{probabilities.shape}"
)

if probabilities.ndim != 2:
    fail(
        "Probability matrix is not 2-dimensional."
    )

shape_check(
    probabilities.shape[0],
    structured_rows,
    "Probability rows"
)

shape_check(
    probabilities.shape[1],
    expected_mitre,
    "Probability columns"
)


# =============================================================================
# 18. PROBABILITY VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating probability values")
print("-" * 78)

if not np.all(
    np.isfinite(probabilities)
):
    fail(
        "Probability matrix contains "
        "NaN or infinite values."
    )

print(
    "[PASS] All probabilities are finite."
)

if (
    np.any(probabilities < 0.0)
    or np.any(probabilities > 1.0)
):
    fail(
        "Probability values outside [0, 1] detected."
    )

print(
    "[PASS] All probabilities are within [0, 1]."
)


# =============================================================================
# 19. MITRE TECHNIQUE CLASS VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating MITRE technique classes")
print("-" * 78)

check_file(
    MULTILABEL_ENCODER_FILE
)

multilabel_encoder = joblib.load(
    MULTILABEL_ENCODER_FILE
)

if not hasattr(
    multilabel_encoder,
    "classes_"
):
    fail(
        "MultiLabelBinarizer does not expose "
        "classes_."
    )

mitre_classes = [
    str(x)
    for x in multilabel_encoder.classes_
]

print(
    f"MultiLabelBinarizer classes: "
    f"{len(mitre_classes)}"
)

if len(mitre_classes) != expected_mitre:
    fail(
        "MITRE class count mismatch. "
        f"Expected={expected_mitre}, "
        f"actual={len(mitre_classes)}"
    )

if not hasattr(
    model,
    "classes_"
):
    fail(
        "OneVsRestClassifier does not expose "
        "classes_."
    )

model_classes = [
    str(x)
    for x in model.classes_
]

print(
    f"OneVsRest model classes: "
    f"{len(model_classes)}"
)

if (
    len(model_classes)
    != len(mitre_classes)
):
    fail(
        "Model class count mismatch. "
        f"Model={len(model_classes)}, "
        f"encoder={len(mitre_classes)}"
    )


# IMPORTANT:
# OneVsRestClassifier.classes_ contains output positions.
# MultiLabelBinarizer.classes_ contains actual MITRE IDs.

schema_class_set = set(
    schema_classes
)

encoder_class_set = set(
    mitre_classes
)

if schema_class_set != encoder_class_set:

    missing = sorted(
        schema_class_set
        - encoder_class_set
    )

    extra = sorted(
        encoder_class_set
        - schema_class_set
    )

    fail(
        "MITRE class mismatch. "
        f"Missing={missing}; "
        f"Extra={extra}"
    )

print(
    "[PASS] MITRE technique classes match "
    "schema and encoder."
)


# =============================================================================
# 20. PROBABILITY COLUMN → MITRE MAPPING
# =============================================================================

print()
print("-" * 78)
print("Validating probability-column mapping")
print("-" * 78)

if (
    len(mitre_classes)
    != probabilities.shape[1]
):
    fail(
        "Probability-to-MITRE mapping "
        "dimension mismatch."
    )

probability_columns = {
    technique: probabilities[:, index]
    for index, technique
    in enumerate(mitre_classes)
}

if (
    len(probability_columns)
    != expected_mitre
):
    fail(
        "Probability mapping does not contain "
        "all MITRE techniques."
    )

print(
    f"[PASS] {len(probability_columns)} "
    "probability columns mapped to "
    "MITRE techniques."
)


# =============================================================================
# 21. THRESHOLD VALIDATION
# =============================================================================

print()
print("-" * 78)
print("Validating thresholds")
print("-" * 78)

thresholds = {}

if THRESHOLD_FILE.exists():

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        threshold_data = json.load(f)

    if (
        isinstance(threshold_data, dict)
        and "thresholds"
        in threshold_data
    ):
        thresholds = (
            threshold_data["thresholds"]
        )

    else:
        thresholds = threshold_data

    if not isinstance(
        thresholds,
        dict
    ):
        fail(
            "Threshold artifact does not contain "
            "a dictionary of thresholds."
        )

else:

    print(
        "[WARN] thresholds.json not found. "
        "Threshold validation skipped."
    )


if thresholds:

    invalid_thresholds = {}

    for technique, threshold in (
        thresholds.items()
    ):

        try:
            value = float(
                threshold
            )

        except Exception:

            invalid_thresholds[
                str(technique)
            ] = threshold

            continue

        if not (
            0.0 <= value <= 1.0
        ):

            invalid_thresholds[
                str(technique)
            ] = threshold

    if invalid_thresholds:
        fail(
            "Invalid threshold values detected: "
            f"{invalid_thresholds}"
        )

    print(
        f"[PASS] Threshold values valid: "
        f"{len(thresholds)} entries"
    )

    threshold_techniques = set(
        str(x)
        for x in thresholds.keys()
    )

    unknown_thresholds = (
        threshold_techniques
        - encoder_class_set
    )

    if unknown_thresholds:
        fail(
            "Threshold artifact contains "
            "unknown MITRE techniques: "
            f"{sorted(unknown_thresholds)}"
        )

    print(
        "[PASS] Threshold technique IDs are valid."
    )


# =============================================================================
# 22. REPRESENTATIVE INFERENCE SAMPLES
# =============================================================================

print()
print("-" * 78)
print("Generating representative inference samples")
print("-" * 78)

sample_count = min(
    20,
    len(structured_test)
)

sample_indices = list(
    range(sample_count)
)

sample_rows = []


for row_index in sample_indices:

    row_probabilities = probabilities[
        row_index
    ]

    ranking = np.argsort(
        row_probabilities
    )[::-1]

    top_n = min(
        5,
        len(ranking)
    )

    top_predictions = []

    for position in ranking[:top_n]:

        technique = mitre_classes[
            int(position)
        ]

        probability = float(
            row_probabilities[
                position
            ]
        )

        threshold = float(
            thresholds.get(
                technique,
                0.5
            )
        )

        top_predictions.append(
            (
                technique,
                probability,
                threshold
            )
        )

    sample = {
        "row_index": int(
            row_index
        ),

        "top_1_technique":
            top_predictions[0][0],

        "top_1_probability":
            top_predictions[0][1],

        "top_1_threshold":
            top_predictions[0][2],
    }

    for rank, (
        technique,
        probability,
        threshold
    ) in enumerate(
        top_predictions,
        start=1
    ):

        sample[
            f"top_{rank}_technique"
        ] = technique

        sample[
            f"top_{rank}_probability"
        ] = probability

        sample[
            f"top_{rank}_threshold"
        ] = threshold

    sample_rows.append(
        sample
    )


samples_df = pd.DataFrame(
    sample_rows
)

samples_df.to_csv(
    SAMPLE_FILE,
    index=False
)

print(
    f"[OK] Representative inference "
    f"samples saved: {SAMPLE_FILE}"
)


# =============================================================================
# 23. BUILD REPORT
# =============================================================================

print()
print("-" * 78)
print("Building Gate 2.2 report")
print("-" * 78)


report = {

    "phase": "Phase 2",

    "gate": "Gate 2.2",

    "gate_name":
        "Feature-Parity Validation",

    "model":
        "SmartSOC Windows-APT Model B v4.1",

    "status":
        "ACCEPTED",

    "dataset": {

        "structured_test_rows":
            int(structured_rows),

        "behavior_test_rows":
            int(behavior_rows),

        "structured_columns":
            int(
                len(
                    expected_structured_columns
                )
            ),

        "behavior_text_column":
            "behavior_text_normalized",
    },

    "schema": {

        "categorical_input_features":
            int(
                len(
                    categorical_columns
                )
            ),

        "numerical_features":
            int(
                len(
                    numerical_columns
                )
            ),

        "structured_features":
            int(
                structured_count
            ),

        "categorical_encoded_features":
            int(
                expected_cat_encoded
            ),

        "tfidf_features":
            int(
                expected_tfidf
            ),

        "final_features":
            int(
                expected_final
            ),

        "mitre_classes":
            int(
                expected_mitre
            ),
    },

    "production_artifacts": {

        "structured_encoder":
            str(ENCODER_FILE),

        "numerical_scaler":
            str(SCALER_FILE),

        "tfidf_vectorizer":
            str(TFIDF_FILE),

        "model":
            str(MODEL_FILE),

        "multilabel_encoder":
            str(
                MULTILABEL_ENCODER_FILE
            ),

        "thresholds":
            str(THRESHOLD_FILE),
    },

    "observed_dimensions": {

        "categorical_input":
            int(
                categorical_input.shape[1]
            ),

        "categorical_encoded":
            int(
                categorical_encoded.shape[1]
            ),

        "numerical_raw":
            int(
                numerical_raw.shape[1]
            ),

        "numerical_scaled":
            int(
                numerical_scaled.shape[1]
            ),

        "tfidf":
            int(
                tfidf_matrix.shape[1]
            ),

        "final_model_input":
            int(
                final_matrix.shape[1]
            ),

        "probability_columns":
            int(
                probabilities.shape[1]
            ),
    },

    "validation": {

        "schema_consistent":
            True,

        "structured_schema_valid":
            True,

        "behavior_text_schema_valid":
            True,

        "row_alignment_valid":
            True,

        "categorical_dimension_valid":
            True,

        "numerical_dimension_valid":
            True,

        "tfidf_dimension_valid":
            True,

        "final_dimension_valid":
            True,

        "numerical_stability_valid":
            True,

        "probabilities_finite":
            True,

        "probabilities_in_range":
            True,

        "mitre_class_mapping_valid":
            True,

        "probability_mapping_valid":
            True,

        "threshold_validation":
            bool(
                bool(thresholds)
            ),
    },

    "notes": [

        "Production artifacts were loaded "
        "read-only.",

        "The held-out test set was used only "
        "for feature-parity and inference "
        "validation.",

        "OneVsRestClassifier.classes_ contains "
        "numeric output positions; MITRE IDs "
        "are validated through "
        "multilabel_encoder.classes_.",

        "The probability column mapping follows "
        "the order of "
        "multilabel_encoder.classes_.",

        "Sparse/object feature matrices are "
        "converted to numeric arrays before "
        "finite-value validation.",
    ],
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print(
    f"[OK] Gate 2.2 report saved: "
    f"{REPORT_FILE}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 78)
print("PHASE 2 GATE 2.2 STATUS: ACCEPTED")
print("=" * 78)

print()

print(
    "Feature-parity validation completed successfully."
)

print()

print(
    f"Structured features : "
    f"{structured_count}"
)

print(
    f"Categorical encoded : "
    f"{categorical_encoded.shape[1]}"
)

print(
    f"Numerical features  : "
    f"{numerical_scaled.shape[1]}"
)

print(
    f"TF-IDF features     : "
    f"{tfidf_matrix.shape[1]}"
)

print(
    f"Final model input   : "
    f"{final_matrix.shape[1]}"
)

print(
    f"MITRE classes       : "
    f"{len(mitre_classes)}"
)

print(
    f"Test rows           : "
    f"{structured_rows}"
)

print()

print(
    "Production artifacts were not modified."
)

print(
    f"Report: {REPORT_FILE}"
)

print(
    f"Samples: {SAMPLE_FILE}"
)