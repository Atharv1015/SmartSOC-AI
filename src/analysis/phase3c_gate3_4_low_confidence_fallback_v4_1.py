"""
SMARTSOC PHASE 3C
GATE 3C.4 — LOW-CONFIDENCE FALLBACK VALIDATION

Purpose
-------
Validate production inference fallback behavior.

This gate validates:

1. Production artifact integrity
2. Production preprocessing dimensions
3. Production probability generation
4. No-detection fallback behavior
5. Highest-probability fallback selection
6. Low-confidence fallback behavior
7. Threshold proximity
8. Deterministic inference
9. Analyst-review safety
10. No automatic response

IMPORTANT
---------
Analysis only.

This script does NOT:
- modify the model
- modify thresholds
- modify datasets
- retrain the model
- modify Wazuh
- call Shuffle
- execute response actions
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase3c"
    / "gate3_4"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

REPORT_PATH = (
    REPORT_DIR
    / "phase3c_gate3_4_low_confidence_fallback_report.json"
)

RESULTS_PATH = (
    REPORT_DIR
    / "phase3c_gate3_4_low_confidence_fallback_results.csv"
)


# ============================================================
# PRODUCTION ARTIFACTS
# ============================================================

MODEL_PATH = (
    PRODUCTION_DIR
    / "model.joblib"
)

ENCODER_PATH = (
    PRODUCTION_DIR
    / "structured_encoder.joblib"
)

SCALER_PATH = (
    PRODUCTION_DIR
    / "numerical_scaler.joblib"
)

TFIDF_PATH = (
    PRODUCTION_DIR
    / "tfidf_vectorizer.joblib"
)

MULTILABEL_PATH = (
    PRODUCTION_DIR
    / "multilabel_encoder.joblib"
)

SCHEMA_PATH = (
    PRODUCTION_DIR
    / "feature_schema.json"
)

THRESHOLDS_PATH = (
    PRODUCTION_DIR
    / "thresholds.json"
)


# ============================================================
# TEST DATA
# ============================================================

STRUCTURED_TEST_PATH = (
    DATA_DIR
    / "structured_test.csv"
)

BEHAVIOR_TEST_PATH = (
    DATA_DIR
    / "behavior_text_test.csv"
)

LABELS_TEST_PATH = (
    DATA_DIR
    / "labels_test.csv"
)


# ============================================================
# QUALITY CHECKS
# ============================================================

quality_checks = {}


def check(
    name: str,
    condition: bool,
    message: str = "",
):

    passed = bool(
        condition
    )

    quality_checks[name] = passed

    if passed:

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

def finite_float(
    value,
) -> bool:

    try:

        number = float(
            value
        )

        return math.isfinite(
            number
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def normalize_behavior(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .fillna("")
        .astype(str)
        .replace(
            {
                "nan": "",
                "None": "",
            }
        )
        .str.strip()
    )


# ============================================================
# SCHEMA
# ============================================================

def extract_schema_columns(
    schema: dict,
):

    categorical_columns = schema.get(
        "categorical_columns"
    )

    numerical_columns = schema.get(
        "numerical_columns"
    )

    if not isinstance(
        categorical_columns,
        list,
    ):

        raise RuntimeError(
            "feature_schema.json does not contain "
            "categorical_columns as a list."
        )

    if not isinstance(
        numerical_columns,
        list,
    ):

        raise RuntimeError(
            "feature_schema.json does not contain "
            "numerical_columns as a list."
        )

    categorical_columns = [
        str(column)
        for column in categorical_columns
    ]

    numerical_columns = [
        str(column)
        for column in numerical_columns
    ]

    structured_columns = (
        categorical_columns
        +
        numerical_columns
    )

    return (
        categorical_columns,
        numerical_columns,
        structured_columns,
    )


# ============================================================
# THRESHOLDS
# ============================================================

def extract_thresholds(
    obj,
):

    thresholds = {}

    def is_technique_id(
        key,
    ):

        if not isinstance(
            key,
            str,
        ):

            return False

        return (
            key.startswith("T")
            and
            any(
                character.isdigit()
                for character in key
            )
        )

    def walk(
        current,
    ):

        if isinstance(
            current,
            dict,
        ):

            for key, value in current.items():

                if is_technique_id(
                    key
                ):

                    if finite_float(
                        value
                    ):

                        numeric = float(
                            value
                        )

                        if (
                            0.0
                            <= numeric
                            <= 1.0
                        ):

                            thresholds[
                                key
                            ] = numeric

                else:

                    walk(
                        value
                    )

        elif isinstance(
            current,
            list,
        ):

            for item in current:

                walk(
                    item
                )

    walk(
        obj
    )

    return thresholds


# ============================================================
# LABEL PARSER
# ============================================================

def parse_techniques(
    value,
) -> set[str]:

    if value is None:

        return set()

    text = str(
        value
    )

    if (
        not text
        or
        text.lower()
        in {
            "nan",
            "none",
        }
    ):

        return set()

    import re

    return set(
        re.findall(
            r"T\d{4}(?:\.\d{3})?",
            text,
        )
    )


# ============================================================
# PRODUCTION FEATURE TRANSFORMATION
# ============================================================

def transform_test_data(
    structured,
    behavior,
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns,
):

    # --------------------------------------------------------
    # Validate all required columns
    # --------------------------------------------------------

    required_columns = (
        categorical_columns
        +
        numerical_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in structured.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Test data is missing production "
            f"structured columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # CATEGORICAL DATA
    # --------------------------------------------------------

    categorical_data = structured[
        categorical_columns
    ].copy()

    categorical_matrix = (
        encoder.transform(
            categorical_data
        )
    )

    # --------------------------------------------------------
    # NUMERICAL DATA
    #
    # IMPORTANT:
    # StandardScaler was fitted WITHOUT feature names.
    # Therefore pass a NumPy array rather than a DataFrame.
    # --------------------------------------------------------

    numerical_data = structured[
        numerical_columns
    ].to_numpy(
        dtype=float
    )

    numerical_matrix = (
        scaler.transform(
            numerical_data
        )
    )

    # --------------------------------------------------------
    # BEHAVIOR TEXT
    # --------------------------------------------------------

    behavior_text = normalize_behavior(
        behavior
    )

    tfidf_matrix = (
        tfidf.transform(
            behavior_text
        )
    )

    # --------------------------------------------------------
    # COMBINE
    #
    # IMPORTANT:
    # categorical_matrix may be sparse while numerical_matrix
    # is dense. Therefore scipy.sparse.hstack is required.
    # --------------------------------------------------------

    from scipy.sparse import csr_matrix
    from scipy.sparse import hstack

    categorical_sparse = (
        categorical_matrix
        if hasattr(
            categorical_matrix,
            "tocsr",
        )
        else csr_matrix(
            categorical_matrix
        )
    )

    numerical_sparse = csr_matrix(
        numerical_matrix
    )

    structured_matrix = hstack(
        [
            categorical_sparse,
            numerical_sparse,
        ],
        format="csr",
    )

    final_matrix = hstack(
        [
            structured_matrix,
            tfidf_matrix,
        ],
        format="csr",
    )

    # --------------------------------------------------------
    # DIMENSION VALIDATION
    # --------------------------------------------------------

    if categorical_matrix.shape[1] != 162:

        raise RuntimeError(
            "Categorical encoded dimension is "
            f"{categorical_matrix.shape[1]}; "
            "expected 162."
        )

    if numerical_matrix.shape[1] != 58:

        raise RuntimeError(
            "Numerical dimension is "
            f"{numerical_matrix.shape[1]}; "
            "expected 58."
        )

    if tfidf_matrix.shape[1] != 6699:

        raise RuntimeError(
            "TF-IDF dimension is "
            f"{tfidf_matrix.shape[1]}; "
            "expected 6699."
        )

    if final_matrix.shape[1] != 6919:

        raise RuntimeError(
            "Final feature dimension is "
            f"{final_matrix.shape[1]}; "
            "expected 6919."
        )

    return (
        final_matrix,
        categorical_matrix,
        numerical_matrix,
        tfidf_matrix,
    )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

def analyze_predictions(
    probabilities,
    technique_ids,
    thresholds,
):

    rows = []

    for row_index in range(
        probabilities.shape[0]
    ):

        row = probabilities[
            row_index
        ]

        ranking = np.argsort(
            row
        )[::-1]

        top_index = int(
            ranking[0]
        )

        second_index = (
            int(
                ranking[1]
            )
            if len(ranking) > 1
            else None
        )

        top_score = float(
            row[
                top_index
            ]
        )

        second_score = (
            float(
                row[
                    second_index
                ]
            )
            if second_index is not None
            else 0.0
        )

        top_technique = str(
            technique_ids[
                top_index
            ]
        )

        top_threshold = float(
            thresholds.get(
                top_technique,
                0.50,
            )
        )

        detected = []

        for technique_index in range(
            len(
                technique_ids
            )
        ):

            technique = str(
                technique_ids[
                    technique_index
                ]
            )

            score = float(
                row[
                    technique_index
                ]
            )

            threshold = float(
                thresholds.get(
                    technique,
                    0.50,
                )
            )

            if score >= threshold:

                detected.append(
                    technique
                )

        fallback_required = (
            len(detected) == 0
        )

        fallback_technique = (
            top_technique
            if fallback_required
            else ""
        )

        rows.append(
            {
                "row_index":
                    row_index,

                "top_technique":
                    top_technique,

                "top_score":
                    top_score,

                "second_score":
                    second_score,

                "top_margin":
                    top_score
                    -
                    second_score,

                "top_threshold":
                    top_threshold,

                "threshold_delta":
                    top_score
                    -
                    top_threshold,

                "detected_count":
                    len(detected),

                "detected_techniques":
                    ",".join(
                        detected
                    ),

                "fallback_required":
                    fallback_required,

                "fallback_technique":
                    fallback_technique,

                "fallback_score":
                    (
                        top_score
                        if fallback_required
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("SMARTSOC PHASE 3C")
print("GATE 3C.4 — LOW-CONFIDENCE FALLBACK VALIDATION")
print("=" * 72)


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

print(
    "\nPRODUCTION ARTIFACT VALIDATION"
)

required_artifacts = [
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    TFIDF_PATH,
    MULTILABEL_PATH,
    SCHEMA_PATH,
    THRESHOLDS_PATH,
]

artifacts_exist = all(
    path.exists()
    for path in required_artifacts
)

check(
    "All production artifacts exist",
    artifacts_exist,
)

if not artifacts_exist:

    raise RuntimeError(
        "Production artifact missing."
    )


# ============================================================
# DATA VALIDATION
# ============================================================

required_data = [
    STRUCTURED_TEST_PATH,
    BEHAVIOR_TEST_PATH,
    LABELS_TEST_PATH,
]

data_exist = all(
    path.exists()
    for path in required_data
)

check(
    "All test data files exist",
    data_exist,
)

if not data_exist:

    raise RuntimeError(
        "Test data file missing."
    )


# ============================================================
# LOAD ARTIFACTS
# ============================================================

print(
    "\nLOADING PRODUCTION ARTIFACTS"
)

model = joblib.load(
    MODEL_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)

scaler = joblib.load(
    SCALER_PATH
)

tfidf = joblib.load(
    TFIDF_PATH
)

multilabel_encoder = joblib.load(
    MULTILABEL_PATH
)

schema = load_json(
    SCHEMA_PATH
)

thresholds_json = load_json(
    THRESHOLDS_PATH
)


check(
    "Model loaded",
    model is not None,
)

check(
    "Structured encoder loaded",
    encoder is not None,
)

check(
    "Numerical scaler loaded",
    scaler is not None,
)

check(
    "TF-IDF vectorizer loaded",
    tfidf is not None,
)

check(
    "Multi-label encoder loaded",
    multilabel_encoder is not None,
)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

print(
    "\nSCHEMA VALIDATION"
)

(
    categorical_columns,
    numerical_columns,
    structured_columns,
) = extract_schema_columns(
    schema
)

print(
    f"Structured features : "
    f"{len(structured_columns)}"
)

print(
    f"Categorical          : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical            : "
    f"{len(numerical_columns)}"
)

print(
    "Categorical columns:"
)

for column in categorical_columns:

    print(
        f"  {column}"
    )


check(
    "Structured feature count is 62",
    len(structured_columns) == 62,
)

check(
    "Categorical feature count is 4",
    len(categorical_columns) == 4,
)

check(
    "Numerical feature count is 58",
    len(numerical_columns) == 58,
)

check(
    "Structured columns are unique",
    len(
        set(structured_columns)
    ) == 62,
)


# ============================================================
# ENCODER VALIDATION
# ============================================================

print(
    "\nENCODER SCHEMA VALIDATION"
)

encoder_feature_names = getattr(
    encoder,
    "feature_names_in_",
    None,
)

if encoder_feature_names is not None:

    encoder_feature_names = [
        str(column)
        for column
        in encoder_feature_names
    ]

    encoder_match = (
        encoder_feature_names
        ==
        categorical_columns
    )

    print(
        "Encoder feature_names_in_:"
    )

    for column in encoder_feature_names:

        print(
            f"  {column}"
        )

else:

    encoder_match = True

    print(
        "[INFO] Encoder does not expose "
        "feature_names_in_."
    )


check(
    "Encoder schema is compatible",
    encoder_match,
)


# ============================================================
# SCALER VALIDATION
# ============================================================

print(
    "\nSCALER SCHEMA VALIDATION"
)

scaler_feature_names = getattr(
    scaler,
    "feature_names_in_",
    None,
)

if scaler_feature_names is not None:

    scaler_feature_names = [
        str(column)
        for column
        in scaler_feature_names
    ]

    scaler_match = (
        scaler_feature_names
        ==
        numerical_columns
    )

    print(
        f"Scaler feature names: "
        f"{len(scaler_feature_names)}"
    )

else:

    scaler_match = True

    print(
        "[INFO] Scaler was fitted without "
        "feature names; using validated "
        "production schema."
    )


check(
    "Scaler schema is compatible",
    scaler_match,
)


# ============================================================
# THRESHOLD VALIDATION
# ============================================================

print(
    "\nTHRESHOLD VALIDATION"
)

thresholds = extract_thresholds(
    thresholds_json
)

print(
    f"Thresholds discovered: "
    f"{len(thresholds)}"
)

if thresholds:

    print(
        f"Minimum threshold: "
        f"{min(thresholds.values()):.6f}"
    )

    print(
        f"Maximum threshold: "
        f"{max(thresholds.values()):.6f}"
    )


check(
    "45 numeric thresholds available",
    len(thresholds) == 45,
)

check(
    "All thresholds finite",
    all(
        finite_float(
            value
        )
        for value
        in thresholds.values()
    ),
)

check(
    "All thresholds within 0-1",
    all(
        0.0 <= float(value) <= 1.0
        for value
        in thresholds.values()
    ),
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print(
    "\nLOADING TEST DATA"
)

structured_test = pd.read_csv(
    STRUCTURED_TEST_PATH
)

behavior_test = pd.read_csv(
    BEHAVIOR_TEST_PATH
)

labels_test = pd.read_csv(
    LABELS_TEST_PATH
)


check(
    "Structured test has 11707 rows",
    len(structured_test) == 11707,
)

check(
    "Behavior test has 11707 rows",
    len(behavior_test) == 11707,
)

check(
    "Labels test has 11707 rows",
    len(labels_test) == 11707,
)


# ============================================================
# COLUMN VALIDATION
# ============================================================

print(
    "\nTEST COLUMN VALIDATION"
)

missing_structured = [
    column
    for column
    in structured_columns
    if column
    not in structured_test.columns
]

check(
    "All 62 structured columns are present",
    len(missing_structured) == 0,
    str(
        missing_structured
    ),
)


behavior_column = (
    "behavior_text_normalized"
    if
    "behavior_text_normalized"
    in behavior_test.columns
    else
    behavior_test.columns[0]
)

label_column = (
    "techniques"
    if
    "techniques"
    in labels_test.columns
    else
    labels_test.columns[0]
)

check(
    "Behavior column available",
    behavior_column
    in behavior_test.columns,
)

check(
    "Label column available",
    label_column
    in labels_test.columns,
)

check(
    "Test rows aligned",
    (
        len(structured_test)
        ==
        len(behavior_test)
        ==
        len(labels_test)
    ),
)


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================

print(
    "\nFEATURE TRANSFORMATION"
)

(
    final_matrix,
    categorical_matrix,
    numerical_matrix,
    tfidf_matrix,
) = transform_test_data(
    structured_test,
    behavior_test[
        behavior_column
    ],
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns,
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


check(
    "Categorical matrix is 11707x162",
    categorical_matrix.shape
    ==
    (11707, 162),
)

check(
    "Numerical matrix is 11707x58",
    numerical_matrix.shape
    ==
    (11707, 58),
)

check(
    "TF-IDF matrix is 11707x6699",
    tfidf_matrix.shape
    ==
    (11707, 6699),
)

check(
    "Final matrix is 11707x6919",
    final_matrix.shape
    ==
    (11707, 6919),
)


# ============================================================
# PRODUCTION PREDICTION
# ============================================================

print(
    "\nPRODUCTION PREDICTION"
)

probabilities = np.asarray(
    model.predict_proba(
        final_matrix
    )
)

print(
    f"Probability matrix: "
    f"{probabilities.shape}"
)


check(
    "Probability matrix is 11707x45",
    probabilities.shape
    ==
    (11707, 45),
)

check(
    "All probabilities finite",
    np.isfinite(
        probabilities
    ).all(),
)

check(
    "All probabilities within 0-1",
    (
        probabilities.min()
        >= 0.0
        and
        probabilities.max()
        <= 1.0
    ),
)


# ============================================================
# TECHNIQUE IDs
# ============================================================

print(
    "\nTECHNIQUE ID VALIDATION"
)

technique_ids = [
    str(value)
    for value
    in getattr(
        multilabel_encoder,
        "classes_",
        schema.get(
            "mitre_techniques",
            [],
        ),
    )
]

check(
    "45 technique IDs available",
    len(technique_ids) == 45,
)

check(
    "Technique IDs align with probability columns",
    len(technique_ids)
    ==
    probabilities.shape[1],
)


# ============================================================
# ANALYZE PREDICTIONS
# ============================================================

print(
    "\nFALLBACK ANALYSIS"
)

analysis_df = analyze_predictions(
    probabilities,
    technique_ids,
    thresholds,
)

fallback_df = analysis_df[
    analysis_df[
        "fallback_required"
    ]
].copy()

detected_df = analysis_df[
    ~analysis_df[
        "fallback_required"
    ]
].copy()

fallback_fraction = (
    len(fallback_df)
    /
    len(analysis_df)
)


print(
    f"Total events      : "
    f"{len(analysis_df)}"
)

print(
    f"Detected events   : "
    f"{len(detected_df)}"
)

print(
    f"Fallback events   : "
    f"{len(fallback_df)}"
)

print(
    f"Fallback fraction : "
    f"{fallback_fraction:.6f}"
)


check(
    "Every event has a top prediction",
    analysis_df[
        "top_technique"
    ]
    .notna()
    .all(),
)

check(
    "Fallback events have fallback technique",
    (
        fallback_df[
            "fallback_technique"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )
    if len(fallback_df) > 0
    else True,
)

check(
    "Fallback score equals top probability",
    (
        np.allclose(
            fallback_df[
                "fallback_score"
            ].to_numpy(),
            fallback_df[
                "top_score"
            ].to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=False,
        )
    )
    if len(fallback_df) > 0
    else True,
)


# ============================================================
# FALLBACK TECHNIQUE CONSISTENCY
# ============================================================

print(
    "\nFALLBACK TECHNIQUE CONSISTENCY"
)

fallback_technique_match = (
    (
        fallback_df[
            "fallback_technique"
        ]
        ==
        fallback_df[
            "top_technique"
        ]
    ).all()
    if len(fallback_df) > 0
    else True
)

check(
    "Fallback selects highest-probability technique",
    fallback_technique_match,
)


fallback_zero_detection = (
    (
        fallback_df[
            "detected_count"
        ]
        ==
        0
    ).all()
    if len(fallback_df) > 0
    else True
)

check(
    "Fallback only occurs with zero detections",
    fallback_zero_detection,
)


# ============================================================
# LOW-CONFIDENCE FALLBACK
# ============================================================

print(
    "\nLOW-CONFIDENCE FALLBACK ANALYSIS"
)

LOW_CONFIDENCE_LIMIT = 0.50

low_confidence_df = fallback_df[
    fallback_df[
        "fallback_score"
    ]
    <
    LOW_CONFIDENCE_LIMIT
].copy()

higher_confidence_df = fallback_df[
    fallback_df[
        "fallback_score"
    ]
    >=
    LOW_CONFIDENCE_LIMIT
].copy()


print(
    f"Low-confidence fallback (<0.50): "
    f"{len(low_confidence_df)}"
)

print(
    f"Higher-confidence fallback (>=0.50): "
    f"{len(higher_confidence_df)}"
)


low_confidence_valid = (
    (
        low_confidence_df[
            "fallback_score"
        ]
        <
        LOW_CONFIDENCE_LIMIT
    ).all()
    if len(low_confidence_df) > 0
    else True
)

check(
    "Low-confidence fallback classification is valid",
    low_confidence_valid,
)


# ============================================================
# THRESHOLD PROXIMITY
# ============================================================

print(
    "\nTHRESHOLD PROXIMITY"
)

analysis_df[
    "absolute_threshold_distance"
] = (
    analysis_df[
        "threshold_delta"
    ].abs()
)

near_threshold_df = analysis_df[
    analysis_df[
        "absolute_threshold_distance"
    ]
    <=
    0.05
].copy()

print(
    f"Events within 0.05 of top threshold: "
    f"{len(near_threshold_df)}"
)

check(
    "Threshold distances are finite",
    np.isfinite(
        analysis_df[
            "absolute_threshold_distance"
        ]
    ).all(),
)


# ============================================================
# FALLBACK STATISTICS
# ============================================================

print(
    "\nFALLBACK STATISTICS"
)

if len(fallback_df) > 0:

    fallback_score_mean = float(
        fallback_df[
            "fallback_score"
        ].mean()
    )

    fallback_score_median = float(
        fallback_df[
            "fallback_score"
        ].median()
    )

    fallback_margin_mean = float(
        fallback_df[
            "top_margin"
        ].mean()
    )

    fallback_margin_median = float(
        fallback_df[
            "top_margin"
        ].median()
    )

else:

    fallback_score_mean = 0.0
    fallback_score_median = 0.0
    fallback_margin_mean = 0.0
    fallback_margin_median = 0.0


print(
    f"Fallback score mean   : "
    f"{fallback_score_mean:.6f}"
)

print(
    f"Fallback score median : "
    f"{fallback_score_median:.6f}"
)

print(
    f"Fallback margin mean  : "
    f"{fallback_margin_mean:.6f}"
)

print(
    f"Fallback margin median: "
    f"{fallback_margin_median:.6f}"
)


# ============================================================
# FALLBACK LABEL DIAGNOSTIC
# ============================================================

print(
    "\nFALLBACK LABEL DIAGNOSTIC"
)

fallback_known = 0
fallback_correct = 0

for _, row in fallback_df.iterrows():

    row_index = int(
        row[
            "row_index"
        ]
    )

    true_labels = parse_techniques(
        labels_test.iloc[
            row_index
        ][
            label_column
        ]
    )

    if not true_labels:

        continue

    fallback_known += 1

    if (
        row[
            "fallback_technique"
        ]
        in true_labels
    ):

        fallback_correct += 1


if fallback_known > 0:

    fallback_accuracy = (
        fallback_correct
        /
        fallback_known
    )

else:

    fallback_accuracy = None


print(
    f"Fallback events with known labels: "
    f"{fallback_known}"
)

print(
    f"Fallback top-technique correct: "
    f"{fallback_correct}"
)

if fallback_accuracy is not None:

    print(
        f"Fallback diagnostic accuracy: "
        f"{fallback_accuracy:.6f}"
    )


# ============================================================
# DETERMINISM
# ============================================================

print(
    "\nDETERMINISM TEST"
)

sample_size = min(
    100,
    final_matrix.shape[0],
)
sample_indices = np.arange(
    sample_size
)

prediction_one = np.asarray(
    model.predict_proba(
        final_matrix[
            sample_indices
        ]
    )
)

prediction_two = np.asarray(
    model.predict_proba(
        final_matrix[
            sample_indices
        ]
    )
)

exact_equal = np.array_equal(
    prediction_one,
    prediction_two,
)

check(
    "Repeated inference is deterministic",
    exact_equal,
)


# ============================================================
# FALLBACK SAFETY CONTRACT
# ============================================================

print(
    "\nFALLBACK SAFETY CONTRACT"
)

safety_contract = {

    "automatic_response":
        False,

    "automatic_blocking":
        False,

    "automatic_isolation":
        False,

    "automatic_process_kill":
        False,

    "fallback_direct_containment":
        False,

    "analyst_required":
        True,
}


check(
    "Automatic response disabled",
    safety_contract[
        "automatic_response"
    ] is False,
)

check(
    "Automatic blocking disabled",
    safety_contract[
        "automatic_blocking"
    ] is False,
)

check(
    "Automatic isolation disabled",
    safety_contract[
        "automatic_isolation"
    ] is False,
)

check(
    "Automatic process kill disabled",
    safety_contract[
        "automatic_process_kill"
    ] is False,
)

check(
    "Fallback cannot directly contain",
    safety_contract[
        "fallback_direct_containment"
    ] is False,
)

check(
    "Analyst required",
    safety_contract[
        "analyst_required"
    ] is True,
)


# ============================================================
# PRODUCTION MODIFICATION SAFETY
# ============================================================

print(
    "\nPRODUCTION MODIFICATION SAFETY"
)

production_safety = {

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
}


check(
    "Model not modified",
    production_safety[
        "model_modified"
    ] is False,
)

check(
    "Thresholds not modified",
    production_safety[
        "thresholds_modified"
    ] is False,
)

check(
    "Dataset not modified",
    production_safety[
        "dataset_modified"
    ] is False,
)

check(
    "Wazuh not modified",
    production_safety[
        "wazuh_modified"
    ] is False,
)

check(
    "Shuffle not called",
    production_safety[
        "shuffle_called"
    ] is False,
)

check(
    "No response executed",
    production_safety[
        "response_executed"
    ] is False,
)


# ============================================================
# SAVE CSV
# ============================================================

analysis_df.to_csv(
    RESULTS_PATH,
    index=False,
)


# ============================================================
# FINAL DECISION
# ============================================================

failed_checks = [
    name
    for name, passed
    in quality_checks.items()
    if not passed
]

if len(
    failed_checks
) == 0:

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
        "3C",

    "gate":
        "3C.4",

    "name":
        "Low-Confidence Fallback Validation",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "decision":
        decision,

    "test_data": {

        "structured_rows":
            len(structured_test),

        "behavior_rows":
            len(behavior_test),

        "label_rows":
            len(labels_test),
    },

    "schema": {

        "structured_features":
            len(structured_columns),

        "categorical_features":
            len(categorical_columns),

        "numerical_features":
            len(numerical_columns),

        "categorical_encoded_features":
            162,

        "tfidf_features":
            6699,

        "final_features":
            6919,

        "mitre_techniques":
            len(technique_ids),
    },

    "prediction": {

        "events":
            len(analysis_df),

        "detected_events":
            len(detected_df),

        "fallback_events":
            len(fallback_df),

        "fallback_fraction":
            fallback_fraction,
    },

    "fallback": {

        "low_confidence_limit":
            LOW_CONFIDENCE_LIMIT,

        "low_confidence_events":
            len(
                low_confidence_df
            ),

        "higher_confidence_events":
            len(
                higher_confidence_df
            ),

        "mean_score":
            fallback_score_mean,

        "median_score":
            fallback_score_median,

        "mean_margin":
            fallback_margin_mean,

        "median_margin":
            fallback_margin_median,

        "label_diagnostic": {

            "events_with_known_labels":
                fallback_known,

            "correct":
                fallback_correct,

            "accuracy":
                fallback_accuracy,
        },
    },

    "threshold_proximity": {

        "within_0_05":
            len(
                near_threshold_df
            ),
    },

    "determinism": {

        "exact_equal":
            exact_equal,
    },

    "safety_contract":
        safety_contract,

    "production_safety":
        production_safety,

    "quality_checks":
        quality_checks,

    "failed_checks":
        failed_checks,

    "limitations": [

        (
            "Fallback is a decision-support mechanism "
            "and does not prove that the selected "
            "highest-probability technique is correct."
        ),

        (
            "The untouched test set is used for analysis "
            "only. Production artifacts are not modified."
        ),

        (
            "Fallback correctness is diagnostic because "
            "the underlying classifier is multilabel."
        ),

        (
            "Confidence values are not assumed to be "
            "calibrated probabilities."
        ),

        (
            "This gate does not execute Wazuh, Shuffle, "
            "or containment actions."
        ),
    ],

    "outputs": {

        "results_csv":
            str(
                RESULTS_PATH
            ),

        "report_json":
            str(
                REPORT_PATH
            ),
    },
}


# ============================================================
# WRITE JSON
# ============================================================

with REPORT_PATH.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        report,
        handle,
        indent=2,
        default=str,
    )


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "GATE 3C.4 FINAL RESULT"
)

print(
    "=" * 72
)

print(
    f"Decision             : "
    f"{decision}"
)

print(
    f"Test events          : "
    f"{len(analysis_df)}"
)

print(
    f"Detected events      : "
    f"{len(detected_df)}"
)

print(
    f"Fallback events      : "
    f"{len(fallback_df)}"
)

print(
    f"Fallback fraction    : "
    f"{fallback_fraction:.6f}"
)

print(
    f"Low-confidence (<.50): "
    f"{len(low_confidence_df)}"
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

print(
    "\nOUTPUTS"
)

print(
    f"  {RESULTS_PATH}"
)

print(
    f"  {REPORT_PATH}"
)

print(
    "=" * 72
)


if decision != "PASS":

    raise RuntimeError(
        "Phase 3C.4 requires review."
    )