"""
SmartSOC Stage 4
92057 Coefficient-Level Forensics

Purpose
-------
Explain why the frozen SmartSOC v4.1 model predicts T1070 instead
of T1059.001 for Wazuh rule 92057.

This is a diagnostic-only experiment.

NO:
- model modification
- retraining
- threshold modification
- production inference modification
- victim contact
- mitigation
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import production_inference_windows_apt_v4_1_aligned as canonical


SOURCE_FILE = (
    PROJECT_ROOT /
    "test_92057_pipeline_forensics_v3.py"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "reports" /
    "integration"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    REPORT_DIR /
    "92057_coefficient_forensics_v1.json"
)


TARGETS = [
    "T1070",
    "T1059.001",
]


# ============================================================
# EXACT EVENT EXTRACTION
# ============================================================

def load_exact_event():

    source = SOURCE_FILE.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(SOURCE_FILE),
    )

    for node in tree.body:

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:

            if (
                isinstance(
                    target,
                    ast.Name,
                )
                and target.id
                == "RAW_WAZUH_EVENT"
            ):

                return ast.literal_eval(
                    node.value
                )

    raise RuntimeError(
        "RAW_WAZUH_EVENT was not found."
    )


# ============================================================
# SAFE SIGMOID
# ============================================================

def sigmoid(x):

    x = float(x)

    if x >= 0:

        z = math.exp(-x)

        return 1.0 / (1.0 + z)

    z = math.exp(x)

    return z / (1.0 + z)


# ============================================================
# MODEL ESTIMATOR EXTRACTION
# ============================================================

def get_estimator(
    model,
    label_encoder,
    technique,
):

    classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    if technique not in classes:

        raise RuntimeError(
            f"{technique} not found in label encoder."
        )

    class_index = classes.index(
        technique
    )

    estimators = getattr(
        model,
        "estimators_",
        None,
    )

    if estimators is None:

        raise RuntimeError(
            "Model does not expose estimators_."
        )

    estimator = estimators[
        class_index
    ]

    return (
        class_index,
        estimator,
    )


# ============================================================
# FEATURE NAME CONSTRUCTION
# ============================================================

def build_feature_names(
    feature_result,
    artifacts,
):

    encoder = artifacts[
        "encoder"
    ]

    tfidf = artifacts[
        "tfidf"
    ]

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    categorical_names = []

    categorical_df = (
        feature_result[
            "categorical_df"
        ]
    )

    try:

        encoded_names = (
            encoder.get_feature_names_out(
                categorical_df.columns
            )
        )

        categorical_names = [
            str(x)
            for x in encoded_names
        ]

    except Exception:

        # Fallback to generic names.
        categorical_names = [
            f"categorical_{i}"
            for i in range(
                canonical.EXPECTED_CATEGORICAL_ENCODED
            )
        ]

    # --------------------------------------------------------
    # NUMERICAL
    # --------------------------------------------------------

    numerical_names = [
        str(x)
        for x in canonical.NUMERICAL_FEATURES
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    try:

        tfidf_names = [
            str(x)
            for x in tfidf.get_feature_names_out()
        ]

    except Exception:

        tfidf_names = [
            f"tfidf_{i}"
            for i in range(
                canonical.EXPECTED_TFIDF
            )
        ]

    names = (
        categorical_names
        + numerical_names
        + tfidf_names
    )

    if len(names) != 6919:

        raise RuntimeError(
            "Feature-name count mismatch: "
            f"{len(names)} != 6919"
        )

    return names


# ============================================================
# CONTRIBUTION EXTRACTION
# ============================================================

def analyze_estimator(
    estimator,
    matrix,
    feature_names,
):

    coefficients = np.asarray(
        estimator.coef_
    ).reshape(-1)

    if coefficients.shape[0] != matrix.shape[1]:

        raise RuntimeError(
            "Coefficient count does not match "
            "feature count."
        )

    dense_row = matrix.toarray()[
        0
    ]

    contributions = (
        coefficients
        * dense_row
    )

    intercept = float(
        np.asarray(
            estimator.intercept_
        ).reshape(-1)[0]
    )

    linear_score = (
        intercept
        + float(
            contributions.sum()
        )
    )

    probability = sigmoid(
        linear_score
    )

    rows = []

    for i in range(
        len(feature_names)
    ):

        value = float(
            dense_row[i]
        )

        coefficient = float(
            coefficients[i]
        )

        contribution = float(
            contributions[i]
        )

        if (
            value == 0.0
            and coefficient == 0.0
        ):
            continue

        rows.append(
            {
                "index": i,
                "feature": feature_names[i],
                "value": value,
                "coefficient": coefficient,
                "contribution": contribution,
                "abs_contribution":
                    abs(contribution),
            }
        )

    positive = sorted(
        rows,
        key=lambda x:
            x["contribution"],
        reverse=True,
    )

    negative = sorted(
        rows,
        key=lambda x:
            x["contribution"],
    )

    absolute = sorted(
        rows,
        key=lambda x:
            x["abs_contribution"],
        reverse=True,
    )

    return {
        "intercept": intercept,
        "linear_score": linear_score,
        "sigmoid_probability": probability,
        "coefficient_count":
            int(len(coefficients)),
        "nonzero_input_features":
            int(np.count_nonzero(dense_row)),
        "nonzero_contributions":
            int(np.count_nonzero(contributions)),
        "all_rows": rows,
        "top_positive": positive[:30],
        "top_negative": negative[:30],
        "top_absolute": absolute[:50],
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "SmartSOC Stage 4 - 92057 COEFFICIENT FORENSICS"
    )
    print("=" * 100)

    print()
    print("PURPOSE")
    print(
        "Determine why the frozen SmartSOC v4.1 model "
        "selects T1070 over T1059.001 for rule 92057."
    )

    print()
    print("SAFETY")
    print(
        "Production modification : NO"
    )
    print(
        "Retraining              : NO"
    )
    print(
        "Threshold modification  : NO"
    )
    print(
        "Victim contact           : NO"
    )
    print(
        "Mitigation               : NO"
    )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("1. LOAD EXACT 92057 EVENT")
    print("=" * 100)

    event = load_exact_event()

    print(
        "Rule:",
        event["rule"]["id"]
    )

    print(
        "MITRE:",
        event["rule"]["mitre"]["id"]
    )

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("2. LOAD FROZEN PRODUCTION ARTIFACTS")
    print("=" * 100)

    artifacts = (
        canonical.load_artifacts()
    )

    model = artifacts[
        "model"
    ]

    label_encoder = artifacts[
        "label_encoder"
    ]

    classes = [
        str(x)
        for x in label_encoder.classes_
    ]

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Model features:",
        model.n_features_in_
    )

    print(
        "MITRE techniques:",
        len(classes)
    )

    # --------------------------------------------------------
    # CANONICAL PIPELINE
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("3. BUILD EXACT CANONICAL FEATURE MATRIX")
    print("=" * 100)

    normalized = (
        canonical.normalize_wazuh_event(
            event
        )
    )

    features = (
        canonical.derive_features(
            normalized
        )
    )

    feature_result = (
        canonical.build_feature_matrix(
            normalized,
            artifacts,
        )
    )

    matrix = feature_result[
        "matrix"
    ]

    print(
        "Matrix:",
        matrix.shape
    )

    print()
    print(
        "Feature layout:"
    )

    print(
        "  categorical : [0:162]"
    )

    print(
        "  numerical   : [162:220]"
    )

    print(
        "  TF-IDF      : [220:6919]"
    )

    # --------------------------------------------------------
    # VERIFY PREDICTION
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("4. VERIFY CANONICAL PREDICTION")
    print("=" * 100)

    prediction = canonical.predict(
        feature_result,
        artifacts,
    )

    for technique in TARGETS:

        for item in prediction[
            "all_scores"
        ]:

            if (
                item["technique"]
                == technique
            ):

                print(
                    f"{technique:<12} "
                    f"confidence="
                    f"{item['confidence']:.12f}"
                )

                break

    # --------------------------------------------------------
    # FEATURE NAMES
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("5. BUILD EXACT 6919 FEATURE NAMES")
    print("=" * 100)

    feature_names = (
        build_feature_names(
            feature_result,
            artifacts,
        )
    )

    print(
        "Feature names:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # SYSTEM LEVEL LOCATION
    # --------------------------------------------------------

    system_index = (
        int(
            canonical.EXPECTED_CATEGORICAL_ENCODED
        )
        + canonical.NUMERICAL_FEATURES.index(
            "system_level"
        )
    )

    system_value = float(
        matrix[
            0,
            system_index
        ]
    )

    print()
    print("=" * 100)
    print("6. SYSTEM_LEVEL FORENSICS")
    print("=" * 100)

    print(
        "system_level matrix index:",
        system_index
    )

    print(
        "system_level feature value:",
        system_value
    )

    # --------------------------------------------------------
    # TARGET ESTIMATORS
    # --------------------------------------------------------

    analysis = {}

    for technique in TARGETS:

        print()
        print("=" * 100)
        print(
            f"7. ANALYZING {technique}"
        )
        print("=" * 100)

        class_index, estimator = (
            get_estimator(
                model,
                label_encoder,
                technique,
            )
        )

        print(
            "Estimator index:",
            class_index
        )

        print(
            "Estimator type:",
            type(estimator).__name__
        )

        result = analyze_estimator(
            estimator,
            matrix,
            feature_names,
        )

        analysis[
            technique
        ] = result

        # ----------------------------------------------------
        # SYSTEM LEVEL COEFFICIENT
        # ----------------------------------------------------

        coefficient = float(
            np.asarray(
                estimator.coef_
            ).reshape(-1)[
                system_index
            ]
        )

        contribution = (
            coefficient
            * system_value
        )

        print()
        print(
            "SYSTEM_LEVEL"
        )

        print(
            "Coefficient:",
            coefficient
        )

        print(
            "Feature value:",
            system_value
        )

        print(
            "Contribution:",
            contribution
        )

        print()
        print(
            "Intercept:",
            result[
                "intercept"
            ]
        )

        print(
            "Linear score:",
            result[
                "linear_score"
            ]
        )

        print(
            "Sigmoid probability:",
            result[
                "sigmoid_probability"
            ]
        )

        # ----------------------------------------------------
        # TOP ABSOLUTE
        # ----------------------------------------------------

        print()
        print(
            "TOP 15 ABSOLUTE CONTRIBUTIONS"
        )

        for row in result[
            "top_absolute"
        ][:15]:

            print(
                f"{row['index']:4d} "
                f"{row['feature']:<45} "
                f"value={row['value']:>12.6f} "
                f"coef={row['coefficient']:>14.9f} "
                f"contrib={row['contribution']:>14.9f}"
            )

        # ----------------------------------------------------
        # TOP POSITIVE
        # ----------------------------------------------------

        print()
        print(
            "TOP 10 POSITIVE CONTRIBUTIONS"
        )

        for row in result[
            "top_positive"
        ][:10]:

            print(
                f"{row['feature']:<45} "
                f"{row['contribution']:+.9f}"
            )

        # ----------------------------------------------------
        # TOP NEGATIVE
        # ----------------------------------------------------

        print()
        print(
            "TOP 10 NEGATIVE CONTRIBUTIONS"
        )

        for row in result[
            "top_negative"
        ][:10]:

            print(
                f"{row['feature']:<45} "
                f"{row['contribution']:+.9f}"
            )

    # --------------------------------------------------------
    # DIRECT COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "8. T1070 vs T1059.001 SYSTEM_LEVEL COMPARISON"
    )
    print("=" * 100)

    comparison = {}

    for technique in TARGETS:

        estimator = get_estimator(
            model,
            label_encoder,
            technique,
        )[1]

        coefficient = float(
            np.asarray(
                estimator.coef_
            ).reshape(-1)[
                system_index
            ]
        )

        contribution = (
            coefficient
            * system_value
        )

        comparison[
            technique
        ] = {
            "coefficient":
                coefficient,
            "feature_value":
                system_value,
            "contribution":
                contribution,
            "intercept":
                analysis[
                    technique
                ][
                    "intercept"
                ],
            "linear_score":
                analysis[
                    technique
                ][
                    "linear_score"
                ],
            "sigmoid_probability":
                analysis[
                    technique
                ][
                    "sigmoid_probability"
                ],
        }

        print(
            f"{technique:<12} "
            f"coef={coefficient:+.12f} "
            f"contribution={contribution:+.12f}"
        )

    # --------------------------------------------------------
    # SCORE DIFFERENCE
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "9. SCORE DIFFERENCE"
    )
    print("=" * 100)

    t1070_score = (
        comparison[
            "T1070"
        ][
            "linear_score"
        ]
    )

    t1059_score = (
        comparison[
            "T1059.001"
        ][
            "linear_score"
        ]
    )

    score_difference = (
        t1070_score
        - t1059_score
    )

    system_contribution_difference = (
        comparison[
            "T1070"
        ][
            "contribution"
        ]
        -
        comparison[
            "T1059.001"
        ][
            "contribution"
        ]
    )

    print(
        "T1070 linear score:",
        t1070_score
    )

    print(
        "T1059.001 linear score:",
        t1059_score
    )

    print(
        "Score difference:",
        score_difference
    )

    print(
        "system_level contribution difference:",
        system_contribution_difference
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "10. SANITY CHECK"
    )
    print("=" * 100)

    sanity_pass = True

    for technique in TARGETS:

        calculated = analysis[
            technique
        ][
            "sigmoid_probability"
        ]

        canonical_value = None

        for item in prediction[
            "all_scores"
        ]:

            if (
                item["technique"]
                == technique
            ):

                canonical_value = float(
                    item["confidence"]
                )

                break

        if canonical_value is None:

            print(
                f"[FAIL] {technique}: "
                "canonical probability missing"
            )

            sanity_pass = False

            continue

        difference = abs(
            calculated
            - canonical_value
        )

        print(
            f"{technique}: "
            f"calculated={calculated:.12f} "
            f"canonical={canonical_value:.12f} "
            f"difference={difference:.12e}"
        )

        if difference > 1e-8:

            sanity_pass = False

    print()

    if sanity_pass:

        print(
            "[PASS] Linear coefficient reconstruction "
            "matches canonical prediction."
        )

    else:

        print(
            "[FAIL] Reconstruction mismatch."
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "experiment":
            "92057_coefficient_forensics_v1",

        "diagnostic_only":
            True,

        "production_modified":
            False,

        "model_retrained":
            False,

        "thresholds_modified":
            False,

        "victim_contacted":
            False,

        "mitigation_executed":
            False,

        "event":
            event,

        "feature_layout":
            {
                "categorical":
                    {
                        "start": 0,
                        "count": 162,
                    },

                "numerical":
                    {
                        "start": 162,
                        "count": 58,
                    },

                "tfidf":
                    {
                        "start": 220,
                        "count": 6699,
                    },

                "total":
                    6919,
            },

        "system_level":
            {
                "matrix_index":
                    system_index,

                "value":
                    system_value,
            },

        "canonical_prediction":
            prediction,

        "analysis":
            analysis,

        "system_level_comparison":
            comparison,

        "score_difference":
            score_difference,

        "system_contribution_difference":
            system_contribution_difference,

        "sanity_check_passed":
            sanity_pass,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "11. STAGE 4 COMPLETE"
    )
    print("=" * 100)

    print(
        "Sanity check:",
        "PASS" if sanity_pass else "FAIL"
    )

    print(
        "Production modified: NO"
    )

    print(
        "Victim contacted: NO"
    )

    print(
        "Mitigation executed: NO"
    )

    print()
    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()