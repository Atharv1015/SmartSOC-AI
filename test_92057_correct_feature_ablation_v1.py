"""
SmartSOC Stage 3B
Correct 92057 Feature Ablation

IMPORTANT
---------
The canonical SmartSOC v4.1 feature matrix layout is:

    categorical : 162
    numerical   : 58
    TF-IDF      : 6699

Therefore:

    categorical = [0:162]
    numerical   = [162:220]
    TF-IDF      = [220:6919]

Previous diagnostic scripts incorrectly assumed:

    TF-IDF -> categorical -> numerical

Those previous feature-offset ablations are therefore NOT
valid as feature-specific evidence.

This script uses the ACTUAL canonical layout.

DIAGNOSTIC ONLY:
- no model modification
- no retraining
- no threshold modification
- no victim contact
- no mitigation
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np


# ============================================================
# PROJECT
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
    "92057_correct_feature_ablation_v1.json"
)


TARGETS = [
    "T1070",
    "T1059.001",
]


# ============================================================
# EXACT EVENT
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
        "RAW_WAZUH_EVENT not found."
    )


# ============================================================
# MODEL HELPERS
# ============================================================

def get_classes(
    label_encoder,
):

    return [
        str(x)
        for x in label_encoder.classes_
    ]


def get_probability(
    probabilities,
    classes,
    technique,
):

    index = classes.index(
        technique
    )

    if probabilities.ndim == 1:

        return float(
            probabilities[index]
        )

    return float(
        probabilities[
            0,
            index,
        ]
    )


def predict_direct(
    matrix,
    model,
    classes,
):

    probabilities = np.asarray(
        model.predict_proba(
            matrix
        )
    )

    result = {}

    for technique in TARGETS:

        result[
            technique
        ] = get_probability(
            probabilities,
            classes,
            technique,
        )

    return result


# ============================================================
# ABLATE ONE FEATURE
# ============================================================

def ablate_feature(
    matrix,
    matrix_index,
):

    altered = matrix.copy()

    altered[
        :,
        matrix_index
    ] = 0.0

    return altered


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print(
        "SmartSOC Stage 3B - CORRECT 92057 FEATURE ABLATION"
    )
    print("=" * 90)

    print()
    print("PROJECT MOTIVE")
    print(
        "Determine which feature representation influences "
        "SmartSOC v4.1 inference while preserving the "
        "controlled autonomous-response architecture."
    )

    print()
    print("SMARTSOC FLOW")
    print(
        "Wazuh -> ML inference -> reasoning -> deterministic "
        "policy -> controlled mitigation -> verification -> audit"
    )

    print()
    print("SAFETY")
    print(
        "Production model modified : NO"
    )
    print(
        "Model retrained            : NO"
    )
    print(
        "Threshold changed          : NO"
    )
    print(
        "Victim contacted           : NO"
    )
    print(
        "Mitigation executed        : NO"
    )

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("1. LOAD EXACT 92057 EVENT")
    print("=" * 90)

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
    print("=" * 90)
    print("2. LOAD FROZEN ARTIFACTS")
    print("=" * 90)

    artifacts = (
        canonical.load_artifacts()
    )

    model = artifacts[
        "model"
    ]

    label_encoder = artifacts[
        "label_encoder"
    ]

    classes = get_classes(
        label_encoder
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Model features:",
        model.n_features_in_
    )

    print(
        "Techniques:",
        len(classes)
    )

    # --------------------------------------------------------
    # CANONICAL PIPELINE
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("3. BUILD CANONICAL FEATURE MATRIX")
    print("=" * 90)

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

    matrix = (
        feature_result[
            "matrix"
        ]
    )

    print(
        "Matrix shape:",
        matrix.shape
    )

    print()
    print("ACTUAL FEATURE LAYOUT")
    print(
        "categorical : [0:162]"
    )
    print(
        "numerical   : [162:220]"
    )
    print(
        "TF-IDF      : [220:6919]"
    )

    # --------------------------------------------------------
    # VERIFY NUMERICAL BLOCK
    # --------------------------------------------------------

    NUMERICAL_START = int(
        canonical.EXPECTED_CATEGORICAL_ENCODED
    )

    NUMERICAL_COUNT = int(
        canonical.EXPECTED_NUMERICAL
    )

    TFIDF_START = (
        NUMERICAL_START
        + NUMERICAL_COUNT
    )

    print()
    print(
        "Numerical start:",
        NUMERICAL_START
    )

    print(
        "Numerical count:",
        NUMERICAL_COUNT
    )

    print(
        "TF-IDF start:",
        TFIDF_START
    )

    print()
    print(
        "CANONICAL NUMERICAL VALUES"
    )

    for i, feature in enumerate(
        canonical.NUMERICAL_FEATURES
    ):

        matrix_index = (
            NUMERICAL_START
            + i
        )

        value = float(
            matrix[
                0,
                matrix_index
            ]
        )

        print(
            f"{i:02d} "
            f"{feature:<35} "
            f"index={matrix_index:<5} "
            f"value={value}"
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("4. BASELINE")
    print("=" * 90)

    baseline = predict_direct(
        matrix,
        model,
        classes,
    )

    for technique in TARGETS:

        print(
            f"{technique:<12} "
            f"{baseline[technique]:.12f}"
        )

    baseline_winner = max(
        baseline,
        key=baseline.get,
    )

    print()
    print(
        "Baseline winner:",
        baseline_winner
    )

    # --------------------------------------------------------
    # NUMERICAL ABLATION
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "5. CORRECT INDIVIDUAL NUMERICAL FEATURE ABLATION"
    )
    print("=" * 90)

    results = {}

    for numerical_index, feature in enumerate(
        canonical.NUMERICAL_FEATURES
    ):

        matrix_index = (
            NUMERICAL_START
            + numerical_index
        )

        original_value = float(
            matrix[
                0,
                matrix_index
            ]
        )

        altered = ablate_feature(
            matrix,
            matrix_index,
        )

        prediction = predict_direct(
            altered,
            model,
            classes,
        )

        winner = max(
            prediction,
            key=prediction.get,
        )

        result = {

            "numerical_index":
                numerical_index,

            "matrix_index":
                matrix_index,

            "original_value":
                original_value,

            "ablated_value":
                0.0,

            "T1070":
                prediction[
                    "T1070"
                ],

            "T1059.001":
                prediction[
                    "T1059.001"
                ],

            "delta_T1070":
                prediction[
                    "T1070"
                ]
                - baseline[
                    "T1070"
                ],

            "delta_T1059.001":
                prediction[
                    "T1059.001"
                ]
                - baseline[
                    "T1059.001"
                ],

            "winner":
                winner,
        }

        results[
            feature
        ] = result

        print(
            f"{feature:<35} "
            f"T1070={result['T1070']:.12f} "
            f"T1059.001={result['T1059.001']:.12f} "
            f"winner={winner}"
        )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "6. TOP FEATURES BY T1059.001 IMPROVEMENT"
    )
    print("=" * 90)

    ranked_t1059 = sorted(
        results.items(),
        key=lambda item:
            item[1][
                "delta_T1059.001"
            ],
        reverse=True,
    )

    for feature, result in (
        ranked_t1059[:15]
    ):

        print(
            f"{feature:<35} "
            f"delta={result['delta_T1059.001']:+.12f} "
            f"ablated={result['T1059.001']:.12f}"
        )

    print()
    print("=" * 90)
    print(
        "7. TOP FEATURES BY T1070 REDUCTION"
    )
    print("=" * 90)

    ranked_t1070 = sorted(
        results.items(),
        key=lambda item:
            item[1][
                "delta_T1070"
            ],
    )

    for feature, result in (
        ranked_t1070[:15]
    ):

        print(
            f"{feature:<35} "
            f"delta={result['delta_T1070']:+.12f} "
            f"ablated={result['T1070']:.12f}"
        )

    # --------------------------------------------------------
    # WINNER FLIPS
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "8. WINNER FLIPS"
    )
    print("=" * 90)

    flips = []

    for feature, result in results.items():

        if (
            result["winner"]
            != baseline_winner
        ):

            flips.append(
                (
                    feature,
                    result,
                )
            )

    if flips:

        for feature, result in flips:

            print(
                f"{feature}: "
                f"{baseline_winner} -> "
                f"{result['winner']}"
            )

    else:

        print(
            "No individual numerical feature "
            "ablation flips the winner."
        )

    # --------------------------------------------------------
    # SYSTEM LEVEL SPECIFIC
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "9. system_level SPECIFIC RESULT"
    )
    print("=" * 90)

    system_result = results[
        "system_level"
    ]

    print(
        "Correct matrix index:",
        system_result[
            "matrix_index"
        ]
    )

    print(
        "Original value:",
        system_result[
            "original_value"
        ]
    )

    print(
        "T1070 baseline:",
        baseline[
            "T1070"
        ]
    )

    print(
        "T1070 after system_level ablation:",
        system_result[
            "T1070"
        ]
    )

    print(
        "T1059.001 baseline:",
        baseline[
            "T1059.001"
        ]
    )

    print(
        "T1059.001 after system_level ablation:",
        system_result[
            "T1059.001"
        ]
    )

    print(
        "Winner:",
        system_result[
            "winner"
        ]
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "experiment":
        "92057_correct_feature_ablation_v1",

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
            "categorical_start":
                0,

            "categorical_count":
                NUMERICAL_START,

            "numerical_start":
                NUMERICAL_START,

            "numerical_count":
                NUMERICAL_COUNT,

            "tfidf_start":
                TFIDF_START,

            "tfidf_count":
                int(
                    canonical.EXPECTED_TFIDF
                ),

            "total":
                int(
                    matrix.shape[1]
                ),
        },

        "baseline":
        baseline,

        "baseline_winner":
        baseline_winner,

        "results":
        results,

        "winner_flips":
        [
            {
                "feature":
                    feature,

                **result,
            }
            for feature, result
            in flips
        ],
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
    print("=" * 90)
    print(
        "10. STAGE 3B COMPLETE"
    )
    print("=" * 90)

    print(
        "Correct matrix layout used : YES"
    )

    print(
        "Production modified        : NO"
    )

    print(
        "Victim contacted           : NO"
    )

    print(
        "Mitigation executed        : NO"
    )

    print()
    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The previous Stage-2 feature-offset ablation "
        "must not be treated as final evidence."
    )

    print(
        "This corrected experiment uses the actual "
        "categorical -> numerical -> TF-IDF layout."
    )


if __name__ == "__main__":

    main()