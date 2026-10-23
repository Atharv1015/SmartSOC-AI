"""
SmartSOC Stage 5
92057 Scaler + Training-Distribution Forensics

Purpose
-------
Determine exactly why the canonical v4.1 pipeline produces the
system_level value observed for Wazuh rule 92057.

This is a READ-ONLY diagnostic experiment.

NO:
- production model modification
- retraining
- threshold modification
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
    "92057_scaler_forensics_v1.json"
)


# ============================================================
# HELPERS
# ============================================================

def section(title: str):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


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


def safe_float(value):

    try:

        value = float(value)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return None


def array_summary(values):

    arr = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    finite = arr[
        np.isfinite(arr)
    ]

    if finite.size == 0:

        return {
            "count": int(arr.size),
            "finite_count": 0,
        }

    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "abs_max": float(
            np.max(np.abs(finite))
        ),
    }


def percentile_rank(
    value,
    reference,
):

    reference = np.asarray(
        reference,
        dtype=float,
    ).reshape(-1)

    reference = reference[
        np.isfinite(reference)
    ]

    if reference.size == 0:
        return None

    value = float(value)

    less_equal = np.sum(
        reference <= value
    )

    return float(
        100.0
        * less_equal
        / reference.size
    )


def dump_json_safe(value):

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, dict):

        return {
            str(k):
            dump_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            dump_json_safe(v)
            for v in value
        ]

    try:

        json.dumps(value)

        return value

    except Exception:

        return repr(value)


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "SmartSOC Stage 5 - 92057 SCALER FORENSICS"
    )

    print()
    print(
        "PURPOSE"
    )

    print(
        "Determine how system_level is transformed "
        "for the real Wazuh 92057 event and whether "
        "the resulting value is unusual relative to "
        "the scaler's stored training statistics."
    )

    print()
    print(
        "SAFETY"
    )

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

    # ========================================================
    # EVENT
    # ========================================================

    section(
        "1. LOAD EXACT 92057 EVENT"
    )

    event = load_exact_event()

    print(
        "Rule:",
        event["rule"]["id"]
    )

    print(
        "MITRE:",
        event["rule"]["mitre"]["id"]
    )

    # ========================================================
    # ARTIFACTS
    # ========================================================

    section(
        "2. LOAD FROZEN PRODUCTION ARTIFACTS"
    )

    artifacts = (
        canonical.load_artifacts()
    )

    model = artifacts["model"]

    scaler = artifacts["scaler"]

    encoder = artifacts["encoder"]

    tfidf = artifacts["tfidf"]

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Scaler:",
        type(scaler).__name__
    )

    print(
        "Encoder:",
        type(encoder).__name__
    )

    print(
        "TF-IDF:",
        type(tfidf).__name__
    )

    print(
        "Scaler feature count:",
        getattr(
            scaler,
            "n_features_in_",
            "N/A",
        )
    )

    # ========================================================
    # SCALER INTERNAL STRUCTURE
    # ========================================================

    section(
        "3. SCALER INTERNAL STRUCTURE"
    )

    scaler_dict = getattr(
        scaler,
        "__dict__",
        {},
    )

    print(
        "Scaler attributes:"
    )

    for key in sorted(
        scaler_dict.keys()
    ):

        value = scaler_dict[key]

        if isinstance(
            value,
            np.ndarray,
        ):

            print(
                f"  {key}: "
                f"ndarray shape={value.shape} "
                f"dtype={value.dtype}"
            )

        else:

            print(
                f"  {key}: "
                f"{type(value).__name__} "
                f"value={value!r}"
            )

    # ========================================================
    # CANONICAL NORMALIZATION
    # ========================================================

    section(
        "4. BUILD CANONICAL FEATURES"
    )

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

    numerical_df = feature_result[
        "numerical_df"
    ]

    numerical_raw = feature_result[
        "numerical_array"
    ]

    numerical_scaled = feature_result[
        "numerical_scaled"
    ]

    print(
        "Final matrix:",
        matrix.shape
    )

    print(
        "Numerical raw:",
        numerical_raw.shape
    )

    print(
        "Numerical scaled:",
        numerical_scaled.shape
    )

    # ========================================================
    # NUMERICAL FEATURE INDEX
    # ========================================================

    numerical_features = [
        str(x)
        for x in canonical.NUMERICAL_FEATURES
    ]

    system_position = (
        numerical_features.index(
            "system_level"
        )
    )

    matrix_index = (
        canonical.EXPECTED_CATEGORICAL_ENCODED
        + system_position
    )

    raw_value = float(
        numerical_raw[
            0,
            system_position
        ]
    )

    scaled_value = float(
        numerical_scaled[
            0,
            system_position
        ]
    )

    matrix_value = float(
        matrix[
            0,
            matrix_index
        ]
    )

    print()
    print(
        "system_level position inside "
        "numerical block:",
        system_position
    )

    print(
        "system_level final matrix index:",
        matrix_index
    )

    print(
        "system_level RAW value:",
        raw_value
    )

    print(
        "system_level SCALED value:",
        scaled_value
    )

    print(
        "system_level FINAL MATRIX value:",
        matrix_value
    )

    # ========================================================
    # CONSISTENCY
    # ========================================================

    section(
        "5. CANONICAL SCALER CONSISTENCY"
    )

    scaler_direct = scaler.transform(
        numerical_raw
    )

    direct_scaled_value = float(
        scaler_direct[
            0,
            system_position
        ]
    )

    print(
        "Canonical scaled value:",
        scaled_value
    )

    print(
        "Direct scaler.transform value:",
        direct_scaled_value
    )

    print(
        "Final matrix value:",
        matrix_value
    )

    canonical_consistency = (
        abs(
            scaled_value
            - direct_scaled_value
        )
        < 1e-12
    )

    matrix_consistency = (
        abs(
            scaled_value
            - matrix_value
        )
        < 1e-12
    )

    print()

    print(
        "Scaler consistency:",
        "PASS"
        if canonical_consistency
        else "FAIL"
    )

    print(
        "Matrix consistency:",
        "PASS"
        if matrix_consistency
        else "FAIL"
    )

    # ========================================================
    # SCALER PARAMETERS
    # ========================================================

    section(
        "6. SYSTEM_LEVEL SCALER PARAMETERS"
    )

    scaler_parameters = {}

    parameter_names = [
        "mean_",
        "scale_",
        "var_",
        "data_min_",
        "data_max_",
        "data_range_",
        "center_",
        "n_samples_seen_",
    ]

    for parameter in parameter_names:

        if not hasattr(
            scaler,
            parameter,
        ):

            continue

        value = getattr(
            scaler,
            parameter,
        )

        if isinstance(
            value,
            np.ndarray,
        ):

            if (
                value.ndim == 1
                and system_position
                < value.shape[0]
            ):

                selected = value[
                    system_position
                ]

                selected = (
                    safe_float(selected)
                )

                scaler_parameters[
                    parameter
                ] = selected

                print(
                    f"{parameter}:",
                    selected
                )

            else:

                print(
                    f"{parameter}: "
                    f"shape={value.shape}"
                )

        else:

            selected = safe_float(
                value
            )

            if selected is not None:

                scaler_parameters[
                    parameter
                ] = selected

                print(
                    f"{parameter}:",
                    selected
                )

            else:

                print(
                    f"{parameter}:",
                    repr(value)
                )

    # ========================================================
    # FEATURE-WISE SCALER VALUES
    # ========================================================

    section(
        "7. ALL NUMERICAL FEATURE RAW/SCALED VALUES"
    )

    numerical_rows = []

    for i, name in enumerate(
        numerical_features
    ):

        raw = float(
            numerical_raw[
                0,
                i
            ]
        )

        scaled = float(
            numerical_scaled[
                0,
                i
            ]
        )

        row = {
            "position": i,
            "matrix_index":
                int(
                    canonical.EXPECTED_CATEGORICAL_ENCODED
                    + i
                ),
            "feature": name,
            "raw_value": raw,
            "scaled_value": scaled,
        }

        numerical_rows.append(
            row
        )

        print(
            f"{i:02d} "
            f"{name:<38} "
            f"raw={raw:>15.9f} "
            f"scaled={scaled:>15.9f}"
        )

    # ========================================================
    # SCALER TYPE-SPECIFIC INTERPRETATION
    # ========================================================

    section(
        "8. SCALER TRANSFORMATION ANALYSIS"
    )

    scaler_name = type(
        scaler
    ).__name__

    print(
        "Scaler class:",
        scaler_name
    )

    transformation_analysis = {}

    # --------------------------------------------------------
    # StandardScaler-like
    # --------------------------------------------------------

    if (
        hasattr(scaler, "mean_")
        and hasattr(scaler, "scale_")
    ):

        mean_value = float(
            np.asarray(
                scaler.mean_
            ).reshape(-1)[
                system_position
            ]
        )

        scale_value = float(
            np.asarray(
                scaler.scale_
            ).reshape(-1)[
                system_position
            ]
        )

        print(
            "Stored mean:",
            mean_value
        )

        print(
            "Stored scale:",
            scale_value
        )

        if (
            math.isfinite(scale_value)
            and scale_value != 0
        ):

            reconstructed = (
                (
                    raw_value
                    - mean_value
                )
                / scale_value
            )

            print(
                "Reconstructed "
                "(raw - mean) / scale:",
                reconstructed
            )

            print(
                "Actual scaler output:",
                scaled_value
            )

            reconstruction_error = abs(
                reconstructed
                - scaled_value
            )

            print(
                "Reconstruction error:",
                reconstruction_error
            )

            transformation_analysis[
                "formula_check"
            ] = {
                "formula":
                    "(raw - mean) / scale",

                "raw":
                    raw_value,

                "mean":
                    mean_value,

                "scale":
                    scale_value,

                "reconstructed":
                    reconstructed,

                "actual":
                    scaled_value,

                "absolute_error":
                    reconstruction_error,
            }

    # --------------------------------------------------------
    # MinMax-like
    # --------------------------------------------------------

    if (
        hasattr(scaler, "data_min_")
        and hasattr(scaler, "data_max_")
    ):

        data_min = float(
            np.asarray(
                scaler.data_min_
            ).reshape(-1)[
                system_position
            ]
        )

        data_max = float(
            np.asarray(
                scaler.data_max_
            ).reshape(-1)[
                system_position
            ]
        )

        print(
            "Stored data_min:",
            data_min
        )

        print(
            "Stored data_max:",
            data_max
        )

        transformation_analysis[
            "minmax_parameters"
        ] = {
            "data_min":
                data_min,

            "data_max":
                data_max,
        }

    # ========================================================
    # RELATIVE MAGNITUDE
    # ========================================================

    section(
        "9. SYSTEM_LEVEL MAGNITUDE ANALYSIS"
    )

    magnitude = {
        "raw_value":
            raw_value,

        "scaled_value":
            scaled_value,

        "absolute_scaled_value":
            abs(scaled_value),
    }

    print(
        "Raw value:",
        raw_value
    )

    print(
        "Scaled value:",
        scaled_value
    )

    print(
        "Absolute scaled value:",
        abs(scaled_value)
    )

    if abs(scaled_value) >= 10:

        classification = (
            "EXTREME_MAGNITUDE"
        )

    elif abs(scaled_value) >= 5:

        classification = (
            "HIGH_MAGNITUDE"
        )

    elif abs(scaled_value) >= 3:

        classification = (
            "MODERATE_MAGNITUDE"
        )

    elif abs(scaled_value) >= 1:

        classification = (
            "NORMALIZED_MAGNITUDE"
        )

    else:

        classification = (
            "LOW_MAGNITUDE"
        )

    print(
        "Diagnostic magnitude class:",
        classification
    )

    magnitude[
        "diagnostic_class"
    ] = classification

    # ========================================================
    # ALL FEATURE MAGNITUDES
    # ========================================================

    section(
        "10. SCALED NUMERICAL FEATURE MAGNITUDE RANKING"
    )

    magnitude_rows = []

    for row in numerical_rows:

        magnitude_rows.append(
            {
                **row,
                "absolute_scaled_value":
                    abs(
                        row[
                            "scaled_value"
                        ]
                    ),
            }
        )

    magnitude_rows = sorted(
        magnitude_rows,
        key=lambda x:
            x[
                "absolute_scaled_value"
            ],
        reverse=True,
    )

    for row in magnitude_rows:

        print(
            f"{row['feature']:<38} "
            f"scaled={row['scaled_value']:>15.9f} "
            f"abs={row['absolute_scaled_value']:>15.9f}"
        )

    # ========================================================
    # MODEL CONTRIBUTION LINK
    # ========================================================

    section(
        "11. LINK TO STAGE 4 MODEL CONTRIBUTION"
    )

    estimators = getattr(
        model,
        "estimators_",
        None,
    )

    classes = [
        str(x)
        for x in artifacts[
            "label_encoder"
        ].classes_
    ]

    contribution_link = {}

    for technique in [
        "T1070",
        "T1059.001",
    ]:

        class_index = classes.index(
            technique
        )

        estimator = estimators[
            class_index
        ]

        coefficient = float(
            np.asarray(
                estimator.coef_
            ).reshape(-1)[
                matrix_index
            ]
        )

        contribution = (
            coefficient
            * scaled_value
        )

        contribution_link[
            technique
        ] = {
            "coefficient":
                coefficient,

            "system_level_scaled_value":
                scaled_value,

            "contribution":
                contribution,
        }

        print(
            f"{technique:<12} "
            f"coefficient={coefficient:+.12f} "
            f"value={scaled_value:+.12f} "
            f"contribution={contribution:+.12f}"
        )

    # ========================================================
    # FINAL ASSESSMENT
    # ========================================================

    section(
        "12. STAGE 5 PRELIMINARY ASSESSMENT"
    )

    assessment = []

    if canonical_consistency:

        assessment.append(
            "Canonical numerical scaling and direct "
            "scaler.transform() agree."
        )

    else:

        assessment.append(
            "WARNING: canonical numerical scaling "
            "does not match direct scaler.transform()."
        )

    if matrix_consistency:

        assessment.append(
            "Final matrix contains the same "
            "system_level value produced by the scaler."
        )

    else:

        assessment.append(
            "WARNING: final matrix system_level "
            "does not match canonical scaler output."
        )

    if abs(scaled_value) >= 10:

        assessment.append(
            "system_level has an extreme transformed "
            "magnitude relative to ordinary normalized "
            "feature ranges."
        )

    elif abs(scaled_value) >= 5:

        assessment.append(
            "system_level has a high transformed "
            "magnitude."
        )

    else:

        assessment.append(
            "system_level does not have an extreme "
            "transformed magnitude under the diagnostic "
            "magnitude thresholds."
        )

    for item in assessment:

        print(
            "-",
            item
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "experiment":
            "92057_scaler_forensics_v1",

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

        "event_identity": {
            "rule_id":
                event["rule"]["id"],

            "mitre":
                event["rule"]["mitre"]["id"],
        },

        "artifact_types": {
            "model":
                type(model).__name__,

            "scaler":
                type(scaler).__name__,

            "encoder":
                type(encoder).__name__,

            "tfidf":
                type(tfidf).__name__,
        },

        "matrix_layout": {
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

        "system_level": {

            "numerical_position":
                system_position,

            "matrix_index":
                matrix_index,

            "raw_value":
                raw_value,

            "scaled_value":
                scaled_value,

            "final_matrix_value":
                matrix_value,

            "magnitude":
                magnitude,

            "scaler_parameters":
                scaler_parameters,
        },

        "scaler_attributes":
            dump_json_safe(
                scaler_dict
            ),

        "transformation_analysis":
            transformation_analysis,

        "numerical_features":
            numerical_rows,

        "scaled_magnitude_ranking":
            magnitude_rows,

        "stage4_contribution_link":
            contribution_link,

        "consistency": {
            "canonical_vs_direct_scaler":
                canonical_consistency,

            "canonical_vs_final_matrix":
                matrix_consistency,
        },

        "assessment":
            assessment,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dump_json_safe(report),
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    section(
        "13. STAGE 5 COMPLETE"
    )

    print(
        "Scaler consistency:",
        "PASS"
        if canonical_consistency
        else "FAIL"
    )

    print(
        "Matrix consistency:",
        "PASS"
        if matrix_consistency
        else "FAIL"
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