"""
SmartSOC Stage 3A
Canonical build_feature_matrix() key and numerical-block forensic analysis.

DIAGNOSTIC ONLY.

No production model changes.
No retraining.
No threshold changes.
No victim contact.
No mitigation execution.

Purpose:
Determine exactly how the canonical SmartSOC v4.1
build_feature_matrix() represents its numerical features,
because Stage 3 discovered:

    raw system_level       = 12.0
    scaler-derived value   = 58.3423
    reported scaled value  = 87.4620
    final matrix value     = 0.0

These values must be reconciled before interpreting
model coefficients.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


FORENSICS_SOURCE = (
    PROJECT_ROOT /
    "test_92057_pipeline_forensics_v3.py"
)


# ============================================================
# CANONICAL MODULE
# ============================================================

import production_inference_windows_apt_v4_1_aligned as canonical


# ============================================================
# EXACT EVENT EXTRACTION
# ============================================================

def extract_raw_event():

    source = FORENSICS_SOURCE.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(FORENSICS_SOURCE),
    )

    for node in tree.body:

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:

            if (
                isinstance(target, ast.Name)
                and target.id == "RAW_WAZUH_EVENT"
            ):

                return ast.literal_eval(
                    node.value
                )

    raise RuntimeError(
        "RAW_WAZUH_EVENT not found."
    )


# ============================================================
# PRINT OBJECT
# ============================================================

def describe_object(
    name: str,
    value: Any,
):

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        "Python type:",
        type(value),
    )

    if isinstance(value, dict):

        print(
            "Dictionary keys:"
        )

        for key in value.keys():

            print(
                f"  {key!r}"
            )

    else:

        try:

            print(
                "Shape:",
                value.shape,
            )

        except Exception:

            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SmartSOC Stage 3A")
    print("Canonical Feature-Matrix Key Forensics")
    print("=" * 80)

    print()
    print("PROJECT FLOW")
    print(
        "Wazuh -> ML inference -> reasoning -> "
        "policy -> controlled mitigation -> verification -> audit"
    )

    print()
    print("SAFETY")
    print("Production modification : NO")
    print("Retraining              : NO")
    print("Victim contact          : NO")
    print("Mitigation              : NO")

    # --------------------------------------------------------
    # EVENT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("1. LOAD EXACT 92057 EVENT")
    print("=" * 80)

    event = extract_raw_event()

    print(
        "Rule:",
        event["rule"]["id"],
    )

    print(
        "MITRE:",
        event["rule"]["mitre"]["id"],
    )

    eventdata = (
        event["data"]
        ["win"]
        ["eventdata"]
    )

    print(
        "Process:",
        eventdata.get("image"),
    )

    print(
        "Parent:",
        eventdata.get("parentImage"),
    )

    print(
        "PID:",
        eventdata.get("processId"),
    )

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("2. LOAD CANONICAL ARTIFACTS")
    print("=" * 80)

    artifacts = (
        canonical.load_artifacts()
    )

    print(
        "Artifact keys:"
    )

    for key in artifacts.keys():

        print(
            f"  {key}"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("3. NORMALIZATION")
    print("=" * 80)

    normalized = (
        canonical.normalize_wazuh_event(
            event
        )
    )

    print(
        json.dumps(
            normalized,
            indent=2,
            default=str,
        )
    )

    # --------------------------------------------------------
    # DERIVED FEATURES
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("4. DERIVED FEATURES")
    print("=" * 80)

    derived = (
        canonical.derive_features(
            normalized
        )
    )

    print(
        "system_level:",
        derived.get(
            "system_level"
        )
    )

    print()

    print(
        "All numerical-looking values:"
    )

    for name in canonical.NUMERICAL_FEATURES:

        print(
            f"  {name:<35} "
            f"{derived.get(name)!r}"
        )

    # --------------------------------------------------------
    # BEHAVIOR TEXT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("5. BEHAVIOR TEXT")
    print("=" * 80)

    behavior = (
        canonical.build_behavior_text(
            normalized,
            derived,
        )
    )

    print(
        behavior
    )

    # --------------------------------------------------------
    # FEATURE MATRIX
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("6. RAW build_feature_matrix() RESULT")
    print("=" * 80)

    result = (
        canonical.build_feature_matrix(
            normalized,
            artifacts,
        )
    )

    describe_object(
        "build_feature_matrix() result",
        result,
    )

    if not isinstance(
        result,
        dict,
    ):

        raise RuntimeError(
            "Expected build_feature_matrix() "
            "to return a dictionary."
        )

    print()
    print(
        "RESULT KEYS:"
    )

    for key in result.keys():

        print(
            f"  {key!r}"
        )

    # --------------------------------------------------------
    # EACH RESULT ITEM
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("7. RESULT OBJECT DETAILS")
    print("=" * 80)

    for key, value in result.items():

        print()
        print(
            f"[{key}]"
        )

        print(
            "type:",
            type(value),
        )

        if hasattr(
            value,
            "shape",
        ):

            print(
                "shape:",
                value.shape,
            )

        if hasattr(
            value,
            "dtype",
        ):

            print(
                "dtype:",
                value.dtype,
            )

        if isinstance(
            value,
            dict,
        ):

            print(
                "dict keys:",
                list(
                    value.keys()
                ),
            )

    # --------------------------------------------------------
    # NUMERICAL CANDIDATE EXTRACTION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("8. NUMERICAL CANDIDATE OBJECTS")
    print("=" * 80)

    candidates = [
        "numerical",
        "numerical_raw",
        "numerical_scaled",
        "scaled_numerical",
        "numerical_features",
        "numerical_features_raw",
        "numerical_features_scaled",
        "numerical_matrix",
        "scaled_numerical_matrix",
        "X_numerical",
        "X_numerical_scaled",
    ]

    for key in candidates:

        if key not in result:

            continue

        value = result[key]

        print()
        print(
            f"FOUND KEY: {key}"
        )

        print(
            "type:",
            type(value),
        )

        try:

            array = np.asarray(
                value
            )

            print(
                "shape:",
                array.shape,
            )

            print(
                "dtype:",
                array.dtype,
            )

            print(
                "system_level candidate value:"
            )

            if array.ndim == 2:

                print(
                    array[
                        0,
                        1,
                    ]
                )

            elif array.ndim == 1:

                print(
                    array[1]
                )

            else:

                print(
                    "Unsupported dimensionality"
                )

        except Exception as exc:

            print(
                "Could not inspect:",
                exc,
            )

    # --------------------------------------------------------
    # FINAL MATRIX
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("9. FINAL MATRIX")
    print("=" * 80)

    matrix = None

    for key in [
        "matrix",
        "feature_matrix",
        "X",
        "final_matrix",
    ]:

        if key in result:

            matrix = result[key]

            print(
                "Matrix key:",
                key,
            )

            break

    if matrix is None:

        raise RuntimeError(
            "Final matrix key could not be identified."
        )

    print(
        "Matrix type:",
        type(matrix),
    )

    print(
        "Matrix shape:",
        matrix.shape,
    )

    # --------------------------------------------------------
    # NUMERICAL BLOCK
    # --------------------------------------------------------

    tfidf_count = int(
        canonical.EXPECTED_TFIDF
    )

    categorical_count = int(
        canonical.EXPECTED_CATEGORICAL_ENCODED
    )

    numerical_start = (
        tfidf_count
        + categorical_count
    )

    print()
    print(
        "TF-IDF count:",
        tfidf_count,
    )

    print(
        "Categorical count:",
        categorical_count,
    )

    print(
        "Numerical block start:",
        numerical_start,
    )

    print()
    print(
        "FINAL NUMERICAL BLOCK:"
    )

    for index, feature in enumerate(
        canonical.NUMERICAL_FEATURES
    ):

        matrix_index = (
            numerical_start
            + index
        )

        try:

            value = float(
                matrix[
                    0,
                    matrix_index,
                ]
            )

        except Exception:

            value = float(
                np.asarray(
                    matrix
                )[
                    0,
                    matrix_index,
                ]
            )

        print(
            f"  [{index:02d}] "
            f"{feature:<35} "
            f"matrix_index={matrix_index:<5} "
            f"value={value}"
        )

    # --------------------------------------------------------
    # SCALER DIRECT TEST
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("10. DIRECT SCALER TEST")
    print("=" * 80)

    numerical = np.zeros(
        (
            1,
            len(
                canonical.NUMERICAL_FEATURES
            ),
        ),
        dtype=float,
    )

    numerical[
        0,
        1,
    ] = float(
        derived[
            "system_level"
        ]
    )

    print(
        "Input numerical vector:"
    )

    print(
        numerical
    )

    scaled = (
        artifacts[
            "scaler"
        ].transform(
            numerical
        )
    )

    print()
    print(
        "Scaler output:"
    )

    print(
        scaled
    )

    print()
    print(
        "Direct scaled system_level:",
        float(
            scaled[
                0,
                1,
            ]
        ),
    )

    # --------------------------------------------------------
    # FINAL DIAGNOSTIC
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("11. DIAGNOSTIC CONCLUSION")
    print("=" * 80)

    print(
        "We are checking whether the final matrix numerical "
        "block actually contains the scaler output."
    )

    final_system_level = float(
        matrix[
            0,
            numerical_start + 1,
        ]
    )

    direct_system_level = float(
        scaled[
            0,
            1,
        ]
    )

    print()
    print(
        "Direct scaler system_level:",
        direct_system_level,
    )

    print(
        "Final matrix system_level:",
        final_system_level,
    )

    print()

    if np.isclose(
        final_system_level,
        direct_system_level,
    ):

        print(
            "RESULT: MATCH"
        )

        print(
            "The final matrix contains the expected "
            "scaled numerical value."
        )

    else:

        print(
            "RESULT: MISMATCH"
        )

        print(
            "The final matrix does NOT contain the "
            "expected scaled system_level value."
        )

        print()

        print(
            "This mismatch must be investigated before "
            "interpreting feature coefficients."
        )

    print()
    print("=" * 80)
    print("STAGE 3A COMPLETE")
    print("=" * 80)

    print(
        "No production files were modified."
    )

    print(
        "No victim endpoint was contacted."
    )

    print(
        "No mitigation was executed."
    )


if __name__ == "__main__":

    main()