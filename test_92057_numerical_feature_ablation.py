import json
import os
import sys

import numpy as np
from scipy import sparse


# ============================================================================
# SmartSOC v4.1 — 92057 INDIVIDUAL NUMERICAL FEATURE ABLATION
# ============================================================================
#
# PURPOSE
# -------
# Identify which individual numerical features influence the 92057
# T1070 vs T1059.001 classification.
#
# DIAGNOSTIC ONLY
# ---------------
# - Frozen model is NOT modified.
# - Production artifacts are NOT modified.
# - Production inference code is NOT modified.
# - Thresholds are NOT modified.
# - All feature ablations happen in memory.
#
# EXPERIMENT
# ----------
# Baseline:
#     Exact canonical v4.1 feature matrix
#
# Then:
#     Zero ONE numerical feature at a time
#     Run canonical v4.1 predict()
#     Compare T1070 and T1059.001
#
# OUTPUT
# ------
# .\reports\integration\92057_numerical_feature_ablation_v1.json
# ============================================================================


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================================
# EXACT 92057 WAZUH EVENT
# ============================================================================

TEST_EVENT = {
    "timestamp": "2026-10-04T11:33:21.625+0000",

    "rule": {
        "id": "92057",
        "level": 12,
        "description": (
            "Powershell.exe spawned a powershell process which "
            "executed a base64 encoded command"
        ),
        "mitre": {
            "id": ["T1059.001"],
            "tactic": ["Execution"],
            "technique": ["PowerShell"],
        },
    },

    "agent": {
        "id": "001",
        "name": "SmartSOC-Windows-Victim",
        "ip": "192.168.50.20",
    },

    "data": {
        "win": {
            "eventdata": {

                "image": (
                    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
                ),

                "parentImage": (
                    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
                ),

                "commandLine": (
                    r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" '
                    r"-EncodedCommand "
                    r"UwB0AGEAcgB0AC0AUAByAG8AYwBlAHMAcwAg"
                    r"AG4AbwB0AGUAcABhAGQALgBlAHgAZQA="
                ),

                "parentCommandLine": (
                    r'"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"'
                ),

                "processId": "7048",
                "parentProcessId": "1940",

                "user": "SYSTEM",

                "integrityLevel": "High",

                "sha256": (
                    "9785001B0DCF755EDDB8AF294A373C0B87B2498660F724E76C4D53F9C217C7A3"
                ),
            },

            "system": {
                "eventID": "1",
            },
        }
    },

    "full_log": (
        "Powershell.exe spawned a powershell process which "
        "executed a base64 encoded command"
    ),
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def fmt(value):
    return f"{float(value):.9f}"


# ============================================================================
# EXTRACT FINAL MATRIX
# ============================================================================

def extract_final_matrix(feature_result):
    """
    Locate the exact final 6919-dimensional matrix.
    """

    possible_keys = [
        "matrix",
        "feature_matrix",
        "X",
        "final_matrix",
        "final_features",
    ]

    for key in possible_keys:

        if key not in feature_result:
            continue

        value = feature_result[key]

        if value is not None and hasattr(
            value,
            "shape"
        ):
            return value

    raise KeyError(
        "Could not locate final feature matrix. "
        f"Available keys: {list(feature_result.keys())}"
    )


# ============================================================================
# NUMERICAL FEATURE NAMES
# ============================================================================

def extract_numerical_feature_names(
    feature_result
):
    """
    Obtain the exact 58 numerical feature names/order.
    """

    numerical_df = feature_result.get(
        "numerical_df"
    )

    if numerical_df is None:

        raise KeyError(
            "numerical_df was not found. "
            f"Available keys: {list(feature_result.keys())}"
        )

    return list(
        numerical_df.columns
    )


# ============================================================================
# NUMERICAL BLOCK LOCATION
# ============================================================================

def locate_numerical_block(
    feature_result
):
    """
    Canonical v4.1 layout:

        categorical = 162
        numerical   = 58
        TF-IDF      = 6699

        total       = 6919
    """

    categorical_encoded = (
        feature_result.get(
            "categorical_encoded"
        )
    )

    numerical_raw = (
        feature_result.get(
            "numerical_raw"
        )
    )

    if (
        categorical_encoded is not None
        and numerical_raw is not None
    ):

        start = int(
            categorical_encoded.shape[1]
        )

        end = (
            start
            + int(
                numerical_raw.shape[1]
            )
        )

        return start, end

    # Canonical fallback.
    return 162, 220


# ============================================================================
# PREDICTION SCORE EXTRACTION
# ============================================================================

def extract_prediction_scores(
    prediction_result
):
    """
    Extract the full technique -> probability mapping.

    Canonical predict() output contains:

        detected
        all_scores
        probabilities

    Depending on the exact representation, all_scores/probabilities may be
    dictionaries, arrays, lists, tuples, pandas objects, or structured
    records. This helper handles the expected v4.1 possibilities.
    """

    # ------------------------------------------------------------------------
    # 1. all_scores
    # ------------------------------------------------------------------------

    if "all_scores" in prediction_result:

        scores = prediction_result[
            "all_scores"
        ]

        # Normal dictionary.
        if isinstance(
            scores,
            dict
        ):

            return {
                str(key): float(value)
                for key, value in scores.items()
            }

        # NumPy array.
        if isinstance(
            scores,
            np.ndarray
        ):

            return convert_array_scores(
                scores
            )

        # List / tuple.
        if isinstance(
            scores,
            (list, tuple)
        ):

            extracted = (
                convert_structured_scores(
                    scores
                )
            )

            if extracted:
                return extracted

    # ------------------------------------------------------------------------
    # 2. probabilities
    # ------------------------------------------------------------------------

    if "probabilities" in prediction_result:

        scores = prediction_result[
            "probabilities"
        ]

        if isinstance(
            scores,
            dict
        ):

            return {
                str(key): float(value)
                for key, value in scores.items()
            }

        if isinstance(
            scores,
            np.ndarray
        ):

            return convert_array_scores(
                scores
            )

        if isinstance(
            scores,
            (list, tuple)
        ):

            extracted = (
                convert_structured_scores(
                    scores
                )
            )

            if extracted:
                return extracted

    # ------------------------------------------------------------------------
    # 3. Detailed failure
    # ------------------------------------------------------------------------

    raise KeyError(
        "Could not extract prediction scores.\n"
        f"Available keys: "
        f"{list(prediction_result.keys())}\n"
        f"all_scores type: "
        f"{type(prediction_result.get('all_scores')).__name__}\n"
        f"probabilities type: "
        f"{type(prediction_result.get('probabilities')).__name__}"
    )


# ============================================================================
# STRUCTURED SCORE CONVERTER
# ============================================================================

def convert_structured_scores(
    scores
):
    """
    Convert list/tuple structured prediction scores into:

        {
            "T1070": probability,
            "T1059.001": probability,
            ...
        }
    """

    extracted = {}

    for item in scores:

        # ------------------------------------------------------------
        # Dictionary record
        # ------------------------------------------------------------

        if isinstance(
            item,
            dict
        ):

            technique = (
                item.get("technique")
                or item.get("id")
                or item.get("label")
            )

            value = None

            if item.get(
                "confidence"
            ) is not None:

                value = item.get(
                    "confidence"
                )

            elif item.get(
                "probability"
            ) is not None:

                value = item.get(
                    "probability"
                )

            elif item.get(
                "score"
            ) is not None:

                value = item.get(
                    "score"
                )

            if (
                technique is not None
                and value is not None
            ):

                extracted[
                    str(technique)
                ] = float(value)

        # ------------------------------------------------------------
        # Two-element tuple/list:
        #     ("T1070", 0.99)
        # ------------------------------------------------------------

        elif isinstance(
            item,
            (tuple, list)
        ):

            if len(item) >= 2:

                first = item[0]
                second = item[1]

                if (
                    isinstance(
                        first,
                        str
                    )
                    and isinstance(
                        second,
                        (int, float, np.number)
                    )
                ):

                    extracted[
                        first
                    ] = float(second)

    return extracted


# ============================================================================
# ARRAY SCORE CONVERTER
# ============================================================================

def convert_array_scores(
    scores
):
    """
    Convert a NumPy probability array using the model label encoder.

    The label encoder is supplied later through the global helper wrapper,
    so this function is intentionally not used directly unless the array
    carries technique metadata.
    """

    # A bare probability array cannot safely be mapped to technique labels
    # without the label encoder. Return an empty result so the caller can
    # continue to the safer structured path.
    return {}


# ============================================================================
# RUN CANONICAL PREDICTION
# ============================================================================

def run_canonical_prediction(
    module,
    feature_result,
    artifacts
):
    """
    Always use the actual production-aligned predict() function.
    """

    prediction = module.predict(
        feature_result,
        artifacts
    )

    if not isinstance(
        prediction,
        dict
    ):

        raise TypeError(
            "Canonical predict() did not return a dictionary. "
            f"Received: {type(prediction).__name__}"
        )

    return prediction


# ============================================================================
# SCORE EXTRACTION WITH LABEL ENCODER FALLBACK
# ============================================================================

def get_all_scores(
    prediction_result,
    label_encoder
):
    """
    Extract probabilities and, if predict() returns an array, map them
    using the exact production label encoder.
    """

    # First try structured representations.
    try:

        scores = (
            extract_prediction_scores(
                prediction_result
            )
        )

        if scores:
            return scores

    except KeyError:
        pass

    # ------------------------------------------------------------------------
    # If probabilities is an array, map it through the production
    # multilabel encoder.
    # ------------------------------------------------------------------------

    probabilities = prediction_result.get(
        "probabilities"
    )

    if probabilities is None:

        raise KeyError(
            "No usable probabilities found in prediction result."
        )

    array = np.asarray(
        probabilities
    )

    # Handle shape (1, n).
    if array.ndim == 2:

        if array.shape[0] != 1:

            raise ValueError(
                "Expected one-row prediction, "
                f"got shape {array.shape}"
            )

        array = array[0]

    classes = list(
        label_encoder.classes_
    )

    if len(array) != len(classes):

        raise ValueError(
            "Probability count does not match "
            "label encoder classes.\n"
            f"Probabilities: {len(array)}\n"
            f"Classes: {len(classes)}"
        )

    return {
        str(classes[index]): float(
            array[index]
        )
        for index in range(
            len(classes)
        )
    }


# ============================================================================
# GET ONE TECHNIQUE SCORE
# ============================================================================

def get_score(
    prediction_result,
    label_encoder,
    technique
):

    scores = get_all_scores(
        prediction_result,
        label_encoder
    )

    return float(
        scores.get(
            technique,
            0.0
        )
    )


# ============================================================================
# GET WINNER
# ============================================================================

def get_winner(
    prediction_result,
    label_encoder
):

    scores = get_all_scores(
        prediction_result,
        label_encoder
    )

    return max(
        scores.items(),
        key=lambda item: item[1]
    )[0]


# ============================================================================
# ABLATE ONE FEATURE
# ============================================================================

def ablate_one_feature(
    baseline_matrix,
    numerical_start,
    feature_index
):
    """
    Zero exactly one numerical feature in an in-memory copy.
    """

    absolute_index = (
        numerical_start
        + feature_index
    )

    if sparse.issparse(
        baseline_matrix
    ):

        modified = (
            baseline_matrix.tolil(
                copy=True
            )
        )

        modified[
            0,
            absolute_index
        ] = 0.0

        modified = (
            modified.tocsr()
        )

    else:

        modified = np.array(
            baseline_matrix,
            copy=True
        )

        modified[
            0,
            absolute_index
        ] = 0.0

    return modified


# ============================================================================
# REPLACE FINAL MATRIX
# ============================================================================

def replace_final_matrix(
    feature_result,
    modified_matrix
):
    """
    Make an in-memory copy of feature_result and replace ONLY its final
    feature matrix.
    """

    modified_result = dict(
        feature_result
    )

    possible_keys = [
        "matrix",
        "feature_matrix",
        "X",
        "final_matrix",
        "final_features",
    ]

    for key in possible_keys:

        if key in modified_result:

            modified_result[
                key
            ] = modified_matrix

            return modified_result

    raise KeyError(
        "Could not find final matrix key. "
        f"Available keys: {list(feature_result.keys())}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 90)
    print(
        "SmartSOC v4.1 — "
        "92057 INDIVIDUAL NUMERICAL FEATURE ABLATION"
    )
    print("=" * 90)

    print()
    print("IMPORTANT:")
    print("This is a diagnostic only.")
    print("Frozen model/artifacts are NOT modified.")
    print("Production inference code is NOT modified.")
    print("Thresholds are NOT modified.")
    print(
        "All ablations are performed on "
        "in-memory feature matrices."
    )
    print()

    # ========================================================================
    # 1. LOAD ALIGNED MODULE
    # ========================================================================

    print(
        "[1] LOADING EXISTING ALIGNED PRODUCTION MODULE"
    )
    print()

    import production_inference_windows_apt_v4_1_aligned as module

    print(
        "Aligned module:"
    )

    print(
        module.__file__
    )

    print()
    print(
        "Checking canonical production functions:"
    )

    required_functions = [
        "load_artifacts",
        "normalize_wazuh_event",
        "derive_features",
        "build_feature_matrix",
        "predict",
    ]

    for name in required_functions:

        if not hasattr(
            module,
            name
        ):

            raise ImportError(
                "Required canonical function missing: "
                f"{name}"
            )

        print(
            f"  [OK] {name}"
        )

    # ========================================================================
    # 2. LOAD ARTIFACTS
    # ========================================================================

    print()
    print(
        "[2] LOADING PRODUCTION ARTIFACTS"
    )
    print()

    artifacts = (
        module.load_artifacts()
    )

    if not isinstance(
        artifacts,
        dict
    ):

        raise TypeError(
            "load_artifacts() did not return dict. "
            f"Received: {type(artifacts).__name__}"
        )

    print(
        "Production artifacts loaded successfully."
    )

    print()
    print(
        "Artifact keys:"
    )

    for key in artifacts.keys():

        print(
            f"  {key}"
        )

    model = artifacts[
        "model"
    ]

    label_encoder = artifacts[
        "label_encoder"
    ]

    print()
    print(
        "Model type: "
        f"{type(model).__name__}"
    )

    print(
        "Model features: "
        f"{model.n_features_in_}"
    )

    print(
        "MITRE techniques: "
        f"{len(label_encoder.classes_)}"
    )

    # ========================================================================
    # 3. BUILD EXACT FEATURE REPRESENTATION
    # ========================================================================

    print()
    print(
        "[3] BUILDING EXACT v4.1 FEATURE REPRESENTATION"
    )
    print()

    normalized_event = (
        module.normalize_wazuh_event(
            TEST_EVENT
        )
    )

    derived_features = (
        module.derive_features(
            normalized_event
        )
    )

    print(
        "Derived feature count: "
        f"{len(derived_features)}"
    )

    feature_result = (
        module.build_feature_matrix(
            normalized_event,
            artifacts
        )
    )

    baseline_matrix = (
        extract_final_matrix(
            feature_result
        )
    )

    print()
    print(
        "Original matrix shape: "
        f"{baseline_matrix.shape}"
    )

    if sparse.issparse(
        baseline_matrix
    ):

        print(
            "Original non-zero: "
            f"{baseline_matrix.nnz}"
        )

    else:

        print(
            "Original non-zero: "
            f"{np.count_nonzero(baseline_matrix)}"
        )

    numerical_start, numerical_end = (
        locate_numerical_block(
            feature_result
        )
    )

    numerical_feature_names = (
        extract_numerical_feature_names(
            feature_result
        )
    )

    if len(
        numerical_feature_names
    ) != 58:

        raise RuntimeError(
            "Expected exactly 58 numerical features. "
            f"Found {len(numerical_feature_names)}."
        )

    print()
    print(
        "Feature layout:"
    )

    print(
        f"  categorical : "
        f"[0:{numerical_start}]"
    )

    print(
        f"  numerical   : "
        f"[{numerical_start}:{numerical_end}]"
    )

    print(
        f"  TF-IDF      : "
        f"[{numerical_end}:{baseline_matrix.shape[1]}]"
    )

    print()
    print(
        "Numerical feature count: "
        f"{len(numerical_feature_names)}"
    )

    # ========================================================================
    # 4. BASELINE
    # ========================================================================

    print()
    print(
        "[4] BASELINE"
    )

    print(
        "-" * 90
    )

    baseline_prediction = (
        run_canonical_prediction(
            module,
            feature_result,
            artifacts
        )
    )

    # Print actual structure once for transparency.
    print(
        "Prediction keys: "
        f"{list(baseline_prediction.keys())}"
    )

    print(
        "all_scores type: "
        f"{type(baseline_prediction.get('all_scores')).__name__}"
    )

    print(
        "probabilities type: "
        f"{type(baseline_prediction.get('probabilities')).__name__}"
    )

    baseline_t1070 = get_score(
        baseline_prediction,
        label_encoder,
        "T1070"
    )

    baseline_t1059 = get_score(
        baseline_prediction,
        label_encoder,
        "T1059.001"
    )

    baseline_winner = get_winner(
        baseline_prediction,
        label_encoder
    )

    print()
    print(
        "BASELINE — ALL FEATURES "
        f"T1070={fmt(baseline_t1070)} "
        f"T1059.001={fmt(baseline_t1059)} "
        f"winner={baseline_winner}"
    )

    # ========================================================================
    # 5. INDIVIDUAL FEATURE ABLATION
    # ========================================================================

    print()
    print(
        "[5] INDIVIDUAL NUMERICAL FEATURE ABLATIONS"
    )

    print(
        "-" * 90
    )

    results = []

    for index, feature_name in enumerate(
        numerical_feature_names
    ):

        modified_matrix = (
            ablate_one_feature(
                baseline_matrix,
                numerical_start,
                index
            )
        )

        modified_feature_result = (
            replace_final_matrix(
                feature_result,
                modified_matrix
            )
        )

        prediction = (
            run_canonical_prediction(
                module,
                modified_feature_result,
                artifacts
            )
        )

        t1070 = get_score(
            prediction,
            label_encoder,
            "T1070"
        )

        t1059 = get_score(
            prediction,
            label_encoder,
            "T1059.001"
        )

        delta_t1070 = (
            t1070
            - baseline_t1070
        )

        delta_t1059 = (
            t1059
            - baseline_t1059
        )

        winner = get_winner(
            prediction,
            label_encoder
        )

        item = {
            "index": index,
            "feature": feature_name,

            "baseline_t1070": baseline_t1070,
            "ablated_t1070": t1070,
            "delta_t1070": delta_t1070,
            "abs_delta_t1070": abs(
                delta_t1070
            ),

            "baseline_t1059_001": baseline_t1059,
            "ablated_t1059_001": t1059,
            "delta_t1059_001": delta_t1059,
            "abs_delta_t1059_001": abs(
                delta_t1059
            ),

            "winner": winner
        }

        results.append(
            item
        )

        print(
            f"{index:02d}  "
            f"{feature_name:<34} "
            f"T1070={fmt(t1070)} "
            f"T1059.001={fmt(t1059)} "
            f"winner={winner}"
        )

    # ========================================================================
    # 6. RANKINGS
    # ========================================================================

    by_abs_t1059 = sorted(
        results,
        key=lambda x: x[
            "abs_delta_t1059_001"
        ],
        reverse=True
    )

    by_abs_t1070 = sorted(
        results,
        key=lambda x: x[
            "abs_delta_t1070"
        ],
        reverse=True
    )

    by_t1059_improvement = sorted(
        results,
        key=lambda x: x[
            "delta_t1059_001"
        ],
        reverse=True
    )

    by_t1070_reduction = sorted(
        results,
        key=lambda x: x[
            "delta_t1070"
        ]
    )

    # ========================================================================
    # 7. TOP T1059.001 IMPACT
    # ========================================================================

    print()
    print("=" * 90)
    print(
        "TOP 15 FEATURES BY ABSOLUTE T1059.001 IMPACT"
    )
    print("=" * 90)

    for rank, item in enumerate(
        by_abs_t1059[:15],
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{item['feature']:<34} "
            f"ΔT1059.001="
            f"{item['delta_t1059_001']:+.9f} "
            f"ablated="
            f"{item['ablated_t1059_001']:.9f}"
        )

    # ========================================================================
    # 8. TOP T1070 IMPACT
    # ========================================================================

    print()
    print("=" * 90)
    print(
        "TOP 15 FEATURES BY ABSOLUTE T1070 IMPACT"
    )
    print("=" * 90)

    for rank, item in enumerate(
        by_abs_t1070[:15],
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{item['feature']:<34} "
            f"ΔT1070="
            f"{item['delta_t1070']:+.9f} "
            f"ablated="
            f"{item['ablated_t1070']:.9f}"
        )

    # ========================================================================
    # 9. FEATURES THAT INCREASE T1059.001
    # ========================================================================

    print()
    print("=" * 90)
    print(
        "TOP 15 FEATURES THAT INCREASE T1059.001 WHEN REMOVED"
    )
    print("=" * 90)

    for rank, item in enumerate(
        by_t1059_improvement[:15],
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{item['feature']:<34} "
            f"ΔT1059.001="
            f"{item['delta_t1059_001']:+.9f} "
            f"ablated="
            f"{item['ablated_t1059_001']:.9f}"
        )

    # ========================================================================
    # 10. FEATURES THAT REDUCE T1070
    # ========================================================================

    print()
    print("=" * 90)
    print(
        "TOP 15 FEATURES WHOSE REMOVAL REDUCES T1070"
    )
    print("=" * 90)

    for rank, item in enumerate(
        by_t1070_reduction[:15],
        start=1
    ):

        print(
            f"{rank:02d}. "
            f"{item['feature']:<34} "
            f"ΔT1070="
            f"{item['delta_t1070']:+.9f} "
            f"ablated="
            f"{item['ablated_t1070']:.9f}"
        )

    # ========================================================================
    # 11. WINNER FLIPS
    # ========================================================================

    winner_flips = [
        item
        for item in results
        if item["winner"]
        != baseline_winner
    ]

    print()
    print("=" * 90)
    print(
        "WINNER FLIPS"
    )
    print("=" * 90)

    if winner_flips:

        print(
            f"Baseline winner: "
            f"{baseline_winner}"
        )

        for item in winner_flips:

            print(
                f"{item['index']:02d} "
                f"{item['feature']:<34} "
                f"winner={item['winner']} "
                f"T1070="
                f"{item['ablated_t1070']:.9f} "
                f"T1059.001="
                f"{item['ablated_t1059_001']:.9f}"
            )

    else:

        print(
            "No individual numerical feature removal "
            "changed the winning technique."
        )

    # ========================================================================
    # 12. STRONG T1059.001 RECOVERY
    # ========================================================================

    strong_recovery = [
        item
        for item in results
        if item[
            "ablated_t1059_001"
        ] >= 0.50
    ]

    print()
    print("=" * 90)
    print(
        "FEATURE REMOVALS PRODUCING T1059.001 >= 0.50"
    )
    print("=" * 90)

    if strong_recovery:

        for item in strong_recovery:

            print(
                f"{item['feature']:<34} "
                f"T1059.001="
                f"{item['ablated_t1059_001']:.9f} "
                f"T1070="
                f"{item['ablated_t1070']:.9f} "
                f"winner={item['winner']}"
            )

    else:

        print(
            "No single numerical feature removal "
            "produced T1059.001 >= 0.50."
        )

    # ========================================================================
    # 13. SAVE REPORT
    # ========================================================================

    output_path = os.path.join(
        PROJECT_ROOT,
        "reports",
        "integration",
        "92057_numerical_feature_ablation_v1.json"
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    report = {
        "experiment":
            "92057_numerical_feature_ablation_v1",

        "diagnostic_only": True,

        "production_model_modified":
            False,

        "production_artifacts_modified":
            False,

        "production_inference_modified":
            False,

        "thresholds_modified":
            False,

        "model_features":
            int(
                model.n_features_in_
            ),

        "mitre_techniques":
            int(
                len(
                    label_encoder.classes_
                )
            ),

        "matrix_shape":
            list(
                baseline_matrix.shape
            ),

        "numerical_feature_count":
            len(
                numerical_feature_names
            ),

        "numerical_feature_names":
            numerical_feature_names,

        "baseline": {
            "T1070":
                baseline_t1070,

            "T1059.001":
                baseline_t1059,

            "winner":
                baseline_winner
        },

        "individual_ablation_results":
            results,

        "top_by_absolute_T1059.001_impact":
            by_abs_t1059[:15],

        "top_by_absolute_T1070_impact":
            by_abs_t1070[:15],

        "top_by_T1059.001_improvement":
            by_t1059_improvement[:15],

        "top_by_T1070_reduction":
            by_t1070_reduction[:15],

        "winner_flips":
            winner_flips,

        "T1059.001_ge_0.50_after_ablation":
            strong_recovery
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2
        )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print()
    print("=" * 90)
    print(
        "INDIVIDUAL NUMERICAL FEATURE ABLATION COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        "Diagnostic report written to:"
    )

    print(
        r".\reports\integration\92057_numerical_feature_ablation_v1.json"
    )

    print()
    print(
        "Production model/artifacts were not modified."
    )

    print(
        "Production inference code was not modified."
    )

    print()
    print(
        "Do NOT make production changes based "
        "on this experiment yet."
    )


if __name__ == "__main__":
    main()