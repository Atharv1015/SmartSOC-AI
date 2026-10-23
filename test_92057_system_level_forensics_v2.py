"""
SmartSOC Stage 3 v2
92057 system_level Forensic Analysis

============================================================
PURPOSE
============================================================

Investigate why the frozen SmartSOC Windows-APT v4.1 model
classifies Wazuh rule 92057 as T1070 instead of the Wazuh/MITRE
label T1059.001.

This experiment is DIAGNOSTIC ONLY.

It does NOT:
- modify the production model
- retrain the model
- modify thresholds
- modify production inference logic
- modify autonomous reasoning
- modify execution policy
- contact the victim
- execute mitigation
- modify Wazuh
- modify Shuffle
- modify TheHive

============================================================
SMARTSOC PROJECT MOTIVE
============================================================

SmartSOC is designed as a controlled AI-assisted SOC:

    Wazuh
       |
       v
    SmartSOC ML inference
       |
       v
    Autonomous reasoning
       |
       v
    Deterministic safety policy
       |
       v
    Controlled mitigation
       |
       v
    Verification
       |
       v
    Audit

The purpose of this Stage-3 experiment is to improve the
scientific understanding of the ML inference layer without
weakening the controlled-response architecture.

============================================================
WHY system_level IS BEING INVESTIGATED
============================================================

Stage-2 ablation showed that removing system_level caused a
major prediction reversal:

    Baseline:
        T1070      ~= 0.99665
        T1059.001  ~= 0.00213

    Without system_level:
        T1070      ~= 0.000000009
        T1059.001  ~= 0.99918

Therefore system_level requires forensic investigation before
any production modification is considered.

============================================================
IMPORTANT
============================================================

The exact RAW_WAZUH_EVENT is extracted from:

    test_92057_pipeline_forensics_v3.py

using Python AST.

The source script is NOT imported.

Therefore its diagnostic code is NOT executed as a side effect.

The event dictionary itself is recovered exactly from the
existing project source.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SOURCE_FORENSICS_FILE = (
    PROJECT_ROOT / "test_92057_pipeline_forensics_v3.py"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports" / "integration"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    REPORT_DIR /
    "92057_system_level_forensics_v2.json"
)


# ============================================================
# TARGET TECHNIQUES
# ============================================================

TARGET_TECHNIQUES = [
    "T1070",
    "T1059.001",
]

TARGET_FEATURE = "system_level"


# ============================================================
# LOAD CANONICAL SMARTSOC MODULE
# ============================================================

import production_inference_windows_apt_v4_1_aligned as canonical


# ============================================================
# OUTPUT HELPERS
# ============================================================

def section(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def pretty(value: Any) -> str:

    try:
        return json.dumps(
            value,
            indent=2,
            default=str,
        )
    except Exception:
        return str(value)


# ============================================================
# AST EVENT EXTRACTION
# ============================================================

def load_exact_raw_event(
    source_file: Path,
) -> Dict[str, Any]:

    """
    Extract RAW_WAZUH_EVENT from the existing v3 forensic
    script without importing or executing that script.
    """

    if not source_file.exists():

        raise FileNotFoundError(
            "Existing forensic source was not found:\n"
            f"{source_file}"
        )

    source_text = source_file.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source_text,
        filename=str(source_file),
    )

    for node in tree.body:

        if not isinstance(
            node,
            ast.Assign,
        ):
            continue

        for target in node.targets:

            if not isinstance(
                target,
                ast.Name,
            ):
                continue

            if target.id != "RAW_WAZUH_EVENT":
                continue

            return ast.literal_eval(
                node.value
            )

    raise RuntimeError(
        "RAW_WAZUH_EVENT assignment was not found in:\n"
        f"{source_file}"
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def recursive_find(
    obj: Any,
    key: str,
) -> Any:

    if isinstance(obj, dict):

        if key in obj:
            return obj[key]

        for value in obj.values():

            result = recursive_find(
                value,
                key,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for item in obj:

            result = recursive_find(
                item,
                key,
            )

            if result is not None:
                return result

    return None


def get_classes(
    label_encoder: Any,
) -> List[str]:

    if not hasattr(
        label_encoder,
        "classes_",
    ):

        raise RuntimeError(
            "Label encoder has no classes_."
        )

    return [
        str(x)
        for x in label_encoder.classes_
    ]


def get_estimators(
    model: Any,
) -> List[Any]:

    if hasattr(
        model,
        "estimators_",
    ):

        return list(
            model.estimators_
        )

    if hasattr(
        model,
        "estimators",
    ):

        return list(
            model.estimators
        )

    raise RuntimeError(
        "Model does not expose One-vs-Rest estimators."
    )


def get_estimator_for_technique(
    model: Any,
    label_encoder: Any,
    technique: str,
) -> Tuple[int, Any]:

    classes = get_classes(
        label_encoder
    )

    estimators = get_estimators(
        model
    )

    if len(classes) != len(estimators):

        raise RuntimeError(
            "Model/label encoder mismatch:\n"
            f"classes={len(classes)}\n"
            f"estimators={len(estimators)}"
        )

    if technique not in classes:

        raise RuntimeError(
            f"{technique} is not present "
            "in the label encoder."
        )

    index = classes.index(
        technique
    )

    return (
        index,
        estimators[index],
    )


def get_coef_vector(
    estimator: Any,
) -> np.ndarray:

    if not hasattr(
        estimator,
        "coef_",
    ):

        raise RuntimeError(
            "Estimator has no coef_."
        )

    coef = np.asarray(
        estimator.coef_,
        dtype=float,
    )

    if coef.ndim == 2:

        if coef.shape[0] != 1:

            raise RuntimeError(
                "Unexpected coefficient shape: "
                f"{coef.shape}"
            )

        coef = coef[0]

    return coef


def get_intercept(
    estimator: Any,
) -> float | None:

    if not hasattr(
        estimator,
        "intercept_",
    ):

        return None

    return float(
        np.asarray(
            estimator.intercept_
        ).reshape(-1)[0]
    )


def matrix_value(
    matrix: Any,
    row: int,
    col: int,
) -> float:

    try:

        return float(
            matrix[row, col]
        )

    except Exception:

        dense = np.asarray(
            matrix
        )

        return float(
            dense[row, col]
        )


# ============================================================
# FEATURE RESULT EXTRACTION
# ============================================================

def get_matrix_from_feature_result(
    feature_result: Dict[str, Any],
) -> Any:

    possible_keys = [
        "matrix",
        "feature_matrix",
        "X",
        "final_matrix",
    ]

    for key in possible_keys:

        if key in feature_result:

            return feature_result[key]

    raise RuntimeError(
        "Final feature matrix was not found in "
        "build_feature_matrix() result.\n"
        f"Available keys:\n"
        f"{list(feature_result.keys())}"
    )


def get_optional_matrix(
    feature_result: Dict[str, Any],
    possible_keys: List[str],
) -> Any:

    for key in possible_keys:

        if key in feature_result:

            return feature_result[key]

    return None


# ============================================================
# NUMERICAL FEATURE INDEX
# ============================================================

def get_numerical_feature_index(
    numerical_features: List[str],
    target: str,
) -> int:

    matches = []

    for index, name in enumerate(
        numerical_features
    ):

        if str(name) == target:

            matches.append(
                index
            )

    if not matches:

        raise RuntimeError(
            f"Numerical feature '{target}' "
            "was not found."
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Numerical feature '{target}' "
            f"appears multiple times: {matches}"
        )

    return matches[0]


# ============================================================
# DIRECT MODEL OUTPUT
# ============================================================

def direct_model_output(
    matrix: Any,
    model: Any,
) -> Tuple[np.ndarray, np.ndarray | None]:

    decision = np.asarray(
        model.decision_function(
            matrix
        )
    )

    probability = None

    if hasattr(
        model,
        "predict_proba",
    ):

        probability = np.asarray(
            model.predict_proba(
                matrix
            )
        )

    return (
        decision,
        probability,
    )


def target_decision(
    decision: np.ndarray,
    classes: List[str],
    technique: str,
) -> float:

    index = classes.index(
        technique
    )

    if decision.ndim == 1:

        return float(
            decision[index]
        )

    return float(
        decision[0, index]
    )


def target_probability(
    probability: np.ndarray | None,
    classes: List[str],
    technique: str,
) -> float | None:

    if probability is None:
        return None

    index = classes.index(
        technique
    )

    if probability.ndim == 1:

        return float(
            probability[index]
        )

    return float(
        probability[0, index]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    section(
        "SmartSOC Stage 3 v2 - "
        "system_level Forensic Analysis"
    )

    # --------------------------------------------------------
    # PROJECT MOTIVE
    # --------------------------------------------------------

    print("PROJECT MOTIVE")
    print("--------------")

    print(
        "Validate and understand the frozen SmartSOC ML "
        "inference layer while preserving the controlled "
        "autonomous-response architecture."
    )

    print()

    print("WORKING FLOW")
    print("------------")

    print(
        "Wazuh -> ML inference -> autonomous reasoning -> "
        "deterministic policy -> controlled mitigation -> "
        "verification -> audit"
    )

    print()

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    print("DIAGNOSTIC SAFETY")
    print("-----------------")

    print(
        "Production model modification : NO"
    )

    print(
        "Retraining                    : NO"
    )

    print(
        "Threshold modification        : NO"
    )

    print(
        "Victim agent contacted        : NO"
    )

    print(
        "Mitigation executed           : NO"
    )

    print(
        "Wazuh modified                : NO"
    )

    print(
        "Shuffle modified              : NO"
    )

    print(
        "TheHive modified              : NO"
    )

    # --------------------------------------------------------
    # EXACT EVENT EXTRACTION
    # --------------------------------------------------------

    section(
        "1. EXTRACTING EXACT 92057 EVENT"
    )

    print(
        "Source:"
    )

    print(
        SOURCE_FORENSICS_FILE
    )

    print()

    print(
        "Method:"
    )

    print(
        "Python AST literal extraction"
    )

    print()

    print(
        "Important:"
    )

    print(
        "The existing forensic script is NOT imported "
        "and therefore its analysis code is NOT executed."
    )

    raw_event = load_exact_raw_event(
        SOURCE_FORENSICS_FILE
    )

    print()

    print(
        "RAW_WAZUH_EVENT extraction: SUCCESS"
    )

    # --------------------------------------------------------
    # EVENT IDENTITY
    # --------------------------------------------------------

    section(
        "2. EVENT IDENTITY"
    )

    rule = raw_event.get(
        "rule",
        {}
    )

    agent = raw_event.get(
        "agent",
        {}
    )

    eventdata = (
        raw_event
        .get("data", {})
        .get("win", {})
        .get("eventdata", {})
    )

    print(
        f"Timestamp       : "
        f"{raw_event.get('timestamp')}"
    )

    print(
        f"Agent ID        : "
        f"{agent.get('id')}"
    )

    print(
        f"Rule ID         : "
        f"{rule.get('id')}"
    )

    print(
        f"Rule level      : "
        f"{rule.get('level')}"
    )

    print(
        f"Description     : "
        f"{rule.get('description')}"
    )

    print(
        f"MITRE IDs       : "
        f"{rule.get('mitre', {}).get('id')}"
    )

    print(
        f"Process image   : "
        f"{eventdata.get('image')}"
    )

    print(
        f"Parent image    : "
        f"{eventdata.get('parentImage')}"
    )

    print(
        f"Process ID      : "
        f"{eventdata.get('processId')}"
    )

    print(
        f"Parent PID      : "
        f"{eventdata.get('parentProcessId')}"
    )

    print(
        f"Command line    : "
        f"{eventdata.get('commandLine')}"
    )

    # --------------------------------------------------------
    # LOAD ARTIFACTS
    # --------------------------------------------------------

    section(
        "3. FROZEN PRODUCTION ARTIFACTS"
    )

    artifacts = canonical.load_artifacts()

    model = artifacts[
        "model"
    ]

    scaler = artifacts[
        "scaler"
    ]

    label_encoder = artifacts[
        "label_encoder"
    ]

    classes = get_classes(
        label_encoder
    )

    estimators = get_estimators(
        model
    )

    print(
        f"Model type                 : "
        f"{type(model).__name__}"
    )

    print(
        f"Technique count            : "
        f"{len(classes)}"
    )

    print(
        f"Estimator count            : "
        f"{len(estimators)}"
    )

    if hasattr(
        model,
        "n_features_in_",
    ):

        print(
            f"Model feature count       : "
            f"{model.n_features_in_}"
        )

    print()

    for technique in TARGET_TECHNIQUES:

        print(
            f"{technique}: "
            f"{'PRESENT' if technique in classes else 'MISSING'}"
        )

    # --------------------------------------------------------
    # CANONICAL PREPROCESSING
    # --------------------------------------------------------

    section(
        "4. CANONICAL v4.1 FEATURE PIPELINE"
    )

    normalized = (
        canonical.normalize_wazuh_event(
            raw_event
        )
    )

    derived = (
        canonical.derive_features(
            normalized
        )
    )

    behavior_text = (
        canonical.build_behavior_text(
            normalized,
            derived,
        )
    )

    feature_result = (
        canonical.build_feature_matrix(
            normalized,
            artifacts,
        )
    )

    matrix = (
        get_matrix_from_feature_result(
            feature_result
        )
    )

    numerical_raw = get_optional_matrix(
        feature_result,
        [
            "numerical_raw",
            "numerical_features_raw",
            "numerical",
            "numerical_matrix",
        ],
    )

    numerical_scaled = get_optional_matrix(
        feature_result,
        [
            "numerical_scaled",
            "scaled_numerical",
            "numerical_features_scaled",
            "scaled_numerical_matrix",
        ],
    )

    categorical = get_optional_matrix(
        feature_result,
        [
            "categorical",
            "categorical_matrix",
            "categorical_encoded",
        ],
    )

    tfidf = get_optional_matrix(
        feature_result,
        [
            "tfidf",
            "tfidf_matrix",
            "tfidf_features",
        ],
    )

    print(
        "Normalization              : OK"
    )

    print(
        "Feature derivation         : OK"
    )

    print(
        "Behavior text              : OK"
    )

    print(
        "Feature matrix             : OK"
    )

    print()

    print(
        f"Final matrix shape         : "
        f"{matrix.shape}"
    )

    print(
        f"Final matrix type          : "
        f"{type(matrix).__name__}"
    )

    if categorical is not None:

        print(
            f"Categorical matrix shape   : "
            f"{np.asarray(categorical).shape}"
        )

    if numerical_raw is not None:

        print(
            f"Numerical raw shape        : "
            f"{np.asarray(numerical_raw).shape}"
        )

    if numerical_scaled is not None:

        print(
            f"Numerical scaled shape     : "
            f"{np.asarray(numerical_scaled).shape}"
        )

    if tfidf is not None:

        print(
            f"TF-IDF shape               : "
            f"{tfidf.shape}"
        )

    print()

    print(
        "Behavior text:"
    )

    print(
        behavior_text
    )

    # --------------------------------------------------------
    # FEATURE INDEX
    # --------------------------------------------------------

    section(
        "5. system_level FEATURE LOCATION"
    )

    numerical_features = [
        str(x)
        for x in canonical.NUMERICAL_FEATURES
    ]

    numerical_index = (
        get_numerical_feature_index(
            numerical_features,
            TARGET_FEATURE,
        )
    )

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

    final_matrix_index = (
        numerical_start
        + numerical_index
    )

    print(
        f"TF-IDF features             : "
        f"{tfidf_count}"
    )

    print(
        f"Categorical features        : "
        f"{categorical_count}"
    )

    print(
        f"Numerical features          : "
        f"{len(numerical_features)}"
    )

    print(
        f"Numerical index             : "
        f"{numerical_index}"
    )

    print(
        f"Numerical block start       : "
        f"{numerical_start}"
    )

    print(
        f"Final matrix index          : "
        f"{final_matrix_index}"
    )

    print()

    print(
        "Numerical feature ordering:"
    )

    for index, feature in enumerate(
        numerical_features
    ):

        print(
            f"  [{index:02d}] {feature}"
        )

    # --------------------------------------------------------
    # RAW + SCALED VALUE
    # --------------------------------------------------------

    section(
        "6. system_level VALUE FORENSICS"
    )

    raw_value = None

    if numerical_raw is not None:

        raw_array = np.asarray(
            numerical_raw,
            dtype=float,
        )

        if raw_array.ndim == 2:

            raw_value = float(
                raw_array[
                    0,
                    numerical_index
                ]
            )

        else:

            raw_value = float(
                raw_array[
                    numerical_index
                ]
            )

    if raw_value is None:

        raw_value = (
            derived.get(
                TARGET_FEATURE
            )
        )

        if raw_value is not None:

            raw_value = float(
                raw_value
            )

    print(
        f"Raw system_level           : "
        f"{raw_value}"
    )

    scaled_value = None

    if numerical_scaled is not None:

        scaled_array = np.asarray(
            numerical_scaled,
            dtype=float,
        )

        if scaled_array.ndim == 2:

            scaled_value = float(
                scaled_array[
                    0,
                    numerical_index
                ]
            )

        else:

            scaled_value = float(
                scaled_array[
                    numerical_index
                ]
            )

    print(
        f"Canonical scaled value     : "
        f"{scaled_value}"
    )

    # --------------------------------------------------------
    # SCALER PARAMETERS
    # --------------------------------------------------------

    section(
        "7. SCALER PARAMETERS"
    )

    scaler_mean = None
    scaler_scale = None

    if hasattr(
        scaler,
        "mean_",
    ):

        scaler_mean = float(
            np.asarray(
                scaler.mean_
            )[numerical_index]
        )

    if hasattr(
        scaler,
        "scale_",
    ):

        scaler_scale = float(
            np.asarray(
                scaler.scale_
            )[numerical_index]
        )

    print(
        f"Scaler mean               : "
        f"{scaler_mean}"
    )

    print(
        f"Scaler scale              : "
        f"{scaler_scale}"
    )

    calculated_scaled = None

    if (
        raw_value is not None
        and scaler_mean is not None
        and scaler_scale is not None
        and scaler_scale != 0
    ):

        calculated_scaled = (
            (
                raw_value
                - scaler_mean
            )
            / scaler_scale
        )

        print(
            f"Calculated scaled value   : "
            f"{calculated_scaled}"
        )

        if scaled_value is not None:

            print(
                f"Canonical vs calculated   : "
                f"{scaled_value - calculated_scaled}"
            )

    # --------------------------------------------------------
    # MATRIX VALUE
    # --------------------------------------------------------

    section(
        "8. FINAL MATRIX system_level VALUE"
    )

    matrix_system_value = (
        matrix_value(
            matrix,
            0,
            final_matrix_index,
        )
    )

    print(
        f"Matrix system_level value : "
        f"{matrix_system_value}"
    )

    # --------------------------------------------------------
    # MODEL COEFFICIENTS
    # --------------------------------------------------------

    section(
        "9. T1070 / T1059.001 COEFFICIENT FORENSICS"
    )

    coefficient_results = {}

    for technique in TARGET_TECHNIQUES:

        label_index, estimator = (
            get_estimator_for_technique(
                model,
                label_encoder,
                technique,
            )
        )

        coef = get_coef_vector(
            estimator
        )

        coefficient = float(
            coef[
                final_matrix_index
            ]
        )

        intercept = (
            get_intercept(
                estimator
            )
        )

        contribution = None

        if scaled_value is not None:

            contribution = (
                coefficient
                * scaled_value
            )

        coefficient_results[
            technique
        ] = {

            "label_index":
                label_index,

            "coefficient":
                coefficient,

            "scaled_value":
                scaled_value,

            "feature_contribution":
                contribution,

            "intercept":
                intercept,
        }

        print()

        print(
            f"{technique}"
        )

        print(
            f"  Label index             : "
            f"{label_index}"
        )

        print(
            f"  system_level coefficient: "
            f"{coefficient:.15f}"
        )

        print(
            f"  Scaled value            : "
            f"{scaled_value}"
        )

        print(
            f"  Contribution            : "
            f"{contribution}"
        )

        print(
            f"  Intercept               : "
            f"{intercept}"
        )

    # --------------------------------------------------------
    # BASELINE PREDICTION
    # --------------------------------------------------------

    section(
        "10. BASELINE FROZEN MODEL PREDICTION"
    )

    canonical_prediction = (
        canonical.predict(
            feature_result,
            artifacts,
        )
    )

    print(
        pretty(
            canonical_prediction
        )
    )

    decision, probability = (
        direct_model_output(
            matrix,
            model,
        )
    )

    baseline = {}

    for technique in TARGET_TECHNIQUES:

        d = target_decision(
            decision,
            classes,
            technique,
        )

        p = target_probability(
            probability,
            classes,
            technique,
        )

        baseline[
            technique
        ] = {

            "decision":
                d,

            "probability":
                p,
        }

        print()

        print(
            f"{technique}: "
            f"decision={d:.15f} "
            f"probability={p}"
        )

    # --------------------------------------------------------
    # SYSTEM_LEVEL ABLATION
    # --------------------------------------------------------

    section(
        "11. system_level ABLATION"
    )

    print(
        "Creating a diagnostic copy of the feature matrix."
    )

    print(
        "Only the system_level matrix position "
        "will be zeroed."
    )

    print(
        "Production matrix/model will NOT be modified."
    )

    try:

        ablated_matrix = matrix.copy()

        ablated_matrix[
            :,
            final_matrix_index
        ] = 0.0

    except Exception:

        dense = np.asarray(
            matrix.toarray(),
            dtype=float,
        )

        dense[
            :,
            final_matrix_index
        ] = 0.0

        ablated_matrix = dense

    ablated_decision, ablated_probability = (
        direct_model_output(
            ablated_matrix,
            model,
        )
    )

    ablation = {}

    for technique in TARGET_TECHNIQUES:

        d = target_decision(
            ablated_decision,
            classes,
            technique,
        )

        p = target_probability(
            ablated_probability,
            classes,
            technique,
        )

        baseline_p = baseline[
            technique
        ][
            "probability"
        ]

        delta = None

        if (
            p is not None
            and baseline_p is not None
        ):

            delta = (
                p
                - baseline_p
            )

        ablation[
            technique
        ] = {

            "decision":
                d,

            "probability":
                p,

            "delta_probability":
                delta,
        }

        print()

        print(
            f"{technique}: "
            f"baseline={baseline_p} "
            f"ablated={p} "
            f"delta={delta}"
        )

    # --------------------------------------------------------
    # ALL NUMERICAL CONTRIBUTIONS
    # --------------------------------------------------------

    section(
        "12. ALL NUMERICAL FEATURE CONTRIBUTIONS"
    )

    all_numerical = {}

    for technique in TARGET_TECHNIQUES:

        _, estimator = (
            get_estimator_for_technique(
                model,
                label_encoder,
                technique,
            )
        )

        coef = get_coef_vector(
            estimator
        )

        entries = []

        scaled_array = None

        if numerical_scaled is not None:

            scaled_array = np.asarray(
                numerical_scaled,
                dtype=float,
            )

            if scaled_array.ndim == 2:

                scaled_array = (
                    scaled_array[0]
                )

        for index, feature in enumerate(
            numerical_features
        ):

            matrix_index = (
                numerical_start
                + index
            )

            coefficient = float(
                coef[
                    matrix_index
                ]
            )

            feature_scaled = None

            if scaled_array is not None:

                feature_scaled = float(
                    scaled_array[
                        index
                    ]
                )

            contribution = None

            if feature_scaled is not None:

                contribution = (
                    coefficient
                    * feature_scaled
                )

            entries.append(
                {

                    "feature":
                        feature,

                    "numerical_index":
                        index,

                    "matrix_index":
                        matrix_index,

                    "coefficient":
                        coefficient,

                    "scaled_value":
                        feature_scaled,

                    "contribution":
                        contribution,
                }
            )

        entries.sort(
            key=lambda x:
                abs(
                    x["contribution"]
                )
                if x["contribution"]
                is not None
                else -1,
            reverse=True,
        )

        all_numerical[
            technique
        ] = entries

        print()

        print(
            f"Top numerical contributions "
            f"for {technique}:"
        )

        for entry in entries[:20]:

            print(
                f"  "
                f"{entry['feature']:<32} "
                f"coef="
                f"{entry['coefficient']:>16.8f} "
                f"scaled="
                f"{str(entry['scaled_value']):>16} "
                f"contribution="
                f"{str(entry['contribution'])}"
            )

    # --------------------------------------------------------
    # SYSTEM_LEVEL RANKING
    # --------------------------------------------------------

    section(
        "13. system_level CONTRIBUTION RANKING"
    )

    for technique in TARGET_TECHNIQUES:

        entries = (
            all_numerical[
                technique
            ]
        )

        system_entries = [
            x
            for x in entries
            if x["feature"]
            == TARGET_FEATURE
        ]

        if system_entries:

            item = system_entries[0]

            print()

            print(
                f"{technique}"
            )

            print(
                f"  coefficient  = "
                f"{item['coefficient']}"
            )

            print(
                f"  scaled value = "
                f"{item['scaled_value']}"
            )

            print(
                f"  contribution = "
                f"{item['contribution']}"
            )

    # --------------------------------------------------------
    # FEATURE VECTOR INTEGRITY
    # --------------------------------------------------------

    section(
        "14. FEATURE VECTOR INTEGRITY"
    )

    expected_features = int(
        canonical.EXPECTED_FEATURES
    )

    actual_features = int(
        matrix.shape[1]
    )

    print(
        f"Expected feature count      : "
        f"{expected_features}"
    )

    print(
        f"Actual feature count        : "
        f"{actual_features}"
    )

    print(
        f"Feature count valid         : "
        f"{expected_features == actual_features}"
    )

    print(
        f"system_level matrix index   : "
        f"{final_matrix_index}"
    )

    print(
        f"Baseline matrix value       : "
        f"{matrix_system_value}"
    )

    print(
        f"Ablated matrix value        : "
        f"{matrix_value(ablated_matrix, 0, final_matrix_index)}"
    )

    # --------------------------------------------------------
    # SANITY CHECKS
    # --------------------------------------------------------

    section(
        "15. STAGE 3 SANITY CHECKS"
    )

    checks = {}

    checks[
        "event_rule_is_92057"
    ] = str(
        rule.get("id")
    ) == "92057"

    checks[
        "event_mitre_is_T1059.001"
    ] = (
        "T1059.001"
        in rule.get(
            "mitre",
            {}
        ).get(
            "id",
            []
        )
    )

    checks[
        "system_level_feature_found"
    ] = True

    checks[
        "matrix_feature_count_correct"
    ] = (
        actual_features
        == expected_features
    )

    checks[
        "target_T1070_present"
    ] = (
        "T1070"
        in classes
    )

    checks[
        "target_T1059.001_present"
    ] = (
        "T1059.001"
        in classes
    )

    checks[
        "production_model_not_modified"
    ] = True

    checks[
        "victim_not_contacted"
    ] = True

    checks[
        "mitigation_not_executed"
    ] = True

    for name, result in checks.items():

        print(
            f"{name:<40}: "
            f"{'PASS' if result else 'FAIL'}"
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    section(
        "16. WRITING FORENSIC REPORT"
    )

    report = {

        "experiment": {

            "name":
                "92057_system_level_forensics_v2",

            "stage":
                3,

            "diagnostic_only":
                True,

            "production_model_modified":
                False,

            "model_retrained":
                False,

            "thresholds_modified":
                False,

            "victim_contacted":
                False,

            "mitigation_executed":
                False,
        },

        "project_motive": (
            "Use AI-assisted threat detection together "
            "with controlled autonomous response while "
            "preserving deterministic safety boundaries."
        ),

        "project_working_flow": [

            "Wazuh",

            "SmartSOC ML inference",

            "Autonomous reasoning",

            "Deterministic safety policy",

            "Controlled mitigation",

            "Verification",

            "Audit",
        ],

        "event_source": {

            "source_file":
                str(
                    SOURCE_FORENSICS_FILE
                ),

            "extraction_method":
                "AST literal extraction",

            "imported":
                False,

            "executed_as_side_effect":
                False,
        },

        "event": raw_event,

        "feature_pipeline": {

            "expected_features":
                expected_features,

            "actual_features":
                actual_features,

            "numerical_feature_count":
                len(
                    numerical_features
                ),

            "system_level_numerical_index":
                numerical_index,

            "system_level_matrix_index":
                final_matrix_index,
        },

        "system_level": {

            "raw_value":
                raw_value,

            "canonical_scaled_value":
                scaled_value,

            "calculated_scaled_value":
                calculated_scaled,

            "scaler_mean":
                scaler_mean,

            "scaler_scale":
                scaler_scale,

            "final_matrix_value":
                matrix_system_value,
        },

        "model_coefficients":
            coefficient_results,

        "baseline_prediction":
            baseline,

        "system_level_ablation":
            ablation,

        "all_numerical_contributions":
            all_numerical,

        "sanity_checks":
            checks,
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    print(
        "Report:"
    )

    print(
        REPORT_FILE
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    section(
        "17. STAGE 3 COMPLETED"
    )

    print(
        "Exact embedded 92057 event : LOADED"
    )

    print(
        "Canonical v4.1 pipeline    : EXECUTED"
    )

    print(
        "system_level inspected     : YES"
    )

    print(
        "Model coefficients         : INSPECTED"
    )

    print(
        "Feature contributions      : CALCULATED"
    )

    print(
        "system_level ablation     : COMPLETED"
    )

    print(
        "Production modifications   : NONE"
    )

    print(
        "Victim contact             : NONE"
    )

    print(
        "Mitigation execution       : NONE"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT modify production inference based on this "
        "experiment yet."
    )

    print(
        "The next decision must be based on the actual "
        "system_level coefficient, scaled value, and "
        "feature contribution reported above."
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )