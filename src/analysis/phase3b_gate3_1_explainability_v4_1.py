from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)


ROOT = Path(__file__).resolve().parents[2]

PROD = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"

REPORT_DIR = ROOT / "reports" / "phase3b_gate3_1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    REPORT_DIR /
    "phase3b_gate3_1_explainability_report.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_schema_columns(schema, encoder, scaler):
    """
    Recover the exact production column definitions.

    The production feature_schema.json does not necessarily use
    the key names assumed by earlier draft scripts, so we first
    inspect the schema and then use the fitted transformers as the
    authoritative source where appropriate.
    """

    print("\nFEATURE SCHEMA KEYS")

    for key in schema.keys():
        print(" ", key)

    # ------------------------------------------------------------
    # Categorical columns
    # ------------------------------------------------------------

    categorical_columns = None

    possible_categorical_keys = [
        "categorical_columns",
        "categorical_features",
        "categorical",
        "categorical_feature_names",
    ]

    for key in possible_categorical_keys:
        value = schema.get(key)

        if isinstance(value, list) and value:
            categorical_columns = [
                str(x) for x in value
            ]
            break

    # The fitted encoder is authoritative if the schema key
    # names differ.
    if categorical_columns is None:
        if hasattr(
            encoder,
            "feature_names_in_"
        ):
            categorical_columns = [
                str(x)
                for x in encoder.feature_names_in_
            ]

    if categorical_columns is None:
        raise RuntimeError(
            "Unable to determine categorical columns."
        )

    # ------------------------------------------------------------
    # Numerical columns
    # ------------------------------------------------------------

    numerical_columns = None

    possible_numerical_keys = [
        "numerical_columns",
        "numerical_features",
        "numerical",
        "numerical_feature_names",
    ]

    for key in possible_numerical_keys:
        value = schema.get(key)

        if isinstance(value, list) and value:
            numerical_columns = [
                str(x) for x in value
            ]
            break

    if numerical_columns is None:
        if hasattr(
            scaler,
            "feature_names_in_"
        ):
            numerical_columns = [
                str(x)
                for x in scaler.feature_names_in_
            ]

    if numerical_columns is None:
        raise RuntimeError(
            "Unable to determine numerical columns."
        )

    return (
        categorical_columns,
        numerical_columns
    )


def get_threshold_map(path):

    obj = load_json(path)

    if not isinstance(obj, dict):
        raise RuntimeError(
            "thresholds.json is not a dictionary."
        )

    # Direct nested thresholds map
    if isinstance(
        obj.get("thresholds"),
        dict
    ):
        values = obj["thresholds"]

        numeric = {
            str(k): float(v)
            for k, v in values.items()
            if isinstance(v, (int, float))
        }

        if numeric:
            return numeric

    # Search nested dictionaries
    for _, value in obj.items():

        if not isinstance(value, dict):
            continue

        numeric = {}

        for k, v in value.items():

            if (
                isinstance(k, str)
                and k.startswith("T")
                and isinstance(v, (int, float))
            ):
                numeric[k] = float(v)

        if len(numeric) >= 5:
            return numeric

    # Search top-level
    numeric = {}

    for k, v in obj.items():

        if (
            isinstance(k, str)
            and k.startswith("T")
            and isinstance(v, (int, float))
        ):
            numeric[k] = float(v)

    if numeric:
        return numeric

    raise RuntimeError(
        "Could not extract technique thresholds."
    )


def get_feature_names(
    encoder,
    numerical_columns,
    tfidf
):
    categorical_names = list(
        encoder.get_feature_names_out()
    )

    numerical_names = [
        str(x)
        for x in numerical_columns
    ]

    tfidf_names = [
        f"tfidf:{x}"
        for x in tfidf.get_feature_names_out()
    ]

    feature_names = (
        categorical_names
        + numerical_names
        + tfidf_names
    )

    return (
        feature_names,
        len(categorical_names),
        len(numerical_names),
        len(tfidf_names)
    )


def get_coefficients(model):

    if not hasattr(
        model,
        "estimators_"
    ):
        raise RuntimeError(
            "Model does not expose estimators_."
        )

    rows = []

    for estimator in model.estimators_:

        if not hasattr(
            estimator,
            "coef_"
        ):
            raise RuntimeError(
                "Estimator does not expose coef_."
            )

        coef = np.asarray(
            estimator.coef_
        )

        if coef.ndim != 2:
            raise RuntimeError(
                f"Unexpected coefficient shape: {coef.shape}"
            )

        rows.append(
            coef[0]
        )

    return np.vstack(rows)


def transform_row(
    structured_row,
    behavior_text,
    categorical_columns,
    numerical_columns,
    encoder,
    scaler,
    tfidf
):

    cat_df = pd.DataFrame(
        [
            {
                column: structured_row[column]
                for column in categorical_columns
            }
        ]
    )

    num_df = pd.DataFrame(
        [
            {
                column: structured_row[column]
                for column in numerical_columns
            }
        ]
    )

    X_cat = encoder.transform(
        cat_df
    )

    X_num = scaler.transform(
        num_df
    )

    X_text = tfidf.transform(
        [behavior_text]
    )

    X_final = sparse.hstack(
        [
            sparse.csr_matrix(X_cat),
            sparse.csr_matrix(X_num),
            X_text
        ],
        format="csr"
    )

    return (
        X_cat,
        X_num,
        X_text,
        X_final
    )


def normalize_probability_output(
    probability
):

    if isinstance(
        probability,
        list
    ):
        probability = np.column_stack(
            [
                p[:, 1]
                if p.ndim == 2
                and p.shape[1] > 1
                else p.ravel()
                for p in probability
            ]
        )

    return np.asarray(
        probability,
        dtype=np.float64
    )


def main():

    print("=" * 72)
    print("SMARTSOC PHASE 3B")
    print("GATE 3B.1 — EXPLAINABILITY & EVIDENCE EXTRACTION")
    print("=" * 72)

    report = {
        "phase": "3B",
        "gate": "3B.1",
        "production_artifacts_modified": False,
        "status": "FAIL"
    }

    # ============================================================
    # LOAD ARTIFACTS
    # ============================================================

    print("\nPRODUCTION ARTIFACTS")

    model = joblib.load(
        PROD / "model.joblib"
    )

    encoder = joblib.load(
        PROD / "structured_encoder.joblib"
    )

    scaler = joblib.load(
        PROD / "numerical_scaler.joblib"
    )

    tfidf = joblib.load(
        PROD / "tfidf_vectorizer.joblib"
    )

    multilabel = joblib.load(
        PROD / "multilabel_encoder.joblib"
    )

    schema = load_json(
        PROD / "feature_schema.json"
    )

    thresholds = get_threshold_map(
        PROD / "thresholds.json"
    )

    for name in [
        "model.joblib",
        "structured_encoder.joblib",
        "numerical_scaler.joblib",
        "tfidf_vectorizer.joblib",
        "multilabel_encoder.joblib",
        "feature_schema.json",
        "thresholds.json"
    ]:
        print(
            "[PASS]",
            name
        )

    # ============================================================
    # COLUMNS
    # ============================================================

    (
        categorical_columns,
        numerical_columns
    ) = find_schema_columns(
        schema,
        encoder,
        scaler
    )

    techniques = [
        str(x)
        for x in multilabel.classes_
    ]

    print("\nSTRUCTURED FEATURES")

    print(
        "  Categorical:",
        len(categorical_columns)
    )

    print(
        "  Numerical  :",
        len(numerical_columns)
    )

    print(
        "  Combined   :",
        len(categorical_columns)
        + len(numerical_columns)
    )

    print(
        "  Techniques :",
        len(techniques)
    )

    if len(categorical_columns) != 4:
        raise RuntimeError(
            "Expected 4 categorical features."
        )

    if len(numerical_columns) != 58:
        raise RuntimeError(
            "Expected 58 numerical features."
        )

    if len(techniques) != 45:
        raise RuntimeError(
            "Expected 45 MITRE techniques."
        )

    # ============================================================
    # FEATURE NAMES
    # ============================================================

    (
        feature_names,
        categorical_encoded_count,
        numerical_count,
        tfidf_count
    ) = get_feature_names(
        encoder,
        numerical_columns,
        tfidf
    )

    print("\nFEATURE MAPPING")

    print(
        "  Categorical encoded:",
        categorical_encoded_count
    )

    print(
        "  Numerical          :",
        numerical_count
    )

    print(
        "  TF-IDF             :",
        tfidf_count
    )

    print(
        "  Final              :",
        len(feature_names)
    )

    if len(feature_names) != 6919:
        raise RuntimeError(
            f"Expected 6919 feature names, "
            f"got {len(feature_names)}"
        )

    print(
        "[PASS] Feature mapping"
    )

    # ============================================================
    # LOAD TEST DATA
    # ============================================================

    structured_path = (
        DATA / "structured_test.csv"
    )

    behavior_path = (
        DATA / "behavior_text_test.csv"
    )

    structured_df = pd.read_csv(
        structured_path
    )

    behavior_df = pd.read_csv(
        behavior_path
    )

    print("\nINPUT DATA")

    print(
        "  Structured rows:",
        len(structured_df)
    )

    print(
        "  Behavior rows  :",
        len(behavior_df)
    )

    if len(structured_df) != len(behavior_df):
        raise RuntimeError(
            "Input row alignment failure."
        )

    # ============================================================
    # BEHAVIOR COLUMN
    # ============================================================

    behavior_column = None

    for candidate in [
        "behavior_text_normalized",
        "behavior_text",
        "behavior",
        "text"
    ]:
        if candidate in behavior_df.columns:
            behavior_column = candidate
            break

    if behavior_column is None:
        raise RuntimeError(
            "Behavior text column not found."
        )

    print(
        "  Behavior column:",
        behavior_column
    )

    # ============================================================
    # SELECT DETERMINISTIC REAL EVENT
    # ============================================================

    row_index = 0

    structured_row = structured_df.iloc[
        row_index
    ]

    behavior_text = behavior_df.iloc[
        row_index
    ][behavior_column]

    if pd.isna(
        behavior_text
    ):
        behavior_text = ""

    behavior_text = str(
        behavior_text
    )

    # ============================================================
    # TRANSFORM
    # ============================================================

    (
        X_cat,
        X_num,
        X_text,
        X_final
    ) = transform_row(
        structured_row,
        behavior_text,
        categorical_columns,
        numerical_columns,
        encoder,
        scaler,
        tfidf
    )

    print("\nTRANSFORMATION")

    print(
        "  Categorical:",
        X_cat.shape
    )

    print(
        "  Numerical  :",
        X_num.shape
    )

    print(
        "  TF-IDF     :",
        X_text.shape
    )

    print(
        "  Final      :",
        X_final.shape
    )

    if X_final.shape != (
        1,
        6919
    ):
        raise RuntimeError(
            f"Unexpected final vector: "
            f"{X_final.shape}"
        )

    print(
        "[PASS] Production feature vector"
    )

    # ============================================================
    # PREDICTION
    # ============================================================

    probabilities = normalize_probability_output(
        model.predict_proba(
            X_final
        )
    )

    if probabilities.shape != (
        1,
        45
    ):
        raise RuntimeError(
            f"Unexpected probability matrix: "
            f"{probabilities.shape}"
        )

    print("\nPREDICTION")

    print(
        "  Shape :",
        probabilities.shape
    )

    print(
        "  Finite:",
        bool(
            np.isfinite(
                probabilities
            ).all()
        )
    )

    # ============================================================
    # COEFFICIENTS
    # ============================================================

    coefficients = get_coefficients(
        model
    )

    print("\nCOEFFICIENT MATRIX")

    print(
        "  Shape:",
        coefficients.shape
    )

    if coefficients.shape != (
        45,
        6919
    ):
        raise RuntimeError(
            "Coefficient dimensions do not "
            "match production dimensions."
        )

    print(
        "[PASS] Coefficient alignment"
    )

    # ============================================================
    # CONTRIBUTIONS
    # ============================================================

    x_dense = X_final.toarray().ravel()

    explanations = []

    for technique_index, technique in enumerate(
        techniques
    ):

        probability = float(
            probabilities[
                0,
                technique_index
            ]
        )

        threshold = float(
            thresholds.get(
                technique,
                0.5
            )
        )

        margin = (
            probability
            - threshold
        )

        coef = coefficients[
            technique_index
        ]

        contributions = (
            x_dense * coef
        )

        positive_indices = np.where(
            contributions > 0
        )[0]

        positive_indices = sorted(
            positive_indices,
            key=lambda i: contributions[i],
            reverse=True
        )

        positive = []

        for feature_index in positive_indices:

            if len(positive) >= 15:
                break

            value = float(
                x_dense[feature_index]
            )

            contribution = float(
                contributions[feature_index]
            )

            if abs(value) < 1e-15:
                continue

            positive.append(
                {
                    "feature_index": int(
                        feature_index
                    ),
                    "feature": feature_names[
                        feature_index
                    ],
                    "value": value,
                    "coefficient": float(
                        coef[feature_index]
                    ),
                    "contribution": contribution
                }
            )

        negative_indices = np.where(
            contributions < 0
        )[0]

        negative_indices = sorted(
            negative_indices,
            key=lambda i: contributions[i]
        )

        negative = []

        for feature_index in negative_indices:

            if len(negative) >= 10:
                break

            value = float(
                x_dense[feature_index]
            )

            contribution = float(
                contributions[feature_index]
            )

            if abs(value) < 1e-15:
                continue

            negative.append(
                {
                    "feature_index": int(
                        feature_index
                    ),
                    "feature": feature_names[
                        feature_index
                    ],
                    "value": value,
                    "coefficient": float(
                        coef[feature_index]
                    ),
                    "contribution": contribution
                }
            )

        explanations.append(
            {
                "technique": technique,
                "probability": probability,
                "threshold": threshold,
                "margin": margin,
                "above_threshold": bool(
                    probability >= threshold
                ),
                "top_positive_features": positive,
                "top_negative_features": negative
            }
        )

    # ============================================================
    # RANK TECHNIQUES
    # ============================================================

    order = np.argsort(
        probabilities[0]
    )[::-1]

    top_predictions = []

    for index in order:

        technique = techniques[
            index
        ]

        probability = float(
            probabilities[
                0,
                index
            ]
        )

        threshold = float(
            thresholds.get(
                technique,
                0.5
            )
        )

        top_predictions.append(
            {
                "technique": technique,
                "probability": probability,
                "threshold": threshold,
                "margin": probability - threshold,
                "above_threshold": bool(
                    probability >= threshold
                )
            }
        )

    detected = [
        item
        for item in top_predictions
        if item["above_threshold"]
    ]

    # ============================================================
    # STRUCTURED EVENT EVIDENCE
    # ============================================================

    structured_evidence = {}

    for column in (
        categorical_columns
        + numerical_columns
    ):

        value = structured_row[
            column
        ]

        if pd.isna(value):

            structured_evidence[
                column
            ] = None

        else:

            if isinstance(
                value,
                np.generic
            ):
                value = value.item()

            structured_evidence[
                column
            ] = value

    # ============================================================
    # REPORT
    # ============================================================

    report.update(
        {
            "status": "PASS",

            "model": {
                "type": type(model).__name__,
                "technique_count": len(techniques)
            },

            "feature_dimensions": {
                "categorical_source_features": len(
                    categorical_columns
                ),
                "categorical_encoded_features": (
                    categorical_encoded_count
                ),
                "numerical_features": numerical_count,
                "tfidf_features": tfidf_count,
                "final_features": len(feature_names)
            },

            "input": {
                "dataset": (
                    "structured_test.csv + "
                    "behavior_text_test.csv"
                ),
                "row_index": row_index,
                "behavior_column": behavior_column,
                "behavior_text": behavior_text,
                "structured_evidence": structured_evidence
            },

            "prediction": {
                "detected_technique_count": len(
                    detected
                ),
                "detected_techniques": detected,
                "top_10": top_predictions[:10]
            },

            "explanations": explanations,

            "methodology": {
                "method": (
                    "per-feature linear contribution "
                    "x_j * coefficient_j"
                ),
                "model": (
                    "OneVsRestClassifier + "
                    "LogisticRegression"
                ),
                "positive_contribution": (
                    "Feature contribution that increases "
                    "the classifier logit for the technique."
                ),
                "negative_contribution": (
                    "Feature contribution that decreases "
                    "the classifier logit for the technique."
                ),
                "causal_warning": (
                    "These are model-level feature "
                    "attributions, not causal proof and "
                    "not independent proof of maliciousness."
                )
            },

            "production_artifacts_modified": False
        }
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ============================================================
    # CONSOLE
    # ============================================================

    print("\n" + "=" * 72)
    print("PHASE 3B GATE 3B.1 SUMMARY")
    print("=" * 72)

    print(
        "Final feature vector:",
        X_final.shape
    )

    print(
        "MITRE techniques:",
        len(techniques)
    )

    print(
        "Detected techniques:",
        len(detected)
    )

    print("\nTOP PREDICTIONS")

    for item in top_predictions[:10]:

        print(
            f"  {item['technique']:10s} "
            f"P={item['probability']:.6f} "
            f"T={item['threshold']:.6f} "
            f"M={item['margin']:.6f} "
            f"Detected={item['above_threshold']}"
        )

    print("\nTOP POSITIVE MODEL EVIDENCE")

    explanation_lookup = {
        item["technique"]: item
        for item in explanations
    }

    for item in top_predictions[:5]:

        explanation = explanation_lookup[
            item["technique"]
        ]

        print(
            f"\n  {item['technique']} "
            f"P={item['probability']:.6f}"
        )

        for evidence in explanation[
            "top_positive_features"
        ][:5]:

            print(
                "    +",
                evidence["feature"],
                "contribution="
                f"{evidence['contribution']:.6f}"
            )

    print("\nDECISION: PASS")

    print(
        "Report:",
        REPORT_PATH
    )

    print(
        "\nNO PRODUCTION ARTIFACTS WERE MODIFIED."
    )


if __name__ == "__main__":
    main()