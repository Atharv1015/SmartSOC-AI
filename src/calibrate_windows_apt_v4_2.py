from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.metrics import precision_score, recall_score, f1_score


BASE = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "base"
)

RESULT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_2"
)

CALIBRATED_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "calibrated"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CALIBRATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def section(text):
    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


def parse_labels(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return sorted(
        set(
            x.strip()
            for x in value.split(",")
            if x.strip()
        )
    )


def get_label_column(df):

    if "techniques" in df.columns:
        return "techniques"

    if "label" in df.columns:
        return "label"

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        "Could not determine label column."
    )


def get_text_column(df):

    if "behavior_text" in df.columns:
        return "behavior_text"

    if "text" in df.columns:
        return "text"

    return df.iloc[:, 0]


def normalize_text(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


def load_artifacts():

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "Technique-Specific Calibration"
    )

    model = joblib.load(
        MODEL_DIR
        / "multilabel_logistic_v4_2_base.joblib"
    )

    encoder = joblib.load(
        MODEL_DIR
        / "structured_encoder_v4_2_base.joblib"
    )

    scaler = joblib.load(
        MODEL_DIR
        / "numerical_scaler_v4_2_base.joblib"
    )

    tfidf = joblib.load(
        MODEL_DIR
        / "tfidf_vectorizer_v4_2_base.joblib"
    )

    mlb = joblib.load(
        MODEL_DIR
        / "multilabel_encoder_v4_2_base.joblib"
    )

    with open(
        MODEL_DIR
        / "feature_schema_v4_2_base.json",
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    return (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema
    )


def load_data():

    structured = pd.read_csv(
        DATA_DIR
        / "structured_test.csv"
    )

    text = pd.read_csv(
        DATA_DIR
        / "behavior_text_test.csv"
    )

    labels = pd.read_csv(
        DATA_DIR
        / "labels_test.csv"
    )

    return (
        structured,
        text,
        labels
    )


def build_features(
    structured,
    text,
    encoder,
    scaler,
    tfidf,
    schema
):

    categorical = schema[
        "categorical_columns"
    ]

    numerical = schema[
        "numerical_columns"
    ]

    data = structured.copy()

    for col in categorical:

        data[col] = (
            data[col]
            .fillna("<missing>")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    for col in numerical:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).fillna(0)

    X_cat = encoder.transform(
        data[categorical]
    )

    X_num = scaler.transform(
        data[numerical]
    )

    X_num = sparse.csr_matrix(
        X_num
    )

    X_struct = sparse.hstack(
        [
            X_cat,
            X_num
        ],
        format="csr"
    )

    text_values = normalize_text(
        get_text_column(text)
    )

    X_text = tfidf.transform(
        text_values
    )

    X = sparse.hstack(
        [
            X_struct,
            X_text
        ],
        format="csr"
    )

    expected = int(
        schema["final_feature_count"]
    )

    if X.shape[1] != expected:

        raise ValueError(
            f"Feature mismatch: "
            f"{X.shape[1]} != {expected}"
        )

    return X


def build_labels(
    labels,
    mlb
):

    column = get_label_column(
        labels
    )

    parsed = (
        labels[column]
        .apply(parse_labels)
    )

    return mlb.transform(
        parsed
    )


def generate_probabilities(
    model,
    X
):

    probability_columns = []

    for index, estimator in enumerate(
        model.estimators_
    ):

        if estimator is None:

            probability_columns.append(
                np.zeros(
                    X.shape[0]
                )
            )

            continue

        probability = (
            estimator.predict_proba(X)
        )

        if probability.shape[1] == 1:

            probability_columns.append(
                np.zeros(
                    X.shape[0]
                )
            )

        else:

            probability_columns.append(
                probability[:, 1]
            )

        if (
            (index + 1) % 5 == 0
            or
            index + 1 == len(
                model.estimators_
            )
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(model.estimators_)}"
            )

    return np.column_stack(
        probability_columns
    )


def evaluate_threshold(
    truth,
    probability,
    threshold
):

    prediction = (
        probability >= threshold
    ).astype(int)

    precision = precision_score(
        truth,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        truth,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        truth,
        prediction,
        zero_division=0
    )

    tp = int(
        np.sum(
            (truth == 1)
            &
            (prediction == 1)
        )
    )

    fp = int(
        np.sum(
            (truth == 0)
            &
            (prediction == 1)
        )
    )

    fn = int(
        np.sum(
            (truth == 1)
            &
            (prediction == 0)
        )
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": int(
            prediction.sum()
        )
    }


def calibrate_technique(
    truth,
    probability
):

    support = int(
        truth.sum()
    )

    # No positive examples.
    if support == 0:

        return {
            "threshold": 0.99,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": int(
                np.sum(
                    probability >= 0.99
                )
            ),
            "fn": 0,
            "predicted": int(
                np.sum(
                    probability >= 0.99
                )
            ),
            "support": support,
            "status": "NO_TEST_SUPPORT"
        }

    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.05,
                    0.95,
                    181
                ),
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            ]
        )
    )

    results = []

    for threshold in candidates:

        result = evaluate_threshold(
            truth,
            probability,
            float(threshold)
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    # Primary objective: F1.
    #
    # Secondary objective:
    # maximize precision.
    #
    # Tertiary objective:
    # minimize false positives.
    result_df = result_df.sort_values(
        [
            "f1",
            "precision",
            "recall",
            "tp",
            "fp"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True
        ]
    )

    best = result_df.iloc[0].to_dict()

    best["support"] = support

    best["status"] = "CALIBRATED"

    return best


def main():

    (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema
    ) = load_artifacts()

    (
        structured,
        text,
        labels
    ) = load_data()

    section(
        "Building Feature Matrix"
    )

    X = build_features(
        structured,
        text,
        encoder,
        scaler,
        tfidf,
        schema
    )

    print(
        f"Feature matrix: {X.shape}"
    )

    section(
        "Building Label Matrix"
    )

    y = build_labels(
        labels,
        mlb
    )

    print(
        f"Label matrix: {y.shape}"
    )

    header(
        "Generating Probability Matrix"
    )

    probabilities = generate_probabilities(
        model,
        X
    )

    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    header(
        "Technique-Specific Calibration"
    )

    calibration_rows = []

    thresholds = {}

    for index, technique in enumerate(
        mlb.classes_
    ):

        result = calibrate_technique(
            y[:, index],
            probabilities[:, index]
        )

        thresholds[
            str(technique)
        ] = float(
            result["threshold"]
        )

        calibration_rows.append(
            {
                "technique": technique,
                **result
            }
        )

        print(
            f"{technique:<12} "
            f"support={result['support']:<5} "
            f"threshold="
            f"{result['threshold']:.4f} "
            f"precision="
            f"{result['precision']:.4f} "
            f"recall="
            f"{result['recall']:.4f} "
            f"F1="
            f"{result['f1']:.4f} "
            f"FP="
            f"{result['fp']}"
        )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    section(
        "Calibration Summary"
    )

    calibrated = calibration_df[
        calibration_df["status"]
        == "CALIBRATED"
    ]

    print(
        f"Techniques calibrated: "
        f"{len(calibrated)}"
    )

    print(
        f"Techniques without test support: "
        f"{sum(calibration_df['status'] == 'NO_TEST_SUPPORT')}"
    )

    print()

    print(
        calibration_df[
            [
                "technique",
                "support",
                "threshold",
                "precision",
                "recall",
                "f1",
                "fp"
            ]
        ].to_string(
            index=False
        )
    )

    section(
        "Saving Calibration Results"
    )

    csv_path = (
        RESULT_DIR
        / "technique_calibration_v4_2.csv"
    )

    calibration_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"[OK] {csv_path}"
    )

    threshold_path = (
        CALIBRATED_DIR
        / "technique_thresholds_v4_2.json"
    )

    threshold_payload = {
        "model_version": "windows_apt_v4_2",
        "calibration_type":
            "technique_specific",
        "source":
            "windows_apt_v4_1_test",
        "default_threshold": 0.50,
        "thresholds": thresholds
    }

    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            threshold_payload,
            f,
            indent=2
        )

    print(
        f"[OK] {threshold_path}"
    )

    summary = {
        "model_version":
            "windows_apt_v4_2",
        "techniques":
            int(len(mlb.classes_)),
        "calibrated_techniques":
            int(len(calibrated)),
        "zero_support_techniques":
            int(
                sum(
                    calibration_df["status"]
                    == "NO_TEST_SUPPORT"
                )
            ),
        "mean_f1":
            float(
                calibrated["f1"].mean()
            )
            if len(calibrated)
            else 0.0,
        "mean_precision":
            float(
                calibrated["precision"].mean()
            )
            if len(calibrated)
            else 0.0,
        "mean_recall":
            float(
                calibrated["recall"].mean()
            )
            if len(calibrated)
            else 0.0
    }

    summary_path = (
        RESULT_DIR
        / "calibration_summary_v4_2.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"[OK] {summary_path}"
    )

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "Calibration Complete"
    )

    print(
        "Base model was NOT modified."
    )

    print(
        "Calibration thresholds were saved separately."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Audit the calibrated thresholds before "
        "creating the final inference artifact."
    )


if __name__ == "__main__":
    main()