from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from scipy import sparse


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

RESULT_DIR.mkdir(
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


def label_column(df):

    if "techniques" in df.columns:
        return "techniques"

    if "label" in df.columns:
        return "label"

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        "Unable to identify label column."
    )


def text_column(df):

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
        "Probability Plateau Diagnosis"
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

    structured = pd.read_csv(
        DATA_DIR / "structured_test.csv"
    )

    text = pd.read_csv(
        DATA_DIR / "behavior_text_test.csv"
    )

    labels = pd.read_csv(
        DATA_DIR / "labels_test.csv"
    )

    return (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
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

    txt = normalize_text(
        text_column(text)
    )

    X_text = tfidf.transform(
        txt
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


def build_labels(labels, mlb):

    col = label_column(labels)

    parsed = (
        labels[col]
        .apply(parse_labels)
    )

    return mlb.transform(parsed)


def generate_probabilities(
    model,
    X
):

    probabilities = []

    for estimator in model.estimators_:

        if estimator is None:

            probabilities.append(
                np.zeros(
                    X.shape[0]
                )
            )

            continue

        p = estimator.predict_proba(X)

        if p.shape[1] == 1:

            probabilities.append(
                np.zeros(
                    X.shape[0]
                )
            )

        else:

            probabilities.append(
                p[:, 1]
            )

    return np.column_stack(
        probabilities
    )


def analyze_class(
    technique,
    index,
    y,
    probabilities
):

    truth = y[:, index]
    prob = probabilities[:, index]

    positive = prob[truth == 1]
    negative = prob[truth == 0]

    unique_values, counts = np.unique(
        np.round(prob, 8),
        return_counts=True
    )

    order = np.argsort(
        counts
    )[::-1]

    top_values = []

    for i in order[:10]:

        top_values.append(
            (
                float(unique_values[i]),
                int(counts[i])
            )
        )

    return {
        "technique": technique,
        "support": int(truth.sum()),
        "positive_unique": int(
            len(np.unique(
                np.round(
                    positive,
                    8
                )
            ))
        ),
        "negative_unique": int(
            len(np.unique(
                np.round(
                    negative,
                    8
                )
            ))
        ),
        "overall_unique": int(
            len(np.unique(
                np.round(
                    prob,
                    8
                )
            ))
        ),
        "positive_min": float(
            positive.min()
        ) if len(positive) else 0.0,
        "positive_max": float(
            positive.max()
        ) if len(positive) else 0.0,
        "positive_mean": float(
            positive.mean()
        ) if len(positive) else 0.0,
        "negative_min": float(
            negative.min()
        ) if len(negative) else 0.0,
        "negative_max": float(
            negative.max()
        ) if len(negative) else 0.0,
        "negative_mean": float(
            negative.mean()
        ) if len(negative) else 0.0,
        "top_probability_values":
            str(top_values)
    }


def main():

    (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        structured,
        text,
        labels
    ) = load_artifacts()

    header(
        "Building Test Matrix"
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
        f"Test feature matrix: {X.shape}"
    )

    y = build_labels(
        labels,
        mlb
    )

    print(
        f"Label matrix: {y.shape}"
    )

    header(
        "Generating Probabilities"
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
        "Analyzing All 45 Techniques"
    )

    rows = []

    for index, technique in enumerate(
        mlb.classes_
    ):

        result = analyze_class(
            technique,
            index,
            y,
            probabilities
        )

        rows.append(result)

    diagnostics = pd.DataFrame(
        rows
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    section(
        "Potential Probability Plateaus"
    )

    plateau = diagnostics[
        (
            diagnostics[
                "overall_unique"
            ]
            <= 5
        )
    ].copy()

    if plateau.empty:

        print(
            "[OK] No severe probability plateaus detected."
        )

    else:

        print(
            plateau[
                [
                    "technique",
                    "support",
                    "overall_unique",
                    "positive_unique",
                    "negative_unique",
                    "positive_min",
                    "positive_max",
                    "negative_min",
                    "negative_max"
                ]
            ]
            .to_string(
                index=False
            )
        )

    section(
        "Zero-Support Test Techniques"
    )

    zero_support = diagnostics[
        diagnostics["support"] == 0
    ]

    print(
        zero_support[
            [
                "technique",
                "overall_unique",
                "negative_mean",
                "negative_max"
            ]
        ]
        .to_string(
            index=False
        )
    )

    section(
        "High-False-Positive Candidates"
    )

    candidates = diagnostics[
        (
            diagnostics["support"] > 0
        )
        &
        (
            diagnostics["negative_max"]
            >
            diagnostics["positive_min"]
        )
    ].copy()

    candidates["overlap"] = (
        candidates["negative_max"]
        -
        candidates["positive_min"]
    )

    print(
        candidates
        .sort_values(
            "overlap",
            ascending=False
        )[
            [
                "technique",
                "support",
                "positive_min",
                "negative_max",
                "positive_mean",
                "negative_mean",
                "overlap"
            ]
        ]
        .to_string(
            index=False
        )
    )

    output = (
        RESULT_DIR
        / "probability_plateau_diagnostics_v4_2.csv"
    )

    diagnostics.to_csv(
        output,
        index=False
    )

    print()
    print(
        f"[OK] Saved: {output}"
    )

    header(
        "Diagnosis Complete"
    )

    print(
        "No model artifacts were modified."
    )

    print(
        "No thresholds were changed."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Use this report to determine whether "
        "the problem is caused by class imbalance, "
        "label leakage, duplicated behavior signatures, "
        "or model probability saturation."
    )


if __name__ == "__main__":
    main()