from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# SmartSOC Windows-APT
# Model B v4.1 vs v4.2
#
# FAIR TEMPORAL TEST COMPARISON
#
# v4.1 = calibrated baseline
# v4.2 = same v4.1 model + recalibrated thresholds
#
# IMPORTANT:
# - Same unseen temporal test set
# - Same feature pipeline
# - Same MITRE label space
# - Base v4.1 model is NOT modified
# =============================================================================


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
    / "windows_apt_v4_1"
    / "calibrated"
)

V42_THRESHOLD_FILE = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "recalibrated"
    / "technique_thresholds_v4_2_precision.json"
)

OUTPUT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


STRUCTURED_TEST = (
    DATA_DIR / "structured_test.csv"
)

TEXT_TEST = (
    DATA_DIR / "behavior_text_test.csv"
)

LABELS_TEST = (
    DATA_DIR / "labels_test.csv"
)

METADATA_TEST = (
    DATA_DIR / "metadata_test.csv"
)

SCHEMA_FILE = (
    DATA_DIR / "schema.json"
)


# =============================================================================
# HELPERS
# =============================================================================

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


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


def find_artifact(directory, keywords):

    directory = Path(directory)

    candidates = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        name = path.name.lower()

        if all(
            keyword.lower() in name
            for keyword in keywords
        ):
            candidates.append(path)

    if not candidates:
        return None

    joblibs = [
        p for p in candidates
        if p.suffix.lower() == ".joblib"
    ]

    if joblibs:
        candidates = joblibs

    candidates.sort(
        key=lambda p: len(str(p))
    )

    return candidates[0]


def normalize_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_labels(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    return sorted(
        set(
            item.strip()
            for item in text.split(",")
            if item.strip()
        )
    )


def load_schema():

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =============================================================================
# FEATURE PIPELINE
# =============================================================================

def get_numerical_columns(
    structured,
    schema
):

    possible_keys = [
        "numerical_columns",
        "numerical_features",
        "numeric_columns",
        "numeric_features",
    ]

    for key in possible_keys:

        values = schema.get(key)

        if isinstance(values, list):

            columns = [
                str(x)
                for x in values
                if str(x) in structured.columns
            ]

            if columns:
                return columns

    categorical = {
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
    }

    return [
        column
        for column in structured.columns
        if (
            column not in categorical
            and pd.api.types.is_numeric_dtype(
                structured[column]
            )
        )
    ]


def build_features(
    structured,
    text,
    encoder,
    scaler,
    vectorizer,
    schema
):

    categorical = [
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
    ]

    categorical_columns = [
        c for c in categorical
        if c in structured.columns
    ]

    cat_data = (
        structured[
            categorical_columns
        ]
        .fillna("<missing>")
        .astype(str)
    )

    X_cat = encoder.transform(
        cat_data
    )

    if not sparse.issparse(X_cat):

        X_cat = sparse.csr_matrix(
            X_cat
        )

    numerical_columns = (
        get_numerical_columns(
            structured,
            schema
        )
    )

    numerical_data = (
        structured[
            numerical_columns
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0.0)
    )

    X_num = scaler.transform(
        numerical_data.values
    )

    X_num = sparse.csr_matrix(
        X_num
    )

    text_values = (
        text.iloc[:, 0]
        .fillna("")
        .map(normalize_text)
        .tolist()
    )

    X_text = vectorizer.transform(
        text_values
    )

    if not sparse.issparse(X_text):

        X_text = sparse.csr_matrix(
            X_text
        )

    X = sparse.hstack(
        [
            X_cat,
            X_num,
            X_text,
        ],
        format="csr"
    )

    return X


# =============================================================================
# PROBABILITIES
# =============================================================================

def get_probabilities(
    model,
    X
):

    probabilities = model.predict_proba(
        X
    )

    if isinstance(
        probabilities,
        list
    ):

        columns = []

        for p in probabilities:

            p = np.asarray(p)

            if p.ndim == 2:

                if p.shape[1] >= 2:
                    columns.append(
                        p[:, 1]
                    )
                else:
                    columns.append(
                        p[:, 0]
                    )

            else:

                columns.append(p)

        probabilities = np.column_stack(
            columns
        )

    return np.asarray(
        probabilities
    )


# =============================================================================
# LABEL MATRIX
# =============================================================================

def build_label_matrix(
    labels,
    label_encoder
):

    techniques = [
        str(x)
        for x in label_encoder.classes_
    ]

    technique_to_index = {
        technique: i
        for i, technique
        in enumerate(techniques)
    }

    column = (
        "techniques"
        if "techniques" in labels.columns
        else labels.columns[0]
    )

    label_lists = (
        labels[column]
        .map(normalize_labels)
        .tolist()
    )

    Y = np.zeros(
        (
            len(label_lists),
            len(techniques)
        ),
        dtype=np.int8
    )

    for row_index, row_labels in enumerate(
        label_lists
    ):

        for technique in row_labels:

            if technique in technique_to_index:

                Y[
                    row_index,
                    technique_to_index[
                        technique
                    ]
                ] = 1

    return Y, techniques


# =============================================================================
# METRICS
# =============================================================================

def calculate_global_metrics(
    Y_true,
    Y_pred
):

    return {
        "micro_precision":
            precision_score(
                Y_true,
                Y_pred,
                average="micro",
                zero_division=0
            ),

        "micro_recall":
            recall_score(
                Y_true,
                Y_pred,
                average="micro",
                zero_division=0
            ),

        "micro_f1":
            f1_score(
                Y_true,
                Y_pred,
                average="micro",
                zero_division=0
            ),

        "macro_precision":
            precision_score(
                Y_true,
                Y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_recall":
            recall_score(
                Y_true,
                Y_pred,
                average="macro",
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                Y_true,
                Y_pred,
                average="macro",
                zero_division=0
            ),

        "weighted_precision":
            precision_score(
                Y_true,
                Y_pred,
                average="weighted",
                zero_division=0
            ),

        "weighted_recall":
            recall_score(
                Y_true,
                Y_pred,
                average="weighted",
                zero_division=0
            ),

        "weighted_f1":
            f1_score(
                Y_true,
                Y_pred,
                average="weighted",
                zero_division=0
            ),
    }


def calculate_per_technique(
    Y_true,
    Y_pred,
    probabilities,
    techniques,
    thresholds
):

    rows = []

    for i, technique in enumerate(
        techniques
    ):

        y_true = Y_true[:, i]
        y_pred = Y_pred[:, i]

        support = int(
            y_true.sum()
        )

        tp = int(
            np.sum(
                (y_true == 1)
                &
                (y_pred == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true == 0)
                &
                (y_pred == 1)
            )
        )

        fn = int(
            np.sum(
                (y_true == 1)
                &
                (y_pred == 0)
            )
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        rows.append(
            {
                "technique": technique,
                "support": support,
                "threshold": thresholds.get(
                    technique,
                    0.5
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_probability":
                    float(
                        probabilities[:, i].mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B "
        "v4.1 vs v4.2"
    )

    # =========================================================================
    # FILES
    # =========================================================================

    section(
        "Checking common unseen temporal test set"
    )

    require(STRUCTURED_TEST)
    require(TEXT_TEST)
    require(LABELS_TEST)
    require(METADATA_TEST)
    require(SCHEMA_FILE)
    require(V42_THRESHOLD_FILE)

    # =========================================================================
    # FIND MODEL ARTIFACTS
    # =========================================================================

    section(
        "Locating v4.1 calibrated artifacts"
    )

    model_path = find_artifact(
        MODEL_DIR,
        ["multilabel", "logistic"]
    )

    encoder_path = find_artifact(
        MODEL_DIR,
        ["structured", "encoder"]
    )

    scaler_path = find_artifact(
        MODEL_DIR,
        ["numerical", "scaler"]
    )

    vectorizer_path = find_artifact(
        MODEL_DIR,
        ["tfidf", "vectorizer"]
    )

    label_encoder_path = find_artifact(
        MODEL_DIR,
        ["multilabel", "encoder"]
    )

    require(model_path)
    require(encoder_path)
    require(scaler_path)
    require(vectorizer_path)
    require(label_encoder_path)

    # =========================================================================
    # LOAD ARTIFACTS
    # =========================================================================

    section(
        "Loading model and preprocessing artifacts"
    )

    model = joblib.load(
        model_path
    )

    encoder = joblib.load(
        encoder_path
    )

    scaler = joblib.load(
        scaler_path
    )

    vectorizer = joblib.load(
        vectorizer_path
    )

    label_encoder = joblib.load(
        label_encoder_path
    )

    schema = load_schema()

    with open(
        V42_THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        v42_data = json.load(f)

    v42_thresholds = {
        str(k): float(v)
        for k, v
        in v42_data[
            "thresholds"
        ].items()
    }

    # =========================================================================
    # V4.1 THRESHOLDS
    # =========================================================================

    v41_thresholds = {
        technique: 0.5
        for technique
        in label_encoder.classes_
    }

    # =========================================================================
    # LOAD TEST SET
    # =========================================================================

    section(
        "Loading SAME unseen temporal test set"
    )

    structured = pd.read_csv(
        STRUCTURED_TEST
    )

    text = pd.read_csv(
        TEXT_TEST
    )

    labels = pd.read_csv(
        LABELS_TEST
    )

    metadata = pd.read_csv(
        METADATA_TEST
    )

    print(
        "Structured:",
        structured.shape
    )

    print(
        "Text:",
        text.shape
    )

    print(
        "Labels:",
        labels.shape
    )

    print(
        "Metadata:",
        metadata.shape
    )

    counts = [
        len(structured),
        len(text),
        len(labels),
        len(metadata),
    ]

    if len(set(counts)) != 1:

        raise ValueError(
            "Test datasets are not aligned."
        )

    print(
        "[OK] Test datasets aligned:",
        counts[0]
    )

    # =========================================================================
    # TIMESTAMP VALIDATION
    # =========================================================================

    section(
        "Validating temporal test set"
    )

    timestamps = pd.to_datetime(
        metadata["timestamp"],
        format="%b %d, %Y @ %H:%M:%S.%f",
        errors="coerce"
    )

    invalid = int(
        timestamps.isna().sum()
    )

    print(
        "Invalid timestamps:",
        invalid
    )

    if invalid:

        raise ValueError(
            "Invalid timestamps in test set."
        )

    print(
        "Test range:"
    )

    print(
        timestamps.min(),
        "->",
        timestamps.max()
    )

    # =========================================================================
    # FEATURES
    # =========================================================================

    section(
        "Building common test feature matrix"
    )

    X = build_features(
        structured,
        text,
        encoder,
        scaler,
        vectorizer,
        schema
    )

    print(
        "Feature matrix:",
        X.shape
    )

    if hasattr(
        model,
        "n_features_in_"
    ):

        if (
            X.shape[1]
            != model.n_features_in_
        ):

            raise ValueError(
                "Feature dimension mismatch."
            )

    # =========================================================================
    # LABELS
    # =========================================================================

    section(
        "Building common test label matrix"
    )

    Y_true, techniques = (
        build_label_matrix(
            labels,
            label_encoder
        )
    )

    print(
        "Label matrix:",
        Y_true.shape
    )

    # =========================================================================
    # PROBABILITIES
    # =========================================================================

    section(
        "Generating probabilities"
    )

    probabilities = get_probabilities(
        model,
        X
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    if (
        probabilities.shape[1]
        != len(techniques)
    ):

        raise ValueError(
            "Probability/technique dimension mismatch."
        )

    # =========================================================================
    # BUILD PREDICTIONS
    # =========================================================================

    section(
        "Generating v4.1 predictions"
    )

    v41_matrix = np.zeros_like(
        probabilities,
        dtype=np.int8
    )

    for i, technique in enumerate(
        techniques
    ):

        threshold = (
            v41_thresholds[
                technique
            ]
        )

        v41_matrix[:, i] = (
            probabilities[:, i]
            >= threshold
        ).astype(np.int8)

    print(
        "[OK] v4.1 predictions generated."
    )

    # =========================================================================
    # V4.2
    # =========================================================================

    section(
        "Generating v4.2 predictions"
    )

    v42_matrix = np.zeros_like(
        probabilities,
        dtype=np.int8
    )

    for i, technique in enumerate(
        techniques
    ):

        threshold = (
            v42_thresholds.get(
                technique,
                0.5
            )
        )

        v42_matrix[:, i] = (
            probabilities[:, i]
            >= threshold
        ).astype(np.int8)

    print(
        "[OK] v4.2 predictions generated."
    )

    # =========================================================================
    # GLOBAL METRICS
    # =========================================================================

    section(
        "Calculating global metrics"
    )

    v41_metrics = (
        calculate_global_metrics(
            Y_true,
            v41_matrix
        )
    )

    v42_metrics = (
        calculate_global_metrics(
            Y_true,
            v42_matrix
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "v4.1",
                **v41_metrics
            },
            {
                "model": "v4.2",
                **v42_metrics
            },
        ]
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # =========================================================================
    # PER TECHNIQUE
    # =========================================================================

    section(
        "Per-technique comparison"
    )

    v41_per = calculate_per_technique(
        Y_true,
        v41_matrix,
        probabilities,
        techniques,
        v41_thresholds
    )

    v42_per = calculate_per_technique(
        Y_true,
        v42_matrix,
        probabilities,
        techniques,
        v42_thresholds
    )

    merged = v41_per.merge(
        v42_per,
        on="technique",
        suffixes=(
            "_v41",
            "_v42"
        )
    )

    merged["precision_delta"] = (
        merged["precision_v42"]
        - merged["precision_v41"]
    )

    merged["recall_delta"] = (
        merged["recall_v42"]
        - merged["recall_v41"]
    )

    merged["f1_delta"] = (
        merged["f1_v42"]
        - merged["f1_v41"]
    )

    merged["fp_delta"] = (
        merged["fp_v42"]
        - merged["fp_v41"]
    )

    # =========================================================================
    # BIGGEST CHANGES
    # =========================================================================

    print()
    print(
        "Largest F1 improvements:"
    )

    print(
        merged[
            [
                "technique",
                "f1_v41",
                "f1_v42",
                "f1_delta"
            ]
        ]
        .sort_values(
            "f1_delta",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Largest F1 declines:"
    )

    print(
        merged[
            [
                "technique",
                "f1_v41",
                "f1_v42",
                "f1_delta"
            ]
        ]
        .sort_values(
            "f1_delta",
            ascending=True
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Largest false-positive reductions:"
    )

    print(
        merged[
            [
                "technique",
                "fp_v41",
                "fp_v42",
                "fp_delta"
            ]
        ]
        .sort_values(
            "fp_delta"
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Largest false-positive increases:"
    )

    print(
        merged[
            [
                "technique",
                "fp_v41",
                "fp_v42",
                "fp_delta"
            ]
        ]
        .sort_values(
            "fp_delta",
            ascending=False
        )
        .head(10)
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # PREDICTION COUNTS
    # =========================================================================

    section(
        "Prediction volume"
    )

    v41_total = int(
        v41_matrix.sum()
    )

    v42_total = int(
        v42_matrix.sum()
    )

    print(
        "v4.1 predicted technique instances:",
        v41_total
    )

    print(
        "v4.2 predicted technique instances:",
        v42_total
    )

    print(
        "Change:",
        v42_total - v41_total
    )

    # =========================================================================
    # ACTIVE TECHNIQUES
    # =========================================================================

    v41_active = int(
        np.sum(
            v41_matrix.sum(
                axis=0
            ) > 0
        )
    )

    v42_active = int(
        np.sum(
            v42_matrix.sum(
                axis=0
            ) > 0
        )
    )

    print(
        "v4.1 active techniques:",
        v41_active
    )

    print(
        "v4.2 active techniques:",
        v42_active
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    section(
        "Saving comparison results"
    )

    comparison_file = (
        OUTPUT_DIR
        / "v4_1_vs_v4_2_global_metrics.csv"
    )

    per_file = (
        OUTPUT_DIR
        / "v4_1_vs_v4_2_per_technique.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "v4_1_vs_v4_2_summary.json"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    merged.to_csv(
        per_file,
        index=False
    )

    # =========================================================================
    # DECISION SUPPORT
    # =========================================================================

    f1_delta = (
        v42_metrics["micro_f1"]
        - v41_metrics["micro_f1"]
    )

    precision_delta = (
        v42_metrics["micro_precision"]
        - v41_metrics["micro_precision"]
    )

    recall_delta = (
        v42_metrics["micro_recall"]
        - v41_metrics["micro_recall"]
    )

    summary = {

        "comparison":
            "SmartSOC Windows-APT Model B v4.1 vs v4.2",

        "test_set":
            "windows_apt_v4_1 temporal test",

        "test_rows":
            int(len(Y_true)),

        "v4_1": v41_metrics,

        "v4_2": v42_metrics,

        "delta": {
            "micro_precision":
                float(precision_delta),

            "micro_recall":
                float(recall_delta),

            "micro_f1":
                float(f1_delta),
        },

        "prediction_volume": {
            "v4_1":
                v41_total,

            "v4_2":
                v42_total,

            "delta":
                v42_total - v41_total,
        },

        "active_techniques": {
            "v4_1":
                v41_active,

            "v4_2":
                v42_active,
        },

        "v4_2_threshold_source":
            str(V42_THRESHOLD_FILE),

        "base_model_modified":
            False,
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"[OK] {comparison_file}"
    )

    print(
        f"[OK] {per_file}"
    )

    print(
        f"[OK] {summary_file}"
    )

    # =========================================================================
    # FINAL INTERPRETATION
    # =========================================================================

    header(
        "v4.1 vs v4.2 Evaluation Summary"
    )

    print(
        f"v4.1 micro-F1 : "
        f"{v41_metrics['micro_f1']:.4f}"
    )

    print(
        f"v4.2 micro-F1 : "
        f"{v42_metrics['micro_f1']:.4f}"
    )

    print(
        f"Micro-F1 delta: "
        f"{f1_delta:+.4f}"
    )

    print()

    print(
        f"v4.1 micro-precision : "
        f"{v41_metrics['micro_precision']:.4f}"
    )

    print(
        f"v4.2 micro-precision : "
        f"{v42_metrics['micro_precision']:.4f}"
    )

    print(
        f"Precision delta      : "
        f"{precision_delta:+.4f}"
    )

    print()

    print(
        f"v4.1 micro-recall : "
        f"{v41_metrics['micro_recall']:.4f}"
    )

    print(
        f"v4.2 micro-recall : "
        f"{v42_metrics['micro_recall']:.4f}"
    )

    print(
        f"Recall delta      : "
        f"{recall_delta:+.4f}"
    )

    print()

    if f1_delta > 0:

        print(
            "[RESULT] v4.2 has higher micro-F1 "
            "on the common temporal test set."
        )

    elif f1_delta < 0:

        print(
            "[RESULT] v4.1 has higher micro-F1 "
            "on the common temporal test set."
        )

    else:

        print(
            "[RESULT] v4.1 and v4.2 have equal "
            "micro-F1."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not deploy v4.2 based on threshold "
        "validation alone."
    )

    print(
        "This comparison is the required "
        "unseen temporal evaluation."
    )

    print(
        "Keep v4.1 as the production candidate "
        "until this comparison is reviewed."
    )


if __name__ == "__main__":
    main()