from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")


# =============================================================================
# SmartSOC Windows-APT Model B v4.2
# Precision-Constrained Recalibration
#
# SOURCE:
#   v4.1 dataset
#   v4.1 calibrated model
#
# OUTPUT:
#   v4.2 precision-constrained threshold artifact
#
# IMPORTANT:
#   - v4.1 model is NOT modified
#   - v4.1 dataset is NOT modified
#   - Test set is NOT used for threshold selection
#   - Temporal validation is used
# =============================================================================


BASE = Path(__file__).resolve().parents[1]


# =============================================================================
# v4.1 DATASET
# =============================================================================

DATA_DIR = (
    BASE
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
TEXT_TRAIN = DATA_DIR / "behavior_text_train.csv"
LABELS_TRAIN = DATA_DIR / "labels_train.csv"
METADATA_TRAIN = DATA_DIR / "metadata_train.csv"
SCHEMA_FILE = DATA_DIR / "schema.json"


# =============================================================================
# v4.1 CALIBRATED MODEL
# =============================================================================

MODEL_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_1"
    / "calibrated"
)


# =============================================================================
# v4.2 OUTPUT
# =============================================================================

OUTPUT_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "recalibrated"
)

RESULT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "precision_recalibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# DISPLAY HELPERS
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


def require(path, description):

    if path is None or not Path(path).exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


# =============================================================================
# ARTIFACT FINDER
# =============================================================================

def find_artifact(directory, keywords):

    directory = Path(directory)

    if not directory.exists():
        return None

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
        path
        for path in candidates
        if path.suffix.lower() == ".joblib"
    ]

    if joblibs:
        candidates = joblibs

    candidates.sort(
        key=lambda p: len(str(p))
    )

    return candidates[0]


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

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

    labels = []

    for item in text.split(","):

        item = item.strip()

        if item:
            labels.append(item)

    return sorted(
        set(labels)
    )


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


# =============================================================================
# SCHEMA LOADER
# =============================================================================

def load_schema():

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# =============================================================================
# NUMERICAL FEATURE DETECTION
# =============================================================================

def get_numerical_columns(
    structured,
    schema
):

    possible_keys = [
        "numerical_columns",
        "numerical_features",
        "numeric_columns",
        "numeric_features"
    ]

    for key in possible_keys:

        value = schema.get(key)

        if isinstance(
            value,
            list
        ):

            columns = [
                str(item)
                for item in value
                if str(item) in structured.columns
            ]

            if columns:
                return columns

    categorical = {
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension"
    }

    columns = []

    for column in structured.columns:

        if column in categorical:
            continue

        if pd.api.types.is_numeric_dtype(
            structured[column]
        ):

            columns.append(column)

    return columns


# =============================================================================
# FEATURE MATRIX
# =============================================================================

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
        "parent_extension"
    ]

    categorical_columns = [
        column
        for column in categorical
        if column in structured.columns
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

    # -------------------------------------------------------------------------
    # NUMERICAL
    # -------------------------------------------------------------------------

    numerical_columns = get_numerical_columns(
        structured,
        schema
    )

    if not numerical_columns:

        X_num = sparse.csr_matrix(
            (
                len(structured),
                0
            )
        )

    else:

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

        X_num_array = scaler.transform(
            numerical_data.values
        )

        X_num = sparse.csr_matrix(
            X_num_array
        )

    # -------------------------------------------------------------------------
    # TF-IDF
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # HYBRID
    # -------------------------------------------------------------------------

    X = sparse.hstack(
        [
            X_cat,
            X_num,
            X_text
        ],
        format="csr"
    )

    return X


# =============================================================================
# PROBABILITY MATRIX
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

        for probability in probabilities:

            probability = np.asarray(
                probability
            )

            if probability.ndim == 2:

                if probability.shape[1] >= 2:

                    columns.append(
                        probability[:, 1]
                    )

                else:

                    columns.append(
                        probability[:, 0]
                    )

            else:

                columns.append(
                    probability
                )

        probabilities = np.column_stack(
            columns
        )

    probabilities = np.asarray(
        probabilities
    )

    return probabilities


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (predictions == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (predictions == 0)
        )
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================

def select_threshold(
    y_true,
    probabilities
):

    support = int(
        y_true.sum()
    )

    # -------------------------------------------------------------------------
    # No positive examples
    # -------------------------------------------------------------------------

    if support == 0:

        return {
            "threshold": 0.95,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": len(y_true),
            "status": "NO_VALIDATION_SUPPORT"
        }

    # -------------------------------------------------------------------------
    # Threshold grid
    # -------------------------------------------------------------------------

    thresholds = np.arange(
        0.20,
        1.001,
        0.01
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold
        )

        results.append(
            metrics
        )

    metrics_df = pd.DataFrame(
        results
    )

    # -------------------------------------------------------------------------
    # Precision target
    # -------------------------------------------------------------------------

    if support >= 20:

        precision_target = 0.90

    elif support >= 10:

        precision_target = 0.90

    else:

        precision_target = 0.95

    acceptable = metrics_df[
        metrics_df["precision"]
        >= precision_target
    ]

    # -------------------------------------------------------------------------
    # Best safe threshold
    # -------------------------------------------------------------------------

    if not acceptable.empty:

        best = (
            acceptable
            .sort_values(
                [
                    "f1",
                    "precision",
                    "recall",
                    "threshold"
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True
                ]
            )
            .iloc[0]
            .to_dict()
        )

        best["status"] = (
            "PRECISION_CONSTRAINED"
        )

        return best

    # -------------------------------------------------------------------------
    # No threshold achieved target
    # -------------------------------------------------------------------------

    best = (
        metrics_df
        .sort_values(
            [
                "precision",
                "f1",
                "recall",
                "threshold"
            ],
            ascending=[
                False,
                False,
                False,
                True
            ]
        )
        .iloc[0]
        .to_dict()
    )

    best["status"] = (
        "PRECISION_TARGET_NOT_REACHED"
    )

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "- Precision-Constrained Recalibration"
    )

    # =========================================================================
    # CHECK DATA
    # =========================================================================

    section(
        "Checking v4.1 dataset artifacts"
    )

    require(
        STRUCTURED_TRAIN,
        "structured_train.csv"
    )

    require(
        TEXT_TRAIN,
        "behavior_text_train.csv"
    )

    require(
        LABELS_TRAIN,
        "labels_train.csv"
    )

    require(
        METADATA_TRAIN,
        "metadata_train.csv"
    )

    require(
        SCHEMA_FILE,
        "schema.json"
    )

    # =========================================================================
    # FIND MODEL ARTIFACTS
    # =========================================================================

    section(
        "Locating v4.1 calibrated model"
    )

    model_path = find_artifact(
        MODEL_DIR,
        [
            "multilabel",
            "logistic"
        ]
    )

    encoder_path = find_artifact(
        MODEL_DIR,
        [
            "structured",
            "encoder"
        ]
    )

    scaler_path = find_artifact(
        MODEL_DIR,
        [
            "numerical",
            "scaler"
        ]
    )

    vectorizer_path = find_artifact(
        MODEL_DIR,
        [
            "tfidf",
            "vectorizer"
        ]
    )

    label_encoder_path = find_artifact(
        MODEL_DIR,
        [
            "multilabel",
            "encoder"
        ]
    )

    require(
        model_path,
        "multilabel logistic model"
    )

    require(
        encoder_path,
        "structured encoder"
    )

    require(
        scaler_path,
        "numerical scaler"
    )

    require(
        vectorizer_path,
        "TF-IDF vectorizer"
    )

    require(
        label_encoder_path,
        "multilabel encoder"
    )

    # =========================================================================
    # LOAD ARTIFACTS
    # =========================================================================

    section(
        "Loading artifacts"
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

    print(
        "Model type:",
        type(model).__name__
    )

    print(
        "MITRE techniques:",
        len(
            label_encoder.classes_
        )
    )

    print(
        "TF-IDF vocabulary:",
        len(
            vectorizer.vocabulary_
        )
    )

    # =========================================================================
    # LOAD TRAINING DATA
    # =========================================================================

    section(
        "Loading v4.1 training data"
    )

    structured = pd.read_csv(
        STRUCTURED_TRAIN
    )

    text = pd.read_csv(
        TEXT_TRAIN
    )

    labels = pd.read_csv(
        LABELS_TRAIN
    )

    metadata = pd.read_csv(
        METADATA_TRAIN
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

    # =========================================================================
    # ALIGNMENT
    # =========================================================================

    section(
        "Checking alignment"
    )

    row_counts = [
        len(structured),
        len(text),
        len(labels),
        len(metadata)
    ]

    if len(set(row_counts)) != 1:

        raise ValueError(
            "Dataset row counts are not aligned."
        )

    print(
        "[OK] Aligned rows:",
        row_counts[0]
    )

    # =========================================================================
    # TIMESTAMP PARSING
    # =========================================================================

    section(
        "Parsing timestamps"
    )

    if "timestamp" not in metadata.columns:

        raise ValueError(
            "Metadata does not contain "
            "'timestamp' column."
        )

    timestamp_format = (
        "%b %d, %Y @ %H:%M:%S.%f"
    )

    print(
        "Timestamp format:",
        timestamp_format
    )

    print(
        "Example:",
        metadata["timestamp"].iloc[0]
    )

    timestamps = pd.to_datetime(
        metadata["timestamp"],
        format=timestamp_format,
        errors="coerce"
    )

    invalid_count = int(
        timestamps.isna().sum()
    )

    print(
        "Invalid timestamps:",
        invalid_count
    )

    if invalid_count:

        print(
            metadata.loc[
                timestamps.isna(),
                "timestamp"
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        raise ValueError(
            "Invalid timestamps found."
        )

    print(
        "Timestamp range:"
    )

    print(
        timestamps.min(),
        "->",
        timestamps.max()
    )

    # =========================================================================
    # TEMPORAL VALIDATION SPLIT
    # =========================================================================

    section(
        "Creating temporal validation split"
    )

    validation_fraction = 0.20

    sorted_indices = np.argsort(
        timestamps.values
    )

    validation_count = max(
        1,
        int(
            len(sorted_indices)
            * validation_fraction
        )
    )

    validation_indices = (
        sorted_indices[
            -validation_count:
        ]
    )

    print(
        "Total rows:",
        len(sorted_indices)
    )

    print(
        "Validation rows:",
        len(validation_indices)
    )

    validation_timestamps = (
        timestamps.iloc[
            validation_indices
        ]
    )

    print(
        "Validation range:"
    )

    print(
        validation_timestamps.min(),
        "->",
        validation_timestamps.max()
    )

    # =========================================================================
    # BUILD VALIDATION DATA
    # =========================================================================

    validation_structured = (
        structured
        .iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    validation_text = (
        text
        .iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    validation_labels = (
        labels
        .iloc[
            validation_indices
        ]
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # FEATURES
    # =========================================================================

    section(
        "Building validation feature matrix"
    )

    X_validation = build_features(
        validation_structured,
        validation_text,
        encoder,
        scaler,
        vectorizer,
        schema
    )

    print(
        "Feature matrix:",
        X_validation.shape
    )

    if hasattr(
        model,
        "n_features_in_"
    ):

        expected_features = int(
            model.n_features_in_
        )

        actual_features = int(
            X_validation.shape[1]
        )

        print(
            "Expected features:",
            expected_features
        )

        print(
            "Generated features:",
            actual_features
        )

        if (
            expected_features
            != actual_features
        ):

            raise ValueError(
                "Feature dimension mismatch."
            )

    # =========================================================================
    # LABEL MATRIX
    # =========================================================================

    section(
        "Building validation label matrix"
    )

    if "techniques" in validation_labels.columns:

        label_column = "techniques"

    else:

        label_column = (
            validation_labels.columns[0]
        )

    label_lists = (
        validation_labels[
            label_column
        ]
        .map(normalize_labels)
        .tolist()
    )

    techniques = [
        str(value)
        for value
        in label_encoder.classes_
    ]

    technique_to_index = {
        technique: index
        for index, technique
        in enumerate(techniques)
    }

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

    print(
        "Label matrix:",
        Y.shape
    )

    # =========================================================================
    # PROBABILITIES
    # =========================================================================

    section(
        "Generating validation probabilities"
    )

    probabilities = get_probabilities(
        model,
        X_validation
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
            "Probability matrix does not match "
            "MITRE technique count."
        )

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    header(
        "Precision-Constrained Threshold Selection"
    )

    rows = []

    for index, technique in enumerate(
        techniques
    ):

        y_true = Y[
            :,
            index
        ]

        probs = probabilities[
            :,
            index
        ]

        support = int(
            y_true.sum()
        )

        result = select_threshold(
            y_true,
            probs
        )

        result["technique"] = technique
        result["support"] = support

        rows.append(
            result
        )

        print(
            f"{technique:<12} "
            f"support={support:<5} "
            f"threshold={result['threshold']:.4f} "
            f"precision={result['precision']:.4f} "
            f"recall={result['recall']:.4f} "
            f"F1={result['f1']:.4f} "
            f"FP={result['fp']:<5} "
            f"{result['status']}"
        )

    calibration = pd.DataFrame(
        rows
    )

    # =========================================================================
    # SAFETY CLAMP
    # =========================================================================

    calibration["threshold"] = (
        calibration["threshold"]
        .clip(
            lower=0.20,
            upper=0.99
        )
    )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_path = (
        RESULT_DIR
        / "precision_recalibration_v4_2.csv"
    )

    calibration.to_csv(
        csv_path,
        index=False
    )

    print()
    print(
        f"[OK] {csv_path}"
    )

    # =========================================================================
    # SAVE JSON
    # =========================================================================

    threshold_map = {}

    for _, row in calibration.iterrows():

        threshold_map[
            str(row["technique"])
        ] = round(
            float(
                row["threshold"]
            ),
            6
        )

    output = {

        "model":
            "SmartSOC Windows-APT Model B v4.2",

        "source_model":
            "SmartSOC Windows-APT Model B v4.1 calibrated",

        "calibration_method":
            "precision_constrained_temporal_validation",

        "timestamp_format":
            timestamp_format,

        "validation_fraction":
            validation_fraction,

        "test_set_used":
            False,

        "base_model_modified":
            False,

        "default_threshold":
            0.50,

        "minimum_threshold":
            0.20,

        "maximum_threshold":
            0.99,

        "techniques":
            techniques,

        "thresholds":
            threshold_map
    }

    json_path = (
        OUTPUT_DIR
        / "technique_thresholds_v4_2_precision.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print(
        f"[OK] {json_path}"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    header(
        "Precision Recalibration Summary"
    )

    supported = calibration[
        calibration["support"] > 0
    ]

    unsupported = calibration[
        calibration["support"] == 0
    ]

    print(
        "Techniques:",
        len(calibration)
    )

    print(
        "Validation-supported techniques:",
        len(supported)
    )

    print(
        "Validation-zero-support techniques:",
        len(unsupported)
    )

    print(
        "Minimum threshold:",
        f"{calibration['threshold'].min():.4f}"
    )

    print(
        "Maximum threshold:",
        f"{calibration['threshold'].max():.4f}"
    )

    # =========================================================================
    # HIGH RISK
    # =========================================================================

    section(
        "High-risk recalibration results"
    )

    high_risk = calibration[
        (
            calibration["precision"] < 0.90
        )
        |
        (
            calibration["f1"] < 0.50
        )
        |
        (
            calibration["fp"] > 50
        )
    ].sort_values(
        [
            "fp",
            "precision"
        ],
        ascending=[
            False,
            True
        ]
    )

    if high_risk.empty:

        print(
            "[OK] No high-risk techniques."
        )

    else:

        print(
            high_risk[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp",
                    "status"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # COMPLETE
    # =========================================================================

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "- Precision Recalibration COMPLETE"
    )

    print(
        "[OK] v4.1 dataset was used."
    )

    print(
        "[OK] v4.1 calibrated model was used."
    )

    print(
        "[OK] Test set was NOT used."
    )

    print(
        "[OK] v4.1 model was NOT modified."
    )

    print()
    print(
        "Threshold artifact:"
    )

    print(
        json_path
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Audit the v4.2 precision-constrained "
        "thresholds before creating the final "
        "inference artifact."
    )


if __name__ == "__main__":
    main()