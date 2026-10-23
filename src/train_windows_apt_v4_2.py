"""
SmartSOC Windows-APT Model B v4.2
=================================

Purpose:
    Train an improved multilabel MITRE ATT&CK classifier using the
    cleaned v4.1 dataset.

Key changes from v4.1:
    - Keeps the v4.1 temporal train/test split.
    - Uses the v4.1.1 cleaned dataset.
    - Uses class-balanced One-vs-Rest Logistic Regression.
    - Uses structured + TF-IDF hybrid features.
    - Does NOT modify v4.1 artifacts.
    - Does NOT perform threshold calibration here.
    - Saves a clean v4.2 model for subsequent evaluation/calibration.

Primary objective:
    Improve macro-level MITRE technique coverage while retaining
    acceptable micro/weighted performance.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE = Path(__file__).resolve().parents[1]

DATA_DIR = BASE / "data" / "processed" / "windows_apt_v4_1"

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

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "structured_train": DATA_DIR / "structured_train.csv",
    "structured_test": DATA_DIR / "structured_test.csv",
    "text_train": DATA_DIR / "behavior_text_train.csv",
    "text_test": DATA_DIR / "behavior_text_test.csv",
    "labels_train": DATA_DIR / "labels_train.csv",
    "labels_test": DATA_DIR / "labels_test.csv",
    "metadata_train": DATA_DIR / "metadata_train.csv",
    "metadata_test": DATA_DIR / "metadata_test.csv",
    "schema": DATA_DIR / "schema.json",
}


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def section(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def require_files():
    section("Checking required artifacts...")

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"[OK] {path}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    header("SmartSOC Windows-APT Model B v4.2 - Training")

    require_files()

    header("Loading v4.1.1 dataset")

    structured_train = pd.read_csv(FILES["structured_train"])
    structured_test = pd.read_csv(FILES["structured_test"])

    text_train = pd.read_csv(FILES["text_train"])
    text_test = pd.read_csv(FILES["text_test"])

    labels_train = pd.read_csv(FILES["labels_train"])
    labels_test = pd.read_csv(FILES["labels_test"])

    metadata_train = pd.read_csv(FILES["metadata_train"])
    metadata_test = pd.read_csv(FILES["metadata_test"])

    with open(FILES["schema"], "r", encoding="utf-8") as f:
        schema = json.load(f)

    print(f"Structured train : {structured_train.shape}")
    print(f"Structured test  : {structured_test.shape}")
    print(f"Text train       : {text_train.shape}")
    print(f"Text test        : {text_test.shape}")
    print(f"Labels train     : {labels_train.shape}")
    print(f"Labels test      : {labels_test.shape}")
    print(f"Metadata train   : {metadata_train.shape}")
    print(f"Metadata test    : {metadata_test.shape}")

    return (
        structured_train,
        structured_test,
        text_train,
        text_test,
        labels_train,
        labels_test,
        metadata_train,
        metadata_test,
        schema,
    )


# ============================================================================
# ALIGNMENT
# ============================================================================

def check_alignment(
    structured_train,
    structured_test,
    text_train,
    text_test,
    labels_train,
    labels_test,
    metadata_train,
    metadata_test,
):
    header("Checking dataset alignment")

    train_sizes = {
        "structured": len(structured_train),
        "text": len(text_train),
        "labels": len(labels_train),
        "metadata": len(metadata_train),
    }

    test_sizes = {
        "structured": len(structured_test),
        "text": len(text_test),
        "labels": len(labels_test),
        "metadata": len(metadata_test),
    }

    print("Training row counts:")
    for k, v in train_sizes.items():
        print(f"  {k:12}: {v}")

    print("Testing row counts:")
    for k, v in test_sizes.items():
        print(f"  {k:12}: {v}")

    if len(set(train_sizes.values())) != 1:
        raise ValueError(
            f"Training datasets are not aligned: {train_sizes}"
        )

    if len(set(test_sizes.values())) != 1:
        raise ValueError(
            f"Testing datasets are not aligned: {test_sizes}"
        )

    print("[OK] Training datasets aligned.")
    print("[OK] Testing datasets aligned.")


# ============================================================================
# COLUMN DETECTION
# ============================================================================

def detect_columns(structured_train, schema):
    section("Detecting feature schema")

    schema_categorical = schema.get("categorical_columns", [])

    if schema_categorical:
        categorical_columns = [
            c for c in schema_categorical
            if c in structured_train.columns
        ]
    else:
        categorical_columns = [
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
        ]

    numerical_columns = [
        c for c in structured_train.columns
        if c not in categorical_columns
    ]

    print(f"Categorical features: {len(categorical_columns)}")

    for c in categorical_columns:
        print(f"  - {c}")

    print(f"Numerical features: {len(numerical_columns)}")

    return categorical_columns, numerical_columns


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def get_text_column(df):
    if df.shape[1] == 1:
        return df.iloc[:, 0].fillna("").astype(str)

    candidates = [
        "behavior_text",
        "text",
        "command_line",
    ]

    for c in candidates:
        if c in df.columns:
            return df[c].fillna("").astype(str)

    return df.iloc[:, 0].fillna("").astype(str)


def normalize_text(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================================
# LABEL PROCESSING
# ============================================================================

def get_label_column(df):
    if "techniques" in df.columns:
        return "techniques"

    if "label" in df.columns:
        return "label"

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        "Could not identify MITRE label column."
    )


def parse_labels(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    return sorted(set(parts))


def prepare_labels(labels_train, labels_test):
    section("Preparing MITRE multilabel targets")

    train_col = get_label_column(labels_train)
    test_col = get_label_column(labels_test)

    train_lists = labels_train[train_col].apply(parse_labels)
    test_lists = labels_test[test_col].apply(parse_labels)

    all_classes = sorted(
        set(x for labels in train_lists for x in labels)
        |
        set(x for labels in test_lists for x in labels)
    )

    print(f"MITRE techniques: {len(all_classes)}")

    print("Techniques:")
    print(all_classes)

    test_only = (
        set(x for labels in test_lists for x in labels)
        -
        set(x for labels in train_lists for x in labels)
    )

    if test_only:
        raise ValueError(
            "Test-only MITRE techniques detected:\n"
            + "\n".join(sorted(test_only))
        )

    mlb = MultiLabelBinarizer(classes=all_classes)

    y_train = mlb.fit_transform(train_lists)
    y_test = mlb.transform(test_lists)

    print(f"Y train: {y_train.shape}")
    print(f"Y test : {y_test.shape}")

    return mlb, y_train, y_test, train_lists, test_lists


# ============================================================================
# STRUCTURED FEATURES
# ============================================================================

def clean_structured(
    train,
    test,
    categorical_columns,
    numerical_columns,
):
    section("Preparing structured features")

    train = train.copy()
    test = test.copy()

    for col in categorical_columns:
        train[col] = (
            train[col]
            .fillna("<missing>")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        test[col] = (
            test[col]
            .fillna("<missing>")
            .astype(str)
            .str.lower()
            .str.strip()
        )

    for col in numerical_columns:
        train[col] = pd.to_numeric(
            train[col],
            errors="coerce"
        ).fillna(0)

        test[col] = pd.to_numeric(
            test[col],
            errors="coerce"
        ).fillna(0)

    return train, test


def build_structured_features(
    train,
    test,
    categorical_columns,
    numerical_columns,
):
    section("Encoding structured features")

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )

    X_cat_train = encoder.fit_transform(
        train[categorical_columns]
    )

    X_cat_test = encoder.transform(
        test[categorical_columns]
    )

    print(
        f"Categorical train: {X_cat_train.shape}"
    )

    print(
        f"Categorical test : {X_cat_test.shape}"
    )

    scaler = StandardScaler()

    X_num_train = scaler.fit_transform(
        train[numerical_columns]
    )

    X_num_test = scaler.transform(
        test[numerical_columns]
    )

    X_num_train = sparse.csr_matrix(
        X_num_train
    )

    X_num_test = sparse.csr_matrix(
        X_num_test
    )

    print(
        f"Numerical train: {X_num_train.shape}"
    )

    print(
        f"Numerical test : {X_num_test.shape}"
    )

    X_struct_train = sparse.hstack(
        [
            X_cat_train,
            X_num_train,
        ],
        format="csr",
    )

    X_struct_test = sparse.hstack(
        [
            X_cat_test,
            X_num_test,
        ],
        format="csr",
    )

    print(
        f"Structured train: {X_struct_train.shape}"
    )

    print(
        f"Structured test : {X_struct_test.shape}"
    )

    return (
        encoder,
        scaler,
        X_struct_train,
        X_struct_test,
    )


# ============================================================================
# TF-IDF
# ============================================================================

def build_text_features(
    text_train,
    text_test,
):
    section("Building TF-IDF behavioral features")

    train_text = normalize_text(
        get_text_column(text_train)
    )

    test_text = normalize_text(
        get_text_column(text_test)
    )

    print(
        f"Empty training behavior text: "
        f"{(train_text == '').sum()}"
    )

    print(
        f"Empty testing behavior text : "
        f"{(test_text == '').sum()}"
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=12000,
    )

    X_text_train = vectorizer.fit_transform(
        train_text
    )

    X_text_test = vectorizer.transform(
        test_text
    )

    print(
        f"TF-IDF vocabulary: "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Text train: {X_text_train.shape}"
    )

    print(
        f"Text test : {X_text_test.shape}"
    )

    return (
        vectorizer,
        X_text_train,
        X_text_test,
    )


# ============================================================================
# HYBRID MATRIX
# ============================================================================

def build_hybrid_matrix(
    X_struct_train,
    X_struct_test,
    X_text_train,
    X_text_test,
):
    section("Building hybrid feature matrix")

    X_train = sparse.hstack(
        [
            X_struct_train,
            X_text_train,
        ],
        format="csr",
    )

    X_test = sparse.hstack(
        [
            X_struct_test,
            X_text_test,
        ],
        format="csr",
    )

    print(
        f"Final train matrix: {X_train.shape}"
    )

    print(
        f"Final test matrix : {X_test.shape}"
    )

    print(
        f"Total model features: {X_train.shape[1]}"
    )

    return X_train, X_test


# ============================================================================
# CLASS BALANCE ANALYSIS
# ============================================================================

def calculate_class_weights(y_train, classes):
    section("Calculating MITRE class weights")

    counts = y_train.sum(axis=0).astype(float)

    total = len(y_train)

    weights = {}

    for idx, technique in enumerate(classes):
        count = counts[idx]

        if count <= 0:
            weight = 1.0
        else:
            # sqrt-based weighting:
            #
            # More stable than raw inverse frequency.
            # Prevents rare classes from receiving extreme weights.
            weight = np.sqrt(
                total / (2.0 * count)
            )

            weight = float(
                np.clip(
                    weight,
                    1.0,
                    8.0,
                )
            )

        weights[technique] = weight

    print(
        f"Minimum class count: {int(counts.min())}"
    )

    print(
        f"Maximum class count: {int(counts.max())}"
    )

    print()
    print("Class weights:")

    for technique, weight in weights.items():
        count = int(
            counts[
                list(classes).index(technique)
            ]
        )

        print(
            f"{technique:12} "
            f"support={count:6d} "
            f"weight={weight:.3f}"
        )

    return weights


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    X_train,
    y_train,
    classes,
    class_weights,
):
    header(
        "Training SmartSOC Windows-APT Model B v4.2"
    )

    print(
        "Classifier: One-vs-Rest Logistic Regression"
    )

    print(
        "Class imbalance strategy: "
        "technique-specific sample weights"
    )

    print(
        "Solver: liblinear"
    )

    print(
        "n_jobs: 1"
    )

    #
    # We use explicit per-label sample weights rather than
    # class_weight='balanced', because this is multilabel data
    # and each event may contain multiple techniques.
    #

    estimators = []

    for idx, technique in enumerate(classes):
        print(
            f"\nTraining [{idx + 1}/{len(classes)}] "
            f"{technique}..."
        )

        positive = y_train[:, idx].astype(bool)

        if positive.sum() == 0:
            print(
                f"[WARNING] No positive examples for {technique}. "
                "Skipping."
            )
            estimators.append(None)
            continue

        sample_weight = np.ones(
            len(y_train),
            dtype=np.float64,
        )

        sample_weight[positive] = (
            class_weights[technique]
        )

        estimator = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1500,
            class_weight=None,
            random_state=42,
        )

        estimator.fit(
            X_train,
            y_train[:, idx],
            sample_weight=sample_weight,
        )

        estimators.append(estimator)

    model = OneVsRestClassifier(
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=1500,
            random_state=42,
        )
    )

    #
    # We cannot directly inject the individually trained estimators
    # into OneVsRestClassifier through its public constructor.
    #
    # Therefore create a normal OneVsRestClassifier first using
    # the strongest common configuration, then replace its estimators
    # with the explicitly class-weighted estimators.
    #

    model.estimators_ = estimators
    model.classes_ = np.arange(len(classes))
    model.n_features_in_ = X_train.shape[1]

    #
    # Store metadata required by sklearn-compatible workflows.
    #

    model.n_jobs = 1

    print()
    print("[OK] All available MITRE classifiers trained.")

    return model


# ============================================================================
# PREDICTION
# ============================================================================

def predict_proba_safe(model, X):
    probabilities = []

    for estimator in model.estimators_:
        if estimator is None:
            probabilities.append(
                np.zeros(X.shape[0])
            )
            continue

        p = estimator.predict_proba(X)

        if p.shape[1] == 1:
            probabilities.append(
                np.zeros(X.shape[0])
            )
        else:
            probabilities.append(
                p[:, 1]
            )

    return np.column_stack(
        probabilities
    )


# ============================================================================
# METRICS
# ============================================================================

def evaluate_model(
    y_true,
    probabilities,
    classes,
):
    header("v4.2 Baseline Evaluation")

    #
    # IMPORTANT:
    # v4.2 training does not perform calibration yet.
    # For an unbiased initial model evaluation, use 0.5.
    #

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {}

    metrics["micro_precision"] = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["micro_recall"] = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["micro_f1"] = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["macro_precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["macro_recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["weighted_precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["weighted_recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["weighted_f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print(
        f"Micro Precision : "
        f"{metrics['micro_precision']:.4f}"
    )

    print(
        f"Micro Recall    : "
        f"{metrics['micro_recall']:.4f}"
    )

    print(
        f"Micro F1        : "
        f"{metrics['micro_f1']:.4f}"
    )

    print()

    print(
        f"Macro Precision : "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall    : "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1        : "
        f"{metrics['macro_f1']:.4f}"
    )

    print()

    print(
        f"Weighted Precision : "
        f"{metrics['weighted_precision']:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{metrics['weighted_recall']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{metrics['weighted_f1']:.4f}"
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        zero_division=0,
    )

    print()
    print("Classification Report:")
    print(report)

    no_prediction = (
        y_pred.sum(axis=1) == 0
    ).sum()

    print(
        f"Events with no prediction: "
        f"{no_prediction}"
    )

    return metrics, y_pred, report


# ============================================================================
# PER-TECHNIQUE DIAGNOSTICS
# ============================================================================

def technique_diagnostics(
    y_true,
    y_pred,
    probabilities,
    classes,
):
    section("Per-technique diagnostics")

    rows = []

    for idx, technique in enumerate(classes):

        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        prob = probabilities[:, idx]

        tp = int(
            np.sum(
                (yt == 1)
                & (yp == 1)
            )
        )

        fp = int(
            np.sum(
                (yt == 0)
                & (yp == 1)
            )
        )

        fn = int(
            np.sum(
                (yt == 1)
                & (yp == 0)
            )
        )

        support = int(
            yt.sum()
        )

        predicted = int(
            yp.sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "technique": technique,
                "support": support,
                "predicted_count": predicted,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_probability": float(
                    np.mean(prob)
                ),
            }
        )

    df = pd.DataFrame(rows)

    print(
        df.sort_values(
            "f1"
        ).to_string(
            index=False
        )
    )

    return df


# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(
    model,
    encoder,
    scaler,
    vectorizer,
    mlb,
    categorical_columns,
    numerical_columns,
    classes,
    metrics,
    report,
    diagnostics,
    schema,
):
    header("Saving v4.2 model artifacts")

    model_path = (
        MODEL_DIR
        / "multilabel_logistic_v4_2_base.joblib"
    )

    encoder_path = (
        MODEL_DIR
        / "structured_encoder_v4_2_base.joblib"
    )

    scaler_path = (
        MODEL_DIR
        / "numerical_scaler_v4_2_base.joblib"
    )

    tfidf_path = (
        MODEL_DIR
        / "tfidf_vectorizer_v4_2_base.joblib"
    )

    label_path = (
        MODEL_DIR
        / "multilabel_encoder_v4_2_base.joblib"
    )

    feature_schema_path = (
        MODEL_DIR
        / "feature_schema_v4_2_base.json"
    )

    metrics_path = (
        RESULT_DIR
        / "metrics_v4_2_base.json"
    )

    report_path = (
        RESULT_DIR
        / "classification_report_v4_2_base.txt"
    )

    diagnostics_path = (
        RESULT_DIR
        / "per_technique_diagnostics_v4_2.csv"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        encoder,
        encoder_path,
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    joblib.dump(
        vectorizer,
        tfidf_path,
    )

    joblib.dump(
        mlb,
        label_path,
    )

    final_feature_count = (
        len(encoder.get_feature_names_out())
        + len(numerical_columns)
        + len(vectorizer.vocabulary_)
    )

    feature_schema = {
        "model": "SmartSOC Windows-APT Model B v4.2",
        "version": "4.2",
        "input_dataset": "windows_apt_v4_1",
        "categorical_columns": list(
            categorical_columns
        ),
        "numerical_columns": list(
            numerical_columns
        ),
        "categorical_input_feature_count": len(
            categorical_columns
        ),
        "categorical_encoded_feature_count": len(
            encoder.get_feature_names_out()
        ),
        "numerical_feature_count": len(
            numerical_columns
        ),
        "tfidf_feature_count": len(
            vectorizer.vocabulary_
        ),
        "final_feature_count": final_feature_count,
        "mitre_technique_count": len(
            classes
        ),
        "mitre_techniques": list(
            classes
        ),
        "threshold_used_for_baseline": 0.5,
        "training_strategy": (
            "per-technique sqrt class weighting"
        ),
        "calibration_status": "NOT_CALIBRATED",
        "source_schema": schema,
    }

    with open(
        feature_schema_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            feature_schema,
            f,
            indent=2,
            default=str,
        )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    print(f"[OK] {model_path}")
    print(f"[OK] {encoder_path}")
    print(f"[OK] {scaler_path}")
    print(f"[OK] {tfidf_path}")
    print(f"[OK] {label_path}")
    print(f"[OK] {feature_schema_path}")
    print(f"[OK] {metrics_path}")
    print(f"[OK] {report_path}")
    print(f"[OK] {diagnostics_path}")

    return feature_schema


# ============================================================================
# MAIN
# ============================================================================

def main():

    warnings.filterwarnings(
        "ignore",
        message="X has feature names",
    )

    (
        structured_train,
        structured_test,
        text_train,
        text_test,
        labels_train,
        labels_test,
        metadata_train,
        metadata_test,
        schema,
    ) = load_data()

    check_alignment(
        structured_train,
        structured_test,
        text_train,
        text_test,
        labels_train,
        labels_test,
        metadata_train,
        metadata_test,
    )

    categorical_columns, numerical_columns = (
        detect_columns(
            structured_train,
            schema,
        )
    )

    (
        mlb,
        y_train,
        y_test,
        train_lists,
        test_lists,
    ) = prepare_labels(
        labels_train,
        labels_test,
    )

    (
        structured_train_clean,
        structured_test_clean,
    ) = clean_structured(
        structured_train,
        structured_test,
        categorical_columns,
        numerical_columns,
    )

    (
        encoder,
        scaler,
        X_struct_train,
        X_struct_test,
    ) = build_structured_features(
        structured_train_clean,
        structured_test_clean,
        categorical_columns,
        numerical_columns,
    )

    (
        vectorizer,
        X_text_train,
        X_text_test,
    ) = build_text_features(
        text_train,
        text_test,
    )

    (
        X_train,
        X_test,
    ) = build_hybrid_matrix(
        X_struct_train,
        X_struct_test,
        X_text_train,
        X_text_test,
    )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Train/test feature dimensions do not match."
        )

    class_weights = calculate_class_weights(
        y_train,
        mlb.classes_,
    )

    model = train_model(
        X_train,
        y_train,
        mlb.classes_,
        class_weights,
    )

    print()
    print(
        "Generating test probabilities..."
    )

    probabilities = predict_proba_safe(
        model,
        X_test,
    )

    print(
        f"Probability matrix: "
        f"{probabilities.shape}"
    )

    (
        metrics,
        y_pred,
        report,
    ) = evaluate_model(
        y_test,
        probabilities,
        mlb.classes_,
    )

    diagnostics = technique_diagnostics(
        y_test,
        y_pred,
        probabilities,
        mlb.classes_,
    )

    feature_schema = save_artifacts(
        model,
        encoder,
        scaler,
        vectorizer,
        mlb,
        categorical_columns,
        numerical_columns,
        mlb.classes_,
        metrics,
        report,
        diagnostics,
        schema,
    )

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "Training Complete"
    )

    print(
        f"Training events       : {len(y_train)}"
    )

    print(
        f"Testing events        : {len(y_test)}"
    )

    print(
        f"Structured features   : "
        f"{len(categorical_columns) + len(numerical_columns)}"
    )

    print(
        f"TF-IDF features       : "
        f"{len(vectorizer.vocabulary_)}"
    )

    print(
        f"Final features        : "
        f"{feature_schema['final_feature_count']}"
    )

    print(
        f"MITRE techniques      : "
        f"{len(mlb.classes_)}"
    )

    print()
    print(
        f"Micro F1              : "
        f"{metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro F1              : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1           : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print()
    print(
        "Calibration status: NOT CALIBRATED"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the v4.2 diagnostic before calibration."
    )

    print(
        "Do NOT deploy v4.2 yet."
    )


if __name__ == "__main__":
    main()