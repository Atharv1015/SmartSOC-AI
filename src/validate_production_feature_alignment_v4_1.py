from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SmartSOC Windows-APT Model B v4.1
# Production Feature Alignment Validator
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

SCHEMA_FILE = PRODUCTION_DIR / "feature_schema.json"
ENCODER_FILE = PRODUCTION_DIR / "structured_encoder.joblib"
SCALER_FILE = PRODUCTION_DIR / "numerical_scaler.joblib"
TFIDF_FILE = PRODUCTION_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = PRODUCTION_DIR / "model.joblib"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
TEXT_TEST = DATA_DIR / "behavior_text_test.csv"
LABEL_TEST = DATA_DIR / "labels_test.csv"


# ============================================================================
# Helpers
# ============================================================================

def fail(message: str):
    print()
    print("[FAIL]", message)
    sys.exit(1)


def ok(message: str):
    print("[OK]", message)


def load_json(path: Path):
    if not path.exists():
        fail(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require(path: Path):
    if not path.exists():
        fail(f"Required artifact not found: {path}")

    ok(str(path))


# ============================================================================
# Main
# ============================================================================

def main():

    print()
    print("=" * 78)
    print("SmartSOC Windows-APT Model B v4.1")
    print("Production Feature Alignment Validation")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Required artifacts
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Checking production artifacts")
    print("-" * 78)

    for path in [
        SCHEMA_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        MODEL_FILE,
    ]:
        require(path)

    print()
    print("-" * 78)
    print("Checking v4.1 temporal test data")
    print("-" * 78)

    for path in [
        STRUCTURED_TEST,
        TEXT_TEST,
        LABEL_TEST,
    ]:
        require(path)

    # ------------------------------------------------------------------------
    # Load schema
    # ------------------------------------------------------------------------

    schema = load_json(SCHEMA_FILE)

    categorical_columns = schema.get(
        "categorical_columns",
        []
    )

    numerical_columns = schema.get(
        "numerical_columns",
        []
    )

    expected_cat_count = int(
        schema.get(
            "categorical_input_feature_count",
            len(categorical_columns),
        )
    )

    expected_encoded_count = int(
        schema.get(
            "categorical_encoded_feature_count",
            0,
        )
    )

    expected_num_count = int(
        schema.get(
            "numerical_feature_count",
            len(numerical_columns),
        )
    )

    expected_tfidf_count = int(
        schema.get(
            "tfidf_feature_count",
            0,
        )
    )

    expected_final_count = int(
        schema.get(
            "final_feature_count",
            0,
        )
    )

    print()
    print("-" * 78)
    print("Production feature schema")
    print("-" * 78)

    print("Categorical input features :", len(categorical_columns))
    print("Expected categorical       :", expected_cat_count)

    print("Numerical features         :", len(numerical_columns))
    print("Expected numerical         :", expected_num_count)

    print("Encoded categorical        :", expected_encoded_count)
    print("TF-IDF features             :", expected_tfidf_count)
    print("Final features              :", expected_final_count)

    if len(categorical_columns) != expected_cat_count:
        fail(
            "Categorical feature count does not match schema."
        )

    if len(numerical_columns) != expected_num_count:
        fail(
            "Numerical feature count does not match schema."
        )

    if (
        expected_encoded_count
        + expected_num_count
        + expected_tfidf_count
        != expected_final_count
    ):
        fail(
            "Feature arithmetic is inconsistent."
        )

    ok("Feature schema arithmetic is consistent.")

    # ------------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Loading production preprocessing artifacts")
    print("-" * 78)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        encoder = joblib.load(
            ENCODER_FILE
        )

        scaler = joblib.load(
            SCALER_FILE
        )

        tfidf = joblib.load(
            TFIDF_FILE
        )

        model = joblib.load(
            MODEL_FILE
        )

    print("Encoder :", type(encoder).__name__)
    print("Scaler  :", type(scaler).__name__)
    print("TF-IDF  :", type(tfidf).__name__)
    print("Model   :", type(model).__name__)

    # ------------------------------------------------------------------------
    # Validate encoder
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Validating categorical encoder")
    print("-" * 78)

    encoder_categories = getattr(
        encoder,
        "categories_",
        None,
    )

    if encoder_categories is None:
        fail(
            "Structured encoder has no categories_."
        )

    print(
        "Encoder input columns :",
        len(encoder_categories),
    )

    print(
        "Encoder output columns:",
        len(
            encoder.get_feature_names_out()
        ),
    )

    if len(encoder_categories) != expected_cat_count:
        fail(
            "Encoder input feature count mismatch."
        )

    actual_encoded_count = len(
        encoder.get_feature_names_out()
    )

    if actual_encoded_count != expected_encoded_count:
        fail(
            "Encoder output feature count mismatch."
        )

    ok(
        f"Encoder output = {actual_encoded_count}"
    )

    # ------------------------------------------------------------------------
    # Validate scaler
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Validating numerical scaler")
    print("-" * 78)

    scaler_count = getattr(
        scaler,
        "n_features_in_",
        None,
    )

    print(
        "Scaler features:",
        scaler_count,
    )

    if scaler_count != expected_num_count:
        fail(
            "Scaler feature count does not match "
            "the 58-feature production schema."
        )

    mean_values = getattr(
        scaler,
        "mean_",
        None,
    )

    scale_values = getattr(
        scaler,
        "scale_",
        None,
    )

    if mean_values is None:
        fail(
            "Scaler has no mean_ attribute."
        )

    if scale_values is None:
        fail(
            "Scaler has no scale_ attribute."
        )

    if len(mean_values) != expected_num_count:
        fail(
            "Scaler mean_ length mismatch."
        )

    if len(scale_values) != expected_num_count:
        fail(
            "Scaler scale_ length mismatch."
        )

    print(
        "Scaler mean length :",
        len(mean_values),
    )

    print(
        "Scaler scale length:",
        len(scale_values),
    )

    ok("Numerical scaler dimensions are correct.")

    # ------------------------------------------------------------------------
    # Validate TF-IDF
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Validating TF-IDF vectorizer")
    print("-" * 78)

    tfidf_vocab = len(
        tfidf.vocabulary_
    )

    print(
        "Vocabulary:",
        tfidf_vocab,
    )

    print(
        "Expected  :",
        expected_tfidf_count,
    )

    if tfidf_vocab != expected_tfidf_count:
        fail(
            "TF-IDF vocabulary size does not match "
            "production schema."
        )

    config = schema.get(
        "tfidf_configuration",
        {},
    )

    print()
    print("TF-IDF configuration:")

    for key in [
        "lowercase",
        "strip_accents",
        "ngram_range",
        "min_df",
        "max_df",
        "sublinear_tf",
        "max_features",
    ]:
        print(
            f"  {key:<15}:",
            getattr(
                tfidf,
                key,
                None,
            ),
        )

    if (
        getattr(tfidf, "lowercase", None)
        != config.get("lowercase")
    ):
        fail("TF-IDF lowercase mismatch.")

    if (
        getattr(tfidf, "strip_accents", None)
        != config.get("strip_accents")
    ):
        fail("TF-IDF strip_accents mismatch.")

    if tuple(
        getattr(tfidf, "ngram_range", ())
    ) != tuple(
        config.get("ngram_range", ())
    ):
        fail("TF-IDF ngram_range mismatch.")

    if (
        getattr(tfidf, "min_df", None)
        != config.get("min_df")
    ):
        fail("TF-IDF min_df mismatch.")

    if (
        getattr(tfidf, "max_df", None)
        != config.get("max_df")
    ):
        fail("TF-IDF max_df mismatch.")

    if (
        getattr(tfidf, "sublinear_tf", None)
        != config.get("sublinear_tf")
    ):
        fail("TF-IDF sublinear_tf mismatch.")

    ok("TF-IDF configuration matches schema.")

    # ------------------------------------------------------------------------
    # Load temporal test set
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Loading v4.1 temporal test set")
    print("-" * 78)

    structured_test = pd.read_csv(
        STRUCTURED_TEST
    )

    text_test = pd.read_csv(
        TEXT_TEST
    )

    labels_test = pd.read_csv(
        LABEL_TEST
    )

    print(
        "Structured:",
        structured_test.shape,
    )

    print(
        "Text      :",
        text_test.shape,
    )

    print(
        "Labels    :",
        labels_test.shape,
    )

    if not (
        len(structured_test)
        == len(text_test)
        == len(labels_test)
    ):
        fail(
            "Test datasets are not row aligned."
        )

    ok(
        f"Test rows aligned: {len(structured_test)}"
    )

    # ------------------------------------------------------------------------
    # Validate structured columns
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Validating structured test schema")
    print("-" * 78)

    actual_columns = list(
        structured_test.columns
    )

    expected_columns = (
        categorical_columns
        + numerical_columns
    )

    print(
        "Structured columns:",
        len(actual_columns),
    )

    print(
        "Expected columns  :",
        len(expected_columns),
    )

    missing_columns = [
        c
        for c in expected_columns
        if c not in actual_columns
    ]

    extra_columns = [
        c
        for c in actual_columns
        if c not in expected_columns
    ]

    if missing_columns:
        print()
        print("Missing columns:")
        for c in missing_columns:
            print("  -", c)

        fail(
            "Structured test data is missing "
            "required production features."
        )

    if extra_columns:
        print()
        print("Extra columns:")
        for c in extra_columns:
            print("  -", c)

    ok(
        "All required structured features exist."
    )

    # ------------------------------------------------------------------------
    # Prepare categorical data exactly as training
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Testing categorical transformation")
    print("-" * 78)

    X_cat = structured_test[
        categorical_columns
    ].copy()

    for col in categorical_columns:

        X_cat[col] = (
            X_cat[col]
            .fillna("<missing>")
            .astype(str)
            .str.strip()
            .replace(
                "",
                "<missing>",
            )
        )

    X_cat_encoded = encoder.transform(
        X_cat
    )

    print(
        "Categorical matrix:",
        X_cat_encoded.shape,
    )

    if X_cat_encoded.shape[1] != expected_encoded_count:
        fail(
            "Categorical encoded dimensions "
            "do not match production."
        )

    ok(
        f"Categorical output = {X_cat_encoded.shape[1]}"
    )

    # ------------------------------------------------------------------------
    # Prepare numerical data exactly as training representation
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Testing numerical transformation")
    print("-" * 78)

    X_num = structured_test[
        numerical_columns
    ].copy()

    for col in numerical_columns:
        X_num[col] = pd.to_numeric(
            X_num[col],
            errors="coerce",
        )

    nan_before = int(
        X_num.isna().sum().sum()
    )

    print(
        "NaN values before fill:",
        nan_before,
    )

    # The v4.1 training data preparation uses
    # numeric conversion followed by training-time
    # missing-value handling. Production events should
    # not pass NaN into the scaler.
    #
    # For the existing v4.1 structured dataset, missing
    # numerical values are represented as zero-equivalent
    # values after preprocessing. Therefore zero is used
    # only for missing live values here.
    X_num = X_num.fillna(0.0)

    nan_after = int(
        X_num.isna().sum().sum()
    )

    print(
        "NaN values after fill :",
        nan_after,
    )

    X_num_array = X_num.to_numpy(
        dtype=np.float64
    )

    if X_num_array.shape[1] != expected_num_count:
        fail(
            "Numerical matrix does not have 58 columns."
        )

    # IMPORTANT:
    # Pass NumPy to the scaler, matching training behavior.
    X_num_scaled = scaler.transform(
        X_num_array
    )

    print(
        "Numerical matrix:",
        X_num_array.shape,
    )

    print(
        "Scaled matrix   :",
        X_num_scaled.shape,
    )

    if X_num_scaled.shape[1] != expected_num_count:
        fail(
            "Scaled numerical matrix dimension mismatch."
        )

    if not np.isfinite(
        X_num_scaled.data
        if hasattr(X_num_scaled, "data")
        else X_num_scaled
    ).all():
        fail(
            "Non-finite values found after scaling."
        )

    ok(
        f"Numerical output = {X_num_scaled.shape[1]}"
    )

    # ------------------------------------------------------------------------
    # Structured feature total
    # ------------------------------------------------------------------------

    structured_feature_count = (
        X_cat_encoded.shape[1]
        + X_num_scaled.shape[1]
    )

    print()
    print(
        "Structured feature count:",
        structured_feature_count,
    )

    if structured_feature_count != (
        expected_encoded_count
        + expected_num_count
    ):
        fail(
            "Structured feature count mismatch."
        )

    ok(
        "Structured feature dimensions are correct."
    )

    # ------------------------------------------------------------------------
    # TF-IDF transformation
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Testing TF-IDF transformation")
    print("-" * 78)

    # Find the normalized text column used by v4.1.
    text_column = None

    for candidate in [
        "behavior_text_normalized",
        "behavior_text",
    ]:
        if candidate in text_test.columns:
            text_column = candidate
            break

    if text_column is None:
        fail(
            "Could not find behavior text column."
        )

    behavior_test = (
        text_test[text_column]
        .fillna("")
        .astype(str)
    )

    print(
        "Text column:",
        text_column,
    )

    print(
        "Empty text:",
        int(
            (behavior_test.str.strip() == "")
            .sum()
        ),
    )

    X_text = tfidf.transform(
        behavior_test
    )

    print(
        "TF-IDF matrix:",
        X_text.shape,
    )

    if X_text.shape[1] != expected_tfidf_count:
        fail(
            "TF-IDF output dimension mismatch."
        )

    ok(
        f"TF-IDF output = {X_text.shape[1]}"
    )

    # ------------------------------------------------------------------------
    # Final feature count
    # ------------------------------------------------------------------------

    final_count = (
        structured_feature_count
        + X_text.shape[1]
    )

    print()
    print("-" * 78)
    print("Final feature alignment")
    print("-" * 78)

    print(
        "Categorical encoded:",
        X_cat_encoded.shape[1],
    )

    print(
        "Numerical scaled   :",
        X_num_scaled.shape[1],
    )

    print(
        "TF-IDF             :",
        X_text.shape[1],
    )

    print(
        "Final expected     :",
        expected_final_count,
    )

    print(
        "Final calculated   :",
        final_count,
    )

    if final_count != expected_final_count:
        fail(
            "FINAL FEATURE COUNT MISMATCH."
        )

    ok(
        f"Final feature count = {final_count}"
    )

    # ------------------------------------------------------------------------
    # Model input dimension
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("Validating model input dimension")
    print("-" * 78)

    model_n_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    print(
        "Model n_features_in_:",
        model_n_features,
    )

    if model_n_features is not None:
        if model_n_features != expected_final_count:
            fail(
                "Model input dimension does not match "
                "production feature schema."
            )

        ok(
            "Model input dimension matches 6919."
        )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("PRODUCTION FEATURE ALIGNMENT VALIDATION COMPLETE")
    print("=" * 78)

    print()
    print("Categorical input       :", expected_cat_count)
    print("Categorical encoded     :", expected_encoded_count)
    print("Numerical               :", expected_num_count)
    print("TF-IDF                  :", expected_tfidf_count)
    print("Final                   :", expected_final_count)
    print("Temporal test rows      :", len(structured_test))
    print()

    print("[PASS] Production schema is internally consistent.")
    print("[PASS] Encoder dimensions are consistent.")
    print("[PASS] Scaler dimensions are consistent.")
    print("[PASS] TF-IDF dimensions/configuration are consistent.")
    print("[PASS] Temporal test data transforms to expected dimensions.")
    print("[PASS] Model expects the expected feature count.")

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This validates preprocessing dimensions and artifact "
        "compatibility."
    )
    print(
        "It does NOT yet prove that raw Wazuh JSON is converted "
        "into the exact v4.1 structured features."
    )
    print()


if __name__ == "__main__":
    main()