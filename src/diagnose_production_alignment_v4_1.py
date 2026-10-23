from pathlib import Path
import json
import hashlib
import warnings

import joblib
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

SCHEMA_FILE = PRODUCTION / "feature_schema.json"
MODEL_FILE = PRODUCTION / "model.joblib"
ENCODER_FILE = PRODUCTION / "structured_encoder.joblib"
SCALER_FILE = PRODUCTION / "numerical_scaler.joblib"
TFIDF_FILE = PRODUCTION / "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = PRODUCTION / "multilabel_encoder.joblib"
THRESHOLDS_FILE = PRODUCTION / "thresholds.json"

TRAIN_PREPARE = (
    ROOT
    / "src"
    / "prepare_windows_apt_v4_1.py"
)

TRAIN_FILE = (
    ROOT
    / "src"
    / "train_windows_apt_v4_1.py"
)

ALIGNED_INFERENCE = (
    ROOT
    / "src"
    / "production_inference_windows_apt_v4_1_aligned.py"
)


# =============================================================================
# HELPERS
# =============================================================================

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


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )
    print(f"[OK] {path}")


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1"
        " - Production Semantic Alignment Diagnostic"
    )

    print()
    print("IMPORTANT:")
    print("This script DOES NOT modify the production model.")
    print("This script DOES NOT retrain anything.")
    print("This script only diagnoses preprocessing alignment.")

    # =========================================================================
    # 1. CHECK ARTIFACTS
    # =========================================================================

    section("1. Checking production artifacts")

    required = [
        SCHEMA_FILE,
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        THRESHOLDS_FILE,
        TRAIN_PREPARE,
        TRAIN_FILE,
        ALIGNED_INFERENCE,
    ]

    for path in required:
        require(path)

    # =========================================================================
    # 2. FILE HASHES
    # =========================================================================

    section("2. Production artifact SHA-256")

    for path in [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        LABEL_ENCODER_FILE,
        SCHEMA_FILE,
        THRESHOLDS_FILE,
    ]:

        print(
            f"{path.name:<35} "
            f"{sha256(path)}"
        )

    # =========================================================================
    # 3. LOAD SCHEMA
    # =========================================================================

    section("3. Loading production feature schema")

    schema = load_json(SCHEMA_FILE)

    categorical = schema.get(
        "categorical_columns",
        []
    )

    numerical = schema.get(
        "numerical_columns",
        []
    )

    print(
        f"Categorical input features : "
        f"{len(categorical)}"
    )

    print(
        f"Numerical features         : "
        f"{len(numerical)}"
    )

    print(
        f"Encoded categorical        : "
        f"{schema.get('categorical_encoded_feature_count')}"
    )

    print(
        f"TF-IDF features            : "
        f"{schema.get('tfidf_feature_count')}"
    )

    print(
        f"Final features             : "
        f"{schema.get('final_feature_count')}"
    )

    print()
    print("Categorical feature order:")

    for i, name in enumerate(categorical):
        print(
            f"{i:02d}: {name}"
        )

    print()
    print("Numerical feature order:")

    for i, name in enumerate(numerical):
        print(
            f"{i:02d}: {name}"
        )

    # =========================================================================
    # 4. LOAD ARTIFACTS
    # =========================================================================

    section("4. Loading fitted artifacts")

    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    label_encoder = joblib.load(
        LABEL_ENCODER_FILE
    )

    print(
        "Model      :",
        type(model).__name__
    )

    print(
        "Encoder    :",
        type(encoder).__name__
    )

    print(
        "Scaler     :",
        type(scaler).__name__
    )

    print(
        "TF-IDF     :",
        type(tfidf).__name__
    )

    print(
        "Label      :",
        type(label_encoder).__name__
    )

    # =========================================================================
    # 5. SCALER ORDER DIAGNOSTIC
    # =========================================================================

    section(
        "5. Numerical scaler semantic-order diagnostic"
    )

    scaler_count = getattr(
        scaler,
        "n_features_in_",
        None
    )

    print(
        "Scaler feature count:",
        scaler_count
    )

    print(
        "Schema numerical count:",
        len(numerical)
    )

    if scaler_count != len(numerical):

        print(
            "[FAIL] Scaler and schema dimensions differ."
        )

    else:

        print(
            "[PASS] Scaler dimension = schema dimension."
        )

    print()
    print(
        "The StandardScaler does NOT contain feature names."
    )

    print(
        "Therefore numerical feature ORDER is critical."
    )

    print()
    print(
        "Scaler means by schema feature:"
    )

    means = getattr(
        scaler,
        "mean_",
        None
    )

    scales = getattr(
        scaler,
        "scale_",
        None
    )

    if means is not None:

        for i, name in enumerate(numerical):

            print(
                f"{i:02d} "
                f"{name:<40} "
                f"mean={means[i]:.8f} "
                f"scale={scales[i]:.8f}"
            )

    # =========================================================================
    # 6. ENCODER DIAGNOSTIC
    # =========================================================================

    section(
        "6. OneHotEncoder semantic diagnostic"
    )

    encoder_features = getattr(
        encoder,
        "feature_names_in_",
        None
    )

    if encoder_features is not None:

        print(
            "Encoder feature names:"
        )

        for i, name in enumerate(
            encoder_features
        ):

            print(
                f"{i:02d}: {name}"
            )

        if list(encoder_features) == list(
            categorical
        ):

            print(
                "[PASS] Encoder input order matches schema."
            )

        else:

            print(
                "[FAIL] Encoder input order DOES NOT match schema."
            )

            print()
            print(
                "Encoder:",
                list(encoder_features)
            )

            print(
                "Schema :",
                categorical
            )

    else:

        print(
            "[WARNING] Encoder does not expose "
            "feature_names_in_."
        )

        print(
            "Checking category structure instead."
        )

    encoded_dimension = getattr(
        encoder,
        "transform",
        None
    )

    categories = getattr(
        encoder,
        "categories_",
        None
    )

    if categories is not None:

        print()
        print(
            "Encoder category counts:"
        )

        total = 0

        for i, cats in enumerate(
            categories
        ):

            count = len(cats)

            total += count

            feature_name = (
                categorical[i]
                if i < len(categorical)
                else f"feature_{i}"
            )

            print(
                f"{i:02d} "
                f"{feature_name:<30} "
                f"{count}"
            )

        print()
        print(
            "Calculated encoded dimension:",
            total
        )

        expected_encoded = schema.get(
            "categorical_encoded_feature_count"
        )

        if total == expected_encoded:

            print(
                "[PASS] Encoder encoded dimension matches schema."
            )

        else:

            print(
                "[FAIL] Encoder encoded dimension mismatch."
            )

    # =========================================================================
    # 7. TF-IDF DIAGNOSTIC
    # =========================================================================

    section(
        "7. TF-IDF semantic diagnostic"
    )

    print(
        "Vocabulary:",
        len(tfidf.vocabulary_)
    )

    print(
        "Expected:",
        schema.get("tfidf_feature_count")
    )

    attributes = [
        "lowercase",
        "strip_accents",
        "ngram_range",
        "min_df",
        "max_df",
        "sublinear_tf",
        "max_features",
        "token_pattern",
    ]

    for attr in attributes:

        print(
            f"{attr:<20}: "
            f"{getattr(tfidf, attr, None)}"
        )

    print()

    schema_tfidf = schema.get(
        "tfidf_configuration",
        {}
    )

    mismatches = []

    for attr in [
        "lowercase",
        "strip_accents",
        "ngram_range",
        "min_df",
        "max_df",
        "sublinear_tf",
        "max_features",
    ]:

        actual = getattr(
            tfidf,
            attr,
            None
        )

        expected = schema_tfidf.get(
            attr
        )

        if actual != expected:

            mismatches.append(
                (
                    attr,
                    expected,
                    actual
                )
            )

    if not mismatches:

        print(
            "[PASS] TF-IDF configuration matches schema."
        )

    else:

        print(
            "[FAIL] TF-IDF configuration mismatch."
        )

        for item in mismatches:

            print(
                f"  {item[0]} "
                f"expected={item[1]!r} "
                f"actual={item[2]!r}"
            )

    # =========================================================================
    # 8. MODEL DIAGNOSTIC
    # =========================================================================

    section(
        "8. Model input/output diagnostic"
    )

    model_features = getattr(
        model,
        "n_features_in_",
        None
    )

    print(
        "Model input features:",
        model_features
    )

    print(
        "Schema final features:",
        schema.get("final_feature_count")
    )

    if (
        model_features
        == schema.get("final_feature_count")
    ):

        print(
            "[PASS] Model input dimension matches schema."
        )

    else:

        print(
            "[FAIL] Model input dimension mismatch."
        )

    classes = getattr(
        label_encoder,
        "classes_",
        []
    )

    print(
        "MITRE classes:",
        len(classes)
    )

    print(
        "Schema MITRE classes:",
        len(schema.get("mitre_techniques", []))
    )

    if list(classes) == list(
        schema.get("mitre_techniques", [])
    ):

        print(
            "[PASS] MITRE class order matches schema."
        )

    else:

        print(
            "[FAIL] MITRE class order mismatch."
        )

        print()
        print("Encoder classes:")

        for i, value in enumerate(classes):
            print(
                f"{i:02d}: {value}"
            )

        print()
        print("Schema classes:")

        for i, value in enumerate(
            schema.get("mitre_techniques", [])
        ):
            print(
                f"{i:02d}: {value}"
            )

    # =========================================================================
    # 9. SOURCE CODE CHECKS
    # =========================================================================

    section(
        "9. Source-code semantic feature checks"
    )

    prepare_text = TRAIN_PREPARE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    train_text = TRAIN_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    inference_text = ALIGNED_INFERENCE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    feature_names = (
        categorical
        + numerical
    )

    print(
        "Checking v4.1 numerical features "
        "against preparation source..."
    )

    prepare_missing = []

    for feature in numerical:

        if feature not in prepare_text:

            prepare_missing.append(
                feature
            )

    if not prepare_missing:

        print(
            "[PASS] Every numerical feature name "
            "exists somewhere in prepare_windows_apt_v4_1.py."
        )

    else:

        print(
            "[WARNING] Numerical features not found "
            "in preparation source:"
        )

        for feature in prepare_missing:

            print(
                " -",
                feature
            )

    print()
    print(
        "Checking inference source for numerical features..."
    )

    inference_missing = []

    for feature in numerical:

        if feature not in inference_text:

            inference_missing.append(
                feature
            )

    if not inference_missing:

        print(
            "[PASS] Every numerical feature name "
            "exists in aligned inference source."
        )

    else:

        print(
            "[WARNING] Numerical features missing "
            "from aligned inference source:"
        )

        for feature in inference_missing:

            print(
                " -",
                feature
            )

    # =========================================================================
    # 10. CRITICAL BEHAVIOR FEATURES
    # =========================================================================

    section(
        "10. Critical behavioral-feature source audit"
    )

    critical = [
        "uses_powershell",
        "uses_cmd",
        "uses_net",
        "uses_whoami",
        "uses_schtasks",
        "uses_reg",
        "uses_rundll32",
        "uses_wmic",
        "uses_certutil",
        "uses_bitsadmin",
        "uses_ipconfig",
        "uses_netstat",
        "uses_tasklist",
        "uses_sc",
        "uses_mshta",
        "uses_regsvr32",
        "uses_cscript",
        "uses_wscript",
        "uses_wevtutil",
        "uses_vssadmin",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "administrative_powershell_context",
        "strong_behavior_signal",
    ]

    print()
    print(
        f"{'FEATURE':<35} "
        f"{'PREPARE':<10} "
        f"{'INFERENCE':<10}"
    )

    print("-" * 60)

    for feature in critical:

        in_prepare = (
            "YES"
            if feature in prepare_text
            else "NO"
        )

        in_inference = (
            "YES"
            if feature in inference_text
            else "NO"
        )

        print(
            f"{feature:<35} "
            f"{in_prepare:<10} "
            f"{in_inference:<10}"
        )

    # =========================================================================
    # 11. BEHAVIOR TEXT CHECK
    # =========================================================================

    section(
        "11. Behavioral text construction audit"
    )

    behavior_tokens = [
        "command_line",
        "parent_command_line",
        "image",
        "parent_image",
        "originalFileName",
        "targetFilename",
        "imageLoaded",
        "targetObject",
    ]

    print(
        "Training preparation source:"
    )

    for token in behavior_tokens:

        print(
            f"  {token:<25} "
            f"{'FOUND' if token in prepare_text else 'MISSING'}"
        )

    print()
    print(
        "Aligned inference source:"
    )

    for token in behavior_tokens:

        print(
            f"  {token:<25} "
            f"{'FOUND' if token in inference_text else 'MISSING'}"
        )

    # =========================================================================
    # 12. PATH CONSISTENCY
    # =========================================================================

    section(
        "12. Production-path consistency"
    )

    print(
        "Production directory:"
    )

    print(PRODUCTION)

    print()
    print(
        "Aligned inference references:"
    )

    path_strings = [
        "models/windows_apt_v4_1/production",
        "models\\windows_apt_v4_1\\production",
        "models/windows_apt_v4_1",
        "models\\windows_apt_v4_1",
    ]

    found_paths = []

    for value in path_strings:

        if value in inference_text:

            found_paths.append(value)

    if found_paths:

        for value in found_paths:

            print(
                "[FOUND]",
                value
            )

    else:

        print(
            "[INFO] No literal normalized production path found."
        )

    # =========================================================================
    # 13. FINAL VERDICT
    # =========================================================================

    header(
        "FINAL DIAGNOSTIC VERDICT"
    )

    problems = 0

    if scaler_count != len(numerical):
        problems += 1

    if len(tfidf.vocabulary_) != schema.get(
        "tfidf_feature_count"
    ):
        problems += 1

    if model_features != schema.get(
        "final_feature_count"
    ):
        problems += 1

    if list(classes) != list(
        schema.get("mitre_techniques", [])
    ):
        problems += 1

    if mismatches:
        problems += len(mismatches)

    if problems == 0:

        print(
            "[PASS] Artifact-level alignment is consistent."
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This means the production artifacts themselves "
            "are internally consistent."
        )

        print()
        print(
            "The remaining question is SEMANTIC alignment:"
        )

        print(
            "Does the aligned inference code calculate "
            "each feature exactly as the training preparation code?"
        )

        print()
        print(
            "Do NOT retrain or modify v4.1 yet."
        )

    else:

        print(
            f"[FAIL] {problems} alignment issue(s) detected."
        )

        print()
        print(
            "Do NOT connect this pipeline to Wazuh yet."
        )

    print()
    print(
        "Diagnostic complete."
    )


if __name__ == "__main__":
    main()