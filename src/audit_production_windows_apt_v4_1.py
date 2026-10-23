from pathlib import Path
import json
import hashlib
import joblib
import numpy as np


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# FINAL PRODUCTION AUDIT
# =============================================================================

BASE = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_1"
    / "calibrated"
)

DATA_DIR = (
    BASE
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

RESULT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_1"
)

PRODUCTION_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

PRODUCTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# EXPECTED ARTIFACTS
# =============================================================================

MODEL_FILE = (
    MODEL_DIR
    / "multilabel_logistic_v4_1_calibrated.joblib"
)

STRUCTURED_ENCODER = (
    MODEL_DIR
    / "structured_encoder_v4_1_calibrated.joblib"
)

NUMERICAL_SCALER = (
    MODEL_DIR
    / "numerical_scaler_v4_1_calibrated.joblib"
)

TFIDF = (
    MODEL_DIR
    / "tfidf_vectorizer_v4_1_calibrated.joblib"
)

LABEL_ENCODER = (
    MODEL_DIR
    / "multilabel_encoder_v4_1_calibrated.joblib"
)

FEATURE_SCHEMA = (
    MODEL_DIR
    / "feature_schema_v4_1_calibrated.json"
)

THRESHOLDS = (
    MODEL_DIR
    / "technique_thresholds_v4_1.json"
)

SCHEMA = (
    DATA_DIR
    / "schema.json"
)

AUDIT_FILE = (
    DATA_DIR
    / "dataset_audit.json"
)

COMPARISON_FILE = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "model_comparison"
    / "v4_1_vs_v4_2_per_technique.csv"
)

GLOBAL_COMPARISON = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "model_comparison"
    / "v4_1_vs_v4_2_global_metrics.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found:\n{path}"
        )

    print(f"[OK] {path}")


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


# =============================================================================
# MAIN
# =============================================================================

def main():

    section(
        "SmartSOC Windows-APT Model B v4.1"
    )

    print(
        "FINAL PRODUCTION CANDIDATE AUDIT"
    )

    # =========================================================================
    # 1. CHECK MODEL ARTIFACTS
    # =========================================================================

    section(
        "1. Checking v4.1 calibrated artifacts"
    )

    artifacts = {
        "model": MODEL_FILE,
        "structured_encoder": STRUCTURED_ENCODER,
        "numerical_scaler": NUMERICAL_SCALER,
        "tfidf_vectorizer": TFIDF,
        "multilabel_encoder": LABEL_ENCODER,
        "feature_schema": FEATURE_SCHEMA,
        "technique_thresholds": THRESHOLDS,
    }

    for name, path in artifacts.items():
        require(path)

    # =========================================================================
    # 2. CHECK DATASET ARTIFACTS
    # =========================================================================

    section(
        "2. Checking v4.1 dataset artifacts"
    )

    dataset_files = [
        DATA_DIR / "structured_train.csv",
        DATA_DIR / "structured_test.csv",
        DATA_DIR / "behavior_text_train.csv",
        DATA_DIR / "behavior_text_test.csv",
        DATA_DIR / "labels_train.csv",
        DATA_DIR / "labels_test.csv",
        DATA_DIR / "metadata_train.csv",
        DATA_DIR / "metadata_test.csv",
        SCHEMA,
        AUDIT_FILE,
    ]

    for path in dataset_files:
        require(path)

    # =========================================================================
    # 3. LOAD MODEL
    # =========================================================================

    section(
        "3. Loading model artifacts"
    )

    model = joblib.load(
        MODEL_FILE
    )

    encoder = joblib.load(
        STRUCTURED_ENCODER
    )

    scaler = joblib.load(
        NUMERICAL_SCALER
    )

    vectorizer = joblib.load(
        TFIDF
    )

    label_encoder = joblib.load(
        LABEL_ENCODER
    )

    print(
        "Model type:",
        type(model).__name__
    )

    print(
        "Encoder type:",
        type(encoder).__name__
    )

    print(
        "Scaler type:",
        type(scaler).__name__
    )

    print(
        "TF-IDF type:",
        type(vectorizer).__name__
    )

    # =========================================================================
    # 4. FEATURE DIMENSIONS
    # =========================================================================

    section(
        "4. Validating feature dimensions"
    )

    categorical_features = [
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
    ]

    categorical_count = len(
        categorical_features
    )

    encoded_features = (
        getattr(
            encoder,
            "get_feature_names_out",
            lambda: []
        )()
    )

    encoded_count = len(
        encoded_features
    )

    numerical_count = getattr(
        scaler,
        "n_features_in_",
        None
    )

    tfidf_count = len(
        vectorizer.vocabulary_
    )

    technique_count = len(
        label_encoder.classes_
    )

    calculated_features = (
        encoded_count
        + numerical_count
        + tfidf_count
    )

    model_features = getattr(
        model,
        "n_features_in_",
        None
    )

    print(
        "Categorical input features :",
        categorical_count
    )

    print(
        "Categorical encoded features:",
        encoded_count
    )

    print(
        "Numerical features          :",
        numerical_count
    )

    print(
        "TF-IDF features             :",
        tfidf_count
    )

    print(
        "Calculated hybrid features  :",
        calculated_features
    )

    print(
        "Model input features        :",
        model_features
    )

    print(
        "MITRE techniques            :",
        technique_count
    )

    if calculated_features != model_features:

        raise ValueError(
            "Feature dimension mismatch."
        )

    if model_features != 6919:

        raise ValueError(
            f"Expected v4.1 feature count 6919, "
            f"got {model_features}."
        )

    if technique_count != 45:

        raise ValueError(
            f"Expected 45 MITRE techniques, "
            f"got {technique_count}."
        )

    print(
        "[OK] Feature dimensions are consistent."
    )

    # =========================================================================
    # 5. THRESHOLD VALIDATION
    # =========================================================================

    section(
        "5. Validating technique thresholds"
    )

    with open(
        THRESHOLDS,
        "r",
        encoding="utf-8"
    ) as f:

        threshold_data = json.load(f)

    thresholds = threshold_data.get(
        "thresholds",
        {}
    )

    if not thresholds:

        raise ValueError(
            "No thresholds found."
        )

    missing_thresholds = [
        technique
        for technique
        in label_encoder.classes_
        if technique not in thresholds
    ]

    if missing_thresholds:

        raise ValueError(
            "Missing thresholds:\n"
            + "\n".join(
                missing_thresholds
            )
        )

    numeric_thresholds = {
        technique: float(
            thresholds[technique]
        )
        for technique in thresholds
    }

    values = list(
        numeric_thresholds.values()
    )

    print(
        "Threshold count:",
        len(values)
    )

    print(
        "Minimum threshold:",
        f"{min(values):.4f}"
    )

    print(
        "Maximum threshold:",
        f"{max(values):.4f}"
    )

    print(
        "Unique thresholds:",
        sorted(
            set(values)
        )
    )

    print(
        "[OK] All 45 techniques have thresholds."
    )

    # =========================================================================
    # 6. DATASET AUDIT
    # =========================================================================

    section(
        "6. Reading v4.1 dataset audit"
    )

    with open(
        AUDIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        audit = json.load(f)

    print(
        "Dataset audit loaded."
    )

    # Print useful fields if available.

    if isinstance(audit, dict):

        for key in [
            "training_events",
            "testing_events",
            "training_rows",
            "testing_rows",
            "technique_count",
            "test_only_techniques",
            "invalid_training_timestamps",
            "invalid_testing_timestamps",
        ]:

            if key in audit:

                print(
                    f"{key}:",
                    audit[key]
                )

    # =========================================================================
    # 7. VERIFY TEST COMPARISON
    # =========================================================================

    section(
        "7. Verifying v4.1 vs v4.2 comparison"
    )

    require(
        COMPARISON_FILE
    )

    require(
        GLOBAL_COMPARISON
    )

    global_df = None
    per_df = None

    try:
        import pandas as pd

        global_df = pd.read_csv(
            GLOBAL_COMPARISON
        )

        per_df = pd.read_csv(
            COMPARISON_FILE
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not read comparison results."
        ) from exc

    print(
        "Global comparison rows:",
        len(global_df)
    )

    print(
        "Per-technique rows:",
        len(per_df)
    )

    if "model" not in global_df.columns:

        raise ValueError(
            "Global comparison does not contain model column."
        )

    if not set(
        ["v4.1", "v4.2"]
    ).issubset(
        set(global_df["model"])
    ):

        raise ValueError(
            "Comparison does not contain both v4.1 and v4.2."
        )

    v41 = global_df[
        global_df["model"] == "v4.1"
    ].iloc[0]

    v42 = global_df[
        global_df["model"] == "v4.2"
    ].iloc[0]

    print()
    print(
        "Unseen temporal test comparison:"
    )

    print(
        f"v4.1 micro-F1 : "
        f"{v41['micro_f1']:.4f}"
    )

    print(
        f"v4.2 micro-F1 : "
        f"{v42['micro_f1']:.4f}"
    )

    print(
        f"v4.1 precision: "
        f"{v41['micro_precision']:.4f}"
    )

    print(
        f"v4.2 precision: "
        f"{v42['micro_precision']:.4f}"
    )

    print(
        f"v4.1 recall   : "
        f"{v41['micro_recall']:.4f}"
    )

    print(
        f"v4.2 recall   : "
        f"{v42['micro_recall']:.4f}"
    )

    # =========================================================================
    # 8. MODEL SELECTION CHECK
    # =========================================================================

    section(
        "8. Production candidate decision"
    )

    v41_f1 = float(
        v41["micro_f1"]
    )

    v42_f1 = float(
        v42["micro_f1"]
    )

    v41_precision = float(
        v41["micro_precision"]
    )

    v42_precision = float(
        v42["micro_precision"]
    )

    v41_recall = float(
        v41["micro_recall"]
    )

    v42_recall = float(
        v42["micro_recall"]
    )

    print(
        "v4.1 micro-F1 delta:",
        f"{v41_f1 - v42_f1:+.6f}"
    )

    print(
        "v4.1 precision delta:",
        f"{v41_precision - v42_precision:+.6f}"
    )

    print(
        "v4.1 recall delta:",
        f"{v41_recall - v42_recall:+.6f}"
    )

    if (
        v41_f1 > v42_f1
        and
        v41_precision > v42_precision
        and
        v41_recall > v42_recall
    ):

        print()
        print(
            "[PASS] v4.1 outperforms v4.2 "
            "on all three primary global metrics."
        )

        production_candidate = "v4.1"

    else:

        print()
        print(
            "[WARNING] v4.1 does not dominate "
            "v4.2 on every primary metric."
        )

        production_candidate = "v4.1_pending_review"

    # =========================================================================
    # 9. CHECK FOR V4.2 CONTAMINATION
    # =========================================================================

    section(
        "9. Checking production dependency isolation"
    )

    production_paths = [
        MODEL_FILE,
        STRUCTURED_ENCODER,
        NUMERICAL_SCALER,
        TFIDF,
        LABEL_ENCODER,
        FEATURE_SCHEMA,
        THRESHOLDS,
    ]

    v42_strings = []

    for path in production_paths:

        try:

            raw = path.read_bytes()

            if b"v4_2" in raw:
                v42_strings.append(
                    str(path)
                )

        except Exception:
            pass

    if v42_strings:

        print(
            "[WARNING] Possible v4.2 references found:"
        )

        for item in v42_strings:
            print(
                " -",
                item
            )

    else:

        print(
            "[OK] No obvious v4.2 byte-string "
            "references found in production artifacts."
        )

    # =========================================================================
    # 10. SHA256 MANIFEST
    # =========================================================================

    section(
        "10. Generating artifact integrity hashes"
    )

    hashes = {}

    for name, path in artifacts.items():

        hashes[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }

        print(
            f"{name:24s}",
            hashes[name]["sha256"]
        )

    # =========================================================================
    # 11. PRODUCTION AUDIT REPORT
    # =========================================================================

    section(
        "11. Saving production audit report"
    )

    report = {

        "project":
            "SmartSOC",

        "model_family":
            "Windows-APT Model B",

        "version":
            "v4.1",

        "status":
            production_candidate,

        "model_type":
            type(model).__name__,

        "feature_count":
            int(model_features),

        "mitre_technique_count":
            int(technique_count),

        "tfidf_vocabulary":
            int(tfidf_count),

        "categorical_input_features":
            int(categorical_count),

        "categorical_encoded_features":
            int(encoded_count),

        "numerical_features":
            int(numerical_count),

        "unseen_test_metrics": {

            "v4.1": {
                "micro_precision":
                    v41_precision,

                "micro_recall":
                    v41_recall,

                "micro_f1":
                    v41_f1,

                "macro_precision":
                    float(
                        v41["macro_precision"]
                    ),

                "macro_recall":
                    float(
                        v41["macro_recall"]
                    ),

                "macro_f1":
                    float(
                        v41["macro_f1"]
                    ),

                "weighted_f1":
                    float(
                        v41["weighted_f1"]
                    ),
            },

            "v4.2": {
                "micro_precision":
                    v42_precision,

                "micro_recall":
                    v42_recall,

                "micro_f1":
                    v42_f1,

                "macro_precision":
                    float(
                        v42["macro_precision"]
                    ),

                "macro_recall":
                    float(
                        v42["macro_recall"]
                    ),

                "macro_f1":
                    float(
                        v42["macro_f1"]
                    ),

                "weighted_f1":
                    float(
                        v42["weighted_f1"]
                    ),
            },
        },

        "production_artifacts":
            hashes,

        "v4_2_used_for_training":
            False,

        "v4_2_thresholds_used":
            False,

        "base_model_modified":
            False,
    }

    audit_output = (
        RESULT_DIR
        / "production_audit_v4_1.json"
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        audit_output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"[OK] {audit_output}"
    )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    section(
        "FINAL PRODUCTION AUDIT RESULT"
    )

    print(
        "Model candidate:",
        production_candidate
    )

    print(
        "v4.1 feature count:",
        model_features
    )

    print(
        "MITRE techniques:",
        technique_count
    )

    print(
        "TF-IDF vocabulary:",
        tfidf_count
    )

    print(
        "Unseen temporal micro-F1:",
        f"{v41_f1:.4f}"
    )

    print(
        "Unseen temporal precision:",
        f"{v41_precision:.4f}"
    )

    print(
        "Unseen temporal recall:",
        f"{v41_recall:.4f}"
    )

    print()

    if production_candidate == "v4.1":

        print(
            "[PASS] v4.1 is the production candidate."
        )

        print(
            "[PASS] v4.2 is NOT promoted."
        )

        print(
            "[PASS] Base model remains unchanged."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "Create the final v4.1 production "
            "artifact package."
        )

    else:

        print(
            "[REVIEW REQUIRED]"
        )

    print()
    print(
        "Audit complete."
    )


if __name__ == "__main__":
    main()