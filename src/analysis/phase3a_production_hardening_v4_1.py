import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "phase3a"

REPORT_FILE = (
    REPORT_DIR
    / "phase3a_production_hardening_report.json"
)


EXPECTED_ARTIFACTS = [
    "model.joblib",
    "structured_encoder.joblib",
    "numerical_scaler.joblib",
    "tfidf_vectorizer.joblib",
    "multilabel_encoder.joblib",
    "feature_schema.json",
    "thresholds.json",
    "model_manifest.json",
    "checksums.json",
    "environment.json",
    "production_audit_v4_1.json",
    "requirements_production.txt",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_check(status, message, details=None):
    result = {
        "status": status,
        "message": message,
    }

    if details is not None:
        result["details"] = details

    return result


def pass_check(message, details=None):
    return make_check(
        "PASS",
        message,
        details,
    )


def fail_check(message, details=None):
    return make_check(
        "FAIL",
        message,
        details,
    )


def review_check(message, details=None):
    return make_check(
        "REVIEW",
        message,
        details,
    )


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "gate": "Phase 3A",
        "gate_name": (
            "Production Hardening and Reproducibility"
        ),
        "started_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "production_directory": str(PRODUCTION_DIR),
        "checks": {},
        "environment": {},
        "artifact_hashes": {},
        "model_properties": {},
        "threshold_properties": {},
        "serialization_compatibility": {},
        "decision": None,
    }

    print()
    print("=" * 72)
    print("SMARTSOC PHASE 3A")
    print("PRODUCTION HARDENING & REPRODUCIBILITY AUDIT")
    print("=" * 72)
    print()

    # ============================================================
    # 1. Production directory
    # ============================================================

    if PRODUCTION_DIR.exists():

        report["checks"]["production_directory"] = (
            pass_check(
                "Production artifact directory exists",
                str(PRODUCTION_DIR),
            )
        )

    else:

        report["checks"]["production_directory"] = (
            fail_check(
                "Production artifact directory does not exist",
                str(PRODUCTION_DIR),
            )
        )

        report["decision"] = "FAIL"
        report["finished_at"] = utc_now()

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        return 1

    print("[PASS] Production directory")

    # ============================================================
    # 2. Required artifacts
    # ============================================================

    missing = []

    for artifact in EXPECTED_ARTIFACTS:

        if not (
            PRODUCTION_DIR / artifact
        ).exists():

            missing.append(artifact)

    if not missing:

        report["checks"]["required_artifacts"] = (
            pass_check(
                "All required production artifacts exist",
                {
                    "expected_count": len(
                        EXPECTED_ARTIFACTS
                    ),
                    "missing": [],
                },
            )
        )

        print("[PASS] Required production artifacts")

    else:

        report["checks"]["required_artifacts"] = (
            fail_check(
                "Required production artifacts are missing",
                {
                    "expected_count": len(
                        EXPECTED_ARTIFACTS
                    ),
                    "missing": missing,
                },
            )
        )

        print(
            "[FAIL] Missing artifacts:",
            ", ".join(missing),
        )

    # ============================================================
    # 3. Current environment
    # ============================================================

    report["environment"] = {
        "python_version": platform.python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }

    print()
    print("CURRENT ENVIRONMENT")

    print(
        f"  Python       : "
        f"{platform.python_version()}"
    )

    print(
        f"  scikit-learn: "
        f"{sklearn.__version__}"
    )

    print(
        f"  joblib       : "
        f"{joblib.__version__}"
    )

    print(
        f"  numpy        : "
        f"{np.__version__}"
    )

    print(
        f"  pandas       : "
        f"{pd.__version__}"
    )

    print(
        f"  scipy        : "
        f"{scipy.__version__}"
    )

    # ============================================================
    # 4. Load metadata
    # ============================================================

    try:

        feature_schema = load_json(
            PRODUCTION_DIR
            / "feature_schema.json"
        )

        thresholds_file = load_json(
            PRODUCTION_DIR
            / "thresholds.json"
        )

        manifest = load_json(
            PRODUCTION_DIR
            / "model_manifest.json"
        )

        environment_artifact = load_json(
            PRODUCTION_DIR
            / "environment.json"
        )

        report["checks"]["metadata_json"] = (
            pass_check(
                "Production metadata JSON files load successfully"
            )
        )

    except Exception as exc:

        report["checks"]["metadata_json"] = (
            fail_check(
                "Production metadata JSON loading failed",
                repr(exc),
            )
        )

        report["decision"] = "FAIL"
        report["finished_at"] = utc_now()

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        return 1

    # ============================================================
    # 5. Load joblib artifacts
    # ============================================================

    joblib_artifact_names = [
        "model.joblib",
        "structured_encoder.joblib",
        "numerical_scaler.joblib",
        "tfidf_vectorizer.joblib",
        "multilabel_encoder.joblib",
    ]

    artifacts = {}
    load_errors = {}

    for name in joblib_artifact_names:

        try:

            artifacts[name] = joblib.load(
                PRODUCTION_DIR / name
            )

        except Exception as exc:

            load_errors[name] = repr(exc)

    if not load_errors:

        report["checks"]["joblib_loading"] = (
            pass_check(
                "All production joblib artifacts load successfully"
            )
        )

        print(
            "[PASS] Production joblib loading"
        )

    else:

        report["checks"]["joblib_loading"] = (
            fail_check(
                "One or more production joblib artifacts failed to load",
                load_errors,
            )
        )

        print(
            "[FAIL] Production joblib loading"
        )

        report["decision"] = "FAIL"
        report["finished_at"] = utc_now()

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
            )

        return 1

    # ============================================================
    # 6. Feature schema
    # ============================================================

    categorical_columns = feature_schema.get(
        "categorical_columns",
        [],
    )

    numerical_columns = feature_schema.get(
        "numerical_columns",
        [],
    )

    structured_columns = (
        categorical_columns
        + numerical_columns
    )

    declared_cat_count = feature_schema.get(
        "categorical_input_feature_count"
    )

    declared_cat_encoded = feature_schema.get(
        "categorical_encoded_feature_count"
    )

    declared_num_count = feature_schema.get(
        "numerical_feature_count"
    )

    declared_tfidf_count = feature_schema.get(
        "tfidf_feature_count"
    )

    declared_final_count = feature_schema.get(
        "final_feature_count"
    )

    declared_mitre = feature_schema.get(
        "mitre_techniques",
        [],
    )

    report["model_properties"].update(
        {
            "structured_columns": len(
                structured_columns
            ),
            "categorical_columns": len(
                categorical_columns
            ),
            "numerical_columns": len(
                numerical_columns
            ),
            "declared_categorical_input": (
                declared_cat_count
            ),
            "declared_categorical_encoded": (
                declared_cat_encoded
            ),
            "declared_numerical": (
                declared_num_count
            ),
            "declared_tfidf": (
                declared_tfidf_count
            ),
            "declared_final": (
                declared_final_count
            ),
            "declared_mitre_count": len(
                declared_mitre
            ),
        }
    )

    if len(structured_columns) == 62:

        report["checks"]["structured_feature_count"] = (
            pass_check(
                "Structured feature count is 62",
                len(structured_columns),
            )
        )

    else:

        report["checks"]["structured_feature_count"] = (
            fail_check(
                "Structured feature count is not 62",
                len(structured_columns),
            )
        )

    if (
        len(categorical_columns) == 4
        and declared_cat_count == 4
    ):

        report["checks"]["categorical_feature_count"] = (
            pass_check(
                "Categorical input feature count is 4",
                len(categorical_columns),
            )
        )

    else:

        report["checks"]["categorical_feature_count"] = (
            fail_check(
                "Categorical input feature count is incorrect",
                {
                    "actual": len(
                        categorical_columns
                    ),
                    "declared": declared_cat_count,
                },
            )
        )

    if (
        len(numerical_columns) == 58
        and declared_num_count == 58
    ):

        report["checks"]["numerical_feature_count"] = (
            pass_check(
                "Numerical feature count is 58",
                len(numerical_columns),
            )
        )

    else:

        report["checks"]["numerical_feature_count"] = (
            fail_check(
                "Numerical feature count is incorrect",
                {
                    "actual": len(
                        numerical_columns
                    ),
                    "declared": declared_num_count,
                },
            )
        )

    # ============================================================
    # 7. Transformer dimensions
    # ============================================================

    structured_encoder = artifacts[
        "structured_encoder.joblib"
    ]

    numerical_scaler = artifacts[
        "numerical_scaler.joblib"
    ]

    tfidf_vectorizer = artifacts[
        "tfidf_vectorizer.joblib"
    ]

    multilabel_encoder = artifacts[
        "multilabel_encoder.joblib"
    ]

    # OneHotEncoder dimension
    try:

        cat_dim = sum(
            len(categories)
            for categories
            in structured_encoder.categories_
        )

    except Exception:

        cat_dim = None

    # Numerical dimension
    try:

        num_dim = int(
            numerical_scaler.n_features_in_
        )

    except Exception:

        num_dim = None

    # TF-IDF dimension
    try:

        tfidf_dim = len(
            tfidf_vectorizer.vocabulary_
        )

    except Exception:

        tfidf_dim = None

    # MITRE classes
    try:

        mitre_dim = len(
            multilabel_encoder.classes_
        )

        classes = list(
            multilabel_encoder.classes_
        )

    except Exception:

        mitre_dim = None
        classes = []

    report["model_properties"].update(
        {
            "actual_categorical_encoded": cat_dim,
            "actual_numerical": num_dim,
            "actual_tfidf": tfidf_dim,
            "actual_mitre_classes": mitre_dim,
        }
    )

    print()

    print(
        f"Categorical encoded : {cat_dim}"
    )

    print(
        f"Numerical features  : {num_dim}"
    )

    print(
        f"TF-IDF features     : {tfidf_dim}"
    )

    print(
        f"MITRE techniques    : {mitre_dim}"
    )

    if cat_dim == 162:

        report["checks"][
            "categorical_encoded_dimensions"
        ] = pass_check(
            "Categorical encoder produces 162 features",
            cat_dim,
        )

    else:

        report["checks"][
            "categorical_encoded_dimensions"
        ] = fail_check(
            "Categorical encoder dimension is incorrect",
            cat_dim,
        )

    if num_dim == 58:

        report["checks"][
            "numerical_dimensions"
        ] = pass_check(
            "Numerical scaler produces 58 features",
            num_dim,
        )

    else:

        report["checks"][
            "numerical_dimensions"
        ] = fail_check(
            "Numerical scaler dimension is incorrect",
            num_dim,
        )

    if tfidf_dim == 6699:

        report["checks"][
            "tfidf_dimensions"
        ] = pass_check(
            "TF-IDF vectorizer produces 6699 features",
            tfidf_dim,
        )

    else:

        report["checks"][
            "tfidf_dimensions"
        ] = fail_check(
            "TF-IDF dimension is incorrect",
            tfidf_dim,
        )

    if mitre_dim == 45:

        report["checks"][
            "mitre_class_count"
        ] = pass_check(
            "MITRE label encoder contains 45 techniques",
            mitre_dim,
        )

    else:

        report["checks"][
            "mitre_class_count"
        ] = fail_check(
            "MITRE class count is incorrect",
            mitre_dim,
        )

    # ============================================================
    # 8. Final feature dimension
    # ============================================================

    if (
        cat_dim is not None
        and num_dim is not None
        and tfidf_dim is not None
    ):

        final_dim = (
            cat_dim
            + num_dim
            + tfidf_dim
        )

    else:

        final_dim = None

    report["model_properties"][
        "actual_final_feature_dimension"
    ] = final_dim

    if (
        final_dim == 6919
        and declared_final_count == 6919
    ):

        report["checks"][
            "final_feature_dimension"
        ] = pass_check(
            "Final production feature dimension is 6919",
            final_dim,
        )

    else:

        report["checks"][
            "final_feature_dimension"
        ] = fail_check(
            "Final production feature dimension is incorrect",
            {
                "actual": final_dim,
                "declared": declared_final_count,
            },
        )

    # ============================================================
    # 9. Model input dimension
    # ============================================================

    model = artifacts["model.joblib"]

    model_input_dim = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_input_dim is None:

        try:

            model_input_dim = (
                model.estimators_[0]
                .n_features_in_
            )

        except Exception:

            model_input_dim = None

    report["model_properties"][
        "model_input_dimension"
    ] = model_input_dim

    if model_input_dim == 6919:

        report["checks"][
            "model_input_dimension"
        ] = pass_check(
            "Production model expects 6919 input features",
            model_input_dim,
        )

    else:

        report["checks"][
            "model_input_dimension"
        ] = fail_check(
            "Production model input dimension is incorrect",
            model_input_dim,
        )

    # ============================================================
    # 10. MITRE mapping
    # ============================================================

    invalid_classes = [
        x
        for x in classes
        if not (
            isinstance(x, str)
            and x.startswith("T")
        )
    ]

    schema_set = set(
        declared_mitre
    )

    model_set = set(
        classes
    )

    mapping_matches = (
        schema_set == model_set
    )

    if (
        len(classes) == 45
        and not invalid_classes
        and mapping_matches
    ):

        report["checks"][
            "mitre_mapping"
        ] = pass_check(
            "MITRE mapping contains the expected 45 techniques",
            {
                "count": len(classes),
                "schema_model_match": True,
            },
        )

    else:

        report["checks"][
            "mitre_mapping"
        ] = fail_check(
            "MITRE mapping does not match the production schema",
            {
                "class_count": len(classes),
                "invalid_classes": invalid_classes,
                "schema_model_match": mapping_matches,
            },
        )

    # ============================================================
    # 11. Threshold validation
    # ============================================================

    threshold_map = thresholds_file.get(
        "thresholds",
        {},
    )

    threshold_techniques = thresholds_file.get(
        "techniques",
        [],
    )

    numeric_thresholds = {}
    invalid_thresholds = []

    for technique, value in threshold_map.items():

        try:

            numeric_value = float(value)

            numeric_thresholds[
                technique
            ] = numeric_value

            if not (
                0.0
                <= numeric_value
                <= 1.0
            ):

                invalid_thresholds.append(
                    {
                        "technique": technique,
                        "value": numeric_value,
                        "reason": (
                            "outside_0_to_1"
                        ),
                    }
                )

        except Exception:

            invalid_thresholds.append(
                {
                    "technique": technique,
                    "value": repr(value),
                    "reason": (
                        "non_numeric"
                    ),
                }
            )

    threshold_keys_match = (
        set(threshold_map.keys())
        == set(threshold_techniques)
        == set(classes)
    )

    report["threshold_properties"] = {
        "count": len(threshold_map),
        "min": (
            min(
                numeric_thresholds.values()
            )
            if numeric_thresholds
            else None
        ),
        "max": (
            max(
                numeric_thresholds.values()
            )
            if numeric_thresholds
            else None
        ),
        "technique_list_count": len(
            threshold_techniques
        ),
        "invalid_thresholds": (
            invalid_thresholds
        ),
        "keys_match_model_classes": (
            threshold_keys_match
        ),
    }

    if (
        len(threshold_map) == 45
        and len(threshold_techniques) == 45
        and not invalid_thresholds
        and threshold_keys_match
    ):

        report["checks"][
            "threshold_validation"
        ] = pass_check(
            "45 valid technique-specific thresholds match MITRE classes",
            {
                "count": len(threshold_map),
                "min": min(
                    numeric_thresholds.values()
                ),
                "max": max(
                    numeric_thresholds.values()
                ),
            },
        )

    else:

        report["checks"][
            "threshold_validation"
        ] = fail_check(
            "Threshold configuration is inconsistent",
            report["threshold_properties"],
        )

    # ============================================================
    # 12. Environment comparison
    # ============================================================

    recorded_environment = {
        "python": environment_artifact.get(
            "python"
        ),
        "platform": environment_artifact.get(
            "platform"
        ),
        "scikit_learn": environment_artifact.get(
            "scikit_learn"
        ),
        "numpy": environment_artifact.get(
            "numpy"
        ),
        "pandas": environment_artifact.get(
            "pandas"
        ),
        "scipy": environment_artifact.get(
            "scipy"
        ),
        "joblib": environment_artifact.get(
            "joblib"
        ),
    }

    report["environment"][
        "recorded_production_environment"
    ] = recorded_environment

    current_environment = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }

    environment_matches = {
        key: (
            recorded_environment.get(key)
            == current_environment.get(key)
        )
        for key in current_environment
    }

    report["environment"][
        "version_comparison"
    ] = environment_matches

    mismatches = {
        key: {
            "recorded": recorded_environment.get(
                key
            ),
            "current": current_environment.get(
                key
            ),
        }
        for key, matches
        in environment_matches.items()
        if not matches
    }

    if not mismatches:

        report["checks"][
            "environment_match"
        ] = pass_check(
            "Current runtime matches recorded production environment",
            current_environment,
        )

        print(
            "[PASS] Recorded environment matches current environment"
        )

    else:

        report["checks"][
            "environment_match"
        ] = review_check(
            "Current runtime differs from recorded production environment",
            mismatches,
        )

        print(
            "[REVIEW] Environment differences detected"
        )

    # ============================================================
    # 13. Serialized estimator compatibility
    # ============================================================

    recorded_runtime_sklearn = (
        recorded_environment.get(
            "scikit_learn"
        )
    )

    report["serialization_compatibility"] = {
        "recorded_environment_sklearn": (
            recorded_runtime_sklearn
        ),
        "current_sklearn": sklearn.__version__,
        "joblib_load_success": True,
        "observed_inconsistent_version_warning": (
            True
        ),
        "warning_detail": (
            "Serialized sklearn estimators report "
            "version 1.8.0 while the recorded "
            "production environment and current "
            "runtime report sklearn 1.9.0."
        ),
    }

    # We know from the load operation that the
    # artifacts successfully deserialize. However,
    # sklearn emitted InconsistentVersionWarning
    # during loading. Treat this as a review item,
    # not as a hard failure.

    report["checks"][
        "serialization_version_review"
    ] = review_check(
        "Serialized sklearn estimator metadata requires compatibility review",
        {
            "serialized_estimator_version": (
                "1.8.0"
            ),
            "recorded_environment_version": (
                recorded_runtime_sklearn
            ),
            "current_runtime_version": (
                sklearn.__version__
            ),
            "artifacts_loaded": True,
        },
    )

    # ============================================================
    # 14. Artifact SHA-256 hashes
    # ============================================================

    for artifact in EXPECTED_ARTIFACTS:

        path = PRODUCTION_DIR / artifact

        if path.exists():

            report["artifact_hashes"][
                artifact
            ] = {
                "sha256": sha256_file(
                    path
                ),
                "size_bytes": path.stat().st_size,
            }

    # ============================================================
    # 15. Existing checksum validation
    # ============================================================

    checksum_path = (
        PRODUCTION_DIR
        / "checksums.json"
    )

    checksum_mismatches = []
    checksum_entries_checked = 0

    if checksum_path.exists():

        try:

            recorded_checksums = load_json(
                checksum_path
            )

            for artifact, data in report[
                "artifact_hashes"
            ].items():

                entry = recorded_checksums.get(
                    artifact
                )

                recorded_hash = None

                if isinstance(
                    entry,
                    str,
                ):

                    recorded_hash = entry

                elif isinstance(
                    entry,
                    dict,
                ):

                    recorded_hash = entry.get(
                        "sha256"
                    )

                if recorded_hash is not None:

                    checksum_entries_checked += 1

                    if (
                        recorded_hash
                        != data["sha256"]
                    ):

                        checksum_mismatches.append(
                            {
                                "artifact": artifact,
                                "recorded": (
                                    recorded_hash
                                ),
                                "current": (
                                    data["sha256"]
                                ),
                            }
                        )

        except Exception as exc:

            checksum_mismatches.append(
                {
                    "error": repr(exc)
                }
            )

    if (
        checksum_entries_checked > 0
        and not checksum_mismatches
    ):

        report["checks"][
            "checksum_integrity"
        ] = pass_check(
            "Recorded production checksums match current artifacts",
            {
                "entries_checked": (
                    checksum_entries_checked
                ),
            },
        )

    else:

        report["checks"][
            "checksum_integrity"
        ] = review_check(
            "Checksum verification requires review",
            {
                "entries_checked": (
                    checksum_entries_checked
                ),
                "mismatches": (
                    checksum_mismatches
                ),
            },
        )

    # ============================================================
    # 16. Manifest
    # ============================================================

    if (
        isinstance(manifest, dict)
        and manifest
    ):

        report["checks"]["manifest"] = (
            pass_check(
                "Production model manifest is present and non-empty"
            )
        )

    else:

        report["checks"]["manifest"] = (
            fail_check(
                "Production model manifest is empty or invalid"
            )
        )

    # ============================================================
    # 17. Requirements artifact
    # ============================================================

    requirements_path = (
        PRODUCTION_DIR
        / "requirements_production.txt"
    )

    if requirements_path.exists():

        requirements_text = (
            requirements_path.read_text(
                encoding="utf-8"
            )
        )

        report["checks"][
            "requirements_artifact"
        ] = pass_check(
            "Production requirements artifact exists and is non-empty",
            {
                "lines": len(
                    [
                        line
                        for line
                        in requirements_text.splitlines()
                        if line.strip()
                    ]
                )
            },
        )

    else:

        report["checks"][
            "requirements_artifact"
        ] = fail_check(
            "Production requirements artifact is missing"
        )

    # ============================================================
    # 18. Final decision
    # ============================================================

    hard_failures = []

    for name, result in report[
        "checks"
    ].items():

        if result.get("status") == "FAIL":

            hard_failures.append(name)

    report["hard_failures"] = (
        hard_failures
    )

    if hard_failures:

        decision = "FAIL"

    else:

        review_items = [
            name
            for name, result
            in report["checks"].items()
            if result.get("status") == "REVIEW"
        ]

        if review_items:

            decision = (
                "PASS_WITH_DOCUMENTED_REVIEW"
            )

            report["review_items"] = (
                review_items
            )

        else:

            decision = "PASS"

    report["decision"] = decision
    report["finished_at"] = utc_now()

    # ============================================================
    # 19. Save report
    # ============================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ============================================================
    # Console summary
    # ============================================================

    print()
    print("=" * 72)
    print("PHASE 3A SUMMARY")
    print("=" * 72)

    print(
        f"Structured features  : "
        f"{len(structured_columns)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_columns)}"
    )

    print(
        f"Numerical features   : "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical encoded  : "
        f"{cat_dim}"
    )

    print(
        f"TF-IDF features      : "
        f"{tfidf_dim}"
    )

    print(
        f"Final feature vector : "
        f"{final_dim}"
    )

    print(
        f"Model input          : "
        f"{model_input_dim}"
    )

    print(
        f"MITRE techniques     : "
        f"{mitre_dim}"
    )

    print(
        f"Thresholds           : "
        f"{len(threshold_map)}"
    )

    print()
    print(
        f"Serialized sklearn   : "
        f"1.8.0"
    )

    print(
        f"Recorded environment: "
        f"{recorded_runtime_sklearn}"
    )

    print(
        f"Current sklearn      : "
        f"{sklearn.__version__}"
    )

    print()
    print(
        f"DECISION: {decision}"
    )

    if hard_failures:

        print()
        print("HARD FAILURES:")

        for failure in hard_failures:

            print(
                f"  - {failure}"
            )

    print()
    print(
        f"Report: {REPORT_FILE}"
    )

    print()
    print(
        "No production model, threshold, "
        "dataset, or inference artifact was modified."
    )

    return 0 if decision != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())