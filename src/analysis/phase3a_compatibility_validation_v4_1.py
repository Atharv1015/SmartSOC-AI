import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase3a"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase3a_compatibility_validation_report.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("SMARTSOC PHASE 3A")
    print("COMPATIBILITY VALIDATION")
    print("=" * 72)

    print()
    print("RUNTIME")
    print("  Python       :", sys.version.split()[0])
    print("  scikit-learn:", sklearn.__version__)

    report = {
        "phase": "3A",
        "test": "compatibility_validation",
        "runtime": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
        },
        "production_directory": str(PRODUCTION_DIR),
        "checks": {},
        "artifacts": {},
        "model_properties": {},
        "prediction_test": {},
        "decision": None,
    }

    # ------------------------------------------------------------
    # Runtime validation
    # ------------------------------------------------------------

    expected_versions = {
        "python": "3.12.10",
        "scikit_learn": "1.8.0",
    }

    runtime_ok = (
        sys.version.split()[0]
        == expected_versions["python"]
        and sklearn.__version__
        == expected_versions["scikit_learn"]
    )

    if runtime_ok:

        report["checks"]["runtime"] = {
            "status": "PASS",
            "message": (
                "Compatibility runtime matches "
                "serialized production environment"
            ),
        }

        print("[PASS] Compatibility runtime")

    else:

        report["checks"]["runtime"] = {
            "status": "FAIL",
            "message": (
                "Compatibility runtime does not "
                "match expected versions"
            ),
            "expected": expected_versions,
        }

        print("[FAIL] Compatibility runtime")

        report["decision"] = "FAIL"

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

    # ------------------------------------------------------------
    # Production artifacts
    # ------------------------------------------------------------

    artifact_names = [
        "model.joblib",
        "structured_encoder.joblib",
        "numerical_scaler.joblib",
        "tfidf_vectorizer.joblib",
        "multilabel_encoder.joblib",
    ]

    artifacts = {}

    load_errors = {}

    for name in artifact_names:

        path = PRODUCTION_DIR / name

        try:

            artifacts[name] = joblib.load(path)

            report["artifacts"][name] = {
                "status": "LOADED",
                "size_bytes": path.stat().st_size,
            }

            print(
                f"[PASS] Loaded {name}"
            )

        except Exception as exc:

            load_errors[name] = repr(exc)

            report["artifacts"][name] = {
                "status": "FAILED",
                "error": repr(exc),
            }

            print(
                f"[FAIL] {name}"
            )

    if load_errors:

        report["checks"]["artifact_loading"] = {
            "status": "FAIL",
            "errors": load_errors,
        }

        report["decision"] = "FAIL"

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

    report["checks"]["artifact_loading"] = {
        "status": "PASS",
        "message": (
            "All production joblib artifacts "
            "loaded under sklearn 1.8.0"
        ),
    }

    # ------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------

    schema = load_json(
        PRODUCTION_DIR
        / "feature_schema.json"
    )

    thresholds = load_json(
        PRODUCTION_DIR
        / "thresholds.json"
    )

    categorical = schema[
        "categorical_columns"
    ]

    numerical = schema[
        "numerical_columns"
    ]

    techniques = schema[
        "mitre_techniques"
    ]

    report["model_properties"] = {
        "structured_features": (
            len(categorical)
            + len(numerical)
        ),
        "categorical_features": len(
            categorical
        ),
        "numerical_features": len(
            numerical
        ),
        "categorical_encoded": schema[
            "categorical_encoded_feature_count"
        ],
        "tfidf_features": schema[
            "tfidf_feature_count"
        ],
        "final_features": schema[
            "final_feature_count"
        ],
        "mitre_techniques": len(
            techniques
        ),
    }

    # ------------------------------------------------------------
    # Model dimension
    # ------------------------------------------------------------

    model = artifacts[
        "model.joblib"
    ]

    model_input = getattr(
        model,
        "n_features_in_",
        None,
    )

    if model_input is None:

        try:

            model_input = (
                model.estimators_[0]
                .n_features_in_
            )

        except Exception:

            model_input = None

    report["model_properties"][
        "model_input_dimension"
    ] = model_input

    dimension_ok = (
        model_input
        == schema["final_feature_count"]
    )

    if dimension_ok:

        report["checks"][
            "model_dimension"
        ] = {
            "status": "PASS",
            "message": (
                "Model input dimension matches "
                "production feature schema"
            ),
            "dimension": model_input,
        }

        print(
            "[PASS] Model input dimension:",
            model_input,
        )

    else:

        report["checks"][
            "model_dimension"
        ] = {
            "status": "FAIL",
            "expected": schema[
                "final_feature_count"
            ],
            "actual": model_input,
        }

        print(
            "[FAIL] Model input dimension"
        )

    # ------------------------------------------------------------
    # Transformer dimensions
    # ------------------------------------------------------------

    structured_encoder = artifacts[
        "structured_encoder.joblib"
    ]

    numerical_scaler = artifacts[
        "numerical_scaler.joblib"
    ]

    tfidf_vectorizer = artifacts[
        "tfidf_vectorizer.joblib"
    ]

    label_encoder = artifacts[
        "multilabel_encoder.joblib"
    ]

    cat_dim = sum(
        len(x)
        for x
        in structured_encoder.categories_
    )

    num_dim = (
        numerical_scaler.n_features_in_
    )

    tfidf_dim = len(
        tfidf_vectorizer.vocabulary_
    )

    label_dim = len(
        label_encoder.classes_
    )

    transformer_ok = (
        cat_dim == 162
        and num_dim == 58
        and tfidf_dim == 6699
        and label_dim == 45
    )

    report["checks"][
        "transformer_dimensions"
    ] = {
        "status": (
            "PASS"
            if transformer_ok
            else "FAIL"
        ),
        "categorical_encoded": cat_dim,
        "numerical": num_dim,
        "tfidf": tfidf_dim,
        "mitre": label_dim,
    }

    print(
        "["
        + ("PASS" if transformer_ok else "FAIL")
        + "] Transformer dimensions"
    )

    # ------------------------------------------------------------
    # Synthetic feature vector
    # ------------------------------------------------------------
    #
    # We construct the final feature vector directly
    # at the production model boundary.
    #
    # This is intentionally NOT used to evaluate
    # model quality. It only proves that the loaded
    # model can execute predict_proba under sklearn 1.8.0.
    # ------------------------------------------------------------

    final_dimension = schema[
        "final_feature_count"
    ]

    synthetic_vector = np.zeros(
        (1, final_dimension),
        dtype=np.float64,
    )

    # Put a tiny non-zero value in the first
    # feature to ensure the matrix is valid
    # numeric data.
    synthetic_vector[0, 0] = 0.001

    try:

        probabilities = (
            model.predict_proba(
                synthetic_vector
            )
        )

        # OneVsRest normally returns a matrix.
        probabilities = np.asarray(
            probabilities
        )

        report["prediction_test"] = {
            "status": "PASS",
            "shape": list(
                probabilities.shape
            ),
            "finite": bool(
                np.isfinite(
                    probabilities
                ).all()
            ),
            "min_probability": float(
                probabilities.min()
            ),
            "max_probability": float(
                probabilities.max()
            ),
        }

        print(
            "[PASS] predict_proba execution"
        )

        print(
            "  Probability shape:",
            probabilities.shape,
        )

        print(
            "  Finite:",
            np.isfinite(
                probabilities
            ).all(),
        )

    except Exception as exc:

        report["prediction_test"] = {
            "status": "FAIL",
            "error": repr(exc),
        }

        print(
            "[FAIL] predict_proba execution"
        )

        report["decision"] = "FAIL"

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

    # ------------------------------------------------------------
    # Threshold validation
    # ------------------------------------------------------------

    threshold_map = thresholds[
        "thresholds"
    ]

    threshold_ok = (
        len(threshold_map) == 45
        and set(threshold_map.keys())
        == set(techniques)
        and all(
            0.0 <= float(v) <= 1.0
            for v in threshold_map.values()
        )
    )

    report["checks"][
        "thresholds"
    ] = {
        "status": (
            "PASS"
            if threshold_ok
            else "FAIL"
        ),
        "count": len(
            threshold_map
        ),
    }

    print(
        "["
        + ("PASS" if threshold_ok else "FAIL")
        + "] Threshold configuration"
    )

    # ------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------

    failures = []

    for name, result in report[
        "checks"
    ].items():

        if result.get("status") == "FAIL":

            failures.append(name)

    report["failures"] = failures

    if failures:

        report["decision"] = "FAIL"

    else:

        report["decision"] = (
            "PASS"
        )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

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

    print()
    print("=" * 72)
    print("COMPATIBILITY VALIDATION SUMMARY")
    print("=" * 72)

    print(
        "Python       :",
        sys.version.split()[0],
    )

    print(
        "scikit-learn:",
        sklearn.__version__,
    )

    print(
        "Model input  :",
        model_input,
    )

    print(
        "Probabilities:",
        probabilities.shape,
    )

    print(
        "MITRE        :",
        label_dim,
    )

    print(
        "Thresholds   :",
        len(threshold_map),
    )

    print()
    print(
        "DECISION:",
        report["decision"],
    )

    print()
    print(
        "Report:",
        REPORT_FILE,
    )

    print()
    print(
        "NO PRODUCTION ARTIFACTS WERE MODIFIED."
    )

    return (
        0
        if report["decision"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())