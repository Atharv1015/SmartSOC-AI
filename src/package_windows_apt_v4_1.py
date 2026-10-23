from pathlib import Path
import json
import hashlib
import shutil
import sys
import platform
import subprocess
from datetime import datetime


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# FINAL PRODUCTION PACKAGE BUILDER
# =============================================================================

BASE = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_1"
    / "calibrated"
)

PRODUCTION_DIR = (
    BASE
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

AUDIT_FILE = (
    BASE
    / "results"
    / "windows_apt_v4_1"
    / "production_audit_v4_1.json"
)

PRODUCTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# ARTIFACT MAP
# =============================================================================

ARTIFACTS = {

    "model.joblib":
        SOURCE_DIR
        / "multilabel_logistic_v4_1_calibrated.joblib",

    "structured_encoder.joblib":
        SOURCE_DIR
        / "structured_encoder_v4_1_calibrated.joblib",

    "numerical_scaler.joblib":
        SOURCE_DIR
        / "numerical_scaler_v4_1_calibrated.joblib",

    "tfidf_vectorizer.joblib":
        SOURCE_DIR
        / "tfidf_vectorizer_v4_1_calibrated.joblib",

    "multilabel_encoder.joblib":
        SOURCE_DIR
        / "multilabel_encoder_v4_1_calibrated.joblib",

    "feature_schema.json":
        SOURCE_DIR
        / "feature_schema_v4_1_calibrated.json",

    "thresholds.json":
        SOURCE_DIR
        / "technique_thresholds_v4_1.json",
}


# =============================================================================
# HELPERS
# =============================================================================

def header(text):
    print()
    print("=" * 78)
    print(text)
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


def get_package_version(package):

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "show",
                package
            ],
            capture_output=True,
            text=True,
            check=False
        )

        for line in result.stdout.splitlines():

            if line.startswith("Version:"):

                return line.split(
                    ":", 1
                )[1].strip()

    except Exception:
        pass

    return "unknown"


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1"
    )

    print(
        "FINAL PRODUCTION ARTIFACT PACKAGING"
    )

    # =========================================================================
    # 1. CHECK SOURCE ARTIFACTS
    # =========================================================================

    header(
        "1. Checking validated source artifacts"
    )

    for name, path in ARTIFACTS.items():

        require(path)

    require(AUDIT_FILE)

    # =========================================================================
    # 2. LOAD AUDIT
    # =========================================================================

    header(
        "2. Loading production audit"
    )

    with open(
        AUDIT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        audit = json.load(f)

    if audit.get("version") != "v4.1":

        raise ValueError(
            "Production audit is not for v4.1."
        )

    if audit.get("status") != "v4.1":

        raise ValueError(
            "v4.1 is not marked as production candidate."
        )

    print(
        "[OK] Production audit confirms v4.1."
    )

    # =========================================================================
    # 3. CLEAN PREVIOUS PACKAGE CONTENT
    # =========================================================================

    header(
        "3. Preparing production directory"
    )

    for path in PRODUCTION_DIR.iterdir():

        if path.is_file() or path.is_symlink():

            path.unlink()

        elif path.is_dir():

            shutil.rmtree(path)

    print(
        "[OK] Production directory prepared."
    )

    # =========================================================================
    # 4. COPY ARTIFACTS
    # =========================================================================

    header(
        "4. Copying production artifacts"
    )

    manifest_artifacts = {}

    for destination_name, source_path in ARTIFACTS.items():

        destination = (
            PRODUCTION_DIR
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination
        )

        digest = sha256(
            destination
        )

        size = destination.stat().st_size

        manifest_artifacts[
            destination_name
        ] = {

            "source":
                str(source_path),

            "sha256":
                digest,

            "size_bytes":
                size,
        }

        print(
            f"[OK] {destination_name}"
        )

    # =========================================================================
    # 5. COPY AUDIT REPORT
    # =========================================================================

    shutil.copy2(
        AUDIT_FILE,
        PRODUCTION_DIR
        / "production_audit_v4_1.json"
    )

    print(
        "[OK] production_audit_v4_1.json"
    )

    # =========================================================================
    # 6. ENVIRONMENT INFORMATION
    # =========================================================================

    header(
        "5. Recording runtime environment"
    )

    python_version = platform.python_version()

    sklearn_version = get_package_version(
        "scikit-learn"
    )

    numpy_version = get_package_version(
        "numpy"
    )

    pandas_version = get_package_version(
        "pandas"
    )

    scipy_version = get_package_version(
        "scipy"
    )

    joblib_version = get_package_version(
        "joblib"
    )

    environment = {

        "python":
            python_version,

        "platform":
            platform.platform(),

        "scikit_learn":
            sklearn_version,

        "numpy":
            numpy_version,

        "pandas":
            pandas_version,

        "scipy":
            scipy_version,

        "joblib":
            joblib_version,
    }

    print(
        "Python:",
        python_version
    )

    print(
        "scikit-learn:",
        sklearn_version
    )

    print(
        "numpy:",
        numpy_version
    )

    print(
        "pandas:",
        pandas_version
    )

    print(
        "scipy:",
        scipy_version
    )

    print(
        "joblib:",
        joblib_version
    )

    with open(
        PRODUCTION_DIR
        / "environment.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            environment,
            f,
            indent=2
        )

    # =========================================================================
    # 7. CREATE MODEL MANIFEST
    # =========================================================================

    header(
        "6. Creating model manifest"
    )

    manifest = {

        "project":
            "SmartSOC",

        "model_name":
            "SmartSOC Windows-APT Model B",

        "version":
            "v4.1",

        "status":
            "production_candidate",

        "created_at":
            datetime.now().isoformat(),

        "model_type":
            "OneVsRestClassifier",

        "feature_count":
            6919,

        "categorical_input_features":
            4,

        "categorical_encoded_features":
            162,

        "numerical_features":
            58,

        "tfidf_features":
            6699,

        "mitre_techniques":
            45,

        "source_dataset":
            "windows_apt_v4_1",

        "base_model_modified":
            False,

        "v4_2_promoted":
            False,

        "v4_2_thresholds_used":
            False,

        "evaluation": {

            "unseen_temporal_test":
                True,

            "micro_precision":
                0.980262,

            "micro_recall":
                0.702975,

            "micro_f1":
                0.818779,

            "macro_f1":
                0.380766,

            "weighted_f1":
                0.711380,
        },

        "artifacts":
            manifest_artifacts,

        "environment":
            environment,

        "notes": [

            "v4.1 selected over v4.2 after unseen temporal comparison.",

            "v4.2 threshold recalibration was not promoted.",

            "Base v4.1 model was not modified during v4.2 experiments.",

            "Artifacts should be loaded with the recorded scikit-learn environment.",

            "Any production environment change should trigger a compatibility test."
        ],
    }

    manifest_file = (
        PRODUCTION_DIR
        / "model_manifest.json"
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        f"[OK] {manifest_file}"
    )

    # =========================================================================
    # 8. CREATE REQUIREMENTS
    # =========================================================================

    requirements_file = (
        PRODUCTION_DIR
        / "requirements_production.txt"
    )

    requirements = [
        f"scikit-learn=={sklearn_version}",
        f"numpy=={numpy_version}",
        f"pandas=={pandas_version}",
        f"scipy=={scipy_version}",
        f"joblib=={joblib_version}",
    ]

    with open(
        requirements_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(requirements)
        )

        f.write("\n")

    print(
        f"[OK] {requirements_file}"
    )

    # =========================================================================
    # 9. FINAL HASH MANIFEST
    # =========================================================================

    header(
        "7. Final artifact integrity verification"
    )

    final_hashes = {}

    for path in sorted(
        PRODUCTION_DIR.iterdir()
    ):

        if path.is_file():

            final_hashes[
                path.name
            ] = {

                "sha256":
                    sha256(path),

                "size_bytes":
                    path.stat().st_size,
            }

            print(
                f"{path.name:40s}"
                f"{final_hashes[path.name]['sha256']}"
            )

    with open(
        PRODUCTION_DIR
        / "checksums.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_hashes,
            f,
            indent=2
        )

    # =========================================================================
    # 10. FINAL DIRECTORY LISTING
    # =========================================================================

    header(
        "8. Production package contents"
    )

    for path in sorted(
        PRODUCTION_DIR.iterdir()
    ):

        if path.is_file():

            print(
                "[OK]",
                path.name
            )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    header(
        "SmartSOC Windows-APT Model B v4.1"
    )

    print(
        "FINAL PRODUCTION PACKAGE CREATED"
    )

    print()
    print(
        "Production directory:"
    )

    print(
        PRODUCTION_DIR
    )

    print()
    print(
        "Model version : v4.1"
    )

    print(
        "Feature count : 6919"
    )

    print(
        "MITRE classes : 45"
    )

    print(
        "TF-IDF vocab  : 6699"
    )

    print(
        "Micro-F1      : 0.8188"
    )

    print(
        "Precision     : 0.9803"
    )

    print(
        "Recall        : 0.7030"
    )

    print()

    print(
        "[PASS] v4.1 packaged."
    )

    print(
        "[PASS] v4.2 not promoted."
    )

    print(
        "[PASS] Environment recorded."
    )

    print(
        "[PASS] SHA-256 checksums recorded."
    )

    print()
    print(
        "Next stage:"
    )

    print(
        "Integrate the production inference pipeline "
        "with SmartSOC/Wazuh."
    )


if __name__ == "__main__":
    main()