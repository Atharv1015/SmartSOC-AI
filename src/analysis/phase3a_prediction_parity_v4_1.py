from pathlib import Path
import json
import sys
import subprocess
import tempfile
import numpy as np
import joblib
import sklearn


ROOT = Path(__file__).resolve().parents[2]

PROD = ROOT / "models" / "windows_apt_v4_1" / "production"
DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"
REPORT_DIR = ROOT / "reports" / "phase3a"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "phase3a_prediction_parity_report.json"

COMPAT_PYTHON = ROOT / ".venv-compat" / "Scripts" / "python.exe"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_test_vector():
    """
    Build a deterministic production-shaped vector.

    This is intentionally independent of the model artifacts.
    The parity test is checking whether the same serialized model
    produces equivalent probabilities in the two runtimes.
    """

    schema = load_json(PROD / "feature_schema.json")

    total_features = int(schema["final_feature_count"])

    rng = np.random.default_rng(202503)

    x = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(1, total_features)
    ).astype(np.float64)

    return x


def run_compat_prediction(input_path, output_path):
    if not COMPAT_PYTHON.exists():
        raise FileNotFoundError(
            f"Compatibility Python not found: {COMPAT_PYTHON}"
        )

    helper = r'''
import sys
import json
import numpy as np
import joblib
from pathlib import Path

prod = Path(sys.argv[1])
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

model = joblib.load(prod / "model.joblib")

x = np.load(input_path)

proba = model.predict_proba(x)

if isinstance(proba, list):
    proba = np.column_stack([
        p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
        for p in proba
    ])

proba = np.asarray(proba, dtype=np.float64)

np.save(output_path, proba)

print("COMPAT_PROBABILITY_SHAPE:", proba.shape)
print("COMPAT_FINITE:", bool(np.isfinite(proba).all()))
'''

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(helper)
        helper_path = f.name

    try:
        result = subprocess.run(
            [
                str(COMPAT_PYTHON),
                helper_path,
                str(PROD),
                str(input_path),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Compatibility prediction failed.\n"
                + result.stdout
                + "\n"
                + result.stderr
            )

        return result.stdout.strip()

    finally:
        try:
            Path(helper_path).unlink()
        except Exception:
            pass


def main():

    print("=" * 72)
    print("SMARTSOC PHASE 3A")
    print("PREDICTION PARITY VALIDATION")
    print("=" * 72)

    report = {
        "phase": "3A",
        "test": "prediction_parity",
        "production_artifacts_modified": False,
        "status": "FAIL",
    }

    # ------------------------------------------------------------------
    # Current runtime
    # ------------------------------------------------------------------

    print("\nCURRENT RUNTIME")
    print("  Python       :", sys.version.split()[0])
    print("  scikit-learn:", sklearn.__version__)
    print("  joblib       :", joblib.__version__)

    # ------------------------------------------------------------------
    # Load production model
    # ------------------------------------------------------------------

    print("\nPRODUCTION MODEL")

    model_path = PROD / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)

    print("  Model type   :", type(model).__name__)
    print("[PASS] Current-runtime model loading")

    # ------------------------------------------------------------------
    # Validate expected input dimension
    # ------------------------------------------------------------------

    schema = load_json(PROD / "feature_schema.json")

    expected_features = int(schema["final_feature_count"])

    if expected_features != 6919:
        raise RuntimeError(
            f"Unexpected production feature count: {expected_features}"
        )

    print("[PASS] Expected feature count:", expected_features)

    # ------------------------------------------------------------------
    # Build deterministic vector
    # ------------------------------------------------------------------

    x = build_test_vector()

    print("[PASS] Deterministic parity vector:", x.shape)

    # ------------------------------------------------------------------
    # Current runtime prediction
    # ------------------------------------------------------------------

    current_proba = model.predict_proba(x)

    if isinstance(current_proba, list):
        current_proba = np.column_stack([
            p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
            for p in current_proba
        ])

    current_proba = np.asarray(current_proba, dtype=np.float64)

    print("[PASS] Current-runtime predict_proba")
    print("  Probability shape:", current_proba.shape)
    print("  Finite           :", bool(np.isfinite(current_proba).all()))

    if current_proba.shape != (1, 45):
        raise RuntimeError(
            f"Unexpected current probability shape: {current_proba.shape}"
        )

    # ------------------------------------------------------------------
    # Save deterministic vector
    # ------------------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        input_path = tmp / "parity_input.npy"
        compat_output = tmp / "compat_probability.npy"

        np.save(input_path, x)

        print("\nCOMPATIBILITY RUNTIME")

        compat_stdout = run_compat_prediction(
            input_path,
            compat_output
        )

        print(compat_stdout)

        compat_proba = np.load(compat_output)

        print("[PASS] Compatibility-runtime prediction")
        print("  Probability shape:", compat_proba.shape)
        print("  Finite           :", bool(np.isfinite(compat_proba).all()))

        # --------------------------------------------------------------
        # Compare
        # --------------------------------------------------------------

        if current_proba.shape != compat_proba.shape:
            raise RuntimeError(
                "Probability shape mismatch between runtimes."
            )

        absolute_diff = np.abs(
            current_proba - compat_proba
        )

        max_abs_diff = float(np.max(absolute_diff))
        mean_abs_diff = float(np.mean(absolute_diff))

        exact_match = bool(
            np.array_equal(current_proba, compat_proba)
        )

        close_match = bool(
            np.allclose(
                current_proba,
                compat_proba,
                rtol=1e-10,
                atol=1e-12
            )
        )

        print("\nPARITY COMPARISON")
        print("  Exact match       :", exact_match)
        print("  Allclose          :", close_match)
        print("  Max abs difference:", f"{max_abs_diff:.12g}")
        print("  Mean abs diff     :", f"{mean_abs_diff:.12g}")

        # --------------------------------------------------------------
        # Top predictions
        # --------------------------------------------------------------

        labels = schema["mitre_techniques"]

        current_order = np.argsort(
            current_proba[0]
        )[::-1]

        compat_order = np.argsort(
            compat_proba[0]
        )[::-1]

        current_top5 = [
            {
                "technique": labels[int(i)],
                "probability": float(current_proba[0, i]),
            }
            for i in current_order[:5]
        ]

        compat_top5 = [
            {
                "technique": labels[int(i)],
                "probability": float(compat_proba[0, i]),
            }
            for i in compat_order[:5]
        ]

        top1_same = bool(
            current_order[0] == compat_order[0]
        )

        print("\nTOP-5 PARITY")

        for rank in range(5):
            c = current_top5[rank]
            k = compat_top5[rank]

            print(
                f"  #{rank + 1}: "
                f"{c['technique']} "
                f"{c['probability']:.12f}"
                f"  |  "
                f"{k['technique']} "
                f"{k['probability']:.12f}"
            )

        print("\n  Top-1 same:", top1_same)

        # --------------------------------------------------------------
        # Final decision
        # --------------------------------------------------------------

        parity_pass = (
            close_match
            and top1_same
            and bool(np.isfinite(current_proba).all())
            and bool(np.isfinite(compat_proba).all())
        )

        if parity_pass:
            status = "PASS"
            print("\nDECISION: PASS")
        else:
            status = "REVIEW_REQUIRED"
            print("\nDECISION: REVIEW_REQUIRED")

        report.update({
            "current_runtime": {
                "python": sys.version.split()[0],
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
            "compatibility_runtime": {
                "python": "3.12.10",
                "scikit_learn": "1.8.0",
            },
            "model_type": type(model).__name__,
            "expected_feature_count": expected_features,
            "current_probability_shape": list(current_proba.shape),
            "compatibility_probability_shape": list(compat_proba.shape),
            "exact_match": exact_match,
            "allclose": close_match,
            "max_absolute_difference": max_abs_diff,
            "mean_absolute_difference": mean_abs_diff,
            "top1_same": top1_same,
            "current_top5": current_top5,
            "compatibility_top5": compat_top5,
            "status": status,
        })

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print("\n" + "=" * 72)
    print("PHASE 3A PARITY SUMMARY")
    print("=" * 72)

    print("Current runtime probability :", report["current_probability_shape"])
    print("Compat runtime probability  :", report["compatibility_probability_shape"])
    print(
        "Max absolute difference     :",
        f"{report['max_absolute_difference']:.12g}"
    )
    print(
        "Mean absolute difference    :",
        f"{report['mean_absolute_difference']:.12g}"
    )
    print("Top-1 same                  :", report["top1_same"])
    print("Allclose                    :", report["allclose"])
    print("\nDECISION:", report["status"])

    print("\nReport:", REPORT_PATH)
    print("\nNO PRODUCTION ARTIFACTS WERE MODIFIED.")


if __name__ == "__main__":
    main()