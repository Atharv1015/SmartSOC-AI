from pathlib import Path
import json
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

THRESHOLD_FILE = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "recalibrated"
    / "technique_thresholds_v4_2_precision.json"
)

RESULT_FILE = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "precision_recalibration"
    / "precision_recalibration_v4_2.csv"
)


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


def require(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


def main():

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "- Precision Recalibration Audit"
    )

    # =========================================================
    # CHECK FILES
    # =========================================================

    section("Checking recalibration artifacts")

    require(
        THRESHOLD_FILE,
        "v4.2 precision threshold file"
    )

    require(
        RESULT_FILE,
        "v4.2 recalibration results"
    )

    # =========================================================
    # LOAD JSON
    # =========================================================

    section("Loading threshold artifact")

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(
        "Model:",
        data.get("model")
    )

    print(
        "Source model:",
        data.get("source_model")
    )

    print(
        "Calibration method:",
        data.get("calibration_method")
    )

    print(
        "Test set used:",
        data.get("test_set_used")
    )

    print(
        "Base model modified:",
        data.get("base_model_modified")
    )

    thresholds = data.get(
        "thresholds",
        {}
    )

    # =========================================================
    # BASIC VALIDATION
    # =========================================================

    section("Basic threshold validation")

    if not thresholds:

        raise ValueError(
            "No thresholds found."
        )

    values = [
        float(v)
        for v in thresholds.values()
    ]

    print(
        "Technique count:",
        len(thresholds)
    )

    print(
        "Minimum threshold:",
        min(values)
    )

    print(
        "Maximum threshold:",
        max(values)
    )

    print(
        "Unique thresholds:",
        sorted(
            set(values)
        )
    )

    invalid = {
        technique: threshold
        for technique, threshold
        in thresholds.items()
        if threshold < 0.0
        or threshold > 1.0
    }

    if invalid:

        print()
        print(
            "[ERROR] Invalid thresholds:"
        )

        for technique, threshold in invalid.items():

            print(
                f"{technique}: {threshold}"
            )

        raise ValueError(
            "Thresholds outside [0,1]."
        )

    print(
        "[OK] All thresholds are within [0,1]."
    )

    # =========================================================
    # LOAD CALIBRATION RESULTS
    # =========================================================

    section("Loading calibration results")

    df = pd.read_csv(
        RESULT_FILE
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )

    required_columns = [
        "technique",
        "support",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fp"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(missing)
        )

    # =========================================================
    # DUPLICATE TECHNIQUES
    # =========================================================

    section("Checking technique uniqueness")

    duplicates = df[
        df["technique"].duplicated(
            keep=False
        )
    ]

    if not duplicates.empty:

        print(
            "[ERROR] Duplicate techniques found:"
        )

        print(
            duplicates[
                ["technique", "threshold"]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            "Duplicate techniques."
        )

    print(
        "[OK] Each technique appears once."
    )

    # =========================================================
    # THRESHOLD CONSISTENCY
    # =========================================================

    section("Checking JSON/CSV threshold consistency")

    mismatches = []

    for _, row in df.iterrows():

        technique = str(
            row["technique"]
        )

        csv_threshold = float(
            row["threshold"]
        )

        json_threshold = float(
            thresholds.get(
                technique,
                -1
            )
        )

        if abs(
            csv_threshold
            - json_threshold
        ) > 1e-6:

            mismatches.append(
                (
                    technique,
                    csv_threshold,
                    json_threshold
                )
            )

    if mismatches:

        print(
            "[ERROR] Threshold mismatches:"
        )

        for item in mismatches:

            print(
                item
            )

        raise ValueError(
            "JSON and CSV thresholds differ."
        )

    print(
        "[OK] JSON and CSV thresholds match."
    )

    # =========================================================
    # PRECISION AUDIT
    # =========================================================

    section("Precision audit")

    supported = df[
        df["support"] > 0
    ].copy()

    unsupported = df[
        df["support"] == 0
    ].copy()

    print(
        "Supported techniques:",
        len(supported)
    )

    print(
        "Zero-support techniques:",
        len(unsupported)
    )

    # =========================================================
    # PRECISION TARGET
    # =========================================================

    precision_failures = supported[
        supported["precision"] < 0.90
    ]

    print(
        "Supported techniques below 90% precision:",
        len(precision_failures)
    )

    if not precision_failures.empty:

        print()

        print(
            precision_failures[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp"
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "[OK] All supported techniques "
            "achieved >= 90% validation precision."
        )

    # =========================================================
    # HIGH FALSE-POSITIVE TECHNIQUES
    # =========================================================

    section("False-positive audit")

    high_fp = supported[
        supported["fp"] > 50
    ].sort_values(
        "fp",
        ascending=False
    )

    if high_fp.empty:

        print(
            "[OK] No technique has more than "
            "50 validation false positives."
        )

    else:

        print(
            "Techniques with >50 false positives:"
        )

        print(
            high_fp[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # VERY LOW PRECISION
    # =========================================================

    section("Critical precision failures")

    critical = supported[
        supported["precision"] < 0.50
    ]

    if critical.empty:

        print(
            "[OK] No supported technique "
            "has precision below 50%."
        )

    else:

        print(
            "[WARNING] Critical precision failures:"
        )

        print(
            critical[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # RECALL AUDIT
    # =========================================================

    section("Recall audit")

    low_recall = supported[
        supported["recall"] < 0.50
    ]

    print(
        "Techniques below 50% recall:",
        len(low_recall)
    )

    if not low_recall.empty:

        print(
            low_recall[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # F1 AUDIT
    # =========================================================

    section("F1 audit")

    low_f1 = supported[
        supported["f1"] < 0.50
    ]

    print(
        "Techniques below 0.50 F1:",
        len(low_f1)
    )

    if not low_f1.empty:

        print(
            low_f1[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # VERY LOW SUPPORT
    # =========================================================

    section("Low-support audit")

    low_support = supported[
        supported["support"] < 10
    ].sort_values(
        "support"
    )

    print(
        "Techniques with support < 10:",
        len(low_support)
    )

    if not low_support.empty:

        print(
            low_support[
                [
                    "technique",
                    "support",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "fp"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # ZERO SUPPORT
    # =========================================================

    section("Zero-support techniques")

    if unsupported.empty:

        print(
            "[OK] Every technique has validation support."
        )

    else:

        print(
            "These techniques cannot be "
            "validated on this split:"
        )

        print(
            unsupported[
                [
                    "technique",
                    "support",
                    "threshold"
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================
    # THRESHOLD DISTRIBUTION
    # =========================================================

    section("Threshold distribution")

    distribution = (
        df[
            "threshold"
        ]
        .round(4)
        .value_counts()
        .sort_index()
    )

    for threshold, count in distribution.items():

        print(
            f"{threshold:.4f}: {count}"
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    header(
        "v4.2 Precision Recalibration Audit Summary"
    )

    print(
        f"Total techniques              : {len(df)}"
    )

    print(
        f"Validation-supported          : {len(supported)}"
    )

    print(
        f"Validation-zero-support      : {len(unsupported)}"
    )

    print(
        f"Precision < 90%              : {len(precision_failures)}"
    )

    print(
        f"Precision < 50%              : {len(critical)}"
    )

    print(
        f"Recall < 50%                 : {len(low_recall)}"
    )

    print(
        f"F1 < 0.50                    : {len(low_f1)}"
    )

    print(
        f"Support < 10                 : {len(low_support)}"
    )

    print(
        f"FP > 50                      : {len(high_fp)}"
    )

    # =========================================================
    # DECISION
    # =========================================================

    section("Audit decision")

    if (
        not invalid
        and not mismatches
        and len(critical) == 0
    ):

        print(
            "[PASS] v4.2 threshold artifact "
            "passes structural and critical precision checks."
        )

        if (
            len(precision_failures) == 0
        ):

            print(
                "[PASS] All supported techniques "
                "meet the 90% validation precision target."
            )

        else:

            print(
                "[WARNING] Some techniques "
                "remain below the 90% precision target."
            )

        if (
            len(high_fp) > 0
        ):

            print(
                "[WARNING] False-positive hotspots remain."
            )

        print()
        print(
            "The artifact is eligible for "
            "controlled inference testing."
        )

    else:

        print(
            "[FAIL] v4.2 should NOT be promoted yet."
        )

        print(
            "Keep v4.1 as the active baseline."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This audit does not evaluate the final "
        "unseen test set."
    )

    print(
        "Do not claim v4.2 is better than v4.1 "
        "until both are compared on the same "
        "unseen temporal test set."
    )


if __name__ == "__main__":
    main()