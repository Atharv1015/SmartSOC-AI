from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = (
    BASE
    / "results"
    / "windows_apt_v4_2"
    / "technique_calibration_v4_2.csv"
)

THRESHOLD_FILE = (
    BASE
    / "models"
    / "windows_apt_v4_2"
    / "calibrated"
    / "technique_thresholds_v4_2.json"
)

OUTPUT_DIR = (
    BASE
    / "results"
    / "windows_apt_v4_2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONSTANTS
# ============================================================

TEST_ROWS = 11707

MIN_ACCEPTABLE_PRECISION = 0.80
MIN_ACCEPTABLE_F1 = 0.50

MAX_FALSE_POSITIVE_RATE = 0.02
MAX_RARE_CLASS_FP = 20

RARE_SUPPORT = 20
VERY_RARE_SUPPORT = 5

MIN_DEPLOYMENT_THRESHOLD = 0.10
MAX_DEPLOYMENT_THRESHOLD = 0.95


# ============================================================
# DISPLAY HELPERS
# ============================================================

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


def fail(message):
    print()
    print("[ERROR]")
    print(message)
    sys.exit(1)


# ============================================================
# FILE CHECK
# ============================================================

def check_file(path, description):

    if not path.exists():

        fail(
            f"{description} not found:\n"
            f"{path}"
        )

    print(
        f"[OK] {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "- Calibration Audit"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    section(
        "Checking required artifacts"
    )

    check_file(
        CALIBRATION_FILE,
        "Calibration CSV"
    )

    check_file(
        THRESHOLD_FILE,
        "Threshold JSON"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    section(
        "Loading calibration results"
    )

    try:

        calibration = pd.read_csv(
            CALIBRATION_FILE
        )

    except Exception as exc:

        fail(
            f"Could not read calibration CSV:\n{exc}"
        )

    print(
        f"Calibration rows: "
        f"{len(calibration)}"
    )

    print(
        f"Columns: "
        f"{list(calibration.columns)}"
    )

    required_columns = [
        "technique",
        "support",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fp",
        "predicted"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in calibration.columns
    ]

    if missing_columns:

        fail(
            "Missing required columns:\n"
            + "\n".join(
                f"- {x}"
                for x in missing_columns
            )
        )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    section(
        "Loading threshold JSON"
    )

    try:

        with open(
            THRESHOLD_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            threshold_data = json.load(f)

    except Exception as exc:

        fail(
            f"Could not read threshold JSON:\n{exc}"
        )

    if not isinstance(
        threshold_data,
        dict
    ):

        fail(
            "Threshold JSON is not a dictionary."
        )

    print(
        f"JSON keys: "
        f"{list(threshold_data.keys())}"
    )

    if "thresholds" not in threshold_data:

        fail(
            "Threshold JSON does not contain "
            "'thresholds'."
        )

    thresholds = threshold_data[
        "thresholds"
    ]

    if not isinstance(
        thresholds,
        dict
    ):

        fail(
            "'thresholds' must be a dictionary."
        )

    print(
        f"Threshold count: "
        f"{len(thresholds)}"
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    section(
        "Normalizing calibration data"
    )

    numeric_columns = [
        "support",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fp",
        "predicted"
    ]

    for column in numeric_columns:

        calibration[column] = pd.to_numeric(
            calibration[column],
            errors="coerce"
        )

    if calibration[
        numeric_columns
    ].isna().any().any():

        print(
            "[WARNING] Missing/non-numeric values "
            "were detected."
        )

    calibration["technique"] = (
        calibration["technique"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    section(
        "Validating threshold mapping"
    )

    calibration_techniques = set(
        calibration["technique"]
    )

    json_techniques = set(
        thresholds.keys()
    )

    missing_in_json = (
        calibration_techniques
        - json_techniques
    )

    missing_in_csv = (
        json_techniques
        - calibration_techniques
    )

    if missing_in_json:

        print(
            "[WARNING] Techniques in CSV "
            "but not JSON:"
        )

        for technique in sorted(
            missing_in_json
        ):

            print(
                f"  - {technique}"
            )

    if missing_in_csv:

        print(
            "[WARNING] Techniques in JSON "
            "but not CSV:"
        )

        for technique in sorted(
            missing_in_csv
        ):

            print(
                f"  - {technique}"
            )

    if not missing_in_json and not missing_in_csv:

        print(
            "[OK] CSV and JSON technique sets match."
        )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    section(
        "Auditing individual techniques"
    )

    audit_rows = []

    for _, row in calibration.iterrows():

        technique = str(
            row["technique"]
        )

        support = int(
            row["support"]
        )

        threshold = float(
            row["threshold"]
        )

        precision = float(
            row["precision"]
        )

        recall = float(
            row["recall"]
        )

        f1 = float(
            row["f1"]
        )

        fp = int(
            row["fp"]
        )

        predicted = int(
            row["predicted"]
        )

        issues = []

        severity = "PASS"

        # ----------------------------------------------------
        # ZERO SUPPORT
        # ----------------------------------------------------

        if support == 0:

            issues.append(
                "NO_TEST_SUPPORT"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # LOW SUPPORT
        # ----------------------------------------------------

        if (
            support > 0
            and support <= VERY_RARE_SUPPORT
        ):

            issues.append(
                "VERY_LOW_SUPPORT"
            )

            if severity == "PASS":
                severity = "REVIEW"

        elif (
            support > 0
            and support <= RARE_SUPPORT
        ):

            issues.append(
                "LOW_SUPPORT"
            )

            if severity == "PASS":
                severity = "REVIEW"

        # ----------------------------------------------------
        # PRECISION
        # ----------------------------------------------------

        if (
            support > 0
            and precision < MIN_ACCEPTABLE_PRECISION
        ):

            issues.append(
                "LOW_PRECISION"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        if (
            support > 0
            and f1 < MIN_ACCEPTABLE_F1
        ):

            issues.append(
                "LOW_F1"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # FALSE POSITIVES
        # ----------------------------------------------------

        negative_count = max(
            1,
            TEST_ROWS - support
        )

        false_positive_rate = (
            fp / negative_count
        )

        if (
            support > 0
            and support <= RARE_SUPPORT
            and fp > MAX_RARE_CLASS_FP
        ):

            issues.append(
                "EXCESSIVE_FP_FOR_RARE_CLASS"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # FPR
        # ----------------------------------------------------

        if (
            support > 0
            and false_positive_rate
            > MAX_FALSE_POSITIVE_RATE
        ):

            issues.append(
                "HIGH_FALSE_POSITIVE_RATE"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # LOW THRESHOLD
        # ----------------------------------------------------

        if threshold < MIN_DEPLOYMENT_THRESHOLD:

            issues.append(
                "VERY_LOW_THRESHOLD"
            )

            if severity == "PASS":
                severity = "REVIEW"

        # ----------------------------------------------------
        # INVALID THRESHOLD
        # ----------------------------------------------------

        if (
            threshold < 0.0
            or threshold > 1.0
        ):

            issues.append(
                "INVALID_THRESHOLD"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # SUSPICIOUS PERFECT RESULT
        # ----------------------------------------------------

        if (
            support >= 20
            and precision >= 0.999
            and recall >= 0.999
            and fp == 0
        ):

            issues.append(
                "PERFECT_RESULT_REQUIRES_REVIEW"
            )

            if severity == "PASS":
                severity = "REVIEW"

        # ----------------------------------------------------
        # THRESHOLD CONSISTENCY
        # ----------------------------------------------------

        json_threshold = thresholds.get(
            technique
        )

        threshold_difference = None

        if json_threshold is not None:

            try:

                json_threshold = float(
                    json_threshold
                )

                threshold_difference = abs(
                    threshold
                    - json_threshold
                )

                if threshold_difference > 1e-6:

                    issues.append(
                        "CSV_JSON_THRESHOLD_MISMATCH"
                    )

                    severity = "HOLD"

            except Exception:

                issues.append(
                    "INVALID_JSON_THRESHOLD"
                )

                severity = "HOLD"

        else:

            issues.append(
                "MISSING_JSON_THRESHOLD"
            )

            severity = "HOLD"

        # ----------------------------------------------------
        # ISSUE STRING
        # ----------------------------------------------------

        if issues:

            issue_text = ";".join(
                issues
            )

        else:

            issue_text = "NONE"

        audit_rows.append(
            {
                "technique": technique,
                "support": support,
                "threshold": threshold,
                "json_threshold":
                    json_threshold,
                "threshold_difference":
                    threshold_difference,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp": fp,
                "predicted": predicted,
                "false_positive_rate":
                    false_positive_rate,
                "severity": severity,
                "issues": issue_text
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    header(
        "Complete Calibration Audit"
    )

    print(
        audit[
            [
                "technique",
                "support",
                "threshold",
                "precision",
                "recall",
                "f1",
                "fp",
                "false_positive_rate",
                "severity",
                "issues"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    header(
        "Audit Summary"
    )

    pass_count = int(
        (
            audit["severity"]
            == "PASS"
        ).sum()
    )

    review_count = int(
        (
            audit["severity"]
            == "REVIEW"
        ).sum()
    )

    hold_count = int(
        (
            audit["severity"]
            == "HOLD"
        ).sum()
    )

    print(
        f"Total techniques : {len(audit)}"
    )

    print(
        f"PASS              : {pass_count}"
    )

    print(
        f"REVIEW            : {review_count}"
    )

    print(
        f"HOLD              : {hold_count}"
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    section(
        "Techniques ON HOLD"
    )

    hold = audit[
        audit["severity"] == "HOLD"
    ]

    if hold.empty:

        print(
            "None."
        )

    else:

        for _, row in hold.iterrows():

            print(
                f"{row['technique']:<12} "
                f"support={int(row['support']):<5} "
                f"threshold={row['threshold']:.4f} "
                f"precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} "
                f"F1={row['f1']:.4f} "
                f"FP={int(row['fp']):<5} "
                f"FPR={row['false_positive_rate']:.4f}"
            )

            print(
                f"             "
                f"issues={row['issues']}"
            )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    section(
        "Techniques REQUIRING REVIEW"
    )

    review = audit[
        audit["severity"] == "REVIEW"
    ]

    if review.empty:

        print(
            "None."
        )

    else:

        for _, row in review.iterrows():

            print(
                f"{row['technique']:<12} "
                f"support={int(row['support']):<5} "
                f"threshold={row['threshold']:.4f} "
                f"precision={row['precision']:.4f} "
                f"recall={row['recall']:.4f} "
                f"F1={row['f1']:.4f}"
            )

            print(
                f"             "
                f"issues={row['issues']}"
            )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    section(
        "Zero-Support Techniques"
    )

    zero_support = audit[
        audit["support"] == 0
    ]

    if zero_support.empty:

        print(
            "None."
        )

    else:

        for technique in zero_support[
            "technique"
        ]:

            print(
                f"- {technique}"
            )

    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    section(
        "Highest False-Positive Techniques"
    )

    high_fp = (
        audit
        .sort_values(
            "fp",
            ascending=False
        )
        .head(15)
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
                "fp",
                "false_positive_rate"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # STEP 13
    # --------------------------------------------------------

    section(
        "Very Low Thresholds"
    )

    low_thresholds = audit[
        audit["threshold"]
        < MIN_DEPLOYMENT_THRESHOLD
    ]

    if low_thresholds.empty:

        print(
            "None."
        )

    else:

        print(
            low_thresholds[
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

    # --------------------------------------------------------
    # STEP 14
    # --------------------------------------------------------

    section(
        "Perfect / Suspicious Results"
    )

    suspicious = audit[
        audit["issues"].str.contains(
            "PERFECT_RESULT_REQUIRES_REVIEW",
            na=False
        )
    ]

    if suspicious.empty:

        print(
            "None."
        )

    else:

        print(
            suspicious[
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

    # --------------------------------------------------------
    # STEP 15
    # --------------------------------------------------------

    section(
        "Deployment Decision"
    )

    if hold_count > 0:

        decision = "DO_NOT_DEPLOY"

        print(
            "[HOLD] Calibration is NOT "
            "safe for deployment."
        )

        print()
        print(
            "Reason:"
        )

        print(
            f"{hold_count} technique(s) "
            "have unacceptable calibration behavior."
        )

    elif review_count > 0:

        decision = (
            "REVIEW_BEFORE_DEPLOYMENT"
        )

        print(
            "[REVIEW] Calibration requires "
            "additional review."
        )

    else:

        decision = (
            "CALIBRATION_AUDIT_PASSED"
        )

        print(
            "[PASS] Calibration audit passed."
        )

    # --------------------------------------------------------
    # STEP 16
    # --------------------------------------------------------

    section(
        "Saving Audit Files"
    )

    audit_csv = (
        OUTPUT_DIR
        / "calibration_audit_v4_2.csv"
    )

    audit.to_csv(
        audit_csv,
        index=False
    )

    print(
        f"[OK] {audit_csv}"
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = {
        "model_version":
            "windows_apt_v4_2",

        "test_rows":
            TEST_ROWS,

        "total_techniques":
            int(len(audit)),

        "pass":
            pass_count,

        "review":
            review_count,

        "hold":
            hold_count,

        "zero_support":
            int(len(zero_support)),

        "decision":
            decision,

        "rules": {
            "minimum_precision":
                MIN_ACCEPTABLE_PRECISION,

            "minimum_f1":
                MIN_ACCEPTABLE_F1,

            "maximum_false_positive_rate":
                MAX_FALSE_POSITIVE_RATE,

            "maximum_rare_class_fp":
                MAX_RARE_CLASS_FP,

            "minimum_deployment_threshold":
                MIN_DEPLOYMENT_THRESHOLD,

            "maximum_deployment_threshold":
                MAX_DEPLOYMENT_THRESHOLD
        },

        "hold_techniques":
            hold[
                "technique"
            ].tolist(),

        "review_techniques":
            review[
                "technique"
            ].tolist(),

        "zero_support_techniques":
            zero_support[
                "technique"
            ].tolist()
    }

    audit_json = (
        OUTPUT_DIR
        / "calibration_audit_v4_2.json"
    )

    with open(
        audit_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print(
        f"[OK] {audit_json}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    header(
        "SmartSOC Windows-APT Model B v4.2 "
        "- Calibration Audit Complete"
    )

    print(
        f"Decision: {decision}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Base model was NOT modified."
    )

    print(
        "Threshold JSON was NOT modified."
    )

    print(
        "No inference artifact was created."
    )

    print()

    if decision == "DO_NOT_DEPLOY":

        print(
            "Next step:"
        )

        print(
            "Fix the rare-class/high-false-positive "
            "calibration problem before deployment."
        )

    elif decision == "REVIEW_BEFORE_DEPLOYMENT":

        print(
            "Next step:"
        )

        print(
            "Review the flagged techniques before "
            "creating the inference artifact."
        )

    else:

        print(
            "Next step:"
        )

        print(
            "Proceed to final inference artifact generation."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()