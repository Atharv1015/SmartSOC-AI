from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

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
    "audit": DATA_DIR / "dataset_audit.json",
}


# ============================================================
# HELPERS
# ============================================================

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


def split_labels(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# LOAD
# ============================================================

def main():

    header(
        "SmartSOC Windows-APT Model B v4.1 - Dataset Audit"
    )

    section("Checking required artifacts")

    for path in FILES.values():
        require(path)

    section("Loading datasets")

    structured_train = pd.read_csv(
        FILES["structured_train"]
    )

    structured_test = pd.read_csv(
        FILES["structured_test"]
    )

    text_train = pd.read_csv(
        FILES["text_train"]
    )

    text_test = pd.read_csv(
        FILES["text_test"]
    )

    labels_train = pd.read_csv(
        FILES["labels_train"]
    )

    labels_test = pd.read_csv(
        FILES["labels_test"]
    )

    metadata_train = pd.read_csv(
        FILES["metadata_train"]
    )

    metadata_test = pd.read_csv(
        FILES["metadata_test"]
    )

    with open(
        FILES["schema"],
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    print(
        "Structured train:",
        structured_train.shape
    )

    print(
        "Structured test :",
        structured_test.shape
    )

    print(
        "Text train      :",
        text_train.shape
    )

    print(
        "Text test       :",
        text_test.shape
    )

    print(
        "Labels train    :",
        labels_train.shape
    )

    print(
        "Labels test     :",
        labels_test.shape
    )

    print(
        "Metadata train  :",
        metadata_train.shape
    )

    print(
        "Metadata test   :",
        metadata_test.shape
    )

    # ========================================================
    # ALIGNMENT
    # ========================================================

    header("1. ROW ALIGNMENT AUDIT")

    train_lengths = {
        "structured": len(structured_train),
        "text": len(text_train),
        "labels": len(labels_train),
        "metadata": len(metadata_train),
    }

    test_lengths = {
        "structured": len(structured_test),
        "text": len(text_test),
        "labels": len(labels_test),
        "metadata": len(metadata_test),
    }

    print("Training:")
    for k, v in train_lengths.items():
        print(f"  {k:<12}: {v}")

    print()
    print("Testing:")
    for k, v in test_lengths.items():
        print(f"  {k:<12}: {v}")

    if len(set(train_lengths.values())) != 1:
        raise ValueError(
            "Training datasets are not aligned."
        )

    if len(set(test_lengths.values())) != 1:
        raise ValueError(
            "Testing datasets are not aligned."
        )

    print()
    print("[PASS] Training rows aligned.")
    print("[PASS] Testing rows aligned.")

    # ========================================================
    # LABEL COLUMN
    # ========================================================

    header("2. LABEL STRUCTURE AUDIT")

    print(
        "Training label columns:",
        labels_train.columns.tolist()
    )

    print(
        "Testing label columns :",
        labels_test.columns.tolist()
    )

    if "techniques" not in labels_train.columns:
        raise ValueError(
            "Training labels do not contain 'techniques'."
        )

    if "techniques" not in labels_test.columns:
        raise ValueError(
            "Testing labels do not contain 'techniques'."
        )

    train_label_lists = (
        labels_train["techniques"]
        .apply(split_labels)
    )

    test_label_lists = (
        labels_test["techniques"]
        .apply(split_labels)
    )

    # ========================================================
    # TECHNIQUE DISTRIBUTION
    # ========================================================

    header("3. MITRE TECHNIQUE DISTRIBUTION")

    train_counts = (
        train_label_lists
        .explode()
        .value_counts()
    )

    test_counts = (
        test_label_lists
        .explode()
        .value_counts()
    )

    all_techniques = sorted(
        set(train_counts.index)
        | set(test_counts.index)
    )

    print(
        "Total MITRE techniques:",
        len(all_techniques)
    )

    print()
    print("Training distribution:")
    print(
        train_counts.to_string()
    )

    print()
    print("Testing distribution:")
    print(
        test_counts.to_string()
    )

    # ========================================================
    # RARE CLASS AUDIT
    # ========================================================

    header("4. RARE TECHNIQUE AUDIT")

    rare_1 = train_counts[
        train_counts <= 1
    ]

    rare_5 = train_counts[
        train_counts <= 5
    ]

    rare_10 = train_counts[
        train_counts <= 10
    ]

    rare_20 = train_counts[
        train_counts <= 20
    ]

    print(
        "Techniques with <= 1 training event :",
        len(rare_1)
    )

    print(
        "Techniques with <= 5 training events:",
        len(rare_5)
    )

    print(
        "Techniques with <=10 training events:",
        len(rare_10)
    )

    print(
        "Techniques with <=20 training events:",
        len(rare_20)
    )

    if len(rare_20) > 0:
        print()
        print(
            "Techniques with <=20 training events:"
        )

        print(
            rare_20.to_string()
        )

    # ========================================================
    # TEST-ONLY TECHNIQUES
    # ========================================================

    header("5. TEST-ONLY TECHNIQUE AUDIT")

    train_set = set(train_counts.index)
    test_set = set(test_counts.index)

    test_only = sorted(
        test_set - train_set
    )

    train_only = sorted(
        train_set - test_set
    )

    print(
        "Test-only techniques:",
        len(test_only)
    )

    if test_only:
        print(
            "\n".join(
                f"  {x}"
                for x in test_only
            )
        )
    else:
        print("  None")

    print()
    print(
        "Training-only techniques:",
        len(train_only)
    )

    if train_only:
        print(
            "\n".join(
                f"  {x}"
                for x in train_only
            )
        )

    # ========================================================
    # MULTI-LABEL AUDIT
    # ========================================================

    header("6. MULTI-LABEL AUDIT")

    train_multi = (
        train_label_lists
        .apply(lambda x: len(x) > 1)
    )

    test_multi = (
        test_label_lists
        .apply(lambda x: len(x) > 1)
    )

    print(
        "Training single-label:",
        int((~train_multi).sum())
    )

    print(
        "Training multi-label :",
        int(train_multi.sum())
    )

    print()

    print(
        "Testing single-label:",
        int((~test_multi).sum())
    )

    print(
        "Testing multi-label :",
        int(test_multi.sum())
    )

    # ========================================================
    # EMPTY TEXT
    # ========================================================

    header("7. BEHAVIOR TEXT AUDIT")

    text_train_col = text_train.columns[0]
    text_test_col = text_test.columns[0]

    empty_train = (
        text_train[text_train_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_test = (
        text_test[text_test_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    print(
        "Empty training behavior:",
        int(empty_train.sum()),
        "/",
        len(text_train)
    )

    print(
        "Empty testing behavior :",
        int(empty_test.sum()),
        "/",
        len(text_test)
    )

    print(
        "Training empty ratio:",
        round(
            empty_train.mean(),
            4
        )
    )

    print(
        "Testing empty ratio :",
        round(
            empty_test.mean(),
            4
        )
    )

    # ========================================================
    # STRUCTURED FEATURE AUDIT
    # ========================================================

    header("8. STRUCTURED FEATURE AUDIT")

    train_columns = list(
        structured_train.columns
    )

    test_columns = list(
        structured_test.columns
    )

    if train_columns != test_columns:
        print(
            "[FAIL] Train/test feature schemas differ."
        )

        print(
            "Train-only:",
            sorted(
                set(train_columns)
                - set(test_columns)
            )
        )

        print(
            "Test-only:",
            sorted(
                set(test_columns)
                - set(train_columns)
            )
        )

        raise ValueError(
            "Structured train/test schema mismatch."
        )

    print(
        "[PASS] Structured schemas match."
    )

    print(
        "Feature count:",
        len(train_columns)
    )

    categorical = [
        x
        for x in [
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
        ]
        if x in train_columns
    ]

    numerical = [
        x
        for x in train_columns
        if x not in categorical
    ]

    print()
    print(
        "Categorical features:",
        len(categorical)
    )

    for x in categorical:
        print(
            f"  - {x}"
        )

    print()
    print(
        "Numerical features:",
        len(numerical)
    )

    # ========================================================
    # CONSTANT FEATURES
    # ========================================================

    header("9. CONSTANT FEATURE AUDIT")

    cardinality = (
        structured_train
        .nunique(dropna=False)
        .sort_values()
    )

    constants = cardinality[
        cardinality <= 1
    ]

    print(
        "Constant features:",
        len(constants)
    )

    if len(constants):
        print(
            constants.to_string()
        )
    else:
        print(
            "[PASS] No constant features."
        )

    # ========================================================
    # DUPLICATE AUDIT
    # ========================================================

    header("10. DUPLICATE AUDIT")

    structured_train_duplicates = (
        structured_train.duplicated().sum()
    )

    structured_test_duplicates = (
        structured_test.duplicated().sum()
    )

    print(
        "Structured training duplicates:",
        int(structured_train_duplicates)
    )

    print(
        "Structured testing duplicates :",
        int(structured_test_duplicates)
    )

    # Combined signatures
    train_struct_sig = (
        structured_train
        .fillna("<missing>")
        .astype(str)
        .agg("|".join, axis=1)
    )

    test_struct_sig = (
        structured_test
        .fillna("<missing>")
        .astype(str)
        .agg("|".join, axis=1)
    )

    train_text_sig = (
        text_train[text_train_col]
        .fillna("")
        .astype(str)
    )

    test_text_sig = (
        text_test[text_test_col]
        .fillna("")
        .astype(str)
    )

    train_combined = (
        train_struct_sig
        + "||"
        + train_text_sig
    )

    test_combined = (
        test_struct_sig
        + "||"
        + test_text_sig
    )

    print()
    print(
        "Unique structured training signatures:",
        train_struct_sig.nunique()
    )

    print(
        "Unique combined training signatures:",
        train_combined.nunique()
    )

    print()
    print(
        "Unique structured testing signatures:",
        test_struct_sig.nunique()
    )

    print(
        "Unique combined testing signatures:",
        test_combined.nunique()
    )

    # ========================================================
    # TRAIN/TEST SIGNATURE LEAKAGE
    # ========================================================

    header("11. TRAIN/TEST SIGNATURE LEAKAGE AUDIT")

    train_signature_set = set(
        train_combined
    )

    test_signature_set = set(
        test_combined
    )

    overlap = (
        train_signature_set
        & test_signature_set
    )

    print(
        "Training combined signatures:",
        len(train_signature_set)
    )

    print(
        "Testing combined signatures :",
        len(test_signature_set)
    )

    print(
        "Exact combined signatures appearing in both:",
        len(overlap)
    )

    if overlap:
        print()
        print(
            "[WARNING] Exact behavior signatures "
            "occur in both train and test."
        )

        # Count rows affected
        train_overlap_rows = (
            train_combined.isin(overlap).sum()
        )

        test_overlap_rows = (
            test_combined.isin(overlap).sum()
        )

        print(
            "Training rows affected:",
            int(train_overlap_rows)
        )

        print(
            "Testing rows affected :",
            int(test_overlap_rows)
        )

    else:
        print(
            "[PASS] No exact combined-signature overlap."
        )

    # ========================================================
    # LABEL CONFLICT AUDIT
    # ========================================================

    header("12. SIGNATURE LABEL-CONFLICT AUDIT")

    train_signature_labels = pd.DataFrame(
        {
            "signature": train_combined,
            "label": labels_train[
                "techniques"
            ].astype(str),
        }
    )

    conflicts = (
        train_signature_labels
        .groupby("signature")["label"]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    print(
        "Training signatures with conflicting labels:",
        len(conflicts)
    )

    if len(conflicts):

        print()
        print(
            "Top conflicts:"
        )

        for signature in conflicts.head(10).index:

            rows = train_signature_labels[
                train_signature_labels[
                    "signature"
                ] == signature
            ]

            print()
            print(
                "Signature:"
            )

            print(
                signature[:500]
            )

            print(
                "\nLabels:"
            )

            print(
                rows["label"]
                .value_counts()
                .to_string()
            )

    # ========================================================
    # TEMPORAL AUDIT
    # ========================================================

    header("13. TEMPORAL AUDIT")

    def parse_timestamp(series):

        cleaned = (
            series
            .astype(str)
            .str.replace(
                " @ ",
                " ",
                regex=False,
            )
            .str.strip()
        )

        return pd.to_datetime(
            cleaned,
            format="%b %d, %Y %H:%M:%S.%f",
            errors="coerce",
        )

    train_time = parse_timestamp(
        metadata_train["timestamp"]
    )

    test_time = parse_timestamp(
        metadata_test["timestamp"]
    )

    print(
        "Training invalid timestamps:",
        int(train_time.isna().sum())
    )

    print(
        "Testing invalid timestamps :",
        int(test_time.isna().sum())
    )

    if train_time.isna().any():
        raise ValueError(
            "Training contains invalid timestamps."
        )

    if test_time.isna().any():
        raise ValueError(
            "Testing contains invalid timestamps."
        )

    print()
    print(
        "Training:",
        train_time.min(),
        "->",
        train_time.max()
    )

    print(
        "Testing :",
        test_time.min(),
        "->",
        test_time.max()
    )

    if train_time.max() < test_time.min():
        print(
            "[PASS] Strict temporal separation."
        )
    else:
        print(
            "[FAIL] Temporal overlap detected."
        )

    # ========================================================
    # FINAL RISK ASSESSMENT
    # ========================================================

    header("14. v4.1 TRAINING READINESS")

    warnings = []

    if len(test_only):
        warnings.append(
            "Test-only MITRE techniques exist."
        )

    if len(conflicts):
        warnings.append(
            "Conflicting behavior signatures exist."
        )

    if len(rare_10):
        warnings.append(
            f"{len(rare_10)} techniques have <=10 "
            "training examples."
        )

    if empty_test.mean() > 0.20:
        warnings.append(
            "More than 20% of test behavior text is empty."
        )

    if overlap:
        warnings.append(
            "Exact combined signatures overlap train/test."
        )

    if train_time.max() >= test_time.min():
        warnings.append(
            "Temporal overlap detected."
        )

    if warnings:

        print(
            "WARNINGS:"
        )

        for warning in warnings:
            print(
                f"[!] {warning}"
            )

        print()
        print(
            "Dataset is NOT clean enough for blind "
            "deployment claims."
        )

        print(
            "Training may proceed only as an experimental "
            "baseline after documenting these limitations."
        )

    else:

        print(
            "[PASS] No major audit warnings."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print("v4.1 AUDIT SUMMARY")
    print("=" * 78)

    print(
        f"Training events       : {len(structured_train)}"
    )

    print(
        f"Testing events        : {len(structured_test)}"
    )

    print(
        f"Structured features   : {len(train_columns)}"
    )

    print(
        f"MITRE techniques      : {len(all_techniques)}"
    )

    print(
        f"Rare classes <=10     : {len(rare_10)}"
    )

    print(
        f"Train/test signature overlap: {len(overlap)}"
    )

    print(
        f"Conflicting signatures: {len(conflicts)}"
    )

    print(
        f"Empty test text       : {int(empty_test.sum())}"
    )

    print(
        f"Test-only techniques  : {len(test_only)}"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Review this audit output before training v4.1."
    )


if __name__ == "__main__":
    main()