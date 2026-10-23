from pathlib import Path
from collections import Counter
import json
import hashlib

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "windows_apt_v4_1"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "dataset_readiness_v4_1"
)

TRAIN_STRUCTURED = DATA_DIR / "structured_train.csv"
TEST_STRUCTURED = DATA_DIR / "structured_test.csv"

TRAIN_LABELS = DATA_DIR / "labels_train.csv"
TEST_LABELS = DATA_DIR / "labels_test.csv"

TRAIN_METADATA = DATA_DIR / "metadata_train.csv"
TEST_METADATA = DATA_DIR / "metadata_test.csv"


# ============================================================
# HELPERS
# ============================================================

def normalize_value(value):
    """
    Convert values into deterministic strings so that identical
    feature rows receive identical fingerprints.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return format(value, ".15g")

    return str(value)


def make_fingerprint(row, feature_columns):
    """
    Create an exact SHA256 fingerprint from all structured features.
    """

    values = [
        normalize_value(row[col])
        for col in feature_columns
    ]

    payload = "\x1f".join(values)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def normalize_label(label):
    """
    Normalize MITRE technique label strings.

    Examples:
        'T1059, T1105'
        'T1059,T1105'
        'T1105,T1059'

    become:

        'T1059,T1105'
    """

    if pd.isna(label):
        return ""

    text = str(label).strip()

    if not text:
        return ""

    parts = [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]

    # Remove accidental duplicates and sort for deterministic comparison
    parts = sorted(set(parts))

    return ",".join(parts)


def label_tuple(label):
    normalized = normalize_label(label)

    if not normalized:
        return tuple()

    return tuple(normalized.split(","))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SmartSOC - Global Duplicate Label Ambiguity Audit v4.1")
    print("=" * 80)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Report directory: {REPORT_DIR}")
    print()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    required_files = [
        TRAIN_STRUCTURED,
        TEST_STRUCTURED,
        TRAIN_LABELS,
        TEST_LABELS,
        TRAIN_METADATA,
        TEST_METADATA,
    ]

    print("[1/8] Checking required files...")

    missing_files = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print()
        print("ERROR: Missing required files:")

        for path in missing_files:
            print(f"  - {path}")

        raise SystemExit(1)

    print("PASS - all required files exist.")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("[2/8] Loading train/test data...")

    train_structured = pd.read_csv(
        TRAIN_STRUCTURED,
        low_memory=False
    )

    test_structured = pd.read_csv(
        TEST_STRUCTURED,
        low_memory=False
    )

    train_labels = pd.read_csv(
        TRAIN_LABELS,
        low_memory=False
    )

    test_labels = pd.read_csv(
        TEST_LABELS,
        low_memory=False
    )

    train_metadata = pd.read_csv(
        TRAIN_METADATA,
        low_memory=False
    )

    test_metadata = pd.read_csv(
        TEST_METADATA,
        low_memory=False
    )

    print(
        f"Train structured rows: {len(train_structured):,}"
    )

    print(
        f"Test structured rows : {len(test_structured):,}"
    )

    print(
        f"Train labels rows    : {len(train_labels):,}"
    )

    print(
        f"Test labels rows     : {len(test_labels):,}"
    )

    print()

    # --------------------------------------------------------
    # Basic alignment validation
    # --------------------------------------------------------

    print("[3/8] Validating row alignment...")

    if len(train_structured) != len(train_labels):
        raise ValueError(
            "Train structured rows do not match train label rows."
        )

    if len(test_structured) != len(test_labels):
        raise ValueError(
            "Test structured rows do not match test label rows."
        )

    if len(train_structured) != len(train_metadata):
        raise ValueError(
            "Train structured rows do not match train metadata rows."
        )

    if len(test_structured) != len(test_metadata):
        raise ValueError(
            "Test structured rows do not match test metadata rows."
        )

    if "techniques" not in train_labels.columns:
        raise ValueError(
            "Train labels file does not contain 'techniques' column."
        )

    if "techniques" not in test_labels.columns:
        raise ValueError(
            "Test labels file does not contain 'techniques' column."
        )

    print("PASS - train/test rows are aligned.")
    print()

    # --------------------------------------------------------
    # Determine exact structured feature columns
    # --------------------------------------------------------

    print("[4/8] Determining the 62 structured feature columns...")

    feature_columns = list(train_structured.columns)

    if len(feature_columns) != 62:
        raise ValueError(
            f"Expected 62 structured features, "
            f"found {len(feature_columns)}."
        )

    if list(test_structured.columns) != feature_columns:
        raise ValueError(
            "Train and test structured schemas do not match exactly."
        )

    print(f"PASS - exactly {len(feature_columns)} features detected.")

    print()

    # --------------------------------------------------------
    # Build combined dataset
    # --------------------------------------------------------

    print("[5/8] Building global dataset and fingerprints...")

    train = train_structured.copy()

    test = test_structured.copy()

    train["label"] = train_labels["techniques"].map(
        normalize_label
    )

    test["label"] = test_labels["techniques"].map(
        normalize_label
    )

    train["split"] = "train"
    test["split"] = "test"

    train["row_index"] = range(len(train))
    test["row_index"] = range(len(test))

    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True
    )

    combined["fingerprint"] = combined.apply(
        lambda row: make_fingerprint(
            row,
            feature_columns
        ),
        axis=1
    )

    print(
        f"Total dataset rows: {len(combined):,}"
    )

    print(
        f"Unique representations: "
        f"{combined['fingerprint'].nunique():,}"
    )

    print()

    # --------------------------------------------------------
    # Group exact duplicate feature representations
    # --------------------------------------------------------

    grouped = (
        combined
        .groupby("fingerprint", sort=False)
        .agg(
            total_rows=("fingerprint", "size"),
            unique_labels=("label", "nunique"),
            unique_splits=("split", "nunique"),
        )
        .reset_index()
    )

    grouped["ambiguous"] = (
        grouped["unique_labels"] > 1
    )

    grouped["repeated"] = (
        grouped["total_rows"] > 1
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    unique_representations = len(grouped)

    repeated_representations = int(
        grouped["repeated"].sum()
    )

    ambiguous_representations = int(
        grouped["ambiguous"].sum()
    )

    repeated_rows = int(
        combined["fingerprint"].map(
            combined["fingerprint"].value_counts()
        ).gt(1).sum()
    )

    ambiguous_fingerprints = set(
        grouped.loc[
            grouped["ambiguous"],
            "fingerprint"
        ]
    )

    ambiguous_rows_df = combined[
        combined["fingerprint"].isin(
            ambiguous_fingerprints
        )
    ].copy()

    ambiguous_rows = len(
        ambiguous_rows_df
    )

    # --------------------------------------------------------
    # Print main results
    # --------------------------------------------------------

    print("[6/8] Global ambiguity results")
    print("-" * 80)

    print(
        f"Total rows                         : "
        f"{len(combined):,}"
    )

    print(
        f"Unique feature representations     : "
        f"{unique_representations:,}"
    )

    print(
        f"Repeated representations           : "
        f"{repeated_representations:,}"
    )

    print(
        f"Rows belonging to repeated reps    : "
        f"{repeated_rows:,}"
    )

    print(
        f"Ambiguous representations          : "
        f"{ambiguous_representations:,}"
    )

    print(
        f"Rows affected by label ambiguity   : "
        f"{ambiguous_rows:,}"
    )

    if unique_representations:
        ambiguity_rate = (
            ambiguous_representations
            / unique_representations
            * 100
        )
    else:
        ambiguity_rate = 0.0

    if len(combined):
        ambiguous_row_rate = (
            ambiguous_rows
            / len(combined)
            * 100
        )
    else:
        ambiguous_row_rate = 0.0

    print(
        f"Ambiguous representation rate      : "
        f"{ambiguity_rate:.4f}%"
    )

    print(
        f"Ambiguous row rate                 : "
        f"{ambiguous_row_rate:.4f}%"
    )

    print()

    # --------------------------------------------------------
    # Label combination analysis
    # --------------------------------------------------------

    print(
        "[7/8] Analyzing conflicting label combinations..."
    )

    ambiguity_records = []

    for fingerprint in ambiguous_fingerprints:

        subset = combined[
            combined["fingerprint"] == fingerprint
        ].copy()

        label_counts = (
            subset["label"]
            .value_counts()
            .to_dict()
        )

        labels_sorted = sorted(
            label_counts.keys()
        )

        train_count = int(
            (subset["split"] == "train").sum()
        )

        test_count = int(
            (subset["split"] == "test").sum()
        )

        ambiguity_records.append(
            {
                "fingerprint": fingerprint,
                "total_rows": len(subset),
                "unique_labels": len(labels_sorted),
                "labels": " | ".join(labels_sorted),
                "label_counts": json.dumps(
                    label_counts,
                    sort_keys=True
                ),
                "train_rows": train_count,
                "test_rows": test_count,
                "cross_split": (
                    train_count > 0
                    and test_count > 0
                ),
            }
        )

    ambiguity_summary = pd.DataFrame(
        ambiguity_records
    )

    if not ambiguity_summary.empty:

        ambiguity_summary = (
            ambiguity_summary
            .sort_values(
                [
                    "total_rows",
                    "unique_labels",
                ],
                ascending=False
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Pairwise label conflicts
    # --------------------------------------------------------

    pair_counter = Counter()

    for fingerprint in ambiguous_fingerprints:

        subset = combined[
            combined["fingerprint"] == fingerprint
        ]

        labels = sorted(
            set(subset["label"])
        )

        for i in range(len(labels)):

            for j in range(i + 1, len(labels)):

                pair = (
                    labels[i],
                    labels[j]
                )

                pair_counter[pair] += 1

    pair_records = []

    for (label_a, label_b), count in pair_counter.items():

        pair_records.append(
            {
                "label_a": label_a,
                "label_b": label_b,
                "ambiguous_representations": count,
            }
        )

    pair_summary = pd.DataFrame(
        pair_records
    )

    if not pair_summary.empty:

        pair_summary = (
            pair_summary
            .sort_values(
                "ambiguous_representations",
                ascending=False
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Split-specific ambiguity
    # --------------------------------------------------------

    train_only_ambiguous = 0
    test_only_ambiguous = 0
    cross_split_ambiguous = 0

    for fingerprint in ambiguous_fingerprints:

        subset = combined[
            combined["fingerprint"] == fingerprint
        ]

        has_train = (
            (subset["split"] == "train").any()
        )

        has_test = (
            (subset["split"] == "test").any()
        )

        if has_train and has_test:
            cross_split_ambiguous += 1

        elif has_train:
            train_only_ambiguous += 1

        elif has_test:
            test_only_ambiguous += 1

    print()
    print("Ambiguous representation location:")
    print(
        f"  Train-only : {train_only_ambiguous:,}"
    )
    print(
        f"  Test-only  : {test_only_ambiguous:,}"
    )
    print(
        f"  Cross-split: {cross_split_ambiguous:,}"
    )

    print()

    # --------------------------------------------------------
    # Detailed ambiguous rows
    # --------------------------------------------------------

    if ambiguous_fingerprints:

        detail_columns = (
            feature_columns
            + [
                "label",
                "split",
                "row_index",
                "fingerprint",
            ]
        )

        ambiguous_detail = (
            combined[
                combined["fingerprint"].isin(
                    ambiguous_fingerprints
                )
            ][detail_columns]
            .sort_values(
                [
                    "fingerprint",
                    "split",
                    "row_index",
                ]
            )
            .reset_index(drop=True)
        )

    else:

        ambiguous_detail = pd.DataFrame(
            columns=(
                feature_columns
                + [
                    "label",
                    "split",
                    "row_index",
                    "fingerprint",
                ]
            )
        )

    # --------------------------------------------------------
    # Split-aware ambiguity details
    # --------------------------------------------------------

    split_records = []

    for fingerprint in ambiguous_fingerprints:

        subset = combined[
            combined["fingerprint"] == fingerprint
        ]

        train_labels_set = sorted(
            set(
                subset.loc[
                    subset["split"] == "train",
                    "label"
                ]
            )
        )

        test_labels_set = sorted(
            set(
                subset.loc[
                    subset["split"] == "test",
                    "label"
                ]
            )
        )

        split_records.append(
            {
                "fingerprint": fingerprint,
                "train_rows": int(
                    (subset["split"] == "train").sum()
                ),
                "test_rows": int(
                    (subset["split"] == "test").sum()
                ),
                "train_labels": " | ".join(
                    train_labels_set
                ),
                "test_labels": " | ".join(
                    test_labels_set
                ),
                "cross_split_conflict": (
                    len(train_labels_set) > 0
                    and len(test_labels_set) > 0
                    and set(train_labels_set)
                    != set(test_labels_set)
                ),
            }
        )

    split_summary = pd.DataFrame(
        split_records
    )

    if not split_summary.empty:

        split_summary = (
            split_summary
            .sort_values(
                [
                    "cross_split_conflict",
                    "train_rows",
                    "test_rows",
                ],
                ascending=False
            )
            .reset_index(drop=True)
        )

    # --------------------------------------------------------
    # Repeated representation statistics
    # --------------------------------------------------------

    representation_frequency = (
        combined["fingerprint"]
        .value_counts()
        .rename_axis("fingerprint")
        .reset_index(name="row_count")
    )

    frequency_distribution = (
        representation_frequency["row_count"]
        .value_counts()
        .sort_index()
        .rename_axis("rows_per_representation")
        .reset_index(name="representation_count")
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    print()
    print("Saving reports...")

    summary = {
        "dataset": "windows_apt_v4_1",

        "total_rows": int(len(combined)),

        "train_rows": int(len(train)),
        "test_rows": int(len(test)),

        "structured_feature_count": int(
            len(feature_columns)
        ),

        "unique_feature_representations": int(
            unique_representations
        ),

        "repeated_feature_representations": int(
            repeated_representations
        ),

        "rows_in_repeated_representations": int(
            repeated_rows
        ),

        "ambiguous_feature_representations": int(
            ambiguous_representations
        ),

        "rows_affected_by_label_ambiguity": int(
            ambiguous_rows
        ),

        "ambiguous_representation_rate_percent": round(
            ambiguity_rate,
            6
        ),

        "ambiguous_row_rate_percent": round(
            ambiguous_row_rate,
            6
        ),

        "train_only_ambiguous_representations": int(
            train_only_ambiguous
        ),

        "test_only_ambiguous_representations": int(
            test_only_ambiguous
        ),

        "cross_split_ambiguous_representations": int(
            cross_split_ambiguous
        ),

        "feature_columns": feature_columns,
    }

    summary_path = (
        REPORT_DIR
        / "duplicate_label_ambiguity_summary_v4_1.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    ambiguity_summary_path = (
        REPORT_DIR
        / "duplicate_label_ambiguity_representations_v4_1.csv"
    )

    ambiguity_summary.to_csv(
        ambiguity_summary_path,
        index=False
    )

    pair_summary_path = (
        REPORT_DIR
        / "duplicate_label_ambiguity_label_pairs_v4_1.csv"
    )

    pair_summary.to_csv(
        pair_summary_path,
        index=False
    )

    detail_path = (
        REPORT_DIR
        / "duplicate_label_ambiguity_details_v4_1.csv"
    )

    ambiguous_detail.to_csv(
        detail_path,
        index=False
    )

    split_summary_path = (
        REPORT_DIR
        / "duplicate_label_ambiguity_split_analysis_v4_1.csv"
    )

    split_summary.to_csv(
        split_summary_path,
        index=False
    )

    frequency_path = (
        REPORT_DIR
        / "duplicate_representation_frequency_v4_1.csv"
    )

    frequency_distribution.to_csv(
        frequency_path,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print()
    print("Key results:")
    print(
        f"  Total rows                         : "
        f"{len(combined):,}"
    )

    print(
        f"  Unique representations             : "
        f"{unique_representations:,}"
    )

    print(
        f"  Repeated representations           : "
        f"{repeated_representations:,}"
    )

    print(
        f"  Ambiguous representations          : "
        f"{ambiguous_representations:,}"
    )

    print(
        f"  Rows affected by ambiguity         : "
        f"{ambiguous_rows:,}"
    )

    print(
        f"  Train-only ambiguity               : "
        f"{train_only_ambiguous:,}"
    )

    print(
        f"  Test-only ambiguity                : "
        f"{test_only_ambiguous:,}"
    )

    print(
        f"  Cross-split ambiguity              : "
        f"{cross_split_ambiguous:,}"
    )

    print()
    print("Reports written to:")
    print(f"  {summary_path}")
    print(f"  {ambiguity_summary_path}")
    print(f"  {pair_summary_path}")
    print(f"  {detail_path}")
    print(f"  {split_summary_path}")
    print(f"  {frequency_path}")

    print()
    print("Next step: review these results before changing")
    print("the dataset split, labels, or model.")
    print()


if __name__ == "__main__":
    main()