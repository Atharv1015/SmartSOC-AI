from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "dataset_readiness_v4_1"
)

INPUT_FILE = (
    REPORT_DIR
    / "train_test_overlap_unique_details.csv"
)

OUTPUT_ALL = (
    REPORT_DIR
    / "train_test_overlap_inspection.csv"
)

OUTPUT_CONFLICTS = (
    REPORT_DIR
    / "train_test_label_conflicts.csv"
)

OUTPUT_LABEL_SUMMARY = (
    REPORT_DIR
    / "train_test_overlap_label_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("SmartSOC-AI v4.1")
print("TRAIN / TEST OVERLAP INSPECTION")
print("=" * 75)


# ============================================================
# 1. LOAD OVERLAP REPORT
# ============================================================

print()
print("[1] Loading overlap report...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Overlap report not found:\n{INPUT_FILE}"
    )


overlap = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)


print(
    f"[PASS] Loaded {len(overlap):,} "
    f"unique overlap records."
)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

required_columns = [
    "train_index",
    "test_index",
    "hash",
    "train_label",
    "test_label",
    "same_label",
]


missing = [
    column
    for column in required_columns
    if column not in overlap.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


print(
    "[PASS] Required columns present."
)


# ============================================================
# 3. DISPLAY OVERALL OVERLAP INFORMATION
# ============================================================

print()
print("=" * 75)
print("OVERALL OVERLAP")
print("=" * 75)

total = len(overlap)

same = int(
    overlap["same_label"].sum()
)

different = int(
    (~overlap["same_label"]).sum()
)


print()
print(
    f"Total unique overlaps : {total:,}"
)

print(
    f"Same-label overlaps   : {same:,}"
)

print(
    f"Different-label       : {different:,}"
)


if total > 0:

    print(
        f"Same-label percentage : "
        f"{same / total * 100:.2f}%"
    )

    print(
        f"Conflict percentage   : "
        f"{different / total * 100:.2f}%"
    )


# ============================================================
# 4. LABEL DISTRIBUTION AMONG OVERLAPS
# ============================================================

print()
print("=" * 75)
print("OVERLAP LABEL DISTRIBUTION")
print("=" * 75)


train_label_counts = (
    overlap["train_label"]
    .value_counts()
)


test_label_counts = (
    overlap["test_label"]
    .value_counts()
)


print()
print("Training labels in overlap:")

for label, count in train_label_counts.items():

    print(
        f"  {label:<40} {count:>5}"
    )


print()
print("Testing labels in overlap:")

for label, count in test_label_counts.items():

    print(
        f"  {label:<40} {count:>5}"
    )


# ============================================================
# 5. LABEL PAIR SUMMARY
# ============================================================

print()
print("=" * 75)
print("TRAIN LABEL → TEST LABEL PAIRS")
print("=" * 75)


pair_summary = (
    overlap
    .groupby(
        [
            "train_label",
            "test_label",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)


print()

for _, row in pair_summary.iterrows():

    train_label = str(
        row["train_label"]
    )

    test_label = str(
        row["test_label"]
    )

    count = int(
        row["count"]
    )

    marker = (
        "SAME"
        if train_label == test_label
        else "CONFLICT"
    )

    print(
        f"[{marker:<8}] "
        f"{count:>4}  "
        f"TRAIN: {train_label}  "
        f"TEST: {test_label}"
    )


# ============================================================
# 6. EXTRACT CONFLICTS
# ============================================================

print()
print("=" * 75)
print("DIFFERENT-LABEL OVERLAPS")
print("=" * 75)


conflicts = (
    overlap[
        ~overlap["same_label"]
    ]
    .copy()
)


print()
print(
    f"Found {len(conflicts):,} "
    f"different-label overlap(s)."
)


if len(conflicts) == 0:

    print()
    print(
        "[PASS] No label conflicts found."
    )

else:

    print()

    for number, (_, row) in enumerate(
        conflicts.iterrows(),
        start=1,
    ):

        print(
            "-" * 75
        )

        print(
            f"CONFLICT #{number}"
        )

        print(
            f"Train index : "
            f"{row['train_index']}"
        )

        print(
            f"Test index  : "
            f"{row['test_index']}"
        )

        print(
            f"Hash        : "
            f"{row['hash']}"
        )

        print(
            f"Train label : "
            f"{row['train_label']}"
        )

        print(
            f"Test label  : "
            f"{row['test_label']}"
        )


        # Display metadata
        train_meta_columns = [
            column
            for column in overlap.columns
            if column.startswith(
                "train_meta_"
            )
        ]

        test_meta_columns = [
            column
            for column in overlap.columns
            if column.startswith(
                "test_meta_"
            )
        ]


        print()
        print("Training metadata:")

        for column in train_meta_columns:

            print(
                f"  {column}: "
                f"{row[column]}"
            )


        print()
        print("Testing metadata:")

        for column in test_meta_columns:

            print(
                f"  {column}: "
                f"{row[column]}"
            )


# ============================================================
# 7. SAVE CONFLICT REPORT
# ============================================================

conflicts.to_csv(
    OUTPUT_CONFLICTS,
    index=False,
)


print()
print(
    f"[SAVED] Conflict report:"
)

print(
    OUTPUT_CONFLICTS
)


# ============================================================
# 8. SAVE ALL OVERLAP RECORDS
# ============================================================

overlap.to_csv(
    OUTPUT_ALL,
    index=False,
)


print()
print(
    f"[SAVED] Full overlap inspection:"
)

print(
    OUTPUT_ALL
)


# ============================================================
# 9. SAVE LABEL SUMMARY
# ============================================================

label_summary = (
    pair_summary
    .copy()
)


label_summary[
    "same_label"
] = (
    label_summary[
        "train_label"
    ]
    ==
    label_summary[
        "test_label"
    ]
)


label_summary.to_csv(
    OUTPUT_LABEL_SUMMARY,
    index=False,
)


print()
print(
    f"[SAVED] Label-pair summary:"
)

print(
    OUTPUT_LABEL_SUMMARY
)


# ============================================================
# 10. FINAL ASSESSMENT
# ============================================================

print()
print("=" * 75)
print("INITIAL ASSESSMENT")
print("=" * 75)

print()

if total == 0:

    print(
        "[PASS] No cross-split overlaps detected."
    )

elif different == 0:

    print(
        "[REVIEW] Cross-split overlaps exist, "
        "but all overlapping representations "
        "have the same label."
    )

    print()
    print(
        "These may represent repeated legitimate "
        "event patterns rather than label leakage."
    )

else:

    print(
        "[WARNING] Cross-split label conflicts detected."
    )

    print()
    print(
        f"{different} unique structured representation(s) "
        "have different labels between train and test."
    )

    print()
    print(
        "These conflicts require investigation before "
        "changing the train/test split or retraining."
    )


print()
print("=" * 75)
print("NEXT DECISION")
print("=" * 75)

print()

print(
    "Do NOT delete rows yet."
)

print(
    "Do NOT retrain the model yet."
)

print(
    "Do NOT change to 80/20 yet."
)

print()
print(
    "First inspect the conflict report and determine "
    "whether the conflicting labels are legitimate "
    "multi-technique/context differences or dataset leakage."
)

print()
print("Finished.")