from pathlib import Path
import time
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


# ============================================================
# TIMER
# ============================================================

SCRIPT_START = time.perf_counter()


def elapsed():
    """Return total elapsed time in seconds."""
    return time.perf_counter() - SCRIPT_START


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stage_start(name):
    """Start a stage timer."""
    print()
    print("=" * 75)
    print(f"START: {name}")
    print(f"Total elapsed: {format_time(elapsed())}")
    print("=" * 75)

    return time.perf_counter()


def stage_end(name, start_time):
    """End a stage timer."""
    duration = time.perf_counter() - start_time

    print()
    print(f"[DONE] {name}")
    print(f"Stage time  : {format_time(duration)}")
    print(f"Total time  : {format_time(elapsed())}")

    return duration


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "dataset_readiness_v4_1"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


STRUCTURED_TRAIN = (
    DATA_DIR / "structured_train.csv"
)

STRUCTURED_TEST = (
    DATA_DIR / "structured_test.csv"
)

LABELS_TRAIN = (
    DATA_DIR / "labels_train.csv"
)

LABELS_TEST = (
    DATA_DIR / "labels_test.csv"
)

METADATA_TRAIN = (
    DATA_DIR / "metadata_train.csv"
)

METADATA_TEST = (
    DATA_DIR / "metadata_test.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OVERLAP_FILE = (
    REPORT_DIR
    / "train_test_overlap_unique_details.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "train_test_overlap_unique_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("SmartSOC-AI v4.1")
print("UNIQUE TRAIN / TEST OVERLAP INVESTIGATION")
print("=" * 75)

print()
print("Purpose:")
print("Detect unique structured feature representations")
print("that occur in BOTH training and testing datasets.")

print()
print("This version avoids the many-to-many merge problem")
print("that produced millions of duplicate overlap pairs.")

print()
print("Timer started.")


# ============================================================
# 1. LOAD DATASETS
# ============================================================

t = stage_start("1. Loading datasets")

print()
print("Loading training structured data...")

train_x = pd.read_csv(
    STRUCTURED_TRAIN,
    low_memory=False,
)

print("Loading testing structured data...")

test_x = pd.read_csv(
    STRUCTURED_TEST,
    low_memory=False,
)

print("Loading training labels...")

train_y = pd.read_csv(
    LABELS_TRAIN,
    low_memory=False,
)

print("Loading testing labels...")

test_y = pd.read_csv(
    LABELS_TEST,
    low_memory=False,
)

print("Loading training metadata...")

train_meta = pd.read_csv(
    METADATA_TRAIN,
    low_memory=False,
)

print("Loading testing metadata...")

test_meta = pd.read_csv(
    METADATA_TEST,
    low_memory=False,
)


print()
print("DATASET SIZES")
print("-" * 50)

print(
    f"Train structured : {train_x.shape}"
)

print(
    f"Test structured  : {test_x.shape}"
)

print(
    f"Train labels     : {train_y.shape}"
)

print(
    f"Test labels      : {test_y.shape}"
)

print(
    f"Train metadata   : {train_meta.shape}"
)

print(
    f"Test metadata    : {test_meta.shape}"
)

stage_end(
    "1. Loading datasets",
    t
)


# ============================================================
# 2. VALIDATE ROW ALIGNMENT
# ============================================================

t = stage_start(
    "2. Validating row alignment"
)


# Training alignment
if len(train_x) != len(train_y):

    raise ValueError(
        "Training structured rows and training labels "
        "do not have the same number of rows."
    )


if len(train_x) != len(train_meta):

    raise ValueError(
        "Training structured rows and training metadata "
        "do not have the same number of rows."
    )


# Testing alignment
if len(test_x) != len(test_y):

    raise ValueError(
        "Testing structured rows and testing labels "
        "do not have the same number of rows."
    )


if len(test_x) != len(test_meta):

    raise ValueError(
        "Testing structured rows and testing metadata "
        "do not have the same number of rows."
    )


print(
    "[PASS] Training structured/label/metadata rows aligned."
)

print(
    "[PASS] Testing structured/label/metadata rows aligned."
)


# Check column alignment
if list(train_x.columns) != list(test_x.columns):

    raise ValueError(
        "Training and testing structured feature columns "
        "are not identical or are in a different order."
    )


print(
    "[PASS] Train/test structured schemas match."
)

print(
    f"[PASS] Structured feature count: "
    f"{len(train_x.columns)}"
)


stage_end(
    "2. Validating row alignment",
    t
)


# ============================================================
# 3. CREATE FAST ROW FINGERPRINTS
# ============================================================

t = stage_start(
    "3. Creating fast row fingerprints"
)

print()
print(
    f"Hashing training rows: "
    f"{len(train_x):,}"
)

train_hash = hash_pandas_object(
    train_x,
    index=False,
).to_numpy()


print(
    f"Hashing testing rows: "
    f"{len(test_x):,}"
)

test_hash = hash_pandas_object(
    test_x,
    index=False,
).to_numpy()


print()
print("[PASS] Row fingerprints created.")


stage_end(
    "3. Creating fast row fingerprints",
    t
)


# ============================================================
# 4. COUNT UNIQUE FINGERPRINTS
# ============================================================

t = stage_start(
    "4. Counting unique fingerprints"
)

train_unique_hashes = np.unique(
    train_hash
)

test_unique_hashes = np.unique(
    test_hash
)


print()
print(
    f"Training rows          : "
    f"{len(train_hash):,}"
)

print(
    f"Unique training rows   : "
    f"{len(train_unique_hashes):,}"
)

print()
print(
    f"Testing rows           : "
    f"{len(test_hash):,}"
)

print(
    f"Unique testing rows    : "
    f"{len(test_unique_hashes):,}"
)


train_duplicate_rows = (
    len(train_hash)
    -
    len(train_unique_hashes)
)

test_duplicate_rows = (
    len(test_hash)
    -
    len(test_unique_hashes)
)


print()
print(
    f"Repeated training rows : "
    f"{train_duplicate_rows:,}"
)

print(
    f"Repeated testing rows  : "
    f"{test_duplicate_rows:,}"
)


stage_end(
    "4. Counting unique fingerprints",
    t
)


# ============================================================
# 5. FIND UNIQUE CROSS-SPLIT OVERLAP
# ============================================================

t = stage_start(
    "5. Finding UNIQUE train/test overlap"
)

print()
print(
    "Finding intersection of unique fingerprints..."
)

# np.intersect1d is efficient and does NOT create
# a many-to-many Cartesian product.
common_hashes = np.intersect1d(
    train_unique_hashes,
    test_unique_hashes,
)


print()
print(
    f"UNIQUE overlapping representations: "
    f"{len(common_hashes):,}"
)


if len(common_hashes) == 0:

    print()
    print(
        "[PASS] No unique structured representations "
        "are shared between train and test."
    )


stage_end(
    "5. Finding UNIQUE train/test overlap",
    t
)


# ============================================================
# 6. MAP EACH UNIQUE OVERLAP TO REPRESENTATIVE ROWS
# ============================================================

t = stage_start(
    "6. Mapping unique overlaps to rows"
)


if len(common_hashes) > 0:

    print()
    print(
        "Locating representative training rows..."
    )

    # Get first occurrence of every training hash
    train_first_positions = {}

    for idx, h in enumerate(train_hash):

        if h in train_first_positions:
            continue

        train_first_positions[h] = idx


    print(
        "Locating representative testing rows..."
    )

    # Get first occurrence of every testing hash
    test_first_positions = {}

    for idx, h in enumerate(test_hash):

        if h in test_first_positions:
            continue

        test_first_positions[h] = idx


    records = []


    total_unique_overlaps = len(
        common_hashes
    )


    print()
    print(
        f"Building report for "
        f"{total_unique_overlaps:,} "
        f"unique overlaps..."
    )


    for counter, h in enumerate(
        common_hashes,
        start=1
    ):

        train_idx = (
            train_first_positions[h]
        )

        test_idx = (
            test_first_positions[h]
        )


        train_label = str(
            train_y.iloc[
                train_idx
            ]["techniques"]
        )

        test_label = str(
            test_y.iloc[
                test_idx
            ]["techniques"]
        )


        record = {
            "train_index":
                train_idx,

            "test_index":
                test_idx,

            "hash":
                int(h),

            "train_label":
                train_label,

            "test_label":
                test_label,

            "same_label":
                train_label == test_label,
        }


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        for column in train_meta.columns:

            record[
                f"train_meta_{column}"
            ] = train_meta.iloc[
                train_idx
            ][column]


        for column in test_meta.columns:

            record[
                f"test_meta_{column}"
            ] = test_meta.iloc[
                test_idx
            ][column]


        records.append(
            record
        )


        # Progress display
        if (
            counter == 1
            or
            counter % 100 == 0
            or
            counter == total_unique_overlaps
        ):

            percentage = (
                counter
                /
                total_unique_overlaps
                *
                100
            )

            print(
                f"\rProgress: "
                f"{counter:,}/"
                f"{total_unique_overlaps:,} "
                f"({percentage:.1f}%)",
                end="",
                flush=True,
            )


    print()


    overlap_report = pd.DataFrame(
        records
    )


else:

    overlap_report = pd.DataFrame(
        columns=[
            "train_index",
            "test_index",
            "hash",
            "train_label",
            "test_label",
            "same_label",
        ]
    )


print()
print(
    f"Unique overlap records created: "
    f"{len(overlap_report):,}"
)


stage_end(
    "6. Mapping unique overlaps to rows",
    t
)


# ============================================================
# 7. LABEL AGREEMENT
# ============================================================

t = stage_start(
    "7. Checking label agreement"
)


if len(overlap_report) > 0:

    same_label_count = int(
        overlap_report[
            "same_label"
        ].sum()
    )

    different_label_count = int(
        (
            ~overlap_report[
                "same_label"
            ]
        ).sum()
    )

else:

    same_label_count = 0
    different_label_count = 0


print()
print(
    f"Unique overlapping representations: "
    f"{len(overlap_report):,}"
)

print(
    f"Same-label overlaps               : "
    f"{same_label_count:,}"
)

print(
    f"Different-label overlaps          : "
    f"{different_label_count:,}"
)


if len(overlap_report) > 0:

    same_label_percentage = (
        same_label_count
        /
        len(overlap_report)
        *
        100
    )

    different_label_percentage = (
        different_label_count
        /
        len(overlap_report)
        *
        100
    )

else:

    same_label_percentage = 0.0
    different_label_percentage = 0.0


print()
print(
    f"Same-label percentage     : "
    f"{same_label_percentage:.2f}%"
)

print(
    f"Different-label percentage: "
    f"{different_label_percentage:.2f}%"
)


stage_end(
    "7. Checking label agreement",
    t
)


# ============================================================
# 8. SAVE UNIQUE OVERLAP REPORT
# ============================================================

t = stage_start(
    "8. Saving unique overlap report"
)


overlap_report.to_csv(
    OVERLAP_FILE,
    index=False,
)


print()
print(
    "Detailed report saved:"
)

print(
    OVERLAP_FILE
)


stage_end(
    "8. Saving unique overlap report",
    t
)


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

t = stage_start(
    "9. Saving summary"
)


summary = pd.DataFrame([
    {
        "metric":
            "structured_train_rows",

        "value":
            len(train_x),
    },

    {
        "metric":
            "structured_test_rows",

        "value":
            len(test_x),
    },

    {
        "metric":
            "unique_train_fingerprints",

        "value":
            len(train_unique_hashes),
    },

    {
        "metric":
            "unique_test_fingerprints",

        "value":
            len(test_unique_hashes),
    },

    {
        "metric":
            "duplicate_train_rows",

        "value":
            train_duplicate_rows,
    },

    {
        "metric":
            "duplicate_test_rows",

        "value":
            test_duplicate_rows,
    },

    {
        "metric":
            "unique_cross_split_overlaps",

        "value":
            len(common_hashes),
    },

    {
        "metric":
            "same_label_overlaps",

        "value":
            same_label_count,
    },

    {
        "metric":
            "different_label_overlaps",

        "value":
            different_label_count,
    },

    {
        "metric":
            "same_label_percentage",

        "value":
            round(
                same_label_percentage,
                4,
            ),
    },

    {
        "metric":
            "different_label_percentage",

        "value":
            round(
                different_label_percentage,
                4,
            ),
    },
])


summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


print()
print(
    "Summary saved:"
)

print(
    SUMMARY_FILE
)


stage_end(
    "9. Saving summary",
    t
)


# ============================================================
# FINAL RESULT
# ============================================================

total_time = elapsed()


print()
print()
print("=" * 75)
print("FINAL RESULT")
print("=" * 75)

print()

print(
    f"Training rows                  : "
    f"{len(train_x):,}"
)

print(
    f"Testing rows                   : "
    f"{len(test_x):,}"
)

print(
    f"Unique training representations: "
    f"{len(train_unique_hashes):,}"
)

print(
    f"Unique testing representations : "
    f"{len(test_unique_hashes):,}"
)

print(
    f"Repeated training rows         : "
    f"{train_duplicate_rows:,}"
)

print(
    f"Repeated testing rows          : "
    f"{test_duplicate_rows:,}"
)

print()

print(
    f"UNIQUE cross-split overlaps    : "
    f"{len(common_hashes):,}"
)

print(
    f"Same-label overlaps            : "
    f"{same_label_count:,}"
)

print(
    f"Different-label overlaps       : "
    f"{different_label_count:,}"
)

print()

print(
    f"Detailed report:"
)

print(
    OVERLAP_FILE
)

print()

print(
    f"Summary:"
)

print(
    SUMMARY_FILE
)

print()

print("=" * 75)

print(
    f"TOTAL SCRIPT TIME: "
    f"{format_time(total_time)}"
)

print("=" * 75)

print()

print(
    "IMPORTANT:"
)

print(
    "Do NOT delete overlapping rows automatically."
)

print(
    "First determine whether identical structured "
    "representations are legitimate repeated events "
    "or evidence of train/test leakage."
)

print()
print(
    "The previous many-to-many merge could produce "
    "millions of overlap PAIRS when duplicate rows "
    "exist. This version reports UNIQUE overlapping "
    "representations instead."
)