"""
SmartSOC Windows-APT Model B v4
PRE-TRAINING DATA QUALITY AUDIT

Purpose:
    - Validate v4 train/test alignment
    - Validate missing values
    - Validate MITRE technique coverage
    - Audit rare techniques
    - Audit label combinations
    - Audit multi-label distribution
    - Audit behavior text
    - Audit PowerShell behavior
    - Audit high-risk behavioral features
    - Audit technique/feature associations
    - Validate timestamps
    - Validate temporal separation
    - Audit agent distribution
    - Audit event-ID distribution
    - Produce final TRAINING READINESS decision

This script does NOT train a model.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE
    / "data"
    / "processed"
    / "windows_apt_v4"
)

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

TEXT_TRAIN = DATA_DIR / "behavior_text_train.csv"
TEXT_TEST = DATA_DIR / "behavior_text_test.csv"

LABEL_TRAIN = DATA_DIR / "labels_train.csv"
LABEL_TEST = DATA_DIR / "labels_test.csv"

META_TRAIN = DATA_DIR / "metadata_train.csv"
META_TEST = DATA_DIR / "metadata_test.csv"

SCHEMA_FILE = DATA_DIR / "schema.json"


# ============================================================
# DISPLAY HELPERS
# ============================================================

WIDTH = 75


def header(title):
    print()
    print("=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def subheader(title):
    print()
    print("=" * WIDTH)
    print(f" {title}")
    print("=" * WIDTH)


def ok(name):
    print(f"[PASS] {name}")


def fail(name):
    print(f"[FAIL] {name}")


def warn(name):
    print(f"[WARN] {name}")


# ============================================================
# FILE CHECK
# ============================================================

header("SmartSOC Windows-APT Model B v4")
print("PRE-TRAINING DATA QUALITY AUDIT")


header("CHECKING REQUIRED FILES")

required_files = [
    STRUCTURED_TRAIN,
    STRUCTURED_TEST,
    TEXT_TRAIN,
    TEXT_TEST,
    LABEL_TRAIN,
    LABEL_TEST,
    META_TRAIN,
    META_TEST,
    SCHEMA_FILE,
]

missing_files = []

for path in required_files:

    if path.exists():
        print(f"[OK] {path}")
    else:
        print(f"[MISSING] {path}")
        missing_files.append(path)


if missing_files:

    print()
    print("Required files are missing.")
    print("Fix the v4 preparation pipeline before continuing.")
    sys.exit(1)


# ============================================================
# LOAD DATASETS
# ============================================================

header("LOADING V4 DATASETS")

try:

    X_train = pd.read_csv(STRUCTURED_TRAIN)
    X_test = pd.read_csv(STRUCTURED_TEST)

    text_train = pd.read_csv(TEXT_TRAIN)
    text_test = pd.read_csv(TEXT_TEST)

    y_train = pd.read_csv(LABEL_TRAIN)
    y_test = pd.read_csv(LABEL_TEST)

    meta_train = pd.read_csv(META_TRAIN)
    meta_test = pd.read_csv(META_TEST)

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

except Exception as exc:

    print()
    print("ERROR while loading v4 datasets:")
    print(exc)
    sys.exit(1)


print(f"Training rows: {len(X_train)}")
print(f"Testing rows : {len(X_test)}")
print(f"Training features: {len(X_train.columns)}")
print(f"Testing features : {len(X_test.columns)}")


# ============================================================
# BASIC CONSISTENCY
# ============================================================

subheader("BASIC CONSISTENCY")

checks = {}


# ------------------------------------------------------------
# Row alignment
# ------------------------------------------------------------

train_lengths = [
    len(X_train),
    len(text_train),
    len(y_train),
    len(meta_train),
]

test_lengths = [
    len(X_test),
    len(text_test),
    len(y_test),
    len(meta_test),
]

checks["row_alignment"] = (
    len(set(train_lengths)) == 1
    and len(set(test_lengths)) == 1
)

if checks["row_alignment"]:
    ok("row_alignment")
else:
    fail("row_alignment")

    print("Training lengths:")
    print(train_lengths)

    print("Testing lengths:")
    print(test_lengths)


# ------------------------------------------------------------
# Feature alignment
# ------------------------------------------------------------

train_columns = list(X_train.columns)
test_columns = list(X_test.columns)

checks["feature_alignment"] = train_columns == test_columns

if checks["feature_alignment"]:
    ok("feature_alignment")
else:
    fail("feature_alignment")

    print("Training-only columns:")
    print(sorted(set(train_columns) - set(test_columns)))

    print("Testing-only columns:")
    print(sorted(set(test_columns) - set(train_columns)))


# ============================================================
# MISSING VALUE AUDIT
# ============================================================

subheader("MISSING VALUE AUDIT")

train_struct_missing = int(X_train.isna().sum().sum())
test_struct_missing = int(X_test.isna().sum().sum())

train_text_missing = int(text_train.isna().sum().sum())
test_text_missing = int(text_test.isna().sum().sum())

print(
    "Training structured missing values:",
    train_struct_missing
)

print(
    "Testing structured missing values:",
    test_struct_missing
)

print(
    "Training text missing values:",
    train_text_missing
)

print(
    "Testing text missing values:",
    test_text_missing
)


checks["structured_missing_values"] = (
    train_struct_missing == 0
    and test_struct_missing == 0
)

if checks["structured_missing_values"]:
    ok("structured_missing_values")
else:
    fail("structured_missing_values")


# ============================================================
# TECHNIQUE EXTRACTION
# ============================================================

def normalize_technique_string(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def extract_techniques(series):

    result = set()

    for value in series:

        for technique in normalize_technique_string(value):
            result.add(technique)

    return result


train_techniques = extract_techniques(y_train["techniques"])
test_techniques = extract_techniques(y_test["techniques"])

all_techniques = sorted(
    train_techniques | test_techniques
)


# ============================================================
# TECHNIQUE COVERAGE
# ============================================================

subheader("TECHNIQUE COVERAGE")

print("Training techniques:", len(train_techniques))
print("Testing techniques :", len(test_techniques))

test_only = sorted(
    test_techniques - train_techniques
)

train_only = sorted(
    train_techniques - test_techniques
)

print()
print("Test-only techniques:")

if test_only:
    print(test_only)
else:
    print("NONE")

print()
print("Train-only techniques:")

if train_only:
    print(train_only)
else:
    print("NONE")


checks["no_test_only_techniques"] = len(test_only) == 0

if checks["no_test_only_techniques"]:
    ok("no_test_only_techniques")
else:
    fail("no_test_only_techniques")


checks["all_techniques_have_training"] = (
    len(test_only) == 0
)

if checks["all_techniques_have_training"]:
    ok("all_techniques_have_training")
else:
    fail("all_techniques_have_training")


# ============================================================
# TRAINING TECHNIQUE DISTRIBUTION
# ============================================================

subheader("TRAINING TECHNIQUE DISTRIBUTION")

train_exploded = (
    y_train["techniques"]
    .apply(normalize_technique_string)
    .explode()
    .dropna()
)

print(
    train_exploded.value_counts().to_string()
)


# ============================================================
# TEST TECHNIQUE DISTRIBUTION
# ============================================================

subheader("TEST TECHNIQUE DISTRIBUTION")

test_exploded = (
    y_test["techniques"]
    .apply(normalize_technique_string)
    .explode()
    .dropna()
)

print(
    test_exploded.value_counts().to_string()
)


# ============================================================
# RARE TECHNIQUE AUDIT
# ============================================================

subheader("RARE TECHNIQUE AUDIT")

train_counts = train_exploded.value_counts()

rare_20 = train_counts[
    train_counts < 20
]

print("Techniques with <20 training examples:")

if len(rare_20):
    print(rare_20.to_string())
else:
    print("NONE")


print()
print("Techniques with <5 training examples:")

rare_5 = train_counts[
    train_counts < 5
]

if len(rare_5):
    print(rare_5.to_string())
else:
    print("NONE")


# ============================================================
# LABEL COMBINATION ANALYSIS
# ============================================================

subheader("LABEL COMBINATION ANALYSIS")

train_combinations = (
    y_train["techniques"]
    .fillna("")
    .astype(str)
    .str.strip()
)

test_combinations = (
    y_test["techniques"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(
    "Unique training combinations:",
    train_combinations.nunique()
)

print(
    "Unique test combinations:",
    test_combinations.nunique()
)


print()
print("Top training combinations:")

print(
    train_combinations
    .value_counts()
    .head(30)
    .to_string()
)


print()
print("Top testing combinations:")

print(
    test_combinations
    .value_counts()
    .head(30)
    .to_string()
)


# ============================================================
# MULTI-LABEL ANALYSIS
# ============================================================

subheader("MULTI-LABEL ANALYSIS")

train_label_count = (
    train_combinations
    .apply(
        lambda x: len(normalize_technique_string(x))
    )
)

test_label_count = (
    test_combinations
    .apply(
        lambda x: len(normalize_technique_string(x))
    )
)

train_single = int(
    (train_label_count == 1).sum()
)

train_multi = int(
    (train_label_count > 1).sum()
)

test_single = int(
    (test_label_count == 1).sum()
)

test_multi = int(
    (test_label_count > 1).sum()
)

print("Training single-label:", train_single)
print("Training multi-label :", train_multi)

print("Testing single-label:", test_single)
print("Testing multi-label :", test_multi)


# ============================================================
# BEHAVIOR TEXT AUDIT
# ============================================================

subheader("BEHAVIOR TEXT AUDIT")


def find_text_column(df):

    candidates = [
        "behavior_text",
        "text",
        "behavior",
        "full_log",
        "full-log",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    # Fallback:
    # Use first object/string column.

    object_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if object_columns:
        return object_columns[0]

    return None


train_text_column = find_text_column(text_train)
test_text_column = find_text_column(text_test)

print("Training text column:", train_text_column)
print("Testing text column :", test_text_column)


if train_text_column and test_text_column:

    train_text_series = (
        text_train[train_text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    test_text_series = (
        text_test[test_text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_train = int(
        (train_text_series == "").sum()
    )

    empty_test = int(
        (test_text_series == "").sum()
    )

    unique_train_text = int(
        train_text_series.nunique()
    )

    unique_test_text = int(
        test_text_series.nunique()
    )

    print(
        "Empty training text:",
        empty_train
    )

    print(
        "Empty testing text:",
        empty_test
    )

    print(
        "Unique training texts:",
        unique_train_text
    )

    print(
        "Unique testing texts:",
        unique_test_text
    )

else:

    warn("Could not identify behavior text column")

    empty_train = 0
    empty_test = 0


# ============================================================
# POWERSHELL / PROCESS AUDIT
# ============================================================

subheader("POWERSHELL / PROCESS AUDIT")


def column_contains(df, column, pattern):

    if column not in df.columns:
        return pd.Series(
            False,
            index=df.index
        )

    return (
        df[column]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            pattern,
            regex=False,
            na=False
        )
    )


powershell_train_mask = (
    column_contains(
        X_train,
        "process_name",
        "powershell"
    )
    |
    column_contains(
        X_train,
        "parent_process_name",
        "powershell"
    )
    |
    column_contains(
        X_train,
        "behavior_text",
        "powershell"
    )
)

powershell_test_mask = (
    column_contains(
        X_test,
        "process_name",
        "powershell"
    )
    |
    column_contains(
        X_test,
        "parent_process_name",
        "powershell"
    )
    |
    column_contains(
        text_test,
        test_text_column,
        "powershell"
    )
)

print(
    "PowerShell training events:",
    int(powershell_train_mask.sum())
)

print(
    "PowerShell testing events:",
    int(powershell_test_mask.sum())
)


print()
print("PowerShell training labels:")

if powershell_train_mask.any():

    print(
        y_train.loc[
            powershell_train_mask,
            "techniques"
        ]
        .apply(normalize_technique_string)
        .explode()
        .dropna()
        .value_counts()
        .to_string()
    )

else:

    print("NONE")


# ============================================================
# ADMINISTRATIVE POWERSHELL AUDIT
# ============================================================

subheader("ADMINISTRATIVE POWERSHELL AUDIT")


def safe_bool_column(df, column):

    if column not in df.columns:

        return pd.Series(
            False,
            index=df.index
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .fillna(0)
        .astype(float)
        > 0
    )


admin_mask = (
    powershell_train_mask
    &
    (
        safe_bool_column(
            X_train,
            "has_user"
        )
        |
        safe_bool_column(
            X_train,
            "is_admin_user"
        )
    )
    &
    ~safe_bool_column(
        X_train,
        "has_encoded_command"
    )
    &
    ~safe_bool_column(
        X_train,
        "has_download_behavior"
    )
)

print(
    "Administrative-looking PowerShell:",
    int(admin_mask.sum())
)

print()

if admin_mask.any():

    print("Labels:")

    print(
        y_train.loc[
            admin_mask,
            "techniques"
        ]
        .apply(normalize_technique_string)
        .explode()
        .dropna()
        .value_counts()
        .to_string()
    )

else:

    print("NONE")


# ============================================================
# HIGH-RISK BEHAVIOR SIGNAL AUDIT
# ============================================================

subheader("HIGH-RISK BEHAVIOR SIGNAL AUDIT")


behavior_columns = [
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_mshta",
    "uses_rundll32",
    "uses_regsvr32",
    "uses_vssadmin",
]


for column in behavior_columns:

    if column not in X_train.columns:

        print(
            f"{column:<30} MISSING"
        )

        continue

    train_count = int(
        safe_bool_column(
            X_train,
            column
        ).sum()
    )

    test_count = int(
        safe_bool_column(
            X_test,
            column
        ).sum()
    )

    print(
        f"{column:<30}"
        f"train={train_count:6d} "
        f"test={test_count:6d}"
    )


# ============================================================
# TECHNIQUE / FEATURE ASSOCIATION AUDIT
# ============================================================

subheader("TECHNIQUE / FEATURE ASSOCIATION AUDIT")


association_features = [
    "uses_powershell",
    "uses_cmd",
    "uses_bitsadmin",
    "uses_mshta",
    "uses_rundll32",
    "uses_regsvr32",
    "uses_vssadmin",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
]


for technique in sorted(train_techniques):

    mask = y_train["techniques"].apply(
        lambda x: technique in normalize_technique_string(x)
    )

    count = int(mask.sum())

    print()
    print(
        f"{technique} (events={count})"
    )

    if count == 0:
        continue

    for feature in association_features:

        if feature not in X_train.columns:
            continue

        values = pd.to_numeric(
            X_train.loc[
                mask,
                feature
            ],
            errors="coerce"
        ).fillna(0)

        if len(values) == 0:
            continue

        ratio = float(
            (values > 0).mean()
        )

        if ratio > 0:

            print(
                f"  {feature:<30}"
                f"{ratio:.3f}"
            )


# ============================================================
# ROBUST TIMESTAMP PARSER
# ============================================================

subheader("TIMESTAMP ANALYSIS")


def parse_wazuh_timestamp(series):

    """
    Windows-APT timestamp format observed in the dataset:

        Dec 1, 2024 @ 06:24:28.026

    We explicitly remove '@' and parse using the known format.
    """

    s = (
        series
        .astype(str)
        .str.strip()
    )

    # Remove @ while preserving the surrounding whitespace.
    s = s.str.replace(
        "@",
        "",
        regex=False
    )

    # Normalize repeated whitespace.
    s = s.str.replace(
        r"\s+",
        " ",
        regex=True
    )

    parsed = pd.to_datetime(
        s,
        format="%b %d, %Y %H:%M:%S.%f",
        errors="coerce"
    )

    return parsed


# ------------------------------------------------------------
# Identify timestamp column
# ------------------------------------------------------------

def find_timestamp_column(df):

    candidates = [
        "timestamp",
        "Timestamp",
        "time",
        "@timestamp",
        "event_time",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    return None


train_timestamp_column = find_timestamp_column(
    meta_train
)

test_timestamp_column = find_timestamp_column(
    meta_test
)

print(
    "Training timestamp column:",
    train_timestamp_column
)

print(
    "Testing timestamp column:",
    test_timestamp_column
)


if (
    train_timestamp_column is None
    or test_timestamp_column is None
):

    fail("timestamp_column_available")

    checks["valid_timestamps"] = False
    checks["temporal_separation"] = False

else:

    train_timestamp = parse_wazuh_timestamp(
        meta_train[
            train_timestamp_column
        ]
    )

    test_timestamp = parse_wazuh_timestamp(
        meta_test[
            test_timestamp_column
        ]
    )

    invalid_train = int(
        train_timestamp.isna().sum()
    )

    invalid_test = int(
        test_timestamp.isna().sum()
    )

    print()
    print(
        "Invalid training timestamps:",
        invalid_train
    )

    print(
        "Invalid testing timestamps:",
        invalid_test
    )

    checks["valid_timestamps"] = (
        invalid_train == 0
        and invalid_test == 0
    )

    if checks["valid_timestamps"]:
        ok("valid_timestamps")
    else:
        fail("valid_timestamps")

    if train_timestamp.notna().any():

        train_min = train_timestamp.min()
        train_max = train_timestamp.max()

    else:

        train_min = None
        train_max = None

    if test_timestamp.notna().any():

        test_min = test_timestamp.min()
        test_max = test_timestamp.max()

    else:

        test_min = None
        test_max = None

    print()

    print(
        "Training:",
        train_min,
        "->",
        train_max
    )

    print(
        "Testing :",
        test_min,
        "->",
        test_max
    )

    # --------------------------------------------------------
    # Temporal separation
    # --------------------------------------------------------

    if checks["valid_timestamps"]:

        checks["temporal_separation"] = (
            train_max < test_min
        )

    else:

        checks["temporal_separation"] = False

    if checks["temporal_separation"]:

        ok("temporal_separation")

    else:

        fail("temporal_separation")

        if (
            train_max is not None
            and test_min is not None
        ):

            print()
            print(
                "Training maximum timestamp:",
                train_max
            )

            print(
                "Testing minimum timestamp:",
                test_min
            )

            print()
            print(
                "WARNING:"
            )

            print(
                "Training and testing timestamps overlap."
            )

            print(
                "Do NOT train the v4 model until the split "
                "is corrected."
            )


# ============================================================
# TEMPORAL OVERLAP DETAILS
# ============================================================

if (
    "train_timestamp" in locals()
    and "test_timestamp" in locals()
    and checks.get("valid_timestamps", False)
):

    print()
    print(
        "Temporal boundary:"
    )

    print(
        "Train max:",
        train_timestamp.max()
    )

    print(
        "Test min :",
        test_timestamp.min()
    )

    if train_timestamp.max() < test_timestamp.min():

        print(
            "[OK] No timestamp overlap."
        )

    else:

        overlap_start = test_timestamp.min()
        overlap_end = train_timestamp.max()

        print(
            "[WARNING] Temporal overlap detected."
        )

        print(
            "Overlap interval:",
            overlap_start,
            "->",
            overlap_end
        )


# ============================================================
# AGENT DISTRIBUTION
# ============================================================

subheader("AGENT DISTRIBUTION")


def print_distribution(df, title):

    if "agent" not in df.columns:

        print(
            f"{title}: agent column unavailable"
        )

        return

    print()
    print(title)

    print(
        df["agent"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
        .to_string()
    )


print_distribution(
    meta_train,
    "Training agents:"
)

print_distribution(
    meta_test,
    "Testing agents:"
)


# ============================================================
# EVENT ID DISTRIBUTION
# ============================================================

subheader("EVENT ID DISTRIBUTION")


def print_event_distribution(df, title):

    if "event_id" not in df.columns:

        print(
            f"{title}: event_id column unavailable"
        )

        return

    print()
    print(title)

    print(
        df["event_id"]
        .value_counts(dropna=False)
        .head(30)
        .to_string()
    )


print_event_distribution(
    X_train,
    "Training:"
)

print_event_distribution(
    X_test,
    "Testing:"
)


# ============================================================
# STRUCTURED FEATURE CARDINALITY
# ============================================================

subheader("STRUCTURED FEATURE CARDINALITY")


cardinality = (
    X_train
    .nunique(dropna=False)
    .sort_values()
)

print(
    cardinality.to_string()
)


# ============================================================
# DUPLICATE AUDIT
# ============================================================

subheader("DUPLICATE AUDIT")


train_struct_duplicates = int(
    X_train.duplicated().sum()
)

test_struct_duplicates = int(
    X_test.duplicated().sum()
)

print(
    "Training structured duplicates:",
    train_struct_duplicates
)

print(
    "Testing structured duplicates:",
    test_struct_duplicates
)


# ============================================================
# EXACT TRAIN/TEST FEATURE OVERLAP
# ============================================================

subheader("TRAIN/TEST STRUCTURED OVERLAP AUDIT")


# Hash rows so this is memory-efficient compared with
# converting all rows into Python tuples.

try:

    train_hashes = pd.util.hash_pandas_object(
        X_train,
        index=False
    )

    test_hashes = pd.util.hash_pandas_object(
        X_test,
        index=False
    )

    train_hash_set = set(
        train_hashes.tolist()
    )

    test_hash_set = set(
        test_hashes.tolist()
    )

    overlap_count = len(
        train_hash_set & test_hash_set
    )

    print(
        "Unique training structured signatures:",
        len(train_hash_set)
    )

    print(
        "Unique testing structured signatures:",
        len(test_hash_set)
    )

    print(
        "Exact structured signatures appearing "
        "in both train and test:",
        overlap_count
    )

except Exception as exc:

    print(
        "Could not calculate structured overlap:"
    )

    print(exc)


# ============================================================
# TRAINING READINESS
# ============================================================

subheader("V4 TRAINING READINESS")


# ------------------------------------------------------------
# Reasonable training size
# ------------------------------------------------------------

checks["reasonable_training_size"] = (
    len(X_train) >= 10000
)

if checks["reasonable_training_size"]:
    ok("reasonable_training_size")
else:
    fail("reasonable_training_size")


# ------------------------------------------------------------
# Test size
# ------------------------------------------------------------

checks["reasonable_test_size"] = (
    len(X_test) >= 1000
)

if checks["reasonable_test_size"]:
    ok("reasonable_test_size")
else:
    fail("reasonable_test_size")


# ------------------------------------------------------------
# Labels exist
# ------------------------------------------------------------

checks["labels_available"] = (
    len(train_techniques) > 0
    and len(test_techniques) > 0
)

if checks["labels_available"]:
    ok("labels_available")
else:
    fail("labels_available")


# ------------------------------------------------------------
# No catastrophic missing structured data
# ------------------------------------------------------------

checks["structured_data_complete"] = (
    train_struct_missing == 0
    and test_struct_missing == 0
)

if checks["structured_data_complete"]:
    ok("structured_data_complete")
else:
    fail("structured_data_complete")


# ------------------------------------------------------------
# Final training readiness
# ------------------------------------------------------------

critical_checks = [
    "row_alignment",
    "feature_alignment",
    "no_test_only_techniques",
    "all_techniques_have_training",
    "valid_timestamps",
    "temporal_separation",
    "reasonable_training_size",
    "reasonable_test_size",
    "labels_available",
    "structured_data_complete",
]


all_pass = all(
    checks.get(name, False)
    for name in critical_checks
)


print()

if all_pass:

    print(
        "[PASS] ALL CRITICAL TRAINING CHECKS"
    )

    print()
    print(
        "V4 DATASET IS READY FOR TRAINING."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the v4 training pipeline."
    )

else:

    print(
        "[FAIL] ONE OR MORE CRITICAL CHECKS"
    )

    print()
    print(
        "DATASET NOT READY FOR TRAINING"
    )

    print(
        "Fix the failed checks first."
    )

    print()
    print(
        "Failed checks:"
    )

    for name in critical_checks:

        if not checks.get(name, False):

            print(
                f"  [FAIL] {name}"
            )


# ============================================================
# SUMMARY
# ============================================================

subheader("AUDIT SUMMARY")

print(
    "Training events:",
    len(X_train)
)

print(
    "Testing events:",
    len(X_test)
)

print(
    "Structured features:",
    len(X_train.columns)
)

print(
    "MITRE techniques:",
    len(all_techniques)
)

print(
    "Training single-label:",
    train_single
)

print(
    "Training multi-label:",
    train_multi
)

print(
    "Testing single-label:",
    test_single
)

print(
    "Testing multi-label:",
    test_multi
)

print(
    "Rare techniques (<20):",
    len(rare_20)
)

print(
    "Very rare techniques (<5):",
    len(rare_5)
)

print(
    "Training text missing:",
    empty_train
)

print(
    "Testing text missing:",
    empty_test
)

print(
    "Temporal separation:",
    checks.get(
        "temporal_separation",
        False
    )
)

print(
    "Overall training readiness:",
    all_pass
)

print()
print("=" * WIDTH)
print(" SmartSOC Windows-APT v4 Audit Complete")
print("=" * WIDTH)