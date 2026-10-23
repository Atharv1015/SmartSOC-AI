import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SMARTSOC
# FINAL DATASET CLEANING, AUDIT & FREEZE GATE
# ============================================================
#
# PURPOSE
# -------
# Formally audit the already validated windows_apt_v4_1 dataset
# and create an exact, hash-preserving FINAL copy.
#
# IMPORTANT
# ---------
# This script is READ-ONLY with respect to the source dataset.
#
# It does NOT:
#   - delete rows
#   - alter labels
#   - alter features
#   - alter train/test membership
#   - retrain the model
#   - modify production thresholds
#   - modify production model artifacts
#
# It creates:
#
# data/final/windows_apt_v4_1/
#
# as an exact byte-for-byte copy of:
#
# data/processed/windows_apt_v4_1/
#
# and produces audit reports.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

FINAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "final_dataset_cleaning"
)


FILES = [
    "structured_train.csv",
    "structured_test.csv",
    "labels_train.csv",
    "labels_test.csv",
    "metadata_train.csv",
    "metadata_test.csv",
    "behavior_text_train.csv",
    "behavior_text_test.csv",
]


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_STRUCTURED_FEATURES = 62
EXPECTED_TECHNIQUES = 45
EXPECTED_TFIDF_FEATURES = 6699
EXPECTED_FINAL_FEATURES = 6919

TIMESTAMP_FORMAT = (
    "%b %d, %Y @ %H:%M:%S.%f"
)


# ============================================================
# DIRECTORIES
# ============================================================

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"[FAIL] {message}")
    raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value):
    """
    Convert NumPy/Pandas scalar types into
    native Python types for JSON serialization.
    """

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(v)
            for v in value
        ]

    return value


def normalize_label_string(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def parse_label_set(value):

    raw = normalize_label_string(value)

    if not raw:
        return frozenset()

    normalized = (
        raw
        .replace(";", ",")
        .replace("|", ",")
    )

    tokens = [
        token.strip()
        for token in normalized.split(",")
        if token.strip()
    ]

    return frozenset(tokens)


def build_full_representation(
    structured_df,
    behavior_df
):

    representation_df = structured_df.copy()

    behavior_values = (
        behavior_df[
            "behavior_text_normalized"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    representation_df[
        "__behavior_text__"
    ] = behavior_values.values

    # Use a stable row representation.
    #
    # Keep column order exactly as supplied.
    return representation_df.astype(
        str
    ).agg(
        "||".join,
        axis=1
    )


def build_label_aware_representation_map(
    representation_series,
    labels_df
):

    result = {}

    for idx, representation in enumerate(
        representation_series
    ):

        label_set = parse_label_set(
            labels_df.iloc[idx]["techniques"]
        )

        if representation not in result:
            result[representation] = set()

        result[representation].add(
            tuple(sorted(label_set))
        )

    return result


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("SMARTSOC")
print("FINAL DATASET CLEANING, AUDIT & FREEZE GATE")
print("=" * 72)


# ============================================================
# 1. SOURCE VALIDATION
# ============================================================

print("\nSOURCE DATASET")

if not SOURCE_DIR.exists():
    fail(
        f"Source dataset does not exist:\n{SOURCE_DIR}"
    )

source_paths = {}

for filename in FILES:

    path = SOURCE_DIR / filename

    if not path.exists():
        fail(
            f"Missing source file:\n{path}"
        )

    source_paths[filename] = path

    print(
        f"[PASS] {filename}"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLOADING DATA")

data = {}

for filename, path in source_paths.items():

    df = pd.read_csv(path)

    data[filename] = df

    print(
        f"{filename:30s} {df.shape}"
    )


# ============================================================
# 3. REFERENCES
# ============================================================

structured_train = data[
    "structured_train.csv"
]

structured_test = data[
    "structured_test.csv"
]

labels_train = data[
    "labels_train.csv"
]

labels_test = data[
    "labels_test.csv"
]

metadata_train = data[
    "metadata_train.csv"
]

metadata_test = data[
    "metadata_test.csv"
]

behavior_train = data[
    "behavior_text_train.csv"
]

behavior_test = data[
    "behavior_text_test.csv"
]


# ============================================================
# 4. ROW ALIGNMENT
# ============================================================

print("\nROW ALIGNMENT")

train_lengths = {
    "structured":
        len(structured_train),

    "labels":
        len(labels_train),

    "metadata":
        len(metadata_train),

    "behavior":
        len(behavior_train),
}

test_lengths = {
    "structured":
        len(structured_test),

    "labels":
        len(labels_test),

    "metadata":
        len(metadata_test),

    "behavior":
        len(behavior_test),
}

print(
    "TRAIN:",
    train_lengths
)

print(
    "TEST :",
    test_lengths
)

train_alignment = (
    len(set(train_lengths.values()))
    == 1
)

test_alignment = (
    len(set(test_lengths.values()))
    == 1
)

if not train_alignment:
    fail(
        "Training files are not row aligned"
    )

if not test_alignment:
    fail(
        "Test files are not row aligned"
    )

print(
    "[PASS] Train row alignment"
)

print(
    "[PASS] Test row alignment"
)


# ============================================================
# 5. EXPECTED ROW COUNTS
# ============================================================

print("\nDATASET SIZE")

train_rows = len(
    structured_train
)

test_rows = len(
    structured_test
)

total_rows = (
    train_rows
    +
    test_rows
)

print(
    f"Train rows : {train_rows}"
)

print(
    f"Test rows  : {test_rows}"
)

print(
    f"Total rows : {total_rows}"
)

if train_rows != 18677:
    fail(
        "Unexpected training row count"
    )

if test_rows != 11707:
    fail(
        "Unexpected test row count"
    )

print(
    "[PASS] Expected dataset sizes"
)


# ============================================================
# 6. SCHEMA AUDIT
# ============================================================

print("\nSCHEMA AUDIT")

train_columns = list(
    structured_train.columns
)

test_columns = list(
    structured_test.columns
)

if (
    len(train_columns)
    != EXPECTED_STRUCTURED_FEATURES
):
    fail(
        "Training structured dataset "
        "does not contain 62 features"
    )

if (
    len(test_columns)
    != EXPECTED_STRUCTURED_FEATURES
):
    fail(
        "Test structured dataset "
        "does not contain 62 features"
    )

if train_columns != test_columns:
    fail(
        "Train/test structured schemas differ"
    )

if (
    "behavior_text_normalized"
    not in behavior_train.columns
):
    fail(
        "Training behavior file missing "
        "behavior_text_normalized"
    )

if (
    "behavior_text_normalized"
    not in behavior_test.columns
):
    fail(
        "Test behavior file missing "
        "behavior_text_normalized"
    )

if "techniques" not in labels_train.columns:
    fail(
        "Training labels missing techniques"
    )

if "techniques" not in labels_test.columns:
    fail(
        "Test labels missing techniques"
    )

if "timestamp" not in metadata_train.columns:
    fail(
        "Training metadata missing timestamp"
    )

if "timestamp" not in metadata_test.columns:
    fail(
        "Test metadata missing timestamp"
    )

print(
    "[PASS] Structured feature count: 62"
)

print(
    "[PASS] Train/test schema parity"
)

print(
    "[PASS] Required columns present"
)


# ============================================================
# 7. MISSING VALUE AUDIT
# ============================================================

print("\nMISSING VALUE AUDIT")

missing_rows = []

for filename, df in data.items():

    missing_cells = int(
        df.isna()
        .sum()
        .sum()
    )

    missing_rows.append({
        "file": filename,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells":
            missing_cells,
    })

    print(
        f"{filename:30s} "
        f"missing={missing_cells}"
    )

missing_df = pd.DataFrame(
    missing_rows
)

print(
    "[PASS] Missing-value audit completed"
)

print(
    "NOTE: missing behavior text is "
    "documented, not silently deleted."
)


# ============================================================
# 8. NUMERIC VALIDITY
# ============================================================

print("\nNUMERIC VALIDITY")

numeric_columns = (
    structured_train
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)

numeric_rows = []

for column in numeric_columns:

    train_numeric = pd.to_numeric(
        structured_train[column],
        errors="coerce"
    )

    test_numeric = pd.to_numeric(
        structured_test[column],
        errors="coerce"
    )

    train_inf = int(
        np.isinf(
            train_numeric.fillna(0)
        ).sum()
    )

    test_inf = int(
        np.isinf(
            test_numeric.fillna(0)
        ).sum()
    )

    train_nan = int(
        train_numeric.isna().sum()
    )

    test_nan = int(
        test_numeric.isna().sum()
    )

    numeric_rows.append({
        "feature": column,
        "train_nan": train_nan,
        "test_nan": test_nan,
        "train_infinite":
            train_inf,
        "test_infinite":
            test_inf,
    })

numeric_df = pd.DataFrame(
    numeric_rows
)

if (
    numeric_df[
        "train_infinite"
    ].sum()
    > 0
):
    fail(
        "Infinite numeric values in training data"
    )

if (
    numeric_df[
        "test_infinite"
    ].sum()
    > 0
):
    fail(
        "Infinite numeric values in test data"
    )

print(
    f"[PASS] Numeric columns audited: "
    f"{len(numeric_columns)}"
)

print(
    "[PASS] No infinite numeric values"
)


# ============================================================
# 9. CONSTANT FEATURE AUDIT
# ============================================================

print("\nCONSTANT FEATURE AUDIT")

constant_rows = []

for column in train_columns:

    train_unique = int(
        structured_train[
            column
        ].nunique(
            dropna=False
        )
    )

    test_unique = int(
        structured_test[
            column
        ].nunique(
            dropna=False
        )
    )

    constant_rows.append({
        "feature": column,
        "train_unique_values":
            train_unique,
        "test_unique_values":
            test_unique,
        "constant_in_train":
            train_unique <= 1,
        "constant_in_test":
            test_unique <= 1,
    })

constant_df = pd.DataFrame(
    constant_rows
)

constant_train_features = (
    constant_df[
        constant_df[
            "constant_in_train"
        ]
    ]["feature"]
    .tolist()
)

constant_test_features = (
    constant_df[
        constant_df[
            "constant_in_test"
        ]
    ]["feature"]
    .tolist()
)

print(
    f"Constant training features: "
    f"{len(constant_train_features)}"
)

if constant_train_features:

    for feature in constant_train_features:
        print(
            f"  {feature}"
        )

else:

    print(
        "[PASS] No constant training features"
    )

print(
    "ACTION: constants are documented; "
    "production schema is not altered."
)


# ============================================================
# 10. EXACT DUPLICATE AUDIT
# ============================================================

print("\nEXACT DUPLICATE INVESTIGATION")

duplicate_rows = []

for filename, df in data.items():

    duplicated = int(
        df.duplicated(
            keep=False
        ).sum()
    )

    unique_rows = int(
        df.drop_duplicates().shape[0]
    )

    duplicate_rows.append({
        "file": filename,
        "rows": int(len(df)),
        "exact_duplicate_rows":
            duplicated,
        "unique_rows":
            unique_rows,
    })

    print(
        f"{filename:30s} "
        f"duplicates={duplicated}"
    )

duplicate_df = pd.DataFrame(
    duplicate_rows
)

print(
    "[PASS] Exact duplicate audit completed"
)

print(
    "ACTION: exact duplicates are "
    "documented, not automatically deleted."
)


# ============================================================
# 11. FULL REPRESENTATION
# ============================================================

print("\nFULL REPRESENTATION AUDIT")

train_representation = (
    build_full_representation(
        structured_train,
        behavior_train
    )
)

test_representation = (
    build_full_representation(
        structured_test,
        behavior_test
    )
)

train_counts = (
    train_representation
    .value_counts()
)

test_counts = (
    test_representation
    .value_counts()
)

train_repeated_rows = int(
    train_counts[
        train_counts > 1
    ].sum()
)

test_repeated_rows = int(
    test_counts[
        test_counts > 1
    ].sum()
)

train_unique_representations = int(
    train_representation.nunique()
)

test_unique_representations = int(
    test_representation.nunique()
)

cross_split_overlap_set = (
    set(
        train_representation.unique()
    )
    &
    set(
        test_representation.unique()
    )
)

cross_split_overlap = len(
    cross_split_overlap_set
)

print(
    f"Train unique representations : "
    f"{train_unique_representations}"
)

print(
    f"Test unique representations  : "
    f"{test_unique_representations}"
)

print(
    f"Repeated train rows           : "
    f"{train_repeated_rows}"
)

print(
    f"Repeated test rows            : "
    f"{test_repeated_rows}"
)

print(
    f"Cross-split identical reps    : "
    f"{cross_split_overlap}"
)


# ============================================================
# 12. CROSS-SPLIT LABEL CONFLICT ANALYSIS
# ============================================================

print("\nCROSS-SPLIT LABEL CONFLICT ANALYSIS")

train_rep_labels = (
    build_label_aware_representation_map(
        train_representation,
        labels_train
    )
)

test_rep_labels = (
    build_label_aware_representation_map(
        test_representation,
        labels_test
    )
)

cross_split_same_label = 0
cross_split_conflict = 0

cross_split_details = []

for representation in (
    cross_split_overlap_set
):

    train_labels_for_rep = (
        train_rep_labels.get(
            representation,
            set()
        )
    )

    test_labels_for_rep = (
        test_rep_labels.get(
            representation,
            set()
        )
    )

    train_label_union = set()

    for label_set in train_labels_for_rep:
        train_label_union.update(
            label_set
        )

    test_label_union = set()

    for label_set in test_labels_for_rep:
        test_label_union.update(
            label_set
        )

    label_conflict = (
        train_label_union
        !=
        test_label_union
    )

    if label_conflict:
        cross_split_conflict += 1
    else:
        cross_split_same_label += 1

    cross_split_details.append({
        "representation_hash":
            hashlib.sha256(
                representation.encode(
                    "utf-8"
                )
            ).hexdigest(),

        "train_label_sets":
            [
                list(x)
                for x in
                sorted(
                    train_labels_for_rep
                )
            ],

        "test_label_sets":
            [
                list(x)
                for x in
                sorted(
                    test_labels_for_rep
                )
            ],

        "label_conflict":
            bool(label_conflict),
    })

print(
    f"Cross-split identical representations: "
    f"{cross_split_overlap}"
)

print(
    f"Same-label cross-split repetitions    : "
    f"{cross_split_same_label}"
)

print(
    f"Different-label cross-split conflicts : "
    f"{cross_split_conflict}"
)

if cross_split_conflict > 0:

    print(
        "[REVIEW] Cross-split label conflicts detected"
    )

else:

    print(
        "[PASS] No cross-split label conflicts"
    )


# ============================================================
# 13. WITHIN-SPLIT LABEL AMBIGUITY
# ============================================================

print("\nWITHIN-SPLIT LABEL AMBIGUITY")

def count_ambiguous_representations(
    representation_series,
    labels_df
):

    mapping = (
        build_label_aware_representation_map(
            representation_series,
            labels_df
        )
    )

    ambiguous_representations = 0
    affected_rows = 0

    counts = (
        representation_series
        .value_counts()
    )

    for representation, label_sets in mapping.items():

        if len(label_sets) > 1:

            ambiguous_representations += 1

            affected_rows += int(
                counts.get(
                    representation,
                    0
                )
            )

    return (
        ambiguous_representations,
        affected_rows,
    )


train_ambiguous_reps, train_ambiguous_rows = (
    count_ambiguous_representations(
        train_representation,
        labels_train
    )
)

test_ambiguous_reps, test_ambiguous_rows = (
    count_ambiguous_representations(
        test_representation,
        labels_test
    )
)

print(
    f"Train ambiguous representations: "
    f"{train_ambiguous_reps}"
)

print(
    f"Train affected rows: "
    f"{train_ambiguous_rows}"
)

print(
    f"Test ambiguous representations: "
    f"{test_ambiguous_reps}"
)

print(
    f"Test affected rows: "
    f"{test_ambiguous_rows}"
)

print(
    "[PASS] Within-split ambiguity audit completed"
)


# ============================================================
# 14. LABEL AUDIT
# ============================================================

print("\nLABEL AUDIT")

train_labels_flat = []
test_labels_flat = []

for value in labels_train[
    "techniques"
].fillna(""):

    train_labels_flat.extend(
        list(
            parse_label_set(value)
        )
    )

for value in labels_test[
    "techniques"
].fillna(""):

    test_labels_flat.extend(
        list(
            parse_label_set(value)
        )
    )

all_techniques = sorted(
    set(
        train_labels_flat
        +
        test_labels_flat
    )
)

print(
    f"Unique MITRE techniques: "
    f"{len(all_techniques)}"
)

if len(all_techniques) != EXPECTED_TECHNIQUES:
    fail(
        f"Expected {EXPECTED_TECHNIQUES} "
        f"techniques, found "
        f"{len(all_techniques)}"
    )

print(
    "[PASS] 45 MITRE techniques"
)

empty_train_labels = int(
    labels_train[
        "techniques"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

empty_test_labels = int(
    labels_test[
        "techniques"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(
    f"Empty train label rows: "
    f"{empty_train_labels}"
)

print(
    f"Empty test label rows : "
    f"{empty_test_labels}"
)


# ============================================================
# 15. TIMESTAMP AUDIT
# ============================================================

print("\nTIMESTAMP AUDIT")

train_timestamps = pd.to_datetime(
    metadata_train[
        "timestamp"
    ].astype(str).str.strip(),
    format=TIMESTAMP_FORMAT,
    errors="coerce",
)

test_timestamps = pd.to_datetime(
    metadata_test[
        "timestamp"
    ].astype(str).str.strip(),
    format=TIMESTAMP_FORMAT,
    errors="coerce",
)

invalid_train_timestamps = int(
    train_timestamps.isna().sum()
)

invalid_test_timestamps = int(
    test_timestamps.isna().sum()
)

print(
    f"Invalid train timestamps: "
    f"{invalid_train_timestamps}"
)

print(
    f"Invalid test timestamps : "
    f"{invalid_test_timestamps}"
)

if invalid_train_timestamps > 0:
    fail(
        "Invalid training timestamps"
    )

if invalid_test_timestamps > 0:
    fail(
        "Invalid test timestamps"
    )

train_latest = (
    train_timestamps.max()
)

test_earliest = (
    test_timestamps.min()
)

temporal_gap = (
    test_earliest
    -
    train_latest
)

print(
    f"Train latest : "
    f"{train_latest}"
)

print(
    f"Test earliest: "
    f"{test_earliest}"
)

print(
    f"Temporal gap : "
    f"{temporal_gap}"
)

if train_latest >= test_earliest:
    fail(
        "Temporal train/test overlap detected"
    )

print(
    "[PASS] Temporal separation"
)


# ============================================================
# 16. IDENTIFIER AUDIT
# ============================================================

print("\nIDENTIFIER / METADATA AUDIT")

identifier_candidates = [
    "event_id",
]

identifier_rows = []

for column in identifier_candidates:

    if column not in structured_train.columns:
        continue

    train_unique = int(
        structured_train[
            column
        ].nunique(
            dropna=False
        )
    )

    test_unique = int(
        structured_test[
            column
        ].nunique(
            dropna=False
        )
    )

    identifier_rows.append({
        "feature": column,
        "train_unique":
            train_unique,
        "test_unique":
            test_unique,
    })

    print(
        f"{column}: "
        f"train_unique={train_unique}, "
        f"test_unique={test_unique}"
    )

print(
    "[PASS] Identifier audit completed"
)

print(
    "NOTE: identifier fields are documented "
    "as metadata and are not interpreted "
    "as causal SOC evidence."
)


# ============================================================
# 17. BEHAVIOR TEXT AUDIT
# ============================================================

print("\nBEHAVIOR TEXT AUDIT")

train_behavior_missing = int(
    behavior_train[
        "behavior_text_normalized"
    ]
    .isna()
    .sum()
)

test_behavior_missing = int(
    behavior_test[
        "behavior_text_normalized"
    ]
    .isna()
    .sum()
)

train_behavior_empty = int(
    behavior_train[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

test_behavior_empty = int(
    behavior_test[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()
)

print(
    f"Train missing behavior values: "
    f"{train_behavior_missing}"
)

print(
    f"Test missing behavior values : "
    f"{test_behavior_missing}"
)

print(
    f"Train empty behavior values  : "
    f"{train_behavior_empty}"
)

print(
    f"Test empty behavior values   : "
    f"{test_behavior_empty}"
)

print(
    "[PASS] Behavior text audit completed"
)


# ============================================================
# 18. SOURCE HASHES
# ============================================================

print("\nSOURCE SHA-256 HASHES")

source_hashes = {}

for filename, path in source_paths.items():

    digest = sha256_file(path)

    source_hashes[filename] = digest

    print(
        f"{filename:30s} "
        f"{digest[:16]}..."
    )


# ============================================================
# 19. CREATE FINAL COPY
# ============================================================

print("\nCREATING FINAL DATASET COPY")

for filename, source_path in source_paths.items():

    destination = (
        FINAL_DIR / filename
    )

    shutil.copy2(
        source_path,
        destination
    )

    print(
        f"[PASS] {filename}"
    )


# ============================================================
# 20. VERIFY FINAL COPY
# ============================================================

print("\nFINAL COPY VERIFICATION")

final_hashes = {}

for filename in FILES:

    source_path = (
        SOURCE_DIR / filename
    )

    final_path = (
        FINAL_DIR / filename
    )

    if not final_path.exists():
        fail(
            f"Final file missing: "
            f"{final_path}"
        )

    source_hash = sha256_file(
        source_path
    )

    final_hash = sha256_file(
        final_path
    )

    final_hashes[filename] = (
        final_hash
    )

    if source_hash != final_hash:
        fail(
            f"Hash mismatch: "
            f"{filename}"
        )

    print(
        f"[PASS] {filename}"
    )

lineage_preserved = (
    source_hashes
    ==
    final_hashes
)

if not lineage_preserved:
    fail(
        "Final dataset is not "
        "hash-identical to source"
    )

print(
    "[PASS] Complete dataset lineage preserved"
)


# ============================================================
# 21. SAVE CSV REPORTS
# ============================================================

duplicate_path = (
    REPORT_DIR
    / "duplicate_audit.csv"
)

constant_path = (
    REPORT_DIR
    / "constant_feature_audit.csv"
)

numeric_path = (
    REPORT_DIR
    / "numeric_validity_audit.csv"
)

missing_path = (
    REPORT_DIR
    / "missing_value_audit.csv"
)

cross_split_path = (
    REPORT_DIR
    / "cross_split_representation_audit.csv"
)

pd.DataFrame(
    duplicate_df
).to_csv(
    duplicate_path,
    index=False
)

constant_df.to_csv(
    constant_path,
    index=False
)

numeric_df.to_csv(
    numeric_path,
    index=False
)

missing_df.to_csv(
    missing_path,
    index=False
)

pd.DataFrame(
    cross_split_details
).to_csv(
    cross_split_path,
    index=False
)


# ============================================================
# 22. QUARANTINE / REVIEW MANIFEST
# ============================================================

print("\nQUARANTINE / REVIEW MANIFEST")

review_records = []

if len(constant_train_features) > 0:

    review_records.append({
        "issue":
            "constant_training_features",

        "count":
            len(constant_train_features),

        "action":
            "document_only_no_deletion",

        "details":
            "|".join(
                constant_train_features
            ),
    })


if train_repeated_rows > 0:

    review_records.append({
        "issue":
            "repeated_train_representations",

        "count":
            train_repeated_rows,

        "action":
            "document_only_no_deletion",

        "details":
            "Repeated full representations "
            "are not automatically leakage.",
    })


if test_repeated_rows > 0:

    review_records.append({
        "issue":
            "repeated_test_representations",

        "count":
            test_repeated_rows,

        "action":
            "document_only_no_deletion",

        "details":
            "Repeated full representations "
            "are not automatically leakage.",
    })


if cross_split_overlap > 0:

    review_records.append({
        "issue":
            "cross_split_identical_representations",

        "count":
            cross_split_overlap,

        "action":
            "review_classification_required",

        "details":
            "Classify identical cross-split "
            "representations by label consistency.",
    })


if cross_split_conflict > 0:

    review_records.append({
        "issue":
            "cross_split_label_conflicts",

        "count":
            cross_split_conflict,

        "action":
            "review_required",

        "details":
            "Identical full representations "
            "carry different labels across splits.",
    })


if train_ambiguous_reps > 0:

    review_records.append({
        "issue":
            "train_label_ambiguity",

        "count":
            train_ambiguous_reps,

        "action":
            "document_only_no_deletion",

        "details":
            f"{train_ambiguous_rows} affected rows.",
    })


if test_ambiguous_reps > 0:

    review_records.append({
        "issue":
            "test_label_ambiguity",

        "count":
            test_ambiguous_reps,

        "action":
            "document_only_no_deletion",

        "details":
            f"{test_ambiguous_rows} affected rows.",
    })


if (
    train_behavior_missing
    +
    test_behavior_missing
) > 0:

    review_records.append({
        "issue":
            "missing_behavior_text",

        "count":
            train_behavior_missing
            +
            test_behavior_missing,

        "action":
            "document_only_no_deletion",

        "details":
            "Existing production pipeline "
            "handles missing text representation.",
    })


review_df = pd.DataFrame(
    review_records
)

if review_df.empty:

    review_df = pd.DataFrame(
        columns=[
            "issue",
            "count",
            "action",
            "details",
        ]
    )

review_path = (
    REPORT_DIR
    / "review_manifest.csv"
)

review_df.to_csv(
    review_path,
    index=False
)

print(
    f"Review items documented: "
    f"{len(review_df)}"
)


# ============================================================
# 23. QUALITY CHECKS
# ============================================================

quality_checks = {

    "source_files_present":
        all(
            path.exists()
            for path in source_paths.values()
        ),

    "train_alignment":
        train_alignment,

    "test_alignment":
        test_alignment,

    "expected_train_rows":
        train_rows == 18677,

    "expected_test_rows":
        test_rows == 11707,

    "structured_features_62":
        structured_train.shape[1]
        == EXPECTED_STRUCTURED_FEATURES,

    "train_test_schema_match":
        train_columns
        ==
        test_columns,

    "required_columns_present":
        (
            "techniques"
            in labels_train.columns
            and
            "techniques"
            in labels_test.columns
            and
            "timestamp"
            in metadata_train.columns
            and
            "timestamp"
            in metadata_test.columns
            and
            "behavior_text_normalized"
            in behavior_train.columns
            and
            "behavior_text_normalized"
            in behavior_test.columns
        ),

    "numeric_values_valid":
        (
            numeric_df[
                "train_infinite"
            ].sum()
            == 0
            and
            numeric_df[
                "test_infinite"
            ].sum()
            == 0
        ),

    "technique_count_45":
        len(all_techniques)
        == EXPECTED_TECHNIQUES,

    "timestamps_valid":
        (
            invalid_train_timestamps
            == 0
            and
            invalid_test_timestamps
            == 0
        ),

    "temporal_separation_valid":
        train_latest
        <
        test_earliest,

    "final_copy_hash_match":
        lineage_preserved,
}


# ============================================================
# 24. DECISION LOGIC
# ============================================================

hard_failures = [
    name
    for name, value
    in quality_checks.items()
    if not value
]

if hard_failures:

    decision = (
        "REVIEW_REQUIRED"
    )

else:

    decision = (
        "PASS_WITH_DOCUMENTED_LIMITATIONS"
    )


# ============================================================
# 25. MANIFEST
# ============================================================

manifest = {

    "dataset_name":
        "windows_apt_v4_1_final",

    "status":
        "FROZEN_COPY_OF_VALIDATED_V4_1",

    "source_directory":
        str(SOURCE_DIR),

    "final_directory":
        str(FINAL_DIR),

    "rows": {

        "train":
            int(train_rows),

        "test":
            int(test_rows),

        "total":
            int(total_rows),
    },

    "structured_features":
        int(
            structured_train.shape[1]
        ),

    "numeric_feature_count":
        int(
            len(numeric_columns)
        ),

    "categorical_feature_count":
        int(
            len(structured_train.columns)
            -
            len(numeric_columns)
        ),

    "tfidf_feature_count":
        EXPECTED_TFIDF_FEATURES,

    "final_model_feature_count":
        EXPECTED_FINAL_FEATURES,

    "mitre_technique_count":
        int(len(all_techniques)),

    "mitre_techniques":
        all_techniques,

    "constant_training_features":
        constant_train_features,

    "duplicate_audit": {

        "repeated_train_rows":
            int(train_repeated_rows),

        "repeated_test_rows":
            int(test_repeated_rows),

        "cross_split_identical_representations":
            int(cross_split_overlap),

        "cross_split_same_label":
            int(cross_split_same_label),

        "cross_split_label_conflicts":
            int(cross_split_conflict),

        "train_ambiguous_representations":
            int(train_ambiguous_reps),

        "test_ambiguous_representations":
            int(test_ambiguous_reps),
    },

    "temporal": {

        "train_latest":
            str(train_latest),

        "test_earliest":
            str(test_earliest),

        "gap":
            str(temporal_gap),

        "separation_valid":
            True,
    },

    "behavior_text": {

        "train_missing":
            train_behavior_missing,

        "test_missing":
            test_behavior_missing,

        "train_empty":
            train_behavior_empty,

        "test_empty":
            test_behavior_empty,
    },

    "source_sha256":
        source_hashes,

    "final_sha256":
        final_hashes,

    "lineage_preserved":
        bool(lineage_preserved),

    "production_lineage_policy": {

        "source_modified":
            False,

        "rows_deleted":
            False,

        "labels_modified":
            False,

        "features_modified":
            False,

        "train_test_membership_modified":
            False,

        "model_modified":
            False,

        "thresholds_modified":
            False,

        "retraining_performed":
            False,
    },

    "review_items":
        review_records,

    "quality_checks":
        quality_checks,

    "decision":
        decision,
}


manifest_path = (
    FINAL_DIR
    / "final_dataset_manifest.json"
)

with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_safe(manifest),
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 26. FINAL REPORT
# ============================================================

report = {

    "gate":
        "FINAL_DATASET_CLEANING",

    "title":
        "SmartSOC Final Dataset Cleaning, Audit & Freeze",

    "decision":
        decision,

    "dataset": {

        "source":
            str(SOURCE_DIR),

        "final":
            str(FINAL_DIR),

        "train_rows":
            int(train_rows),

        "test_rows":
            int(test_rows),

        "total_rows":
            int(total_rows),

        "structured_features":
            int(
                structured_train.shape[1]
            ),

        "tfidf_features":
            EXPECTED_TFIDF_FEATURES,

        "final_model_features":
            EXPECTED_FINAL_FEATURES,

        "mitre_techniques":
            int(len(all_techniques)),
    },

    "alignment": {

        "train":
            train_alignment,

        "test":
            test_alignment,

        "schema_match":
            train_columns
            ==
            test_columns,
    },

    "duplicates": {

        "exact_duplicate_audit":
            duplicate_df.to_dict(
                orient="records"
            ),

        "train_repeated_rows":
            int(train_repeated_rows),

        "test_repeated_rows":
            int(test_repeated_rows),

        "cross_split_identical":
            int(cross_split_overlap),

        "cross_split_same_label":
            int(cross_split_same_label),

        "cross_split_label_conflicts":
            int(cross_split_conflict),

        "train_ambiguous_representations":
            int(train_ambiguous_reps),

        "test_ambiguous_representations":
            int(test_ambiguous_reps),
    },

    "constants":
        constant_df.to_dict(
            orient="records"
        ),

    "missing_values":
        missing_df.to_dict(
            orient="records"
        ),

    "numeric_validity":
        numeric_df.to_dict(
            orient="records"
        ),

    "labels": {

        "technique_count":
            int(len(all_techniques)),

        "techniques":
            all_techniques,

        "empty_train_rows":
            int(empty_train_labels),

        "empty_test_rows":
            int(empty_test_labels),
    },

    "timestamps": {

        "invalid_train":
            int(invalid_train_timestamps),

        "invalid_test":
            int(invalid_test_timestamps),

        "train_latest":
            str(train_latest),

        "test_earliest":
            str(test_earliest),

        "temporal_gap":
            str(temporal_gap),

        "separation_valid":
            True,
    },

    "identifiers":
        identifier_rows,

    "behavior": {

        "train_missing":
            int(train_behavior_missing),

        "test_missing":
            int(test_behavior_missing),

        "train_empty":
            int(train_behavior_empty),

        "test_empty":
            int(test_behavior_empty),
    },

    "lineage": {

        "hash_preserved":
            bool(lineage_preserved),

        "source_sha256":
            source_hashes,

        "final_sha256":
            final_hashes,
    },

    "quality_checks":
        quality_checks,

    "hard_failures":
        hard_failures,

    "review_manifest":
        review_records,

    "policy": {

        "automatic_deletion":
            False,

        "automatic_label_correction":
            False,

        "automatic_feature_removal":
            False,

        "source_dataset_modified":
            False,

        "production_model_modified":
            False,

        "production_thresholds_modified":
            False,

        "retraining":
            False,
    },

    "outputs": {

        "final_dataset":
            str(FINAL_DIR),

        "manifest":
            str(manifest_path),

        "duplicate_audit":
            str(duplicate_path),

        "constant_feature_audit":
            str(constant_path),

        "numeric_audit":
            str(numeric_path),

        "missing_value_audit":
            str(missing_path),

        "cross_split_audit":
            str(cross_split_path),

        "review_manifest":
            str(review_path),

        "final_report":
            str(
                REPORT_DIR
                / "final_dataset_cleaning_report.json"
            ),
    },

    "methodology_notes": [

        "The validated windows_apt_v4_1 source dataset "
        "was treated as immutable.",

        "The final dataset is an exact hash-preserving "
        "copy of the source dataset.",

        "Duplicate rows were investigated but not "
        "automatically deleted.",

        "Identical representations across train and "
        "test were separately checked for label conflicts.",

        "Constant features were documented rather than "
        "removed because the production model uses the "
        "existing 62-feature schema.",

        "Timestamp parsing explicitly uses the dataset's "
        "Elasticsearch/Kibana timestamp format.",

        "The temporal split remains train ending before "
        "test begins.",

        "No production model, threshold, or inference "
        "artifact was modified.",
    ],
}


report_path = (
    REPORT_DIR
    / "final_dataset_cleaning_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_safe(report),
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 27. FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("FINAL DATASET CLEANING SUMMARY")
print("=" * 72)

print(
    f"Train rows              : "
    f"{train_rows}"
)

print(
    f"Test rows               : "
    f"{test_rows}"
)

print(
    f"Total rows              : "
    f"{total_rows}"
)

print(
    f"Structured features     : "
    f"{structured_train.shape[1]}"
)

print(
    f"TF-IDF features         : "
    f"{EXPECTED_TFIDF_FEATURES}"
)

print(
    f"Final model features    : "
    f"{EXPECTED_FINAL_FEATURES}"
)

print(
    f"MITRE techniques        : "
    f"{len(all_techniques)}"
)

print(
    f"Constant train features: "
    f"{len(constant_train_features)}"
)

print(
    f"Repeated train rows     : "
    f"{train_repeated_rows}"
)

print(
    f"Repeated test rows      : "
    f"{test_repeated_rows}"
)

print(
    f"Cross-split identical   : "
    f"{cross_split_overlap}"
)

print(
    f"Cross-split conflicts   : "
    f"{cross_split_conflict}"
)

print(
    f"Train ambiguous reps    : "
    f"{train_ambiguous_reps}"
)

print(
    f"Test ambiguous reps     : "
    f"{test_ambiguous_reps}"
)

print(
    f"Train latest            : "
    f"{train_latest}"
)

print(
    f"Test earliest           : "
    f"{test_earliest}"
)

print(
    f"Temporal gap            : "
    f"{temporal_gap}"
)

print(
    f"Lineage hash preserved  : "
    f"{lineage_preserved}"
)


print("\nQUALITY CHECKS")

for name, passed in quality_checks.items():

    print(
        f"  [{'PASS' if passed else 'FAIL'}] "
        f"{name}"
    )


print(
    f"\nDECISION: {decision}"
)

print(
    "\nFINAL DATASET:"
)

print(
    FINAL_DIR
)

print(
    "\nMANIFEST:"
)

print(
    manifest_path
)

print(
    "\nREPORT:"
)

print(
    report_path
)

print(
    "\nNO PRODUCTION MODEL, THRESHOLD, "
    "OR SOURCE DATASET WAS MODIFIED."
)

print("=" * 72)