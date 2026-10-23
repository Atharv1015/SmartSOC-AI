from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


# ============================================================
# SmartSOC-AI v4.1
# COMPLETE DATASET READINESS AUDIT
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

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


STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

LABELS_TRAIN = DATA_DIR / "labels_train.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"

METADATA_TRAIN = DATA_DIR / "metadata_train.csv"
METADATA_TEST = DATA_DIR / "metadata_test.csv"


EXPECTED_STRUCTURED_FEATURES = 62


# ============================================================
# EXPECTED V4.1 STRUCTURED SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "event_id",
    "system_level",
    "is_sysmon",
    "is_security_channel",
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
    "has_process_image",
    "has_parent_image",
    "has_user",
    "has_parent_user",
    "is_system_user",
    "is_admin_user",
    "integrity_high",
    "integrity_medium",
    "command_length",
    "parent_command_length",
    "has_command_line",
    "has_parent_command",
    "uses_powershell",
    "uses_cmd",
    "uses_net",
    "uses_whoami",
    "uses_schtasks",
    "uses_reg",
    "uses_rundll32",
    "uses_wmic",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "uses_sc",
    "uses_mshta",
    "uses_regsvr32",
    "uses_cscript",
    "uses_wscript",
    "uses_wevtutil",
    "uses_vssadmin",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "same_process_parent",
    "suspicious_parent_wazuh",
    "has_hash",
    "has_sha256",
    "has_md5",
    "has_source_ip",
    "has_destination_ip",
    "has_source_port",
    "has_destination_port",
    "has_protocol",
    "is_network_event",
    "has_target_filename",
    "has_image_loaded",
    "has_target_object",
    "administrative_powershell_context",
    "strong_behavior_signal",
]


BINARY_COLUMNS = [
    "is_sysmon",
    "is_security_channel",
    "has_process_image",
    "has_parent_image",
    "has_user",
    "has_parent_user",
    "is_system_user",
    "is_admin_user",
    "integrity_high",
    "integrity_medium",
    "has_command_line",
    "has_parent_command",
    "uses_powershell",
    "uses_cmd",
    "uses_net",
    "uses_whoami",
    "uses_schtasks",
    "uses_reg",
    "uses_rundll32",
    "uses_wmic",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "uses_sc",
    "uses_mshta",
    "uses_regsvr32",
    "uses_cscript",
    "uses_wscript",
    "uses_wevtutil",
    "uses_vssadmin",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "same_process_parent",
    "suspicious_parent_wazuh",
    "has_hash",
    "has_sha256",
    "has_md5",
    "has_source_ip",
    "has_destination_ip",
    "has_source_port",
    "has_destination_port",
    "has_protocol",
    "is_network_event",
    "has_target_filename",
    "has_image_loaded",
    "has_target_object",
    "administrative_powershell_context",
    "strong_behavior_signal",
]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def file_status(path):
    return path.exists() and path.is_file()


def load_csv(path):
    if not file_status(path):
        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def calculate_iqr_outliers(df, numeric_columns):
    """
    IQR analysis is used ONLY for reporting.

    IMPORTANT:
    We do NOT delete these rows automatically.
    Rare cybersecurity behavior can be legitimate
    attack behavior.
    """

    matrix = pd.DataFrame(
        False,
        index=df.index,
        columns=numeric_columns,
    )

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = series.dropna()

        if valid.empty:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        matrix[column] = (
            (series < lower)
            | (series > upper)
        )

    return matrix


def extract_mitre_ids(value):
    if pd.isna(value):
        return []

    return re.findall(
        r"T\d{4}(?:\.\d{3})?",
        str(value),
    )


# ============================================================
# START
# ============================================================

print_header(
    "SmartSOC-AI v4.1 DATASET READINESS AUDIT"
)

print(
    f"Project root:\n{PROJECT_ROOT}"
)

print(
    f"\nDataset directory:\n{DATA_DIR}"
)

# ============================================================
# FILE CHECK
# ============================================================

print_header("1. REQUIRED FILE CHECK")

required_files = [
    STRUCTURED_TRAIN,
    STRUCTURED_TEST,
    LABELS_TRAIN,
    LABELS_TEST,
    METADATA_TRAIN,
    METADATA_TEST,
]

all_files_exist = True

for path in required_files:

    exists = file_status(path)

    print(
        f"[{'OK' if exists else 'FAIL'}] "
        f"{path.name}"
    )

    if not exists:
        all_files_exist = False

if not all_files_exist:
    raise FileNotFoundError(
        "\nOne or more required v4.1 files are missing."
    )


# ============================================================
# LOAD DATA
# ============================================================

print_header("2. LOADING DATASETS")

structured_train = load_csv(
    STRUCTURED_TRAIN
)

structured_test = load_csv(
    STRUCTURED_TEST
)

labels_train = load_csv(
    LABELS_TRAIN
)

labels_test = load_csv(
    LABELS_TEST
)

metadata_train = load_csv(
    METADATA_TRAIN
)

metadata_test = load_csv(
    METADATA_TEST
)

print(
    f"Structured train : {structured_train.shape}"
)

print(
    f"Structured test  : {structured_test.shape}"
)

print(
    f"Labels train     : {labels_train.shape}"
)

print(
    f"Labels test      : {labels_test.shape}"
)

print(
    f"Metadata train   : {metadata_train.shape}"
)

print(
    f"Metadata test    : {metadata_test.shape}"
)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

print_header("3. STRUCTURED FEATURE SCHEMA")

schema_ok = True

actual_columns = list(
    structured_train.columns
)

print(
    f"Expected features : "
    f"{EXPECTED_STRUCTURED_FEATURES}"
)

print(
    f"Actual features   : "
    f"{len(actual_columns)}"
)

if actual_columns != EXPECTED_COLUMNS:

    schema_ok = False

    missing = [
        c
        for c in EXPECTED_COLUMNS
        if c not in actual_columns
    ]

    extra = [
        c
        for c in actual_columns
        if c not in EXPECTED_COLUMNS
    ]

    print("\n[FAIL] Schema mismatch.")

    if missing:
        print("\nMissing columns:")
        for c in missing:
            print(" -", c)

    if extra:
        print("\nUnexpected columns:")
        for c in extra:
            print(" -", c)

else:

    print(
        "[PASS] Training schema exactly matches v4.1."
    )


if list(structured_test.columns) != EXPECTED_COLUMNS:

    schema_ok = False

    print(
        "[FAIL] Test schema does not match training schema."
    )

else:

    print(
        "[PASS] Test schema matches training schema."
    )


# ============================================================
# ROW ALIGNMENT
# ============================================================

print_header("4. TRAIN / TEST ROW ALIGNMENT")

alignment_checks = {
    "train_structured_vs_labels":
        len(structured_train) == len(labels_train),

    "train_structured_vs_metadata":
        len(structured_train) == len(metadata_train),

    "test_structured_vs_labels":
        len(structured_test) == len(labels_test),

    "test_structured_vs_metadata":
        len(structured_test) == len(metadata_test),
}

alignment_ok = True

for name, passed in alignment_checks.items():

    print(
        f"[{'PASS' if passed else 'FAIL'}] "
        f"{name}"
    )

    if not passed:
        alignment_ok = False


# ============================================================
# MISSING VALUES
# ============================================================

print_header("5. MISSING VALUE AUDIT")

missing_train = (
    structured_train.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_test = (
    structured_test.isna()
    .sum()
    .sort_values(ascending=False)
)

missing_report = pd.DataFrame({
    "feature": EXPECTED_COLUMNS,
    "train_missing": [
        int(missing_train.get(c, 0))
        for c in EXPECTED_COLUMNS
    ],
    "test_missing": [
        int(missing_test.get(c, 0))
        for c in EXPECTED_COLUMNS
    ],
})

missing_report[
    "train_missing_percentage"
] = (
    missing_report["train_missing"]
    / len(structured_train)
    * 100
)

missing_report[
    "test_missing_percentage"
] = (
    missing_report["test_missing"]
    / len(structured_test)
    * 100
)

missing_report.to_csv(
    REPORT_DIR / "missing_values.csv",
    index=False,
)

total_missing_train = int(
    structured_train.isna().sum().sum()
)

total_missing_test = int(
    structured_test.isna().sum().sum()
)

print(
    f"Train missing cells : "
    f"{total_missing_train:,}"
)

print(
    f"Test missing cells  : "
    f"{total_missing_test:,}"
)

if total_missing_train == 0:
    print("[PASS] No missing structured training values.")
else:
    print("[REVIEW] Training missing values found.")

if total_missing_test == 0:
    print("[PASS] No missing structured test values.")
else:
    print("[REVIEW] Test missing values found.")


# ============================================================
# DUPLICATES
# ============================================================

print_header("6. DUPLICATE AUDIT")

train_duplicates = int(
    structured_train.duplicated().sum()
)

test_duplicates = int(
    structured_test.duplicated().sum()
)

print(
    f"Training duplicate rows : "
    f"{train_duplicates:,}"
)

print(
    f"Test duplicate rows     : "
    f"{test_duplicates:,}"
)

duplicate_report = pd.DataFrame({
    "dataset": [
        "train",
        "test",
    ],
    "duplicate_rows": [
        train_duplicates,
        test_duplicates,
    ],
})

duplicate_report.to_csv(
    REPORT_DIR / "duplicate_summary.csv",
    index=False,
)


# ============================================================
# BINARY FEATURE VALIDATION
# ============================================================

print_header("7. BINARY FEATURE VALIDATION")

binary_issues = []

for column in BINARY_COLUMNS:

    if column not in structured_train.columns:
        continue

    values = (
        pd.to_numeric(
            structured_train[column],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    invalid = [
        value
        for value in values
        if value not in (0, 1)
    ]

    if invalid:

        binary_issues.append({
            "feature": column,
            "invalid_values": str(
                sorted(invalid)
            ),
        })

if binary_issues:

    binary_ok = False

    print(
        f"[REVIEW] "
        f"{len(binary_issues)} binary features "
        f"contain non-0/1 values."
    )

    pd.DataFrame(
        binary_issues
    ).to_csv(
        REPORT_DIR / "binary_feature_issues.csv",
        index=False,
    )

else:

    binary_ok = True

    print(
        "[PASS] Binary features contain only 0/1 values."
    )


# ============================================================
# NUMERIC VALIDATION
# ============================================================

print_header("8. NUMERICAL VALUE VALIDATION")

numeric_columns = list(
    structured_train.select_dtypes(
        include=[np.number]
    ).columns
)

print(
    f"Numerical columns detected: "
    f"{len(numeric_columns)}"
)

numeric_report_rows = []

for column in numeric_columns:

    train_numeric = pd.to_numeric(
        structured_train[column],
        errors="coerce",
    )

    test_numeric = pd.to_numeric(
        structured_test[column],
        errors="coerce",
    )

    train_inf = int(
        np.isinf(
            train_numeric
        ).sum()
    )

    test_inf = int(
        np.isinf(
            test_numeric
        ).sum()
    )

    train_invalid = int(
        train_numeric.isna().sum()
        - structured_train[column].isna().sum()
    )

    test_invalid = int(
        test_numeric.isna().sum()
        - structured_test[column].isna().sum()
    )

    numeric_report_rows.append({
        "feature": column,
        "train_infinite": train_inf,
        "test_infinite": test_inf,
        "train_non_numeric": train_invalid,
        "test_non_numeric": test_invalid,
    })

numeric_report = pd.DataFrame(
    numeric_report_rows
)

numeric_report.to_csv(
    REPORT_DIR / "numeric_validation.csv",
    index=False,
)

total_infinite = int(
    numeric_report[
        [
            "train_infinite",
            "test_infinite",
        ]
    ].sum().sum()
)

total_non_numeric = int(
    numeric_report[
        [
            "train_non_numeric",
            "test_non_numeric",
        ]
    ].sum().sum()
)

print(
    f"Infinite values : "
    f"{total_infinite:,}"
)

print(
    f"Non-numeric values : "
    f"{total_non_numeric:,}"
)

if total_infinite == 0 and total_non_numeric == 0:
    print(
        "[PASS] Numerical values are valid."
    )
else:
    print(
        "[REVIEW] Numerical issues found."
    )


# ============================================================
# CONSTANT FEATURES
# ============================================================

print_header("9. CONSTANT / LOW-VARIANCE FEATURES")

constant_features = []

for column in structured_train.columns:

    unique_count = (
        structured_train[column]
        .nunique(dropna=False)
    )

    if unique_count <= 1:

        constant_features.append(
            column
        )

print(
    f"Constant features: "
    f"{len(constant_features)}"
)

for column in constant_features:
    print(
        f" - {column}"
    )

pd.DataFrame({
    "constant_feature": constant_features
}).to_csv(
    REPORT_DIR / "constant_features.csv",
    index=False,
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

print_header("10. OUTLIER ANALYSIS")

print(
    "Using IQR only for statistical analysis."
)

print(
    "Outliers will NOT be automatically deleted."
)

outlier_matrix = calculate_iqr_outliers(
    structured_train,
    numeric_columns,
)

total_outlier_cells = int(
    outlier_matrix.sum().sum()
)

outlier_per_event = (
    outlier_matrix.sum(axis=1)
)

total_outlier_events = int(
    (outlier_per_event > 0).sum()
)

outlier_event_percentage = (
    total_outlier_events
    / len(structured_train)
    * 100
)

outlier_cell_percentage = (
    total_outlier_cells
    / outlier_matrix.size
    * 100
)

print(
    f"\nTotal outlier cells : "
    f"{total_outlier_cells:,}"
)

print(
    f"Events with >=1 outlier : "
    f"{total_outlier_events:,}"
)

print(
    f"Outlier events percentage : "
    f"{outlier_event_percentage:.2f}%"
)

print(
    f"Outlier cell percentage : "
    f"{outlier_cell_percentage:.2f}%"
)

feature_outliers = (
    outlier_matrix.sum()
    .sort_values(
        ascending=False
    )
)

outlier_feature_report = pd.DataFrame({
    "feature":
        feature_outliers.index,

    "outlier_count":
        feature_outliers.values,
})

outlier_feature_report[
    "outlier_percentage"
] = (
    outlier_feature_report["outlier_count"]
    / len(structured_train)
    * 100
)

outlier_feature_report.to_csv(
    REPORT_DIR / "outlier_summary.csv",
    index=False,
)


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

print_header("11. FEATURE DISTRIBUTION")

distribution_rows = []

for column in structured_train.columns:

    series = structured_train[column]

    distribution_rows.append({
        "feature": column,
        "dtype": str(series.dtype),
        "unique_values":
            int(series.nunique(dropna=False)),
        "missing":
            int(series.isna().sum()),
    })

distribution_report = pd.DataFrame(
    distribution_rows
)

distribution_report.to_csv(
    REPORT_DIR / "feature_distribution.csv",
    index=False,
)


# ============================================================
# LABEL ANALYSIS
# ============================================================

print_header("12. MITRE LABEL ANALYSIS")

label_counter = {}

all_techniques = set()

for value in labels_train["techniques"].fillna(""):

    techniques = extract_mitre_ids(
        value
    )

    for technique in set(techniques):

        all_techniques.add(
            technique
        )

        label_counter[technique] = (
            label_counter.get(
                technique,
                0,
            )
            + 1
        )

label_distribution = pd.DataFrame([
    {
        "technique": technique,
        "training_rows": count,
        "percentage": (
            count
            / len(labels_train)
            * 100
        ),
    }
    for technique, count
    in sorted(
        label_counter.items(),
        key=lambda x: x[1],
        reverse=True,
    )
])

label_distribution.to_csv(
    REPORT_DIR / "label_distribution.csv",
    index=False,
)

print(
    f"Unique MITRE techniques found: "
    f"{len(all_techniques)}"
)

print("\nTop techniques:")

for _, row in label_distribution.head(20).iterrows():

    print(
        f"{row['technique']:<15}"
        f"{int(row['training_rows']):>8,}"
        f" "
        f"{row['percentage']:>7.2f}%"
    )


# ============================================================
# TRAIN / TEST FEATURE OVERLAP
# ============================================================

print_header("13. TRAIN / TEST DUPLICATE FEATURE ROW CHECK")

train_hash = pd.util.hash_pandas_object(
    structured_train,
    index=False,
)

test_hash = pd.util.hash_pandas_object(
    structured_test,
    index=False,
)

train_hash_set = set(
    train_hash.tolist()
)

test_hash_set = set(
    test_hash.tolist()
)

overlap_count = len(
    train_hash_set
    & test_hash_set
)

print(
    f"Exact structured rows appearing "
    f"in both train and test: "
    f"{overlap_count:,}"
)

if overlap_count == 0:
    print(
        "[PASS] No exact structured-row overlap detected."
    )
else:
    print(
        "[REVIEW] Potential train/test overlap detected."
    )


# ============================================================
# CORRELATION MATRIX
# ============================================================

print_header("14. CORRELATION ANALYSIS")

correlation = structured_train[
    numeric_columns
].corr(
    method="pearson"
)

correlation.to_csv(
    REPORT_DIR / "correlation_matrix.csv"
)

print(
    "Correlation matrix saved."
)


# ============================================================
# SUMMARY
# ============================================================

print_header("15. FINAL DATASET SUMMARY")

summary = {
    "project_root":
        str(PROJECT_ROOT),

    "data_directory":
        str(DATA_DIR),

    "train_rows":
        int(len(structured_train)),

    "test_rows":
        int(len(structured_test)),

    "structured_features":
        int(len(structured_train.columns)),

    "numeric_features":
        int(len(numeric_columns)),

    "categorical_features":
        int(
            len(structured_train.columns)
            - len(numeric_columns)
        ),

    "mitre_techniques":
        int(len(all_techniques)),

    "train_missing_cells":
        total_missing_train,

    "test_missing_cells":
        total_missing_test,

    "train_duplicate_rows":
        train_duplicates,

    "test_duplicate_rows":
        test_duplicates,

    "constant_features":
        len(constant_features),

    "outlier_cells":
        total_outlier_cells,

    "outlier_events":
        total_outlier_events,

    "outlier_event_percentage":
        round(
            outlier_event_percentage,
            4,
        ),

    "train_test_exact_row_overlap":
        overlap_count,
}

with open(
    REPORT_DIR / "data_quality_report.json",
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        summary,
        handle,
        indent=2,
    )


# ============================================================
# FINAL STATUS
# ============================================================

print_header("FINAL STATUS")

checks = [
    (
        "Required files",
        all_files_exist,
    ),

    (
        "Structured schema",
        schema_ok,
    ),

    (
        "Train/test row alignment",
        alignment_ok,
    ),

    (
        "Binary features",
        binary_ok,
    ),

    (
        "Numerical validity",
        total_infinite == 0
        and total_non_numeric == 0,
    ),

    (
        "Train/test exact overlap",
        overlap_count == 0,
    ),
]

all_passed = True

for name, passed in checks:

    print(
        f"[{'PASS' if passed else 'REVIEW'}] "
        f"{name}"
    )

    if not passed:
        all_passed = False


print()
print(
    "IMPORTANT:"
)

print(
    "Outliers were analyzed but NOT deleted."
)

print(
    "Rare cybersecurity behaviors are preserved."
)

if all_passed:

    print(
        "\nDATASET STATUS: "
        "STRUCTURALLY READY FOR FINAL REVIEW"
    )

else:

    print(
        "\nDATASET STATUS: "
        "NEEDS REVIEW BEFORE MODEL TRAINING"
    )

print(
    f"\nReports saved to:\n{REPORT_DIR}"
)