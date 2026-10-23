import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.3B
# BENIGN DATASET COVERAGE & FEATURE SEMANTICS AUDIT
#
# READ-ONLY AUDIT
#
# This script:
#   - analyzes the existing training dataset
#   - identifies common Windows administrative utilities
#   - examines their associated MITRE labels
#   - examines suspicious-feature activation
#   - examines behavior-text associations
#   - identifies possible benign-negative-class deficiency
#
# It DOES NOT:
#   - modify the dataset
#   - modify the model
#   - modify thresholds
#   - retrain anything
#   - execute Windows commands
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3b"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase2_gate2_3b_benign_dataset_audit_report.json"
)

UTILITY_ANALYSIS_FILE = (
    REPORT_DIR
    / "utility_label_analysis.csv"
)

FEATURE_ANALYSIS_FILE = (
    REPORT_DIR
    / "utility_feature_analysis.csv"
)

TEXT_ANALYSIS_FILE = (
    REPORT_DIR
    / "utility_text_analysis.csv"
)

LABEL_CONTEXT_FILE = (
    REPORT_DIR
    / "training_label_context_analysis.csv"
)


# =============================================================================
# FILES
# =============================================================================

STRUCTURED_TRAIN_FILE = (
    DATA_DIR
    / "structured_train.csv"
)

LABELS_TRAIN_FILE = (
    DATA_DIR
    / "labels_train.csv"
)

BEHAVIOR_TRAIN_FILE = (
    DATA_DIR
    / "behavior_text_train.csv"
)

SCHEMA_FILE = (
    MODEL_DIR
    / "feature_schema.json"
)


# =============================================================================
# WARNINGS
# =============================================================================

warnings.filterwarnings(
    "ignore"
)


# =============================================================================
# WINDOWS UTILITIES OF INTEREST
# =============================================================================
#
# These are dual-use utilities:
# legitimate administration AND common attacker tooling.
#
# We are NOT declaring them malicious or benign.
# We are measuring how the dataset labels them.
# =============================================================================

UTILITY_PATTERNS = {

    "powershell":
        [
            "powershell",
            "pwsh",
        ],

    "cmd":
        [
            "cmd.exe",
            "cmd",
        ],

    "whoami":
        [
            "whoami",
        ],

    "ipconfig":
        [
            "ipconfig",
        ],

    "netstat":
        [
            "netstat",
        ],

    "tasklist":
        [
            "tasklist",
        ],

    "net":
        [
            "net.exe",
            "net ",
        ],

    "sc":
        [
            "sc.exe",
            "sc ",
        ],

    "schtasks":
        [
            "schtasks",
        ],

    "reg":
        [
            "reg.exe",
            "reg ",
        ],

    "wscript":
        [
            "wscript",
        ],

    "cscript":
        [
            "cscript",
        ],

    "rundll32":
        [
            "rundll32",
        ],

    "wmic":
        [
            "wmic",
        ],

    "certutil":
        [
            "certutil",
        ],

    "bitsadmin":
        [
            "bitsadmin",
        ],

    "mshta":
        [
            "mshta",
        ],

    "regsvr32":
        [
            "regsvr32",
        ],

    "wevtutil":
        [
            "wevtutil",
        ],

    "vssadmin":
        [
            "vssadmin",
        ],
}


# =============================================================================
# SUSPICIOUS FEATURES
# =============================================================================

SUSPICIOUS_FEATURES = [

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

    "suspicious_parent_wazuh",
    "administrative_powershell_context",
    "strong_behavior_signal",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fail(message):

    print()
    print(
        f"[FAIL] {message}"
    )

    raise SystemExit(1)


def require_file(path):

    if not path.exists():

        fail(
            f"Required file missing: {path}"
        )

    print(
        f"[OK] {path}"
    )


def normalize(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .lower()
        .strip()
    )


def parse_labels(value):

    text = normalize(value)

    if not text:
        return []

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def contains_any(text, patterns):

    text = normalize(text)

    return any(
        pattern.lower() in text
        for pattern in patterns
    )


def safe_mean(series):

    if len(series) == 0:
        return 0.0

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(0)
        .mean()
    )


def percentage(numerator, denominator):

    if denominator == 0:
        return 0.0

    return (
        100.0
        * numerator
        / denominator
    )


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3B")
print("BENIGN DATASET COVERAGE & FEATURE SEMANTICS AUDIT")
print("=" * 80)


REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 1. CHECK FILES
# =============================================================================

print()
print("-" * 80)
print("Checking required files")
print("-" * 80)


for path in [

    STRUCTURED_TRAIN_FILE,
    LABELS_TRAIN_FILE,
    BEHAVIOR_TRAIN_FILE,
    SCHEMA_FILE,

]:

    require_file(path)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

print()
print("-" * 80)
print("Loading training data")
print("-" * 80)


structured = pd.read_csv(
    STRUCTURED_TRAIN_FILE
)

labels = pd.read_csv(
    LABELS_TRAIN_FILE
)

behavior = pd.read_csv(
    BEHAVIOR_TRAIN_FILE
)


print(
    f"Structured rows : {len(structured)}"
)

print(
    f"Label rows      : {len(labels)}"
)

print(
    f"Behavior rows   : {len(behavior)}"
)


if not (
    len(structured)
    == len(labels)
    == len(behavior)
):

    fail(
        "Training row counts are not aligned."
    )


if "techniques" not in labels.columns:

    fail(
        "labels_train.csv does not contain "
        "'techniques'."
    )


if (
    "behavior_text_normalized"
    not in behavior.columns
):

    fail(
        "behavior_text_train.csv does not contain "
        "'behavior_text_normalized'."
    )


# =============================================================================
# 3. LOAD SCHEMA
# =============================================================================

print()
print("-" * 80)
print("Loading production schema")
print("-" * 80)


with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as f:

    schema = json.load(f)


categorical_columns = [
    str(x)
    for x in schema[
        "categorical_columns"
    ]
]

numerical_columns = [
    str(x)
    for x in schema[
        "numerical_columns"
    ]
]

structured_columns = (
    categorical_columns
    + numerical_columns
)


missing_columns = [
    col
    for col in structured_columns
    if col not in structured.columns
]


if missing_columns:

    fail(
        "Missing structured columns: "
        f"{missing_columns}"
    )


print(
    f"Structured features: "
    f"{len(structured_columns)}"
)


# =============================================================================
# 4. COMBINE DATA
# =============================================================================

data = structured.copy()

data[
    "techniques"
] = labels[
    "techniques"
].fillna("").astype(str)

data[
    "behavior_text_normalized"
] = behavior[
    "behavior_text_normalized"
].fillna("").astype(str)


data[
    "_labels_list"
] = data[
    "techniques"
].apply(
    parse_labels
)


data[
    "_behavior_lower"
] = data[
    "behavior_text_normalized"
].apply(
    normalize
)


# =============================================================================
# 5. BASIC LABEL STATISTICS
# =============================================================================

print()
print("-" * 80)
print("Calculating label statistics")
print("-" * 80)


label_counts = {}


for labels_list in data[
    "_labels_list"
]:

    for label in labels_list:

        label_counts[label] = (
            label_counts.get(
                label,
                0
            )
            + 1
        )


unique_labels = sorted(
    label_counts.keys()
)


print(
    f"Unique MITRE labels: "
    f"{len(unique_labels)}"
)


# =============================================================================
# 6. UTILITY DETECTION
# =============================================================================

print()
print("-" * 80)
print("Analyzing dual-use Windows utilities")
print("-" * 80)


utility_records = []


for utility, patterns in (
    UTILITY_PATTERNS.items()
):

    # -------------------------------------------------------------------------
    # Search process name
    # -------------------------------------------------------------------------

    process_mask = (
        data[
            "process_name"
        ]
        .fillna("")
        .astype(str)
        .apply(
            lambda x:
                contains_any(
                    x,
                    patterns
                )
        )
    )


    # -------------------------------------------------------------------------
    # Search parent process
    # -------------------------------------------------------------------------

    parent_mask = (
        data[
            "parent_process_name"
        ]
        .fillna("")
        .astype(str)
        .apply(
            lambda x:
                contains_any(
                    x,
                    patterns
                )
        )
    )


    # -------------------------------------------------------------------------
    # Search behavior text
    # -------------------------------------------------------------------------

    behavior_mask = (
        data[
            "_behavior_lower"
        ]
        .apply(
            lambda x:
                contains_any(
                    x,
                    patterns
                )
        )
    )


    # -------------------------------------------------------------------------
    # Search combined categorical/text context
    # -------------------------------------------------------------------------

    utility_mask = (
        process_mask
        | parent_mask
        | behavior_mask
    )


    subset = data[
        utility_mask
    ]


    row_count = len(subset)


    # -------------------------------------------------------------------------
    # Label distribution
    # -------------------------------------------------------------------------

    local_label_counts = {}


    for labels_list in subset[
        "_labels_list"
    ]:

        for label in labels_list:

            local_label_counts[label] = (
                local_label_counts.get(
                    label,
                    0
                )
                + 1
            )


    sorted_local_labels = sorted(
        local_label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    top_labels = [
        label
        for label, count
        in sorted_local_labels[:10]
    ]


    top_label_counts = [
        count
        for label, count
        in sorted_local_labels[:10]
    ]


    # -------------------------------------------------------------------------
    # Multi-label prevalence
    # -------------------------------------------------------------------------

    multi_label_rows = int(
        sum(
            len(labels_list) > 1
            for labels_list
            in subset[
                "_labels_list"
            ]
        )
    )


    empty_label_rows = int(
        sum(
            len(labels_list) == 0
            for labels_list
            in subset[
                "_labels_list"
            ]
        )
    )


    # -------------------------------------------------------------------------
    # Suspicious feature rates
    # -------------------------------------------------------------------------

    feature_rates = {}


    for feature in SUSPICIOUS_FEATURES:

        if feature in subset.columns:

            values = pd.to_numeric(
                subset[feature],
                errors="coerce"
            ).fillna(0)

            feature_rates[
                feature
            ] = percentage(
                int(
                    (values > 0).sum()
                ),
                row_count
            )


    high_feature_rates = sorted(
        feature_rates.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # -------------------------------------------------------------------------
    # Most common labels
    # -------------------------------------------------------------------------

    top_label_1 = (
        sorted_local_labels[0][0]
        if sorted_local_labels
        else ""
    )

    top_label_1_count = (
        sorted_local_labels[0][1]
        if sorted_local_labels
        else 0
    )


    top_label_2 = (
        sorted_local_labels[1][0]
        if len(sorted_local_labels) > 1
        else ""
    )

    top_label_2_count = (
        sorted_local_labels[1][1]
        if len(sorted_local_labels) > 1
        else 0
    )


    top_label_3 = (
        sorted_local_labels[2][0]
        if len(sorted_local_labels) > 2
        else ""
    )

    top_label_3_count = (
        sorted_local_labels[2][1]
        if len(sorted_local_labels) > 2
        else 0
    )


    utility_records.append(
        {
            "utility":
                utility,

            "matching_rows":
                row_count,

            "matching_row_percentage":
                percentage(
                    row_count,
                    len(data)
                ),

            "rows_with_multiple_labels":
                multi_label_rows,

            "multiple_label_percentage":
                percentage(
                    multi_label_rows,
                    row_count
                ),

            "rows_with_empty_labels":
                empty_label_rows,

            "empty_label_percentage":
                percentage(
                    empty_label_rows,
                    row_count
                ),

            "unique_labels":
                len(local_label_counts),

            "top_label_1":
                top_label_1,

            "top_label_1_count":
                top_label_1_count,

            "top_label_1_percentage":
                percentage(
                    top_label_1_count,
                    row_count
                ),

            "top_label_2":
                top_label_2,

            "top_label_2_count":
                top_label_2_count,

            "top_label_2_percentage":
                percentage(
                    top_label_2_count,
                    row_count
                ),

            "top_label_3":
                top_label_3,

            "top_label_3_count":
                top_label_3_count,

            "top_label_3_percentage":
                percentage(
                    top_label_3_count,
                    row_count
                ),

            "top_suspicious_feature":
                (
                    high_feature_rates[0][0]
                    if high_feature_rates
                    else ""
                ),

            "top_suspicious_feature_rate":
                (
                    high_feature_rates[0][1]
                    if high_feature_rates
                    else 0.0
                ),

            "label_distribution":
                json.dumps(
                    local_label_counts,
                    sort_keys=True
                ),
        }
    )


utility_df = pd.DataFrame(
    utility_records
)


# =============================================================================
# 7. FEATURE SEMANTICS ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing suspicious feature semantics")
print("-" * 80)


feature_records = []


for feature in SUSPICIOUS_FEATURES:

    if feature not in data.columns:
        continue


    values = pd.to_numeric(
        data[feature],
        errors="coerce"
    ).fillna(0)


    activated = (
        values > 0
    )


    total_activated = int(
        activated.sum()
    )


    if total_activated == 0:

        feature_records.append(
            {
                "feature":
                    feature,

                "activated_rows":
                    0,

                "activation_rate":
                    0.0,

                "unique_labels":
                    0,

                "dominant_label":
                    "",

                "dominant_label_count":
                    0,

                "dominant_label_percentage":
                    0.0,

                "multi_label_rows":
                    0,

                "multi_label_percentage":
                    0.0,

                "empty_label_rows":
                    0,

                "empty_label_percentage":
                    0.0,

                "label_distribution":
                    "{}",
            }
        )

        continue


    subset = data[
        activated
    ]


    local_label_counts = {}


    for labels_list in subset[
        "_labels_list"
    ]:

        for label in labels_list:

            local_label_counts[label] = (
                local_label_counts.get(
                    label,
                    0
                )
                + 1
            )


    sorted_labels = sorted(
        local_label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    dominant_label = (
        sorted_labels[0][0]
        if sorted_labels
        else ""
    )

    dominant_count = (
        sorted_labels[0][1]
        if sorted_labels
        else 0
    )


    multi_label_rows = int(
        sum(
            len(x) > 1
            for x in subset[
                "_labels_list"
            ]
        )
    )


    empty_label_rows = int(
        sum(
            len(x) == 0
            for x in subset[
                "_labels_list"
            ]
        )
    )


    feature_records.append(
        {
            "feature":
                feature,

            "activated_rows":
                total_activated,

            "activation_rate":
                percentage(
                    total_activated,
                    len(data)
                ),

            "unique_labels":
                len(local_label_counts),

            "dominant_label":
                dominant_label,

            "dominant_label_count":
                dominant_count,

            "dominant_label_percentage":
                percentage(
                    dominant_count,
                    total_activated
                ),

            "multi_label_rows":
                multi_label_rows,

            "multi_label_percentage":
                percentage(
                    multi_label_rows,
                    total_activated
                ),

            "empty_label_rows":
                empty_label_rows,

            "empty_label_percentage":
                percentage(
                    empty_label_rows,
                    total_activated
                ),

            "label_distribution":
                json.dumps(
                    local_label_counts,
                    sort_keys=True
                ),
        }
    )


feature_df = pd.DataFrame(
    feature_records
)


# =============================================================================
# 8. TEXT ASSOCIATION ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Analyzing behavior-text associations")
print("-" * 80)


text_records = []


for utility, patterns in (
    UTILITY_PATTERNS.items()
):

    mask = (
        data[
            "_behavior_lower"
        ]
        .apply(
            lambda x:
                contains_any(
                    x,
                    patterns
                )
        )
    )


    subset = data[
        mask
    ]


    local_label_counts = {}


    for labels_list in subset[
        "_labels_list"
    ]:

        for label in labels_list:

            local_label_counts[label] = (
                local_label_counts.get(
                    label,
                    0
                )
                + 1
            )


    sorted_labels = sorted(
        local_label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    top_labels_text = (
        "; ".join(
            f"{label}:{count}"
            for label, count
            in sorted_labels[:15]
        )
    )


    # Number of distinct behavior texts.
    unique_texts = int(
        subset[
            "_behavior_lower"
        ]
        .nunique()
    )


    empty_texts = int(
        (
            subset[
                "_behavior_lower"
            ]
            .str.len()
            == 0
        ).sum()
    )


    text_records.append(
        {
            "utility":
                utility,

            "matching_behavior_rows":
                len(subset),

            "unique_behavior_texts":
                unique_texts,

            "empty_behavior_rows":
                empty_texts,

            "empty_behavior_percentage":
                percentage(
                    empty_texts,
                    len(subset)
                ),

            "top_label_distribution":
                top_labels_text,
        }
    )


text_df = pd.DataFrame(
    text_records
)


# =============================================================================
# 9. LABEL CONTEXT ANALYSIS
# =============================================================================
#
# For each MITRE label, determine which utilities occur in its examples.
# This helps distinguish:
#
#   utility -> almost always attack label
#
# from:
#
#   utility -> mixed labels
#
# =============================================================================

print()
print("-" * 80)
print("Analyzing MITRE label context")
print("-" * 80)


label_context_records = []


for technique in sorted(
    label_counts.keys()
):

    technique_mask = data[
        "_labels_list"
    ].apply(
        lambda labels_list:
            technique in labels_list
    )


    subset = data[
        technique_mask
    ]


    utility_counts = {}


    for utility, patterns in (
        UTILITY_PATTERNS.items()
    ):

        utility_mask = (
            subset[
                "process_name"
            ]
            .fillna("")
            .astype(str)
            .apply(
                lambda x:
                    contains_any(
                        x,
                        patterns
                    )
            )
            |
            subset[
                "parent_process_name"
            ]
            .fillna("")
            .astype(str)
            .apply(
                lambda x:
                    contains_any(
                        x,
                        patterns
                    )
            )
            |
            subset[
                "_behavior_lower"
            ]
            .apply(
                lambda x:
                    contains_any(
                        x,
                        patterns
                    )
            )
        )


        count = int(
            utility_mask.sum()
        )


        if count > 0:

            utility_counts[
                utility
            ] = count


    top_utilities = sorted(
        utility_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )


    label_context_records.append(
        {
            "technique":
                technique,

            "training_rows":
                int(
                    label_counts[
                        technique
                    ]
                ),

            "utilities_present":
                len(utility_counts),

            "top_utility_1":
                (
                    top_utilities[0][0]
                    if top_utilities
                    else ""
                ),

            "top_utility_1_count":
                (
                    top_utilities[0][1]
                    if top_utilities
                    else 0
                ),

            "top_utility_2":
                (
                    top_utilities[1][0]
                    if len(top_utilities) > 1
                    else ""
                ),

            "top_utility_2_count":
                (
                    top_utilities[1][1]
                    if len(top_utilities) > 1
                    else 0
                ),

            "top_utility_3":
                (
                    top_utilities[2][0]
                    if len(top_utilities) > 2
                    else ""
                ),

            "top_utility_3_count":
                (
                    top_utilities[2][1]
                    if len(top_utilities) > 2
                    else 0
                ),

            "utility_distribution":
                json.dumps(
                    utility_counts,
                    sort_keys=True
                ),
        }
    )


label_context_df = pd.DataFrame(
    label_context_records
)


# =============================================================================
# 10. IDENTIFY HIGH-RISK DATASET PATTERNS
# =============================================================================

print()
print("-" * 80)
print("Identifying potential dataset issues")
print("-" * 80)


# -------------------------------------------------------------------------
# A. Utilities strongly associated with a single label
# -------------------------------------------------------------------------

single_label_utilities = []


for _, row in utility_df.iterrows():

    if (
        row["matching_rows"] >= 10
        and row["top_label_1_percentage"] >= 90.0
    ):

        single_label_utilities.append(
            row["utility"]
        )


# -------------------------------------------------------------------------
# B. Utilities with broad label diversity
# -------------------------------------------------------------------------

high_label_diversity_utilities = []


for _, row in utility_df.iterrows():

    if (
        row["matching_rows"] >= 10
        and row["unique_labels"] >= 5
    ):

        high_label_diversity_utilities.append(
            row["utility"]
        )


# -------------------------------------------------------------------------
# C. Strong suspicious feature activation
# -------------------------------------------------------------------------

strong_feature_associations = []


for _, row in feature_df.iterrows():

    if (
        row["activated_rows"] >= 10
        and row["dominant_label_percentage"] >= 80.0
    ):

        strong_feature_associations.append(
            row["feature"]
        )


# -------------------------------------------------------------------------
# D. Empty behavior text
# -------------------------------------------------------------------------

empty_behavior_rate = percentage(
    int(
        (
            data[
                "_behavior_lower"
            ]
            .str.len()
            == 0
        ).sum()
    ),
    len(data)
)


# =============================================================================
# 11. COMMON LABELS IN UTILITY DATA
# =============================================================================

utility_label_union = {}


for _, row in utility_df.iterrows():

    distribution = json.loads(
        row[
            "label_distribution"
        ]
    )


    for label, count in distribution.items():

        utility_label_union[
            label
        ] = (
            utility_label_union.get(
                label,
                0
            )
            + int(count)
        )


top_utility_labels = sorted(
    utility_label_union.items(),
    key=lambda x: x[1],
    reverse=True
)


# =============================================================================
# 12. PRINT RESULTS
# =============================================================================

print()
print("=" * 80)
print("DATASET AUDIT RESULTS")
print("=" * 80)


print()
print(
    f"Training rows              : "
    f"{len(data)}"
)

print(
    f"Unique MITRE labels        : "
    f"{len(unique_labels)}"
)

print(
    f"Empty behavior-text rows   : "
    f"{empty_behavior_rate:.2f}%"
)


print()
print("-" * 80)
print("DUAL-USE UTILITY ANALYSIS")
print("-" * 80)


for _, row in utility_df.iterrows():

    if row[
        "matching_rows"
    ] == 0:

        continue


    print()
    print(
        f"{row['utility']}"
    )

    print(
        f"  Matching rows       : "
        f"{int(row['matching_rows'])}"
    )

    print(
        f"  Unique labels       : "
        f"{int(row['unique_labels'])}"
    )

    print(
        f"  Top label           : "
        f"{row['top_label_1']} "
        f"({row['top_label_1_percentage']:.2f}%)"
    )

    print(
        f"  Second label        : "
        f"{row['top_label_2']} "
        f"({row['top_label_2_percentage']:.2f}%)"
    )

    print(
        f"  Multi-label rows    : "
        f"{int(row['rows_with_multiple_labels'])}"
    )

    print(
        f"  Top feature         : "
        f"{row['top_suspicious_feature']} "
        f"({row['top_suspicious_feature_rate']:.2f}%)"
    )


print()
print("-" * 80)
print("POTENTIAL DATASET RISK INDICATORS")
print("-" * 80)


print()
print(
    "Utilities strongly dominated by one label:"
)

if single_label_utilities:

    for utility in (
        single_label_utilities
    ):

        print(
            f"  - {utility}"
        )

else:

    print(
        "  None"
    )


print()
print(
    "Utilities with broad label diversity:"
)

if high_label_diversity_utilities:

    for utility in (
        high_label_diversity_utilities
    ):

        print(
            f"  - {utility}"
        )

else:

    print(
        "  None"
    )


print()
print(
    "Suspicious features strongly associated "
    "with one label:"
)

if strong_feature_associations:

    for feature in (
        strong_feature_associations
    ):

        print(
            f"  - {feature}"
        )

else:

    print(
        "  None"
    )


print()
print("-" * 80)
print("MOST COMMON LABELS IN DUAL-USE UTILITY DATA")
print("-" * 80)


for label, count in (
    top_utility_labels[:20]
):

    print(
        f"  {label:<15} {count}"
    )


# =============================================================================
# 13. SAVE CSV REPORTS
# =============================================================================

print()
print("-" * 80)
print("Saving CSV reports")
print("-" * 80)


utility_df.to_csv(
    UTILITY_ANALYSIS_FILE,
    index=False
)

feature_df.to_csv(
    FEATURE_ANALYSIS_FILE,
    index=False
)

text_df.to_csv(
    TEXT_ANALYSIS_FILE,
    index=False
)

label_context_df.to_csv(
    LABEL_CONTEXT_FILE,
    index=False
)


# =============================================================================
# 14. BUILD JSON REPORT
# =============================================================================

report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3B",

    "gate_name":
        "Benign Dataset Coverage & Feature Semantics Audit",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "read_only":
        True,

    "production_artifacts_modified":
        False,

    "model_retrained":
        False,

    "thresholds_modified":
        False,

    "datasets_modified":
        False,

    "commands_executed":
        False,

    "training_rows":
        int(
            len(data)
        ),

    "unique_mitre_labels":
        int(
            len(unique_labels)
        ),

    "empty_behavior_text_rate":
        float(
            empty_behavior_rate
        ),

    "utility_analysis": {

        "utilities_analyzed":
            int(
                len(UTILITY_PATTERNS)
            ),

        "utilities_present_in_training":
            int(
                (
                    utility_df[
                        "matching_rows"
                    ]
                    > 0
                ).sum()
            ),

        "strong_single_label_utilities":
            single_label_utilities,

        "high_label_diversity_utilities":
            high_label_diversity_utilities,
    },

    "feature_analysis": {

        "features_analyzed":
            int(
                len(
                    feature_df
                )
            ),

        "strong_label_associations":
            strong_feature_associations,
    },

    "top_labels_in_dual_use_data":
        [
            {
                "technique":
                    label,

                "count":
                    int(count),
            }
            for label, count
            in top_utility_labels[:20]
        ],

    "interpretation_guidance": {

        "single_label_utility":
            "A dual-use utility appearing overwhelmingly "
            "with one MITRE label can cause strong lexical "
            "or categorical association in the classifier.",

        "high_label_diversity":
            "A utility appearing with many MITRE labels "
            "indicates contextual ambiguity and should not "
            "be handled using a simple hard-coded exclusion.",

        "empty_behavior_text":
            "A high empty-text rate reduces contextual "
            "information available to the TF-IDF component.",

        "feature_association":
            "A structured feature that is strongly associated "
            "with one label can dominate predictions even "
            "when the underlying activity is benign.",
    },

    "reports": {

        "utility_analysis":
            str(
                UTILITY_ANALYSIS_FILE
            ),

        "feature_analysis":
            str(
                FEATURE_ANALYSIS_FILE
            ),

        "text_analysis":
            str(
                TEXT_ANALYSIS_FILE
            ),

        "label_context":
            str(
                LABEL_CONTEXT_FILE
            ),
    },
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# =============================================================================
# 15. FINAL STATUS
# =============================================================================

print()
print("=" * 80)
print(
    "PHASE 2 GATE 2.3B STATUS: "
    "DIAGNOSTIC_COMPLETE"
)
print("=" * 80)

print()
print(
    f"JSON report : {REPORT_FILE}"
)

print(
    f"Utilities   : {UTILITY_ANALYSIS_FILE}"
)

print(
    f"Features    : {FEATURE_ANALYSIS_FILE}"
)

print(
    f"Text        : {TEXT_ANALYSIS_FILE}"
)

print(
    f"Labels      : {LABEL_CONTEXT_FILE}"
)

print()
print(
    "Production model artifacts were NOT modified."
)

print(
    "Thresholds were NOT modified."
)

print(
    "Training data was NOT modified."
)

print(
    "No Windows commands were executed."
)

print()
print(
    "Next action: review this diagnostic before "
    "any retraining or threshold changes."
)