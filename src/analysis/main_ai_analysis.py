"""
SmartSOC-AI
Master AI Dataset Analysis Pipeline

Purpose:
    Generate a complete analysis of the Windows-APT v4.1 dataset
    in one execution.

Outputs:
    - Dataset statistics
    - Feature statistics
    - Label distribution
    - Outlier analysis
    - Correlation matrix
    - Correlation heatmap
    - Outlier heatmap
    - Feature variance analysis
    - Feature distribution
    - JSON summary
    - Human-readable summary

This script DOES NOT modify:
    - Training data
    - Model artifacts
    - Production inference artifacts
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
STRUCTURED_TEST = DATA_DIR / "structured_test.csv"

LABELS_TRAIN = DATA_DIR / "labels_train.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"

METADATA_TRAIN = DATA_DIR / "metadata_train.csv"
METADATA_TEST = DATA_DIR / "metadata_test.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "ai_analysis"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

warnings.filterwarnings("ignore")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def save_dataframe(df, filename):
    path = REPORT_DIR / filename
    df.to_csv(path, index=True)
    return path


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ============================================================
# FILE VALIDATION
# ============================================================

section("SMARTSOC-AI MASTER DATASET ANALYSIS")

print("Project root:")
print(PROJECT_ROOT)

print()
print("Checking required files...")

required_files = [
    STRUCTURED_TRAIN,
    STRUCTURED_TEST,
    LABELS_TRAIN,
    LABELS_TEST,
]

missing_files = []

for path in required_files:

    if path.exists():
        print("[OK]", path)

    else:
        print("[MISSING]", path)
        missing_files.append(path)


if missing_files:

    print()
    print("ERROR: Required dataset files are missing.")

    for path in missing_files:
        print(" -", path)

    raise SystemExit(1)


# ============================================================
# LOAD DATASETS
# ============================================================

section("LOADING DATASETS")

structured_train = pd.read_csv(
    STRUCTURED_TRAIN
)

structured_test = pd.read_csv(
    STRUCTURED_TEST
)

labels_train = pd.read_csv(
    LABELS_TRAIN
)

labels_test = pd.read_csv(
    LABELS_TEST
)

print(
    "Structured train:",
    structured_train.shape
)

print(
    "Structured test :",
    structured_test.shape
)

print(
    "Labels train    :",
    labels_train.shape
)

print(
    "Labels test     :",
    labels_test.shape
)


# ============================================================
# DATASET ALIGNMENT
# ============================================================

section("DATASET ALIGNMENT")

print(
    "Training rows:",
    len(structured_train)
)

print(
    "Training label rows:",
    len(labels_train)
)

print(
    "Test rows:",
    len(structured_test)
)

print(
    "Test label rows:",
    len(labels_test)
)

if len(structured_train) == len(labels_train):
    print("[PASS] Training X/Y alignment")
else:
    print("[FAIL] Training X/Y alignment")

if len(structured_test) == len(labels_test):
    print("[PASS] Test X/Y alignment")
else:
    print("[FAIL] Test X/Y alignment")


# ============================================================
# FEATURE ANALYSIS
# ============================================================

section("FEATURE ANALYSIS")

all_features = list(
    structured_train.columns
)

numeric_df = structured_train.select_dtypes(
    include=np.number
)

categorical_df = structured_train.select_dtypes(
    exclude=np.number
)

print(
    "Total structured features:",
    len(all_features)
)

print(
    "Numerical features:",
    len(numeric_df.columns)
)

print(
    "Categorical features:",
    len(categorical_df.columns)
)

print()

print("Feature list:")

for i, feature in enumerate(
    all_features,
    start=1
):

    print(
        f"{i:03d}. {feature}"
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

section("FEATURE STATISTICS")

feature_stats = pd.DataFrame(
    index=numeric_df.columns
)

feature_stats["dtype"] = (
    numeric_df.dtypes.astype(str)
)

feature_stats["missing_count"] = (
    numeric_df.isna().sum()
)

feature_stats["missing_percent"] = (
    numeric_df.isna().mean() * 100
)

feature_stats["unique_values"] = (
    numeric_df.nunique()
)

feature_stats["mean"] = (
    numeric_df.mean()
)

feature_stats["std"] = (
    numeric_df.std()
)

feature_stats["min"] = (
    numeric_df.min()
)

feature_stats["max"] = (
    numeric_df.max()
)

feature_stats["variance"] = (
    numeric_df.var()
)

feature_stats = feature_stats.sort_index()

feature_stats.to_csv(
    REPORT_DIR / "feature_statistics.csv"
)

print(
    "[OK] feature_statistics.csv"
)


# ============================================================
# MISSING VALUES
# ============================================================

section("MISSING VALUE ANALYSIS")

missing = (
    structured_train.isna().sum()
    .sort_values(ascending=False)
)

missing_percent = (
    structured_train.isna().mean() * 100
).sort_values(
    ascending=False
)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
})

missing_report.to_csv(
    REPORT_DIR / "missing_values.csv"
)

print(
    missing_report[
        missing_report["missing_count"] > 0
    ].head(20)
)

if missing.sum() == 0:
    print()
    print("[PASS] No missing values found.")


# ============================================================
# CONSTANT / LOW VARIANCE FEATURES
# ============================================================

section("CONSTANT / LOW VARIANCE FEATURES")

constant_features = []

for column in numeric_df.columns:

    if numeric_df[column].nunique() <= 1:
        constant_features.append(column)

print(
    "Constant features:",
    len(constant_features)
)

for feature in constant_features:
    print(
        " -",
        feature
    )

low_variance = (
    numeric_df.var()
    .sort_values()
)

low_variance.to_csv(
    REPORT_DIR / "feature_variance.csv"
)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

section("OUTLIER ANALYSIS — IQR METHOD")

Q1 = numeric_df.quantile(
    0.25
)

Q3 = numeric_df.quantile(
    0.75
)

IQR = Q3 - Q1

lower_bound = (
    Q1 - 1.5 * IQR
)

upper_bound = (
    Q3 + 1.5 * IQR
)

outlier_mask = (
    numeric_df.lt(lower_bound)
    |
    numeric_df.gt(upper_bound)
)

outlier_count = (
    outlier_mask.sum()
    .sort_values(
        ascending=False
    )
)

outlier_percent = (
    outlier_mask.mean()
    * 100
)

outlier_summary = pd.DataFrame({
    "outlier_count": outlier_count,
    "outlier_percent": outlier_percent
})

outlier_summary.to_csv(
    REPORT_DIR / "outlier_summary.csv"
)

total_outlier_values = int(
    outlier_mask.sum().sum()
)

rows_with_outliers = int(
    outlier_mask.any(axis=1).sum()
)

features_with_outliers = int(
    outlier_mask.any(axis=0).sum()
)

print(
    "Total outlier values:",
    total_outlier_values
)

print(
    "Rows with >=1 outlier:",
    rows_with_outliers
)

print(
    "Features with outliers:",
    features_with_outliers
)

print()

print(
    "Top outlier-producing features:"
)

print(
    outlier_summary.head(20)
)


# ============================================================
# OUTLIER HEATMAP
# ============================================================

section("GENERATING OUTLIER HEATMAP")

# Limit visualization to features that actually contain outliers.
outlier_features = [
    column
    for column in numeric_df.columns
    if outlier_mask[column].any()
]

if outlier_features:

    # To keep the visualization readable, show the first
    # 1000 training events.
    display_rows = min(
        len(numeric_df),
        1000
    )

    outlier_visual = (
        outlier_mask[
            outlier_features
        ]
        .iloc[:display_rows]
        .astype(int)
    )

    plt.figure(
        figsize=(18, 10)
    )

    plt.imshow(
        outlier_visual.T,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar(
        label="Outlier (1 = Yes)"
    )

    plt.xlabel(
        "Training Event"
    )

    plt.ylabel(
        "Feature"
    )

    plt.yticks(
        range(len(outlier_features)),
        outlier_features,
        fontsize=7
    )

    plt.title(
        "SmartSOC-AI v4.1 — Outlier Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "outlier_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[OK] outlier_heatmap.png"
    )

else:

    print(
        "[INFO] No outlier features found."
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

section("CORRELATION ANALYSIS")

correlation = numeric_df.corr(
    method="pearson"
)

correlation.to_csv(
    REPORT_DIR
    / "correlation_matrix.csv"
)

print(
    "[OK] correlation_matrix.csv"
)


# ============================================================
# HIGH CORRELATION PAIRS
# ============================================================

section("HIGH CORRELATION PAIRS")

pairs = []

columns = list(
    correlation.columns
)

for i in range(
    len(columns)
):

    for j in range(
        i + 1,
        len(columns)
    ):

        value = correlation.iloc[
            i,
            j
        ]

        if pd.notna(value):

            if abs(value) >= 0.90:

                pairs.append({
                    "feature_1":
                        columns[i],

                    "feature_2":
                        columns[j],

                    "correlation":
                        value
                })

high_corr = pd.DataFrame(
    pairs
)

if not high_corr.empty:

    high_corr = high_corr.sort_values(
        "correlation",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(
        high_corr.head(30).to_string(
            index=False
        )
    )

else:

    print(
        "No feature pairs with |correlation| >= 0.90"
    )

high_corr.to_csv(
    REPORT_DIR
    / "high_correlation_pairs.csv",
    index=False
)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

section("GENERATING CORRELATION HEATMAP")

plt.figure(
    figsize=(24, 20)
)

plt.imshow(
    correlation.values,
    aspect="auto",
    interpolation="nearest",
    vmin=-1,
    vmax=1
)

plt.colorbar(
    label="Pearson Correlation"
)

plt.xticks(
    range(len(columns)),
    columns,
    rotation=90,
    fontsize=6
)

plt.yticks(
    range(len(columns)),
    columns,
    fontsize=6
)

plt.title(
    "SmartSOC-AI v4.1 — Numerical Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "[OK] correlation_heatmap.png"
)


# ============================================================
# LABEL ANALYSIS
# ============================================================

section("MITRE LABEL ANALYSIS")

if "techniques" in labels_train.columns:

    import re
    from collections import Counter

    technique_counter = Counter()

    rows_with_labels = 0

    for value in labels_train[
        "techniques"
    ].fillna(""):

        text = str(value)

        techniques = re.findall(
            r"T\d{4}(?:\.\d{3})?",
            text
        )

        if techniques:
            rows_with_labels += 1

        for technique in set(
            techniques
        ):
            technique_counter[
                technique
            ] += 1

    label_distribution = pd.DataFrame(
        technique_counter.items(),
        columns=[
            "technique",
            "event_count"
        ]
    )

    label_distribution = (
        label_distribution
        .sort_values(
            "event_count",
            ascending=False
        )
        .reset_index(drop=True)
    )

    label_distribution[
        "percentage"
    ] = (
        label_distribution[
            "event_count"
        ]
        / len(labels_train)
        * 100
    )

    label_distribution.to_csv(
        REPORT_DIR
        / "label_distribution.csv",
        index=False
    )

    print(
        "Training rows:",
        len(labels_train)
    )

    print(
        "Rows containing labels:",
        rows_with_labels
    )

    print(
        "Unique MITRE techniques:",
        len(label_distribution)
    )

    print()

    print(
        label_distribution.head(30).to_string(
            index=False
        )
    )

else:

    print(
        "[WARNING] 'techniques' column not found."
    )

    label_distribution = pd.DataFrame()


# ============================================================
# LABEL DISTRIBUTION PLOT
# ============================================================

if not label_distribution.empty:

    section("GENERATING LABEL DISTRIBUTION PLOT")

    top_labels = (
        label_distribution
        .head(20)
        .sort_values(
            "event_count"
        )
    )

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        top_labels["technique"],
        top_labels["event_count"]
    )

    plt.xlabel(
        "Number of Training Events"
    )

    plt.ylabel(
        "MITRE ATT&CK Technique"
    )

    plt.title(
        "SmartSOC-AI v4.1 — Top 20 MITRE Techniques"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "label_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[OK] label_distribution.png"
    )


# ============================================================
# FEATURE VARIANCE PLOT
# ============================================================

section("GENERATING FEATURE VARIANCE PLOT")

variance_sorted = (
    numeric_df.var()
    .sort_values(
        ascending=False
    )
)

top_variance = (
    variance_sorted
    .head(20)
    .sort_values()
)

plt.figure(
    figsize=(12, 8)
)

plt.barh(
    top_variance.index,
    top_variance.values
)

plt.xlabel(
    "Variance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "SmartSOC-AI v4.1 — Top 20 Feature Variances"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR
    / "feature_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "[OK] feature_variance.png"
)


# ============================================================
# BINARY FEATURE ANALYSIS
# ============================================================

section("BINARY FEATURE ANALYSIS")

binary_features = []

for column in numeric_df.columns:

    unique_values = set(
        numeric_df[column]
        .dropna()
        .unique()
    )

    if unique_values.issubset(
        {0, 1}
    ):

        binary_features.append(
            column
        )

print(
    "Binary features:",
    len(binary_features)
)

for feature in binary_features:

    count = int(
        numeric_df[
            feature
        ].sum()
    )

    print(
        f"{feature:<40} {count:>8}"
    )


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

section("SMARTSOC BEHAVIORAL SIGNALS")

behavioral_features = [
    "uses_powershell",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "administrative_powershell_context",
    "strong_behavior_signal",
]

behavioral_report = []

for feature in behavioral_features:

    if feature in structured_train.columns:

        count = int(
            structured_train[
                feature
            ].sum()
        )

        percentage = (
            count
            / len(structured_train)
            * 100
        )

        behavioral_report.append({
            "feature":
                feature,

            "event_count":
                count,

            "percentage":
                percentage
        })

behavioral_df = pd.DataFrame(
    behavioral_report
)

if not behavioral_df.empty:

    print(
        behavioral_df.to_string(
            index=False
        )
    )

    behavioral_df.to_csv(
        REPORT_DIR
        / "behavioral_signals.csv",
        index=False
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

section("FINAL DATASET SUMMARY")

summary = {

    "project":
        "SmartSOC-AI",

    "dataset":
        "Windows-APT v4.1",

    "training_rows":
        int(len(structured_train)),

    "test_rows":
        int(len(structured_test)),

    "structured_features":
        int(len(structured_train.columns)),

    "numerical_features":
        int(len(numeric_df.columns)),

    "categorical_features":
        int(len(categorical_df.columns)),

    "total_outlier_values":
        total_outlier_values,

    "rows_with_outliers":
        rows_with_outliers,

    "features_with_outliers":
        features_with_outliers,

    "constant_features":
        len(constant_features),

    "high_correlation_pairs":
        len(high_corr),

    "binary_features":
        len(binary_features),

    "unique_mitre_techniques":
        (
            len(label_distribution)
            if not label_distribution.empty
            else 0
        ),
}


# ============================================================
# JSON REPORT
# ============================================================

with open(
    REPORT_DIR / "analysis_report.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        summary,
        handle,
        indent=4
    )


# ============================================================
# TEXT REPORT
# ============================================================

with open(
    REPORT_DIR / "summary.txt",
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "SMARTSOC-AI v4.1\n"
    )

    handle.write(
        "MASTER DATASET ANALYSIS REPORT\n"
    )

    handle.write(
        "=" * 70
        + "\n\n"
    )

    for key, value in summary.items():

        handle.write(
            f"{key}: {value}\n"
        )

    handle.write(
        "\n\n"
    )

    handle.write(
        "Behavioral Signals\n"
    )

    handle.write(
        "-" * 70
        + "\n"
    )

    if not behavioral_df.empty:

        handle.write(
            behavioral_df.to_string(
                index=False
            )
        )

    handle.write(
        "\n\n"
    )

    handle.write(
        "Top MITRE Techniques\n"
    )

    handle.write(
        "-" * 70
        + "\n"
    )

    if not label_distribution.empty:

        handle.write(
            label_distribution
            .head(30)
            .to_string(index=False)
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

section("ANALYSIS COMPLETED")

print(
    "Reports generated in:"
)

print(
    REPORT_DIR
)

print()

for path in sorted(
    REPORT_DIR.iterdir()
):

    if path.is_file():

        print(
            "[OUTPUT]",
            path.name
        )

print()
print(
    "SmartSOC-AI dataset analysis completed successfully."
)

print(
    "No training data or production model artifacts were modified."
)