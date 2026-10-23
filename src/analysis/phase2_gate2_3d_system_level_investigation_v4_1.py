import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT Model B v4.1
# PHASE 2 - GATE 2.3D
# SYSTEM_LEVEL NUMERICAL FEATURE INVESTIGATION
#
# READ-ONLY DIAGNOSTIC
#
# This script does NOT:
#   - retrain the model
#   - modify the model
#   - modify thresholds
#   - modify datasets
#   - execute Windows commands
#
# Purpose:
#   Determine why system_level produced a very large T1070 attribution
#   during Gate 2.3C.
# =============================================================================

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"C:\Users\Atharv\SmartSOC-AI"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3d"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase2_gate2_3d_system_level_report.json"
)

DISTRIBUTION_FILE = (
    REPORT_DIR
    / "system_level_distribution.csv"
)

LABEL_FILE = (
    REPORT_DIR
    / "system_level_label_distribution.csv"
)

COEFFICIENT_FILE = (
    REPORT_DIR
    / "system_level_coefficients.csv"
)

BENIGN_FILE = (
    REPORT_DIR
    / "system_level_benign_analysis.csv"
)


# =============================================================================
# ARTIFACTS
# =============================================================================

SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"


# =============================================================================
# DATA
# =============================================================================

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
LABELS_TRAIN = DATA_DIR / "labels_train.csv"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    print()
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def require_file(path):
    if not path.exists():
        fail(f"Missing required file: {path}")

    print(f"[OK] {path}")


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def normalize_label(value):
    return str(value).strip()


def label_contains(label, technique):
    parts = [
        x.strip()
        for x in str(label).split(",")
    ]

    return technique in parts


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3D")
print("SYSTEM_LEVEL NUMERICAL FEATURE INVESTIGATION")
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
    SCHEMA_FILE,
    ENCODER_FILE,
    SCALER_FILE,
    MODEL_FILE,
    MLB_FILE,
    STRUCTURED_TRAIN,
    LABELS_TRAIN,
    STRUCTURED_TEST,
    LABELS_TEST,
]:
    require_file(path)


# =============================================================================
# 2. LOAD SCHEMA
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


print(
    f"Categorical features : "
    f"{len(categorical_columns)}"
)

print(
    f"Numerical features   : "
    f"{len(numerical_columns)}"
)

print(
    f"Structured features  : "
    f"{len(structured_columns)}"
)


if "system_level" not in numerical_columns:

    fail(
        "system_level is NOT listed as a numerical "
        "feature in feature_schema.json."
    )


system_index = numerical_columns.index(
    "system_level"
)

print(
    f"system_level numerical index: "
    f"{system_index}"
)


# =============================================================================
# 3. LOAD DATA
# =============================================================================

print()
print("-" * 80)
print("Loading dataset")
print("-" * 80)

train = pd.read_csv(
    STRUCTURED_TRAIN
)

labels_train = pd.read_csv(
    LABELS_TRAIN
)

test = pd.read_csv(
    STRUCTURED_TEST
)

labels_test = pd.read_csv(
    LABELS_TEST
)


print(
    f"Training rows : {len(train)}"
)

print(
    f"Test rows     : {len(test)}"
)


# =============================================================================
# 4. VALIDATE SYSTEM_LEVEL
# =============================================================================

print()
print("-" * 80)
print("Validating system_level")
print("-" * 80)

train_system = pd.to_numeric(
    train["system_level"],
    errors="coerce"
)

test_system = pd.to_numeric(
    test["system_level"],
    errors="coerce"
)


print(
    f"Train missing : "
    f"{int(train_system.isna().sum())}"
)

print(
    f"Test missing  : "
    f"{int(test_system.isna().sum())}"
)


print()
print("TRAIN system_level statistics")

print(
    train_system.describe()
)


print()
print("TEST system_level statistics")

print(
    test_system.describe()
)


# =============================================================================
# 5. UNIQUE VALUES
# =============================================================================

print()
print("-" * 80)
print("Unique system_level values")
print("-" * 80)


train_values = (
    train_system
    .value_counts(dropna=False)
    .sort_index()
)

test_values = (
    test_system
    .value_counts(dropna=False)
    .sort_index()
)


print()
print("TRAIN")

for value, count in train_values.items():

    print(
        f"  {value}: {count}"
    )


print()
print("TEST")

for value, count in test_values.items():

    print(
        f"  {value}: {count}"
    )


# =============================================================================
# 6. DISTRIBUTION REPORT
# =============================================================================

distribution_rows = []


for split_name, series in [
    ("train", train_system),
    ("test", test_system),
]:

    counts = (
        series
        .value_counts(dropna=False)
        .sort_index()
    )

    for value, count in counts.items():

        distribution_rows.append(
            {
                "split":
                    split_name,

                "system_level":
                    safe_float(value),

                "row_count":
                    int(count),

                "row_percentage":
                    float(
                        count
                        / len(series)
                        * 100
                    ),
            }
        )


distribution_df = pd.DataFrame(
    distribution_rows
)


# =============================================================================
# 7. LABEL DISTRIBUTION
# =============================================================================

print()
print("-" * 80)
print("Analyzing labels by system_level")
print("-" * 80)


if "techniques" not in labels_train.columns:

    fail(
        "labels_train.csv does not contain "
        "the techniques column."
    )


combined_train = train.copy()

combined_train["techniques"] = (
    labels_train["techniques"]
    .astype(str)
)


label_rows = []


for system_value, group in (
    combined_train
    .groupby(
        "system_level",
        dropna=False
    )
):

    total = len(group)

    counts = (
        group["techniques"]
        .value_counts()
    )

    for label, count in counts.items():

        label_rows.append(
            {
                "system_level":
                    safe_float(system_value),

                "technique":
                    str(label),

                "row_count":
                    int(count),

                "percentage_within_system_level":
                    float(
                        count
                        / total
                        * 100
                    ),

                "total_rows_at_system_level":
                    int(total),
            }
        )


label_df = pd.DataFrame(
    label_rows
)


# =============================================================================
# 8. T1070 DISTRIBUTION
# =============================================================================

print()
print("-" * 80)
print("T1070 distribution across system_level")
print("-" * 80)


t1070_mask = (
    combined_train["techniques"]
    .apply(
        lambda x:
            label_contains(
                x,
                "T1070"
            )
    )
)


t1070_by_level = (
    combined_train[
        t1070_mask
    ]
    .groupby(
        "system_level",
        dropna=False
    )
    .size()
    .reset_index(
        name="t1070_rows"
    )
)


total_by_level = (
    combined_train
    .groupby(
        "system_level",
        dropna=False
    )
    .size()
    .reset_index(
        name="total_rows"
    )
)


t1070_analysis = (
    total_by_level
    .merge(
        t1070_by_level,
        on="system_level",
        how="left"
    )
)


t1070_analysis[
    "t1070_rows"
] = (
    t1070_analysis[
        "t1070_rows"
    ]
    .fillna(0)
)


t1070_analysis[
    "t1070_percentage"
] = (
    t1070_analysis[
        "t1070_rows"
    ]
    /
    t1070_analysis[
        "total_rows"
    ]
    * 100
)


for _, row in (
    t1070_analysis
    .sort_values(
        "t1070_percentage",
        ascending=False
    )
    .iterrows()
):

    print(
        f"system_level={row['system_level']} "
        f"total={int(row['total_rows'])} "
        f"T1070={int(row['t1070_rows'])} "
        f"T1070_pct={row['t1070_percentage']:.4f}%"
    )


# =============================================================================
# 9. LOAD MODEL ARTIFACTS
# =============================================================================

print()
print("-" * 80)
print("Loading production artifacts")
print("-" * 80)


encoder = joblib.load(
    ENCODER_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

model = joblib.load(
    MODEL_FILE
)

mlb = joblib.load(
    MLB_FILE
)


mitre_classes = [
    str(x)
    for x in mlb.classes_
]


if "T1070" not in mitre_classes:

    fail(
        "T1070 is not present in MultiLabelBinarizer classes."
    )


t1070_index = (
    mitre_classes.index(
        "T1070"
    )
)


t1070_estimator = (
    model.estimators_[
        t1070_index
    ]
)


print(
    f"T1070 model class index : "
    f"{t1070_index}"
)


# =============================================================================
# 10. INSPECT NUMERICAL SCALER
# =============================================================================

print()
print("-" * 80)
print("Inspecting numerical scaler")
print("-" * 80)


if not hasattr(
    scaler,
    "mean_"
):

    fail(
        "Production numerical scaler does not expose mean_."
    )


if not hasattr(
    scaler,
    "scale_"
):

    fail(
        "Production numerical scaler does not expose scale_."
    )


if len(scaler.mean_) != len(
    numerical_columns
):

    fail(
        "Scaler feature count does not match "
        "numerical_columns."
    )


system_mean = float(
    scaler.mean_[
        system_index
    ]
)

system_scale = float(
    scaler.scale_[
        system_index
    ]
)


print(
    f"system_level scaler mean  : "
    f"{system_mean:.12f}"
)

print(
    f"system_level scaler scale : "
    f"{system_scale:.12f}"
)


# =============================================================================
# 11. T1070 NUMERICAL COEFFICIENT
# =============================================================================

print()
print("-" * 80)
print("Inspecting T1070 numerical coefficient")
print("-" * 80)


t1070_coef = (
    np.asarray(
        t1070_estimator.coef_
    )
    .reshape(-1)
)


expected_numerical_count = (
    len(numerical_columns)
)


print(
    f"T1070 coefficient count : "
    f"{len(t1070_coef)}"
)


print(
    f"Numerical feature count : "
    f"{expected_numerical_count}"
)


# The final model vector is:
#
#   categorical encoded features
#   +
#   scaled numerical features
#   +
#   TF-IDF features
#
# Therefore system_level does NOT necessarily occupy numerical index
# system_index in the final vector.
#
# Determine the final-vector offset from encoder output.

categorical_feature_count = len(
    encoder.get_feature_names_out()
)

numerical_offset = (
    categorical_feature_count
)

system_final_index = (
    numerical_offset
    + system_index
)


print(
    f"Categorical encoded features : "
    f"{categorical_feature_count}"
)

print(
    f"Numerical offset              : "
    f"{numerical_offset}"
)

print(
    f"system_level final index      : "
    f"{system_final_index}"
)


if system_final_index >= len(
    t1070_coef
):

    fail(
        "Calculated system_level final feature index "
        "is outside the T1070 coefficient vector."
    )


system_coefficient = float(
    t1070_coef[
        system_final_index
    ]
)


print(
    f"T1070 system_level coefficient: "
    f"{system_coefficient:.12f}"
)


# =============================================================================
# 12. CALCULATE CONTRIBUTION FOR VALUES
# =============================================================================

print()
print("-" * 80)
print("Calculating system_level contribution")
print("-" * 80)


candidate_values = sorted(
    set(
        [
            0.0,
            2.0,
            4.0,
            12.0
        ]
        + [
            float(x)
            for x in train_system
            .dropna()
            .unique()
        ]
        + [
            float(x)
            for x in test_system
            .dropna()
            .unique()
        ]
    )
)


contribution_rows = []


for raw_value in candidate_values:

    scaled_value = (
        raw_value
        - system_mean
    ) / system_scale


    contribution = (
        scaled_value
        * system_coefficient
    )


    contribution_rows.append(
        {
            "raw_system_level":
                raw_value,

            "scaled_system_level":
                float(
                    scaled_value
                ),

            "t1070_coefficient":
                system_coefficient,

            "t1070_contribution":
                float(
                    contribution
                ),
        }
    )


contribution_df = pd.DataFrame(
    contribution_rows
)


for _, row in (
    contribution_df
    .iterrows()
):

    print(
        f"raw={row['raw_system_level']:<8} "
        f"scaled={row['scaled_system_level']:>12.6f} "
        f"coef={row['t1070_coefficient']:>14.8f} "
        f"contribution={row['t1070_contribution']:>14.6f}"
    )


# =============================================================================
# 13. CHECK TRAINING VALUES AGAINST MODEL CONTRIBUTION
# =============================================================================

print()
print("-" * 80)
print("Training system_level contribution range")
print("-" * 80)


train_contributions = []


for value in train_system.dropna():

    raw_value = float(value)

    scaled_value = (
        raw_value
        - system_mean
    ) / system_scale

    contribution = (
        scaled_value
        * system_coefficient
    )

    train_contributions.append(
        contribution
    )


if train_contributions:

    print(
        f"Minimum contribution : "
        f"{min(train_contributions):.8f}"
    )

    print(
        f"Maximum contribution : "
        f"{max(train_contributions):.8f}"
    )

    print(
        f"Mean contribution    : "
        f"{np.mean(train_contributions):.8f}"
    )


# =============================================================================
# 14. CONTROLLED BENIGN VALUE ANALYSIS
# =============================================================================

print()
print("-" * 80)
print("Controlled benign system_level analysis")
print("-" * 80)


benign_values = [
    12.0,
    4.0,
    2.0,
    0.0,
]


benign_rows = []


for raw_value in benign_values:

    scaled_value = (
        raw_value
        - system_mean
    ) / system_scale


    contribution = (
        scaled_value
        * system_coefficient
    )


    benign_rows.append(
        {
            "controlled_raw_system_level":
                raw_value,

            "scaled_system_level":
                float(
                    scaled_value
                ),

            "t1070_coefficient":
                system_coefficient,

            "t1070_contribution":
                float(
                    contribution
                ),

            "present_in_training":
                bool(
                    raw_value
                    in set(
                        train_system
                        .dropna()
                        .astype(float)
                        .tolist()
                    )
                ),

            "present_in_test":
                bool(
                    raw_value
                    in set(
                        test_system
                        .dropna()
                        .astype(float)
                        .tolist()
                    )
                ),
        }
    )


benign_df = pd.DataFrame(
    benign_rows
)


for _, row in (
    benign_df
    .iterrows()
):

    print(
        f"raw={row['controlled_raw_system_level']:<6} "
        f"scaled={row['scaled_system_level']:>12.6f} "
        f"contribution={row['t1070_contribution']:>14.6f} "
        f"train={row['present_in_training']} "
        f"test={row['present_in_test']}"
    )


# =============================================================================
# 15. FIND ALL LARGE T1070 COEFFICIENTS
# =============================================================================

print()
print("-" * 80)
print("Largest T1070 numerical coefficients")
print("-" * 80)


numerical_coefficient_rows = []


for index, feature_name in enumerate(
    numerical_columns
):

    final_index = (
        numerical_offset
        + index
    )


    coefficient = float(
        t1070_coef[
            final_index
        ]
    )


    numerical_coefficient_rows.append(
        {
            "feature":
                feature_name,

            "numerical_index":
                int(index),

            "final_feature_index":
                int(final_index),

            "t1070_coefficient":
                coefficient,

            "absolute_coefficient":
                abs(coefficient),
        }
    )


numerical_coefficient_df = (
    pd.DataFrame(
        numerical_coefficient_rows
    )
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)


for _, row in (
    numerical_coefficient_df
    .head(30)
    .iterrows()
):

    print(
        f"{row['feature']:<45} "
        f"coef={row['t1070_coefficient']:>14.8f}"
    )


# =============================================================================
# 16. LABEL PRIOR FOR SYSTEM_LEVEL VALUES
# =============================================================================

print()
print("-" * 80)
print("Calculating T1070 prior by system_level")
print("-" * 80)


prior_rows = []


for raw_value in candidate_values:

    group = combined_train[
        combined_train[
            "system_level"
        ] == raw_value
    ]


    total = len(group)


    if total == 0:

        t1070_count = 0
        percentage = 0.0

    else:

        mask = (
            group[
                "techniques"
            ]
            .apply(
                lambda x:
                    label_contains(
                        x,
                        "T1070"
                    )
            )
        )

        t1070_count = int(
            mask.sum()
        )

        percentage = (
            t1070_count
            / total
            * 100
        )


    prior_rows.append(
        {
            "system_level":
                raw_value,

            "training_rows":
                int(total),

            "t1070_rows":
                int(t1070_count),

            "t1070_percentage":
                float(percentage),
        }
    )


prior_df = pd.DataFrame(
    prior_rows
)


for _, row in (
    prior_df
    .iterrows()
):

    print(
        f"system_level={row['system_level']:<6} "
        f"rows={int(row['training_rows']):<8} "
        f"T1070={int(row['t1070_rows']):<6} "
        f"pct={row['t1070_percentage']:.6f}%"
    )


# =============================================================================
# 17. SAVE CSV REPORTS
# =============================================================================

print()
print("-" * 80)
print("Saving reports")
print("-" * 80)


distribution_df.to_csv(
    DISTRIBUTION_FILE,
    index=False
)


label_df.to_csv(
    LABEL_FILE,
    index=False
)


# Combine numerical coefficients with system-level-specific metadata.

coefficient_df = (
    numerical_coefficient_df
    .copy()
)


coefficient_df[
    "system_level_feature"
] = (
    coefficient_df[
        "feature"
    ]
    == "system_level"
)


coefficient_df.to_csv(
    COEFFICIENT_FILE,
    index=False
)


benign_df.to_csv(
    BENIGN_FILE,
    index=False
)


# =============================================================================
# 18. DIAGNOSTIC CONCLUSION
# =============================================================================

system_present_train = bool(
    12.0
    in set(
        train_system
        .dropna()
        .astype(float)
        .tolist()
    )
)


system_present_test = bool(
    12.0
    in set(
        test_system
        .dropna()
        .astype(float)
        .tolist()
    )
)


system_12_contribution = float(
    benign_df[
        benign_df[
            "controlled_raw_system_level"
        ] == 12.0
    ][
        "t1070_contribution"
    ].iloc[0]
)


system_4_contribution = float(
    benign_df[
        benign_df[
            "controlled_raw_system_level"
        ] == 4.0
    ][
        "t1070_contribution"
    ].iloc[0]
)


system_0_contribution = float(
    benign_df[
        benign_df[
            "controlled_raw_system_level"
        ] == 0.0
    ][
        "t1070_contribution"
    ].iloc[0]
)


if abs(system_12_contribution) > 5:

    conclusion = (
        "system_level=12 produces a large T1070 model "
        "contribution even though value 12 is absent "
        "from both the training and test distributions. "
        "This indicates the controlled benign feature "
        "value is out-of-distribution and may be a major "
        "source of the observed false positive."
    )

elif not system_present_train:

    conclusion = (
        "system_level=12 is absent from training. "
        "Its model contribution should not be interpreted "
        "as evidence that system_level=12 is a learned "
        "normal or malicious operating state."
    )

else:

    conclusion = (
        "system_level requires further investigation. "
        "The observed T1070 contribution is not explained "
        "solely by an out-of-distribution system_level=12 "
        "value."
    )


# =============================================================================
# 19. JSON REPORT
# =============================================================================

report = {

    "phase":
        "Phase 2",

    "gate":
        "Gate 2.3D",

    "gate_name":
        "system_level Numerical Feature Investigation",

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

    "dataset": {

        "training_rows":
            int(len(train)),

        "test_rows":
            int(len(test)),

        "train_system_level_values":
            [
                safe_float(x)
                for x in train_values.index
            ],

        "test_system_level_values":
            [
                safe_float(x)
                for x in test_values.index
            ],
    },

    "feature_schema": {

        "categorical_features":
            len(categorical_columns),

        "numerical_features":
            len(numerical_columns),

        "system_level_type":
            "numerical",

        "system_level_numerical_index":
            int(system_index),

        "categorical_system_level_features":
            0,
    },

    "scaler": {

        "system_level_mean":
            system_mean,

        "system_level_scale":
            system_scale,
    },

    "model": {

        "t1070_present":
            True,

        "t1070_class_index":
            int(t1070_index),

        "categorical_encoded_features":
            int(categorical_feature_count),

        "system_level_final_feature_index":
            int(system_final_index),

        "t1070_system_level_coefficient":
            system_coefficient,
    },

    "controlled_benign": {

        "system_level_12_present_in_training":
            system_present_train,

        "system_level_12_present_in_test":
            system_present_test,

        "system_level_12_t1070_contribution":
            system_12_contribution,

        "system_level_4_t1070_contribution":
            system_4_contribution,

        "system_level_0_t1070_contribution":
            system_0_contribution,
    },

    "conclusion":
        conclusion,

    "files": {

        "distribution":
            str(DISTRIBUTION_FILE),

        "label_distribution":
            str(LABEL_FILE),

        "coefficients":
            str(COEFFICIENT_FILE),

        "benign_analysis":
            str(BENIGN_FILE),
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
# FINAL
# =============================================================================

print()
print("=" * 80)
print("PHASE 2 GATE 2.3D STATUS: DIAGNOSTIC_COMPLETE")
print("=" * 80)

print()
print(
    "IMPORTANT FINDING:"
)

print(
    f"Training system_level values : "
    f"{list(train_values.index)}"
)

print(
    f"Test system_level values     : "
    f"{list(test_values.index)}"
)

print(
    f"system_level=12 in training  : "
    f"{system_present_train}"
)

print(
    f"system_level=12 in test      : "
    f"{system_present_test}"
)

print(
    f"T1070 coefficient            : "
    f"{system_coefficient:.12f}"
)

print(
    f"T1070 contribution @ 12      : "
    f"{system_12_contribution:.8f}"
)

print(
    f"T1070 contribution @ 4       : "
    f"{system_4_contribution:.8f}"
)

print(
    f"T1070 contribution @ 0       : "
    f"{system_0_contribution:.8f}"
)

print()
print(
    "Conclusion:"
)

print(
    conclusion
)

print()
print(
    f"JSON report : {REPORT_FILE}"
)

print(
    f"Distribution: {DISTRIBUTION_FILE}"
)

print(
    f"Labels      : {LABEL_FILE}"
)

print(
    f"Coefficients: {COEFFICIENT_FILE}"
)

print(
    f"Benign      : {BENIGN_FILE}"
)

print()
print(
    "Production artifacts were NOT modified."
)

print(
    "Training data was NOT modified."
)

print(
    "Thresholds were NOT modified."
)

print(
    "No Windows commands were executed."
)