from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
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

CONFLICT_FILE = (
    REPORT_DIR
    / "train_test_label_conflicts.csv"
)

OUTPUT_FILE = (
    REPORT_DIR
    / "conflicting_rows_62_feature_comparison.csv"
)

OUTPUT_SUMMARY = (
    REPORT_DIR
    / "conflicting_rows_feature_difference_summary.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("SmartSOC-AI v4.1")
print("FULL 62-FEATURE CONFLICT COMPARISON")
print("=" * 75)


# ============================================================
# 1. CHECK FILES
# ============================================================

print()
print("[1] Checking required files...")

required_files = [
    STRUCTURED_TRAIN,
    STRUCTURED_TEST,
    LABELS_TRAIN,
    LABELS_TEST,
    CONFLICT_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        f"[FOUND] {file_path}"
    )


# ============================================================
# 2. LOAD DATA
# ============================================================

print()
print("[2] Loading datasets...")

train_x = pd.read_csv(
    STRUCTURED_TRAIN,
    low_memory=False,
)

test_x = pd.read_csv(
    STRUCTURED_TEST,
    low_memory=False,
)

train_y = pd.read_csv(
    LABELS_TRAIN,
    low_memory=False,
)

test_y = pd.read_csv(
    LABELS_TEST,
    low_memory=False,
)

conflicts = pd.read_csv(
    CONFLICT_FILE,
    low_memory=False,
)


print(
    f"Train structured: {train_x.shape}"
)

print(
    f"Test structured : {test_x.shape}"
)

print(
    f"Conflicts       : {len(conflicts)}"
)


# ============================================================
# 3. VALIDATE SCHEMA
# ============================================================

print()
print("[3] Validating structured schema...")

if list(train_x.columns) != list(test_x.columns):

    raise ValueError(
        "Train/test structured columns do not match."
    )


feature_columns = list(
    train_x.columns
)

print(
    f"[PASS] Structured feature count: "
    f"{len(feature_columns)}"
)


if len(feature_columns) != 62:

    print(
        "[WARNING] Expected 62 features, "
        f"but found {len(feature_columns)}."
    )


# ============================================================
# 4. COMPARE EACH CONFLICT
# ============================================================

print()
print("=" * 75)
print("FULL FEATURE COMPARISON")
print("=" * 75)


all_records = []
summary_records = []


for conflict_number, conflict_row in conflicts.iterrows():

    conflict_id = conflict_number + 1

    train_index = int(
        conflict_row["train_index"]
    )

    test_index = int(
        conflict_row["test_index"]
    )

    train_label = str(
        conflict_row["train_label"]
    )

    test_label = str(
        conflict_row["test_label"]
    )


    train_row = train_x.iloc[
        train_index
    ]

    test_row = test_x.iloc[
        test_index
    ]


    print()
    print("-" * 75)

    print(
        f"CONFLICT #{conflict_id}"
    )

    print(
        f"Train index : {train_index}"
    )

    print(
        f"Test index  : {test_index}"
    )

    print(
        f"Train label : {train_label}"
    )

    print(
        f"Test label  : {test_label}"
    )

    print("-" * 75)


    differences = 0


    for feature in feature_columns:

        train_value = train_row[
            feature
        ]

        test_value = test_row[
            feature
        ]


        # ----------------------------------------------------
        # Robust comparison
        # ----------------------------------------------------

        if pd.isna(train_value) and pd.isna(test_value):

            equal = True

        elif pd.isna(train_value) != pd.isna(test_value):

            equal = False

        else:

            equal = (
                train_value == test_value
            )


        if not equal:

            differences += 1


        record = {
            "conflict_number":
                conflict_id,

            "train_index":
                train_index,

            "test_index":
                test_index,

            "train_label":
                train_label,

            "test_label":
                test_label,

            "feature":
                feature,

            "train_value":
                train_value,

            "test_value":
                test_value,

            "values_equal":
                equal,
        }


        all_records.append(
            record
        )


    # --------------------------------------------------------
    # Print conflict summary
    # --------------------------------------------------------

    print()
    print(
        f"Different features: "
        f"{differences} / "
        f"{len(feature_columns)}"
    )


    if differences == 0:

        print(
            "[IDENTICAL] All 62 structured "
            "features are identical."
        )

    else:

        print(
            "[DIFFERENT] Structured features "
            "are not identical."
        )

        print()
        print(
            "Feature differences:"
        )

        for feature in feature_columns:

            train_value = train_row[
                feature
            ]

            test_value = test_row[
                feature
            ]


            if pd.isna(train_value) and pd.isna(test_value):

                equal = True

            elif pd.isna(train_value) != pd.isna(test_value):

                equal = False

            else:

                equal = (
                    train_value == test_value
                )


            if not equal:

                print(
                    f"  {feature}: "
                    f"TRAIN={train_value!r} "
                    f"TEST={test_value!r}"
                )


    summary_records.append(
        {
            "conflict_number":
                conflict_id,

            "train_index":
                train_index,

            "test_index":
                test_index,

            "train_label":
                train_label,

            "test_label":
                test_label,

            "different_feature_count":
                differences,

            "total_features":
                len(feature_columns),

            "all_features_identical":
                differences == 0,
        }
    )


# ============================================================
# 5. CREATE DATAFRAMES
# ============================================================

print()
print("=" * 75)
print("[5] Creating comparison reports")
print("=" * 75)


comparison_df = pd.DataFrame(
    all_records
)

summary_df = pd.DataFrame(
    summary_records
)


# ============================================================
# 6. SAVE FULL COMPARISON
# ============================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print(
    "[SAVED] Full 62-feature comparison:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 7. SAVE SUMMARY
# ============================================================

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


print()
print(
    "[SAVED] Feature difference summary:"
)

print(
    OUTPUT_SUMMARY
)


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("FINAL SUMMARY")
print("=" * 75)


identical_count = int(
    summary_df[
        "all_features_identical"
    ].sum()
)

different_count = (
    len(summary_df)
    -
    identical_count
)


print()
print(
    f"Conflicting rows analyzed : "
    f"{len(summary_df)}"
)

print(
    f"All 62 features identical : "
    f"{identical_count}"
)

print(
    f"Features differ           : "
    f"{different_count}"
)


print()
print(
    "Per-conflict results:"
)

for _, row in summary_df.iterrows():

    status = (
        "IDENTICAL"
        if row[
            "all_features_identical"
        ]
        else "DIFFERENT"
    )

    print(
        f"  Conflict #{int(row['conflict_number'])}: "
        f"{status} | "
        f"{int(row['different_feature_count'])} "
        f"different feature(s)"
    )


# ============================================================
# 9. INTERPRETATION
# ============================================================

print()
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

if identical_count == len(summary_df):

    print()
    print(
        "RESULT: All conflicting rows have "
        "identical 62-feature representations "
        "but different labels."
    )

    print()
    print(
        "This indicates LABEL AMBIGUITY / "
        "ANNOTATION CONFLICT rather than "
        "a structured-feature mismatch."
    )

    print()
    print(
        "The same observable feature vector "
        "is being mapped to different MITRE "
        "technique labels."
    )

elif different_count == len(summary_df):

    print()
    print(
        "RESULT: All conflicting pairs have "
        "at least one different structured feature."
    )

    print()
    print(
        "The overlap hash investigation should "
        "therefore be examined for data-type or "
        "fingerprint behavior."
    )

else:

    print()
    print(
        "RESULT: Mixed behavior."
    )

    print(
        f"{identical_count} conflict(s) have "
        "identical feature vectors."
    )

    print(
        f"{different_count} conflict(s) have "
        "different feature vectors."
    )


print()
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print()
print(
    "Do NOT delete rows."
)

print(
    "Do NOT retrain."
)

print(
    "Do NOT change the split yet."
)

print()
print(
    "Use this report to determine the correct "
    "dataset-cleaning strategy."
)

print()
print("Finished.")