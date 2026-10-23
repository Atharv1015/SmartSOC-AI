"""
SmartSOC Windows-APT Model B v4.1
Dataset Preparation

Purpose
-------
Prepare and validate the v4 temporal dataset for Model B v4.1.

Important:
- Does NOT modify the original v4 dataset.
- Preserves the temporal train/test split.
- Preserves row alignment between structured/text/labels/metadata.
- Correctly parses Windows-APT timestamps containing " @ ".
- Validates labels and duplicate behavior signatures.
- Produces a clean v4.1 dataset for the next audit/training stage.

Project root:
C:/Users/vedantika/SmartSOC-AI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V4_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4"
)

V41_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

# Input files
STRUCTURED_TRAIN = V4_DIR / "structured_train.csv"
STRUCTURED_TEST = V4_DIR / "structured_test.csv"

TEXT_TRAIN = V4_DIR / "behavior_text_train.csv"
TEXT_TEST = V4_DIR / "behavior_text_test.csv"

LABELS_TRAIN = V4_DIR / "labels_train.csv"
LABELS_TEST = V4_DIR / "labels_test.csv"

METADATA_TRAIN = V4_DIR / "metadata_train.csv"
METADATA_TEST = V4_DIR / "metadata_test.csv"

SCHEMA_FILE = V4_DIR / "schema.json"

# Output files
OUT_STRUCTURED_TRAIN = V41_DIR / "structured_train.csv"
OUT_STRUCTURED_TEST = V41_DIR / "structured_test.csv"

OUT_TEXT_TRAIN = V41_DIR / "behavior_text_train.csv"
OUT_TEXT_TEST = V41_DIR / "behavior_text_test.csv"

OUT_LABELS_TRAIN = V41_DIR / "labels_train.csv"
OUT_LABELS_TEST = V41_DIR / "labels_test.csv"

OUT_METADATA_TRAIN = V41_DIR / "metadata_train.csv"
OUT_METADATA_TEST = V41_DIR / "metadata_test.csv"

OUT_SCHEMA = V41_DIR / "schema.json"

OUT_AUDIT = V41_DIR / "dataset_audit.json"


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_section(title: str) -> None:
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================
# TIMESTAMP PARSING
# ============================================================

TIMESTAMP_FORMAT = "%b %d, %Y %H:%M:%S.%f"


def parse_windows_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse Windows-APT timestamps.

    Original format:
        Dec 1, 2024 @ 06:24:28.026

    Normalized format:
        Dec 1, 2024 06:24:28.026

    This avoids pandas/dateutil ambiguity around the literal '@'.
    """

    cleaned = (
        series
        .astype("string")
        .str.replace(" @ ", " ", regex=False)
        .str.strip()
    )

    parsed = pd.to_datetime(
        cleaned,
        format=TIMESTAMP_FORMAT,
        errors="coerce",
    )

    return parsed


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(value) -> str:
    """
    Normalize behavior text while preserving meaningful content.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = (
        text
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    # Collapse repeated whitespace
    text = " ".join(text.split())

    return text.strip().lower()


def normalize_behavior_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the behavior text column and normalize it.

    Expected v4 structure:
        one text column.
    """

    if df.shape[1] != 1:
        raise ValueError(
            "Behavior text dataframe must contain exactly one column. "
            f"Found {df.shape[1]} columns: {df.columns.tolist()}"
        )

    column = df.columns[0]

    df = df.copy()

    df[column] = df[column].apply(normalize_text)

    return df


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def find_label_column(df: pd.DataFrame) -> str:
    """
    Locate the MITRE technique label column.
    """

    preferred = [
        "techniques",
        "technique",
        "label",
        "labels",
    ]

    for column in preferred:
        if column in df.columns:
            return column

    if df.shape[1] == 1:
        return df.columns[0]

    raise ValueError(
        "Could not determine label column. "
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels into a single 'techniques' column.
    """

    column = find_label_column(df)

    result = pd.DataFrame()

    result["techniques"] = (
        df[column]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    return result


def split_techniques(value: str) -> list[str]:
    if not value:
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


# ============================================================
# METADATA VALIDATION
# ============================================================

def validate_metadata(
    metadata: pd.DataFrame,
    name: str,
) -> tuple[pd.DataFrame, dict]:

    if "timestamp" not in metadata.columns:
        raise ValueError(
            f"{name} metadata does not contain 'timestamp'. "
            f"Columns: {metadata.columns.tolist()}"
        )

    result = metadata.copy()

    parsed = parse_windows_timestamp(
        result["timestamp"]
    )

    invalid_mask = parsed.isna()

    invalid_count = int(invalid_mask.sum())

    print(
        f"{name.capitalize()} invalid timestamps: "
        f"{invalid_count}"
    )

    if invalid_count > 0:
        print("Examples:")

        examples = (
            result.loc[invalid_mask, "timestamp"]
            .head(10)
        )

        print(
            examples.to_string(index=False)
        )

        raise ValueError(
            f"Invalid timestamps found in {name} metadata."
        )

    result["timestamp_parsed"] = parsed

    print(
        f"{name.capitalize()} timestamp range: "
        f"{parsed.min()} -> {parsed.max()}"
    )

    audit = {
        "rows": len(result),
        "invalid_timestamps": invalid_count,
        "earliest": str(parsed.min()),
        "latest": str(parsed.max()),
    }

    return result, audit


# ============================================================
# ALIGNMENT
# ============================================================

def check_alignment(
    structured: pd.DataFrame,
    text: pd.DataFrame,
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    name: str,
) -> None:

    sizes = {
        "structured": len(structured),
        "text": len(text),
        "labels": len(labels),
        "metadata": len(metadata),
    }

    print(
        f"{name.capitalize()} row counts:"
    )

    for key, value in sizes.items():
        print(
            f"  {key:<12}: {value}"
        )

    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"{name.capitalize()} datasets are not aligned:\n"
            f"{sizes}"
        )

    print(
        f"{name.capitalize()} aligned rows: "
        f"{len(structured)}"
    )


# ============================================================
# STRUCTURED DATA VALIDATION
# ============================================================

def inspect_structured_features(
    structured: pd.DataFrame,
) -> dict:

    print_section(
        "Structured feature inspection"
    )

    print(
        f"Structured rows : {len(structured)}"
    )

    print(
        f"Structured cols : {len(structured.columns)}"
    )

    categorical_candidates = [
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
    ]

    categorical = [
        col
        for col in categorical_candidates
        if col in structured.columns
    ]

    numerical = [
        col
        for col in structured.columns
        if col not in categorical
    ]

    print(
        f"Categorical features: {len(categorical)}"
    )

    for col in categorical:
        print(f"  - {col}")

    print(
        f"Numerical features: {len(numerical)}"
    )

    # Check missingness
    missing = structured.isna().sum()

    missing_features = (
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    print(
        f"Features with missing values: "
        f"{len(missing_features)}"
    )

    if len(missing_features) > 0:
        print(
            missing_features.head(20).to_string()
        )

    return {
        "feature_count": len(structured.columns),
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "missing_features": {
            str(k): int(v)
            for k, v in missing_features.items()
        },
    }


# ============================================================
# DUPLICATE / SIGNATURE ANALYSIS
# ============================================================

def build_behavior_signature(
    structured: pd.DataFrame,
    text: pd.DataFrame,
) -> pd.Series:

    structured_string = (
        structured
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )

    text_column = text.columns[0]

    behavior_string = (
        text[text_column]
        .fillna("")
        .astype(str)
    )

    return (
        structured_string
        + "||"
        + behavior_string
    )


def analyze_duplicates(
    structured: pd.DataFrame,
    text: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict:

    print_section(
        "Duplicate and behavior-signature analysis"
    )

    structured_signature = (
        structured
        .fillna("")
        .astype(str)
        .agg("|".join, axis=1)
    )

    combined_signature = (
        build_behavior_signature(
            structured,
            text,
        )
    )

    structured_unique = (
        structured_signature.nunique()
    )

    combined_unique = (
        combined_signature.nunique()
    )

    structured_duplicates = (
        len(structured)
        - structured_unique
    )

    combined_duplicates = (
        len(combined_signature)
        - combined_unique
    )

    print(
        f"Structured duplicate rows : "
        f"{structured_duplicates}"
    )

    print(
        f"Combined behavior duplicate rows: "
        f"{combined_duplicates}"
    )

    print(
        f"Unique structured signatures: "
        f"{structured_unique}"
    )

    print(
        f"Unique combined signatures: "
        f"{combined_unique}"
    )

    # Check signatures having conflicting label combinations
    label_series = (
        labels["techniques"]
        .fillna("")
        .astype(str)
    )

    signature_df = pd.DataFrame(
        {
            "signature": combined_signature,
            "techniques": label_series,
        }
    )

    conflicts = (
        signature_df
        .groupby("signature")["techniques"]
        .nunique()
    )

    conflicting_signatures = (
        conflicts[conflicts > 1]
    )

    print(
        f"Behavior signatures with multiple "
        f"label combinations: "
        f"{len(conflicting_signatures)}"
    )

    if len(conflicting_signatures) > 0:
        print()
        print("Top conflicting signatures:")

        top_conflicts = (
            conflicting_signatures
            .sort_values(ascending=False)
            .head(10)
        )

        for signature in top_conflicts.index:

            print()
            print("Signature:")
            print(signature)

            labels_for_signature = (
                signature_df.loc[
                    signature_df["signature"] == signature,
                    "techniques",
                ]
                .value_counts()
            )

            print("Labels:")
            print(
                labels_for_signature.to_string()
            )

    return {
        "structured_unique_signatures": int(
            structured_unique
        ),
        "structured_duplicate_rows": int(
            structured_duplicates
        ),
        "combined_unique_signatures": int(
            combined_unique
        ),
        "combined_duplicate_rows": int(
            combined_duplicates
        ),
        "conflicting_behavior_signatures": int(
            len(conflicting_signatures)
        ),
    }


# ============================================================
# TECHNIQUE ANALYSIS
# ============================================================

def analyze_labels(
    labels_train: pd.DataFrame,
    labels_test: pd.DataFrame,
) -> dict:

    print_section(
        "MITRE label analysis"
    )

    train_sets = (
        labels_train["techniques"]
        .apply(split_techniques)
    )

    test_sets = (
        labels_test["techniques"]
        .apply(split_techniques)
    )

    train_techniques = sorted(
        {
            technique
            for values in train_sets
            for technique in values
        }
    )

    test_techniques = sorted(
        {
            technique
            for values in test_sets
            for technique in values
        }
    )

    all_techniques = sorted(
        set(train_techniques)
        | set(test_techniques)
    )

    test_only = sorted(
        set(test_techniques)
        - set(train_techniques)
    )

    train_only = sorted(
        set(train_techniques)
        - set(test_techniques)
    )

    print(
        f"Training techniques: "
        f"{len(train_techniques)}"
    )

    print(
        f"Testing techniques : "
        f"{len(test_techniques)}"
    )

    print(
        f"Total techniques   : "
        f"{len(all_techniques)}"
    )

    if test_only:
        print()
        print(
            "WARNING: Test-only techniques:"
        )

        for technique in test_only:
            print(
                f"  - {technique}"
            )

    else:
        print(
            "No test-only techniques."
        )

    print()
    print(
        "Training-only techniques:"
    )

    if train_only:
        for technique in train_only:
            print(
                f"  - {technique}"
            )
    else:
        print("  None")

    train_counts = (
        labels_train["techniques"]
        .apply(split_techniques)
        .explode()
        .value_counts()
    )

    print()
    print("Top training techniques:")

    print(
        train_counts.head(25).to_string()
    )

    multi_train = int(
        train_sets.apply(
            lambda x: len(x) > 1
        ).sum()
    )

    multi_test = int(
        test_sets.apply(
            lambda x: len(x) > 1
        ).sum()
    )

    print()
    print(
        f"Multi-label training events: "
        f"{multi_train}"
    )

    print(
        f"Multi-label testing events : "
        f"{multi_test}"
    )

    return {
        "training_techniques": train_techniques,
        "testing_techniques": test_techniques,
        "all_techniques": all_techniques,
        "test_only_techniques": test_only,
        "train_only_techniques": train_only,
        "multi_label_training_events": multi_train,
        "multi_label_testing_events": multi_test,
    }


# ============================================================
# EMPTY TEXT ANALYSIS
# ============================================================

def analyze_text(
    text_train: pd.DataFrame,
    text_test: pd.DataFrame,
) -> dict:

    print_section(
        "Behavior text analysis"
    )

    column_train = text_train.columns[0]
    column_test = text_test.columns[0]

    empty_train = int(
        (
            text_train[column_train]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        ).sum()
    )

    empty_test = int(
        (
            text_test[column_test]
            .fillna("")
            .astype(str)
            .str.strip()
            == ""
        ).sum()
    )

    print(
        f"Empty training behavior text: "
        f"{empty_train}"
    )

    print(
        f"Empty testing behavior text : "
        f"{empty_test}"
    )

    return {
        "empty_training_text": empty_train,
        "empty_testing_text": empty_test,
    }


# ============================================================
# TEMPORAL INTEGRITY
# ============================================================

def analyze_temporal_split(
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
) -> dict:

    print_section(
        "Temporal split validation"
    )

    train_max = (
        meta_train["timestamp_parsed"].max()
    )

    test_min = (
        meta_test["timestamp_parsed"].min()
    )

    train_min = (
        meta_train["timestamp_parsed"].min()
    )

    test_max = (
        meta_test["timestamp_parsed"].max()
    )

    print(
        f"Training range:"
    )

    print(
        f"  {train_min} -> {train_max}"
    )

    print(
        f"Testing range:"
    )

    print(
        f"  {test_min} -> {test_max}"
    )

    temporal_overlap = (
        test_min <= train_max
    )

    if temporal_overlap:
        print()
        print(
            "WARNING: Temporal overlap detected."
        )

        print(
            f"Latest training event: {train_max}"
        )

        print(
            f"Earliest testing event: {test_min}"
        )

    else:
        print(
            "[OK] Training/testing timestamps "
            "are temporally separated."
        )

    return {
        "training_earliest": str(train_min),
        "training_latest": str(train_max),
        "testing_earliest": str(test_min),
        "testing_latest": str(test_max),
        "temporal_overlap": bool(
            temporal_overlap
        ),
    }


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def check_duplicate_event_ids(
    metadata: pd.DataFrame,
    name: str,
) -> dict:

    if "event_id" not in metadata.columns:
        return {
            "event_id_present": False,
            "duplicate_event_ids": None,
        }

    duplicate_count = int(
        metadata["event_id"]
        .duplicated()
        .sum()
    )

    print(
        f"{name.capitalize()} duplicate event_id rows: "
        f"{duplicate_count}"
    )

    return {
        "event_id_present": True,
        "duplicate_event_ids": duplicate_count,
    }


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    structured_train,
    structured_test,
    text_train,
    text_test,
    labels_train,
    labels_test,
    metadata_train,
    metadata_test,
    schema,
    audit,
) -> None:

    print_section(
        "Saving v4.1 dataset"
    )

    V41_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Never save internal parsed timestamp unless explicitly useful.
    # Preserve original metadata schema.
    metadata_train_out = metadata_train.drop(
        columns=["timestamp_parsed"],
        errors="ignore",
    )

    metadata_test_out = metadata_test.drop(
        columns=["timestamp_parsed"],
        errors="ignore",
    )

    structured_train.to_csv(
        OUT_STRUCTURED_TRAIN,
        index=False,
    )

    structured_test.to_csv(
        OUT_STRUCTURED_TEST,
        index=False,
    )

    text_train.to_csv(
        OUT_TEXT_TRAIN,
        index=False,
    )

    text_test.to_csv(
        OUT_TEXT_TEST,
        index=False,
    )

    labels_train.to_csv(
        OUT_LABELS_TRAIN,
        index=False,
    )

    labels_test.to_csv(
        OUT_LABELS_TEST,
        index=False,
    )

    metadata_train_out.to_csv(
        OUT_METADATA_TRAIN,
        index=False,
    )

    metadata_test_out.to_csv(
        OUT_METADATA_TEST,
        index=False,
    )

    with open(
        OUT_SCHEMA,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            schema,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        OUT_AUDIT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[OK] {OUT_STRUCTURED_TRAIN}"
    )

    print(
        f"[OK] {OUT_STRUCTURED_TEST}"
    )

    print(
        f"[OK] {OUT_TEXT_TRAIN}"
    )

    print(
        f"[OK] {OUT_TEXT_TEST}"
    )

    print(
        f"[OK] {OUT_LABELS_TRAIN}"
    )

    print(
        f"[OK] {OUT_LABELS_TEST}"
    )

    print(
        f"[OK] {OUT_METADATA_TRAIN}"
    )

    print(
        f"[OK] {OUT_METADATA_TEST}"
    )

    print(
        f"[OK] {OUT_SCHEMA}"
    )

    print(
        f"[OK] {OUT_AUDIT}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    structured_train,
    structured_test,
    text_train,
    text_test,
    labels_train,
    labels_test,
    metadata_train,
    metadata_test,
) -> None:

    print_section(
        "Final v4.1 alignment validation"
    )

    datasets = {
        "structured_train": structured_train,
        "text_train": text_train,
        "labels_train": labels_train,
        "metadata_train": metadata_train,

        "structured_test": structured_test,
        "text_test": text_test,
        "labels_test": labels_test,
        "metadata_test": metadata_test,
    }

    for name, df in datasets.items():
        print(
            f"{name:<20}: {len(df)} rows"
        )

    train_lengths = {
        len(structured_train),
        len(text_train),
        len(labels_train),
        len(metadata_train),
    }

    test_lengths = {
        len(structured_test),
        len(text_test),
        len(labels_test),
        len(metadata_test),
    }

    if len(train_lengths) != 1:
        raise ValueError(
            "Final training datasets are not aligned."
        )

    if len(test_lengths) != 1:
        raise ValueError(
            "Final testing datasets are not aligned."
        )

    print()
    print(
        "[OK] Training datasets aligned."
    )

    print(
        "[OK] Testing datasets aligned."
    )

    # Verify no accidental index column
    suspicious_columns = []

    for name, df in datasets.items():

        for column in df.columns:

            if str(column).lower() in {
                "unnamed: 0",
                "index",
            }:

                suspicious_columns.append(
                    (name, column)
                )

    if suspicious_columns:

        print()
        print(
            "WARNING: Suspicious index columns:"
        )

        for item in suspicious_columns:
            print(
                f"  {item[0]} -> {item[1]}"
            )

    else:
        print(
            "[OK] No accidental index columns."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "SmartSOC Windows-APT Model B v4.1 - Dataset Preparation"
    )

    # --------------------------------------------------------
    # Required artifacts
    # --------------------------------------------------------

    print_section(
        "Checking required artifacts..."
    )

    required_files = [
        STRUCTURED_TRAIN,
        STRUCTURED_TEST,
        TEXT_TRAIN,
        TEXT_TEST,
        LABELS_TRAIN,
        LABELS_TEST,
        METADATA_TRAIN,
        METADATA_TEST,
        SCHEMA_FILE,
    ]

    for path in required_files:
        require_file(path)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print_header(
        "Loading v4 dataset"
    )

    structured_train = pd.read_csv(
        STRUCTURED_TRAIN
    )

    structured_test = pd.read_csv(
        STRUCTURED_TEST
    )

    text_train = pd.read_csv(
        TEXT_TRAIN
    )

    text_test = pd.read_csv(
        TEXT_TEST
    )

    labels_train_raw = pd.read_csv(
        LABELS_TRAIN
    )

    labels_test_raw = pd.read_csv(
        LABELS_TEST
    )

    metadata_train = pd.read_csv(
        METADATA_TRAIN
    )

    metadata_test = pd.read_csv(
        METADATA_TEST
    )

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        original_schema = json.load(f)

    print(
        f"Structured train: {structured_train.shape}"
    )

    print(
        f"Structured test : {structured_test.shape}"
    )

    print(
        f"Text train      : {text_train.shape}"
    )

    print(
        f"Text test       : {text_test.shape}"
    )

    print(
        f"Labels train    : {labels_train_raw.shape}"
    )

    print(
        f"Labels test     : {labels_test_raw.shape}"
    )

    print(
        f"Metadata train  : {metadata_train.shape}"
    )

    print(
        f"Metadata test   : {metadata_test.shape}"
    )

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    print_section(
        "Normalizing behavior text"
    )

    text_train = normalize_behavior_column(
        text_train
    )

    text_test = normalize_behavior_column(
        text_test
    )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    print_section(
        "Normalizing MITRE labels"
    )

    labels_train = normalize_label_dataframe(
        labels_train_raw
    )

    labels_test = normalize_label_dataframe(
        labels_test_raw
    )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    print_header(
        "Checking alignment"
    )

    check_alignment(
        structured_train,
        text_train,
        labels_train,
        metadata_train,
        "training",
    )

    check_alignment(
        structured_test,
        text_test,
        labels_test,
        metadata_test,
        "testing",
    )

    # --------------------------------------------------------
    # Timestamp parsing
    # --------------------------------------------------------

    print_header(
        "Parsing timestamps"
    )

    metadata_train, train_timestamp_audit = (
        validate_metadata(
            metadata_train,
            "training",
        )
    )

    metadata_test, test_timestamp_audit = (
        validate_metadata(
            metadata_test,
            "testing",
        )
    )

    # --------------------------------------------------------
    # Structured features
    # --------------------------------------------------------

    structured_audit = (
        inspect_structured_features(
            structured_train
        )
    )

    # Ensure test has identical schema
    if list(structured_train.columns) != list(
        structured_test.columns
    ):
        raise ValueError(
            "Structured train/test columns do not match."
        )

    print(
        "[OK] Structured train/test schemas match."
    )

    # --------------------------------------------------------
    # Duplicate analysis
    # --------------------------------------------------------

    train_duplicate_audit = analyze_duplicates(
        structured_train,
        text_train,
        labels_train,
    )

    test_duplicate_audit = analyze_duplicates(
        structured_test,
        text_test,
        labels_test,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_audit = analyze_labels(
        labels_train,
        labels_test,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text_audit = analyze_text(
        text_train,
        text_test,
    )

    # --------------------------------------------------------
    # Temporal validation
    # --------------------------------------------------------

    temporal_audit = analyze_temporal_split(
        metadata_train,
        metadata_test,
    )

    # --------------------------------------------------------
    # Event ID validation
    # --------------------------------------------------------

    train_event_audit = check_duplicate_event_ids(
        metadata_train,
        "training",
    )

    test_event_audit = check_duplicate_event_ids(
        metadata_test,
        "testing",
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_validation(
        structured_train,
        structured_test,
        text_train,
        text_test,
        labels_train,
        labels_test,
        metadata_train,
        metadata_test,
    )

    # --------------------------------------------------------
    # Build schema
    # --------------------------------------------------------

    categorical_columns = [
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
    ]

    categorical_columns = [
        col
        for col in categorical_columns
        if col in structured_train.columns
    ]

    numerical_columns = [
        col
        for col in structured_train.columns
        if col not in categorical_columns
    ]

    schema = {
        "model": "SmartSOC Windows-APT Model B v4.1",
        "source_dataset": "windows_apt_v4",
        "timestamp_format": TIMESTAMP_FORMAT,

        "structured_feature_count": int(
            len(structured_train.columns)
        ),

        "categorical_columns": categorical_columns,

        "numerical_columns": numerical_columns,

        "categorical_feature_count": int(
            len(categorical_columns)
        ),

        "numerical_feature_count": int(
            len(numerical_columns)
        ),

        "text_column": text_train.columns[0],

        "label_column": "techniques",

        "training_rows": int(
            len(structured_train)
        ),

        "testing_rows": int(
            len(structured_test)
        ),

        "mitre_techniques": label_audit[
            "all_techniques"
        ],

        "test_only_techniques": label_audit[
            "test_only_techniques"
        ],

        "temporal_split": {
            "training_earliest":
                temporal_audit[
                    "training_earliest"
                ],

            "training_latest":
                temporal_audit[
                    "training_latest"
                ],

            "testing_earliest":
                temporal_audit[
                    "testing_earliest"
                ],

            "testing_latest":
                temporal_audit[
                    "testing_latest"
                ],
        },

        "original_schema": original_schema,
    }

    # --------------------------------------------------------
    # Audit object
    # --------------------------------------------------------

    audit = {
        "model": (
            "SmartSOC Windows-APT Model B v4.1"
        ),

        "dataset": {
            "training_rows": int(
                len(structured_train)
            ),

            "testing_rows": int(
                len(structured_test)
            ),

            "training_features": int(
                len(structured_train.columns)
            ),
        },

        "timestamps": {
            "training":
                train_timestamp_audit,

            "testing":
                test_timestamp_audit,
        },

        "structured_features":
            structured_audit,

        "duplicates": {
            "training":
                train_duplicate_audit,

            "testing":
                test_duplicate_audit,
        },

        "labels":
            label_audit,

        "text":
            text_audit,

        "temporal":
            temporal_audit,

        "event_ids": {
            "training":
                train_event_audit,

            "testing":
                test_event_audit,
        },

        "status": "PASS",
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        structured_train,
        structured_test,
        text_train,
        text_test,
        labels_train,
        labels_test,
        metadata_train,
        metadata_test,
        schema,
        audit,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "SmartSOC Windows-APT Model B v4.1 "
        "Dataset Preparation COMPLETE"
    )

    print(
        f"Training events           : "
        f"{len(structured_train)}"
    )

    print(
        f"Testing events            : "
        f"{len(structured_test)}"
    )

    print(
        f"Structured features       : "
        f"{len(structured_train.columns)}"
    )

    print(
        f"MITRE techniques          : "
        f"{len(label_audit['all_techniques'])}"
    )

    print(
        f"Test-only techniques      : "
        f"{len(label_audit['test_only_techniques'])}"
    )

    print(
        f"Conflicting signatures    : "
        f"{train_duplicate_audit['conflicting_behavior_signatures']}"
    )

    print(
        f"Invalid training timestamps: "
        f"{train_timestamp_audit['invalid_timestamps']}"
    )

    print(
        f"Invalid testing timestamps : "
        f"{test_timestamp_audit['invalid_timestamps']}"
    )

    print()

    if temporal_audit["temporal_overlap"]:
        print(
            "[WARNING] Temporal overlap exists."
        )
    else:
        print(
            "[OK] Temporal separation verified."
        )

    if label_audit["test_only_techniques"]:
        print(
            "[WARNING] Test-only MITRE techniques exist."
        )
    else:
        print(
            "[OK] No test-only MITRE techniques."
        )

    print()
    print(
        "v4.1 dataset written to:"
    )

    print(
        V41_DIR
    )

    print()
    print(
        "Next step:"
    )

    print(
        "Run the v4.1 audit before training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nProcess interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 78)
        print("ERROR")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)