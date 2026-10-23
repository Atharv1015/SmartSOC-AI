"""
SmartSOC
Windows-APT Model B v3
Inference Alignment Audit

Purpose:
    Investigate whether production inference features/text are
    semantically aligned with the training representation.

This script DOES NOT train or modify the model.

It analyzes:

1. Training structured features
2. Training behavior text
3. Training labels
4. Feature distributions by MITRE technique
5. PowerShell-related training examples
6. Suspicious technique correlations
7. Current inference feature construction
8. Current model predictions
9. Feature-space dimensionality
10. Potential causes of false-positive technique predictions
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v3"

MODEL_DIR = ROOT / "models" / "windows_apt_v3"

STRUCTURED_PATH = DATA_DIR / "structured_features.csv"
TEXT_PATH = DATA_DIR / "behavior_text.csv"
LABEL_PATH = DATA_DIR / "labels.csv"
METADATA_PATH = DATA_DIR / "metadata.csv"

MODEL_PATH = (
    MODEL_DIR /
    "multilabel_logistic_v3_calibrated.joblib"
)

ENCODER_PATH = (
    MODEL_DIR /
    "structured_encoder_calibrated.joblib"
)

TFIDF_PATH = (
    MODEL_DIR /
    "tfidf_vectorizer_calibrated.joblib"
)

LABEL_ENCODER_PATH = (
    MODEL_DIR /
    "multilabel_encoder_calibrated.joblib"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "technique_thresholds.json"
)

SCHEMA_PATH = (
    MODEL_DIR /
    "feature_schema_calibrated_v3.json"
)


# ============================================================
# PRINT HELPERS
# ============================================================

def section(title):

    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)


def subsection(title):

    print()
    print("-" * 78)
    print(f" {title}")
    print("-" * 78)


# ============================================================
# FILE CHECK
# ============================================================

def check_files():

    section("CHECKING REQUIRED FILES")

    files = [
        STRUCTURED_PATH,
        TEXT_PATH,
        LABEL_PATH,
        METADATA_PATH,
        MODEL_PATH,
        ENCODER_PATH,
        TFIDF_PATH,
        LABEL_ENCODER_PATH,
        THRESHOLD_PATH,
        SCHEMA_PATH,
    ]

    for path in files:

        if path.exists():

            print(f"[OK] {path}")

        else:

            print(f"[MISSING] {path}")

            raise FileNotFoundError(
                f"Required file missing:\n{path}"
            )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    section("LOADING WINDOWS-APT V3 DATA")

    print("Loading structured features...")

    X = pd.read_csv(
        STRUCTURED_PATH
    )

    print(
        f"Structured rows: {len(X)}"
    )

    print("Loading behavior text...")

    text_df = pd.read_csv(
        TEXT_PATH
    )

    print(
        f"Text rows: {len(text_df)}"
    )

    print("Loading labels...")

    labels = pd.read_csv(
        LABEL_PATH
    )

    print(
        f"Label rows: {len(labels)}"
    )

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    if not (
        len(X)
        == len(text_df)
        == len(labels)
        == len(metadata)
    ):

        raise RuntimeError(
            "Dataset row counts do not match."
        )

    return (
        X,
        text_df,
        labels,
        metadata,
    )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

def load_artifacts():

    section("LOADING CALIBRATED MODEL ARTIFACTS")

    model = joblib.load(
        MODEL_PATH
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    label_encoder = joblib.load(
        LABEL_ENCODER_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        thresholds = json.load(f)

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "Structured encoder:",
        type(encoder).__name__
    )

    print(
        "TF-IDF vocabulary:",
        len(tfidf.vocabulary_)
    )

    print(
        "MITRE classes:",
        len(label_encoder.classes_)
    )

    print(
        "Schema feature count:",
        schema.get(
            "final_feature_count",
            "UNKNOWN"
        )
    )

    print(
        "Categorical columns:"
    )

    for column in schema[
        "categorical_columns"
    ]:

        print(
            f"  - {column}"
        )

    print(
        "Numerical columns:",
        len(
            schema[
                "numerical_columns"
            ]
        )
    )

    return (
        model,
        encoder,
        tfidf,
        label_encoder,
        thresholds,
        schema,
    )


# ============================================================
# BASIC DATA AUDIT
# ============================================================

def basic_audit(
    X,
    text_df,
    labels,
    metadata,
):

    section("BASIC DATA AUDIT")

    print(
        "Events:",
        len(X)
    )

    print(
        "Structured features:",
        len(X.columns)
    )

    print(
        "Behavior text columns:",
        text_df.columns.tolist()
    )

    print(
        "Label columns:",
        labels.columns.tolist()
    )

    print(
        "Metadata columns:",
        metadata.columns.tolist()
    )

    print(
        "Duplicate structured rows:",
        X.duplicated().sum()
    )

    # Find likely behavior text column.
    text_candidates = [
        "behavior_text",
        "text",
        "full_log",
        "behavior",
    ]

    text_column = None

    for candidate in text_candidates:

        if candidate in text_df.columns:

            text_column = candidate
            break

    if text_column is None:

        print(
            "\nWARNING:"
        )

        print(
            "Could not automatically identify "
            "behavior text column."
        )

        print(
            "Available columns:",
            text_df.columns.tolist()
        )

    else:

        empty = (
            text_df[text_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(
            "\nBehavior text column:",
            text_column
        )

        print(
            "Empty behavior texts:",
            empty
        )

    return text_column


# ============================================================
# LABEL PARSING
# ============================================================

def parse_techniques(value):

    if pd.isna(value):

        return []

    value = str(value).strip()

    if not value:

        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def build_label_sets(labels):

    technique_column = "techniques"

    if technique_column not in labels.columns:

        raise RuntimeError(
            "Expected labels.csv to contain "
            "'techniques' column."
        )

    return labels[
        technique_column
    ].apply(
        parse_techniques
    )


# ============================================================
# POWERSELL AUDIT
# ============================================================

def powershell_audit(
    X,
    text_df,
    labels,
):

    section("POWERSHELL TRAINING DATA AUDIT")

    if "process_name" not in X.columns:

        print(
            "process_name not found."
        )

        return

    process = (
        X["process_name"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    parent = (
        X["parent_process_name"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    power_mask = (
        process.str.contains(
            "powershell",
            regex=False
        )
        |
        parent.str.contains(
            "powershell",
            regex=False
        )
    )

    print(
        "PowerShell-related events:",
        int(power_mask.sum())
    )

    if power_mask.sum() == 0:

        print(
            "No PowerShell events found."
        )

        return

    power_indices = X.index[
        power_mask
    ]

    subsection(
        "PowerShell Technique Distribution"
    )

    technique_counts = {}

    label_sets = build_label_sets(
        labels
    )

    for index in power_indices:

        for technique in label_sets.loc[
            index
        ]:

            technique_counts[
                technique
            ] = technique_counts.get(
                technique,
                0
            ) + 1

    if technique_counts:

        result = pd.Series(
            technique_counts
        ).sort_values(
            ascending=False
        )

        print(
            result.to_string()
        )

    else:

        print(
            "No labels found for PowerShell events."
        )

    subsection(
        "PowerShell Structured Feature Statistics"
    )

    interesting = [
        "uses_powershell",
        "uses_cmd",
        "uses_wmic",
        "uses_certutil",
        "uses_bitsadmin",
        "uses_reg",
        "uses_rundll32",
        "uses_schtasks",
        "uses_vssadmin",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "is_admin_user",
        "integrity_high",
        "is_network_event",
    ]

    available = [
        c for c in interesting
        if c in X.columns
    ]

    if available:

        stats = (
            X.loc[
                power_indices,
                available
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        print(
            stats.to_string()
        )

    subsection(
        "Sample PowerShell Training Events"
    )

    display_columns = [
        c
        for c in [
            "event_id",
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
            "uses_powershell",
            "has_encoded_command",
            "has_download_behavior",
            "has_registry_operation",
            "has_file_deletion",
            "has_obfuscation_hint",
        ]
        if c in X.columns
    ]

    samples = X.loc[
        power_indices,
        display_columns
    ].head(20)

    print(
        samples.to_string(
            index=True
        )
    )


# ============================================================
# TARGET TECHNIQUE AUDIT
# ============================================================

def technique_feature_audit(
    X,
    labels,
):

    section(
        "TARGET TECHNIQUE FEATURE AUDIT"
    )

    label_sets = build_label_sets(
        labels
    )

    targets = [
        "T1059.001",
        "T1027",
        "T1112",
        "T1070.004",
        "T1548.002",
        "T1105",
        "T1059.003",
    ]

    interesting = [
        "uses_powershell",
        "uses_cmd",
        "uses_reg",
        "uses_certutil",
        "uses_bitsadmin",
        "uses_schtasks",
        "uses_wmic",
        "has_encoded_command",
        "has_download_behavior",
        "has_remote_execution_hint",
        "has_registry_operation",
        "has_file_deletion",
        "has_obfuscation_hint",
        "is_admin_user",
        "integrity_high",
        "is_network_event",
    ]

    available = [
        c for c in interesting
        if c in X.columns
    ]

    for technique in targets:

        mask = label_sets.apply(
            lambda values:
            technique in values
        )

        count = int(mask.sum())

        subsection(
            f"{technique} - {count} events"
        )

        if count == 0:

            print(
                "No examples."
            )

            continue

        stats = (
            X.loc[
                mask,
                available
            ]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        print(
            stats.to_string()
        )


# ============================================================
# LABEL CONFLICT AUDIT
# ============================================================

def label_conflict_audit(
    X,
    labels,
):

    section(
        "STRUCTURED FEATURE / LABEL CONFLICT AUDIT"
    )

    label_sets = build_label_sets(
        labels
    )

    # Use process-level behavioral grouping.
    grouping_columns = [
        c
        for c in [
            "process_name",
            "parent_process_name",
            "process_extension",
            "parent_extension",
            "event_id",
            "uses_powershell",
            "uses_cmd",
            "has_encoded_command",
            "has_download_behavior",
            "has_registry_operation",
            "has_file_deletion",
            "has_obfuscation_hint",
        ]
        if c in X.columns
    ]

    if not grouping_columns:

        print(
            "No suitable grouping columns."
        )

        return

    temp = X[
        grouping_columns
    ].copy()

    temp[
        "_label"
    ] = label_sets.apply(
        lambda x: ",".join(
            sorted(x)
        )
    )

    grouped = (
        temp
        .groupby(
            grouping_columns,
            dropna=False
        )["_label"]
        .nunique()
    )

    conflicts = (
        grouped > 1
    ).sum()

    print(
        "Behavior groups:",
        len(grouped)
    )

    print(
        "Groups with multiple label combinations:",
        int(conflicts)
    )

    if conflicts:

        print(
            "\nWARNING:"
        )

        print(
            "Some structurally similar events "
            "have different MITRE labels."
        )

        print(
            "This can cause the model to learn "
            "spurious correlations."
        )


# ============================================================
# TEXT AUDIT
# ============================================================

def text_audit(
    text_df,
    labels,
    text_column,
):

    section(
        "BEHAVIOR TEXT AUDIT"
    )

    if text_column is None:

        print(
            "Behavior text column unavailable."
        )

        return

    text = (
        text_df[text_column]
        .fillna("")
        .astype(str)
    )

    print(
        "Total text rows:",
        len(text)
    )

    print(
        "Empty text rows:",
        int(
            text.str.strip()
            .eq("")
            .sum()
        )
    )

    print(
        "Average text length:",
        round(
            text.str.len().mean(),
            2
        )
    )

    print(
        "Median text length:",
        round(
            text.str.len().median(),
            2
        )
    )

    subsection(
        "PowerShell Text Examples"
    )

    mask = text.str.contains(
        "powershell",
        case=False,
        regex=False
    )

    for index in text.index[mask].tolist()[:10]:

        print(
            f"\n[{index}]"
        )

        print(
            text.loc[index][:1000]
        )

        if index < len(labels):

            print(
                "LABEL:",
                labels.iloc[index][
                    "techniques"
                ]
            )


# ============================================================
# TF-IDF TOKEN AUDIT
# ============================================================

def tfidf_audit(
    tfidf,
):

    section(
        "TF-IDF TOKEN AUDIT"
    )

    vocabulary = tfidf.vocabulary_

    suspicious_terms = [
        "powershell",
        "powershell.exe",
        "encodedcommand",
        "encoded",
        "invoke-webrequest",
        "download",
        "reg",
        "registry",
        "delete",
        "wevtutil",
        "vssadmin",
        "get-service",
        "cmd.exe",
        "winword",
    ]

    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    print(
        "\nRelevant vocabulary tokens:"
    )

    for term in suspicious_terms:

        matches = [
            token
            for token in vocabulary
            if term in token.lower()
        ]

        print(
            f"\n{term}:"
        )

        if matches:

            print(
                matches[:30]
            )

        else:

            print(
                "  NOT FOUND"
            )


# ============================================================
# INFERENCE VECTOR AUDIT
# ============================================================

def inference_vector_audit(
    X,
    text_df,
    labels,
    model,
    encoder,
    tfidf,
    label_encoder,
    thresholds,
    schema,
):

    section(
        "MODEL FEATURE-SPACE AUDIT"
    )

    categorical_columns = schema[
        "categorical_columns"
    ]

    numerical_columns = schema[
        "numerical_columns"
    ]

    print(
        "Categorical feature count:",
        len(categorical_columns)
    )

    print(
        "Numerical feature count:",
        len(numerical_columns)
    )

    print(
        "TF-IDF feature count:",
        len(tfidf.vocabulary_)
    )

    encoded_width = encoder.transform(
        X[categorical_columns]
        .fillna("")
        .astype(str)
    ).shape[1]

    print(
        "OneHotEncoder output:",
        encoded_width
    )

    numerical_width = len(
        numerical_columns
    )

    expected = (
        encoded_width
        + numerical_width
        + len(tfidf.vocabulary_)
    )

    print(
        "Calculated final feature count:",
        expected
    )

    print(
        "Schema final feature count:",
        schema[
            "final_feature_count"
        ]
    )

    model_features = getattr(
        model,
        "n_features_in_",
        None
    )

    print(
        "Model feature count:",
        model_features
    )

    if (
        model_features is not None
        and model_features != expected
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Model feature dimension does "
            "not equal reconstructed dimension."
        )

    else:

        print(
            "\nFeature-space dimension is consistent."
        )


# ============================================================
# MODEL COEFFICIENT AUDIT
# ============================================================

def coefficient_audit(
    model,
    encoder,
    tfidf,
    label_encoder,
):

    section(
        "MODEL COEFFICIENT AUDIT"
    )

    # OneVsRestClassifier
    if not hasattr(
        model,
        "estimators_"
    ):

        print(
            "Model does not expose "
            "One-vs-Rest estimators."
        )

        return

    try:

        onehot_names = (
            encoder
            .get_feature_names_out()
            .tolist()
        )

    except Exception:

        onehot_names = []

    numerical_names = []

    # These are known from the schema.
    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(f)

    numerical_names = schema[
        "numerical_columns"
    ]

    tfidf_names = [
        f"TEXT::{x}"
        for x in tfidf.get_feature_names_out()
    ]

    feature_names = (
        onehot_names
        + numerical_names
        + tfidf_names
    )

    print(
        "Reconstructed feature names:",
        len(feature_names)
    )

    targets = [
        "T1059.001",
        "T1027",
        "T1112",
        "T1070.004",
        "T1548.002",
    ]

    class_names = list(
        label_encoder.classes_
    )

    for technique in targets:

        if technique not in class_names:

            continue

        index = class_names.index(
            technique
        )

        estimator = model.estimators_[
            index
        ]

        coefficients = (
            estimator.coef_[0]
        )

        if len(coefficients) != len(
            feature_names
        ):

            print(
                f"\n{technique}: "
                "coefficient/feature mismatch"
            )

            continue

        top_positive = np.argsort(
            coefficients
        )[-20:][::-1]

        subsection(
            f"Top positive features for {technique}"
        )

        for position in top_positive:

            print(
                f"{feature_names[position]:60s}"
                f" {coefficients[position]: .6f}"
            )


# ============================================================
# FIND REAL TRAINING EVENTS SIMILAR TO DEMO
# ============================================================

def find_similar_training_events(
    X,
    text_df,
    labels,
):

    section(
        "REAL TRAINING EVENTS SIMILAR TO DEMO EVENTS"
    )

    text_column_candidates = [
        "behavior_text",
        "text",
        "full_log",
        "behavior",
    ]

    text_column = None

    for candidate in text_column_candidates:

        if candidate in text_df.columns:

            text_column = candidate
            break

    if text_column is None:

        print(
            "No behavior text column detected."
        )

        return

    text = (
        text_df[text_column]
        .fillna("")
        .astype(str)
    )

    queries = {
        "NORMAL POWERSHELL":
            ["powershell", "get-service"],

        "ENCODED POWERSHELL":
            ["powershell", "encodedcommand"],

        "DOWNLOAD POWERSHELL":
            ["powershell", "invoke-webrequest"],
    }

    for name, terms in queries.items():

        subsection(name)

        mask = pd.Series(
            True,
            index=text.index
        )

        for term in terms:

            mask &= text.str.contains(
                term,
                case=False,
                regex=False
            )

        indices = text.index[
            mask
        ]

        print(
            "Matching training events:",
            len(indices)
        )

        for index in indices[:5]:

            print(
                f"\nINDEX: {index}"
            )

            print(
                "TEXT:"
            )

            print(
                text.loc[index][:1200]
            )

            print(
                "\nLABEL:"
            )

            print(
                labels.iloc[index][
                    "techniques"
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        " SmartSOC Windows-APT Model B v3"
    )
    print(
        " INFERENCE ALIGNMENT AUDIT"
    )
    print("=" * 78)

    try:

        check_files()

        (
            X,
            text_df,
            labels,
            metadata,
        ) = load_data()

        (
            model,
            encoder,
            tfidf,
            label_encoder,
            thresholds,
            schema,
        ) = load_artifacts()

        text_column = basic_audit(
            X,
            text_df,
            labels,
            metadata,
        )

        powershell_audit(
            X,
            text_df,
            labels,
        )

        technique_feature_audit(
            X,
            labels,
        )

        label_conflict_audit(
            X,
            labels,
        )

        text_audit(
            text_df,
            labels,
            text_column,
        )

        tfidf_audit(
            tfidf,
        )

        inference_vector_audit(
            X,
            text_df,
            labels,
            model,
            encoder,
            tfidf,
            label_encoder,
            thresholds,
            schema,
        )

        coefficient_audit(
            model,
            encoder,
            tfidf,
            label_encoder,
        )

        find_similar_training_events(
            X,
            text_df,
            labels,
        )

        section(
            "AUDIT COMPLETE"
        )

        print(
            "No files were modified."
        )

        print(
            "No model was retrained."
        )

        print(
            "No thresholds were changed."
        )

        print(
            "\nUse this output to determine whether "
            "the inference representation is "
            "semantically aligned with training."
        )

    except Exception as exc:

        print()
        print("=" * 78)
        print(" AUDIT ERROR")
        print("=" * 78)

        print(
            f"\n{type(exc).__name__}: {exc}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()