import json
import re
from pathlib import Path

import pandas as pd


# =============================================================================
# SmartSOC Windows-APT v4.1
# Full-Input Ambiguity Investigation
#
# Purpose:
#   Investigate the exact full-input representations that have conflicting
#   MITRE labels.
#
# Full representation:
#   62 structured features + behavior_text_normalized
#
# IMPORTANT:
#   This script is READ-ONLY.
#   It does NOT modify the dataset or model.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"

REPORT_DIR = ROOT / "reports" / "phase1_final"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = (
    REPORT_DIR
    / "full_input_ambiguity_investigation_v4_1.json"
)

REPORT_CSV = (
    REPORT_DIR
    / "full_input_ambiguity_investigation_v4_1.csv"
)


# =============================================================================
# FILES
# =============================================================================

FILES = {
    "structured_train": DATA / "structured_train.csv",
    "structured_test": DATA / "structured_test.csv",
    "labels_train": DATA / "labels_train.csv",
    "labels_test": DATA / "labels_test.csv",
    "metadata_train": DATA / "metadata_train.csv",
    "metadata_test": DATA / "metadata_test.csv",
    "behavior_train": DATA / "behavior_text_train.csv",
    "behavior_test": DATA / "behavior_text_test.csv",
}


# =============================================================================
# EXPECTED STRUCTURED FEATURES
# =============================================================================

EXPECTED_STRUCTURED_COLUMNS = [
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


# =============================================================================
# HELPERS
# =============================================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def detect_behavior_column(df):
    columns = list(df.columns)

    if "behavior_text_normalized" in columns:
        return "behavior_text_normalized"

    if len(columns) == 1:
        return columns[0]

    candidates = [
        c for c in columns
        if "behavior" in c.lower()
        or "text" in c.lower()
    ]

    if candidates:
        return candidates[0]

    return columns[0]


def build_full_representation(
    structured,
    behavior
):
    """
    Build exactly the same conceptual representation used by the
    Phase 1 final gate:
        62 structured features + behavior text.
    """

    structured_part = structured.copy()

    for column in structured_part.columns:
        structured_part[column] = (
            structured_part[column]
            .map(normalize_text)
        )

    behavior_column = detect_behavior_column(
        behavior
    )

    behavior_text = (
        behavior[behavior_column]
        .map(normalize_text)
    )

    structured_strings = (
        structured_part
        .astype(str)
        .agg("||".join, axis=1)
    )

    return (
        structured_strings
        + "||BEHAVIOR_TEXT="
        + behavior_text
    )


def parse_timestamp(series):
    """
    Parse:
        Dec 1, 2024 @ 06:24:28.026
    """

    raw = (
        series
        .astype(str)
        .str.strip()
    )

    cleaned = raw.str.replace(
        " @ ",
        " ",
        regex=False
    )

    parsed = pd.to_datetime(
        cleaned,
        format="%b %d, %Y %H:%M:%S.%f",
        errors="coerce"
    )

    missing = parsed.isna()

    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            cleaned.loc[missing],
            format="%b %d, %Y %H:%M:%S",
            errors="coerce"
        )

    return parsed


def label_counts(group):
    counts = (
        group["techniques"]
        .fillna("")
        .astype(str)
        .value_counts()
    )

    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def unique_values(group, column):
    if column not in group.columns:
        return []

    values = (
        group[column]
        .fillna("")
        .astype(str)
        .map(str.strip)
        .unique()
        .tolist()
    )

    return [
        v for v in values
        if v
    ]


def truncate_text(value, maximum=1000):
    value = str(value)

    if len(value) <= maximum:
        return value

    return value[:maximum] + "...[TRUNCATED]"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT v4.1")
print("FULL-INPUT AMBIGUITY INVESTIGATION")
print("=" * 80)

print("\nThis script is READ-ONLY.")
print("No dataset or model files will be modified.")


# =============================================================================
# LOAD
# =============================================================================

print("\nLoading datasets...")

structured_train = pd.read_csv(
    FILES["structured_train"]
)

structured_test = pd.read_csv(
    FILES["structured_test"]
)

labels_train = pd.read_csv(
    FILES["labels_train"]
)

labels_test = pd.read_csv(
    FILES["labels_test"]
)

metadata_train = pd.read_csv(
    FILES["metadata_train"]
)

metadata_test = pd.read_csv(
    FILES["metadata_test"]
)

behavior_train = pd.read_csv(
    FILES["behavior_train"]
)

behavior_test = pd.read_csv(
    FILES["behavior_test"]
)


# =============================================================================
# BASIC VALIDATION
# =============================================================================

print("\nDataset sizes:")

print(
    f"Train structured : {len(structured_train):,}"
)

print(
    f"Test structured  : {len(structured_test):,}"
)

print(
    f"Train labels     : {len(labels_train):,}"
)

print(
    f"Test labels      : {len(labels_test):,}"
)


if list(structured_train.columns) != EXPECTED_STRUCTURED_COLUMNS:
    raise RuntimeError(
        "Train structured schema does not match expected 62 columns."
    )


if list(structured_test.columns) != EXPECTED_STRUCTURED_COLUMNS:
    raise RuntimeError(
        "Test structured schema does not match expected 62 columns."
    )


if len(structured_train) != len(labels_train):
    raise RuntimeError(
        "Train structured/label row count mismatch."
    )


if len(structured_test) != len(labels_test):
    raise RuntimeError(
        "Test structured/label row count mismatch."
    )


if len(structured_train) != len(metadata_train):
    raise RuntimeError(
        "Train structured/metadata row count mismatch."
    )


if len(structured_test) != len(metadata_test):
    raise RuntimeError(
        "Test structured/metadata row count mismatch."
    )


if len(structured_train) != len(behavior_train):
    raise RuntimeError(
        "Train structured/behavior row count mismatch."
    )


if len(structured_test) != len(behavior_test):
    raise RuntimeError(
        "Test structured/behavior row count mismatch."
    )


# =============================================================================
# BUILD FULL REPRESENTATIONS
# =============================================================================

print("\nBuilding full-input representations...")

train_representation = build_full_representation(
    structured_train,
    behavior_train
)

test_representation = build_full_representation(
    structured_test,
    behavior_test
)


# =============================================================================
# COMBINE EVERYTHING
# =============================================================================

train_records = structured_train.copy()

train_records["techniques"] = (
    labels_train["techniques"]
    .fillna("")
    .astype(str)
)

train_records["timestamp"] = (
    metadata_train["timestamp"]
)

train_records["agent"] = (
    metadata_train["agent"]
)

train_records["metadata_event_id"] = (
    metadata_train["event_id"]
)

train_records["behavior_text"] = (
    behavior_train[
        detect_behavior_column(behavior_train)
    ]
    .fillna("")
    .astype(str)
)

train_records["split"] = "train"

train_records["full_representation"] = (
    train_representation
)


test_records = structured_test.copy()

test_records["techniques"] = (
    labels_test["techniques"]
    .fillna("")
    .astype(str)
)

test_records["timestamp"] = (
    metadata_test["timestamp"]
)

test_records["agent"] = (
    metadata_test["agent"]
)

test_records["metadata_event_id"] = (
    metadata_test["event_id"]
)

test_records["behavior_text"] = (
    behavior_test[
        detect_behavior_column(behavior_test)
    ]
    .fillna("")
    .astype(str)
)

test_records["split"] = "test"

test_records["full_representation"] = (
    test_representation
)


all_records = pd.concat(
    [
        train_records,
        test_records,
    ],
    ignore_index=True
)


# =============================================================================
# FIND AMBIGUOUS FULL REPRESENTATIONS
# =============================================================================

print("\nSearching for ambiguous full-input representations...")

ambiguous = []

grouped = all_records.groupby(
    "full_representation",
    sort=False
)


for representation, group in grouped:

    labels = (
        group["techniques"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    unique_labels = sorted(
        set(labels)
    )

    if len(unique_labels) <= 1:
        continue

    splits = set(
        group["split"]
    )

    ambiguous.append(
        {
            "representation": representation,
            "group": group.copy(),
            "labels": unique_labels,
            "splits": sorted(splits),
        }
    )


print(
    f"\nAmbiguous full-input representations found: "
    f"{len(ambiguous)}"
)


# =============================================================================
# EXPECTED RESULT
# =============================================================================

if len(ambiguous) == 0:

    print(
        "\n[PASS] No ambiguous full-input representations found."
    )

elif len(ambiguous) == 3:

    print(
        "\n[INFO] Exactly 3 ambiguous representations found, "
        "matching the Phase 1 final gate."
    )

else:

    print(
        "\n[WARNING] Number differs from the Phase 1 final gate."
    )


# =============================================================================
# INVESTIGATION
# =============================================================================

investigation = []
summary_rows = []


for index, item in enumerate(
    ambiguous,
    start=1
):

    representation = item["representation"]
    group = item["group"].copy()
    labels = item["labels"]
    splits = item["splits"]

    # ---------------------------------------------------------
    # Parse timestamps
    # ---------------------------------------------------------

    parsed_ts = parse_timestamp(
        group["timestamp"]
    )

    group["_parsed_timestamp"] = parsed_ts

    valid_ts = parsed_ts.dropna()

    if len(valid_ts) > 0:
        earliest = valid_ts.min()
        latest = valid_ts.max()
    else:
        earliest = pd.NaT
        latest = pd.NaT

    # ---------------------------------------------------------
    # Structured feature consistency
    # ---------------------------------------------------------

    structured_differences = []

    for column in EXPECTED_STRUCTURED_COLUMNS:

        values = (
            group[column]
            .map(normalize_text)
            .unique()
        )

        if len(values) > 1:
            structured_differences.append(
                {
                    "feature": column,
                    "values": [
                        str(v)
                        for v in values
                    ],
                }
            )

    # ---------------------------------------------------------
    # Behavior text consistency
    # ---------------------------------------------------------

    behavior_values = (
        group["behavior_text"]
        .map(normalize_text)
        .unique()
    )

    behavior_identical = (
        len(behavior_values) == 1
    )

    # ---------------------------------------------------------
    # Label counts
    # ---------------------------------------------------------

    counts = label_counts(
        group
    )

    # ---------------------------------------------------------
    # Split-specific label counts
    # ---------------------------------------------------------

    train_group = group[
        group["split"] == "train"
    ]

    test_group = group[
        group["split"] == "test"
    ]

    train_label_counts = label_counts(
        train_group
    ) if len(train_group) else {}

    test_label_counts = label_counts(
        test_group
    ) if len(test_group) else {}

    # ---------------------------------------------------------
    # Agent distribution
    # ---------------------------------------------------------

    agents = unique_values(
        group,
        "agent"
    )

    # ---------------------------------------------------------
    # Event IDs
    # ---------------------------------------------------------

    event_ids = unique_values(
        group,
        "metadata_event_id"
    )

    # ---------------------------------------------------------
    # Process information
    # ---------------------------------------------------------

    process_names = unique_values(
        group,
        "process_name"
    )

    parent_process_names = unique_values(
        group,
        "parent_process_name"
    )

    # ---------------------------------------------------------
    # Command information
    # ---------------------------------------------------------

    command_lengths = unique_values(
        group,
        "command_length"
    )

    # ---------------------------------------------------------
    # Key behavioral flags
    # ---------------------------------------------------------

    behavioral_flags = {}

    important_flags = [
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
        "administrative_powershell_context",
        "strong_behavior_signal",
    ]

    for flag in important_flags:

        if flag not in group.columns:
            continue

        values = (
            group[flag]
            .map(normalize_text)
            .unique()
            .tolist()
        )

        behavioral_flags[flag] = values

    # ---------------------------------------------------------
    # Behavior text
    # ---------------------------------------------------------

    if len(behavior_values) == 1:
        behavior_text = truncate_text(
            behavior_values[0],
            maximum=2000
        )
    else:
        behavior_text = [
            truncate_text(
                value,
                maximum=1000
            )
            for value in behavior_values
        ]

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    if (
        len(structured_differences) == 0
        and behavior_identical
    ):

        classification = (
            "TRUE_FULL_INPUT_AMBIGUITY"
        )

    else:

        classification = (
            "REPRESENTATION_CONSTRUCTION_MISMATCH"
        )

    # ---------------------------------------------------------
    # Cross split
    # ---------------------------------------------------------

    cross_split = (
        "train" in splits
        and "test" in splits
    )

    # ---------------------------------------------------------
    # Print investigation
    # ---------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        f"AMBIGUOUS REPRESENTATION #{index}"
    )
    print("=" * 80)

    print(
        f"Classification       : {classification}"
    )

    print(
        f"Rows                 : {len(group):,}"
    )

    print(
        f"Labels               : {', '.join(labels)}"
    )

    print(
        f"Cross-split          : {cross_split}"
    )

    print(
        f"Splits               : {', '.join(splits)}"
    )

    print(
        f"Earliest timestamp   : {earliest}"
    )

    print(
        f"Latest timestamp     : {latest}"
    )

    print(
        f"Agents               : {', '.join(agents)}"
    )

    print(
        f"Event IDs            : {', '.join(event_ids)}"
    )

    print(
        f"Process names        : {', '.join(process_names)}"
    )

    print(
        f"Parent processes     : "
        f"{', '.join(parent_process_names)}"
    )

    print(
        f"Command lengths      : "
        f"{', '.join(command_lengths)}"
    )

    print("\nLabel counts:")
    print(
        json.dumps(
            counts,
            indent=2
        )
    )

    print("\nTrain label counts:")
    print(
        json.dumps(
            train_label_counts,
            indent=2
        )
    )

    print("\nTest label counts:")
    print(
        json.dumps(
            test_label_counts,
            indent=2
        )
    )

    print(
        "\nStructured features different: "
        f"{len(structured_differences)}"
    )

    if structured_differences:

        for difference in structured_differences:

            print(
                f"  {difference['feature']}: "
                f"{difference['values']}"
            )

    print(
        "\nBehavior text identical: "
        f"{behavior_identical}"
    )

    print("\nBehavior text:")
    print(
        behavior_text
    )

    print("\nImportant behavioral flags:")

    for flag, values in behavioral_flags.items():

        print(
            f"  {flag}: {values}"
        )

    # ---------------------------------------------------------
    # Add detailed JSON record
    # ---------------------------------------------------------

    investigation.append(
        {
            "ambiguity_number": index,
            "classification": classification,
            "row_count": int(len(group)),
            "labels": labels,
            "label_counts": counts,
            "train_label_counts": train_label_counts,
            "test_label_counts": test_label_counts,
            "splits": splits,
            "cross_split": cross_split,
            "earliest_timestamp": (
                None
                if pd.isna(earliest)
                else str(earliest)
            ),
            "latest_timestamp": (
                None
                if pd.isna(latest)
                else str(latest)
            ),
            "agents": agents,
            "event_ids": event_ids,
            "process_names": process_names,
            "parent_process_names": parent_process_names,
            "command_lengths": command_lengths,
            "structured_feature_differences":
                structured_differences,
            "behavior_text_identical":
                behavior_identical,
            "behavior_text": behavior_text,
            "behavioral_flags": behavioral_flags,
            "representation": representation,
        }
    )

    # ---------------------------------------------------------
    # Summary row
    # ---------------------------------------------------------

    summary_rows.append(
        {
            "ambiguity_number": index,
            "classification": classification,
            "row_count": int(len(group)),
            "unique_labels": " | ".join(labels),
            "cross_split": cross_split,
            "splits": " | ".join(splits),
            "earliest_timestamp": (
                None
                if pd.isna(earliest)
                else str(earliest)
            ),
            "latest_timestamp": (
                None
                if pd.isna(latest)
                else str(latest)
            ),
            "agent_count": len(agents),
            "event_id_count": len(event_ids),
            "process_name_count": len(process_names),
            "structured_feature_difference_count":
                len(structured_differences),
            "behavior_text_identical":
                behavior_identical,
        }
    )


# =============================================================================
# SAVE REPORT
# =============================================================================

final_report = {
    "phase": "Phase 1",
    "investigation": "Full-input ambiguity",
    "read_only": True,

    "dataset": {
        "train_rows": int(len(structured_train)),
        "test_rows": int(len(structured_test)),
        "total_rows": int(len(all_records)),
    },

    "ambiguous_full_representations": {
        "count": len(ambiguous),
        "representations": investigation,
    },

    "conclusion": {
        "no_dataset_changes_made": True,
        "requires_manual_review": (
            len(ambiguous) > 0
        ),
    },
}


REPORT_JSON.write_text(
    json.dumps(
        final_report,
        indent=2,
        ensure_ascii=False,
        default=str
    ),
    encoding="utf-8"
)


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    REPORT_CSV,
    index=False,
    encoding="utf-8"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)

print(
    f"Ambiguous full representations : "
    f"{len(ambiguous)}"
)

if ambiguous:

    true_ambiguities = sum(
        1
        for item in investigation
        if item["classification"]
        == "TRUE_FULL_INPUT_AMBIGUITY"
    )

    construction_mismatches = sum(
        1
        for item in investigation
        if item["classification"]
        == "REPRESENTATION_CONSTRUCTION_MISMATCH"
    )

    print(
        f"True full-input ambiguities    : "
        f"{true_ambiguities}"
    )

    print(
        f"Construction mismatches       : "
        f"{construction_mismatches}"
    )

    print(
        "\n[REVIEW] Do not modify the dataset yet."
    )

else:

    print(
        "\n[PASS] No full-input ambiguities found."
    )


print("\nReports:")
print(
    REPORT_JSON
)

print(
    REPORT_CSV
)

print("\n" + "=" * 80)