import ast
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# SmartSOC - Windows-APT 2025
# Model B: MITRE ATT&CK Multi-Label Dataset Preparation
# ============================================================

print("\n" + "=" * 70)
print(" SmartSOC Windows-APT 2025 - Model B Preparation")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = Path("data/raw/Windows-APT 2025 A Dataset for APT-Inspired Attack")

INPUT_FILE = BASE_DIR / "combined.csv"

OUTPUT_DIR = Path("data/processed/windows_apt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = OUTPUT_DIR / "endpoint_features.csv"
LABEL_FILE = OUTPUT_DIR / "technique_labels.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHUNK_SIZE = 10000

# Fields we absolutely do NOT want as model features.
# These contain the answers or Wazuh's interpretation.
LEAKAGE_COLUMNS = {
    "_source.rule.mitre.id",
    "_source.rule.mitre.technique",
    "_source.rule.mitre.tactic",
    "_source.rule.id",
    "_source.rule.description",
    "_source.rule.groups",
}

# Metadata fields that identify an event but should not be
# treated as behavioral features.
METADATA_COLUMNS = {
    "timestamp",
    "_source.timestamp",
    "_source.agent.name",
    "_source.agent.id",
    "_source.location",
    "_source.decoder.name",
}

# Candidate telemetry fields.
CANDIDATE_FEATURES = [
    "_source.data.win.system.eventID",
    "_source.data.win.system.providerName",
    "_source.data.win.system.channel",
    "_source.data.win.system.computer",
    "_source.data.win.system.level",
    "_source.data.win.eventdata.image",
    "_source.data.win.eventdata.parentImage",
    "_source.data.win.eventdata.commandLine",
    "_source.data.win.eventdata.parentCommandLine",
    "_source.data.win.eventdata.user",
    "_source.data.win.eventdata.parentUser",
    "_source.data.win.eventdata.integrityLevel",
    "_source.data.win.eventdata.processId",
    "_source.data.win.eventdata.parentProcessId",
    "_source.data.win.eventdata.currentDirectory",
    "_source.data.win.eventdata.originalFileName",
    "_source.data.win.eventdata.hashes",
    "_source.data.win.eventdata.targetFilename",
    "_source.data.win.eventdata.imageLoaded",
    "_source.data.win.eventdata.sourceIp",
    "_source.data.win.eventdata.sourcePort",
    "_source.data.win.eventdata.destinationIp",
    "_source.data.win.eventdata.destinationPort",
    "_source.data.win.eventdata.protocol",
    "_source.data.win.eventdata.initiated",
    "_source.data.win.eventdata.queryName",
    "_source.data.win.eventdata.queryStatus",
    "_source.data.win.eventdata.registryKey",
    "_source.data.win.eventdata.targetObject",
    "_source.full_log",
]


# ------------------------------------------------------------
# Helper: parse MITRE labels
# ------------------------------------------------------------

def parse_mitre_ids(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    # Example:
    # ["T1087","T1059.003"]

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            result = []

            for item in parsed:

                item = str(item).strip()

                if re.fullmatch(r"T\d{4}(?:\.\d{3})?", item):
                    result.append(item)

            return sorted(set(result))

    except Exception:
        pass

    # Fallback extraction
    matches = re.findall(
        r"T\d{4}(?:\.\d{3})?",
        value
    )

    return sorted(set(matches))


# ------------------------------------------------------------
# Determine available columns
# ------------------------------------------------------------

print("\nReading CSV header...")

header = pd.read_csv(
    INPUT_FILE,
    nrows=0
)

all_columns = list(header.columns)

print(
    f"Total raw columns: {len(all_columns)}"
)


# ------------------------------------------------------------
# Select useful columns
# ------------------------------------------------------------

available_features = [
    column
    for column in CANDIDATE_FEATURES
    if column in all_columns
]

missing_candidates = [
    column
    for column in CANDIDATE_FEATURES
    if column not in all_columns
]

print(
    f"Candidate feature columns found: "
    f"{len(available_features)}"
)

if missing_candidates:

    print("\nCandidate fields not found:")

    for column in missing_candidates:
        print(" -", column)


# MITRE labels
MITRE_ID_COLUMN = "_source.rule.mitre.id"

if MITRE_ID_COLUMN not in all_columns:

    raise RuntimeError(
        "MITRE ID column was not found."
    )


# ------------------------------------------------------------
# Load data in chunks
# ------------------------------------------------------------

print("\nProcessing dataset in chunks...")

usecols = available_features + [
    MITRE_ID_COLUMN
]

feature_chunks = []
label_chunks = []

total_rows = 0
labeled_rows = 0
unlabeled_rows = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        engine="python",
        on_bad_lines="skip"
    ),
    start=1
):

    total_rows += len(chunk)

    print(
        f"\rProcessed rows: {total_rows:,}",
        end=""
    )

    # Parse MITRE labels
    chunk["__techniques"] = (
        chunk[MITRE_ID_COLUMN]
        .apply(parse_mitre_ids)
    )

    # Keep only events with at least one technique.
    labeled_mask = (
        chunk["__techniques"].str.len() > 0
    )

    labeled = chunk.loc[
        labeled_mask
    ].copy()

    unlabeled_rows += (
        (~labeled_mask).sum()
    )

    labeled_rows += len(labeled)

    if len(labeled) == 0:
        continue

    # Feature data
    feature_data = labeled[
        available_features
    ].copy()

    # Label data
    label_data = pd.DataFrame({
        "techniques": labeled["__techniques"]
            .apply(lambda x: ",".join(x))
    })

    feature_chunks.append(
        feature_data
    )

    label_chunks.append(
        label_data
    )


print("\n\nChunk processing complete.")


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

if not feature_chunks:

    raise RuntimeError(
        "No MITRE-labeled records were found."
    )


features = pd.concat(
    feature_chunks,
    ignore_index=True
)

labels = pd.concat(
    label_chunks,
    ignore_index=True
)


# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------

print("\nCleaning data...")

features = features.replace(
    [r"^\s*$"],
    pd.NA,
    regex=True
)

features = features.fillna("")

# Convert everything to string for the first preprocessing
# stage. We will perform proper encoding later.
for column in features.columns:

    features[column] = (
        features[column]
        .astype(str)
        .str.strip()
    )


# ------------------------------------------------------------
# Remove completely empty features
# ------------------------------------------------------------

empty_columns = []

for column in features.columns:

    if (
        features[column]
        .replace("", pd.NA)
        .isna()
        .all()
    ):

        empty_columns.append(column)


if empty_columns:

    features = features.drop(
        columns=empty_columns
    )

    print(
        f"Removed completely empty columns: "
        f"{len(empty_columns)}"
    )


# ------------------------------------------------------------
# Remove duplicate feature/label combinations
# ------------------------------------------------------------

combined = pd.concat(
    [
        features,
        labels
    ],
    axis=1
)

before_duplicates = len(combined)

combined = combined.drop_duplicates()

after_duplicates = len(combined)

print(
    f"Duplicate rows removed: "
    f"{before_duplicates - after_duplicates:,}"
)


features = combined[
    features.columns
]

labels = combined[
    ["techniques"]
]


# ------------------------------------------------------------
# Technique statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print(" MITRE TECHNIQUE ANALYSIS")
print("=" * 70)


technique_counts = {}

for value in labels["techniques"]:

    techniques = value.split(",")

    for technique in techniques:

        technique_counts[technique] = (
            technique_counts.get(
                technique,
                0
            ) + 1
        )


technique_distribution = (
    pd.Series(technique_counts)
    .sort_values(
        ascending=False
    )
)


print(
    "\nNumber of unique MITRE techniques:",
    len(technique_distribution)
)

print(
    "\nTechnique distribution:"
)

print(
    technique_distribution
    .to_string()
)


# ------------------------------------------------------------
# Multi-label statistics
# ------------------------------------------------------------

label_count = (
    labels["techniques"]
    .apply(
        lambda x: len(x.split(","))
    )
)

print(
    "\nAverage techniques per event:",
    round(
        label_count.mean(),
        3
    )
)

print(
    "Maximum techniques per event:",
    label_count.max()
)

print(
    "Single-technique events:",
    (label_count == 1).sum()
)

print(
    "Multi-technique events:",
    (label_count > 1).sum()
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

print("\nSaving processed datasets...")


features.to_csv(
    FEATURE_FILE,
    index=False
)

labels.to_csv(
    LABEL_FILE,
    index=False
)


metadata = pd.DataFrame({

    "metric": [
        "raw_rows_processed",
        "mitre_labeled_rows",
        "unlabeled_rows",
        "final_rows",
        "raw_columns",
        "final_feature_columns",
        "unique_mitre_techniques",
        "duplicate_rows_removed"
    ],

    "value": [
        total_rows,
        labeled_rows,
        unlabeled_rows,
        len(features),
        len(all_columns),
        len(features.columns),
        len(technique_distribution),
        before_duplicates - after_duplicates
    ]

})

metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print(" WINDOWS-APT PREPARATION COMPLETE")
print("=" * 70)

print(
    f"\nRows processed       : {total_rows:,}"
)

print(
    f"MITRE-labeled rows   : {labeled_rows:,}"
)

print(
    f"Unlabeled rows       : {unlabeled_rows:,}"
)

print(
    f"Final rows           : {len(features):,}"
)

print(
    f"Feature columns      : {len(features.columns)}"
)

print(
    f"MITRE techniques     : "
    f"{len(technique_distribution)}"
)

print("\nFiles created:")

print(
    f"Features : {FEATURE_FILE}"
)

print(
    f"Labels   : {LABEL_FILE}"
)

print(
    f"Metadata : {METADATA_FILE}"
)

print(
    "\nNext step: inspect the processed dataset "
    "before training the multi-label model."
)