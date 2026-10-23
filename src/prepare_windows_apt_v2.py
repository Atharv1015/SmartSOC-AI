import ast
import re
from pathlib import Path

import pandas as pd


# ============================================================
# SmartSOC
# Windows-APT 2025
# Model B - Endpoint Feature Engineering v2.1
# ============================================================

print("\n" + "=" * 70)
print(" SmartSOC Windows-APT - Feature Engineering v2.1")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

RAW_DIR = Path(
    "data/raw/Windows-APT 2025 A Dataset for APT-Inspired Attack"
)

INPUT_FILE = RAW_DIR / "combined.csv"

OUTPUT_DIR = Path(
    "data/processed/windows_apt_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_FILE = OUTPUT_DIR / "features.csv"
LABEL_FILE = OUTPUT_DIR / "labels.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CHUNK_SIZE = 10000


USE_COLUMNS = [
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

    "_source.data.win.eventdata.targetObject",

    "_source.rule.mitre.id",

    "_source.timestamp",
    "_source.agent.name",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_filename(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = value.replace("/", "\\")

    return value.split("\\")[-1].lower()


def extract_extension(value):

    if not value:
        return ""

    if "." not in value:
        return ""

    return value.rsplit(".", 1)[-1].lower()


def parse_mitre_ids(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            result = []

            for item in parsed:

                item = str(item).strip()

                if re.fullmatch(
                    r"T\d{4}(?:\.\d{3})?",
                    item
                ):
                    result.append(item)

            return sorted(set(result))

    except Exception:
        pass

    return sorted(
        set(
            re.findall(
                r"T\d{4}(?:\.\d{3})?",
                value
            )
        )
    )


def contains_any(series, keywords):

    pattern = "|".join(
        re.escape(k)
        for k in keywords
    )

    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
        .astype("int8")
    )


def present(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype("int8")
    )


# ------------------------------------------------------------
# Check schema
# ------------------------------------------------------------

print("\nChecking CSV schema...")

header = pd.read_csv(
    INPUT_FILE,
    nrows=0
)

available = [
    column
    for column in USE_COLUMNS
    if column in header.columns
]

missing = [
    column
    for column in USE_COLUMNS
    if column not in header.columns
]

print(
    f"Requested columns : {len(USE_COLUMNS)}"
)

print(
    f"Available columns : {len(available)}"
)

if missing:

    print("\nMissing columns:")

    for column in missing:
        print(" -", column)


# ------------------------------------------------------------
# Process
# ------------------------------------------------------------

feature_chunks = []
label_chunks = []
metadata_chunks = []

total_rows = 0
labeled_rows = 0


print("\nProcessing raw dataset...")


for df in pd.read_csv(
    INPUT_FILE,
    usecols=available,
    chunksize=CHUNK_SIZE,
    engine="python",
    on_bad_lines="skip"
):

    total_rows += len(df)

    print(
        f"\rProcessed: {total_rows:,}",
        end=""
    )

    # --------------------------------------------------------
    # Normalize strings
    # --------------------------------------------------------

    for column in df.columns:

        if column not in [
            "_source.data.win.system.eventID",
            "_source.data.win.system.level",
            "_source.data.win.eventdata.sourcePort",
            "_source.data.win.eventdata.destinationPort",
            "_source.data.win.eventdata.processId",
            "_source.data.win.eventdata.parentProcessId",
        ]:

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )


    # --------------------------------------------------------
    # MITRE labels
    # --------------------------------------------------------

    labels = (
        df["_source.rule.mitre.id"]
        .apply(parse_mitre_ids)
    )

    valid_mask = (
        labels.map(len) > 0
    )

    if not valid_mask.any():
        continue

    labeled_rows += int(valid_mask.sum())


    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

    event_id = pd.to_numeric(
        df["_source.data.win.system.eventID"],
        errors="coerce"
    ).fillna(-1)

    provider = (
        df["_source.data.win.system.providerName"]
        .str.lower()
    )

    channel = (
        df["_source.data.win.system.channel"]
        .str.lower()
    )

    image = df[
        "_source.data.win.eventdata.image"
    ]

    parent_image = df[
        "_source.data.win.eventdata.parentImage"
    ]

    command_line = df[
        "_source.data.win.eventdata.commandLine"
    ]

    parent_command = df[
        "_source.data.win.eventdata.parentCommandLine"
    ]

    image_name = image.apply(
        extract_filename
    )

    parent_name = parent_image.apply(
        extract_filename
    )


    # --------------------------------------------------------
    # Command line
    # --------------------------------------------------------

    cmd = command_line.str.lower()

    parent_cmd = parent_command.str.lower()

    combined_cmd = (
        cmd + " " + parent_cmd
    )


    # --------------------------------------------------------
    # User / privilege
    # --------------------------------------------------------

    user = (
        df[
            "_source.data.win.eventdata.user"
        ]
        .str.lower()
    )

    parent_user = (
        df[
            "_source.data.win.eventdata.parentUser"
        ]
        .str.lower()
    )

    integrity = (
        df[
            "_source.data.win.eventdata.integrityLevel"
        ]
        .str.lower()
    )


    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    features = pd.DataFrame(
        index=df.index
    )


    # Event
    features["event_id"] = event_id

    features["system_level"] = pd.to_numeric(
        df[
            "_source.data.win.system.level"
        ],
        errors="coerce"
    ).fillna(0)

    features["is_sysmon"] = (
        provider
        .str.contains(
            "sysmon",
            na=False
        )
        .astype("int8")
    )

    features["is_security_channel"] = (
        channel
        .str.contains(
            "security",
            na=False
        )
        .astype("int8")
    )


    # Process
    features["has_process_image"] = (
        image.ne("")
    ).astype("int8")

    features["has_parent_image"] = (
        parent_image.ne("")
    ).astype("int8")

    features["process_name"] = image_name

    features["parent_process_name"] = parent_name

    features["process_extension"] = (
        image_name.apply(
            extract_extension
        )
    )

    features["parent_extension"] = (
        parent_name.apply(
            extract_extension
        )
    )


    # Command
    features["command_length"] = (
        cmd.str.len()
    )

    features["has_command_line"] = (
        cmd.ne("")
    ).astype("int8")


    command_indicators = {

        "uses_powershell":
            ["powershell", "pwsh"],

        "uses_cmd":
            ["cmd.exe", "\\cmd ", "cmd /"],

        "uses_net":
            ["net.exe", "net1.exe", " net "],

        "uses_whoami":
            ["whoami"],

        "uses_schtasks":
            ["schtasks"],

        "uses_reg":
            ["reg.exe", " reg "],

        "uses_rundll32":
            ["rundll32"],

        "uses_wmic":
            ["wmic"],

        "uses_certutil":
            ["certutil"],

        "uses_bitsadmin":
            ["bitsadmin"],

        "uses_ipconfig":
            ["ipconfig"],

        "uses_netstat":
            ["netstat"],

        "uses_tasklist":
            ["tasklist"],

        "uses_sc":
            ["sc.exe", " sc "],

        "uses_mshta":
            ["mshta"],

        "uses_cscript":
            ["cscript"],

        "uses_wscript":
            ["wscript"],

        "uses_regsvr32":
            ["regsvr32"],

        "uses_forfiles":
            ["forfiles"],

        "uses_wevtutil":
            ["wevtutil"],

        "uses_vssadmin":
            ["vssadmin"],
    }


    for feature_name, keywords in command_indicators.items():

        features[feature_name] = contains_any(
            combined_cmd,
            keywords
        )


    features["has_encoded_command"] = (
        combined_cmd.str.contains(
            "-enc",
            na=False
        )
    ).astype("int8")


    features["has_download_behavior"] = (
        combined_cmd.str.contains(
            "download|invoke-webrequest|curl|wget|bitsadmin",
            regex=True,
            na=False
        )
    ).astype("int8")


    features["has_remote_execution_hint"] = (
        combined_cmd.str.contains(
            "psexec|winrm|wmic|invoke-command|enter-pssession",
            regex=True,
            na=False
        )
    ).astype("int8")


    features["has_registry_operation"] = (
        combined_cmd.str.contains(
            r"\breg\b|reg\.exe|hkey_|currentversion\\run",
            regex=True,
            na=False
        )
    ).astype("int8")


    features["has_file_deletion"] = (
        combined_cmd.str.contains(
            r"del |erase |remove-item|rmdir|rd ",
            regex=True,
            na=False
        )
    ).astype("int8")


    features["has_obfuscation_hint"] = (
        combined_cmd.str.contains(
            r"base64|frombase64string|encodedcommand|`",
            regex=True,
            na=False
        )
    ).astype("int8")


    # User
    features["has_user"] = (
        user.ne("")
    ).astype("int8")

    features["is_system_user"] = (
        user.str.contains(
            r"nt authority\\system|system",
            regex=True,
            na=False
        )
    ).astype("int8")

    features["is_admin_user"] = (
        user.str.contains(
            "administrator|admin",
            regex=True,
            na=False
        )
    ).astype("int8")

    features["has_parent_user"] = (
        parent_user.ne("")
    ).astype("int8")

    features["integrity_high"] = (
        integrity.str.contains(
            "high|system",
            regex=True,
            na=False
        )
    ).astype("int8")

    features["integrity_medium"] = (
        integrity.str.contains(
            "medium",
            na=False
        )
    ).astype("int8")


    # Process relationship
    features["same_process_parent"] = (
        image_name == parent_name
    ).astype("int8")


    features["suspicious_parent_wazuh"] = (
        parent_name.str.contains(
            "wazuh-agent|ossec-agent",
            regex=True,
            na=False
        )
    ).astype("int8")


    # Hash
    hashes = (
        df[
            "_source.data.win.eventdata.hashes"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    features["has_hash"] = (
        hashes.ne("")
    ).astype("int8")

    features["has_sha256"] = (
        hashes.str.contains(
            "sha256=",
            na=False
        )
    ).astype("int8")

    features["has_md5"] = (
        hashes.str.contains(
            "md5=",
            na=False
        )
    ).astype("int8")


    # Network
    source_ip = df[
        "_source.data.win.eventdata.sourceIp"
    ]

    destination_ip = df[
        "_source.data.win.eventdata.destinationIp"
    ]

    source_port = df[
        "_source.data.win.eventdata.sourcePort"
    ]

    destination_port = df[
        "_source.data.win.eventdata.destinationPort"
    ]

    protocol = df[
        "_source.data.win.eventdata.protocol"
    ]


    features["has_source_ip"] = present(
        source_ip
    )

    features["has_destination_ip"] = present(
        destination_ip
    )

    features["has_source_port"] = present(
        source_port
    )

    features["has_destination_port"] = present(
        destination_port
    )

    features["has_protocol"] = present(
        protocol
    )

    features["is_network_event"] = (
        (
            features["has_source_ip"]
            + features["has_destination_ip"]
            + features["has_destination_port"]
        ) > 0
    ).astype("int8")


    # Other telemetry
    features["has_target_filename"] = present(
        df[
            "_source.data.win.eventdata.targetFilename"
        ]
    )

    features["has_image_loaded"] = present(
        df[
            "_source.data.win.eventdata.imageLoaded"
        ]
    )

    features["has_target_object"] = present(
        df[
            "_source.data.win.eventdata.targetObject"
        ]
    )


    # --------------------------------------------------------
    # Keep only labeled events
    # --------------------------------------------------------

    features = features.loc[
        valid_mask
    ]

    label_values = labels.loc[
        valid_mask
    ].apply(
        lambda x: ",".join(x)
    )

    metadata = pd.DataFrame({

        "timestamp":
            df.loc[
                valid_mask,
                "_source.timestamp"
            ].values,

        "agent":
            df.loc[
                valid_mask,
                "_source.agent.name"
            ].values,

        "event_id":
            event_id.loc[
                valid_mask
            ].values,

        "techniques":
            label_values.values

    })


    feature_chunks.append(
        features
    )

    label_chunks.append(
        pd.DataFrame({
            "techniques":
                label_values.values
        })
    )

    metadata_chunks.append(
        metadata
    )


print("\n\nFeature engineering complete.")


# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

features = pd.concat(
    feature_chunks,
    ignore_index=True
)

labels = pd.concat(
    label_chunks,
    ignore_index=True
)

metadata = pd.concat(
    metadata_chunks,
    ignore_index=True
)


# ------------------------------------------------------------
# IMPORTANT:
# Do NOT aggressively deduplicate engineered features.
#
# Multiple attack events can legitimately have identical
# behavioral indicators but different labels.
# ------------------------------------------------------------

print(
    f"\nLabeled events before deduplication: "
    f"{len(features):,}"
)

duplicate_count = features.duplicated().sum()

print(
    f"Duplicate feature rows: "
    f"{duplicate_count:,}"
)

print(
    "Keeping duplicate behavioral events "
    "for multi-label training."
)


# ------------------------------------------------------------
# Clean categorical fields
# ------------------------------------------------------------

features = features.fillna("")

labels = labels.fillna("")

metadata = metadata.fillna("")


# ------------------------------------------------------------
# Technique statistics
# ------------------------------------------------------------

technique_counts = {}

for value in labels["techniques"].tolist():

    for technique in value.split(","):

        technique_counts[technique] = (
            technique_counts.get(
                technique,
                0
            ) + 1
        )


technique_distribution = (
    pd.Series(
        technique_counts,
        dtype="int64"
    )
    .sort_values(
        ascending=False
    )
)


# ------------------------------------------------------------
# Multi-label statistics
# ------------------------------------------------------------

label_series = labels[
    "techniques"
]

multi_label_count = (
    label_series
    .str.contains(
        ",",
        regex=False,
        na=False
    )
    .sum()
)


single_label_count = (
    len(label_series)
    - multi_label_count
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

features.to_csv(
    FEATURE_FILE,
    index=False
)

labels.to_csv(
    LABEL_FILE,
    index=False
)

metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print(" FINAL WINDOWS-APT v2.1 DATASET")
print("=" * 70)

print(
    f"\nRaw rows processed       : "
    f"{total_rows:,}"
)

print(
    f"MITRE-labeled rows      : "
    f"{labeled_rows:,}"
)

print(
    f"Final events            : "
    f"{len(features):,}"
)

print(
    f"Feature count           : "
    f"{len(features.columns)}"
)

print(
    f"Unique techniques       : "
    f"{len(technique_distribution)}"
)

print(
    f"Single-label events     : "
    f"{single_label_count:,}"
)

print(
    f"Multi-label events      : "
    f"{multi_label_count:,}"
)

print(
    f"Duplicate feature rows  : "
    f"{duplicate_count:,}"
)

print("\nTop techniques:")

print(
    technique_distribution
    .head(20)
    .to_string()
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

print("\n" + "=" * 70)
print(
    " Windows-APT Feature Engineering Complete"
)
print("=" * 70)