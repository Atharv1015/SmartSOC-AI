import ast
import re
from pathlib import Path

import pandas as pd


# ============================================================
# SmartSOC
# Windows-APT 2025
# Model B - Endpoint Dataset v3
#
# Hybrid representation:
#   1. Structured behavioral features
#   2. Sanitized behavioral text
#   3. Multi-label MITRE ATT&CK targets
#
# IMPORTANT:
# MITRE/Wazuh detection fields are NEVER model features.
# ============================================================

print("\n" + "=" * 75)
print(" SmartSOC Windows-APT - Model B v3 Preparation")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

RAW_DIR = Path(
    "data/raw/Windows-APT 2025 A Dataset for APT-Inspired Attack"
)

INPUT_FILE = RAW_DIR / "combined.csv"

OUTPUT_DIR = Path(
    "data/processed/windows_apt_v5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_FILE = OUTPUT_DIR / "structured_features.csv"
TEXT_FILE = OUTPUT_DIR / "behavior_text.csv"
LABEL_FILE = OUTPUT_DIR / "labels.csv"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 10000


# These are the raw fields we need.
USE_COLUMNS = [

    # Windows system information
    "_source.data.win.system.eventID",
    "_source.data.win.system.providerName",
    "_source.data.win.system.channel",
    "_source.data.win.system.computer",
    "_source.data.win.system.level",

    # Process
    "_source.data.win.eventdata.image",
    "_source.data.win.eventdata.parentImage",
    "_source.data.win.eventdata.commandLine",
    "_source.data.win.eventdata.parentCommandLine",

    # User / privilege
    "_source.data.win.eventdata.user",
    "_source.data.win.eventdata.parentUser",
    "_source.data.win.eventdata.integrityLevel",

    # Process IDs
    "_source.data.win.eventdata.processId",
    "_source.data.win.eventdata.parentProcessId",

    # File information
    "_source.data.win.eventdata.currentDirectory",
    "_source.data.win.eventdata.originalFileName",
    "_source.data.win.eventdata.targetFilename",
    "_source.data.win.eventdata.imageLoaded",

    # Hashes
    "_source.data.win.eventdata.hashes",

    # Network
    "_source.data.win.eventdata.sourceIp",
    "_source.data.win.eventdata.sourcePort",
    "_source.data.win.eventdata.destinationIp",
    "_source.data.win.eventdata.destinationPort",
    "_source.data.win.eventdata.protocol",
    "_source.data.win.eventdata.initiated",

    # Registry / object
    "_source.data.win.eventdata.targetObject",

    # --- v5.0 FIX: non-Sysmon behavioral text sources ---
    # The original 8 text fields below are Sysmon-only and are empty for
    # Security-channel and syscheck/FIM events, silently zeroing ~97% of
    # the feature vector for those techniques. These four fields carry
    # the equivalent behavioral signal for those event types.
    "_source.rule.description",
    "_source.data.win.system.message",
    "_source.syscheck.path",
    "_source.syscheck.value_name",

    # Ground-truth labels
    "_source.rule.mitre.id",

    # Evaluation metadata
    "_source.timestamp",
    "_source.agent.name",
]


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def filename(path):

    path = clean(path)

    if not path:
        return ""

    path = path.replace("/", "\\")

    return path.split("\\")[-1].lower()


def extension(name):

    name = clean(name)

    if "." not in name:
        return ""

    return name.rsplit(".", 1)[-1].lower()


def parse_mitre_ids(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            output = []

            for item in parsed:

                item = str(item).strip()

                if re.fullmatch(
                    r"T\d{4}(?:\.\d{3})?",
                    item
                ):
                    output.append(item)

            return sorted(set(output))

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


def presence(series):

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype("int8")
    )


def contains(series, pattern):

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


def sanitize_text(value):

    """
    Preserve useful behavioral tokens while removing
    highly specific artifacts that encourage memorization.

    Examples:

    C:\\Windows\\System32\\net.exe
        ->
    net.exe

    192.168.1.50
        ->
    <IP>

    SHA256=abcdef...
        ->
    <HASH>

    Process IDs / ports / arbitrary numbers
        ->
    <NUM>
    """

    value = clean(value).lower()

    if not value:
        return ""

    # Remove hashes
    value = re.sub(
        r"\b(?:md5|sha1|sha256|imphash)=[a-f0-9]{8,}\b",
        " <HASH> ",
        value,
        flags=re.IGNORECASE
    )

    # Generic long hexadecimal hashes
    value = re.sub(
        r"\b[a-f0-9]{32,64}\b",
        " <HASH> ",
        value
    )

    # IPv4
    value = re.sub(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        " <IP> ",
        value
    )

    # IPv6-ish addresses
    value = re.sub(
        r"\b[0-9a-f]{0,4}(?::[0-9a-f]{1,4}){2,7}\b",
        " <IP6> ",
        value,
        flags=re.IGNORECASE
    )

    # Windows paths:
    # preserve executable filename but remove directory.
    value = re.sub(
        r"[a-z]:\\(?:[^\\\s]+\\)*([^\\\s]+)",
        r" \1 ",
        value,
        flags=re.IGNORECASE
    )

    # UNC paths
    value = re.sub(
        r"\\\\[^\\\s]+\\(?:[^\\\s]+\\)*([^\\\s]+)",
        r" \1 ",
        value
    )

    # URLs
    value = re.sub(
        r"https?://[^\s]+",
        " <URL> ",
        value
    )

    # Long numeric values
    value = re.sub(
        r"\b\d{4,}\b",
        " <NUM> ",
        value
    )

    # Normalize separators
    value = re.sub(
        r"[\\/:,;|]+",
        " ",
        value
    )

    # Keep common command characters
    value = re.sub(
        r"[^a-z0-9_.<>\-+=' ]+",
        " ",
        value
    )

    # Collapse whitespace
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# SCHEMA
# ============================================================

print("\nChecking CSV schema...")

header = pd.read_csv(
    INPUT_FILE,
    nrows=0
)

available = [
    col
    for col in USE_COLUMNS
    if col in header.columns
]

missing = [
    col
    for col in USE_COLUMNS
    if col not in header.columns
]

print(
    f"Requested columns : {len(USE_COLUMNS)}"
)

print(
    f"Available columns : {len(available)}"
)

if missing:

    print("\nMissing columns:")

    for col in missing:
        print(" -", col)


# ============================================================
# CHUNK PROCESSING
# ============================================================

structured_chunks = []
text_chunks = []
label_chunks = []
metadata_chunks = []

total_rows = 0
labeled_rows = 0


print("\nProcessing raw Windows-APT dataset...")


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
    # Normalize object fields
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
    # LABEL
    # --------------------------------------------------------

    mitre = (
        df["_source.rule.mitre.id"]
        .apply(parse_mitre_ids)
    )

    valid = (
        mitre.map(len) > 0
    )

    if not valid.any():
        continue

    labeled_rows += int(valid.sum())


    # --------------------------------------------------------
    # BASIC RAW FIELDS
    # --------------------------------------------------------

    event_id = pd.to_numeric(
        df["_source.data.win.system.eventID"],
        errors="coerce"
    ).fillna(-1)

    system_level = pd.to_numeric(
        df["_source.data.win.system.level"],
        errors="coerce"
    ).fillna(0)

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
    # DERIVED PROCESS FEATURES
    # --------------------------------------------------------

    process_name = image.apply(
        filename
    )

    parent_process_name = parent_image.apply(
        filename
    )

    process_ext = process_name.apply(
        extension
    )

    parent_ext = parent_process_name.apply(
        extension
    )


    # --------------------------------------------------------
    # DERIVED COMMAND FEATURES
    # --------------------------------------------------------

    cmd = command_line.str.lower()

    parent_cmd = parent_command.str.lower()

    combined_cmd = (
        cmd + " " + parent_cmd
    )


    # ========================================================
    # STRUCTURED FEATURES
    # ========================================================

    X = pd.DataFrame(index=df.index)

    # ---- Event ----

    X["event_id"] = event_id

    X["system_level"] = system_level

    X["is_sysmon"] = contains(
        provider,
        r"sysmon"
    )

    X["is_security_channel"] = contains(
        channel,
        r"security"
    )


    # ---- Process ----

    X["process_name"] = process_name

    X["parent_process_name"] = parent_process_name

    X["process_extension"] = process_ext

    X["parent_extension"] = parent_ext

    X["has_process_image"] = presence(
        image
    )

    X["has_parent_image"] = presence(
        parent_image
    )


    # ---- User / privilege ----

    X["has_user"] = presence(
        user
    )

    X["has_parent_user"] = presence(
        parent_user
    )

    X["is_system_user"] = contains(
        user,
        r"nt authority\\system|^system$"
    )

    X["is_admin_user"] = contains(
        user,
        r"administrator|admin"
    )

    X["integrity_high"] = contains(
        integrity,
        r"high|system"
    )

    X["integrity_medium"] = contains(
        integrity,
        r"medium"
    )


    # ---- Command statistics ----

    X["command_length"] = (
        cmd.str.len()
    )

    X["parent_command_length"] = (
        parent_cmd.str.len()
    )

    X["has_command_line"] = presence(
        cmd
    )

    X["has_parent_command"] = presence(
        parent_cmd
    )


    # ---- Tool indicators ----

    X["uses_powershell"] = contains(
        combined_cmd,
        r"\bpowershell(?:\.exe)?\b|\bpwsh(?:\.exe)?\b"
    )

    X["uses_cmd"] = contains(
        combined_cmd,
        r"\bcmd(?:\.exe)?\b"
    )

    X["uses_net"] = contains(
        combined_cmd,
        r"\bnet(?:\.exe)?\b|\bnet1(?:\.exe)?\b"
    )

    X["uses_whoami"] = contains(
        combined_cmd,
        r"\bwhoami(?:\.exe)?\b"
    )

    X["uses_schtasks"] = contains(
        combined_cmd,
        r"\bschtasks(?:\.exe)?\b"
    )

    X["uses_reg"] = contains(
        combined_cmd,
        r"\breg(?:\.exe)?\b"
    )

    X["uses_rundll32"] = contains(
        combined_cmd,
        r"\brundll32(?:\.exe)?\b"
    )

    X["uses_wmic"] = contains(
        combined_cmd,
        r"\bwmic(?:\.exe)?\b"
    )

    X["uses_certutil"] = contains(
        combined_cmd,
        r"\bcertutil(?:\.exe)?\b"
    )

    X["uses_bitsadmin"] = contains(
        combined_cmd,
        r"\bbitsadmin(?:\.exe)?\b"
    )

    X["uses_ipconfig"] = contains(
        combined_cmd,
        r"\bipconfig(?:\.exe)?\b"
    )

    X["uses_netstat"] = contains(
        combined_cmd,
        r"\bnetstat(?:\.exe)?\b"
    )

    X["uses_tasklist"] = contains(
        combined_cmd,
        r"\btasklist(?:\.exe)?\b"
    )

    X["uses_sc"] = contains(
        combined_cmd,
        r"\bsc(?:\.exe)?\b"
    )

    X["uses_mshta"] = contains(
        combined_cmd,
        r"\bmshta(?:\.exe)?\b"
    )

    X["uses_regsvr32"] = contains(
        combined_cmd,
        r"\bregsvr32(?:\.exe)?\b"
    )

    X["uses_cscript"] = contains(
        combined_cmd,
        r"\bcscript(?:\.exe)?\b"
    )

    X["uses_wscript"] = contains(
        combined_cmd,
        r"\bwscript(?:\.exe)?\b"
    )

    X["uses_wevtutil"] = contains(
        combined_cmd,
        r"\bwevtutil(?:\.exe)?\b"
    )

    X["uses_vssadmin"] = contains(
        combined_cmd,
        r"\bvssadmin(?:\.exe)?\b"
    )


    # ---- Behavioral indicators ----

    X["has_encoded_command"] = contains(
        combined_cmd,
        r"-enc(?:odedcommand)?\b"
    )

    X["has_download_behavior"] = contains(
        combined_cmd,
        r"invoke-webrequest|invoke-restmethod|download|curl|wget|bitsadmin"
    )

    X["has_remote_execution_hint"] = contains(
        combined_cmd,
        r"psexec|winrm|wmic|invoke-command|enter-pssession"
    )

    X["has_registry_operation"] = contains(
        combined_cmd,
        r"\breg(?:\.exe)?\b|hkey_|currentversion\\run"
    )

    X["has_file_deletion"] = contains(
        combined_cmd,
        r"\bdel\b|\berase\b|remove-item|rmdir|\brd\b"
    )

    X["has_obfuscation_hint"] = contains(
        combined_cmd,
        r"base64|frombase64string|encodedcommand"
    )


    # ---- Process relationship ----

    X["same_process_parent"] = (
        process_name == parent_process_name
    ).astype("int8")

    X["suspicious_parent_wazuh"] = contains(
        parent_process_name,
        r"wazuh-agent|ossec-agent"
    )


    # ---- Hash ----

    hashes = (
        df[
            "_source.data.win.eventdata.hashes"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    X["has_hash"] = presence(
        hashes
    )

    X["has_sha256"] = contains(
        hashes,
        r"sha256="
    )

    X["has_md5"] = contains(
        hashes,
        r"md5="
    )


    # ---- Network ----

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

    X["has_source_ip"] = presence(
        source_ip
    )

    X["has_destination_ip"] = presence(
        destination_ip
    )

    X["has_source_port"] = presence(
        source_port
    )

    X["has_destination_port"] = presence(
        destination_port
    )

    X["has_protocol"] = presence(
        protocol
    )

    X["is_network_event"] = (
        (
            X["has_source_ip"]
            + X["has_destination_ip"]
            + X["has_destination_port"]
        ) > 0
    ).astype("int8")


    # ---- Other telemetry ----

    X["has_target_filename"] = presence(
        df[
            "_source.data.win.eventdata.targetFilename"
        ]
    )

    X["has_image_loaded"] = presence(
        df[
            "_source.data.win.eventdata.imageLoaded"
        ]
    )

    X["has_target_object"] = presence(
        df[
            "_source.data.win.eventdata.targetObject"
        ]
    )


    # ========================================================
    # BEHAVIORAL TEXT
    # ========================================================

    text_columns = [

        command_line,
        parent_command,

        image,
        parent_image,

        df[
            "_source.data.win.eventdata.originalFileName"
        ],

        df[
            "_source.data.win.eventdata.targetFilename"
        ],

        df[
            "_source.data.win.eventdata.imageLoaded"
        ],

        df[
            "_source.data.win.eventdata.targetObject"
        ],

        # --- v5.0 FIX: non-Sysmon fallback text ---
        df[
            "_source.rule.description"
        ],

        df[
            "_source.data.win.system.message"
        ],

        df[
            "_source.syscheck.path"
        ],

        df[
            "_source.syscheck.value_name"
        ],
    ]


    behavior_text = pd.Series(
        "",
        index=df.index,
        dtype="object"
    )


    for column in text_columns:

        sanitized = column.apply(
            sanitize_text
        )

        behavior_text = (
            behavior_text
            + " "
            + sanitized
        )


    behavior_text = (
        behavior_text
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_strings = (
        mitre.loc[valid]
        .apply(
            lambda x: ",".join(x)
        )
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame({

        "timestamp":
            df.loc[
                valid,
                "_source.timestamp"
            ].values,

        "agent":
            df.loc[
                valid,
                "_source.agent.name"
            ].values,

        "event_id":
            event_id.loc[
                valid
            ].values,

        "techniques":
            label_strings.values
    })


    # --------------------------------------------------------
    # Keep only labeled events
    # --------------------------------------------------------

    X = X.loc[
        valid
    ].copy()

    behavior_text = behavior_text.loc[
        valid
    ].copy()


    structured_chunks.append(
        X
    )

    text_chunks.append(
        pd.DataFrame({
            "behavior_text":
                behavior_text.values
        })
    )

    label_chunks.append(
        pd.DataFrame({
            "techniques":
                label_strings.values
        })
    )

    metadata_chunks.append(
        metadata
    )


print("\n\nRaw processing complete.")


# ============================================================
# COMBINE
# ============================================================

X = pd.concat(
    structured_chunks,
    ignore_index=True
)

text = pd.concat(
    text_chunks,
    ignore_index=True
)

y = pd.concat(
    label_chunks,
    ignore_index=True
)

metadata = pd.concat(
    metadata_chunks,
    ignore_index=True
)


# ============================================================
# CLEAN
# ============================================================

X = X.fillna("")

text = text.fillna("")

y = y.fillna("")

metadata = metadata.fillna("")


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print(" DUPLICATE ANALYSIS")
print("=" * 75)

print(
    "\nTotal labeled events:",
    len(X)
)

print(
    "Unique structured rows:",
    len(X.drop_duplicates())
)

print(
    "Duplicate structured rows:",
    X.duplicated().sum()
)

print(
    "Unique behavior texts:",
    text["behavior_text"].nunique()
)

print(
    "Empty behavior texts:",
    (
        text["behavior_text"]
        .eq("")
        .sum()
    )
)


# ============================================================
# LABEL ANALYSIS
# ============================================================

technique_counts = {}

for value in y["techniques"]:

    for technique in value.split(","):

        technique_counts[technique] = (
            technique_counts.get(
                technique,
                0
            ) + 1
        )


distribution = (
    pd.Series(
        technique_counts,
        dtype="int64"
    )
    .sort_values(
        ascending=False
    )
)


multi_label = (
    y["techniques"]
    .str.contains(
        ",",
        regex=False,
        na=False
    )
)


print("\n" + "=" * 75)
print(" LABEL ANALYSIS")
print("=" * 75)

print(
    "\nUnique MITRE techniques:",
    len(distribution)
)

print(
    "Single-label events:",
    (~multi_label).sum()
)

print(
    "Multi-label events:",
    multi_label.sum()
)

print(
    "\nTop techniques:"
)

print(
    distribution
    .head(25)
    .to_string()
)


# ============================================================
# TIMESTAMP ANALYSIS
# ============================================================

timestamps = pd.to_datetime(
    metadata["timestamp"],
    format="%b %d, %Y @ %H:%M:%S.%f",
    errors="coerce"
)

print("\n" + "=" * 75)
print(" TIMESTAMP ANALYSIS")
print("=" * 75)

print(
    "\nValid timestamps:",
    timestamps.notna().sum()
)

print(
    "Invalid timestamps:",
    timestamps.isna().sum()
)

if timestamps.notna().any():

    print(
        "Earliest:",
        timestamps.min()
    )

    print(
        "Latest:",
        timestamps.max()
    )

    print(
        "\nEvents per date:"
    )

    print(
        timestamps
        .dt.date
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# SAVE
# ============================================================

X.to_csv(
    FEATURE_FILE,
    index=False
)

text.to_csv(
    TEXT_FILE,
    index=False
)

y.to_csv(
    LABEL_FILE,
    index=False
)

metadata.to_csv(
    METADATA_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(" WINDOWS-APT MODEL B v3 READY")
print("=" * 75)

print(
    f"\nRaw rows processed       : {total_rows:,}"
)

print(
    f"MITRE-labeled events    : {labeled_rows:,}"
)

print(
    f"Final events            : {len(X):,}"
)

print(
    f"Structured features     : {len(X.columns)}"
)

print(
    f"Behavior text rows      : {len(text):,}"
)

print(
    f"MITRE techniques        : {len(distribution)}"
)

print(
    f"Multi-label events      : {multi_label.sum():,}"
)

print("\nFiles created:")

print(
    f"Structured : {FEATURE_FILE}"
)

print(
    f"Text       : {TEXT_FILE}"
)

print(
    f"Labels     : {LABEL_FILE}"
)

print(
    f"Metadata   : {METADATA_FILE}"
)

print("\n" + "=" * 75)
print(
    "Next step: leakage-safe split + TF-IDF + MultiLabelBinarizer"
)
print("=" * 75)