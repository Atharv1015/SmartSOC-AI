import pandas as pd
import numpy as np
import json
import re
from pathlib import Path


# ============================================================
# SmartSOC Windows-APT Model B v4
# Label / Duplication / Behavior Preparation
# ============================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_STRUCTURED = (
    BASE / "data" / "processed" / "windows_apt_v3" /
    "structured_features.csv"
)

INPUT_TEXT = (
    BASE / "data" / "processed" / "windows_apt_v3" /
    "behavior_text.csv"
)

INPUT_LABELS = (
    BASE / "data" / "processed" / "windows_apt_v3" /
    "labels.csv"
)

INPUT_METADATA = (
    BASE / "data" / "processed" / "windows_apt_v3" /
    "metadata.csv"
)

OUTPUT_DIR = (
    BASE / "data" / "processed" / "windows_apt_v4"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 75)
print(" SmartSOC Windows-APT - Model B v4 Preparation")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading v3 datasets...")

X = pd.read_csv(INPUT_STRUCTURED)
T = pd.read_csv(INPUT_TEXT)
Y = pd.read_csv(INPUT_LABELS)
M = pd.read_csv(INPUT_METADATA)

print("Structured rows :", len(X))
print("Text rows       :", len(T))
print("Label rows      :", len(Y))
print("Metadata rows   :", len(M))


if not (
    len(X) == len(T) == len(Y) == len(M)
):
    raise RuntimeError(
        "Dataset row counts do not match."
    )


# ============================================================
# 2. TIMESTAMP NORMALIZATION
# ============================================================

print("\nParsing timestamps...")

M["timestamp_parsed"] = pd.to_datetime(
    M["timestamp"],
    format="%b %d, %Y @ %H:%M:%S.%f",
    errors="coerce"
)

invalid = M["timestamp_parsed"].isna().sum()

if invalid:
    raise RuntimeError(
        f"Invalid timestamps detected: {invalid}"
    )

print(
    "Earliest:",
    M["timestamp_parsed"].min()
)

print(
    "Latest  :",
    M["timestamp_parsed"].max()
)


# ============================================================
# 3. NORMALIZE TECHNIQUE LABELS
# ============================================================

def normalize_techniques(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    techniques = [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]

    techniques = sorted(set(techniques))

    return ",".join(techniques)


Y["techniques"] = (
    Y["techniques"]
    .apply(normalize_techniques)
)


# Remove completely unlabeled events.
labeled_mask = Y["techniques"].ne("")

X = X.loc[labeled_mask].reset_index(drop=True)
T = T.loc[labeled_mask].reset_index(drop=True)
Y = Y.loc[labeled_mask].reset_index(drop=True)
M = M.loc[labeled_mask].reset_index(drop=True)

print("\nLabeled events:", len(X))


# ============================================================
# 4. BUILD BEHAVIOR TEXT
# ============================================================

print("\nNormalizing behavior text...")

if "behavior_text" in T.columns:

    behavior_col = "behavior_text"

elif len(T.columns) == 1:

    behavior_col = T.columns[0]

else:

    raise RuntimeError(
        "Could not determine behavior-text column."
    )


def normalize_text(value):

    if pd.isna(value):
        return ""

    text = str(value).lower()

    # Normalize Windows paths.
    text = re.sub(
        r"[a-z]:\\[^ ]+",
        " <path> ",
        text
    )

    # Normalize URLs.
    text = re.sub(
        r"https?://\S+",
        " <url> ",
        text
    )

    # Normalize IPv4 addresses.
    text = re.sub(
        r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        " <ip> ",
        text
    )

    # Normalize hashes.
    text = re.sub(
        r"\b[a-f0-9]{32,64}\b",
        " <hash> ",
        text
    )

    # Normalize long hexadecimal values.
    text = re.sub(
        r"\b0x[0-9a-f]+\b",
        " <hex> ",
        text
    )

    # Normalize excessive whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


T["behavior_text_normalized"] = (
    T[behavior_col]
    .apply(normalize_text)
)


empty_text = (
    T["behavior_text_normalized"]
    .eq("")
    .sum()
)

print(
    "Empty behavior text:",
    empty_text
)


# ============================================================
# 5. STRUCTURED BEHAVIOR SIGNATURE
# ============================================================

print("\nCreating behavioral signatures...")


categorical_columns = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension"
]


# Ensure missing values are consistent.
for col in categorical_columns:

    if col in X.columns:

        X[col] = (
            X[col]
            .fillna("<missing>")
            .astype(str)
            .str.lower()
            .str.strip()
        )


# Build a compact structured signature.
signature_columns = [
    "event_id",
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
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
    "has_hash",
    "has_source_ip",
    "has_destination_ip",
    "has_source_port",
    "has_destination_port",
    "has_protocol",
    "is_network_event",
    "has_target_filename",
    "has_image_loaded",
    "has_target_object"
]


signature_columns = [
    c for c in signature_columns
    if c in X.columns
]


X["_behavior_signature"] = (
    X[signature_columns]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================
# 6. COMBINE STRUCTURED + TEXT SIGNATURE
# ============================================================

combined_signature = (
    X["_behavior_signature"]
    + "||"
    + T["behavior_text_normalized"]
)

X["_combined_signature"] = combined_signature


# ============================================================
# 7. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print(" DUPLICATE ANALYSIS")
print("=" * 75)

structured_duplicates = (
    X["_behavior_signature"].duplicated().sum()
)

combined_duplicates = (
    X["_combined_signature"].duplicated().sum()
)

print(
    "Structured duplicate rows:",
    structured_duplicates
)

print(
    "Combined behavior duplicate rows:",
    combined_duplicates
)

print(
    "Unique structured signatures:",
    X["_behavior_signature"].nunique()
)

print(
    "Unique combined signatures:",
    X["_combined_signature"].nunique()
)


# ============================================================
# 8. LABEL-CONFLICT ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print(" LABEL CONFLICT ANALYSIS")
print("=" * 75)


signature_labels = (
    pd.DataFrame({
        "signature": X["_combined_signature"],
        "techniques": Y["techniques"]
    })
    .groupby("signature")["techniques"]
    .nunique()
)


conflicting_signatures = (
    signature_labels[signature_labels > 1]
)

print(
    "Behavior signatures with multiple label combinations:",
    len(conflicting_signatures)
)


if len(conflicting_signatures) > 0:

    print(
        "\nTop conflicting signatures:"
    )

    conflict_ids = (
        conflicting_signatures
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    for sig in conflict_ids:

        labels = (
            Y.loc[
                X["_combined_signature"] == sig,
                "techniques"
            ]
            .value_counts()
        )

        print("\nSignature:")
        print(sig[:500])

        print("Labels:")
        print(labels.to_string())


# ============================================================
# 9. POWERHELL LABEL AUDIT
# ============================================================

print("\n" + "=" * 75)
print(" POWERSHELL LABEL AUDIT")
print("=" * 75)


if "uses_powershell" in X.columns:

    powershell_mask = (
        X["uses_powershell"] == 1
    )

    powershell_labels = (
        Y.loc[
            powershell_mask,
            "techniques"
        ]
        .str.split(",")
        .explode()
        .value_counts()
    )

    print(
        "PowerShell events:",
        powershell_mask.sum()
    )

    print(
        "\nTop MITRE techniques associated with PowerShell:"
    )

    print(
        powershell_labels
        .head(25)
        .to_string()
    )


# ============================================================
# 10. ADMINISTRATIVE POWERHELL AUDIT
# ============================================================

print("\n" + "=" * 75)
print(" ADMINISTRATIVE POWERSHELL AUDIT")
print("=" * 75)


admin_commands = [
    "get-service",
    "get-process",
    "get-childitem",
    "get-item",
    "get-content",
    "get-command",
    "get-help",
    "get-computerinfo",
    "get-wmiobject",
    "get-ciminstance",
    "get-eventlog",
    "get-hotfix",
    "get-netadapter",
    "get-netipconfiguration",
    "get-nettcpconnection"
]


def looks_administrative(text):

    if not text:
        return False

    return any(
        command in text
        for command in admin_commands
    )


T["_administrative_powershell"] = (
    T["behavior_text_normalized"]
    .apply(looks_administrative)
)


admin_ps = (
    X.get(
        "uses_powershell",
        pd.Series(0, index=X.index)
    ).eq(1)
    &
    T["_administrative_powershell"]
)


print(
    "Potential administrative PowerShell events:",
    admin_ps.sum()
)

if admin_ps.sum() > 0:

    print(
        "\nLabels attached to administrative-looking PowerShell:"
    )

    print(
        Y.loc[
            admin_ps,
            "techniques"
        ]
        .str.split(",")
        .explode()
        .value_counts()
        .head(25)
        .to_string()
    )


# ============================================================
# 11. CREATE BENIGN-CONTEXT FLAG
# ============================================================

X["administrative_powershell_context"] = (
    admin_ps.astype(int)
)


# ============================================================
# 12. HIGH-CONFIDENCE ATTACK SIGNALS
# ============================================================

attack_signal_columns = [
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "uses_certutil",
    "uses_bitsadmin",
    "uses_mshta",
    "uses_rundll32",
    "uses_regsvr32",
    "uses_vssadmin"
]


attack_signal_columns = [
    c for c in attack_signal_columns
    if c in X.columns
]


X["strong_behavior_signal"] = (
    X[attack_signal_columns]
    .sum(axis=1)
    .gt(0)
    .astype(int)
)


print(
    "\nStrong behavioral-signal events:",
    X["strong_behavior_signal"].sum()
)


# ============================================================
# 13. TEMPORAL SPLIT
# ============================================================

print("\n" + "=" * 75)
print(" TEMPORAL SPLIT")
print("=" * 75)


dates = sorted(
    M["timestamp_parsed"]
    .dt.date
    .unique()
)

candidate_cutoff = dates[
    int(len(dates) * 0.80)
]

train_mask = (
    M["timestamp_parsed"]
    .dt.date
    <= candidate_cutoff
)

test_mask = ~train_mask


print(
    "Temporal cutoff:",
    candidate_cutoff
)

print(
    "Training events:",
    train_mask.sum()
)

print(
    "Testing events :",
    test_mask.sum()
)


# ============================================================
# 14. TRAINING DUPLICATE CONTROL
# ============================================================

print("\n" + "=" * 75)
print(" TRAINING DUPLICATE CONTROL")
print("=" * 75)


# We do NOT randomly remove duplicates.
#
# Instead:
# - temporal split happens first
# - duplicate signatures are identified
# - identical behavior + identical label can be collapsed
# - conflicting labels are retained for audit


train_indices = X.index[train_mask]

train_signature_labels = (
    pd.DataFrame({
        "signature": X.loc[
            train_indices,
            "_combined_signature"
        ],
        "techniques": Y.loc[
            train_indices,
            "techniques"
        ]
    })
)


# Keep one representative for identical
# signature + identical label.
keep_train = (
    ~train_signature_labels.duplicated(
        subset=["signature", "techniques"]
    )
)


kept_indices = (
    train_signature_labels
    .loc[keep_train]
    .index
)


removed_duplicate_count = (
    len(train_indices)
    - len(kept_indices)
)


print(
    "Original training events:",
    len(train_indices)
)

print(
    "Training representatives:",
    len(kept_indices)
)

print(
    "Collapsed exact duplicates:",
    removed_duplicate_count
)


# ============================================================
# 15. IMPORTANT: DO NOT DEDUPLICATE TEST SET
# ============================================================

print(
    "\nTesting events are retained in full."
)

print(
    "This preserves an honest temporal evaluation."
)


# ============================================================
# 16. BUILD OUTPUT DATASETS
# ============================================================

output_train_idx = kept_indices

output_test_idx = X.index[test_mask]


# Remove helper columns from model features.
helper_columns = [
    "_behavior_signature",
    "_combined_signature"
]


structured_output_columns = [
    c for c in X.columns
    if c not in helper_columns
]


X_train = X.loc[
    output_train_idx,
    structured_output_columns
].copy()

X_test = X.loc[
    output_test_idx,
    structured_output_columns
].copy()


T_train = T.loc[
    output_train_idx,
    ["behavior_text_normalized"]
].copy()

T_test = T.loc[
    output_test_idx,
    ["behavior_text_normalized"]
].copy()


Y_train = Y.loc[
    output_train_idx
].copy()

Y_test = Y.loc[
    output_test_idx
].copy()


M_train = M.loc[
    output_train_idx
].copy()

M_test = M.loc[
    output_test_idx
].copy()


# ============================================================
# 17. SAVE
# ============================================================

print("\n" + "=" * 75)
print(" SAVING MODEL B v4 DATA")
print("=" * 75)


X_train.to_csv(
    OUTPUT_DIR / "structured_train.csv",
    index=False
)

X_test.to_csv(
    OUTPUT_DIR / "structured_test.csv",
    index=False
)

T_train.to_csv(
    OUTPUT_DIR / "behavior_text_train.csv",
    index=False
)

T_test.to_csv(
    OUTPUT_DIR / "behavior_text_test.csv",
    index=False
)

Y_train.to_csv(
    OUTPUT_DIR / "labels_train.csv",
    index=False
)

Y_test.to_csv(
    OUTPUT_DIR / "labels_test.csv",
    index=False
)

M_train.to_csv(
    OUTPUT_DIR / "metadata_train.csv",
    index=False
)

M_test.to_csv(
    OUTPUT_DIR / "metadata_test.csv",
    index=False
)


# ============================================================
# 18. SAVE SCHEMA
# ============================================================

schema = {

    "version": "SmartSOC Windows-APT Model B v4",

    "structured_feature_count":
        len(structured_output_columns),

    "structured_features":
        structured_output_columns,

    "categorical_features":
        [
            c for c in categorical_columns
            if c in structured_output_columns
        ],

    "behavior_text_column":
        "behavior_text_normalized",

    "target":
        "techniques",

    "mitre_technique_count":
        int(
            Y["techniques"]
            .str.split(",")
            .explode()
            .nunique()
        ),

    "temporal_cutoff":
        str(candidate_cutoff),

    "training_events":
        int(len(X_train)),

    "testing_events":
        int(len(X_test)),

    "collapsed_training_duplicates":
        int(removed_duplicate_count),

    "label_conflicting_signatures":
        int(len(conflicting_signatures))
}


with open(
    OUTPUT_DIR / "schema.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        schema,
        f,
        indent=4
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(" WINDOWS-APT MODEL B v4 READY")
print("=" * 75)

print(
    "\nTraining events:",
    len(X_train)
)

print(
    "Testing events :",
    len(X_test)
)

print(
    "Structured features:",
    len(structured_output_columns)
)

print(
    "MITRE techniques:",
    schema["mitre_technique_count"]
)

print(
    "Collapsed training duplicates:",
    removed_duplicate_count
)

print(
    "Conflicting behavior signatures:",
    len(conflicting_signatures)
)

print(
    "\nFiles created:"
)

for file in sorted(OUTPUT_DIR.iterdir()):

    print(
        " -",
        file
    )


print(
    "\nNext step:"
)

print(
    "Run the v4 audit before training."
)

print(
    "Do NOT deploy this model yet."
)