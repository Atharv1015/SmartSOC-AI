from pathlib import Path
import pandas as pd


# ============================================================
# SmartSOC Windows-APT v4
# Problem Label Inspection - FIXED
# ============================================================

BASE = Path(__file__).resolve().parents[1]

DATA = BASE / "data" / "processed" / "windows_apt_v4"

STRUCTURED = DATA / "structured_train.csv"
TEXT = DATA / "behavior_text_train.csv"
LABELS = DATA / "labels_train.csv"
META = DATA / "metadata_train.csv"

OUT = BASE / "results" / "windows_apt_v4" / "audit"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"\nRequired file not found:\n{path}")


def unique_column(df, column, source_name):
    """
    Return exactly one column.

    Prevents the pandas DataFrame-vs-Series problem when duplicate
    column names exist.
    """

    matches = [
        i for i, c in enumerate(df.columns)
        if c == column
    ]

    if len(matches) == 0:
        raise KeyError(
            f"Column '{column}' not found in {source_name}"
        )

    if len(matches) > 1:
        print(
            f"[WARNING] {source_name} contains "
            f"{len(matches)} copies of '{column}'. "
            f"Using the first copy."
        )

    return df.iloc[:, matches[0]]


def contains_any(series, pattern):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False,
        )
    )


# ============================================================
# CHECK FILES
# ============================================================

header("CHECKING V4 DATASET")

for path in [
    STRUCTURED,
    TEXT,
    LABELS,
    META,
]:
    require(path)
    print("[OK]", path)


# ============================================================
# LOAD FILES
# ============================================================

header("LOADING V4 TRAINING DATA")

structured = pd.read_csv(STRUCTURED)
text = pd.read_csv(TEXT)
labels = pd.read_csv(LABELS)
meta = pd.read_csv(META)

print("Structured:", structured.shape)
print("Text      :", text.shape)
print("Labels    :", labels.shape)
print("Metadata  :", meta.shape)


# ============================================================
# SHOW SCHEMAS
# ============================================================

print("\nStructured columns:")
print(structured.columns.tolist())

print("\nText columns:")
print(text.columns.tolist())

print("\nLabel columns:")
print(labels.columns.tolist())

print("\nMetadata columns:")
print(meta.columns.tolist())


# ============================================================
# ALIGNMENT
# ============================================================

n = len(labels)

if len(structured) != n:
    raise ValueError(
        f"Structured row mismatch: "
        f"{len(structured)} != {n}"
    )

if len(text) != n:
    raise ValueError(
        f"Text row mismatch: "
        f"{len(text)} != {n}"
    )

if len(meta) != n:
    raise ValueError(
        f"Metadata row mismatch: "
        f"{len(meta)} != {n}"
    )


print("\nAligned rows:", n)


# ============================================================
# IMPORTANT:
# DO NOT CONCAT WHOLE FILES
#
# Instead, explicitly select the required columns.
# This prevents duplicate-column-name collisions.
# ============================================================

label_series = unique_column(
    labels,
    "techniques",
    "labels_train.csv",
)

text_series = unique_column(
    text,
    text.columns[0],
    "behavior_text_train.csv",
)


# Detect actual behavior-text column
text_column_name = text.columns[0]

print("\nBehavior text column:", text_column_name)


# ============================================================
# BUILD ANALYSIS DATAFRAME
# ============================================================

df = structured.copy().reset_index(drop=True)

df["techniques"] = (
    label_series
    .reset_index(drop=True)
    .fillna("")
    .astype(str)
    .str.strip()
)

df["behavior_text_normalized"] = (
    text_series
    .reset_index(drop=True)
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# Add metadata safely
for col in meta.columns:

    # Never overwrite techniques.
    if col == "techniques":
        continue

    if col in df.columns:
        continue

    df[col] = meta[col].reset_index(drop=True)


print(
    "\nFinal analysis dataframe:",
    df.shape
)


# ============================================================
# LABEL SANITY CHECK
# ============================================================

header("LABEL SANITY CHECK")

print(
    "Unique label combinations:",
    df["techniques"].nunique()
)

print("\nTop labels:")

print(
    df["techniques"]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# T1070.004
# ============================================================

header("T1070.004 + POWERSHELL INSPECTION")

t1070_mask = (
    df["techniques"]
    .str.split(",")
    .apply(
        lambda values:
        "T1070.004" in values
    )
)

t1070 = df[t1070_mask].copy()

print(
    "T1070.004 events:",
    len(t1070)
)


powershell_mask = contains_any(
    t1070["behavior_text_normalized"],
    r"powershell"
)

ps = t1070[powershell_mask].copy()

print(
    "T1070.004 + PowerShell:",
    len(ps)
)


# ============================================================
# POWERHELL BEHAVIOR FLAGS
# ============================================================

ps["has_clear_file_deletion"] = contains_any(
    ps["behavior_text_normalized"],
    r"remove-item|delete|deleted|erase|"
    r"\bdel\b|\brd\b|\brmdir\b"
)

ps["has_clear_log_clearing"] = contains_any(
    ps["behavior_text_normalized"],
    r"wevtutil.*cl|clear-eventlog|"
    r"clear.*log|event.*log.*clear"
)

ps["has_encoded"] = contains_any(
    ps["behavior_text_normalized"],
    r"encodedcommand|encoded.command|"
    r"frombase64string"
)

ps["has_download"] = contains_any(
    ps["behavior_text_normalized"],
    r"invoke-webrequest|invoke-restmethod|"
    r"download|bitsadmin|certutil"
)

ps["has_suspicious_parent"] = contains_any(
    ps["behavior_text_normalized"],
    r"winword|excel|outlook|"
    r"wscript|cscript|mshta|rundll32"
)


# ============================================================
# PRINT GROUP
# ============================================================

def print_group(name, subset):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print("Count:", len(subset))

    if len(subset) == 0:
        return

    columns = [
        "techniques",
        "behavior_text_normalized",
    ]

    columns = [
        c for c in columns
        if c in subset.columns
    ]

    print(
        subset[columns]
        .head(30)
        .to_string(index=False)
    )


print_group(
    "Clear file deletion indicators",
    ps[ps["has_clear_file_deletion"]],
)

print_group(
    "Windows event-log clearing indicators",
    ps[ps["has_clear_log_clearing"]],
)

print_group(
    "Encoded PowerShell",
    ps[ps["has_encoded"]],
)

print_group(
    "Download behavior",
    ps[ps["has_download"]],
)

print_group(
    "Suspicious PowerShell parent",
    ps[ps["has_suspicious_parent"]],
)


# ============================================================
# ADMINISTRATIVE POWERSHELL
# ============================================================

header(
    "ADMINISTRATIVE POWERSHELL + T1070.004"
)

admin_pattern = (
    r"get-service|"
    r"get-process|"
    r"get-computerinfo|"
    r"get-wmiobject|"
    r"get-ciminstance|"
    r"get-childitem|"
    r"test-connection|"
    r"get-netadapter|"
    r"get-netipconfiguration|"
    r"get-hotfix|"
    r"get-eventlog|"
    r"get-localuser|"
    r"get-localgroup"
)

admin_mask = contains_any(
    ps["behavior_text_normalized"],
    admin_pattern,
)

admin_ps = ps[admin_mask].copy()

print(
    "Administrative-looking PowerShell:",
    len(admin_ps)
)

if len(admin_ps):

    print(
        admin_ps[
            [
                "techniques",
                "behavior_text_normalized",
            ]
        ]
        .head(100)
        .to_string(index=False)
    )


# ============================================================
# T1485
# ============================================================

header("T1485 INSPECTION")

t1485_mask = (
    df["techniques"]
    .str.split(",")
    .apply(
        lambda values:
        "T1485" in values
    )
)

t1485 = df[t1485_mask].copy()

print(
    "T1485 events:",
    len(t1485)
)

if len(t1485):

    print(
        t1485[
            [
                "techniques",
                "behavior_text_normalized",
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# DESTRUCTIVE BEHAVIOR
# ============================================================

header("DESTRUCTIVE BEHAVIOR SEARCH")

destructive_pattern = (
    r"mass file|"
    r"mass modification|"
    r"file destruction|"
    r"destroy files|"
    r"delete files|"
    r"deleted files|"
    r"remove files|"
    r"encrypt files|"
    r"ransom|"
    r"ransomware|"
    r"vssadmin.*delete|"
    r"shadow.*delete|"
    r"cipher.*/w"
)

destructive_mask = contains_any(
    df["behavior_text_normalized"],
    destructive_pattern,
)

destructive = df[destructive_mask].copy()

print(
    "Potential destructive events:",
    len(destructive)
)

if len(destructive):

    print(
        destructive[
            [
                "techniques",
                "behavior_text_normalized",
            ]
        ]
        .head(100)
        .to_string(index=False)
    )


# ============================================================
# T1485 + DESTRUCTIVE
# ============================================================

t1485_destructive = t1485[
    contains_any(
        t1485["behavior_text_normalized"],
        destructive_pattern,
    )
]

print(
    "\nT1485 + destructive behavior:",
    len(t1485_destructive)
)


# ============================================================
# CONFLICTING BEHAVIOR SIGNATURES
# ============================================================

header("CONFLICTING BEHAVIOR SIGNATURES")

signature_counts = (
    df
    .groupby(
        "behavior_text_normalized"
    )["techniques"]
    .nunique()
)

conflicting = signature_counts[
    signature_counts > 1
]

print(
    "Unique behavior signatures:",
    len(signature_counts)
)

print(
    "Conflicting signatures:",
    len(conflicting)
)


conflict_rows = []

for signature in conflicting.index:

    subset = df[
        df["behavior_text_normalized"]
        == signature
    ]

    label_counts = (
        subset["techniques"]
        .value_counts()
        .to_dict()
    )

    conflict_rows.append(
        {
            "signature": signature,
            "event_count": len(subset),
            "label_combinations": str(
                label_counts
            ),
        }
    )


conflict_df = pd.DataFrame(
    conflict_rows
)


if len(conflict_df):

    print(
        conflict_df
        .head(20)
        .to_string(index=False)
    )

    conflict_df.to_csv(
        OUT / "conflicting_signatures.csv",
        index=False,
    )


# ============================================================
# SAVE INSPECTION FILES
# ============================================================

ps.to_csv(
    OUT / "t1070_004_powershell_events.csv",
    index=False,
)

admin_ps.to_csv(
    OUT / "administrative_powershell_events.csv",
    index=False,
)

t1485.to_csv(
    OUT / "t1485_events.csv",
    index=False,
)

destructive.to_csv(
    OUT / "destructive_behavior_events.csv",
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

header("V4 DATA QUALITY SUMMARY")

print(
    "Total training events             :",
    len(df),
)

print(
    "T1070.004 events                  :",
    len(t1070),
)

print(
    "T1070.004 + PowerShell            :",
    len(ps),
)

print(
    "Administrative-looking PowerShell :",
    len(admin_ps),
)

print(
    "T1485 events                      :",
    len(t1485),
)

print(
    "T1485 + destructive behavior      :",
    len(t1485_destructive),
)

print(
    "Potential destructive events      :",
    len(destructive),
)

print(
    "Conflicting signatures            :",
    len(conflicting),
)

print()
print("Inspection files saved to:")
print(OUT)

print()
print("IMPORTANT:")
print("This script is READ-ONLY.")
print("No labels or training data were modified.")
print()
print(
    "Use this output to determine the v4.1 "
    "data correction policy."
)