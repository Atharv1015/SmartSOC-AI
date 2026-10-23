import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3f_t1059_003_training_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

STRUCTURED_FILE = DATA_DIR / "structured_train.csv"
BEHAVIOR_FILE = DATA_DIR / "behavior_text_train.csv"
LABEL_FILE = DATA_DIR / "labels_train.csv"

TARGET = "T1059.003"


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def has_target(value):
    if pd.isna(value):
        return False

    labels = [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]

    return TARGET in labels


section("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3F")
print("T1059.003 TRAINING DATA AUDIT")
print()
print("READ-ONLY DIAGNOSTIC")


# -------------------------------------------------------------------------
# LOAD
# -------------------------------------------------------------------------

section("Loading training data")

structured = pd.read_csv(
    STRUCTURED_FILE
)

behavior = pd.read_csv(
    BEHAVIOR_FILE
)

labels = pd.read_csv(
    LABEL_FILE
)

print(
    f"Structured rows : {len(structured)}"
)

print(
    f"Behavior rows   : {len(behavior)}"
)

print(
    f"Label rows      : {len(labels)}"
)


if not (
    len(structured)
    == len(behavior)
    == len(labels)
):

    raise RuntimeError(
        "Training datasets are not aligned."
    )


# -------------------------------------------------------------------------
# TARGET
# -------------------------------------------------------------------------

section(
    f"Identifying {TARGET} training examples"
)

mask = labels[
    "techniques"
].apply(
    has_target
)

target_structured = structured.loc[
    mask
].copy()

target_behavior = behavior.loc[
    mask
].copy()

target_labels = labels.loc[
    mask
].copy()

print(
    f"{TARGET} training rows: "
    f"{len(target_structured)}"
)


# -------------------------------------------------------------------------
# PROCESS
# -------------------------------------------------------------------------

section("Process distribution")

print(
    target_structured[
        "process_name"
    ]
    .fillna("<missing>")
    .astype(str)
    .str.lower()
    .value_counts()
    .head(30)
    .to_string()
)


# -------------------------------------------------------------------------
# PARENT
# -------------------------------------------------------------------------

section("Parent-process distribution")

print(
    target_structured[
        "parent_process_name"
    ]
    .fillna("<missing>")
    .astype(str)
    .str.lower()
    .value_counts()
    .head(30)
    .to_string()
)


# -------------------------------------------------------------------------
# USES CMD
# -------------------------------------------------------------------------

section("uses_cmd distribution")

print(
    target_structured[
        "uses_cmd"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
    .to_string()
)


# -------------------------------------------------------------------------
# OTHER BEHAVIOR FEATURES
# -------------------------------------------------------------------------

section("Behavior feature distribution")

behavior_features = [
    "uses_powershell",
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

rows = []

for feature in behavior_features:

    if feature not in target_structured.columns:
        continue

    counts = (
        target_structured[
            feature
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    ones = int(
        counts.get(1, 0)
    )

    zeros = int(
        counts.get(0, 0)
    )

    rows.append(
        {
            "feature": feature,
            "ones": ones,
            "zeros": zeros,
            "total": len(target_structured),
            "positive_rate":
                ones / len(target_structured),
        }
    )

feature_df = pd.DataFrame(
    rows
)

print(
    feature_df[
        [
            "feature",
            "ones",
            "zeros",
            "positive_rate",
        ]
    ]
    .sort_values(
        "positive_rate",
        ascending=False
    )
    .to_string(index=False)
)


# -------------------------------------------------------------------------
# LABEL COMBINATIONS
# -------------------------------------------------------------------------

section(
    f"{TARGET} label combinations"
)

print(
    target_labels[
        "techniques"
    ]
    .value_counts()
    .head(30)
    .to_string()
)


# -------------------------------------------------------------------------
# COMMAND LENGTH
# -------------------------------------------------------------------------

section("Command length statistics")

print(
    target_structured[
        "command_length"
    ]
    .describe()
    .to_string()
)


# -------------------------------------------------------------------------
# EMPTY COMMANDS
# -------------------------------------------------------------------------

empty_command = (
    target_structured[
        "has_command_line"
    ]
    == 0
)

print()
print(
    "Rows without command line:",
    int(empty_command.sum())
)

print(
    "Rows with command line:",
    int((~empty_command).sum())
)


# -------------------------------------------------------------------------
# BENIGN-LIKE COMMANDS
# -------------------------------------------------------------------------

section(
    "Searching for benign-like command examples"
)

benign_terms = [
    "ipconfig",
    "whoami",
    "tasklist",
    "netstat",
    "net user",
    "net accounts",
    "hostname",
    "systeminfo",
    "ver",
    "set ",
    "echo ",
    "dir ",
]

text_series = (
    target_behavior[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
    .str.lower()
)

benign_matches = []

for term in benign_terms:

    m = text_series.str.contains(
        term,
        regex=False,
        na=False
    )

    count = int(m.sum())

    benign_matches.append(
        {
            "term": term,
            "count": count,
        }
    )

    print(
        f"{term:20s} : {count}"
    )


# -------------------------------------------------------------------------
# SHOW EXAMPLES
# -------------------------------------------------------------------------

section(
    "Representative T1059.003 training examples"
)

display_columns = [
    "process_name",
    "parent_process_name",
    "uses_cmd",
    "uses_powershell",
    "uses_net",
    "uses_whoami",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "command_length",
]

display_columns = [
    c
    for c in display_columns
    if c in target_structured.columns
]

sample_indices = (
    target_structured
    .sample(
        min(50, len(target_structured)),
        random_state=42
    )
    .index
)

sample_table = target_structured.loc[
    sample_indices,
    display_columns
].copy()

sample_table[
    "behavior_text"
] = target_behavior.loc[
    sample_indices,
    "behavior_text_normalized"
].fillna("").astype(str).values

sample_table[
    "label"
] = target_labels.loc[
    sample_indices,
    "techniques"
].values

print(
    sample_table.to_string(
        index=False
    )
)


# -------------------------------------------------------------------------
# SAVE FEATURE AUDIT
# -------------------------------------------------------------------------

feature_file = (
    REPORT_DIR
    / "t1059_003_feature_distribution.csv"
)

feature_df.to_csv(
    feature_file,
    index=False
)


# -------------------------------------------------------------------------
# SAVE BENIGN-LIKE SEARCH
# -------------------------------------------------------------------------

benign_file = (
    REPORT_DIR
    / "t1059_003_benign_like_terms.csv"
)

pd.DataFrame(
    benign_matches
).to_csv(
    benign_file,
    index=False
)


# -------------------------------------------------------------------------
# SAVE ALL TARGET EXAMPLES
# -------------------------------------------------------------------------

all_examples_file = (
    REPORT_DIR
    / "t1059_003_training_examples.csv"
)

full_examples = target_structured.copy()

full_examples[
    "behavior_text_normalized"
] = target_behavior[
    "behavior_text_normalized"
].fillna("").astype(str).values

full_examples[
    "techniques"
] = target_labels[
    "techniques"
].values

full_examples.to_csv(
    all_examples_file,
    index=False
)


# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------

summary = {
    "phase": "Phase 2",
    "gate": "Gate 2.3F",
    "status": "DIAGNOSTIC_COMPLETE",
    "target_technique": TARGET,
    "target_training_rows":
        int(len(target_structured)),
    "uses_cmd_ones":
        int(
            (
                target_structured["uses_cmd"]
                == 1
            ).sum()
        ),
    "uses_cmd_zeros":
        int(
            (
                target_structured["uses_cmd"]
                == 0
            ).sum()
        ),
    "command_line_present":
        int(
            (
                target_structured[
                    "has_command_line"
                ]
                == 1
            ).sum()
        ),
    "command_line_missing":
        int(
            (
                target_structured[
                    "has_command_line"
                ]
                == 0
            ).sum()
        ),
    "benign_like_terms":
        benign_matches,
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
    "files": {
        "feature_distribution":
            str(feature_file),
        "benign_like_terms":
            str(benign_file),
        "training_examples":
            str(all_examples_file),
    },
}


report_file = (
    REPORT_DIR
    / "phase2_gate2_3f_t1059_003_training_audit_report.json"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


section("DIAGNOSTIC COMPLETE")

print(
    f"Target training rows : "
    f"{len(target_structured)}"
)

print(
    f"uses_cmd=1           : "
    f"{summary['uses_cmd_ones']}"
)

print(
    f"uses_cmd=0           : "
    f"{summary['uses_cmd_zeros']}"
)

print()
print(
    f"Report: {report_file}"
)

print(
    "No model artifacts modified."
)

print(
    "No thresholds modified."
)

print(
    "No dataset modified."
)

print(
    "No commands executed."
)