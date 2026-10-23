import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"

REPORT_DIR = (
    ROOT
    / "reports"
    / "phase2_gate2_3g_benign_separation"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

STRUCTURED_TRAIN = DATA_DIR / "structured_train.csv"
BEHAVIOR_TRAIN = DATA_DIR / "behavior_text_train.csv"
LABELS_TRAIN = DATA_DIR / "labels_train.csv"

SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "multilabel_encoder.joblib"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"

TARGET = "T1059.003"


def has_target(value):
    if pd.isna(value):
        return False

    return TARGET in [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


def show(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


show("SmartSOC Windows-APT Model B v4.1")
print("PHASE 2 - GATE 2.3G")
print("BENIGN / T1059.003 SEPARATION ANALYSIS")
print()
print("READ-ONLY")
print("No retraining")
print("No threshold modification")
print("No dataset modification")


# -------------------------------------------------------------------------
# LOAD
# -------------------------------------------------------------------------

show("Loading production artifacts")

schema = json.loads(
    SCHEMA_FILE.read_text(encoding="utf-8")
)

encoder = joblib.load(ENCODER_FILE)
scaler = joblib.load(SCALER_FILE)
tfidf = joblib.load(TFIDF_FILE)
model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(LABEL_ENCODER_FILE)

threshold_data = json.loads(
    THRESHOLD_FILE.read_text(
        encoding="utf-8"
    )
)

thresholds = threshold_data["thresholds"]

categorical_columns = schema["categorical_columns"]
numerical_columns = schema["numerical_columns"]

print(
    f"Categorical columns : {len(categorical_columns)}"
)

print(
    f"Numerical columns   : {len(numerical_columns)}"
)

print(
    f"MITRE classes       : {len(label_encoder.classes_)}"
)


# -------------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------------

show("Loading training data")

structured = pd.read_csv(
    STRUCTURED_TRAIN
)

behavior = pd.read_csv(
    BEHAVIOR_TRAIN
)

labels = pd.read_csv(
    LABELS_TRAIN
)

print(
    f"Training rows: {len(structured)}"
)


if not (
    len(structured)
    == len(behavior)
    == len(labels)
):

    raise RuntimeError(
        "Training files are not aligned."
    )


# -------------------------------------------------------------------------
# SELECT T1059.003
# -------------------------------------------------------------------------

show(
    f"Selecting {TARGET} training population"
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
# TARGET MODEL PROBABILITY
# -------------------------------------------------------------------------

show(
    "Generating model probabilities for T1059.003 training population"
)

X_cat = target_structured[
    categorical_columns
].copy()

X_num = target_structured[
    numerical_columns
].copy()

X_cat = X_cat.fillna("")
X_num = X_num.apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0)

cat_matrix = encoder.transform(
    X_cat
)

num_matrix = scaler.transform(
    X_num
)

text = (
    target_behavior[
        "behavior_text_normalized"
    ]
    .fillna("")
    .astype(str)
)

text_matrix = tfidf.transform(text)

from scipy.sparse import hstack

X_final = hstack(
    [
        cat_matrix,
        num_matrix,
        text_matrix,
    ],
    format="csr",
)

probabilities = model.predict_proba(
    X_final
)

classes = list(
    label_encoder.classes_
)

target_index = classes.index(
    TARGET
)

target_probabilities = (
    probabilities[:, target_index]
)

target_threshold = float(
    thresholds.get(
        TARGET,
        0.5
    )
)

print(
    f"Threshold for {TARGET}: "
    f"{target_threshold}"
)


# -------------------------------------------------------------------------
# FEATURE PROFILE
# -------------------------------------------------------------------------

show(
    f"{TARGET} feature profile"
)

feature_list = [
    "uses_cmd",
    "uses_powershell",
    "uses_net",
    "uses_whoami",
    "uses_ipconfig",
    "uses_netstat",
    "uses_tasklist",
    "uses_sc",
    "uses_wmic",
    "uses_rundll32",
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "administrative_powershell_context",
    "strong_behavior_signal",
]

profile_rows = []

for feature in feature_list:

    if feature not in target_structured.columns:
        continue

    values = pd.to_numeric(
        target_structured[feature],
        errors="coerce"
    ).fillna(0)

    profile_rows.append(
        {
            "feature": feature,
            "positive_count": int(
                (values == 1).sum()
            ),
            "total": len(values),
            "positive_rate":
                float(values.mean()),
        }
    )

profile = pd.DataFrame(
    profile_rows
)

print(
    profile.sort_values(
        "positive_rate",
        ascending=False
    ).to_string(
        index=False
    )
)


# -------------------------------------------------------------------------
# COMMAND-ONLY VS BEHAVIORAL
# -------------------------------------------------------------------------

show(
    "Command-shell behavioral separation"
)

behavioral_features = [
    "has_encoded_command",
    "has_download_behavior",
    "has_remote_execution_hint",
    "has_registry_operation",
    "has_file_deletion",
    "has_obfuscation_hint",
    "strong_behavior_signal",
]

behavioral_signal = np.zeros(
    len(target_structured),
    dtype=bool,
)

for feature in behavioral_features:

    if feature not in target_structured.columns:
        continue

    values = pd.to_numeric(
        target_structured[feature],
        errors="coerce"
    ).fillna(0)

    behavioral_signal |= (
        values > 0
    ).to_numpy()


command_only = (
    target_structured["uses_cmd"]
    .fillna(0)
    .eq(1)
    .to_numpy()
    & ~behavioral_signal
)

command_behavioral = (
    target_structured["uses_cmd"]
    .fillna(0)
    .eq(1)
    .to_numpy()
    & behavioral_signal
)

no_cmd = (
    ~target_structured["uses_cmd"]
    .fillna(0)
    .eq(1)
    .to_numpy()
)

groups = {
    "cmd_only_no_strong_behavior":
        command_only,
    "cmd_with_behavior_signal":
        command_behavioral,
    "no_cmd":
        no_cmd,
}

group_rows = []

for name, group_mask in groups.items():

    probs = target_probabilities[
        group_mask
    ]

    if len(probs) == 0:
        continue

    group_rows.append(
        {
            "group": name,
            "rows": int(len(probs)),
            "mean_probability":
                float(probs.mean()),
            "median_probability":
                float(np.median(probs)),
            "min_probability":
                float(probs.min()),
            "max_probability":
                float(probs.max()),
            "above_threshold":
                int(
                    (
                        probs
                        >= target_threshold
                    ).sum()
                ),
            "above_threshold_rate":
                float(
                    (
                        probs
                        >= target_threshold
                    ).mean()
                ),
        }
    )

group_df = pd.DataFrame(
    group_rows
)

print(
    group_df.to_string(
        index=False
    )
)


# -------------------------------------------------------------------------
# LOW-BEHAVIOR / HIGH-PROBABILITY CASES
# -------------------------------------------------------------------------

show(
    "High-confidence T1059.003 examples without strong behavior"
)

candidate_mask = (
    command_only
    & (
        target_probabilities
        >= target_threshold
    )
)

candidate_indices = np.where(
    candidate_mask
)[0]

candidate_rows = []

for idx in candidate_indices[:100]:

    row = target_structured.iloc[idx]

    candidate_rows.append(
        {
            "index": int(
                target_structured.index[idx]
            ),
            "probability":
                float(
                    target_probabilities[idx]
                ),
            "process_name":
                row.get(
                    "process_name",
                    ""
                ),
            "parent_process_name":
                row.get(
                    "parent_process_name",
                    ""
                ),
            "uses_cmd":
                row.get(
                    "uses_cmd",
                    0
                ),
            "uses_ipconfig":
                row.get(
                    "uses_ipconfig",
                    0
                ),
            "uses_netstat":
                row.get(
                    "uses_netstat",
                    0
                ),
            "uses_tasklist":
                row.get(
                    "uses_tasklist",
                    0
                ),
            "uses_whoami":
                row.get(
                    "uses_whoami",
                    0
                ),
            "command_length":
                row.get(
                    "command_length",
                    0
                ),
            "behavior_text":
                target_behavior.iloc[
                    idx
                ].get(
                    "behavior_text_normalized",
                    ""
                ),
            "techniques":
                target_labels.iloc[
                    idx
                ]["techniques"],
        }
    )

candidate_df = pd.DataFrame(
    candidate_rows
)

if len(candidate_df):

    print(
        candidate_df.head(30).to_string(
            index=False
        )
    )

else:

    print(
        "No command-only high-confidence "
        "T1059.003 examples found."
    )


# -------------------------------------------------------------------------
# SAVE
# -------------------------------------------------------------------------

profile_file = (
    REPORT_DIR
    / "t1059_003_feature_profile.csv"
)

group_file = (
    REPORT_DIR
    / "t1059_003_behavioral_separation.csv"
)

candidate_file = (
    REPORT_DIR
    / "t1059_003_command_only_high_confidence.csv"
)

profile.to_csv(
    profile_file,
    index=False
)

group_df.to_csv(
    group_file,
    index=False
)

candidate_df.to_csv(
    candidate_file,
    index=False
)


# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------

summary = {
    "phase": "Phase 2",
    "gate": "Gate 2.3G",
    "status": "DIAGNOSTIC_COMPLETE",
    "target": TARGET,
    "target_training_rows":
        int(len(target_structured)),
    "threshold":
        target_threshold,
    "groups": group_rows,
    "high_confidence_command_only_rows":
        int(len(candidate_df)),
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
        "feature_profile":
            str(profile_file),
        "behavioral_separation":
            str(group_file),
        "command_only_high_confidence":
            str(candidate_file),
    },
}

report_file = (
    REPORT_DIR
    / "phase2_gate2_3g_report.json"
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


show("DIAGNOSTIC COMPLETE")

print(
    "No production artifacts modified."
)

print(
    "No model retraining performed."
)

print(
    "No thresholds modified."
)

print(
    "No dataset modified."
)

print()
print(
    f"Report: {report_file}"
)