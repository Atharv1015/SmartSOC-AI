import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# SmartSOC Windows-APT v4.1
# PHASE 1 FINAL GATE
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "windows_apt_v4_1"

# IMPORTANT:
# Production artifacts are stored inside the production directory.
MODEL = ROOT / "models" / "windows_apt_v4_1" / "production"

REPORT_DIR = ROOT / "reports" / "phase1_final"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "phase1_final_gate_report.json"


# =============================================================================
# REQUIRED DATASET FILES
# =============================================================================

DATA_FILES = {
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
# REQUIRED MODEL ARTIFACTS
# =============================================================================

MODEL_FILES = {
    "model": MODEL / "model.joblib",
    "encoder": MODEL / "structured_encoder.joblib",
    "scaler": MODEL / "numerical_scaler.joblib",
    "tfidf": MODEL / "tfidf_vectorizer.joblib",
    "labels_encoder": MODEL / "multilabel_encoder.joblib",
    "schema": MODEL / "feature_schema.json",
    "thresholds": MODEL / "thresholds.json",
    "manifest": MODEL / "model_manifest.json",
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

def sha256_file(path: Path) -> str:
    """Return SHA256 hash for a file."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def print_status(name: str, passed: bool):
    """Print PASS/FAIL status."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")


def normalize_text(value) -> str:
    """Normalize arbitrary values into stable text."""
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value)

    return value


def parse_dataset_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse SmartSOC dataset timestamps.

    Dataset format:
        Dec 1, 2024 @ 06:24:28.026

    The '@' separator is removed before pandas datetime parsing.
    """

    raw = series.astype(str).str.strip()

    # Convert:
    # Dec 1, 2024 @ 06:24:28.026
    #
    # into:
    # Dec 1, 2024 06:24:28.026
    cleaned = raw.str.replace(
        " @ ",
        " ",
        regex=False
    )

    # Explicit format keeps parsing deterministic.
    parsed = pd.to_datetime(
        cleaned,
        format="%b %d, %Y %H:%M:%S.%f",
        errors="coerce"
    )

    # Fallback parser for any rows that do not contain milliseconds.
    missing = parsed.isna()

    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            cleaned.loc[missing],
            format="%b %d, %Y %H:%M:%S",
            errors="coerce"
        )

    return parsed


def make_full_representation(
    structured: pd.DataFrame,
    behavior: pd.DataFrame
) -> pd.Series:
    """
    Build a fingerprint representing the complete model input:
        62 structured features + behavior text.
    """

    structured_part = structured.copy()

    # Convert every structured value into stable text.
    for col in structured_part.columns:
        structured_part[col] = structured_part[col].map(normalize_text)

    # Find behavior column.
    behavior_columns = list(behavior.columns)

    if "behavior_text_normalized" in behavior_columns:
        behavior_col = "behavior_text_normalized"
    elif len(behavior_columns) == 1:
        behavior_col = behavior_columns[0]
    else:
        candidates = [
            c for c in behavior_columns
            if "behavior" in c.lower() or "text" in c.lower()
        ]

        if candidates:
            behavior_col = candidates[0]
        else:
            behavior_col = behavior_columns[0]

    behavior_text = behavior[behavior_col].map(normalize_text)

    # Build one representation per row.
    structured_strings = structured_part.astype(str).agg(
        "||".join,
        axis=1
    )

    full_representation = (
        structured_strings
        + "||BEHAVIOR_TEXT="
        + behavior_text
    )

    return full_representation


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("SmartSOC Windows-APT v4.1 - PHASE 1 FINAL GATE")
print("=" * 80)


# =============================================================================
# GATE 1 - FILE INTEGRITY
# =============================================================================

print("\n[1/8] FILE INTEGRITY")

file_gate = True

for name, path in DATA_FILES.items():
    exists = path.exists()

    if exists:
        print(f"[OK]   {name}")
    else:
        print(f"[FAIL] {name}")
        file_gate = False


for name, path in MODEL_FILES.items():
    exists = path.exists()

    if exists:
        print(f"[OK]   {name}")
    else:
        print(f"[FAIL] {name}")
        file_gate = False


if not file_gate:
    print("\nPhase 1 cannot close: required files are missing.")

    report = {
        "phase": "Phase 1",
        "phase1_status": "REVIEW_REQUIRED",
        "reason": "Required files missing",
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"\nFinal report:\n{REPORT_PATH}")
    raise SystemExit(1)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading datasets...")

structured_train = pd.read_csv(
    DATA_FILES["structured_train"]
)

structured_test = pd.read_csv(
    DATA_FILES["structured_test"]
)

labels_train = pd.read_csv(
    DATA_FILES["labels_train"]
)

labels_test = pd.read_csv(
    DATA_FILES["labels_test"]
)

metadata_train = pd.read_csv(
    DATA_FILES["metadata_train"]
)

metadata_test = pd.read_csv(
    DATA_FILES["metadata_test"]
)

behavior_train = pd.read_csv(
    DATA_FILES["behavior_train"]
)

behavior_test = pd.read_csv(
    DATA_FILES["behavior_test"]
)


# =============================================================================
# GATE 2 - DATASET ALIGNMENT
# =============================================================================

print("\n[2/8] DATASET ALIGNMENT")

alignment_pass = True

expected_train_rows = len(structured_train)
expected_test_rows = len(structured_test)

train_alignment = (
    len(labels_train) == expected_train_rows
    and len(metadata_train) == expected_train_rows
    and len(behavior_train) == expected_train_rows
)

test_alignment = (
    len(labels_test) == expected_test_rows
    and len(metadata_test) == expected_test_rows
    and len(behavior_test) == expected_test_rows
)

if train_alignment and test_alignment:
    print("[PASS] Train/test alignment")
else:
    print("[FAIL] Train/test alignment")
    alignment_pass = False


# Structured schema.
structured_schema_pass = (
    list(structured_train.columns) == EXPECTED_STRUCTURED_COLUMNS
    and list(structured_test.columns) == EXPECTED_STRUCTURED_COLUMNS
)

if structured_schema_pass:
    print("[PASS] 62 structured features")
else:
    print("[FAIL] 62 structured features")
    alignment_pass = False


# Behavior text column.
behavior_columns = list(behavior_train.columns)

if "behavior_text_normalized" in behavior_columns:
    behavior_column = "behavior_text_normalized"
elif len(behavior_columns) == 1:
    behavior_column = behavior_columns[0]
else:
    candidates = [
        c for c in behavior_columns
        if "behavior" in c.lower() or "text" in c.lower()
    ]

    behavior_column = candidates[0] if candidates else None


behavior_column_pass = behavior_column is not None

if behavior_column_pass:
    print(f"[PASS] Behavior text column: {behavior_column}")
else:
    print("[FAIL] Behavior text column")
    alignment_pass = False


# =============================================================================
# GATE 3 - FULL INPUT DUPLICATION / AMBIGUITY
# =============================================================================

print("\n[3/8] FULL INPUT DUPLICATION / AMBIGUITY")

full_train = make_full_representation(
    structured_train,
    behavior_train
)

full_test = make_full_representation(
    structured_test,
    behavior_test
)

full_all = pd.concat(
    [
        full_train.rename("representation"),
        full_test.rename("representation"),
    ],
    ignore_index=True
)

total_rows = len(full_all)

unique_full_representations = int(
    full_all.nunique()
)

representation_counts = (
    full_all.value_counts()
)

repeated_full_representations = int(
    (representation_counts > 1).sum()
)

# Build combined labels.
train_labels_series = (
    labels_train["techniques"]
    .fillna("")
    .astype(str)
    .str.strip()
)

test_labels_series = (
    labels_test["techniques"]
    .fillna("")
    .astype(str)
    .str.strip()
)

all_labels = pd.concat(
    [
        train_labels_series,
        test_labels_series
    ],
    ignore_index=True
)

full_analysis = pd.DataFrame({
    "representation": full_all,
    "label": all_labels,
    "split": (
        ["train"] * len(train_labels_series)
        + ["test"] * len(test_labels_series)
    )
})

ambiguous_full_representations = 0
ambiguous_rows = 0
cross_split_ambiguous = 0

for representation, group in full_analysis.groupby(
    "representation",
    sort=False
):
    unique_labels = set(group["label"])

    if len(unique_labels) > 1:
        ambiguous_full_representations += 1
        ambiguous_rows += len(group)

        splits = set(group["split"])

        if "train" in splits and "test" in splits:
            cross_split_ambiguous += 1


print(
    f"Total rows                    : {total_rows:,}"
)

print(
    f"Unique full representations  : "
    f"{unique_full_representations:,}"
)

print(
    f"Repeated full representations : "
    f"{repeated_full_representations:,}"
)

print(
    f"Ambiguous full representations: "
    f"{ambiguous_full_representations:,}"
)

print(
    f"Rows affected                : "
    f"{ambiguous_rows:,}"
)

print(
    f"Cross-split ambiguous        : "
    f"{cross_split_ambiguous:,}"
)


# =============================================================================
# GATE 4 - MITRE LABEL AUDIT
# =============================================================================

print("\n[4/8] MITRE LABEL AUDIT")


def parse_techniques(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Standard format is comma-separated.
    parts = [
        p.strip()
        for p in text.split(",")
        if p.strip()
    ]

    return parts


train_techniques = []

for value in labels_train["techniques"]:
    train_techniques.extend(
        parse_techniques(value)
    )


test_techniques = []

for value in labels_test["techniques"]:
    test_techniques.extend(
        parse_techniques(value)
    )


train_counts = pd.Series(
    train_techniques
).value_counts()

test_counts = pd.Series(
    test_techniques
).value_counts()

all_techniques = sorted(
    set(train_counts.index)
    | set(test_counts.index)
)

mitre_classes = len(all_techniques)

mitre_class_pass = (
    mitre_classes == 45
)

if mitre_class_pass:
    print("[PASS] 45 MITRE techniques")
else:
    print(
        f"[FAIL] Expected 45 MITRE techniques, "
        f"found {mitre_classes}"
    )


rare_rows = []

for technique in all_techniques:
    train_support = int(
        train_counts.get(technique, 0)
    )

    test_support = int(
        test_counts.get(technique, 0)
    )

    if train_support < 5:
        rare_rows.append(
            {
                "technique": technique,
                "train_support": train_support,
                "test_support": test_support,
            }
        )


if rare_rows:
    rare_df = pd.DataFrame(
        rare_rows
    )

    print("\nRare techniques (<5 training examples):")
    print(
        rare_df.to_string(index=False)
    )
else:
    print("\nNo techniques have fewer than 5 training examples.")


# =============================================================================
# GATE 5 - MODEL ARTIFACT VALIDATION
# =============================================================================

print("\n[5/8] MODEL ARTIFACT VALIDATION")

model = joblib.load(
    MODEL_FILES["model"]
)

encoder = joblib.load(
    MODEL_FILES["encoder"]
)

scaler = joblib.load(
    MODEL_FILES["scaler"]
)

tfidf = joblib.load(
    MODEL_FILES["tfidf"]
)

labels_encoder = joblib.load(
    MODEL_FILES["labels_encoder"]
)

with MODEL_FILES["schema"].open(
    "r",
    encoding="utf-8"
) as f:
    schema = json.load(f)

with MODEL_FILES["thresholds"].open(
    "r",
    encoding="utf-8"
) as f:
    thresholds = json.load(f)

with MODEL_FILES["manifest"].open(
    "r",
    encoding="utf-8"
) as f:
    manifest = json.load(f)


model_type = type(model).__name__

tfidf_vocabulary = len(
    getattr(
        tfidf,
        "vocabulary_",
        {}
    )
)

if hasattr(
    labels_encoder,
    "classes_"
):
    model_classes = len(
        labels_encoder.classes_
    )
else:
    model_classes = 0


# Numerical columns from schema.
numerical_columns = schema.get(
    "numerical_columns",
    []
)

categorical_columns = schema.get(
    "categorical_columns",
    []
)

expected_final_features = (
    len(numerical_columns)
    + len(categorical_columns)
)

# Prefer explicit schema value if available.
schema_final_features = schema.get(
    "final_feature_count",
    expected_final_features
)

try:
    scaler_features = int(
        getattr(
            scaler,
            "n_features_in_",
            len(numerical_columns)
        )
    )
except Exception:
    scaler_features = len(numerical_columns)


try:
    encoder_features = int(
        len(
            encoder.get_feature_names_out(
                categorical_columns
            )
        )
    )
except Exception:
    encoder_features = 0


# The trained model's estimator input dimension.
model_features = getattr(
    model,
    "n_features_in_",
    None
)

if model_features is None:
    try:
        model_features = model.estimators_[0].n_features_in_
    except Exception:
        model_features = None


print(
    f"Model type       : {model_type}"
)

print(
    f"TF-IDF vocabulary : {tfidf_vocabulary:,}"
)

print(
    f"MITRE classes     : {model_classes}"
)

print(
    f"Numerical columns : {len(numerical_columns)}"
)

print(
    f"Categorical cols  : {len(categorical_columns)}"
)

# v4.1 expected values:
# 58 numerical
# 162 one-hot categorical
# 6699 TF-IDF
# 6919 final model features

# Calculate actual structured feature contribution.
try:
    categorical_encoded_features = encoder_features
except Exception:
    categorical_encoded_features = 0

actual_final_features = (
    categorical_encoded_features
    + scaler_features
    + tfidf_vocabulary
)

print(
    f"Final features    : {actual_final_features}"
)

artifact_dimensions_pass = (
    model_type == "OneVsRestClassifier"
    and tfidf_vocabulary == 6699
    and model_classes == 45
    and scaler_features == 58
    and len(categorical_columns) == 4
    and categorical_encoded_features == 162
    and actual_final_features == 6919
)

if artifact_dimensions_pass:
    print("[PASS] Production artifact dimensions")
else:
    print("[FAIL] Production artifact dimensions")


# =============================================================================
# GATE 6 - THRESHOLD CALIBRATION
# =============================================================================

print("\n[6/8] THRESHOLD CALIBRATION")

threshold_count = 0

if isinstance(thresholds, dict):
    threshold_count = len(thresholds)

elif isinstance(thresholds, list):
    threshold_count = len(thresholds)


thresholds_pass = threshold_count > 0

if thresholds_pass:
    print(
        f"[PASS] Threshold artifact loaded "
        f"({threshold_count} entries)"
    )
else:
    print("[FAIL] Threshold artifact is empty")


# =============================================================================
# GATE 7 - FINAL TEST ISOLATION
# =============================================================================

print("\n[7/8] FINAL TEST ISOLATION")

timestamp_col = "timestamp"

temporal_order_pass = False
temporal_gap = pd.NaT
train_latest = pd.NaT
test_earliest = pd.NaT

if timestamp_col not in metadata_train.columns:
    print(
        f"[FAIL] Missing timestamp column in train metadata: "
        f"{timestamp_col}"
    )

elif timestamp_col not in metadata_test.columns:
    print(
        f"[FAIL] Missing timestamp column in test metadata: "
        f"{timestamp_col}"
    )

else:

    # -------------------------------------------------------------------------
    # CRITICAL FIX
    #
    # Actual dataset timestamps look like:
    #
    # Dec 1, 2024 @ 06:24:28.026
    #
    # The '@' is not part of a standard datetime format.
    # parse_dataset_timestamp() removes it and parses the timestamps.
    # -------------------------------------------------------------------------

    train_ts = parse_dataset_timestamp(
        metadata_train[timestamp_col]
    )

    test_ts = parse_dataset_timestamp(
        metadata_test[timestamp_col]
    )

    train_invalid = int(
        train_ts.isna().sum()
    )

    test_invalid = int(
        test_ts.isna().sum()
    )

    print(
        f"Invalid train timestamps : {train_invalid}"
    )

    print(
        f"Invalid test timestamps  : {test_invalid}"
    )

    if train_invalid > 0 or test_invalid > 0:

        print(
            "[FAIL] Invalid timestamps remain after parsing"
        )

        # Show examples to make debugging easy.
        if train_invalid > 0:
            bad_train = metadata_train.loc[
                train_ts.isna(),
                timestamp_col
            ].head(5)

            print("\nExample invalid train timestamps:")
            print(
                bad_train.to_string(index=False)
            )

        if test_invalid > 0:
            bad_test = metadata_test.loc[
                test_ts.isna(),
                timestamp_col
            ].head(5)

            print("\nExample invalid test timestamps:")
            print(
                bad_test.to_string(index=False)
            )

    else:

        train_latest = train_ts.max()
        test_earliest = test_ts.min()

        temporal_gap = (
            test_earliest
            - train_latest
        )

        print(
            f"Timestamp column : {timestamp_col}"
        )

        print(
            f"Train latest     : {train_latest}"
        )

        print(
            f"Test earliest    : {test_earliest}"
        )

        print(
            f"Temporal gap     : {temporal_gap}"
        )

        if train_latest < test_earliest:

            print(
                "[PASS] Clean temporal separation"
            )

            temporal_order_pass = True

        else:

            print(
                "[FAIL] Temporal overlap detected"
            )

            temporal_order_pass = False


# =============================================================================
# GATE 8 - PHASE 1 ACCEPTANCE
# =============================================================================

print("\n[8/8] PHASE 1 ACCEPTANCE")

gates = {
    "dataset_files": file_gate,
    "row_alignment": alignment_pass,
    "structured_schema": structured_schema_pass,
    "model_artifacts": artifact_dimensions_pass,
    "mitre_classes": mitre_class_pass,
    "tfidf_features": tfidf_vocabulary == 6699,
    "temporal_order": temporal_order_pass,
    "thresholds_present": thresholds_pass,
}

for gate_name, passed in gates.items():
    print_status(
        gate_name,
        passed
    )


# =============================================================================
# OVERALL STATUS
# =============================================================================

phase1_pass = all(
    gates.values()
)


if phase1_pass:

    phase1_status = "ACCEPTED"

    print("\n" + "=" * 80)
    print(
        "PHASE 1 STATUS: ACCEPTED"
    )
    print("=" * 80)

else:

    phase1_status = "REVIEW_REQUIRED"

    print("\n" + "=" * 80)
    print(
        "PHASE 1 STATUS: REVIEW REQUIRED"
    )
    print(
        "Do not proceed until failed gates are resolved."
    )
    print("=" * 80)


# =============================================================================
# ARTIFACT HASHES
# =============================================================================

artifact_sha256 = {}

for name, path in {
    **DATA_FILES,
    **MODEL_FILES,
}.items():

    try:
        artifact_sha256[name] = sha256_file(path)

    except Exception as exc:
        artifact_sha256[name] = (
            f"ERROR: {exc}"
        )


# =============================================================================
# FINAL REPORT
# =============================================================================

report = {
    "phase": "Phase 1",

    "model": {
        "type": model_type,
        "mitre_classes": model_classes,
        "tfidf_features": tfidf_vocabulary,
        "final_features": actual_final_features,
        "numerical_columns": len(numerical_columns),
        "categorical_columns": len(categorical_columns),
        "categorical_encoded_features": categorical_encoded_features,
    },

    "dataset": {
        "train_rows": len(structured_train),
        "test_rows": len(structured_test),
        "total_rows": total_rows,
        "structured_features": len(
            structured_train.columns
        ),
        "unique_full_representations":
            unique_full_representations,
        "repeated_full_representations":
            repeated_full_representations,
        "ambiguous_full_representations":
            ambiguous_full_representations,
        "ambiguous_rows":
            ambiguous_rows,
        "cross_split_ambiguous":
            cross_split_ambiguous,
    },

    "mitre_label_audit": {
        "mitre_classes": mitre_classes,
        "rare_training_techniques": rare_rows,
    },

    "temporal_validation": {
        "timestamp_column": timestamp_col,
        "train_latest": (
            None
            if pd.isna(train_latest)
            else str(train_latest)
        ),
        "test_earliest": (
            None
            if pd.isna(test_earliest)
            else str(test_earliest)
        ),
        "temporal_gap": (
            None
            if pd.isna(temporal_gap)
            else str(temporal_gap)
        ),
        "clean_temporal_order":
            temporal_order_pass,
    },

    "threshold_validation": {
        "threshold_entries":
            threshold_count,
        "thresholds_present":
            thresholds_pass,
    },

    "gates": gates,

    "phase1_status": phase1_status,

    "artifact_sha256": artifact_sha256,
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        default=str
    ),
    encoding="utf-8"
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\nFinal report:")
print(REPORT_PATH)