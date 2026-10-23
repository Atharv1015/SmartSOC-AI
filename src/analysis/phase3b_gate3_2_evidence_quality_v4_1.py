from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SMARTSOC PHASE 3B
# GATE 3B.2 — EVIDENCE QUALITY FILTERING & SOC EVIDENCE EXTRACTION
#
# Purpose:
#   1. Reuse the production model and production feature pipeline.
#   2. Extract coefficient-based feature contributions.
#   3. Separate SOC-relevant evidence from identifier/metadata artifacts.
#   4. Verify that event identifiers do not appear in analyst-facing evidence.
#   5. Produce a read-only evidence-quality report.
#
# IMPORTANT:
#   - No model modification.
#   - No threshold modification.
#   - No dataset modification.
#   - No production artifact modification.
# ============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIR = (
    PROJECT_ROOT
    / "models"
    / "windows_apt_v4_1"
    / "production"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "windows_apt_v4_1"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "phase3b_gate3_2"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PRODUCTION_DIR / "model.joblib"
ENCODER_PATH = PRODUCTION_DIR / "structured_encoder.joblib"
SCALER_PATH = PRODUCTION_DIR / "numerical_scaler.joblib"
TFIDF_PATH = PRODUCTION_DIR / "tfidf_vectorizer.joblib"
MLB_PATH = PRODUCTION_DIR / "multilabel_encoder.joblib"
SCHEMA_PATH = PRODUCTION_DIR / "feature_schema.json"
THRESHOLD_PATH = PRODUCTION_DIR / "thresholds.json"

STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(value):
    return str(value).strip().lower()


def is_identifier_or_metadata(feature_name):
    """
    Features that should NOT be exposed as behavioral SOC evidence.

    These can still participate in model attribution internally.
    They are simply excluded from analyst-facing evidence.
    """
    name = normalize_name(feature_name)

    blocked_exact = {
        "event_id",
    }

    blocked_prefixes = {
        "event_id",
    }

    if name in blocked_exact:
        return True

    for prefix in blocked_prefixes:
        if name.startswith(prefix):
            return True

    return False


def evidence_category(feature_name):
    """
    Categorize a feature according to SOC evidence usefulness.
    """

    name = normalize_name(feature_name)

    if is_identifier_or_metadata(name):
        return "identifier_or_metadata"

    # TF-IDF
    if name.startswith("tfidf:") or name.startswith("tfidf__"):
        return "text_behavior"

    # Process identity / execution context
    process_terms = [
        "process_name",
        "parent_process_name",
        "process_extension",
        "parent_extension",
        "has_process_image",
        "has_parent_image",
        "same_process_parent",
        "suspicious_parent_wazuh",
    ]

    if any(term in name for term in process_terms):
        return "process_evidence"

    # User / privilege / integrity context
    context_terms = [
        "has_user",
        "has_parent_user",
        "is_system_user",
        "is_admin_user",
        "integrity_high",
        "integrity_medium",
        "administrative_powershell_context",
        "system_level",
        "is_sysmon",
        "is_security_channel",
    ]

    if any(term in name for term in context_terms):
        return "contextual_evidence"

    # Command-line behavior
    command_terms = [
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
    ]

    if any(term in name for term in command_terms):
        return "command_behavior"

    # Network evidence
    network_terms = [
        "has_source_ip",
        "has_destination_ip",
        "has_source_port",
        "has_destination_port",
        "has_protocol",
        "is_network_event",
    ]

    if any(term in name for term in network_terms):
        return "network_evidence"

    # File/object/hash evidence
    artifact_terms = [
        "has_hash",
        "has_sha256",
        "has_md5",
        "has_target_filename",
        "has_image_loaded",
        "has_target_object",
        "has_registry_operation",
        "has_file_deletion",
    ]

    if any(term in name for term in artifact_terms):
        return "artifact_evidence"

    # Strong behavior signal
    if "strong_behavior_signal" in name:
        return "behavior_signal"

    # Default
    return "other"


def get_column_list(schema, keys):
    for key in keys:
        value = schema.get(key)
        if isinstance(value, list):
            return value

    return None


# ============================================================================
# HEADER
# ============================================================================

print("=" * 72)
print("SMARTSOC PHASE 3B")
print("GATE 3B.2 — EVIDENCE QUALITY FILTERING & SOC EVIDENCE EXTRACTION")
print("=" * 72)


# ============================================================================
# PRODUCTION ARTIFACT VALIDATION
# ============================================================================

required_artifacts = [
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    TFIDF_PATH,
    MLB_PATH,
    SCHEMA_PATH,
    THRESHOLD_PATH,
]

print("\nPRODUCTION ARTIFACTS")

for path in required_artifacts:
    if path.exists():
        print(f"[PASS] {path.name}")
    else:
        fail(f"Missing production artifact: {path}")


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

print("\nLOADING PRODUCTION ARTIFACTS")

try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    mlb = joblib.load(MLB_PATH)
    schema = load_json(SCHEMA_PATH)
    thresholds = load_json(THRESHOLD_PATH)
except Exception as exc:
    fail(f"Artifact loading failed: {exc}")

print("[PASS] model")
print("[PASS] structured encoder")
print("[PASS] numerical scaler")
print("[PASS] TF-IDF vectorizer")
print("[PASS] multilabel encoder")
print("[PASS] feature schema")
print("[PASS] thresholds")


# ============================================================================
# FEATURE SCHEMA
# ============================================================================

print("\nFEATURE SCHEMA")

categorical_columns = get_column_list(
    schema,
    [
        "categorical_columns",
        "categorical_features",
    ],
)

numerical_columns = get_column_list(
    schema,
    [
        "numerical_columns",
        "numerical_features",
    ],
)

if categorical_columns is None:
    if hasattr(encoder, "feature_names_in_"):
        categorical_columns = list(encoder.feature_names_in_)

if numerical_columns is None:
    if hasattr(scaler, "feature_names_in_"):
        numerical_columns = list(scaler.feature_names_in_)

if categorical_columns is None:
    fail("Unable to determine categorical columns.")

if numerical_columns is None:
    fail("Unable to determine numerical columns.")

mitre_techniques = list(mlb.classes_)

print(f"  Categorical : {len(categorical_columns)}")
print(f"  Numerical   : {len(numerical_columns)}")
print(f"  Techniques  : {len(mitre_techniques)}")


# ============================================================================
# BUILD FEATURE NAMES
# ============================================================================

print("\nBUILDING FEATURE MAP")

try:
    categorical_feature_names = list(
        encoder.get_feature_names_out(categorical_columns)
    )
except TypeError:
    categorical_feature_names = list(
        encoder.get_feature_names_out()
    )

numerical_feature_names = list(numerical_columns)

tfidf_feature_names = [
    f"tfidf:{term}"
    for term in tfidf.get_feature_names_out()
]

feature_names = (
    categorical_feature_names
    + numerical_feature_names
    + tfidf_feature_names
)

expected_feature_count = int(
    schema.get("final_feature_count", 6919)
)

print(f"  Categorical encoded : {len(categorical_feature_names)}")
print(f"  Numerical           : {len(numerical_feature_names)}")
print(f"  TF-IDF              : {len(tfidf_feature_names)}")
print(f"  Final               : {len(feature_names)}")

if len(feature_names) != expected_feature_count:
    fail(
        f"Feature mapping mismatch: "
        f"{len(feature_names)} != {expected_feature_count}"
    )

print("[PASS] Feature mapping")


# ============================================================================
# LOAD TEST INPUT
# ============================================================================

print("\nINPUT DATA")

if not STRUCTURED_TEST.exists():
    fail(f"Missing structured test file: {STRUCTURED_TEST}")

if not BEHAVIOR_TEST.exists():
    fail(f"Missing behavior test file: {BEHAVIOR_TEST}")

structured_df = pd.read_csv(STRUCTURED_TEST)
behavior_df = pd.read_csv(BEHAVIOR_TEST)

if len(structured_df) != len(behavior_df):
    fail(
        f"Row mismatch: structured={len(structured_df)}, "
        f"behavior={len(behavior_df)}"
    )

behavior_column = "behavior_text_normalized"

if behavior_column not in behavior_df.columns:
    fail(f"Missing behavior column: {behavior_column}")

print(f"  Structured rows: {len(structured_df)}")
print(f"  Behavior rows  : {len(behavior_df)}")
print(f"  Behavior column: {behavior_column}")


# ============================================================================
# DETERMINISTIC SAMPLE
# ============================================================================

sample_index = 0

structured_row = structured_df.iloc[[sample_index]].copy()
behavior_row = behavior_df.iloc[[sample_index]].copy()

behavior_text = str(
    behavior_row.iloc[0][behavior_column]
)

print("\nSAMPLE EVENT")
print(f"  Test row : {sample_index}")

if "event_id" in structured_row.columns:
    print(f"  Event ID : {structured_row.iloc[0]['event_id']}")

print(f"  Behavior : {behavior_text[:300]}")


# ============================================================================
# TRANSFORMATION
# ============================================================================

print("\nTRANSFORMATION")

try:
    cat_input = structured_row[categorical_columns]

    # Avoid StandardScaler feature-name warning.
    num_input = structured_row[numerical_columns].to_numpy(
        dtype=np.float64
    )

    X_cat = encoder.transform(cat_input)
    X_num = scaler.transform(num_input)
    X_text = tfidf.transform([behavior_text])

    X_final = np.hstack(
        [
            X_cat.toarray()
            if hasattr(X_cat, "toarray")
            else np.asarray(X_cat),
            np.asarray(X_num),
            X_text.toarray()
            if hasattr(X_text, "toarray")
            else np.asarray(X_text),
        ]
    )

except Exception as exc:
    fail(f"Transformation failed: {exc}")

print(f"  Categorical: {X_cat.shape}")
print(f"  Numerical  : {X_num.shape}")
print(f"  TF-IDF     : {X_text.shape}")
print(f"  Final      : {X_final.shape}")

if X_final.shape[1] != expected_feature_count:
    fail(
        f"Final feature dimension mismatch: "
        f"{X_final.shape[1]} != {expected_feature_count}"
    )

print("[PASS] Production feature vector")


# ============================================================================
# PREDICTION
# ============================================================================

print("\nPREDICTION")

try:
    probabilities = model.predict_proba(X_final)
except Exception as exc:
    fail(f"Prediction failed: {exc}")

probabilities = np.asarray(probabilities)

if probabilities.ndim != 2:
    fail(f"Unexpected probability shape: {probabilities.shape}")

if probabilities.shape[1] != len(mitre_techniques):
    fail(
        f"Probability/technique mismatch: "
        f"{probabilities.shape[1]} != {len(mitre_techniques)}"
    )

if not np.isfinite(probabilities).all():
    fail("Non-finite probabilities detected.")

print(f"  Shape : {probabilities.shape}")
print("  Finite: True")


# ============================================================================
# COEFFICIENT EXTRACTION
# ============================================================================

print("\nCOEFFICIENT MATRIX")

if not hasattr(model, "estimators_"):
    fail("Production model does not expose estimators_.")

estimators = model.estimators_

if len(estimators) != len(mitre_techniques):
    fail(
        f"Estimator/technique mismatch: "
        f"{len(estimators)} != {len(mitre_techniques)}"
    )

coefficient_matrix = np.vstack(
    [
        np.asarray(est.coef_).reshape(-1)
        for est in estimators
    ]
)

print(f"  Shape: {coefficient_matrix.shape}")

if coefficient_matrix.shape != (
    len(mitre_techniques),
    expected_feature_count,
):
    fail(
        "Coefficient alignment mismatch: "
        f"{coefficient_matrix.shape}"
    )

print("[PASS] Coefficient alignment")


# ============================================================================
# CONTRIBUTIONS
# ============================================================================

contribution_matrix = X_final * coefficient_matrix


# ============================================================================
# TECHNIQUE RANKING
# ============================================================================

top_indices = np.argsort(
    probabilities[0]
)[::-1]

print("\nTOP PREDICTIONS")

top_predictions = []

for rank, technique_index in enumerate(top_indices[:10], start=1):

    technique = str(
        mitre_techniques[technique_index]
    )

    probability = float(
        probabilities[0, technique_index]
    )

    threshold_value = thresholds.get(
        technique,
        thresholds.get(
            "default_threshold",
            0.5
        )
    )

    try:
        threshold_value = float(threshold_value)
    except Exception:
        threshold_value = 0.5

    margin = probability - threshold_value

    detected = probability >= threshold_value

    print(
        f"  #{rank}: {technique:<10} "
        f"P={probability:.6f} "
        f"T={threshold_value:.6f} "
        f"M={margin:.6f} "
        f"Detected={detected}"
    )

    top_predictions.append(
        {
            "rank": rank,
            "technique": technique,
            "probability": probability,
            "threshold": threshold_value,
            "margin": margin,
            "detected": bool(detected),
        }
    )


# ============================================================================
# EVIDENCE QUALITY ANALYSIS
# ============================================================================

print("\nEVIDENCE QUALITY ANALYSIS")

evidence_records = []

for technique_index in top_indices[:10]:

    technique = str(
        mitre_techniques[technique_index]
    )

    probability = float(
        probabilities[0, technique_index]
    )

    threshold_value = thresholds.get(
        technique,
        thresholds.get(
            "default_threshold",
            0.5
        )
    )

    try:
        threshold_value = float(threshold_value)
    except Exception:
        threshold_value = 0.5

    contribution_vector = contribution_matrix[
        technique_index
    ]

    positive_indices = np.where(
        contribution_vector > 0
    )[0]

    positive_indices = sorted(
        positive_indices,
        key=lambda i: contribution_vector[i],
        reverse=True,
    )

    # Analyze top 30 raw positive contributions.
    raw_top = positive_indices[:30]

    raw_evidence_count = 0
    filtered_evidence = []
    filtered_out = []

    for feature_index in raw_top:

        feature_name = str(
            feature_names[feature_index]
        )

        contribution = float(
            contribution_vector[feature_index]
        )

        category = evidence_category(
            feature_name
        )

        value = float(
            X_final[0, feature_index]
        )

        record = {
            "feature": feature_name,
            "value": value,
            "contribution": contribution,
            "category": category,
        }

        if category == "identifier_or_metadata":
            filtered_out.append(record)
        else:
            raw_evidence_count += 1
            filtered_evidence.append(record)

    # Sort retained evidence by contribution.
    filtered_evidence = sorted(
        filtered_evidence,
        key=lambda x: x["contribution"],
        reverse=True,
    )[:10]

    # Top raw evidence regardless of filtering.
    raw_top_evidence = []

    for feature_index in raw_top[:10]:

        feature_name = str(
            feature_names[feature_index]
        )

        raw_top_evidence.append(
            {
                "feature": feature_name,
                "value": float(
                    X_final[0, feature_index]
                ),
                "contribution": float(
                    contribution_vector[feature_index]
                ),
                "category": evidence_category(
                    feature_name
                ),
            }
        )

    evidence_records.append(
        {
            "technique": technique,
            "probability": probability,
            "threshold": threshold_value,
            "detected": bool(
                probability >= threshold_value
            ),
            "raw_top_positive_evidence": raw_top_evidence,
            "soc_relevant_evidence": filtered_evidence,
            "filtered_identifier_metadata": filtered_out,
        }
    )


# ============================================================================
# PRINT FILTERED EVIDENCE
# ============================================================================

for record in evidence_records:

    print("\n" + "-" * 72)

    print(
        f"TECHNIQUE: {record['technique']} "
        f"P={record['probability']:.6f}"
    )

    print(
        "SOC-RELEVANT POSITIVE EVIDENCE"
    )

    if not record["soc_relevant_evidence"]:
        print("  [NONE]")

    else:

        for evidence in record[
            "soc_relevant_evidence"
        ]:

            print(
                f"  + {evidence['feature']}"
                f" | value={evidence['value']:.6f}"
                f" | contribution={evidence['contribution']:.6f}"
                f" | category={evidence['category']}"
            )

    filtered = record[
        "filtered_identifier_metadata"
    ]

    if filtered:

        print(
            "FILTERED IDENTIFIER/METADATA FEATURES"
        )

        for evidence in filtered[:5]:

            print(
                f"  [FILTERED] {evidence['feature']}"
                f" | contribution={evidence['contribution']:.6f}"
            )


# ============================================================================
# GLOBAL QUALITY COUNTS
# ============================================================================

raw_identifier_contributions = 0
raw_soc_evidence_contributions = 0

for record in evidence_records:

    raw_identifier_contributions += len(
        record["filtered_identifier_metadata"]
    )

    raw_soc_evidence_contributions += len(
        record["soc_relevant_evidence"]
    )


# ============================================================================
# VERIFY EVENT_ID EXCLUSION
# ============================================================================

print("\nEVIDENCE SAFETY CHECK")

analyst_features = []

for record in evidence_records:

    for evidence in record[
        "soc_relevant_evidence"
    ]:

        analyst_features.append(
            evidence["feature"]
        )

event_id_leakage = [
    feature
    for feature in analyst_features
    if is_identifier_or_metadata(feature)
]

if event_id_leakage:
    print("[FAIL] Identifier leakage detected.")
    print(event_id_leakage)
    fail(
        "Identifier/metadata feature leaked into "
        "SOC-relevant evidence."
    )

print(
    "[PASS] Identifier/metadata features excluded "
    "from SOC-relevant evidence."
)


# ============================================================================
# CATEGORY DISTRIBUTION
# ============================================================================

category_counts = {}

for record in evidence_records:

    for evidence in record[
        "soc_relevant_evidence"
    ]:

        category = evidence["category"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )


print("\nSOC EVIDENCE CATEGORY DISTRIBUTION")

if category_counts:

    for category, count in sorted(
        category_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        print(
            f"  {category:<24} {count}"
        )

else:

    print("  [NONE]")


# ============================================================================
# QUALITY DECISION
# ============================================================================

print("\n" + "=" * 72)
print("PHASE 3B GATE 3B.2 SUMMARY")
print("=" * 72)

print(
    f"Techniques analyzed              : "
    f"{len(evidence_records)}"
)

print(
    f"Raw identifier/metadata items    : "
    f"{raw_identifier_contributions}"
)

print(
    f"SOC-relevant evidence items      : "
    f"{raw_soc_evidence_contributions}"
)

print(
    f"Analyst evidence identifier leak : "
    f"{len(event_id_leakage)}"
)

quality_checks = {
    "feature_mapping_valid": True,
    "prediction_valid": True,
    "coefficient_alignment_valid": True,
    "identifier_metadata_filtered": True,
    "analyst_evidence_identifier_leakage": len(
        event_id_leakage
    ) == 0,
}

all_pass = all(
    quality_checks.values()
)

if all_pass:
    decision = "PASS"
else:
    decision = "REVIEW_REQUIRED"

print("\nQUALITY CHECKS")

for key, value in quality_checks.items():

    status = "PASS" if value else "FAIL"

    print(
        f"  [{status}] {key}"
    )

print(
    f"\nDECISION: {decision}"
)


# ============================================================================
# REPORT
# ============================================================================

report = {
    "phase": "3B",
    "gate": "3B.2",
    "title": "Evidence Quality Filtering & SOC Evidence Extraction",
    "decision": decision,
    "production_artifacts_modified": False,
    "sample_index": sample_index,
    "expected_feature_count": expected_feature_count,
    "actual_feature_count": int(X_final.shape[1]),
    "mitre_technique_count": len(mitre_techniques),
    "top_predictions": top_predictions,
    "evidence_quality": {
        "raw_identifier_metadata_items": raw_identifier_contributions,
        "soc_relevant_evidence_items": raw_soc_evidence_contributions,
        "analyst_evidence_identifier_leakage": len(
            event_id_leakage
        ),
        "category_distribution": category_counts,
    },
    "evidence_records": evidence_records,
    "quality_checks": quality_checks,
    "limitations": [
        "Coefficient contribution indicates model association, not causality.",
        "Identifier and metadata features may still influence the model internally but are excluded from analyst-facing evidence.",
        "The Gate 3B.2 result is based on a deterministic test-row sample.",
        "This gate validates evidence extraction quality, not overall model detection performance.",
    ],
}

REPORT_PATH = (
    REPORT_DIR
    / "phase3b_gate3_2_evidence_quality_report.json"
)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================================
# FINAL
# ============================================================================

print(
    f"\nReport: {REPORT_PATH}"
)

print(
    "NO PRODUCTION ARTIFACTS WERE MODIFIED."
)

print("=" * 72)