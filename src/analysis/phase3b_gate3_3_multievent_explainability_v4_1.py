from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# SMARTSOC PHASE 3B
# GATE 3B.3 — MULTI-EVENT EXPLAINABILITY VALIDATION
#
# Purpose:
#   Validate explainability across multiple informative test events rather
#   than relying on a single arbitrary event.
#
# This gate:
#   - Uses the existing production model.
#   - Uses the existing production preprocessing artifacts.
#   - Does NOT retrain.
#   - Does NOT modify thresholds.
#   - Does NOT modify production artifacts.
#   - Uses the untouched test dataset only for read-only analysis.
#
# Evidence categories:
#   - process_evidence
#   - command_behavior
#   - network_evidence
#   - artifact_evidence
#   - text_behavior
#   - contextual_evidence
#   - behavior_signal
#   - other
#
# Identifier / metadata features are excluded from analyst-facing evidence.
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
    / "phase3b_gate3_3"
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
# CONFIGURATION
# ============================================================================

# Number of informative events to analyze.
MAX_EVENTS = 20

# We first search for events with meaningful structured/text evidence.
# Then we rank them by the maximum model probability.
MIN_BEHAVIOR_TEXT_LENGTH = 5

# Number of top model techniques considered per event.
TOP_TECHNIQUES_PER_EVENT = 3

# Number of positive model contributions retained per technique.
TOP_EVIDENCE_PER_TECHNIQUE = 8


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
    Features that must never be presented as analyst-facing attack evidence.
    """

    name = normalize_name(feature_name)

    blocked = {
        "event_id",
    }

    return name in blocked or name.startswith("event_id")


def evidence_category(feature_name):
    """
    Classify a feature according to SOC usefulness.
    """

    name = normalize_name(feature_name)

    if is_identifier_or_metadata(name):
        return "identifier_or_metadata"

    if name.startswith("tfidf:") or name.startswith("tfidf__"):
        return "text_behavior"

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

    if "strong_behavior_signal" in name:
        return "behavior_signal"

    return "other"


def get_column_list(schema, keys):

    for key in keys:

        value = schema.get(key)

        if isinstance(value, list):
            return value

    return None


def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================================
# HEADER
# ============================================================================

print("=" * 72)
print("SMARTSOC PHASE 3B")
print("GATE 3B.3 — MULTI-EVENT EXPLAINABILITY VALIDATION")
print("=" * 72)


# ============================================================================
# ARTIFACT VALIDATION
# ============================================================================

print("\nPRODUCTION ARTIFACTS")

required_artifacts = [
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    TFIDF_PATH,
    MLB_PATH,
    SCHEMA_PATH,
    THRESHOLD_PATH,
]

for path in required_artifacts:

    if path.exists():
        print(f"[PASS] {path.name}")

    else:
        fail(f"Missing artifact: {path}")


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

print("[PASS] Production artifacts loaded")


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
        categorical_columns = list(
            encoder.feature_names_in_
        )

if numerical_columns is None:

    if hasattr(scaler, "feature_names_in_"):
        numerical_columns = list(
            scaler.feature_names_in_
        )

if categorical_columns is None:
    fail("Could not determine categorical columns.")

if numerical_columns is None:
    fail("Could not determine numerical columns.")

mitre_techniques = list(mlb.classes_)

expected_feature_count = int(
    schema.get("final_feature_count", 6919)
)

print(
    f"  Categorical : {len(categorical_columns)}"
)

print(
    f"  Numerical   : {len(numerical_columns)}"
)

print(
    f"  Techniques  : {len(mitre_techniques)}"
)

print(
    f"  Final features: {expected_feature_count}"
)


# ============================================================================
# FEATURE NAME MAP
# ============================================================================

print("\nBUILDING FEATURE MAP")

try:

    categorical_feature_names = list(
        encoder.get_feature_names_out(
            categorical_columns
        )
    )

except TypeError:

    categorical_feature_names = list(
        encoder.get_feature_names_out()
    )


numerical_feature_names = list(
    numerical_columns
)

tfidf_feature_names = [
    f"tfidf:{term}"
    for term in tfidf.get_feature_names_out()
]


feature_names = (
    categorical_feature_names
    + numerical_feature_names
    + tfidf_feature_names
)


print(
    f"  Categorical encoded : "
    f"{len(categorical_feature_names)}"
)

print(
    f"  Numerical           : "
    f"{len(numerical_feature_names)}"
)

print(
    f"  TF-IDF              : "
    f"{len(tfidf_feature_names)}"
)

print(
    f"  Final               : "
    f"{len(feature_names)}"
)


if len(feature_names) != expected_feature_count:

    fail(
        "Feature-name count does not match "
        f"production expectation: "
        f"{len(feature_names)} != "
        f"{expected_feature_count}"
    )

print("[PASS] Feature mapping")


# ============================================================================
# LOAD DATA
# ============================================================================

print("\nINPUT DATA")

if not STRUCTURED_TEST.exists():
    fail(f"Missing test dataset: {STRUCTURED_TEST}")

if not BEHAVIOR_TEST.exists():
    fail(f"Missing behavior dataset: {BEHAVIOR_TEST}")


structured_df = pd.read_csv(
    STRUCTURED_TEST
)

behavior_df = pd.read_csv(
    BEHAVIOR_TEST
)


if len(structured_df) != len(behavior_df):

    fail(
        "Structured/behavior row count mismatch: "
        f"{len(structured_df)} vs "
        f"{len(behavior_df)}"
    )


behavior_column = "behavior_text_normalized"


if behavior_column not in behavior_df.columns:

    fail(
        f"Missing behavior column: "
        f"{behavior_column}"
    )


print(
    f"  Structured rows: "
    f"{len(structured_df)}"
)

print(
    f"  Behavior rows  : "
    f"{len(behavior_df)}"
)


# ============================================================================
# FIND INFORMATIVE EVENTS
# ============================================================================

print("\nSEARCHING FOR INFORMATIVE EVENTS")

candidate_records = []


for index in range(len(structured_df)):

    row = structured_df.iloc[index]
    behavior_value = safe_text(
        behavior_df.iloc[index][behavior_column]
    )

    behavior_length = len(behavior_value)

    # Count meaningful behavioral indicators.
    signal_count = 0

    signal_columns = [
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
        "is_network_event",
        "has_source_ip",
        "has_destination_ip",
        "has_source_port",
        "has_destination_port",
        "has_protocol",
        "has_target_filename",
        "has_image_loaded",
        "has_target_object",
        "strong_behavior_signal",
    ]

    for column in signal_columns:

        if column in row.index:

            try:

                value = float(row[column])

                if np.isfinite(value) and value != 0:
                    signal_count += 1

            except Exception:
                pass


    # Events with text or explicit structured behavioral signals
    # are preferred over empty administrative/security records.
    informative = (
        behavior_length >= MIN_BEHAVIOR_TEXT_LENGTH
        or signal_count > 0
    )

    if informative:

        candidate_records.append(
            {
                "index": index,
                "behavior_length": behavior_length,
                "signal_count": signal_count,
            }
        )


print(
    f"  Informative candidates: "
    f"{len(candidate_records)}"
)


if not candidate_records:

    fail(
        "No informative events found in test set."
    )


# ============================================================================
# PRE-SCREEN CANDIDATES
#
# We now run production inference on candidate events and rank them by
# maximum probability. This is read-only analysis.
# ============================================================================

print("\nPRE-SCREENING CANDIDATES")

candidate_scores = []


for position, candidate in enumerate(
    candidate_records
):

    index = candidate["index"]

    row_df = structured_df.iloc[
        [index]
    ].copy()

    behavior_value = safe_text(
        behavior_df.iloc[index][behavior_column]
    )

    try:

        cat_input = row_df[
            categorical_columns
        ]

        num_input = row_df[
            numerical_columns
        ].to_numpy(
            dtype=np.float64
        )

        X_cat = encoder.transform(
            cat_input
        )

        X_num = scaler.transform(
            num_input
        )

        X_text = tfidf.transform(
            [behavior_value]
        )

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

        probabilities = np.asarray(
            model.predict_proba(X_final)
        )

        max_probability = float(
            np.max(probabilities[0])
        )

        top_index = int(
            np.argmax(probabilities[0])
        )

        top_technique = str(
            mitre_techniques[top_index]
        )

        candidate_scores.append(
            {
                **candidate,
                "max_probability": max_probability,
                "top_technique": top_technique,
            }
        )

    except Exception as exc:

        print(
            f"[WARN] Candidate {index} skipped: "
            f"{exc}"
        )


if not candidate_scores:

    fail(
        "No candidates survived production "
        "inference."
    )


# ============================================================================
# SELECT DIVERSE HIGH-INFORMATION EVENTS
# ============================================================================

candidate_scores = sorted(
    candidate_scores,
    key=lambda x: (
        x["max_probability"],
        x["signal_count"],
        x["behavior_length"],
    ),
    reverse=True,
)


# Prefer different top techniques when possible.
selected = []
seen_top_techniques = set()

for candidate in candidate_scores:

    if len(selected) >= MAX_EVENTS:
        break

    technique = candidate[
        "top_technique"
    ]

    if technique not in seen_top_techniques:

        selected.append(candidate)

        seen_top_techniques.add(
            technique
        )


# Fill remaining slots if fewer than MAX_EVENTS
# have unique top techniques.
if len(selected) < MAX_EVENTS:

    selected_indices = {
        item["index"]
        for item in selected
    }

    for candidate in candidate_scores:

        if len(selected) >= MAX_EVENTS:
            break

        if candidate["index"] not in selected_indices:

            selected.append(candidate)

            selected_indices.add(
                candidate["index"]
            )


print(
    f"  Selected events: "
    f"{len(selected)}"
)

for item in selected:

    print(
        f"  Row {item['index']:<5} "
        f"Top={item['top_technique']:<10} "
        f"P={item['max_probability']:.6f} "
        f"Signals={item['signal_count']:<2} "
        f"TextLen={item['behavior_length']}"
    )


# ============================================================================
# COEFFICIENT MATRIX
# ============================================================================

print("\nCOEFFICIENT MATRIX")

if not hasattr(model, "estimators_"):

    fail(
        "Production model does not expose "
        "estimators_."
    )


estimators = model.estimators_


if len(estimators) != len(mitre_techniques):

    fail(
        "Estimator/technique count mismatch."
    )


coefficient_matrix = np.vstack(
    [
        np.asarray(
            estimator.coef_
        ).reshape(-1)

        for estimator in estimators
    ]
)


if coefficient_matrix.shape != (
    len(mitre_techniques),
    expected_feature_count,
):

    fail(
        "Coefficient matrix alignment failed: "
        f"{coefficient_matrix.shape}"
    )


print(
    f"  Shape: "
    f"{coefficient_matrix.shape}"
)

print("[PASS] Coefficient alignment")


# ============================================================================
# EXPLAIN SELECTED EVENTS
# ============================================================================

print("\nMULTI-EVENT EXPLAINABILITY")

event_results = []

global_category_counts = {}

identifier_leakage = []

missing_value_evidence = []

total_technique_explanations = 0

technique_explanations_with_evidence = 0

technique_explanations_with_behavioral_evidence = 0


for event_number, candidate in enumerate(
    selected,
    start=1
):

    index = candidate["index"]

    row_df = structured_df.iloc[
        [index]
    ].copy()

    behavior_value = safe_text(
        behavior_df.iloc[index][behavior_column]
    )

    # Rebuild production vector.
    cat_input = row_df[
        categorical_columns
    ]

    num_input = row_df[
        numerical_columns
    ].to_numpy(
        dtype=np.float64
    )

    X_cat = encoder.transform(
        cat_input
    )

    X_num = scaler.transform(
        num_input
    )

    X_text = tfidf.transform(
        [behavior_value]
    )

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

    probabilities = np.asarray(
        model.predict_proba(X_final)
    )[0]

    contribution_matrix = (
        X_final
        * coefficient_matrix
    )

    ranking = np.argsort(
        probabilities
    )[::-1]

    top_indices = ranking[
        :TOP_TECHNIQUES_PER_EVENT
    ]

    event_explanations = []

    print("\n" + "=" * 72)

    print(
        f"EVENT #{event_number} "
        f"(test row {index})"
    )

    if "event_id" in row_df.columns:

        print(
            f"Event ID: "
            f"{row_df.iloc[0]['event_id']}"
        )

    print(
        f"Behavior text: "
        f"{behavior_value[:250]}"
    )

    for rank, technique_index in enumerate(
        top_indices,
        start=1
    ):

        technique = str(
            mitre_techniques[
                technique_index
            ]
        )

        probability = float(
            probabilities[
                technique_index
            ]
        )

        threshold_value = thresholds.get(
            technique,
            thresholds.get(
                "default_threshold",
                0.5
            )
        )

        try:

            threshold_value = float(
                threshold_value
            )

        except Exception:

            threshold_value = 0.5


        detected = (
            probability >= threshold_value
        )


        contribution_vector = (
            contribution_matrix[
                technique_index
            ]
        )

        positive_indices = np.where(
            contribution_vector > 0
        )[0]

        positive_indices = sorted(
            positive_indices,
            key=lambda i:
                contribution_vector[i],
            reverse=True,
        )


        evidence = []

        for feature_index in positive_indices:

            feature_name = str(
                feature_names[
                    feature_index
                ]
            )

            contribution = float(
                contribution_vector[
                    feature_index
                ]
            )

            value = float(
                X_final[
                    0,
                    feature_index
                ]
            )

            category = evidence_category(
                feature_name
            )


            if category == (
                "identifier_or_metadata"
            ):

                identifier_leakage.append(
                    {
                        "event_row": index,
                        "technique": technique,
                        "feature": feature_name,
                        "contribution":
                            contribution,
                    }
                )

                continue


            # Exclude zero/non-finite values.
            if not np.isfinite(value):
                continue

            if not np.isfinite(contribution):
                continue

            if value == 0:
                continue


            # Track missing-value-derived evidence.
            if "<missing>" in feature_name.lower():

                missing_value_evidence.append(
                    {
                        "event_row": index,
                        "technique": technique,
                        "feature": feature_name,
                        "contribution":
                            contribution,
                    }
                )


            evidence.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "contribution":
                        contribution,
                    "category": category,
                }
            )


            global_category_counts[
                category
            ] = (
                global_category_counts.get(
                    category,
                    0
                ) + 1
            )


            if len(evidence) >= (
                TOP_EVIDENCE_PER_TECHNIQUE
            ):

                break


        has_evidence = (
            len(evidence) > 0
        )

        has_behavioral_evidence = any(
            item["category"] in {
                "command_behavior",
                "process_evidence",
                "network_evidence",
                "artifact_evidence",
                "text_behavior",
                "behavior_signal",
            }
            for item in evidence
        )


        total_technique_explanations += 1

        if has_evidence:

            technique_explanations_with_evidence += 1

        if has_behavioral_evidence:

            technique_explanations_with_behavioral_evidence += 1


        print(
            f"\n  #{rank} {technique} "
            f"P={probability:.6f} "
            f"T={threshold_value:.6f} "
            f"Detected={detected}"
        )


        if evidence:

            print(
                "    SOC evidence:"
            )

            for item in evidence[:5]:

                print(
                    f"      + "
                    f"{item['feature']}"
                    f" | value="
                    f"{item['value']:.6f}"
                    f" | contribution="
                    f"{item['contribution']:.6f}"
                    f" | "
                    f"{item['category']}"
                )

        else:

            print(
                "    SOC evidence: [NONE]"
            )


        event_explanations.append(
            {
                "rank": rank,
                "technique": technique,
                "probability": probability,
                "threshold": threshold_value,
                "margin":
                    probability -
                    threshold_value,
                "detected": bool(
                    detected
                ),
                "soc_relevant_evidence":
                    evidence,
                "has_evidence":
                    has_evidence,
                "has_behavioral_evidence":
                    has_behavioral_evidence,
            }
        )


    event_results.append(
        {
            "event_number":
                event_number,
            "test_row":
                index,
            "event_id":
                (
                    str(
                        row_df.iloc[0][
                            "event_id"
                        ]
                    )
                    if "event_id"
                    in row_df.columns
                    else None
                ),
            "behavior_text":
                behavior_value,
            "behavior_text_length":
                len(behavior_value),
            "structured_signal_count":
                candidate[
                    "signal_count"
                ],
            "top_technique":
                str(
                    mitre_techniques[
                        ranking[0]
                    ]
                ),
            "max_probability":
                float(
                    probabilities[
                        ranking[0]
                    ]
                ),
            "explanations":
                event_explanations,
        }
    )


# ============================================================================
# COVERAGE METRICS
# ============================================================================

print("\n" + "=" * 72)

print(
    "MULTI-EVENT EXPLAINABILITY METRICS"
)

print("=" * 72)


evidence_coverage = (
    technique_explanations_with_evidence
    / total_technique_explanations
    if total_technique_explanations
    else 0.0
)

behavioral_evidence_coverage = (
    technique_explanations_with_behavioral_evidence
    / total_technique_explanations
    if total_technique_explanations
    else 0.0
)


detected_techniques = 0

for event in event_results:

    for explanation in event[
        "explanations"
    ]:

        if explanation["detected"]:

            detected_techniques += 1


print(
    f"Events analyzed                    : "
    f"{len(event_results)}"
)

print(
    f"Technique explanations              : "
    f"{total_technique_explanations}"
)

print(
    f"Detected technique predictions      : "
    f"{detected_techniques}"
)

print(
    f"Explanations with SOC evidence      : "
    f"{technique_explanations_with_evidence}"
)

print(
    f"Evidence coverage                   : "
    f"{evidence_coverage:.4f}"
)

print(
    f"Explanations with behavioral evidence: "
    f"{technique_explanations_with_behavioral_evidence}"
)

print(
    f"Behavioral evidence coverage        : "
    f"{behavioral_evidence_coverage:.4f}"
)

print(
    f"Identifier leakage                  : "
    f"{len(identifier_leakage)}"
)

print(
    f"Missing-value evidence items        : "
    f"{len(missing_value_evidence)}"
)


# ============================================================================
# CATEGORY DISTRIBUTION
# ============================================================================

print("\nSOC EVIDENCE CATEGORY DISTRIBUTION")

if global_category_counts:

    for category, count in sorted(
        global_category_counts.items(),
        key=lambda item:
            (-item[1], item[0])
    ):

        print(
            f"  {category:<24} "
            f"{count}"
        )

else:

    print("  [NONE]")


# ============================================================================
# QUALITY CHECKS
# ============================================================================

print("\nQUALITY CHECKS")

checks = {
    "production_feature_dimension_valid":
        expected_feature_count == 6919,

    "technique_count_valid":
        len(mitre_techniques) == 45,

    "multi_event_analysis_completed":
        len(event_results) > 0,

    "evidence_extraction_completed":
        total_technique_explanations > 0,

    "identifier_metadata_not_exposed":
        len(identifier_leakage) == 0,

    "behavioral_evidence_available":
        technique_explanations_with_behavioral_evidence > 0,
}


for key, value in checks.items():

    status = (
        "PASS"
        if value
        else "FAIL"
    )

    print(
        f"  [{status}] {key}"
    )


all_checks_pass = all(
    checks.values()
)


# ============================================================================
# DECISION
# ============================================================================

if all_checks_pass:

    decision = "PASS"

else:

    decision = "REVIEW_REQUIRED"


print(
    f"\nDECISION: {decision}"
)


# ============================================================================
# REPORT
# ============================================================================

report = {
    "phase": "3B",
    "gate": "3B.3",
    "title":
        "Multi-Event Explainability Validation",

    "decision": decision,

    "production_artifacts_modified":
        False,

    "configuration": {
        "max_events": MAX_EVENTS,
        "min_behavior_text_length":
            MIN_BEHAVIOR_TEXT_LENGTH,
        "top_techniques_per_event":
            TOP_TECHNIQUES_PER_EVENT,
        "top_evidence_per_technique":
            TOP_EVIDENCE_PER_TECHNIQUE,
    },

    "dataset": {
        "structured_rows":
            int(len(structured_df)),
        "behavior_rows":
            int(len(behavior_df)),
        "informative_candidates":
            int(len(candidate_records)),
        "selected_events":
            int(len(event_results)),
    },

    "feature_dimensions": {
        "categorical":
            len(categorical_columns),
        "categorical_encoded":
            len(categorical_feature_names),
        "numerical":
            len(numerical_feature_names),
        "tfidf":
            len(tfidf_feature_names),
        "final":
            len(feature_names),
        "mitre_techniques":
            len(mitre_techniques),
    },

    "metrics": {
        "events_analyzed":
            len(event_results),
        "technique_explanations":
            total_technique_explanations,
        "detected_technique_predictions":
            detected_techniques,
        "explanations_with_soc_evidence":
            technique_explanations_with_evidence,
        "evidence_coverage":
            evidence_coverage,
        "explanations_with_behavioral_evidence":
            technique_explanations_with_behavioral_evidence,
        "behavioral_evidence_coverage":
            behavioral_evidence_coverage,
        "identifier_leakage":
            len(identifier_leakage),
        "missing_value_evidence_items":
            len(missing_value_evidence),
    },

    "evidence_category_distribution":
        global_category_counts,

    "selected_events":
        event_results,

    "quality_checks":
        checks,

    "identifier_leakage":
        identifier_leakage,

    "missing_value_evidence":
        missing_value_evidence,

    "limitations": [
        "Coefficient contributions indicate model association, not causality.",
        "This gate evaluates explainability behavior, not overall detection performance.",
        "Events are selected from the untouched test set for read-only analysis.",
        "Event identifiers are deliberately excluded from analyst-facing evidence.",
        "Missing-value features may be technically predictive but should be treated cautiously as SOC evidence.",
    ],
}


REPORT_PATH = (
    REPORT_DIR
    / "phase3b_gate3_3_multievent_explainability_report.json"
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