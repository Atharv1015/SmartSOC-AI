import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# SMARTSOC PHASE 2 GATE 2.4I
# LABEL / FEATURE SEPARABILITY ANALYSIS
# ============================================================================
#
# Purpose:
#   Determine whether weak MITRE ATT&CK techniques in the untouched test set
#   have distinguishable feature patterns or are effectively indistinguishable
#   from other labels.
#
# IMPORTANT:
#   - No model modification
#   - No threshold modification
#   - No dataset modification
#   - No retraining
#   - Untouched test set remains untouched
#
# Outputs:
#   reports\phase2_gate2_4i\
# ============================================================================


ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATA_DIR = ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
REPORT_DIR = ROOT / "reports" / "phase2_gate2_4i"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_TEST = DATA_DIR / "structured_test.csv"
LABELS_TEST = DATA_DIR / "labels_test.csv"
BEHAVIOR_TEST = DATA_DIR / "behavior_text_test.csv"

SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"


# ============================================================================
# TECHNIQUES REQUIRING INVESTIGATION
# ============================================================================

TARGET_TECHNIQUES = [
    "T1078",
    "T1112",
    "T1565.001",
    "T1485",
    "T1531",
    "T1562.001",
    "T1070.004",
    "T1548.002",
    "T1059.001",
    "T1543.003",
]


# ============================================================================
# HELPERS
# ============================================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def parse_labels(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []

    return [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]


def one_hot_presence(labels_series, technique):
    return labels_series.apply(
        lambda x: 1 if technique in x else 0
    ).astype(int)


def numeric_summary(df, mask, columns):
    rows = []

    for col in columns:
        if col not in df.columns:
            continue

        values = safe_numeric(df.loc[mask, col])

        if len(values) == 0:
            continue

        rows.append({
            "feature": col,
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        })

    return rows


def effect_size(mean_pos, mean_neg, std_pos, std_neg):
    denominator = np.sqrt(
        ((std_pos ** 2) + (std_neg ** 2)) / 2.0
    )

    if denominator == 0:
        if mean_pos == mean_neg:
            return 0.0
        return float("inf")

    return float((mean_pos - mean_neg) / denominator)


# ============================================================================
# LOAD ARTIFACTS
# ============================================================================

def load_artifacts():

    banner("SMARTSOC PHASE 2 GATE 2.4I")
    print("LABEL / FEATURE SEPARABILITY ANALYSIS")
    print("=" * 78)

    print()
    print("Loading production artifacts...")
    print("-" * 78)

    for path in [
        STRUCTURED_TEST,
        LABELS_TEST,
        BEHAVIOR_TEST,
        SCHEMA_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        MLB_FILE,
        MODEL_FILE,
        THRESHOLD_FILE,
    ]:
        require_file(path)

    schema = json.loads(
        SCHEMA_FILE.read_text(encoding="utf-8")
    )

    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    mlb = joblib.load(MLB_FILE)
    model = joblib.load(MODEL_FILE)

    thresholds = json.loads(
        THRESHOLD_FILE.read_text(encoding="utf-8")
    )

    categorical_columns = list(
        schema.get("categorical_columns", [])
    )

    numerical_columns = list(
        schema.get("numerical_columns", [])
    )

    structured_columns = (
        categorical_columns +
        numerical_columns
    )

    print(f"Model type           : {type(model).__name__}")
    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns   : {len(structured_columns)}")
    print(f"TF-IDF vocabulary    : {len(tfidf.vocabulary_)}")
    print(f"MITRE classes        : {len(mlb.classes_)}")
    print(
        f"Threshold entries    : "
        f"{len(thresholds.get('thresholds', {}))}"
    )

    return (
        schema,
        encoder,
        scaler,
        tfidf,
        mlb,
        model,
        thresholds,
        categorical_columns,
        numerical_columns,
        structured_columns,
    )


# ============================================================================
# LOAD TEST DATA
# ============================================================================

def load_test_data(structured_columns):

    banner("LOADING UNTOUCHED TEST DATA")

    structured = pd.read_csv(STRUCTURED_TEST)
    labels = pd.read_csv(LABELS_TEST)
    behavior = pd.read_csv(BEHAVIOR_TEST)

    print(f"Test structured rows : {len(structured)}")
    print(f"Test label rows      : {len(labels)}")
    print(f"Test behavior rows   : {len(behavior)}")

    if len(structured) != len(labels):
        raise RuntimeError("Structured/label row count mismatch.")

    if len(structured) != len(behavior):
        raise RuntimeError("Structured/behavior row count mismatch.")

    missing = [
        col for col in structured_columns
        if col not in structured.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing structured columns: {missing}"
        )

    if "techniques" not in labels.columns:
        raise RuntimeError(
            "labels_test.csv does not contain 'techniques'."
        )

    behavior_column = "behavior_text_normalized"

    if behavior_column not in behavior.columns:
        raise RuntimeError(
            f"Missing behavior column: {behavior_column}"
        )

    structured = structured[structured_columns].copy()

    parsed_labels = labels["techniques"].apply(parse_labels)

    behavior_text = (
        behavior[behavior_column]
        .fillna("")
        .astype(str)
    )

    print("[PASS] Test data alignment validated")

    return structured, parsed_labels, behavior_text


# ============================================================================
# PRODUCTION TRANSFORMATION
# ============================================================================

def transform_production(
    structured,
    behavior_text,
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns,
):
    banner("PRODUCTION TRANSFORMATION")

    X_cat = encoder.transform(
        structured[categorical_columns]
    )

    X_num = scaler.transform(
        structured[numerical_columns]
    )

    X_text = tfidf.transform(
        behavior_text
    )

    from scipy import sparse

    X_final = sparse.hstack(
        [
            X_cat,
            sparse.csr_matrix(X_num),
            X_text,
        ],
        format="csr",
    )

    print(f"Categorical matrix : {X_cat.shape}")
    print(f"Numerical matrix   : {X_num.shape}")
    print(f"TF-IDF matrix      : {X_text.shape}")
    print(f"Final matrix       : {X_final.shape}")

    expected = (
        X_cat.shape[1]
        + X_num.shape[1]
        + X_text.shape[1]
    )

    print(f"Expected features  : {expected}")

    if X_final.shape[1] != expected:
        raise RuntimeError(
            "Final production feature dimension mismatch."
        )

    print("[PASS] Feature dimensions validated")

    return X_final


# ============================================================================
# FEATURE SET
# ============================================================================

def choose_analysis_features(
    structured,
    numerical_columns,
    categorical_columns,
):
    """
    Focus on interpretable structured features.

    These are the same production structured features used by the model.
    """

    preferred = [
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

    return [
        col for col in preferred
        if col in structured.columns
    ]


# ============================================================================
# TECHNIQUE PROFILE
# ============================================================================

def analyze_technique(
    technique,
    structured,
    labels,
    probabilities,
    mlb_classes,
    thresholds,
    analysis_features,
):

    class_list = list(mlb_classes)

    if technique not in class_list:
        print(f"[WARNING] {technique} not present in model classes.")
        return None, [], []

    class_index = class_list.index(technique)

    y_true = np.array([
        1 if technique in row else 0
        for row in labels
    ])

    y_prob = probabilities[:, class_index]

    threshold = float(
        thresholds.get("thresholds", {}).get(
            technique,
            thresholds.get("default_threshold", 0.5),
        )
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    positive_mask = y_true == 1
    negative_mask = y_true == 0

    support = int(positive_mask.sum())

    predicted_positive = int(y_pred.sum())

    tp = int(
        np.logical_and(
            y_true == 1,
            y_pred == 1
        ).sum()
    )

    fn = int(
        np.logical_and(
            y_true == 1,
            y_pred == 0
        ).sum()
    )

    fp = int(
        np.logical_and(
            y_true == 0,
            y_pred == 1
        ).sum()
    )

    tn = int(
        np.logical_and(
            y_true == 0,
            y_pred == 0
        ).sum()
    )

    recall = (
        tp / support
        if support > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    true_mean = (
        float(y_prob[positive_mask].mean())
        if support > 0
        else np.nan
    )

    false_mean = (
        float(y_prob[negative_mask].mean())
        if negative_mask.sum() > 0
        else np.nan
    )

    print()
    print("-" * 78)
    print(f"{technique}")
    print("-" * 78)

    print(f"Support                 : {support}")
    print(f"TP                      : {tp}")
    print(f"FN                      : {fn}")
    print(f"FP                      : {fp}")
    print(f"TN                      : {tn}")
    print(f"Recall                  : {recall:.4f}")
    print(f"Precision               : {precision:.4f}")
    print(f"F1                      : {f1:.4f}")
    print(f"Threshold               : {threshold:.4f}")
    print(f"True-label mean P       : {true_mean:.6f}")
    print(f"False-label mean P      : {false_mean:.6f}")

    feature_rows = []

    for feature in analysis_features:

        pos = safe_numeric(
            structured.loc[positive_mask, feature]
        )

        neg = safe_numeric(
            structured.loc[negative_mask, feature]
        )

        if len(pos) == 0 or len(neg) == 0:
            continue

        pos_mean = float(pos.mean())
        neg_mean = float(neg.mean())

        pos_std = float(pos.std(ddof=0))
        neg_std = float(neg.std(ddof=0))

        difference = pos_mean - neg_mean

        abs_difference = abs(difference)

        es = effect_size(
            pos_mean,
            neg_mean,
            pos_std,
            neg_std,
        )

        feature_rows.append({
            "technique": technique,
            "feature": feature,
            "positive_count": int(len(pos)),
            "negative_count": int(len(neg)),
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "mean_difference": difference,
            "absolute_mean_difference": abs_difference,
            "positive_std": pos_std,
            "negative_std": neg_std,
            "standardized_effect_size": es,
        })

    feature_rows_sorted = sorted(
        feature_rows,
        key=lambda x: x["absolute_mean_difference"],
        reverse=True,
    )

    print()
    print("Strongest structured-feature differences:")
    print()

    for row in feature_rows_sorted[:10]:

        print(
            f"  {row['feature']:<35} "
            f"positive={row['positive_mean']:.4f} "
            f"negative={row['negative_mean']:.4f} "
            f"diff={row['mean_difference']:+.4f}"
        )

    profile = {
        "technique": technique,
        "support": support,
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "predicted_positive": predicted_positive,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "threshold": threshold,
        "true_label_probability_mean": true_mean,
        "false_label_probability_mean": false_mean,
        "probability_separation": (
            true_mean - false_mean
            if not np.isnan(true_mean)
            and not np.isnan(false_mean)
            else np.nan
        ),
    }

    return profile, feature_rows_sorted, y_prob


# ============================================================================
# LABEL CO-OCCURRENCE ANALYSIS
# ============================================================================

def analyze_label_context(
    technique,
    labels,
):

    positive_rows = [
        row for row in labels
        if technique in row
    ]

    context_counts = {}

    for row in positive_rows:

        for other in row:

            if other == technique:
                continue

            context_counts[other] = (
                context_counts.get(other, 0) + 1
            )

    sorted_context = sorted(
        context_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    rows = []

    for other, count in sorted_context:

        rows.append({
            "technique": technique,
            "cooccurring_technique": other,
            "count": int(count),
            "fraction_of_positive_rows": (
                count / len(positive_rows)
                if positive_rows
                else 0.0
            ),
        })

    return rows


# ============================================================================
# FEATURE OVERLAP ANALYSIS
# ============================================================================

def calculate_feature_overlap(
    technique,
    structured,
    labels,
    analysis_features,
):

    y_true = np.array([
        1 if technique in row else 0
        for row in labels
    ])

    positive_mask = y_true == 1
    negative_mask = y_true == 0

    rows = []

    for feature in analysis_features:

        pos = safe_numeric(
            structured.loc[positive_mask, feature]
        )

        neg = safe_numeric(
            structured.loc[negative_mask, feature]
        )

        if len(pos) == 0 or len(neg) == 0:
            continue

        # For binary features, calculate positive prevalence.
        unique_values = set(
            pd.concat(
                [pos, neg],
                ignore_index=True
            ).round(8).unique()
        )

        is_binary = unique_values.issubset(
            {0.0, 1.0}
        )

        if is_binary:

            pos_rate = float(pos.mean())
            neg_rate = float(neg.mean())

            rows.append({
                "technique": technique,
                "feature": feature,
                "feature_type": "binary",
                "positive_rate": pos_rate,
                "negative_rate": neg_rate,
                "absolute_rate_difference": abs(
                    pos_rate - neg_rate
                ),
            })

        else:

            pos_mean = float(pos.mean())
            neg_mean = float(neg.mean())

            pos_std = float(pos.std(ddof=0))
            neg_std = float(neg.std(ddof=0))

            pooled = np.sqrt(
                (pos_std ** 2 + neg_std ** 2) / 2.0
            )

            if pooled == 0:
                overlap_ratio = (
                    1.0
                    if pos_mean == neg_mean
                    else 0.0
                )
            else:
                overlap_ratio = max(
                    0.0,
                    1.0 -
                    abs(pos_mean - neg_mean)
                    / (
                        pooled
                        + 1e-12
                    )
                )

            rows.append({
                "technique": technique,
                "feature": feature,
                "feature_type": "numeric",
                "positive_rate": np.nan,
                "negative_rate": np.nan,
                "absolute_rate_difference": abs(
                    pos_mean - neg_mean
                ),
                "positive_mean": pos_mean,
                "negative_mean": neg_mean,
                "overlap_indicator": overlap_ratio,
            })

    return rows


# ============================================================================
# MODEL PROBABILITY DISTRIBUTION
# ============================================================================

def analyze_probability_distribution(
    technique,
    probabilities,
    labels,
    mlb_classes,
    thresholds,
):

    class_list = list(mlb_classes)
    idx = class_list.index(technique)

    y_true = np.array([
        1 if technique in row else 0
        for row in labels
    ])

    probs = probabilities[:, idx]

    positive = probs[y_true == 1]
    negative = probs[y_true == 0]

    threshold = float(
        thresholds.get("thresholds", {}).get(
            technique,
            thresholds.get("default_threshold", 0.5),
        )
    )

    quantiles = [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.00,
    ]

    rows = []

    for q in quantiles:

        rows.append({
            "technique": technique,
            "group": "positive",
            "quantile": q,
            "probability": (
                float(np.quantile(positive, q))
                if len(positive)
                else np.nan
            ),
        })

        rows.append({
            "technique": technique,
            "group": "negative",
            "quantile": q,
            "probability": (
                float(np.quantile(negative, q))
                if len(negative)
                else np.nan
            ),
        })

    rows.append({
        "technique": technique,
        "group": "threshold",
        "quantile": np.nan,
        "probability": threshold,
    })

    return rows


# ============================================================================
# SAMPLE CONFUSION ANALYSIS
# ============================================================================

def collect_misclassified_examples(
    technique,
    structured,
    labels,
    probabilities,
    mlb_classes,
    thresholds,
    max_examples=25,
):

    class_list = list(mlb_classes)
    idx = class_list.index(technique)

    probs = probabilities[:, idx]

    y_true = np.array([
        1 if technique in row else 0
        for row in labels
    ])

    threshold = float(
        thresholds.get("thresholds", {}).get(
            technique,
            thresholds.get("default_threshold", 0.5),
        )
    )

    false_negative_indices = np.where(
        (y_true == 1) &
        (probs < threshold)
    )[0]

    # Most suspicious FNs are those closest to threshold.
    sorted_indices = sorted(
        false_negative_indices,
        key=lambda i: probs[i],
        reverse=True,
    )

    rows = []

    for idx in sorted_indices[:max_examples]:

        row = {
            "technique": technique,
            "row_index": int(idx),
            "probability": float(probs[idx]),
            "threshold": threshold,
        }

        for feature in [
            "process_name",
            "parent_process_name",
            "uses_powershell",
            "uses_cmd",
            "uses_net",
            "uses_reg",
            "uses_wevtutil",
            "uses_certutil",
            "has_encoded_command",
            "has_download_behavior",
            "has_registry_operation",
            "has_file_deletion",
            "has_obfuscation_hint",
            "has_target_filename",
            "has_target_object",
            "strong_behavior_signal",
        ]:

            if feature in structured.columns:
                value = structured.iloc[idx][feature]

                if pd.isna(value):
                    value = ""

                row[feature] = value

        row["true_labels"] = ",".join(
            labels[idx]
        )

        rows.append(row)

    return rows


# ============================================================================
# OVERALL DIAGNOSIS
# ============================================================================

def classify_diagnosis(profile):

    recall = profile["recall"]
    f1 = profile["f1"]
    separation = profile["probability_separation"]

    if profile["support"] == 0:
        return "NO_TEST_SUPPORT"

    if (
        recall == 0.0
        and separation < 0.15
    ):
        return "SEVERE_SEPARABILITY_FAILURE"

    if (
        recall == 0.0
        and separation >= 0.15
    ):
        return "THRESHOLD_OR_CALIBRATION_PROBLEM"

    if recall < 0.25:
        return "SEVERE_LOW_RECALL"

    if recall < 0.50:
        return "MODERATE_LOW_RECALL"

    if f1 < 0.70:
        return "PARTIAL_PERFORMANCE"

    return "RELATIVELY_HEALTHY"


# ============================================================================
# MAIN
# ============================================================================

def evaluate():

    (
        schema,
        encoder,
        scaler,
        tfidf,
        mlb,
        model,
        thresholds,
        categorical_columns,
        numerical_columns,
        structured_columns,
    ) = load_artifacts()

    structured, labels, behavior_text = load_test_data(
        structured_columns
    )

    X_final = transform_production(
        structured,
        behavior_text,
        encoder,
        scaler,
        tfidf,
        categorical_columns,
        numerical_columns,
    )

    banner("PRODUCTION INFERENCE")

    probabilities = model.predict_proba(X_final)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    print(
        f"Probability matrix : {probabilities.shape}"
    )

    if probabilities.shape[0] != len(structured):
        raise RuntimeError(
            "Probability row count mismatch."
        )

    if probabilities.shape[1] != len(mlb.classes_):
        raise RuntimeError(
            "Probability class count mismatch."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probability matrix contains non-finite values."
        )

    if (
        probabilities.min() < 0
        or probabilities.max() > 1
    ):
        raise RuntimeError(
            "Probability values outside [0,1]."
        )

    print("[PASS] Probability matrix valid")

    analysis_features = choose_analysis_features(
        structured,
        numerical_columns,
        categorical_columns,
    )

    print()
    print(
        f"Interpretable structured features analyzed: "
        f"{len(analysis_features)}"
    )

    # ------------------------------------------------------------------------
    # TECHNIQUE ANALYSIS
    # ------------------------------------------------------------------------

    banner("TARGET TECHNIQUE SEPARABILITY")

    profiles = []
    feature_rows = []
    context_rows = []
    probability_rows = []
    fn_example_rows = []

    for technique in TARGET_TECHNIQUES:

        profile, feature_profile, _ = analyze_technique(
            technique,
            structured,
            labels,
            probabilities,
            mlb.classes_,
            thresholds,
            analysis_features,
        )

        if profile is None:
            continue

        diagnosis = classify_diagnosis(profile)

        profile["diagnosis"] = diagnosis

        profiles.append(profile)
        feature_rows.extend(feature_profile)

        context_rows.extend(
            analyze_label_context(
                technique,
                labels,
            )
        )

        probability_rows.extend(
            analyze_probability_distribution(
                technique,
                probabilities,
                labels,
                mlb.classes_,
                thresholds,
            )
        )

        fn_example_rows.extend(
            collect_misclassified_examples(
                technique,
                structured,
                labels,
                probabilities,
                mlb.classes_,
                thresholds,
            )
        )

        print(
            f"\nDIAGNOSIS: {diagnosis}"
        )

    # ------------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------------

    banner("SEPARABILITY SUMMARY")

    profile_df = pd.DataFrame(profiles)

    if not profile_df.empty:

        columns = [
            "technique",
            "support",
            "true_positive",
            "false_negative",
            "false_positive",
            "true_negative",
            "predicted_positive",
            "recall",
            "precision",
            "f1",
            "threshold",
            "true_label_probability_mean",
            "false_label_probability_mean",
            "probability_separation",
            "diagnosis",
        ]

        profile_df = profile_df[
            [
                c for c in columns
                if c in profile_df.columns
            ]
        ]

        print(
            profile_df.to_string(
                index=False
            )
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    context_df = pd.DataFrame(
        context_rows
    )

    probability_df = pd.DataFrame(
        probability_rows
    )

    fn_df = pd.DataFrame(
        fn_example_rows
    )

    profile_path = (
        REPORT_DIR /
        "label_feature_separability_summary.csv"
    )

    feature_path = (
        REPORT_DIR /
        "label_feature_separability_features.csv"
    )

    context_path = (
        REPORT_DIR /
        "label_cooccurrence_analysis.csv"
    )

    probability_path = (
        REPORT_DIR /
        "probability_distribution_analysis.csv"
    )

    fn_path = (
        REPORT_DIR /
        "false_negative_examples.csv"
    )

    report_path = (
        REPORT_DIR /
        "phase2_gate2_4i_report.json"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    feature_df.to_csv(
        feature_path,
        index=False,
    )

    context_df.to_csv(
        context_path,
        index=False,
    )

    probability_df.to_csv(
        probability_path,
        index=False,
    )

    fn_df.to_csv(
        fn_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # HIGH-LEVEL DIAGNOSIS
    # ------------------------------------------------------------------------

    severe = profile_df[
        profile_df["diagnosis"] ==
        "SEVERE_SEPARABILITY_FAILURE"
    ] if not profile_df.empty else pd.DataFrame()

    calibration = profile_df[
        profile_df["diagnosis"] ==
        "THRESHOLD_OR_CALIBRATION_PROBLEM"
    ] if not profile_df.empty else pd.DataFrame()

    low_recall = profile_df[
        profile_df["diagnosis"].isin([
            "SEVERE_LOW_RECALL",
            "MODERATE_LOW_RECALL",
            "PARTIAL_PERFORMANCE",
        ])
    ] if not profile_df.empty else pd.DataFrame()

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4I",
        "gate_name": "Label / Feature Separability Analysis",
        "status": "ANALYSIS_COMPLETE",

        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,

        "test_rows": int(len(structured)),
        "techniques_analyzed": int(len(profiles)),
        "structured_features_analyzed": int(
            len(analysis_features)
        ),

        "diagnostic_counts": {
            "severe_separability_failure": int(
                len(severe)
            ),
            "threshold_or_calibration_problem": int(
                len(calibration)
            ),
            "low_recall_or_partial": int(
                len(low_recall)
            ),
        },

        "technique_profiles": profiles,

        "interpretation": {
            "severe_separability_failure": [
                str(x)
                for x in severe["technique"].tolist()
            ] if not severe.empty else [],

            "threshold_or_calibration_problem": [
                str(x)
                for x in calibration["technique"].tolist()
            ] if not calibration.empty else [],

            "low_recall_or_partial": [
                str(x)
                for x in low_recall["technique"].tolist()
            ] if not low_recall.empty else [],
        },

        "next_step": (
            "Use this analysis to determine whether weak MITRE "
            "techniques require dataset/label remediation, feature "
            "engineering, class balancing, threshold recalibration, "
            "or model retraining. Do not modify production artifacts "
            "until remediation is explicitly selected and validated."
        ),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------------

    banner("GATE 2.4I COMPLETE")

    print(
        f"Summary       : {profile_path}"
    )

    print(
        f"Features      : {feature_path}"
    )

    print(
        f"Co-occurrence : {context_path}"
    )

    print(
        f"Probabilities : {probability_path}"
    )

    print(
        f"FN examples   : {fn_path}"
    )

    print(
        f"Report        : {report_path}"
    )

    print()
    print(
        "No model, threshold, dataset, or production artifact "
        "was modified."
    )

    print()
    print("STATUS: ANALYSIS_COMPLETE")


if __name__ == "__main__":
    evaluate()