import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# SMARTSOC PHASE 2 GATE 2.4G
# FALSE-NEGATIVE DEEP DIVE
# =============================================================================

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATASET = ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
OUT_DIR = ROOT / "reports" / "phase2_gate2_4g"

OUT_DIR.mkdir(parents=True, exist_ok=True)


STRUCTURED_TEST = DATASET / "structured_test.csv"
LABELS_TEST = DATASET / "labels_test.csv"
BEHAVIOR_TEST = DATASET / "behavior_text_test.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
STRUCTURED_ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"


# Techniques with the largest FN impact from Gate 2.4F.
TARGET_TECHNIQUES = [
    "T1078",
    "T1112",
    "T1565.001",
    "T1070.004",
    "T1485",
    "T1059.001",
    "T1543.003",
    "T1548.002",
    "T1562.001",
    "T1531",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_label(value):
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


def validate_artifacts():
    print("=" * 78)
    print("PRODUCTION ARTIFACT VALIDATION")
    print("=" * 78)

    required = [
        STRUCTURED_TEST,
        LABELS_TEST,
        BEHAVIOR_TEST,
        MODEL_FILE,
        STRUCTURED_ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        MLB_FILE,
        SCHEMA_FILE,
        THRESHOLD_FILE,
    ]

    missing = [str(x) for x in required if not x.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(STRUCTURED_ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    mlb = joblib.load(MLB_FILE)
    schema = load_json(SCHEMA_FILE)
    thresholds = load_json(THRESHOLD_FILE)

    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]
    structured_columns = (
        categorical_columns + numerical_columns
    )

    if hasattr(model, "estimators_"):
        model_classes = len(model.estimators_)
    else:
        model_classes = len(getattr(model, "classes_", []))

    mitre_classes = list(mlb.classes_)

    threshold_map = thresholds.get("thresholds", {})

    print(f"Model type           : {type(model).__name__}")
    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns   : {len(structured_columns)}")
    print(f"TF-IDF vocabulary    : {len(tfidf.vocabulary_)}")
    print(f"MITRE classes        : {len(mitre_classes)}")
    print(f"Threshold entries    : {len(threshold_map)}")

    if len(structured_columns) != 62:
        raise RuntimeError(
            f"Expected 62 structured columns, got {len(structured_columns)}"
        )

    if len(numerical_columns) != 58:
        raise RuntimeError(
            f"Expected 58 numerical columns, got {len(numerical_columns)}"
        )

    if len(mitre_classes) != 45:
        raise RuntimeError(
            f"Expected 45 MITRE classes, got {len(mitre_classes)}"
        )

    print("[PASS] Production artifacts validated")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        threshold_map,
    )


def transform_test_data(
    structured,
    behavior,
    model,
    encoder,
    scaler,
    tfidf,
    schema,
):
    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]

    structured_columns = (
        categorical_columns + numerical_columns
    )

    structured = structured.copy()

    structured = structured[
        structured_columns
    ]

    X_cat_input = structured[
        categorical_columns
    ].copy()

    X_num_input = structured[
        numerical_columns
    ].copy()

    X_cat = encoder.transform(X_cat_input)

    X_num = scaler.transform(X_num_input)

    behavior_text = (
        behavior["behavior_text_normalized"]
        .fillna("")
        .astype(str)
    )

    X_text = tfidf.transform(behavior_text)

    try:
        from scipy.sparse import hstack

        X = hstack(
            [
                X_cat,
                X_num,
                X_text,
            ],
            format="csr",
        )

    except Exception:
        X = np.hstack(
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

    expected = (
        X_cat.shape[1]
        + X_num.shape[1]
        + X_text.shape[1]
    )

    print()
    print("TRANSFORMATION")
    print("-" * 78)
    print(f"Categorical matrix : {X_cat.shape}")
    print(f"Numerical matrix   : {X_num.shape}")
    print(f"TF-IDF matrix      : {X_text.shape}")
    print(f"Final matrix       : {X.shape}")
    print(f"Expected features  : {expected}")

    if X.shape[1] != expected:
        raise RuntimeError(
            "Feature dimension mismatch."
        )

    if X.shape[1] != 6919:
        raise RuntimeError(
            f"Production feature count expected 6919, got {X.shape[1]}"
        )

    print("[PASS] Feature dimension = 6919")

    return X


def build_truth_matrix(labels, classes):
    class_to_index = {
        technique: i
        for i, technique in enumerate(classes)
    }

    y = np.zeros(
        (len(labels), len(classes)),
        dtype=int,
    )

    for row_idx, row_labels in enumerate(labels):
        for technique in row_labels:
            if technique in class_to_index:
                y[
                    row_idx,
                    class_to_index[technique]
                ] = 1

    return y


def build_prediction_matrix(probabilities, classes, threshold_map):
    y_pred = np.zeros_like(
        probabilities,
        dtype=int,
    )

    for col_idx, technique in enumerate(classes):
        threshold = float(
            threshold_map.get(
                technique,
                0.5,
            )
        )

        y_pred[:, col_idx] = (
            probabilities[:, col_idx] >= threshold
        ).astype(int)

    return y_pred


def calculate_rank(probabilities, classes, target):
    if target not in classes:
        return None

    idx = classes.index(target)

    target_probability = probabilities[idx]

    rank = (
        1
        + int(
            np.sum(
                probabilities > target_probability
            )
        )
    )

    return rank


def get_top_predictions(
    probability_row,
    classes,
    top_n=5,
):
    order = np.argsort(
        probability_row
    )[::-1][:top_n]

    return [
        (
            classes[i],
            float(probability_row[i]),
        )
        for i in order
    ]


def analyze_target(
    technique,
    structured,
    behavior,
    labels,
    probabilities,
    y_true,
    y_pred,
    classes,
    threshold_map,
):
    if technique not in classes:
        return None

    idx = classes.index(technique)

    true_mask = y_true[:, idx] == 1

    predicted_mask = y_pred[:, idx] == 1

    fn_mask = (
        true_mask
        & ~predicted_mask
    )

    tp_mask = (
        true_mask
        & predicted_mask
    )

    support = int(true_mask.sum())
    tp = int(tp_mask.sum())
    fn = int(fn_mask.sum())

    recall = (
        tp / support
        if support > 0
        else 0.0
    )

    threshold = float(
        threshold_map.get(
            technique,
            0.5,
        )
    )

    target_probabilities = (
        probabilities[:, idx]
    )

    fn_probabilities = (
        target_probabilities[fn_mask]
        if fn > 0
        else np.array([])
    )

    tp_probabilities = (
        target_probabilities[tp_mask]
        if tp > 0
        else np.array([])
    )

    print()
    print("=" * 78)
    print(f"{technique}")
    print("=" * 78)

    print(f"Support              : {support}")
    print(f"True positives       : {tp}")
    print(f"False negatives      : {fn}")
    print(f"Recall               : {recall:.4f}")
    print(f"Threshold            : {threshold:.4f}")

    if fn > 0:
        print(
            f"FN probability mean : "
            f"{fn_probabilities.mean():.4f}"
        )

        print(
            f"FN probability median : "
            f"{np.median(fn_probabilities):.4f}"
        )

        print(
            f"FN probability min  : "
            f"{fn_probabilities.min():.4f}"
        )

        print(
            f"FN probability max  : "
            f"{fn_probabilities.max():.4f}"
        )

    if tp > 0:
        print(
            f"TP probability mean : "
            f"{tp_probabilities.mean():.4f}"
        )

        print(
            f"TP probability median : "
            f"{np.median(tp_probabilities):.4f}"
        )

    rows = []

    fn_indices = np.where(fn_mask)[0]

    for row_idx in fn_indices:
        probability_row = probabilities[
            row_idx
        ]

        rank = calculate_rank(
            probability_row,
            classes,
            technique,
        )

        top_predictions = get_top_predictions(
            probability_row,
            classes,
            top_n=5,
        )

        structured_row = structured.iloc[
            row_idx
        ]

        behavior_text = str(
            behavior.iloc[row_idx][
                "behavior_text_normalized"
            ]
        )

        top1_technique, top1_probability = (
            top_predictions[0]
        )

        rows.append(
            {
                "test_index": int(row_idx),
                "technique": technique,
                "target_probability": float(
                    probability_row[idx]
                ),
                "threshold": threshold,
                "rank": rank,
                "top1_technique": top1_technique,
                "top1_probability": float(
                    top1_probability
                ),
                "top2_technique": top_predictions[1][0],
                "top2_probability": float(
                    top_predictions[1][1]
                ),
                "top3_technique": top_predictions[2][0],
                "top3_probability": float(
                    top_predictions[2][1]
                ),
                "top4_technique": top_predictions[3][0],
                "top4_probability": float(
                    top_predictions[3][1]
                ),
                "top5_technique": top_predictions[4][0],
                "top5_probability": float(
                    top_predictions[4][1]
                ),
                "process_name": structured_row.get(
                    "process_name",
                    "",
                ),
                "parent_process_name": structured_row.get(
                    "parent_process_name",
                    "",
                ),
                "command_length": structured_row.get(
                    "command_length",
                    0,
                ),
                "has_command_line": structured_row.get(
                    "has_command_line",
                    0,
                ),
                "uses_powershell": structured_row.get(
                    "uses_powershell",
                    0,
                ),
                "uses_cmd": structured_row.get(
                    "uses_cmd",
                    0,
                ),
                "uses_net": structured_row.get(
                    "uses_net",
                    0,
                ),
                "uses_whoami": structured_row.get(
                    "uses_whoami",
                    0,
                ),
                "uses_schtasks": structured_row.get(
                    "uses_schtasks",
                    0,
                ),
                "uses_reg": structured_row.get(
                    "uses_reg",
                    0,
                ),
                "uses_rundll32": structured_row.get(
                    "uses_rundll32",
                    0,
                ),
                "uses_wmic": structured_row.get(
                    "uses_wmic",
                    0,
                ),
                "uses_certutil": structured_row.get(
                    "uses_certutil",
                    0,
                ),
                "uses_bitsadmin": structured_row.get(
                    "uses_bitsadmin",
                    0,
                ),
                "uses_mshta": structured_row.get(
                    "uses_mshta",
                    0,
                ),
                "uses_regsvr32": structured_row.get(
                    "uses_regsvr32",
                    0,
                ),
                "uses_cscript": structured_row.get(
                    "uses_cscript",
                    0,
                ),
                "uses_wscript": structured_row.get(
                    "uses_wscript",
                    0,
                ),
                "uses_wevtutil": structured_row.get(
                    "uses_wevtutil",
                    0,
                ),
                "uses_vssadmin": structured_row.get(
                    "uses_vssadmin",
                    0,
                ),
                "has_encoded_command": structured_row.get(
                    "has_encoded_command",
                    0,
                ),
                "has_download_behavior": structured_row.get(
                    "has_download_behavior",
                    0,
                ),
                "has_remote_execution_hint": structured_row.get(
                    "has_remote_execution_hint",
                    0,
                ),
                "has_registry_operation": structured_row.get(
                    "has_registry_operation",
                    0,
                ),
                "has_file_deletion": structured_row.get(
                    "has_file_deletion",
                    0,
                ),
                "has_obfuscation_hint": structured_row.get(
                    "has_obfuscation_hint",
                    0,
                ),
                "administrative_powershell_context": structured_row.get(
                    "administrative_powershell_context",
                    0,
                ),
                "strong_behavior_signal": structured_row.get(
                    "strong_behavior_signal",
                    0,
                ),
                "behavior_text": behavior_text,
                "true_labels": ",".join(
                    labels[row_idx]
                ),
            }
        )

    result_df = pd.DataFrame(rows)

    return {
        "technique": technique,
        "support": support,
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "threshold": threshold,
        "fn_probability_mean": (
            float(fn_probabilities.mean())
            if fn > 0
            else None
        ),
        "fn_probability_median": (
            float(np.median(fn_probabilities))
            if fn > 0
            else None
        ),
        "fn_probability_min": (
            float(fn_probabilities.min())
            if fn > 0
            else None
        ),
        "fn_probability_max": (
            float(fn_probabilities.max())
            if fn > 0
            else None
        ),
        "tp_probability_mean": (
            float(tp_probabilities.mean())
            if tp > 0
            else None
        ),
        "tp_probability_median": (
            float(np.median(tp_probabilities))
            if tp > 0
            else None
        ),
        "fn_rows": result_df,
    }


def feature_profile(
    fn_rows,
    technique,
):
    if fn_rows.empty:
        return pd.DataFrame()

    feature_columns = [
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

    for feature in feature_columns:
        values = pd.to_numeric(
            fn_rows[feature],
            errors="coerce",
        ).fillna(0)

        rows.append(
            {
                "technique": technique,
                "feature": feature,
                "fn_mean": float(values.mean()),
                "fn_active_count": int(
                    (values > 0).sum()
                ),
                "fn_active_rate": float(
                    (values > 0).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 78)
    print("SMARTSOC PHASE 2 GATE 2.4G")
    print("FALSE-NEGATIVE DEEP DIVE")
    print("=" * 78)

    (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        threshold_map,
    ) = validate_artifacts()

    classes = [
        str(x)
        for x in mlb.classes_
    ]

    print()
    print("Loading untouched test data...")
    print("-" * 78)

    structured = pd.read_csv(
        STRUCTURED_TEST
    )

    labels_df = pd.read_csv(
        LABELS_TEST
    )

    behavior = pd.read_csv(
        BEHAVIOR_TEST
    )

    print(
        f"Test rows       : {len(structured)}"
    )

    print(
        f"Label rows      : {len(labels_df)}"
    )

    print(
        f"Behavior rows   : {len(behavior)}"
    )

    if not (
        len(structured)
        == len(labels_df)
        == len(behavior)
    ):
        raise RuntimeError(
            "Test datasets are not row aligned."
        )

    label_column = (
        labels_df.columns[0]
    )

    labels = [
        parse_label(x)
        for x in labels_df[label_column]
    ]

    X = transform_test_data(
        structured,
        behavior,
        model,
        encoder,
        scaler,
        tfidf,
        schema,
    )

    print()
    print("Running production inference...")
    print("-" * 78)

    probabilities = model.predict_proba(
        X
    )

    if isinstance(
        probabilities,
        list,
    ):
        probabilities = np.column_stack(
            [
                p[:, 1]
                if np.asarray(p).ndim == 2
                else np.asarray(p)
                for p in probabilities
            ]
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    print(
        f"Probability matrix : "
        f"{probabilities.shape}"
    )

    if probabilities.shape != (
        len(structured),
        len(classes),
    ):
        raise RuntimeError(
            "Probability matrix dimensions "
            "do not match test/classes."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Probability matrix contains "
            "non-finite values."
        )

    print(
        "[PASS] Probability matrix valid"
    )

    y_true = build_truth_matrix(
        labels,
        classes,
    )

    y_pred = build_prediction_matrix(
        probabilities,
        classes,
        threshold_map,
    )

    all_summaries = []
    all_fn_rows = []
    all_profiles = []

    for technique in TARGET_TECHNIQUES:
        result = analyze_target(
            technique,
            structured,
            behavior,
            labels,
            probabilities,
            y_true,
            y_pred,
            classes,
            threshold_map,
        )

        if result is None:
            continue

        all_summaries.append(
            {
                "technique": result["technique"],
                "support": result["support"],
                "tp": result["tp"],
                "fn": result["fn"],
                "recall": result["recall"],
                "threshold": result["threshold"],
                "fn_probability_mean": result[
                    "fn_probability_mean"
                ],
                "fn_probability_median": result[
                    "fn_probability_median"
                ],
                "fn_probability_min": result[
                    "fn_probability_min"
                ],
                "fn_probability_max": result[
                    "fn_probability_max"
                ],
                "tp_probability_mean": result[
                    "tp_probability_mean"
                ],
                "tp_probability_median": result[
                    "tp_probability_median"
                ],
            }
        )

        fn_rows = result["fn_rows"]

        if not fn_rows.empty:
            all_fn_rows.append(
                fn_rows
            )

            profile = feature_profile(
                fn_rows,
                technique,
            )

            if not profile.empty:
                all_profiles.append(
                    profile
                )

    summary_df = pd.DataFrame(
        all_summaries
    )

    if all_fn_rows:
        fn_df = pd.concat(
            all_fn_rows,
            ignore_index=True,
        )
    else:
        fn_df = pd.DataFrame()

    if all_profiles:
        profile_df = pd.concat(
            all_profiles,
            ignore_index=True,
        )
    else:
        profile_df = pd.DataFrame()

    summary_path = (
        OUT_DIR
        / "false_negative_deep_dive_summary.csv"
    )

    fn_path = (
        OUT_DIR
        / "false_negative_examples.csv"
    )

    profile_path = (
        OUT_DIR
        / "false_negative_feature_profile.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    fn_df.to_csv(
        fn_path,
        index=False,
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4G",
        "gate_name": "False-Negative Deep Dive",
        "status": "ANALYSIS_COMPLETE",
        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,
        "test_rows": int(len(structured)),
        "techniques_analyzed": TARGET_TECHNIQUES,
        "summary": all_summaries,
        "outputs": {
            "summary": str(summary_path),
            "false_negative_examples": str(fn_path),
            "feature_profile": str(profile_path),
        },
    }

    report_path = (
        OUT_DIR
        / "phase2_gate2_4g_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    print()
    print("=" * 78)
    print("GATE 2.4G SUMMARY")
    print("=" * 78)

    print(
        f"Techniques analyzed : "
        f"{len(all_summaries)}"
    )

    print(
        f"FN rows analyzed    : "
        f"{len(fn_df)}"
    )

    print()
    print(
        "FALSE-NEGATIVE SUMMARY"
    )

    if not summary_df.empty:
        display_columns = [
            "technique",
            "support",
            "fn",
            "recall",
            "threshold",
            "fn_probability_mean",
            "fn_probability_median",
            "fn_probability_max",
        ]

        print(
            summary_df[
                display_columns
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 78)
    print("GATE 2.4G COMPLETE")
    print("=" * 78)

    print(
        f"Summary : {summary_path}"
    )

    print(
        f"FN data : {fn_path}"
    )

    print(
        f"Profile : {profile_path}"
    )

    print(
        f"Report  : {report_path}"
    )

    print()
    print(
        "No model, threshold, dataset, "
        "or production artifact was modified."
    )

    print()
    print(
        "STATUS: ANALYSIS_COMPLETE"
    )


if __name__ == "__main__":
    main()