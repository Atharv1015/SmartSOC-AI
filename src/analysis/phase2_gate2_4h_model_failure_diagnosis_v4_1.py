import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(r"C:\Users\Atharv\SmartSOC-AI")

DATASET = ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = ROOT / "models" / "windows_apt_v4_1" / "production"
OUT_DIR = ROOT / "reports" / "phase2_gate2_4h"

STRUCTURED_TEST = DATASET / "structured_test.csv"
LABELS_TEST = DATASET / "labels_test.csv"
BEHAVIOR_TEST = DATASET / "behavior_text_test.csv"

STRUCTURED_TRAIN = DATASET / "structured_train.csv"
LABELS_TRAIN = DATASET / "labels_train.csv"
BEHAVIOR_TRAIN = DATASET / "behavior_text_train.csv"

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "structured_encoder.joblib"
SCALER_FILE = MODEL_DIR / "numerical_scaler.joblib"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"
MLB_FILE = MODEL_DIR / "multilabel_encoder.joblib"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"
THRESHOLD_FILE = MODEL_DIR / "thresholds.json"

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

MAX_EXAMPLES_PER_TECHNIQUE = 100


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def load_artifacts():
    banner("SMARTSOC PHASE 2 GATE 2.4H")
    print("MODEL FAILURE / PROBABILITY DIAGNOSIS")
    print("=" * 78)

    required = [
        MODEL_FILE,
        ENCODER_FILE,
        SCALER_FILE,
        TFIDF_FILE,
        MLB_FILE,
        SCHEMA_FILE,
        THRESHOLD_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    tfidf = joblib.load(TFIDF_FILE)
    mlb = joblib.load(MLB_FILE)

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        thresholds_data = json.load(f)

    thresholds = thresholds_data.get("thresholds", {})

    categorical_columns = schema["categorical_columns"]
    numerical_columns = schema["numerical_columns"]

    structured_columns = categorical_columns + numerical_columns

    print()
    print("Production artifact validation")
    print("-" * 78)
    print(f"Model type           : {type(model).__name__}")
    print(f"Categorical columns  : {len(categorical_columns)}")
    print(f"Numerical columns    : {len(numerical_columns)}")
    print(f"Structured columns  : {len(structured_columns)}")
    print(f"TF-IDF vocabulary    : {len(tfidf.vocabulary_)}")
    print(f"MITRE classes        : {len(mlb.classes_)}")
    print(f"Threshold entries    : {len(thresholds)}")

    if len(structured_columns) != 62:
        raise RuntimeError("Expected 62 structured columns.")

    if len(numerical_columns) != 58:
        raise RuntimeError("Expected 58 numerical columns.")

    return (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        thresholds,
        categorical_columns,
        numerical_columns,
        structured_columns,
    )


def load_dataset(structured_path, labels_path, behavior_path):
    structured = pd.read_csv(structured_path)
    labels = pd.read_csv(labels_path)
    behavior = pd.read_csv(behavior_path)

    if len(structured) != len(labels) or len(structured) != len(behavior):
        raise RuntimeError(
            f"Row alignment failure: "
            f"structured={len(structured)}, "
            f"labels={len(labels)}, "
            f"behavior={len(behavior)}"
        )

    return structured, labels, behavior


def parse_labels(value):
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [x.strip() for x in text.split(",") if x.strip()]


def build_matrix(
    structured,
    behavior,
    encoder,
    scaler,
    tfidf,
    categorical_columns,
    numerical_columns,
):
    structured = structured.copy()

    for col in categorical_columns:
        if col not in structured.columns:
            structured[col] = ""

    for col in numerical_columns:
        if col not in structured.columns:
            structured[col] = 0.0

    structured = structured[categorical_columns + numerical_columns]

    cat_input = structured[categorical_columns].copy()

    for col in categorical_columns:
        cat_input[col] = (
            cat_input[col]
            .fillna("")
            .astype(str)
        )

    cat_matrix = encoder.transform(cat_input)

    num_input = structured[numerical_columns].copy()

    for col in numerical_columns:
        num_input[col] = pd.to_numeric(
            num_input[col],
            errors="coerce",
        ).fillna(0.0)

    num_matrix = scaler.transform(num_input)

    behavior_column = "behavior_text_normalized"

    if behavior_column not in behavior.columns:
        raise RuntimeError(
            f"Missing behavior column: {behavior_column}"
        )

    behavior_text = (
        behavior[behavior_column]
        .fillna("")
        .astype(str)
    )

    tfidf_matrix = tfidf.transform(behavior_text)

    final_matrix = np.hstack(
        [
            cat_matrix.toarray()
            if hasattr(cat_matrix, "toarray")
            else np.asarray(cat_matrix),
            np.asarray(num_matrix),
            tfidf_matrix.toarray(),
        ]
    )

    return (
        structured,
        cat_matrix,
        num_matrix,
        tfidf_matrix,
        final_matrix,
    )


def probability_statistics(probabilities):
    values = np.asarray(probabilities, dtype=float)

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "unique_rounded_6": int(
            len(np.unique(np.round(values, 6)))
        ),
    }


def calculate_feature_statistics(
    train_structured,
    test_structured,
    train_behavior,
    test_behavior,
    numerical_columns,
    categorical_columns,
):
    rows = []

    for col in numerical_columns:
        train_values = pd.to_numeric(
            train_structured[col],
            errors="coerce",
        ).fillna(0.0)

        test_values = pd.to_numeric(
            test_structured[col],
            errors="coerce",
        ).fillna(0.0)

        rows.append(
            {
                "feature": col,
                "feature_type": "numerical",
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "train_std": float(train_values.std()),
                "test_std": float(test_values.std()),
                "mean_difference": float(
                    test_values.mean() - train_values.mean()
                ),
            }
        )

    for col in categorical_columns:
        train_values = (
            train_structured[col]
            .fillna("")
            .astype(str)
        )

        test_values = (
            test_structured[col]
            .fillna("")
            .astype(str)
        )

        train_mode = (
            train_values.mode().iloc[0]
            if not train_values.mode().empty
            else ""
        )

        test_mode = (
            test_values.mode().iloc[0]
            if not test_values.mode().empty
            else ""
        )

        rows.append(
            {
                "feature": col,
                "feature_type": "categorical",
                "train_mean": np.nan,
                "test_mean": np.nan,
                "train_std": np.nan,
                "test_std": np.nan,
                "mean_difference": np.nan,
                "train_mode": train_mode,
                "test_mode": test_mode,
            }
        )

    behavior_column = "behavior_text_normalized"

    train_lengths = (
        train_behavior[behavior_column]
        .fillna("")
        .astype(str)
        .str.len()
    )

    test_lengths = (
        test_behavior[behavior_column]
        .fillna("")
        .astype(str)
        .str.len()
    )

    rows.append(
        {
            "feature": "behavior_text_length",
            "feature_type": "text",
            "train_mean": float(train_lengths.mean()),
            "test_mean": float(test_lengths.mean()),
            "train_std": float(train_lengths.std()),
            "test_std": float(test_lengths.std()),
            "mean_difference": float(
                test_lengths.mean() - train_lengths.mean()
            ),
        }
    )

    return pd.DataFrame(rows)


def extract_classifier_coefficients(model):
    """
    Extract coefficient information from the OneVsRest LogisticRegression model.
    """
    estimator = getattr(model, "estimator", None)

    if estimator is None:
        return None

    if not hasattr(estimator, "coef_"):
        return None

    return estimator.coef_


def feature_names(
    encoder,
    numerical_columns,
    tfidf,
):
    names = []

    try:
        names.extend(
            encoder.get_feature_names_out()
        )
    except Exception:
        names.extend(
            [f"categorical_{i}" for i in range(162)]
        )

    names.extend(
        [f"NUM::{x}" for x in numerical_columns]
    )

    tfidf_names = tfidf.get_feature_names_out()

    names.extend(
        [f"TFIDF::{x}" for x in tfidf_names]
    )

    return np.asarray(names, dtype=object)


def analyze_coefficients(
    model,
    mlb,
    names,
    target_techniques,
):
    coef = extract_classifier_coefficients(model)

    if coef is None:
        return pd.DataFrame()

    classes = list(mlb.classes_)

    rows = []

    for technique in target_techniques:
        if technique not in classes:
            continue

        class_index = classes.index(technique)

        if class_index >= coef.shape[0]:
            continue

        class_coef = coef[class_index]

        positive_indices = np.argsort(
            class_coef
        )[-15:][::-1]

        negative_indices = np.argsort(
            class_coef
        )[:15]

        for rank, idx in enumerate(positive_indices, start=1):
            rows.append(
                {
                    "technique": technique,
                    "direction": "positive",
                    "rank": rank,
                    "feature": str(names[idx]),
                    "coefficient": float(class_coef[idx]),
                }
            )

        for rank, idx in enumerate(negative_indices, start=1):
            rows.append(
                {
                    "technique": technique,
                    "direction": "negative",
                    "rank": rank,
                    "feature": str(names[idx]),
                    "coefficient": float(class_coef[idx]),
                }
            )

    return pd.DataFrame(rows)


def evaluate():
    (
        model,
        encoder,
        scaler,
        tfidf,
        mlb,
        schema,
        thresholds,
        categorical_columns,
        numerical_columns,
        structured_columns,
    ) = load_artifacts()

    banner("LOADING TRAIN / TEST DATA")

    train_structured, train_labels, train_behavior = load_dataset(
        STRUCTURED_TRAIN,
        LABELS_TRAIN,
        BEHAVIOR_TRAIN,
    )

    test_structured, test_labels, test_behavior = load_dataset(
        STRUCTURED_TEST,
        LABELS_TEST,
        BEHAVIOR_TEST,
    )

    print(f"Training rows : {len(train_structured)}")
    print(f"Test rows     : {len(test_structured)}")

    banner("TRANSFORMING TRAIN / TEST")

    (
        train_structured_clean,
        train_cat,
        train_num,
        train_tfidf,
        train_X,
    ) = build_matrix(
        train_structured,
        train_behavior,
        encoder,
        scaler,
        tfidf,
        categorical_columns,
        numerical_columns,
    )

    (
        test_structured_clean,
        test_cat,
        test_num,
        test_tfidf,
        test_X,
    ) = build_matrix(
        test_structured,
        test_behavior,
        encoder,
        scaler,
        tfidf,
        categorical_columns,
        numerical_columns,
    )

    print(f"Train final matrix : {train_X.shape}")
    print(f"Test final matrix  : {test_X.shape}")

    if train_X.shape[1] != 6919:
        raise RuntimeError(
            f"Unexpected train feature dimension: {train_X.shape[1]}"
        )

    if test_X.shape[1] != 6919:
        raise RuntimeError(
            f"Unexpected test feature dimension: {test_X.shape[1]}"
        )

    print("[PASS] Feature dimensions = 6919")

    banner("PRODUCTION INFERENCE")

    probabilities = model.predict_proba(test_X)

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Probability matrix contains non-finite values.")

    if np.min(probabilities) < 0 or np.max(probabilities) > 1:
        raise RuntimeError("Probability values outside [0,1].")

    print(f"Probability matrix : {probabilities.shape}")
    print("[PASS] Probability matrix valid")

    technique_to_index = {
        technique: index
        for index, technique in enumerate(mlb.classes_)
    }

    test_label_lists = [
        parse_labels(x)
        for x in test_labels["techniques"]
    ]

    summary_rows = []
    example_rows = []

    banner("TARGET TECHNIQUE DIAGNOSIS")

    for technique in TARGET_TECHNIQUES:

        if technique not in technique_to_index:
            print(f"{technique}: NOT FOUND")
            continue

        idx = technique_to_index[technique]

        true_mask = np.array(
            [
                technique in labels
                for labels in test_label_lists
            ],
            dtype=bool,
        )

        false_mask = ~true_mask

        target_prob = probabilities[:, idx]

        tp_mask = true_mask & (
            target_prob >= thresholds.get(
                technique,
                0.5,
            )
        )

        fn_mask = true_mask & ~(
            target_prob >= thresholds.get(
                technique,
                0.5,
            )
        )

        fp_mask = false_mask & (
            target_prob >= thresholds.get(
                technique,
                0.5,
            )
        )

        threshold = float(
            thresholds.get(
                technique,
                0.5,
            )
        )

        print()
        print(f"{technique}")
        print("-" * 78)
        print(f"Support             : {int(true_mask.sum())}")
        print(f"TP                  : {int(tp_mask.sum())}")
        print(f"FN                  : {int(fn_mask.sum())}")
        print(f"FP                  : {int(fp_mask.sum())}")
        print(
            f"Threshold           : {threshold:.4f}"
        )

        print(
            f"All-test probability: "
            f"mean={target_prob.mean():.6f} "
            f"median={np.median(target_prob):.6f} "
            f"std={target_prob.std():.6f}"
        )

        print(
            f"True-label P       : "
            f"mean={target_prob[true_mask].mean():.6f} "
            f"median={np.median(target_prob[true_mask]):.6f}"
        )

        print(
            f"False-label P      : "
            f"mean={target_prob[false_mask].mean():.6f} "
            f"median={np.median(target_prob[false_mask]):.6f}"
        )

        print(
            f"FN P range         : "
            f"{target_prob[fn_mask].min() if fn_mask.any() else np.nan:.6f}"
            f" - "
            f"{target_prob[fn_mask].max() if fn_mask.any() else np.nan:.6f}"
        )

        print(
            f"TP P range         : "
            f"{target_prob[tp_mask].min() if tp_mask.any() else np.nan:.6f}"
            f" - "
            f"{target_prob[tp_mask].max() if tp_mask.any() else np.nan:.6f}"
        )

        stats = probability_statistics(
            target_prob[true_mask]
            if true_mask.any()
            else target_prob
        )

        summary_rows.append(
            {
                "technique": technique,
                "support": int(true_mask.sum()),
                "tp": int(tp_mask.sum()),
                "fn": int(fn_mask.sum()),
                "fp": int(fp_mask.sum()),
                "threshold": threshold,
                "recall": float(
                    tp_mask.sum() / true_mask.sum()
                )
                if true_mask.sum() > 0
                else 0.0,
                "true_probability_mean": float(
                    target_prob[true_mask].mean()
                )
                if true_mask.any()
                else np.nan,
                "true_probability_median": float(
                    np.median(target_prob[true_mask])
                )
                if true_mask.any()
                else np.nan,
                "true_probability_std": float(
                    target_prob[true_mask].std()
                )
                if true_mask.any()
                else np.nan,
                "true_probability_unique_rounded_6": stats[
                    "unique_rounded_6"
                ],
                "fn_probability_mean": float(
                    target_prob[fn_mask].mean()
                )
                if fn_mask.any()
                else np.nan,
                "fn_probability_max": float(
                    target_prob[fn_mask].max()
                )
                if fn_mask.any()
                else np.nan,
                "fn_probability_min": float(
                    target_prob[fn_mask].min()
                )
                if fn_mask.any()
                else np.nan,
            }
        )

        fn_indices = np.where(fn_mask)[0]

        if len(fn_indices) > MAX_EXAMPLES_PER_TECHNIQUE:
            fn_indices = fn_indices[
                np.argsort(
                    target_prob[fn_indices]
                )[:MAX_EXAMPLES_PER_TECHNIQUE]
            ]

        for row_index in fn_indices:
            example_rows.append(
                {
                    "test_index": int(row_index),
                    "technique": technique,
                    "probability": float(
                        target_prob[row_index]
                    ),
                    "threshold": threshold,
                    "process_name": test_structured.iloc[
                        row_index
                    ].get("process_name", ""),
                    "parent_process_name": test_structured.iloc[
                        row_index
                    ].get("parent_process_name", ""),
                    "command_length": test_structured.iloc[
                        row_index
                    ].get("command_length", 0),
                    "uses_powershell": test_structured.iloc[
                        row_index
                    ].get("uses_powershell", 0),
                    "uses_cmd": test_structured.iloc[
                        row_index
                    ].get("uses_cmd", 0),
                    "uses_net": test_structured.iloc[
                        row_index
                    ].get("uses_net", 0),
                    "uses_reg": test_structured.iloc[
                        row_index
                    ].get("uses_reg", 0),
                    "has_encoded_command": test_structured.iloc[
                        row_index
                    ].get("has_encoded_command", 0),
                    "has_download_behavior": test_structured.iloc[
                        row_index
                    ].get("has_download_behavior", 0),
                    "has_registry_operation": test_structured.iloc[
                        row_index
                    ].get("has_registry_operation", 0),
                    "has_file_deletion": test_structured.iloc[
                        row_index
                    ].get("has_file_deletion", 0),
                    "has_obfuscation_hint": test_structured.iloc[
                        row_index
                    ].get("has_obfuscation_hint", 0),
                    "administrative_powershell_context": test_structured.iloc[
                        row_index
                    ].get(
                        "administrative_powershell_context",
                        0,
                    ),
                    "strong_behavior_signal": test_structured.iloc[
                        row_index
                    ].get(
                        "strong_behavior_signal",
                        0,
                    ),
                    "behavior_text": str(
                        test_behavior.iloc[
                            row_index
                        ]["behavior_text_normalized"]
                    ),
                    "true_labels": str(
                        test_labels.iloc[
                            row_index
                        ]["techniques"]
                    ),
                }
            )

    banner("TRAIN / TEST FEATURE DISTRIBUTION")

    feature_profile = calculate_feature_statistics(
        train_structured_clean,
        test_structured_clean,
        train_behavior,
        test_behavior,
        numerical_columns,
        categorical_columns,
    )

    print(
        feature_profile.sort_values(
            "mean_difference",
            key=lambda x: x.abs()
            if x.notna().any()
            else x,
            ascending=False,
        ).head(20).to_string(index=False)
    )

    banner("MODEL COEFFICIENT ANALYSIS")

    names = feature_names(
        encoder,
        numerical_columns,
        tfidf,
    )

    coefficient_df = analyze_coefficients(
        model,
        mlb,
        names,
        TARGET_TECHNIQUES,
    )

    if coefficient_df.empty:
        print(
            "[WARNING] LogisticRegression coefficients unavailable."
        )
    else:
        for technique in TARGET_TECHNIQUES:
            subset = coefficient_df[
                coefficient_df["technique"] == technique
            ]

            if subset.empty:
                continue

            print()
            print(f"{technique}")
            print("-" * 78)

            print("Top positive features:")
            print(
                subset[
                    subset["direction"] == "positive"
                ][
                    [
                        "rank",
                        "feature",
                        "coefficient",
                    ]
                ].head(10).to_string(index=False)
            )

            print("Top negative features:")
            print(
                subset[
                    subset["direction"] == "negative"
                ][
                    [
                        "rank",
                        "feature",
                        "coefficient",
                    ]
                ].head(10).to_string(index=False)
            )

    banner("DIAGNOSTIC INTERPRETATION")

    summary_df = pd.DataFrame(summary_rows)

    for _, row in summary_df.iterrows():

        technique = row["technique"]
        support = int(row["support"])
        recall = float(row["recall"])
        true_mean = float(row["true_probability_mean"])
        true_std = float(row["true_probability_std"])

        if support == 0:
            diagnosis = "NO_TEST_SUPPORT"

        elif recall == 0 and true_std < 1e-4:
            diagnosis = "DEGENERATE_CONSTANT_LOW_PROBABILITY"

        elif recall == 0:
            diagnosis = "SEVERE_CLASS_SEPARATION_FAILURE"

        elif recall < 0.25:
            diagnosis = "SEVERE_LOW_RECALL"

        elif recall < 0.50:
            diagnosis = "MODERATE_LOW_RECALL"

        elif recall < 0.80:
            diagnosis = "PARTIAL_RECALL"

        else:
            diagnosis = "RELATIVELY_HEALTHY"

        print(
            f"{technique:12s} "
            f"support={support:5d} "
            f"recall={recall:.4f} "
            f"true_mean={true_mean:.4f} "
            f"true_std={true_std:.6f} "
            f"=> {diagnosis}"
        )

        summary_df.loc[
            summary_df["technique"] == technique,
            "diagnosis",
        ] = diagnosis

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUT_DIR /
        "phase2_gate2_4h_technique_diagnosis.csv"
    )

    examples_path = (
        OUT_DIR /
        "phase2_gate2_4h_false_negative_examples.csv"
    )

    feature_path = (
        OUT_DIR /
        "phase2_gate2_4h_train_test_feature_profile.csv"
    )

    coefficient_path = (
        OUT_DIR /
        "phase2_gate2_4h_model_coefficients.csv"
    )

    report_path = (
        OUT_DIR /
        "phase2_gate2_4h_report.json"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(example_rows).to_csv(
        examples_path,
        index=False,
    )

    feature_profile.to_csv(
        feature_path,
        index=False,
    )

    coefficient_df.to_csv(
        coefficient_path,
        index=False,
    )

    report = {
        "phase": "Phase 2",
        "gate": "Gate 2.4H",
        "gate_name": "Model Failure / Probability Diagnosis",
        "status": "ANALYSIS_COMPLETE",
        "production_artifacts_modified": False,
        "model_retrained": False,
        "thresholds_modified": False,
        "datasets_modified": False,
        "test_rows": int(len(test_structured)),
        "train_rows": int(len(train_structured)),
        "feature_count": int(test_X.shape[1]),
        "techniques_analyzed": TARGET_TECHNIQUES,
        "diagnosis_counts": summary_df[
            "diagnosis"
        ].value_counts().to_dict(),
        "outputs": {
            "technique_diagnosis": str(summary_path),
            "false_negative_examples": str(examples_path),
            "train_test_feature_profile": str(feature_path),
            "model_coefficients": str(coefficient_path),
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    banner("GATE 2.4H COMPLETE")

    print(f"Diagnosis : {summary_path}")
    print(f"FN data   : {examples_path}")
    print(f"Features  : {feature_path}")
    print(f"Coeffs    : {coefficient_path}")
    print(f"Report    : {report_path}")

    print()
    print(
        "No model, threshold, dataset, or production artifact was modified."
    )
    print()
    print("STATUS: ANALYSIS_COMPLETE")


if __name__ == "__main__":
    evaluate()