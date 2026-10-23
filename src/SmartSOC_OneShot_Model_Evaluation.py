"""
SmartSOC Windows-APT v4.1 - One-Shot Evaluation
================================================

Computes:
1. Test-set accuracy (subset/exact-match + label-wise accuracy)
2. Micro precision / recall / F1
3. Macro precision / recall / F1
4. Weighted precision / recall / F1
5. Hamming loss
6. 5-fold cross-validation mean +/- std for:
   - subset accuracy
   - label-wise accuracy
   - micro precision / recall / F1
   - macro precision / recall / F1

The test-set evaluation uses the saved v4.1 production artifacts when available.
Cross-validation retrains the v4.1 hybrid One-vs-Rest Logistic Regression baseline
on the training partition only, with preprocessing fitted inside each fold to avoid
data leakage.

Expected project layout:
SmartSOC-AI/
  data/processed/windows_apt_v4_1/
    structured_train.csv
    structured_test.csv
    behavior_text_train.csv
    behavior_text_test.csv
    labels_train.csv
    labels_test.csv
  models/windows_apt_v4_1/production/
    model.joblib
    structured_encoder.joblib
    numerical_scaler.joblib
    tfidf_vectorizer.joblib
    multilabel_encoder.joblib
    feature_schema.json
    thresholds.json

If production artifacts are not found, the script can still run CV and will
evaluate the saved predictions.csv if it contains usable y_true/y_pred columns.
"""

from pathlib import Path
import json
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
    classification_report,
)
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42
MAX_ITER = 1500
TFIDF_MAX_FEATURES = 6699
MIN_DF = 1

CAT_COLUMNS = [
    "process_name",
    "parent_process_name",
    "process_extension",
    "parent_extension",
]

EXPECTED_TECHNIQUES = 45

# ============================================================
# PATH DISCOVERY
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

# If this script is saved inside src/, move one level up.
if PROJECT_ROOT.name.lower() == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "windows_apt_v4_1"
MODEL_DIR = PROJECT_ROOT / "models" / "windows_apt_v4_1"
PROD_DIR = MODEL_DIR / "production"
RESULT_DIR = PROJECT_ROOT / "results" / "windows_apt_v4_1"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def stop(msg):
    print("\n[ERROR] " + msg)
    sys.exit(1)


def find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def require_files():
    required = {
        "structured_train": DATA_DIR / "structured_train.csv",
        "structured_test": DATA_DIR / "structured_test.csv",
        "text_train": DATA_DIR / "behavior_text_train.csv",
        "text_test": DATA_DIR / "behavior_text_test.csv",
        "labels_train": DATA_DIR / "labels_train.csv",
        "labels_test": DATA_DIR / "labels_test.csv",
    }

    missing = [str(p) for p in required.values() if not p.exists()]

    if missing:
        stop(
            "Missing required files:\n  - "
            + "\n  - ".join(missing)
            + "\n\nExpected them under:\n"
            + str(DATA_DIR)
        )

    return required


def load_text(path):
    df = pd.read_csv(path)

    if len(df.columns) == 0:
        stop(f"No columns found in {path}")

    # Existing v4.1 behavior_text files contain one text column.
    if len(df.columns) == 1:
        return df.iloc[:, 0].fillna("").astype(str)

    # Robust fallback: select the likely text column.
    preferred = [
        "behavior_text",
        "command_line",
        "text",
        "behavior",
    ]

    for col in preferred:
        if col in df.columns:
            return df[col].fillna("").astype(str)

    # Last fallback: join all columns.
    return (
        df.fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )


def extract_techniques(value):
    """
    Extract MITRE ATT&CK technique IDs from a label cell.

    Handles:
      T1105
      T1059.003
      comma/semicolon/pipe/newline separated labels
      Python-list-like strings
    """
    if pd.isna(value):
        return []

    text = str(value)

    found = re.findall(r"T\d{4}(?:\.\d{3})?", text.upper())

    # Preserve order and remove duplicates.
    result = []
    seen = set()

    for x in found:
        if x not in seen:
            result.append(x)
            seen.add(x)

    return result


def load_labels(path):
    df = pd.read_csv(path)

    if "techniques" in df.columns:
        col = "techniques"
    elif len(df.columns) == 1:
        col = df.columns[0]
    else:
        stop(
            f"Could not identify the label column in {path}. "
            f"Columns: {list(df.columns)}"
        )

    labels = df[col].apply(extract_techniques).tolist()

    return labels


def build_label_matrix(train_labels, test_labels):
    all_labels = sorted(
        set(x for row in train_labels + test_labels for x in row)
    )

    if len(all_labels) != EXPECTED_TECHNIQUES:
        print(
            f"[WARNING] Found {len(all_labels)} MITRE techniques; "
            f"expected {EXPECTED_TECHNIQUES}."
        )

    mlb = MultiLabelBinarizer(classes=all_labels)

    y_train = mlb.fit_transform(train_labels)
    y_test = mlb.transform(test_labels)

    return mlb, y_train.astype(np.int8), y_test.astype(np.int8)


def get_feature_columns(structured_df):
    missing = [c for c in CAT_COLUMNS if c not in structured_df.columns]

    if missing:
        stop(
            "Missing expected categorical columns: "
            + ", ".join(missing)
        )

    numeric_columns = [
        c for c in structured_df.columns
        if c not in CAT_COLUMNS
    ]

    return CAT_COLUMNS, numeric_columns


def make_preprocessor(cat_cols, num_cols):
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        # Compatibility with older sklearn.
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    scaler = StandardScaler()

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=MIN_DF,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    return encoder, scaler, tfidf


def build_features(
    train_struct,
    valid_struct,
    train_text,
    valid_text,
    cat_cols,
    num_cols,
):
    encoder, scaler, tfidf = make_preprocessor(cat_cols, num_cols)

    # --------------------------
    # Categorical
    # --------------------------
    Xcat_train = encoder.fit_transform(
        train_struct[cat_cols].fillna("").astype(str)
    )

    Xcat_valid = encoder.transform(
        valid_struct[cat_cols].fillna("").astype(str)
    )

    # --------------------------
    # Numerical
    # --------------------------
    Xnum_train_raw = (
        train_struct[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .values
    )

    Xnum_valid_raw = (
        valid_struct[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .values
    )

    Xnum_train = scaler.fit_transform(Xnum_train_raw)
    Xnum_valid = scaler.transform(Xnum_valid_raw)

    # Sparse conversion allows safe hstack with TF-IDF.
    Xnum_train = csr_matrix(Xnum_train)
    Xnum_valid = csr_matrix(Xnum_valid)

    # --------------------------
    # TF-IDF
    # --------------------------
    Xtext_train = tfidf.fit_transform(
        train_text.fillna("").astype(str)
    )

    Xtext_valid = tfidf.transform(
        valid_text.fillna("").astype(str)
    )

    # --------------------------
    # Hybrid matrix
    # --------------------------
    X_train = hstack(
        [Xcat_train, Xnum_train, Xtext_train],
        format="csr",
    )

    X_valid = hstack(
        [Xcat_valid, Xnum_valid, Xtext_valid],
        format="csr",
    )

    return X_train, X_valid, encoder, scaler, tfidf


def train_ovr_model(X_train, y_train):
    """
    Train 45 independent binary logistic regressors.

    This explicitly handles rare labels where a CV training fold may contain
    only one class. In that case, the classifier becomes a constant predictor.
    """

    models = []

    for j in range(y_train.shape[1]):
        y = y_train[:, j]

        unique = np.unique(y)

        if len(unique) == 1:
            models.append(("constant", int(unique[0])))
            continue

        model = LogisticRegression(
            solver="liblinear",
            max_iter=MAX_ITER,
            class_weight=None,
            random_state=RANDOM_STATE,
        )

        model.fit(X_train, y)
        models.append(("model", model))

    return models


def predict_ovr(models, X):
    predictions = np.zeros((X.shape[0], len(models)), dtype=np.int8)

    for j, item in enumerate(models):
        kind, model = item

        if kind == "constant":
            predictions[:, j] = model
        else:
            predictions[:, j] = model.predict(X)

    return predictions


def calculate_metrics(y_true, y_pred):
    """
    For multilabel classification:
      - subset accuracy = exact event-level match
      - label accuracy = element-wise binary accuracy
    """

    metrics = {
        "subset_accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "label_accuracy": float(
            np.mean(y_true == y_pred)
        ),
        "micro_precision": float(
            precision_score(
                y_true, y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_recall": float(
            recall_score(
                y_true, y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y_true, y_pred,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "hamming_loss": float(
            hamming_loss(y_true, y_pred)
        ),
    }

    return metrics


def print_metrics(title, metrics):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    print(
        f"Subset Accuracy      : {metrics['subset_accuracy']:.4f} "
        f"({metrics['subset_accuracy'] * 100:.2f}%)"
    )

    print(
        f"Label-wise Accuracy  : {metrics['label_accuracy']:.4f} "
        f"({metrics['label_accuracy'] * 100:.2f}%)"
    )

    print(
        f"Micro Precision      : {metrics['micro_precision']:.4f}"
    )
    print(
        f"Micro Recall         : {metrics['micro_recall']:.4f}"
    )
    print(
        f"Micro F1             : {metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro Precision      : {metrics['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall         : {metrics['macro_recall']:.4f}"
    )
    print(
        f"Macro F1             : {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted Precision   : {metrics['weighted_precision']:.4f}"
    )
    print(
        f"Weighted Recall      : {metrics['weighted_recall']:.4f}"
    )
    print(
        f"Weighted F1          : {metrics['weighted_f1']:.4f}"
    )

    print(
        f"Hamming Loss        : {metrics['hamming_loss']:.4f}"
    )


def try_saved_predictions():
    """
    Optional shortcut: if predictions.csv contains recognizable true/predicted
    technique columns, calculate metrics from it.

    This does not replace the production-model evaluation below.
    """
    pred_path = RESULT_DIR / "predictions.csv"

    if not pred_path.exists():
        return None

    try:
        df = pd.read_csv(pred_path)

        cols = {c.lower(): c for c in df.columns}

        true_candidates = [
            "y_true",
            "true",
            "actual",
            "actual_techniques",
            "true_techniques",
            "ground_truth",
        ]

        pred_candidates = [
            "y_pred",
            "pred",
            "prediction",
            "predicted",
            "predicted_techniques",
        ]

        true_col = next(
            (cols[x] for x in true_candidates if x in cols),
            None,
        )

        pred_col = next(
            (cols[x] for x in pred_candidates if x in cols),
            None,
        )

        if true_col is None or pred_col is None:
            return None

        true_lists = df[true_col].apply(extract_techniques).tolist()
        pred_lists = df[pred_col].apply(extract_techniques).tolist()

        classes = sorted(
            set(x for row in true_lists + pred_lists for x in row)
        )

        mlb = MultiLabelBinarizer(classes=classes)

        y_true = mlb.fit_transform(true_lists)
        y_pred = mlb.transform(pred_lists)

        return calculate_metrics(y_true, y_pred)

    except Exception as exc:
        print(
            f"[WARNING] Could not evaluate saved predictions.csv: {exc}"
        )
        return None


# ============================================================
# TEST-SET EVALUATION USING SAVED PRODUCTION ARTIFACTS
# ============================================================

def evaluate_saved_production(
    structured_test,
    text_test,
    y_test,
    mlb,
):
    model_path = PROD_DIR / "model.joblib"
    encoder_path = PROD_DIR / "structured_encoder.joblib"
    scaler_path = PROD_DIR / "numerical_scaler.joblib"
    tfidf_path = PROD_DIR / "tfidf_vectorizer.joblib"
    label_encoder_path = PROD_DIR / "multilabel_encoder.joblib"
    schema_path = PROD_DIR / "feature_schema.json"
    thresholds_path = PROD_DIR / "thresholds.json"

    required = [
        model_path,
        encoder_path,
        scaler_path,
        tfidf_path,
        label_encoder_path,
    ]

    if not all(p.exists() for p in required):
        return None

    try:
        import joblib

        print("\nLoading saved v4.1 production artifacts...")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        tfidf = joblib.load(tfidf_path)
        saved_mlb = joblib.load(label_encoder_path)

        # --------------------------
        # Use saved schema if present
        # --------------------------
        if schema_path.exists():
            try:
                schema = json.loads(
                    schema_path.read_text(encoding="utf-8")
                )

                cat_cols = schema.get(
                    "categorical_features",
                    CAT_COLUMNS,
                )

                num_cols = schema.get(
                    "numerical_features",
                    [
                        c for c in structured_test.columns
                        if c not in cat_cols
                    ],
                )
            except Exception:
                cat_cols, num_cols = get_feature_columns(
                    structured_test
                )
        else:
            cat_cols, num_cols = get_feature_columns(
                structured_test
            )

        # --------------------------
        # Build test matrix
        # --------------------------
        Xcat = encoder.transform(
            structured_test[cat_cols].fillna("").astype(str)
        )

        Xnum_raw = (
            structured_test[num_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .astype(np.float32)
            .values
        )

        Xnum = scaler.transform(Xnum_raw)
        Xnum = csr_matrix(Xnum)

        Xtext = tfidf.transform(
            text_test.fillna("").astype(str)
        )

        X = hstack(
            [Xcat, Xnum, Xtext],
            format="csr",
        )

        print(f"Production test matrix: {X.shape}")

        # --------------------------
        # Predictions
        # --------------------------
        if hasattr(model, "predict"):
            raw_pred = model.predict(X)

            if hasattr(raw_pred, "toarray"):
                raw_pred = raw_pred.toarray()

            raw_pred = np.asarray(raw_pred)

        else:
            raise RuntimeError("Production model has no predict() method.")

        # Align label count/order.
        saved_classes = list(
            getattr(saved_mlb, "classes_", mlb.classes_)
        )

        if raw_pred.shape[1] != len(saved_classes):
            raise RuntimeError(
                "Prediction label count does not match saved label encoder."
            )

        y_pred = np.zeros(
            (len(structured_test), len(mlb.classes_)),
            dtype=np.int8,
        )

        class_to_index = {
            c: i for i, c in enumerate(mlb.classes_)
        }

        for j, technique in enumerate(saved_classes):
            if technique in class_to_index:
                y_pred[:, class_to_index[technique]] = (
                    raw_pred[:, j] > 0
                ).astype(np.int8)

        # --------------------------
        # Optional thresholding
        # --------------------------
        # If the saved production model itself already contains calibrated
        # decision behavior, its predict() result is used as-is. We do not
        # silently invent new thresholds.
        #
        # thresholds.json is reported for auditability but is not applied
        # here unless the model exposes probabilities and the file is clearly
        # aligned.

        metrics = calculate_metrics(y_test, y_pred)

        return {
            "metrics": metrics,
            "y_pred": y_pred,
            "source": "saved production model",
        }

    except Exception as exc:
        print(
            "\n[WARNING] Production artifact evaluation failed:"
        )
        print(str(exc))
        print(
            "The script will continue with cross-validation."
        )
        return None


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    structured_train,
    text_train,
    y_train,
    cat_cols,
    num_cols,
):
    print("\n")
    print("=" * 72)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 72)

    print(
        f"Samples: {len(structured_train)} | "
        f"Labels: {y_train.shape[1]} | "
        f"Folds: {N_SPLITS}"
    )

    print(
        "\nIMPORTANT: preprocessing is fitted separately inside "
        "each fold to prevent data leakage."
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold_no, (train_idx, valid_idx) in enumerate(
        kfold.split(structured_train),
        start=1,
    ):
        start = time.time()

        print("\n" + "-" * 72)
        print(f"FOLD {fold_no}/{N_SPLITS}")
        print("-" * 72)

        fold_struct_train = structured_train.iloc[train_idx]
        fold_struct_valid = structured_train.iloc[valid_idx]

        fold_text_train = text_train.iloc[train_idx]
        fold_text_valid = text_train.iloc[valid_idx]

        fold_y_train = y_train[train_idx]
        fold_y_valid = y_train[valid_idx]

        print(
            f"Training rows: {len(train_idx)} | "
            f"Validation rows: {len(valid_idx)}"
        )

        # Feature construction
        X_train, X_valid, _, _, _ = build_features(
            fold_struct_train,
            fold_struct_valid,
            fold_text_train,
            fold_text_valid,
            cat_cols,
            num_cols,
        )

        print(
            f"Feature matrix: train={X_train.shape}, "
            f"valid={X_valid.shape}"
        )

        # Model
        models = train_ovr_model(
            X_train,
            fold_y_train,
        )

        # Predict
        fold_pred = predict_ovr(
            models,
            X_valid,
        )

        # Metrics
        metrics = calculate_metrics(
            fold_y_valid,
            fold_pred,
        )

        metrics["fold"] = fold_no
        metrics["seconds"] = round(
            time.time() - start,
            2,
        )

        fold_results.append(metrics)

        print_metrics(
            f"FOLD {fold_no} RESULTS",
            metrics,
        )

    return pd.DataFrame(fold_results)


def summarize_cv(fold_df):
    metric_columns = [
        "subset_accuracy",
        "label_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "hamming_loss",
    ]

    summary = []

    for metric in metric_columns:
        values = fold_df[metric].astype(float)

        summary.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "mean_percent": values.mean() * 100,
                "std_percent": values.std(ddof=1) * 100,
            }
        )

    return pd.DataFrame(summary)


# ============================================================
# MAIN
# ============================================================

def main():
    total_start = time.time()

    print("=" * 72)
    print("SmartSOC Windows-APT v4.1")
    print("ONE-SHOT MODEL EVALUATION")
    print("=" * 72)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")

    files = require_files()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    structured_train = pd.read_csv(files["structured_train"])
    structured_test = pd.read_csv(files["structured_test"])

    text_train = load_text(files["text_train"])
    text_test = load_text(files["text_test"])

    train_labels = load_labels(files["labels_train"])
    test_labels = load_labels(files["labels_test"])

    # Alignment
    if not (
        len(structured_train)
        == len(text_train)
        == len(train_labels)
    ):
        stop("Training datasets are not row-aligned.")

    if not (
        len(structured_test)
        == len(text_test)
        == len(test_labels)
    ):
        stop("Testing datasets are not row-aligned.")

    print(f"Training events: {len(structured_train)}")
    print(f"Testing events : {len(structured_test)}")
    print(f"Structured features: {structured_train.shape[1]}")

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    mlb, y_train, y_test = build_label_matrix(
        train_labels,
        test_labels,
    )

    print(f"MITRE techniques: {len(mlb.classes_)}")
    print(f"Y train shape: {y_train.shape}")
    print(f"Y test shape : {y_test.shape}")

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    cat_cols, num_cols = get_feature_columns(
        structured_train
    )

    print(f"Categorical features: {len(cat_cols)}")
    print(f"Numerical features  : {len(num_cols)}")

    # --------------------------------------------------------
    # TEST-SET METRICS
    # --------------------------------------------------------

    production_result = evaluate_saved_production(
        structured_test,
        text_test,
        y_test,
        mlb,
    )

    if production_result is not None:
        test_metrics = production_result["metrics"]

        print_metrics(
            "FINAL TEST-SET METRICS — SAVED v4.1 PRODUCTION MODEL",
            test_metrics,
        )

    else:
        print(
            "\n[WARNING] Saved production model could not be evaluated."
        )

        saved_metrics = try_saved_predictions()

        if saved_metrics is not None:
            print_metrics(
                "TEST METRICS — SAVED predictions.csv",
                saved_metrics,
            )

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    cv_folds = run_cross_validation(
        structured_train,
        text_train,
        y_train,
        cat_cols,
        num_cols,
    )

    cv_summary = summarize_cv(cv_folds)

    # --------------------------------------------------------
    # PRINT FINAL CV SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("FINAL 5-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 72)

    for _, row in cv_summary.iterrows():
        print(
            f"{row['metric']:22s}: "
            f"{row['mean']:.4f} ± {row['std']:.4f} "
            f"({row['mean_percent']:.2f}% ± "
            f"{row['std_percent']:.2f}%)"
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    fold_path = RESULT_DIR / "evaluation_5fold_results.csv"
    summary_path = RESULT_DIR / "evaluation_5fold_summary.csv"
    json_path = RESULT_DIR / "evaluation_final.json"

    cv_folds.to_csv(
        fold_path,
        index=False,
    )

    cv_summary.to_csv(
        summary_path,
        index=False,
    )

    final_output = {
        "project": "SmartSOC Windows-APT v4.1",
        "test_set": {
            "events": int(len(structured_test)),
            "techniques": int(len(mlb.classes_)),
        },
        "cross_validation": {
            "folds": N_SPLITS,
            "random_state": RANDOM_STATE,
            "results": cv_summary.to_dict(
                orient="records"
            ),
        },
    }

    if production_result is not None:
        final_output["test_metrics"] = test_metrics

    json_path.write_text(
        json.dumps(
            final_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    elapsed = time.time() - total_start

    print("\n")
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    print(f"Total runtime: {elapsed / 60:.2f} minutes")

    print("\nSaved files:")
    print(f"  1. {fold_path}")
    print(f"  2. {summary_path}")
    print(f"  3. {json_path}")

    print("\nUse the FINAL TEST-SET metrics for the held-out test result.")
    print(
        "Use the 5-FOLD mean ± std values to report "
        "cross-validation robustness."
    )

    print("\nRecommended paper reporting:")
    print(
        "Report micro F1, macro F1, precision, recall, "
        "subset accuracy, and 5-fold mean ± std."
    )


if __name__ == "__main__":
    main()
