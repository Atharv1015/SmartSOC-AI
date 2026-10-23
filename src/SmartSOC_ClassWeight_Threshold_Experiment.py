
"""
SmartSOC Windows-APT v4.1
Class-Weighted + OOF Threshold Optimization Experiment

IMPORTANT:
- Does NOT modify production model artifacts.
- Uses only training data for threshold optimization.
- Uses the untouched test set only for final evaluation.
- Reuses the exact v4.1 feature construction from the existing evaluator.
"""

from pathlib import Path
import json
import time
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, hamming_loss
)

# Reuse the verified v4.1 preprocessing/data pipeline.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import SmartSOC_OneShot_Model_Evaluation as base

N_SPLITS = 5
RANDOM_STATE = 42
MAX_ITER = 1500

RESULT_DIR = PROJECT_ROOT / "results" / "windows_apt_v4_1"
OUT_DIR = RESULT_DIR / "class_weight_threshold_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_weighted_ovr(X_train, y_train):
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
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y)
        models.append(("model", model))
    return models


def predict_probabilities(models, X):
    probs = np.zeros((X.shape[0], len(models)), dtype=np.float64)

    for j, (kind, model) in enumerate(models):
        if kind == "constant":
            probs[:, j] = float(model)
        else:
            probs[:, j] = model.predict_proba(X)[:, 1]

    return probs


def predict_with_thresholds(probs, thresholds):
    return (probs >= thresholds.reshape(1, -1)).astype(np.int8)


def metrics(y_true, y_pred):
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "label_accuracy": float(np.mean(y_true == y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def optimize_per_class_thresholds(y_true, oof_probs):
    """
    Optimize each technique threshold using OOF training predictions only.
    Candidate thresholds are deliberately coarse enough to remain practical.
    """
    candidates = np.linspace(0.05, 0.95, 91)
    thresholds = np.full(y_true.shape[1], 0.50, dtype=np.float64)

    rows = []

    for j in range(y_true.shape[1]):
        best_t = 0.50
        best_f1 = -1.0

        for t in candidates:
            pred = (oof_probs[:, j] >= t).astype(np.int8)
            score = f1_score(
                y_true[:, j], pred,
                average="binary",
                zero_division=0
            )

            if score > best_f1:
                best_f1 = float(score)
                best_t = float(t)

        thresholds[j] = best_t

        rows.append({
            "technique_index": j,
            "threshold": best_t,
            "oof_f1": best_f1,
            "positive_count": int(y_true[:, j].sum()),
        })

    return thresholds, pd.DataFrame(rows)


def print_metrics(title, m):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"Subset Accuracy      : {m['subset_accuracy']:.4f} ({m['subset_accuracy']*100:.2f}%)")
    print(f"Label-wise Accuracy  : {m['label_accuracy']:.4f} ({m['label_accuracy']*100:.2f}%)")
    print(f"Micro Precision      : {m['micro_precision']:.4f}")
    print(f"Micro Recall         : {m['micro_recall']:.4f}")
    print(f"Micro F1             : {m['micro_f1']:.4f}")
    print(f"Macro Precision      : {m['macro_precision']:.4f}")
    print(f"Macro Recall         : {m['macro_recall']:.4f}")
    print(f"Macro F1             : {m['macro_f1']:.4f}")
    print(f"Weighted Precision   : {m['weighted_precision']:.4f}")
    print(f"Weighted Recall      : {m['weighted_recall']:.4f}")
    print(f"Weighted F1          : {m['weighted_f1']:.4f}")
    print(f"Hamming Loss         : {m['hamming_loss']:.4f}")


def main():
    start = time.time()

    print("=" * 72)
    print("SmartSOC v4.1 — CLASS WEIGHT + THRESHOLD OPTIMIZATION")
    print("=" * 72)
    print("Production artifacts will NOT be modified.")
    print("Thresholds are optimized from OOF training predictions only.")

    files = base.require_files()

    structured_train = pd.read_csv(files["structured_train"])
    structured_test = pd.read_csv(files["structured_test"])
    text_train = base.load_text(files["text_train"])
    text_test = base.load_text(files["text_test"])
    train_labels = base.load_labels(files["labels_train"])
    test_labels = base.load_labels(files["labels_test"])

    mlb, y_train, y_test = base.build_label_matrix(train_labels, test_labels)
    cat_cols, num_cols = base.get_feature_columns(structured_train)

    print(f"\nTraining events : {len(structured_train)}")
    print(f"Testing events  : {len(structured_test)}")
    print(f"MITRE techniques: {len(mlb.classes_)}")

    # ------------------------------------------------------------
    # 5-fold OOF probabilities for threshold selection
    # ------------------------------------------------------------
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_probs = np.zeros_like(y_train, dtype=np.float64)
    fold_rows = []

    for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(structured_train), 1):
        fold_start = time.time()
        print("\n" + "-" * 72)
        print(f"FOLD {fold_no}/{N_SPLITS}")
        print("-" * 72)

        X_tr, X_va, _, _, _ = base.build_features(
            structured_train.iloc[train_idx],
            structured_train.iloc[valid_idx],
            text_train.iloc[train_idx],
            text_train.iloc[valid_idx],
            cat_cols,
            num_cols,
        )

        print(f"Feature matrix: train={X_tr.shape}, valid={X_va.shape}")

        models = train_weighted_ovr(X_tr, y_train[train_idx])
        probs = predict_probabilities(models, X_va)
        oof_probs[valid_idx] = probs

        default_pred = (probs >= 0.50).astype(np.int8)
        fm = metrics(y_train[valid_idx], default_pred)
        fm["fold"] = fold_no
        fm["seconds"] = round(time.time() - fold_start, 2)
        fold_rows.append(fm)

        print_metrics(f"FOLD {fold_no} — CLASS-WEIGHTED, DEFAULT 0.50", fm)

    fold_df = pd.DataFrame(fold_rows)

    # ------------------------------------------------------------
    # Threshold optimization on OOF predictions
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("OPTIMIZING PER-TECHNIQUE THRESHOLDS")
    print("=" * 72)
    print("Method: maximize binary F1 for each technique using OOF predictions.")

    thresholds, threshold_df = optimize_per_class_thresholds(y_train, oof_probs)
    threshold_df["technique"] = list(mlb.classes_)
    threshold_df = threshold_df[
        ["technique", "technique_index", "positive_count", "threshold", "oof_f1"]
    ]

    oof_default = (oof_probs >= 0.50).astype(np.int8)
    oof_optimized = predict_with_thresholds(oof_probs, thresholds)

    default_oof_metrics = metrics(y_train, oof_default)
    optimized_oof_metrics = metrics(y_train, oof_optimized)

    print_metrics("OOF — CLASS-WEIGHTED, DEFAULT THRESHOLD 0.50", default_oof_metrics)
    print_metrics("OOF — CLASS-WEIGHTED + OPTIMIZED THRESHOLDS", optimized_oof_metrics)

    # ------------------------------------------------------------
    # Train final class-weighted model on ALL training data
    # ------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TRAINING FINAL CLASS-WEIGHTED MODEL ON FULL TRAINING SET")
    print("=" * 72)

    X_train_full, X_test_full, _, _, _ = base.build_features(
        structured_train,
        structured_test,
        text_train,
        text_test,
        cat_cols,
        num_cols,
    )

    print(f"Final feature matrix: train={X_train_full.shape}, test={X_test_full.shape}")

    final_models = train_weighted_ovr(X_train_full, y_train)
    test_probs = predict_probabilities(final_models, X_test_full)

    test_default = (test_probs >= 0.50).astype(np.int8)
    test_optimized = predict_with_thresholds(test_probs, thresholds)

    test_default_metrics = metrics(y_test, test_default)
    test_optimized_metrics = metrics(y_test, test_optimized)

    print_metrics(
        "FINAL TEST — CLASS-WEIGHTED, DEFAULT THRESHOLD 0.50",
        test_default_metrics,
    )

    print_metrics(
        "FINAL TEST — CLASS-WEIGHTED + OOF-OPTIMIZED THRESHOLDS",
        test_optimized_metrics,
    )

    # ------------------------------------------------------------
    # Per-technique comparison
    # ------------------------------------------------------------
    per_rows = []

    for j, technique in enumerate(mlb.classes_):
        f1_default = f1_score(
            y_test[:, j], test_default[:, j],
            zero_division=0
        )
        f1_opt = f1_score(
            y_test[:, j], test_optimized[:, j],
            zero_division=0
        )

        per_rows.append({
            "technique": technique,
            "test_f1_default_0.50": float(f1_default),
            "test_f1_optimized": float(f1_opt),
            "delta_f1": float(f1_opt - f1_default),
            "test_support": int(y_test[:, j].sum()),
            "selected_threshold": float(thresholds[j]),
        })

    per_df = pd.DataFrame(per_rows).sort_values(
        "delta_f1", ascending=False
    )

    print("\n" + "=" * 72)
    print("RARE / REVIEWER-RELEVANT TECHNIQUES")
    print("=" * 72)

    for technique in ["T1078", "T1565.001", "T1485"]:
        row = per_df[per_df["technique"] == technique]
        if not row.empty:
            print(row.to_string(index=False))

    # ------------------------------------------------------------
    # Save experiment artifacts
    # ------------------------------------------------------------
    threshold_path = OUT_DIR / "optimized_thresholds.csv"
    per_path = OUT_DIR / "per_technique_test_comparison.csv"
    fold_path = OUT_DIR / "class_weight_oof_folds.csv"
    json_path = OUT_DIR / "experiment_summary.json"

    threshold_df.to_csv(threshold_path, index=False)
    per_df.to_csv(per_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    summary = {
        "experiment": "class_weighted_logistic_regression_plus_oof_threshold_optimization",
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "threshold_source": "OOF_training_predictions_only",
        "test_set_used_for_threshold_selection": False,
        "production_artifacts_modified": False,
        "oof_default_metrics": default_oof_metrics,
        "oof_optimized_metrics": optimized_oof_metrics,
        "test_default_metrics": test_default_metrics,
        "test_optimized_metrics": test_optimized_metrics,
        "techniques": int(len(mlb.classes_)),
        "training_events": int(len(y_train)),
        "test_events": int(len(y_test)),
    }

    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    elapsed = time.time() - start

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"Runtime: {elapsed/60:.2f} minutes")
    print("\nSaved:")
    print(threshold_path)
    print(per_path)
    print(fold_path)
    print(json_path)

    print("\nProduction model was NOT modified.")
    print("Use the optimized TEST result only after reviewing the comparison.")


if __name__ == "__main__":
    main()
