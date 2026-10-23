"""
SmartSOC v4.1 — Logistic Regression C Sweep + Global Threshold Calibration

Purpose:
- Evaluate regularization strength C without modifying production artifacts.
- Use 5-fold OOF training data only for model selection.
- Select C using OOF metrics.
- For the selected C, calibrate ONE global decision threshold on OOF scores.
- Retrain selected configuration on the full training partition.
- Evaluate exactly once on the untouched held-out test set.
- Save all results under results/windows_apt_v4_1/hyperparameter_sweep/.

Production artifacts are never modified.
"""

from pathlib import Path
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
N_SPLITS = 5
RANDOM_STATE = 42
MAX_ITER = 1500

# Deliberately moderate sweep to keep runtime practical.
C_VALUES = [0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0]

# One global threshold, selected only from OOF scores.
THRESHOLDS = np.round(np.arange(0.30, 0.71, 0.02), 2)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "src" else SCRIPT_DIR

RESULT_DIR = (
    PROJECT_ROOT / "results" / "windows_apt_v4_1" / "hyperparameter_sweep"
)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Import the validated v4.1 evaluation pipeline.
sys.path.insert(0, str(SCRIPT_DIR))
try:
    import evaluate_smartsoc_v4_1 as base
except Exception as exc:
    print("[ERROR] Could not import evaluate_smartsoc_v4_1.py")
    print(exc)
    sys.exit(1)


def train_ovr(X_train, y_train, C):
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
            C=C,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y)
        models.append(("model", model))

    return models


def score_ovr(models, X):
    """Return decision scores with one column per technique."""
    scores = np.zeros((X.shape[0], len(models)), dtype=np.float32)

    for j, (kind, model) in enumerate(models):
        if kind == "constant":
            # Keep a strong positive/negative score for constant classifiers.
            scores[:, j] = 10.0 if model == 1 else -10.0
        else:
            scores[:, j] = model.decision_function(X)

    return scores


def predict_from_scores(scores, threshold):
    # Logistic decision_function threshold 0 corresponds to probability 0.50.
    # Convert probability threshold to an equivalent logit threshold.
    p = float(threshold)
    logit = np.log(p / (1.0 - p))
    return (scores >= logit).astype(np.int8)


def metrics(y_true, y_pred):
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "label_accuracy": float(np.mean(y_true == y_pred)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


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
    print(f"Weighted F1          : {m['weighted_f1']:.4f}")
    print(f"Hamming Loss         : {m['hamming_loss']:.4f}")


def main():
    start_all = time.time()

    print("=" * 72)
    print("SmartSOC v4.1 — LOGISTIC C SWEEP + GLOBAL THRESHOLD CALIBRATION")
    print("=" * 72)
    print("Production artifacts will NOT be modified.")
    print("Test data is used only for the final evaluation.")

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
    print(f"C candidates     : {C_VALUES}")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # C sweep: OOF only
    # --------------------------------------------------------
    all_rows = []

    for C in C_VALUES:
        print("\n" + "#" * 72)
        print(f"C = {C}")
        print("#" * 72)

        oof_scores = np.zeros((len(y_train), y_train.shape[1]), dtype=np.float32)

        for fold_no, (train_idx, valid_idx) in enumerate(
            kfold.split(structured_train), start=1
        ):
            fold_start = time.time()
            print(f"\nFOLD {fold_no}/{N_SPLITS} | C={C}")

            X_tr, X_va, _, _, _ = base.build_features(
                structured_train.iloc[train_idx],
                structured_train.iloc[valid_idx],
                text_train.iloc[train_idx],
                text_train.iloc[valid_idx],
                cat_cols,
                num_cols,
            )

            models = train_ovr(X_tr, y_train[train_idx], C)
            oof_scores[valid_idx] = score_ovr(models, X_va)

            print(f"Feature matrix: train={X_tr.shape}, valid={X_va.shape}")
            print(f"Fold runtime: {(time.time()-fold_start)/60:.2f} min")

        # Default 0.50 equivalent.
        default_pred = predict_from_scores(oof_scores, 0.50)
        default_m = metrics(y_train, default_pred)
        default_m.update({"C": C, "threshold": 0.50, "mode": "OOF_default"})
        all_rows.append(default_m)

        print_metrics(f"OOF — C={C} — THRESHOLD 0.50", default_m)

        # Global threshold selection on OOF only.
        best = None
        for threshold in THRESHOLDS:
            pred = predict_from_scores(oof_scores, threshold)
            m = metrics(y_train, pred)

            # Selection objective: exact-match first, then micro-F1, then macro-F1.
            key = (m["subset_accuracy"], m["micro_f1"], m["macro_f1"])

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "threshold": float(threshold),
                    "metrics": m,
                }

        calibrated = dict(best["metrics"])
        calibrated.update(
            {
                "C": C,
                "threshold": best["threshold"],
                "mode": "OOF_global_threshold",
            }
        )
        all_rows.append(calibrated)

        print_metrics(
            f"OOF — C={C} — OPTIMIZED GLOBAL THRESHOLD={best['threshold']:.2f}",
            calibrated,
        )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULT_DIR / "c_sweep_oof_results.csv", index=False)

    # Select C based on calibrated OOF exact-match, with micro-F1 and macro-F1
    # as deterministic tie-breakers.
    calibrated_df = results_df[results_df["mode"] == "OOF_global_threshold"].copy()
    calibrated_df = calibrated_df.sort_values(
        ["subset_accuracy", "micro_f1", "macro_f1"],
        ascending=False,
    )

    best_row = calibrated_df.iloc[0]
    best_C = float(best_row["C"])
    best_threshold = float(best_row["threshold"])

    print("\n" + "=" * 72)
    print("SELECTED CONFIGURATION FROM OOF ONLY")
    print("=" * 72)
    print(f"Best C         : {best_C}")
    print(f"Global threshold: {best_threshold:.2f}")
    print(
        f"OOF Exact-match: {best_row['subset_accuracy']*100:.2f}% | "
        f"Micro-F1: {best_row['micro_f1']*100:.2f}% | "
        f"Macro-F1: {best_row['macro_f1']*100:.2f}%"
    )

    # --------------------------------------------------------
    # Final training on full training partition
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL TRAINING — SELECTED C")
    print("=" * 72)

    X_train, X_test, _, _, _ = base.build_features(
        structured_train,
        structured_test,
        text_train,
        text_test,
        cat_cols,
        num_cols,
    )

    print(f"Final feature matrix: train={X_train.shape}, test={X_test.shape}")

    final_models = train_ovr(X_train, y_train, best_C)
    test_scores = score_ovr(final_models, X_test)

    # Evaluate both default and selected global threshold.
    test_default = metrics(y_test, predict_from_scores(test_scores, 0.50))
    test_default.update({"C": best_C, "threshold": 0.50})

    test_calibrated = metrics(
        y_test, predict_from_scores(test_scores, best_threshold)
    )
    test_calibrated.update({"C": best_C, "threshold": best_threshold})

    print_metrics(
        f"FINAL TEST — C={best_C} — DEFAULT THRESHOLD 0.50",
        test_default,
    )

    print_metrics(
        f"FINAL TEST — C={best_C} — OOF-SELECTED GLOBAL THRESHOLD {best_threshold:.2f}",
        test_calibrated,
    )

    # Save final results.
    summary = {
        "project": "SmartSOC Windows-APT v4.1",
        "selection": {
            "method": "5-fold OOF; exact-match primary, micro-F1 then macro-F1 tie-break",
            "C_candidates": C_VALUES,
            "threshold_candidates": THRESHOLDS.tolist(),
            "selected_C": best_C,
            "selected_global_threshold": best_threshold,
        },
        "test_default_threshold": test_default,
        "test_oof_selected_threshold": test_calibrated,
        "production_modified": False,
    }

    (RESULT_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"Runtime: {(time.time()-start_all)/60:.2f} minutes")
    print("Saved:")
    print(RESULT_DIR / "c_sweep_oof_results.csv")
    print(RESULT_DIR / "experiment_summary.json")
    print("\nProduction artifacts were NOT modified.")


if __name__ == "__main__":
    main()
