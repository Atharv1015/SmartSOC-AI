import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# SmartSOC - XGBoost Multiclass Attack Classification
# ============================================================

print("\n" + "=" * 70)
print("       SmartSOC XGBoost - Attack Classification")
print("=" * 70)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = PROCESSED_DIR / "test_clean.csv"

PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"

REPORT_FILE = RESULTS_DIR / "xgboost_report.txt"
CM_FILE = RESULTS_DIR / "xgboost_confusion_matrix.csv"
PREDICTIONS_FILE = RESULTS_DIR / "xgboost_predictions.csv"


# ------------------------------------------------------------
# 2. Verify required files
# ------------------------------------------------------------

print("\nChecking required files...")

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    PREPROCESSOR_FILE
]

for file in required_files:

    if not file.exists():

        print("\nERROR: Required file not found:")
        print(file)

        raise SystemExit(1)

print("All required files found.")


# ------------------------------------------------------------
# 3. Load datasets
# ------------------------------------------------------------

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training rows : {len(train):,}")
print(f"Testing rows  : {len(test):,}")


# ------------------------------------------------------------
# 4. Define target and input features
# ------------------------------------------------------------

TARGET = "attack_cat"

DROP_COLUMNS = [
    "attack_cat",
    "label"
]

X_train = train.drop(columns=DROP_COLUMNS)
y_train_text = train[TARGET]

X_test = test.drop(columns=DROP_COLUMNS)
y_test_text = test[TARGET]

print(f"\nInput features : {X_train.shape[1]}")
print(f"Target classes : {y_train_text.nunique()}")


# ------------------------------------------------------------
# 5. Display class distribution
# ------------------------------------------------------------

print("\nTraining class distribution:")
print(
    y_train_text
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 6. Encode target labels
# ------------------------------------------------------------

print("\nEncoding target classes...")

label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(
    y_train_text
)

y_test = label_encoder.transform(
    y_test_text
)

print("\nClass mapping:")

for number, name in enumerate(label_encoder.classes_):

    print(
        f"{number:2d} -> {name}"
    )


# ------------------------------------------------------------
# 7. Calculate sample weights
# ------------------------------------------------------------

print("\nCalculating class-balanced sample weights...")

class_counts = np.bincount(
    y_train
)

total_samples = len(y_train)
number_of_classes = len(class_counts)

class_weights = {}

for class_id, count in enumerate(class_counts):

    weight = (
        total_samples /
        (number_of_classes * count)
    )

    class_weights[class_id] = weight


print("\nClass weights:")

for class_id, weight in class_weights.items():

    print(
        f"{label_encoder.classes_[class_id]:15s} "
        f"{weight:.4f}"
    )


sample_weights = np.array([
    class_weights[class_id]
    for class_id in y_train
])


# ------------------------------------------------------------
# 8. Load preprocessing pipeline
# ------------------------------------------------------------

print("\nLoading preprocessor...")

preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


# ------------------------------------------------------------
# 9. Transform features
# ------------------------------------------------------------

print("\nTransforming training data...")

X_train_processed = preprocessor.transform(
    X_train
)

print("Transforming testing data...")

X_test_processed = preprocessor.transform(
    X_test
)

print("\nProcessed feature shapes:")

print(
    f"Training : {X_train_processed.shape}"
)

print(
    f"Testing  : {X_test_processed.shape}"
)


# ------------------------------------------------------------
# 10. Train XGBoost
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("Training XGBoost...")
print("=" * 70)

model = XGBClassifier(

    objective="multi:softprob",

    num_class=number_of_classes,

    n_estimators=500,

    max_depth=8,

    learning_rate=0.08,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=2,

    gamma=0.0,

    reg_alpha=0.1,

    reg_lambda=1.0,

    tree_method="hist",

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1

)


model.fit(

    X_train_processed,

    y_train,

    sample_weight=sample_weights

)


print("\nXGBoost training complete.")


# ------------------------------------------------------------
# 11. Generate predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_processed
)

y_probability = model.predict_proba(
    X_test_processed
)


# ------------------------------------------------------------
# 12. Calculate confidence
# ------------------------------------------------------------

confidence = np.max(
    y_probability,
    axis=1
)


predicted_attack = label_encoder.inverse_transform(
    y_pred.astype(int)
)


# ------------------------------------------------------------
# 13. Overall metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Accuracy        : {accuracy:.4f}"
)

print(
    f"Macro Precision : {precision:.4f}"
)

print(
    f"Macro Recall    : {recall:.4f}"
)

print(
    f"Macro F1        : {f1:.4f}"
)


# ------------------------------------------------------------
# 14. Detailed classification report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
)

print(report)


# ------------------------------------------------------------
# 15. Save classification report
# ------------------------------------------------------------

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "SmartSOC XGBoost Attack Classification\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Training rows: {len(train):,}\n"
    )

    file.write(
        f"Testing rows: {len(test):,}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Macro Precision: {precision:.4f}\n"
    )

    file.write(
        f"Macro Recall: {recall:.4f}\n"
    )

    file.write(
        f"Macro F1: {f1:.4f}\n\n"
    )

    file.write(
        report
    )


# ------------------------------------------------------------
# 16. Confusion matrix
# ------------------------------------------------------------

print("\nGenerating confusion matrix...")

labels = np.arange(
    number_of_classes
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

cm_df.to_csv(
    CM_FILE
)

print(
    f"Confusion matrix saved to:"
)

print(CM_FILE)


# ------------------------------------------------------------
# 17. Save prediction results
# ------------------------------------------------------------

print("\nSaving prediction results...")

prediction_df = pd.DataFrame({

    "actual_attack": y_test_text.values,

    "predicted_attack": predicted_attack,

    "confidence": confidence

})

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"Predictions saved to:"
)

print(PREDICTIONS_FILE)


# ------------------------------------------------------------
# 18. Save model
# ------------------------------------------------------------

print("\nSaving XGBoost model...")

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"Model saved to:"
)

print(MODEL_FILE)


# ------------------------------------------------------------
# 19. Save label encoder
# ------------------------------------------------------------

print("\nSaving label encoder...")

joblib.dump(
    label_encoder,
    LABEL_ENCODER_FILE
)

print(
    f"Label encoder saved to:"
)

print(LABEL_ENCODER_FILE)


# ------------------------------------------------------------
# 20. Confidence statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONFIDENCE STATISTICS")
print("=" * 70)

print(
    f"Mean confidence   : {confidence.mean():.4f}"
)

print(
    f"Median confidence : {np.median(confidence):.4f}"
)

print(
    f"Minimum confidence: {confidence.min():.4f}"
)

print(
    f"Maximum confidence: {confidence.max():.4f}"
)


# ------------------------------------------------------------
# 21. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("XGBOOST EXPERIMENT COMPLETE")
print("=" * 70)

print(
    f"\nAccuracy        : {accuracy:.4f}"
)

print(
    f"Macro Precision : {precision:.4f}"
)

print(
    f"Macro Recall    : {recall:.4f}"
)

print(
    f"Macro F1        : {f1:.4f}"
)

print("\nFiles created:")

print(
    f"  Model          : {MODEL_FILE}"
)

print(
    f"  Label Encoder  : {LABEL_ENCODER_FILE}"
)

print(
    f"  Report         : {REPORT_FILE}"
)

print(
    f"  Confusion      : {CM_FILE}"
)

print(
    f"  Predictions    : {PREDICTIONS_FILE}"
)

print("\nSmartSOC XGBoost training finished.")