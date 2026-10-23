import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# SmartSOC - Random Forest Original Training Baseline
# ============================================================

print("\n" + "=" * 65)
print("       SmartSOC Random Forest - Original Training")
print("=" * 65)

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# IMPORTANT:
# Use the ORIGINAL UNSW-NB15 training dataset here.
TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"

# Use our leakage-aware test set.
TEST_FILE = PROCESSED_DIR / "test_clean.csv"

PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

# Output files
MODEL_FILE = MODEL_DIR / "random_forest_original.joblib"
REPORT_FILE = RESULTS_DIR / "random_forest_original_report.txt"
CM_FILE = RESULTS_DIR / "random_forest_original_confusion_matrix.csv"

# ------------------------------------------------------------
# 2. Verify files
# ------------------------------------------------------------

print("\nChecking files...")

required_files = [
    TRAIN_FILE,
    TEST_FILE,
    PREPROCESSOR_FILE
]

for file in required_files:
    if not file.exists():
        print(f"\nERROR: File not found:")
        print(file)
        raise SystemExit(1)

print("All required files found.")

# ------------------------------------------------------------
# 3. Load datasets
# ------------------------------------------------------------

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Original training rows : {len(train):,}")
print(f"Testing rows            : {len(test):,}")

# ------------------------------------------------------------
# 4. Define target and input features
# ------------------------------------------------------------

TARGET = "attack_cat"

# These columns must NOT be used as model inputs.
#
# attack_cat = target
# label      = binary target information
#
# Including label would cause target leakage.

DROP_COLUMNS = [
    "attack_cat",
    "label"
]

X_train = train.drop(columns=DROP_COLUMNS)
y_train = train[TARGET]

X_test = test.drop(columns=DROP_COLUMNS)
y_test = test[TARGET]

print(f"\nInput features : {X_train.shape[1]}")
print(f"Target classes : {y_train.nunique()}")

print("\nTraining class distribution:")
print(y_train.value_counts())

# ------------------------------------------------------------
# 5. Load preprocessing pipeline
# ------------------------------------------------------------

print("\nLoading preprocessor...")

preprocessor = joblib.load(PREPROCESSOR_FILE)

# ------------------------------------------------------------
# 6. Transform datasets
# ------------------------------------------------------------

print("\nTransforming training data...")

X_train_processed = preprocessor.transform(X_train)

print("Transforming testing data...")

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed data:")
print(f"Training shape : {X_train_processed.shape}")
print(f"Testing shape  : {X_test_processed.shape}")

# ------------------------------------------------------------
# 7. Train Random Forest
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("Training Random Forest...")
print("=" * 65)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

print("\nTraining complete.")

# ------------------------------------------------------------
# 8. Generate predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test_processed)

# ------------------------------------------------------------
# 9. Overall metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 65)
print("OVERALL RESULTS")
print("=" * 65)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")

# ------------------------------------------------------------
# 10. Detailed classification report
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# Save report
with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write(
        "SmartSOC Random Forest - Original Training\n"
    )
    file.write("=" * 65 + "\n\n")

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
        "Classification Report\n"
    )

    file.write(
        report
    )

# ------------------------------------------------------------
# 11. Confusion matrix
# ------------------------------------------------------------

print("\nGenerating confusion matrix...")

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

cm_df.to_csv(
    CM_FILE
)

print(f"Confusion matrix saved to:")
print(CM_FILE)

# ------------------------------------------------------------
# 12. Save model
# ------------------------------------------------------------

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_FILE
)

print(f"Model saved to:")
print(MODEL_FILE)

# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print("=" * 65)

print("\nRandom Forest using ORIGINAL training data")

print(f"\nAccuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")

print("\nFiles created:")
print(f"  Model  : {MODEL_FILE}")
print(f"  Report : {REPORT_FILE}")
print(f"  Matrix : {CM_FILE}")

print("\nNext step:")
print("Compare this result with the deduplicated Random Forest")
print("before moving to XGBoost.")

print("\nSmartSOC Random Forest experiment finished.")