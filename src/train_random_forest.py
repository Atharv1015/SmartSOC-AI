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

# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_clean.csv"
TEST_FILE = DATA_DIR / "test_clean.csv"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("\n========== SmartSOC Random Forest Baseline ==========\n")

print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")

# --------------------------------------------------
# Define target and features
# --------------------------------------------------

TARGET = "attack_cat"

DROP_COLUMNS = ["attack_cat", "label"]

X_train = train.drop(columns=DROP_COLUMNS)
y_train = train[TARGET]

X_test = test.drop(columns=DROP_COLUMNS)
y_test = test[TARGET]

print(f"\nInput features: {X_train.shape[1]}")
print(f"Target classes: {y_train.nunique()}")

# --------------------------------------------------
# Load preprocessor
# --------------------------------------------------

print("\nLoading preprocessor...")

preprocessor = joblib.load(PREPROCESSOR_FILE)

# --------------------------------------------------
# Transform data
# --------------------------------------------------

print("Transforming training data...")

X_train_processed = preprocessor.transform(X_train)

print("Transforming testing data...")

X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed testing shape:  {X_test_processed.shape}")

# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_processed, y_train)

print("Training complete.")

# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test_processed)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n========== Overall Results ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {f1:.4f}")

# --------------------------------------------------
# Detailed classification report
# --------------------------------------------------

print("\n========== Classification Report ==========\n")

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)

# Save report
with open(RESULTS_DIR / "random_forest_report.txt", "w") as file:
    file.write(report)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = sorted(y_test.unique())

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

cm_df.to_csv(
    RESULTS_DIR / "random_forest_confusion_matrix.csv"
)

# --------------------------------------------------
# Save model
# --------------------------------------------------

model_file = MODEL_DIR / "random_forest_baseline.joblib"

joblib.dump(model, model_file)

print(f"\nModel saved to: {model_file}")
print("Confusion matrix saved.")
print("\nRandom Forest baseline complete.")