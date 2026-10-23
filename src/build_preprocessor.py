import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DIR / "train_clean.csv"

TARGET = "attack_cat"

# Load cleaned training data
df = pd.read_csv(TRAIN_FILE)

# Columns that must never be model inputs
DROP_COLUMNS = ["attack_cat", "label"]

X = df.drop(columns=DROP_COLUMNS)

# Categorical features
categorical_features = [
    "proto",
    "service",
    "state"
]

# Everything else is numerical
numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

print("\n========== SmartSOC Preprocessor ==========\n")

print(f"Total input features: {len(X.columns)}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

print("\nCategorical:")
print(categorical_features)

print("\nNumerical:")
print(numerical_features)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Fit ONLY on training data
preprocessor.fit(X)

# Save preprocessing object
output_file = MODELS_DIR / "preprocessor.joblib"

joblib.dump(preprocessor, output_file)

print("\nPreprocessor fitted successfully.")
print(f"Saved to: {output_file}")

print("\nSmartSOC preprocessing setup complete.")