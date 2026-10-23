import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

train_file = DATA_DIR / "UNSW_NB15_training-set.csv"

df = pd.read_csv(train_file)

print("\n========== SmartSOC Feature Analysis ==========\n")

categorical = ["proto", "service", "state"]

for column in categorical:
    print(f"\n--- {column} ---")
    print(f"Unique values: {df[column].nunique()}")

    print("\nValues:")
    print(df[column].value_counts().head(30))

print("\n========== Numerical Feature Summary ==========\n")

numeric = df.select_dtypes(include=["int64", "float64"]).columns

print(f"Numerical columns: {len(numeric)}")

print("\nColumns:")
print(numeric.tolist())

print("\n========== Target Information ==========\n")

print("Target: attack_cat")
print(df["attack_cat"].unique().tolist())

print("\nFeature analysis complete.")