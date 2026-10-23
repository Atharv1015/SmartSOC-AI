import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

train_file = DATA_DIR / "UNSW_NB15_training-set.csv"
test_file = DATA_DIR / "UNSW_NB15_testing-set.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

print("\n========== SmartSOC Class Analysis ==========\n")

for name, df in [("TRAINING", train), ("TESTING", test)]:

    print(f"\n{name} DATASET")
    print("=" * 60)

    counts = df["attack_cat"].value_counts()
    percentages = df["attack_cat"].value_counts(normalize=True) * 100

    result = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    print(result)

    print("\nBinary label:")
    print(df["label"].value_counts())

print("\n========== Duplicate Analysis ==========\n")

print(f"Training duplicates: {train.duplicated().sum():,}")
print(f"Testing duplicates:  {test.duplicated().sum():,}")

# Remove target columns to check feature-level overlap
feature_columns = [
    c for c in train.columns
    if c not in ["label", "attack_cat"]
]

train_features = train[feature_columns].drop_duplicates()
test_features = test[feature_columns].drop_duplicates()

train_feature_keys = set(
    train_features.astype(str).agg("|".join, axis=1)
)

test_feature_keys = set(
    test_features.astype(str).agg("|".join, axis=1)
)

overlap = train_feature_keys.intersection(test_feature_keys)

print(f"\nUnique training feature rows: {len(train_feature_keys):,}")
print(f"Unique testing feature rows:  {len(test_feature_keys):,}")
print(f"Cross-split feature overlap:  {len(overlap):,}")

print("\nAnalysis complete.")