import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

print("\n========== SmartSOC Dataset Preparation ==========\n")

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Original training rows: {len(train):,}")
print(f"Original testing rows:  {len(test):,}")

# --------------------------------------------------
# 2. Define feature columns
# --------------------------------------------------

TARGET = "attack_cat"

DROP_COLUMNS = ["attack_cat", "label"]

FEATURE_COLUMNS = [
    column for column in train.columns
    if column not in DROP_COLUMNS
]

# --------------------------------------------------
# 3. Remove exact duplicate rows
# --------------------------------------------------

train_before = len(train)
test_before = len(test)

train = train.drop_duplicates().reset_index(drop=True)
test = test.drop_duplicates().reset_index(drop=True)

print(f"\nTraining duplicates removed: "
      f"{train_before - len(train):,}")

print(f"Testing duplicates removed: "
      f"{test_before - len(test):,}")

print(f"Clean training rows: {len(train):,}")
print(f"Clean testing rows:  {len(test):,}")

# --------------------------------------------------
# 4. Detect cross-split feature overlap
# --------------------------------------------------

print("\nChecking cross-split feature overlap...")

train_features = train[FEATURE_COLUMNS].copy()
test_features = test[FEATURE_COLUMNS].copy()

# Create stable row keys
train_keys = pd.util.hash_pandas_object(
    train_features,
    index=False
)

test_keys = pd.util.hash_pandas_object(
    test_features,
    index=False
)

train_key_set = set(train_keys)

overlap_mask = test_keys.isin(train_key_set)

overlap_count = overlap_mask.sum()

print(f"Cross-split overlapping test rows: {overlap_count:,}")

# Remove overlapping feature vectors from primary evaluation set
test_clean = test.loc[~overlap_mask].reset_index(drop=True)

print(
    f"Leakage-aware testing rows: "
    f"{len(test_clean):,}"
)

# --------------------------------------------------
# 5. Save cleaned datasets
# --------------------------------------------------

train_output = PROCESSED_DIR / "train_clean.csv"
test_output = PROCESSED_DIR / "test_clean.csv"

train.to_csv(train_output, index=False)
test_clean.to_csv(test_output, index=False)

print("\nSaved:")
print(f"  {train_output}")
print(f"  {test_output}")

# --------------------------------------------------
# 6. Print class distributions
# --------------------------------------------------

print("\n========== Training Class Distribution ==========")

print(
    train[TARGET]
    .value_counts()
    .to_string()
)

print("\n========== Evaluation Class Distribution ==========")

print(
    test_clean[TARGET]
    .value_counts()
    .to_string()
)

print("\nDataset preparation complete.")