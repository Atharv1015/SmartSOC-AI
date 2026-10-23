import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

print("\n========== SmartSOC Dataset Inspection ==========\n")

for file in sorted(DATA_DIR.glob("*.csv")):
    print(f"\nFILE: {file.name}")
    print("-" * 70)

    df = pd.read_csv(file)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print(f"\nTotal missing values: {df.isnull().sum().sum():,}")

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    # Look for likely target columns
    for column in ["label", "attack_cat"]:
        if column in df.columns:
            print(f"\n--- Distribution of {column} ---")
            print(df[column].value_counts(dropna=False))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\n" + "=" * 70)