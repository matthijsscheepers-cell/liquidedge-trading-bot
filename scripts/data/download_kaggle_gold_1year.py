"""
Download and explore Kaggle gold prices dataset (1 year)
"""

import kagglehub
import pandas as pd
import os

print("=" * 70)
print("DOWNLOADING KAGGLE GOLD 1-YEAR DATASET")
print("=" * 70)
print()

# Download latest version
print("Downloading from Kaggle...")
path = kagglehub.dataset_download("zkskhurram/gold-price-analysis-last-1-year")

print(f"✓ Dataset downloaded to: {path}")
print()

# List files in the dataset
print("Files in dataset:")
for root, dirs, files in os.walk(path):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {file} ({file_size:.1f} KB)")

        # If it's a CSV, load and preview it
        if file.endswith('.csv'):
            print()
            print(f"Preview of {file}:")
            df = pd.read_csv(file_path)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print()
            print("  First 10 rows:")
            print(df.head(10))
            print()
            print("  Last 10 rows:")
            print(df.tail(10))
            print()
            print("  Data types:")
            print(df.dtypes)
            print()
            print("  Summary statistics:")
            print(df.describe())
            print()

            # Check for date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                print(f"  Date columns found: {date_cols}")
                for col in date_cols:
                    print(f"    {col} sample: {df[col].iloc[0]}")
            print()

print("=" * 70)
print("DATASET READY")
print("=" * 70)
print(f"Path: {path}")
