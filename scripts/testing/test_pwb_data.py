"""
Test pwb-toolbox commodity data
"""

from pwb_toolbox import datasets as pwb_ds
import pandas as pd

print("=" * 70)
print("PWB-TOOLBOX COMMODITY DATA TEST")
print("=" * 70)
print()

# Load commodities dataset
print("Loading Commodities-Daily-Price dataset...")
df = pwb_ds.load_dataset("Commodities-Daily-Price")

print(f"✓ Loaded {len(df)} rows")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Columns: {list(df.columns)}")
print()

# Show unique symbols
print(f"Available symbols ({df['symbol'].nunique()}):")
symbols = sorted(df['symbol'].unique())
for i in range(0, len(symbols), 10):
    print(f"  {', '.join(symbols[i:i+10])}")
print()

# Load specific commodity (Crude Oil)
print("=" * 70)
print("Loading CL1 (Crude Oil Futures)...")
print("=" * 70)
df_cl = pwb_ds.load_dataset("Commodities-Daily-Price", symbols=["CL1"])

print(f"✓ CL1: {len(df_cl)} rows")
print(f"  Date range: {df_cl['date'].min()} to {df_cl['date'].max()}")
print()
print("First 5 rows:")
print(df_cl.head())
print()
print("Latest 5 rows:")
print(df_cl.tail())
print()

# Load multiple commodities relevant to our backtest
print("=" * 70)
print("Loading Gold (GC1) and Silver (SI1)...")
print("=" * 70)

try:
    df_gold = pwb_ds.load_dataset("Commodities-Daily-Price", symbols=["GC1"])
    print(f"✓ Gold (GC1): {len(df_gold)} rows from {df_gold['date'].min()} to {df_gold['date'].max()}")
except Exception as e:
    print(f"✗ Gold: {e}")

try:
    df_silver = pwb_ds.load_dataset("Commodities-Daily-Price", symbols=["SI1"])
    print(f"✓ Silver (SI1): {len(df_silver)} rows from {df_silver['date'].min()} to {df_silver['date'].max()}")
except Exception as e:
    print(f"✗ Silver: {e}")

print()
print("=" * 70)
print("PWB-TOOLBOX READY FOR BACKTESTING")
print("=" * 70)
