"""
Explore Databento DBN file structure
"""

import databento as db
from pathlib import Path

print("=" * 70)
print("EXPLORING DBN FILE")
print("=" * 70)
print()

# Load DBN file
dbn_file = Path("GLBX-20260201-B83UY6MA47/glbx-mdp3-20100912-20260129.ohlcv-1m.dbn.zst")
print(f"Loading: {dbn_file}")
print()

# Open DBN store
store = db.DBNStore.from_file(dbn_file)

# Get metadata
print("DBN Metadata:")
print(f"  Schema: {store.schema}")
print(f"  Dataset: {store.dataset}")
print(f"  Start: {store.start}")
print(f"  End: {store.end}")
print(f"  Symbols: {store.symbols}")
print()

# Convert to DataFrame
print("Converting to DataFrame...")
df = store.to_df()
print(f"Full dataset has {len(df)} records")
print("Taking first 10000 for exploration...")
df = df.head(10000)

print(f"✓ Loaded {len(df)} records")
print()

print("DataFrame columns:")
print(f"  {list(df.columns)}")
print()

print("DataFrame dtypes:")
print(df.dtypes)
print()

print("First 10 rows:")
print(df.head(10))
print()

print("Unique values in key columns:")
if 'symbol' in df.columns:
    print(f"  Symbols: {df['symbol'].unique()}")
if 'instrument_id' in df.columns:
    print(f"  Instrument IDs (first 10): {df['instrument_id'].unique()[:10]}")
print()

print("Index type:")
print(f"  {type(df.index)}")
print(f"  {df.index[:5]}")
print()

print("Sample data:")
print(df.iloc[0])
