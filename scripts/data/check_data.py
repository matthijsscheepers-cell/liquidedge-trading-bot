"""
Check if data has NaN values
"""

import sys
sys.path.insert(0, '.')

from src.execution.capital_connector import CapitalConnector

config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

connector = CapitalConnector(config)
connector.connect()

print("Fetching GOLD data...")
df = connector.get_historical_data('GOLD', '15m', 300)

print(f"\nDataFrame info:")
print(f"  Shape: {df.shape}")
print(f"  Index: {df.index[0]} to {df.index[-1]}")
print()

print("Checking for NaN values:")
print(f"  Close NaN count: {df['close'].isna().sum()}")
print(f"  Open NaN count: {df['open'].isna().sum()}")
print(f"  High NaN count: {df['high'].isna().sum()}")
print(f"  Low NaN count: {df['low'].isna().sum()}")
print(f"  Volume NaN count: {df['volume'].isna().sum()}")
print()

print("First 10 rows:")
print(df.head(10))
print()

print("Last 10 rows:")
print(df.tail(10))
print()

print("Data types:")
print(df.dtypes)
