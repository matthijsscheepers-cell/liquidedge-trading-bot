"""
Test user's Capital.com credentials
"""

import sys
sys.path.insert(0, '.')

from src.execution.capital_connector import CapitalConnector

print("=" * 70)
print("TESTING CAPITAL.COM CONNECTION")
print("=" * 70)
print()

config = {
    'api_key': 'Hh6exwe6rrnYDgQe',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

try:
    connector = CapitalConnector(config)
    print("Attempting to connect...")
    print()

    result = connector.connect()

    if result:
        print()
        print("=" * 70)
        print("✓ CONNECTION SUCCESSFUL!")
        print("=" * 70)
        print()

        # Test data retrieval
        print("Testing historical data retrieval for GOLD (15m)...")
        try:
            df = connector.get_historical_data('GOLD', '15m', 100)
            if not df.empty:
                print(f"✓ Successfully fetched {len(df)} bars")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                print()
                print("First 5 bars:")
                print(df.head())
                print()
                print("=" * 70)
                print("Ready to run backtest!")
                print("=" * 70)
            else:
                print("⚠ Warning: No data returned")
        except Exception as e:
            print(f"✗ Data fetch failed: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print()
    print("=" * 70)
    print("✗ CONNECTION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
