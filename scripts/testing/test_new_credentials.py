"""
Test with new API key
"""

import sys
sys.path.insert(0, '.')

from src.execution.capital_connector import CapitalConnector

print("=" * 70)
print("TESTING NEW CAPITAL.COM CREDENTIALS")
print("=" * 70)
print()

config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

try:
    connector = CapitalConnector(config)
    print("Connecting to Capital.com...")
    print()

    result = connector.connect()

    if result:
        print()
        print("=" * 70)
        print("✓ CONNECTION SUCCESSFUL!")
        print("=" * 70)
        print()

        # Test data retrieval for multiple assets
        assets = ['GOLD', 'SILVER', 'US100', 'US500']

        for asset in assets:
            print(f"\nTesting {asset} historical data (15m)...")
            try:
                df = connector.get_historical_data(asset, '15m', 100)
                if not df.empty:
                    print(f"  ✓ {asset}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
                else:
                    print(f"  ✗ {asset}: No data returned")
            except Exception as e:
                print(f"  ✗ {asset}: Failed - {e}")

        print()
        print("=" * 70)
        print("READY TO RUN BACKTEST!")
        print("=" * 70)
        print()
        print("Run the backtest with:")
        print()
        print('python3 scripts/run_backtest.py --mode capital \\')
        print('  --symbols GOLD SILVER US100 US500 \\')
        print('  --timeframe 15m \\')
        print('  --start 2024-12-01 --end 2024-12-31 \\')
        print('  --api-key "jvJoOhauq6w7Yub0" \\')
        print('  --password "Vergeten22!" \\')
        print('  --identifier "matthijsscheepers@gmail.com"')

except Exception as e:
    print()
    print("=" * 70)
    print("✗ CONNECTION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
