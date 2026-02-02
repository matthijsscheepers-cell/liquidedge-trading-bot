"""
Test Capital.com connection with updated API methods
"""

import sys
sys.path.insert(0, '.')

from src.execution.capital_connector import CapitalConnector

# Test 1: Demo connection (expected to fail but tests API compatibility)
print("=" * 60)
print("Test 1: Demo Connection")
print("=" * 60)

try:
    config = {
        'api_key': 'demo',
        'password': 'demo',
        'identifier': 'demo@demo.com'
    }

    connector = CapitalConnector(config)
    print("✓ Connector created")

    # Try to connect (will fail but shows error message)
    try:
        connector.connect()
        print("✓ Connected successfully (unexpected!)")
    except Exception as e:
        print(f"✗ Connection failed (expected): {e}")

except Exception as e:
    print(f"✗ Connector creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Real Credentials (without email)")
print("=" * 60)

# Test 2: User's credentials without email
try:
    config = {
        'api_key': 'Hh6exwe6rrnYDgQe',
        'password': 'Liquidedge22!',
        'identifier': 'unknown@placeholder.com'  # Placeholder
    }

    connector = CapitalConnector(config)
    print("✓ Connector created with user credentials")

    try:
        result = connector.connect()
        if result:
            print("✓ Connected successfully!")

            # Try to get historical data
            print("\nTrying to fetch GOLD historical data...")
            df = connector.get_historical_data('GOLD', '15m', 100)

            if not df.empty:
                print(f"✓ Got {len(df)} bars of data!")
                print(f"Date range: {df.index[0]} to {df.index[-1]}")
                print(df.head())
            else:
                print("✗ No data returned")
        else:
            print("✗ Connection failed")

    except Exception as e:
        print(f"✗ Connection/data failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"✗ Setup failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("If connection fails with 'invalid.details', you need to provide")
print("the actual Capital.com email address associated with your account.")
print("=" * 60)
