"""
Capital.com Credentials Test

This script helps you verify your Capital.com API credentials before running backtests.

Required Information:
1. API Key - Your Capital.com API key
2. Password - Your Capital.com account password
3. Email/Identifier - The email address you used to register your Capital.com account

Where to find these:
- Log into Capital.com web platform
- Go to Settings > API Keys
- Generate or view your API key
- Use the SAME email and password you use to log into the web platform
"""

import sys
sys.path.insert(0, '.')

from src.execution.capital_connector import CapitalConnector

print("=" * 70)
print("CAPITAL.COM CREDENTIALS TEST")
print("=" * 70)
print()

# Get credentials from user
print("Please provide your Capital.com credentials:")
print()

api_key = input("API Key: ").strip()
password = input("Password: ").strip()
identifier = input("Email/Identifier: ").strip()

if not api_key or not password or not identifier:
    print("\n✗ Error: All three fields (API Key, Password, Email) are required.")
    sys.exit(1)

print()
print("=" * 70)
print("Testing connection...")
print("=" * 70)
print()

try:
    config = {
        'api_key': api_key,
        'password': password,
        'identifier': identifier
    }

    connector = CapitalConnector(config)

    # Try to connect
    result = connector.connect()

    if result:
        print()
        print("=" * 70)
        print("✓ CONNECTION SUCCESSFUL!")
        print("=" * 70)
        print()
        print("Your credentials are working correctly.")
        print()
        print("You can now run backtests using:")
        print()
        print(f"python3 scripts/run_backtest.py --mode capital \\")
        print(f"  --symbols GOLD SILVER US100 US500 \\")
        print(f"  --timeframe 15m \\")
        print(f"  --start 2024-10-01 --end 2024-12-31 \\")
        print(f"  --api-key \"{api_key}\" \\")
        print(f"  --password \"{password}\" \\")
        print(f"  --identifier \"{identifier}\"")
        print()

        # Try to fetch sample data
        print("Testing data retrieval...")
        try:
            df = connector.get_historical_data('GOLD', '15m', 100)
            if not df.empty:
                print(f"✓ Successfully fetched {len(df)} bars of GOLD 15m data")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            else:
                print("⚠ Warning: Connected but no data returned")
        except Exception as e:
            print(f"⚠ Warning: Connected but data fetch failed: {e}")

    else:
        print("✗ Connection failed")

except Exception as e:
    print()
    print("=" * 70)
    print("✗ CONNECTION FAILED")
    print("=" * 70)
    print()
    print(f"Error: {e}")
    print()
    print("Common issues:")
    print("1. Wrong email - use the exact email you registered with Capital.com")
    print("2. Wrong password - use your Capital.com account password")
    print("3. Wrong API key - generate a new one from Capital.com settings")
    print("4. Account not verified - check your email for verification")
    print("5. Using demo credentials with live API key (or vice versa)")
    print()
    sys.exit(1)

print()
print("=" * 70)
