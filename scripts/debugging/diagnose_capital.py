"""
Capital.com Authentication Diagnostic Tool

This script helps diagnose authentication issues by testing different scenarios.
"""

import sys
sys.path.insert(0, '.')

from capitalcom import Client

print("=" * 70)
print("CAPITAL.COM AUTHENTICATION DIAGNOSTICS")
print("=" * 70)
print()

# Test credentials
api_key = 'Hh6exwe6rrnYDgQe'
password = 'Vergeten22!'
email = 'matthijsscheepers@gmail.com'

print("Testing with provided credentials:")
print(f"  API Key: {api_key[:10]}...")
print(f"  Email: {email}")
print(f"  Password: {'*' * len(password)}")
print()

# Test 1: Direct client creation
print("=" * 70)
print("Test 1: Direct Client Creation")
print("=" * 70)

try:
    client = Client(
        log=email,
        pas=password,
        api_key=api_key
    )

    print(f"Status Code: {client.response.status_code}")

    if client.response.status_code == 200:
        print("✓ Authentication successful!")
        print(f"  CST Token: {client.cst[:20]}...")
        print(f"  Security Token: {client.x_security_token[:20]}...")

        # Try to get accounts
        print("\nTrying to fetch account information...")
        accounts = client.all_accounts()
        print(f"✓ Accounts: {accounts}")

    else:
        print("✗ Authentication failed")
        try:
            error_data = client.response.json()
            print(f"  Error: {error_data}")

            error_code = error_data.get('errorCode', '')

            if error_code == 'error.invalid.details':
                print()
                print("DIAGNOSIS:")
                print("  The credentials don't match. Please verify:")
                print("  1. Is this the correct email you used to register?")
                print("  2. Is this the correct password for your Capital.com account?")
                print("  3. Is your API key from the same account (not demo)?")
                print()
                print("TROUBLESHOOTING STEPS:")
                print("  1. Log into Capital.com website with these credentials")
                print("  2. Go to Settings > API Keys")
                print("  3. Generate a NEW API key")
                print("  4. Use that new API key with the same email/password")

            elif error_code == 'error.invalid.api.key':
                print()
                print("DIAGNOSIS:")
                print("  The API key is invalid or expired.")
                print()
                print("TROUBLESHOOTING STEPS:")
                print("  1. Log into Capital.com")
                print("  2. Go to Settings > API Keys")
                print("  3. Delete the old API key")
                print("  4. Generate a NEW API key")
                print("  5. Try again with the new key")

        except:
            print(f"  Response: {client.response.text}")

except Exception as e:
    print(f"✗ Client creation failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print()
print("If authentication is failing:")
print()
print("1. Verify you can log into Capital.com website with:")
print(f"   Email: {email}")
print(f"   Password: (same password you provided)")
print()
print("2. Once logged in, go to: Settings > API > API Keys")
print()
print("3. Generate a NEW API key (delete old one if exists)")
print()
print("4. Make sure you're using the LIVE account, not DEMO")
print("   (or use demo credentials if you want demo data)")
print()
print("5. Try the connection again with the new API key")
print()
