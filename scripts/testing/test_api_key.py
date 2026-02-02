"""
Capital.com API Key Diagnostics Tool
"""

import os
from dotenv import load_dotenv
from capitalcom import Client

load_dotenv()

api_key = os.getenv('CAPITAL_API_KEY')
email = os.getenv('CAPITAL_IDENTIFIER')
password = os.getenv('CAPITAL_PASSWORD')

print("=" * 70)
print("CAPITAL.COM API DIAGNOSTICS")
print("=" * 70)
print()

print("CREDENTIALS CHECK:")
print(f"  Email:    {email}")
print(f"  Password: {'*' * len(password)} ({len(password)} chars)")
print(f"  API Key:  {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''} ({len(api_key)} chars)")
print()

print("VALIDATION:")
if not api_key or len(api_key) < 10:
    print("  ⚠️  API Key seems too short (expected 16-32 chars)")
if ' ' in api_key:
    print("  ⚠️  API Key contains spaces (should be removed)")
if not email or '@' not in email:
    print("  ⚠️  Email format seems invalid")

print()
print("ATTEMPTING CONNECTION:")
print("-" * 70)

try:
    client = Client(log=email, pas=password, api_key=api_key)

    print(f"Status: {client.response.status_code}")

    try:
        response_data = client.response.json()
        print(f"Response: {response_data}")

        if 'errorCode' in response_data:
            error = response_data['errorCode']
            print()
            print("ERROR DIAGNOSIS:")

            if error == 'error.invalid.api.key':
                print("  ❌ API Key is INVALID")
                print()
                print("  Possible causes:")
                print("    1. Key was copied incorrectly (check for spaces/extra chars)")
                print("    2. Key is for LIVE account but you're using DEMO mode")
                print("    3. Key hasn't been activated in Capital.com portal")
                print("    4. Key was regenerated/revoked")
                print()
                print("  Action: Please go to Capital.com → Settings → API")
                print("          and REGENERATE a new Demo API key")

            elif error == 'error.invalid.details':
                print("  ❌ Email or Password is INCORRECT")
                print()
                print("  Action: Verify your login credentials")

    except:
        print(f"Raw response: {client.response.text}")

    if hasattr(client, 'cst') and client.cst:
        print()
        print("✅ SUCCESS! Connection established")
        print(f"   Session token: {client.cst[:30]}...")
    else:
        print()
        print("❌ FAILED: No session token received")

except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
