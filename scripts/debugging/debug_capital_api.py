"""
Debug Capital.com API response
"""

import sys
sys.path.insert(0, '.')

from capitalcom import Client
from capitalcom.client_demo import ResolutionType
import pandas as pd

config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

print("Connecting...")
client = Client(
    log=config['identifier'],
    pas=config['password'],
    api_key=config['api_key']
)

print(f"✓ Connected (CST token: {client.cst[:20]}...)")
print()

print("Fetching GOLD historical data...")
data = client.historical_price(
    epic='GOLD',
    resolution=ResolutionType.MINUTE_15,
    max=10
)

print(f"\nRaw API response:")
print(f"  Type: {type(data)}")
print(f"  Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
print()

if 'prices' in data:
    prices = data['prices']
    print(f"Prices list length: {len(prices)}")
    print()
    print("First price entry:")
    print(prices[0] if prices else "Empty")
    print()
    print("Price entry keys:")
    print(prices[0].keys() if prices else "Empty")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(prices)
    print("DataFrame columns:")
    print(df.columns.tolist())
    print()
    print("First 3 rows (raw):")
    print(df.head(3))
    print()
    print("Data types:")
    print(df.dtypes)
