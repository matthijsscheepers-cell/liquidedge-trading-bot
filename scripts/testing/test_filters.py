"""
Test session filtering and RSI filters
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from datetime import datetime, timezone
from src.filters.session_filter import is_liquid_session, get_session_name, is_news_blackout

print("=" * 70)
print("SESSION FILTER TESTS")
print("=" * 70)
print()

# Test various times (UTC)
test_times = [
    "2026-01-30 08:00:00",  # Asian session
    "2026-01-30 12:00:00",  # London only
    "2026-01-30 14:00:00",  # NY Overlap (LIQUID)
    "2026-01-30 16:00:00",  # NY Overlap (LIQUID)
    "2026-01-30 18:00:00",  # NY Afternoon (LIQUID)
    "2026-01-30 20:00:00",  # NY Afternoon (LIQUID)
    "2026-01-30 22:00:00",  # Closed
]

for time_str in test_times:
    ts = pd.Timestamp(time_str, tz='UTC')
    session = get_session_name(ts)
    is_liquid_gold = is_liquid_session(ts, 'GOLD')
    is_liquid_us = is_liquid_session(ts, 'US100')

    print(f"{time_str} UTC")
    print(f"  Session: {session}")
    print(f"  GOLD liquid: {is_liquid_gold}")
    print(f"  US100 liquid: {is_liquid_us}")
    print()

print("=" * 70)
print("NEWS BLACKOUT TESTS")
print("=" * 70)
print()

# Test news blackout times (UTC)
news_test_times = [
    "2026-01-30 13:00:00",  # 8:30 AM EST release - BLACKOUT
    "2026-01-30 13:30:00",  # 8:30 AM EST release - BLACKOUT
    "2026-01-30 14:00:00",  # Safe
    "2026-01-30 14:30:00",  # 10:00 AM EST release - BLACKOUT
    "2026-01-30 15:00:00",  # 10:00 AM EST release - BLACKOUT
    "2026-01-30 15:30:00",  # Safe
    "2026-01-30 18:30:00",  # 2:00 PM EST release - BLACKOUT
    "2026-01-30 19:00:00",  # 2:00 PM EST release - BLACKOUT
    "2026-01-30 19:30:00",  # Safe
]

for time_str in news_test_times:
    ts = pd.Timestamp(time_str, tz='UTC')
    is_blackout = is_news_blackout(ts)
    status = "BLACKOUT" if is_blackout else "SAFE"
    print(f"{time_str} UTC - {status}")

print()
print("=" * 70)
print("RSI FILTER TEST")
print("=" * 70)
print()

from src.execution.capital_connector import CapitalConnector
from src.regime.detector import RegimeDetector

config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

connector = CapitalConnector(config)
connector.connect()

print("Fetching GOLD data...")
df = connector.get_historical_data('GOLD', '15m', 100)
print(f"✓ Got {len(df)} bars")
print()

# Add indicators including RSI
detector = RegimeDetector()
df = detector.add_all_indicators(df)

# Check if RSI was added
if 'rsi_14' in df.columns:
    print("✓ RSI indicator added successfully")
    print()
    print("Last 10 bars with RSI:")
    print(df[['close', 'rsi_14']].tail(10))
else:
    print("✗ RSI indicator NOT found in dataframe")
    print(f"Available columns: {list(df.columns)}")

print()
print("=" * 70)
print("FILTERS READY")
print("=" * 70)
