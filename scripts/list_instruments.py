#!/usr/bin/env python3
"""
List Available Instruments

This script demonstrates how to use the instrument mappings to convert
between common names and Capital.com EPIC codes.

Usage:
    python scripts/list_instruments.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    INSTRUMENTS,
    get_epic,
    get_name,
    list_instruments,
    get_rate_limit,
)


def main():
    """Display available instruments and rate limits."""

    print("\n" + "=" * 60)
    print("CAPITAL.COM INSTRUMENT REFERENCE")
    print("=" * 60)

    # Display indices
    print("\n📊 INDICES:")
    print("-" * 60)
    indices = ['NAS100', 'US500', 'US30', 'GER40', 'UK100', 'JPN225']
    for name in indices:
        epic = get_epic(name)
        print(f"  {name:<12} → {epic}")

    # Display commodities
    print("\n🥇 COMMODITIES:")
    print("-" * 60)
    commodities = ['XAUUSD', 'XAGUSD', 'WTIUSD', 'NATGAS']
    for name in commodities:
        epic = get_epic(name)
        print(f"  {name:<12} → {epic}")

    # Display forex
    print("\n💱 FOREX:")
    print("-" * 60)
    forex = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    for name in forex:
        epic = get_epic(name)
        print(f"  {name:<12} → {epic}")

    # Display crypto
    print("\n₿ CRYPTOCURRENCIES:")
    print("-" * 60)
    crypto = ['BTCUSD', 'ETHUSD']
    for name in crypto:
        epic = get_epic(name)
        print(f"  {name:<12} → {epic}")

    # Display rate limits
    print("\n⏱️  RATE LIMITS:")
    print("-" * 60)
    for env in ['demo', 'live']:
        limits = get_rate_limit(env)
        print(f"\n  {env.upper()} Account:")
        print(f"    Requests/minute: {limits['requests_per_minute']}")
        print(f"    Recommended interval: {limits['recommended_interval']}s")
        print(f"    Note: {limits['description']}")

    # Usage examples
    print("\n📖 USAGE EXAMPLES:")
    print("-" * 60)
    print("""
  from src.utils import get_epic, INSTRUMENTS

  # Method 1: Using get_epic()
  epic = get_epic('NAS100')  # Returns 'US_TECH_100'

  # Method 2: Direct dictionary access
  epic = INSTRUMENTS['NAS100']  # Returns 'US_TECH_100'

  # Method 3: Use in your trading bot
  connector.get_markets([get_epic('NAS100')])
""")

    print("=" * 60)
    print(f"Total instruments available: {len(INSTRUMENTS)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
