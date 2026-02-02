"""
Test Kaggle gold dataset loader and compare with Capital.com
"""

import sys
sys.path.insert(0, '.')

from src.execution.kaggle_data_loader import KaggleGoldLoader, compare_data_sources

print("=" * 70)
print("KAGGLE GOLD DATA LOADER TEST")
print("=" * 70)
print()

# Initialize loader
loader = KaggleGoldLoader()

# Load monthly data
print("Loading monthly data...")
monthly = loader.load_monthly_data()
print(f"✓ Loaded {len(monthly)} months of gold prices")
print()

# Show summary stats
print("Summary Statistics:")
stats = loader.summary_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

# Show first and last 5 rows
print("First 5 months:")
print(monthly.head())
print()

print("Last 5 months:")
print(monthly.tail())
print()

# Create OHLC for last 10 years
print("Creating OHLC data for 2010-2020...")
ohlc = loader.create_ohlc_from_monthly(
    start_date='2010-01-01',
    end_date='2020-12-31'
)
print(f"✓ Created {len(ohlc)} bars")
print()
print("OHLC Preview (last 10 bars):")
print(ohlc.tail(10))
print()

# Get specific price
print("Price lookup examples:")
price_2015 = loader.get_price_at_date('2015-01-01')
price_2020 = loader.get_price_at_date('2020-01-01')
print(f"  Jan 2015: ${price_2015:.2f}")
print(f"  Jan 2020: ${price_2020:.2f}")
print(f"  % Change: {((price_2020 - price_2015) / price_2015 * 100):.1f}%")
print()

# Compare data sources
compare_data_sources()

print()
print("=" * 70)
print("RECOMMENDATION FOR YOUR STRATEGY")
print("=" * 70)
print()
print("Your strategy trades 15-minute bars with:")
print("  - Session filtering (liquid hours)")
print("  - RSI momentum filters")
print("  - Intraday entries and exits")
print()
print("✓ BEST DATA SOURCE: Capital.com API")
print("  Provides real 15m bars with OHLCV data")
print("  Limited to ~500 bars but perfect for intraday")
print()
print("Alternative use for Kaggle data:")
print("  - Build a long-term trend filter")
print("  - Identify major market regimes (bull/bear cycles)")
print("  - Understand historical volatility patterns")
print("  - Validate regime detection over 70 years")
print()
