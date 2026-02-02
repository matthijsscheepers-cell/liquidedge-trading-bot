"""
Quick test of Databento loader - just load last 7 days
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader

print("=" * 70)
print("DATABENTO LOADER QUICK TEST")
print("=" * 70)
print()

# Initialize
loader = DatabentoMicroFuturesLoader()

# Load recent GOLD data (last 7 days, 15-minute bars)
print("Loading GOLD 15-minute bars (last 7 days)...")
print()

gold_15m = loader.load_symbol(
    'GOLD',
    start_date='2026-01-23',
    end_date='2026-01-29',
    resample='15min'
)

print()
print(f"✓ Loaded {len(gold_15m)} bars")
print()
print("First 10 bars:")
print(gold_15m.head(10))
print()
print("Last 10 bars:")
print(gold_15m.tail(10))
print()
print(f"Price range: ${gold_15m['close'].min():.2f} - ${gold_15m['close'].max():.2f}")
print(f"Current price: ${gold_15m['close'].iloc[-1]:.2f}")
print()
print("=" * 70)
print("SUCCESS!")
print("=" * 70)
