"""
Analyze what's blocking more frequent entries in the strategy
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
import pandas as pd

print("=" * 70)
print("ENTRY FREQUENCY ANALYSIS")
print("=" * 70)
print()

# Load recent SILVER data (most active asset)
loader = DatabentoMicroFuturesLoader()

print("Loading SILVER data (most active asset)...")
silver_15m = loader.load_symbol('SILVER', start_date='2025-12-01', end_date='2026-01-29', resample='15min')
silver_1h = loader.load_symbol('SILVER', start_date='2025-12-01', end_date='2026-01-29', resample='1h')
print(f"✓ 15min: {len(silver_15m)} bars")
print(f"✓ 1H: {len(silver_1h)} bars")
print()

# Add indicators
silver_15m['ema_21'] = calculate_ema(silver_15m['close'], period=21)
silver_15m['atr_20'] = calculate_atr(silver_15m['high'], silver_15m['low'], silver_15m['close'], period=20)

squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
    silver_15m['high'], silver_15m['low'], silver_15m['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
silver_15m['squeeze_on'] = squeeze_on
silver_15m['ttm_momentum'] = momentum

squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
    silver_1h['high'], silver_1h['low'], silver_1h['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
silver_1h['ttm_momentum'] = momentum_1h

print("=" * 70)
print("CONDITION ANALYSIS (Last 2 months SILVER)")
print("=" * 70)
print()

# Analyze conditions
cond1_count = 0  # 1H bullish
cond2_count = 0  # 15M squeeze
cond3_exact = 0  # Price EXACTLY at -1 ATR
cond3_within_05 = 0  # Price within 0.5 ATR of -1 ATR
cond3_within_10 = 0  # Price within 1.0 ATR of -1 ATR
all_conditions = 0

for i in range(50, len(silver_15m)):
    bar_15m = silver_15m.iloc[i]
    timestamp = silver_15m.index[i]

    # Find 1H bar
    bars_1h = silver_1h[silver_1h.index <= timestamp]
    if len(bars_1h) == 0:
        continue
    bar_1h = bars_1h.iloc[-1]

    # Condition 1: 1H bullish
    if bar_1h['ttm_momentum'] > 0:
        cond1_count += 1

    # Condition 2: 15M squeeze
    if bar_15m['squeeze_on']:
        cond2_count += 1

    # Condition 3: Price near -1 ATR (check different tolerances)
    ema = bar_15m['ema_21']
    atr = bar_15m['atr_20']

    if pd.isna(ema) or pd.isna(atr):
        continue

    entry_level = ema - atr
    bar_low = bar_15m['low']
    bar_high = bar_15m['high']

    # Exact touch (current logic)
    if bar_low <= entry_level and bar_high >= entry_level - (0.5 * atr):
        cond3_exact += 1

    # Within 0.5 ATR
    distance = (bar_low - entry_level) / atr
    if distance <= 0.5:
        cond3_within_05 += 1

    # Within 1.0 ATR
    if distance <= 1.0:
        cond3_within_10 += 1

    # All conditions (current logic)
    if (bar_1h['ttm_momentum'] > 0 and
        bar_15m['squeeze_on'] and
        bar_low <= entry_level and bar_high >= entry_level - (0.5 * atr)):
        all_conditions += 1

total = len(silver_15m) - 50

print(f"Total bars analyzed: {total}")
print()
print("CONDITION FREQUENCY:")
print(f"  Condition 1 (1H bullish):              {cond1_count:4d} bars ({cond1_count/total*100:5.1f}%)")
print(f"  Condition 2 (15M squeeze active):      {cond2_count:4d} bars ({cond2_count/total*100:5.1f}%)")
print()
print("CONDITION 3 - PRICE LEVELS:")
print(f"  EXACT touch at -1 ATR (current):       {cond3_exact:4d} bars ({cond3_exact/total*100:5.1f}%)")
print(f"  Within 0.5 ATR of -1 ATR:              {cond3_within_05:4d} bars ({cond3_within_05/total*100:5.1f}%)")
print(f"  Within 1.0 ATR of -1 ATR:              {cond3_within_10:4d} bars ({cond3_within_10/total*100:5.1f}%)")
print()
print(f"ALL CONDITIONS TOGETHER (current):       {all_conditions:4d} bars ({all_conditions/total*100:5.1f}%)")
print()

# Calculate potential trades with looser entry
potential_05 = 0
potential_10 = 0

for i in range(50, len(silver_15m)):
    bar_15m = silver_15m.iloc[i]
    timestamp = silver_15m.index[i]

    bars_1h = silver_1h[silver_1h.index <= timestamp]
    if len(bars_1h) == 0:
        continue
    bar_1h = bars_1h.iloc[-1]

    ema = bar_15m['ema_21']
    atr = bar_15m['atr_20']

    if pd.isna(ema) or pd.isna(atr):
        continue

    entry_level = ema - atr
    bar_low = bar_15m['low']
    distance = (bar_low - entry_level) / atr

    # If 1H bullish + squeeze + within 0.5 ATR
    if (bar_1h['ttm_momentum'] > 0 and
        bar_15m['squeeze_on'] and
        distance <= 0.5):
        potential_05 += 1

    # If 1H bullish + squeeze + within 1.0 ATR
    if (bar_1h['ttm_momentum'] > 0 and
        bar_15m['squeeze_on'] and
        distance <= 1.0):
        potential_10 += 1

print("=" * 70)
print("POTENTIAL IMPROVEMENT")
print("=" * 70)
print()
print(f"Current strategy (exact -1 ATR):         {all_conditions:4d} bars ({all_conditions/total*100:5.1f}%)")
print(f"With 0.5 ATR tolerance:                  {potential_05:4d} bars ({potential_05/total*100:5.1f}%)")
print(f"With 1.0 ATR tolerance:                  {potential_10:4d} bars ({potential_10/total*100:5.1f}%)")
print()

# Calculate daily frequency
days = (silver_15m.index[-1] - silver_15m.index[50]).days
print(f"Period: {days} days")
print()
print("ESTIMATED TRADES PER DAY:")
print(f"  Current (exact -1 ATR):     {all_conditions/days:.2f} trades/day")
print(f"  With 0.5 ATR tolerance:     {potential_05/days:.2f} trades/day")
print(f"  With 1.0 ATR tolerance:     {potential_10/days:.2f} trades/day")
print()

# Show what percentage of bars have each condition failing
print("=" * 70)
print("CONDITION BOTTLENECKS")
print("=" * 70)
print()

passes_1h_squeeze = 0
for i in range(50, len(silver_15m)):
    bar_15m = silver_15m.iloc[i]
    timestamp = silver_15m.index[i]

    bars_1h = silver_1h[silver_1h.index <= timestamp]
    if len(bars_1h) == 0:
        continue
    bar_1h = bars_1h.iloc[-1]

    if bar_1h['ttm_momentum'] > 0 and bar_15m['squeeze_on']:
        passes_1h_squeeze += 1

print(f"Bars passing 1H bullish + 15M squeeze:   {passes_1h_squeeze:4d} bars ({passes_1h_squeeze/total*100:5.1f}%)")
print(f"  But blocked by exact -1 ATR touch:     {passes_1h_squeeze - all_conditions:4d} bars")
print()
print("CONCLUSION:")
print(f"  {(passes_1h_squeeze - all_conditions)/total*100:.1f}% of bars have right conditions but price doesn't touch -1 ATR exactly")
print()

print("=" * 70)
