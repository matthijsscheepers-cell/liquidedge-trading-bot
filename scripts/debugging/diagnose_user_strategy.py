"""
Diagnose: Waarom geen trades met de user's strategie?

Check elke conditie afzonderlijk:
1. 1H momentum bullish?
2. 15min momentum bullish?
3. 15min squeeze active?
4. Price near -1 ATR?
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
import pandas as pd

print("=" * 70)
print("DIAGNOSE USER STRATEGY")
print("=" * 70)
print()

# Load data
loader = DatabentoMicroFuturesLoader()

print("Laden data...")
gold_15m = loader.load_symbol('GOLD', start_date='2026-01-01', end_date='2026-01-29', resample='15min')
gold_1h = loader.load_symbol('GOLD', start_date='2026-01-01', end_date='2026-01-29', resample='1h')
print(f"✓ 15min: {len(gold_15m)} bars")
print(f"✓ 1H: {len(gold_1h)} bars")
print()

# Add indicators (matching user's KC settings: 21-EMA, 20-ATR)
gold_15m['ema_21'] = calculate_ema(gold_15m['close'], period=21)
gold_15m['atr_20'] = calculate_atr(gold_15m['high'], gold_15m['low'], gold_15m['close'], period=20)

squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
    gold_15m['high'], gold_15m['low'], gold_15m['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
gold_15m['squeeze_on'] = squeeze_on
gold_15m['ttm_momentum'] = momentum

squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
    gold_1h['high'], gold_1h['low'], gold_1h['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
gold_1h['ttm_momentum'] = momentum_1h

print("=" * 70)
print("ANALYSE (laatste 500 bars)")
print("=" * 70)
print()

# First, let's look at a few sample bars to understand the numbers
print("Sample bars (laatste 10 bars):")
print("-" * 70)
for i in range(-10, 0):
    bar = gold_15m.iloc[i]
    ema = bar['ema_21']
    atr = bar['atr_20']
    entry_level = ema - atr

    print(f"{gold_15m.index[i]}")
    print(f"  Close: ${bar['close']:.2f}")
    print(f"  Low: ${bar['low']:.2f}, High: ${bar['high']:.2f}")
    print(f"  EMA21: ${ema:.2f}, ATR: ${atr:.2f}")
    print(f"  Entry level (-1 ATR): ${entry_level:.2f}")
    print(f"  Distance: Low is ${bar['low'] - entry_level:.2f} above entry")
    print()

print("=" * 70)
print()

# Analyze ALL bars (not just last 500)
print("Analyzing ALL bars to find ANY that touch -1 ATR...")
print()

touch_count = 0
touch_examples = []

for i in range(50, len(gold_15m)):  # Start at 50 to have enough history
    bar = gold_15m.iloc[i]
    ema = bar['ema_21']
    atr = bar['atr_20']

    if pd.isna(ema) or pd.isna(atr):
        continue

    entry_level = ema - atr
    bar_low = bar['low']
    bar_high = bar['high']

    if bar_low <= entry_level and bar_high >= entry_level - (0.5 * atr):
        touch_count += 1
        if len(touch_examples) < 5:  # Keep first 5 examples
            touch_examples.append({
                'timestamp': gold_15m.index[i],
                'close': bar['close'],
                'low': bar_low,
                'high': bar_high,
                'ema': ema,
                'atr': atr,
                'entry_level': entry_level,
            })

print(f"Total bars with -1 ATR touch: {touch_count} out of {len(gold_15m)} bars ({touch_count/len(gold_15m)*100:.2f}%)")
print()

if touch_count > 0:
    print("Examples of bars that touched -1 ATR:")
    print("-" * 70)
    for ex in touch_examples:
        print(f"{ex['timestamp']}")
        print(f"  Close: ${ex['close']:.2f}, Low: ${ex['low']:.2f}, High: ${ex['high']:.2f}")
        print(f"  EMA21: ${ex['ema']:.2f}, ATR: ${ex['atr']:.2f}")
        print(f"  Entry level: ${ex['entry_level']:.2f}")
        print()
else:
    print("⚠ NO BARS IN ENTIRE DATASET TOUCHED -1 ATR LEVEL!")
    print()
    print("This suggests the entry condition is TOO STRICT.")
    print("Options:")
    print("  1. Use a shallower pullback (e.g., -0.5 ATR instead of -1 ATR)")
    print("  2. Use wider tolerance (e.g., within 0.5 ATR of -1 ATR level)")
    print("  3. Reconsider the strategy parameters")
    print()

print("=" * 70)
print()

# Let's check with different tolerance levels
print("Checking different distance tolerances from -1 ATR...")
print()

tolerances = [0.1, 0.2, 0.3, 0.5, 1.0]
for tol in tolerances:
    count = 0
    for i in range(50, len(gold_15m)):
        bar = gold_15m.iloc[i]
        ema = bar['ema_21']
        atr = bar['atr_20']

        if pd.isna(ema) or pd.isna(atr):
            continue

        entry_level = ema - atr
        bar_low = bar['low']
        bar_high = bar['high']

        # Check if price within tolerance of entry level
        distance_from_entry = (bar_low - entry_level) / atr

        if distance_from_entry <= tol:
            count += 1

    pct = (count / len(gold_15m)) * 100
    print(f"  Within {tol:.1f} ATR of entry level: {count} bars ({pct:.1f}%)")

print()
print("=" * 70)
print()

# Analyze last 500 bars
analysis_bars = 500
recent_15m = gold_15m.iloc[-analysis_bars:]

# Check conditions
cond1_count = 0  # 1H bullish
cond2_count = 0  # 15min bullish
cond3_count = 0  # 15min squeeze active
cond4_count = 0  # Price near -1 ATR
all_cond_count = 0  # All conditions met

for i in range(len(recent_15m)):
    bar_15m = recent_15m.iloc[i]
    timestamp = recent_15m.index[i]

    # Find corresponding 1H bar
    bars_1h = gold_1h[gold_1h.index <= timestamp]
    if len(bars_1h) == 0:
        continue

    bar_1h = bars_1h.iloc[-1]

    # Condition 1: 1H momentum bullish (> 0)
    if bar_1h['ttm_momentum'] > 0:
        cond1_count += 1

    # Condition 2: 15min momentum bullish (> 0)
    if bar_15m['ttm_momentum'] > 0:
        cond2_count += 1

    # Condition 3: 15min squeeze active
    if bar_15m['squeeze_on']:
        cond3_count += 1

    # Condition 4: Bar touches -1 ATR
    ema_21 = bar_15m['ema_21']
    atr = bar_15m['atr_20']
    bar_low = bar_15m['low']
    bar_high = bar_15m['high']

    entry_level = ema_21 - atr  # -1 ATR

    # Bar must touch -1 ATR (low <= entry) AND not be completely below
    if bar_low <= entry_level and bar_high >= entry_level - (0.5 * atr):
        cond4_count += 1

    # All conditions
    if (bar_1h['ttm_momentum'] > 0 and
        bar_15m['ttm_momentum'] > 0 and
        bar_15m['squeeze_on'] and
        bar_low <= entry_level and bar_high >= entry_level - (0.5 * atr)):
        all_cond_count += 1

        print(f"✓ SETUP @ {timestamp}")
        print(f"  1H momentum: {bar_1h['ttm_momentum']:.3f}")
        print(f"  15M momentum: {bar_15m['ttm_momentum']:.3f}")
        print(f"  Squeeze: {bar_15m['squeeze_on']}")
        print(f"  Entry level: ${entry_level:.2f}")
        print(f"  Bar low: ${bar_low:.2f}, Bar high: ${bar_high:.2f}")
        print()

total = len(recent_15m)

print()
print("=" * 70)
print("RESULTATEN")
print("=" * 70)
print()

print(f"Total bars analyzed: {total}")
print()

print(f"Conditie 1 (1H bullish):        {cond1_count:4d} bars ({cond1_count/total*100:5.1f}%)")
print(f"Conditie 2 (15M bullish):       {cond2_count:4d} bars ({cond2_count/total*100:5.1f}%)")
print(f"Conditie 3 (15M squeeze):       {cond3_count:4d} bars ({cond3_count/total*100:5.1f}%)")
print(f"Conditie 4 (Price @ -1 ATR):    {cond4_count:4d} bars ({cond4_count/total*100:5.1f}%)")
print()
print(f"ALLE CONDITIES SAMEN:           {all_cond_count:4d} bars ({all_cond_count/total*100:5.1f}%)")
print()

if all_cond_count == 0:
    print("=" * 70)
    print("PROBLEEM ANALYSE")
    print("=" * 70)
    print()

    # Find weakest link
    conditions = [
        ('1H bullish', cond1_count),
        ('15M bullish', cond2_count),
        ('15M squeeze', cond3_count),
        ('Price @ -1 ATR', cond4_count),
    ]

    weakest = min(conditions, key=lambda x: x[1])

    print(f"Zwakste schakel: {weakest[0]} ({weakest[1]} bars)")
    print()

    if weakest[0] == '15M squeeze':
        print("DIAGNOSE: Squeeze komt te weinig voor")
        print(f"  Squeeze is slechts {cond3_count/total*100:.1f}% van de tijd actief")
        print(f"  Mogelijke oplossing: Verruim squeeze detectie parameters")
    elif weakest[0] == 'Price @ -1 ATR':
        print("DIAGNOSE: Prijs komt zelden exact bij -1 ATR")
        print(f"  Prijs is slechts {cond4_count/total*100:.1f}% van de tijd binnen 0.2 ATR van entry level")
        print(f"  Mogelijke oplossing: Verruim tolerance (bijv. 0.5 ATR)")
    elif 'bullish' in weakest[0]:
        print(f"DIAGNOSE: {weakest[0]} momentum komt te weinig voor")
        print(f"  Momentum is slechts {weakest[1]/total*100:.1f}% van de tijd bullish")
        print(f"  Mogelijke oplossing: Verlaag momentum threshold")

print()
print("=" * 70)
