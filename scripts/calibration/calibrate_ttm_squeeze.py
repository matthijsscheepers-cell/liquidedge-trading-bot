"""
Calibrate TTM Squeeze parameters for intraday futures data

Problem: 99.5% of bars have squeeze_on = True (abnormal!)
Expected: 10-30% of bars should have squeeze_on = True

Root cause: Default parameters designed for daily stock data, not 15-min futures
- BB std: 2.0
- KC multiplier: 1.5

For intraday futures, we need narrower Keltner Channels or wider Bollinger Bands.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.volatility import calculate_bollinger_bands, calculate_keltner_channels, detect_squeeze
import pandas as pd

print("=" * 70)
print("TTM SQUEEZE PARAMETER CALIBRATION")
print("=" * 70)
print()

# Load recent GOLD data
loader = DatabentoMicroFuturesLoader()
gold = loader.load_symbol('GOLD', start_date='2026-01-01', end_date='2026-01-29', resample='15min')

print(f"Loaded: {len(gold)} bars of GOLD (15min)")
print()

# Test different parameter combinations
test_configs = [
    # (bb_std, kc_multiplier, description)
    (2.0, 1.5, "Default (stock daily)"),
    (2.0, 1.0, "Narrower KC (1.0x)"),
    (2.0, 0.8, "Much narrower KC (0.8x)"),
    (2.5, 1.5, "Wider BB (2.5 std)"),
    (3.0, 1.5, "Much wider BB (3.0 std)"),
    (2.5, 1.0, "Balanced (BB 2.5, KC 1.0)"),
    (2.0, 1.2, "Slightly narrower KC (1.2x)"),
]

print("=" * 70)
print("TESTING PARAMETER COMBINATIONS")
print("=" * 70)
print()

results = []

for bb_std, kc_mult, desc in test_configs:
    # Calculate indicators
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        gold['close'], period=20, num_std=bb_std
    )

    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        gold['high'], gold['low'], gold['close'],
        period=20, atr_period=20, atr_multiplier=kc_mult
    )

    # Detect squeeze
    squeeze_on = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

    # Calculate stats
    total_bars = len(squeeze_on)
    squeeze_bars = squeeze_on.sum()
    squeeze_pct = (squeeze_bars / total_bars * 100) if total_bars > 0 else 0

    # Count squeeze releases
    squeeze_releases = ((squeeze_on.shift(1) == True) & (squeeze_on == False)).sum()

    # Average squeeze duration
    durations = []
    current_duration = 0
    for val in squeeze_on:
        if val:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

    avg_duration = sum(durations) / len(durations) if durations else 0

    results.append({
        'bb_std': bb_std,
        'kc_mult': kc_mult,
        'description': desc,
        'squeeze_pct': squeeze_pct,
        'releases': squeeze_releases,
        'avg_duration': avg_duration,
    })

    print(f"{desc:30s}")
    print(f"  BB std: {bb_std:.1f}, KC mult: {kc_mult:.1f}")
    print(f"  Squeeze: {squeeze_bars}/{total_bars} bars ({squeeze_pct:.1f}%)")
    print(f"  Releases: {squeeze_releases}")
    print(f"  Avg duration: {avg_duration:.1f} bars")
    print()

print()
print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()

# Find best config (10-30% squeeze)
best = None
best_diff = float('inf')

for r in results:
    # Target is 15-25% squeeze (middle of ideal range)
    target = 20
    diff = abs(r['squeeze_pct'] - target)

    if diff < best_diff:
        best_diff = diff
        best = r

if best:
    print(f"✓ BEST CONFIG: {best['description']}")
    print(f"  BB std: {best['bb_std']:.1f}")
    print(f"  KC multiplier: {best['kc_mult']:.1f}")
    print(f"  Squeeze %: {best['squeeze_pct']:.1f}% (target: 15-25%)")
    print(f"  Squeeze releases: {best['releases']}")
    print(f"  Average duration: {best['avg_duration']:.1f} bars")
    print()
    print(f"To apply this configuration, update src/regime/detector.py:")
    print(f"  bb_std: float = {best['bb_std']}")
    print(f"  kc_multiplier: float = {best['kc_mult']}")

print()
print("=" * 70)
print("EXPLANATION")
print("=" * 70)
print()
print("TTM Squeeze ON = Bollinger Bands INSIDE Keltner Channels")
print("TTM Squeeze OFF = Bollinger Bands OUTSIDE Keltner Channels")
print()
print("For healthy squeeze behavior:")
print("  • 10-30% of bars should be in squeeze (low volatility)")
print("  • 70-90% of bars should be out of squeeze (normal/high volatility)")
print("  • Squeezes should last 5-20 bars on average")
print("  • Plenty of squeeze releases for trading opportunities")
print()
print("Current problem: 99.5% squeeze = Keltner Channels too wide for futures")
print("Solution: Narrow KC (reduce multiplier) or widen BB (increase std)")
