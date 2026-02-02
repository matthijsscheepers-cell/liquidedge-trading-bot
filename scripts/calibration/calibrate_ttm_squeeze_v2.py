"""
Calibrate TTM Squeeze parameters for intraday futures data - EXTREME TEST

Even with KC multiplier 0.8, we still have 95% squeeze!
Need to test more extreme values.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.volatility import calculate_bollinger_bands, calculate_keltner_channels, detect_squeeze
import pandas as pd
import numpy as np

print("=" * 70)
print("TTM SQUEEZE EXTREME CALIBRATION")
print("=" * 70)
print()

# Load recent GOLD data
loader = DatabentoMicroFuturesLoader()
gold = loader.load_symbol('GOLD', start_date='2026-01-01', end_date='2026-01-29', resample='15min')

print(f"Loaded: {len(gold)} bars of GOLD (15min)")
print()

# Test EXTREME parameter combinations
test_configs = [
    # (bb_std, kc_multiplier, description)
    (2.0, 0.5, "Very narrow KC (0.5x)"),
    (2.0, 0.3, "Extremely narrow KC (0.3x)"),
    (3.0, 0.8, "Wide BB (3.0) + Narrow KC (0.8)"),
    (3.5, 1.0, "Very wide BB (3.5 std)"),
    (4.0, 1.0, "Extremely wide BB (4.0 std)"),
    (3.0, 0.5, "Wide BB (3.0) + Very narrow KC (0.5)"),
    (2.5, 0.5, "BB 2.5, KC 0.5"),
]

print("=" * 70)
print("TESTING EXTREME PARAMETERS")
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

    # Also calculate some sample values to debug
    sample_idx = 1000
    if sample_idx < len(gold):
        sample_bb_upper = bb_upper.iloc[sample_idx]
        sample_bb_lower = bb_lower.iloc[sample_idx]
        sample_kc_upper = kc_upper.iloc[sample_idx]
        sample_kc_lower = kc_lower.iloc[sample_idx]
        sample_close = gold['close'].iloc[sample_idx]

        results.append({
            'bb_std': bb_std,
            'kc_mult': kc_mult,
            'description': desc,
            'squeeze_pct': squeeze_pct,
            'releases': squeeze_releases,
            'avg_duration': avg_duration,
            'sample_bb_range': sample_bb_upper - sample_bb_lower,
            'sample_kc_range': sample_kc_upper - sample_kc_lower,
        })

        print(f"{desc:40s}")
        print(f"  BB std: {bb_std:.1f}, KC mult: {kc_mult:.1f}")
        print(f"  Squeeze: {squeeze_bars}/{total_bars} bars ({squeeze_pct:.1f}%)")
        print(f"  Releases: {squeeze_releases}")
        print(f"  Avg duration: {avg_duration:.1f} bars")
        print(f"  Sample @ bar {sample_idx}:")
        print(f"    Close: ${sample_close:.2f}")
        print(f"    BB range: ${sample_bb_upper - sample_bb_lower:.2f} (${sample_bb_lower:.2f} - ${sample_bb_upper:.2f})")
        print(f"    KC range: ${sample_kc_upper - sample_kc_lower:.2f} (${sample_kc_lower:.2f} - ${sample_kc_upper:.2f})")
        print(f"    BB inside KC: {sample_bb_upper < sample_kc_upper and sample_bb_lower > sample_kc_lower}")
        print()

print()
print("=" * 70)
print("BEST CONFIGURATION")
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

    if best['squeeze_pct'] > 50:
        print("⚠ WARNING: Still too much squeeze! Need even more extreme parameters.")
        print("   OR there may be a fundamental issue with using TTM Squeeze on")
        print("   15-minute futures data. Consider:")
        print("   1. Using different timeframe (1-hour bars)")
        print("   2. Using different indicators for futures")
        print("   3. Accepting that futures are in compression most of the time")
    else:
        print(f"To apply this configuration, update src/regime/detector.py:")
        print(f"  bb_std: float = {best['bb_std']}")
        print(f"  kc_multiplier: float = {best['kc_mult']}")

print()
print("=" * 70)
