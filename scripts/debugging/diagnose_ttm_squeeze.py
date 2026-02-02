"""
Diagnostics: Waarom geen TTM Squeeze trades?
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.regime.detector import RegimeDetector
import pandas as pd

print("=" * 70)
print("TTM SQUEEZE DIAGNOSTICS")
print("=" * 70)
print()

# Laad data
loader = DatabentoMicroFuturesLoader()
gold = loader.load_symbol('GOLD', start_date='2025-12-01', end_date='2026-01-29', resample='15min')

print(f"Data: {len(gold)} bars")
print()

# Voeg indicatoren toe
detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

# Analyseer squeeze condities
print("=" * 70)
print("SQUEEZE ANALYSE")
print("=" * 70)
print()

squeeze_bars = gold[gold['squeeze_on'] == True]
print(f"Totaal bars: {len(gold)}")
print(f"Squeeze bars: {len(squeeze_bars)} ({len(squeeze_bars)/len(gold)*100:.1f}%)")
print()

if len(squeeze_bars) > 0:
    print("Squeeze durations:")
    print(f"  Min: {squeeze_bars['squeeze_duration'].min():.0f} bars")
    print(f"  Max: {squeeze_bars['squeeze_duration'].max():.0f} bars")
    print(f"  Avg: {squeeze_bars['squeeze_duration'].mean():.0f} bars")
    print()

    # Bars met squeeze duration >= 5
    long_squeeze = squeeze_bars[squeeze_bars['squeeze_duration'] >= 5]
    print(f"Long squeezes (>= 5 bars): {len(long_squeeze)} ({len(long_squeeze)/len(gold)*100:.1f}%)")
    print()

# Check regime recommendations
print("=" * 70)
print("REGIME AANBEVELINGEN (laatste 200 bars)")
print("=" * 70)
print()

regime_counts = {}
strategy_counts = {}

for i in range(-200, 0):
    if len(gold) + i < 200:
        continue

    window = gold.iloc[:i]
    regime, confidence, strat_rec = detector.detect_regime(window)

    regime_str = regime.value
    regime_counts[regime_str] = regime_counts.get(regime_str, 0) + 1
    strategy_counts[strat_rec] = strategy_counts.get(strat_rec, 0) + 1

print("Regime verdeling:")
for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / 200 * 100
    print(f"  {regime:25s}: {count:3d} bars ({pct:5.1f}%)")

print()
print("Strategy aanbevelingen:")
for strat, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
    pct = count / 200 * 100
    print(f"  {strat:25s}: {count:3d} bars ({pct:5.1f}%)")

print()

# Zoek voorbeelden van TTM Squeeze setups
print("=" * 70)
print("TTM SQUEEZE VOORBEELDEN")
print("=" * 70)
print()

squeeze_setups = []

for i in range(-500, 0):
    if len(gold) + i < 200:
        continue

    window = gold.iloc[:i]
    bar = gold.iloc[i]
    regime, confidence, strat_rec = detector.detect_regime(window)

    if strat_rec in ['TTM_SQUEEZE', 'TTM_BREAKOUT']:
        squeeze_setups.append({
            'time': gold.index[i],
            'regime': regime.value,
            'strat': strat_rec,
            'confidence': confidence,
            'squeeze_on': bar['squeeze_on'],
            'squeeze_duration': bar['squeeze_duration'],
            'momentum': bar['ttm_momentum'],
            'compression': bar['compression_score']
        })

if squeeze_setups:
    print(f"Gevonden: {len(squeeze_setups)} TTM Squeeze opportunities (laatste 500 bars)")
    print()
    print("Eerste 10:")
    for setup in squeeze_setups[:10]:
        print(f"\n{setup['time']}")
        print(f"  Regime: {setup['regime']}")
        print(f"  Strategy: {setup['strat']}")
        print(f"  Confidence: {setup['confidence']:.0f}%")
        print(f"  Squeeze on: {setup['squeeze_on']}")
        print(f"  Duration: {setup['squeeze_duration']:.0f} bars")
        print(f"  Momentum: {setup['momentum']:.4f}")
        print(f"  Compression: {setup['compression']:.1f}")
else:
    print("⚠ GEEN TTM Squeeze opportunities gevonden!")
    print()
    print("Mogelijke redenen:")
    print("  1. Squeeze komt te weinig voor")
    print("  2. Squeeze duration < 5 bars")
    print("  3. Compression score te laag")
    print("  4. Momentum niet sterk genoeg bij release")

print()
print("=" * 70)
