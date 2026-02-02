"""
Test: Officiële PineScript TTM Squeeze vs Onze Gecalibreerde Versie

Vergelijk:
1. Onze versie: BB std 2.5, KC mult 0.5 (EMA basis, Wilder's ATR)
2. PineScript: BB std 2.0, KC mult 1.0 (SMA basis, SMA van TR)
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.ttm import calculate_ttm_squeeze, calculate_ttm_squeeze_pinescript
import pandas as pd

print("=" * 70)
print("TTM SQUEEZE: PINESCRIPT VS CALIBRATED VERSIE")
print("=" * 70)
print()

# Laad GOLD data
loader = DatabentoMicroFuturesLoader()
print("Laden GOLD data...")
gold = loader.load_symbol('GOLD', start_date='2026-01-01', end_date='2026-01-29', resample='15min')
print(f"✓ {len(gold)} bars geladen")
print()

# Test beide versies
print("=" * 70)
print("VERSIE 1: ONZE GECALIBREERDE VERSIE")
print("=" * 70)
print()
print("Parameters:")
print("  BB std: 2.5")
print("  KC multiplier: 0.5")
print("  KC basis: EMA(close, 20)")
print("  KC range: Wilder's ATR")
print()

squeeze_on_ours, momentum_ours, color_ours = calculate_ttm_squeeze(
    gold['high'], gold['low'], gold['close'],
    bb_period=20, bb_std=2.5,
    kc_period=20, kc_atr_period=20, kc_multiplier=0.5,
    momentum_period=12
)

total_bars = len(squeeze_on_ours)
squeeze_bars_ours = squeeze_on_ours.sum()
squeeze_pct_ours = (squeeze_bars_ours / total_bars * 100) if total_bars > 0 else 0

# Count releases
releases_ours = ((squeeze_on_ours.shift(1) == True) & (squeeze_on_ours == False)).sum()

print(f"Squeeze: {squeeze_bars_ours}/{total_bars} bars ({squeeze_pct_ours:.1f}%)")
print(f"Releases: {releases_ours}")
print()

# Test PineScript versie
print("=" * 70)
print("VERSIE 2: OFFICIËLE PINESCRIPT VERSIE")
print("=" * 70)
print()
print("Parameters:")
print("  BB std: 2.0")
print("  KC multiplier: 1.0")
print("  KC basis: SMA(close, 20)")
print("  KC range: SMA(TR, 20)")
print()

squeeze_on_pine, momentum_pine, color_pine = calculate_ttm_squeeze_pinescript(
    gold['high'], gold['low'], gold['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=1.0,
    momentum_period=12
)

squeeze_bars_pine = squeeze_on_pine.sum()
squeeze_pct_pine = (squeeze_bars_pine / total_bars * 100) if total_bars > 0 else 0

# Count releases
releases_pine = ((squeeze_on_pine.shift(1) == True) & (squeeze_on_pine == False)).sum()

print(f"Squeeze: {squeeze_bars_pine}/{total_bars} bars ({squeeze_pct_pine:.1f}%)")
print(f"Releases: {releases_pine}")
print()

# Vergelijking
print("=" * 70)
print("VERGELIJKING")
print("=" * 70)
print()

print(f"{'Metric':<30} {'Ours':>15} {'PineScript':>15} {'Verschil':>15}")
print("-" * 77)
print(f"{'Squeeze percentage':<30} {squeeze_pct_ours:>14.1f}% {squeeze_pct_pine:>14.1f}% {squeeze_pct_pine - squeeze_pct_ours:>14.1f}%")
print(f"{'Squeeze releases':<30} {releases_ours:>15d} {releases_pine:>15d} {releases_pine - releases_ours:>15d}")
print()

# Ideal target: 15-25% squeeze
print("TARGET: 15-25% squeeze (ideal)")
print()

if 15 <= squeeze_pct_ours <= 25:
    print(f"✓ Onze versie: {squeeze_pct_ours:.1f}% - BINNEN TARGET")
else:
    print(f"✗ Onze versie: {squeeze_pct_ours:.1f}% - BUITEN TARGET")

if 15 <= squeeze_pct_pine <= 25:
    print(f"✓ PineScript: {squeeze_pct_pine:.1f}% - BINNEN TARGET")
else:
    print(f"✗ PineScript: {squeeze_pct_pine:.1f}% - BUITEN TARGET")

print()
print("=" * 70)
print("CONCLUSIE")
print("=" * 70)
print()

if abs(squeeze_pct_ours - 20) < abs(squeeze_pct_pine - 20):
    print("✓ Onze gecalibreerde versie is dichter bij het ideaal (20%)")
    print(f"  Verschil: {abs(squeeze_pct_ours - 20):.1f}% vs {abs(squeeze_pct_pine - 20):.1f}%")
else:
    print("✓ PineScript versie is dichter bij het ideaal (20%)")
    print(f"  Verschil: {abs(squeeze_pct_pine - 20):.1f}% vs {abs(squeeze_pct_ours - 20):.1f}%")

print()
print("Volgende stap: Backtest beide versies om performance te vergelijken")
print("=" * 70)
