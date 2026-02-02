"""
Diagnostics: Waarom zo weinig trades met Databento data?
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.regime.detector import RegimeDetector
from src.strategies.selector import StrategySelector
import pandas as pd

print("=" * 70)
print("STRATEGY DIAGNOSTICS - DATABENTO DATA")
print("=" * 70)
print()

# Laad recente GOLD data (laatste maand)
loader = DatabentoMicroFuturesLoader()

print("Laden GOLD data (laatste 30 dagen, 15min bars)...")
gold = loader.load_symbol(
    'GOLD',
    start_date='2026-01-01',
    end_date='2026-01-29',
    resample='15min'
)

print(f"✓ Geladen: {len(gold)} bars")
print()

# Voeg indicatoren toe
detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

print("Indicatoren toegevoegd")
print()

# Analyseer laatste 100 bars
print("=" * 70)
print("ANALYSEER LAATSTE 100 BARS")
print("=" * 70)
print()

selector = StrategySelector('GOLD')

# Tellers voor filter redenen
filter_counts = {
    'total_bars': 0,
    'insufficient_data': 0,
    'regime_not_trending': 0,
    'session_filtered': 0,
    'news_blackout': 0,
    'adx_too_low': 0,
    'rsi_filtered': 0,
    'no_pullback': 0,
    'no_confirmation': 0,
    'setups_found': 0,
}

for i in range(-100, 0):
    filter_counts['total_bars'] += 1

    if len(gold) + i < 200:
        filter_counts['insufficient_data'] += 1
        continue

    window = gold.iloc[:i]
    bar = gold.iloc[i]
    bar_time = gold.index[i]

    # Detecteer regime
    regime, confidence, strat_rec = detector.detect_regime(window)

    # Check filters
    if regime.value not in ['STRONG_TREND', 'WEAK_TREND']:
        filter_counts['regime_not_trending'] += 1
        continue

    # Session filter check
    from src.filters.session_filter import is_liquid_session, is_news_blackout

    if not is_liquid_session(bar_time, 'GOLD'):
        filter_counts['session_filtered'] += 1
        continue

    if is_news_blackout(bar_time):
        filter_counts['news_blackout'] += 1
        continue

    # ADX check
    if bar['adx_14'] < 6:  # Min ADX for GOLD
        filter_counts['adx_too_low'] += 1
        continue

    # RSI check
    rsi = bar.get('rsi_14', None)
    if rsi is not None:
        if rsi > 70 or rsi < 30:  # Overbought/oversold
            filter_counts['rsi_filtered'] += 1
            continue

    # Check for setup
    setup = selector.check_entry(window, regime.value, confidence, strat_rec)

    if setup:
        filter_counts['setups_found'] += 1
        print(f"✓ SETUP @ {bar_time}")
        print(f"  Type: {setup.setup_type}")
        print(f"  Entry: ${setup.entry_price:.2f}")
        print(f"  RRR: {setup.reward_risk_ratio():.2f}")
        print()
    else:
        # Moet zijn door pullback of confirmation
        filter_counts['no_pullback'] += 1

print()
print("=" * 70)
print("FILTER STATISTIEKEN (laatste 100 bars)")
print("=" * 70)
print()

total = filter_counts['total_bars']

for reason, count in filter_counts.items():
    if reason == 'total_bars':
        continue
    pct = (count / total * 100) if total > 0 else 0
    print(f"{reason:25s}: {count:4d} bars ({pct:5.1f}%)")

print()
print("=" * 70)
print("CONCLUSIE")
print("=" * 70)
print()

if filter_counts['setups_found'] == 0:
    print("⚠ GEEN SETUPS gevonden in laatste 100 bars!")
    print()

    # Identificeer grootste filter
    filters_only = {k: v for k, v in filter_counts.items()
                   if k not in ['total_bars', 'setups_found', 'insufficient_data']}

    biggest_filter = max(filters_only, key=filters_only.get)

    print(f"Grootste filter: {biggest_filter} ({filter_counts[biggest_filter]} bars)")
    print()

    if biggest_filter == 'regime_not_trending':
        print("PROBLEEM: Markt wordt niet als trending geclassificeerd")
        print("OPLOSSING: Verlaag regime detection thresholds")
    elif biggest_filter == 'session_filtered':
        print("PROBLEEM: Te veel bars buiten handelssessie")
        print("OPLOSSING: Verruim sessie tijden of test op andere periode")
    elif biggest_filter == 'adx_too_low':
        print("PROBLEEM: ADX te laag (< 6)")
        print("OPLOSSING: Verlaag min_adx threshold")
    elif biggest_filter == 'no_pullback':
        print("PROBLEEM: Geen pullback patronen gevonden")
        print("OPLOSSING: Verruim pullback_tolerance parameter")

else:
    print(f"✓ {filter_counts['setups_found']} setups gevonden!")
    print(f"  Setup rate: {filter_counts['setups_found']/total*100:.1f}%")

print()
print("=" * 70)
print("AANBEVELING")
print("=" * 70)
print()
print("De huidige filters zijn zeer streng:")
print("  • Session filtering (alleen NY sessie)")
print("  • News blackout periods")
print("  • RSI overbought/oversold filters")
print("  • Lage ADX thresholds (6)")
print("  • Strikte pullback vereisten")
print()
print("Dit resulteert in weinig maar hoogwaardige trades.")
print("Voor meer trades: verlaag één of meer filters.")
