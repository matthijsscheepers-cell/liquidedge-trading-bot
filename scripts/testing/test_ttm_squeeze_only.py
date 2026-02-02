"""
Test: Pure TTM Squeeze trades zonder regime filtering

Dit script test wat er gebeurt als we ALLEEN TTM Squeeze trades nemen,
ongeacht het regime. We forceren de TTM Squeeze strategy te checken op
elke bar en negeren de regime recommendation.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceCalculator
from src.regime.detector import RegimeDetector
from src.strategies.ttm_squeeze import TTMSqueezeStrategy
import pandas as pd

print("=" * 70)
print("TTM SQUEEZE ONLY TEST - GEEN REGIME FILTERING")
print("=" * 70)
print()

# Configuratie
START_DATE = '2025-11-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0

print(f"Periode: {START_DATE} tot {END_DATE}")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print()

# Laad GOLD data
loader = DatabentoMicroFuturesLoader()
print("Laden GOLD data...")
gold = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE, resample='15min')
print(f"✓ {len(gold)} bars geladen")
print()

# Voeg indicatoren toe
detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

# Maak TTM Squeeze strategy
ttm_strategy = TTMSqueezeStrategy('GOLD')

# Scan alle bars voor TTM Squeeze setups
print("=" * 70)
print("SCANNEN VOOR TTM SQUEEZE SETUPS")
print("=" * 70)
print()

setups_found = []
squeeze_releases = 0
squeeze_retests = 0

for i in range(200, len(gold)):  # Start na 200 bars voor indicatoren
    window = gold.iloc[:i+1]

    # Forceer TTM Squeeze check (regime = RANGE_COMPRESSION zodat strategy werkt)
    # Nu we de regime filter hebben uitgeschakeld, kunnen we elke regime gebruiken
    setup = ttm_strategy.check_entry(window, regime='ANY', confidence=80.0)

    if setup:
        setups_found.append({
            'timestamp': gold.index[i],
            'setup_type': setup.setup_type,
            'direction': setup.direction.value,
            'entry_price': setup.entry_price,
            'stop_loss': setup.stop_loss,
            'target': setup.target,
            'rrr': setup.reward_risk_ratio(),
        })

        if 'RELEASE' in setup.setup_type:
            squeeze_releases += 1
        elif 'RETEST' in setup.setup_type:
            squeeze_retests += 1

print(f"Totaal setups gevonden: {len(setups_found)}")
print(f"  Squeeze releases: {squeeze_releases}")
print(f"  Squeeze retests: {squeeze_retests}")
print()

if len(setups_found) > 0:
    print("Eerste 10 setups:")
    print()

    for i, setup in enumerate(setups_found[:10]):
        print(f"{i+1}. {setup['timestamp']}")
        print(f"   Type: {setup['setup_type']}")
        print(f"   Entry: ${setup['entry_price']:.2f}")
        print(f"   Stop: ${setup['stop_loss']:.2f}")
        print(f"   Target: ${setup['target']:.2f}")
        print(f"   RRR: {setup['rrr']:.2f}")
        print()

    print()
    print("=" * 70)
    print("BACKTEST MET ALLEEN TTM SQUEEZE TRADES")
    print("=" * 70)
    print()

    # Nu run een aangepaste backtest die ALLEEN TTM Squeeze trades neemt
    # We moeten de backtest engine aanpassen om dit te doen...

    print("⚠ Om deze trades daadwerkelijk te backtesten, moeten we de")
    print("  BacktestEngine aanpassen om TTM Squeeze te forceren.")
    print()
    print(f"✓ Maar we weten nu dat er {len(setups_found)} TTM Squeeze setups zijn")
    print(f"  in de periode {START_DATE} tot {END_DATE}")
    print(f"  Dit is ~{len(setups_found)/3:.0f} trades per maand")

else:
    print("⚠ GEEN TTM Squeeze setups gevonden!")
    print()
    print("Mogelijk redenen:")
    print("  1. Momentum filters zijn te strikt")
    print("  2. Rejection candle filters zijn te strikt")
    print("  3. Squeeze komt te weinig voor (of te vaak)")
    print()

    # Debug: check squeeze statistieken
    print("Squeeze statistieken:")
    squeeze_on_count = gold['squeeze_on'].sum()
    total_bars = len(gold)
    print(f"  Squeeze ON: {squeeze_on_count}/{total_bars} bars ({squeeze_on_count/total_bars*100:.1f}%)")

    # Check for releases
    releases = ((gold['squeeze_on'].shift(1) == True) & (gold['squeeze_on'] == False)).sum()
    print(f"  Squeeze releases: {releases}")

    # Check momentum distribution
    print(f"  Momentum range: {gold['ttm_momentum'].min():.3f} to {gold['ttm_momentum'].max():.3f}")
    print(f"  Momentum abs mean: {gold['ttm_momentum'].abs().mean():.3f}")

print()
print("=" * 70)
