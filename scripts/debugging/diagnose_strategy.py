"""
Diagnostic script to understand why no trades are generated
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.execution.capital_connector import CapitalConnector
from src.regime.detector import RegimeDetector
from src.strategies.selector import StrategySelector

print("=" * 70)
print("STRATEGY DIAGNOSTIC")
print("=" * 70)
print()

# Connect and get data
config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

connector = CapitalConnector(config)
connector.connect()

# Get GOLD data
print("Fetching GOLD 15m data...")
df = connector.get_historical_data('GOLD', '15m', 300)
print(f"✓ Got {len(df)} bars")
print()

# Add indicators
regime_detector = RegimeDetector()
df = regime_detector.add_all_indicators(df)

print("=" * 70)
print("ANALYZING LAST 10 BARS")
print("=" * 70)
print()

# Check each of the last 10 bars
for i in range(-10, 0):
    bar = df.iloc[i]
    bar_time = df.index[i]

    print(f"\n[{bar_time}]")
    print(f"  Close: {bar['close']:.2f}")
    print(f"  ADX: {bar.get('adx_14', 0):.1f}")
    print(f"  EMA 20: {bar.get('ema_20', 0):.2f}")
    print(f"  EMA 200: {bar.get('ema_200', 0):.2f}")
    print(f"  ATR: {bar.get('atr_14', 0):.2f}")

    # Check regime
    window = df.iloc[:i+1]  # Data up to this bar
    if len(window) >= 200:
        regime, confidence, strat_rec = regime_detector.detect_regime(window)
        print(f"  Regime: {regime.value} ({confidence:.1f}%)")
        print(f"  Strategy: {strat_rec}")

        # Check for setup
        selector = StrategySelector('GOLD')
        setup = selector.check_entry(window, regime.value, confidence, strat_rec)

        if setup:
            print(f"  ✓ SETUP FOUND!")
            print(f"    Type: {setup.setup_type}")
            print(f"    Entry: {setup.entry_price:.2f}")
            print(f"    Stop: {setup.stop_loss:.2f}")
            print(f"    Target: {setup.target:.2f}")
            print(f"    RRR: {setup.reward_risk_ratio():.2f}")
        else:
            print(f"  ✗ No setup")

            # Debug why no setup
            if strat_rec == 'NONE':
                print(f"    Reason: No strategy recommended")
            elif regime.value in ['HIGH_VOLATILITY', 'NO_TRADE']:
                print(f"    Reason: Unfavorable regime")
            else:
                print(f"    Reason: Setup conditions not met (check ADX, pullback, etc.)")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"Last 10 bars analyzed for GOLD 15m")
print(f"If all bars show 'No setup', the strategy is too strict")
print(f"Check ADX values - if all < 6, increase ADX thresholds won't help")
print(f"Check regimes - if all 'NO_TRADE' or 'HIGH_VOLATILITY', regime detection needs tuning")
