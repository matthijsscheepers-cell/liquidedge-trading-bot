"""
Backtest: PineScript TTM Squeeze vs Onze Gecalibreerde Versie

Vergelijk performance van:
1. Onze versie: BB 2.5, KC 0.5 (22.8% squeeze, 50 releases)
2. PineScript: BB 2.0, KC 1.0 (98.9% squeeze, 0 releases)
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.ttm import calculate_ttm_squeeze, calculate_ttm_squeeze_pinescript
from src.strategies.ttm_squeeze import TTMSqueezeStrategy
import pandas as pd

print("=" * 70)
print("BACKTEST: PINESCRIPT VS CALIBRATED TTM SQUEEZE")
print("=" * 70)
print()

# Config
START_DATE = '2025-11-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001

# Laad data
loader = DatabentoMicroFuturesLoader()
print("Laden GOLD data...")
gold = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE, resample='15min')
print(f"✓ {len(gold)} bars geladen")
print()

def run_backtest(squeeze_on, momentum, name):
    """Run backtest with given squeeze indicator."""
    print(f"=" * 70)
    print(f"BACKTEST: {name}")
    print(f"=" * 70)
    print()

    # Add indicators to dataframe
    test_data = gold.copy()
    test_data['squeeze_on'] = squeeze_on
    test_data['ttm_momentum'] = momentum

    # Calculate KC middle for strategy
    from src.indicators.volatility import calculate_keltner_channels_pinescript
    _, kc_middle, _ = calculate_keltner_channels_pinescript(
        test_data['high'], test_data['low'], test_data['close'], period=20, multiplier=1.0
    )
    test_data['kc_middle'] = kc_middle

    # Calculate ATR for strategy
    from src.indicators.volatility import calculate_atr
    test_data['atr_14'] = calculate_atr(
        test_data['high'], test_data['low'], test_data['close'], period=14
    )

    # Run backtest
    strategy = TTMSqueezeStrategy('GOLD')
    capital = INITIAL_CAPITAL
    trades = []
    current_position = None

    for i in range(200, len(test_data)):
        bar = test_data.iloc[i]
        window = test_data.iloc[:i+1]
        timestamp = test_data.index[i]

        # Position management
        if current_position:
            direction = current_position['direction']
            entry_price = current_position['entry_price']
            stop_loss = current_position['stop_loss']
            target = current_position['target']
            size = current_position['size']

            # Check stop
            if direction == 'LONG' and bar['low'] <= stop_loss:
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'STOP'})
                current_position = None
                continue

            # Check target
            if direction == 'LONG' and bar['high'] >= target:
                exit_price = target
                pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'TARGET'})
                current_position = None
                continue

        # Entry scan
        if not current_position:
            setup = strategy.check_entry(window, regime='ANY', confidence=80.0)

            if setup and setup.direction.value == 'LONG':  # Only LONG
                risk_per_trade = capital * 0.01
                size = risk_per_trade / setup.risk_per_share
                entry_cost = setup.entry_price * size

                if entry_cost <= capital:
                    current_position = {
                        'direction': 'LONG',
                        'entry_price': setup.entry_price,
                        'stop_loss': setup.stop_loss,
                        'target': setup.target,
                        'size': size,
                    }

    # Results
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = (len(wins) / len(trades)) * 100

        print(f"Trades: {len(trades)}")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Return: {total_return:.2f}%")
        print(f"Final capital: ${capital:,.2f}")

        if len(wins) > 0 and len(losses) > 0:
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
            print(f"Profit factor: {profit_factor:.2f}")
    else:
        print(f"Trades: 0")
        print(f"Return: {total_return:.2f}%")

    print()
    return {
        'trades': len(trades),
        'return': total_return,
        'capital': capital,
        'win_rate': win_rate if len(trades) > 0 else 0,
    }

# Test 1: Onze gecalibreerde versie
print("VERSIE 1: ONZE GECALIBREERDE VERSIE")
print("  BB std: 2.5, KC mult: 0.5")
print()

squeeze_ours, momentum_ours, _ = calculate_ttm_squeeze(
    gold['high'], gold['low'], gold['close'],
    bb_period=20, bb_std=2.5,
    kc_period=20, kc_atr_period=20, kc_multiplier=0.5,
    momentum_period=12
)

results_ours = run_backtest(squeeze_ours, momentum_ours, "CALIBRATED VERSION")

# Test 2: PineScript versie
print("VERSIE 2: PINESCRIPT VERSIE")
print("  BB std: 2.0, KC mult: 1.0")
print()

squeeze_pine, momentum_pine, _ = calculate_ttm_squeeze_pinescript(
    gold['high'], gold['low'], gold['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=1.0,
    momentum_period=12
)

results_pine = run_backtest(squeeze_pine, momentum_pine, "PINESCRIPT VERSION")

# Vergelijking
print("=" * 70)
print("VERGELIJKING")
print("=" * 70)
print()

print(f"{'Metric':<25} {'Calibrated':>15} {'PineScript':>15} {'Winner':>15}")
print("-" * 72)
print(f"{'Trades':<25} {results_ours['trades']:>15d} {results_pine['trades']:>15d} {'Calibrated' if results_ours['trades'] > results_pine['trades'] else 'PineScript':>15}")
print(f"{'Return':<25} {results_ours['return']:>14.2f}% {results_pine['return']:>14.2f}% {'Calibrated' if results_ours['return'] > results_pine['return'] else 'PineScript':>15}")
print(f"{'Win Rate':<25} {results_ours['win_rate']:>14.1f}% {results_pine['win_rate']:>14.1f}% {'Calibrated' if results_ours['win_rate'] > results_pine['win_rate'] else 'PineScript':>15}")
print()

print("=" * 70)
print("CONCLUSIE")
print("=" * 70)
print()

if results_ours['return'] > results_pine['return']:
    print("✓ Onze gecalibreerde versie presteert BETER")
    print(f"  Return verschil: {results_ours['return'] - results_pine['return']:.2f}%")
else:
    print("✓ PineScript versie presteert BETER")
    print(f"  Return verschil: {results_pine['return'] - results_ours['return']:.2f}%")

print()
print("=" * 70)
