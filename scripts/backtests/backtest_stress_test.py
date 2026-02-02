"""
Progressive Risk Cap - Stress Testing

Tests the winning Progressive Risk Cap strategy under:
1. Various slippage conditions (0.1% - 1.0%)
2. Various commission rates (0.1% - 0.5%)
3. Drawdown analysis (max DD, recovery time)

This validates robustness of the strategy under realistic market conditions.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
from src.strategies.ttm_pullback import TTMSqueezePullbackStrategy
import pandas as pd
import numpy as np

def get_risk_cap(capital):
    """Progressive risk cap"""
    if capital < 1000:
        return 50.0
    elif capital < 5000:
        return 100.0
    elif capital < 20000:
        return 200.0
    elif capital < 100000:
        return 500.0
    else:
        return 1000.0

def run_backtest(commission_pct, slippage_pct, name):
    """Run backtest with specific parameters"""

    START_DATE = '2010-09-12'
    END_DATE = '2026-01-29'
    INITIAL_CAPITAL = 300.0
    LEVERAGE = 20
    MAX_POSITIONS = 4
    RISK_PER_TRADE = 0.02

    ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

    # Load data
    loader = DatabentoMicroFuturesLoader()
    all_data_15m = {}
    all_data_1h = {}

    for asset in ASSETS:
        try:
            df_15m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
            df_1h = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='1h')

            df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
            df_15m['atr_20'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=20)

            squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
                df_15m['high'], df_15m['low'], df_15m['close'],
                bb_period=20, bb_std=2.0, kc_period=20, kc_multiplier=2.0, momentum_period=20
            )
            df_15m['squeeze_on'] = squeeze_on
            df_15m['ttm_momentum'] = momentum

            squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
                df_1h['high'], df_1h['low'], df_1h['close'],
                bb_period=20, bb_std=2.0, kc_period=20, kc_multiplier=2.0, momentum_period=20
            )
            df_1h['ttm_momentum'] = momentum_1h

            all_data_15m[asset] = df_15m
            all_data_1h[asset] = df_1h
        except Exception as e:
            print(f"Error loading {asset}: {e}")

    # Align timestamps
    common_index = None
    for asset in all_data_15m.keys():
        if common_index is None:
            common_index = all_data_15m[asset].index
        else:
            common_index = common_index.intersection(all_data_15m[asset].index)

    for asset in all_data_15m.keys():
        all_data_15m[asset] = all_data_15m[asset].loc[common_index]

    # Initialize strategies
    strategies = {asset: TTMSqueezePullbackStrategy(asset) for asset in all_data_15m.keys()}

    # Backtest
    capital = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = []
    peak_equity = INITIAL_CAPITAL
    max_drawdown = 0
    max_drawdown_pct = 0

    for i in range(200, len(common_index)):
        timestamp = common_index[i]

        # Update equity
        equity = capital
        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]
            unrealized = (bar['close'] - pos['entry_price']) * pos['size']
            equity += unrealized

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity

        drawdown = peak_equity - equity
        drawdown_pct = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0

        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = drawdown_pct

        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': capital,
            'positions': len(positions),
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct
        })

        # Position management
        positions_to_close = []

        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]

            # Check stop loss (with slippage)
            if bar['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss'] * (1 - slippage_pct)  # Slippage on stop
                pnl = (exit_price - pos['entry_price']) * pos['size'] - (exit_price * pos['size'] * commission_pct)
                capital += pnl

                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'symbol': asset,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'STOP',
                    'bars_held': i - pos['entry_bar'],
                })

                positions_to_close.append(asset)
                continue

            # Check target (with slippage)
            if bar['high'] >= pos['target']:
                exit_price = pos['target'] * (1 - slippage_pct)  # Slippage on target
                pnl = (exit_price - pos['entry_price']) * pos['size'] - (exit_price * pos['size'] * commission_pct)
                capital += pnl

                trades.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': timestamp,
                    'symbol': asset,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'exit_reason': 'TARGET',
                    'bars_held': i - pos['entry_bar'],
                })

                positions_to_close.append(asset)

        for asset in positions_to_close:
            del positions[asset]

        # Entry scan
        if len(positions) < MAX_POSITIONS:
            for asset, strategy in strategies.items():
                if asset in positions:
                    continue

                window_15m = all_data_15m[asset].iloc[:i+1]
                current_timestamp = common_index[i]
                bars_1h = all_data_1h[asset][all_data_1h[asset].index <= current_timestamp]

                if len(bars_1h) < 20:
                    continue

                setup = strategy.check_entry(window_15m, bars_1h, regime='ANY', confidence=80.0)

                if setup and setup.direction.value == 'LONG':
                    current_risk_cap = get_risk_cap(capital)
                    risk_amount = min(capital * RISK_PER_TRADE, current_risk_cap)
                    stop_distance = setup.entry_price - setup.stop_loss

                    position_size_units = risk_amount / stop_distance
                    position_value = position_size_units * setup.entry_price
                    margin_required = position_value / LEVERAGE

                    if margin_required <= capital and len(positions) < MAX_POSITIONS:
                        # Apply slippage on entry
                        entry_price_with_slippage = setup.entry_price * (1 + slippage_pct)

                        positions[asset] = {
                            'entry_time': timestamp,
                            'entry_bar': i,
                            'entry_price': entry_price_with_slippage,
                            'stop_loss': setup.stop_loss,
                            'target': setup.target,
                            'size': position_size_units,
                        }
                        break

    # Close remaining positions
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        exit_price = bar['close'] * (1 - slippage_pct)
        pnl = (exit_price - pos['entry_price']) * pos['size'] - (exit_price * pos['size'] * commission_pct)
        capital += pnl

        trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': common_index[-1],
            'symbol': asset,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': 'END',
            'bars_held': len(common_index) - 1 - pos['entry_bar'],
        })

    # Calculate stats
    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 else 0

    return {
        'name': name,
        'commission': commission_pct,
        'slippage': slippage_pct,
        'final_capital': capital,
        'total_return': total_return,
        'total_trades': len(trades_df),
        'win_rate': len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
        'equity_curve': equity_df
    }

# =====================================================
# MAIN - RUN ALL TESTS
# =====================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROGRESSIVE RISK CAP - STRESS TESTING")
    print("=" * 70)
    print()

    results = []

    # Test 1: Slippage variations (fixed commission 0.1%)
    print("TEST 1: SLIPPAGE IMPACT")
    print("-" * 70)
    for slippage in [0.001, 0.002, 0.005, 0.010]:
        print(f"Running with {slippage*100:.1f}% slippage...")
        result = run_backtest(commission_pct=0.001, slippage_pct=slippage,
                             name=f"Slippage {slippage*100:.1f}%")
        results.append(result)
        print(f"  Final: ${result['final_capital']:,.0f}, Return: {result['total_return']:.0f}%, Max DD: {result['max_drawdown_pct']:.1f}%")
    print()

    # Test 2: Commission variations (fixed slippage 0.1%)
    print("TEST 2: COMMISSION IMPACT")
    print("-" * 70)
    for commission in [0.001, 0.003, 0.005]:
        print(f"Running with {commission*100:.1f}% commission...")
        result = run_backtest(commission_pct=commission, slippage_pct=0.001,
                             name=f"Commission {commission*100:.1f}%")
        results.append(result)
        print(f"  Final: ${result['final_capital']:,.0f}, Return: {result['total_return']:.0f}%, Max DD: {result['max_drawdown_pct']:.1f}%")
    print()

    # Test 3: Realistic worst-case (higher slippage + commission)
    print("TEST 3: WORST-CASE SCENARIO")
    print("-" * 70)
    print("Running with 0.5% slippage + 0.3% commission...")
    result = run_backtest(commission_pct=0.003, slippage_pct=0.005,
                         name="Worst Case")
    results.append(result)
    print(f"  Final: ${result['final_capital']:,.0f}, Return: {result['total_return']:.0f}%, Max DD: {result['max_drawdown_pct']:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    print()

    df_results = pd.DataFrame(results)

    print("SLIPPAGE IMPACT:")
    print("-" * 70)
    slippage_tests = df_results[df_results['name'].str.contains('Slippage')]
    for _, row in slippage_tests.iterrows():
        print(f"{row['name']:20s} → ${row['final_capital']:12,.0f} ({row['total_return']:8.0f}%) | DD: {row['max_drawdown_pct']:5.1f}%")
    print()

    print("COMMISSION IMPACT:")
    print("-" * 70)
    commission_tests = df_results[df_results['name'].str.contains('Commission')]
    for _, row in commission_tests.iterrows():
        print(f"{row['name']:20s} → ${row['final_capital']:12,.0f} ({row['total_return']:8.0f}%) | DD: {row['max_drawdown_pct']:5.1f}%")
    print()

    print("WORST-CASE:")
    print("-" * 70)
    worst_case = df_results[df_results['name'] == 'Worst Case'].iloc[0]
    print(f"Worst Case (0.5% slip + 0.3% comm) → ${worst_case['final_capital']:,.0f} ({worst_case['total_return']:.0f}%)")
    print(f"  Max Drawdown: {worst_case['max_drawdown_pct']:.1f}% (${worst_case['max_drawdown']:,.0f})")
    print(f"  Win Rate: {worst_case['win_rate']:.1f}%")
    print(f"  Profit Factor: {worst_case['profit_factor']:.2f}")
    print()

    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print()

    baseline = df_results[df_results['slippage'] == 0.001].iloc[0]
    worst = worst_case

    degradation = ((baseline['final_capital'] - worst['final_capital']) / baseline['final_capital']) * 100

    print(f"Baseline (0.1% slip + 0.1% comm): ${baseline['final_capital']:,.0f}")
    print(f"Worst Case (0.5% slip + 0.3% comm): ${worst['final_capital']:,.0f}")
    print(f"Performance Degradation: {degradation:.1f}%")
    print()

    if worst['final_capital'] > INITIAL_CAPITAL * 100:  # Still 100x return
        print("✅ Strategy remains HIGHLY PROFITABLE even under worst-case conditions!")
    elif worst['final_capital'] > INITIAL_CAPITAL * 10:  # Still 10x return
        print("✅ Strategy remains PROFITABLE under worst-case conditions")
    else:
        print("⚠️  Strategy shows significant degradation under worst-case conditions")

    print()
    print("=" * 70)
