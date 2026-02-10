"""
Optimization Backtest: Test different entry levels and circuit breakers

Tests combinations of:
- Entry ATR: -0.5, -0.75, -1.0
- Circuit breakers: ON/OFF
- Assets: All 4 vs GOLD+SILVER only

Uses corrected backtest logic (no look-ahead bias).
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
import pandas as pd
import numpy as np

# Config
START_DATE = '2010-09-12'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 300.0
COMMISSION_PCT = 0.001
STOP_SLIPPAGE_PCT = 0.001
RISK_PER_TRADE = 0.02
RISK_CAP = 50.0
LEVERAGE = 20
MAX_POSITIONS = 4

# Circuit breaker settings
CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16        # 4 hours = 16 × 15min bars
CB_DAILY_LOSS_MULTIPLIER = 3  # 3x risk cap

ASSETS_ALL = ['GOLD', 'SILVER', 'US100', 'US500']

# Load data once
print("Loading data (this takes a few minutes)...")
loader = DatabentoMicroFuturesLoader()

all_data_15m = {}
all_data_1h = {}

for asset in ASSETS_ALL:
    print(f"  {asset}...", end=" ", flush=True)
    try:
        df_15m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
        df_1h = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='1h')

        df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
        df_15m['atr_20'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=20)

        squeeze_on, momentum, color, intensity = calculate_ttm_squeeze_pinescript(
            df_15m['high'], df_15m['low'], df_15m['close'],
            bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
        )
        df_15m['squeeze_on'] = squeeze_on
        df_15m['ttm_momentum'] = momentum

        _, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
            df_1h['high'], df_1h['low'], df_1h['close'],
            bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
        )
        df_1h['ttm_momentum'] = momentum_1h

        all_data_15m[asset] = df_15m
        all_data_1h[asset] = df_1h
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

# Align timestamps
common_index = None
for asset in all_data_15m.keys():
    if common_index is None:
        common_index = all_data_15m[asset].index
    else:
        common_index = common_index.intersection(all_data_15m[asset].index)

for asset in all_data_15m.keys():
    all_data_15m[asset] = all_data_15m[asset].loc[common_index]

print(f"\nCommon 15min bars: {len(common_index)}")
print()


def run_backtest(entry_atr, stop_atr, target_atr, tolerance, assets, use_circuit_breakers):
    """Run a single backtest configuration."""
    capital = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_values = []

    # Circuit breaker state
    consecutive_stops = {}
    asset_cooldowns = {}
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    for i in range(201, len(common_index)):
        timestamp = common_index[i]

        # Track equity
        equity = capital
        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]
            equity += (bar['close'] - pos['entry_price']) * pos['size']
        equity_values.append(equity)

        # Daily loss reset
        if use_circuit_breakers:
            current_date = timestamp.date() if hasattr(timestamp, 'date') else None
            if current_date and current_date != daily_loss_date:
                daily_losses = 0.0
                daily_loss_date = current_date
                trading_halted = False

        # === POSITION MANAGEMENT ===
        positions_to_close = []
        for asset, pos in list(positions.items()):
            bar = all_data_15m[asset].iloc[i]

            if bar['low'] <= pos['stop_loss']:
                exit_price = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                exit_comm = exit_price * pos['size'] * COMMISSION_PCT
                pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
                capital += pnl
                trades.append({
                    'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                    'bars_held': i - pos['entry_bar'],
                    'entry_time': pos['entry_time'], 'exit_time': timestamp,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                })
                positions_to_close.append(asset)

                if use_circuit_breakers:
                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= RISK_CAP * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True
                continue

            if bar['high'] >= pos['target']:
                exit_price = pos['target']
                entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                exit_comm = exit_price * pos['size'] * COMMISSION_PCT
                pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
                capital += pnl
                trades.append({
                    'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                    'bars_held': i - pos['entry_bar'],
                    'entry_time': pos['entry_time'], 'exit_time': timestamp,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                })
                positions_to_close.append(asset)

                if use_circuit_breakers:
                    consecutive_stops[asset] = 0
                continue

        for asset in positions_to_close:
            del positions[asset]

        # === ENTRY SCAN ===
        if len(positions) >= MAX_POSITIONS:
            continue
        if use_circuit_breakers and trading_halted:
            continue

        for asset in assets:
            if asset in positions or len(positions) >= MAX_POSITIONS:
                continue

            # Circuit breaker cooldown
            if use_circuit_breakers and asset in asset_cooldowns:
                if i < asset_cooldowns[asset]:
                    continue
                else:
                    del asset_cooldowns[asset]
                    consecutive_stops.pop(asset, None)

            df_15m = all_data_15m[asset]
            df_1h = all_data_1h[asset]

            prev_bar = df_15m.iloc[i - 1]
            prev_timestamp = common_index[i - 1]
            bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            # Signal conditions (bar i-1)
            if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
                continue
            if pd.isna(prev_bar['squeeze_on']) or not prev_bar['squeeze_on']:
                continue

            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']
            if pd.isna(ema) or pd.isna(atr) or atr <= 0:
                continue

            entry_level = ema + (entry_atr * atr)
            stop_level = ema + (stop_atr * atr)
            target_level = ema + (target_atr * atr)

            risk = entry_level - stop_level
            if risk <= 0:
                continue

            # Fill check (bar i)
            curr_bar = df_15m.iloc[i]
            distance = (curr_bar['low'] - entry_level) / atr
            if distance > tolerance or distance < -tolerance:
                continue

            fill_price = entry_level
            risk_amount = min(capital * RISK_PER_TRADE, RISK_CAP)
            size = risk_amount / risk
            margin = size * fill_price / LEVERAGE
            if margin > capital:
                continue

            # Same-bar stop check
            if curr_bar['low'] <= stop_level:
                exit_price = stop_level * (1 - STOP_SLIPPAGE_PCT)
                entry_comm = fill_price * size * COMMISSION_PCT
                exit_comm = exit_price * size * COMMISSION_PCT
                pnl = (exit_price - fill_price) * size - entry_comm - exit_comm
                capital += pnl
                trades.append({
                    'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME_BAR',
                    'bars_held': 0,
                    'entry_time': timestamp, 'exit_time': timestamp,
                    'entry_price': fill_price, 'exit_price': exit_price,
                })
                if use_circuit_breakers:
                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(pnl)
                continue

            positions[asset] = {
                'entry_time': timestamp, 'entry_bar': i,
                'entry_price': fill_price, 'stop_loss': stop_level,
                'target': target_level, 'size': size,
            }
            break

    # Close remaining
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        exit_price = bar['close']
        pnl = (exit_price - pos['entry_price']) * pos['size']
        capital += pnl
        trades.append({'symbol': asset, 'pnl': pnl, 'exit_reason': 'END', 'bars_held': 0,
                       'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
                       'entry_price': pos['entry_price'], 'exit_price': exit_price})

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq = pd.Series(equity_values)
    max_dd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100 if len(eq) > 0 else 0

    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        wr = len(wins) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    else:
        wr = pf = avg_win = avg_loss = 0

    # Per-asset breakdown
    asset_stats = {}
    for a in assets:
        at = trades_df[trades_df['symbol'] == a] if len(trades_df) > 0 else pd.DataFrame()
        if len(at) > 0:
            aw = at[at['pnl'] > 0]
            asset_stats[a] = {
                'trades': len(at),
                'wr': len(aw) / len(at) * 100,
                'pnl': at['pnl'].sum(),
            }

    return {
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'asset_stats': asset_stats,
    }


# =====================================================
# RUN OPTIMIZATION GRID
# =====================================================
print("=" * 90)
print("OPTIMIZATION GRID")
print("=" * 90)
print()

# Test configurations
configs = [
    # (name, entry_atr, stop_atr, target_atr, tolerance, assets, circuit_breakers)
    ("Baseline -1.0 ATR",        -1.0, -2.1, 2.0, 0.5, ASSETS_ALL, False),
    ("CB: -1.0 ATR",             -1.0, -2.1, 2.0, 0.5, ASSETS_ALL, True),
    ("-0.75 ATR",                -0.75, -1.75, 1.75, 0.5, ASSETS_ALL, False),
    ("CB: -0.75 ATR",            -0.75, -1.75, 1.75, 0.5, ASSETS_ALL, True),
    ("-0.5 ATR",                 -0.5, -1.5, 1.5, 0.5, ASSETS_ALL, False),
    ("CB: -0.5 ATR",             -0.5, -1.5, 1.5, 0.5, ASSETS_ALL, True),
    ("CB: -0.75 GOLD+SILVER",   -0.75, -1.75, 1.75, 0.5, ['GOLD', 'SILVER'], True),
    ("CB: -1.0 GOLD+SILVER",    -1.0, -2.1, 2.0, 0.5, ['GOLD', 'SILVER'], True),
    ("-0.5 GOLD+SILVER",         -0.5, -1.5, 1.5, 0.5, ['GOLD', 'SILVER'], False),
    ("CB: -0.5 GOLD+SILVER",    -0.5, -1.5, 1.5, 0.5, ['GOLD', 'SILVER'], True),
]

results = []

for name, entry_atr, stop_atr, target_atr, tol, assets, cb in configs:
    print(f"Running: {name}...", end=" ", flush=True)
    r = run_backtest(entry_atr, stop_atr, target_atr, tol, assets, cb)
    r['name'] = name
    results.append(r)
    print(f"Trades: {r['trades']}, WR: {r['win_rate']:.1f}%, PF: {r['profit_factor']:.2f}, "
          f"Capital: ${r['capital']:,.0f}, MaxDD: {r['max_drawdown']:.1f}%")

# Summary table
print()
print("=" * 90)
print("OPTIMIZATION RESULTS SUMMARY")
print("=" * 90)
print()
print(f"{'Config':<30s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'Capital':>12s} {'MaxDD':>8s} {'AvgW':>8s} {'AvgL':>8s}")
print("-" * 90)
for r in results:
    print(f"{r['name']:<30s} {r['trades']:>7d} {r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} "
          f"${r['capital']:>11,.0f} {r['max_drawdown']:>7.1f}% ${r['avg_win']:>6.0f} ${r['avg_loss']:>7.0f}")

print()
print("=" * 90)
print("PER-ASSET BREAKDOWN (best configs)")
print("=" * 90)
print()

for r in results:
    if r['asset_stats']:
        print(f"\n{r['name']}:")
        for a, s in sorted(r['asset_stats'].items()):
            print(f"  {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, ${s['pnl']:>12,.2f}")

print()
print("=" * 90)
