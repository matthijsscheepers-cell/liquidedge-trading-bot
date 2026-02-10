"""
Profit Maximization Backtest: Progressive Risk Cap + Partial Exits

Tests three strategies on GOLD+SILVER with CB -0.5 ATR:
1. Progressive risk cap (scaling with account growth)
2. Flat cap + partial exits (50% at target, trail remaining 50%)
3. Progressive cap + partial exits (the full combo)

Baseline: CB -0.5 ATR GOLD+SILVER flat $50 cap = 88.2% WR, PF 7.44, $643K
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
FLAT_RISK_CAP = 50.0
LEVERAGE = 20
MAX_POSITIONS = 2

# Circuit breaker settings
CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16
CB_DAILY_LOSS_MULTIPLIER = 3

# Entry parameters (optimized)
ENTRY_ATR = -0.5
STOP_ATR = -1.5
TARGET_ATR = 1.5
TOLERANCE = 0.5

# Partial exit settings
PARTIAL_EXIT_PCT = 0.50       # Close 50% at first target
TRAIL_ATR_DISTANCE = 1.5      # Trail stop at highest_high - 1.5 ATR

ASSETS = ['GOLD', 'SILVER']


def get_progressive_risk_cap(capital):
    """Return risk cap based on current capital"""
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


# Load data
print("Loading data (GOLD + SILVER only)...")
loader = DatabentoMicroFuturesLoader()

all_data_15m = {}
all_data_1h = {}

for asset in ASSETS:
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


def run_backtest(use_progressive_cap, use_partial_exits):
    """Run backtest with optional progressive cap and partial exits."""
    capital = INITIAL_CAPITAL
    positions = {}  # asset -> position dict
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
            remaining_size = pos['size'] * (1.0 - pos.get('partial_closed_pct', 0.0))
            equity += (bar['close'] - pos['entry_price']) * remaining_size
        equity_values.append(equity)

        # Daily loss reset
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date and current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # === POSITION MANAGEMENT ===
        positions_to_close = []
        for asset, pos in list(positions.items()):
            bar = all_data_15m[asset].iloc[i]
            remaining_pct = 1.0 - pos.get('partial_closed_pct', 0.0)
            remaining_size = pos['size'] * remaining_pct
            current_stop = pos.get('trailing_stop', pos['stop_loss'])

            # Update highest high for trailing
            if bar['high'] > pos.get('highest_high', pos['entry_price']):
                pos['highest_high'] = bar['high']

            # Update trailing stop if in trail mode
            if pos.get('trailing', False):
                atr = all_data_15m[asset].iloc[i - 1]['atr_20']
                if not pd.isna(atr) and atr > 0:
                    new_trail_stop = pos['highest_high'] - (TRAIL_ATR_DISTANCE * atr)
                    # Only move stop up, never down
                    if new_trail_stop > current_stop:
                        current_stop = new_trail_stop
                        pos['trailing_stop'] = current_stop

            # Check stop loss
            if bar['low'] <= current_stop:
                exit_price = current_stop * (1 - STOP_SLIPPAGE_PCT)
                entry_comm = pos['entry_price'] * remaining_size * COMMISSION_PCT
                exit_comm = exit_price * remaining_size * COMMISSION_PCT
                pnl = (exit_price - pos['entry_price']) * remaining_size - entry_comm - exit_comm
                # Add any already-realized partial P&L
                pnl += pos.get('realized_pnl', 0.0)
                capital += pnl

                exit_reason = 'TRAIL_STOP' if pos.get('trailing', False) else 'STOP'
                trades.append({
                    'symbol': asset, 'pnl': pnl, 'exit_reason': exit_reason,
                    'bars_held': i - pos['entry_bar'],
                    'entry_time': pos['entry_time'], 'exit_time': timestamp,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                })
                positions_to_close.append(asset)

                consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                daily_losses += abs(pnl)
                if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                    asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                if daily_losses >= (get_progressive_risk_cap(capital) if use_progressive_cap else FLAT_RISK_CAP) * CB_DAILY_LOSS_MULTIPLIER:
                    trading_halted = True
                continue

            # Check target
            if bar['high'] >= pos['target'] and not pos.get('target_hit', False):
                if use_partial_exits:
                    # === PARTIAL EXIT: close 50%, trail rest ===
                    partial_size = pos['size'] * PARTIAL_EXIT_PCT
                    exit_price = pos['target']
                    entry_comm = pos['entry_price'] * partial_size * COMMISSION_PCT
                    exit_comm = exit_price * partial_size * COMMISSION_PCT
                    partial_pnl = (exit_price - pos['entry_price']) * partial_size - entry_comm - exit_comm
                    capital += partial_pnl

                    pos['target_hit'] = True
                    pos['partial_closed_pct'] = PARTIAL_EXIT_PCT
                    pos['realized_pnl'] = 0.0  # Already added to capital
                    pos['trailing'] = True
                    pos['trailing_stop'] = pos['entry_price']  # Move stop to breakeven
                    pos['highest_high'] = bar['high']

                    consecutive_stops[asset] = 0
                else:
                    # === FULL EXIT at target ===
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

                    consecutive_stops[asset] = 0
                continue

        for asset in positions_to_close:
            del positions[asset]

        # === ENTRY SCAN ===
        if len(positions) >= MAX_POSITIONS:
            continue
        if trading_halted:
            continue

        for asset in ASSETS:
            if asset in positions or len(positions) >= MAX_POSITIONS:
                continue

            # Circuit breaker cooldown
            if asset in asset_cooldowns:
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

            entry_level = ema + (ENTRY_ATR * atr)
            stop_level = ema + (STOP_ATR * atr)
            target_level = ema + (TARGET_ATR * atr)

            risk = entry_level - stop_level
            if risk <= 0:
                continue

            # Fill check (bar i)
            curr_bar = df_15m.iloc[i]
            distance = (curr_bar['low'] - entry_level) / atr
            if distance > TOLERANCE or distance < -TOLERANCE:
                continue

            # Position sizing
            if use_progressive_cap:
                risk_cap = get_progressive_risk_cap(capital)
            else:
                risk_cap = FLAT_RISK_CAP

            fill_price = entry_level
            risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
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
                consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                daily_losses += abs(pnl)
                continue

            positions[asset] = {
                'entry_time': timestamp, 'entry_bar': i,
                'entry_price': fill_price, 'stop_loss': stop_level,
                'target': target_level, 'size': size,
                'highest_high': curr_bar['high'],
                'target_hit': False, 'trailing': False,
                'partial_closed_pct': 0.0, 'realized_pnl': 0.0,
            }
            break

    # Close remaining positions
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        exit_price = bar['close']
        remaining_pct = 1.0 - pos.get('partial_closed_pct', 0.0)
        remaining_size = pos['size'] * remaining_pct
        pnl = (exit_price - pos['entry_price']) * remaining_size + pos.get('realized_pnl', 0.0)
        capital += pnl
        trades.append({
            'symbol': asset, 'pnl': pnl, 'exit_reason': 'END', 'bars_held': 0,
            'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
        })

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq = pd.Series(equity_values)
    max_dd = ((eq - eq.expanding().max()) / eq.expanding().max()).min() * 100 if len(eq) > 0 else 0

    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        wr = len(wins) / len(trades_df) * 100
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    else:
        wr = pf = avg_win = avg_loss = 0

    # Exit reason breakdown
    exit_reasons = {}
    if len(trades_df) > 0:
        for reason, group in trades_df.groupby('exit_reason'):
            exit_reasons[reason] = {
                'count': len(group),
                'pnl': group['pnl'].sum(),
                'avg_pnl': group['pnl'].mean(),
            }

    # Per-asset breakdown
    asset_stats = {}
    for a in ASSETS:
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
        'exit_reasons': exit_reasons,
    }


# =====================================================
# RUN TESTS
# =====================================================
print("=" * 90)
print("PROFIT MAXIMIZATION: PROGRESSIVE CAP + PARTIAL EXITS")
print("=" * 90)
print()
print("Base config: CB -0.5 ATR, GOLD+SILVER, entry=-0.5, stop=-1.5, target=+1.5")
print()

configs = [
    ("1. Baseline (flat $50 cap)",          False, False),
    ("2. Progressive risk cap",              True,  False),
    ("3. Flat cap + partial exits",          False, True),
    ("4. Progressive cap + partial exits",   True,  True),
]

results = []

for name, prog_cap, partial in configs:
    print(f"Running: {name}...", end=" ", flush=True)
    r = run_backtest(use_progressive_cap=prog_cap, use_partial_exits=partial)
    r['name'] = name
    results.append(r)
    print(f"Trades: {r['trades']}, WR: {r['win_rate']:.1f}%, PF: {r['profit_factor']:.2f}, "
          f"Capital: ${r['capital']:,.0f}, MaxDD: {r['max_drawdown']:.1f}%")

# Summary
print()
print("=" * 90)
print("RESULTS SUMMARY")
print("=" * 90)
print()
print(f"{'Config':<42s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'Capital':>14s} {'MaxDD':>8s} {'AvgW':>10s} {'AvgL':>10s}")
print("-" * 105)
for r in results:
    print(f"{r['name']:<42s} {r['trades']:>7d} {r['win_rate']:>6.1f}% {r['profit_factor']:>7.2f} "
          f"${r['capital']:>13,.0f} {r['max_drawdown']:>7.1f}% ${r['avg_win']:>8,.0f} ${r['avg_loss']:>9,.0f}")

# Exit reason breakdown
print()
print("=" * 90)
print("EXIT REASON BREAKDOWN")
print("=" * 90)
for r in results:
    print(f"\n{r['name']}:")
    for reason, stats in sorted(r['exit_reasons'].items()):
        print(f"  {reason:15s}: {stats['count']:6d} trades, ${stats['pnl']:>12,.2f} total, ${stats['avg_pnl']:>8,.2f} avg")

# Per-asset breakdown
print()
print("=" * 90)
print("PER-ASSET BREAKDOWN")
print("=" * 90)
for r in results:
    if r['asset_stats']:
        print(f"\n{r['name']}:")
        for a, s in sorted(r['asset_stats'].items()):
            print(f"  {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, ${s['pnl']:>14,.2f}")

# Comparison
print()
print("=" * 90)
print("IMPROVEMENT vs BASELINE")
print("=" * 90)
print()
baseline = results[0]
for r in results[1:]:
    cap_diff = r['capital'] - baseline['capital']
    cap_mult = r['capital'] / baseline['capital'] if baseline['capital'] > 0 else 0
    dd_diff = r['max_drawdown'] - baseline['max_drawdown']
    print(f"{r['name']}:")
    print(f"  Capital: ${r['capital']:>13,.0f} ({cap_mult:.1f}x baseline, +${cap_diff:>13,.0f})")
    print(f"  MaxDD:   {r['max_drawdown']:>7.1f}% ({dd_diff:+.1f}% vs baseline)")
    print(f"  WR:      {r['win_rate']:>6.1f}% (baseline: {baseline['win_rate']:.1f}%)")
    print(f"  PF:      {r['profit_factor']:>6.2f} (baseline: {baseline['profit_factor']:.2f})")
    print()

print("=" * 90)
