"""
1-Minute Execution Backtest: High-Fidelity Fill Simulation

Uses 1-minute candles for ALL execution (fills, stop/target exits), while
keeping signal generation on 15-minute + 1-hour data.

Tests optimization grid:
- Entry ATR: -0.5, -0.75, -1.0
- All with circuit breakers ON, progressive risk cap, GOLD+SILVER

Signal timing:
- Signal evaluated on previous closed 15-min bar (bar i-1)
- Execution window: 1-min candles within the NEXT 15-min bar (bar i)
- Limit order expires if unfilled within that 15-min window

Exit handling:
- From the 1-min candle AFTER fill, check stop/target on each 1-min candle
- If both stop and target hit in same 1-min candle: stop wins (conservative)
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
import pandas as pd
import numpy as np
import time

# =====================================================
# CONFIGURATION
# =====================================================

START_DATE = '2010-09-12'
END_DATE = '2026-01-29'
COMMISSION_PCT = 0.001       # 0.1% per side
STOP_SLIPPAGE_PCT = 0.001   # 0.1% slippage on stop-market orders
RISK_PER_TRADE = 0.02       # 2% base risk
LEVERAGE = 20
MAX_POSITIONS = 2

# Circuit breaker settings
CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16        # 4 hours = 16 x 15min bars
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']

# Test -0.5 ATR (best config) across multiple starting capitals
ENTRY_ATR, STOP_ATR, TARGET_ATR = -0.5, -1.5, 1.5

CONFIGS = [
    ("$1K start",   1_000.0),
    ("$5K start",   5_000.0),
    ("$10K start", 10_000.0),
    ("$25K start", 25_000.0),
]


def get_progressive_risk_cap(capital):
    """Return risk cap based on current capital tier."""
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


def resample_ohlcv(df, freq):
    """Resample 1-minute OHLCV data to a higher timeframe."""
    return df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()


# =====================================================
# DATA LOADING (once for all configs)
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — CAPITAL SENSITIVITY")
print("=" * 90)
print(f"Entry: {ENTRY_ATR} ATR | Stop: {STOP_ATR} ATR | Target: +{TARGET_ATR} ATR")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Capital levels: {[f'${c[1]:,.0f}' for c in CONFIGS]}")
print()
print("Loading data (1-minute resolution, resampling in-script)...")
print()

loader = DatabentoMicroFuturesLoader()

all_data_15m = {}
all_data_1h = {}
grouped_1m = {}

t_start = time.time()

for asset in ASSETS:
    print(f"  {asset}...", end=" ", flush=True)

    # Load raw 1-minute data (single DBN read per asset)
    df_1m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE)
    print(f"{len(df_1m)} 1m bars", end=" -> ", flush=True)

    # Resample to 15-min and 1-hour
    df_15m = resample_ohlcv(df_1m, '15min')
    df_1h = resample_ohlcv(df_1m, '1h')
    print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars", end=" ", flush=True)

    # Calculate indicators on 15-min
    df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
    df_15m['atr_20'] = calculate_atr(
        df_15m['high'], df_15m['low'], df_15m['close'], period=20
    )

    squeeze_on, momentum, color, intensity = calculate_ttm_squeeze_pinescript(
        df_15m['high'], df_15m['low'], df_15m['close'],
        bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
    )
    df_15m['squeeze_on'] = squeeze_on
    df_15m['ttm_momentum'] = momentum

    # Calculate 1H momentum
    _, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
        df_1h['high'], df_1h['low'], df_1h['close'],
        bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
    )
    df_1h['ttm_momentum'] = momentum_1h

    # Pre-group 1-minute candles by their 15-minute period for O(1) lookup
    period_labels = df_1m.index.floor('15min')
    grouped = {ts: group for ts, group in df_1m.groupby(period_labels)}

    all_data_15m[asset] = df_15m
    all_data_1h[asset] = df_1h
    grouped_1m[asset] = grouped

    # Free raw 1m DataFrame (grouped dict holds the data now)
    del df_1m

    print("OK")

t_load = time.time() - t_start
print(f"\nData loaded in {t_load:.0f}s")

# Align 15-min timestamps across assets
common_index = None
for asset in all_data_15m.keys():
    if common_index is None:
        common_index = all_data_15m[asset].index
    else:
        common_index = common_index.intersection(all_data_15m[asset].index)

for asset in all_data_15m.keys():
    all_data_15m[asset] = all_data_15m[asset].loc[common_index]

print(f"Common 15min bars: {len(common_index)}")
print()


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(initial_capital, entry_atr=ENTRY_ATR, stop_atr=STOP_ATR, target_atr=TARGET_ATR):
    """Run a single 1-minute execution backtest configuration."""
    capital = initial_capital
    positions = {}       # {asset: position_dict}
    closed_this_bar = set()  # safeguard: assets already closed this bar
    trades = []
    equity_values = []

    # Circuit breaker state
    consecutive_stops = {}
    asset_cooldowns = {}
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False
    account_blown = False
    account_blown_ts = None

    # Execution statistics
    stats = {
        'signals_generated': 0,
        'entries_attempted': 0,
        'entries_filled': 0,
        'entries_no_fill': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
    }

    # Diagnostics
    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'forced_stops_capital': 0,
        'max_simultaneous_open': 0,
    }

    empty_df = pd.DataFrame()
    MIN_CAPITAL = 1.0  # hard floor: stop trading below $1

    for i in range(201, len(common_index)):
        timestamp = common_index[i]

        # --- Hard capital floor ---
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                account_blown_ts = timestamp
                diag['forced_stops_capital'] += 1
                # Force-close any remaining positions
                for asset, pos in list(positions.items()):
                    bar = all_data_15m[asset].iloc[i]
                    exit_price = bar['close']
                    entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    exit_comm = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
                    capital += pnl
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    })
                    diag['positions_closed'] += 1
                positions.clear()
            # Track equity but skip all trading
            equity_values.append(max(capital, 0))
            continue

        # --- Track equity (15-min close) ---
        equity = capital
        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]
            unrealised = (bar['close'] - pos['entry_price']) * pos['size']
            equity += unrealised
        equity_values.append(max(equity, 0))

        # Track max simultaneous open
        if len(positions) > diag['max_simultaneous_open']:
            diag['max_simultaneous_open'] = len(positions)

        # --- Daily loss reset ---
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date and current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ===========================================
        # POSITION MANAGEMENT — 1-minute exits
        # ===========================================
        closed_this_bar.clear()

        for asset in list(positions.keys()):
            if asset in closed_this_bar:
                continue  # already closed this bar

            pos = positions[asset]
            candles = grouped_1m[asset].get(timestamp, empty_df)

            if candles.empty:
                # Fallback: use 15-min bar for exit check (data gap)
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
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_stop'] += 1

                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True

                elif bar['high'] >= pos['target']:
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
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0

                continue

            # --- 1-minute exit scanning ---
            candle_lows = candles['low'].values
            candle_highs = candles['high'].values
            candle_times = candles.index.values

            for k in range(len(candle_lows)):
                # Skip candles at or before last processed timestamp
                if candle_times[k] <= pos['last_processed_ts']:
                    continue

                low = candle_lows[k]
                high = candle_highs[k]

                # Conservative: check stop FIRST (even if both hit same candle)
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    exit_comm = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
                    capital += pnl
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(candle_times[k]),
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_stop'] += 1

                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True
                    break

                # Then check target
                if high >= pos['target']:
                    exit_price = pos['target']
                    entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    exit_comm = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
                    capital += pnl
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(candle_times[k]),
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0
                    break
            else:
                # No exit found — advance the last_processed_ts to end of this bar
                if len(candle_times) > 0:
                    pos['last_processed_ts'] = candle_times[-1]

        # Remove closed positions
        for asset in closed_this_bar:
            positions.pop(asset, None)

        # ===========================================
        # ENTRY SCAN — 1-minute fills
        # ===========================================
        if capital <= MIN_CAPITAL:
            continue
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

            # Signal from bar i-1 (previous closed bar)
            prev_bar = df_15m.iloc[i - 1]
            prev_timestamp = common_index[i - 1]
            bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            # --- Signal conditions (identical to existing) ---
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

            stats['signals_generated'] += 1

            # --- Pre-filter: 15-min bar low must reach entry level ---
            curr_bar_15m = df_15m.iloc[i]
            if curr_bar_15m['low'] > entry_level:
                continue

            stats['entries_attempted'] += 1

            # --- 1-minute fill scanning ---
            candles = grouped_1m[asset].get(timestamp, empty_df)

            if candles.empty:
                continue

            candle_lows = candles['low'].values
            candle_highs = candles['high'].values
            candle_times = candles.index.values

            filled = False

            for j in range(len(candle_lows)):
                low = candle_lows[j]
                high = candle_highs[j]

                # Fill condition: 1-min candle range contains entry_level
                if low <= entry_level <= high:
                    fill_price = entry_level
                    fill_time = candle_times[j]

                    # Position sizing with capital guard
                    risk_cap = get_progressive_risk_cap(capital)
                    risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
                    if risk_amount <= 0:
                        break
                    size = risk_amount / risk
                    if size <= 0:
                        break
                    margin = size * fill_price / LEVERAGE
                    if margin > capital:
                        break

                    # Post-fill R:R check
                    reward = target_level - fill_price
                    risk_actual = fill_price - stop_level
                    if risk_actual > 0 and reward / risk_actual < 1.5:
                        break

                    # Check exits on REMAINING 1-min candles after fill
                    same_bar_exit = False

                    for k in range(j + 1, len(candle_lows)):
                        rc_low = candle_lows[k]
                        rc_high = candle_highs[k]

                        # Stop check first (conservative — even if both hit same candle)
                        if rc_low <= stop_level:
                            exit_price = stop_level * (1 - STOP_SLIPPAGE_PCT)
                            entry_comm = fill_price * size * COMMISSION_PCT
                            exit_comm = exit_price * size * COMMISSION_PCT
                            pnl = (exit_price - fill_price) * size - entry_comm - exit_comm
                            capital += pnl
                            trades.append({
                                'symbol': asset, 'pnl': pnl,
                                'exit_reason': 'STOP_SAME_15M',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(fill_time),
                                'exit_time': pd.Timestamp(candle_times[k]),
                                'entry_price': fill_price,
                                'exit_price': exit_price,
                            })
                            stats['same_bar_stops'] += 1
                            diag['positions_opened'] += 1
                            diag['positions_closed'] += 1

                            consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                            daily_losses += abs(pnl)
                            if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                                asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                            if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                                trading_halted = True
                            same_bar_exit = True
                            break

                        if rc_high >= target_level:
                            exit_price = target_level
                            entry_comm = fill_price * size * COMMISSION_PCT
                            exit_comm = exit_price * size * COMMISSION_PCT
                            pnl = (exit_price - fill_price) * size - entry_comm - exit_comm
                            capital += pnl
                            trades.append({
                                'symbol': asset, 'pnl': pnl,
                                'exit_reason': 'TARGET_SAME_15M',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(fill_time),
                                'exit_time': pd.Timestamp(candle_times[k]),
                                'entry_price': fill_price,
                                'exit_price': exit_price,
                            })
                            stats['same_bar_targets'] += 1
                            diag['positions_opened'] += 1
                            diag['positions_closed'] += 1
                            consecutive_stops[asset] = 0
                            same_bar_exit = True
                            break

                    if not same_bar_exit:
                        # Position stays open — store with last_processed_ts
                        last_ts = candle_times[-1] if len(candle_times) > 0 else fill_time
                        positions[asset] = {
                            'entry_time': pd.Timestamp(fill_time),
                            'entry_bar': i,
                            'entry_price': fill_price,
                            'stop_loss': stop_level,
                            'target': target_level,
                            'size': size,
                            'last_processed_ts': last_ts,
                        }
                        diag['positions_opened'] += 1

                    stats['entries_filled'] += 1
                    filled = True
                    break

            if not filled:
                stats['entries_no_fill'] += 1

            if filled:
                break  # Only one entry per 15-min bar

        # Progress reporting
        if i % 10000 == 0:
            progress = (i - 201) / (len(common_index) - 201) * 100
            print(f"    {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}")

    # Close remaining positions at market
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        exit_price = bar['close']
        entry_comm = pos['entry_price'] * pos['size'] * COMMISSION_PCT
        exit_comm = exit_price * pos['size'] * COMMISSION_PCT
        pnl = (exit_price - pos['entry_price']) * pos['size'] - entry_comm - exit_comm
        capital += pnl
        trades.append({
            'symbol': asset, 'pnl': pnl, 'exit_reason': 'END',
            'bars_held': len(common_index) - 1 - pos['entry_bar'],
            'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
        })
        diag['positions_closed'] += 1

    # Calculate metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq = pd.Series(equity_values)

    # Robust drawdown: guard against zero/negative equity
    if len(eq) > 0:
        running_max = eq.expanding().max()
        running_max = running_max.clip(lower=1e-6)  # prevent div by zero
        drawdowns = (eq - running_max) / running_max
        max_dd = drawdowns.min() * 100
    else:
        max_dd = 0

    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        wr = len(wins) / len(trades_df) * 100
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    else:
        wr = pf = avg_win = avg_loss = 0

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
        'stats': stats,
        'diag': diag,
        'account_blown_ts': account_blown_ts,
        'trades_df': trades_df,
        'equity': equity_values,
    }


# =====================================================
# RUN OPTIMIZATION GRID
# =====================================================

print("=" * 90)
print("RUNNING CAPITAL SENSITIVITY TEST (1-minute execution)")
print("=" * 90)
print()

risk_atr = abs(ENTRY_ATR - STOP_ATR)
reward_atr = TARGET_ATR - ENTRY_ATR
rr = reward_atr / risk_atr if risk_atr > 0 else 0
print(f"Strategy: Entry {ENTRY_ATR} ATR | Stop {STOP_ATR} ATR | Target +{TARGET_ATR} ATR | R:R = {rr:.1f}:1")
print()

results = []
t_total = time.time()

for name, init_cap in CONFIGS:
    print(f"--- {name} (${init_cap:,.0f}) ---")

    t_run = time.time()
    r = run_backtest(init_cap)
    elapsed = time.time() - t_run

    r['name'] = name
    r['initial_capital'] = init_cap
    r['runtime'] = elapsed
    results.append(r)

    blown_str = f", blown {str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
    print(f"    -> {r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
          f"${r['capital']:,.0f}, DD {r['max_drawdown']:.1f}%{blown_str}, {elapsed:.0f}s")
    print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS SUMMARY
# =====================================================

print()
print("=" * 90)
print("RESULTS — 1-MINUTE EXECUTION, CAPITAL SENSITIVITY")
print("=" * 90)
print(f"Strategy: Entry {ENTRY_ATR} ATR | Stop {STOP_ATR} ATR | Target +{TARGET_ATR} ATR")
print(f"Period: {START_DATE} to {END_DATE} | Runtime: {t_total_elapsed:.0f}s")
print()

print(f"{'Config':<14s} {'Start':>10s} {'Final':>14s} {'Return':>9s} {'Trades':>7s} "
      f"{'WR':>7s} {'PF':>6s} {'MaxDD':>8s} {'FillRate':>9s} {'Blown':>12s}")
print("-" * 105)

for r in results:
    fill_rate = r['stats']['entries_filled'] / r['stats']['entries_attempted'] * 100 if r['stats']['entries_attempted'] > 0 else 0
    init_cap = r['initial_capital']
    ret = (r['capital'] - init_cap) / init_cap * 100
    blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
    print(f"{r['name']:<14s} ${init_cap:>9,.0f} ${r['capital']:>13,.0f} {ret:>+8.1f}% {r['trades']:>7d} "
          f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
          f"{fill_rate:>8.1f}% {blown:>12s}")

# Per-asset breakdown
print()
print("=" * 90)
print("PER-ASSET BREAKDOWN")
print("=" * 90)
print()

for r in results:
    if r['asset_stats']:
        print(f"{r['name']}:")
        for a, s in sorted(r['asset_stats'].items()):
            print(f"  {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, ${s['pnl']:>14,.2f}")
        print()

# Execution statistics
print("=" * 90)
print("EXECUTION STATISTICS")
print("=" * 90)
print()

print(f"{'Config':<20s} {'Signals':>8s} {'Attempted':>10s} {'Filled':>8s} {'NoFill':>8s} "
      f"{'SameBarSL':>10s} {'SameBarTP':>10s}")
print("-" * 80)

for r in results:
    s = r['stats']
    print(f"{r['name']:<20s} {s['signals_generated']:>8d} {s['entries_attempted']:>10d} "
          f"{s['entries_filled']:>8d} {s['entries_no_fill']:>8d} "
          f"{s['same_bar_stops']:>10d} {s['same_bar_targets']:>10d}")

# Diagnostics
print()
print("=" * 90)
print("EXECUTION DIAGNOSTICS")
print("=" * 90)
print()

print(f"{'Config':<20s} {'Opened':>8s} {'Closed':>8s} {'ForcedSL':>9s} {'MaxOpen':>8s} {'AccountBlown':>20s}")
print("-" * 80)

for r in results:
    d = r['diag']
    blown = str(r['account_blown_ts'])[:19] if r['account_blown_ts'] else 'No'
    print(f"{r['name']:<20s} {d['positions_opened']:>8d} {d['positions_closed']:>8d} "
          f"{d['forced_stops_capital']:>9d} {d['max_simultaneous_open']:>8d} {blown:>20s}")

# Reference comparison
print()
print("=" * 90)
print("REFERENCE: 15-MINUTE EXECUTION (CB -0.5 ATR, $300 start)")
print("=" * 90)
print()
print("  15-min backtest:  9,759 trades | 88.2% WR | PF 7.44 | DD -12.7% | $300 → $14.5M")
print()
# Find best surviving config
survivors = [r for r in results if r['account_blown_ts'] is None]
if survivors:
    best_surv = max(survivors, key=lambda r: r['capital'])
    print(f"  Best 1-min result: {best_surv['trades']:,d} trades | {best_surv['win_rate']:.1f}% WR | "
          f"PF {best_surv['profit_factor']:.2f} | DD {best_surv['max_drawdown']:.1f}% | "
          f"${best_surv['initial_capital']:,.0f} → ${best_surv['capital']:,.0f}")
else:
    print("  All capital levels blew up. Strategy unprofitable with 1-min execution.")
print()

# Save results to file
output_file = 'results/1m_execution_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION BACKTEST — CAPITAL SENSITIVITY\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Strategy: Entry {ENTRY_ATR} ATR | Stop {STOP_ATR} ATR | Target +{TARGET_ATR} ATR\n")
    f.write(f"Period: {START_DATE} to {END_DATE}\n")
    f.write(f"Assets: {', '.join(ASSETS)}\n")
    f.write(f"Circuit breakers: ON | Progressive risk cap: ON\n")
    f.write(f"Total runtime: {t_total_elapsed:.0f}s\n\n")

    f.write(f"{'Config':<14s} {'Start':>10s} {'Final':>14s} {'Return':>9s} {'Trades':>7s} "
            f"{'WR':>7s} {'PF':>6s} {'MaxDD':>8s} {'FillRate':>9s} {'Blown':>12s}\n")
    f.write("-" * 105 + "\n")

    for r in results:
        fill_rate = r['stats']['entries_filled'] / r['stats']['entries_attempted'] * 100 if r['stats']['entries_attempted'] > 0 else 0
        init_cap = r['initial_capital']
        ret = (r['capital'] - init_cap) / init_cap * 100
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        f.write(f"{r['name']:<14s} ${init_cap:>9,.0f} ${r['capital']:>13,.0f} {ret:>+8.1f}% {r['trades']:>7d} "
                f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
                f"{fill_rate:>8.1f}% {blown:>12s}\n")

    f.write("\n\nPER-ASSET BREAKDOWN:\n")
    for r in results:
        if r['asset_stats']:
            f.write(f"\n{r['name']}:\n")
            for a, s in sorted(r['asset_stats'].items()):
                f.write(f"  {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, ${s['pnl']:>14,.2f}\n")

    f.write(f"\n\nEXECUTION DIAGNOSTICS:\n")
    f.write(f"{'Config':<14s} {'Opened':>8s} {'Closed':>8s} {'ForcedSL':>9s} "
            f"{'MaxOpen':>8s} {'AccountBlown':>20s}\n")
    f.write("-" * 80 + "\n")
    for r in results:
        d = r['diag']
        blown = str(r['account_blown_ts'])[:19] if r['account_blown_ts'] else 'No'
        f.write(f"{r['name']:<14s} {d['positions_opened']:>8d} {d['positions_closed']:>8d} "
                f"{d['forced_stops_capital']:>9d} {d['max_simultaneous_open']:>8d} {blown:>20s}\n")

    f.write("\n" + "=" * 90 + "\n")

print(f"\nResults saved: {output_file}")

# Save best config trade log
best = max(results, key=lambda r: r['profit_factor'])
best['trades_df'].to_csv('results/1m_execution_trades.csv', index=False)
print(f"Best config ({best['name']}) trades saved: results/1m_execution_trades.csv")

print()
print("=" * 90)
