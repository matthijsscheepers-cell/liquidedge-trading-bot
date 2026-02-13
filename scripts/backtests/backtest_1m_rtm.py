"""
1-Minute Execution Backtest: Return-to-Mean (RTM) Value Entry

EMA-reclaim strategy with persistent limit orders and partial exits.

Filters:
  1H:  TTM momentum > 0
  15m: Squeeze active OR ended within last 3 closed bars (regime filter)

Entry:
  Buy limit at EMA21 − 0.75 × ATR(20)
  Limit order persists across multiple 1m/15m bars until regime invalidates.
  Fill when 1m low <= limit price. Fill at limit price (exact).

Stop:
  EMA21 − 1.75 × ATR(20)  (fixed at entry, no trail)

Targets (partial exit):
  T1: EMA21              → close 50%
  T2: EMA21 + 0.75 × ATR → close remaining 50%

Risk per entry = 1.0 ATR.  Blended R:R if both targets hit = 1.125:1.
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
COMMISSION_PCT = 0.001
STOP_SLIPPAGE_PCT = 0.001
RISK_PER_TRADE = 0.02
LEVERAGE = 20
MAX_POSITIONS = 2
MIN_CAPITAL = 1.0

CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']
SQUEEZE_LOOKBACK = 4  # current bar + 3 previous = "ended within last 3 bars"

CONFIGS = [
    ("$1K start",   1_000.0),
    ("$5K start",   5_000.0),
    ("$10K start", 10_000.0),
    ("$25K start", 25_000.0),
]


def get_progressive_risk_cap(capital):
    if capital < 1000: return 50.0
    elif capital < 5000: return 100.0
    elif capital < 20000: return 200.0
    elif capital < 100000: return 500.0
    else: return 1000.0


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — RTM VALUE ENTRY")
print("=" * 90)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Entry: limit at EMA-0.75×ATR | Stop: EMA-1.75×ATR")
print(f"T1: EMA (50%) | T2: EMA+0.75×ATR (50%) | Direction: LONG only")
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
    df_1m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE)
    print(f"{len(df_1m)} 1m bars", end=" -> ", flush=True)

    df_15m = resample_ohlcv(df_1m, '15min')
    df_1h = resample_ohlcv(df_1m, '1h')
    print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars", end=" ", flush=True)

    df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
    df_15m['atr_20'] = calculate_atr(
        df_15m['high'], df_15m['low'], df_15m['close'], period=20
    )

    squeeze_on, momentum, _, _ = calculate_ttm_squeeze_pinescript(
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

    period_labels = df_1m.index.floor('15min')
    grouped = {ts: group for ts, group in df_1m.groupby(period_labels)}

    all_data_15m[asset] = df_15m
    all_data_1h[asset] = df_1h
    grouped_1m[asset] = grouped
    del df_1m
    print("OK")

t_load = time.time() - t_start
print(f"\nData loaded in {t_load:.0f}s")

common_index = None
for asset in all_data_15m:
    idx = all_data_15m[asset].index
    common_index = idx if common_index is None else common_index.intersection(idx)

for asset in all_data_15m:
    all_data_15m[asset] = all_data_15m[asset].loc[common_index]

print(f"Common 15min bars: {len(common_index)}")
print()


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(initial_capital):
    capital = initial_capital
    positions = {}
    pending_limits = {}   # {asset: {'limit': ..., 'stop': ..., 't1': ..., 't2': ..., 'ema': ..., 'atr': ...}}
    trades = []
    equity_values = []

    consecutive_stops = {}
    asset_cooldowns = {}
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False
    account_blown = False
    account_blown_ts = None

    stats = {
        'regimes_active': 0,
        'limits_placed': 0,
        'limits_updated': 0,
        'limits_cancelled': 0,
        'limits_filled': 0,
        'exits_stop': 0,
        'exits_t1_stop': 0,
        'exits_t1_t2': 0,
        'same_bar_stops': 0,
        'same_bar_t1': 0,
        'same_bar_t1_t2': 0,
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'forced_stops_capital': 0,
        'max_simultaneous_open': 0,
        'zero_size_blocked': 0,
    }

    empty_df = pd.DataFrame()

    for i in range(201, len(common_index)):
        timestamp = common_index[i]

        # --- Hard capital floor ---
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                account_blown_ts = timestamp
                diag['forced_stops_capital'] += 1
                for asset, pos in list(positions.items()):
                    bar = all_data_15m[asset].iloc[i]
                    ep = bar['close']
                    rem = pos['remaining_size']
                    ec = pos['entry_price'] * rem * COMMISSION_PCT
                    xc = ep * rem * COMMISSION_PCT
                    final_pnl = (ep - pos['entry_price']) * rem - ec - xc
                    capital += final_pnl
                    total_pnl = pos['t1_pnl'] + final_pnl
                    risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
                    trades.append({
                        'symbol': asset, 'pnl': total_pnl,
                        'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        't1_hit': pos['t1_hit'],
                        'r_multiple': total_pnl / risk if risk > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                positions.clear()
                pending_limits.clear()
            equity_values.append(max(capital, 0))
            continue

        # --- Track equity ---
        equity = capital
        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]
            equity += (bar['close'] - pos['entry_price']) * pos['remaining_size']
        equity_values.append(max(equity, 0))

        if len(positions) > diag['max_simultaneous_open']:
            diag['max_simultaneous_open'] = len(positions)

        # --- Daily reset ---
        current_date = timestamp.date() if hasattr(timestamp, 'date') else None
        if current_date and current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ========================================
        # PHASE 1: EXIT MANAGEMENT (1-minute)
        # ========================================
        closed_this_bar = set()
        closed_at_ts = {}

        for asset in list(positions.keys()):
            if asset in closed_this_bar:
                continue

            pos = positions[asset]
            candles = grouped_1m[asset].get(timestamp, empty_df)

            if candles.empty:
                # Fallback: 15m bar exit check
                bar = all_data_15m[asset].iloc[i]

                if bar['low'] <= pos['stop_loss']:
                    ep = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    rem = pos['remaining_size']
                    ec = pos['entry_price'] * rem * COMMISSION_PCT
                    xc = ep * rem * COMMISSION_PCT
                    final_pnl = (ep - pos['entry_price']) * rem - ec - xc
                    capital += final_pnl
                    total_pnl = pos['t1_pnl'] + final_pnl
                    risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
                    reason = 'T1_STOP' if pos['t1_hit'] else 'STOP'
                    trades.append({
                        'symbol': asset, 'pnl': total_pnl, 'exit_reason': reason,
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        't1_hit': pos['t1_hit'],
                        'r_multiple': total_pnl / risk if risk > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    if pos['t1_hit']:
                        stats['exits_t1_stop'] += 1
                    else:
                        stats['exits_stop'] += 1
                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(total_pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True

                else:
                    # Check targets on 15m bar (no stop hit)
                    if not pos['t1_hit'] and bar['high'] >= pos['target_1']:
                        t1s = pos['initial_size'] * 0.5
                        t1ec = pos['entry_price'] * t1s * COMMISSION_PCT
                        t1xc = pos['target_1'] * t1s * COMMISSION_PCT
                        t1p = (pos['target_1'] - pos['entry_price']) * t1s - t1ec - t1xc
                        capital += t1p
                        pos['t1_pnl'] += t1p
                        pos['remaining_size'] = pos['initial_size'] * 0.5
                        pos['t1_hit'] = True

                    if pos['t1_hit'] and bar['high'] >= pos['target_2']:
                        rem = pos['remaining_size']
                        ec = pos['entry_price'] * rem * COMMISSION_PCT
                        xc = pos['target_2'] * rem * COMMISSION_PCT
                        final_pnl = (pos['target_2'] - pos['entry_price']) * rem - ec - xc
                        capital += final_pnl
                        total_pnl = pos['t1_pnl'] + final_pnl
                        risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
                        trades.append({
                            'symbol': asset, 'pnl': total_pnl, 'exit_reason': 'T1_T2',
                            'bars_held': i - pos['entry_bar'],
                            'entry_time': pos['entry_time'], 'exit_time': timestamp,
                            'entry_price': pos['entry_price'], 'exit_price': pos['target_2'],
                            't1_hit': True,
                            'r_multiple': total_pnl / risk if risk > 0 else 0,
                        })
                        closed_this_bar.add(asset)
                        diag['positions_closed'] += 1
                        stats['exits_t1_t2'] += 1
                        consecutive_stops[asset] = 0

                continue

            # --- 1-minute exit scanning ---
            cl = candles['low'].values
            ch = candles['high'].values
            ct = candles.index.values

            for k in range(len(cl)):
                if ct[k] <= pos['last_processed_ts']:
                    continue

                # Stop first (worst case)
                if cl[k] <= pos['stop_loss']:
                    ep = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    rem = pos['remaining_size']
                    ec = pos['entry_price'] * rem * COMMISSION_PCT
                    xc = ep * rem * COMMISSION_PCT
                    final_pnl = (ep - pos['entry_price']) * rem - ec - xc
                    capital += final_pnl
                    total_pnl = pos['t1_pnl'] + final_pnl
                    risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
                    reason = 'T1_STOP' if pos['t1_hit'] else 'STOP'
                    trades.append({
                        'symbol': asset, 'pnl': total_pnl, 'exit_reason': reason,
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        't1_hit': pos['t1_hit'],
                        'r_multiple': total_pnl / risk if risk > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    closed_at_ts[asset] = ct[k]
                    diag['positions_closed'] += 1
                    if pos['t1_hit']:
                        stats['exits_t1_stop'] += 1
                    else:
                        stats['exits_stop'] += 1
                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(total_pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True
                    break

                # T1: partial close
                if not pos['t1_hit'] and ch[k] >= pos['target_1']:
                    t1s = pos['initial_size'] * 0.5
                    t1ec = pos['entry_price'] * t1s * COMMISSION_PCT
                    t1xc = pos['target_1'] * t1s * COMMISSION_PCT
                    t1p = (pos['target_1'] - pos['entry_price']) * t1s - t1ec - t1xc
                    capital += t1p
                    pos['t1_pnl'] += t1p
                    pos['remaining_size'] = pos['initial_size'] * 0.5
                    pos['t1_hit'] = True
                    stats['same_bar_t1'] += 1  # tracked for any T1 hit

                # T2: close runner (only after T1)
                if pos['t1_hit'] and ch[k] >= pos['target_2']:
                    rem = pos['remaining_size']
                    ec = pos['entry_price'] * rem * COMMISSION_PCT
                    xc = pos['target_2'] * rem * COMMISSION_PCT
                    final_pnl = (pos['target_2'] - pos['entry_price']) * rem - ec - xc
                    capital += final_pnl
                    total_pnl = pos['t1_pnl'] + final_pnl
                    risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
                    trades.append({
                        'symbol': asset, 'pnl': total_pnl, 'exit_reason': 'T1_T2',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': pos['target_2'],
                        't1_hit': True,
                        'r_multiple': total_pnl / risk if risk > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    closed_at_ts[asset] = ct[k]
                    diag['positions_closed'] += 1
                    stats['exits_t1_t2'] += 1
                    consecutive_stops[asset] = 0
                    break
            else:
                # No full close — advance timestamp
                if len(ct) > 0:
                    pos['last_processed_ts'] = ct[-1]

        for asset in closed_this_bar:
            positions.pop(asset, None)

        # ========================================
        # PHASE 2: LIMIT ORDER MANAGEMENT & ENTRY
        # ========================================
        if capital <= MIN_CAPITAL or trading_halted:
            pending_limits.clear()
            continue

        for asset in ASSETS:
            if asset in positions or len(positions) >= MAX_POSITIONS:
                # Cancel pending if we already have a position
                pending_limits.pop(asset, None)
                continue

            # Circuit breaker cooldown
            if asset in asset_cooldowns:
                if i < asset_cooldowns[asset]:
                    pending_limits.pop(asset, None)
                    continue
                else:
                    del asset_cooldowns[asset]
                    consecutive_stops.pop(asset, None)

            df_15m = all_data_15m[asset]
            df_1h = all_data_1h[asset]

            prev_bar = df_15m.iloc[i - 1]
            prev_timestamp = common_index[i - 1]

            # --- 1H filter ---
            bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                pending_limits.pop(asset, None)
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
                pending_limits.pop(asset, None)
                if asset in pending_limits:
                    stats['limits_cancelled'] += 1
                continue

            # --- Squeeze regime: active OR ended within last 3 bars ---
            squeeze_regime = False
            for lb in range(SQUEEZE_LOOKBACK):
                bar_idx = i - 1 - lb
                if bar_idx >= 0:
                    sq = df_15m.iloc[bar_idx]['squeeze_on']
                    if not pd.isna(sq) and sq:
                        squeeze_regime = True
                        break

            if not squeeze_regime:
                if asset in pending_limits:
                    stats['limits_cancelled'] += 1
                    del pending_limits[asset]
                continue

            # --- Calculate levels from closed bar i-1 ---
            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']
            if pd.isna(ema) or pd.isna(atr) or atr <= 0 or ema <= 0:
                pending_limits.pop(asset, None)
                continue

            limit_price = ema - 0.75 * atr
            stop_level = ema - 1.75 * atr
            target_1 = ema
            target_2 = ema + 0.75 * atr
            risk_dist = limit_price - stop_level  # = 1.0 * ATR

            if risk_dist <= 0 or limit_price <= 0 or stop_level <= 0:
                pending_limits.pop(asset, None)
                continue

            stats['regimes_active'] += 1

            # Update or create pending limit
            was_pending = asset in pending_limits
            pending_limits[asset] = {
                'limit': limit_price,
                'stop': stop_level,
                't1': target_1,
                't2': target_2,
                'risk_dist': risk_dist,
            }
            if was_pending:
                stats['limits_updated'] += 1
            else:
                stats['limits_placed'] += 1

            # --- Pre-filter: can 15m bar i reach the limit? ---
            curr_bar_15m = df_15m.iloc[i]
            if curr_bar_15m['low'] > limit_price:
                continue  # no 1m candle can fill

            # --- Scan 1m candles for limit fill ---
            candles = grouped_1m[asset].get(timestamp, empty_df)
            if candles.empty:
                continue

            cl = candles['low'].values
            ch = candles['high'].values
            ct = candles.index.values

            for j in range(len(cl)):
                if asset in closed_at_ts and ct[j] <= closed_at_ts[asset]:
                    continue

                if cl[j] <= limit_price:
                    # === LIMIT FILL ===
                    fill_price = limit_price  # exact limit fill

                    # Position sizing
                    risk_cap = get_progressive_risk_cap(capital)
                    risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
                    if risk_amount <= 0:
                        diag['zero_size_blocked'] += 1
                        break
                    size = risk_amount / risk_dist
                    if size <= 0:
                        diag['zero_size_blocked'] += 1
                        break
                    margin = size * fill_price / LEVERAGE
                    if margin > capital:
                        break

                    stats['limits_filled'] += 1
                    diag['positions_opened'] += 1
                    del pending_limits[asset]

                    # Check same-bar exits on remaining candles
                    same_bar_exit = False
                    sb_t1_hit = False
                    sb_t1_pnl = 0.0
                    sb_remaining = size

                    for k in range(j + 1, len(cl)):
                        # Stop first (worst case)
                        if cl[k] <= stop_level:
                            ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * sb_remaining * COMMISSION_PCT
                            xc = ep * sb_remaining * COMMISSION_PCT
                            final_pnl = (ep - fill_price) * sb_remaining - ec - xc
                            capital += final_pnl
                            total_pnl = sb_t1_pnl + final_pnl
                            risk = risk_dist * size
                            reason = 'T1_STOP' if sb_t1_hit else 'STOP'
                            trades.append({
                                'symbol': asset, 'pnl': total_pnl,
                                'exit_reason': reason + '_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                't1_hit': sb_t1_hit,
                                'r_multiple': total_pnl / risk if risk > 0 else 0,
                            })
                            stats['same_bar_stops'] += 1
                            diag['positions_closed'] += 1
                            consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                            daily_losses += abs(total_pnl)
                            if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                                asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                            if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                                trading_halted = True
                            same_bar_exit = True
                            break

                        # T1
                        if not sb_t1_hit and ch[k] >= target_1:
                            t1s = size * 0.5
                            t1ec = fill_price * t1s * COMMISSION_PCT
                            t1xc = target_1 * t1s * COMMISSION_PCT
                            t1p = (target_1 - fill_price) * t1s - t1ec - t1xc
                            capital += t1p
                            sb_t1_pnl += t1p
                            sb_remaining = size * 0.5
                            sb_t1_hit = True
                            stats['same_bar_t1'] += 1

                        # T2 (after T1)
                        if sb_t1_hit and ch[k] >= target_2:
                            ec = fill_price * sb_remaining * COMMISSION_PCT
                            xc = target_2 * sb_remaining * COMMISSION_PCT
                            final_pnl = (target_2 - fill_price) * sb_remaining - ec - xc
                            capital += final_pnl
                            total_pnl = sb_t1_pnl + final_pnl
                            risk = risk_dist * size
                            trades.append({
                                'symbol': asset, 'pnl': total_pnl,
                                'exit_reason': 'T1_T2_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': target_2,
                                't1_hit': True,
                                'r_multiple': total_pnl / risk if risk > 0 else 0,
                            })
                            stats['same_bar_t1_t2'] += 1
                            diag['positions_closed'] += 1
                            consecutive_stops[asset] = 0
                            same_bar_exit = True
                            break

                    if not same_bar_exit:
                        last_ts = ct[-1] if len(ct) > 0 else ct[j]
                        positions[asset] = {
                            'entry_time': pd.Timestamp(ct[j]),
                            'entry_bar': i,
                            'entry_price': fill_price,
                            'stop_loss': stop_level,
                            'target_1': target_1,
                            'target_2': target_2,
                            'initial_size': size,
                            'remaining_size': sb_remaining,
                            't1_hit': sb_t1_hit,
                            't1_pnl': sb_t1_pnl,
                            'last_processed_ts': last_ts,
                        }

                    break  # one fill per asset per 15m bar

        # Progress
        if i % 10000 == 0:
            progress = (i - 201) / (len(common_index) - 201) * 100
            n_pending = len(pending_limits)
            print(f"    {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}  pending:{n_pending}")

    # Close remaining positions at market
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        ep = bar['close']
        rem = pos['remaining_size']
        ec = pos['entry_price'] * rem * COMMISSION_PCT
        xc = ep * rem * COMMISSION_PCT
        final_pnl = (ep - pos['entry_price']) * rem - ec - xc
        capital += final_pnl
        total_pnl = pos['t1_pnl'] + final_pnl
        risk = (pos['entry_price'] - pos['stop_loss']) * pos['initial_size']
        trades.append({
            'symbol': asset, 'pnl': total_pnl, 'exit_reason': 'END',
            'bars_held': len(common_index) - 1 - pos['entry_bar'],
            'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
            'entry_price': pos['entry_price'], 'exit_price': ep,
            't1_hit': pos['t1_hit'],
            'r_multiple': total_pnl / risk if risk > 0 else 0,
        })
        diag['positions_closed'] += 1

    # Metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq = pd.Series(equity_values)

    if len(eq) > 0:
        rm = eq.expanding().max().clip(lower=1e-6)
        max_dd = ((eq - rm) / rm).min() * 100
    else:
        max_dd = 0

    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        wr = len(wins) / len(trades_df) * 100
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        avg_r = trades_df['r_multiple'].mean() if 'r_multiple' in trades_df.columns else 0
    else:
        wr = pf = avg_win = avg_loss = avg_r = 0

    asset_stats = {}
    for a in ASSETS:
        at = trades_df[trades_df['symbol'] == a] if len(trades_df) > 0 else pd.DataFrame()
        if len(at) > 0:
            aw = at[at['pnl'] > 0]
            asset_stats[a] = {
                'trades': len(at),
                'wr': len(aw) / len(at) * 100,
                'pnl': at['pnl'].sum(),
                'avg_r': at['r_multiple'].mean() if 'r_multiple' in at.columns else 0,
            }

    return {
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_r': avg_r,
        'asset_stats': asset_stats,
        'stats': stats,
        'diag': diag,
        'account_blown_ts': account_blown_ts,
        'trades_df': trades_df,
        'equity': equity_values,
        'initial_capital': initial_capital,
    }


# =====================================================
# RUN
# =====================================================

print("=" * 90)
print("RUNNING CAPITAL SENSITIVITY TEST")
print("=" * 90)
print()

results = []
t_total = time.time()

for name, init_cap in CONFIGS:
    print(f"--- {name} (${init_cap:,.0f}) ---")
    t_run = time.time()
    r = run_backtest(init_cap)
    elapsed = time.time() - t_run
    r['name'] = name
    r['runtime'] = elapsed
    results.append(r)

    blown_str = f", blown {str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
    print(f"    -> {r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
          f"${r['capital']:,.0f}, DD {r['max_drawdown']:.1f}%, avgR {r['avg_r']:+.2f}"
          f"{blown_str}, {elapsed:.0f}s")
    print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS
# =====================================================

print()
print("=" * 90)
print("RESULTS — RTM VALUE ENTRY (1-minute execution)")
print("=" * 90)
print(f"Entry: limit EMA-0.75×ATR | Stop: EMA-1.75×ATR | T1: EMA | T2: EMA+0.75×ATR")
print(f"Period: {START_DATE} to {END_DATE} | Runtime: {t_total_elapsed:.0f}s")
print()

print(f"{'Config':<14s} {'Start':>10s} {'Final':>14s} {'Return':>9s} {'Trades':>7s} "
      f"{'WR':>7s} {'PF':>6s} {'MaxDD':>8s} {'AvgR':>6s} {'Blown':>12s}")
print("-" * 105)

for r in results:
    ic = r['initial_capital']
    ret = (r['capital'] - ic) / ic * 100
    blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
    print(f"{r['name']:<14s} ${ic:>9,.0f} ${r['capital']:>13,.0f} {ret:>+8.1f}% {r['trades']:>7d} "
          f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
          f"{r['avg_r']:>+5.2f}R {blown:>12s}")

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
            print(f"  {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, "
                  f"${s['pnl']:>14,.2f}, avgR {s['avg_r']:>+.2f}")
        print()

# Exit reason breakdown
print("=" * 90)
print("EXIT REASONS (best config by PF)")
print("=" * 90)
print()

best = max(results, key=lambda r: r['profit_factor'])
if len(best['trades_df']) > 0:
    tdf = best['trades_df']
    print(f"Config: {best['name']} | {len(tdf)} trades")
    print()
    for reason in sorted(tdf['exit_reason'].unique()):
        sub = tdf[tdf['exit_reason'] == reason]
        sw = sub[sub['pnl'] > 0]
        print(f"  {reason:<18s}: {len(sub):5d} trades, {len(sw)/len(sub)*100:5.1f}% WR, "
              f"avgR {sub['r_multiple'].mean():>+.2f}, totalPnL ${sub['pnl'].sum():>12,.2f}")

# Execution funnel
print()
print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

print(f"{'Config':<14s} {'Regimes':>8s} {'Placed':>8s} {'Updated':>8s} "
      f"{'Cancel':>8s} {'Filled':>8s} {'SameSlp':>8s} {'SameT1':>8s} {'SameT12':>8s}")
print("-" * 95)

for r in results:
    s = r['stats']
    print(f"{r['name']:<14s} {s['regimes_active']:>8d} {s['limits_placed']:>8d} "
          f"{s['limits_updated']:>8d} {s['limits_cancelled']:>8d} "
          f"{s['limits_filled']:>8d} {s['same_bar_stops']:>8d} "
          f"{s['same_bar_t1']:>8d} {s['same_bar_t1_t2']:>8d}")

# Diagnostics
print()
print("=" * 90)
print("DIAGNOSTICS")
print("=" * 90)
print()

print(f"{'Config':<14s} {'Opened':>8s} {'Closed':>8s} {'ForcedSL':>9s} "
      f"{'MaxOpen':>8s} {'ZeroBlk':>8s} {'AccountBlown':>20s}")
print("-" * 85)

for r in results:
    d = r['diag']
    blown = str(r['account_blown_ts'])[:19] if r['account_blown_ts'] else 'No'
    print(f"{r['name']:<14s} {d['positions_opened']:>8d} {d['positions_closed']:>8d} "
          f"{d['forced_stops_capital']:>9d} {d['max_simultaneous_open']:>8d} "
          f"{d['zero_size_blocked']:>8d} {blown:>20s}")

# R-multiple distribution
print()
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (best config by PF)")
print("=" * 90)
print()

if len(best['trades_df']) > 0 and 'r_multiple' in best['trades_df'].columns:
    tdf = best['trades_df']
    print(f"Config: {best['name']} | {len(tdf)} trades")
    print()
    bins = [(-10, -1.5), (-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 10)]
    for lo, hi in bins:
        n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
        pct = n / len(tdf) * 100 if len(tdf) > 0 else 0
        bar = '#' * int(pct)
        print(f"  {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")

# Validation checks
print()
print("=" * 90)
print("VALIDATION CHECKS")
print("=" * 90)
print()

for r in results:
    d = r['diag']
    s = r['stats']
    ok_balance = d['positions_opened'] == d['positions_closed']
    ok_zero = d['zero_size_blocked'] >= 0  # just report count
    ok_limits = s['limits_filled'] <= s['limits_placed'] + s['limits_updated']
    print(f"{r['name']:<14s}  "
          f"open==close: {'OK' if ok_balance else 'FAIL'} ({d['positions_opened']}/{d['positions_closed']})  "
          f"zero_blocked: {d['zero_size_blocked']}  "
          f"fills<=orders: {'OK' if ok_limits else 'FAIL'}")

print()
print("Limit orders persist across 1m bars: YES (pending_limits dict persists)")
print("No unfinished 15m bar data used:     YES (all indicators from bar i-1)")
print("Worst-case intra-candle ordering:     YES (stop checked before targets)")
print("Ghost trade prevention:               YES (MIN_CAPITAL=$1, size>0 guards)")

# Save results
output_file = 'results/1m_rtm_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION — RTM VALUE ENTRY\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Entry: limit EMA-0.75×ATR | Stop: EMA-1.75×ATR\n")
    f.write(f"T1: EMA (50%) | T2: EMA+0.75×ATR (50%)\n")
    f.write(f"Period: {START_DATE} to {END_DATE}\n")
    f.write(f"Assets: {', '.join(ASSETS)}\n\n")

    f.write(f"{'Config':<14s} {'Start':>10s} {'Final':>14s} {'Return':>9s} {'Trades':>7s} "
            f"{'WR':>7s} {'PF':>6s} {'MaxDD':>8s} {'AvgR':>6s} {'Blown':>12s}\n")
    f.write("-" * 105 + "\n")
    for r in results:
        ic = r['initial_capital']
        ret = (r['capital'] - ic) / ic * 100
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        f.write(f"{r['name']:<14s} ${ic:>9,.0f} ${r['capital']:>13,.0f} {ret:>+8.1f}% {r['trades']:>7d} "
                f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
                f"{r['avg_r']:>+5.2f}R {blown:>12s}\n")

    f.write("\n" + "=" * 90 + "\n")

print(f"\nResults saved: {output_file}")

if len(best['trades_df']) > 0:
    best['trades_df'].to_csv('results/1m_rtm_trades.csv', index=False)
    print(f"Trade log ({best['name']}): results/1m_rtm_trades.csv")

print()
print("=" * 90)
