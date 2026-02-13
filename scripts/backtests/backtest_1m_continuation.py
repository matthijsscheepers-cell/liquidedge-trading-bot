"""
1-Minute Execution Backtest: Squeeze Continuation Setup

New strategy designed for realistic 1-minute execution using buy stop orders.

Filters:
  1H:  TTM momentum > 0
  15m: squeeze_on AND momentum > 0 AND close > EMA21

Entry (1-minute):
  1. Detect upward impulse (2+ consecutive bullish 1m candles)
  2. Detect pullback (2+ consecutive bearish 1m candles, lows above 15m EMA21)
  3. Buy stop at: high of last bearish candle + 1 tick
  4. Fill on NEXT 1m candle (gap fill at open if opens above)

Exit:
  Stop  = lowest low of pullback candles
  Target = entry + 2 × (entry - stop)   → 2R
  Conservative: stop wins if both hit same candle
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
ENTRY_SLIPPAGE_PCT = 0.001   # buy stop fill slippage
RISK_PER_TRADE = 0.02
LEVERAGE = 20
MAX_POSITIONS = 2
MIN_CAPITAL = 1.0

CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']
TICK_SIZES = {'GOLD': 0.10, 'SILVER': 0.005}

MIN_BULLISH_FOR_IMPULSE = 2
MIN_BEARISH_FOR_PULLBACK = 2

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


def reset_entry_state():
    return {
        'impulse_seen': False,
        'bullish_streak': 0,
        'bearish_streak': 0,
        'pullback_lows': [],
        'pending_entry': None,
        'pending_stop': None,
        'pending_candle_count': 0,
    }


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — SQUEEZE CONTINUATION")
print("=" * 90)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Impulse: {MIN_BULLISH_FOR_IMPULSE}+ bullish | Pullback: {MIN_BEARISH_FOR_PULLBACK}+ bearish")
print(f"Target: 2R | Entry: buy stop | Direction: LONG only")
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
    trades = []
    equity_values = []

    entry_state = {a: reset_entry_state() for a in ASSETS}

    consecutive_stops = {}
    asset_cooldowns = {}
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False
    account_blown = False
    account_blown_ts = None

    stats = {
        'setups_active': 0,
        'impulses_seen': 0,
        'pullbacks_formed': 0,
        'buy_stops_filled': 0,
        'buy_stops_cancelled': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'forced_stops_capital': 0,
        'max_simultaneous_open': 0,
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
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = ep * pos['size'] * COMMISSION_PCT
                    pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        'pullback_candles': pos.get('pullback_candles', 0),
                        'stop_ticks': pos.get('stop_ticks', 0),
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                positions.clear()
            equity_values.append(max(capital, 0))
            continue

        # --- Track equity ---
        equity = capital
        for asset, pos in positions.items():
            bar = all_data_15m[asset].iloc[i]
            equity += (bar['close'] - pos['entry_price']) * pos['size']
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
                bar = all_data_15m[asset].iloc[i]
                hit_stop = bar['low'] <= pos['stop_loss']
                hit_target = bar['high'] >= pos['target']

                if hit_stop:
                    ep = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = ep * pos['size'] * COMMISSION_PCT
                    pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        'pullback_candles': pos.get('pullback_candles', 0),
                        'stop_ticks': pos.get('stop_ticks', 0),
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
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

                elif hit_target:
                    ep = pos['target']
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = ep * pos['size'] * COMMISSION_PCT
                    pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        'pullback_candles': pos.get('pullback_candles', 0),
                        'stop_ticks': pos.get('stop_ticks', 0),
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0

                continue

            # --- 1-minute exit scanning ---
            cl = candles['low'].values
            ch = candles['high'].values
            ct = candles.index.values

            for k in range(len(cl)):
                if ct[k] <= pos['last_processed_ts']:
                    continue

                if cl[k] <= pos['stop_loss']:
                    ep = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = ep * pos['size'] * COMMISSION_PCT
                    pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        'pullback_candles': pos.get('pullback_candles', 0),
                        'stop_ticks': pos.get('stop_ticks', 0),
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    closed_at_ts[asset] = ct[k]
                    diag['positions_closed'] += 1
                    stats['exits_stop'] += 1
                    consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                    daily_losses += abs(pnl)
                    if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                        asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                    if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                        trading_halted = True
                    break

                if ch[k] >= pos['target']:
                    ep = pos['target']
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = ep * pos['size'] * COMMISSION_PCT
                    pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': ep,
                        'pullback_candles': pos.get('pullback_candles', 0),
                        'stop_ticks': pos.get('stop_ticks', 0),
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    closed_at_ts[asset] = ct[k]
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0
                    break
            else:
                if len(ct) > 0:
                    pos['last_processed_ts'] = ct[-1]

        for asset in closed_this_bar:
            positions.pop(asset, None)
            entry_state[asset] = reset_entry_state()

        # ========================================
        # PHASE 2: ENTRY SCANNING (1-minute)
        # ========================================
        if capital <= MIN_CAPITAL or len(positions) >= MAX_POSITIONS or trading_halted:
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

            # 1H filter
            bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                entry_state[asset] = reset_entry_state()
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
                entry_state[asset] = reset_entry_state()
                continue

            # 15m setup: squeeze_on AND momentum > 0 AND close > EMA21
            ema = prev_bar['ema_21']
            if (pd.isna(prev_bar['squeeze_on']) or not prev_bar['squeeze_on']
                    or pd.isna(prev_bar['ttm_momentum']) or prev_bar['ttm_momentum'] <= 0
                    or pd.isna(ema) or ema <= 0
                    or prev_bar['close'] <= ema):
                entry_state[asset] = reset_entry_state()
                continue

            stats['setups_active'] += 1

            candles = grouped_1m[asset].get(timestamp, empty_df)
            if candles.empty:
                continue

            tick = TICK_SIZES[asset]
            st = entry_state[asset]

            co = candles['open'].values
            ch = candles['high'].values
            cl = candles['low'].values
            cc = candles['close'].values
            ct = candles.index.values

            for j in range(len(cc)):
                if asset in closed_at_ts and ct[j] <= closed_at_ts[asset]:
                    continue

                is_bull = cc[j] > co[j]
                is_bear = cc[j] < co[j]

                # ---- PENDING BUY STOP CHECK ----
                if st['pending_entry'] is not None:

                    if ch[j] >= st['pending_entry']:
                        # === BUY STOP FILLED ===
                        if co[j] >= st['pending_entry']:
                            fill_price = co[j] * (1 + ENTRY_SLIPPAGE_PCT)
                        else:
                            fill_price = st['pending_entry'] * (1 + ENTRY_SLIPPAGE_PCT)

                        stop = st['pending_stop']
                        risk_dist = fill_price - stop
                        if risk_dist <= 0:
                            st = reset_entry_state()
                            entry_state[asset] = st
                            break

                        target = fill_price + 2.0 * risk_dist
                        pb_count = st['pending_candle_count']
                        stop_ticks = risk_dist / tick

                        risk_cap = get_progressive_risk_cap(capital)
                        risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
                        if risk_amount <= 0:
                            break
                        size = risk_amount / risk_dist
                        if size <= 0:
                            break
                        margin = size * fill_price / LEVERAGE
                        if margin > capital:
                            break

                        stats['buy_stops_filled'] += 1
                        diag['positions_opened'] += 1

                        # Check same-bar exits on remaining candles
                        same_bar_exit = False
                        for k in range(j + 1, len(cl)):
                            if cl[k] <= stop:
                                ep = stop * (1 - STOP_SLIPPAGE_PCT)
                                ec = fill_price * size * COMMISSION_PCT
                                xc = ep * size * COMMISSION_PCT
                                pnl = (ep - fill_price) * size - ec - xc
                                capital += pnl
                                trades.append({
                                    'symbol': asset, 'pnl': pnl,
                                    'exit_reason': 'STOP_SAME_15M',
                                    'bars_held': 0,
                                    'entry_time': pd.Timestamp(ct[j]),
                                    'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': ep,
                                    'pullback_candles': pb_count,
                                    'stop_ticks': stop_ticks,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                })
                                stats['same_bar_stops'] += 1
                                diag['positions_closed'] += 1
                                consecutive_stops[asset] = consecutive_stops.get(asset, 0) + 1
                                daily_losses += abs(pnl)
                                if consecutive_stops[asset] >= CB_CONSECUTIVE_STOP_LIMIT:
                                    asset_cooldowns[asset] = i + CB_COOLDOWN_BARS
                                if daily_losses >= get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULTIPLIER:
                                    trading_halted = True
                                same_bar_exit = True
                                break

                            if ch[k] >= target:
                                ep = target
                                ec = fill_price * size * COMMISSION_PCT
                                xc = ep * size * COMMISSION_PCT
                                pnl = (ep - fill_price) * size - ec - xc
                                capital += pnl
                                trades.append({
                                    'symbol': asset, 'pnl': pnl,
                                    'exit_reason': 'TARGET_SAME_15M',
                                    'bars_held': 0,
                                    'entry_time': pd.Timestamp(ct[j]),
                                    'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': ep,
                                    'pullback_candles': pb_count,
                                    'stop_ticks': stop_ticks,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                })
                                stats['same_bar_targets'] += 1
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
                                'stop_loss': stop,
                                'target': target,
                                'size': size,
                                'last_processed_ts': last_ts,
                                'pullback_candles': pb_count,
                                'stop_ticks': stop_ticks,
                            }

                        st = reset_entry_state()
                        entry_state[asset] = st
                        break

                    elif is_bear:
                        # Pullback extends — update buy stop
                        if cl[j] >= ema:
                            st['pullback_lows'].append(cl[j])
                            st['bearish_streak'] += 1
                            st['bullish_streak'] = 0
                            st['pending_entry'] = ch[j] + tick
                            st['pending_stop'] = min(st['pullback_lows'])
                            st['pending_candle_count'] = len(st['pullback_lows'])
                        else:
                            stats['buy_stops_cancelled'] += 1
                            st = reset_entry_state()
                            entry_state[asset] = st

                    elif is_bull:
                        # Failed breakout — cancel pending
                        stats['buy_stops_cancelled'] += 1
                        st['pending_entry'] = None
                        st['pending_stop'] = None
                        st['pending_candle_count'] = 0
                        st['pullback_lows'] = []
                        st['bearish_streak'] = 0
                        st['bullish_streak'] = 1
                    else:
                        # Doji — keep pending, reset streaks
                        st['bullish_streak'] = 0
                        st['bearish_streak'] = 0

                    continue

                # ---- STATE MACHINE: impulse & pullback ----
                if is_bull:
                    st['bullish_streak'] += 1
                    st['bearish_streak'] = 0
                    st['pullback_lows'] = []
                    if st['bullish_streak'] >= MIN_BULLISH_FOR_IMPULSE:
                        if not st['impulse_seen']:
                            stats['impulses_seen'] += 1
                        st['impulse_seen'] = True

                elif is_bear:
                    st['bearish_streak'] += 1
                    st['bullish_streak'] = 0

                    if st['impulse_seen']:
                        if cl[j] < ema:
                            st['bearish_streak'] = 0
                            st['pullback_lows'] = []
                            st['impulse_seen'] = False
                        else:
                            st['pullback_lows'].append(cl[j])
                            if st['bearish_streak'] >= MIN_BEARISH_FOR_PULLBACK:
                                was_pending = st['pending_entry'] is not None
                                st['pending_entry'] = ch[j] + tick
                                st['pending_stop'] = min(st['pullback_lows'])
                                st['pending_candle_count'] = len(st['pullback_lows'])
                                if not was_pending:
                                    stats['pullbacks_formed'] += 1
                else:
                    # Doji
                    st['bullish_streak'] = 0
                    st['bearish_streak'] = 0
                    st['pullback_lows'] = []

            entry_state[asset] = st

        # Progress
        if i % 10000 == 0:
            progress = (i - 201) / (len(common_index) - 201) * 100
            print(f"    {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}")

    # Close remaining positions
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        ep = bar['close']
        ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
        xc = ep * pos['size'] * COMMISSION_PCT
        pnl = (ep - pos['entry_price']) * pos['size'] - ec - xc
        capital += pnl
        rd = pos['entry_price'] - pos['stop_loss']
        trades.append({
            'symbol': asset, 'pnl': pnl, 'exit_reason': 'END',
            'bars_held': len(common_index) - 1 - pos['entry_bar'],
            'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
            'entry_price': pos['entry_price'], 'exit_price': ep,
            'pullback_candles': pos.get('pullback_candles', 0),
            'stop_ticks': pos.get('stop_ticks', 0),
            'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
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
          f"${r['capital']:,.0f}, DD {r['max_drawdown']:.1f}%{blown_str}, {elapsed:.0f}s")
    print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS
# =====================================================

print()
print("=" * 90)
print("RESULTS — SQUEEZE CONTINUATION (1-minute execution)")
print("=" * 90)
print(f"Target: 2R | Entry: buy stop | Period: {START_DATE} to {END_DATE}")
print(f"Runtime: {t_total_elapsed:.0f}s")
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

# Execution funnel
print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

print(f"{'Config':<14s} {'Setups':>8s} {'Impulse':>8s} {'Pullbck':>8s} "
      f"{'Filled':>8s} {'Cancel':>8s} {'SameSlp':>8s} {'SameTgt':>8s}")
print("-" * 85)

for r in results:
    s = r['stats']
    print(f"{r['name']:<14s} {s['setups_active']:>8d} {s['impulses_seen']:>8d} "
          f"{s['pullbacks_formed']:>8d} {s['buy_stops_filled']:>8d} "
          f"{s['buy_stops_cancelled']:>8d} {s['same_bar_stops']:>8d} "
          f"{s['same_bar_targets']:>8d}")

# Diagnostics
print()
print("=" * 90)
print("DIAGNOSTICS")
print("=" * 90)
print()

print(f"{'Config':<14s} {'Opened':>8s} {'Closed':>8s} {'ForcedSL':>9s} "
      f"{'MaxOpen':>8s} {'AccountBlown':>20s}")
print("-" * 75)

for r in results:
    d = r['diag']
    blown = str(r['account_blown_ts'])[:19] if r['account_blown_ts'] else 'No'
    print(f"{r['name']:<14s} {d['positions_opened']:>8d} {d['positions_closed']:>8d} "
          f"{d['forced_stops_capital']:>9d} {d['max_simultaneous_open']:>8d} {blown:>20s}")

# R-multiple distribution (best config)
print()
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (best config by PF)")
print("=" * 90)
print()

best = max(results, key=lambda r: r['profit_factor'])
if len(best['trades_df']) > 0 and 'r_multiple' in best['trades_df'].columns:
    tdf = best['trades_df']
    print(f"Config: {best['name']} | {len(tdf)} trades")
    print()
    bins = [(-10, -1.5), (-1.5, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 10)]
    for lo, hi in bins:
        n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
        pct = n / len(tdf) * 100 if len(tdf) > 0 else 0
        bar = '#' * int(pct)
        print(f"  {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")

    if 'pullback_candles' in tdf.columns:
        print()
        print("Pullback depth (bearish candles):")
        for pc in sorted(tdf['pullback_candles'].unique()):
            sub = tdf[tdf['pullback_candles'] == pc]
            sw = sub[sub['pnl'] > 0]
            print(f"  {int(pc):2d} candles: {len(sub):5d} trades, "
                  f"{len(sw)/len(sub)*100:.1f}% WR, avgR {sub['r_multiple'].mean():>+.2f}")

# Save results
output_file = 'results/1m_continuation_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION — SQUEEZE CONTINUATION\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Target: 2R | Entry: buy stop | Period: {START_DATE} to {END_DATE}\n")
    f.write(f"Impulse: {MIN_BULLISH_FOR_IMPULSE}+ bullish | Pullback: {MIN_BEARISH_FOR_PULLBACK}+ bearish\n")
    f.write(f"Assets: {', '.join(ASSETS)}\n")
    f.write(f"Circuit breakers: ON | Progressive risk cap: ON\n\n")

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

# Save trade log
if len(best['trades_df']) > 0:
    best['trades_df'].to_csv('results/1m_continuation_trades.csv', index=False)
    print(f"Trade log ({best['name']}): results/1m_continuation_trades.csv")

print()
print("=" * 90)
