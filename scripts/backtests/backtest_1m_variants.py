"""
1-Minute Execution Backtest: Multi-Variant Grid

Tests multiple hypotheses in a single run:

A) Structure breakout + swing low stop (wider structural stop)
   - Entry: buy stop at release_bar_high
   - Stop: last 15m swing low before release bar
   - Target: R-multiple (1.5R, 2.0R, 3.0R)

B) ATR limit orders with wider stops (user's live trading observation)
   - Entry: persistent limit at EMA - 1.0 ATR
   - Stop: EMA - 2.0 ATR (1.0 ATR risk, wider than previous -1.5 ATR)
   - Target: EMA + 2.0 ATR (3R) or EMA + 1.5 ATR (2.5R)

C) Momentum filter variants
   - Standard: squeeze_on + 1H mom > 0
   - Turning: above + 15m momentum rising (yellow histogram)
   - Positive: above + 15m momentum > 0

All with 1-minute execution, persistent orders, worst-case ordering.
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
ENTRY_SLIPPAGE_PCT = 0.001  # for buy stops only; limits fill at exact price
RISK_PER_TRADE = 0.02
LEVERAGE = 20
MAX_POSITIONS = 2
MIN_CAPITAL = 1.0

CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']
SWING_LOW_LOOKBACK = 20  # bars to scan for swing low
SQUEEZE_LOOKBACK = 4     # bars to check for recent squeeze (for limit orders)

# Each config is a dict describing the variant
CONFIGS = [
    # --- A) Structure breakout + swing low stop ---
    {
        'name': 'SB SwLo 1.5R',
        'entry_type': 'buystop',  # buy stop at release_bar_high
        'stop_type': 'swing_low',
        'target_type': 'r_mult',
        'target_r': 1.5,
        'mom_filter': 'standard',
    },
    {
        'name': 'SB SwLo 2.0R',
        'entry_type': 'buystop',
        'stop_type': 'swing_low',
        'target_type': 'r_mult',
        'target_r': 2.0,
        'mom_filter': 'standard',
    },
    {
        'name': 'SB SwLo 3.0R',
        'entry_type': 'buystop',
        'stop_type': 'swing_low',
        'target_type': 'r_mult',
        'target_r': 3.0,
        'mom_filter': 'standard',
    },

    # --- B) ATR limit order variants (user's live observation) ---
    {
        'name': 'Lim-1 S-2 T+2',
        'entry_type': 'limit',
        'entry_atr': -1.0,       # limit at EMA - 1.0 ATR
        'stop_type': 'atr',
        'stop_atr': -2.0,        # stop at EMA - 2.0 ATR
        'target_type': 'atr_abs',
        'target_atr': 2.0,       # target at EMA + 2.0 ATR
        'mom_filter': 'standard',
    },
    {
        'name': 'Lim-1 S-2 T+1.5',
        'entry_type': 'limit',
        'entry_atr': -1.0,
        'stop_type': 'atr',
        'stop_atr': -2.0,
        'target_type': 'atr_abs',
        'target_atr': 1.5,       # lower target
        'mom_filter': 'standard',
    },
    {
        'name': 'Lim-0.5 S-2 T+2',
        'entry_type': 'limit',
        'entry_atr': -0.5,       # original entry level
        'stop_type': 'atr',
        'stop_atr': -2.0,        # but wider stop
        'target_type': 'atr_abs',
        'target_atr': 2.0,
        'mom_filter': 'standard',
    },

    # --- C) Momentum filter variants on best ATR setup ---
    {
        'name': 'Lim-1 S-2 T+2 turn',
        'entry_type': 'limit',
        'entry_atr': -1.0,
        'stop_type': 'atr',
        'stop_atr': -2.0,
        'target_type': 'atr_abs',
        'target_atr': 2.0,
        'mom_filter': 'turning',  # 15m momentum must be rising
    },
    {
        'name': 'Lim-1 S-2 T+2 pos',
        'entry_type': 'limit',
        'entry_atr': -1.0,
        'stop_type': 'atr',
        'stop_atr': -2.0,
        'target_type': 'atr_abs',
        'target_atr': 2.0,
        'mom_filter': 'positive',  # 15m momentum > 0
    },
]

INITIAL_CAPITAL = 10_000.0


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


def find_swing_low(df_15m, before_idx, lookback=SWING_LOW_LOOKBACK):
    """
    Find the most recent 15m swing low before bar at before_idx.
    Swing low: low[k] < low[k-1] AND low[k] < low[k+1].
    Returns the low value, or None if not found.
    """
    # k must be < before_idx, and k-1 >= 0, and k+1 < before_idx
    for k in range(before_idx - 2, max(before_idx - lookback, 0), -1):
        if k < 1:
            break
        low_k = df_15m.iloc[k]['low']
        low_prev = df_15m.iloc[k - 1]['low']
        low_next = df_15m.iloc[k + 1]['low']
        if low_k < low_prev and low_k < low_next:
            return low_k
    return None


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — MULTI-VARIANT GRID")
print("=" * 90)
print(f"Configs: {len(CONFIGS)} variants")
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Capital: ${INITIAL_CAPITAL:,.0f}")
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

def run_backtest(config, initial_capital=INITIAL_CAPITAL, start_bar=201):
    """Run a single variant backtest."""
    capital = initial_capital
    positions = {}
    pending_orders = {}  # {asset: {'entry_level', 'stop_level', 'target_level', ...}}
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
        'signals': 0,
        'orders_placed': 0,
        'orders_cancelled': 0,
        'orders_filled': 0,
        'orders_persisted': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
        'stop_distances_atr': [],
        'no_swing_low': 0,
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'forced_stops_capital': 0,
        'max_simultaneous_open': 0,
    }

    is_buystop = config['entry_type'] == 'buystop'
    is_limit = config['entry_type'] == 'limit'

    empty_df = pd.DataFrame()

    for i in range(start_bar, len(common_index)):
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
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                positions.clear()
                pending_orders.clear()
            equity_values.append(max(capital, 0))
            continue

        # --- Equity tracking ---
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

        for asset in list(positions.keys()):
            if asset in closed_this_bar:
                continue

            pos = positions[asset]
            candles = grouped_1m[asset].get(timestamp, empty_df)

            if candles.empty:
                bar = all_data_15m[asset].iloc[i]
                if bar['low'] <= pos['stop_loss']:
                    exit_price = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
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
                elif bar['high'] >= pos['target']:
                    exit_price = pos['target']
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
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
                    exit_price = pos['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
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
                    break

                if ch[k] >= pos['target']:
                    exit_price = pos['target']
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'],
                        'exit_time': pd.Timestamp(ct[k]),
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0
                    break
            else:
                if len(ct) > 0:
                    pos['last_processed_ts'] = ct[-1]

        for asset in closed_this_bar:
            positions.pop(asset, None)

        # ========================================
        # PHASE 2: ORDER MANAGEMENT & ENTRIES
        # ========================================
        if capital <= MIN_CAPITAL or trading_halted:
            continue

        for asset in ASSETS:
            if asset in positions or len(positions) >= MAX_POSITIONS:
                continue

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

            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']

            if pd.isna(ema) or pd.isna(atr) or atr <= 0:
                pending_orders.pop(asset, None)
                continue

            # 1H momentum filter (required for all configs)
            bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                pending_orders.pop(asset, None)
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
                pending_orders.pop(asset, None)
                continue

            # ============================================================
            # ENTRY TYPE A: BUY STOP (structure breakout + swing low stop)
            # ============================================================
            if is_buystop:
                # Invalidation: bar i-1 closed below EMA21
                if asset in pending_orders and prev_bar['close'] < ema:
                    stats['orders_cancelled'] += 1
                    pending_orders.pop(asset)

                # New signal: release bar (squeeze_on + close > EMA + close > prev close)
                if i >= 2:
                    prev_prev_bar = df_15m.iloc[i - 2]

                    squeeze_ok = not pd.isna(prev_bar['squeeze_on']) and prev_bar['squeeze_on']
                    close_above_ema = prev_bar['close'] > ema
                    close_rising = (not pd.isna(prev_prev_bar['close'])
                                    and prev_bar['close'] > prev_prev_bar['close'])

                    if squeeze_ok and close_above_ema and close_rising:
                        entry_level = prev_bar['high']

                        # Find swing low for stop
                        swing_low = find_swing_low(df_15m, i - 1)

                        if swing_low is None:
                            stats['no_swing_low'] += 1
                            # Fallback: skip this signal (no valid structural stop)
                        else:
                            risk = entry_level - swing_low
                            if risk > 0 and risk / atr < 10:  # sanity: max 10 ATR stop
                                stats['signals'] += 1
                                stats['stop_distances_atr'].append(risk / atr)

                                if asset not in pending_orders:
                                    stats['orders_placed'] += 1

                                target_r = config['target_r']
                                target_level = entry_level + target_r * risk

                                pending_orders[asset] = {
                                    'entry_level': entry_level,
                                    'stop_level': swing_low,
                                    'target_level': target_level,
                                    'signal_bar': i - 1,
                                    'stop_dist_atr': risk / atr,
                                }

                # Fill check
                if asset not in pending_orders:
                    continue

                order = pending_orders[asset]
                curr_bar = df_15m.iloc[i]
                if curr_bar['high'] < order['entry_level']:
                    stats['orders_persisted'] += 1
                    continue

                candles = grouped_1m[asset].get(timestamp, empty_df)
                if candles.empty:
                    stats['orders_persisted'] += 1
                    continue

                co = candles['open'].values
                ch = candles['high'].values
                cl = candles['low'].values
                ct = candles.index.values

                filled = False
                for j in range(len(ch)):
                    if ch[j] >= order['entry_level']:
                        # Buy stop triggered
                        if co[j] >= order['entry_level']:
                            fill_price = co[j]  # gap fill
                        else:
                            fill_price = order['entry_level'] * (1 + ENTRY_SLIPPAGE_PCT)

                        stop_level = order['stop_level']
                        risk_dist = fill_price - stop_level
                        if risk_dist <= 0:
                            break

                        target_level = order['target_level']
                        # Adjust target for actual fill price
                        target_level = fill_price + config['target_r'] * risk_dist

                        # Position sizing
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

                        stats['orders_filled'] += 1
                        diag['positions_opened'] += 1

                        # Same-candle stop check (fill candle)
                        same_bar_exit = False
                        if cl[j] <= stop_level:
                            ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[j]),
                                'entry_price': fill_price, 'exit_price': ep,
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

                        # Remaining candles same-bar check
                        if not same_bar_exit:
                            for k in range(j + 1, len(cl)):
                                if cl[k] <= stop_level:
                                    ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                                    ec = fill_price * size * COMMISSION_PCT
                                    xc = ep * size * COMMISSION_PCT
                                    pnl = (ep - fill_price) * size - ec - xc
                                    capital += pnl
                                    trades.append({
                                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                        'bars_held': 0,
                                        'entry_time': pd.Timestamp(ct[j]),
                                        'exit_time': pd.Timestamp(ct[k]),
                                        'entry_price': fill_price, 'exit_price': ep,
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

                                if ch[k] >= target_level:
                                    ep = target_level
                                    ec = fill_price * size * COMMISSION_PCT
                                    xc = ep * size * COMMISSION_PCT
                                    pnl = (ep - fill_price) * size - ec - xc
                                    capital += pnl
                                    trades.append({
                                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                        'bars_held': 0,
                                        'entry_time': pd.Timestamp(ct[j]),
                                        'exit_time': pd.Timestamp(ct[k]),
                                        'entry_price': fill_price, 'exit_price': ep,
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
                                'stop_loss': stop_level,
                                'target': target_level,
                                'size': size,
                                'last_processed_ts': last_ts,
                            }

                        pending_orders.pop(asset)
                        filled = True
                        break

                if not filled and asset in pending_orders:
                    stats['orders_persisted'] += 1

            # ============================================================
            # ENTRY TYPE B: LIMIT ORDER (ATR-based, persistent)
            # ============================================================
            elif is_limit:
                mom_filter = config.get('mom_filter', 'standard')

                # Check squeeze condition (active in last N bars)
                squeeze_active = False
                for lb in range(0, SQUEEZE_LOOKBACK):
                    idx = i - 1 - lb
                    if idx >= 0:
                        bar_lb = df_15m.iloc[idx]
                        if not pd.isna(bar_lb['squeeze_on']) and bar_lb['squeeze_on']:
                            squeeze_active = True
                            break

                if not squeeze_active:
                    if asset in pending_orders:
                        stats['orders_cancelled'] += 1
                    pending_orders.pop(asset, None)
                    continue

                # Momentum filter
                mom_ok = True
                if mom_filter == 'turning':
                    # 15m momentum must be rising (current > previous)
                    if i >= 2:
                        m_curr = prev_bar['ttm_momentum']
                        m_prev = df_15m.iloc[i - 2]['ttm_momentum']
                        if pd.isna(m_curr) or pd.isna(m_prev) or m_curr <= m_prev:
                            mom_ok = False
                    else:
                        mom_ok = False
                elif mom_filter == 'positive':
                    # 15m momentum must be > 0
                    m = prev_bar['ttm_momentum']
                    if pd.isna(m) or m <= 0:
                        mom_ok = False

                if not mom_ok:
                    # Don't cancel pending — just don't create new ones
                    # But still try to fill existing pending orders
                    pass
                else:
                    # Calculate limit levels from current bar i-1 indicators
                    entry_atr = config['entry_atr']
                    stop_atr = config['stop_atr']
                    target_atr = config['target_atr']

                    limit_price = ema + entry_atr * atr
                    stop_level = ema + stop_atr * atr
                    target_level = ema + target_atr * atr

                    risk_dist = limit_price - stop_level
                    reward = target_level - limit_price

                    if risk_dist > 0 and reward > 0:
                        if asset not in pending_orders:
                            stats['orders_placed'] += 1
                            stats['signals'] += 1
                            stats['stop_distances_atr'].append(risk_dist / atr)

                        pending_orders[asset] = {
                            'entry_level': limit_price,
                            'stop_level': stop_level,
                            'target_level': target_level,
                            'signal_bar': i - 1,
                            'stop_dist_atr': risk_dist / atr,
                        }

                # Fill check for pending limit
                if asset not in pending_orders:
                    continue

                order = pending_orders[asset]

                # Pre-filter: 15m bar low must reach limit price
                curr_bar = df_15m.iloc[i]
                if curr_bar['low'] > order['entry_level']:
                    stats['orders_persisted'] += 1
                    continue

                candles = grouped_1m[asset].get(timestamp, empty_df)
                if candles.empty:
                    stats['orders_persisted'] += 1
                    continue

                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values

                filled = False
                for j in range(len(cl)):
                    if cl[j] <= order['entry_level']:
                        # Limit fill — exact price (no slippage on limits)
                        fill_price = order['entry_level']
                        stop_level = order['stop_level']
                        target_level = order['target_level']

                        risk_dist = fill_price - stop_level
                        if risk_dist <= 0:
                            break

                        # Position sizing
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

                        stats['orders_filled'] += 1
                        diag['positions_opened'] += 1

                        # Same-candle stop check (conservative: stop wins if both hit)
                        same_bar_exit = False

                        # For limit orders: fill is at the LOW side of the candle.
                        # If cl[j] <= entry AND cl[j] <= stop: both fill and stop on same candle.
                        # Worst case: fill first, then stop.
                        if cl[j] <= stop_level:
                            ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[j]),
                                'entry_price': fill_price, 'exit_price': ep,
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

                        # Remaining candles
                        if not same_bar_exit:
                            for k in range(j + 1, len(cl)):
                                if cl[k] <= stop_level:
                                    ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                                    ec = fill_price * size * COMMISSION_PCT
                                    xc = ep * size * COMMISSION_PCT
                                    pnl = (ep - fill_price) * size - ec - xc
                                    capital += pnl
                                    trades.append({
                                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                        'bars_held': 0,
                                        'entry_time': pd.Timestamp(ct[j]),
                                        'exit_time': pd.Timestamp(ct[k]),
                                        'entry_price': fill_price, 'exit_price': ep,
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

                                if ch[k] >= target_level:
                                    ep = target_level
                                    ec = fill_price * size * COMMISSION_PCT
                                    xc = ep * size * COMMISSION_PCT
                                    pnl = (ep - fill_price) * size - ec - xc
                                    capital += pnl
                                    trades.append({
                                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                        'bars_held': 0,
                                        'entry_time': pd.Timestamp(ct[j]),
                                        'exit_time': pd.Timestamp(ct[k]),
                                        'entry_price': fill_price, 'exit_price': ep,
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
                                'stop_loss': stop_level,
                                'target': target_level,
                                'size': size,
                                'last_processed_ts': last_ts,
                            }

                        pending_orders.pop(asset)
                        filled = True
                        break

                if not filled and asset in pending_orders:
                    stats['orders_persisted'] += 1

        # Progress
        if i % 20000 == 0:
            progress = (i - start_bar) / (len(common_index) - start_bar) * 100
            print(f"      {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}", flush=True)

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
            'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
        })
        diag['positions_closed'] += 1

    # ---- Metrics ----
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
        avg_r = trades_df['r_multiple'].mean()
    else:
        wr = pf = avg_r = 0

    asset_stats = {}
    for a in ASSETS:
        at = trades_df[trades_df['symbol'] == a] if len(trades_df) > 0 else pd.DataFrame()
        if len(at) > 0:
            aw = at[at['pnl'] > 0]
            asset_stats[a] = {
                'trades': len(at),
                'wr': len(aw) / len(at) * 100,
                'pnl': at['pnl'].sum(),
                'avg_r': at['r_multiple'].mean(),
            }

    fill_rate = stats['orders_filled'] / stats['signals'] * 100 if stats['signals'] > 0 else 0

    avg_stop_atr = np.mean(stats['stop_distances_atr']) if stats['stop_distances_atr'] else 0
    med_stop_atr = np.median(stats['stop_distances_atr']) if stats['stop_distances_atr'] else 0

    return {
        'config_name': config['name'],
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_r': avg_r,
        'fill_rate': fill_rate,
        'asset_stats': asset_stats,
        'stats': stats,
        'diag': diag,
        'account_blown_ts': account_blown_ts,
        'trades_df': trades_df,
        'initial_capital': initial_capital,
        'avg_stop_atr': avg_stop_atr,
        'med_stop_atr': med_stop_atr,
    }


# =====================================================
# RUN ALL CONFIGS × PERIODS
# =====================================================

cutoff_2020 = pd.Timestamp('2020-01-01', tz='UTC')
start_bar_2020 = max(201, int(common_index.searchsorted(cutoff_2020)))

PERIODS = [
    ("Full 2010-2026", 201),
    ("Recent 2020-2026", start_bar_2020),
]

all_results = {}
t_total = time.time()

for period_name, start_bar in PERIODS:
    print("=" * 90)
    print(f"PERIOD: {period_name}")
    print("=" * 90)
    print()

    period_results = []
    for config in CONFIGS:
        print(f"  {config['name']}...", end=" ", flush=True)
        t_run = time.time()
        r = run_backtest(config, start_bar=start_bar)
        elapsed = time.time() - t_run

        r['runtime'] = elapsed
        period_results.append(r)

        blown_str = f" blown:{str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
        print(f"{r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
              f"avgR {r['avg_r']:+.2f}, ${r['capital']:,.0f}, DD {r['max_drawdown']:.1f}%{blown_str} ({elapsed:.0f}s)")

    all_results[period_name] = period_results
    print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS REPORT
# =====================================================

print()
print("=" * 90)
print("RESULTS — MULTI-VARIANT GRID (1-minute execution, $10K start)")
print("=" * 90)
print(f"Total runtime: {t_total_elapsed:.0f}s")
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print()
    print(f"  {'Variant':<22s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
          f"{'MaxDD':>8s} {'Fill%':>6s} {'StopATR':>8s} {'SameSL':>7s} {'Final':>10s} {'Blown':>12s}")
    print("  " + "-" * 115)

    for r in results:
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        print(f"  {r['config_name']:<22s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
              f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
              f"{r['med_stop_atr']:>7.2f}a "
              f"{r['stats']['same_bar_stops']:>7d} "
              f"${r['capital']:>9,.0f} {blown:>12s}")
    print()

# Per-asset breakdown for best configs
print("=" * 90)
print("PER-ASSET BREAKDOWN (selected configs)")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    for r in results:
        if r['asset_stats']:
            print(f"  {r['config_name']}:")
            for a, s in sorted(r['asset_stats'].items()):
                print(f"    {a:8s}: {s['trades']:4d} trades, {s['wr']:5.1f}% WR, "
                      f"${s['pnl']:>12,.2f}, avgR {s['avg_r']:>+.2f}")
    print()

# Execution funnel
print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print(f"  {'Variant':<22s} {'Signals':>8s} {'Placed':>8s} {'Cancel':>8s} "
          f"{'Filled':>8s} {'Persist':>8s} {'NoSwLo':>8s}")
    print("  " + "-" * 75)
    for r in results:
        s = r['stats']
        print(f"  {r['config_name']:<22s} {s['signals']:>8d} {s['orders_placed']:>8d} "
              f"{s['orders_cancelled']:>8d} {s['orders_filled']:>8d} "
              f"{s['orders_persisted']:>8d} {s['no_swing_low']:>8d}")
    print()

# R-multiple distribution for best config
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (best PF per period)")
print("=" * 90)
print()

for period_name, results in all_results.items():
    best = max(results, key=lambda r: r['profit_factor'])
    if len(best['trades_df']) > 0 and 'r_multiple' in best['trades_df'].columns:
        tdf = best['trades_df']
        print(f"  {period_name} — {best['config_name']} | {len(tdf)} trades")
        print()
        bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
                (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 10)]
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100 if len(tdf) > 0 else 0
            bar = '#' * int(pct)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# Comparison with all previous strategies
print("=" * 90)
print("COMPARISON: ALL 1-MINUTE STRATEGIES TESTED")
print("=" * 90)
print()
print(f"  {'Strategy':<35s} {'WR':>6s} {'PF':>6s} {'AvgR':>7s} {'StopATR':>8s} {'Notes'}")
print("  " + "-" * 85)
print(f"  {'Limit -0.5 S-1.5 T+1.5 (orig)':<35s} {'31%':>6s} {'0.20':>6s} {'-1.5R':>7s} {'1.0a':>8s} {'blown'}")
print(f"  {'Buy stop continuation 2R':<35s} {'18%':>6s} {'0.06':>6s} {'-2.0R':>7s} {'~0.5a':>8s} {'blown'}")
print(f"  {'RTM limit -0.75 partials':<35s} {'13%':>6s} {'0.02':>6s} {'-2.0R':>7s} {'1.0a':>8s} {'blown'}")
print(f"  {'Structure breakout 1.5R':<35s} {'32%':>6s} {'0.25':>6s} {'-1.3R':>7s} {'0.8a':>8s} {'blown'}")

# Add current best results
for period_name, results in all_results.items():
    best = max(results, key=lambda r: r['profit_factor'])
    tag = "full" if "Full" in period_name else "2020+"
    blown = "blown" if best['account_blown_ts'] else "survived"
    print(f"  {best['config_name'] + ' (' + tag + ')':<35s} "
          f"{best['win_rate']:>5.0f}% {best['profit_factor']:>6.2f} "
          f"{best['avg_r']:>+6.2f}R {best['med_stop_atr']:>7.2f}a {blown}")
print()

# Validation
print("=" * 90)
print("VALIDATION")
print("=" * 90)
print()
for period_name, results in all_results.items():
    for r in results:
        d = r['diag']
        ok = "OK" if d['positions_opened'] == d['positions_closed'] else "MISMATCH"
        print(f"  {period_name[:8]} {r['config_name']:<22s} open==close: {ok} ({d['positions_opened']}/{d['positions_closed']})")
print()

# Save results
output_file = 'results/1m_variants_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION — MULTI-VARIANT GRID\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,.0f} | Runtime: {t_total_elapsed:.0f}s\n\n")

    for period_name, results in all_results.items():
        f.write(f"\n{period_name}\n")
        f.write("-" * 50 + "\n\n")

        f.write(f"{'Variant':<22s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
                f"{'MaxDD':>8s} {'Fill%':>6s} {'StopATR':>8s} {'Blown':>12s}\n")
        f.write("-" * 100 + "\n")

        for r in results:
            blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
            f.write(f"{r['config_name']:<22s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                    f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
                    f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
                    f"{r['med_stop_atr']:>7.2f}a {blown:>12s}\n")

        f.write("\nPer-asset:\n")
        for r in results:
            if r['asset_stats']:
                f.write(f"\n  {r['config_name']}:\n")
                for a, s in sorted(r['asset_stats'].items()):
                    f.write(f"    {a:8s}: {s['trades']:4d} trades, {s['wr']:5.1f}% WR, "
                            f"${s['pnl']:>12,.2f}, avgR {s['avg_r']:>+.2f}\n")
        f.write("\n")

    f.write("=" * 90 + "\n")

print(f"Results saved: {output_file}")

# Save best trade log
for period_name, results in all_results.items():
    best = max(results, key=lambda r: r['profit_factor'])
    if len(best['trades_df']) > 0:
        tag = "full" if "Full" in period_name else "2020"
        fname = f'results/1m_variants_trades_{tag}.csv'
        best['trades_df'].to_csv(fname, index=False)
        print(f"Trade log ({period_name}, {best['config_name']}): {fname}")

print()
print("=" * 90)
