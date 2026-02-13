"""
1-Minute Execution Backtest: TTM Squeeze Structure Breakout

Hypothesis: The TTM Squeeze is only a regime filter.
Edge comes from continuation structure after release, not mean-reversion pullbacks.

Signal (15m bar close, "release bar"):
  - squeeze_on is true
  - close > EMA21
  - close > previous 15m close
  + 1H TTM momentum > 0

Entry: Buy stop at release_bar_high
  - Persists across 15m bars until filled or invalidated
  - Invalidation: new 15m bar closes below EMA21

Stop:  release_bar_low (structural, fixed)
Target: entry + 1.5 x (entry - stop) = 1.5R

Execution: 1-minute candles only.
No partial exits, no trailing stops. Long only.
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
STOP_SLIPPAGE_PCT = 0.001   # 0.1% slippage on stop-market exits
ENTRY_SLIPPAGE_PCT = 0.001  # 0.1% slippage on buy stop fills
RISK_PER_TRADE = 0.02       # 2% base risk
LEVERAGE = 20
MAX_POSITIONS = 2
MIN_CAPITAL = 1.0
R_TARGET = 1.5              # target = entry + 1.5 x risk

CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16       # 4 hours = 16 x 15min
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']

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
# DATA LOADING (once for all configs & periods)
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — STRUCTURE BREAKOUT")
print("=" * 90)
print(f"Signal: squeeze_on + close > EMA21 + close > prev_close")
print(f"Entry: buy stop at release_bar_high | Stop: release_bar_low")
print(f"Target: entry + {R_TARGET}R | Direction: LONG only")
print(f"Period: {START_DATE} to {END_DATE}")
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

def run_backtest(initial_capital, start_bar=201):
    """Run structure breakout backtest from start_bar onwards."""
    capital = initial_capital
    positions = {}          # {asset: position_dict}
    pending_orders = {}     # {asset: {'entry_level', 'stop_level', 'signal_bar', ...}}
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
        'signals_generated': 0,
        'orders_placed': 0,
        'orders_updated': 0,
        'orders_cancelled_ema': 0,
        'orders_filled': 0,
        'orders_persisted': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
        'release_bar_ranges_atr': [],
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'forced_stops_capital': 0,
        'max_simultaneous_open': 0,
        'zero_blocked': 0,
    }

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
                    exit_price = bar['close']
                    ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
                    xc = exit_price * pos['size'] * COMMISSION_PCT
                    pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
                    capital += pnl
                    rd = pos['entry_price'] - pos['stop_loss']
                    trades.append({
                        'symbol': asset, 'pnl': pnl, 'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - pos['entry_bar'],
                        'entry_time': pos['entry_time'], 'exit_time': timestamp,
                        'entry_price': pos['entry_price'], 'exit_price': exit_price,
                        'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
                        'stop_dist_atr': pos.get('stop_dist_atr', 0),
                    })
                    diag['positions_closed'] += 1
                positions.clear()
                pending_orders.clear()
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

        for asset in list(positions.keys()):
            if asset in closed_this_bar:
                continue

            pos = positions[asset]
            candles = grouped_1m[asset].get(timestamp, empty_df)

            if candles.empty:
                # Fallback: 15m bar check
                bar = all_data_15m[asset].iloc[i]
                hit_stop = bar['low'] <= pos['stop_loss']
                hit_target = bar['high'] >= pos['target']

                if hit_stop:
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
                        'stop_dist_atr': pos.get('stop_dist_atr', 0),
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
                        'stop_dist_atr': pos.get('stop_dist_atr', 0),
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

                # Stop first (conservative)
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
                        'stop_dist_atr': pos.get('stop_dist_atr', 0),
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
                        'stop_dist_atr': pos.get('stop_dist_atr', 0),
                    })
                    closed_this_bar.add(asset)
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    consecutive_stops[asset] = 0
                    break
            else:
                # No exit — advance last_processed_ts
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

            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']

            if pd.isna(ema) or pd.isna(atr) or atr <= 0:
                pending_orders.pop(asset, None)
                continue

            # === Step A: Invalidation check ===
            # If bar i-1 closed below EMA21, cancel any pending order
            if asset in pending_orders and prev_bar['close'] < ema:
                stats['orders_cancelled_ema'] += 1
                pending_orders.pop(asset)

            # === Step B: New signal check ===
            # Release bar: squeeze_on + close > EMA21 + close > prev_close + 1H mom > 0
            if i >= 2:
                prev_prev_bar = df_15m.iloc[i - 2]

                bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
                has_1h = len(bars_1h_avail) >= 20
                mom_1h_ok = False
                if has_1h:
                    prev_1h = bars_1h_avail.iloc[-1]
                    mom_1h_ok = (not pd.isna(prev_1h['ttm_momentum'])
                                 and prev_1h['ttm_momentum'] > 0)

                squeeze_ok = (not pd.isna(prev_bar['squeeze_on'])
                              and prev_bar['squeeze_on'])
                close_above_ema = prev_bar['close'] > ema
                close_rising = (not pd.isna(prev_prev_bar['close'])
                                and prev_bar['close'] > prev_prev_bar['close'])

                if mom_1h_ok and squeeze_ok and close_above_ema and close_rising:
                    entry_level = prev_bar['high']
                    stop_level = prev_bar['low']
                    risk = entry_level - stop_level

                    if risk > 0:
                        stats['signals_generated'] += 1
                        stats['release_bar_ranges_atr'].append(risk / atr)

                        if asset in pending_orders:
                            stats['orders_updated'] += 1
                        else:
                            stats['orders_placed'] += 1

                        pending_orders[asset] = {
                            'entry_level': entry_level,
                            'stop_level': stop_level,
                            'signal_bar': i - 1,
                            'atr_at_signal': atr,
                            'stop_dist_atr': risk / atr,
                        }

            # === Step C: Fill check ===
            if asset not in pending_orders:
                continue

            order = pending_orders[asset]

            # Pre-filter: 15m bar high must reach entry level
            curr_bar_15m = df_15m.iloc[i]
            if curr_bar_15m['high'] < order['entry_level']:
                stats['orders_persisted'] += 1
                continue

            # Scan 1m candles for fill
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
                        # Gap fill — open above buy stop level
                        fill_price = co[j]
                    else:
                        fill_price = order['entry_level'] * (1 + ENTRY_SLIPPAGE_PCT)

                    stop_level = order['stop_level']
                    risk_dist = fill_price - stop_level
                    if risk_dist <= 0:
                        break

                    target = fill_price + R_TARGET * risk_dist

                    # Position sizing
                    risk_cap = get_progressive_risk_cap(capital)
                    risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
                    if risk_amount <= 0:
                        diag['zero_blocked'] += 1
                        break
                    size = risk_amount / risk_dist
                    if size <= 0:
                        diag['zero_blocked'] += 1
                        break
                    margin = size * fill_price / LEVERAGE
                    if margin > capital:
                        break

                    stats['orders_filled'] += 1
                    diag['positions_opened'] += 1

                    # === Same-candle stop check on fill candle ===
                    # For a buy stop: fill at high, if same candle's low <= stop
                    same_bar_exit = False

                    if cl[j] <= stop_level:
                        # Fill candle itself hits stop — worst case: fill then stop
                        exit_price = stop_level * (1 - STOP_SLIPPAGE_PCT)
                        ec = fill_price * size * COMMISSION_PCT
                        xc = exit_price * size * COMMISSION_PCT
                        pnl = (exit_price - fill_price) * size - ec - xc
                        capital += pnl
                        trades.append({
                            'symbol': asset, 'pnl': pnl,
                            'exit_reason': 'STOP_SAME',
                            'bars_held': 0,
                            'entry_time': pd.Timestamp(ct[j]),
                            'exit_time': pd.Timestamp(ct[j]),
                            'entry_price': fill_price, 'exit_price': exit_price,
                            'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                            'stop_dist_atr': order['stop_dist_atr'],
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

                    # === Check remaining 1m candles for same-15m-bar exits ===
                    if not same_bar_exit:
                        for k in range(j + 1, len(cl)):
                            # Stop first (conservative)
                            if cl[k] <= stop_level:
                                exit_price = stop_level * (1 - STOP_SLIPPAGE_PCT)
                                ec = fill_price * size * COMMISSION_PCT
                                xc = exit_price * size * COMMISSION_PCT
                                pnl = (exit_price - fill_price) * size - ec - xc
                                capital += pnl
                                trades.append({
                                    'symbol': asset, 'pnl': pnl,
                                    'exit_reason': 'STOP_SAME',
                                    'bars_held': 0,
                                    'entry_time': pd.Timestamp(ct[j]),
                                    'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': exit_price,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                    'stop_dist_atr': order['stop_dist_atr'],
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
                                exit_price = target
                                ec = fill_price * size * COMMISSION_PCT
                                xc = exit_price * size * COMMISSION_PCT
                                pnl = (exit_price - fill_price) * size - ec - xc
                                capital += pnl
                                trades.append({
                                    'symbol': asset, 'pnl': pnl,
                                    'exit_reason': 'TARGET_SAME',
                                    'bars_held': 0,
                                    'entry_time': pd.Timestamp(ct[j]),
                                    'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': exit_price,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                    'stop_dist_atr': order['stop_dist_atr'],
                                })
                                stats['same_bar_targets'] += 1
                                diag['positions_closed'] += 1
                                consecutive_stops[asset] = 0
                                same_bar_exit = True
                                break

                    if not same_bar_exit:
                        # Position stays open
                        last_ts = ct[-1] if len(ct) > 0 else ct[j]
                        positions[asset] = {
                            'entry_time': pd.Timestamp(ct[j]),
                            'entry_bar': i,
                            'entry_price': fill_price,
                            'stop_loss': stop_level,
                            'target': target,
                            'size': size,
                            'last_processed_ts': last_ts,
                            'stop_dist_atr': order['stop_dist_atr'],
                        }

                    # Remove pending order (filled)
                    pending_orders.pop(asset)
                    filled = True
                    break

            if not filled and asset in pending_orders:
                stats['orders_persisted'] += 1

        # Progress
        if i % 10000 == 0:
            progress = (i - start_bar) / (len(common_index) - start_bar) * 100
            print(f"    {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}")

    # Close remaining positions
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[-1]
        exit_price = bar['close']
        ec = pos['entry_price'] * pos['size'] * COMMISSION_PCT
        xc = exit_price * pos['size'] * COMMISSION_PCT
        pnl = (exit_price - pos['entry_price']) * pos['size'] - ec - xc
        capital += pnl
        rd = pos['entry_price'] - pos['stop_loss']
        trades.append({
            'symbol': asset, 'pnl': pnl, 'exit_reason': 'END',
            'bars_held': len(common_index) - 1 - pos['entry_bar'],
            'entry_time': pos['entry_time'], 'exit_time': common_index[-1],
            'entry_price': pos['entry_price'], 'exit_price': exit_price,
            'r_multiple': pnl / (rd * pos['size']) if rd > 0 and pos['size'] > 0 else 0,
            'stop_dist_atr': pos.get('stop_dist_atr', 0),
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
                'same_bar_stops': len(at[at['exit_reason'].str.startswith('STOP_SAME')]),
            }

    if stats['release_bar_ranges_atr']:
        avg_range_atr = np.mean(stats['release_bar_ranges_atr'])
        med_range_atr = np.median(stats['release_bar_ranges_atr'])
        min_range_atr = np.min(stats['release_bar_ranges_atr'])
        max_range_atr = np.max(stats['release_bar_ranges_atr'])
        p25_range_atr = np.percentile(stats['release_bar_ranges_atr'], 25)
        p75_range_atr = np.percentile(stats['release_bar_ranges_atr'], 75)
    else:
        avg_range_atr = med_range_atr = min_range_atr = max_range_atr = 0
        p25_range_atr = p75_range_atr = 0

    fill_rate = stats['orders_filled'] / stats['signals_generated'] * 100 if stats['signals_generated'] > 0 else 0

    return {
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
        'equity': equity_values,
        'initial_capital': initial_capital,
        'avg_range_atr': avg_range_atr,
        'med_range_atr': med_range_atr,
        'min_range_atr': min_range_atr,
        'max_range_atr': max_range_atr,
        'p25_range_atr': p25_range_atr,
        'p75_range_atr': p75_range_atr,
    }


# =====================================================
# RUN ALL PERIODS x CONFIGS
# =====================================================

# Determine start bars for each period
cutoff_2020 = pd.Timestamp('2020-01-01', tz='UTC')
start_bar_2020 = max(201, int(common_index.searchsorted(cutoff_2020)))

PERIODS = [
    ("Full 2010-2026", 201),
    ("Recent 2020-2026", start_bar_2020),
]

print(f"Start bars: Full={201}, Recent 2020+={start_bar_2020}")
print()

all_results = {}
t_total = time.time()

for period_name, start_bar in PERIODS:
    print("=" * 90)
    print(f"RUNNING: {period_name}")
    print("=" * 90)
    print()

    period_results = []
    for config_name, init_cap in CONFIGS:
        print(f"  --- {config_name} (${init_cap:,.0f}) ---")
        t_run = time.time()
        r = run_backtest(init_cap, start_bar=start_bar)
        elapsed = time.time() - t_run

        r['name'] = config_name
        r['runtime'] = elapsed
        period_results.append(r)

        blown_str = f", blown {str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
        print(f"      -> {r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
              f"${r['capital']:,.0f}, DD {r['max_drawdown']:.1f}%, avgR {r['avg_r']:+.2f}{blown_str}, {elapsed:.0f}s")
        print()

    all_results[period_name] = period_results

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS REPORTING
# =====================================================

print()
print("=" * 90)
print("RESULTS — TTM SQUEEZE STRUCTURE BREAKOUT (1-minute execution)")
print("=" * 90)
print(f"Entry: buy stop at release_bar_high | Stop: release_bar_low | Target: 1.5R")
print(f"Total runtime: {t_total_elapsed:.0f}s")
print()

for period_name, results in all_results.items():
    print("-" * 90)
    print(f"  {period_name}")
    print("-" * 90)
    print()

    print(f"  {'Config':<14s} {'Start':>10s} {'Final':>14s} {'Return':>9s} {'Trades':>7s} "
          f"{'WR':>7s} {'PF':>6s} {'MaxDD':>8s} {'AvgR':>6s} {'Fill%':>6s} {'Blown':>12s}")
    print("  " + "-" * 105)

    for r in results:
        ic = r['initial_capital']
        ret = (r['capital'] - ic) / ic * 100
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        print(f"  {r['name']:<14s} ${ic:>9,.0f} ${r['capital']:>13,.0f} {ret:>+8.1f}% {r['trades']:>7d} "
              f"{r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['max_drawdown']:>7.1f}% "
              f"{r['avg_r']:>+5.2f}R {r['fill_rate']:>5.1f}% {blown:>12s}")
    print()

# =====================================================
# PER-ASSET BREAKDOWN
# =====================================================

print("=" * 90)
print("PER-ASSET BREAKDOWN")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print()

    for r in results:
        if r['asset_stats']:
            print(f"  {r['name']}:")
            for a, s in sorted(r['asset_stats'].items()):
                print(f"    {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, "
                      f"${s['pnl']:>14,.2f}, avgR {s['avg_r']:>+.2f}, "
                      f"same-bar SL: {s['same_bar_stops']}")
            print()

# =====================================================
# EXECUTION FUNNEL
# =====================================================

print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print()
    print(f"  {'Config':<14s} {'Signals':>8s} {'Placed':>8s} {'Updated':>8s} "
          f"{'Cancel':>8s} {'Filled':>8s} {'Persist':>8s} {'SameSL':>8s} {'SameTP':>8s}")
    print("  " + "-" * 85)

    for r in results:
        s = r['stats']
        print(f"  {r['name']:<14s} {s['signals_generated']:>8d} {s['orders_placed']:>8d} "
              f"{s['orders_updated']:>8d} {s['orders_cancelled_ema']:>8d} "
              f"{s['orders_filled']:>8d} {s['orders_persisted']:>8d} "
              f"{s['same_bar_stops']:>8d} {s['same_bar_targets']:>8d}")
    print()

# =====================================================
# RELEASE BAR RANGE DIAGNOSTIC
# =====================================================

print("=" * 90)
print("RELEASE BAR RANGE (stop distance in ATR units)")
print("=" * 90)
print()
print("  This measures (release_bar_high - release_bar_low) / ATR(20)")
print("  = the structural stop distance in ATR units")
print()

for period_name, results in all_results.items():
    # Use the result with most signals (highest capital = most trades before blow-up)
    best = max(results, key=lambda r: r['stats']['signals_generated'])
    print(f"  {period_name} ({best['name']}, {best['stats']['signals_generated']} signals):")
    print(f"    Mean:   {best['avg_range_atr']:.3f} ATR")
    print(f"    Median: {best['med_range_atr']:.3f} ATR")
    print(f"    P25:    {best['p25_range_atr']:.3f} ATR")
    print(f"    P75:    {best['p75_range_atr']:.3f} ATR")
    print(f"    Min:    {best['min_range_atr']:.3f} ATR")
    print(f"    Max:    {best['max_range_atr']:.3f} ATR")
    print()

# =====================================================
# DIAGNOSTICS
# =====================================================

print("=" * 90)
print("DIAGNOSTICS")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print()
    print(f"  {'Config':<14s} {'Opened':>8s} {'Closed':>8s} {'ForcedSL':>9s} "
          f"{'MaxOpen':>8s} {'ZeroBlk':>8s} {'AccountBlown':>20s}")
    print("  " + "-" * 80)

    for r in results:
        d = r['diag']
        blown = str(r['account_blown_ts'])[:19] if r['account_blown_ts'] else 'No'
        print(f"  {r['name']:<14s} {d['positions_opened']:>8d} {d['positions_closed']:>8d} "
              f"{d['forced_stops_capital']:>9d} {d['max_simultaneous_open']:>8d} "
              f"{d['zero_blocked']:>8d} {blown:>20s}")
    print()

# =====================================================
# R-MULTIPLE DISTRIBUTION (best config per period)
# =====================================================

print("=" * 90)
print("R-MULTIPLE DISTRIBUTION")
print("=" * 90)
print()

for period_name, results in all_results.items():
    best = max(results, key=lambda r: r['profit_factor'])
    if len(best['trades_df']) > 0 and 'r_multiple' in best['trades_df'].columns:
        tdf = best['trades_df']
        print(f"  {period_name} — {best['name']} | {len(tdf)} trades")
        print()
        bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
                (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 10)]
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100 if len(tdf) > 0 else 0
            bar = '#' * int(pct)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# =====================================================
# COMPARISON TO PREVIOUS STRATEGIES
# =====================================================

print("=" * 90)
print("COMPARISON: ALL 1-MINUTE EXECUTION STRATEGIES")
print("=" * 90)
print()
print("  Strategy                       WR      PF     AvgR    Notes")
print("  " + "-" * 75)
print("  Limit at -0.5 ATR              31%    0.20    ~-1.5R  (previous test)")
print("  Buy stop continuation          18%    0.06    -2.00R  (previous test)")
print("  RTM value entry                13%    0.02    -1.96R  (previous test)")

# Add current results (full period, $10K)
for period_name, results in all_results.items():
    for r in results:
        if r['name'] == '$10K start':
            blown_note = f"blown {str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else "survived"
            print(f"  Structure breakout ({period_name[:8]})"
                  f"  {r['win_rate']:4.0f}%  {r['profit_factor']:5.2f}  {r['avg_r']:>+5.2f}R  ({blown_note})")

print()

# =====================================================
# VALIDATION CHECKS
# =====================================================

print("=" * 90)
print("VALIDATION CHECKS")
print("=" * 90)
print()

for period_name, results in all_results.items():
    for r in results:
        d = r['diag']
        opened = d['positions_opened']
        closed = d['positions_closed']
        ok_oc = "OK" if opened == closed else f"MISMATCH ({opened} vs {closed})"
        ok_fills = "OK" if r['stats']['orders_filled'] <= r['stats']['signals_generated'] else "FAIL"
        print(f"  {period_name[:8]} {r['name']:<14s} open==close: {ok_oc}  "
              f"fills<=signals: {ok_fills}  zero_blocked: {d['zero_blocked']}")

print()
print("  No same-bar signal execution:   YES (signal bar i-1, fill bar i+)")
print("  Persistent buy stop orders:     YES (pending_orders dict persists across bars)")
print("  Invalidation on EMA close:      YES (cancel if 15m close < EMA21)")
print("  Worst-case intra-candle order:  YES (stop checked before target, fill-candle stop check)")
print("  Ghost trade prevention:         YES (MIN_CAPITAL=$1, size>0 guards)")
print()

# =====================================================
# SAVE RESULTS
# =====================================================

output_file = 'results/1m_structure_breakout_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION — TTM SQUEEZE STRUCTURE BREAKOUT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Entry: buy stop at release_bar_high | Stop: release_bar_low | Target: {R_TARGET}R\n")
    f.write(f"Signal: squeeze_on + close > EMA21 + close > prev_close + 1H mom > 0\n")
    f.write(f"Assets: {', '.join(ASSETS)}\n")
    f.write(f"Commission: {COMMISSION_PCT*100:.1f}% per side | Stop slippage: {STOP_SLIPPAGE_PCT*100:.1f}%\n")
    f.write(f"Circuit breakers: ON | Progressive risk cap: ON\n")
    f.write(f"Total runtime: {t_total_elapsed:.0f}s\n\n")

    for period_name, results in all_results.items():
        f.write(f"\n{period_name}\n")
        f.write("-" * 50 + "\n\n")

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

        f.write("\nPer-asset:\n")
        for r in results:
            if r['asset_stats']:
                f.write(f"\n  {r['name']}:\n")
                for a, s in sorted(r['asset_stats'].items()):
                    f.write(f"    {a:8s}: {s['trades']:5d} trades, {s['wr']:5.1f}% WR, "
                            f"${s['pnl']:>14,.2f}, avgR {s['avg_r']:>+.2f}\n")

        f.write("\n")

    # Range diagnostic
    f.write("\nRelease Bar Range (ATR units):\n")
    for period_name, results in all_results.items():
        best = max(results, key=lambda r: r['stats']['signals_generated'])
        f.write(f"  {period_name}: mean={best['avg_range_atr']:.3f}, "
                f"median={best['med_range_atr']:.3f}, "
                f"p25={best['p25_range_atr']:.3f}, p75={best['p75_range_atr']:.3f}\n")

    f.write("\n" + "=" * 90 + "\n")

print(f"\nResults saved: {output_file}")

# Save trade logs
for period_name, results in all_results.items():
    best = max(results, key=lambda r: r['profit_factor'])
    if len(best['trades_df']) > 0:
        tag = "full" if "Full" in period_name else "2020"
        fname = f'results/1m_structure_breakout_trades_{tag}.csv'
        best['trades_df'].to_csv(fname, index=False)
        print(f"Trade log ({period_name}, {best['name']}): {fname}")

print()
print("=" * 90)
