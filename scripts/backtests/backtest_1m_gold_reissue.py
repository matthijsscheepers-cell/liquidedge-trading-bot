"""
1-Minute Execution Backtest: GOLD Order Handling A/B/C Test

Clean comparison of three order-handling modes, all else equal:

  A: TRUE PERSISTENT
     First valid bar → place limit order with FIXED levels.
     Levels NEVER change. Cancel only on hard invalidation
     (1H <= 0, squeeze gone, 2+ consecutive red bars).

  B: RE-ISSUE PER BAR
     Each valid 15m bar → cancel prior unfilled order, issue fresh limit
     at CURRENT EMA/ATR levels. Valid only during next 15m bar's candles.
     Expired if not filled.

  C: DYNAMIC UPDATE (reference — what produced PF 2.51 previously)
     Order persists across bars, but levels update to current EMA/ATR
     each bar. This is what was previously tested.

All three share:
- Entry: EMA(21) - 1.0 ATR | Stop: EMA(21) - 2.0 ATR | Target: EMA(21) + 2.0 ATR
- 1H mom > 0, 15m squeeze in last 4 bars, color != red
- GOLD only, post-2020, $10K start, 2% risk
- Frictions: 0.1% commission each side + 0.1% stop slippage
- 1-minute execution, worst-case ordering
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

START_DATE = '2020-01-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.02
LEVERAGE = 20
MIN_CAPITAL = 1.0

COMMISSION_PCT = 0.001       # 0.1% each side
STOP_SLIPPAGE_PCT = 0.001   # 0.1% adverse on stops

SQUEEZE_LOOKBACK = 4

# ATR-based levels
ENTRY_ATR_MULT = -1.0    # EMA - 1.0 ATR
STOP_ATR_MULT = -2.0     # EMA - 2.0 ATR
TARGET_ATR_MULT = 2.0    # EMA + 2.0 ATR
# Risk = 1.0 ATR, Reward = 3.0 ATR, R:R = 1:3

RED_BAR_TOLERANCE = 2  # cancel persistent order after N consecutive red bars


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def get_histogram_color(mom_curr, mom_prev):
    if pd.isna(mom_curr) or pd.isna(mom_prev):
        return 'red'
    if mom_curr > 0 and mom_curr > mom_prev:
        return 'light_blue'
    elif mom_curr > 0:
        return 'dark_blue'
    elif mom_curr > mom_prev:
        return 'yellow'
    else:
        return 'red'


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION — GOLD ORDER HANDLING A/B/C TEST")
print("=" * 90)
print(f"Entry: EMA - 1.0 ATR | Stop: EMA - 2.0 ATR | Target: EMA + 2.0 ATR (1:3 R:R)")
print(f"Frictions: {COMMISSION_PCT*100:.1f}% commission + {STOP_SLIPPAGE_PCT*100:.1f}% stop slip")
print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}%")
print(f"Period: {START_DATE} to {END_DATE}")
print()
print("Loading GOLD data...")

loader = DatabentoMicroFuturesLoader()
t_start = time.time()

df_1m_raw = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE)
print(f"  {len(df_1m_raw)} 1m bars", end=" -> ", flush=True)

df_15m = resample_ohlcv(df_1m_raw, '15min')
df_1h = resample_ohlcv(df_1m_raw, '1h')
print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars")

# 15m indicators
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

# 1H TTM momentum
_, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
    df_1h['high'], df_1h['low'], df_1h['close'],
    bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
)
df_1h['ttm_momentum'] = momentum_1h

# Pre-group 1m candles
period_labels = df_1m_raw.index.floor('15min')
grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}
del df_1m_raw

t_load = time.time() - t_start
print(f"Data loaded in {t_load:.0f}s | {len(df_15m)} bars\n")


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(mode='persistent'):
    """
    mode: 'persistent' | 'reissue' | 'dynamic'
    """
    capital = INITIAL_CAPITAL
    position = None
    pending_order = None   # {entry_level, stop_level, target_level, placed_bar}
    trades = []
    equity_values = []

    account_blown = False
    account_blown_ts = None
    consecutive_red = 0

    stats = {
        'signals_checked': 0,
        'conditions_valid': 0,
        'orders_placed': 0,
        'orders_filled': 0,
        'orders_cancelled': 0,
        'orders_expired': 0,
        'orders_persisted': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
        'cancel_reasons': {'1h': 0, 'squeeze': 0, 'red2': 0},
        'order_ages': [],     # bars from placement to fill
        'reissue_counts': [], # how many re-issues per regime (B only)
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
    }

    empty_df = pd.DataFrame()
    start_bar = 201
    current_regime_reissues = 0  # track re-issues in current regime (B only)

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]

        # --- Hard capital floor ---
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                account_blown_ts = timestamp
                if position is not None:
                    bar = df_15m.iloc[i]
                    ep = bar['close']
                    ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                    xc = ep * position['size'] * COMMISSION_PCT
                    pnl = (ep - position['entry_price']) * position['size'] - ec - xc
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                    position = None
                pending_order = None
            equity_values.append(max(capital, 0))
            continue

        # --- Equity tracking ---
        equity = capital
        if position is not None:
            bar = df_15m.iloc[i]
            equity += (bar['close'] - position['entry_price']) * position['size']
        equity_values.append(max(equity, 0))

        # ========================================
        # PHASE 1: EXIT MANAGEMENT (1-minute)
        # ========================================
        if position is not None:
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                bar = df_15m.iloc[i]
                # Worst case: stop first
                if bar['low'] <= position['stop_loss']:
                    exit_price = position['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                    xc = exit_price * position['size'] * COMMISSION_PCT
                    pnl = (exit_price - position['entry_price']) * position['size'] - ec - xc
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                    stats['exits_stop'] += 1
                    position = None
                elif bar['high'] >= position['target']:
                    exit_price = position['target']
                    ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                    xc = exit_price * position['size'] * COMMISSION_PCT
                    pnl = (exit_price - position['entry_price']) * position['size'] - ec - xc
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': exit_price,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                    })
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    position = None
            else:
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values

                exited = False
                for k in range(len(cl)):
                    if ct[k] <= position['last_processed_ts']:
                        continue

                    if cl[k] <= position['stop_loss']:
                        exit_price = position['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                        xc = exit_price * position['size'] * COMMISSION_PCT
                        pnl = (exit_price - position['entry_price']) * position['size'] - ec - xc
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': exit_price,
                            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                        })
                        diag['positions_closed'] += 1
                        stats['exits_stop'] += 1
                        position = None
                        exited = True
                        break

                    if ch[k] >= position['target']:
                        exit_price = position['target']
                        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                        xc = exit_price * position['size'] * COMMISSION_PCT
                        pnl = (exit_price - position['entry_price']) * position['size'] - ec - xc
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': exit_price,
                            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                        })
                        diag['positions_closed'] += 1
                        stats['exits_target'] += 1
                        position = None
                        exited = True
                        break

                if not exited and position is not None and len(ct) > 0:
                    position['last_processed_ts'] = ct[-1]

        # ========================================
        # PHASE 2: ORDER MANAGEMENT & ENTRIES
        # ========================================
        if capital <= MIN_CAPITAL or position is not None:
            continue

        stats['signals_checked'] += 1

        prev_bar = df_15m.iloc[i - 1]
        prev_timestamp = df_15m.index[i - 1]

        ema = prev_bar['ema_21']
        atr = prev_bar['atr_20']

        if pd.isna(ema) or pd.isna(atr) or atr <= 0:
            if pending_order is not None:
                stats['orders_cancelled'] += 1
            pending_order = None
            consecutive_red = 0
            continue

        # --- Check conditions ---
        # 1H momentum > 0
        bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
        trend_valid = (len(bars_1h_avail) >= 20
                       and not pd.isna(bars_1h_avail.iloc[-1]['ttm_momentum'])
                       and bars_1h_avail.iloc[-1]['ttm_momentum'] > 0)

        # 15m squeeze in last 4 bars
        squeeze_valid = False
        for lb in range(0, SQUEEZE_LOOKBACK):
            idx = i - 1 - lb
            if idx >= 0:
                bar_lb = df_15m.iloc[idx]
                if not pd.isna(bar_lb['squeeze_on']) and bar_lb['squeeze_on']:
                    squeeze_valid = True
                    break

        # 15m histogram color
        mom_curr = prev_bar['ttm_momentum']
        mom_prev = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
        color = get_histogram_color(mom_curr, mom_prev)

        if color == 'red':
            consecutive_red += 1
        else:
            consecutive_red = 0

        color_valid = color != 'red'

        # Full conditions for placing a new order
        all_conditions = trend_valid and squeeze_valid and color_valid

        if all_conditions:
            stats['conditions_valid'] += 1

        # Calculate levels from current (bar i-1) indicators
        limit_price = ema + ENTRY_ATR_MULT * atr
        stop_level = ema + STOP_ATR_MULT * atr
        target_level = ema + TARGET_ATR_MULT * atr

        risk_dist = limit_price - stop_level
        reward_dist = target_level - limit_price

        levels_valid = risk_dist > 0 and reward_dist > 0

        # ========================================
        # MODE-SPECIFIC ORDER MANAGEMENT
        # ========================================

        if mode == 'persistent':
            # A: True Persistent — levels frozen at first signal
            if pending_order is not None:
                # Check hard invalidation
                cancel = False
                if not trend_valid:
                    cancel = True
                    stats['cancel_reasons']['1h'] += 1
                elif not squeeze_valid:
                    cancel = True
                    stats['cancel_reasons']['squeeze'] += 1
                elif consecutive_red >= RED_BAR_TOLERANCE:
                    cancel = True
                    stats['cancel_reasons']['red2'] += 1

                if cancel:
                    stats['orders_cancelled'] += 1
                    pending_order = None
                # else: keep order with ORIGINAL levels (no update)

            elif all_conditions and levels_valid:
                # First signal in this regime
                pending_order = {
                    'entry_level': limit_price,
                    'stop_level': stop_level,
                    'target_level': target_level,
                    'placed_bar': i,
                }
                stats['orders_placed'] += 1

        elif mode == 'reissue':
            # B: Re-Issue Per Bar — fresh order each valid bar, expires after 1 bar
            # Previous order expired at end of last bar (handled below after fill check)
            if all_conditions and levels_valid:
                pending_order = {
                    'entry_level': limit_price,
                    'stop_level': stop_level,
                    'target_level': target_level,
                    'placed_bar': i,
                }
                stats['orders_placed'] += 1
                current_regime_reissues += 1
            else:
                if current_regime_reissues > 0:
                    stats['reissue_counts'].append(current_regime_reissues)
                    current_regime_reissues = 0
                pending_order = None

        elif mode == 'dynamic':
            # C: Dynamic Update — levels update each bar, order persists
            if all_conditions and levels_valid:
                was_new = pending_order is None
                pending_order = {
                    'entry_level': limit_price,
                    'stop_level': stop_level,
                    'target_level': target_level,
                    'placed_bar': pending_order['placed_bar'] if not was_new else i,
                }
                if was_new:
                    stats['orders_placed'] += 1
            else:
                if pending_order is not None:
                    cancel_reason = 'squeeze' if not squeeze_valid else '1h' if not trend_valid else 'red2'
                    stats['cancel_reasons'][cancel_reason] += 1
                    stats['orders_cancelled'] += 1
                pending_order = None

        # ========================================
        # FILL CHECK
        # ========================================
        if pending_order is None:
            continue

        order = pending_order

        # Pre-filter: 15m bar low must reach limit
        curr_bar = df_15m.iloc[i]
        if curr_bar['low'] > order['entry_level']:
            if mode == 'reissue':
                stats['orders_expired'] += 1
                pending_order = None
            else:
                stats['orders_persisted'] += 1
            continue

        candles = grouped_1m.get(timestamp, empty_df)
        if candles.empty:
            if mode == 'reissue':
                stats['orders_expired'] += 1
                pending_order = None
            else:
                stats['orders_persisted'] += 1
            continue

        cl = candles['low'].values
        ch = candles['high'].values
        ct = candles.index.values

        filled = False
        for j in range(len(cl)):
            if cl[j] <= order['entry_level']:
                fill_price = order['entry_level']  # limit: exact fill
                stop_lev = order['stop_level']
                target_lev = order['target_level']

                risk_d = fill_price - stop_lev
                if risk_d <= 0:
                    break

                # Position sizing
                risk_amount = capital * RISK_PER_TRADE
                if risk_amount <= 0:
                    break
                size = risk_amount / risk_d
                if size <= 0:
                    break
                margin = size * fill_price / LEVERAGE
                if margin > capital:
                    break

                stats['orders_filled'] += 1
                stats['order_ages'].append(i - order['placed_bar'])
                diag['positions_opened'] += 1

                # Same-candle stop check
                same_bar_exit = False
                if cl[j] <= stop_lev:
                    ep = stop_lev * (1 - STOP_SLIPPAGE_PCT)
                    ec = fill_price * size * COMMISSION_PCT
                    xc = ep * size * COMMISSION_PCT
                    pnl = (ep - fill_price) * size - ec - xc
                    capital += pnl
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP_SAME',
                        'bars_held': 0,
                        'entry_time': pd.Timestamp(ct[j]),
                        'exit_time': pd.Timestamp(ct[j]),
                        'entry_price': fill_price, 'exit_price': ep,
                        'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                    })
                    stats['same_bar_stops'] += 1
                    diag['positions_closed'] += 1
                    same_bar_exit = True

                if not same_bar_exit:
                    for k in range(j + 1, len(cl)):
                        if cl[k] <= stop_lev:
                            ep = stop_lev * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                            })
                            stats['same_bar_stops'] += 1
                            diag['positions_closed'] += 1
                            same_bar_exit = True
                            break

                        if ch[k] >= target_lev:
                            ep = target_lev
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                            })
                            stats['same_bar_targets'] += 1
                            diag['positions_closed'] += 1
                            same_bar_exit = True
                            break

                if not same_bar_exit:
                    last_ts = ct[-1] if len(ct) > 0 else ct[j]
                    position = {
                        'entry_time': pd.Timestamp(ct[j]),
                        'entry_bar': i,
                        'entry_price': fill_price,
                        'stop_loss': stop_lev,
                        'target': target_lev,
                        'size': size,
                        'last_processed_ts': last_ts,
                    }

                pending_order = None
                if mode == 'reissue':
                    current_regime_reissues = 0
                filled = True
                break

        if not filled:
            if mode == 'reissue':
                stats['orders_expired'] += 1
                pending_order = None
            elif pending_order is not None:
                stats['orders_persisted'] += 1

        # Progress
        if i % 10000 == 0:
            progress = (i - start_bar) / (len(df_15m) - start_bar) * 100
            print(f"      {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}", flush=True)

    # Close remaining position
    if position is not None:
        bar = df_15m.iloc[-1]
        ep = bar['close']
        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
        xc = ep * position['size'] * COMMISSION_PCT
        pnl = (ep - position['entry_price']) * position['size'] - ec - xc
        capital += pnl
        rd = position['entry_price'] - position['stop_loss']
        trades.append({
            'pnl': pnl, 'exit_reason': 'END',
            'bars_held': len(df_15m) - 1 - position['entry_bar'],
            'entry_time': position['entry_time'], 'exit_time': df_15m.index[-1],
            'entry_price': position['entry_price'], 'exit_price': ep,
            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
        })
        diag['positions_closed'] += 1

    # Final reissue count
    if mode == 'reissue' and current_regime_reissues > 0:
        stats['reissue_counts'].append(current_regime_reissues)

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
        med_r = trades_df['r_multiple'].median()
    else:
        wr = pf = avg_r = med_r = 0

    fill_rate = stats['orders_filled'] / stats['orders_placed'] * 100 if stats['orders_placed'] > 0 else 0

    return {
        'mode': mode,
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_r': avg_r,
        'med_r': med_r,
        'fill_rate': fill_rate,
        'stats': stats,
        'diag': diag,
        'account_blown_ts': account_blown_ts,
        'trades_df': trades_df,
    }


# =====================================================
# RUN ALL THREE MODES
# =====================================================

MODES = [
    ('persistent', 'A: True Persistent'),
    ('reissue', 'B: Re-Issue Per Bar'),
    ('dynamic', 'C: Dynamic Update (ref)'),
]

all_results = {}
t_total = time.time()

for mode_key, mode_label in MODES:
    print("=" * 90)
    print(f"{mode_label}")
    print("=" * 90)
    t_run = time.time()
    r = run_backtest(mode=mode_key)
    elapsed = time.time() - t_run
    r['label'] = mode_label
    r['runtime'] = elapsed
    all_results[mode_key] = r

    blown_str = f" blown:{str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
    print(f"  {r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
          f"avgR {r['avg_r']:+.2f}, ${r['capital']:,.0f}{blown_str} ({elapsed:.0f}s)")
    print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS
# =====================================================

print()
print("=" * 90)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 90)
print(f"GOLD only | Post-2020 | $10K start | 2% risk | 0.1% comm+slip")
print(f"Runtime: {t_total_elapsed:.0f}s")
print()

print(f"  {'Mode':<28s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} {'MedR':>7s} "
      f"{'MaxDD':>8s} {'Fill%':>6s} {'Final':>12s}")
print("  " + "-" * 100)

for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    blown = "*" if r['account_blown_ts'] else ""
    print(f"  {mode_label:<28s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R {r['med_r']:>+6.2f}R "
          f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
          f"${r['capital']:>11,.0f}{blown}")
print()

# Delta analysis
ra = all_results['persistent']
rb = all_results['reissue']
rc = all_results['dynamic']

print("  Delta B vs A (re-issue vs persistent):")
print(f"    Trades:  {rb['trades'] - ra['trades']:+d}")
print(f"    WR:      {rb['win_rate'] - ra['win_rate']:+.1f}%")
print(f"    PF:      {rb['profit_factor'] - ra['profit_factor']:+.2f}")
print(f"    AvgR:    {rb['avg_r'] - ra['avg_r']:+.2f}R")
print(f"    Final:   ${rb['capital'] - ra['capital']:+,.0f}")
print()

print("  Delta C vs A (dynamic vs persistent):")
print(f"    Trades:  {rc['trades'] - ra['trades']:+d}")
print(f"    WR:      {rc['win_rate'] - ra['win_rate']:+.1f}%")
print(f"    PF:      {rc['profit_factor'] - ra['profit_factor']:+.2f}")
print(f"    AvgR:    {rc['avg_r'] - ra['avg_r']:+.2f}R")
print(f"    Final:   ${rc['capital'] - ra['capital']:+,.0f}")
print()

# Execution funnel
print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    s = r['stats']
    print(f"  {mode_label}:")
    print(f"    Bars checked:     {s['signals_checked']:>8d}")
    print(f"    Conditions valid: {s['conditions_valid']:>8d}")
    print(f"    Orders placed:    {s['orders_placed']:>8d}")
    print(f"    Orders filled:    {s['orders_filled']:>8d}")
    print(f"    Orders persisted: {s['orders_persisted']:>8d}")
    print(f"    Orders cancelled: {s['orders_cancelled']:>8d}")
    if s['orders_expired'] > 0:
        print(f"    Orders expired:   {s['orders_expired']:>8d}")
    print(f"    Fill rate:        {r['fill_rate']:>7.1f}%")
    print()
    if s['cancel_reasons']['1h'] or s['cancel_reasons']['squeeze'] or s['cancel_reasons']['red2']:
        print(f"    Cancel reasons: 1H={s['cancel_reasons']['1h']}, "
              f"squeeze={s['cancel_reasons']['squeeze']}, "
              f"2+red={s['cancel_reasons']['red2']}")
    print()
    print(f"    Exits — stop:     {s['exits_stop']:>8d}")
    print(f"    Exits — target:   {s['exits_target']:>8d}")
    print(f"    Same-bar stop:    {s['same_bar_stops']:>8d}")
    print(f"    Same-bar target:  {s['same_bar_targets']:>8d}")
    sbp = s['same_bar_stops'] / s['orders_filled'] * 100 if s['orders_filled'] > 0 else 0
    print(f"    Same-bar SL %:    {sbp:>7.1f}%")
    print()

    # Order age
    if s['order_ages']:
        ages = np.array(s['order_ages'])
        print(f"    Order age at fill: mean={np.mean(ages):.1f} bars, "
              f"median={np.median(ages):.0f}, max={np.max(ages)} bars")

    # Re-issue regime stats (B only)
    if s['reissue_counts']:
        rc_arr = np.array(s['reissue_counts'])
        print(f"    Re-issues per regime: mean={np.mean(rc_arr):.1f}, "
              f"median={np.median(rc_arr):.0f}, max={np.max(rc_arr)}")
    print()

# R-multiple distribution
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION")
print("=" * 90)
print()

bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
        (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 5.0), (5.0, 20.0)]

for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    tdf = r['trades_df']
    if len(tdf) > 0 and 'r_multiple' in tdf.columns:
        print(f"  {mode_label} | {len(tdf)} trades")
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100
            bar = '#' * int(pct / 2)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# Yearly performance
print("=" * 90)
print("YEARLY PERFORMANCE (PnL)")
print("=" * 90)
print()

for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    tdf = r['trades_df']
    if len(tdf) > 0:
        tdf_copy = tdf.copy()
        tdf_copy['year'] = pd.to_datetime(tdf_copy['entry_time']).dt.year
        yearly = tdf_copy.groupby('year').agg(
            trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_r=('r_multiple', 'mean'),
            wins=('pnl', lambda x: (x > 0).sum()),
        )
        yearly['wr'] = yearly['wins'] / yearly['trades'] * 100

        print(f"  {mode_label}:")
        print(f"    {'Year':>6s} {'Trades':>7s} {'WR':>7s} {'PnL':>12s} {'AvgR':>7s}")
        print(f"    " + "-" * 45)
        for year, row in yearly.iterrows():
            print(f"    {year:>6d} {row['trades']:>7.0f} {row['wr']:>6.1f}% "
                  f"${row['total_pnl']:>11,.0f} {row['avg_r']:>+6.2f}R")
        print()

# Validation
print("=" * 90)
print("VALIDATION")
print("=" * 90)
print()
for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    d = r['diag']
    ok = "OK" if d['positions_opened'] == d['positions_closed'] else "MISMATCH"
    print(f"  {mode_label}: open==close: {ok} ({d['positions_opened']}/{d['positions_closed']})")
print()

# Overall comparison
print("=" * 90)
print("KEY INSIGHT")
print("=" * 90)
print()
print("  If C >> A, the PF 2.51 result was inflated by dynamic stop-shifting.")
print("  If B > A, re-issuance captures legitimate second-chance setups.")
print("  If B ≈ A, the edge is in the first setup; re-issuance adds noise.")
print()
print("  True baseline = A (persistent). Any strategy must beat A to be valid.")
print()

# Save results
output_file = 'results/1m_gold_reissue_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("GOLD ORDER HANDLING A/B/C TEST — 1-MINUTE EXECUTION\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"GOLD only | Post-2020 | $10K start | 2% risk\n")
    f.write(f"Frictions: {COMMISSION_PCT*100:.1f}% comm + {STOP_SLIPPAGE_PCT*100:.1f}% slip\n\n")

    f.write(f"{'Mode':<28s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
            f"{'MaxDD':>8s} {'Fill%':>6s} {'Final':>12s}\n")
    f.write("-" * 90 + "\n")
    for mode_key, mode_label in MODES:
        r = all_results[mode_key]
        f.write(f"{mode_label:<28s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
                f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
                f"${r['capital']:>11,.0f}\n")
    f.write("\n" + "=" * 90 + "\n")

print(f"Results saved: {output_file}")

# Save trade logs
for mode_key, mode_label in MODES:
    r = all_results[mode_key]
    if len(r['trades_df']) > 0:
        fname = f'results/1m_gold_reissue_{mode_key}_trades.csv'
        r['trades_df'].to_csv(fname, index=False)
        print(f"Trade log ({mode_key}): {fname}")

print()
print("=" * 90)
