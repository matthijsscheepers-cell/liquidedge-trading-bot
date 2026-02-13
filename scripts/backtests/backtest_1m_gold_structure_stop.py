"""
1-Minute Execution Backtest: GOLD Structure Stop vs Fixed ATR Stop

Tests whether a structural swing-low stop improves expectancy vs fixed ATR stop.

Universe: GOLD only
Period: 2020-01-01 to latest

Signal conditions:
- 1H TTM momentum > 0
- 15m squeeze active in last 4 closed bars
- 15m histogram color != red (yellow, light_blue, dark_blue allowed)

Entry: persistent limit at EMA(21) - 1.0 ATR
Target: EMA(21) + 2.0 ATR

Two stop variants:
  A) Structure stop: last 15m swing low, capped at 2.5 ATR from entry
  B) Fixed stop: entry - 1.0 ATR (reference)

No commission/slippage (focus on structural edge).
$10K start, 2% risk per trade, no caps.
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

SQUEEZE_LOOKBACK = 4      # bars to check for recent squeeze
SWING_LOW_LOOKBACK = 20   # bars to scan for swing low
MAX_STOP_ATR = 2.5        # cap structure stop at 2.5 ATR from entry

# Entry/Target ATR offsets
ENTRY_ATR = -1.0           # EMA - 1.0 ATR
TARGET_ATR = 2.0           # EMA + 2.0 ATR

# Fixed stop for comparison
FIXED_STOP_ATR = -1.0      # entry - 1.0 ATR (so EMA - 2.0 ATR)


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def get_histogram_color(mom_curr, mom_prev):
    """
    - light_blue: mom > 0 and rising
    - dark_blue:  mom > 0 and falling
    - yellow:     mom <= 0 but rising
    - red:        mom <= 0 and falling
    """
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


def find_swing_low(df_15m, before_idx, lookback=SWING_LOW_LOOKBACK):
    """
    Find the most recent 15m swing low before bar at before_idx.
    Swing low: low[k] < low[k-1] AND low[k] < low[k+1].
    """
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
print("1-MINUTE EXECUTION — GOLD STRUCTURE STOP vs FIXED ATR STOP")
print("=" * 90)
print(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR")
print(f"Stop A: swing low (capped {MAX_STOP_ATR:.1f} ATR)")
print(f"Stop B: entry - {abs(FIXED_STOP_ATR):.1f} ATR (fixed)")
print(f"Filters: 1H mom > 0, 15m squeeze (4 bars), color != red")
print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}%")
print(f"No commission/slippage | Period: {START_DATE} to {END_DATE}")
print()
print("Loading data...")
print()

loader = DatabentoMicroFuturesLoader()

t_start = time.time()

# Load only GOLD
print("  GOLD...", end=" ", flush=True)
df_1m_raw = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE)
print(f"{len(df_1m_raw)} 1m bars", end=" -> ", flush=True)

df_15m = resample_ohlcv(df_1m_raw, '15min')
df_1h = resample_ohlcv(df_1m_raw, '1h')
print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars", end=" ", flush=True)

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

print("OK")
t_load = time.time() - t_start
print(f"\nData loaded in {t_load:.0f}s")
print(f"15m bars: {len(df_15m)}")
print()


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(use_structure_stop=True):
    """
    Run the backtest with either structure stop or fixed ATR stop.
    """
    capital = INITIAL_CAPITAL
    position = None
    pending_order = None
    trades = []
    equity_values = []

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
        'no_swing_low': 0,
        'stop_distances_atr': [],
        'r_ratios': [],
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
    }

    empty_df = pd.DataFrame()
    start_bar = 201  # warmup for indicators

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
                    pnl = (ep - position['entry_price']) * position['size']
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'ACCOUNT_BLOWN',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                        'stop_dist_atr': position['stop_dist_atr'],
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
                # Fallback: 15m bar
                bar = df_15m.iloc[i]
                if bar['low'] <= position['stop_loss']:
                    ep = position['stop_loss']
                    pnl = (ep - position['entry_price']) * position['size']
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                        'stop_dist_atr': position['stop_dist_atr'],
                    })
                    diag['positions_closed'] += 1
                    stats['exits_stop'] += 1
                    position = None
                elif bar['high'] >= position['target']:
                    ep = position['target']
                    pnl = (ep - position['entry_price']) * position['size']
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                        'stop_dist_atr': position['stop_dist_atr'],
                    })
                    diag['positions_closed'] += 1
                    stats['exits_target'] += 1
                    position = None
            else:
                # 1-minute exit scanning
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values

                exited = False
                for k in range(len(cl)):
                    if ct[k] <= position['last_processed_ts']:
                        continue

                    # Stop first (worst case)
                    if cl[k] <= position['stop_loss']:
                        ep = position['stop_loss']
                        pnl = (ep - position['entry_price']) * position['size']
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                            'stop_dist_atr': position['stop_dist_atr'],
                        })
                        diag['positions_closed'] += 1
                        stats['exits_stop'] += 1
                        position = None
                        exited = True
                        break

                    if ch[k] >= position['target']:
                        ep = position['target']
                        pnl = (ep - position['entry_price']) * position['size']
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
                            'stop_dist_atr': position['stop_dist_atr'],
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

        prev_bar = df_15m.iloc[i - 1]
        prev_timestamp = df_15m.index[i - 1]

        ema = prev_bar['ema_21']
        atr = prev_bar['atr_20']

        if pd.isna(ema) or pd.isna(atr) or atr <= 0:
            pending_order = None
            continue

        # --- 1H momentum filter (> 0) ---
        bars_1h_avail = df_1h[df_1h.index <= prev_timestamp]
        if len(bars_1h_avail) < 20:
            pending_order = None
            continue
        prev_1h = bars_1h_avail.iloc[-1]

        if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
            if pending_order is not None:
                stats['orders_cancelled'] += 1
            pending_order = None
            continue

        # --- 15m squeeze filter (active in last 4 bars) ---
        squeeze_active = False
        for lb in range(0, SQUEEZE_LOOKBACK):
            idx = i - 1 - lb
            if idx >= 0:
                bar_lb = df_15m.iloc[idx]
                if not pd.isna(bar_lb['squeeze_on']) and bar_lb['squeeze_on']:
                    squeeze_active = True
                    break

        if not squeeze_active:
            if pending_order is not None:
                stats['orders_cancelled'] += 1
            pending_order = None
            continue

        # --- 15m histogram color filter (not red) ---
        mom_curr = prev_bar['ttm_momentum']
        mom_prev = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
        color = get_histogram_color(mom_curr, mom_prev)

        if color == 'red':
            if pending_order is not None:
                stats['orders_cancelled'] += 1
            pending_order = None
            continue

        # --- All filters passed ---
        limit_price = ema + ENTRY_ATR * atr   # EMA - 1.0 ATR
        target_level = ema + TARGET_ATR * atr  # EMA + 2.0 ATR

        # Determine stop level
        if use_structure_stop:
            swing_low = find_swing_low(df_15m, i - 1)
            if swing_low is None:
                stats['no_swing_low'] += 1
                # No valid swing low found — skip signal
                if pending_order is not None:
                    stats['orders_cancelled'] += 1
                pending_order = None
                continue

            stop_level = swing_low
            # Cap at MAX_STOP_ATR from entry
            max_stop = limit_price - MAX_STOP_ATR * atr
            if stop_level < max_stop:
                stop_level = max_stop
        else:
            # Fixed stop: entry - 1.0 ATR (so EMA - 2.0 ATR)
            stop_level = limit_price + FIXED_STOP_ATR * atr

        risk_dist = limit_price - stop_level
        reward_dist = target_level - limit_price

        if risk_dist <= 0 or reward_dist <= 0:
            pending_order = None
            continue

        # R:R check (at least 1.0:1 for structure stop which can be wider)
        rr = reward_dist / risk_dist
        if rr < 1.0:
            pending_order = None
            continue

        if pending_order is None:
            stats['orders_placed'] += 1
            stats['signals'] += 1

        stop_dist_atr = risk_dist / atr

        pending_order = {
            'entry_level': limit_price,
            'stop_level': stop_level,
            'target_level': target_level,
            'signal_bar': i - 1,
            'stop_dist_atr': stop_dist_atr,
            'rr': rr,
        }

        # --- Fill check ---
        curr_bar = df_15m.iloc[i]
        if curr_bar['low'] > pending_order['entry_level']:
            stats['orders_persisted'] += 1
            continue

        candles = grouped_1m.get(timestamp, empty_df)
        if candles.empty:
            stats['orders_persisted'] += 1
            continue

        cl = candles['low'].values
        ch = candles['high'].values
        ct = candles.index.values

        filled = False
        for j in range(len(cl)):
            if cl[j] <= pending_order['entry_level']:
                fill_price = pending_order['entry_level']
                stop_lev = pending_order['stop_level']
                target_lev = pending_order['target_level']

                risk_d = fill_price - stop_lev
                if risk_d <= 0:
                    break

                # Position sizing: pure 2% risk
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
                stats['stop_distances_atr'].append(pending_order['stop_dist_atr'])
                stats['r_ratios'].append(pending_order['rr'])
                diag['positions_opened'] += 1

                # Same-candle stop check
                same_bar_exit = False
                if cl[j] <= stop_lev:
                    ep = stop_lev
                    pnl = (ep - fill_price) * size
                    capital += pnl
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP_SAME',
                        'bars_held': 0,
                        'entry_time': pd.Timestamp(ct[j]),
                        'exit_time': pd.Timestamp(ct[j]),
                        'entry_price': fill_price, 'exit_price': ep,
                        'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                        'stop_dist_atr': pending_order['stop_dist_atr'],
                    })
                    stats['same_bar_stops'] += 1
                    diag['positions_closed'] += 1
                    same_bar_exit = True

                # Remaining candles after fill
                if not same_bar_exit:
                    for k in range(j + 1, len(cl)):
                        if cl[k] <= stop_lev:
                            ep = stop_lev
                            pnl = (ep - fill_price) * size
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                                'stop_dist_atr': pending_order['stop_dist_atr'],
                            })
                            stats['same_bar_stops'] += 1
                            diag['positions_closed'] += 1
                            same_bar_exit = True
                            break

                        if ch[k] >= target_lev:
                            ep = target_lev
                            pnl = (ep - fill_price) * size
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                                'stop_dist_atr': pending_order['stop_dist_atr'],
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
                        'stop_dist_atr': pending_order['stop_dist_atr'],
                    }

                pending_order = None
                filled = True
                break

        if not filled and pending_order is not None:
            stats['orders_persisted'] += 1

        # Progress
        if i % 10000 == 0:
            progress = (i - start_bar) / (len(df_15m) - start_bar) * 100
            print(f"      {progress:.0f}%  ${capital:,.0f}  trades:{len(trades)}", flush=True)

    # Close remaining position
    if position is not None:
        bar = df_15m.iloc[-1]
        ep = bar['close']
        pnl = (ep - position['entry_price']) * position['size']
        capital += pnl
        rd = position['entry_price'] - position['stop_loss']
        trades.append({
            'pnl': pnl, 'exit_reason': 'END',
            'bars_held': len(df_15m) - 1 - position['entry_bar'],
            'entry_time': position['entry_time'], 'exit_time': df_15m.index[-1],
            'entry_price': position['entry_price'], 'exit_price': ep,
            'r_multiple': pnl / (rd * position['size']) if rd > 0 and position['size'] > 0 else 0,
            'stop_dist_atr': position['stop_dist_atr'],
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
        med_r = trades_df['r_multiple'].median()
    else:
        wr = pf = avg_r = med_r = 0

    fill_rate = stats['orders_filled'] / stats['signals'] * 100 if stats['signals'] > 0 else 0

    return {
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
# RUN BOTH VARIANTS
# =====================================================

t_total = time.time()

print("=" * 90)
print("VARIANT A: STRUCTURE STOP (swing low, capped 2.5 ATR)")
print("=" * 90)
print()
result_structure = run_backtest(use_structure_stop=True)
print()

print("=" * 90)
print("VARIANT B: FIXED ATR STOP (entry - 1.0 ATR)")
print("=" * 90)
print()
result_fixed = run_backtest(use_structure_stop=False)
print()

t_total_elapsed = time.time() - t_total

# =====================================================
# RESULTS REPORT
# =====================================================

print()
print("=" * 90)
print("RESULTS — GOLD STRUCTURE STOP vs FIXED ATR STOP")
print("=" * 90)
print(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR")
print(f"Filters: 1H mom > 0, 15m squeeze (4 bars), color != red")
print(f"Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}% | No commissions/slippage")
print(f"Total runtime: {t_total_elapsed:.0f}s")
print()

results = [
    ("A: Swing Low (cap 2.5 ATR)", result_structure),
    ("B: Fixed -1.0 ATR", result_fixed),
]

print(f"  {'Variant':<30s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} {'MedR':>7s} "
      f"{'MaxDD':>8s} {'Fill%':>6s} {'Final':>12s} {'Blown':>12s}")
print("  " + "-" * 110)

for label, r in results:
    blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
    print(f"  {label:<30s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R {r['med_r']:>+6.2f}R "
          f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
          f"${r['capital']:>11,.0f} {blown:>12s}")
print()

# Stop distance distribution
print("=" * 90)
print("STOP DISTANCE DISTRIBUTION (in ATR)")
print("=" * 90)
print()

for label, r in results:
    sd = r['stats']['stop_distances_atr']
    if sd:
        arr = np.array(sd)
        print(f"  {label}:")
        print(f"    Mean:   {np.mean(arr):.2f} ATR")
        print(f"    Median: {np.median(arr):.2f} ATR")
        print(f"    Min:    {np.min(arr):.2f} ATR")
        print(f"    Max:    {np.max(arr):.2f} ATR")
        print(f"    Std:    {np.std(arr):.2f} ATR")
        print()
        # Buckets
        buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 5.0)]
        for lo, hi in buckets:
            n = np.sum((arr >= lo) & (arr < hi))
            pct = n / len(arr) * 100
            bar = '#' * int(pct)
            print(f"    {lo:.1f}-{hi:.1f} ATR: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# R:R distribution for structure stop
print("=" * 90)
print("R:R RATIO DISTRIBUTION (structure stop)")
print("=" * 90)
print()

sd_rr = result_structure['stats']['r_ratios']
if sd_rr:
    arr_rr = np.array(sd_rr)
    print(f"  Mean R:R:   {np.mean(arr_rr):.2f}")
    print(f"  Median R:R: {np.median(arr_rr):.2f}")
    print(f"  Min R:R:    {np.min(arr_rr):.2f}")
    print(f"  Max R:R:    {np.max(arr_rr):.2f}")
    print()
    rr_buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 100.0)]
    for lo, hi in rr_buckets:
        n = np.sum((arr_rr >= lo) & (arr_rr < hi))
        pct = n / len(arr_rr) * 100
        bar = '#' * int(pct)
        print(f"    {lo:.1f}-{hi:.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
    print()

# R-multiple distribution
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (trade outcomes)")
print("=" * 90)
print()

bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
        (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 5.0), (5.0, 20.0)]

for label, r in results:
    tdf = r['trades_df']
    if len(tdf) > 0 and 'r_multiple' in tdf.columns:
        print(f"  {label} | {len(tdf)} trades")
        print()
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100
            bar = '#' * int(pct)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# Execution funnel
print("=" * 90)
print("EXECUTION FUNNEL")
print("=" * 90)
print()

for label, r in results:
    s = r['stats']
    print(f"  {label}:")
    print(f"    Signals:         {s['signals']:>8d}")
    print(f"    Orders placed:   {s['orders_placed']:>8d}")
    print(f"    Orders filled:   {s['orders_filled']:>8d}")
    print(f"    Orders persisted:{s['orders_persisted']:>8d}")
    print(f"    Orders cancelled:{s['orders_cancelled']:>8d}")
    print(f"    Fill rate:       {r['fill_rate']:>7.1f}%")
    if s['no_swing_low'] > 0:
        print(f"    No swing low:    {s['no_swing_low']:>8d}")
    print()
    print(f"    Exits — stop:    {s['exits_stop']:>8d}")
    print(f"    Exits — target:  {s['exits_target']:>8d}")
    print(f"    Same-bar stop:   {s['same_bar_stops']:>8d}")
    print(f"    Same-bar target: {s['same_bar_targets']:>8d}")
    same_bar_pct = (s['same_bar_stops'] / s['orders_filled'] * 100) if s['orders_filled'] > 0 else 0
    print(f"    Same-bar SL %:   {same_bar_pct:>7.1f}%")
    print()

# Validation
print("=" * 90)
print("VALIDATION")
print("=" * 90)
print()
for label, r in results:
    d = r['diag']
    ok = "OK" if d['positions_opened'] == d['positions_closed'] else "MISMATCH"
    print(f"  {label}: open==close: {ok} ({d['positions_opened']}/{d['positions_closed']})")
print()

# Comparison table
print("=" * 90)
print("COMPARISON: ALL 1-MINUTE STRATEGIES TESTED (GOLD, post-2020)")
print("=" * 90)
print()
print(f"  {'Strategy':<40s} {'WR':>6s} {'PF':>6s} {'AvgR':>7s} {'Notes'}")
print("  " + "-" * 75)
print(f"  {'Lim-1 S-2 T+2 pos (prev best)':<40s} {'43%':>6s} {'1.01':>6s} {'+0.05R':>7s} {'break-even (G+S)'}")
print(f"  {'4H-filt Lim-1 S-2 T+2 (GOLD)':<40s} {'40%':>6s} {'0.67':>6s} {'-0.09R':>7s} {'worse'}")

for label, r in results:
    blown = "blown" if r['account_blown_ts'] else "survived"
    print(f"  {label:<40s} "
          f"{r['win_rate']:>5.0f}% {r['profit_factor']:>6.2f} "
          f"{r['avg_r']:>+6.2f}R {blown}")
print()

# Save results
output_file = 'results/1m_gold_structure_stop_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("GOLD STRUCTURE STOP vs FIXED ATR STOP — 1-MINUTE EXECUTION\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR\n")
    f.write(f"Filters: 1H > 0, 15m squeeze (4 bars), color != red\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,.0f} | Risk: {RISK_PER_TRADE*100:.0f}% | No comm/slip\n\n")

    f.write(f"{'Variant':<30s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} {'MedR':>7s} "
            f"{'MaxDD':>8s} {'Final':>12s}\n")
    f.write("-" * 95 + "\n")

    for label, r in results:
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        f.write(f"{label:<30s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R {r['med_r']:>+6.2f}R "
                f"{r['max_drawdown']:>7.1f}% "
                f"${r['capital']:>11,.0f}\n")
    f.write("\n" + "=" * 90 + "\n")

print(f"Results saved: {output_file}")

# Save trade logs
for label, r in [("structure", result_structure), ("fixed", result_fixed)]:
    if len(r['trades_df']) > 0:
        fname = f'results/1m_gold_{label}_stop_trades.csv'
        r['trades_df'].to_csv(fname, index=False)
        print(f"Trade log ({label}): {fname}")

print()
print("=" * 90)
