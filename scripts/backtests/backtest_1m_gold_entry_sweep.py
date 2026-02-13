"""
1-Minute Execution: GOLD Entry Level Sweep + Signal Diagnostics

Observation: 2022-2025 shows 100% WR but only 18 trades in 4 years.
Question: Can we tune entry depth to get more fills while preserving the edge?

Sweep: Entry at EMA - X ATR for X in [0.0, 0.25, 0.50, 0.75, 1.00]
Fixed: Stop at EMA - 2.0 ATR, Target at EMA + 2.0 ATR
       1H > 0, 15m squeeze (4 bars), color != red
       Re-issue per bar, GOLD only, post-2020, $10K, 2% risk
       Frictions: 0.1% commission + 0.1% stop slippage

Also tests a "relaxed" filter variant: 1H > 0 + color != red (no squeeze)
to see if squeeze requirement is the bottleneck.

R:R varies by entry depth:
  Entry EMA-0.00: risk=2.0 ATR, reward=2.0 ATR, R:R=1:1
  Entry EMA-0.25: risk=1.75, reward=2.25, R:R=1:1.29
  Entry EMA-0.50: risk=1.50, reward=2.50, R:R=1:1.67
  Entry EMA-0.75: risk=1.25, reward=2.75, R:R=1:2.20
  Entry EMA-1.00: risk=1.00, reward=3.00, R:R=1:3.00
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

COMMISSION_PCT = 0.001
STOP_SLIPPAGE_PCT = 0.001

SQUEEZE_LOOKBACK = 4

STOP_ATR = -2.0     # fixed: EMA - 2.0 ATR
TARGET_ATR = 2.0    # fixed: EMA + 2.0 ATR

# Sweep these entry depths
ENTRY_DEPTHS = [0.00, -0.25, -0.50, -0.75, -1.00]

# Filter variants
FILTER_MODES = ['full', 'no_squeeze']


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
print("1-MINUTE EXECUTION — GOLD ENTRY LEVEL SWEEP + DIAGNOSTICS")
print("=" * 90)
print(f"Stop: EMA - 2.0 ATR | Target: EMA + 2.0 ATR (fixed)")
print(f"Entry depths: {', '.join(f'{d:+.2f}' for d in ENTRY_DEPTHS)} ATR")
print(f"Filters: full (1H>0 + squeeze + color) vs relaxed (1H>0 + color)")
print(f"Frictions: {COMMISSION_PCT*100:.1f}% comm + {STOP_SLIPPAGE_PCT*100:.1f}% slip")
print(f"Period: {START_DATE} to {END_DATE}")
print()

loader = DatabentoMicroFuturesLoader()
t_start = time.time()

df_1m_raw = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE)
print(f"  {len(df_1m_raw)} 1m bars", end=" -> ", flush=True)

df_15m = resample_ohlcv(df_1m_raw, '15min')
df_1h = resample_ohlcv(df_1m_raw, '1h')
print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars")

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

period_labels = df_1m_raw.index.floor('15min')
grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}
del df_1m_raw

t_load = time.time() - t_start
print(f"Data loaded in {t_load:.0f}s\n")


# =====================================================
# SIGNAL DIAGNOSTICS (no execution — just counting)
# =====================================================

print("=" * 90)
print("PHASE 1: SIGNAL DIAGNOSTICS (how often are conditions met?)")
print("=" * 90)
print()

start_bar = 201

# Count per year: conditions met, price within entry range
diag_years = {}

for i in range(start_bar, len(df_15m)):
    timestamp = df_15m.index[i]
    year = timestamp.year
    if year not in diag_years:
        diag_years[year] = {
            'bars': 0,
            'trend_valid': 0,
            'squeeze_valid': 0,
            'color_valid': 0,
            'full_signal': 0,
            'relaxed_signal': 0,
            'price_at_ema': 0,         # low touches EMA
            'price_within_025': 0,     # low reaches EMA - 0.25 ATR
            'price_within_050': 0,
            'price_within_075': 0,
            'price_within_100': 0,
            'trading_days': set(),
        }

    d = diag_years[year]
    d['bars'] += 1
    d['trading_days'].add(timestamp.date())

    prev_bar = df_15m.iloc[i - 1]
    prev_ts = df_15m.index[i - 1]

    ema = prev_bar['ema_21']
    atr = prev_bar['atr_20']
    if pd.isna(ema) or pd.isna(atr) or atr <= 0:
        continue

    # 1H trend
    bars_1h = df_1h[df_1h.index <= prev_ts]
    trend_ok = (len(bars_1h) >= 20
                and not pd.isna(bars_1h.iloc[-1]['ttm_momentum'])
                and bars_1h.iloc[-1]['ttm_momentum'] > 0)
    if trend_ok:
        d['trend_valid'] += 1

    # Squeeze
    sq_ok = False
    for lb in range(SQUEEZE_LOOKBACK):
        idx = i - 1 - lb
        if idx >= 0 and not pd.isna(df_15m.iloc[idx]['squeeze_on']) and df_15m.iloc[idx]['squeeze_on']:
            sq_ok = True
            break
    if sq_ok:
        d['squeeze_valid'] += 1

    # Color
    mom_c = prev_bar['ttm_momentum']
    mom_p = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
    color = get_histogram_color(mom_c, mom_p)
    color_ok = color != 'red'
    if color_ok:
        d['color_valid'] += 1

    if trend_ok and sq_ok and color_ok:
        d['full_signal'] += 1

    if trend_ok and color_ok:
        d['relaxed_signal'] += 1

    # How far does price pull back during signal bars?
    # Use current bar's low (bar i) vs EMA/ATR from bar i-1
    curr_bar = df_15m.iloc[i]
    if trend_ok and color_ok:
        if curr_bar['low'] <= ema:
            d['price_at_ema'] += 1
        if curr_bar['low'] <= ema - 0.25 * atr:
            d['price_within_025'] += 1
        if curr_bar['low'] <= ema - 0.50 * atr:
            d['price_within_050'] += 1
        if curr_bar['low'] <= ema - 0.75 * atr:
            d['price_within_075'] += 1
        if curr_bar['low'] <= ema - 1.00 * atr:
            d['price_within_100'] += 1

print(f"  {'Year':>6s} {'Days':>5s} {'Bars':>7s} {'1H>0':>7s} {'Sqz':>7s} {'!Red':>7s} "
      f"{'Full':>7s} {'Relax':>7s} | {'@EMA':>6s} {'-0.25':>6s} {'-0.50':>6s} {'-0.75':>6s} {'-1.00':>6s}")
print("  " + "-" * 100)

for year in sorted(diag_years.keys()):
    d = diag_years[year]
    days = len(d['trading_days'])
    print(f"  {year:>6d} {days:>5d} {d['bars']:>7d} {d['trend_valid']:>7d} {d['squeeze_valid']:>7d} "
          f"{d['color_valid']:>7d} {d['full_signal']:>7d} {d['relaxed_signal']:>7d} | "
          f"{d['price_at_ema']:>6d} {d['price_within_025']:>6d} {d['price_within_050']:>6d} "
          f"{d['price_within_075']:>6d} {d['price_within_100']:>6d}")

print()
print("  'Full' = 1H>0 + squeeze(4) + color!=red")
print("  'Relax' = 1H>0 + color!=red (no squeeze)")
print("  Price columns: when Relax signal active, how often does bar low reach entry level")
print()

# Per day rates
print("  Signal frequency (per trading day):")
print(f"  {'Year':>6s} {'Full/day':>9s} {'Relax/day':>10s} | {'@EMA/day':>9s} {'-.25/day':>9s} "
      f"{'-.50/day':>9s} {'-.75/day':>9s} {'-1.0/day':>9s}")
print("  " + "-" * 90)
for year in sorted(diag_years.keys()):
    d = diag_years[year]
    days = max(len(d['trading_days']), 1)
    print(f"  {year:>6d} {d['full_signal']/days:>9.1f} {d['relaxed_signal']/days:>10.1f} | "
          f"{d['price_at_ema']/days:>9.1f} {d['price_within_025']/days:>9.1f} "
          f"{d['price_within_050']/days:>9.1f} {d['price_within_075']/days:>9.1f} "
          f"{d['price_within_100']/days:>9.1f}")
print()


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(entry_depth, require_squeeze=True):
    """
    Re-issue per bar with given entry depth.
    entry_depth: ATR multiplier (e.g., -0.5 means EMA - 0.5 ATR)
    """
    capital = INITIAL_CAPITAL
    position = None
    pending_order = None
    trades = []
    equity_values = []

    account_blown = False
    account_blown_ts = None

    n_signals = 0
    n_placed = 0
    n_filled = 0
    n_expired = 0

    empty_df = pd.DataFrame()

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]

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
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'BLOWN', 'bars_held': 0,
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': 0,
                    })
                    position = None
                pending_order = None
            equity_values.append(max(capital, 0))
            continue

        equity = capital
        if position is not None:
            bar = df_15m.iloc[i]
            equity += (bar['close'] - position['entry_price']) * position['size']
        equity_values.append(max(equity, 0))

        # PHASE 1: EXITS
        if position is not None:
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                bar = df_15m.iloc[i]
                if bar['low'] <= position['stop_loss']:
                    ep = position['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                    xc = ep * position['size'] * COMMISSION_PCT
                    pnl = (ep - position['entry_price']) * position['size'] - ec - xc
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd * position['size'] > 0 else 0,
                    })
                    position = None
                elif bar['high'] >= position['target']:
                    ep = position['target']
                    ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                    xc = ep * position['size'] * COMMISSION_PCT
                    pnl = (ep - position['entry_price']) * position['size'] - ec - xc
                    capital += pnl
                    rd = position['entry_price'] - position['stop_loss']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - position['entry_bar'],
                        'entry_time': position['entry_time'], 'exit_time': timestamp,
                        'entry_price': position['entry_price'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['size']) if rd * position['size'] > 0 else 0,
                    })
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
                        ep = position['stop_loss'] * (1 - STOP_SLIPPAGE_PCT)
                        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                        xc = ep * position['size'] * COMMISSION_PCT
                        pnl = (ep - position['entry_price']) * position['size'] - ec - xc
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['size']) if rd * position['size'] > 0 else 0,
                        })
                        position = None
                        exited = True
                        break

                    if ch[k] >= position['target']:
                        ep = position['target']
                        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
                        xc = ep * position['size'] * COMMISSION_PCT
                        pnl = (ep - position['entry_price']) * position['size'] - ec - xc
                        capital += pnl
                        rd = position['entry_price'] - position['stop_loss']
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - position['entry_bar'],
                            'entry_time': position['entry_time'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['entry_price'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['size']) if rd * position['size'] > 0 else 0,
                        })
                        position = None
                        exited = True
                        break

                if not exited and position is not None and len(ct) > 0:
                    position['last_processed_ts'] = ct[-1]

        # PHASE 2: ENTRIES (re-issue per bar)
        if capital <= MIN_CAPITAL or position is not None:
            continue

        prev_bar = df_15m.iloc[i - 1]
        prev_ts = df_15m.index[i - 1]

        ema = prev_bar['ema_21']
        atr = prev_bar['atr_20']
        if pd.isna(ema) or pd.isna(atr) or atr <= 0:
            continue

        # 1H trend
        bars_1h = df_1h[df_1h.index <= prev_ts]
        if len(bars_1h) < 20:
            continue
        if pd.isna(bars_1h.iloc[-1]['ttm_momentum']) or bars_1h.iloc[-1]['ttm_momentum'] <= 0:
            continue

        # Squeeze (optional)
        if require_squeeze:
            sq_ok = False
            for lb in range(SQUEEZE_LOOKBACK):
                idx = i - 1 - lb
                if idx >= 0 and not pd.isna(df_15m.iloc[idx]['squeeze_on']) and df_15m.iloc[idx]['squeeze_on']:
                    sq_ok = True
                    break
            if not sq_ok:
                continue

        # Color
        mom_c = prev_bar['ttm_momentum']
        mom_p = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
        color = get_histogram_color(mom_c, mom_p)
        if color == 'red':
            continue

        n_signals += 1

        # Calculate levels
        limit_price = ema + entry_depth * atr
        stop_level = ema + STOP_ATR * atr
        target_level = ema + TARGET_ATR * atr

        risk_dist = limit_price - stop_level
        reward_dist = target_level - limit_price

        if risk_dist <= 0 or reward_dist <= 0:
            continue

        n_placed += 1

        # Re-issue: fresh order, valid this bar only
        pending_order = {
            'entry_level': limit_price,
            'stop_level': stop_level,
            'target_level': target_level,
        }

        # Fill check
        curr_bar = df_15m.iloc[i]
        if curr_bar['low'] > pending_order['entry_level']:
            n_expired += 1
            pending_order = None
            continue

        candles = grouped_1m.get(timestamp, empty_df)
        if candles.empty:
            n_expired += 1
            pending_order = None
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

                risk_amount = capital * RISK_PER_TRADE
                if risk_amount <= 0:
                    break
                size = risk_amount / risk_d
                if size <= 0:
                    break
                margin = size * fill_price / LEVERAGE
                if margin > capital:
                    break

                n_filled += 1

                # Same-candle checks
                same_exit = False
                if cl[j] <= stop_lev:
                    ep = stop_lev * (1 - STOP_SLIPPAGE_PCT)
                    ec = fill_price * size * COMMISSION_PCT
                    xc = ep * size * COMMISSION_PCT
                    pnl = (ep - fill_price) * size - ec - xc
                    capital += pnl
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP_SAME', 'bars_held': 0,
                        'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[j]),
                        'entry_price': fill_price, 'exit_price': ep,
                        'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                    })
                    same_exit = True

                if not same_exit:
                    for k in range(j + 1, len(cl)):
                        if cl[k] <= stop_lev:
                            ep = stop_lev * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'STOP_SAME', 'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                            })
                            same_exit = True
                            break

                        if ch[k] >= target_lev:
                            ep = target_lev
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'TARGET_SAME', 'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_d * size) if risk_d * size > 0 else 0,
                            })
                            same_exit = True
                            break

                if not same_exit:
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
                filled = True
                break

        if not filled:
            n_expired += 1
            pending_order = None

    # Close remaining
    if position is not None:
        bar = df_15m.iloc[-1]
        ep = bar['close']
        ec = position['entry_price'] * position['size'] * COMMISSION_PCT
        xc = ep * position['size'] * COMMISSION_PCT
        pnl = (ep - position['entry_price']) * position['size'] - ec - xc
        capital += pnl
        trades.append({
            'pnl': pnl, 'exit_reason': 'END', 'bars_held': 0,
            'entry_time': position['entry_time'], 'exit_time': df_15m.index[-1],
            'entry_price': position['entry_price'], 'exit_price': ep,
            'r_multiple': 0,
        })

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

    fill_rate = n_filled / n_placed * 100 if n_placed > 0 else 0

    # Per-year breakdown
    yearly = {}
    if len(trades_df) > 0:
        tdf = trades_df.copy()
        tdf['year'] = pd.to_datetime(tdf['entry_time']).dt.year
        for year, grp in tdf.groupby('year'):
            w = (grp['pnl'] > 0).sum()
            yearly[year] = {
                'trades': len(grp),
                'wr': w / len(grp) * 100,
                'pnl': grp['pnl'].sum(),
                'avg_r': grp['r_multiple'].mean(),
            }

    rr = reward_dist / risk_dist if risk_dist > 0 else 0
    # R:R for this entry depth
    risk_atr = abs(entry_depth - STOP_ATR)
    reward_atr = TARGET_ATR - entry_depth
    rr_ratio = reward_atr / risk_atr if risk_atr > 0 else 0

    return {
        'entry_depth': entry_depth,
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'avg_r': avg_r,
        'fill_rate': fill_rate,
        'n_signals': n_signals,
        'n_placed': n_placed,
        'n_filled': n_filled,
        'n_expired': n_expired,
        'account_blown_ts': account_blown_ts,
        'yearly': yearly,
        'trades_df': trades_df,
        'rr_ratio': rr_ratio,
        'risk_atr': risk_atr,
        'require_squeeze': require_squeeze,
    }


# =====================================================
# RUN SWEEP
# =====================================================

print("=" * 90)
print("PHASE 2: ENTRY LEVEL SWEEP (with execution)")
print("=" * 90)
print()

all_results = {}
t_total = time.time()

for filter_mode in FILTER_MODES:
    require_sq = filter_mode == 'full'
    filter_label = "Full filter" if require_sq else "No squeeze"
    print(f"--- {filter_label} ---")
    print()

    for depth in ENTRY_DEPTHS:
        label = f"EMA{depth:+.2f}"
        print(f"  {label}...", end=" ", flush=True)
        t_run = time.time()

        r = run_backtest(entry_depth=depth, require_squeeze=require_sq)
        elapsed = time.time() - t_run

        r['label'] = label
        r['filter'] = filter_mode
        key = (filter_mode, depth)
        all_results[key] = r

        blown = f" blown:{str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
        print(f"{r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
              f"avgR {r['avg_r']:+.2f}, ${r['capital']:,.0f}, "
              f"fill {r['fill_rate']:.1f}%{blown} ({elapsed:.0f}s)")

    print()

t_elapsed = time.time() - t_total

# =====================================================
# RESULTS
# =====================================================

print()
print("=" * 90)
print("ENTRY LEVEL SWEEP — RESULTS")
print("=" * 90)
print(f"Runtime: {t_elapsed:.0f}s | Frictions: 0.1% comm + 0.1% slip")
print()

for filter_mode in FILTER_MODES:
    filter_label = "Full (1H>0 + squeeze + color)" if filter_mode == 'full' else "Relaxed (1H>0 + color, no squeeze)"
    print(f"--- {filter_label} ---")
    print()
    print(f"  {'Entry':<12s} {'R:R':>5s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
          f"{'MaxDD':>8s} {'Fill%':>7s} {'Signals':>8s} {'Final':>12s}")
    print("  " + "-" * 100)

    for depth in ENTRY_DEPTHS:
        r = all_results[(filter_mode, depth)]
        rr_str = f"1:{r['rr_ratio']:.1f}"
        blown = "*" if r['account_blown_ts'] else ""
        print(f"  EMA{depth:+.2f} ATR {rr_str:>5s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
              f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>6.1f}% "
              f"{r['n_signals']:>8d} ${r['capital']:>11,.0f}{blown}")
    print()

# Per-year breakdown for each entry level
print("=" * 90)
print("YEARLY BREAKDOWN (trades / WR / PnL)")
print("=" * 90)
print()

years = sorted(set(y for r in all_results.values() for y in r['yearly'].keys()))

for filter_mode in FILTER_MODES:
    filter_label = "Full" if filter_mode == 'full' else "Relaxed"
    print(f"--- {filter_label} filter ---")
    print()

    # Header
    hdr_parts = [f"  {'Entry':<12s}"]
    for y in years:
        hdr_parts.append(f"{'  ' + str(y):>18s}")
    hdr_parts.append(f"{'  TOTAL':>12s}")
    print(''.join(hdr_parts))
    print("  " + "-" * (12 + 18 * len(years) + 12))

    for depth in ENTRY_DEPTHS:
        r = all_results[(filter_mode, depth)]
        parts = [f"  EMA{depth:+.2f} ATR"]
        total_trades = 0
        total_pnl = 0
        for y in years:
            if y in r['yearly']:
                yd = r['yearly'][y]
                total_trades += yd['trades']
                total_pnl += yd['pnl']
                parts.append(f"  {yd['trades']:>3d} {yd['wr']:>4.0f}% ${yd['pnl']:>7,.0f}")
            else:
                parts.append(f"  {'---':>3s} {'':>5s} {'':>8s}")
        parts.append(f"  {total_trades:>3d} ${total_pnl:>7,.0f}")
        print(''.join(parts))
    print()

# Trades per day analysis
print("=" * 90)
print("TRADES PER TRADING DAY (target: 1-2)")
print("=" * 90)
print()

for filter_mode in FILTER_MODES:
    filter_label = "Full" if filter_mode == 'full' else "Relaxed"
    print(f"  {'Entry':<12s} {'Filter':<8s}", end="")
    for y in years:
        print(f" {y:>8d}", end="")
    print()
    print("  " + "-" * (20 + 9 * len(years)))

    for depth in ENTRY_DEPTHS:
        r = all_results[(filter_mode, depth)]
        print(f"  EMA{depth:+.2f} ATR {filter_label:<8s}", end="")
        for y in years:
            if y in r['yearly']:
                days = len(diag_years[y]['trading_days'])
                tpd = r['yearly'][y]['trades'] / days if days > 0 else 0
                print(f" {tpd:>8.2f}", end="")
            else:
                print(f" {'0.00':>8s}", end="")
        print()
    print()

# Best config identification
print("=" * 90)
print("BEST CONFIGURATIONS (by PF, min 20 trades)")
print("=" * 90)
print()

ranked = sorted(
    [(k, r) for k, r in all_results.items() if r['trades'] >= 20],
    key=lambda x: x[1]['profit_factor'],
    reverse=True
)

print(f"  {'Rank':>4s} {'Entry':<12s} {'Filter':<8s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} "
      f"{'AvgR':>7s} {'MaxDD':>8s} {'Final':>12s}")
print("  " + "-" * 85)

for rank, (key, r) in enumerate(ranked[:10], 1):
    fl = "Full" if key[0] == 'full' else "Relax"
    print(f"  {rank:>4d} EMA{key[1]:+.2f} ATR {fl:<8s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
          f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
          f"{r['max_drawdown']:>7.1f}% ${r['capital']:>11,.0f}")
print()

# R-multiple distribution for top configs
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (top 3 by PF)")
print("=" * 90)
print()

bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
        (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 5.0), (5.0, 20.0)]

for rank, (key, r) in enumerate(ranked[:3], 1):
    tdf = r['trades_df']
    fl = "Full" if key[0] == 'full' else "Relax"
    if len(tdf) > 0:
        print(f"  #{rank} EMA{key[1]:+.2f} {fl} | {len(tdf)} trades, PF {r['profit_factor']:.2f}")
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100
            bar = '#' * int(pct / 2)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# Save
output_file = 'results/1m_gold_entry_sweep_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("GOLD ENTRY LEVEL SWEEP — 1-MINUTE EXECUTION\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Stop: EMA-2.0 | Target: EMA+2.0 | Risk: 2% | Frictions: 0.1%+0.1%\n\n")

    for filter_mode in FILTER_MODES:
        fl = "Full" if filter_mode == 'full' else "Relaxed"
        f.write(f"\n{fl} filter\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Entry':<12s} {'R:R':>5s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
                f"{'MaxDD':>8s} {'Final':>12s}\n")
        f.write("-" * 70 + "\n")
        for depth in ENTRY_DEPTHS:
            r = all_results[(filter_mode, depth)]
            rr_str = f"1:{r['rr_ratio']:.1f}"
            f.write(f"EMA{depth:+.2f} ATR {rr_str:>5s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                    f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
                    f"{r['max_drawdown']:>7.1f}% ${r['capital']:>11,.0f}\n")
        f.write("\n")
    f.write("=" * 90 + "\n")

print(f"\nResults saved: {output_file}")
print()
print("=" * 90)
