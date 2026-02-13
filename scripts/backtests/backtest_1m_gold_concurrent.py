"""
1-Minute Execution: GOLD — Concurrent Positions + Alternative 1H Filters

Tests the impact of allowing multiple simultaneous positions and
replacing the 1H TTM momentum filter with alternatives.

Configs:
  1. single_ttm    — 1 position,  1H TTM momentum > 0 (baseline)
  2. single_ema    — 1 position,  1H close > EMA(21)
  3. multi3_ttm    — 3 positions, 1H TTM momentum > 0
  4. multi3_ema    — 3 positions, 1H close > EMA(21)
  5. multi5_ema    — 5 positions, 1H close > EMA(21)
  6. multi5_none   — 5 positions, no 1H filter (15m color only)

Strategy (all configs):
  Entry:  Limit at EMA(21)
  Stop:   EMA(21) - 2.0 ATR
  Target: EMA(21) + 2.0 ATR
  15m filter: histogram color != red
  NO squeeze requirement.
  Re-issue per bar. GOLD only. 2020-2026.
  Scale A risk (2% < $1M, 1% < $10M, 0.5% >= $10M).
  Frictions: 0.1% commission + 0.1% stop slippage.
  Circuit breakers: 2 consecutive stops -> 4h cooldown, daily loss >= 3x risk -> halt.
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
INITIAL_CAPITAL = 3000
LEVERAGE = 20
MIN_CAPITAL = 1.0

COMMISSION_PCT = 0.001
STOP_SLIPPAGE_PCT = 0.001

ENTRY_ATR = 0.0       # EMA itself
STOP_ATR = -2.0        # EMA - 2 ATR
TARGET_ATR = 2.0       # EMA + 2 ATR

CB_CONSEC_STOP = 2
CB_COOLDOWN_BARS = 16  # 4 hours
CB_DAILY_LOSS_MULT = 3

# Scale A risk tiers
SCALING_TIERS = [(1_000_000, 0.02), (10_000_000, 0.01), (float('inf'), 0.005)]

CONFIGS = {
    'single_ttm': {
        'label': '1 pos, TTM mom>0',
        'max_positions': 1,
        'filter_1h': 'ttm_momentum',
    },
    'single_ema': {
        'label': '1 pos, Close>EMA',
        'max_positions': 1,
        'filter_1h': 'close_above_ema',
    },
    'multi3_ttm': {
        'label': '3 pos, TTM mom>0',
        'max_positions': 3,
        'filter_1h': 'ttm_momentum',
    },
    'multi3_ema': {
        'label': '3 pos, Close>EMA',
        'max_positions': 3,
        'filter_1h': 'close_above_ema',
    },
    'multi5_ema': {
        'label': '5 pos, Close>EMA',
        'max_positions': 5,
        'filter_1h': 'close_above_ema',
    },
    'multi5_none': {
        'label': '5 pos, no 1H filter',
        'max_positions': 5,
        'filter_1h': 'none',
    },
}


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


def effective_risk_pct(capital):
    for threshold, pct in SCALING_TIERS:
        if capital < threshold:
            return pct
    return SCALING_TIERS[-1][1]


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION — CONCURRENT POSITIONS + ALTERNATIVE FILTERS")
print("=" * 90)
print(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1")
print(f"Risk: Scale A (2%→1%→0.5%) | Frictions: 0.1%+0.1%")
print(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}")
print()

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
_, momentum_15m, _, _ = calculate_ttm_squeeze_pinescript(
    df_15m['high'], df_15m['low'], df_15m['close'],
    bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
)
df_15m['ttm_momentum'] = momentum_15m

# 1H indicators
_, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
    df_1h['high'], df_1h['low'], df_1h['close'],
    bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
)
df_1h['ttm_momentum'] = momentum_1h
df_1h['ema_21'] = calculate_ema(df_1h['close'], period=21)

# Pre-align 1H data to 15m index (ffill = use last completed 1H bar)
mom_1h_arr = df_1h['ttm_momentum'].reindex(df_15m.index, method='ffill').values
close_1h_arr = df_1h['close'].reindex(df_15m.index, method='ffill').values
ema_1h_arr = df_1h['ema_21'].reindex(df_15m.index, method='ffill').values

# Pre-group 1m candles
period_labels = df_1m_raw.index.floor('15min')
grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}
del df_1m_raw

t_load = time.time() - t_start
print(f"Data loaded in {t_load:.0f}s\n")


# =====================================================
# BACKTEST ENGINE — MULTI-POSITION
# =====================================================

def run_backtest(config):
    capital = float(INITIAL_CAPITAL)
    positions = []   # list of position dicts
    trades = []
    equity_values = []
    equity_timestamps = []
    account_blown = False

    consec_stops = 0
    cooldown_until = 0
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    max_pos = config['max_positions']
    filter_1h = config['filter_1h']

    stats = {
        'signals': 0, 'placed': 0, 'filled': 0, 'expired': 0,
        'skipped_cooldown': 0, 'skipped_halted': 0, 'skipped_maxpos': 0,
        'exits_stop': 0, 'exits_target': 0,
        'same_bar_stops': 0, 'same_bar_targets': 0,
        'max_concurrent': 0,
    }

    empty_df = pd.DataFrame()
    start_bar = 201

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]
        bar = df_15m.iloc[i]

        # ---- Capital floor ----
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                for pos in positions:
                    ep = bar['close']
                    pnl = (ep - pos['ep']) * pos['sz'] \
                          - pos['ep'] * pos['sz'] * COMMISSION_PCT \
                          - ep * pos['sz'] * COMMISSION_PCT
                    capital += pnl
                    trades.append({'pnl': pnl, 'exit_reason': 'BLOWN',
                        'bars_held': i - pos['bar'],
                        'entry_time': pos['et'], 'exit_time': timestamp,
                        'entry_price': pos['ep'], 'exit_price': ep,
                        'r_multiple': 0, 'risk_used': pos['risk_used']})
                positions = []
            equity_values.append(max(capital, 0))
            equity_timestamps.append(timestamp)
            continue

        # ---- Equity tracking (mark to market all positions) ----
        equity = capital
        for pos in positions:
            equity += (bar['close'] - pos['ep']) * pos['sz']
        equity_values.append(max(equity, 0))
        equity_timestamps.append(timestamp)

        # Track max concurrent
        if len(positions) > stats['max_concurrent']:
            stats['max_concurrent'] = len(positions)

        # ---- Daily reset ----
        current_date = timestamp.date()
        if current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ========== PHASE 1: EXIT ALL OPEN POSITIONS ==========
        closed_indices = []
        for pidx, pos in enumerate(positions):
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                # Fallback: 15m bar (stop first)
                if bar['low'] <= pos['sl']:
                    ep = pos['sl'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                    xc = ep * pos['sz'] * COMMISSION_PCT
                    pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                    capital += pnl
                    rd = pos['ep'] - pos['sl']
                    trades.append({'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - pos['bar'],
                        'entry_time': pos['et'], 'exit_time': timestamp,
                        'entry_price': pos['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                        'risk_used': pos['risk_used']})
                    consec_stops += 1
                    daily_losses += abs(pnl)
                    stats['exits_stop'] += 1
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                    if daily_losses >= cb_limit:
                        trading_halted = True
                    closed_indices.append(pidx)

                elif bar['high'] >= pos['tp']:
                    ep = pos['tp']
                    ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                    xc = ep * pos['sz'] * COMMISSION_PCT
                    pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                    capital += pnl
                    rd = pos['ep'] - pos['sl']
                    trades.append({'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - pos['bar'],
                        'entry_time': pos['et'], 'exit_time': timestamp,
                        'entry_price': pos['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                        'risk_used': pos['risk_used']})
                    consec_stops = 0
                    stats['exits_target'] += 1
                    closed_indices.append(pidx)
            else:
                # 1m candle scanning
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values
                exited = False

                for k in range(len(cl)):
                    if ct[k] <= pos['lts']:
                        continue

                    # Stop first (conservative)
                    if cl[k] <= pos['sl']:
                        ep = pos['sl'] * (1 - STOP_SLIPPAGE_PCT)
                        ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                        xc = ep * pos['sz'] * COMMISSION_PCT
                        pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                        capital += pnl
                        rd = pos['ep'] - pos['sl']
                        trades.append({'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - pos['bar'],
                            'entry_time': pos['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': pos['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                            'risk_used': pos['risk_used']})
                        consec_stops += 1
                        daily_losses += abs(pnl)
                        stats['exits_stop'] += 1
                        if consec_stops >= CB_CONSEC_STOP:
                            cooldown_until = i + CB_COOLDOWN_BARS
                        cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                        if daily_losses >= cb_limit:
                            trading_halted = True
                        closed_indices.append(pidx)
                        exited = True
                        break

                    # Target hit
                    if ch[k] >= pos['tp']:
                        ep = pos['tp']
                        ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                        xc = ep * pos['sz'] * COMMISSION_PCT
                        pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                        capital += pnl
                        rd = pos['ep'] - pos['sl']
                        trades.append({'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - pos['bar'],
                            'entry_time': pos['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': pos['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                            'risk_used': pos['risk_used']})
                        consec_stops = 0
                        stats['exits_target'] += 1
                        closed_indices.append(pidx)
                        exited = True
                        break

                if not exited and len(ct) > 0:
                    pos['lts'] = ct[-1]

        # Remove closed positions (reverse order to preserve indices)
        for pidx in sorted(closed_indices, reverse=True):
            positions.pop(pidx)

        # ========== PHASE 2: ENTRIES (re-issue per bar) ==========
        if capital <= MIN_CAPITAL:
            continue

        if len(positions) >= max_pos:
            stats['skipped_maxpos'] += 1
            continue

        if trading_halted:
            stats['skipped_halted'] += 1
            continue
        if i < cooldown_until:
            stats['skipped_cooldown'] += 1
            continue

        prev_bar = df_15m.iloc[i - 1]

        ema = prev_bar['ema_21']
        atr = prev_bar['atr_20']
        if pd.isna(ema) or pd.isna(atr) or atr <= 0:
            continue

        # 1H trend filter (using pre-aligned arrays, prev bar index)
        if filter_1h == 'ttm_momentum':
            mom_val = mom_1h_arr[i - 1]
            if pd.isna(mom_val) or mom_val <= 0:
                continue
        elif filter_1h == 'close_above_ema':
            c_val = close_1h_arr[i - 1]
            e_val = ema_1h_arr[i - 1]
            if pd.isna(c_val) or pd.isna(e_val) or c_val <= e_val:
                continue
        # filter_1h == 'none': skip 1H filter entirely

        # 15m histogram color filter
        mom_c = prev_bar['ttm_momentum']
        mom_p = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
        if get_histogram_color(mom_c, mom_p) == 'red':
            continue

        stats['signals'] += 1

        # Calculate levels
        limit_price = ema + ENTRY_ATR * atr
        stop_level = ema + STOP_ATR * atr
        target_level = ema + TARGET_ATR * atr
        risk_dist = limit_price - stop_level
        reward_dist = target_level - limit_price

        if risk_dist <= 0 or reward_dist <= 0:
            continue

        # Check if we already have a position at very similar levels (avoid stacking identical trades)
        duplicate = False
        for pos in positions:
            if abs(pos['ep'] - limit_price) < 0.1 * atr and abs(pos['sl'] - stop_level) < 0.1 * atr:
                duplicate = True
                break
        if duplicate:
            continue

        stats['placed'] += 1

        # Position sizing (Scale A)
        eff_pct = effective_risk_pct(capital)
        risk_amount = capital * eff_pct

        if risk_amount <= 0:
            continue

        size = risk_amount / risk_dist
        if size <= 0:
            continue
        margin = size * limit_price / LEVERAGE
        if margin > capital:
            size = capital * LEVERAGE / limit_price * 0.95
            if size <= 0:
                continue

        # Fill check: 15m pre-filter
        curr_bar = df_15m.iloc[i]
        if curr_bar['low'] > limit_price:
            stats['expired'] += 1
            continue

        candles = grouped_1m.get(timestamp, empty_df)
        if candles.empty:
            stats['expired'] += 1
            continue

        cl = candles['low'].values
        ch = candles['high'].values
        ct = candles.index.values

        filled = False
        for j in range(len(cl)):
            if cl[j] <= limit_price:
                fill_price = limit_price
                stats['filled'] += 1

                # Same-candle stop check
                same_exit = False
                if cl[j] <= stop_level:
                    ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                    ec = fill_price * size * COMMISSION_PCT
                    xc = ep * size * COMMISSION_PCT
                    pnl = (ep - fill_price) * size - ec - xc
                    capital += pnl
                    trades.append({'pnl': pnl, 'exit_reason': 'STOP_SAME',
                        'bars_held': 0,
                        'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[j]),
                        'entry_price': fill_price, 'exit_price': ep,
                        'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                        'risk_used': risk_amount})
                    stats['same_bar_stops'] += 1
                    consec_stops += 1
                    daily_losses += abs(pnl)
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                    if daily_losses >= cb_limit:
                        trading_halted = True
                    same_exit = True

                # Check subsequent candles for stop/target
                if not same_exit:
                    for k in range(j + 1, len(cl)):
                        if cl[k] <= stop_level:
                            ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                'risk_used': risk_amount})
                            stats['same_bar_stops'] += 1
                            consec_stops += 1
                            daily_losses += abs(pnl)
                            if consec_stops >= CB_CONSEC_STOP:
                                cooldown_until = i + CB_COOLDOWN_BARS
                            cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                            if daily_losses >= cb_limit:
                                trading_halted = True
                            same_exit = True
                            break

                        if ch[k] >= target_level:
                            ep = target_level
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                'risk_used': risk_amount})
                            stats['same_bar_targets'] += 1
                            consec_stops = 0
                            same_exit = True
                            break

                if not same_exit:
                    last_ts = ct[-1] if len(ct) > 0 else ct[j]
                    positions.append({
                        'et': pd.Timestamp(ct[j]), 'bar': i,
                        'ep': fill_price, 'sl': stop_level, 'tp': target_level,
                        'sz': size, 'lts': last_ts, 'risk_used': risk_amount,
                    })

                filled = True
                break

        if not filled:
            stats['expired'] += 1

    # ---- Close remaining positions at end ----
    if positions:
        bar = df_15m.iloc[-1]
        for pos in positions:
            ep = bar['close']
            ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
            xc = ep * pos['sz'] * COMMISSION_PCT
            pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
            capital += pnl
            rd = pos['ep'] - pos['sl']
            trades.append({'pnl': pnl, 'exit_reason': 'END',
                'bars_held': len(df_15m) - 1 - pos['bar'],
                'entry_time': pos['et'], 'exit_time': df_15m.index[-1],
                'entry_price': pos['ep'], 'exit_price': ep,
                'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                'risk_used': pos['risk_used']})

    # ---- Calculate results ----
    eq = np.array(equity_values)
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = drawdowns.min() * 100

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    n_trades = len(trades)
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_r = np.mean([t['r_multiple'] for t in trades]) if trades else 0

    # Yearly breakdown
    yearly = {}
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['year'] = pd.to_datetime(df_trades['exit_time']).dt.year
        for y, grp in df_trades.groupby('year'):
            w = grp[grp['pnl'] > 0]
            yearly[y] = {
                'trades': len(grp),
                'wr': len(w) / len(grp) * 100 if len(grp) > 0 else 0,
                'pnl': grp['pnl'].sum(),
            }

    return {
        'capital': capital,
        'trades': trades,
        'overall': {'trades': n_trades, 'wr': wr, 'pf': pf, 'avg_r': avg_r},
        'max_drawdown': max_dd,
        'peak_equity': peak.max() if len(peak) > 0 else INITIAL_CAPITAL,
        'stats': stats,
        'fill_rate': stats['filled'] / stats['placed'] * 100 if stats['placed'] > 0 else 0,
        'yearly': yearly,
    }


# =====================================================
# RUN ALL CONFIGS
# =====================================================

all_results = {}
t_total = time.time()

for key, config in CONFIGS.items():
    label = config['label']
    print(f"  {label}...", end=" ", flush=True)
    t_run = time.time()
    r = run_backtest(config)
    elapsed = time.time() - t_run
    all_results[key] = r
    o = r['overall']
    pf_str = f"{o['pf']:.2f}" if o['pf'] < 1000 else "inf"
    print(f"{o['trades']} trades, {o['wr']:.1f}% WR, PF {pf_str}, "
          f"DD {r['max_drawdown']:.1f}%, ${r['capital']:,.0f} ({elapsed:.1f}s)")

t_elapsed = time.time() - t_total
print(f"\nAll configs done in {t_elapsed:.0f}s")


# =====================================================
# COMPARISON TABLE
# =====================================================

print()
print("=" * 110)
print("COMPARISON TABLE")
print("=" * 110)
print(f"Baseline: EMA entry, -2/+2 ATR stop/target, Scale A risk, 0.1%+0.1% frictions")
print()

header = f"  {'Config':<24s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} " \
         f"{'MaxDD':>8s} {'Fill%':>6s} {'MaxPos':>7s} {'Final':>14s} {'vs Base':>8s}"
print(header)
print("  " + "-" * 105)

baseline_capital = all_results['single_ttm']['capital']
for key, config in CONFIGS.items():
    r = all_results[key]
    o = r['overall']
    pf_str = f"{o['pf']:>7.2f}" if o['pf'] < 1000 else f"{'inf':>7s}"
    vs_base = (r['capital'] / baseline_capital - 1) * 100 if key != 'single_ttm' else 0
    vs_str = f"{vs_base:>+7.1f}%" if key != 'single_ttm' else f"{'—':>8s}"
    print(f"  {config['label']:<24s} {o['trades']:>7d} {o['wr']:>6.1f}% "
          f"{pf_str} {o['avg_r']:>+6.2f}R "
          f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
          f"{r['stats']['max_concurrent']:>7d} "
          f"${r['capital']:>13,.0f} {vs_str}")

print()

# =====================================================
# YEARLY BREAKDOWN
# =====================================================

print("=" * 110)
print("YEARLY PERFORMANCE BY CONFIG")
print("=" * 110)
print()

years = sorted(set(y for r in all_results.values() for y in r['yearly'].keys()))

for key, config in CONFIGS.items():
    r = all_results[key]
    print(f"  {config['label']}:")
    print(f"    {'Year':>6s} {'Trades':>7s} {'WR':>7s} {'PnL':>14s} {'Cumul':>14s}")
    print(f"    " + "-" * 55)
    cumul = INITIAL_CAPITAL
    for y in years:
        if y in r['yearly']:
            yd = r['yearly'][y]
            cumul += yd['pnl']
            print(f"    {y:>6d} {yd['trades']:>7d} {yd['wr']:>6.1f}% "
                  f"${yd['pnl']:>13,.0f} ${cumul:>13,.0f}")
    print()


# =====================================================
# FILTER & EXECUTION STATS
# =====================================================

print("=" * 110)
print("EXECUTION & FILTER STATS")
print("=" * 110)
print()

for key, config in CONFIGS.items():
    r = all_results[key]
    s = r['stats']
    print(f"  {config['label']}:")
    print(f"    Signals:         {s['signals']:>7d}    Filled:         {s['filled']:>7d} ({r['fill_rate']:.1f}%)")
    print(f"    Placed:          {s['placed']:>7d}    Expired:        {s['expired']:>7d}")
    print(f"    Skip cooldown:   {s['skipped_cooldown']:>7d}    Skip halted:    {s['skipped_halted']:>7d}")
    print(f"    Skip max pos:    {s['skipped_maxpos']:>7d}    Max concurrent: {s['max_concurrent']:>7d}")
    print(f"    Same-bar stops:  {s['same_bar_stops']:>7d}    Same-bar tgts:  {s['same_bar_targets']:>7d}")
    print()


# =====================================================
# SAVE RESULTS
# =====================================================

results_file = 'results/1m_gold_concurrent_results.txt'
with open(results_file, 'w') as f:
    f.write("=" * 110 + "\n")
    f.write("GOLD EMA-ENTRY — CONCURRENT POSITIONS + ALTERNATIVE 1H FILTERS\n")
    f.write("=" * 110 + "\n\n")
    f.write(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR\n")
    f.write(f"Risk: Scale A (2%→1%→0.5%) | Frictions: 0.1%+0.1% | CBs ON\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,} | Runtime: {t_elapsed:.0f}s\n\n")

    f.write("COMPARISON TABLE\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'Config':<24s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} "
            f"{'MaxDD':>8s} {'MaxPos':>7s} {'Final':>14s}\n")
    f.write("-" * 80 + "\n")
    for key, config in CONFIGS.items():
        r = all_results[key]
        o = r['overall']
        pf_str = f"{o['pf']:>7.2f}" if o['pf'] < 1000 else f"{'inf':>7s}"
        f.write(f"{config['label']:<24s} {o['trades']:>7d} {o['wr']:>6.1f}% "
                f"{pf_str} {o['avg_r']:>+6.2f}R "
                f"{r['max_drawdown']:>7.1f}% {r['stats']['max_concurrent']:>7d} "
                f"${r['capital']:>13,.0f}\n")
    f.write("\n")

    # Yearly
    for key, config in CONFIGS.items():
        r = all_results[key]
        f.write(f"\n{config['label']} — Yearly:\n")
        cumul = INITIAL_CAPITAL
        for y in years:
            if y in r['yearly']:
                yd = r['yearly'][y]
                cumul += yd['pnl']
                f.write(f"  {y}: {yd['trades']:>4d} trades, {yd['wr']:>5.1f}% WR, "
                        f"${yd['pnl']:>13,.0f}, cumul ${cumul:>13,.0f}\n")
    f.write("\n" + "=" * 110 + "\n")

print(f"Results saved to {results_file}")

# Save trade logs
for key in CONFIGS.keys():
    r = all_results[key]
    if r['trades']:
        df_t = pd.DataFrame(r['trades'])
        csv_file = f'results/1m_gold_concurrent_{key}_trades.csv'
        df_t.to_csv(csv_file, index=False)

print("Trade logs saved.")
print(f"\nBaseline: EMA entry, -2/+2 ATR, 1H>0, color!=red, Scale A risk")
print("Done.")
