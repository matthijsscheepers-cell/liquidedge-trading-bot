"""
1-Minute Execution: Multi-Asset — TTM Momentum vs Close > EMA(21)

Tests the 1H filter change (TTM momentum > 0 vs Close > EMA) across all 4 assets:
  GOLD, SILVER, US100, US500

All single-position, Scale A risk, same strategy:
  Entry:  Limit at EMA(21)
  Stop:   EMA(21) - 2.0 ATR
  Target: EMA(21) + 2.0 ATR
  15m filter: histogram color != red
  NO squeeze requirement.
  Re-issue per bar. 2020-2026.
  Scale A risk (2% < $1M, 1% < $10M, 0.5% >= $10M).
  Frictions: 0.1% commission + 0.1% stop slippage.
  Circuit breakers: 2 consecutive stops -> 4h cooldown.
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

ENTRY_ATR = 0.0
STOP_ATR = -2.0
TARGET_ATR = 2.0

CB_CONSEC_STOP = 2
CB_COOLDOWN_BARS = 16
CB_DAILY_LOSS_MULT = 3

SCALING_TIERS = [(1_000_000, 0.02), (10_000_000, 0.01), (float('inf'), 0.005)]

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

FILTERS = {
    'ttm_momentum': 'TTM mom>0',
    'close_above_ema': 'Close>EMA',
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
# BACKTEST ENGINE (single position per asset)
# =====================================================

def run_backtest(df_15m, df_1h, grouped_1m, mom_1h_arr, close_1h_arr, ema_1h_arr, filter_1h):
    capital = float(INITIAL_CAPITAL)
    position = None
    trades = []
    equity_values = []
    equity_timestamps = []
    account_blown = False

    consec_stops = 0
    cooldown_until = 0
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    stats = {
        'signals': 0, 'placed': 0, 'filled': 0, 'expired': 0,
        'skipped_cooldown': 0, 'skipped_halted': 0,
        'exits_stop': 0, 'exits_target': 0,
        'same_bar_stops': 0, 'same_bar_targets': 0,
    }

    empty_df = pd.DataFrame()
    start_bar = 201

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]
        bar = df_15m.iloc[i]

        # Capital floor
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                if position is not None:
                    ep = bar['close']
                    pnl = (ep - position['ep']) * position['sz'] \
                          - position['ep'] * position['sz'] * COMMISSION_PCT \
                          - ep * position['sz'] * COMMISSION_PCT
                    capital += pnl
                    trades.append({'pnl': pnl, 'exit_reason': 'BLOWN',
                        'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': 0, 'risk_used': position['risk_used']})
                    position = None
            equity_values.append(max(capital, 0))
            equity_timestamps.append(timestamp)
            continue

        # Equity tracking
        equity = capital
        if position is not None:
            equity += (bar['close'] - position['ep']) * position['sz']
        equity_values.append(max(equity, 0))
        equity_timestamps.append(timestamp)

        # Daily reset
        current_date = timestamp.date()
        if current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ========== PHASE 1: EXITS ==========
        if position is not None:
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                if bar['low'] <= position['sl']:
                    ep = position['sl'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = position['ep'] * position['sz'] * COMMISSION_PCT
                    xc = ep * position['sz'] * COMMISSION_PCT
                    pnl = (ep - position['ep']) * position['sz'] - ec - xc
                    capital += pnl
                    rd = position['ep'] - position['sl']
                    trades.append({'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                        'risk_used': position['risk_used']})
                    consec_stops += 1
                    daily_losses += abs(pnl)
                    stats['exits_stop'] += 1
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                    if daily_losses >= cb_limit:
                        trading_halted = True
                    position = None

                elif bar['high'] >= position['tp']:
                    ep = position['tp']
                    ec = position['ep'] * position['sz'] * COMMISSION_PCT
                    xc = ep * position['sz'] * COMMISSION_PCT
                    pnl = (ep - position['ep']) * position['sz'] - ec - xc
                    capital += pnl
                    rd = position['ep'] - position['sl']
                    trades.append({'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                        'risk_used': position['risk_used']})
                    consec_stops = 0
                    stats['exits_target'] += 1
                    position = None
            else:
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values
                exited = False

                for k in range(len(cl)):
                    if ct[k] <= position['lts']:
                        continue

                    if cl[k] <= position['sl']:
                        ep = position['sl'] * (1 - STOP_SLIPPAGE_PCT)
                        ec = position['ep'] * position['sz'] * COMMISSION_PCT
                        xc = ep * position['sz'] * COMMISSION_PCT
                        pnl = (ep - position['ep']) * position['sz'] - ec - xc
                        capital += pnl
                        rd = position['ep'] - position['sl']
                        trades.append({'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - position['bar'],
                            'entry_time': position['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                            'risk_used': position['risk_used']})
                        consec_stops += 1
                        daily_losses += abs(pnl)
                        stats['exits_stop'] += 1
                        if consec_stops >= CB_CONSEC_STOP:
                            cooldown_until = i + CB_COOLDOWN_BARS
                        cb_limit = capital * effective_risk_pct(capital) * CB_DAILY_LOSS_MULT
                        if daily_losses >= cb_limit:
                            trading_halted = True
                        position = None
                        exited = True
                        break

                    if ch[k] >= position['tp']:
                        ep = position['tp']
                        ec = position['ep'] * position['sz'] * COMMISSION_PCT
                        xc = ep * position['sz'] * COMMISSION_PCT
                        pnl = (ep - position['ep']) * position['sz'] - ec - xc
                        capital += pnl
                        rd = position['ep'] - position['sl']
                        trades.append({'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - position['bar'],
                            'entry_time': position['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                            'risk_used': position['risk_used']})
                        consec_stops = 0
                        stats['exits_target'] += 1
                        position = None
                        exited = True
                        break

                if not exited and position is not None and len(ct) > 0:
                    position['lts'] = ct[-1]

        # ========== PHASE 2: ENTRIES ==========
        if capital <= MIN_CAPITAL or position is not None:
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

        # 1H trend filter
        if filter_1h == 'ttm_momentum':
            mom_val = mom_1h_arr[i - 1]
            if pd.isna(mom_val) or mom_val <= 0:
                continue
        elif filter_1h == 'close_above_ema':
            c_val = close_1h_arr[i - 1]
            e_val = ema_1h_arr[i - 1]
            if pd.isna(c_val) or pd.isna(e_val) or c_val <= e_val:
                continue

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

        stats['placed'] += 1

        # Position sizing
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

        # Fill check
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
                    position = {
                        'et': pd.Timestamp(ct[j]), 'bar': i,
                        'ep': fill_price, 'sl': stop_level, 'tp': target_level,
                        'sz': size, 'lts': last_ts, 'risk_used': risk_amount,
                    }

                filled = True
                break

        if not filled:
            stats['expired'] += 1

    # Close remaining
    if position is not None:
        bar = df_15m.iloc[-1]
        ep = bar['close']
        ec = position['ep'] * position['sz'] * COMMISSION_PCT
        xc = ep * position['sz'] * COMMISSION_PCT
        pnl = (ep - position['ep']) * position['sz'] - ec - xc
        capital += pnl
        rd = position['ep'] - position['sl']
        trades.append({'pnl': pnl, 'exit_reason': 'END',
            'bars_held': 0,
            'entry_time': position['et'], 'exit_time': df_15m.index[-1],
            'entry_price': position['ep'], 'exit_price': ep,
            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
            'risk_used': position['risk_used']})

    # Results
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

    # Yearly
    yearly = {}
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['year'] = pd.to_datetime(df_trades['exit_time'], utc=True).dt.year
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
        'stats': stats,
        'fill_rate': stats['filled'] / stats['placed'] * 100 if stats['placed'] > 0 else 0,
        'yearly': yearly,
    }


# =====================================================
# MAIN
# =====================================================

print("=" * 100)
print("MULTI-ASSET FILTER COMPARISON: TTM Momentum vs Close > EMA(21)")
print("=" * 100)
print(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1")
print(f"Risk: Scale A (2%→1%→0.5%) | Frictions: 0.1%+0.1% | Single position")
print(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}")
print()

loader = DatabentoMicroFuturesLoader()
all_results = {}

for asset in ASSETS:
    print(f"\n{'='*60}")
    print(f"  Loading {asset}...")
    t_start = time.time()

    try:
        df_1m_raw = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        print(f"  ERROR loading {asset}: {e}")
        continue

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

    # Pre-align 1H data
    mom_1h_arr = df_1h['ttm_momentum'].reindex(df_15m.index, method='ffill').values
    close_1h_arr = df_1h['close'].reindex(df_15m.index, method='ffill').values
    ema_1h_arr = df_1h['ema_21'].reindex(df_15m.index, method='ffill').values

    # Pre-group 1m candles
    period_labels = df_1m_raw.index.floor('15min')
    grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}
    del df_1m_raw

    t_load = time.time() - t_start
    print(f"  Data ready in {t_load:.0f}s")

    # Run both filters
    for filter_key, filter_label in FILTERS.items():
        print(f"    {filter_label}...", end=" ", flush=True)
        t_run = time.time()
        r = run_backtest(df_15m, df_1h, grouped_1m, mom_1h_arr, close_1h_arr, ema_1h_arr, filter_key)
        elapsed = time.time() - t_run
        result_key = f"{asset}_{filter_key}"
        all_results[result_key] = r
        o = r['overall']
        pf_str = f"{o['pf']:.2f}" if o['pf'] < 1000 else "inf"
        print(f"{o['trades']} trades, {o['wr']:.1f}% WR, PF {pf_str}, "
              f"DD {r['max_drawdown']:.1f}%, ${r['capital']:,.0f} ({elapsed:.1f}s)")

    # Cleanup
    del df_15m, df_1h, grouped_1m

print()
print()
print("=" * 120)
print("FINAL COMPARISON — ALL ASSETS × BOTH FILTERS")
print("=" * 120)
print()

header = f"  {'Asset':<8s} {'Filter':<14s} {'Trades':>7s} {'WR':>7s} {'PF':>8s} {'AvgR':>7s} " \
         f"{'MaxDD':>8s} {'Final':>16s} {'Signals':>8s} {'Fill%':>6s}"
print(header)
print("  " + "-" * 110)

for asset in ASSETS:
    key_ttm = f"{asset}_ttm_momentum"
    key_ema = f"{asset}_close_above_ema"
    if key_ttm not in all_results or key_ema not in all_results:
        continue

    for filter_key, filter_label in FILTERS.items():
        result_key = f"{asset}_{filter_key}"
        r = all_results[result_key]
        o = r['overall']
        pf_str = f"{o['pf']:>8.2f}" if o['pf'] < 1000 else f"{'inf':>8s}"
        print(f"  {asset:<8s} {filter_label:<14s} {o['trades']:>7d} {o['wr']:>6.1f}% "
              f"{pf_str} {o['avg_r']:>+6.2f}R "
              f"{r['max_drawdown']:>7.1f}% ${r['capital']:>15,.0f} "
              f"{r['stats']['signals']:>8d} {r['fill_rate']:>5.1f}%")

    # Delta row
    r_ttm = all_results[key_ttm]
    r_ema = all_results[key_ema]
    wr_delta = r_ema['overall']['wr'] - r_ttm['overall']['wr']
    dd_delta = r_ema['max_drawdown'] - r_ttm['max_drawdown']
    trade_delta = r_ema['overall']['trades'] - r_ttm['overall']['trades']
    if r_ttm['capital'] > 0:
        cap_ratio = r_ema['capital'] / r_ttm['capital']
    else:
        cap_ratio = float('inf')
    print(f"  {'':8s} {'DELTA':>14s} {trade_delta:>+7d} {wr_delta:>+6.1f}pp "
          f"{'':>8s} {'':>7s} "
          f"{dd_delta:>+7.1f}pp {'x' + f'{cap_ratio:.1f}':>16s}")
    print("  " + "-" * 110)

# Yearly detail for each asset
print()
print("=" * 120)
print("YEARLY DETAIL")
print("=" * 120)
print()

years = sorted(set(y for r in all_results.values() for y in r['yearly'].keys()))

for asset in ASSETS:
    key_ttm = f"{asset}_ttm_momentum"
    key_ema = f"{asset}_close_above_ema"
    if key_ttm not in all_results or key_ema not in all_results:
        continue

    print(f"  {asset}:")
    print(f"    {'Year':>6s} | {'TTM mom>0':^28s} | {'Close>EMA':^28s} |")
    print(f"    {'':>6s} | {'Trades':>7s} {'WR':>6s} {'PnL':>12s} | {'Trades':>7s} {'WR':>6s} {'PnL':>12s} |")
    print(f"    " + "-" * 70)

    r_ttm = all_results[key_ttm]
    r_ema = all_results[key_ema]

    for y in years:
        yt = r_ttm['yearly'].get(y, {'trades': 0, 'wr': 0, 'pnl': 0})
        ye = r_ema['yearly'].get(y, {'trades': 0, 'wr': 0, 'pnl': 0})
        print(f"    {y:>6d} | {yt['trades']:>7d} {yt['wr']:>5.1f}% ${yt['pnl']:>11,.0f} "
              f"| {ye['trades']:>7d} {ye['wr']:>5.1f}% ${ye['pnl']:>11,.0f} |")
    print()


# Save results
results_file = 'results/1m_multiasset_filter_results.txt'
with open(results_file, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write("MULTI-ASSET FILTER COMPARISON: TTM Momentum vs Close > EMA(21)\n")
    f.write("=" * 120 + "\n\n")
    f.write(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1\n")
    f.write(f"Risk: Scale A | Frictions: 0.1%+0.1% | Single position | CBs ON\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}\n\n")

    f.write(f"{'Asset':<8s} {'Filter':<14s} {'Trades':>7s} {'WR':>7s} {'PF':>8s} {'AvgR':>7s} "
            f"{'MaxDD':>8s} {'Final':>16s}\n")
    f.write("-" * 80 + "\n")
    for asset in ASSETS:
        for filter_key, filter_label in FILTERS.items():
            result_key = f"{asset}_{filter_key}"
            if result_key not in all_results:
                continue
            r = all_results[result_key]
            o = r['overall']
            pf_str = f"{o['pf']:>8.2f}" if o['pf'] < 1000 else f"{'inf':>8s}"
            f.write(f"{asset:<8s} {filter_label:<14s} {o['trades']:>7d} {o['wr']:>6.1f}% "
                    f"{pf_str} {o['avg_r']:>+6.2f}R "
                    f"{r['max_drawdown']:>7.1f}% ${r['capital']:>15,.0f}\n")
        f.write("\n")

    # Yearly
    for asset in ASSETS:
        for filter_key, filter_label in FILTERS.items():
            result_key = f"{asset}_{filter_key}"
            if result_key not in all_results:
                continue
            r = all_results[result_key]
            f.write(f"\n{asset} {filter_label} — Yearly:\n")
            cumul = INITIAL_CAPITAL
            for y in years:
                if y in r['yearly']:
                    yd = r['yearly'][y]
                    cumul += yd['pnl']
                    f.write(f"  {y}: {yd['trades']:>4d} trades, {yd['wr']:>5.1f}% WR, "
                            f"${yd['pnl']:>13,.0f}, cumul ${cumul:>13,.0f}\n")
    f.write("\n" + "=" * 120 + "\n")

print(f"\nResults saved to {results_file}")
print("Done.")
