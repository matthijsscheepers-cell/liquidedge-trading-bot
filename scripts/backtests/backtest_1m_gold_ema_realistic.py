"""
1-Minute Execution: GOLD EMA-Entry — Realistic Capital Test

Best finding from sweep: Entry at EMA(21) itself, 85% WR, PF 18+.
Now test with realistic constraints:

Entry:  Limit at EMA(21)
Stop:   EMA(21) - 2.0 ATR
Target: EMA(21) + 2.0 ATR
R:R = 1:1 (risk = reward = 2 ATR). Edge from 85% win rate.

Filters: 1H momentum > 0, 15m squeeze in last 4 bars, color != red
Re-issue per bar. GOLD only. Post-2020.

Risk modes:
  A) Progressive cap: min(2% × capital, tier_cap)
     <$1K→$50, <$5K→$100, <$20K→$200, <$100K→$500, ≥$100K→$1000
  B) Pure 2% (no cap)

Capital levels: $300, $1,000, $3,000

Circuit breakers:
  - 2 consecutive stops → 4h cooldown (16 bars)
  - Daily loss >= 3× risk cap → halt until next day

Frictions: 0.1% commission each side + 0.1% stop slippage
Max 1 position at a time.
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
RISK_PER_TRADE = 0.02
LEVERAGE = 20
MIN_CAPITAL = 1.0

COMMISSION_PCT = 0.001
STOP_SLIPPAGE_PCT = 0.001

SQUEEZE_LOOKBACK = 4

# Entry at EMA, stop/target ±2 ATR
ENTRY_ATR = 0.0      # EMA itself
STOP_ATR = -2.0      # EMA - 2.0 ATR
TARGET_ATR = 2.0     # EMA + 2.0 ATR

# Circuit breakers
CB_CONSEC_STOP = 2
CB_COOLDOWN_BARS = 16   # 4 hours
CB_DAILY_LOSS_MULT = 3

CAPITAL_LEVELS = [300, 1_000, 3_000]
RISK_MODES = ['progressive', 'pure_2pct']


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
print("1-MINUTE EXECUTION — GOLD EMA-ENTRY REALISTIC TEST")
print("=" * 90)
print(f"Entry: EMA(21) | Stop: EMA - 2.0 ATR | Target: EMA + 2.0 ATR (1:1 R:R)")
print(f"Frictions: {COMMISSION_PCT*100:.1f}% comm + {STOP_SLIPPAGE_PCT*100:.1f}% slip")
print(f"Capitals: {', '.join(f'${c:,}' for c in CAPITAL_LEVELS)}")
print(f"Risk modes: progressive cap, pure 2%")
print(f"Circuit breakers: {CB_CONSEC_STOP} stops → {CB_COOLDOWN_BARS*15//60}h cooldown")
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
# BACKTEST ENGINE
# =====================================================

def run_backtest(initial_capital, risk_mode='progressive'):
    capital = initial_capital
    position = None
    trades = []
    equity_values = []
    equity_timestamps = []

    account_blown = False
    account_blown_ts = None

    consec_stops = 0
    cooldown_until = 0
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    stats = {
        'signals': 0,
        'placed': 0,
        'filled': 0,
        'expired': 0,
        'skipped_cooldown': 0,
        'skipped_halted': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
    }

    empty_df = pd.DataFrame()
    start_bar = 201

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]

        # Capital floor
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                account_blown_ts = timestamp
                if position is not None:
                    bar = df_15m.iloc[i]
                    ep = bar['close']
                    ec = position['ep'] * position['sz'] * COMMISSION_PCT
                    xc = ep * position['sz'] * COMMISSION_PCT
                    pnl = (ep - position['ep']) * position['sz'] - ec - xc
                    capital += pnl
                    rd = position['ep'] - position['sl']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'BLOWN', 'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                        'risk_used': position['risk_used'],
                    })
                    position = None
            equity_values.append(max(capital, 0))
            equity_timestamps.append(timestamp)
            continue

        # Equity tracking
        equity = capital
        if position is not None:
            bar = df_15m.iloc[i]
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
                bar = df_15m.iloc[i]
                if bar['low'] <= position['sl']:
                    ep = position['sl'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = position['ep'] * position['sz'] * COMMISSION_PCT
                    xc = ep * position['sz'] * COMMISSION_PCT
                    pnl = (ep - position['ep']) * position['sz'] - ec - xc
                    capital += pnl
                    rd = position['ep'] - position['sl']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP',
                        'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                        'risk_used': position['risk_used'],
                    })
                    consec_stops += 1
                    daily_losses += abs(pnl)
                    stats['exits_stop'] += 1
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cap = get_progressive_risk_cap(capital) if risk_mode == 'progressive' else capital * RISK_PER_TRADE
                    if daily_losses >= cap * CB_DAILY_LOSS_MULT:
                        trading_halted = True
                    position = None
                elif bar['high'] >= position['tp']:
                    ep = position['tp']
                    ec = position['ep'] * position['sz'] * COMMISSION_PCT
                    xc = ep * position['sz'] * COMMISSION_PCT
                    pnl = (ep - position['ep']) * position['sz'] - ec - xc
                    capital += pnl
                    rd = position['ep'] - position['sl']
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'TARGET',
                        'bars_held': i - position['bar'],
                        'entry_time': position['et'], 'exit_time': timestamp,
                        'entry_price': position['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                        'risk_used': position['risk_used'],
                    })
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
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'STOP',
                            'bars_held': i - position['bar'],
                            'entry_time': position['et'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                            'risk_used': position['risk_used'],
                        })
                        consec_stops += 1
                        daily_losses += abs(pnl)
                        stats['exits_stop'] += 1
                        if consec_stops >= CB_CONSEC_STOP:
                            cooldown_until = i + CB_COOLDOWN_BARS
                        cap = get_progressive_risk_cap(capital) if risk_mode == 'progressive' else capital * RISK_PER_TRADE
                        if daily_losses >= cap * CB_DAILY_LOSS_MULT:
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
                        trades.append({
                            'pnl': pnl, 'exit_reason': 'TARGET',
                            'bars_held': i - position['bar'],
                            'entry_time': position['et'],
                            'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': position['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
                            'risk_used': position['risk_used'],
                        })
                        consec_stops = 0
                        stats['exits_target'] += 1
                        position = None
                        exited = True
                        break

                if not exited and position is not None and len(ct) > 0:
                    position['lts'] = ct[-1]

        # ========== PHASE 2: ENTRIES (re-issue per bar) ==========
        if capital <= MIN_CAPITAL or position is not None:
            continue

        if trading_halted:
            stats['skipped_halted'] += 1
            continue

        if i < cooldown_until:
            stats['skipped_cooldown'] += 1
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

        # Squeeze
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
        if get_histogram_color(mom_c, mom_p) == 'red':
            continue

        stats['signals'] += 1

        # Calculate levels
        limit_price = ema + ENTRY_ATR * atr     # = EMA
        stop_level = ema + STOP_ATR * atr        # EMA - 2 ATR
        target_level = ema + TARGET_ATR * atr    # EMA + 2 ATR

        risk_dist = limit_price - stop_level     # 2 ATR
        reward_dist = target_level - limit_price  # 2 ATR

        if risk_dist <= 0 or reward_dist <= 0:
            continue

        stats['placed'] += 1

        # Position sizing
        if risk_mode == 'progressive':
            risk_cap = get_progressive_risk_cap(capital)
            risk_amount = min(capital * RISK_PER_TRADE, risk_cap)
        else:
            risk_amount = capital * RISK_PER_TRADE

        if risk_amount <= 0:
            continue

        size = risk_amount / risk_dist
        if size <= 0:
            continue
        margin = size * limit_price / LEVERAGE
        if margin > capital:
            # Reduce size to fit margin
            size = capital * LEVERAGE / limit_price * 0.95  # 95% margin use
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

                # Same-candle checks
                same_exit = False
                if cl[j] <= stop_level:
                    ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                    ec = fill_price * size * COMMISSION_PCT
                    xc = ep * size * COMMISSION_PCT
                    pnl = (ep - fill_price) * size - ec - xc
                    capital += pnl
                    trades.append({
                        'pnl': pnl, 'exit_reason': 'STOP_SAME', 'bars_held': 0,
                        'entry_time': pd.Timestamp(ct[j]),
                        'exit_time': pd.Timestamp(ct[j]),
                        'entry_price': fill_price, 'exit_price': ep,
                        'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                        'risk_used': risk_amount,
                    })
                    stats['same_bar_stops'] += 1
                    consec_stops += 1
                    daily_losses += abs(pnl)
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cap_check = get_progressive_risk_cap(capital) if risk_mode == 'progressive' else capital * RISK_PER_TRADE
                    if daily_losses >= cap_check * CB_DAILY_LOSS_MULT:
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
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'STOP_SAME', 'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                'risk_used': risk_amount,
                            })
                            stats['same_bar_stops'] += 1
                            consec_stops += 1
                            daily_losses += abs(pnl)
                            if consec_stops >= CB_CONSEC_STOP:
                                cooldown_until = i + CB_COOLDOWN_BARS
                            cap_check = get_progressive_risk_cap(capital) if risk_mode == 'progressive' else capital * RISK_PER_TRADE
                            if daily_losses >= cap_check * CB_DAILY_LOSS_MULT:
                                trading_halted = True
                            same_exit = True
                            break

                        if ch[k] >= target_level:
                            ep = target_level
                            ec = fill_price * size * COMMISSION_PCT
                            xc = ep * size * COMMISSION_PCT
                            pnl = (ep - fill_price) * size - ec - xc
                            capital += pnl
                            trades.append({
                                'pnl': pnl, 'exit_reason': 'TARGET_SAME', 'bars_held': 0,
                                'entry_time': pd.Timestamp(ct[j]),
                                'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': fill_price, 'exit_price': ep,
                                'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                'risk_used': risk_amount,
                            })
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
        trades.append({
            'pnl': pnl, 'exit_reason': 'END', 'bars_held': 0,
            'entry_time': position['et'], 'exit_time': df_15m.index[-1],
            'entry_price': position['ep'], 'exit_price': ep,
            'r_multiple': pnl / (rd * position['sz']) if rd * position['sz'] > 0 else 0,
            'risk_used': position['risk_used'],
        })

    # Metrics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    eq = pd.Series(equity_values)

    if len(eq) > 0:
        rm = eq.expanding().max().clip(lower=1e-6)
        max_dd = ((eq - rm) / rm).min() * 100
        # Find peak equity
        peak_eq = eq.max()
    else:
        max_dd = 0
        peak_eq = initial_capital

    if len(trades_df) > 0:
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        wr = len(wins) / len(trades_df) * 100
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')
        avg_r = trades_df['r_multiple'].mean()
        avg_risk = trades_df['risk_used'].mean()
    else:
        wr = pf = avg_r = avg_risk = 0

    fill_rate = stats['filled'] / stats['placed'] * 100 if stats['placed'] > 0 else 0

    # Yearly
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
                'avg_risk': grp['risk_used'].mean(),
            }

    return {
        'trades': len(trades_df),
        'capital': capital,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'peak_equity': peak_eq,
        'avg_r': avg_r,
        'avg_risk': avg_risk,
        'fill_rate': fill_rate,
        'stats': stats,
        'account_blown_ts': account_blown_ts,
        'yearly': yearly,
        'trades_df': trades_df,
        'initial_capital': initial_capital,
        'risk_mode': risk_mode,
        'equity_curve': eq,
        'equity_timestamps': equity_timestamps,
    }


# =====================================================
# RUN ALL COMBINATIONS
# =====================================================

all_results = {}
t_total = time.time()

for risk_mode in RISK_MODES:
    mode_label = "Progressive Cap" if risk_mode == 'progressive' else "Pure 2%"
    print(f"--- {mode_label} ---")
    for cap in CAPITAL_LEVELS:
        label = f"${cap:,} {mode_label}"
        print(f"  ${cap:,}...", end=" ", flush=True)
        t_run = time.time()
        r = run_backtest(initial_capital=cap, risk_mode=risk_mode)
        elapsed = time.time() - t_run
        r['label'] = label
        all_results[(risk_mode, cap)] = r

        blown = f" blown:{str(r['account_blown_ts'])[:10]}" if r['account_blown_ts'] else ""
        print(f"{r['trades']} trades, {r['win_rate']:.1f}% WR, PF {r['profit_factor']:.2f}, "
              f"${r['capital']:,.0f} (peak ${r['peak_equity']:,.0f}) "
              f"DD {r['max_drawdown']:.1f}%{blown} ({elapsed:.0f}s)")
    print()

t_elapsed = time.time() - t_total

# =====================================================
# RESULTS
# =====================================================

print()
print("=" * 90)
print("RESULTS — GOLD EMA-ENTRY REALISTIC TEST")
print("=" * 90)
print(f"Entry: EMA(21) | Stop: EMA - 2.0 ATR | Target: EMA + 2.0 ATR")
print(f"Frictions: 0.1% comm + 0.1% slip | Circuit breakers ON")
print(f"Runtime: {t_elapsed:.0f}s")
print()

for risk_mode in RISK_MODES:
    mode_label = "Progressive Cap" if risk_mode == 'progressive' else "Pure 2%"
    print(f"--- {mode_label} ---")
    print()
    print(f"  {'Start':>8s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} "
          f"{'AvgRisk':>8s} {'MaxDD':>8s} {'Fill%':>6s} {'Peak':>12s} {'Final':>12s}")
    print("  " + "-" * 100)

    for cap in CAPITAL_LEVELS:
        r = all_results[(risk_mode, cap)]
        blown = "*" if r['account_blown_ts'] else ""
        pf_str = f"{r['profit_factor']:>7.2f}" if r['profit_factor'] < 1000 else f"{'∞':>7s}"
        print(f"  ${cap:>7,} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{pf_str} {r['avg_r']:>+6.2f}R "
              f"${r['avg_risk']:>7.0f} {r['max_drawdown']:>7.1f}% "
              f"{r['fill_rate']:>5.1f}% ${r['peak_equity']:>11,.0f} "
              f"${r['capital']:>11,.0f}{blown}")
    print()

# Yearly breakdown
print("=" * 90)
print("YEARLY PERFORMANCE")
print("=" * 90)
print()

years = sorted(set(y for r in all_results.values() for y in r['yearly'].keys()))

for risk_mode in RISK_MODES:
    mode_label = "Progressive Cap" if risk_mode == 'progressive' else "Pure 2%"
    print(f"--- {mode_label} ---")
    print()

    for cap in CAPITAL_LEVELS:
        r = all_results[(risk_mode, cap)]
        print(f"  ${cap:,} start:")
        print(f"    {'Year':>6s} {'Trades':>7s} {'WR':>7s} {'AvgR':>7s} {'AvgRisk$':>9s} {'PnL':>12s} {'Cumul':>12s}")
        print(f"    " + "-" * 65)
        cumul = cap
        for y in years:
            if y in r['yearly']:
                yd = r['yearly'][y]
                cumul += yd['pnl']
                print(f"    {y:>6d} {yd['trades']:>7.0f} {yd['wr']:>6.1f}% "
                      f"{yd['avg_r']:>+6.2f}R ${yd['avg_risk']:>8.0f} "
                      f"${yd['pnl']:>11,.0f} ${cumul:>11,.0f}")
        print()

# Circuit breaker stats
print("=" * 90)
print("EXECUTION & CIRCUIT BREAKER STATS")
print("=" * 90)
print()

for risk_mode in RISK_MODES:
    mode_label = "Prog Cap" if risk_mode == 'progressive' else "Pure 2%"
    for cap in CAPITAL_LEVELS:
        r = all_results[(risk_mode, cap)]
        s = r['stats']
        print(f"  ${cap:,} {mode_label}:")
        print(f"    Signals:           {s['signals']:>7d}")
        print(f"    Placed:            {s['placed']:>7d}")
        print(f"    Filled:            {s['filled']:>7d}  ({r['fill_rate']:.1f}%)")
        print(f"    Expired:           {s['expired']:>7d}")
        print(f"    Skipped (cooldown):{s['skipped_cooldown']:>7d}")
        print(f"    Skipped (halted):  {s['skipped_halted']:>7d}")
        print(f"    Same-bar stops:    {s['same_bar_stops']:>7d}")
        print(f"    Same-bar targets:  {s['same_bar_targets']:>7d}")
        sb_pct = s['same_bar_stops'] / s['filled'] * 100 if s['filled'] > 0 else 0
        print(f"    Same-bar SL %:     {sb_pct:>6.1f}%")
        print()

# R-multiple distribution
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION (Progressive Cap, $3K start)")
print("=" * 90)
print()

r = all_results[('progressive', 3000)]
tdf = r['trades_df']
if len(tdf) > 0:
    bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
            (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
    print(f"  {len(tdf)} trades")
    for lo, hi in bins:
        n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
        pct = n / len(tdf) * 100
        bar = '#' * int(pct)
        print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
    print()

# Growth comparison
print("=" * 90)
print("GROWTH SUMMARY")
print("=" * 90)
print()
print(f"  {'Config':<30s} {'Start':>8s} {'Final':>12s} {'Peak':>12s} {'Return':>8s} {'MaxDD':>8s}")
print("  " + "-" * 85)

for risk_mode in RISK_MODES:
    mode_short = "Cap" if risk_mode == 'progressive' else "2%"
    for cap in CAPITAL_LEVELS:
        r = all_results[(risk_mode, cap)]
        ret = (r['capital'] / cap - 1) * 100
        label = f"${cap:,} {mode_short}"
        blown = " (blown)" if r['account_blown_ts'] else ""
        print(f"  {label:<30s} ${cap:>7,} ${r['capital']:>11,.0f} ${r['peak_equity']:>11,.0f} "
              f"{ret:>+7.0f}% {r['max_drawdown']:>7.1f}%{blown}")
print()

# Key finding
print("=" * 90)
print("KEY FINDINGS")
print("=" * 90)
print()

best = max(all_results.values(), key=lambda r: r['profit_factor'] if r['profit_factor'] < 1000 else 0)
print(f"  Best PF: {best['label']} → PF {best['profit_factor']:.2f}, "
      f"{best['win_rate']:.1f}% WR, ${best['initial_capital']:,} → ${best['capital']:,.0f}")
print()

# Check if profitable across all configs
all_profitable = all(r['capital'] > r['initial_capital'] for r in all_results.values())
print(f"  All configs profitable: {'YES' if all_profitable else 'NO'}")
for risk_mode in RISK_MODES:
    mode_short = "Cap" if risk_mode == 'progressive' else "2%"
    profitable = [cap for cap in CAPITAL_LEVELS
                  if all_results[(risk_mode, cap)]['capital'] > cap]
    blown = [cap for cap in CAPITAL_LEVELS
             if all_results[(risk_mode, cap)]['account_blown_ts'] is not None]
    print(f"    {mode_short}: profitable at {', '.join(f'${c:,}' for c in profitable) or 'none'}"
          f"{', blown at ' + ', '.join(f'${c:,}' for c in blown) if blown else ''}")
print()

# Save results
output_file = 'results/1m_gold_ema_realistic_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("GOLD EMA-ENTRY REALISTIC TEST — 1-MINUTE EXECUTION\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR\n")
    f.write(f"Frictions: 0.1%+0.1% | CBs ON | Runtime: {t_elapsed:.0f}s\n\n")

    for risk_mode in RISK_MODES:
        mode_label = "Progressive Cap" if risk_mode == 'progressive' else "Pure 2%"
        f.write(f"\n{mode_label}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Start':>8s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} "
                f"{'MaxDD':>8s} {'Peak':>12s} {'Final':>12s}\n")
        f.write("-" * 80 + "\n")
        for cap in CAPITAL_LEVELS:
            r = all_results[(risk_mode, cap)]
            pf_str = f"{r['profit_factor']:>7.2f}" if r['profit_factor'] < 1000 else f"{'inf':>7s}"
            f.write(f"${cap:>7,} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                    f"{pf_str} {r['avg_r']:>+6.2f}R "
                    f"{r['max_drawdown']:>7.1f}% ${r['peak_equity']:>11,.0f} "
                    f"${r['capital']:>11,.0f}\n")
        f.write("\n")

        # Yearly
        for cap in CAPITAL_LEVELS:
            r = all_results[(risk_mode, cap)]
            f.write(f"\n  ${cap:,} yearly:\n")
            cumul = cap
            for y in years:
                if y in r['yearly']:
                    yd = r['yearly'][y]
                    cumul += yd['pnl']
                    f.write(f"    {y}: {yd['trades']:>3d} trades, {yd['wr']:>5.1f}% WR, "
                            f"${yd['pnl']:>11,.0f}, cumul ${cumul:>11,.0f}\n")
        f.write("\n")

    f.write("=" * 90 + "\n")

print(f"Results saved: {output_file}")

# Save trade logs
for risk_mode in RISK_MODES:
    for cap in CAPITAL_LEVELS:
        r = all_results[(risk_mode, cap)]
        if len(r['trades_df']) > 0:
            mode_short = "cap" if risk_mode == 'progressive' else "2pct"
            fname = f'results/1m_gold_ema_{mode_short}_{cap}_trades.csv'
            r['trades_df'].to_csv(fname, index=False)

print("Trade logs saved.")
print()
print("=" * 90)
