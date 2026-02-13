"""
1-Minute Execution: GOLD EMA-Entry — Optimization Suite

Tests 11 configurations against the winning EMA-entry baseline:
  1. Baseline (progressive cap, no extras)
  2. Runner A (80/20, trail at EMA(21))
  3. Runner B (80/20, trail at entry + 0.5 ATR)
  4. Session filter (07:00-17:00 CET entries only)
  5. ATR percentile >= 40%
  6. ATR percentile >= 50%
  7. Dynamic risk (progressive cap, 50% after 2 consecutive losses)
  8. Fixed 2% risk (no cap)
  9. Fixed 2% + streak guard (3 losses -> 1% for 2 trades)
  10. Fixed 3% risk (no cap)
  11. Fixed 3% + streak guard (3 losses -> 1% for 2 trades)

Strategy (all configs):
  Entry:  Limit at EMA(21)
  Stop:   EMA(21) - 2.0 ATR
  Target: EMA(21) + 2.0 ATR
  Filters: 1H momentum > 0, histogram color != red
  NO squeeze requirement.
  Re-issue per bar. GOLD only. 2020-2026.
  Progressive risk cap, $3,000 start.
  Frictions: 0.1% commission + 0.1% stop slippage.
  Circuit breakers: 2 consecutive stops -> 4h cooldown, daily loss >= 3x cap -> halt.
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
from zoneinfo import ZoneInfo

# =====================================================
# CONFIGURATION
# =====================================================

START_DATE = '2020-01-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 3000
RISK_PER_TRADE = 0.02
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

ATR_PCT_WINDOW = 460   # ~5 trading days of 15m bars

CONFIGS = {
    'baseline': {
        'label': 'Baseline',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
    },
    'runner_a': {
        'label': 'Runner A (EMA trail)',
        'runner': True, 'runner_trail': 'ema', 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
    },
    'runner_b': {
        'label': 'Runner B (entry+0.5ATR)',
        'runner': True, 'runner_trail': 'fixed', 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
    },
    'session': {
        'label': 'Session 07-17 CET',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': True, 'atr_filter': None, 'dynamic_risk': False,
    },
    'atr_40': {
        'label': 'ATR pct >= 40%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': 0.40, 'dynamic_risk': False,
    },
    'atr_50': {
        'label': 'ATR pct >= 50%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': 0.50, 'dynamic_risk': False,
    },
    'dyn_risk': {
        'label': 'Dynamic risk (50%)',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': True,
    },
    'fixed_2pct': {
        'label': 'Fixed 2%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'risk_pct': 0.02, 'use_cap': False,
    },
    'fixed_2pct_dyn': {
        'label': 'Fixed 2% + streak guard',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'risk_pct': 0.02, 'use_cap': False, 'dynamic_risk_v2': True,
    },
    'fixed_3pct': {
        'label': 'Fixed 3%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'risk_pct': 0.03, 'use_cap': False,
    },
    'fixed_3pct_dyn': {
        'label': 'Fixed 3% + streak guard',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'risk_pct': 0.03, 'use_cap': False, 'dynamic_risk_v2': True,
    },
    'scaling_a': {
        'label': 'Scale 2→1→0.5%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'use_cap': False,
        'scaling_tiers': [(1_000_000, 0.02), (10_000_000, 0.01), (float('inf'), 0.005)],
    },
    'scaling_b': {
        'label': 'Scale 2→1→0.5→0.25%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'use_cap': False,
        'scaling_tiers': [(500_000, 0.02), (5_000_000, 0.01), (50_000_000, 0.005), (float('inf'), 0.0025)],
    },
    'scaling_c': {
        'label': 'Scale 3→2→1→0.5%',
        'runner': False, 'runner_trail': None, 'runner_split': 0.8,
        'session_filter': False, 'atr_filter': None, 'dynamic_risk': False,
        'use_cap': False,
        'scaling_tiers': [(100_000, 0.03), (1_000_000, 0.02), (10_000_000, 0.01), (float('inf'), 0.005)],
    },
}

# Only run these configs (set to None to run all)
RUN_ONLY = ['baseline', 'scaling_a', 'scaling_b', 'scaling_c']


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
print("1-MINUTE EXECUTION — GOLD EMA-ENTRY OPTIMIZATION SUITE")
print("=" * 90)
print(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1")
print(f"Frictions: 0.1% comm + 0.1% stop slip | Progressive risk cap")
print(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}")
print(f"Configs: {len(CONFIGS)} variants")
print()

loader = DatabentoMicroFuturesLoader()
t_start = time.time()

df_1m_raw = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE)
print(f"  {len(df_1m_raw)} 1m bars", end=" -> ", flush=True)

df_15m = resample_ohlcv(df_1m_raw, '15min')
df_1h = resample_ohlcv(df_1m_raw, '1h')
print(f"{len(df_15m)} 15m, {len(df_1h)} 1h bars")

# Indicators
df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
df_15m['atr_20'] = calculate_atr(
    df_15m['high'], df_15m['low'], df_15m['close'], period=20
)
_, momentum_15m, _, _ = calculate_ttm_squeeze_pinescript(
    df_15m['high'], df_15m['low'], df_15m['close'],
    bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
)
df_15m['ttm_momentum'] = momentum_15m

_, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
    df_1h['high'], df_1h['low'], df_1h['close'],
    bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
)
df_1h['ttm_momentum'] = momentum_1h

# Pre-group 1m candles
period_labels = df_1m_raw.index.floor('15min')
grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}
del df_1m_raw

# Precompute session filter (CET/CEST 07:00-17:00)
print("Precomputing session filter...", end=" ", flush=True)
cet = ZoneInfo('Europe/Amsterdam')
try:
    cet_times = df_15m.index.tz_localize('UTC').tz_convert(cet)
except TypeError:
    cet_times = df_15m.index.tz_convert(cet)
session_ok = np.array((cet_times.hour >= 7) & (cet_times.hour < 17))
print("done")

# Precompute ATR percentile (rolling rank)
print("Precomputing ATR percentile...", end=" ", flush=True)
t_atr = time.time()
atr_vals = df_15m['atr_20'].values
atr_pct = np.full(len(atr_vals), np.nan)
for idx in range(ATR_PCT_WINDOW, len(atr_vals)):
    w = atr_vals[idx - ATR_PCT_WINDOW:idx]
    valid = w[~np.isnan(w)]
    if len(valid) > 0:
        atr_pct[idx] = np.sum(atr_vals[idx] > valid) / len(valid)
print(f"done ({time.time() - t_atr:.1f}s)")

t_load = time.time() - t_start
print(f"Data loaded in {t_load:.0f}s\n")


# =====================================================
# BACKTEST ENGINE
# =====================================================

def run_backtest(config):
    capital = float(INITIAL_CAPITAL)
    position = None
    runner = None
    trades = []
    runner_trades = []
    equity_values = []
    equity_timestamps = []
    account_blown = False

    consec_stops = 0
    cooldown_until = 0
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    # Dynamic risk state
    consec_losses = 0
    reduced_risk_remaining = 0
    dyn_v2_reduced_remaining = 0

    stats = {
        'signals': 0, 'placed': 0, 'filled': 0, 'expired': 0,
        'skipped_cooldown': 0, 'skipped_halted': 0,
        'skipped_session': 0, 'skipped_atr': 0,
        'exits_stop': 0, 'exits_target': 0,
        'same_bar_stops': 0, 'same_bar_targets': 0,
        'runners_started': 0, 'runners_exited': 0,
    }

    use_runner = config['runner']
    runner_trail_mode = config['runner_trail']
    runner_split = config['runner_split']
    use_session = config['session_filter']
    atr_threshold = config['atr_filter']
    use_dynamic_risk = config['dynamic_risk']
    use_dynamic_risk_v2 = config.get('dynamic_risk_v2', False)
    risk_pct = config.get('risk_pct', RISK_PER_TRADE)
    use_cap = config.get('use_cap', True)
    scaling_tiers = config.get('scaling_tiers', None)

    def effective_risk_pct():
        if scaling_tiers:
            for threshold, pct in scaling_tiers:
                if capital < threshold:
                    return pct
            return scaling_tiers[-1][1]
        return risk_pct

    empty_df = pd.DataFrame()
    start_bar = 201

    for i in range(start_bar, len(df_15m)):
        timestamp = df_15m.index[i]
        bar = df_15m.iloc[i]

        # ---- Capital floor ----
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
                if runner is not None:
                    ep = bar['close']
                    pnl = (ep - runner['ep']) * runner['sz'] \
                          - ep * runner['sz'] * COMMISSION_PCT
                    capital += pnl
                    runner_trades.append({'pnl': pnl, 'exit_reason': 'BLOWN',
                        'bars_held': i - runner['bar'],
                        'entry_time': runner['et'], 'exit_time': timestamp,
                        'entry_price': runner['ep'], 'exit_price': ep,
                        'r_multiple': 0, 'risk_used': 0})
                    runner = None
            equity_values.append(max(capital, 0))
            equity_timestamps.append(timestamp)
            continue

        # ---- Equity tracking ----
        equity = capital
        if position is not None:
            equity += (bar['close'] - position['ep']) * position['sz']
        if runner is not None:
            equity += (bar['close'] - runner['ep']) * runner['sz']
        equity_values.append(max(equity, 0))
        equity_timestamps.append(timestamp)

        # ---- Daily reset ----
        current_date = timestamp.date()
        if current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ========== PHASE 1A: RUNNER EXITS ==========
        if runner is not None:
            # Update EMA trail (ratchet up only)
            if runner_trail_mode == 'ema' and i > 0:
                prev_ema = df_15m.iloc[i - 1]['ema_21']
                if not pd.isna(prev_ema) and prev_ema > runner['trail']:
                    runner['trail'] = prev_ema

            candles = grouped_1m.get(timestamp, empty_df)
            if candles.empty:
                # Fallback: 15m bar
                if bar['low'] <= runner['trail']:
                    ep = runner['trail'] * (1 - STOP_SLIPPAGE_PCT)
                    xc = ep * runner['sz'] * COMMISSION_PCT
                    ec_r = runner['ep'] * runner['sz'] * COMMISSION_PCT
                    pnl = (ep - runner['ep']) * runner['sz'] - ec_r - xc
                    capital += pnl
                    runner_trades.append({'pnl': pnl, 'exit_reason': 'TRAIL_STOP',
                        'bars_held': i - runner['bar'],
                        'entry_time': runner['et'], 'exit_time': timestamp,
                        'entry_price': runner['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (runner['rd'] * runner['sz']) if runner['rd'] * runner['sz'] > 0 else 0,
                        'risk_used': 0})
                    stats['runners_exited'] += 1
                    runner = None
            else:
                cl = candles['low'].values
                ct = candles.index.values
                for k in range(len(cl)):
                    if ct[k] <= runner.get('lts', np.datetime64(0, 'ns')):
                        continue
                    if cl[k] <= runner['trail']:
                        ep = runner['trail'] * (1 - STOP_SLIPPAGE_PCT)
                        xc = ep * runner['sz'] * COMMISSION_PCT
                        ec_r = runner['ep'] * runner['sz'] * COMMISSION_PCT
                        pnl = (ep - runner['ep']) * runner['sz'] - ec_r - xc
                        capital += pnl
                        runner_trades.append({'pnl': pnl, 'exit_reason': 'TRAIL_STOP',
                            'bars_held': i - runner['bar'],
                            'entry_time': runner['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': runner['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (runner['rd'] * runner['sz']) if runner['rd'] * runner['sz'] > 0 else 0,
                            'risk_used': 0})
                        stats['runners_exited'] += 1
                        runner = None
                        break
                if runner is not None and len(ct) > 0:
                    runner['lts'] = ct[-1]

        # ========== PHASE 1B: POSITION EXITS ==========
        if position is not None:
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                # Fallback: 15m bar (stop first)
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
                    consec_losses += 1
                    if use_dynamic_risk and consec_losses >= 2:
                        reduced_risk_remaining = 2
                    if use_dynamic_risk_v2 and consec_losses >= 3:
                        dyn_v2_reduced_remaining = 2
                    daily_losses += abs(pnl)
                    stats['exits_stop'] += 1
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cb_limit = get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULT if use_cap else capital * effective_risk_pct() * CB_DAILY_LOSS_MULT
                    if daily_losses >= cb_limit:
                        trading_halted = True
                    position = None

                elif bar['high'] >= position['tp']:
                    # Target hit — runner split or full exit
                    if use_runner and runner is None:
                        core_sz = position['sz'] * runner_split
                        ep = position['tp']
                        ec = position['ep'] * core_sz * COMMISSION_PCT
                        xc = ep * core_sz * COMMISSION_PCT
                        pnl = (ep - position['ep']) * core_sz - ec - xc
                        capital += pnl
                        rd = position['ep'] - position['sl']
                        trades.append({'pnl': pnl, 'exit_reason': 'TARGET_CORE',
                            'bars_held': i - position['bar'],
                            'entry_time': position['et'], 'exit_time': timestamp,
                            'entry_price': position['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * core_sz) if rd * core_sz > 0 else 0,
                            'risk_used': position['risk_used']})
                        # Create runner
                        r_sz = position['sz'] * (1 - runner_split)
                        if runner_trail_mode == 'ema':
                            trail = df_15m.iloc[i - 1]['ema_21']
                            if pd.isna(trail): trail = position['ep']
                        else:
                            trail = position['ep'] + 0.5 * position['atr_entry']
                        runner = {'et': position['et'], 'bar': position['bar'],
                                  'ep': position['ep'], 'sz': r_sz,
                                  'trail': trail, 'rd': rd,
                                  'lts': np.datetime64(0, 'ns')}
                        stats['runners_started'] += 1
                    else:
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
                        stats['exits_target'] += 1

                    consec_stops = 0
                    consec_losses = 0
                    position = None
            else:
                # 1m candle scanning
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values
                exited = False

                for k in range(len(cl)):
                    if ct[k] <= position['lts']:
                        continue

                    # Stop first (conservative)
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
                        consec_losses += 1
                        if use_dynamic_risk and consec_losses >= 2:
                            reduced_risk_remaining = 2
                        if use_dynamic_risk_v2 and consec_losses >= 3:
                            dyn_v2_reduced_remaining = 2
                        daily_losses += abs(pnl)
                        stats['exits_stop'] += 1
                        if consec_stops >= CB_CONSEC_STOP:
                            cooldown_until = i + CB_COOLDOWN_BARS
                        cb_limit = get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULT if use_cap else capital * effective_risk_pct() * CB_DAILY_LOSS_MULT
                        if daily_losses >= cb_limit:
                            trading_halted = True
                        position = None
                        exited = True
                        break

                    # Target hit
                    if ch[k] >= position['tp']:
                        if use_runner and runner is None:
                            core_sz = position['sz'] * runner_split
                            ep = position['tp']
                            ec = position['ep'] * core_sz * COMMISSION_PCT
                            xc = ep * core_sz * COMMISSION_PCT
                            pnl = (ep - position['ep']) * core_sz - ec - xc
                            capital += pnl
                            rd = position['ep'] - position['sl']
                            trades.append({'pnl': pnl, 'exit_reason': 'TARGET_CORE',
                                'bars_held': i - position['bar'],
                                'entry_time': position['et'], 'exit_time': pd.Timestamp(ct[k]),
                                'entry_price': position['ep'], 'exit_price': ep,
                                'r_multiple': pnl / (rd * core_sz) if rd * core_sz > 0 else 0,
                                'risk_used': position['risk_used']})
                            stats['exits_target'] += 1

                            # Create runner
                            r_sz = position['sz'] * (1 - runner_split)
                            if runner_trail_mode == 'ema':
                                trail = df_15m.iloc[i - 1]['ema_21']
                                if pd.isna(trail): trail = position['ep']
                            else:
                                trail = position['ep'] + 0.5 * position['atr_entry']
                            runner = {'et': position['et'], 'bar': position['bar'],
                                      'ep': position['ep'], 'sz': r_sz,
                                      'trail': trail, 'rd': rd,
                                      'lts': ct[k]}
                            stats['runners_started'] += 1

                            # Check remaining candles for runner trail hit
                            for m in range(k + 1, len(cl)):
                                if cl[m] <= runner['trail']:
                                    ep_r = runner['trail'] * (1 - STOP_SLIPPAGE_PCT)
                                    ec_r = runner['ep'] * runner['sz'] * COMMISSION_PCT
                                    xc_r = ep_r * runner['sz'] * COMMISSION_PCT
                                    pnl_r = (ep_r - runner['ep']) * runner['sz'] - ec_r - xc_r
                                    capital += pnl_r
                                    runner_trades.append({'pnl': pnl_r, 'exit_reason': 'TRAIL_STOP',
                                        'bars_held': i - runner['bar'],
                                        'entry_time': runner['et'], 'exit_time': pd.Timestamp(ct[m]),
                                        'entry_price': runner['ep'], 'exit_price': ep_r,
                                        'r_multiple': pnl_r / (runner['rd'] * runner['sz']) if runner['rd'] * runner['sz'] > 0 else 0,
                                        'risk_used': 0})
                                    stats['runners_exited'] += 1
                                    runner = None
                                    break
                            if runner is not None and len(ct) > 0:
                                runner['lts'] = ct[-1]
                        else:
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
                            stats['exits_target'] += 1

                        consec_stops = 0
                        consec_losses = 0
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

        # 1H trend filter
        bars_1h = df_1h[df_1h.index <= prev_ts]
        if len(bars_1h) < 20:
            continue
        if pd.isna(bars_1h.iloc[-1]['ttm_momentum']) or bars_1h.iloc[-1]['ttm_momentum'] <= 0:
            continue

        # Histogram color filter (no squeeze requirement)
        mom_c = prev_bar['ttm_momentum']
        mom_p = df_15m.iloc[i - 2]['ttm_momentum'] if i >= 2 else np.nan
        if get_histogram_color(mom_c, mom_p) == 'red':
            continue

        stats['signals'] += 1

        # Session filter
        if use_session and not session_ok[i]:
            stats['skipped_session'] += 1
            continue

        # ATR percentile filter
        if atr_threshold is not None:
            pct = atr_pct[i - 1]  # Use previous bar's ATR percentile
            if pd.isna(pct) or pct < atr_threshold:
                stats['skipped_atr'] += 1
                continue

        # Calculate levels
        limit_price = ema + ENTRY_ATR * atr    # = EMA
        stop_level = ema + STOP_ATR * atr       # EMA - 2 ATR
        target_level = ema + TARGET_ATR * atr   # EMA + 2 ATR
        risk_dist = limit_price - stop_level     # 2 ATR
        reward_dist = target_level - limit_price  # 2 ATR

        if risk_dist <= 0 or reward_dist <= 0:
            continue

        stats['placed'] += 1

        # Position sizing
        eff_pct = effective_risk_pct()
        if use_cap:
            risk_cap = get_progressive_risk_cap(capital)
            risk_amount = min(capital * eff_pct, risk_cap)
        else:
            risk_amount = capital * eff_pct

        if use_dynamic_risk and reduced_risk_remaining > 0:
            risk_amount *= 0.5
            reduced_risk_remaining -= 1
        elif use_dynamic_risk_v2 and dyn_v2_reduced_remaining > 0:
            risk_amount = capital * 0.01
            dyn_v2_reduced_remaining -= 1

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
                    consec_losses += 1
                    if use_dynamic_risk and consec_losses >= 2:
                        reduced_risk_remaining = 2
                    if use_dynamic_risk_v2 and consec_losses >= 3:
                        dyn_v2_reduced_remaining = 2
                    daily_losses += abs(pnl)
                    if consec_stops >= CB_CONSEC_STOP:
                        cooldown_until = i + CB_COOLDOWN_BARS
                    cb_limit = get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULT if use_cap else capital * effective_risk_pct() * CB_DAILY_LOSS_MULT
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
                            consec_losses += 1
                            if use_dynamic_risk and consec_losses >= 2:
                                reduced_risk_remaining = 2
                            if use_dynamic_risk_v2 and consec_losses >= 3:
                                dyn_v2_reduced_remaining = 2
                            daily_losses += abs(pnl)
                            if consec_stops >= CB_CONSEC_STOP:
                                cooldown_until = i + CB_COOLDOWN_BARS
                            cb_limit = get_progressive_risk_cap(capital) * CB_DAILY_LOSS_MULT if use_cap else capital * effective_risk_pct() * CB_DAILY_LOSS_MULT
                            if daily_losses >= cb_limit:
                                trading_halted = True
                            same_exit = True
                            break

                        if ch[k] >= target_level:
                            if use_runner and runner is None:
                                # Core exit (80%)
                                core_sz = size * runner_split
                                ep = target_level
                                ec = fill_price * core_sz * COMMISSION_PCT
                                xc = ep * core_sz * COMMISSION_PCT
                                pnl = (ep - fill_price) * core_sz - ec - xc
                                capital += pnl
                                trades.append({'pnl': pnl, 'exit_reason': 'TARGET_CORE_SAME',
                                    'bars_held': 0,
                                    'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': ep,
                                    'r_multiple': pnl / (risk_dist * core_sz) if risk_dist * core_sz > 0 else 0,
                                    'risk_used': risk_amount})
                                stats['same_bar_targets'] += 1
                                consec_stops = 0
                                consec_losses = 0

                                # Create runner
                                r_sz = size * (1 - runner_split)
                                if runner_trail_mode == 'ema':
                                    trail = df_15m.iloc[i - 1]['ema_21']
                                    if pd.isna(trail): trail = fill_price
                                else:
                                    trail = fill_price + 0.5 * atr
                                runner = {'et': pd.Timestamp(ct[j]), 'bar': i,
                                          'ep': fill_price, 'sz': r_sz,
                                          'trail': trail, 'rd': risk_dist,
                                          'lts': ct[k]}
                                stats['runners_started'] += 1

                                # Check remaining candles for runner trail
                                for m in range(k + 1, len(cl)):
                                    if cl[m] <= runner['trail']:
                                        ep_r = runner['trail'] * (1 - STOP_SLIPPAGE_PCT)
                                        ec_r = runner['ep'] * runner['sz'] * COMMISSION_PCT
                                        xc_r = ep_r * runner['sz'] * COMMISSION_PCT
                                        pnl_r = (ep_r - runner['ep']) * runner['sz'] - ec_r - xc_r
                                        capital += pnl_r
                                        runner_trades.append({'pnl': pnl_r, 'exit_reason': 'TRAIL_STOP',
                                            'bars_held': 0,
                                            'entry_time': runner['et'], 'exit_time': pd.Timestamp(ct[m]),
                                            'entry_price': runner['ep'], 'exit_price': ep_r,
                                            'r_multiple': pnl_r / (runner['rd'] * runner['sz']) if runner['rd'] * runner['sz'] > 0 else 0,
                                            'risk_used': 0})
                                        stats['runners_exited'] += 1
                                        runner = None
                                        break
                                if runner is not None and len(ct) > 0:
                                    runner['lts'] = ct[-1]
                            else:
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
                                consec_losses = 0

                            same_exit = True
                            break

                if not same_exit:
                    last_ts = ct[-1] if len(ct) > 0 else ct[j]
                    position = {
                        'et': pd.Timestamp(ct[j]), 'bar': i,
                        'ep': fill_price, 'sl': stop_level, 'tp': target_level,
                        'sz': size, 'lts': last_ts, 'risk_used': risk_amount,
                        'atr_entry': atr,
                    }

                filled = True
                break

        if not filled:
            stats['expired'] += 1

    # ---- Close remaining positions ----
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

    if runner is not None:
        bar = df_15m.iloc[-1]
        ep = bar['close']
        ec_r = runner['ep'] * runner['sz'] * COMMISSION_PCT
        xc = ep * runner['sz'] * COMMISSION_PCT
        pnl = (ep - runner['ep']) * runner['sz'] - ec_r - xc
        capital += pnl
        runner_trades.append({'pnl': pnl, 'exit_reason': 'END',
            'bars_held': 0,
            'entry_time': runner['et'], 'exit_time': df_15m.index[-1],
            'entry_price': runner['ep'], 'exit_price': ep,
            'r_multiple': pnl / (runner['rd'] * runner['sz']) if runner['rd'] * runner['sz'] > 0 else 0,
            'risk_used': 0})

    # ---- Compute metrics ----
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    runner_df = pd.DataFrame(runner_trades) if runner_trades else pd.DataFrame()
    eq = pd.Series(equity_values)

    if len(eq) > 0:
        rm = eq.expanding().max().clip(lower=1e-6)
        max_dd = ((eq - rm) / rm).min() * 100
        peak_eq = eq.max()
    else:
        max_dd = 0
        peak_eq = INITIAL_CAPITAL

    # Combined trades for overall metrics
    all_trades_df = pd.concat([trades_df, runner_df], ignore_index=True) if len(runner_df) > 0 else trades_df

    def compute_metrics(tdf):
        if len(tdf) == 0:
            return {'trades': 0, 'wr': 0, 'pf': 0, 'avg_r': 0, 'total_pnl': 0}
        wins = tdf[tdf['pnl'] > 0]
        losses = tdf[tdf['pnl'] <= 0]
        wr = len(wins) / len(tdf) * 100
        pf = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')
        return {'trades': len(tdf), 'wr': wr, 'pf': pf, 'avg_r': tdf['r_multiple'].mean(), 'total_pnl': tdf['pnl'].sum()}

    overall = compute_metrics(all_trades_df)
    core_metrics = compute_metrics(trades_df)
    runner_metrics = compute_metrics(runner_df)

    fill_rate = stats['filled'] / stats['placed'] * 100 if stats['placed'] > 0 else 0

    # Yearly breakdown
    yearly = {}
    if len(all_trades_df) > 0:
        tdf = all_trades_df.copy()
        tdf['year'] = pd.to_datetime(tdf['entry_time']).dt.year
        for year, grp in tdf.groupby('year'):
            w = (grp['pnl'] > 0).sum()
            yearly[year] = {
                'trades': len(grp), 'wr': w / len(grp) * 100,
                'pnl': grp['pnl'].sum(), 'avg_r': grp['r_multiple'].mean(),
            }

    return {
        'capital': capital, 'max_drawdown': max_dd, 'peak_equity': peak_eq,
        'fill_rate': fill_rate, 'stats': stats,
        'overall': overall, 'core': core_metrics, 'runner': runner_metrics,
        'yearly': yearly, 'trades_df': all_trades_df,
        'equity_curve': eq, 'equity_timestamps': equity_timestamps,
    }


# =====================================================
# RUN ALL CONFIGS
# =====================================================

all_results = {}
t_total = time.time()

for key, config in CONFIGS.items():
    if RUN_ONLY and key not in RUN_ONLY:
        continue
    label = config['label']
    print(f"  {label}...", end=" ", flush=True)
    t_run = time.time()
    r = run_backtest(config)
    elapsed = time.time() - t_run
    all_results[key] = r
    o = r['overall']
    pf_str = f"{o['pf']:.2f}" if o['pf'] < 1000 else "inf"
    print(f"{o['trades']} trades, {o['wr']:.1f}% WR, PF {pf_str}, "
          f"${r['capital']:,.0f} (DD {r['max_drawdown']:.1f}%) ({elapsed:.0f}s)")

t_elapsed = time.time() - t_total
print(f"\nTotal runtime: {t_elapsed:.0f}s\n")


# =====================================================
# RESULTS — COMPARISON TABLE
# =====================================================

print()
print("=" * 110)
print("OPTIMIZATION RESULTS — COMPARISON TABLE")
print("=" * 110)
print(f"Baseline: EMA entry, -2/+2 ATR stop/target, 1H>0, color!=red, no squeeze")
print(f"All: $3K start, progressive cap, 0.1%+0.1% frictions, circuit breakers")
print()

header = f"  {'Config':<28s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} " \
         f"{'MaxDD':>8s} {'Fill%':>6s} {'Final':>12s} {'vs Base':>8s}"
print(header)
print("  " + "-" * 105)

baseline_capital = all_results['baseline']['capital'] if 'baseline' in all_results else 1
for key, config in CONFIGS.items():
    if key not in all_results:
        continue
    r = all_results[key]
    o = r['overall']
    pf_str = f"{o['pf']:>7.2f}" if o['pf'] < 1000 else f"{'inf':>7s}"
    vs_base = (r['capital'] / baseline_capital - 1) * 100 if key != 'baseline' else 0
    vs_str = f"{vs_base:>+7.1f}%" if key != 'baseline' else f"{'—':>8s}"
    marker = " *" if key != 'baseline' and r['capital'] > baseline_capital else ""
    print(f"  {config['label']:<28s} {o['trades']:>7d} {o['wr']:>6.1f}% "
          f"{pf_str} {o['avg_r']:>+6.2f}R "
          f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
          f"${r['capital']:>11,.0f} {vs_str}{marker}")

print()
print("  * = beats baseline")
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
    if key not in all_results:
        continue
    r = all_results[key]
    print(f"  {config['label']}:")
    print(f"    {'Year':>6s} {'Trades':>7s} {'WR':>7s} {'AvgR':>7s} {'PnL':>12s} {'Cumul':>12s}")
    print(f"    " + "-" * 55)
    cumul = INITIAL_CAPITAL
    for y in years:
        if y in r['yearly']:
            yd = r['yearly'][y]
            cumul += yd['pnl']
            print(f"    {y:>6d} {yd['trades']:>7.0f} {yd['wr']:>6.1f}% "
                  f"{yd['avg_r']:>+6.2f}R ${yd['pnl']:>11,.0f} ${cumul:>11,.0f}")
    print()


# =====================================================
# RUNNER ANALYSIS (Test Set 1)
# =====================================================

print("=" * 110)
print("RUNNER ANALYSIS — CORE vs RUNNER LEG")
print("=" * 110)
print()

for key in ['runner_a', 'runner_b']:
    if key not in all_results:
        continue
    r = all_results[key]
    config = CONFIGS[key]
    c = r['core']
    rn = r['runner']
    print(f"  {config['label']}:")
    print(f"    {'Leg':<15s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} {'Total PnL':>12s}")
    print(f"    " + "-" * 60)

    pf_c = f"{c['pf']:>7.2f}" if c['pf'] < 1000 else f"{'inf':>7s}"
    pf_r = f"{rn['pf']:>7.2f}" if rn['pf'] < 1000 else f"{'inf':>7s}"
    print(f"    {'Core (80%)':<15s} {c['trades']:>7d} {c['wr']:>6.1f}% "
          f"{pf_c} {c['avg_r']:>+6.2f}R ${c['total_pnl']:>11,.0f}")
    print(f"    {'Runner (20%)':<15s} {rn['trades']:>7d} {rn['wr']:>6.1f}% "
          f"{pf_r} {rn['avg_r']:>+6.2f}R ${rn['total_pnl']:>11,.0f}")
    print(f"    {'Combined':<15s} {c['trades']+rn['trades']:>7d}")
    print(f"    Runners started: {r['stats']['runners_started']}, "
          f"exited: {r['stats']['runners_exited']}")
    print()


# =====================================================
# EXECUTION STATS
# =====================================================

print("=" * 110)
print("EXECUTION & FILTER STATS")
print("=" * 110)
print()

for key, config in CONFIGS.items():
    if key not in all_results:
        continue
    r = all_results[key]
    s = r['stats']
    print(f"  {config['label']}:")
    print(f"    Signals:         {s['signals']:>7d}    Filled:         {s['filled']:>7d} ({r['fill_rate']:.1f}%)")
    print(f"    Placed:          {s['placed']:>7d}    Expired:        {s['expired']:>7d}")
    print(f"    Skip cooldown:   {s['skipped_cooldown']:>7d}    Skip halted:    {s['skipped_halted']:>7d}")
    print(f"    Skip session:    {s['skipped_session']:>7d}    Skip ATR:       {s['skipped_atr']:>7d}")
    print(f"    Same-bar stops:  {s['same_bar_stops']:>7d}    Same-bar tgts:  {s['same_bar_targets']:>7d}")
    sb_pct = s['same_bar_stops'] / s['filled'] * 100 if s['filled'] > 0 else 0
    print(f"    Same-bar SL %:   {sb_pct:>6.1f}%")
    print()


# =====================================================
# DRAWDOWN COMPARISON
# =====================================================

print("=" * 110)
print("DRAWDOWN & RISK COMPARISON")
print("=" * 110)
print()

print(f"  {'Config':<28s} {'MaxDD':>8s} {'Final':>12s} {'Peak':>12s} {'Return':>10s}")
print("  " + "-" * 75)

for key, config in CONFIGS.items():
    if key not in all_results:
        continue
    r = all_results[key]
    ret = (r['capital'] / INITIAL_CAPITAL - 1) * 100
    print(f"  {config['label']:<28s} {r['max_drawdown']:>7.1f}% "
          f"${r['capital']:>11,.0f} ${r['peak_equity']:>11,.0f} {ret:>+9.0f}%")
print()


# =====================================================
# RECOMMENDATION
# =====================================================

print("=" * 110)
print("SUMMARY & RECOMMENDATION")
print("=" * 110)
print()

# Find best by PF (excluding inf)
results_sorted = sorted(
    [(k, all_results[k]) for k in all_results],
    key=lambda x: x[1]['overall']['pf'] if x[1]['overall']['pf'] < 1000 else 0,
    reverse=True
)

print("  Ranked by Profit Factor:")
for rank, (key, r) in enumerate(results_sorted, 1):
    o = r['overall']
    pf_str = f"{o['pf']:.2f}" if o['pf'] < 1000 else "inf"
    marker = " <-- baseline" if key == 'baseline' else ""
    better = " BETTER" if key != 'baseline' and r['capital'] > baseline_capital else ""
    print(f"    {rank}. {CONFIGS[key]['label']:<28s} PF {pf_str:>7s}  "
          f"WR {o['wr']:.1f}%  DD {r['max_drawdown']:.1f}%  "
          f"${r['capital']:>11,.0f}{marker}{better}")

print()

# Beats baseline?
better_configs = [k for k in all_results if k != 'baseline'
                  and all_results[k]['capital'] > baseline_capital]
worse_configs = [k for k in all_results if k != 'baseline'
                 and all_results[k]['capital'] <= baseline_capital]

if better_configs:
    print(f"  Configs that BEAT baseline: {', '.join(CONFIGS[k]['label'] for k in better_configs)}")
if worse_configs:
    print(f"  Configs that HURT baseline: {', '.join(CONFIGS[k]['label'] for k in worse_configs)}")
print()

# Best overall
best_key = results_sorted[0][0]
best_r = all_results[best_key]
print(f"  Best config: {CONFIGS[best_key]['label']}")
print(f"    PF {best_r['overall']['pf']:.2f}, WR {best_r['overall']['wr']:.1f}%, "
      f"MaxDD {best_r['max_drawdown']:.1f}%, ${INITIAL_CAPITAL:,} -> ${best_r['capital']:,.0f}")
print()


# =====================================================
# SAVE RESULTS
# =====================================================

output_file = 'results/1m_gold_optimization_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 110 + "\n")
    f.write("GOLD EMA-ENTRY OPTIMIZATION SUITE — 1-MINUTE EXECUTION\n")
    f.write("=" * 110 + "\n\n")
    f.write(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR\n")
    f.write(f"Frictions: 0.1%+0.1% | Progressive cap | CBs ON\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,} | Runtime: {t_elapsed:.0f}s\n\n")

    f.write("COMPARISON TABLE\n")
    f.write("-" * 110 + "\n")
    f.write(f"{'Config':<28s} {'Trades':>7s} {'WR':>7s} {'PF':>7s} {'AvgR':>7s} "
            f"{'MaxDD':>8s} {'Final':>12s}\n")
    f.write("-" * 80 + "\n")
    for key, config in CONFIGS.items():
        if key not in all_results:
            continue
        r = all_results[key]
        o = r['overall']
        pf_str = f"{o['pf']:>7.2f}" if o['pf'] < 1000 else f"{'inf':>7s}"
        f.write(f"{config['label']:<28s} {o['trades']:>7d} {o['wr']:>6.1f}% "
                f"{pf_str} {o['avg_r']:>+6.2f}R "
                f"{r['max_drawdown']:>7.1f}% ${r['capital']:>11,.0f}\n")
    f.write("\n")

    # Yearly
    for key, config in CONFIGS.items():
        if key not in all_results:
            continue
        r = all_results[key]
        f.write(f"\n{config['label']} — Yearly:\n")
        cumul = INITIAL_CAPITAL
        for y in years:
            if y in r['yearly']:
                yd = r['yearly'][y]
                cumul += yd['pnl']
                f.write(f"  {y}: {yd['trades']:>4d} trades, {yd['wr']:>5.1f}% WR, "
                        f"${yd['pnl']:>11,.0f}, cumul ${cumul:>11,.0f}\n")
    f.write("\n")

    # Runner analysis
    for key in ['runner_a', 'runner_b']:
        if key not in all_results:
            continue
        r = all_results[key]
        c = r['core']
        rn = r['runner']
        f.write(f"\n{CONFIGS[key]['label']} — Leg Analysis:\n")
        f.write(f"  Core:   {c['trades']} trades, {c['wr']:.1f}% WR, PF {c['pf']:.2f}, ${c['total_pnl']:,.0f}\n")
        f.write(f"  Runner: {rn['trades']} trades, {rn['wr']:.1f}% WR, PF {rn['pf']:.2f}, ${rn['total_pnl']:,.0f}\n")
    f.write("\n" + "=" * 110 + "\n")

print(f"Results saved: {output_file}")

# Save trade logs
for key in all_results:
    r = all_results[key]
    if len(r['trades_df']) > 0:
        fname = f'results/1m_gold_opt_{key}_trades.csv'
        r['trades_df'].to_csv(fname, index=False)

print("Trade logs saved.")
print()
print("=" * 110)
