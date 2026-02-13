"""
1-Minute Execution: Combined Portfolio — GOLD + SILVER + US500

Tests shared equity curve with asset-weighted risk allocations.
All use Close > EMA(21) as 1H filter.

Configs:
  1. gold_only       — GOLD 2% (baseline)
  2. gold_us500      — GOLD 2%, US500 2% (current live)
  3. all_2pct        — GOLD 2%, SILVER 2%, US500 2%
  4. all_w1          — GOLD 2%, SILVER 1%, US500 1%
  5. all_w05         — GOLD 2%, SILVER 0.5%, US500 0.5%
  6. gold_silver     — GOLD 2%, SILVER 1%
  7. all_delayed_50k — GOLD 2%, SILVER 1%, US500 1% (start others after $50K)

Same strategy: Entry EMA(21), Stop EMA-2ATR, Target EMA+2ATR.
Close>EMA 1H filter. 15m histogram != red. No squeeze.
Scale A tiers applied per-asset risk. 1-min execution. CBs ON.
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

ASSETS = ['GOLD', 'SILVER', 'US500']

# Configurations: {name: {asset: base_risk_pct, ...}, delay_threshold}
CONFIGS = {
    'gold_only': {
        'assets': {'GOLD': 0.02},
        'delay_threshold': 0,
        'label': 'GOLD 2%',
    },
    'gold_us500': {
        'assets': {'GOLD': 0.02, 'US500': 0.02},
        'delay_threshold': 0,
        'label': 'GOLD 2% + US500 2%',
    },
    'all_2pct': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.02, 'US500': 0.02},
        'delay_threshold': 0,
        'label': 'ALL 2%',
    },
    'all_w1': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 0,
        'label': 'GOLD 2% + SLV/US5 1%',
    },
    'all_w05': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.005, 'US500': 0.005},
        'delay_threshold': 0,
        'label': 'GOLD 2% + SLV/US5 0.5%',
    },
    'gold_silver': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01},
        'delay_threshold': 0,
        'label': 'GOLD 2% + SILVER 1%',
    },
    'all_delayed_50k': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 50_000,
        'label': 'ALL weighted, delayed $50K',
    },
    # Aggressive start: use 2% on all assets until step_up_threshold,
    # then drop SILVER/US500 to their configured base_risk
    'all_step_2k': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 0,
        'step_up_threshold': 2_000,
        'step_up_risk': 0.02,
        'label': 'ALL w1, aggressive <$2K',
    },
    'all_step_5k': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 0,
        'step_up_threshold': 5_000,
        'step_up_risk': 0.02,
        'label': 'ALL w1, aggressive <$5K',
    },
    'all_step_10k': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 0,
        'step_up_threshold': 10_000,
        'step_up_risk': 0.02,
        'label': 'ALL w1, aggressive <$10K',
    },
    'all_step_50k': {
        'assets': {'GOLD': 0.02, 'SILVER': 0.01, 'US500': 0.01},
        'delay_threshold': 0,
        'step_up_threshold': 50_000,
        'step_up_risk': 0.02,
        'label': 'ALL w1, aggressive <$50K',
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


def scale_risk(base_pct, capital):
    """Apply Scale A tiers to base risk pct."""
    for threshold, tier_pct in SCALING_TIERS:
        if capital < threshold:
            # Scale proportionally: if base is 1% and tier is 2%, use 1%
            # If base is 2% and tier is 2%, use 2%
            # If base is 2% and tier is 1%, use 1% (cap at tier)
            return min(base_pct, tier_pct)
    return min(base_pct, SCALING_TIERS[-1][1])


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


# =====================================================
# COMBINED PORTFOLIO BACKTEST
# =====================================================

def run_portfolio(config, asset_data):
    """
    Run combined portfolio backtest with shared equity curve.

    asset_data: {asset: {df_15m, grouped_1m, close_1h_arr, ema_1h_arr}}
    config: {assets: {asset: base_risk_pct}, delay_threshold}
    """
    capital = float(INITIAL_CAPITAL)
    peak_capital = capital
    positions = {}  # {asset: position_dict}
    all_trades = []
    equity_values = []
    equity_timestamps = []

    # Per-asset circuit breakers
    consec_stops = {a: 0 for a in config['assets']}
    cooldown_until = {a: 0 for a in config['assets']}

    # Portfolio-wide
    daily_losses = 0.0
    daily_loss_date = None
    trading_halted = False

    delay_threshold = config.get('delay_threshold', 0)

    # Per-asset stats
    asset_stats = {}
    for asset in config['assets']:
        asset_stats[asset] = {
            'signals': 0, 'placed': 0, 'filled': 0, 'expired': 0,
            'exits_stop': 0, 'exits_target': 0,
            'same_bar_stops': 0, 'same_bar_targets': 0,
            'trades': 0, 'wins': 0,
        }

    # Build unified timeline from all assets
    all_timestamps = set()
    for asset in config['assets']:
        if asset in asset_data:
            all_timestamps.update(asset_data[asset]['df_15m'].index[201:])
    unified_index = sorted(all_timestamps)

    empty_df = pd.DataFrame()

    for timestamp in unified_index:
        # Capital floor
        if capital <= MIN_CAPITAL:
            equity_values.append(max(capital, 0))
            equity_timestamps.append(timestamp)
            # Close all positions
            for asset in list(positions.keys()):
                pos = positions[asset]
                ad = asset_data[asset]
                df_15m = ad['df_15m']
                if timestamp in df_15m.index:
                    bar = df_15m.loc[timestamp]
                    ep = bar['close']
                else:
                    ep = pos['ep']
                pnl = (ep - pos['ep']) * pos['sz'] \
                      - pos['ep'] * pos['sz'] * COMMISSION_PCT \
                      - ep * pos['sz'] * COMMISSION_PCT
                capital += pnl
                all_trades.append({
                    'asset': asset, 'pnl': pnl, 'exit_reason': 'BLOWN',
                    'entry_time': pos['et'], 'exit_time': timestamp,
                    'entry_price': pos['ep'], 'exit_price': ep,
                    'r_multiple': 0, 'risk_used': pos['risk_used'],
                })
                del positions[asset]
            continue

        # Equity tracking (mark-to-market)
        equity = capital
        for asset, pos in positions.items():
            ad = asset_data[asset]
            df_15m = ad['df_15m']
            if timestamp in df_15m.index:
                equity += (df_15m.loc[timestamp, 'close'] - pos['ep']) * pos['sz']
            else:
                # Use last known close
                idx = df_15m.index.searchsorted(timestamp) - 1
                if idx >= 0:
                    equity += (df_15m.iloc[idx]['close'] - pos['ep']) * pos['sz']
        equity_values.append(max(equity, 0))
        equity_timestamps.append(timestamp)

        # Daily reset
        current_date = timestamp.date()
        if current_date != daily_loss_date:
            daily_losses = 0.0
            daily_loss_date = current_date
            trading_halted = False

        # ========== PHASE 1: EXITS (all assets) ==========
        for asset in list(positions.keys()):
            pos = positions[asset]
            ad = asset_data[asset]
            df_15m = ad['df_15m']
            grouped_1m = ad['grouped_1m']
            stats = asset_stats[asset]

            if timestamp not in df_15m.index:
                continue

            bar = df_15m.loc[timestamp]
            candles = grouped_1m.get(timestamp, empty_df)

            if candles.empty:
                # Fallback: 15m bar
                if bar['low'] <= pos['sl']:
                    ep = pos['sl'] * (1 - STOP_SLIPPAGE_PCT)
                    ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                    xc = ep * pos['sz'] * COMMISSION_PCT
                    pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                    capital += pnl
                    rd = pos['ep'] - pos['sl']
                    all_trades.append({
                        'asset': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                        'entry_time': pos['et'], 'exit_time': timestamp,
                        'entry_price': pos['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                        'risk_used': pos['risk_used'],
                    })
                    consec_stops[asset] += 1
                    daily_losses += abs(pnl)
                    stats['exits_stop'] += 1
                    if consec_stops[asset] >= CB_CONSEC_STOP:
                        # Find bar index for cooldown
                        bar_idx = df_15m.index.get_loc(timestamp)
                        cooldown_until[asset] = bar_idx + CB_COOLDOWN_BARS
                    cb_limit = capital * scale_risk(config['assets'][asset], capital) * CB_DAILY_LOSS_MULT
                    if daily_losses >= cb_limit:
                        trading_halted = True
                    del positions[asset]

                elif bar['high'] >= pos['tp']:
                    ep = pos['tp']
                    ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                    xc = ep * pos['sz'] * COMMISSION_PCT
                    pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                    capital += pnl
                    rd = pos['ep'] - pos['sl']
                    all_trades.append({
                        'asset': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                        'entry_time': pos['et'], 'exit_time': timestamp,
                        'entry_price': pos['ep'], 'exit_price': ep,
                        'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                        'risk_used': pos['risk_used'],
                    })
                    consec_stops[asset] = 0
                    stats['exits_target'] += 1
                    del positions[asset]
            else:
                # 1-min candle scan
                cl = candles['low'].values
                ch = candles['high'].values
                ct = candles.index.values
                exited = False

                for k in range(len(cl)):
                    if ct[k] <= pos['lts']:
                        continue

                    if cl[k] <= pos['sl']:
                        ep = pos['sl'] * (1 - STOP_SLIPPAGE_PCT)
                        ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                        xc = ep * pos['sz'] * COMMISSION_PCT
                        pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                        capital += pnl
                        rd = pos['ep'] - pos['sl']
                        all_trades.append({
                            'asset': asset, 'pnl': pnl, 'exit_reason': 'STOP',
                            'entry_time': pos['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': pos['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                            'risk_used': pos['risk_used'],
                        })
                        consec_stops[asset] += 1
                        daily_losses += abs(pnl)
                        stats['exits_stop'] += 1
                        if consec_stops[asset] >= CB_CONSEC_STOP:
                            bar_idx = df_15m.index.get_loc(timestamp)
                            cooldown_until[asset] = bar_idx + CB_COOLDOWN_BARS
                        cb_limit = capital * scale_risk(config['assets'][asset], capital) * CB_DAILY_LOSS_MULT
                        if daily_losses >= cb_limit:
                            trading_halted = True
                        del positions[asset]
                        exited = True
                        break

                    if ch[k] >= pos['tp']:
                        ep = pos['tp']
                        ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
                        xc = ep * pos['sz'] * COMMISSION_PCT
                        pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
                        capital += pnl
                        rd = pos['ep'] - pos['sl']
                        all_trades.append({
                            'asset': asset, 'pnl': pnl, 'exit_reason': 'TARGET',
                            'entry_time': pos['et'], 'exit_time': pd.Timestamp(ct[k]),
                            'entry_price': pos['ep'], 'exit_price': ep,
                            'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
                            'risk_used': pos['risk_used'],
                        })
                        consec_stops[asset] = 0
                        stats['exits_target'] += 1
                        del positions[asset]
                        exited = True
                        break

                if not exited and asset in positions and len(ct) > 0:
                    positions[asset]['lts'] = ct[-1]

        # ========== PHASE 2: ENTRIES (all assets) ==========
        if capital <= MIN_CAPITAL or trading_halted:
            continue

        # Step-up logic: use aggressive risk when capital is small
        step_up_threshold = config.get('step_up_threshold', 0)
        step_up_risk = config.get('step_up_risk', 0.02)

        for asset, base_risk in config['assets'].items():
            if asset in positions:
                continue  # Already holding this asset
            if asset not in asset_data:
                continue

            # Apply step-up: if capital below threshold, use aggressive risk
            if step_up_threshold > 0 and capital < step_up_threshold:
                effective_base_risk = step_up_risk
            else:
                effective_base_risk = base_risk

            ad = asset_data[asset]
            df_15m = ad['df_15m']

            if timestamp not in df_15m.index:
                continue

            bar_idx = df_15m.index.get_loc(timestamp)
            if bar_idx < 201:
                continue

            stats = asset_stats[asset]

            # Cooldown
            if bar_idx < cooldown_until.get(asset, 0):
                continue

            # Delay threshold (don't trade this asset until equity > threshold)
            if delay_threshold > 0 and asset != 'GOLD':
                if capital < delay_threshold:
                    continue

            # Signal from previous bar
            prev_bar = df_15m.iloc[bar_idx - 1]
            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']
            if pd.isna(ema) or pd.isna(atr) or atr <= 0:
                continue

            # 1H Close > EMA filter
            close_1h_arr = ad['close_1h_arr']
            ema_1h_arr = ad['ema_1h_arr']
            c_val = close_1h_arr[bar_idx - 1]
            e_val = ema_1h_arr[bar_idx - 1]
            if pd.isna(c_val) or pd.isna(e_val) or c_val <= e_val:
                continue

            # 15m histogram color filter
            mom_c = prev_bar['ttm_momentum']
            mom_p = df_15m.iloc[bar_idx - 2]['ttm_momentum'] if bar_idx >= 2 else np.nan
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

            # Position sizing with asset-weighted risk
            eff_pct = scale_risk(effective_base_risk, capital)
            risk_amount = capital * eff_pct
            if risk_amount <= 0:
                continue

            size = risk_amount / risk_dist
            if size <= 0:
                continue
            margin = size * limit_price / LEVERAGE
            if margin > capital * 0.5:  # Don't use more than 50% margin per position
                size = capital * 0.5 * LEVERAGE / limit_price * 0.95
                if size <= 0:
                    continue

            # Fill check
            curr_bar = df_15m.iloc[bar_idx]
            if curr_bar['low'] > limit_price:
                stats['expired'] += 1
                continue

            grouped_1m = ad['grouped_1m']
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
                    # Same-candle stop
                    if cl[j] <= stop_level:
                        ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                        ec = fill_price * size * COMMISSION_PCT
                        xc = ep * size * COMMISSION_PCT
                        pnl = (ep - fill_price) * size - ec - xc
                        capital += pnl
                        all_trades.append({
                            'asset': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                            'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[j]),
                            'entry_price': fill_price, 'exit_price': ep,
                            'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                            'risk_used': risk_amount,
                        })
                        stats['same_bar_stops'] += 1
                        consec_stops[asset] += 1
                        daily_losses += abs(pnl)
                        if consec_stops[asset] >= CB_CONSEC_STOP:
                            cooldown_until[asset] = bar_idx + CB_COOLDOWN_BARS
                        cb_limit = capital * scale_risk(effective_base_risk, capital) * CB_DAILY_LOSS_MULT
                        if daily_losses >= cb_limit:
                            trading_halted = True
                        same_exit = True

                    # Scan remaining 1m candles
                    if not same_exit:
                        for k in range(j + 1, len(cl)):
                            if cl[k] <= stop_level:
                                ep = stop_level * (1 - STOP_SLIPPAGE_PCT)
                                ec = fill_price * size * COMMISSION_PCT
                                xc = ep * size * COMMISSION_PCT
                                pnl = (ep - fill_price) * size - ec - xc
                                capital += pnl
                                all_trades.append({
                                    'asset': asset, 'pnl': pnl, 'exit_reason': 'STOP_SAME',
                                    'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': ep,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                    'risk_used': risk_amount,
                                })
                                stats['same_bar_stops'] += 1
                                consec_stops[asset] += 1
                                daily_losses += abs(pnl)
                                if consec_stops[asset] >= CB_CONSEC_STOP:
                                    cooldown_until[asset] = bar_idx + CB_COOLDOWN_BARS
                                cb_limit = capital * scale_risk(effective_base_risk, capital) * CB_DAILY_LOSS_MULT
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
                                all_trades.append({
                                    'asset': asset, 'pnl': pnl, 'exit_reason': 'TARGET_SAME',
                                    'entry_time': pd.Timestamp(ct[j]), 'exit_time': pd.Timestamp(ct[k]),
                                    'entry_price': fill_price, 'exit_price': ep,
                                    'r_multiple': pnl / (risk_dist * size) if risk_dist * size > 0 else 0,
                                    'risk_used': risk_amount,
                                })
                                stats['same_bar_targets'] += 1
                                consec_stops[asset] = 0
                                same_exit = True
                                break

                    if not same_exit:
                        last_ts = ct[-1] if len(ct) > 0 else ct[j]
                        positions[asset] = {
                            'et': pd.Timestamp(ct[j]), 'bar': bar_idx,
                            'ep': fill_price, 'sl': stop_level, 'tp': target_level,
                            'sz': size, 'lts': last_ts, 'risk_used': risk_amount,
                        }

                    filled = True
                    break

            if not filled:
                stats['expired'] += 1

    # Close remaining positions
    for asset in list(positions.keys()):
        pos = positions[asset]
        ad = asset_data[asset]
        df_15m = ad['df_15m']
        bar = df_15m.iloc[-1]
        ep = bar['close']
        ec = pos['ep'] * pos['sz'] * COMMISSION_PCT
        xc = ep * pos['sz'] * COMMISSION_PCT
        pnl = (ep - pos['ep']) * pos['sz'] - ec - xc
        capital += pnl
        rd = pos['ep'] - pos['sl']
        all_trades.append({
            'asset': asset, 'pnl': pnl, 'exit_reason': 'END',
            'entry_time': pos['et'], 'exit_time': df_15m.index[-1],
            'entry_price': pos['ep'], 'exit_price': ep,
            'r_multiple': pnl / (rd * pos['sz']) if rd * pos['sz'] > 0 else 0,
            'risk_used': pos['risk_used'],
        })

    # Compute results
    eq = np.array(equity_values)
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / np.where(peak > 0, peak, 1)
    max_dd = drawdowns.min() * 100

    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    gross_win = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    n_trades = len(all_trades)
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_r = np.mean([t['r_multiple'] for t in all_trades]) if all_trades else 0

    # Per-asset breakdown
    asset_breakdown = {}
    for asset in config['assets']:
        at = [t for t in all_trades if t['asset'] == asset]
        aw = [t for t in at if t['pnl'] > 0]
        al = [t for t in at if t['pnl'] <= 0]
        gw = sum(t['pnl'] for t in aw)
        gl = abs(sum(t['pnl'] for t in al))
        asset_breakdown[asset] = {
            'trades': len(at),
            'wr': len(aw) / len(at) * 100 if at else 0,
            'pf': gw / gl if gl > 0 else float('inf'),
            'pnl': sum(t['pnl'] for t in at),
            'avg_r': np.mean([t['r_multiple'] for t in at]) if at else 0,
        }

    # Yearly breakdown
    yearly = {}
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        df_trades['year'] = pd.to_datetime(df_trades['exit_time'], utc=True).dt.year
        for y, grp in df_trades.groupby('year'):
            w = grp[grp['pnl'] > 0]
            yearly[y] = {
                'trades': len(grp),
                'wr': len(w) / len(grp) * 100 if len(grp) > 0 else 0,
                'pnl': grp['pnl'].sum(),
            }
            # Per-asset yearly
            for asset in config['assets']:
                agrp = grp[grp['asset'] == asset]
                aw = agrp[agrp['pnl'] > 0]
                yearly[y][f'{asset}_trades'] = len(agrp)
                yearly[y][f'{asset}_wr'] = len(aw) / len(agrp) * 100 if len(agrp) > 0 else 0
                yearly[y][f'{asset}_pnl'] = agrp['pnl'].sum()

    return {
        'capital': capital,
        'trades': all_trades,
        'overall': {'trades': n_trades, 'wr': wr, 'pf': pf, 'avg_r': avg_r},
        'max_drawdown': max_dd,
        'asset_breakdown': asset_breakdown,
        'asset_stats': asset_stats,
        'yearly': yearly,
    }


# =====================================================
# MAIN
# =====================================================

print("=" * 120)
print("COMBINED PORTFOLIO BACKTEST: GOLD + SILVER + US500")
print("=" * 120)
print(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1")
print(f"1H filter: Close > EMA(21) | 15m: histogram != red | No squeeze")
print(f"Frictions: 0.1% commission + 0.1% stop slippage | CBs ON")
print(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}")
print()

# Load all asset data
loader = DatabentoMicroFuturesLoader()
asset_data = {}

for asset in ASSETS:
    print(f"Loading {asset}...", end=" ", flush=True)
    t_start = time.time()

    try:
        df_1m_raw = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE)
    except Exception as e:
        print(f"ERROR: {e}")
        continue

    df_15m = resample_ohlcv(df_1m_raw, '15min')
    df_1h = resample_ohlcv(df_1m_raw, '1h')
    print(f"{len(df_1m_raw)} 1m -> {len(df_15m)} 15m, {len(df_1h)} 1h", end=" ", flush=True)

    # 15m indicators
    df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
    df_15m['atr_20'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=20)
    _, momentum_15m, _, _ = calculate_ttm_squeeze_pinescript(
        df_15m['high'], df_15m['low'], df_15m['close'],
        bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
    )
    df_15m['ttm_momentum'] = momentum_15m

    # 1H indicators
    df_1h['ema_21'] = calculate_ema(df_1h['close'], period=21)

    # Pre-align 1H to 15m
    close_1h_arr = df_1h['close'].reindex(df_15m.index, method='ffill').values
    ema_1h_arr = df_1h['ema_21'].reindex(df_15m.index, method='ffill').values

    # Pre-group 1m candles
    period_labels = df_1m_raw.index.floor('15min')
    grouped_1m = {ts: group for ts, group in df_1m_raw.groupby(period_labels)}

    asset_data[asset] = {
        'df_15m': df_15m,
        'df_1h': df_1h,
        'grouped_1m': grouped_1m,
        'close_1h_arr': close_1h_arr,
        'ema_1h_arr': ema_1h_arr,
    }

    elapsed = time.time() - t_start
    print(f"({elapsed:.0f}s)")
    del df_1m_raw

print()

# Run all configs
all_results = {}

for config_name, config in CONFIGS.items():
    # Check all required assets are loaded
    missing = [a for a in config['assets'] if a not in asset_data]
    if missing:
        print(f"  SKIP {config_name}: missing {missing}")
        continue

    print(f"Running {config_name} ({config['label']})...", end=" ", flush=True)
    t_start = time.time()
    result = run_portfolio(config, asset_data)
    elapsed = time.time() - t_start
    all_results[config_name] = result

    o = result['overall']
    pf_str = f"{o['pf']:.2f}" if o['pf'] < 1000 else "inf"
    print(f"{o['trades']} trades, {o['wr']:.1f}% WR, PF {pf_str}, "
          f"DD {result['max_drawdown']:.1f}%, ${result['capital']:,.0f} ({elapsed:.0f}s)")


# =====================================================
# RESULTS
# =====================================================

print()
print()
print("=" * 120)
print("COMBINED PORTFOLIO RESULTS")
print("=" * 120)
print()

# Summary table
header = f"  {'Config':<24s} {'Trades':>7s} {'WR':>7s} {'PF':>8s} {'AvgR':>7s} {'MaxDD':>8s} {'Final':>16s}"
print(header)
print("  " + "-" * 80)

for config_name, result in all_results.items():
    o = result['overall']
    pf_str = f"{o['pf']:>8.2f}" if o['pf'] < 1000 else f"{'inf':>8s}"
    label = CONFIGS[config_name]['label']
    print(f"  {label:<24s} {o['trades']:>7d} {o['wr']:>6.1f}% {pf_str} {o['avg_r']:>+6.2f}R "
          f"{result['max_drawdown']:>7.1f}% ${result['capital']:>15,.0f}")

print()

# Per-asset breakdown for each config
print("=" * 120)
print("PER-ASSET BREAKDOWN")
print("=" * 120)
print()

for config_name, result in all_results.items():
    label = CONFIGS[config_name]['label']
    print(f"  {label}:")
    print(f"    {'Asset':<8s} {'Trades':>7s} {'WR':>7s} {'PF':>8s} {'AvgR':>7s} {'PnL':>16s}")
    print(f"    " + "-" * 55)
    for asset, ab in result['asset_breakdown'].items():
        pf_str = f"{ab['pf']:>8.2f}" if ab['pf'] < 1000 else f"{'inf':>8s}"
        print(f"    {asset:<8s} {ab['trades']:>7d} {ab['wr']:>6.1f}% {pf_str} "
              f"{ab['avg_r']:>+6.2f}R ${ab['pnl']:>15,.0f}")
    print()

# Yearly detail for key configs
print("=" * 120)
print("YEARLY DETAIL")
print("=" * 120)
print()

years = sorted(set(y for r in all_results.values() for y in r['yearly'].keys()))

for config_name in all_results:
    result = all_results[config_name]
    label = CONFIGS[config_name]['label']
    config = CONFIGS[config_name]
    assets_in_config = list(config['assets'].keys())

    print(f"  {label}:")
    # Header
    asset_headers = "".join(f" | {a:^20s}" for a in assets_in_config)
    print(f"    {'Year':>6s} {'Trades':>7s} {'WR':>6s} {'PnL':>14s} {'Cumul':>14s}{asset_headers}")
    print(f"    " + "-" * (55 + 23 * len(assets_in_config)))

    cumul = INITIAL_CAPITAL
    for y in years:
        if y not in result['yearly']:
            continue
        yd = result['yearly'][y]
        cumul += yd['pnl']
        asset_cols = ""
        for a in assets_in_config:
            at = yd.get(f'{a}_trades', 0)
            awr = yd.get(f'{a}_wr', 0)
            if at > 0:
                asset_cols += f" | {at:>4d}t {awr:>5.1f}%      "
            else:
                asset_cols += f" | {'--':>20s}"
        print(f"    {y:>6d} {yd['trades']:>7d} {yd['wr']:>5.1f}% ${yd['pnl']:>13,.0f} ${cumul:>13,.0f}{asset_cols}")
    print()


# Save results
results_file = 'results/1m_combined_portfolio_results.txt'
with open(results_file, 'w') as f:
    f.write("=" * 120 + "\n")
    f.write("COMBINED PORTFOLIO BACKTEST: GOLD + SILVER + US500\n")
    f.write("=" * 120 + "\n\n")
    f.write(f"Entry: EMA(21) | Stop: EMA-2ATR | Target: EMA+2ATR | R:R 1:1\n")
    f.write(f"1H filter: Close > EMA(21) | 15m: histogram != red | No squeeze\n")
    f.write(f"Frictions: 0.1% commission + 0.1% stop slippage | CBs ON\n")
    f.write(f"Capital: ${INITIAL_CAPITAL:,} | Period: {START_DATE} to {END_DATE}\n\n")

    # Summary
    f.write(f"{'Config':<24s} {'Trades':>7s} {'WR':>7s} {'PF':>8s} {'AvgR':>7s} {'MaxDD':>8s} {'Final':>16s}\n")
    f.write("-" * 80 + "\n")
    for config_name, result in all_results.items():
        o = result['overall']
        pf_str = f"{o['pf']:>8.2f}" if o['pf'] < 1000 else f"{'inf':>8s}"
        label = CONFIGS[config_name]['label']
        f.write(f"{label:<24s} {o['trades']:>7d} {o['wr']:>6.1f}% {pf_str} {o['avg_r']:>+6.2f}R "
                f"{result['max_drawdown']:>7.1f}% ${result['capital']:>15,.0f}\n")
    f.write("\n")

    # Per-asset
    for config_name, result in all_results.items():
        label = CONFIGS[config_name]['label']
        f.write(f"\n{label} — Per-asset:\n")
        for asset, ab in result['asset_breakdown'].items():
            pf_str = f"{ab['pf']:.2f}" if ab['pf'] < 1000 else "inf"
            f.write(f"  {asset:<8s} {ab['trades']:>7d} trades, {ab['wr']:>5.1f}% WR, "
                    f"PF {pf_str}, ${ab['pnl']:>13,.0f}\n")

    # Yearly
    f.write("\n\nYEARLY DETAIL:\n")
    f.write("-" * 80 + "\n")
    for config_name in all_results:
        result = all_results[config_name]
        label = CONFIGS[config_name]['label']
        f.write(f"\n{label}:\n")
        cumul = INITIAL_CAPITAL
        for y in years:
            if y not in result['yearly']:
                continue
            yd = result['yearly'][y]
            cumul += yd['pnl']
            f.write(f"  {y}: {yd['trades']:>5d} trades, {yd['wr']:>5.1f}% WR, "
                    f"${yd['pnl']:>14,.0f}, cumul ${cumul:>14,.0f}\n")

    f.write("\n" + "=" * 120 + "\n")

# Save trade logs for key configs
for config_name in ['all_w1', 'gold_silver', 'all_delayed_50k', 'all_step_2k', 'all_step_5k']:
    if config_name not in all_results:
        continue
    trades = all_results[config_name]['trades']
    if trades:
        df_out = pd.DataFrame(trades)
        csv_file = f'results/1m_combined_{config_name}_trades.csv'
        df_out.to_csv(csv_file, index=False)
        print(f"Trade log saved: {csv_file}")

print(f"\nResults saved to {results_file}")
print("Done.")
