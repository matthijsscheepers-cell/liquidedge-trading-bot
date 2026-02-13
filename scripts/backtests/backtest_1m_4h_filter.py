"""
1-Minute Execution Backtest: 4H-Filtered Limit Order

Adds 4H momentum filter on top of 1H filter.
Tests whether higher timeframe trend alignment improves expectancy.

Signal conditions (ALL must be true):
- 4H TTM momentum >= 0 (bullish on highest timeframe)
- 1H TTM momentum >= 0 (bullish on anchor timeframe)
- 15m squeeze ON (volatility compression)
- 15m histogram color: yellow, light-blue, or dark-blue (NOT red)

Entry: persistent limit at EMA(21) - 1.0 ATR
Stop:  EMA(21) - 2.0 ATR
Target: EMA(21) + 2.0 ATR
R:R = 3:1

Risk: pure 2% of capital (NO progressive risk cap)

Histogram color definitions:
- light_blue: momentum > 0 AND rising (> prev bar)
- dark_blue:  momentum > 0 AND falling (<= prev bar)
- yellow:     momentum <= 0 BUT rising (> prev bar)
- red:        momentum <= 0 AND falling (<= prev bar)

Invalidation (cancel pending order):
  1H < 0 OR 4H < 0 OR squeeze off OR color = red

Capital levels: $300, $1,000, $5,000
Periods: Full 2010-2026 and Recent 2020-2026
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
RISK_PER_TRADE = 0.02  # Pure 2%, no progressive cap
LEVERAGE = 20
MAX_POSITIONS = 2
MIN_CAPITAL = 1.0

# Circuit breakers
CB_CONSECUTIVE_STOP_LIMIT = 2
CB_COOLDOWN_BARS = 16  # 4 hours at 15min
CB_DAILY_LOSS_MULTIPLIER = 3

ASSETS = ['GOLD', 'SILVER']

# Entry/Stop/Target (ATR units relative to EMA)
ENTRY_ATR = -1.0   # EMA - 1.0 ATR
STOP_ATR = -2.0    # EMA - 2.0 ATR
TARGET_ATR = 2.0   # EMA + 2.0 ATR

CAPITAL_LEVELS = [300, 1_000, 5_000]


def resample_ohlcv(df, freq):
    return df.resample(freq).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()


def get_histogram_color(mom_curr, mom_prev):
    """
    Determine histogram color based on momentum value and direction.
    - light_blue: mom > 0 and rising (> prev)
    - dark_blue:  mom > 0 and falling (<= prev)
    - yellow:     mom <= 0 but rising (> prev)
    - red:        mom <= 0 and falling (<= prev)
    """
    if pd.isna(mom_curr) or pd.isna(mom_prev):
        return 'red'  # conservative: treat unknown as red

    rising = mom_curr > mom_prev
    positive = mom_curr > 0

    if positive and rising:
        return 'light_blue'
    elif positive and not rising:
        return 'dark_blue'
    elif not positive and rising:
        return 'yellow'
    else:
        return 'red'


def get_daily_risk_limit(capital):
    """Circuit breaker daily loss limit: 3x single trade risk."""
    return capital * RISK_PER_TRADE * CB_DAILY_LOSS_MULTIPLIER


# =====================================================
# DATA LOADING
# =====================================================

print("=" * 90)
print("1-MINUTE EXECUTION BACKTEST — 4H-FILTERED LIMIT ORDER")
print("=" * 90)
print(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Stop: EMA - {abs(STOP_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR")
print(f"Risk: pure {RISK_PER_TRADE*100:.0f}% (no cap)")
print(f"Filters: 4H mom >= 0, 1H mom >= 0, 15m squeeze ON, color != red")
print(f"Capital levels: {', '.join(f'${c:,}' for c in CAPITAL_LEVELS)}")
print(f"Period: {START_DATE} to {END_DATE}")
print()
print("Loading data (1-minute resolution, resampling in-script)...")
print()

loader = DatabentoMicroFuturesLoader()

all_data_15m = {}
all_data_1h = {}
all_data_4h = {}
grouped_1m = {}

t_start = time.time()

for asset in ASSETS:
    print(f"  {asset}...", end=" ", flush=True)
    df_1m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE)
    print(f"{len(df_1m)} 1m bars", end=" -> ", flush=True)

    df_15m = resample_ohlcv(df_1m, '15min')
    df_1h = resample_ohlcv(df_1m, '1h')
    df_4h = resample_ohlcv(df_1m, '4h')
    print(f"{len(df_15m)} 15m, {len(df_1h)} 1h, {len(df_4h)} 4h bars", end=" ", flush=True)

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

    # 4H TTM momentum
    _, momentum_4h, _, _ = calculate_ttm_squeeze_pinescript(
        df_4h['high'], df_4h['low'], df_4h['close'],
        bb_period=20, bb_std=2.0, kc_period=20, momentum_period=20
    )
    df_4h['ttm_momentum'] = momentum_4h

    # Pre-group 1m candles by 15m period
    period_labels = df_1m.index.floor('15min')
    grouped = {ts: group for ts, group in df_1m.groupby(period_labels)}

    all_data_15m[asset] = df_15m
    all_data_1h[asset] = df_1h
    all_data_4h[asset] = df_4h
    grouped_1m[asset] = grouped
    del df_1m
    print("OK")

t_load = time.time() - t_start
print(f"\nData loaded in {t_load:.0f}s")

# Common index for 15m bars
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
    """Run the 4H-filtered limit order backtest."""
    capital = initial_capital
    positions = {}
    pending_orders = {}
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
        'orders_cancelled_1h': 0,
        'orders_cancelled_4h': 0,
        'orders_cancelled_squeeze': 0,
        'orders_cancelled_color': 0,
        'orders_filled': 0,
        'orders_persisted': 0,
        'exits_stop': 0,
        'exits_target': 0,
        'same_bar_stops': 0,
        'same_bar_targets': 0,
        'color_counts': {'light_blue': 0, 'dark_blue': 0, 'yellow': 0, 'red': 0},
    }

    diag = {
        'positions_opened': 0,
        'positions_closed': 0,
        'max_simultaneous_open': 0,
    }

    empty_df = pd.DataFrame()

    for i in range(start_bar, len(common_index)):
        timestamp = common_index[i]

        # --- Hard capital floor ---
        if capital <= MIN_CAPITAL:
            if not account_blown:
                account_blown = True
                account_blown_ts = timestamp
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
                # Fallback: use 15m bar
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
                    if daily_losses >= get_daily_risk_limit(capital):
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

                # Stop check first (worst case)
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
                    if daily_losses >= get_daily_risk_limit(capital):
                        trading_halted = True
                    break

                # Target check
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
                # No exit: update last processed timestamp
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

            df_15m_asset = all_data_15m[asset]
            df_1h_asset = all_data_1h[asset]
            df_4h_asset = all_data_4h[asset]

            prev_bar = df_15m_asset.iloc[i - 1]
            prev_timestamp = common_index[i - 1]

            ema = prev_bar['ema_21']
            atr = prev_bar['atr_20']

            if pd.isna(ema) or pd.isna(atr) or atr <= 0:
                pending_orders.pop(asset, None)
                continue

            # --- 1H momentum filter (>= 0) ---
            bars_1h_avail = df_1h_asset[df_1h_asset.index <= prev_timestamp]
            if len(bars_1h_avail) < 20:
                pending_orders.pop(asset, None)
                continue
            prev_1h = bars_1h_avail.iloc[-1]

            if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] < 0:
                if asset in pending_orders:
                    stats['orders_cancelled_1h'] += 1
                pending_orders.pop(asset, None)
                continue

            # --- 4H momentum filter (>= 0) ---
            bars_4h_avail = df_4h_asset[df_4h_asset.index <= prev_timestamp]
            if len(bars_4h_avail) < 20:
                pending_orders.pop(asset, None)
                continue
            prev_4h = bars_4h_avail.iloc[-1]

            if pd.isna(prev_4h['ttm_momentum']) or prev_4h['ttm_momentum'] < 0:
                if asset in pending_orders:
                    stats['orders_cancelled_4h'] += 1
                pending_orders.pop(asset, None)
                continue

            # --- 15m squeeze filter ---
            squeeze_active = not pd.isna(prev_bar['squeeze_on']) and prev_bar['squeeze_on']
            if not squeeze_active:
                if asset in pending_orders:
                    stats['orders_cancelled_squeeze'] += 1
                pending_orders.pop(asset, None)
                continue

            # --- 15m histogram color filter ---
            mom_curr = prev_bar['ttm_momentum']
            if i >= 2:
                mom_prev = df_15m_asset.iloc[i - 2]['ttm_momentum']
            else:
                mom_prev = np.nan

            color = get_histogram_color(mom_curr, mom_prev)
            stats['color_counts'][color] += 1

            if color == 'red':
                if asset in pending_orders:
                    stats['orders_cancelled_color'] += 1
                pending_orders.pop(asset, None)
                continue

            # --- All filters passed: place/update limit order ---
            limit_price = ema + ENTRY_ATR * atr    # EMA - 1.0 * ATR
            stop_level = ema + STOP_ATR * atr       # EMA - 2.0 * ATR
            target_level = ema + TARGET_ATR * atr   # EMA + 2.0 * ATR

            risk_dist = limit_price - stop_level    # 1.0 * ATR
            reward = target_level - limit_price     # 3.0 * ATR

            if risk_dist > 0 and reward > 0:
                if asset not in pending_orders:
                    stats['orders_placed'] += 1
                    stats['signals'] += 1

                pending_orders[asset] = {
                    'entry_level': limit_price,
                    'stop_level': stop_level,
                    'target_level': target_level,
                    'signal_bar': i - 1,
                    'color': color,
                }

            # --- Fill check for pending limit ---
            if asset not in pending_orders:
                continue

            order = pending_orders[asset]

            # Pre-filter: 15m bar low must reach limit price
            curr_bar = df_15m_asset.iloc[i]
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

                    # R:R check
                    reward_dist = target_level - fill_price
                    if reward_dist / risk_dist < 1.5:
                        break

                    # Position sizing: pure 2% risk
                    risk_amount = capital * RISK_PER_TRADE
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

                    # Same-candle stop check (conservative: stop wins)
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
                        if daily_losses >= get_daily_risk_limit(capital):
                            trading_halted = True
                        same_bar_exit = True

                    # Remaining candles after fill
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
                                if daily_losses >= get_daily_risk_limit(capital):
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
        'initial_capital': initial_capital,
    }


# =====================================================
# RUN ALL CAPITAL LEVELS × PERIODS
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
    for cap in CAPITAL_LEVELS:
        print(f"  ${cap:,}...", end=" ", flush=True)
        t_run = time.time()
        r = run_backtest(initial_capital=cap, start_bar=start_bar)
        elapsed = time.time() - t_run

        r['runtime'] = elapsed
        r['label'] = f"${cap:,}"
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
print("RESULTS — 4H-FILTERED LIMIT ORDER (1-minute execution)")
print("=" * 90)
print(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Stop: EMA - {abs(STOP_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR")
print(f"Filters: 4H >= 0, 1H >= 0, 15m squeeze ON, color != red")
print(f"Risk: pure {RISK_PER_TRADE*100:.0f}% per trade (no cap)")
print(f"Total runtime: {t_total_elapsed:.0f}s")
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    print()
    print(f"  {'Capital':<10s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
          f"{'MaxDD':>8s} {'Fill%':>6s} {'Final':>12s} {'Blown':>12s}")
    print("  " + "-" * 90)

    for r in results:
        blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
        print(f"  {r['label']:<10s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
              f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
              f"{r['max_drawdown']:>7.1f}% {r['fill_rate']:>5.1f}% "
              f"${r['capital']:>11,.0f} {blown:>12s}")
    print()

# Per-asset breakdown
print("=" * 90)
print("PER-ASSET BREAKDOWN")
print("=" * 90)
print()

for period_name, results in all_results.items():
    print(f"--- {period_name} ---")
    for r in results:
        if r['asset_stats']:
            print(f"  {r['label']}:")
            for a, s in sorted(r['asset_stats'].items()):
                print(f"    {a:8s}: {s['trades']:4d} trades, {s['wr']:5.1f}% WR, "
                      f"${s['pnl']:>12,.2f}, avgR {s['avg_r']:>+.2f}")
    print()

# Execution funnel
print("=" * 90)
print("EXECUTION FUNNEL & FILTER STATS")
print("=" * 90)
print()

for period_name, results in all_results.items():
    # Use the largest capital level for most representative stats
    r = results[-1]  # $5K
    s = r['stats']
    print(f"--- {period_name} (${CAPITAL_LEVELS[-1]:,} capital) ---")
    print(f"  Signals generated:       {s['signals']:>8d}")
    print(f"  Orders placed:           {s['orders_placed']:>8d}")
    print(f"  Orders filled:           {s['orders_filled']:>8d}")
    print(f"  Orders persisted:        {s['orders_persisted']:>8d}")
    print(f"  Cancelled (1H < 0):      {s['orders_cancelled_1h']:>8d}")
    print(f"  Cancelled (4H < 0):      {s['orders_cancelled_4h']:>8d}")
    print(f"  Cancelled (squeeze off):  {s['orders_cancelled_squeeze']:>8d}")
    print(f"  Cancelled (color=red):   {s['orders_cancelled_color']:>8d}")
    print(f"  Fill rate:               {r['fill_rate']:>7.1f}%")
    print()
    print(f"  Exit breakdown:")
    print(f"    Stop:                  {s['exits_stop']:>8d}")
    print(f"    Target:                {s['exits_target']:>8d}")
    print(f"    Same-bar stop:         {s['same_bar_stops']:>8d}")
    print(f"    Same-bar target:       {s['same_bar_targets']:>8d}")
    print()
    total_colors = sum(s['color_counts'].values())
    print(f"  Histogram color at signal ({total_colors} bars checked):")
    for c in ['light_blue', 'dark_blue', 'yellow', 'red']:
        n = s['color_counts'][c]
        pct = n / total_colors * 100 if total_colors > 0 else 0
        print(f"    {c:<12s}: {n:>8d} ({pct:>5.1f}%)")
    print()

# R-multiple distribution
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION")
print("=" * 90)
print()

for period_name, results in all_results.items():
    r = results[-1]  # $5K
    if len(r['trades_df']) > 0 and 'r_multiple' in r['trades_df'].columns:
        tdf = r['trades_df']
        print(f"  {period_name} — {r['label']} | {len(tdf)} trades")
        print()
        bins = [(-10, -1.5), (-1.5, -1.0), (-1.0, -0.5), (-0.5, 0),
                (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 10)]
        for lo, hi in bins:
            n = len(tdf[(tdf['r_multiple'] >= lo) & (tdf['r_multiple'] < hi)])
            pct = n / len(tdf) * 100 if len(tdf) > 0 else 0
            bar = '#' * int(pct)
            print(f"    {lo:>+5.1f}R to {hi:>+5.1f}R: {n:>5d} ({pct:>5.1f}%) {bar}")
        print()

# Comparison with previous strategies
print("=" * 90)
print("COMPARISON: ALL 1-MINUTE STRATEGIES TESTED")
print("=" * 90)
print()
print(f"  {'Strategy':<40s} {'WR':>6s} {'PF':>6s} {'AvgR':>7s} {'Notes'}")
print("  " + "-" * 75)
print(f"  {'Limit -0.5 S-1.5 T+1.5 (orig)':<40s} {'31%':>6s} {'0.20':>6s} {'-1.5R':>7s} {'blown'}")
print(f"  {'Buy stop continuation 2R':<40s} {'18%':>6s} {'0.06':>6s} {'-2.0R':>7s} {'blown'}")
print(f"  {'RTM limit -0.75 partials':<40s} {'13%':>6s} {'0.02':>6s} {'-2.0R':>7s} {'blown'}")
print(f"  {'Structure breakout 1.5R':<40s} {'32%':>6s} {'0.25':>6s} {'-1.3R':>7s} {'blown'}")
print(f"  {'Lim-1 S-2 T+2 pos (full)':<40s} {'34%':>6s} {'0.16':>6s} {'-1.0R':>7s} {'blown'}")
print(f"  {'Lim-1 S-2 T+2 pos (2020+)':<40s} {'43%':>6s} {'1.01':>6s} {'+0.05R':>7s} {'break-even'}")

# Add current results
for period_name, results in all_results.items():
    r = results[-1]  # $5K
    tag = "full" if "Full" in period_name else "2020+"
    blown = "blown" if r['account_blown_ts'] else "survived"
    label = f"4H-filt Lim-1 S-2 T+2 ({tag})"
    print(f"  {label:<40s} "
          f"{r['win_rate']:>5.0f}% {r['profit_factor']:>6.2f} "
          f"{r['avg_r']:>+6.2f}R {blown}")
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
        print(f"  {period_name[:8]} {r['label']:<10s} open==close: {ok} "
              f"({d['positions_opened']}/{d['positions_closed']})")
print()

# Save results
output_file = 'results/1m_4h_filter_results.txt'
with open(output_file, 'w') as f:
    f.write("=" * 90 + "\n")
    f.write("1-MINUTE EXECUTION — 4H-FILTERED LIMIT ORDER\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Entry: EMA - {abs(ENTRY_ATR):.1f} ATR | Stop: EMA - {abs(STOP_ATR):.1f} ATR | Target: EMA + {TARGET_ATR:.1f} ATR\n")
    f.write(f"Filters: 4H >= 0, 1H >= 0, 15m squeeze ON, color != red\n")
    f.write(f"Risk: pure {RISK_PER_TRADE*100:.0f}% per trade | Runtime: {t_total_elapsed:.0f}s\n\n")

    for period_name, results in all_results.items():
        f.write(f"\n{period_name}\n")
        f.write("-" * 50 + "\n\n")

        f.write(f"{'Capital':<10s} {'Trades':>7s} {'WR':>7s} {'PF':>6s} {'AvgR':>7s} "
                f"{'MaxDD':>8s} {'Final':>12s} {'Blown':>12s}\n")
        f.write("-" * 85 + "\n")

        for r in results:
            blown = str(r['account_blown_ts'])[:10] if r['account_blown_ts'] else '-'
            f.write(f"{r['label']:<10s} {r['trades']:>7d} {r['win_rate']:>6.1f}% "
                    f"{r['profit_factor']:>6.2f} {r['avg_r']:>+6.2f}R "
                    f"{r['max_drawdown']:>7.1f}% "
                    f"${r['capital']:>11,.0f} {blown:>12s}\n")

        f.write("\nPer-asset:\n")
        for r in results:
            if r['asset_stats']:
                f.write(f"\n  {r['label']}:\n")
                for a, s in sorted(r['asset_stats'].items()):
                    f.write(f"    {a:8s}: {s['trades']:4d} trades, {s['wr']:5.1f}% WR, "
                            f"${s['pnl']:>12,.2f}, avgR {s['avg_r']:>+.2f}\n")
        f.write("\n")

    f.write("=" * 90 + "\n")

print(f"Results saved: {output_file}")

# Save trade logs
for period_name, results in all_results.items():
    r = results[-1]  # $5K
    if len(r['trades_df']) > 0:
        tag = "full" if "Full" in period_name else "2020"
        fname = f'results/1m_4h_filter_trades_{tag}.csv'
        r['trades_df'].to_csv(fname, index=False)
        print(f"Trade log ({period_name}, {r['label']}): {fname}")

print()
print("=" * 90)
