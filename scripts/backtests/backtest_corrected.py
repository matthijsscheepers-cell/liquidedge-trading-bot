"""
CORRECTED Backtest: Multi-Asset TTM Pullback Strategy

Fixes vs original backtest_user_strategy_2yr.py:
1. NO LOOK-AHEAD BIAS: Signal from bar i-1 close, fill check on bar i
2. SLIPPAGE APPLIED: Stop-loss exits include slippage (stop-market orders)
3. COMMISSION ON BOTH SIDES: Entry + exit commission
4. SAME-BAR STOP CHECK: If entry bar also hits stop, count as loss

Strategy:
- 1H chart: Trend confirmation (TTM momentum > 0)
- 15min chart: Squeeze active + price pulls back to -1 ATR
- Entry: -1 ATR below 21-EMA (limit order)
- Stop: -2.1 ATR below 21-EMA (stop-market order)
- Target: +2 ATR above 21-EMA (limit order)
- LONG ONLY
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
import pandas as pd
import numpy as np

print("=" * 70)
print("CORRECTED BACKTEST - TTM PULLBACK STRATEGY")
print("=" * 70)
print()
print("Fixes applied:")
print("  1. No look-ahead bias (signal bar i-1, fill bar i)")
print("  2. Slippage on stop-loss exits (stop-market orders)")
print("  3. Commission on BOTH entry and exit")
print("  4. Same-bar stop check (entry + stop on same bar = loss)")
print()

# Config
START_DATE = '2010-09-12'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 300.0
COMMISSION_PCT = 0.001      # 0.1% per side (entry + exit)
STOP_SLIPPAGE_PCT = 0.001   # 0.1% slippage on stop-market orders
RISK_PER_TRADE = 0.02       # 2% risk per trade
RISK_CAP = 50.0             # $50 max risk per trade (matches live bot)
LEVERAGE = 20               # CFD leverage 1:20
MAX_POSITIONS = 4

# Strategy parameters (must match ttm_pullback.py exactly)
ENTRY_ATR = -1.0            # Entry at EMA - 1*ATR
STOP_ATR = -2.1             # Stop at EMA - 2.1*ATR
TARGET_ATR = 2.0            # Target at EMA + 2*ATR
ENTRY_TOLERANCE = 0.5       # Accept within 0.5 ATR of entry level

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print(f"Period: {START_DATE} to {END_DATE} (~15.5 years)")
print(f"Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Risk cap: ${RISK_CAP:.0f}/trade")
print(f"Commission: {COMMISSION_PCT*100:.1f}% per side")
print(f"Stop slippage: {STOP_SLIPPAGE_PCT*100:.1f}%")
print(f"Assets: {', '.join(ASSETS)}")
print()

# Load data
loader = DatabentoMicroFuturesLoader()

print("Loading data...")
all_data_15m = {}
all_data_1h = {}

for asset in ASSETS:
    print(f"  {asset}...", end=" ", flush=True)
    try:
        df_15m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
        df_1h = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='1h')

        # Add indicators to 15min
        df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
        df_15m['atr_20'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=20)

        squeeze_on, momentum, color, intensity = calculate_ttm_squeeze_pinescript(
            df_15m['high'], df_15m['low'], df_15m['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20,
            momentum_period=20
        )
        df_15m['squeeze_on'] = squeeze_on
        df_15m['ttm_momentum'] = momentum

        # Add indicators to 1H
        _, momentum_1h, _, _ = calculate_ttm_squeeze_pinescript(
            df_1h['high'], df_1h['low'], df_1h['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20,
            momentum_period=20
        )
        df_1h['ttm_momentum'] = momentum_1h

        all_data_15m[asset] = df_15m
        all_data_1h[asset] = df_1h

        print(f"OK ({len(df_15m)} 15m, {len(df_1h)} 1H)")
    except Exception as e:
        print(f"ERROR: {e}")

print()

if not all_data_15m:
    print("No data loaded!")
    sys.exit(1)

# Align to common timestamps
print("Aligning timestamps...")
common_index = None
for asset in all_data_15m.keys():
    if common_index is None:
        common_index = all_data_15m[asset].index
    else:
        common_index = common_index.intersection(all_data_15m[asset].index)

print(f"Common 15min bars: {len(common_index)}")
print()

for asset in all_data_15m.keys():
    all_data_15m[asset] = all_data_15m[asset].loc[common_index]

# =====================================================
# BACKTEST LOOP
# =====================================================
print("=" * 70)
print("RUNNING BACKTEST")
print("=" * 70)
print()

capital = INITIAL_CAPITAL
positions = {}      # {asset: position_dict}
trades = []
equity_curve = []

# Start at bar 201 so we have bar 200 as first signal bar
for i in range(201, len(common_index)):
    timestamp = common_index[i]

    # Track equity
    equity = capital
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[i]
        unrealized = (bar['close'] - pos['entry_price']) * pos['size']
        equity += unrealized

    equity_curve.append({
        'timestamp': timestamp,
        'equity': equity,
        'capital': capital,
        'positions': len(positions),
    })

    # === 1. POSITION MANAGEMENT (check stops/targets on bar i) ===
    positions_to_close = []

    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[i]
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        target = pos['target']
        size = pos['size']

        # Check stop loss (stop-market order: include slippage)
        if bar['low'] <= stop_loss:
            exit_price = stop_loss * (1 - STOP_SLIPPAGE_PCT)  # Slip worse
            entry_commission = entry_price * size * COMMISSION_PCT
            exit_commission = exit_price * size * COMMISSION_PCT
            pnl = (exit_price - entry_price) * size - entry_commission - exit_commission
            capital += pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'symbol': asset,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'STOP',
                'bars_held': i - pos['entry_bar'],
            })
            positions_to_close.append(asset)
            continue

        # Check target (limit order: fills at exact target)
        if bar['high'] >= target:
            exit_price = target
            entry_commission = entry_price * size * COMMISSION_PCT
            exit_commission = exit_price * size * COMMISSION_PCT
            pnl = (exit_price - entry_price) * size - entry_commission - exit_commission
            capital += pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'symbol': asset,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'TARGET',
                'bars_held': i - pos['entry_bar'],
            })
            positions_to_close.append(asset)

    for asset in positions_to_close:
        del positions[asset]

    # === 2. ENTRY SCAN (signal from bar i-1, fill on bar i) ===
    if len(positions) >= MAX_POSITIONS:
        continue

    for asset in ASSETS:
        if asset in positions:
            continue
        if len(positions) >= MAX_POSITIONS:
            break

        df_15m = all_data_15m[asset]
        df_1h = all_data_1h[asset]

        # --- SIGNAL CHECK: Use bar i-1 (last CLOSED bar) ---
        prev_bar = df_15m.iloc[i - 1]

        # Find 1H bar that was closed at bar i-1's timestamp
        prev_timestamp = common_index[i - 1]
        bars_1h_available = df_1h[df_1h.index <= prev_timestamp]
        if len(bars_1h_available) < 20:
            continue
        prev_1h = bars_1h_available.iloc[-1]

        # Condition 1: 1H momentum bullish (positive)
        if pd.isna(prev_1h['ttm_momentum']) or prev_1h['ttm_momentum'] <= 0:
            continue

        # Condition 2: 15M squeeze active
        if pd.isna(prev_bar['squeeze_on']) or not prev_bar['squeeze_on']:
            continue

        # Condition 3: Calculate entry levels from bar i-1's indicators
        ema_21 = prev_bar['ema_21']
        atr = prev_bar['atr_20']

        if pd.isna(ema_21) or pd.isna(atr) or atr <= 0:
            continue

        entry_level = ema_21 + (ENTRY_ATR * atr)    # EMA - 1*ATR
        stop_level = ema_21 + (STOP_ATR * atr)       # EMA - 2.1*ATR
        target_level = ema_21 + (TARGET_ATR * atr)    # EMA + 2*ATR

        risk_per_share = entry_level - stop_level
        if risk_per_share <= 0:
            continue

        # --- FILL CHECK: Use bar i (current bar) ---
        curr_bar = df_15m.iloc[i]
        bar_low = curr_bar['low']

        # Check if bar i's low reaches within tolerance of entry level
        distance_atr = (bar_low - entry_level) / atr
        if distance_atr > ENTRY_TOLERANCE:    # Didn't pull back enough
            continue
        if distance_atr < -ENTRY_TOLERANCE:   # Too far below (crash)
            continue

        # LIMIT ORDER FILLS at entry_level (no slippage on limit fills)
        fill_price = entry_level

        # Position sizing
        risk_amount = min(capital * RISK_PER_TRADE, RISK_CAP)
        position_size = risk_amount / risk_per_share
        position_value = position_size * fill_price
        margin_required = position_value / LEVERAGE

        if margin_required > capital:
            continue

        # SAME-BAR STOP CHECK: if bar i also hits the stop, it's a loss
        if bar_low <= stop_level:
            # Entry AND stop hit on same bar → worst case: stopped out
            exit_price = stop_level * (1 - STOP_SLIPPAGE_PCT)
            entry_commission = fill_price * position_size * COMMISSION_PCT
            exit_commission = exit_price * position_size * COMMISSION_PCT
            pnl = (exit_price - fill_price) * position_size - entry_commission - exit_commission
            capital += pnl

            trades.append({
                'entry_time': timestamp,
                'exit_time': timestamp,
                'symbol': asset,
                'entry_price': fill_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'STOP_SAME_BAR',
                'bars_held': 0,
            })
            continue

        # Valid entry - open position
        positions[asset] = {
            'entry_time': timestamp,
            'entry_bar': i,
            'direction': 'LONG',
            'entry_price': fill_price,
            'stop_loss': stop_level,
            'target': target_level,
            'size': position_size,
        }
        break  # Only 1 new entry per bar

    # Progress
    if i % 5000 == 0:
        progress = (i / len(common_index)) * 100
        print(f"Progress: {progress:.1f}%, Capital: ${capital:,.2f}, Trades: {len(trades)}")

# Close remaining positions at market
for asset, pos in list(positions.items()):
    bar = all_data_15m[asset].iloc[-1]
    exit_price = bar['close']
    size = pos['size']
    entry_price = pos['entry_price']
    entry_commission = entry_price * size * COMMISSION_PCT
    exit_commission = exit_price * size * COMMISSION_PCT
    pnl = (exit_price - entry_price) * size - entry_commission - exit_commission
    capital += pnl

    trades.append({
        'entry_time': pos['entry_time'],
        'exit_time': common_index[-1],
        'symbol': asset,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'exit_reason': 'END',
        'bars_held': len(common_index) - 1 - pos['entry_bar'],
    })

# =====================================================
# RESULTS
# =====================================================
print()
print("=" * 70)
print("RESULTS - CORRECTED BACKTEST")
print("=" * 70)
print()

trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

print(f"Total trades: {len(trades_df)}")
print(f"Final capital: ${capital:,.2f}")
print()

if len(trades_df) > 0:
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("PERFORMANCE METRICS")
    print("-" * 70)
    print(f"Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Capital:       ${capital:,.2f}")
    print(f"Total Return:        {total_return:.2f}%")
    print()
    print(f"Total Trades:        {len(trades_df)}")
    print(f"Win Rate:            {len(wins)/len(trades_df)*100:.1f}%")

    if len(wins) > 0:
        print(f"Avg Win:             ${wins['pnl'].mean():,.2f}")
    if len(losses) > 0:
        print(f"Avg Loss:            ${losses['pnl'].mean():,.2f}")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"Profit Factor:       {profit_factor:.2f}")

    avg_rr = wins['pnl'].mean() / abs(losses['pnl'].mean()) if len(losses) > 0 and len(wins) > 0 else 0
    print(f"Avg R:R:             {avg_rr:.2f}:1")
    print()

    # Drawdown analysis
    eq = equity_df['equity']
    running_max = eq.expanding().max()
    drawdown = (eq - running_max) / running_max * 100
    max_dd = drawdown.min()
    print(f"Max Drawdown:        {max_dd:.2f}%")
    print()

    # Exit reason breakdown
    print("EXIT REASONS")
    print("-" * 70)
    for reason in sorted(trades_df['exit_reason'].unique()):
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        reason_wins = reason_trades[reason_trades['pnl'] > 0]
        print(f"  {reason}: {len(reason_trades)} trades ({len(reason_wins)} wins, {len(reason_trades)-len(reason_wins)} losses)")
    print()

    # Per asset breakdown
    print("PER ASSET")
    print("-" * 70)
    for asset in ASSETS:
        at = trades_df[trades_df['symbol'] == asset]
        if len(at) > 0:
            aw = at[at['pnl'] > 0]
            wr = len(aw) / len(at) * 100
            total_pnl = at['pnl'].sum()
            avg_held = at['bars_held'].mean()
            print(f"  {asset:8s}: {len(at):5d} trades, {wr:5.1f}% WR, ${total_pnl:>12,.2f} P&L, avg {avg_held:.0f} bars held")
        else:
            print(f"  {asset:8s}: 0 trades")
    print()

    # Compare with original results
    print("=" * 70)
    print("COMPARISON WITH ORIGINAL (BUGGY) BACKTEST")
    print("=" * 70)
    print()
    print(f"{'Metric':<25s} {'Original':>15s} {'Corrected':>15s}")
    print("-" * 55)
    print(f"{'Win Rate':<25s} {'77.0%':>15s} {f'{len(wins)/len(trades_df)*100:.1f}%':>15s}")
    print(f"{'Total Trades':<25s} {'10,895':>15s} {f'{len(trades_df):,}':>15s}")
    print(f"{'Final Capital':<25s} {'$850,051':>15s} {f'${capital:,.0f}':>15s}")
    if len(wins) > 0 and len(losses) > 0:
        print(f"{'Profit Factor':<25s} {'5.99':>15s} {f'{profit_factor:.2f}':>15s}")
    print(f"{'Max Drawdown':<25s} {'N/A':>15s} {f'{max_dd:.1f}%':>15s}")
    print()

    # Save results
    trades_df.to_csv('results/corrected_backtest_trades.csv', index=False)
    equity_df.to_csv('results/corrected_backtest_equity.csv', index=False)
    print("Trade log saved to results/corrected_backtest_trades.csv")
    print("Equity curve saved to results/corrected_backtest_equity.csv")

else:
    print("NO TRADES GENERATED")
    print()
    print("This means the strategy conditions were too strict with the")
    print("look-ahead bias fix. The original 77% win rate relied on")
    print("using future bar information.")

print()
print("=" * 70)
