"""
Backtest: Kelly Criterion Position Sizing

Tests mathematically optimal position sizing using Kelly Criterion:
Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win

Uses rolling 100-trade window to calculate Kelly percentage.
For first 100 trades, uses baseline 2% risk.
Kelly fraction is capped at 50% (half-Kelly for safety).
Risk is still capped at $50 max to prevent explosive growth.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
from src.strategies.ttm_pullback import TTMSqueezePullbackStrategy
import pandas as pd
import numpy as np

print("=" * 70)
print("OPTIMIZATION 3: KELLY CRITERION POSITION SIZING")
print("=" * 70)
print()

# Config
START_DATE = '2010-09-12'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 300.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001
LEVERAGE = 20
MAX_POSITIONS = 4
RISK_CAP = 50.0  # Fixed cap at $50
KELLY_LOOKBACK = 100  # Rolling window for Kelly calculation
KELLY_FRACTION = 0.5  # Use half-Kelly for safety

def calculate_kelly_percentage(recent_trades):
    """Calculate Kelly percentage from recent trade history"""
    if len(recent_trades) < 10:
        return 0.02  # Default 2% if insufficient data

    trades_df = pd.DataFrame(recent_trades)
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.02  # Default if no wins or losses

    win_rate = len(wins) / len(trades_df)
    loss_rate = 1 - win_rate
    avg_win = wins['pnl'].mean()
    avg_loss = abs(losses['pnl'].mean())

    # Kelly formula: (WinRate × AvgWin - LossRate × AvgLoss) / AvgWin
    kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win

    # Apply Kelly fraction and ensure it's positive and reasonable
    kelly_pct = max(0.01, min(kelly_pct * KELLY_FRACTION, 0.10))  # Cap between 1% and 10%

    return kelly_pct

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print(f"Periode: {START_DATE} tot {END_DATE} (~15.5 years)")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Assets: {', '.join(ASSETS)}")
print(f"Max Risk Cap: ${RISK_CAP:,.0f}")
print(f"Kelly Lookback: {KELLY_LOOKBACK} trades")
print(f"Kelly Fraction: {KELLY_FRACTION} (half-Kelly)")
print()
print("Kelly Criterion adjusts position size based on recent performance")
print("Formula: (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win")
print()

# Load data for all assets
loader = DatabentoMicroFuturesLoader()

print("Laden data...")
all_data_15m = {}
all_data_1h = {}

for asset in ASSETS:
    print(f"  {asset}...", end=" ")
    try:
        df_15m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
        df_1h = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='1h')

        df_15m['ema_21'] = calculate_ema(df_15m['close'], period=21)
        df_15m['atr_20'] = calculate_atr(df_15m['high'], df_15m['low'], df_15m['close'], period=20)

        squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
            df_15m['high'], df_15m['low'], df_15m['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20, kc_multiplier=2.0,
            momentum_period=20
        )
        df_15m['squeeze_on'] = squeeze_on
        df_15m['ttm_momentum'] = momentum

        squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
            df_1h['high'], df_1h['low'], df_1h['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20, kc_multiplier=2.0,
            momentum_period=20
        )
        df_1h['ttm_momentum'] = momentum_1h

        all_data_15m[asset] = df_15m
        all_data_1h[asset] = df_1h

        print(f"✓")
    except Exception as e:
        print(f"✗ Error: {e}")

print()

if not all_data_15m:
    print("⚠ Geen data geladen!")
    sys.exit(1)

# Align timestamps
print("Aligning timestamps...")
common_index = None
for asset in all_data_15m.keys():
    if common_index is None:
        common_index = all_data_15m[asset].index
    else:
        common_index = common_index.intersection(all_data_15m[asset].index)

print(f"✓ Common 15min timestamps: {len(common_index)}")
print()

for asset in all_data_15m.keys():
    all_data_15m[asset] = all_data_15m[asset].loc[common_index]

# Initialize strategies
strategies = {asset: TTMSqueezePullbackStrategy(asset) for asset in all_data_15m.keys()}

# Backtest
print("=" * 70)
print("BACKTEST STARTEN")
print("=" * 70)
print()

capital = INITIAL_CAPITAL
positions = {}
trades = []
equity_curve = []

for i in range(200, len(common_index)):
    timestamp = common_index[i]

    # Update equity
    equity = capital
    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[i]
        if pos['direction'] == 'LONG':
            unrealized = (bar['close'] - pos['entry_price']) * pos['size']
        else:
            unrealized = (pos['entry_price'] - bar['close']) * pos['size']
        equity += unrealized

    equity_curve.append({
        'timestamp': timestamp,
        'equity': equity,
        'capital': capital,
        'positions': len(positions),
    })

    # === POSITION MANAGEMENT ===
    positions_to_close = []

    for asset, pos in positions.items():
        bar = all_data_15m[asset].iloc[i]
        direction = pos['direction']
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        target = pos['target']
        size = pos['size']

        # Check stop loss
        if direction == 'LONG' and bar['low'] <= stop_loss:
            exit_price = stop_loss
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'symbol': asset,
                'direction': direction,
                'setup_type': pos.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'STOP',
                'bars_held': i - pos['entry_bar'],
            })

            positions_to_close.append(asset)
            continue

        # Check target
        if direction == 'LONG' and bar['high'] >= target:
            exit_price = target
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'symbol': asset,
                'direction': direction,
                'setup_type': pos.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'TARGET',
                'bars_held': i - pos['entry_bar'],
            })

            positions_to_close.append(asset)

    # Close positions
    for asset in positions_to_close:
        del positions[asset]

    # === ENTRY SCAN ===
    if len(positions) < MAX_POSITIONS:
        for asset, strategy in strategies.items():
            if asset in positions:
                continue

            window_15m = all_data_15m[asset].iloc[:i+1]
            current_timestamp = common_index[i]
            bars_1h = all_data_1h[asset][all_data_1h[asset].index <= current_timestamp]

            if len(bars_1h) < 20:
                continue

            setup = strategy.check_entry(window_15m, bars_1h, regime='ANY', confidence=80.0)

            if setup and setup.direction.value == 'LONG':
                # === KELLY CRITERION SIZING ===
                # Use rolling window of recent trades to calculate Kelly percentage
                recent_trades = trades[-KELLY_LOOKBACK:] if len(trades) >= KELLY_LOOKBACK else trades
                kelly_pct = calculate_kelly_percentage(recent_trades)

                risk_amount = min(capital * kelly_pct, RISK_CAP)
                stop_distance = setup.entry_price - setup.stop_loss

                position_size_units = risk_amount / stop_distance
                position_value = position_size_units * setup.entry_price
                margin_required = position_value / LEVERAGE

                if margin_required <= capital and len(positions) < MAX_POSITIONS:
                    size = position_size_units
                    positions[asset] = {
                        'entry_time': timestamp,
                        'entry_bar': i,
                        'direction': 'LONG',
                        'entry_price': setup.entry_price,
                        'stop_loss': setup.stop_loss,
                        'target': setup.target,
                        'size': size,
                        'setup_type': setup.setup_type,
                    }
                    break

    # Progress
    if i % 5000 == 0:
        progress = (i / len(common_index)) * 100
        recent_trades = trades[-KELLY_LOOKBACK:] if len(trades) >= KELLY_LOOKBACK else trades
        kelly_pct = calculate_kelly_percentage(recent_trades)
        print(f"Progress: {progress:.1f}%, Capital: ${capital:,.2f}, Kelly: {kelly_pct*100:.2f}%, Trades: {len(trades)}")

# Close remaining positions
for asset, pos in positions.items():
    bar = all_data_15m[asset].iloc[-1]
    exit_price = bar['close']
    direction = pos['direction']
    entry_price = pos['entry_price']
    size = pos['size']

    if direction == 'LONG':
        pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
    else:
        pnl = (entry_price - exit_price) * size - (exit_price * size * COMMISSION_PCT)

    capital += pnl

    trades.append({
        'entry_time': pos['entry_time'],
        'exit_time': common_index[-1],
        'symbol': asset,
        'direction': direction,
        'setup_type': pos.get('setup_type', 'UNKNOWN'),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': (pnl / (entry_price * size)) * 100,
        'exit_reason': 'END',
        'bars_held': len(common_index) - 1 - pos['entry_bar'],
    })

print()
print("=" * 70)
print("RESULTS - KELLY CRITERION")
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
    print(f"Avg Win:             ${wins['pnl'].mean():,.2f}" if len(wins) > 0 else "Avg Win: $0.00")
    print(f"Avg Loss:            ${losses['pnl'].mean():,.2f}" if len(losses) > 0 else "Avg Loss: $0.00")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"Profit Factor:       {profit_factor:.2f}")

    # Calculate final Kelly percentage
    final_kelly = calculate_kelly_percentage(trades[-KELLY_LOOKBACK:])
    print()
    print(f"Final Kelly %:       {final_kelly*100:.2f}%")

print()
print("=" * 70)
