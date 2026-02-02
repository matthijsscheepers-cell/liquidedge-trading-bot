"""
Backtest: Multi-Asset FULL HISTORICAL Test of User's TTM Pullback Strategy

Strategy:
- 1H chart: Trend confirmation (TTM bullish)
- 15min chart: Execution (pullback to -1 ATR)
- Entry: -1 ATR below 21-EMA
- Stop: -2.1 ATR below 21-EMA
- Target: +2 ATR above 21-EMA
- LONG ONLY

Test Period: 2010-2026 (Full dataset - 15+ years)
Assets: GOLD, SILVER, US100, US500
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript
from src.strategies.ttm_pullback import TTMSqueezePullbackStrategy
import pandas as pd

print("=" * 70)
print("MULTI-ASSET FULL HISTORICAL BACKTEST - USER'S TTM PULLBACK STRATEGY")
print("=" * 70)
print()

# Config
START_DATE = '2010-09-12'  # Full historical data from 2010
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 300.0  # Starting with $300
COMMISSION_PCT = 0.001  # 0.1% commission per trade
SLIPPAGE_PCT = 0.001    # 0.1% slippage
RISK_PER_TRADE = 0.02   # 2% risk per trade (CFD risk-based sizing)
LEVERAGE = 20           # CFD leverage 1:20 (5% margin)
MAX_POSITIONS = 4       # Max 4 simultaneous positions

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print(f"Periode: {START_DATE} tot {END_DATE} (~15.5 years)")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Assets: {', '.join(ASSETS)}")
print(f"Max positions: {MAX_POSITIONS}")
print()
print("Testing performance across:")
print("  - 2010-2015: Post-crisis bull market")
print("  - 2015-2016: Correction period")
print("  - 2016-2020: Extended bull market")
print("  - 2020: COVID crash & recovery")
print("  - 2022: Bear market")
print("  - 2023-2026: Recent period")
print()

# Load data for all assets
loader = DatabentoMicroFuturesLoader()

print("Laden data...")
all_data_15m = {}
all_data_1h = {}

for asset in ASSETS:
    print(f"  {asset}...", end=" ")
    try:
        # Load 15min data
        df_15m = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
        df_1h = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='1h')

        # Add indicators to 15min
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

        # Add indicators to 1H
        squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
            df_1h['high'], df_1h['low'], df_1h['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20, kc_multiplier=2.0,
            momentum_period=20
        )
        df_1h['ttm_momentum'] = momentum_1h

        all_data_15m[asset] = df_15m
        all_data_1h[asset] = df_1h

        print(f"✓ 15m: {len(df_15m)}, 1H: {len(df_1h)}")
    except Exception as e:
        print(f"✗ Error: {e}")

print()

if not all_data_15m:
    print("⚠ Geen data geladen!")
    sys.exit(1)

# Align all data to common timestamps
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
positions = {}  # {asset: position_dict}
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
                continue  # Already in position

            # Get data windows
            window_15m = all_data_15m[asset].iloc[:i+1]

            # Find corresponding 1H data
            current_timestamp = common_index[i]
            bars_1h = all_data_1h[asset][all_data_1h[asset].index <= current_timestamp]

            if len(bars_1h) < 20:
                continue

            # Check for setup
            setup = strategy.check_entry(window_15m, bars_1h, regime='ANY', confidence=80.0)

            if setup and setup.direction.value == 'LONG':
                # === CFD POSITION SIZING (with risk cap) ===
                # Calculate position size based on risk and stop distance
                # Position size (units) = Risk / Stop distance
                # Margin required = Position value / Leverage

                # Use 2% risk but cap at $50 max per trade (prevents explosive compounding)
                risk_amount = min(capital * RISK_PER_TRADE, 50.0)
                stop_distance = setup.entry_price - setup.stop_loss  # Points at risk

                # Position size in units (can be fractional for CFDs)
                position_size_units = risk_amount / stop_distance

                # Calculate position value and margin requirement
                position_value = position_size_units * setup.entry_price
                margin_required = position_value / LEVERAGE  # 1:20 leverage = 5% margin

                # Check if we have enough margin
                if margin_required <= capital and len(positions) < MAX_POSITIONS:
                    size = position_size_units
                    entry_cost = margin_required  # For capital tracking
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
                    print(f"✓ ENTRY: {asset} @ {timestamp} - Entry: ${setup.entry_price:.2f}, Stop: ${setup.stop_loss:.2f}, Target: ${setup.target:.2f}")
                    break  # Only 1 new position per bar

    # Progress
    if i % 5000 == 0:
        progress = (i / len(common_index)) * 100
        print(f"Progress: {progress:.1f}% ({i}/{len(common_index)}), Positions: {len(positions)}, Trades: {len(trades)}, Capital: ${capital:,.2f}")

# Close any remaining positions
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
print("BACKTEST COMPLEET")
print("=" * 70)
print()

# Results
trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

print(f"Totaal trades: {len(trades_df)}")
print(f"Final capital: ${capital:,.2f}")
print()

if len(trades_df) > 0:
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    print("=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print()

    print(f"📊 RETURNS:")
    print(f"   Initial Capital:     ${INITIAL_CAPITAL:,.2f}")
    print(f"   Final Capital:       ${capital:,.2f}")
    print(f"   Total Return:        {total_return:.2f}%")
    print()

    print(f"🎯 TRADES:")
    print(f"   Total Trades:        {len(trades_df)}")
    print(f"   Winning Trades:      {len(wins)}")
    print(f"   Losing Trades:       {len(losses)}")
    print(f"   Win Rate:            {len(wins)/len(trades_df)*100:.1f}%")
    print()

    print(f"💰 WIN/LOSS:")
    print(f"   Total Profit:        ${wins['pnl'].sum():,.2f}")
    print(f"   Total Loss:          ${losses['pnl'].sum():,.2f}")
    print(f"   Avg Win:             ${wins['pnl'].mean():,.2f}" if len(wins) > 0 else "   Avg Win:             $0.00")
    print(f"   Avg Loss:            ${losses['pnl'].mean():,.2f}" if len(losses) > 0 else "   Avg Loss:            $0.00")
    print(f"   Largest Win:         ${wins['pnl'].max():,.2f}" if len(wins) > 0 else "   Largest Win:         $0.00")
    print(f"   Largest Loss:        ${losses['pnl'].min():,.2f}" if len(losses) > 0 else "   Largest Loss:        $0.00")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"   Profit Factor:       {profit_factor:.2f}")
    print()

    # Per asset breakdown
    print("=" * 70)
    print("PERFORMANCE PER ASSET")
    print("=" * 70)
    print()

    for asset in ASSETS:
        asset_trades = trades_df[trades_df['symbol'] == asset]
        if len(asset_trades) > 0:
            asset_wins = asset_trades[asset_trades['pnl'] > 0]
            win_rate = len(asset_wins) / len(asset_trades) * 100
            total_pnl = asset_trades['pnl'].sum()

            print(f"{asset}:")
            print(f"  Trades: {len(asset_trades)}")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Total P&L: ${total_pnl:,.2f}")
            print(f"  Avg bars held: {asset_trades['bars_held'].mean():.1f}")
            print()
        else:
            print(f"{asset}: 0 trades")
            print()

    # Exit reason breakdown
    print("=" * 70)
    print("EXIT REASON BREAKDOWN")
    print("=" * 70)
    print()

    for reason in trades_df['exit_reason'].unique():
        reason_trades = trades_df[trades_df['exit_reason'] == reason]
        print(f"{reason}: {len(reason_trades)} trades")
    print()

    # Max positions
    max_positions = equity_df['positions'].max()
    avg_positions = equity_df['positions'].mean()

    print("=" * 70)
    print("POSITION UTILIZATION")
    print("=" * 70)
    print()
    print(f"Max positions: {max_positions} (limit: {MAX_POSITIONS})")
    print(f"Avg positions: {avg_positions:.2f}")
    print()
else:
    print("⚠ NO TRADES GENERATED!")
    print()
    print("This confirms the -1 ATR entry condition is too strict.")
    print("Price rarely pulls back 1 full ATR below the 21-EMA even over 2 years.")
    print()

print("=" * 70)
