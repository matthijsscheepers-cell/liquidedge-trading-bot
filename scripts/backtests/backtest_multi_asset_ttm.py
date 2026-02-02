"""
Multi-Asset TTM Squeeze Backtest

Test TTM Squeeze strategy over meerdere assets:
- GOLD
- SILVER
- US100 (Nasdaq)
- US500 (S&P 500)
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.regime.detector import RegimeDetector
from src.strategies.ttm_squeeze import TTMSqueezeStrategy
import pandas as pd

print("=" * 70)
print("MULTI-ASSET TTM SQUEEZE BACKTEST")
print("=" * 70)
print()

# Config
START_DATE = '2025-11-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001
RISK_PER_TRADE = 0.01  # 1% risk per trade
MAX_POSITIONS = 4  # Max 4 simultaneous positions

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print(f"Periode: {START_DATE} tot {END_DATE}")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Assets: {', '.join(ASSETS)}")
print(f"Max positions: {MAX_POSITIONS}")
print()

# Laad data voor alle assets
loader = DatabentoMicroFuturesLoader()
detector = RegimeDetector()

print("Laden data...")
all_data = {}

for asset in ASSETS:
    print(f"  {asset}...", end=" ")
    try:
        df = loader.load_symbol(asset, start_date=START_DATE, end_date=END_DATE, resample='15min')
        df = detector.add_all_indicators(df)
        all_data[asset] = df
        print(f"✓ {len(df)} bars")
    except Exception as e:
        print(f"✗ Error: {e}")

print()

if not all_data:
    print("⚠ Geen data geladen!")
    sys.exit(1)

# Align all data to same timestamps
print("Aligning timestamps...")
common_index = None
for asset, df in all_data.items():
    if common_index is None:
        common_index = df.index
    else:
        common_index = common_index.intersection(df.index)

print(f"✓ Common timestamps: {len(common_index)}")
print()

for asset in all_data.keys():
    all_data[asset] = all_data[asset].loc[common_index]

# Initialize strategies
strategies = {asset: TTMSqueezeStrategy(asset) for asset in all_data.keys()}

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
        bar = all_data[asset].iloc[i]
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
        bar = all_data[asset].iloc[i]
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

            window = all_data[asset].iloc[:i+1]
            setup = strategy.check_entry(window, regime='ANY', confidence=80.0)

            if setup and setup.direction.value == 'LONG':
                # Calculate position size
                risk_per_trade = capital * RISK_PER_TRADE
                size = risk_per_trade / setup.risk_per_share
                entry_cost = setup.entry_price * size

                if entry_cost <= capital and len(positions) < MAX_POSITIONS:
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
                    break  # Only 1 new position per bar

    # Progress
    if i % 500 == 0:
        progress = (i / len(common_index)) * 100
        print(f"Progress: {progress:.1f}% ({i}/{len(common_index)}), Positions: {len(positions)}, Trades: {len(trades)}")

# Close any remaining positions
for asset, pos in positions.items():
    bar = all_data[asset].iloc[-1]
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
            print()

    # Setup type breakdown
    print("=" * 70)
    print("SETUP TYPE BREAKDOWN")
    print("=" * 70)
    print()

    for setup_type in trades_df['setup_type'].unique():
        setup_trades = trades_df[trades_df['setup_type'] == setup_type]
        setup_wins = setup_trades[setup_trades['pnl'] > 0]
        win_rate = len(setup_wins) / len(setup_trades) * 100

        print(f"{setup_type}:")
        print(f"  Trades: {len(setup_trades)}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${setup_trades['pnl'].sum():,.2f}")
        print()

    # Max positions at any time
    max_positions = equity_df['positions'].max()
    avg_positions = equity_df['positions'].mean()

    print("=" * 70)
    print("POSITION UTILIZATION")
    print("=" * 70)
    print()
    print(f"Max positions: {max_positions} (limit: {MAX_POSITIONS})")
    print(f"Avg positions: {avg_positions:.2f}")
    print()

print("=" * 70)
