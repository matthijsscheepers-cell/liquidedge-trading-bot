"""
Backtest: Pure TTM Squeeze trades zonder regime filtering

Dit script runt een volledige backtest met ALLEEN TTM Squeeze trades.
De regime filter is uitgeschakeld - we nemen elke TTM Squeeze setup die voldoet.
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.regime.detector import RegimeDetector
from src.strategies.ttm_squeeze import TTMSqueezeStrategy
import pandas as pd
from datetime import datetime

print("=" * 70)
print("PURE TTM SQUEEZE BACKTEST")
print("=" * 70)
print()

# Configuratie
START_DATE = '2025-11-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001

print(f"Periode: {START_DATE} tot {END_DATE}")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Commissie: {COMMISSION_PCT*100}%")
print(f"Slippage: {SLIPPAGE_PCT*100}%")
print()

# Laad data
loader = DatabentoMicroFuturesLoader()
print("Laden GOLD data...")
gold = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE, resample='15min')
print(f"✓ {len(gold)} bars geladen")
print()

# Voeg indicatoren toe
detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

# Maak strategy
strategy = TTMSqueezeStrategy('GOLD')
capital = INITIAL_CAPITAL

# Tracking
trades = []
equity_curve = []
current_position = None

print("=" * 70)
print("BACKTEST STARTEN")
print("=" * 70)
print()

# Backtest loop
for i in range(200, len(gold)):
    timestamp = gold.index[i]
    bar = gold.iloc[i]
    window = gold.iloc[:i+1]

    # Update equity
    equity = capital
    if current_position:
        # Mark-to-market
        if current_position['direction'] == 'LONG':
            unrealized = (bar['close'] - current_position['entry_price']) * current_position['size']
        else:
            unrealized = (current_position['entry_price'] - bar['close']) * current_position['size']
        equity += unrealized

    equity_curve.append({
        'timestamp': timestamp,
        'equity': equity,
        'capital': capital,
    })

    # === POSITION MANAGEMENT ===
    if current_position:
        direction = current_position['direction']
        entry_price = current_position['entry_price']
        stop_loss = current_position['stop_loss']
        target = current_position['target']
        size = current_position['size']

        # Check stop loss
        if direction == 'LONG' and bar['low'] <= stop_loss:
            # Stopped out
            exit_price = stop_loss
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': direction,
                'setup_type': current_position.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'STOP',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

        elif direction == 'SHORT' and bar['high'] >= stop_loss:
            # Stopped out
            exit_price = stop_loss
            pnl = (entry_price - exit_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': direction,
                'setup_type': current_position.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'STOP',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

        # Check target
        if direction == 'LONG' and bar['high'] >= target:
            # Target hit
            exit_price = target
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': direction,
                'setup_type': current_position.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'TARGET',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

        elif direction == 'SHORT' and bar['low'] <= target:
            # Target hit
            exit_price = target
            pnl = (entry_price - exit_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': direction,
                'setup_type': current_position.get('setup_type', 'UNKNOWN'),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'size': size,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'TARGET',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

    # === ENTRY SCAN ===
    if not current_position:
        # Check for TTM Squeeze setup (any regime)
        setup = strategy.check_entry(window, regime='ANY', confidence=80.0)

        if setup:
            # Calculate position size (1% risk)
            risk_per_trade = capital * 0.01
            size = risk_per_trade / setup.risk_per_share

            # Entry cost
            entry_cost = setup.entry_price * size
            commission = entry_cost * COMMISSION_PCT
            slippage = entry_cost * SLIPPAGE_PCT
            total_cost = entry_cost + commission + slippage

            # Check if we have enough capital
            if total_cost <= capital:
                current_position = {
                    'entry_time': timestamp,
                    'entry_bar': i,
                    'direction': setup.direction.value,
                    'entry_price': setup.entry_price,
                    'stop_loss': setup.stop_loss,
                    'target': setup.target,
                    'size': size,
                    'setup_type': setup.setup_type,
                }

    # Progress
    if i % 500 == 0:
        progress = (i / len(gold)) * 100
        print(f"Progress: {progress:.1f}% ({i}/{len(gold)})")

# Close any open position at end
if current_position:
    exit_price = gold.iloc[-1]['close']
    direction = current_position['direction']
    entry_price = current_position['entry_price']
    size = current_position['size']

    if direction == 'LONG':
        pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
    else:
        pnl = (entry_price - exit_price) * size - (exit_price * size * COMMISSION_PCT)

    capital += pnl

    trades.append({
        'entry_time': current_position['entry_time'],
        'exit_time': gold.index[-1],
        'symbol': 'GOLD',
        'direction': direction,
        'setup_type': current_position.get('setup_type', 'UNKNOWN'),
        'entry_price': entry_price,
        'exit_price': exit_price,
        'size': size,
        'pnl': pnl,
        'pnl_pct': (pnl / (entry_price * size)) * 100,
        'exit_reason': 'END',
        'bars_held': len(gold) - 1 - current_position['entry_bar'],
    })

print()
print("=" * 70)
print("BACKTEST COMPLEET")
print("=" * 70)
print()

# Converteer naar DataFrame
trades_df = pd.DataFrame(trades)
equity_df = pd.DataFrame(equity_curve)

# Print resultaten
print(f"Totaal trades: {len(trades_df)}")
print(f"Final capital: ${capital:,.2f}")
print()

if len(trades_df) > 0:
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]

    print("=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print()

    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

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
    print()

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"   Profit Factor:       {profit_factor:.2f}")
    print()

    # Setup type breakdown
    print("=" * 70)
    print("SETUP TYPE BREAKDOWN")
    print("=" * 70)
    print()

    for setup_type in trades_df['setup_type'].unique():
        setup_trades = trades_df[trades_df['setup_type'] == setup_type]
        setup_wins = setup_trades[setup_trades['pnl'] > 0]
        win_rate = len(setup_wins) / len(setup_trades) * 100 if len(setup_trades) > 0 else 0

        print(f"{setup_type}:")
        print(f"  Trades: {len(setup_trades)}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${setup_trades['pnl'].sum():,.2f}")
        print()

    # Laatste 10 trades
    print("=" * 70)
    print("LAATSTE 10 TRADES")
    print("=" * 70)
    print()
    print(trades_df[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].tail(10).to_string(index=False))

print()
print("=" * 70)
