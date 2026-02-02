"""
Backtest: Gebruiker's Multi-Timeframe TTM Squeeze Pullback Strategy

Strategy:
- 1H chart: Trend confirmation (TTM bullish)
- 15min chart: Execution (pullback to -1 ATR)
- Entry: -1 ATR
- Stop: -2 ATR
- Target: +2 ATR
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
print("USER'S TTM SQUEEZE PULLBACK STRATEGY")
print("=" * 70)
print()

# Config
START_DATE = '2025-11-01'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001
RISK_PER_TRADE = 0.01

print(f"Periode: {START_DATE} tot {END_DATE}")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Strategy: Multi-Timeframe (1H + 15min)")
print()

# Load data
loader = DatabentoMicroFuturesLoader()

print("Laden GOLD data...")
print("  15min data...", end=" ")
gold_15m = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE, resample='15min')
print(f"✓ {len(gold_15m)} bars")

print("  1H data...", end=" ")
gold_1h = loader.load_symbol('GOLD', start_date=START_DATE, end_date=END_DATE, resample='1h')
print(f"✓ {len(gold_1h)} bars")
print()

# Add indicators to 15min data
print("Adding indicators to 15min data...")
gold_15m['ema_21'] = calculate_ema(gold_15m['close'], period=21)
gold_15m['atr_20'] = calculate_atr(gold_15m['high'], gold_15m['low'], gold_15m['close'], period=20)

squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
    gold_15m['high'], gold_15m['low'], gold_15m['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
gold_15m['squeeze_on'] = squeeze_on
gold_15m['ttm_momentum'] = momentum
gold_15m['ttm_color'] = color
print("✓ 15min indicators added")

# Add indicators to 1H data
print("Adding indicators to 1H data...")
squeeze_on_1h, momentum_1h, color_1h = calculate_ttm_squeeze_pinescript(
    gold_1h['high'], gold_1h['low'], gold_1h['close'],
    bb_period=20, bb_std=2.0,
    kc_period=20, kc_multiplier=2.0,
    momentum_period=20
)
gold_1h['ttm_momentum'] = momentum_1h
gold_1h['ttm_color'] = color_1h
print("✓ 1H indicators added")
print()

# Initialize strategy
strategy = TTMSqueezePullbackStrategy('GOLD')

# Backtest
print("=" * 70)
print("BACKTEST STARTEN")
print("=" * 70)
print()

capital = INITIAL_CAPITAL
trades = []
equity_curve = []
current_position = None

# Start from bar 200 (enough data for indicators)
for i in range(200, len(gold_15m)):
    timestamp = gold_15m.index[i]

    # Get corresponding 1H bar
    # Find the 1H bar that contains this 15min bar
    current_1h_bars = gold_1h[gold_1h.index <= timestamp]
    if len(current_1h_bars) == 0:
        continue

    # Get 1H data up to current time
    df_1h = current_1h_bars

    # Get 15min data up to current bar
    df_15m = gold_15m.iloc[:i+1]

    bar = df_15m.iloc[-1]

    # Update equity
    equity = capital
    if current_position:
        unrealized = (bar['close'] - current_position['entry_price']) * current_position['size']
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
        if bar['low'] <= stop_loss:
            exit_price = stop_loss
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': 'LONG',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'STOP',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

        # Check target
        if bar['high'] >= target:
            exit_price = target
            pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
            capital += pnl

            trades.append({
                'entry_time': current_position['entry_time'],
                'exit_time': timestamp,
                'symbol': 'GOLD',
                'direction': 'LONG',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': (pnl / (entry_price * size)) * 100,
                'exit_reason': 'TARGET',
                'bars_held': i - current_position['entry_bar'],
            })

            current_position = None
            continue

    # === ENTRY SCAN ===
    if not current_position:
        # Check for setup
        setup = strategy.check_entry(df_15m, df_1h, regime='ANY', confidence=80.0)

        if setup:
            # Calculate position size (1% risk)
            risk_per_trade = capital * RISK_PER_TRADE
            size = risk_per_trade / setup.risk_per_share

            entry_cost = setup.entry_price * size

            if entry_cost <= capital:
                current_position = {
                    'entry_time': timestamp,
                    'entry_bar': i,
                    'entry_price': setup.entry_price,
                    'stop_loss': setup.stop_loss,
                    'target': setup.target,
                    'size': size,
                }

                print(f"[ENTRY] {timestamp}")
                print(f"  Entry: ${setup.entry_price:.2f}")
                print(f"  Stop: ${setup.stop_loss:.2f} ({((setup.stop_loss/setup.entry_price - 1) * 100):.1f}%)")
                print(f"  Target: ${setup.target:.2f} ({((setup.target/setup.entry_price - 1) * 100):.1f}%)")
                print(f"  Size: {size:.3f}")
                print(f"  1H Momentum: {setup.metadata['momentum_1h']:.3f}")
                print(f"  15M Momentum: {setup.metadata['momentum_15m']:.3f}")
                print()

    # Progress
    if i % 500 == 0:
        progress = (i / len(gold_15m)) * 100
        print(f"Progress: {progress:.1f}% ({i}/{len(gold_15m)}), Trades: {len(trades)}")

# Close any open position
if current_position:
    exit_price = gold_15m.iloc[-1]['close']
    entry_price = current_position['entry_price']
    size = current_position['size']

    pnl = (exit_price - entry_price) * size - (exit_price * size * COMMISSION_PCT)
    capital += pnl

    trades.append({
        'entry_time': current_position['entry_time'],
        'exit_time': gold_15m.index[-1],
        'symbol': 'GOLD',
        'direction': 'LONG',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': (pnl / (entry_price * size)) * 100,
        'exit_reason': 'END',
        'bars_held': len(gold_15m) - 1 - current_position['entry_bar'],
    })

print()
print("=" * 70)
print("BACKTEST COMPLEET")
print("=" * 70)
print()

# Results
trades_df = pd.DataFrame(trades)

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
    if len(wins) > 0:
        print(f"   Avg Win:             ${wins['pnl'].mean():,.2f}")
        print(f"   Largest Win:         ${wins['pnl'].max():,.2f}")
    if len(losses) > 0:
        print(f"   Avg Loss:            ${losses['pnl'].mean():,.2f}")
        print(f"   Largest Loss:        ${losses['pnl'].min():,.2f}")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())
        print(f"   Profit Factor:       {profit_factor:.2f}")

    print()

    # Show all trades
    if len(trades_df) > 0:
        print("=" * 70)
        print("ALL TRADES")
        print("=" * 70)
        print()
        for idx, trade in trades_df.iterrows():
            print(f"{idx+1}. {trade['entry_time']} -> {trade['exit_time']}")
            print(f"   Entry: ${trade['entry_price']:.2f}")
            print(f"   Exit: ${trade['exit_price']:.2f} ({trade['exit_reason']})")
            print(f"   P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
            print(f"   Bars held: {trade['bars_held']}")
            print()

else:
    print("⚠ Geen trades gegenereerd!")
    print()
    print("Mogelijke redenen:")
    print("  - Multi-timeframe alignment te strikt")
    print("  - Pullback naar -1 ATR komt zelden voor")
    print("  - 1H en 15min momentum alignment zeldzaam")

print("=" * 70)
