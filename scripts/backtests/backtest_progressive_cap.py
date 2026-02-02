"""
Backtest: Progressive Risk Cap Optimization

Tests progressive risk cap that scales with account growth:
- $50 cap when capital < $1,000
- $100 cap when capital < $5,000
- $200 cap when capital < $20,000
- $500 cap when capital < $100,000
- $1,000 cap when capital >= $100,000

This allows compounding while preventing explosive unrealistic growth.
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
print("OPTIMIZATION 1: PROGRESSIVE RISK CAP")
print("=" * 70)
print()

# Config
START_DATE = '2010-09-12'
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 300.0
COMMISSION_PCT = 0.001
SLIPPAGE_PCT = 0.001
RISK_PER_TRADE = 0.02  # 2% base risk
LEVERAGE = 20
MAX_POSITIONS = 4

# Progressive risk cap tiers
def get_risk_cap(capital):
    """Return risk cap based on current capital"""
    if capital < 1000:
        return 50.0
    elif capital < 5000:
        return 100.0
    elif capital < 20000:
        return 200.0
    elif capital < 100000:
        return 500.0
    else:
        return 1000.0

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print(f"Periode: {START_DATE} tot {END_DATE} (~15.5 years)")
print(f"Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"Assets: {', '.join(ASSETS)}")
print()
print("Progressive Risk Cap Tiers:")
print("  < $1,000:     $50 max risk per trade")
print("  < $5,000:     $100 max risk per trade")
print("  < $20,000:    $200 max risk per trade")
print("  < $100,000:   $500 max risk per trade")
print("  >= $100,000:  $1,000 max risk per trade")
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
                # === PROGRESSIVE RISK CAP ===
                current_risk_cap = get_risk_cap(capital)
                risk_amount = min(capital * RISK_PER_TRADE, current_risk_cap)
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
        current_cap = get_risk_cap(capital)
        print(f"Progress: {progress:.1f}%, Capital: ${capital:,.2f}, Risk Cap: ${current_cap:.0f}, Trades: {len(trades)}")

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
print("RESULTS - PROGRESSIVE RISK CAP")
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

print()
print("=" * 70)
