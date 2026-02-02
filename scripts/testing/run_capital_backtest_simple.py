"""
Simple Capital.com backtest runner
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.execution.capital_connector import CapitalConnector
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceCalculator
from src.backtest.visualizer import BacktestVisualizer

print("=" * 70)
print("CAPITAL.COM SIMPLE BACKTEST")
print("=" * 70)
print()

# Connect once
config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

connector = CapitalConnector(config)
print("Connecting to Capital.com...")
connector.connect()
print()

# Fetch data for all assets
assets = ['GOLD', 'SILVER', 'US100', 'US500']
data = {}

for asset in assets:
    print(f"Fetching {asset}...")
    try:
        df = connector.get_historical_data(asset, '15m', 500)  # Get last 500 bars
        if not df.empty:
            data[asset] = df
            print(f"  ✓ {asset}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"  ✗ {asset}: No data")
    except Exception as e:
        print(f"  ✗ {asset}: Error - {e}")

print()

if not data:
    print("No data loaded. Exiting.")
    sys.exit(1)

print(f"Loaded {len(data)} assets")
print()

# Run backtest
print("=" * 70)
print("RUNNING BACKTEST")
print("=" * 70)
print()

engine = BacktestEngine(
    data=data,
    initial_capital=10000,
    assets=list(data.keys()),
    commission_pct=0.001,
    slippage_pct=0.001
)

results = engine.run()

# Print results
print()
calc = PerformanceCalculator(
    results['equity_curve'],
    results['trades'],
    10000
)
calc.print_summary(results)

# Visualize
print("\nGenerating visualizations...")
viz = BacktestVisualizer(results)
viz.plot_all(save_path='data/backtest_results/capital_simple_backtest.png')

print("\n✓ Backtest complete!")
print("✓ Chart saved to: data/backtest_results/capital_simple_backtest.png")
