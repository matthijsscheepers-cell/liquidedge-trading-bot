"""
Quick Databento backtest - laatste 3 maanden
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceCalculator
from src.backtest.visualizer import BacktestVisualizer
from src.regime.detector import RegimeDetector

print("=" * 70)
print("LIQUIDEDGE QUICK BACKTEST - 3 MAANDEN")
print("=" * 70)
print()

# Configuratie
START_DATE = '2025-11-01'  # 3 maanden
END_DATE = '2026-01-29'
INITIAL_CAPITAL = 10000.0
ASSETS = ['GOLD']  # Eerst alleen GOLD testen

print(f"Periode: {START_DATE} tot {END_DATE}")
print(f"Assets: {', '.join(ASSETS)}")
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

data = {'GOLD': gold}

# Run backtest
print("=" * 70)
print("BACKTEST STARTEN")
print("=" * 70)
print()

engine = BacktestEngine(
    data=data,
    initial_capital=INITIAL_CAPITAL,
    assets=ASSETS,
    commission_pct=0.001,
    slippage_pct=0.001
)

results = engine.run()

# Print resultaten
print()
print("=" * 70)
print("RESULTATEN")
print("=" * 70)
print()

calc = PerformanceCalculator(
    results['equity_curve'],
    results['trades'],
    INITIAL_CAPITAL
)
calc.print_summary(results)

print()
print(f"Aantal trades: {len(results['trades'])}")

if results['trades']:
    import pandas as pd
    trades_df = pd.DataFrame(results['trades'])

    print()
    print("Laatste 10 trades:")
    print(trades_df[['entry_time', 'symbol', 'direction', 'pnl', 'pnl_pct']].tail(10))

print()
print("=" * 70)
print("MET NIEUWE REGIME CALIBRATIE!")
print("=" * 70)
