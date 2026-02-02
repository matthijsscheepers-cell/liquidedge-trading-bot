"""
Inspecteer de trades om te zien waarom er duplicates zijn
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.backtest.engine import BacktestEngine
from src.regime.detector import RegimeDetector
import pandas as pd

print("=" * 70)
print("TRADE INSPECTION")
print("=" * 70)
print()

# Laad kleine dataset
loader = DatabentoMicroFuturesLoader()
gold = loader.load_symbol('GOLD', start_date='2025-12-01', end_date='2026-01-29', resample='15min')

# Voeg indicatoren toe
detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

# Run backtest
engine = BacktestEngine(
    data={'GOLD': gold},
    initial_capital=10000,
    assets=['GOLD'],
    commission_pct=0.001,
    slippage_pct=0.001
)

results = engine.run()

print()
print(f"Totaal trades: {len(results['trades'])}")
print()

if len(results['trades']) > 0:
    trades = results['trades']

    print("Trade DataFrame info:")
    print(f"  Type: {type(trades)}")
    print(f"  Shape: {trades.shape}")
    print(f"  Columns: {list(trades.columns)}")
    print()

    print("Alle trades:")
    print(trades)
    print()

    print("Trade details:")
    for idx, trade in trades.iterrows():
        print(f"\nTrade {idx}:")
        print(f"  Data: {trade.to_dict()}")

print()
print("=" * 70)
