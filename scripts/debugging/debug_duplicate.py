"""
Debug duplicate trades issue
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.backtest.engine import BacktestEngine
from src.regime.detector import RegimeDetector

# Patch the engine to add logging
original_scan = BacktestEngine._scan_for_setups
original_manage = BacktestEngine._manage_positions

def logged_scan(self, timestamp):
    print(f"\n[SCAN @ {timestamp}] Positions: {list(self.broker.positions.keys())}, Closed: {self.closed_this_bar}")
    return original_scan(self, timestamp)

def logged_manage(self, timestamp):
    print(f"[MANAGE @ {timestamp}] Positions: {list(self.broker.positions.keys())}")
    result = original_manage(self, timestamp)
    print(f"[AFTER MANAGE] Positions: {list(self.broker.positions.keys())}, Closed: {self.closed_this_bar}")
    return result

BacktestEngine._scan_for_setups = logged_scan
BacktestEngine._manage_positions = logged_manage

# Run small backtest
loader = DatabentoMicroFuturesLoader()
gold = loader.load_symbol('GOLD', start_date='2025-12-15', end_date='2025-12-20', resample='15min')

detector = RegimeDetector()
gold = detector.add_all_indicators(gold)

engine = BacktestEngine(
    data={'GOLD': gold},
    initial_capital=10000,
    assets=['GOLD'],
    commission_pct=0.001,
    slippage_pct=0.001
)

results = engine.run()

print(f"\n\nTotal trades: {len(results['trades'])}")
