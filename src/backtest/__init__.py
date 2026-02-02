"""
Backtesting framework for LiquidEdge.

Components:
- DataLoader: Load historical data
- BacktestEngine: Run backtests
- PerformanceCalculator: Calculate metrics
- BacktestVisualizer: Visualize results
"""

from .data_loader import DataLoader, CSVDataLoader, YahooFinanceLoader, CapitalDataLoader
from .engine import BacktestEngine
from .performance import PerformanceCalculator
from .visualizer import BacktestVisualizer

__all__ = [
    'DataLoader',
    'CSVDataLoader',
    'YahooFinanceLoader',
    'CapitalDataLoader',
    'BacktestEngine',
    'PerformanceCalculator',
    'BacktestVisualizer'
]
