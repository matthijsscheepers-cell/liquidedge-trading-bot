"""
LIQUIDEDGE Trading Bot - Main Package

A hybrid trading bot combining technical analysis, regime detection,
and risk management for automated forex trading.

Main modules:
    - indicators: Technical indicators (trend, volatility, TTM squeeze)
    - regime: Market regime detection (the BRAIN of the bot)

Quick start:
    from src.regime import RegimeDetector, MarketRegime
    from src.indicators import calculate_adx, calculate_atr

    detector = RegimeDetector()
    df = detector.add_all_indicators(df)
    regime, confidence, strategy = detector.detect_regime(df)
"""

__version__ = "0.1.0"
__author__ = "LIQUIDEDGE Team"

from typing import Final

# Package metadata
PACKAGE_NAME: Final[str] = "liquidedge"

# Export main modules for easy access
from . import indicators
from . import regime

__all__ = [
    'indicators',
    'regime',
    '__version__',
]
