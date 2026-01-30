"""
Trading Strategies Module

This module contains various trading strategies that combine indicators,
regime detection, and risk management for signal generation.

Strategy Types:
    - Trend Following: Momentum-based strategies
    - Mean Reversion: Counter-trend strategies
    - Breakout: Support/resistance breakout strategies
    - Hybrid: Multi-strategy combinations
    - Regime-Adaptive: Strategies that adapt to market conditions

Each strategy implements:
    - Entry/exit logic
    - Position sizing
    - Stop-loss and take-profit levels
    - Signal confidence scoring
"""

from typing import List

__all__: List[str] = []
