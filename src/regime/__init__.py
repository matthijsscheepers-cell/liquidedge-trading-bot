"""
Market Regime Detection Module

Main class: RegimeDetector - The BRAIN of the trading bot
Main enum: MarketRegime - Trading regime classifications

Usage:
    from src.regime import RegimeDetector, MarketRegime, RegimeConfig

    # Initialize detector
    detector = RegimeDetector()

    # Add all indicators to your DataFrame
    df = detector.add_all_indicators(df)

    # Get trading decision
    regime, confidence, strategy = detector.detect_regime(df)

    # Use confidence for position sizing
    if confidence > 70:
        position_size = base_size * (confidence / 100)
        execute_strategy(strategy)
"""

from .detector import RegimeDetector, MarketRegime, RegimeConfig

__all__ = ['RegimeDetector', 'MarketRegime', 'RegimeConfig']
