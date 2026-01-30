"""
Market Regime Detection Module

This module implements algorithms to detect and classify market conditions
into distinct regimes for adaptive trading strategies.

Regime Types:
    - Trending Bullish: Strong upward trend (ADX > 25, +DI > -DI)
    - Trending Bearish: Strong downward trend (ADX > 25, -DI > +DI)
    - Ranging: Sideways movement (ADX < 20)
    - Volatile: High volatility without clear trend
    - Squeeze: Low volatility compression (potential breakout)

Detection Methods:
    - ADX-based trend strength analysis
    - Volatility measurement (ATR, Bollinger Bands)
    - TTM Squeeze indicator
    - Multi-indicator regime classification

Usage:
    from src.regime import RegimeDetector, RegimeType

    # Initialize detector
    detector = RegimeDetector()

    # Detect regime
    regime_df = detector.detect(high, low, close)

    # Get current regime
    current = detector.get_current_regime(high, low, close)
    print(f"Current regime: {current['regime']}")
"""

from typing import List
from src.regime.detector import (
    RegimeDetector,
    RegimeType,
    TrendStrength,
    VolatilityState,
    RegimeConfig,
)

__all__: List[str] = [
    "RegimeDetector",
    "RegimeType",
    "TrendStrength",
    "VolatilityState",
    "RegimeConfig",
]
