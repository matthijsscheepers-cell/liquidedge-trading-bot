"""
Technical Indicators Module

This module provides a comprehensive suite of technical analysis indicators
for trend detection, volatility measurement, and momentum analysis.

Indicator Categories:
    - Trend: ADX, EMA, SMA, slopes, moving average crosses
    - Volatility: ATR, Bollinger Bands, Keltner Channels
    - TTM Squeeze: John Carter's squeeze indicator with momentum

Usage:
    from src.indicators import (
        calculate_adx,
        calculate_ema,
        calculate_atr,
        calculate_bollinger_bands,
        calculate_ttm_squeeze
    )

    # Trend analysis
    adx, plus_di, minus_di = calculate_adx(high, low, close)
    ema_20 = calculate_ema(close, period=20)

    # Volatility analysis
    atr = calculate_atr(high, low, close)
    bb_upper, bb_mid, bb_lower, bb_width = calculate_bollinger_bands(close)

    # TTM Squeeze
    squeeze_on, momentum, color = calculate_ttm_squeeze(high, low, close)
"""

from typing import List

# Trend indicators
from src.indicators.trend import (
    calculate_ema,
    calculate_sma,
    calculate_slope,
    calculate_adx,
    calculate_trend_strength,
    calculate_trend_direction,
    calculate_moving_average_cross,
    # DataFrame-based convenience functions
    calculate_adx_df,
    calculate_ema_slope,
    calculate_multiple_emas,
    check_ema_alignment,
    add_trend_indicators,
)

# Volatility indicators
from src.indicators.volatility import (
    calculate_true_range,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_volatility_ratio,
    calculate_band_position,
    detect_squeeze,
    calculate_squeeze_duration,
    calculate_historical_volatility,
)

# TTM Squeeze indicators
from src.indicators.ttm import (
    calculate_momentum,
    calculate_ttm_squeeze,
    calculate_momentum_color,
    identify_squeeze_setups,
    calculate_squeeze_strength,
    get_ttm_signals,
)

__all__: List[str] = [
    # Trend
    "calculate_ema",
    "calculate_sma",
    "calculate_slope",
    "calculate_adx",
    "calculate_trend_strength",
    "calculate_trend_direction",
    "calculate_moving_average_cross",
    "calculate_adx_df",
    "calculate_ema_slope",
    "calculate_multiple_emas",
    "check_ema_alignment",
    "add_trend_indicators",
    # Volatility
    "calculate_true_range",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_keltner_channels",
    "calculate_volatility_ratio",
    "calculate_band_position",
    "detect_squeeze",
    "calculate_squeeze_duration",
    "calculate_historical_volatility",
    # TTM Squeeze
    "calculate_momentum",
    "calculate_ttm_squeeze",
    "calculate_momentum_color",
    "identify_squeeze_setups",
    "calculate_squeeze_strength",
    "get_ttm_signals",
]
