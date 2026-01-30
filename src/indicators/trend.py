"""
Trend Indicators Module

This module implements trend-following indicators for identifying
market direction and momentum.

Indicators:
    - ADX (Average Directional Index): Measures trend strength
    - EMA (Exponential Moving Average): Smoothed moving average
    - Slope: Rate of change for trend momentum
    - +DI/-DI: Directional indicators for ADX

Usage:
    from src.indicators.trend import calculate_adx, calculate_ema, calculate_slope

    # Calculate ADX
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)

    # Calculate EMA
    ema = calculate_ema(close, period=20)

    # Calculate slope
    slope = calculate_slope(close, period=14)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def calculate_ema(
    data: Union[pd.Series, np.ndarray],
    period: int = 20
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive
    to new information than simple moving average.

    Args:
        data: Price data (typically close prices)
        period: Number of periods for EMA calculation

    Returns:
        Pandas Series containing EMA values

    Example:
        >>> close = pd.Series([100, 102, 101, 103, 105])
        >>> ema = calculate_ema(close, period=3)
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    return data.ewm(span=period, adjust=False).mean()


def calculate_sma(
    data: Union[pd.Series, np.ndarray],
    period: int = 20
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        data: Price data
        period: Number of periods for SMA calculation

    Returns:
        Pandas Series containing SMA values
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    return data.rolling(window=period).mean()


def calculate_slope(
    data: Union[pd.Series, np.ndarray],
    period: int = 14,
    normalize: bool = True
) -> pd.Series:
    """
    Calculate slope (rate of change) of data over a period.

    The slope indicates the momentum and direction of the trend.
    Positive slope = uptrend, negative slope = downtrend.

    Args:
        data: Price data (typically close or moving average)
        period: Number of periods to calculate slope over
        normalize: If True, normalize slope by price level (percentage change)

    Returns:
        Pandas Series containing slope values

    Example:
        >>> close = pd.Series([100, 102, 104, 103, 105])
        >>> slope = calculate_slope(close, period=3)
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Calculate simple linear regression slope over rolling window
    slopes = pd.Series(index=data.index, dtype=float)

    for i in range(period - 1, len(data)):
        y = data.iloc[i - period + 1:i + 1].values
        x = np.arange(period)

        # Linear regression: y = mx + b
        # Slope (m) = covariance(x,y) / variance(x)
        m = np.polyfit(x, y, 1)[0]

        if normalize and data.iloc[i] != 0:
            # Normalize by current price to get percentage slope
            m = (m / data.iloc[i]) * 100

        slopes.iloc[i] = m

    return slopes


def calculate_adx(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Average Directional Index (ADX) and directional indicators.

    ADX measures the strength of a trend (regardless of direction).
    - ADX > 25: Strong trend
    - ADX < 20: Weak/no trend
    - +DI > -DI: Bullish trend
    - +DI < -DI: Bearish trend

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ADX calculation (typically 14)

    Returns:
        Tuple of (adx, plus_di, minus_di)
        - adx: Average Directional Index (trend strength)
        - plus_di: Positive Directional Indicator
        - minus_di: Negative Directional Indicator

    Example:
        >>> adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
        >>> strong_trend = adx > 25
        >>> bullish = (adx > 25) & (plus_di > minus_di)
    """
    # Convert to pandas Series if numpy arrays
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Calculate True Range (TR)
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Calculate Directional Movement
    high_diff = high - high.shift(1)
    low_diff = low.shift(1) - low

    # Positive Directional Movement (+DM)
    plus_dm = pd.Series(0.0, index=high.index)
    plus_dm[high_diff > low_diff] = high_diff[high_diff > low_diff]
    plus_dm[plus_dm < 0] = 0

    # Negative Directional Movement (-DM)
    minus_dm = pd.Series(0.0, index=low.index)
    minus_dm[low_diff > high_diff] = low_diff[low_diff > high_diff]
    minus_dm[minus_dm < 0] = 0

    # Smooth TR and DM using Wilder's smoothing (exponential moving average)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # Calculate Directional Index (DX)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)

    # Calculate ADX (smoothed DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


def calculate_trend_strength(
    adx: pd.Series,
    threshold_strong: float = 25.0,
    threshold_weak: float = 20.0
) -> pd.Series:
    """
    Categorize trend strength based on ADX values.

    Args:
        adx: ADX values
        threshold_strong: ADX value above which trend is considered strong
        threshold_weak: ADX value below which trend is considered weak

    Returns:
        Series with values: 'strong', 'moderate', 'weak'

    Example:
        >>> adx = calculate_adx(high, low, close)[0]
        >>> strength = calculate_trend_strength(adx)
    """
    strength = pd.Series('moderate', index=adx.index)
    strength[adx >= threshold_strong] = 'strong'
    strength[adx < threshold_weak] = 'weak'

    return strength


def calculate_trend_direction(
    plus_di: pd.Series,
    minus_di: pd.Series,
    min_difference: float = 5.0
) -> pd.Series:
    """
    Determine trend direction from directional indicators.

    Args:
        plus_di: Positive Directional Indicator
        minus_di: Negative Directional Indicator
        min_difference: Minimum difference between +DI and -DI to consider directional

    Returns:
        Series with values: 'bullish', 'bearish', 'neutral'

    Example:
        >>> adx, plus_di, minus_di = calculate_adx(high, low, close)
        >>> direction = calculate_trend_direction(plus_di, minus_di)
    """
    direction = pd.Series('neutral', index=plus_di.index)

    di_diff = plus_di - minus_di
    direction[di_diff > min_difference] = 'bullish'
    direction[di_diff < -min_difference] = 'bearish'

    return direction


def calculate_moving_average_cross(
    fast: pd.Series,
    slow: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect moving average crossovers.

    Args:
        fast: Fast moving average (e.g., EMA 10)
        slow: Slow moving average (e.g., EMA 20)

    Returns:
        Tuple of (bullish_cross, bearish_cross)
        - bullish_cross: True when fast crosses above slow
        - bearish_cross: True when fast crosses below slow

    Example:
        >>> ema_10 = calculate_ema(close, 10)
        >>> ema_20 = calculate_ema(close, 20)
        >>> bull_cross, bear_cross = calculate_moving_average_cross(ema_10, ema_20)
    """
    # Fast above slow
    fast_above = fast > slow

    # Detect crossovers (change in relationship)
    bullish_cross = (fast_above) & (~fast_above.shift(1))
    bearish_cross = (~fast_above) & (fast_above.shift(1))

    return bullish_cross, bearish_cross


# Export all functions
__all__ = [
    'calculate_ema',
    'calculate_sma',
    'calculate_slope',
    'calculate_adx',
    'calculate_trend_strength',
    'calculate_trend_direction',
    'calculate_moving_average_cross',
]
