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


# === DataFrame-based Convenience Functions ===

def calculate_adx_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate ADX with directional indicators from DataFrame.

    ADX measures trend strength (0-100):
    - 0-20: No trend / weak trend
    - 20-40: Strong trend emerging
    - 40+: Very strong trend

    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default 14)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        DataFrame with columns 'adx', 'plus_di', 'minus_di'

    Raises:
        KeyError: If required columns don't exist

    Example:
        >>> df_adx = calculate_adx_df(df, period=14)
        >>> strong_trend = df_adx['adx'] > 25
    """
    # Validate columns exist
    required = [high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Calculate ADX using Series-based function
    adx, plus_di, minus_di = calculate_adx(
        df[high_col], df[low_col], df[close_col], period=period
    )

    # Return as DataFrame
    result = pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }, index=df.index)

    return result


def calculate_ema_slope(
    df: pd.DataFrame,
    ema_period: int,
    slope_lookback: int = 20,
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate slope of EMA (rate of change).

    Positive slope = uptrend
    Negative slope = downtrend

    Args:
        df: DataFrame with price data
        ema_period: Period for EMA calculation
        slope_lookback: Bars to look back for slope calculation
        close_col: Name of close column

    Returns:
        Series with normalized slope (percentage)

    Example:
        >>> slope = calculate_ema_slope(df, ema_period=20, slope_lookback=20)
        >>> uptrend = slope > 0.1  # Positive slope > 0.1%
    """
    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not found in DataFrame")

    # Calculate EMA
    ema = calculate_ema(df[close_col], period=ema_period)

    # Calculate slope as percentage change
    ema_lookback = ema.shift(slope_lookback)
    slope = ((ema - ema_lookback) / df[close_col]) * 100

    slope.name = f'ema_{ema_period}_slope'
    return slope


def calculate_multiple_emas(
    df: pd.DataFrame,
    periods: list = None,
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate multiple EMAs at once.

    Args:
        df: DataFrame with price data
        periods: List of periods (default [20, 50, 200])
        close_col: Name of close column

    Returns:
        DataFrame with columns 'ema_20', 'ema_50', 'ema_200'

    Example:
        >>> df_emas = calculate_multiple_emas(df, periods=[20, 50, 200])
        >>> print(df_emas.columns)
        Index(['ema_20', 'ema_50', 'ema_200'], dtype='object')
    """
    if periods is None:
        periods = [20, 50, 200]

    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not found in DataFrame")

    result = pd.DataFrame(index=df.index)

    for period in periods:
        ema = calculate_ema(df[close_col], period=period)
        result[f'ema_{period}'] = ema

    return result


def check_ema_alignment(
    df: pd.DataFrame,
    periods: list = None,
    close_col: str = 'close'
) -> pd.Series:
    """
    Check if EMAs are aligned (bullish or bearish).

    Bullish alignment: EMA20 > EMA50 > EMA200 (fast above slow)
    Bearish alignment: EMA20 < EMA50 < EMA200 (fast below slow)

    Args:
        df: DataFrame with price data
        periods: List of periods (default [20, 50, 200])
        close_col: Name of close column

    Returns:
        Series with values: 'bullish', 'bearish', 'mixed'

    Example:
        >>> alignment = check_ema_alignment(df, periods=[20, 50, 200])
        >>> bullish_periods = alignment == 'bullish'
        >>> print(f"Bullish alignment: {bullish_periods.sum()} bars")
    """
    if periods is None:
        periods = [20, 50, 200]

    if len(periods) < 2:
        raise ValueError("Need at least 2 periods to check alignment")

    # Calculate EMAs
    emas = calculate_multiple_emas(df, periods=periods, close_col=close_col)

    # Sort periods to get fast to slow
    sorted_periods = sorted(periods)
    ema_cols = [f'ema_{p}' for p in sorted_periods]

    # Initialize alignment series
    alignment = pd.Series('mixed', index=df.index)

    # Check bullish alignment (fast > slow progressively)
    bullish = pd.Series(True, index=df.index)
    for i in range(len(ema_cols) - 1):
        bullish &= (emas[ema_cols[i]] > emas[ema_cols[i + 1]])

    # Check bearish alignment (fast < slow progressively)
    bearish = pd.Series(True, index=df.index)
    for i in range(len(ema_cols) - 1):
        bearish &= (emas[ema_cols[i]] < emas[ema_cols[i + 1]])

    # Set alignment
    alignment[bullish] = 'bullish'
    alignment[bearish] = 'bearish'

    alignment.name = 'ema_alignment'
    return alignment


def add_trend_indicators(
    df: pd.DataFrame,
    adx_period: int = 14,
    ema_periods: list = None,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Add all trend indicators to DataFrame in one call.

    This is a convenience function that calculates and adds:
    - ADX, +DI, -DI
    - Multiple EMAs (20, 50, 200)
    - EMA alignment

    Args:
        df: DataFrame with OHLC data
        adx_period: Period for ADX calculation
        ema_periods: List of EMA periods (default [20, 50, 200])
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        DataFrame with original data plus trend indicators

    Example:
        >>> df = add_trend_indicators(df)
        >>> print(df.columns)
        Index(['high', 'low', 'close', 'adx', 'plus_di', 'minus_di',
               'ema_20', 'ema_50', 'ema_200', 'ema_alignment'], dtype='object')
    """
    if ema_periods is None:
        ema_periods = [20, 50, 200]

    # Create copy to avoid modifying original
    result = df.copy()

    # Add ADX indicators
    adx_df = calculate_adx_df(
        df, period=adx_period,
        high_col=high_col, low_col=low_col, close_col=close_col
    )
    result = pd.concat([result, adx_df], axis=1)

    # Add EMAs
    emas_df = calculate_multiple_emas(df, periods=ema_periods, close_col=close_col)
    result = pd.concat([result, emas_df], axis=1)

    # Add EMA alignment
    alignment = check_ema_alignment(df, periods=ema_periods, close_col=close_col)
    result['ema_alignment'] = alignment

    return result


# Export all functions
__all__ = [
    # Series-based functions (used by regime detector)
    'calculate_ema',
    'calculate_sma',
    'calculate_slope',
    'calculate_adx',
    'calculate_trend_strength',
    'calculate_trend_direction',
    'calculate_moving_average_cross',
    # DataFrame-based convenience functions
    'calculate_adx_df',
    'calculate_ema_slope',
    'calculate_multiple_emas',
    'check_ema_alignment',
    'add_trend_indicators',
]
