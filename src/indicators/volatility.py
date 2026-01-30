"""
Volatility Indicators Module

This module implements volatility-based indicators for measuring
market volatility and identifying potential breakout/breakdown conditions.

Indicators:
    - ATR (Average True Range): Measures market volatility
    - Bollinger Bands: Volatility bands based on standard deviation
    - Keltner Channels: Volatility bands based on ATR
    - Volatility Ratio: Normalized volatility measure

Usage:
    from src.indicators.volatility import (
        calculate_atr,
        calculate_bollinger_bands,
        calculate_keltner_channels
    )

    # Calculate ATR
    atr = calculate_atr(high, low, close, period=14)

    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, period=20)

    # Calculate Keltner Channels
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(high, low, close)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union


def calculate_true_range(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """
    Calculate True Range (TR).

    True Range is the greatest of:
    - Current High - Current Low
    - Abs(Current High - Previous Close)
    - Abs(Current Low - Previous Close)

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        Pandas Series containing True Range values
    """
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return tr


def calculate_atr(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range
    of price movement. Higher ATR = higher volatility.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation (typically 14)

    Returns:
        Pandas Series containing ATR values

    Example:
        >>> atr = calculate_atr(high, low, close, period=14)
        >>> high_volatility = atr > atr.rolling(50).mean()
    """
    tr = calculate_true_range(high, low, close)

    # Use Wilder's smoothing (exponential moving average)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calculate_bollinger_bands(
    data: Union[pd.Series, np.ndarray],
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of:
    - Middle Band: Simple moving average
    - Upper Band: Middle + (num_std * standard deviation)
    - Lower Band: Middle - (num_std * standard deviation)

    Interpretation:
    - Price near upper band: Potentially overbought
    - Price near lower band: Potentially oversold
    - Band squeeze: Low volatility, potential breakout coming
    - Band expansion: High volatility

    Args:
        data: Price data (typically close prices)
        period: Period for moving average and std dev calculation
        num_std: Number of standard deviations for bands (typically 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band, band_width)

    Example:
        >>> bb_upper, bb_mid, bb_lower, bb_width = calculate_bollinger_bands(close)
        >>> squeeze = bb_width < bb_width.rolling(50).mean()
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    # Middle band is the SMA
    middle_band = data.rolling(window=period).mean()

    # Calculate standard deviation
    std = data.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)

    # Band width (normalized measure of volatility)
    band_width = (upper_band - lower_band) / middle_band

    return upper_band, middle_band, lower_band, band_width


def calculate_keltner_channels(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 20,
    atr_period: int = 10,
    atr_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.

    Keltner Channels are volatility-based bands:
    - Middle Line: EMA of close
    - Upper Channel: Middle + (atr_multiplier * ATR)
    - Lower Channel: Middle - (atr_multiplier * ATR)

    Similar to Bollinger Bands but uses ATR instead of standard deviation.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for EMA calculation (typically 20)
        atr_period: Period for ATR calculation (typically 10)
        atr_multiplier: Multiplier for ATR (typically 2.0)

    Returns:
        Tuple of (upper_channel, middle_line, lower_channel)

    Example:
        >>> kc_upper, kc_mid, kc_lower = calculate_keltner_channels(high, low, close)
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Middle line is EMA of close
    from src.indicators.trend import calculate_ema
    middle_line = calculate_ema(close, period=period)

    # Calculate ATR
    atr = calculate_atr(high, low, close, period=atr_period)

    # Upper and lower channels
    upper_channel = middle_line + (atr_multiplier * atr)
    lower_channel = middle_line - (atr_multiplier * atr)

    return upper_channel, middle_line, lower_channel


def calculate_volatility_ratio(
    atr: pd.Series,
    close: Union[pd.Series, np.ndarray],
    period: int = 50
) -> pd.Series:
    """
    Calculate normalized volatility ratio.

    Compares current ATR to historical average ATR to identify
    high/low volatility periods.

    Args:
        atr: Average True Range values
        close: Close prices (for normalization)
        period: Period for historical average (typically 50)

    Returns:
        Pandas Series with volatility ratio
        - Ratio > 1.0: Above average volatility
        - Ratio < 1.0: Below average volatility

    Example:
        >>> atr = calculate_atr(high, low, close)
        >>> vol_ratio = calculate_volatility_ratio(atr, close)
        >>> high_vol = vol_ratio > 1.2
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Normalize ATR by price to get percentage volatility
    atr_pct = (atr / close) * 100

    # Compare to historical average
    atr_avg = atr_pct.rolling(window=period).mean()
    volatility_ratio = atr_pct / atr_avg

    return volatility_ratio


def calculate_band_position(
    close: Union[pd.Series, np.ndarray],
    upper_band: pd.Series,
    lower_band: pd.Series
) -> pd.Series:
    """
    Calculate price position within bands.

    Returns a value between 0 and 1:
    - 1.0: Price at upper band
    - 0.5: Price at middle of bands
    - 0.0: Price at lower band

    Args:
        close: Close prices
        upper_band: Upper band values (BB or KC)
        lower_band: Lower band values (BB or KC)

    Returns:
        Pandas Series with band position (0 to 1)

    Example:
        >>> bb_upper, _, bb_lower, _ = calculate_bollinger_bands(close)
        >>> position = calculate_band_position(close, bb_upper, bb_lower)
        >>> overbought = position > 0.8
        >>> oversold = position < 0.2
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    band_range = upper_band - lower_band
    price_from_lower = close - lower_band

    # Avoid division by zero
    band_position = price_from_lower / band_range.replace(0, np.nan)
    band_position = band_position.clip(0, 1)  # Clamp to [0, 1]

    return band_position


def detect_squeeze(
    bb_upper: pd.Series,
    bb_lower: pd.Series,
    kc_upper: pd.Series,
    kc_lower: pd.Series
) -> pd.Series:
    """
    Detect squeeze condition (Bollinger Bands inside Keltner Channels).

    A squeeze occurs when Bollinger Bands contract inside Keltner Channels,
    indicating very low volatility. This often precedes a significant move.

    Args:
        bb_upper: Bollinger Bands upper band
        bb_lower: Bollinger Bands lower band
        kc_upper: Keltner Channels upper band
        kc_lower: Keltner Channels lower band

    Returns:
        Boolean Series: True when squeeze is active

    Example:
        >>> bb_upper, _, bb_lower, _ = calculate_bollinger_bands(close)
        >>> kc_upper, _, kc_lower = calculate_keltner_channels(high, low, close)
        >>> squeeze = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
    """
    # Squeeze is active when BBs are completely inside KCs
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    return squeeze


def calculate_squeeze_duration(squeeze: pd.Series) -> pd.Series:
    """
    Calculate how long the squeeze has been active.

    Args:
        squeeze: Boolean series indicating squeeze condition

    Returns:
        Series with number of consecutive periods in squeeze

    Example:
        >>> squeeze = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)
        >>> duration = calculate_squeeze_duration(squeeze)
        >>> long_squeeze = duration > 20  # Squeeze lasting > 20 bars
    """
    # Create groups of consecutive True values
    squeeze_groups = (squeeze != squeeze.shift()).cumsum()

    # Count consecutive periods only when squeeze is True
    duration = squeeze.groupby(squeeze_groups).cumsum()
    duration[~squeeze] = 0

    return duration


def calculate_historical_volatility(
    returns: Union[pd.Series, np.ndarray],
    period: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculate historical volatility (standard deviation of returns).

    Args:
        returns: Percentage returns
        period: Rolling window period
        annualize: If True, annualize the volatility
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly)

    Returns:
        Pandas Series with historical volatility

    Example:
        >>> returns = close.pct_change()
        >>> hvol = calculate_historical_volatility(returns, period=20)
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    # Calculate rolling standard deviation
    hvol = returns.rolling(window=period).std()

    # Annualize if requested
    if annualize:
        hvol = hvol * np.sqrt(periods_per_year)

    return hvol


# Export all functions
__all__ = [
    'calculate_true_range',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_keltner_channels',
    'calculate_volatility_ratio',
    'calculate_band_position',
    'detect_squeeze',
    'calculate_squeeze_duration',
    'calculate_historical_volatility',
]
