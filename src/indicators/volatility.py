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


def calculate_keltner_channels_pinescript(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    period: int = 20,
    multiplier: float = 1.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels using PineScript/TradingView method.

    This matches the official TTM Squeeze indicator implementation:
    - Middle Line: SMA of close (not EMA!)
    - Range: SMA of True Range (not Wilder's ATR!)
    - Upper Channel: Middle + (multiplier * SMA(TR))
    - Lower Channel: Middle - (multiplier * SMA(TR))

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for SMA calculation (typically 20)
        multiplier: Multiplier for True Range SMA (typically 1.0, 1.5, or 2.0)

    Returns:
        Tuple of (upper_channel, middle_line, lower_channel)

    Example:
        >>> # Official TTM Squeeze uses 3 levels
        >>> kc_upper_1, kc_mid, kc_lower_1 = calculate_keltner_channels_pinescript(high, low, close, 20, 1.0)
        >>> kc_upper_2, _, kc_lower_2 = calculate_keltner_channels_pinescript(high, low, close, 20, 1.5)
        >>> kc_upper_3, _, kc_lower_3 = calculate_keltner_channels_pinescript(high, low, close, 20, 2.0)
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if isinstance(high, np.ndarray):
        high = pd.Series(high)
    if isinstance(low, np.ndarray):
        low = pd.Series(low)

    # Middle line is SMA of close (PineScript uses SMA, not EMA!)
    middle_line = close.rolling(window=period).mean()

    # Calculate True Range
    tr = calculate_true_range(high, low, close)

    # SMA of True Range (PineScript uses simple average, not Wilder's smoothing!)
    tr_sma = tr.rolling(window=period).mean()

    # Upper and lower channels
    upper_channel = middle_line + (multiplier * tr_sma)
    lower_channel = middle_line - (multiplier * tr_sma)

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


# === DataFrame-based Convenience Functions ===

def calculate_atr_df(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate Average True Range from DataFrame.

    ATR measures market volatility:
    - High ATR = high volatility (larger price swings)
    - Low ATR = low volatility (smaller price swings)

    Args:
        df: DataFrame with OHLC data
        period: Lookback period (default 14)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series with ATR values

    Raises:
        KeyError: If required columns don't exist

    Example:
        >>> atr = calculate_atr_df(df, period=14)
        >>> high_volatility = atr > atr.rolling(50).mean()
    """
    # Validate columns exist
    required = [high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Calculate ATR using Series-based function
    atr = calculate_atr(
        df[high_col], df[low_col], df[close_col], period=period
    )

    atr.name = f'atr_{period}'
    return atr


def calculate_atr_percentage(
    df: pd.DataFrame,
    period: int = 14,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate ATR as percentage of price.

    Normalized ATR for comparing volatility across instruments
    with different price levels.

    Args:
        df: DataFrame with OHLC data
        period: ATR period
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series with ATR percentage (ATR / close * 100)

    Example:
        >>> atr_pct = calculate_atr_percentage(df, period=14)
        >>> print(f"Volatility: {atr_pct.iloc[-1]:.2f}%")
    """
    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not found in DataFrame")

    # Calculate ATR
    atr = calculate_atr_df(df, period, high_col, low_col, close_col)

    # Convert to percentage
    atr_pct = (atr / df[close_col]) * 100
    atr_pct.name = f'atr_pct_{period}'

    return atr_pct


def calculate_atr_percentile(
    df: pd.DataFrame,
    atr_period: int = 14,
    percentile_period: int = 252,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate ATR percentile ranking.

    Shows if current ATR is high or low relative to recent history.
    Percentile of 100 = highest ATR in lookback period.
    Percentile of 0 = lowest ATR in lookback period.

    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation (default 14)
        percentile_period: Lookback for percentile rank (default 252)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series with percentile rank (0-100)

    Example:
        >>> atr_pct_rank = calculate_atr_percentile(df, atr_period=14, percentile_period=252)
        >>> extreme_low_vol = atr_pct_rank < 20  # ATR in bottom 20%
        >>> extreme_high_vol = atr_pct_rank > 80  # ATR in top 20%
    """
    # Calculate ATR
    atr = calculate_atr_df(df, atr_period, high_col, low_col, close_col)

    # Calculate percentile rank over rolling window
    percentile = atr.rolling(window=percentile_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False
    )

    percentile.name = f'atr_percentile_{atr_period}_{percentile_period}'
    return percentile


def calculate_bollinger_bands_df(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands from DataFrame.

    Bollinger Bands expand/contract with volatility:
    - Narrow bands = low volatility (potential breakout)
    - Wide bands = high volatility

    Args:
        df: DataFrame with price data
        period: Period for SMA (default 20)
        std_dev: Standard deviations (default 2.0)
        close_col: Name of close column

    Returns:
        DataFrame with columns 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width'

    Example:
        >>> bb = calculate_bollinger_bands_df(df, period=20)
        >>> price_at_upper = df['close'] > bb['bb_upper']
        >>> squeeze = bb['bb_width'] < bb['bb_width'].rolling(50).mean()
    """
    if close_col not in df.columns:
        raise KeyError(f"Column '{close_col}' not found in DataFrame")

    # Calculate Bollinger Bands using Series-based function
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        df[close_col], period=period, num_std=std_dev
    )

    # Return as DataFrame
    result = pd.DataFrame({
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'bb_width': bb_width
    }, index=df.index)

    return result


def calculate_bollinger_width(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate Bollinger Band width.

    Width = (upper - lower) / middle

    - Narrow bands = low volatility (squeeze building)
    - Wide bands = high volatility (expansion)

    Args:
        df: DataFrame with price data
        period: Bollinger Band period
        std_dev: Standard deviations
        close_col: Name of close column

    Returns:
        Series with BB width

    Example:
        >>> bb_width = calculate_bollinger_width(df, period=20)
        >>> squeeze = bb_width < bb_width.rolling(100).quantile(0.2)
    """
    # Calculate Bollinger Bands
    _, _, _, bb_width = calculate_bollinger_bands(
        df[close_col], period=period, num_std=std_dev
    )

    bb_width.name = f'bb_width_{period}'
    return bb_width


def calculate_keltner_channels_df(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 20,
    atr_multiplier: float = 1.5,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Calculate Keltner Channels from DataFrame.

    Keltner Channels use ATR instead of standard deviation
    (more responsive to volatility changes than Bollinger Bands).

    Args:
        df: DataFrame with OHLC data
        ema_period: Period for EMA (default 20)
        atr_period: Period for ATR (default 20)
        atr_multiplier: ATR multiplier (default 1.5)
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        DataFrame with columns 'kc_upper', 'kc_middle', 'kc_lower'

    Example:
        >>> kc = calculate_keltner_channels_df(df)
        >>> breakout_up = df['close'] > kc['kc_upper']
        >>> breakout_down = df['close'] < kc['kc_lower']
    """
    # Validate columns
    required = [high_col, low_col, close_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Calculate Keltner Channels using Series-based function
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        df[high_col], df[low_col], df[close_col],
        period=ema_period,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier
    )

    # Return as DataFrame
    result = pd.DataFrame({
        'kc_upper': kc_upper,
        'kc_middle': kc_middle,
        'kc_lower': kc_lower
    }, index=df.index)

    return result


def calculate_keltner_width(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 20,
    atr_multiplier: float = 1.5,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.Series:
    """
    Calculate Keltner Channel width.

    Width = (upper - lower) / middle

    Args:
        df: DataFrame with OHLC data
        ema_period: Period for EMA
        atr_period: Period for ATR
        atr_multiplier: ATR multiplier
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        Series with KC width

    Example:
        >>> kc_width = calculate_keltner_width(df)
        >>> volatility_expansion = kc_width > kc_width.rolling(50).mean()
    """
    # Calculate Keltner Channels
    kc = calculate_keltner_channels_df(
        df, ema_period, atr_period, atr_multiplier,
        high_col, low_col, close_col
    )

    # Calculate width
    width = (kc['kc_upper'] - kc['kc_lower']) / kc['kc_middle']
    width.name = f'kc_width_{ema_period}_{atr_period}'

    return width


def add_volatility_indicators(
    df: pd.DataFrame,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_ema_period: int = 20,
    kc_atr_period: int = 20,
    kc_multiplier: float = 1.5,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close'
) -> pd.DataFrame:
    """
    Add all volatility indicators to DataFrame in one call.

    This convenience function calculates and adds:
    - ATR and ATR percentage
    - Bollinger Bands (upper, middle, lower, width)
    - Keltner Channels (upper, middle, lower)

    Args:
        df: DataFrame with OHLC data
        atr_period: Period for ATR calculation
        bb_period: Period for Bollinger Bands
        bb_std: Bollinger Bands standard deviations
        kc_ema_period: Keltner Channel EMA period
        kc_atr_period: Keltner Channel ATR period
        kc_multiplier: Keltner Channel ATR multiplier
        high_col: Name of high column
        low_col: Name of low column
        close_col: Name of close column

    Returns:
        DataFrame with original data plus volatility indicators

    Example:
        >>> df = add_volatility_indicators(df)
        >>> print(df.columns)
        Index(['high', 'low', 'close', 'atr_14', 'atr_pct_14',
               'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
               'kc_upper', 'kc_middle', 'kc_lower'], dtype='object')
    """
    # Create copy to avoid modifying original
    result = df.copy()

    # Add ATR
    atr = calculate_atr_df(df, atr_period, high_col, low_col, close_col)
    result[atr.name] = atr

    # Add ATR percentage
    atr_pct = calculate_atr_percentage(df, atr_period, high_col, low_col, close_col)
    result[atr_pct.name] = atr_pct

    # Add Bollinger Bands
    bb = calculate_bollinger_bands_df(df, bb_period, bb_std, close_col)
    result = pd.concat([result, bb], axis=1)

    # Add Keltner Channels
    kc = calculate_keltner_channels_df(
        df, kc_ema_period, kc_atr_period, kc_multiplier,
        high_col, low_col, close_col
    )
    result = pd.concat([result, kc], axis=1)

    return result


# Export all functions
__all__ = [
    # Series-based functions (used by regime detector and TTM)
    'calculate_true_range',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_keltner_channels',
    'calculate_volatility_ratio',
    'calculate_band_position',
    'detect_squeeze',
    'calculate_squeeze_duration',
    'calculate_historical_volatility',
    # DataFrame-based convenience functions
    'calculate_atr_df',
    'calculate_atr_percentage',
    'calculate_atr_percentile',
    'calculate_bollinger_bands_df',
    'calculate_bollinger_width',
    'calculate_keltner_channels_df',
    'calculate_keltner_width',
    'add_volatility_indicators',
]
