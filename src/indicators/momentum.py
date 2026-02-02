"""
Momentum Indicators Module

Provides momentum and oscillator indicators like RSI, MACD, Stochastic, etc.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes.
    Values range from 0-100:
        - RSI > 70: Overbought (potential reversal)
        - RSI < 30: Oversold (potential reversal)
        - RSI > 50: Bullish momentum
        - RSI < 50: Bearish momentum

    Args:
        close: Close prices
        period: RSI period (default: 14)

    Returns:
        RSI values (0-100)

    Formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss

    Example:
        >>> rsi_14 = calculate_rsi(df['close'], period=14)
        >>> overbought = rsi_14 > 70
        >>> oversold = rsi_14 < 30
    """
    # Calculate price changes
    delta = close.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate exponential moving average of gains and losses
    # Using Wilder's smoothing method (equivalent to EWM with alpha=1/period)
    avg_gains = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_rsi_df(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.DataFrame:
    """
    Calculate RSI and add to DataFrame.

    Args:
        df: DataFrame with OHLC data
        period: RSI period (default: 14)
        column: Column to calculate RSI from (default: 'close')

    Returns:
        DataFrame with RSI column added

    Example:
        >>> df = calculate_rsi_df(df, period=14)
        >>> print(df[['close', 'rsi_14']].tail())
    """
    df = df.copy()
    df[f'rsi_{period}'] = calculate_rsi(df[column], period=period)
    return df


def is_rsi_overbought(rsi: float, threshold: float = 70.0) -> bool:
    """Check if RSI is in overbought territory."""
    return rsi > threshold


def is_rsi_oversold(rsi: float, threshold: float = 30.0) -> bool:
    """Check if RSI is in oversold territory."""
    return rsi < threshold


def is_rsi_bullish(rsi: float) -> bool:
    """Check if RSI shows bullish momentum (> 50)."""
    return rsi > 50.0


def is_rsi_bearish(rsi: float) -> bool:
    """Check if RSI shows bearish momentum (< 50)."""
    return rsi < 50.0


def detect_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 14
) -> Tuple[bool, bool]:
    """
    Detect bullish and bearish RSI divergences.

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high

    Args:
        close: Close prices
        rsi: RSI values
        lookback: Periods to look back for divergence

    Returns:
        (bullish_divergence, bearish_divergence)

    Example:
        >>> bullish_div, bearish_div = detect_rsi_divergence(df['close'], df['rsi_14'])
    """
    if len(close) < lookback + 1:
        return False, False

    # Get recent data
    recent_close = close.iloc[-lookback:]
    recent_rsi = rsi.iloc[-lookback:]

    # Find local lows and highs
    close_min_idx = recent_close.idxmin()
    close_max_idx = recent_close.idxmax()
    rsi_min_idx = recent_rsi.idxmin()
    rsi_max_idx = recent_rsi.idxmax()

    # Bullish divergence: price lower low, RSI higher low
    bullish_div = False
    if close_min_idx == recent_close.index[-1]:  # Recent low in price
        prev_close_lows = recent_close.iloc[:-1]
        prev_rsi_lows = recent_rsi.iloc[:-1]
        if len(prev_close_lows) > 0:
            if recent_close.iloc[-1] < prev_close_lows.min() and recent_rsi.iloc[-1] > prev_rsi_lows.min():
                bullish_div = True

    # Bearish divergence: price higher high, RSI lower high
    bearish_div = False
    if close_max_idx == recent_close.index[-1]:  # Recent high in price
        prev_close_highs = recent_close.iloc[:-1]
        prev_rsi_highs = recent_rsi.iloc[:-1]
        if len(prev_close_highs) > 0:
            if recent_close.iloc[-1] > prev_close_highs.max() and recent_rsi.iloc[-1] < prev_rsi_highs.max():
                bearish_div = True

    return bullish_div, bearish_div
