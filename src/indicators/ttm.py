"""
TTM Squeeze Indicator Module

This module implements John Carter's TTM Squeeze indicator, which identifies
periods of low volatility (squeeze) followed by potential explosive moves (release).

The TTM Squeeze combines:
- Bollinger Bands and Keltner Channels for volatility compression detection
- Linear regression momentum for directional bias
- Histogram visualization showing momentum strength and direction

Indicator States:
- Squeeze ON: Bollinger Bands inside Keltner Channels (low volatility)
- Squeeze OFF: Bollinger Bands outside Keltner Channels (normal/high volatility)
- Momentum: Colored bars showing directional momentum and strength

Usage:
    from src.indicators.ttm import calculate_ttm_squeeze

    # Calculate TTM Squeeze
    squeeze_on, momentum, momentum_color = calculate_ttm_squeeze(
        high, low, close, bb_period=20, kc_period=20
    )

    # Identify squeeze release (potential trade setup)
    squeeze_release = (squeeze_on.shift(1)) & (~squeeze_on)

References:
    - John Carter's "Mastering the Trade"
    - TTM Squeeze indicator (Thinkorswim platform)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union

from src.indicators.volatility import (
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_keltner_channels_pinescript,
    detect_squeeze
)


def calculate_momentum(
    close: Union[pd.Series, np.ndarray],
    period: int = 12,
    normalize: bool = True
) -> pd.Series:
    """
    Calculate linear regression momentum for TTM Squeeze.

    This momentum calculation uses linear regression to smooth price
    and then calculates the difference between current price and
    the regression line.

    Args:
        close: Close prices
        period: Period for momentum calculation (typically 12)
        normalize: If True, normalize by ATR or price range

    Returns:
        Pandas Series containing momentum values

    Example:
        >>> momentum = calculate_momentum(close, period=12)
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)

    # Calculate highest high and lowest low over period
    highest = close.rolling(window=period).max()
    lowest = close.rolling(window=period).min()
    avg = (highest + lowest) / 2

    # Linear regression of (close - avg)
    # This shows momentum relative to the middle of the range
    momentum_values = pd.Series(index=close.index, dtype=float)

    for i in range(period - 1, len(close)):
        # Get data window
        y = (close.iloc[i - period + 1:i + 1] - avg.iloc[i - period + 1:i + 1]).values
        x = np.arange(period)

        # Fit linear regression
        if len(y) == period:
            # y = mx + b, we want the predicted value at the end
            coeffs = np.polyfit(x, y, 1)
            momentum_values.iloc[i] = coeffs[0] * (period - 1) + coeffs[1]

    if normalize:
        # Normalize by recent price range
        price_range = highest - lowest
        momentum_values = momentum_values / price_range.replace(0, np.nan)

    return momentum_values


def calculate_ttm_squeeze(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_period: int = 20,
    kc_multiplier: float = 1.5,
    momentum_period: int = 12
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate complete TTM Squeeze indicator.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        bb_period: Bollinger Bands period (typically 20)
        bb_std: Bollinger Bands standard deviations (typically 2.0)
        kc_period: Keltner Channels EMA period (typically 20)
        kc_atr_period: Keltner Channels ATR period (typically 20)
        kc_multiplier: Keltner Channels ATR multiplier (typically 1.5)
        momentum_period: Momentum calculation period (typically 12)

    Returns:
        Tuple of (squeeze_on, momentum, momentum_color)
        - squeeze_on: Boolean series (True = squeeze active)
        - momentum: Momentum values (histogram height)
        - momentum_color: Color indicator for histogram
          * 'lime': Increasing positive momentum
          * 'green': Decreasing positive momentum
          * 'red': Decreasing negative momentum
          * 'dark_red': Increasing negative momentum

    Example:
        >>> squeeze_on, momentum, color = calculate_ttm_squeeze(high, low, close)
        >>> # Find squeeze release with bullish momentum
        >>> release = (~squeeze_on) & (squeeze_on.shift(1))
        >>> bullish_setup = release & (momentum > 0) & (color == 'lime')
    """
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, period=bb_period, num_std=bb_std
    )

    # Calculate Keltner Channels
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
        high, low, close,
        period=kc_period,
        atr_period=kc_atr_period,
        atr_multiplier=kc_multiplier
    )

    # Detect squeeze
    squeeze_on = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

    # Calculate momentum
    momentum = calculate_momentum(close, period=momentum_period)

    # Determine momentum color (trend direction)
    momentum_color = calculate_momentum_color(momentum)

    return squeeze_on, momentum, momentum_color


def calculate_ttm_squeeze_pinescript(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    momentum_period: int = 12
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate TTM Squeeze using Beardy Squeeze Pro / John Carter method.

    Uses three Keltner Channel levels (multipliers 1.0, 1.5, 2.0) to
    determine squeeze intensity, matching Beardy Squeeze Pro exactly:
    - Bollinger Bands: SMA(close, 20) ± 2.0 * stdev
    - KC Level 1: SMA(close, 20) ± 1.0 * SMA(TR, 20)  (narrowest)
    - KC Level 2: SMA(close, 20) ± 1.5 * SMA(TR, 20)
    - KC Level 3: SMA(close, 20) ± 2.0 * SMA(TR, 20)  (widest)

    Squeeze states (matching Beardy Squeeze Pro dot colors):
    - No squeeze (green dots): BB outside all KC levels → intensity 0
    - Light squeeze (black dots): BB inside KC×2.0 only → intensity 1
    - Medium squeeze (red dots): BB inside KC×1.5 → intensity 2
    - Tight squeeze (orange dots): BB inside KC×1.0 → intensity 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        bb_period: Bollinger Bands period (typically 20)
        bb_std: Bollinger Bands standard deviations (typically 2.0)
        kc_period: Keltner Channels period (typically 20)
        momentum_period: Momentum calculation period (typically 12)

    Returns:
        Tuple of (squeeze_on, momentum, momentum_color, squeeze_intensity)
        - squeeze_on: Boolean (True when BB inside any KC level)
        - momentum: Momentum values
        - momentum_color: Color indicator
        - squeeze_intensity: 0=none(green), 1=light(black), 2=medium(red), 3=tight(orange)

    Example:
        >>> squeeze_on, momentum, color, intensity = calculate_ttm_squeeze_pinescript(
        ...     high, low, close, bb_period=20, bb_std=2.0, kc_period=20
        ... )
        >>> # Tight squeeze (orange dots) about to release
        >>> tight_release = squeeze_on.shift(1) & ~squeeze_on & (intensity.shift(1) == 3)
    """
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, period=bb_period, num_std=bb_std
    )

    # Calculate three Keltner Channel levels (Beardy Squeeze Pro)
    kc_upper_1, kc_middle, kc_lower_1 = calculate_keltner_channels_pinescript(
        high, low, close, period=kc_period, multiplier=1.0
    )
    kc_upper_15, _, kc_lower_15 = calculate_keltner_channels_pinescript(
        high, low, close, period=kc_period, multiplier=1.5
    )
    kc_upper_2, _, kc_lower_2 = calculate_keltner_channels_pinescript(
        high, low, close, period=kc_period, multiplier=2.0
    )

    # Detect squeeze at each level
    squeeze_narrow = detect_squeeze(bb_upper, bb_lower, kc_upper_1, kc_lower_1)   # Orange dots (tightest)
    squeeze_mid = detect_squeeze(bb_upper, bb_lower, kc_upper_15, kc_lower_15)    # Red dots
    squeeze_wide = detect_squeeze(bb_upper, bb_lower, kc_upper_2, kc_lower_2)     # Black dots (first level)

    # squeeze_on = BB inside widest KC (any squeeze active, black dots or deeper)
    squeeze_on = squeeze_wide

    # Squeeze intensity: 0=green, 1=black, 2=red, 3=orange
    squeeze_intensity = (
        squeeze_wide.astype(int) +
        squeeze_mid.astype(int) +
        squeeze_narrow.astype(int)
    )

    # Calculate momentum
    momentum = calculate_momentum(close, period=momentum_period)

    # Determine momentum color
    momentum_color = calculate_momentum_color(momentum)

    return squeeze_on, momentum, momentum_color, squeeze_intensity


def calculate_momentum_color(momentum: pd.Series) -> pd.Series:
    """
    Calculate color indicator for TTM Squeeze momentum histogram.

    Color scheme:
    - Lime: Positive momentum increasing (bullish acceleration)
    - Green: Positive momentum decreasing (bullish deceleration)
    - Red: Negative momentum decreasing (bearish deceleration)
    - Dark Red: Negative momentum increasing (bearish acceleration)

    Args:
        momentum: Momentum values from calculate_momentum()

    Returns:
        Series with color values

    Example:
        >>> momentum = calculate_momentum(close)
        >>> colors = calculate_momentum_color(momentum)
    """
    colors = pd.Series('gray', index=momentum.index)

    # Calculate momentum change
    momentum_increasing = momentum > momentum.shift(1)

    # Positive momentum
    colors[(momentum > 0) & momentum_increasing] = 'lime'
    colors[(momentum > 0) & ~momentum_increasing] = 'green'

    # Negative momentum
    colors[(momentum < 0) & ~momentum_increasing] = 'red'
    colors[(momentum < 0) & momentum_increasing] = 'dark_red'

    return colors


def identify_squeeze_setups(
    squeeze_on: pd.Series,
    momentum: pd.Series,
    momentum_color: pd.Series,
    min_squeeze_duration: int = 6
) -> Tuple[pd.Series, pd.Series]:
    """
    Identify potential trading setups from TTM Squeeze.

    A setup occurs when:
    1. Squeeze has been active for minimum duration
    2. Squeeze releases (turns off)
    3. Momentum shows directional strength

    Args:
        squeeze_on: Boolean series indicating squeeze state
        momentum: Momentum values
        momentum_color: Momentum color indicator
        min_squeeze_duration: Minimum squeeze bars before considering setup

    Returns:
        Tuple of (bullish_setup, bearish_setup)

    Example:
        >>> squeeze_on, momentum, color = calculate_ttm_squeeze(high, low, close)
        >>> bull_setup, bear_setup = identify_squeeze_setups(
        ...     squeeze_on, momentum, color, min_squeeze_duration=6
        ... )
    """
    from src.indicators.volatility import calculate_squeeze_duration

    # Calculate squeeze duration
    duration = calculate_squeeze_duration(squeeze_on)

    # Identify squeeze release (was on, now off)
    squeeze_release = (squeeze_on.shift(1)) & (~squeeze_on)

    # Release must come after sufficient squeeze duration
    valid_release = squeeze_release & (duration.shift(1) >= min_squeeze_duration)

    # Bullish setup: Release with positive accelerating momentum
    bullish_setup = valid_release & (momentum > 0) & (momentum_color == 'lime')

    # Bearish setup: Release with negative accelerating momentum
    bearish_setup = valid_release & (momentum < 0) & (momentum_color == 'dark_red')

    return bullish_setup, bearish_setup


def calculate_squeeze_strength(
    bb_width: pd.Series,
    lookback: int = 100
) -> pd.Series:
    """
    Calculate squeeze strength relative to historical volatility.

    A stronger squeeze (lower volatility relative to history) often
    leads to a more explosive move when it releases.

    Args:
        bb_width: Bollinger Band width from calculate_bollinger_bands()
        lookback: Lookback period for historical comparison

    Returns:
        Series with squeeze strength (0 to 1)
        - 0: Current BB width at historical maximum (no squeeze)
        - 1: Current BB width at historical minimum (strongest squeeze)

    Example:
        >>> _, _, _, bb_width = calculate_bollinger_bands(close)
        >>> strength = calculate_squeeze_strength(bb_width)
        >>> strong_squeeze = strength > 0.7
    """
    # Calculate percentile rank of BB width
    rolling_min = bb_width.rolling(window=lookback).min()
    rolling_max = bb_width.rolling(window=lookback).max()

    # Invert: lower width = stronger squeeze
    strength = 1 - ((bb_width - rolling_min) / (rolling_max - rolling_min))
    strength = strength.fillna(0.5)  # Neutral if not enough data

    return strength


def get_ttm_signals(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    """
    Calculate complete TTM Squeeze with all signals and metadata.

    This is a convenience function that returns a DataFrame with
    all TTM Squeeze components for easy analysis.

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        DataFrame with columns:
        - squeeze_on: Boolean squeeze indicator
        - momentum: Momentum values
        - momentum_color: Color indicator
        - squeeze_duration: Consecutive bars in squeeze
        - squeeze_strength: Relative squeeze strength
        - bullish_setup: Bullish trade setup signal
        - bearish_setup: Bearish trade setup signal

    Example:
        >>> signals = get_ttm_signals(high, low, close)
        >>> # Filter for high-probability bullish setups
        >>> high_prob = signals[
        ...     (signals['bullish_setup']) &
        ...     (signals['squeeze_strength'] > 0.7) &
        ...     (signals['squeeze_duration'] > 10)
        ... ]
    """
    from src.indicators.volatility import calculate_squeeze_duration

    # Calculate base TTM Squeeze
    squeeze_on, momentum, momentum_color = calculate_ttm_squeeze(high, low, close)

    # Calculate Bollinger Bands for width
    _, _, _, bb_width = calculate_bollinger_bands(close, period=20)

    # Calculate additional metrics
    squeeze_duration = calculate_squeeze_duration(squeeze_on)
    squeeze_strength = calculate_squeeze_strength(bb_width)

    # Identify setups
    bullish_setup, bearish_setup = identify_squeeze_setups(
        squeeze_on, momentum, momentum_color
    )

    # Combine into DataFrame
    signals = pd.DataFrame({
        'squeeze_on': squeeze_on,
        'momentum': momentum,
        'momentum_color': momentum_color,
        'squeeze_duration': squeeze_duration,
        'squeeze_strength': squeeze_strength,
        'bullish_setup': bullish_setup,
        'bearish_setup': bearish_setup
    })

    return signals


# Export all functions
__all__ = [
    'calculate_momentum',
    'calculate_ttm_squeeze',
    'calculate_ttm_squeeze_pinescript',
    'calculate_momentum_color',
    'identify_squeeze_setups',
    'calculate_squeeze_strength',
    'get_ttm_signals',
]
