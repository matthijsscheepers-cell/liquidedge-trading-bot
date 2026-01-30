"""
Market Regime Detector Module

This module identifies the current market regime by analyzing trend strength,
volatility, and momentum indicators. Different market regimes require different
trading strategies for optimal performance.

Market Regimes:
    - Trending: Strong directional movement (ADX > 25)
        * Bullish Trend: +DI > -DI, positive momentum
        * Bearish Trend: -DI > +DI, negative momentum
    - Ranging: Sideways movement with low trend strength (ADX < 20)
    - Volatile: High volatility without clear direction
    - Squeeze: Low volatility compression (potential breakout setup)

The detector combines multiple indicators:
    - ADX: Trend strength measurement
    - Volatility: ATR and Bollinger Band width
    - TTM Squeeze: Compression/expansion cycles
    - Momentum: Directional bias

Usage:
    from src.regime.detector import RegimeDetector

    # Initialize detector
    detector = RegimeDetector()

    # Detect regime
    regime = detector.detect(high, low, close)

    # Access regime components
    print(f"Current regime: {regime['regime']}")
    print(f"Trend strength: {regime['trend_strength']}")
    print(f"Volatility state: {regime['volatility_state']}")
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
from dataclasses import dataclass
from enum import Enum

from src.indicators import (
    calculate_adx,
    calculate_trend_strength,
    calculate_trend_direction,
    calculate_atr,
    calculate_volatility_ratio,
    calculate_bollinger_bands,
    calculate_ttm_squeeze,
    calculate_squeeze_duration,
)


class RegimeType(Enum):
    """Market regime classifications."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    SQUEEZE = "squeeze"
    UNDEFINED = "undefined"


class TrendStrength(Enum):
    """Trend strength classifications."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class VolatilityState(Enum):
    """Volatility state classifications."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SQUEEZE = "squeeze"


@dataclass
class RegimeConfig:
    """Configuration for regime detection parameters."""
    # ADX parameters
    adx_period: int = 14
    adx_strong_threshold: float = 25.0
    adx_weak_threshold: float = 20.0

    # Directional indicator parameters
    di_difference_threshold: float = 5.0

    # Volatility parameters
    atr_period: int = 14
    volatility_lookback: int = 50
    volatility_high_threshold: float = 1.2
    volatility_low_threshold: float = 0.8

    # TTM Squeeze parameters
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_period: int = 20
    kc_multiplier: float = 1.5
    momentum_period: int = 12

    # Regime detection
    min_squeeze_duration: int = 5


class RegimeDetector:
    """
    Market regime detector using multiple technical indicators.

    The detector analyzes price action to classify the market into
    different regimes, each requiring specific trading strategies.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime detector.

        Args:
            config: Configuration parameters (uses defaults if None)
        """
        self.config = config or RegimeConfig()

    def detect(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Detect market regime for each bar.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            DataFrame with regime information:
            - regime: Primary regime classification
            - trend_strength: Trend strength category
            - trend_direction: Trend direction (bullish/bearish/neutral)
            - volatility_state: Volatility classification
            - squeeze_active: Boolean squeeze indicator
            - adx: ADX values
            - plus_di: +DI values
            - minus_di: -DI values
            - atr: ATR values
            - volatility_ratio: Normalized volatility

        Example:
            >>> detector = RegimeDetector()
            >>> regime_df = detector.detect(high, low, close)
            >>> current_regime = regime_df['regime'].iloc[-1]
        """
        # Calculate all indicators
        indicators = self._calculate_indicators(high, low, close)

        # Classify regimes
        regime_df = self._classify_regimes(indicators)

        return regime_df

    def _calculate_indicators(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate all required indicators for regime detection."""
        indicators = {}

        # Trend indicators
        adx, plus_di, minus_di = calculate_adx(
            high, low, close, period=self.config.adx_period
        )
        indicators['adx'] = adx
        indicators['plus_di'] = plus_di
        indicators['minus_di'] = minus_di

        # Trend strength and direction
        indicators['trend_strength'] = calculate_trend_strength(
            adx,
            threshold_strong=self.config.adx_strong_threshold,
            threshold_weak=self.config.adx_weak_threshold
        )
        indicators['trend_direction'] = calculate_trend_direction(
            plus_di, minus_di,
            min_difference=self.config.di_difference_threshold
        )

        # Volatility indicators
        atr = calculate_atr(high, low, close, period=self.config.atr_period)
        indicators['atr'] = atr

        volatility_ratio = calculate_volatility_ratio(
            atr, close, period=self.config.volatility_lookback
        )
        indicators['volatility_ratio'] = volatility_ratio

        # Bollinger Bands (for squeeze and volatility)
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
            close, period=self.config.bb_period, num_std=self.config.bb_std
        )
        indicators['bb_width'] = bb_width

        # TTM Squeeze
        squeeze_on, momentum, momentum_color = calculate_ttm_squeeze(
            high, low, close,
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            kc_period=self.config.kc_period,
            kc_atr_period=self.config.kc_atr_period,
            kc_multiplier=self.config.kc_multiplier,
            momentum_period=self.config.momentum_period
        )
        indicators['squeeze_active'] = squeeze_on
        indicators['momentum'] = momentum
        indicators['momentum_color'] = momentum_color

        # Squeeze duration
        indicators['squeeze_duration'] = calculate_squeeze_duration(squeeze_on)

        return indicators

    def _classify_regimes(self, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """Classify market regime based on indicators."""
        regime_data = {
            'adx': indicators['adx'],
            'plus_di': indicators['plus_di'],
            'minus_di': indicators['minus_di'],
            'trend_strength': indicators['trend_strength'],
            'trend_direction': indicators['trend_direction'],
            'atr': indicators['atr'],
            'volatility_ratio': indicators['volatility_ratio'],
            'squeeze_active': indicators['squeeze_active'],
            'squeeze_duration': indicators['squeeze_duration'],
            'momentum': indicators['momentum'],
        }

        df = pd.DataFrame(regime_data)

        # Classify volatility state
        df['volatility_state'] = self._classify_volatility(
            indicators['volatility_ratio'],
            indicators['squeeze_active']
        )

        # Classify primary regime
        df['regime'] = self._classify_primary_regime(
            df['trend_strength'],
            df['trend_direction'],
            df['volatility_state'],
            df['squeeze_active']
        )

        return df

    def _classify_volatility(
        self,
        volatility_ratio: pd.Series,
        squeeze_active: pd.Series
    ) -> pd.Series:
        """Classify volatility state."""
        volatility_state = pd.Series('normal', index=volatility_ratio.index)

        # Squeeze takes precedence
        volatility_state[squeeze_active] = 'squeeze'

        # High/low volatility (only when not in squeeze)
        volatility_state[
            (~squeeze_active) &
            (volatility_ratio > self.config.volatility_high_threshold)
        ] = 'high'

        volatility_state[
            (~squeeze_active) &
            (volatility_ratio < self.config.volatility_low_threshold)
        ] = 'low'

        return volatility_state

    def _classify_primary_regime(
        self,
        trend_strength: pd.Series,
        trend_direction: pd.Series,
        volatility_state: pd.Series,
        squeeze_active: pd.Series
    ) -> pd.Series:
        """Classify primary market regime."""
        regime = pd.Series(RegimeType.UNDEFINED.value, index=trend_strength.index)

        # Squeeze regime (highest priority)
        regime[squeeze_active] = RegimeType.SQUEEZE.value

        # Trending regimes (strong trend)
        trending_bullish = (
            (trend_strength == 'strong') &
            (trend_direction == 'bullish') &
            (~squeeze_active)
        )
        regime[trending_bullish] = RegimeType.TRENDING_BULLISH.value

        trending_bearish = (
            (trend_strength == 'strong') &
            (trend_direction == 'bearish') &
            (~squeeze_active)
        )
        regime[trending_bearish] = RegimeType.TRENDING_BEARISH.value

        # Ranging regime (weak trend, normal/low volatility)
        ranging = (
            (trend_strength == 'weak') &
            (volatility_state.isin(['normal', 'low'])) &
            (~squeeze_active)
        )
        regime[ranging] = RegimeType.RANGING.value

        # Volatile regime (high volatility without clear trend)
        volatile = (
            (volatility_state == 'high') &
            (trend_strength != 'strong') &
            (~squeeze_active)
        )
        regime[volatile] = RegimeType.VOLATILE.value

        return regime

    def get_regime_stats(self, regime_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics about regime distribution.

        Args:
            regime_df: DataFrame from detect() method

        Returns:
            Dictionary with regime percentages and metrics

        Example:
            >>> regime_df = detector.detect(high, low, close)
            >>> stats = detector.get_regime_stats(regime_df)
            >>> print(f"Trending: {stats['trending_pct']:.1f}%")
        """
        total_bars = len(regime_df)

        stats = {
            'total_bars': total_bars,
            'trending_bullish_pct': (
                (regime_df['regime'] == RegimeType.TRENDING_BULLISH.value).sum() / total_bars * 100
            ),
            'trending_bearish_pct': (
                (regime_df['regime'] == RegimeType.TRENDING_BEARISH.value).sum() / total_bars * 100
            ),
            'ranging_pct': (
                (regime_df['regime'] == RegimeType.RANGING.value).sum() / total_bars * 100
            ),
            'volatile_pct': (
                (regime_df['regime'] == RegimeType.VOLATILE.value).sum() / total_bars * 100
            ),
            'squeeze_pct': (
                (regime_df['regime'] == RegimeType.SQUEEZE.value).sum() / total_bars * 100
            ),
            'avg_adx': regime_df['adx'].mean(),
            'avg_volatility_ratio': regime_df['volatility_ratio'].mean(),
            'max_squeeze_duration': regime_df['squeeze_duration'].max(),
        }

        return stats

    def get_current_regime(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, any]:
        """
        Get current (most recent) market regime information.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Dictionary with current regime details

        Example:
            >>> current = detector.get_current_regime(high, low, close)
            >>> if current['regime'] == 'trending_bullish':
            ...     print("Strong bullish trend detected!")
        """
        regime_df = self.detect(high, low, close)

        current = {
            'regime': regime_df['regime'].iloc[-1],
            'trend_strength': regime_df['trend_strength'].iloc[-1],
            'trend_direction': regime_df['trend_direction'].iloc[-1],
            'volatility_state': regime_df['volatility_state'].iloc[-1],
            'squeeze_active': regime_df['squeeze_active'].iloc[-1],
            'adx': regime_df['adx'].iloc[-1],
            'atr': regime_df['atr'].iloc[-1],
            'volatility_ratio': regime_df['volatility_ratio'].iloc[-1],
        }

        if current['squeeze_active']:
            current['squeeze_duration'] = regime_df['squeeze_duration'].iloc[-1]

        return current


# Export main components
__all__ = [
    'RegimeDetector',
    'RegimeType',
    'TrendStrength',
    'VolatilityState',
    'RegimeConfig',
]
