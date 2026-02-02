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
    calculate_ema,
    calculate_slope,
    calculate_trend_strength,
    calculate_trend_direction,
    calculate_atr,
    calculate_volatility_ratio,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_ttm_squeeze,
    calculate_squeeze_duration,
    detect_squeeze,
    calculate_rsi,
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
    adx_strong_threshold: float = 10.0  # Intraday 15m: very low threshold
    adx_weak_threshold: float = 6.0     # Intraday 15m: catch weak trends

    # Directional indicator parameters
    di_difference_threshold: float = 5.0

    # Volatility parameters
    atr_period: int = 14
    volatility_lookback: int = 50
    volatility_high_threshold: float = 2.5  # Intraday 15m: veel hoger voor natuurlijke intraday volatility
    volatility_low_threshold: float = 0.6   # Intraday 15m: lager voor squeeze detection

    # TTM Squeeze parameters
    bb_period: int = 20
    bb_std: float = 2.5  # Intraday 15m: wider BB for futures volatility
    kc_period: int = 20
    kc_atr_period: int = 20
    kc_multiplier: float = 0.5  # Intraday 15m: much narrower KC for proper squeeze detection
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

    def add_all_indicators(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        ema_periods: list = None
    ) -> pd.DataFrame:
        """
        Add ALL regime-relevant indicators to DataFrame.

        This convenience method calculates and adds all indicators needed
        for regime detection and strategy decisions. Useful for live trading
        where you want all context in a single DataFrame.

        Args:
            df: DataFrame with OHLCV data
            high_col: Name of high column (default 'high')
            low_col: Name of low column (default 'low')
            close_col: Name of close column (default 'close')
            ema_periods: List of EMA periods (default [20, 50, 200])

        Returns:
            DataFrame with all indicators added as columns:

            Trend Indicators:
            - adx_{period}: ADX trend strength
            - plus_di: Positive directional indicator
            - minus_di: Negative directional indicator
            - ema_{period}: Exponential moving averages
            - ema_{period}_slope: EMA slope (momentum)
            - ema_alignment: Bullish/bearish/mixed alignment

            Volatility Indicators:
            - atr_{period}: Average True Range
            - atr_pct: ATR as percentage of price
            - atr_percentile: ATR percentile ranking
            - bb_upper, bb_middle, bb_lower: Bollinger Bands
            - bb_width: Bollinger Band width
            - kc_upper, kc_middle, kc_lower: Keltner Channels

            TTM Squeeze Indicators:
            - squeeze_on: Boolean squeeze indicator
            - squeeze_duration: Bars since squeeze started
            - ttm_momentum: TTM momentum value
            - momentum_color: Momentum color (green/red)

            Momentum Indicators:
            - rsi_14: Relative Strength Index (14-period)

            Derived Metrics:
            - volatility_ratio: Normalized volatility
            - compression_score: Squeeze compression intensity (0-100)
            - bb_width_percentile: BB width percentile

        Example:
            >>> detector = RegimeDetector()
            >>> df = detector.add_all_indicators(df)
            >>> regime, conf, strat = detector.detect_regime(df)
            >>> print(f"{regime.value}: {conf}% confidence → {strat}")
        """
        if ema_periods is None:
            ema_periods = [20, 50, 200]

        # Validate columns
        required = [high_col, low_col, close_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Create copy to avoid modifying original
        result = df.copy()

        high = df[high_col]
        low = df[low_col]
        close = df[close_col]

        # === Trend Indicators ===

        # ADX and directional indicators
        adx, plus_di, minus_di = calculate_adx(
            high, low, close, period=self.config.adx_period
        )
        result[f'adx_{self.config.adx_period}'] = adx
        result['plus_di'] = plus_di
        result['minus_di'] = minus_di

        # EMAs
        for period in ema_periods:
            ema = calculate_ema(close, period=period)
            result[f'ema_{period}'] = ema

        # EMA slopes (momentum)
        for period in ema_periods:
            if f'ema_{period}' in result.columns:
                ema = result[f'ema_{period}']
                slope = calculate_slope(ema, period=20, normalize=False)
                # Normalize slope as percentage
                slope_pct = (slope / close) * 100
                result[f'ema_{period}_slope'] = slope_pct

        # EMA alignment
        if all(f'ema_{p}' in result.columns for p in ema_periods):
            sorted_periods = sorted(ema_periods)
            if len(sorted_periods) >= 2:
                # Bullish: fast > slow
                bullish = pd.Series(True, index=result.index)
                for i in range(len(sorted_periods) - 1):
                    fast_col = f'ema_{sorted_periods[i]}'
                    slow_col = f'ema_{sorted_periods[i + 1]}'
                    bullish &= (result[fast_col] > result[slow_col])

                # Bearish: fast < slow
                bearish = pd.Series(True, index=result.index)
                for i in range(len(sorted_periods) - 1):
                    fast_col = f'ema_{sorted_periods[i]}'
                    slow_col = f'ema_{sorted_periods[i + 1]}'
                    bearish &= (result[fast_col] < result[slow_col])

                alignment = pd.Series('mixed', index=result.index)
                alignment[bullish] = 'bullish'
                alignment[bearish] = 'bearish'
                result['ema_alignment'] = alignment

        # === Volatility Indicators ===

        # ATR
        atr = calculate_atr(high, low, close, period=self.config.atr_period)
        result[f'atr_{self.config.atr_period}'] = atr

        # ATR percentage
        atr_pct = (atr / close) * 100
        result['atr_pct'] = atr_pct

        # ATR percentile (rolling 252 bars = 1 year daily)
        atr_percentile = atr.rolling(window=252, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 2 else np.nan,
            raw=False
        )
        result['atr_percentile'] = atr_percentile

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
            close, period=self.config.bb_period, num_std=self.config.bb_std
        )
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower
        result['bb_width'] = bb_width

        # BB width percentile
        bb_width_percentile = bb_width.rolling(window=100, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= 2 else np.nan,
            raw=False
        )
        result['bb_width_percentile'] = bb_width_percentile

        # Keltner Channels
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
            high, low, close,
            period=self.config.kc_period,
            atr_period=self.config.kc_atr_period,
            atr_multiplier=self.config.kc_multiplier
        )
        result['kc_upper'] = kc_upper
        result['kc_middle'] = kc_middle
        result['kc_lower'] = kc_lower

        # Volatility ratio (normalized)
        volatility_ratio = calculate_volatility_ratio(
            atr, close, period=self.config.volatility_lookback
        )
        result['volatility_ratio'] = volatility_ratio

        # === TTM Squeeze Indicators ===

        # TTM Squeeze
        squeeze_on, ttm_momentum, momentum_color = calculate_ttm_squeeze(
            high, low, close,
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            kc_period=self.config.kc_period,
            kc_atr_period=self.config.kc_atr_period,
            kc_multiplier=self.config.kc_multiplier,
            momentum_period=self.config.momentum_period
        )
        result['squeeze_on'] = squeeze_on
        result['ttm_momentum'] = ttm_momentum
        result['momentum_color'] = momentum_color

        # Squeeze duration
        squeeze_duration = calculate_squeeze_duration(squeeze_on)
        result['squeeze_duration'] = squeeze_duration

        # Compression score (0-100): how compressed is the squeeze
        # Higher score = tighter squeeze = more explosive breakout potential
        compression_score = pd.Series(0.0, index=result.index)

        # When squeeze is on, calculate compression
        squeeze_mask = squeeze_on.fillna(False).astype(bool)
        if squeeze_mask.any():
            # Ratio of BB width to KC width (both normalized)
            bb_range = bb_upper - bb_lower
            kc_range = kc_upper - kc_lower

            # Avoid division by zero
            kc_range_safe = kc_range.replace(0, np.nan)

            # Compression ratio: smaller BB relative to KC = higher compression
            compression_ratio = bb_range / kc_range_safe

            # Convert to 0-100 score (inverted: lower ratio = higher score)
            # Typical range: 0.5 to 1.0, map to 100 to 0
            compression_normalized = (1.0 - compression_ratio.clip(0.5, 1.0)) / 0.5 * 100

            compression_score[squeeze_mask] = compression_normalized[squeeze_mask]

        result['compression_score'] = compression_score

        # === Momentum Indicators ===

        # RSI (14-period)
        rsi = calculate_rsi(close, period=14)
        result['rsi_14'] = rsi

        return result

    def detect_regime(
        self,
        df: pd.DataFrame,
        lookback_bars: int = 2
    ) -> tuple:
        """
        Detect current market regime with confidence and strategy recommendation.

        This is the PRIMARY method for live trading decisions. It analyzes
        the current market state using a priority-based system and returns
        actionable information.

        Priority hierarchy (checked in order):
        1. HIGH_VOLATILITY - Safety check, avoid trading
        2. RANGE_COMPRESSION - TTM Squeeze setup opportunities
        3. STRONG_TREND - High ADX trending opportunities
        4. WEAK_TREND - Lower ADX trending opportunities
        5. NO_TRADE - No clear trading opportunity

        Args:
            df: DataFrame with indicators (from add_all_indicators)
            lookback_bars: Number of bars to look back for analysis

        Returns:
            Tuple of (regime, confidence, strategy):
            - regime: MarketRegime enum value
            - confidence: Float 0-100 indicating setup quality
            - strategy: String with recommended strategy name

        Raises:
            ValueError: If required indicators are missing from DataFrame

        Example:
            >>> detector = RegimeDetector()
            >>> df = detector.add_all_indicators(df)
            >>> regime, conf, strat = detector.detect_regime(df)
            >>>
            >>> if conf > 70:
            ...     print(f"High confidence {regime.value}: {strat}")
            ...     # Execute trading strategy
            >>> elif conf > 50:
            ...     print(f"Moderate confidence, be cautious")
            >>> else:
            ...     print(f"Low confidence, wait for better setup")
        """
        # Validate required indicators
        required_indicators = [
            'close', 'atr_percentile', 'bb_width_percentile',
            'squeeze_on', 'squeeze_duration', 'compression_score',
            'ttm_momentum', 'ema_alignment', 'volatility_ratio'
        ]

        # Check for ADX (could be any period)
        adx_cols = [col for col in df.columns if col.startswith('adx_')]
        if not adx_cols:
            required_indicators.append('adx_14')  # Add specific one to error message

        # Check for EMA 200 (or any 200 period EMA)
        ema_200_cols = [col for col in df.columns if 'ema_200' in col]
        if not ema_200_cols:
            required_indicators.append('ema_200')

        missing = [col for col in required_indicators if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required indicators: {missing}. "
                f"Call add_all_indicators() first to calculate indicators."
            )

        # Get current and recent bars
        if len(df) < lookback_bars:
            lookback_bars = len(df)

        current = df.iloc[-1]
        recent = df.iloc[-lookback_bars:]

        # Get ADX column (use first one found)
        adx_col = adx_cols[0] if adx_cols else 'adx_14'

        # Get EMA 200 column
        ema_200_col = ema_200_cols[0] if ema_200_cols else 'ema_200'

        # === PRIORITY 1: Safety - Volatility Spike Check ===
        if self._is_volatility_spike(current):
            return MarketRegime.HIGH_VOLATILITY, 0, 'NONE'

        # === PRIORITY 2: TTM Squeeze Setup Check ===
        regime, conf, strat = self._check_ttm_squeeze_setup(
            df, current, recent, ema_200_col
        )
        if regime:
            return regime, conf, strat

        # === PRIORITY 3: Trend Regime Check ===
        regime, conf, strat = self._check_trend_regime(
            current, recent, adx_col, ema_200_col
        )
        if regime:
            return regime, conf, strat

        # === DEFAULT: No Clear Trade Setup ===
        return MarketRegime.NO_TRADE, 0, 'NONE'

    def _is_volatility_spike(self, current: pd.Series) -> bool:
        """
        Check if volatility is dangerously high.

        High volatility = unpredictable price action = increased risk.
        Better to wait for volatility to normalize.

        Args:
            current: Current bar data

        Returns:
            True if volatility is too high to trade safely

        Note:
            Thresholds adjusted for intraday 15m data - more lenient than daily.
        """
        # ATR in top 5% of historical range (was 15% - too strict for intraday)
        atr_extreme = current.get('atr_percentile', 50) > 95

        # BB width in top 5% (extreme expansion - was 10%)
        bb_extreme = current.get('bb_width_percentile', 50) > 95

        # Volatility ratio significantly elevated (was 1.5, now 2.5 for intraday)
        vol_elevated = current.get('volatility_ratio', 1.0) > 2.5

        return atr_extreme or bb_extreme or vol_elevated

    def _check_ttm_squeeze_setup(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        recent: pd.DataFrame,
        ema_200_col: str
    ) -> tuple:
        """
        Check for TTM Squeeze trading opportunities.

        The TTM Squeeze identifies periods of low volatility (compression)
        that often precede explosive moves (expansion). We want to enter
        just before or right as the squeeze releases.

        Two setups:
        1. Squeeze Building - Compression increasing, waiting for breakout
        2. Squeeze Released - Breakout happening now, momentum confirming

        Args:
            df: Full DataFrame with indicators
            current: Current bar
            recent: Recent bars for context
            ema_200_col: Name of EMA 200 column

        Returns:
            (regime, confidence, strategy) or (None, 0, 'NONE')
        """
        # === SETUP 1: Squeeze Building (Pre-breakout) ===

        if current['squeeze_on']:
            duration = current['squeeze_duration']
            compression = current['compression_score']

            # Minimum squeeze duration (avoid false signals)
            if duration >= self.config.min_squeeze_duration:

                # Quality filters
                price_above_200ema = current['close'] > current[ema_200_col]
                atr_reasonable = current.get('atr_percentile', 50) < 70
                ema_aligned = current.get('ema_alignment', 'mixed') in ['bullish', 'bearish']

                # Check if compression is increasing (squeeze tightening)
                if len(recent) >= 2:
                    compression_increasing = (
                        current['compression_score'] > recent['compression_score'].iloc[-2]
                    )
                else:
                    compression_increasing = True  # Not enough data, be permissive

                # All filters must pass
                if price_above_200ema and atr_reasonable and ema_aligned:
                    # Confidence based on:
                    # - Duration (longer squeeze = more explosive breakout)
                    # - Compression level (tighter squeeze = more energy)
                    base_confidence = 60
                    duration_bonus = min(20, duration * 2)  # +2% per bar, max +20%
                    compression_bonus = min(15, compression * 0.15)  # Scale compression score

                    confidence = base_confidence + duration_bonus + compression_bonus
                    confidence = min(100, confidence)  # Cap at 100

                    return MarketRegime.RANGE_COMPRESSION, confidence, 'TTM_SQUEEZE'

        # === SETUP 2: Squeeze Released (Breakout) ===

        # Check if squeeze just released (was on, now off)
        if len(recent) >= 2:
            prev_squeeze = recent['squeeze_on'].iloc[-2]
            curr_squeeze = current['squeeze_on']

            squeeze_released = prev_squeeze and not curr_squeeze

            if squeeze_released:
                # Confirm with momentum
                prev_momentum = recent['ttm_momentum'].iloc[-2]
                curr_momentum = current['ttm_momentum']

                # Momentum should be accelerating in the breakout direction
                momentum_increasing = abs(curr_momentum) > abs(prev_momentum)

                # Strong momentum confirmation
                momentum_strong = abs(curr_momentum) > 0.3

                if momentum_increasing or momentum_strong:
                    # High confidence for confirmed breakout
                    confidence = 85

                    # Bonus if momentum is very strong
                    if abs(curr_momentum) > 0.5:
                        confidence = min(95, confidence + 10)

                    return MarketRegime.RANGE_COMPRESSION, confidence, 'TTM_BREAKOUT'

        return None, 0, 'NONE'

    def _check_trend_regime(
        self,
        current: pd.Series,
        recent: pd.DataFrame,
        adx_col: str,
        ema_200_col: str
    ) -> tuple:
        """
        Check for trend trading opportunities.

        Trend following works best when:
        1. Clear trend strength (ADX > threshold)
        2. EMA alignment confirms direction
        3. EMA 200 slope shows sustained momentum
        4. Volatility is reasonable

        Strategy: Wait for pullbacks in strong trends, then enter
        in the direction of the trend (REGIME_PULLBACK strategy).

        Args:
            current: Current bar
            recent: Recent bars
            adx_col: Name of ADX column
            ema_200_col: Name of EMA 200 column

        Returns:
            (regime, confidence, strategy) or (None, 0, 'NONE')
        """
        adx = current[adx_col]
        ema_alignment = current.get('ema_alignment', 'mixed')

        # Require EMA alignment (bullish or bearish, not mixed)
        if ema_alignment not in ['bullish', 'bearish']:
            return None, 0, 'NONE'

        # Get EMA 200 slope
        ema_200_slope_col = f'{ema_200_col}_slope'
        ema_200_slope = current.get(ema_200_slope_col, 0)

        # Volatility should be reasonable (not too high)
        vol_ratio = current.get('volatility_ratio', 1.0)
        if vol_ratio > 1.3:
            return None, 0, 'NONE'  # Too volatile for trend following

        # === STRONG TREND ===
        if adx > self.config.adx_strong_threshold:
            # Require sustained EMA 200 momentum
            slope_threshold = 0.0003  # Intraday 15m: very small moves acceptable

            if abs(ema_200_slope) > slope_threshold:
                # Base confidence for strong trend
                base_confidence = 70

                # Bonus for very strong ADX
                adx_bonus = min(20, (adx - self.config.adx_strong_threshold) * 2)

                # Bonus for strong slope
                slope_bonus = min(10, abs(ema_200_slope) * 500)  # Scale up small slope values

                confidence = base_confidence + adx_bonus + slope_bonus
                confidence = min(100, confidence)

                return MarketRegime.STRONG_TREND, confidence, 'REGIME_PULLBACK'

        # === WEAK TREND ===
        if adx > self.config.adx_weak_threshold:
            # Lower confidence for weak trends
            base_confidence = 50

            # Bonus for ADX strength within weak range
            adx_bonus = (adx - self.config.adx_weak_threshold) * 3
            adx_bonus = min(20, adx_bonus)

            confidence = base_confidence + adx_bonus

            return MarketRegime.WEAK_TREND, confidence, 'REGIME_PULLBACK'

        return None, 0, 'NONE'


# Add new MarketRegime enum for trading decisions
class MarketRegime(Enum):
    """
    Market regime classifications for trading decisions.

    These regimes map directly to trading strategies with
    confidence scores indicating setup quality.
    """
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    RANGE_COMPRESSION = "RANGE_COMPRESSION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    NO_TRADE = "NO_TRADE"


# Export main components
__all__ = [
    'RegimeDetector',
    'RegimeType',
    'MarketRegime',
    'TrendStrength',
    'VolatilityState',
    'RegimeConfig',
]
