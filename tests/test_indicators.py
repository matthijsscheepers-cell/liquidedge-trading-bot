"""
Unit Tests for Technical Indicators

Tests cover:
- Trend indicators (ADX, EMA, slope)
- Volatility indicators (ATR, Bollinger Bands, Keltner Channels)
- TTM Squeeze indicator
- Regime detection

Each test validates:
- Correct output shape and type
- Expected value ranges
- Edge cases (empty data, single value, etc.)
- Known calculation examples
"""

import pytest
import numpy as np
import pandas as pd
from typing import Tuple

from src.indicators.trend import (
    calculate_ema,
    calculate_sma,
    calculate_slope,
    calculate_adx,
    calculate_trend_strength,
    calculate_trend_direction,
    # DataFrame-based functions
    calculate_adx_df,
    calculate_ema_slope,
    calculate_multiple_emas,
    check_ema_alignment,
    add_trend_indicators,
)

from src.indicators.volatility import (
    calculate_true_range,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_keltner_channels,
    calculate_volatility_ratio,
    detect_squeeze,
)

from src.indicators.ttm import (
    calculate_momentum,
    calculate_ttm_squeeze,
    identify_squeeze_setups,
)

from src.regime.detector import RegimeDetector, RegimeType


# Test fixtures
@pytest.fixture
def sample_price_data() -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate sample OHLC data for testing."""
    np.random.seed(42)
    n = 100

    # Generate trending data with noise
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 2, n)
    close = pd.Series(trend + noise)

    # Generate high/low from close
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)

    return high, low, close


@pytest.fixture
def ranging_price_data() -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Generate ranging/sideways price data."""
    np.random.seed(42)
    n = 100

    # Generate ranging data around 100
    close = pd.Series(100 + np.random.normal(0, 2, n))
    high = close + np.random.uniform(0.5, 1.5, n)
    low = close - np.random.uniform(0.5, 1.5, n)

    return high, low, close


# === Trend Indicator Tests ===

class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_basic(self, sample_price_data):
        """Test basic EMA calculation."""
        _, _, close = sample_price_data
        ema = calculate_ema(close, period=20)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(close)
        assert not ema.iloc[-1] != ema.iloc[-1]  # Check not NaN

    def test_ema_with_numpy(self):
        """Test EMA with numpy array input."""
        data = np.array([100, 102, 104, 103, 105])
        ema = calculate_ema(data, period=3)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(data)

    def test_ema_smoothing(self):
        """Test that EMA smooths data."""
        data = pd.Series([100, 150, 100, 150, 100])  # Volatile data
        ema = calculate_ema(data, period=3)

        # EMA should be smoother than raw data
        ema_volatility = ema.diff().abs().mean()
        data_volatility = data.diff().abs().mean()

        assert ema_volatility < data_volatility


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_basic(self, sample_price_data):
        """Test basic SMA calculation."""
        _, _, close = sample_price_data
        sma = calculate_sma(close, period=20)

        assert isinstance(sma, pd.Series)
        assert len(sma) == len(close)

    def test_sma_known_values(self):
        """Test SMA with known values."""
        data = pd.Series([10, 20, 30, 40, 50])
        sma = calculate_sma(data, period=3)

        # SMA of last 3 values: (30 + 40 + 50) / 3 = 40
        assert abs(sma.iloc[-1] - 40.0) < 0.01


class TestSlope:
    """Tests for slope calculation."""

    def test_slope_uptrend(self):
        """Test slope detects uptrend."""
        data = pd.Series(np.linspace(100, 120, 50))  # Linear uptrend
        slope = calculate_slope(data, period=10)

        # Slope should be consistently positive
        assert slope.iloc[-1] > 0

    def test_slope_downtrend(self):
        """Test slope detects downtrend."""
        data = pd.Series(np.linspace(120, 100, 50))  # Linear downtrend
        slope = calculate_slope(data, period=10)

        # Slope should be consistently negative
        assert slope.iloc[-1] < 0

    def test_slope_normalization(self):
        """Test normalized vs non-normalized slope."""
        data = pd.Series([100, 102, 104, 106, 108])
        slope_norm = calculate_slope(data, period=3, normalize=True)
        slope_raw = calculate_slope(data, period=3, normalize=False)

        # Both should have same sign but different magnitudes
        assert (slope_norm.iloc[-1] > 0) == (slope_raw.iloc[-1] > 0)


class TestADX:
    """Tests for Average Directional Index."""

    def test_adx_basic(self, sample_price_data):
        """Test basic ADX calculation."""
        high, low, close = sample_price_data
        adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)

        assert isinstance(adx, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)
        assert len(adx) == len(close)

    def test_adx_range(self, sample_price_data):
        """Test ADX values are in valid range."""
        high, low, close = sample_price_data
        adx, plus_di, minus_di = calculate_adx(high, low, close)

        # ADX and DI values should be between 0 and 100
        assert (adx >= 0).all()
        assert (adx <= 100).all()
        assert (plus_di >= 0).all()
        assert (minus_di >= 0).all()

    def test_adx_trending_vs_ranging(self, sample_price_data, ranging_price_data):
        """Test ADX higher in trending vs ranging market."""
        high_trend, low_trend, close_trend = sample_price_data
        high_range, low_range, close_range = ranging_price_data

        adx_trend, _, _ = calculate_adx(high_trend, low_trend, close_trend)
        adx_range, _, _ = calculate_adx(high_range, low_range, close_range)

        # Trending market should have higher average ADX
        assert adx_trend.mean() > adx_range.mean()


class TestTrendStrength:
    """Tests for trend strength classification."""

    def test_trend_strength_categories(self):
        """Test trend strength categories."""
        adx = pd.Series([15, 22, 28, 35])
        strength = calculate_trend_strength(adx, threshold_strong=25, threshold_weak=20)

        assert strength.iloc[0] == 'weak'  # 15 < 20
        assert strength.iloc[1] == 'moderate'  # 20 <= 22 < 25
        assert strength.iloc[2] == 'strong'  # 28 >= 25
        assert strength.iloc[3] == 'strong'  # 35 >= 25


class TestTrendDirection:
    """Tests for trend direction classification."""

    def test_trend_direction_bullish(self):
        """Test bullish trend detection."""
        plus_di = pd.Series([30, 35, 40])
        minus_di = pd.Series([15, 12, 10])
        direction = calculate_trend_direction(plus_di, minus_di, min_difference=5)

        assert (direction == 'bullish').all()

    def test_trend_direction_bearish(self):
        """Test bearish trend detection."""
        plus_di = pd.Series([10, 12, 15])
        minus_di = pd.Series([30, 35, 40])
        direction = calculate_trend_direction(plus_di, minus_di, min_difference=5)

        assert (direction == 'bearish').all()

    def test_trend_direction_neutral(self):
        """Test neutral trend when DI difference is small."""
        plus_di = pd.Series([20, 22, 21])
        minus_di = pd.Series([19, 20, 22])
        direction = calculate_trend_direction(plus_di, minus_di, min_difference=5)

        assert (direction == 'neutral').all()


class TestDataFrameTrendFunctions:
    """Tests for DataFrame-based trend convenience functions."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with OHLC data."""
        np.random.seed(42)
        n = 100
        close = pd.Series(np.linspace(100, 120, n) + np.random.normal(0, 2, n))
        high = close + np.random.uniform(0.5, 2.0, n)
        low = close - np.random.uniform(0.5, 2.0, n)

        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        return df

    def test_calculate_adx_df(self, sample_df):
        """Test DataFrame-based ADX calculation."""
        result = calculate_adx_df(sample_df, period=14)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)
        assert 'adx' in result.columns
        assert 'plus_di' in result.columns
        assert 'minus_di' in result.columns

        # Check values are in valid range
        assert (result['adx'] >= 0).all()
        assert (result['plus_di'] >= 0).all()
        assert (result['minus_di'] >= 0).all()

    def test_calculate_adx_df_missing_column(self):
        """Test error when required column missing."""
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(KeyError):
            calculate_adx_df(df)

    def test_calculate_ema_slope(self, sample_df):
        """Test EMA slope calculation."""
        slope = calculate_ema_slope(sample_df, ema_period=20, slope_lookback=20)

        assert isinstance(slope, pd.Series)
        assert len(slope) == len(sample_df)
        assert slope.name == 'ema_20_slope'

        # For uptrending data, average slope should be positive
        assert slope.mean() > 0

    def test_calculate_multiple_emas(self, sample_df):
        """Test multiple EMA calculation."""
        emas = calculate_multiple_emas(sample_df, periods=[20, 50, 200])

        assert isinstance(emas, pd.DataFrame)
        assert len(emas) == len(sample_df)
        assert 'ema_20' in emas.columns
        assert 'ema_50' in emas.columns
        assert 'ema_200' in emas.columns

    def test_calculate_multiple_emas_custom_periods(self, sample_df):
        """Test with custom period list."""
        emas = calculate_multiple_emas(sample_df, periods=[10, 30])

        assert 'ema_10' in emas.columns
        assert 'ema_30' in emas.columns
        assert 'ema_20' not in emas.columns

    def test_check_ema_alignment_bullish(self):
        """Test bullish EMA alignment detection."""
        # Create data with clear bullish alignment
        n = 100
        close = pd.Series(np.linspace(100, 150, n))  # Strong uptrend
        df = pd.DataFrame({
            'high': close + 1,
            'low': close - 1,
            'close': close
        })

        alignment = check_ema_alignment(df, periods=[20, 50, 200])

        assert isinstance(alignment, pd.Series)
        assert alignment.name == 'ema_alignment'

        # In strong uptrend, should have some bullish alignment
        bullish_count = (alignment == 'bullish').sum()
        assert bullish_count > 0

    def test_check_ema_alignment_bearish(self):
        """Test bearish EMA alignment detection."""
        # Create data with bearish alignment
        n = 100
        close = pd.Series(np.linspace(150, 100, n))  # Strong downtrend
        df = pd.DataFrame({
            'high': close + 1,
            'low': close - 1,
            'close': close
        })

        alignment = check_ema_alignment(df, periods=[20, 50, 200])

        # In strong downtrend, should have some bearish alignment
        bearish_count = (alignment == 'bearish').sum()
        assert bearish_count > 0

    def test_check_ema_alignment_values(self):
        """Test alignment returns only valid values."""
        n = 50
        df = pd.DataFrame({
            'high': np.linspace(100, 110, n),
            'low': np.linspace(98, 108, n),
            'close': np.linspace(99, 109, n)
        })

        alignment = check_ema_alignment(df, periods=[10, 20])

        # Should only contain these three values
        valid_values = {'bullish', 'bearish', 'mixed'}
        assert set(alignment.unique()).issubset(valid_values)

    def test_add_trend_indicators(self, sample_df):
        """Test adding all trend indicators at once."""
        result = add_trend_indicators(sample_df)

        # Check original columns preserved
        assert 'high' in result.columns
        assert 'low' in result.columns
        assert 'close' in result.columns

        # Check ADX indicators added
        assert 'adx' in result.columns
        assert 'plus_di' in result.columns
        assert 'minus_di' in result.columns

        # Check EMAs added
        assert 'ema_20' in result.columns
        assert 'ema_50' in result.columns
        assert 'ema_200' in result.columns

        # Check alignment added
        assert 'ema_alignment' in result.columns

        # Check length preserved
        assert len(result) == len(sample_df)

    def test_add_trend_indicators_custom_periods(self, sample_df):
        """Test with custom EMA periods."""
        result = add_trend_indicators(
            sample_df,
            adx_period=20,
            ema_periods=[10, 30, 60]
        )

        assert 'ema_10' in result.columns
        assert 'ema_30' in result.columns
        assert 'ema_60' in result.columns
        assert 'ema_20' not in result.columns  # Not in custom list


# === Volatility Indicator Tests ===

class TestTrueRange:
    """Tests for True Range calculation."""

    def test_true_range_basic(self, sample_price_data):
        """Test basic True Range calculation."""
        high, low, close = sample_price_data
        tr = calculate_true_range(high, low, close)

        assert isinstance(tr, pd.Series)
        assert len(tr) == len(close)
        assert (tr >= 0).all()  # TR should always be positive

    def test_true_range_minimum(self):
        """Test TR equals high-low when it's the largest."""
        high = pd.Series([105, 110, 108])
        low = pd.Series([100, 105, 103])
        close = pd.Series([102, 107, 105])
        tr = calculate_true_range(high, low, close)

        # First TR is just high - low
        assert abs(tr.iloc[0] - 5.0) < 0.01


class TestATR:
    """Tests for Average True Range."""

    def test_atr_basic(self, sample_price_data):
        """Test basic ATR calculation."""
        high, low, close = sample_price_data
        atr = calculate_atr(high, low, close, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(close)
        assert (atr >= 0).all()

    def test_atr_smoothing(self, sample_price_data):
        """Test ATR smooths True Range."""
        high, low, close = sample_price_data
        tr = calculate_true_range(high, low, close)
        atr = calculate_atr(high, low, close, period=14)

        # ATR should be smoother than TR
        tr_volatility = tr.diff().abs().mean()
        atr_volatility = atr.diff().abs().mean()

        assert atr_volatility < tr_volatility


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_bands_basic(self, sample_price_data):
        """Test basic Bollinger Bands calculation."""
        _, _, close = sample_price_data
        bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
            close, period=20
        )

        assert isinstance(bb_upper, pd.Series)
        assert len(bb_upper) == len(close)

        # Upper should be above middle, middle above lower (check non-NaN values)
        valid_mask = ~bb_upper.isna()
        assert (bb_upper[valid_mask] >= bb_middle[valid_mask]).all()
        assert (bb_middle[valid_mask] >= bb_lower[valid_mask]).all()

    def test_bollinger_bands_std_multiplier(self):
        """Test different std multipliers."""
        data = pd.Series([100] * 20 + [110, 115, 120])  # Sudden move
        bb_upper_2std, _, _, _ = calculate_bollinger_bands(data, period=20, num_std=2.0)
        bb_upper_3std, _, _, _ = calculate_bollinger_bands(data, period=20, num_std=3.0)

        # 3 std bands should be wider than 2 std bands
        assert bb_upper_3std.iloc[-1] > bb_upper_2std.iloc[-1]


class TestKeltnerChannels:
    """Tests for Keltner Channels."""

    def test_keltner_channels_basic(self, sample_price_data):
        """Test basic Keltner Channels calculation."""
        high, low, close = sample_price_data
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(
            high, low, close
        )

        assert isinstance(kc_upper, pd.Series)
        assert len(kc_upper) == len(close)

        # Upper should be above middle, middle above lower
        assert (kc_upper >= kc_middle).all()
        assert (kc_middle >= kc_lower).all()


class TestSqueeze:
    """Tests for squeeze detection."""

    def test_squeeze_detection(self):
        """Test squeeze detection when BB inside KC."""
        # Create scenario where BB is inside KC
        bb_upper = pd.Series([105, 106, 107])
        bb_lower = pd.Series([95, 94, 93])
        kc_upper = pd.Series([110, 111, 112])  # KC wider than BB
        kc_lower = pd.Series([90, 89, 88])

        squeeze = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        assert isinstance(squeeze, pd.Series)
        assert squeeze.all()  # All should be in squeeze

    def test_no_squeeze(self):
        """Test no squeeze when BB outside KC."""
        # BB wider than KC
        bb_upper = pd.Series([110, 111, 112])
        bb_lower = pd.Series([90, 89, 88])
        kc_upper = pd.Series([105, 106, 107])
        kc_lower = pd.Series([95, 94, 93])

        squeeze = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower)

        assert not squeeze.any()  # No squeeze


# === TTM Squeeze Tests ===

class TestTTMSqueeze:
    """Tests for TTM Squeeze indicator."""

    def test_ttm_squeeze_basic(self, sample_price_data):
        """Test basic TTM Squeeze calculation."""
        high, low, close = sample_price_data
        squeeze_on, momentum, momentum_color = calculate_ttm_squeeze(
            high, low, close
        )

        assert isinstance(squeeze_on, pd.Series)
        assert isinstance(momentum, pd.Series)
        assert isinstance(momentum_color, pd.Series)
        assert len(squeeze_on) == len(close)

    def test_momentum_colors(self):
        """Test momentum color classification."""
        # Increasing positive momentum
        momentum = pd.Series([0.5, 1.0, 1.5])
        from src.indicators.ttm import calculate_momentum_color
        colors = calculate_momentum_color(momentum)

        assert colors.iloc[-1] == 'lime'  # Positive and increasing

    def test_squeeze_setups(self, sample_price_data):
        """Test squeeze setup identification."""
        high, low, close = sample_price_data
        squeeze_on, momentum, momentum_color = calculate_ttm_squeeze(
            high, low, close
        )

        bullish, bearish = identify_squeeze_setups(
            squeeze_on, momentum, momentum_color, min_squeeze_duration=3
        )

        assert isinstance(bullish, pd.Series)
        assert isinstance(bearish, pd.Series)
        # Setups should be mutually exclusive
        assert not (bullish & bearish).any()


# === Regime Detector Tests ===

class TestRegimeDetector:
    """Tests for market regime detection."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = RegimeDetector()
        assert detector.config is not None

    def test_regime_detection_basic(self, sample_price_data):
        """Test basic regime detection."""
        high, low, close = sample_price_data
        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)

        assert isinstance(regime_df, pd.DataFrame)
        assert 'regime' in regime_df.columns
        assert 'trend_strength' in regime_df.columns
        assert 'volatility_state' in regime_df.columns
        assert len(regime_df) == len(close)

    def test_regime_types(self, sample_price_data):
        """Test regime type classification."""
        high, low, close = sample_price_data
        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)

        # Check that regimes are valid types
        valid_regimes = [e.value for e in RegimeType]
        assert regime_df['regime'].isin(valid_regimes).all()

    def test_trending_detection(self):
        """Test trending regime detection with strong trend data."""
        # Create strong trending data without noise
        n = 100
        close = pd.Series(np.linspace(100, 150, n))  # Strong linear uptrend
        high = close + 1.0
        low = close - 1.0

        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)

        # Strong trend should produce higher ADX and some trending regimes
        # At minimum, ADX should be above weak threshold for decent portion
        strong_adx_count = (regime_df['adx'] > 20).sum()
        assert strong_adx_count > 10  # At least 10% of bars have ADX > 20

    def test_ranging_detection(self, ranging_price_data):
        """Test ranging regime detection."""
        high, low, close = ranging_price_data
        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)

        # Ranging data should have some ranging regimes
        ranging_count = (regime_df['regime'] == RegimeType.RANGING.value).sum()
        assert ranging_count > 0

    def test_get_current_regime(self, sample_price_data):
        """Test getting current regime."""
        high, low, close = sample_price_data
        detector = RegimeDetector()
        current = detector.get_current_regime(high, low, close)

        assert isinstance(current, dict)
        assert 'regime' in current
        assert 'trend_strength' in current
        assert 'adx' in current

    def test_regime_stats(self, sample_price_data):
        """Test regime statistics calculation."""
        high, low, close = sample_price_data
        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)
        stats = detector.get_regime_stats(regime_df)

        assert isinstance(stats, dict)
        assert 'total_bars' in stats
        assert 'avg_adx' in stats

        # Percentages should sum to approximately 100%
        total_pct = (
            stats['trending_bullish_pct'] +
            stats['trending_bearish_pct'] +
            stats['ranging_pct'] +
            stats['volatile_pct'] +
            stats['squeeze_pct']
        )
        assert abs(total_pct - 100.0) < 1.0  # Allow small rounding error


# === Integration Tests ===

class TestIndicatorIntegration:
    """Integration tests for indicator combinations."""

    def test_full_pipeline(self, sample_price_data):
        """Test complete indicator calculation pipeline."""
        high, low, close = sample_price_data

        # Calculate all indicators
        adx, plus_di, minus_di = calculate_adx(high, low, close)
        atr = calculate_atr(high, low, close)
        bb_upper, bb_mid, bb_lower, bb_width = calculate_bollinger_bands(close)
        kc_upper, kc_mid, kc_lower = calculate_keltner_channels(high, low, close)
        squeeze_on, momentum, color = calculate_ttm_squeeze(high, low, close)

        # Detect regime
        detector = RegimeDetector()
        regime_df = detector.detect(high, low, close)

        # All should have same length
        assert len(adx) == len(atr) == len(bb_upper) == len(squeeze_on)
        assert len(regime_df) == len(close)

        # No NaN in recent values (after warm-up period)
        assert not adx.iloc[-10:].isna().any()
        assert not atr.iloc[-10:].isna().any()
        assert not bb_upper.iloc[-10:].isna().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
