"""
Comprehensive Strategy Test Suite

This module provides complete test coverage for the LIQUIDEDGE strategy system:
- Base strategy abstract class validation
- RegimePullbackStrategy (trend following)
- TTMSqueezeStrategy (breakout trading)
- StrategySelector (intelligent routing)

Run with: pytest tests/test_strategies.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.base import (
    BaseStrategy,
    TradeSetup,
    Position,
    SignalDirection,
    ExitAction,
)
from src.strategies.regime_pullback import RegimePullbackStrategy
from src.strategies.ttm_squeeze import TTMSqueezeStrategy
from src.strategies.selector import StrategySelector


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def trending_uptrend_df():
    """Create clear uptrend data for pullback testing."""
    np.random.seed(42)
    n = 100

    # Strong uptrend
    trend = np.linspace(100, 110, n)
    noise = np.random.normal(0, 0.2, n)
    close = trend + noise

    df = pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.3,
        'low': close - 0.3,
        'close': close,
        'ema_20': trend,  # EMA follows trend
        'atr_14': np.full(n, 2.0),
        'adx_14': np.full(n, 30.0),  # Strong trend
    })

    return df


@pytest.fixture
def trending_downtrend_df():
    """Create clear downtrend data for pullback testing."""
    np.random.seed(43)
    n = 100

    # Strong downtrend
    trend = np.linspace(100, 90, n)
    noise = np.random.normal(0, 0.2, n)
    close = trend + noise

    df = pd.DataFrame({
        'open': close + 0.1,
        'high': close + 0.3,
        'low': close - 0.3,
        'close': close,
        'ema_20': trend,  # EMA follows trend
        'atr_14': np.full(n, 2.0),
        'adx_14': np.full(n, 30.0),  # Strong trend
    })

    return df


@pytest.fixture
def ranging_df():
    """Create tight ranging data."""
    np.random.seed(44)
    n = 100

    # Sideways movement
    base = 100
    noise = np.random.normal(0, 0.15, n)
    close = base + noise

    df = pd.DataFrame({
        'open': close - 0.05,
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
        'ema_20': np.full(n, 100.0),
        'atr_14': np.full(n, 0.3),  # Low volatility
        'adx_14': np.full(n, 15.0),  # Weak trend
    })

    return df


@pytest.fixture
def squeeze_release_df():
    """Create squeeze release scenario."""
    np.random.seed(45)
    n = 50

    # Compression then release
    close = np.concatenate([
        np.full(40, 100.0) + np.random.normal(0, 0.1, 40),  # Compression
        np.linspace(100, 105, 10)  # Breakout
    ])

    df = pd.DataFrame({
        'open': close - 0.05,
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
        'squeeze_on': np.concatenate([np.full(40, True), np.full(10, False)]),
        'ttm_momentum': np.linspace(0.1, 0.5, n),
        'kc_middle': np.full(n, 100.0),
        'atr_14': np.full(n, 0.5),
    })

    return df


@pytest.fixture
def pullback_strategy():
    """Create RegimePullbackStrategy instance."""
    return RegimePullbackStrategy(asset="US_TECH_100")


@pytest.fixture
def squeeze_strategy():
    """Create TTMSqueezeStrategy instance."""
    return TTMSqueezeStrategy(asset="US_TECH_100")


@pytest.fixture
def strategy_selector():
    """Create StrategySelector instance."""
    return StrategySelector(asset="US_TECH_100")


@pytest.fixture
def sample_long_position():
    """Create sample LONG position for testing."""
    return Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=96.0,
        target=114.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="PULLBACK_LONG"
    )


@pytest.fixture
def sample_short_position():
    """Create sample SHORT position for testing."""
    return Position(
        asset="US_TECH_100",
        direction=SignalDirection.SHORT,
        entry_price=100.0,
        stop_loss=104.0,
        target=86.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="PULLBACK_SHORT"
    )


# ============================================================================
# TEST BASE STRATEGY
# ============================================================================


class TestBaseStrategy:
    """Test abstract BaseStrategy class."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy(asset="US_TECH_100")

    def test_trade_setup_validation_long(self):
        """TradeSetup validates LONG setup correctly."""
        # Valid LONG setup
        setup = TradeSetup(
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=98.0,
            target=106.0,
            risk_per_share=2.0,
            confidence=85.0,
            setup_type="TEST_LONG"
        )

        assert setup.direction == SignalDirection.LONG
        assert setup.entry_price > setup.stop_loss
        assert setup.target > setup.entry_price
        assert setup.reward_risk_ratio() == 3.0  # (106-100) / 2.0

    def test_trade_setup_validation_long_invalid(self):
        """TradeSetup rejects invalid LONG setup."""
        # Entry below stop (invalid for LONG)
        with pytest.raises(AssertionError):
            TradeSetup(
                direction=SignalDirection.LONG,
                entry_price=98.0,  # Below stop
                stop_loss=100.0,
                target=106.0,
                risk_per_share=2.0,
                confidence=85.0,
                setup_type="INVALID_LONG"
            )

    def test_trade_setup_validation_short(self):
        """TradeSetup validates SHORT setup correctly."""
        # Valid SHORT setup
        setup = TradeSetup(
            direction=SignalDirection.SHORT,
            entry_price=100.0,
            stop_loss=102.0,
            target=94.0,
            risk_per_share=2.0,
            confidence=85.0,
            setup_type="TEST_SHORT"
        )

        assert setup.direction == SignalDirection.SHORT
        assert setup.entry_price < setup.stop_loss
        assert setup.target < setup.entry_price
        assert setup.reward_risk_ratio() == 3.0  # (100-94) / 2.0

    def test_position_current_pnl_long(self, sample_long_position):
        """Position calculates P&L correctly for LONG."""
        # At entry
        pnl = sample_long_position.current_pnl(100.0)
        assert pnl == 0.0

        # In profit
        pnl = sample_long_position.current_pnl(105.0)
        assert pnl == 5.0

        # In loss
        pnl = sample_long_position.current_pnl(97.0)
        assert pnl == -3.0

    def test_position_current_r_long(self, sample_long_position):
        """Position calculates R multiple correctly for LONG."""
        # At entry
        r = sample_long_position.current_r(100.0)
        assert r == 0.0

        # At 1R
        r = sample_long_position.current_r(104.0)
        assert r == 1.0

        # At 2.5R
        r = sample_long_position.current_r(110.0)
        assert r == 2.5

        # At -1R (full loss)
        r = sample_long_position.current_r(96.0)
        assert r == -1.0


# ============================================================================
# TEST REGIME PULLBACK STRATEGY
# ============================================================================


class TestRegimePullbackStrategy:
    """Test RegimePullbackStrategy (trend following)."""

    def test_asset_specific_parameters(self):
        """Strategy loads correct parameters for each asset."""
        # US indices
        nas_strategy = RegimePullbackStrategy(asset="US_TECH_100")
        assert nas_strategy.params['initial_stop_atr'] == 2.0
        assert nas_strategy.params['min_rrr'] == 2.5

        # Gold
        gold_strategy = RegimePullbackStrategy(asset="GOLD")
        assert gold_strategy.params['initial_stop_atr'] == 2.5
        assert gold_strategy.params['min_rrr'] == 2.0

    def test_check_entry_in_uptrend(self, pullback_strategy):
        """Detects bullish pullback entry in uptrend."""
        # Create bullish pullback scenario
        df = pd.DataFrame({
            'open': [100, 102, 101, 101, 100],
            'high': [101, 103, 102, 101.5, 101],
            'low': [99, 101, 100, 99, 97],  # Long lower wick (rejection)
            'close': [100.5, 102.5, 101, 100, 100.5],
            'ema_20': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = pullback_strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)

        assert setup is not None
        assert setup.direction == SignalDirection.LONG
        assert setup.setup_type == "PULLBACK_LONG"
        assert setup.entry_price == 100.5

    def test_check_entry_in_downtrend(self, pullback_strategy):
        """Detects bearish pullback entry in downtrend."""
        # Create bearish pullback scenario
        df = pd.DataFrame({
            'open': [100, 98, 99, 99, 100],
            'high': [101, 99, 100, 100.5, 103],  # Long upper wick (rejection)
            'low': [99, 97, 98, 99, 99],
            'close': [99.5, 97.5, 99, 100, 99.5],
            'ema_20': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = pullback_strategy.check_entry(df, regime="STRONG_TREND", confidence=80.0)

        assert setup is not None
        assert setup.direction == SignalDirection.SHORT
        assert setup.setup_type == "PULLBACK_SHORT"
        assert setup.entry_price == 99.5

    def test_no_entry_without_pullback(self, pullback_strategy):
        """No entry when price is too far from EMA."""
        # Price far from EMA
        df = pd.DataFrame({
            'open': [100, 102, 104, 106, 108],
            'high': [101, 103, 105, 107, 109],
            'low': [99, 101, 103, 105, 107],
            'close': [100.5, 102.5, 104.5, 106.5, 108.5],
            'ema_20': [100, 100, 100, 100, 100],  # Price way above EMA
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = pullback_strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)

        assert setup is None  # Too far from EMA

    def test_no_entry_without_confirmation(self, pullback_strategy):
        """No entry without confirmation candle."""
        # Price at EMA but no confirmation
        df = pd.DataFrame({
            'open': [100, 102, 101, 100.5, 100.3],
            'high': [101, 103, 102, 101, 100.8],
            'low': [99, 101, 100, 99.8, 99.8],
            'close': [100.5, 102.5, 101, 100, 100.5],  # Small candle, no clear confirmation
            'ema_20': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = pullback_strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)

        assert setup is None  # No confirmation candle

    def test_stop_loss_below_entry_for_long(self, pullback_strategy):
        """LONG setup has stop loss below entry."""
        df = pd.DataFrame({
            'open': [100, 102, 101, 101, 100],
            'high': [101, 103, 102, 101.5, 101],
            'low': [99, 101, 100, 99, 97],
            'close': [100.5, 102.5, 101, 100, 100.5],
            'ema_20': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = pullback_strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)

        assert setup is not None
        assert setup.stop_loss < setup.entry_price
        assert setup.stop_loss == 100.5 - (2.0 * 2.0)  # entry - (stop_atr * atr)

    def test_breakeven_move(self, pullback_strategy, sample_long_position):
        """Position moves to breakeven at 1.5R."""
        df = pd.DataFrame({
            'close': [106.0],  # 1.5R profit
            'atr_14': [2.0],
        })

        action, value = pullback_strategy.manage_exit(df, sample_long_position)

        assert action == ExitAction.BREAKEVEN
        assert value == sample_long_position.entry_price

    def test_trailing_stop(self, pullback_strategy, sample_long_position):
        """Position trails stop at 2.5R."""
        # Move to breakeven first
        sample_long_position.stop_loss = sample_long_position.entry_price

        df = pd.DataFrame({
            'close': [110.0],  # 2.5R profit
            'atr_14': [2.0],
        })

        action, value = pullback_strategy.manage_exit(df, sample_long_position)

        assert action == ExitAction.TRAIL
        assert value > sample_long_position.stop_loss
        expected_trail = 110.0 - (1.5 * 2.0)
        assert abs(value - expected_trail) < 0.01

    def test_target_exit(self, pullback_strategy, sample_long_position):
        """Position exits at target."""
        df = pd.DataFrame({
            'close': [114.5],  # Above target
            'atr_14': [2.0],
        })

        action, value = pullback_strategy.manage_exit(df, sample_long_position)

        assert action == ExitAction.TARGET
        assert value == sample_long_position.target

    def test_time_based_exit(self, pullback_strategy):
        """Position exits after max bars."""
        position = Position(
            asset="US_TECH_100",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=96.0,
            target=114.0,
            units=1.0,
            risk_per_share=4.0,
            entry_time=pd.Timestamp.now(),
            entry_bar=0,
            entry_strategy="PULLBACK_LONG"
        )

        # Create DataFrame with 21 bars (exceeds max_bars=20)
        df = pd.DataFrame({
            'close': [101.0] * 21,
            'atr_14': [2.0] * 21,
        })

        action, value = pullback_strategy.manage_exit(df, position)

        assert action == ExitAction.TIME_EXIT
        assert value == 101.0  # Current price


# ============================================================================
# TEST TTM SQUEEZE STRATEGY
# ============================================================================


class TestTTMSqueezeStrategy:
    """Test TTMSqueezeStrategy (breakout trading)."""

    def test_tighter_stops_than_regime(self):
        """TTM strategy has tighter stops than pullback."""
        pullback = RegimePullbackStrategy(asset="US_TECH_100")
        squeeze = TTMSqueezeStrategy(asset="US_TECH_100")

        assert squeeze.params['initial_stop_atr'] < pullback.params['initial_stop_atr']
        assert squeeze.params['min_rrr'] < pullback.params['min_rrr']

    def test_earlier_breakeven(self):
        """TTM strategy moves to breakeven earlier."""
        pullback = RegimePullbackStrategy(asset="US_TECH_100")
        squeeze = TTMSqueezeStrategy(asset="US_TECH_100")

        assert squeeze.params['breakeven_r'] < pullback.params['breakeven_r']
        assert squeeze.params['trail_start_r'] < pullback.params['trail_start_r']

    def test_entry_on_squeeze_release(self, squeeze_strategy):
        """Detects entry on squeeze release."""
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [101, 101, 101, 101, 102],
            'low': [99, 99, 99, 99, 99],
            'close': [100, 100, 100, 100, 101],
            'squeeze_on': [True, True, True, True, False],  # Just released
            'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],
            'kc_middle': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        setup = squeeze_strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=90.0)

        assert setup is not None
        assert setup.direction == SignalDirection.LONG
        assert setup.setup_type == "SQUEEZE_RELEASE_LONG"

    def test_entry_on_retest(self, squeeze_strategy):
        """Detects entry on squeeze retest."""
        df = pd.DataFrame({
            'open': [100, 100, 102, 101, 100.2],
            'high': [101, 101, 103, 102, 101],
            'low': [99, 99, 101, 99.5, 98],  # Long lower wick (rejection)
            'close': [100, 100, 102.5, 100.5, 100.3],
            'squeeze_on': [True, True, False, False, False],  # Released 2 bars ago
            'ttm_momentum': [0.1, 0.15, 0.3, 0.25, 0.28],
            'kc_middle': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        setup = squeeze_strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)

        assert setup is not None
        assert setup.setup_type == "SQUEEZE_RETEST_LONG"

    def test_no_entry_during_squeeze(self, squeeze_strategy):
        """No entry while squeeze is still active."""
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [101, 101, 101, 101, 101],
            'low': [99, 99, 99, 99, 99],
            'close': [100, 100, 100, 100, 100],
            'squeeze_on': [True, True, True, True, True],  # Still ON
            'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],
            'kc_middle': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        setup = squeeze_strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)

        assert setup is None  # Squeeze still active

    def test_momentum_reversal_exit(self, squeeze_strategy):
        """Position exits on momentum reversal when in profit."""
        position = Position(
            asset="US_TECH_100",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=97.0,
            target=105.4,
            units=1.0,
            risk_per_share=3.0,
            entry_time=pd.Timestamp.now(),
            entry_bar=0,
            entry_strategy="SQUEEZE_RELEASE_LONG",
            metadata={'ttm_momentum': 0.3}  # Positive at entry
        )

        df = pd.DataFrame({
            'close': [102.0],  # 0.67R profit
            'atr_14': [2.0],
            'ttm_momentum': [-0.1],  # Reversed to negative
        })

        action, value = squeeze_strategy.manage_exit(df, position)

        assert action == ExitAction.TARGET  # Exit at market
        assert value == 102.0


# ============================================================================
# TEST STRATEGY SELECTOR
# ============================================================================


class TestStrategySelector:
    """Test StrategySelector (intelligent routing)."""

    def test_initialization(self, strategy_selector):
        """Selector initializes with both strategies."""
        assert strategy_selector.regime_strategy is not None
        assert strategy_selector.ttm_strategy is not None
        assert strategy_selector.asset == "US_TECH_100"

    def test_routes_to_regime_in_trend(self, strategy_selector):
        """Routes to RegimePullbackStrategy in trending regime."""
        df = pd.DataFrame({
            'open': [100, 102, 101, 101, 100],
            'high': [101, 103, 102, 101.5, 101],
            'low': [99, 101, 100, 99, 97],
            'close': [100.5, 102.5, 101, 100, 100.5],
            'ema_20': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
            'adx_14': [25, 25, 25, 25, 25],
        })

        setup = strategy_selector.check_entry(
            df=df,
            regime="STRONG_TREND",
            confidence=85.0,
            strategy_recommendation="REGIME_PULLBACK"
        )

        assert setup is not None
        assert "PULLBACK" in setup.setup_type

    def test_routes_to_ttm_in_squeeze(self, strategy_selector):
        """Routes to TTMSqueezeStrategy in compression regime."""
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 100],
            'high': [101, 101, 101, 101, 102],
            'low': [99, 99, 99, 99, 99],
            'close': [100, 100, 100, 100, 101],
            'squeeze_on': [True, True, True, True, False],
            'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],
            'kc_middle': [100, 100, 100, 100, 100],
            'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        })

        setup = strategy_selector.check_entry(
            df=df,
            regime="RANGE_COMPRESSION",
            confidence=90.0,
            strategy_recommendation="TTM_SQUEEZE"
        )

        assert setup is not None
        assert "SQUEEZE" in setup.setup_type

    def test_no_trade_in_high_volatility(self, strategy_selector):
        """No setup in HIGH_VOLATILITY regime."""
        df = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100],
        })

        setup = strategy_selector.check_entry(
            df=df,
            regime="HIGH_VOLATILITY",
            confidence=0.0,
            strategy_recommendation="NONE"
        )

        assert setup is None

    def test_exit_uses_correct_strategy(self, strategy_selector):
        """Exit routing uses correct strategy based on entry_strategy."""
        # PULLBACK position
        pullback_pos = Position(
            asset="US_TECH_100",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=96.0,
            target=114.0,
            units=1.0,
            risk_per_share=4.0,
            entry_time=pd.Timestamp.now(),
            entry_bar=0,
            entry_strategy="PULLBACK_LONG"
        )

        df = pd.DataFrame({
            'close': [101.0],
            'atr_14': [2.0],
            'ttm_momentum': [0.3],
        })

        action, value = strategy_selector.manage_exit(df, pullback_pos)
        assert action == ExitAction.HOLD

        # SQUEEZE position
        squeeze_pos = Position(
            asset="US_TECH_100",
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=97.0,
            target=105.4,
            units=1.0,
            risk_per_share=3.0,
            entry_time=pd.Timestamp.now(),
            entry_bar=0,
            entry_strategy="SQUEEZE_RELEASE_LONG",
            metadata={'ttm_momentum': 0.3}
        )

        action, value = strategy_selector.manage_exit(df, squeeze_pos)
        assert action == ExitAction.HOLD

    def test_get_strategy_stats(self, strategy_selector):
        """get_strategy_stats returns both strategy parameters."""
        stats = strategy_selector.get_strategy_stats()

        assert 'asset' in stats
        assert 'regime_params' in stats
        assert 'ttm_params' in stats
        assert stats['asset'] == "US_TECH_100"
        assert stats['regime_params']['initial_stop_atr'] == 2.0
        assert stats['ttm_params']['initial_stop_atr'] == 1.5

    def test_validate_setup(self, strategy_selector):
        """validate_setup checks RRR meets minimum."""
        # Valid setup
        valid_setup = TradeSetup(
            direction=SignalDirection.LONG,
            entry_price=100.0,
            stop_loss=96.0,
            target=110.0,
            risk_per_share=4.0,
            confidence=85.0,
            setup_type="PULLBACK_LONG"
        )

        is_valid = strategy_selector.validate_setup(valid_setup)
        assert is_valid is True


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
