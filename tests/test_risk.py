import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.risk.limits import RiskLimits, RiskProfile
from src.risk.position_sizing import PositionSizer
from src.risk.governor import RiskGovernor


class TestRiskLimits:
    """Test risk limit definitions"""

    def test_micro_profile_for_small_capital(self):
        """Test MICRO profile assigned to small capital"""
        profile = RiskLimits.get_profile(1000)
        assert profile.profile_name == 'MICRO'
        assert profile.base_risk_pct == 0.020

    def test_medium_profile_for_medium_capital(self):
        """Test MEDIUM profile assigned"""
        profile = RiskLimits.get_profile(7500)
        assert profile.profile_name == 'MEDIUM'
        assert profile.base_risk_pct == 0.015

    def test_profile_validation_prevents_invalid(self):
        """Test that invalid profiles are rejected"""
        with pytest.raises(AssertionError):
            RiskProfile(
                profile_name='TEST',
                base_risk_pct=0.10,  # Too high!
                max_daily_loss_pct=0.03,
                max_drawdown_pct=0.12,
                min_position_value=50,
                dd_scale_threshold_1=0.05,
                dd_scale_threshold_2=0.08,
                max_concurrent_positions=2,
                max_position_size_pct=0.25
            )

    def test_dd_thresholds_ordered_correctly(self):
        """Test drawdown thresholds in correct order"""
        profile = RiskLimits.get_profile(5000)
        assert profile.dd_scale_threshold_1 < profile.dd_scale_threshold_2
        assert profile.dd_scale_threshold_2 < profile.max_drawdown_pct


class TestPositionSizer:
    """Test position sizing calculations"""

    @pytest.fixture
    def profile(self):
        return RiskLimits.get_profile(10000)

    @pytest.fixture
    def sizer(self, profile):
        return PositionSizer(profile)

    def test_calculate_basic_size(self, sizer):
        """Test basic position size calculation"""
        units = sizer.calculate_size(
            capital=10000,
            risk_pct=0.015,  # 1.5%
            risk_per_share=2.0,
            price=100.0,
            asset='US_TECH_100'
        )

        # Risk capital = 10000 * 0.015 = 150
        # Units = 150 / 2.0 = 75
        # But capped at max_position_size_pct (20% for LARGE profile)
        # Max position value = 10000 * 0.20 = 2000
        # Max units = 2000 / 100 = 20
        assert units == pytest.approx(20.0, rel=0.1)

    def test_respects_broker_minimum(self, sizer):
        """Test broker minimum is respected"""
        # Very small position
        units = sizer.calculate_size(
            capital=1000,
            risk_pct=0.005,  # 0.5%
            risk_per_share=50.0,  # Large stop
            price=100.0,
            asset='US_TECH_100'
        )

        # Would calculate to 0.1, which is the minimum
        assert units >= 0.1

    def test_respects_max_position_size(self, sizer):
        """Test max position size limit"""
        units = sizer.calculate_size(
            capital=10000,
            risk_pct=0.03,  # 3%
            risk_per_share=0.5,  # Very tight stop
            price=100.0,
            asset='US_TECH_100'
        )

        # Position value should not exceed 25% of capital
        position_value = units * 100.0
        assert position_value <= 10000 * 0.25

    def test_forex_rounding(self, sizer):
        """Test forex units rounded to 1000s"""
        units = sizer.calculate_size(
            capital=10000,
            risk_pct=0.015,
            risk_per_share=0.001,
            price=1.1,
            asset='EUR_USD'
        )

        # Should be multiple of 1000
        assert units % 1000 == 0

    def test_validation_detects_oversized_position(self, sizer):
        """Test position validation catches issues"""
        result = sizer.validate_position_size(
            units=1000,
            price=100.0,
            capital=10000,
            risk_per_share=5.0
        )

        # Position value = 100,000 (way too large)
        # But risk check happens first: 1000 * 5.0 / 10000 = 50% risk
        assert not result['valid']
        assert 'risk too high' in result['reason'].lower()


class TestRiskGovernor:
    """Test risk governor"""

    @pytest.fixture
    def governor(self):
        return RiskGovernor(initial_capital=10000)

    def test_initial_state(self, governor):
        """Test governor initializes correctly"""
        assert governor.current_capital == 10000
        assert governor.peak_capital == 10000
        assert governor.current_dd_pct == 0.0
        assert governor.consecutive_wins == 0
        assert governor.consecutive_losses == 0

    def test_allows_trade_in_normal_conditions(self, governor):
        """Test trade allowed in normal conditions"""
        allowed, reason, risk = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
        )

        assert allowed
        assert reason == "APPROVED"
        assert risk > 0

    def test_blocks_trade_after_max_drawdown(self, governor):
        """Test trading blocked at max drawdown"""
        # Simulate losses to trigger DD
        governor.current_capital = 8800  # 12% loss
        governor.current_dd_pct = 0.12

        allowed, reason, risk = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85}
        )

        assert not allowed
        assert reason == "MAX_DRAWDOWN_REACHED"

    def test_blocks_trade_after_daily_loss_limit(self, governor):
        """Test trading blocked after daily loss limit"""
        governor.current_date = datetime.now().date()
        governor.daily_pnl = -300  # 3% loss

        allowed, reason, risk = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85}
        )

        assert not allowed
        assert reason == "DAILY_LOSS_LIMIT"

    def test_blocks_trade_after_loss_streak(self, governor):
        """Test trading blocked after 3 losses"""
        governor.consecutive_losses = 4

        allowed, reason, risk = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85}
        )

        assert not allowed
        assert reason == "LOSS_STREAK_LIMIT"

    def test_reduces_risk_in_drawdown(self, governor):
        """Test risk is reduced in drawdown"""
        # Normal risk first
        _, _, risk_normal = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
        )

        # Simulate drawdown
        governor.current_capital = 9500
        governor.current_dd_pct = 0.05

        # Risk in DD
        _, _, risk_dd = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
        )

        assert risk_dd < risk_normal

    def test_increases_risk_on_win_streak(self, governor):
        """Test risk increases slightly on win streak"""
        # Normal risk
        _, _, risk_normal = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
        )

        # Win streak
        governor.consecutive_wins = 3

        _, _, risk_streak = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
        )

        assert risk_streak > risk_normal

    def test_daily_reset_clears_counters(self, governor):
        """Test daily reset clears daily counters"""
        # Set some daily data
        governor.current_date = datetime.now().date()
        governor.daily_pnl = -100
        governor.daily_trades = [{'test': 'data'}]

        # Reset with new date
        tomorrow = datetime.now() + timedelta(days=1)
        governor.check_daily_reset(tomorrow)

        assert governor.daily_pnl == 0
        assert len(governor.daily_trades) == 0

    def test_record_trade_updates_metrics(self, governor):
        """Test trade recording updates all metrics"""
        initial_capital = governor.current_capital

        # Record a winning trade
        governor.record_trade(
            asset='US_TECH_100',
            direction='LONG',
            entry_price=100.0,
            exit_price=105.0,
            units=10,
            outcome='WIN',
            setup_type='PULLBACK_LONG',
            exit_reason='TARGET'
        )

        # Check updates
        assert governor.current_capital > initial_capital
        assert governor.consecutive_wins == 1
        assert governor.consecutive_losses == 0
        assert len(governor.trade_log) == 1

    def test_health_status_degrades_with_losses(self, governor):
        """Test health status reflects problems"""
        # Initial health
        health_initial = governor.get_health_status()
        assert health_initial['status'] == 'HEALTHY'

        # Simulate problems
        governor.current_dd_pct = 0.08
        governor.consecutive_losses = 2

        health_problem = governor.get_health_status()
        assert health_problem['score'] < health_initial['score']
        assert len(health_problem['warnings']) > 0

    def test_should_pause_detects_issues(self, governor):
        """Test pause detection"""
        # Normal
        pause, reason, _ = governor.should_pause_trading()
        assert not pause

        # Max DD
        governor.current_dd_pct = 0.12
        pause, reason, _ = governor.should_pause_trading()
        assert pause
        assert reason == "MAX_DRAWDOWN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
