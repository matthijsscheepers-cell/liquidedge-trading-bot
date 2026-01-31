from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskProfile:
    """
    Risk profile based on account size.

    Different capital sizes need different risk parameters.
    Smaller accounts can be slightly more aggressive (within safe limits).
    """
    profile_name: str
    base_risk_pct: float          # Base risk per trade
    max_daily_loss_pct: float     # Max daily loss before stop
    max_drawdown_pct: float       # Max drawdown before shutdown
    min_position_value: float     # Minimum position size (broker limits)
    dd_scale_threshold_1: float   # First drawdown scaling level
    dd_scale_threshold_2: float   # Second drawdown scaling level
    max_concurrent_positions: int # Max positions at once
    max_position_size_pct: float  # Max % of capital per position

    def __post_init__(self):
        """Validate risk profile"""
        assert 0 < self.base_risk_pct <= 0.03, "Base risk must be 0-3%"
        assert 0 < self.max_daily_loss_pct <= 0.05, "Daily loss must be 0-5%"
        assert 0 < self.max_drawdown_pct <= 0.20, "Max DD must be 0-20%"
        assert self.dd_scale_threshold_1 < self.dd_scale_threshold_2, "DD thresholds wrong order"
        assert self.dd_scale_threshold_2 < self.max_drawdown_pct, "DD threshold 2 must be < max DD"


class RiskLimits:
    """
    Risk limit definitions for different account sizes.

    Provides appropriate risk profiles based on capital.
    """

    @staticmethod
    def get_profile(capital: float) -> RiskProfile:
        """
        Get risk profile for given capital

        Args:
            capital: Account size

        Returns:
            RiskProfile with appropriate limits
        """
        if capital < 2000:
            # MICRO account (€1000 - €2000)
            return RiskProfile(
                profile_name='MICRO',
                base_risk_pct=0.020,        # 2.0% per trade
                max_daily_loss_pct=0.04,    # 4% max daily loss
                max_drawdown_pct=0.15,      # 15% max drawdown
                min_position_value=50,      # €50 minimum
                dd_scale_threshold_1=0.06,  # Scale at 6% DD
                dd_scale_threshold_2=0.10,  # More scaling at 10% DD
                max_concurrent_positions=2,
                max_position_size_pct=0.30  # Max 30% per position
            )

        elif capital < 5000:
            # SMALL account (€2000 - €5000)
            return RiskProfile(
                profile_name='SMALL',
                base_risk_pct=0.018,
                max_daily_loss_pct=0.035,
                max_drawdown_pct=0.13,
                min_position_value=50,
                dd_scale_threshold_1=0.05,
                dd_scale_threshold_2=0.08,
                max_concurrent_positions=2,
                max_position_size_pct=0.25
            )

        elif capital < 10000:
            # MEDIUM account (€5000 - €10000)
            return RiskProfile(
                profile_name='MEDIUM',
                base_risk_pct=0.015,
                max_daily_loss_pct=0.03,
                max_drawdown_pct=0.12,
                min_position_value=100,
                dd_scale_threshold_1=0.05,
                dd_scale_threshold_2=0.08,
                max_concurrent_positions=3,
                max_position_size_pct=0.25
            )

        else:
            # LARGE account (>€10000)
            return RiskProfile(
                profile_name='LARGE',
                base_risk_pct=0.012,
                max_daily_loss_pct=0.025,
                max_drawdown_pct=0.10,
                min_position_value=200,
                dd_scale_threshold_1=0.04,
                dd_scale_threshold_2=0.07,
                max_concurrent_positions=3,
                max_position_size_pct=0.20
            )

    @staticmethod
    def get_all_profiles() -> Dict[str, RiskProfile]:
        """Get all available risk profiles"""
        return {
            'MICRO': RiskLimits.get_profile(1000),
            'SMALL': RiskLimits.get_profile(3000),
            'MEDIUM': RiskLimits.get_profile(7500),
            'LARGE': RiskLimits.get_profile(15000)
        }
