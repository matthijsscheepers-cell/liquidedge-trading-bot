from typing import Dict, Optional
import pandas as pd
from src.risk.limits import RiskProfile


class PositionSizer:
    """
    Calculate position sizes based on risk parameters.

    Handles:
    - Risk-based sizing
    - Broker minimum requirements
    - Portfolio concentration limits
    - Margin requirements

    Example:
        sizer = PositionSizer(profile)
        units = sizer.calculate_size(
            capital=10000,
            risk_pct=0.015,
            risk_per_share=2.0,
            price=100.0
        )
    """

    def __init__(self, profile: RiskProfile):
        """
        Initialize position sizer

        Args:
            profile: Risk profile with limits
        """
        self.profile = profile

        # Broker minimums (Capital.com specific)
        self.BROKER_MINIMUMS = {
            'US_TECH_100': 0.1,    # 0.1 contracts minimum
            'US_SPX_500': 0.1,
            'GOLD': 0.1,
            'EUR_USD': 1000,       # 1000 units (micro lot)
            'GBP_USD': 1000,
        }

    def calculate_size(self,
                      capital: float,
                      risk_pct: float,
                      risk_per_share: float,
                      price: float,
                      asset: str = 'US_TECH_100') -> float:
        """
        Calculate position size in units

        Formula:
        1. Risk capital = capital * risk_pct
        2. Units = risk_capital / risk_per_share
        3. Adjust for minimums
        4. Cap at max position size

        Args:
            capital: Current account capital
            risk_pct: Percentage of capital to risk
            risk_per_share: Risk per unit (e.g., distance to stop)
            price: Current market price
            asset: Trading instrument

        Returns:
            Position size in units (rounded to valid increment)
        """

        # Step 1: Calculate risk capital
        risk_capital = capital * risk_pct

        # Step 2: Calculate units based on risk
        if risk_per_share <= 0:
            raise ValueError(f"Invalid risk_per_share: {risk_per_share}")

        units = risk_capital / risk_per_share

        # Step 3: Apply broker minimums
        min_units = self._get_minimum_units(asset)

        if units < min_units:
            # Position too small
            # Option 1: Return 0 (no trade)
            # Option 2: Use minimum (increases risk slightly)

            # Check if using minimum violates max risk
            min_risk = (min_units * risk_per_share) / capital

            if min_risk > 0.04:  # Hard cap at 4% risk
                return 0.0  # Position too risky even at minimum

            units = min_units

        # Step 4: Round to valid increment
        units = self._round_to_increment(units, asset)

        # Step 5: Apply max position size limit
        max_position_value = capital * self.profile.max_position_size_pct
        max_units = max_position_value / price

        units = min(units, max_units)

        # Step 6: Final rounding
        units = self._round_to_increment(units, asset)

        return units

    def _get_minimum_units(self, asset: str) -> float:
        """Get minimum units for asset"""

        # Try exact match
        if asset in self.BROKER_MINIMUMS:
            return self.BROKER_MINIMUMS[asset]

        # Try partial match
        for key in self.BROKER_MINIMUMS:
            if key in asset or asset in key:
                return self.BROKER_MINIMUMS[key]

        # Default to 0.1 for indices/commodities
        return 0.1

    def _round_to_increment(self, units: float, asset: str) -> float:
        """
        Round units to valid increment

        Capital.com allows:
        - Indices: 0.1 increment
        - Forex: 1000 increment (micro lots)
        """

        min_units = self._get_minimum_units(asset)

        if min_units >= 1000:
            # Forex - round to nearest 1000
            return round(units / 1000) * 1000
        else:
            # Indices/commodities - round to nearest 0.1
            return round(units / 0.1) * 0.1

    def validate_position_size(self,
                              units: float,
                              price: float,
                              capital: float,
                              risk_per_share: float) -> Dict[str, any]:
        """
        Validate if position size is acceptable

        Returns dict with:
        - valid: bool
        - reason: str (if invalid)
        - effective_risk_pct: float
        - position_value: float
        - position_pct: float
        """

        position_value = units * price
        position_pct = position_value / capital

        # Calculate effective risk
        risk_capital = units * risk_per_share
        effective_risk_pct = risk_capital / capital

        # Validations
        if units == 0:
            return {
                'valid': False,
                'reason': 'Position size is zero',
                'effective_risk_pct': 0,
                'position_value': 0,
                'position_pct': 0
            }

        if effective_risk_pct > 0.04:
            return {
                'valid': False,
                'reason': f'Risk too high: {effective_risk_pct*100:.1f}%',
                'effective_risk_pct': effective_risk_pct,
                'position_value': position_value,
                'position_pct': position_pct
            }

        if position_pct > self.profile.max_position_size_pct:
            return {
                'valid': False,
                'reason': f'Position too large: {position_pct*100:.1f}% of capital',
                'effective_risk_pct': effective_risk_pct,
                'position_value': position_value,
                'position_pct': position_pct
            }

        if position_value < self.profile.min_position_value:
            return {
                'valid': False,
                'reason': f'Position too small: €{position_value:.0f} (min €{self.profile.min_position_value})',
                'effective_risk_pct': effective_risk_pct,
                'position_value': position_value,
                'position_pct': position_pct
            }

        return {
            'valid': True,
            'reason': 'OK',
            'effective_risk_pct': effective_risk_pct,
            'position_value': position_value,
            'position_pct': position_pct
        }
