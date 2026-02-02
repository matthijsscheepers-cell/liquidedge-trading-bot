"""
Multi-Timeframe TTM Squeeze Pullback Strategy

Gebruiker's eigen strategie:
- 1H chart als anchor (trend confirmatie)
- 15min chart voor execution (timing entries)
- Entry op pullback naar -1 ATR
- Target op +2 ATR expansion

Entry Criteria:
    1. 1H TTM histogram: Bullish firing (yellow/light blue/dark blue)
    2. 15min TTM histogram: Bullish firing (yellow/light blue/dark blue)
    3. 15min Squeeze: Active (red/orange/black dots)
    4. Price: Pullback to -1 ATR level

Entry Execution:
    Entry: -1 ATR (below 21-EMA)
    Stop: Below -2 ATR
    Target: +2 ATR (above 21-EMA)
    Direction: LONG ONLY

The Edge:
    - Buy the dip in a confirmed uptrend (multi-timeframe alignment)
    - Squeeze indicates compression (energy building)
    - Pullback entry = better risk/reward
    - Target expansion = ride the move
"""

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from .base import (
    BaseStrategy,
    TradeSetup,
    Position,
    SignalDirection,
    ExitAction,
)


class TTMSqueezePullbackStrategy(BaseStrategy):
    """
    Multi-timeframe TTM Squeeze pullback strategy.

    Uses 1H chart for trend confirmation and 15min chart for execution.
    Enters on pullback to -1 ATR, exits at +2 ATR target.
    """

    def _get_asset_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters.

        All assets use same parameters for this strategy:
        - 21-EMA basis for Keltner Channels
        - ATR-based entry/stop/target levels
        - No RRR calculation (fixed ATR levels)
        """
        # Universal parameters (same for all assets)
        return {
            'ema_period': 21,           # EMA basis for Keltner Channels
            'atr_period': 20,           # ATR period (matches user's KC settings)
            'entry_atr': -1.0,          # Entry at -1 ATR (pullback)
            'entry_tolerance': 0.5,     # Accept entries within 0.5 ATR of target
            'stop_atr': -2.1,           # Stop below -2 ATR
            'target_atr': 2.0,          # Target at +2 ATR
            'min_momentum_1h': 0.0,     # Minimum 1H momentum (positive = bullish)
            'min_momentum_15m': 0.0,    # Minimum 15min momentum (positive = bullish)
        }

    def check_entry(self,
                   df_15min: pd.DataFrame,
                   df_1h: pd.DataFrame,
                   regime: str,
                   confidence: float) -> Optional[TradeSetup]:
        """
        Check for pullback entry setup.

        Multi-timeframe logic:
        1. Align 1H and 15min data to current 15min bar
        2. Check 1H histogram is bullish firing (trend confirmation)
        3. Check 15min squeeze is active (compression)
        4. Check price touches -1 ATR level (pullback entry)
        5. Calculate entry/stop/target based on 21-EMA + ATR

        Note: No 15min momentum requirement - pullbacks often show
        temporary bearish momentum (red bars) which is expected behavior.

        Args:
            df_15min: 15-minute DataFrame with indicators
            df_1h: 1-hour DataFrame with indicators
            regime: Current market regime (not used)
            confidence: Regime confidence (not used)

        Returns:
            TradeSetup if valid setup found, None otherwise
        """
        # Get ATR column name based on period
        atr_col = f'atr_{self.params["atr_period"]}'

        # Validate 15min indicators
        self._validate_indicators_present(
            df_15min,
            ['open', 'high', 'low', 'close', 'ema_21', atr_col,
             'ttm_momentum', 'squeeze_on']
        )

        # Validate 1H indicators
        self._validate_indicators_present(
            df_1h,
            ['ttm_momentum']
        )

        # Need sufficient data
        if len(df_15min) < 50 or len(df_1h) < 20:
            return None

        # Get current bars
        current_15m = df_15min.iloc[-1]
        current_1h = df_1h.iloc[-1]

        # Get previous momentum for direction detection
        prev_15m = df_15min.iloc[-2]
        prev_1h = df_1h.iloc[-2]

        # === 1. CHECK 1H HISTOGRAM IS BULLISH FIRING ===
        # Bullish = positive momentum AND increasing OR high enough
        momentum_1h = current_1h['ttm_momentum']
        prev_momentum_1h = prev_1h['ttm_momentum']

        # Must be positive (bullish)
        if momentum_1h <= self.params['min_momentum_1h']:
            return None

        # Bullish firing = increasing OR staying strong
        # Colors: dark blue (decreasing) -> yellow (flat) -> light blue (increasing)
        # We want: yellow, light blue, or dark blue (all positive momentum)
        is_bullish_1h = momentum_1h > 0

        if not is_bullish_1h:
            return None

        # === 2. CHECK 15MIN SQUEEZE IS ACTIVE ===
        # During a pullback, 15M momentum can be red/yellow (bearish/neutral)
        # The 1H bullish provides trend confirmation
        # No 15M momentum requirement - we want to catch the pullback LOW

        # Get 15M momentum for metadata (not used as filter)
        momentum_15m = current_15m['ttm_momentum']

        # Squeeze active = red/orange/black dots = squeeze_on = True
        squeeze_active = current_15m['squeeze_on']

        if not squeeze_active:
            return None

        # === 3. CHECK PRICE NEAR -1 ATR LEVEL ===
        ema_21 = current_15m['ema_21']
        atr = current_15m[atr_col]

        # Calculate ATR levels
        entry_target = ema_21 + (self.params['entry_atr'] * atr)  # -1 ATR target
        tolerance = self.params['entry_tolerance'] * atr           # 0.5 ATR tolerance

        stop_level = ema_21 + (self.params['stop_atr'] * atr)    # -2.1 ATR
        target_level = ema_21 + (self.params['target_atr'] * atr) # +2 ATR

        # Check if bar's low is within tolerance of -1 ATR level (pullback)
        # Accept entries within ±0.5 ATR of the target level
        bar_low = current_15m['low']
        bar_high = current_15m['high']

        # Distance from bar's low to entry target (in ATR units)
        distance_from_target = (bar_low - entry_target) / atr

        # Accept if within tolerance (e.g., within 0.5 ATR of -1 ATR target)
        if distance_from_target > tolerance:  # Didn't pull back enough
            return None

        # But not be completely below (want to catch the pullback, not a crash)
        if distance_from_target < -tolerance:  # Too far below
            return None

        # Entry price is at the actual bar low (within tolerance range)
        entry_level = entry_target  # Still enter at -1 ATR level for consistency

        # === 4. CREATE TRADE SETUP ===
        # Entry at -1 ATR level (limit order filled at pullback)
        entry_price = entry_level
        stop_loss = stop_level
        target = target_level

        # Risk calculation
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:  # Invalid setup
            return None

        return TradeSetup(
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            risk_per_share=risk_per_share,
            confidence=confidence,
            setup_type="TTM_PULLBACK_LONG",
            metadata={
                'ema_21': ema_21,
                'atr': atr,
                'atr_period': self.params['atr_period'],
                'momentum_1h': momentum_1h,
                'momentum_15m': momentum_15m,
                'squeeze_active': squeeze_active,
                'entry_level': entry_level,
                'regime': regime,
            }
        )

    def manage_exit(self,
                   df: pd.DataFrame,
                   position: Position) -> tuple[ExitAction, Optional[float]]:
        """
        Manage exit for pullback position.

        Simple exit logic:
        1. Stop loss hit -> exit
        2. Target hit -> exit
        3. Hold otherwise

        No trailing stops, no time exits - just stop and target.

        Args:
            df: DataFrame with current price data
            position: Active position to manage

        Returns:
            Tuple of (ExitAction, new_value)
        """
        # Validate required data
        self._validate_indicators_present(df, ['close', 'high', 'low'])

        current = df.iloc[-1]
        current_price = current['close']

        # LONG position management
        if position.direction == SignalDirection.LONG:
            # Check stop loss
            if current['low'] <= position.stop_loss:
                return ExitAction.STOP, position.stop_loss

            # Check target
            if current['high'] >= position.target:
                return ExitAction.TARGET, position.target

        # Hold position
        return ExitAction.HOLD, None


# Export
__all__ = ['TTMSqueezePullbackStrategy']
