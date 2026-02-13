"""
Multi-Timeframe EMA Pullback Strategy

Entry on pullback to 21-EMA in an uptrend:
- 1H chart als anchor (trend confirmatie via Close > EMA(21))
- 15min chart voor execution (timing entries)
- Entry at EMA(21) level (limit order)
- Stop at EMA - 2.0 ATR
- Target at EMA + 2.0 ATR

Entry Criteria:
    1. 1H Close > EMA(21) (bullish trend)
    2. 15min histogram color != red (momentum not negative AND falling)
    3. Price pulls back to EMA(21) level
    4. NO squeeze requirement

Optimized parameters (backtest 2020-2026, GOLD+US500, 1-min execution):
    - GOLD: 89.0% win rate, PF 24.94 (Scale A), max drawdown -20.9%
    - US500: 70.6% win rate, PF 6.35 (Scale A)
    - Risk per trade: 2.0 ATR, Reward: 2.0 ATR (R:R = 1:1)
    - Circuit breakers: 2 consecutive stops -> 4h cooldown
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


def get_histogram_color(mom_curr, mom_prev):
    """Determine TTM histogram bar color.

    Colors:
        light_blue: momentum > 0 AND rising
        dark_blue:  momentum > 0 AND falling
        yellow:     momentum <= 0 AND rising
        red:        momentum <= 0 AND falling
    """
    if pd.isna(mom_curr) or pd.isna(mom_prev):
        return 'red'
    if mom_curr > 0 and mom_curr > mom_prev:
        return 'light_blue'
    elif mom_curr > 0:
        return 'dark_blue'
    elif mom_curr > mom_prev:
        return 'yellow'
    else:
        return 'red'


class TTMSqueezePullbackStrategy(BaseStrategy):
    """
    Multi-timeframe EMA pullback strategy.

    Uses 1H chart for trend confirmation and 15min chart for execution.
    Enters on pullback to EMA(21), exits at +2.0 ATR target.
    """

    def _get_asset_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters.

        All assets use same parameters:
        - 21-EMA basis for entry level
        - ATR-based stop/target levels
        """
        # Optimized via 1-minute execution backtest (2020-2026, GOLD+US500)
        # GOLD: 89.0% WR, PF 24.94 (Scale A), -20.9% max DD
        # US500: 70.6% WR, PF 6.35 (Scale A)
        return {
            'ema_period': 21,           # EMA basis
            'atr_period': 20,           # ATR period
            'entry_atr': 0.0,           # Entry at EMA(21) itself
            'entry_tolerance': 0.5,     # Accept entries within 0.5 ATR of target
            'stop_atr': -2.0,           # Stop at EMA - 2.0 ATR
            'target_atr': 2.0,          # Target at EMA + 2.0 ATR
        }

    def check_entry(self,
                   df_15min: pd.DataFrame,
                   df_1h: pd.DataFrame,
                   regime: str,
                   confidence: float) -> Optional[TradeSetup]:
        """
        Check for pullback entry setup.

        Multi-timeframe logic:
        1. Check 1H Close > EMA(21) (bullish trend)
        2. Check 15min histogram color is NOT red (not negative AND falling)
        3. Check price touches EMA(21) level (pullback entry)
        4. Calculate entry/stop/target based on 21-EMA + ATR

        No squeeze requirement — removed after optimization showed
        it filters out profitable trades without improving win rate.
        """
        # Get ATR column name based on period
        atr_col = f'atr_{self.params["atr_period"]}'

        # Validate 15min indicators
        self._validate_indicators_present(
            df_15min,
            ['open', 'high', 'low', 'close', 'ema_21', atr_col,
             'ttm_momentum']
        )

        # Validate 1H indicators
        self._validate_indicators_present(
            df_1h,
            ['close', 'ema_21']
        )

        # Need sufficient data
        if len(df_15min) < 50 or len(df_1h) < 20:
            return None

        # Get current and previous bars
        current_15m = df_15min.iloc[-1]
        prev_15m = df_15min.iloc[-2]
        current_1h = df_1h.iloc[-1]

        # === 1. CHECK 1H CLOSE > EMA(21) ===
        close_1h = current_1h['close']
        ema_1h = current_1h['ema_21']
        if pd.isna(close_1h) or pd.isna(ema_1h) or close_1h <= ema_1h:
            return None

        # === 2. CHECK 15MIN HISTOGRAM COLOR != RED ===
        # Red = momentum <= 0 AND falling (worst case for entry)
        momentum_15m = current_15m['ttm_momentum']
        prev_momentum_15m = prev_15m['ttm_momentum']
        color = get_histogram_color(momentum_15m, prev_momentum_15m)

        if color == 'red':
            return None

        # === 3. CHECK PRICE NEAR EMA LEVEL ===
        ema_21 = current_15m['ema_21']
        atr = current_15m[atr_col]

        # Calculate ATR levels
        entry_target = ema_21 + (self.params['entry_atr'] * atr)   # EMA itself (0.0 ATR)
        tolerance = self.params['entry_tolerance'] * atr            # 0.5 ATR tolerance

        stop_level = ema_21 + (self.params['stop_atr'] * atr)     # EMA - 2.0 ATR
        target_level = ema_21 + (self.params['target_atr'] * atr)  # EMA + 2.0 ATR

        # Check if bar's low is within tolerance of EMA level (pullback)
        bar_low = current_15m['low']

        # Distance from bar's low to entry target (in ATR units)
        distance_from_target = (bar_low - entry_target) / atr

        # Accept if within tolerance (within 0.5 ATR of EMA)
        if distance_from_target > tolerance:  # Didn't pull back enough
            return None

        # But not too far below (want pullback, not crash)
        if distance_from_target < -tolerance:  # Too far below
            return None

        # Entry at EMA level
        entry_price = entry_target
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
                'close_1h': close_1h,
                'ema_1h': ema_1h,
                'momentum_15m': momentum_15m,
                'histogram_color': color,
                'entry_level': entry_target,
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
        """
        # Validate required data
        self._validate_indicators_present(df, ['close', 'high', 'low'])

        current = df.iloc[-1]

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
