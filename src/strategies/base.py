"""
Base Strategy Classes and Interfaces

This module defines the abstract base class and data structures for all trading
strategies in the LIQUIDEDGE system.

Architecture:
    - SignalDirection: Trade direction enum (LONG/SHORT/NONE)
    - ExitAction: Exit action types (HOLD/STOP/BREAKEVEN/TRAIL/TARGET/TIME_EXIT)
    - TradeSetup: Entry setup with validation
    - Position: Active position tracking
    - BaseStrategy: Abstract base class all strategies inherit from

All strategies must implement:
    - check_entry(): Scan for entry opportunities
    - manage_exit(): Manage open positions

Usage:
    from src.strategies.base import BaseStrategy, TradeSetup, SignalDirection

    class MyStrategy(BaseStrategy):
        def _get_asset_params(self, asset):
            return {'initial_stop_atr': 2.0, 'min_rrr': 2.5}

        def check_entry(self, df, regime, confidence):
            # Entry logic here
            if valid_setup:
                return TradeSetup(...)
            return None

        def manage_exit(self, df, position):
            # Exit logic here
            return ExitAction.HOLD, None
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from enum import Enum


class SignalDirection(Enum):
    """
    Trade direction.

    Values:
        LONG: Buy/long position
        SHORT: Sell/short position
        NONE: No trade signal
    """
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class ExitAction(Enum):
    """
    Exit action types for position management.

    Values:
        HOLD: Keep position as-is
        STOP: Exit at stop loss
        BREAKEVEN: Move stop to breakeven
        TRAIL: Trail stop loss
        TARGET: Exit at target
        TIME_EXIT: Exit due to time limit
    """
    HOLD = "HOLD"
    STOP = "STOP"
    BREAKEVEN = "BREAKEVEN"
    TRAIL = "TRAIL"
    TARGET = "TARGET"
    TIME_EXIT = "TIME_EXIT"


@dataclass
class TradeSetup:
    """
    Trade setup returned by check_entry().

    Represents a valid entry opportunity with all necessary parameters
    for position sizing and risk management.

    Attributes:
        direction: LONG or SHORT
        entry_price: Suggested entry price
        stop_loss: Initial stop loss price
        target: Take profit target price
        risk_per_share: Risk per unit (entry - stop for LONG)
        confidence: Setup confidence score (0-100)
        setup_type: Name of setup (e.g., "PULLBACK_LONG", "SQUEEZE_BREAKOUT")
        metadata: Additional info (optional dict)

    Validation:
        - LONG: entry > stop, target > entry
        - SHORT: entry < stop, target < entry
        - Confidence: 0-100
        - Risk: positive

    Example:
        >>> setup = TradeSetup(
        ...     direction=SignalDirection.LONG,
        ...     entry_price=100.0,
        ...     stop_loss=98.0,
        ...     target=106.0,
        ...     risk_per_share=2.0,
        ...     confidence=85.0,
        ...     setup_type="PULLBACK_LONG",
        ...     metadata={'ema_slope': 0.5}
        ... )
    """
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    target: float
    risk_per_share: float
    confidence: float
    setup_type: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Validate setup parameters after initialization."""
        if self.direction == SignalDirection.LONG:
            assert self.entry_price > self.stop_loss, \
                f"LONG: entry ({self.entry_price}) must be > stop ({self.stop_loss})"
            assert self.target > self.entry_price, \
                f"LONG: target ({self.target}) must be > entry ({self.entry_price})"
        elif self.direction == SignalDirection.SHORT:
            assert self.entry_price < self.stop_loss, \
                f"SHORT: entry ({self.entry_price}) must be < stop ({self.stop_loss})"
            assert self.target < self.entry_price, \
                f"SHORT: target ({self.target}) must be < entry ({self.entry_price})"

        assert 0 <= self.confidence <= 100, \
            f"Confidence must be 0-100, got {self.confidence}"
        assert self.risk_per_share > 0, \
            f"Risk must be positive, got {self.risk_per_share}"

    def reward_risk_ratio(self) -> float:
        """
        Calculate reward-to-risk ratio.

        Returns:
            R:R ratio (e.g., 3.0 means 3x reward for 1x risk)
        """
        if self.direction == SignalDirection.LONG:
            reward = self.target - self.entry_price
        else:
            reward = self.entry_price - self.target

        return reward / self.risk_per_share


@dataclass
class Position:
    """
    Active position tracking.

    Maintains all information needed to manage an open position including
    entry details, current stops, and performance metrics.

    Attributes:
        asset: Instrument (e.g., "US_TECH_100", "EUR_USD")
        direction: LONG or SHORT
        entry_price: Actual entry price
        stop_loss: Current stop loss price
        target: Take profit target price
        units: Position size (units/contracts)
        risk_per_share: Risk per unit (distance from entry to initial stop)
        entry_time: Entry timestamp
        entry_bar: Entry bar index in DataFrame
        max_r: Maximum R multiple achieved (for tracking)
        entry_strategy: Strategy name that opened this position
        metadata: Additional tracking info (optional dict)

    Example:
        >>> position = Position(
        ...     asset="US_TECH_100",
        ...     direction=SignalDirection.LONG,
        ...     entry_price=100.0,
        ...     stop_loss=98.0,
        ...     target=106.0,
        ...     units=10.0,
        ...     risk_per_share=2.0,
        ...     entry_time=pd.Timestamp.now(),
        ...     entry_bar=150,
        ...     entry_strategy="RegimePullback"
        ... )
    """
    asset: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    target: float
    units: float
    risk_per_share: float
    entry_time: pd.Timestamp
    entry_bar: int
    max_r: float = 0.0
    entry_strategy: str = ""
    metadata: Dict[str, Any] = None

    def current_pnl(self, current_price: float) -> float:
        """
        Calculate current P&L in currency units.

        Args:
            current_price: Current market price

        Returns:
            P&L (positive = profit, negative = loss)
        """
        if self.direction == SignalDirection.LONG:
            pnl_per_unit = current_price - self.entry_price
        else:
            pnl_per_unit = self.entry_price - current_price

        return pnl_per_unit * self.units

    def current_r(self, current_price: float) -> float:
        """
        Calculate current R multiple.

        Args:
            current_price: Current market price

        Returns:
            R multiple (e.g., 2.5 = up 2.5x initial risk)
        """
        if self.direction == SignalDirection.LONG:
            pnl_per_unit = current_price - self.entry_price
        else:
            pnl_per_unit = self.entry_price - current_price

        return pnl_per_unit / self.risk_per_share


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies inherit from this class and must implement:
        - _get_asset_params(): Asset-specific parameters
        - check_entry(): Entry logic
        - manage_exit(): Exit logic

    The strategy system is designed around:
        1. Asset-specific parameters (each asset has different characteristics)
        2. Entry scanning (find valid setups)
        3. Position management (stops, targets, trails)

    Subclasses should:
        - Define asset-specific parameters in _get_asset_params()
        - Implement entry rules in check_entry()
        - Implement exit rules in manage_exit()
        - Add strategy-specific helper methods as needed

    Example:
        >>> class MyStrategy(BaseStrategy):
        ...     def _get_asset_params(self, asset):
        ...         return {
        ...             'initial_stop_atr': 2.0,
        ...             'min_rrr': 2.5,
        ...             'breakeven_r': 1.5
        ...         }
        ...
        ...     def check_entry(self, df, regime, confidence):
        ...         # Entry logic
        ...         if valid_setup:
        ...             return TradeSetup(...)
        ...         return None
        ...
        ...     def manage_exit(self, df, position):
        ...         # Exit logic
        ...         return ExitAction.HOLD, None
    """

    def __init__(self, asset: str):
        """
        Initialize strategy.

        Args:
            asset: Trading instrument (e.g., "US_TECH_100", "EUR_USD")
        """
        self.asset = asset
        self.params = self._get_asset_params(asset)

    @abstractmethod
    def _get_asset_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters.

        Different assets need different parameters:
        - Indices (NAS100, US30): Tighter stops, bigger targets
        - Commodities (Gold, Oil): Wider stops, smaller targets
        - Forex (EUR/USD, GBP/USD): Varies by pair volatility

        Common parameters:
            - initial_stop_atr: Initial stop distance in ATR multiples
            - min_rrr: Minimum reward-to-risk ratio
            - breakeven_r: R level to move stop to breakeven
            - trail_start_r: R level to start trailing stop
            - trail_distance_atr: Trail distance in ATR multiples
            - max_bars: Maximum bars to hold position

        Args:
            asset: Asset symbol

        Returns:
            Dict with asset-specific parameters

        Example:
            >>> def _get_asset_params(self, asset):
            ...     if asset == "US_TECH_100":
            ...         return {
            ...             'initial_stop_atr': 2.0,
            ...             'min_rrr': 2.5,
            ...             'breakeven_r': 1.5,
            ...             'trail_start_r': 2.5,
            ...             'trail_distance_atr': 1.5,
            ...             'max_bars': 20
            ...         }
        """
        pass

    @abstractmethod
    def check_entry(self,
                   df: pd.DataFrame,
                   regime: str,
                   confidence: float) -> Optional[TradeSetup]:
        """
        Check for entry setup.

        This method scans the market for valid entry opportunities based on
        the current regime and market conditions.

        Args:
            df: DataFrame with price data and indicators
                Must include: open, high, low, close, and strategy-specific indicators
            regime: Current market regime (from RegimeDetector)
            confidence: Regime confidence score (0-100)

        Returns:
            TradeSetup if valid setup found, None otherwise

        Implementation guidelines:
            - Validate required indicators are present
            - Check if regime matches strategy requirements
            - Look for entry trigger (e.g., pullback, breakout)
            - Calculate entry, stop, and target prices
            - Verify reward-to-risk ratio meets minimum
            - Return TradeSetup with all parameters

        Example:
            >>> def check_entry(self, df, regime, confidence):
            ...     # Validate indicators
            ...     self._validate_indicators_present(df, ['ema_20', 'atr_14'])
            ...
            ...     # Check regime
            ...     if regime != 'STRONG_TREND' or confidence < 70:
            ...         return None
            ...
            ...     # Check for pullback entry
            ...     current = df.iloc[-1]
            ...     if current['close'] > current['ema_20']:
            ...         entry = current['close']
            ...         stop = entry - (2.0 * current['atr_14'])
            ...         target = entry + (5.0 * current['atr_14'])
            ...
            ...         return TradeSetup(
            ...             direction=SignalDirection.LONG,
            ...             entry_price=entry,
            ...             stop_loss=stop,
            ...             target=target,
            ...             risk_per_share=entry - stop,
            ...             confidence=confidence,
            ...             setup_type="PULLBACK_LONG"
            ...         )
            ...     return None
        """
        pass

    @abstractmethod
    def manage_exit(self,
                   df: pd.DataFrame,
                   position: Position) -> tuple[ExitAction, Optional[float]]:
        """
        Manage exit for open position.

        This method handles all exit logic including stop loss, profit targets,
        breakeven moves, and trailing stops.

        Args:
            df: DataFrame with current price data and indicators
            position: Active position to manage

        Returns:
            Tuple of (action, new_value):
                - action: ExitAction enum (HOLD, STOP, BREAKEVEN, TRAIL, TARGET, TIME_EXIT)
                - new_value: New stop price (for BREAKEVEN/TRAIL) or exit price (for STOP/TARGET)
                            None if action is HOLD

        Implementation guidelines:
            - Calculate current R multiple
            - Check for stop loss hit
            - Check for target hit
            - Move to breakeven when appropriate
            - Trail stop when in profit
            - Implement time-based exits if needed

        Example:
            >>> def manage_exit(self, df, position):
            ...     current = df.iloc[-1]
            ...     current_price = current['close']
            ...     r = self.calculate_r_multiple(current_price, position)
            ...
            ...     # Check stop loss
            ...     if position.direction == SignalDirection.LONG:
            ...         if current_price <= position.stop_loss:
            ...             return ExitAction.STOP, position.stop_loss
            ...         if current_price >= position.target:
            ...             return ExitAction.TARGET, position.target
            ...
            ...     # Move to breakeven at 1.5R
            ...     if r >= 1.5 and position.stop_loss != position.entry_price:
            ...         return ExitAction.BREAKEVEN, position.entry_price
            ...
            ...     # Trail at 2.5R
            ...     if r >= 2.5:
            ...         atr = current['atr_14']
            ...         if position.direction == SignalDirection.LONG:
            ...             new_stop = current_price - (1.5 * atr)
            ...             if new_stop > position.stop_loss:
            ...                 return ExitAction.TRAIL, new_stop
            ...
            ...     return ExitAction.HOLD, None
        """
        pass

    def calculate_r_multiple(self,
                            current_price: float,
                            position: Position) -> float:
        """
        Calculate current R multiple for position.

        R multiple = (current P&L per unit) / initial risk per unit

        This is the standard way to measure position performance:
            - R = 1.0: At 1x initial risk (breakeven region)
            - R = 2.0: Up 2x initial risk (2R profit)
            - R = -1.0: Down 1x initial risk (full loss)

        Args:
            current_price: Current market price
            position: Position to calculate for

        Returns:
            R multiple (e.g., 2.5 means up 2.5x initial risk)

        Example:
            >>> # Position: LONG at 100, stop at 98 (2.0 risk)
            >>> # Current price: 106
            >>> r = self.calculate_r_multiple(106, position)
            >>> print(f"Position at {r}R")  # "Position at 3.0R"
        """
        if position.direction == SignalDirection.LONG:
            pnl = current_price - position.entry_price
        else:
            pnl = position.entry_price - current_price

        return pnl / position.risk_per_share

    def _validate_indicators_present(self,
                                    df: pd.DataFrame,
                                    required: list[str]) -> bool:
        """
        Check if required indicators are present in DataFrame.

        Args:
            df: DataFrame to check
            required: List of required column names

        Returns:
            True if all present

        Raises:
            ValueError: If any indicators are missing

        Example:
            >>> self._validate_indicators_present(df, ['ema_20', 'atr_14', 'adx_14'])
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required indicators: {missing}. "
                f"Ensure DataFrame has been processed with add_all_indicators()"
            )
        return True

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(asset={self.asset})"


# Export main components
__all__ = [
    'SignalDirection',
    'ExitAction',
    'TradeSetup',
    'Position',
    'BaseStrategy',
]
