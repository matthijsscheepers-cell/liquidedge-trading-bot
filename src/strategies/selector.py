"""
Strategy Selector - Intelligent Strategy Routing

This module provides automatic strategy selection based on market regime.
It routes entry and exit logic to the appropriate strategy, ensuring the
correct strategy manages each position throughout its lifecycle.

Architecture:
    - Single interface for all strategy operations
    - Automatic routing based on regime and strategy recommendation
    - Maintains strategy context for position management
    - Centralized logging and statistics

Routing Logic:
    STRONG_TREND → RegimePullbackStrategy
    WEAK_TREND → RegimePullbackStrategy
    RANGE_COMPRESSION → TTMSqueezeStrategy
    HIGH_VOLATILITY → No trade
    NO_TRADE → No trade

Usage:
    from src.strategies.selector import StrategySelector

    # Initialize
    selector = StrategySelector(asset="US_TECH_100")

    # Check for entry (selector routes automatically)
    setup = selector.check_entry(
        df=df,
        regime="STRONG_TREND",
        confidence=85.0,
        strategy_recommendation="REGIME_PULLBACK"
    )

    # Manage exit (selector remembers which strategy opened the position)
    action, value = selector.manage_exit(df, position)
"""

from typing import Optional, Dict, Any
import pandas as pd
import logging
from .base import BaseStrategy, TradeSetup, Position, ExitAction
from .regime_pullback import RegimePullbackStrategy
from .ttm_squeeze import TTMSqueezeStrategy

# Configure logging
logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Intelligent strategy router based on market regime.

    This class acts as a unified interface to all trading strategies,
    automatically routing entry and exit logic to the appropriate strategy
    based on market regime and detector recommendations.

    The selector maintains strategy context for positions, ensuring that
    the same strategy that opened a position manages its exits.

    Attributes:
        asset: Trading instrument (e.g., "US_TECH_100")
        regime_strategy: RegimePullbackStrategy instance
        ttm_strategy: TTMSqueezeStrategy instance
        active_strategies: Mapping of position IDs to strategies

    Example:
        >>> selector = StrategySelector(asset="US_TECH_100")
        >>>
        >>> # Check for entry
        >>> setup = selector.check_entry(
        ...     df=df,
        ...     regime="STRONG_TREND",
        ...     confidence=85.0,
        ...     strategy_recommendation="REGIME_PULLBACK"
        ... )
        >>>
        >>> if setup:
        ...     print(f"Entry: {setup.entry_price}, Stop: {setup.stop_loss}")
        >>>
        >>> # Later, manage exit
        >>> action, value = selector.manage_exit(df, position)
    """

    def __init__(self, asset: str):
        """
        Initialize strategy selector with all available strategies.

        Args:
            asset: Trading instrument (e.g., "US_TECH_100", "GOLD", "EUR_USD")

        Example:
            >>> selector = StrategySelector(asset="US_TECH_100")
            >>> print(selector.asset)
            US_TECH_100
        """
        self.asset = asset

        # Initialize both strategies
        self.regime_strategy = RegimePullbackStrategy(asset)
        self.ttm_strategy = TTMSqueezeStrategy(asset)

        # Track which strategy opened each position
        # Key: position identifier (asset + entry_time combo)
        # Value: strategy instance that opened the position
        self.active_strategies: Dict[str, BaseStrategy] = {}

        logger.info(f"StrategySelector initialized for {asset}")
        logger.debug(f"Regime strategy params: {self.regime_strategy.params}")
        logger.debug(f"TTM strategy params: {self.ttm_strategy.params}")

    def check_entry(
        self,
        df: pd.DataFrame,
        regime: str,
        confidence: float,
        strategy_recommendation: str,
    ) -> Optional[TradeSetup]:
        """
        Route to appropriate strategy for entry signal.

        The selector uses the strategy recommendation from the RegimeDetector
        to determine which strategy should evaluate the current market for
        entry opportunities.

        Routing rules:
            - 'REGIME_PULLBACK' → RegimePullbackStrategy
            - 'TTM_SQUEEZE' → TTMSqueezeStrategy
            - 'TTM_BREAKOUT' → TTMSqueezeStrategy
            - 'NONE' → No trade

        Args:
            df: DataFrame with price data and indicators
                Must include columns required by the selected strategy
            regime: Current market regime (e.g., "STRONG_TREND", "RANGE_COMPRESSION")
            confidence: Regime confidence score (0-100)
            strategy_recommendation: Strategy name from RegimeDetector

        Returns:
            TradeSetup if valid entry found, None otherwise

        Example:
            >>> selector = StrategySelector(asset="US_TECH_100")
            >>> setup = selector.check_entry(
            ...     df=df,
            ...     regime="STRONG_TREND",
            ...     confidence=85.0,
            ...     strategy_recommendation="REGIME_PULLBACK"
            ... )
            >>> if setup:
            ...     print(f"Found {setup.setup_type} setup")
        """
        # Route based on strategy recommendation
        if strategy_recommendation == "REGIME_PULLBACK":
            logger.debug(
                f"Routing to RegimePullbackStrategy (regime={regime}, conf={confidence:.0f}%)"
            )
            return self.regime_strategy.check_entry(df, regime, confidence)

        elif strategy_recommendation in ["TTM_SQUEEZE", "TTM_BREAKOUT"]:
            logger.debug(
                f"Routing to TTMSqueezeStrategy (regime={regime}, conf={confidence:.0f}%)"
            )
            return self.ttm_strategy.check_entry(df, regime, confidence)

        else:
            logger.debug(f"No strategy for recommendation: {strategy_recommendation}")
            return None

    def manage_exit(
        self, df: pd.DataFrame, position: Position
    ) -> tuple[ExitAction, Optional[float]]:
        """
        Route to correct strategy for exit management.

        The selector uses the position's entry_strategy attribute to determine
        which strategy should manage the exit. This ensures consistent exit
        management throughout the position's lifecycle.

        Strategy detection:
            - If entry_strategy contains 'PULLBACK' → RegimePullbackStrategy
            - If entry_strategy contains 'SQUEEZE' or 'BREAKOUT' → TTMSqueezeStrategy
            - Otherwise → RegimePullbackStrategy (fallback)

        Args:
            df: DataFrame with current price data and indicators
                Required columns depend on the strategy managing the position
            position: Active position to manage
                Must have entry_strategy attribute set

        Returns:
            Tuple of (ExitAction, new_value):
                - action: ExitAction enum (HOLD, STOP, BREAKEVEN, TRAIL, TARGET, TIME_EXIT)
                - new_value: New stop price (for BREAKEVEN/TRAIL) or exit price (for STOP/TARGET)
                            None if action is HOLD

        Example:
            >>> action, value = selector.manage_exit(df, position)
            >>>
            >>> if action == ExitAction.BREAKEVEN:
            ...     position.stop_loss = value
            ...     print(f"Moved to breakeven at {value}")
            >>> elif action == ExitAction.STOP:
            ...     close_position(position, exit_price=value)
            ...     print(f"Stopped out at {value}")
        """
        # Determine which strategy opened this position
        entry_strategy = position.entry_strategy

        if "PULLBACK" in entry_strategy.upper():
            logger.debug(
                f"Routing exit to RegimePullbackStrategy for {position.asset}"
            )
            return self.regime_strategy.manage_exit(df, position)

        elif "SQUEEZE" in entry_strategy.upper() or "BREAKOUT" in entry_strategy.upper():
            logger.debug(f"Routing exit to TTMSqueezeStrategy for {position.asset}")
            return self.ttm_strategy.manage_exit(df, position)

        else:
            # Unknown strategy → use regime as fallback
            logger.warning(
                f"Unknown entry_strategy '{entry_strategy}', using RegimePullbackStrategy as fallback"
            )
            return self.regime_strategy.manage_exit(df, position)

    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get statistics and parameters for all strategies.

        Returns detailed information about both strategies including
        their parameters and configuration.

        Returns:
            Dict with strategy information:
                - asset: Trading instrument
                - regime_params: RegimePullbackStrategy parameters
                - ttm_params: TTMSqueezeStrategy parameters

        Example:
            >>> selector = StrategySelector(asset="US_TECH_100")
            >>> stats = selector.get_strategy_stats()
            >>> print(f"Asset: {stats['asset']}")
            >>> print(f"Regime stop: {stats['regime_params']['initial_stop_atr']} ATR")
            >>> print(f"TTM stop: {stats['ttm_params']['initial_stop_atr']} ATR")
        """
        return {
            "asset": self.asset,
            "regime_params": self.regime_strategy.params,
            "ttm_params": self.ttm_strategy.params,
        }

    def validate_setup(self, setup: TradeSetup) -> bool:
        """
        Validate a trade setup before execution.

        Performs sanity checks on the setup to ensure it's valid:
            - Stop loss is on correct side of entry
            - Target is on correct side of entry
            - Risk per share is positive
            - Confidence is within bounds
            - RRR meets minimum requirements

        Args:
            setup: TradeSetup to validate

        Returns:
            True if setup is valid, False otherwise

        Example:
            >>> setup = selector.check_entry(df, regime, confidence, strategy)
            >>> if setup and selector.validate_setup(setup):
            ...     # Execute trade
            ...     execute_trade(setup)
        """
        try:
            # All validation is already done in TradeSetup.__post_init__()
            # This method exists for potential additional validation
            # or to explicitly check before execution

            # Verify RRR meets minimum for the strategy
            rrr = setup.reward_risk_ratio()

            # Different minimums for different setup types
            if "PULLBACK" in setup.setup_type:
                min_rrr = self.regime_strategy.params["min_rrr"]
            elif "SQUEEZE" in setup.setup_type:
                min_rrr = self.ttm_strategy.params["min_rrr"]
            else:
                min_rrr = 1.5  # Default minimum

            if rrr < min_rrr:
                logger.warning(
                    f"Setup RRR {rrr:.2f} below minimum {min_rrr:.2f} for {setup.setup_type}"
                )
                return False

            logger.debug(f"Setup validated: {setup.setup_type} with {rrr:.2f}R")
            return True

        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            return False

    def get_strategy_for_regime(self, regime: str) -> Optional[BaseStrategy]:
        """
        Get the appropriate strategy instance for a given regime.

        Useful for accessing strategy-specific methods or parameters
        based on the current market regime.

        Args:
            regime: Market regime name

        Returns:
            Strategy instance or None if no strategy for regime

        Example:
            >>> strategy = selector.get_strategy_for_regime("STRONG_TREND")
            >>> if strategy:
            ...     print(f"Max hold time: {strategy.params['max_bars']} bars")
        """
        if regime in ["STRONG_TREND", "WEAK_TREND"]:
            return self.regime_strategy
        elif regime == "RANGE_COMPRESSION":
            return self.ttm_strategy
        else:
            return None

    def __repr__(self) -> str:
        """String representation of the selector."""
        return f"StrategySelector(asset={self.asset})"


# Export
__all__ = ["StrategySelector"]
