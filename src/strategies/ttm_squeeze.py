"""
TTM Squeeze Breakout Strategy

This strategy trades volatility compression releases (~30% of trades).

Strategy logic:
    - Trades squeeze releases in RANGE_COMPRESSION regime
    - Two entry types: immediate breakout or pullback retest
    - Uses momentum confirmation for entries
    - Tighter risk management than trend-following (faster exits)
    - Exits on momentum reversal

Entry conditions:
    SQUEEZE_RELEASE (Immediate Breakout):
        - Squeeze just turned OFF (prev bar ON, current OFF)
        - Momentum increasing in breakout direction
        - Enter at market close

    SQUEEZE_RETEST (Pullback Entry):
        - Squeeze released 1-3 bars ago
        - Price pulled back to Keltner Channel basis
        - Rejection candle confirms entry

Exit management:
    - Tighter stops (1.5 ATR vs 2.0 ATR for pullback)
    - Lower targets (1.8R vs 2.5R)
    - Earlier breakeven (1.0R vs 1.5R)
    - Faster trailing (1.5R vs 2.5R)
    - Shorter max hold time (48 bars vs 120)
    - Exit on momentum reversal

WHY different from pullback strategy:
    Breakouts are explosive but short-lived. We need:
    - Faster exits to capture the initial move
    - Tighter stops because breakouts either work quickly or fail
    - Momentum reversal exits because expansion phase ends abruptly
    - Shorter hold times because energy dissipates faster

Usage:
    from src.strategies.ttm_squeeze import TTMSqueezeStrategy

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")
    df = detector.add_all_indicators(df)

    setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)
    if setup:
        # Execute breakout trade
        position = create_position(setup)

    # Later, manage exit
    action, value = strategy.manage_exit(df, position)
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import (
    BaseStrategy,
    TradeSetup,
    Position,
    SignalDirection,
    ExitAction,
)


class TTMSqueezeStrategy(BaseStrategy):
    """
    Volatility compression breakout strategy using TTM Squeeze.

    This strategy capitalizes on the explosive moves that follow periods
    of low volatility (squeezes). It enters on confirmed breakouts or
    pullback retests after the squeeze releases.

    Key characteristics:
        - Faster entries and exits than trend following
        - Tighter risk management (stops, targets, timeframes)
        - Momentum-based confirmations and exits
        - Best in RANGE_COMPRESSION regime

    Asset-specific parameters are tuned for breakout characteristics:
        - Tighter stops for quick failure detection
        - Lower targets for rapid expansion capture
        - Faster management (breakeven, trail, time exits)

    Example:
        >>> strategy = TTMSqueezeStrategy(asset="US_TECH_100")
        >>> setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)
        >>> if setup:
        ...     print(f"Breakout at {setup.entry_price}, Stop: {setup.stop_loss}")
    """

    def _get_asset_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters for squeeze breakout strategy.

        Breakout parameters are tighter than pullback strategy because:
        - Breakouts either work fast or fail fast
        - Energy from compression dissipates quickly
        - Momentum reversals happen abruptly

        Parameters:
            initial_stop_atr: Initial stop distance in ATR (tighter: 1.5 vs 2.0)
            min_rrr: Minimum reward-to-risk ratio (lower: 1.8 vs 2.5)
            breakeven_r: R level to move stop to breakeven (faster: 1.0 vs 1.5)
            trail_start_r: R level to start trailing stop (faster: 1.5 vs 2.5)
            trail_distance_atr: Trail distance in ATR multiples
            max_bars: Maximum bars to hold position (shorter: 48 vs 120)
            retest_tolerance: How close to KC basis for retest (ATR multiples)
            min_momentum: Minimum momentum for entry confirmation

        Args:
            asset: Asset symbol (e.g., "US_TECH_100", "GOLD", "EUR_USD")

        Returns:
            Dict with asset-specific parameters

        Example:
            >>> strategy = TTMSqueezeStrategy(asset="US_TECH_100")
            >>> print(strategy.params['initial_stop_atr'])  # 1.5 (tighter than pullback)
        """
        # US Indices - Fast, explosive breakouts
        if asset in ["US_TECH_100", "US_30", "US_SPX_500"]:
            return {
                'initial_stop_atr': 1.5,  # Tighter stop
                'min_rrr': 1.8,           # Lower target
                'breakeven_r': 1.0,       # Earlier breakeven
                'trail_start_r': 1.5,     # Earlier trail
                'trail_distance_atr': 1.2,
                'max_bars': 48,           # Shorter hold time
                'retest_tolerance': 0.3,
                'min_momentum': 0.2,
            }

        # Gold - Medium parameters
        elif asset == "GOLD":
            return {
                'initial_stop_atr': 1.8,
                'min_rrr': 1.6,
                'breakeven_r': 1.0,
                'trail_start_r': 1.5,
                'trail_distance_atr': 1.3,
                'max_bars': 60,
                'retest_tolerance': 0.4,
                'min_momentum': 0.15,
            }

        # Forex - varies by pair
        elif asset in ["EUR_USD", "GBP_USD"]:
            return {
                'initial_stop_atr': 1.5,
                'min_rrr': 1.8,
                'breakeven_r': 1.0,
                'trail_start_r': 1.5,
                'trail_distance_atr': 1.2,
                'max_bars': 48,
                'retest_tolerance': 0.3,
                'min_momentum': 0.2,
            }

        # Default parameters
        else:
            return {
                'initial_stop_atr': 1.8,
                'min_rrr': 1.6,
                'breakeven_r': 1.0,
                'trail_start_r': 1.5,
                'trail_distance_atr': 1.3,
                'max_bars': 48,
                'retest_tolerance': 0.4,
                'min_momentum': 0.15,
            }

    def check_entry(self,
                   df: pd.DataFrame,
                   regime: str,
                   confidence: float) -> Optional[TradeSetup]:
        """
        Check for squeeze breakout entry setup.

        Looks for two types of entries:
        1. Immediate breakout when squeeze releases
        2. Retest of Keltner basis after squeeze release

        Entry logic:
            1. Validate regime is RANGE_COMPRESSION
            2. Check for squeeze release (SETUP 1) OR retest (SETUP 2)
            3. Confirm with momentum
            4. Calculate entry, stop, and target
            5. Verify RRR meets minimum

        Args:
            df: DataFrame with price data and indicators
                Required columns: open, high, low, close, squeeze_on, ttm_momentum,
                                kc_middle, atr_14
            regime: Current market regime (from RegimeDetector)
            confidence: Regime confidence score (0-100)

        Returns:
            TradeSetup if valid setup found, None otherwise

        Example:
            >>> strategy = TTMSqueezeStrategy(asset="US_TECH_100")
            >>> setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)
            >>> if setup:
            ...     print(f"Found {setup.setup_type} at {setup.entry_price}")
        """
        # Validate required indicators
        self._validate_indicators_present(
            df,
            ['open', 'high', 'low', 'close', 'squeeze_on', 'ttm_momentum',
             'kc_middle', 'atr_14']
        )

        # REGIME FILTER DISABLED - Accept TTM Squeeze trades in any regime
        # Previously: Only trade in compression/squeeze regime
        # if regime != 'RANGE_COMPRESSION':
        #     return None

        # Need at least 4 bars for pattern detection
        if len(df) < 4:
            return None

        # Get current and previous bars
        current = df.iloc[-1]
        prev = df.iloc[-2]

        atr = current['atr_14']
        kc_basis = current['kc_middle']

        # === SETUP 1: SQUEEZE_RELEASE (Immediate Breakout) ===
        setup = self._check_squeeze_release(df, current, prev, atr, confidence)
        if setup:
            return setup

        # === SETUP 2: SQUEEZE_RETEST (Pullback Entry) ===
        setup = self._check_squeeze_retest(df, current, atr, kc_basis, confidence)
        if setup:
            return setup

        return None

    def _check_squeeze_release(self,
                               df: pd.DataFrame,
                               current: pd.Series,
                               prev: pd.Series,
                               atr: float,
                               confidence: float) -> Optional[TradeSetup]:
        """
        Check for immediate squeeze release breakout.

        Squeeze just turned off → explosive move likely starting.

        Args:
            df: Full DataFrame
            current: Current bar
            prev: Previous bar
            atr: Current ATR
            confidence: Regime confidence

        Returns:
            TradeSetup if valid release found, None otherwise
        """
        # Check if squeeze just released (was ON, now OFF)
        prev_squeeze = prev['squeeze_on']
        curr_squeeze = current['squeeze_on']

        # Handle NaN values
        if pd.isna(prev_squeeze) or pd.isna(curr_squeeze):
            return None

        squeeze_released = prev_squeeze and not curr_squeeze

        if not squeeze_released:
            return None

        # Get momentum values
        prev_momentum = prev['ttm_momentum']
        curr_momentum = current['ttm_momentum']

        # Momentum should be increasing (breakout accelerating)
        momentum_increasing = abs(curr_momentum) > abs(prev_momentum)

        # Momentum should be above minimum threshold
        momentum_strong = abs(curr_momentum) >= self.params['min_momentum']

        if not (momentum_increasing or momentum_strong):
            return None

        # Determine direction from momentum
        # ONLY LONG TRADES - SHORT trades have 0% win rate
        if curr_momentum > 0:
            # Bullish breakout
            direction = SignalDirection.LONG
            entry = current['close']
            stop = entry - (self.params['initial_stop_atr'] * atr)
            risk = entry - stop
            target = entry + (risk * self.params['min_rrr'])
            setup_type = "SQUEEZE_RELEASE_LONG"

        else:
            # Skip SHORT trades (0% win rate in backtests)
            return None

        return TradeSetup(
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            risk_per_share=risk,
            confidence=confidence,
            setup_type=setup_type,
            metadata={
                'kc_middle': current['kc_middle'],
                'atr_14': atr,
                'ttm_momentum': curr_momentum,
                'regime': 'RANGE_COMPRESSION',
            }
        )

    def _check_squeeze_retest(self,
                              df: pd.DataFrame,
                              current: pd.Series,
                              atr: float,
                              kc_basis: float,
                              confidence: float) -> Optional[TradeSetup]:
        """
        Check for pullback retest after squeeze release.

        Squeeze released 1-3 bars ago, price pulls back to KC basis,
        then confirms with rejection candle.

        Args:
            df: Full DataFrame
            current: Current bar
            atr: Current ATR
            kc_basis: Keltner Channel middle line
            confidence: Regime confidence

        Returns:
            TradeSetup if valid retest found, None otherwise
        """
        # Check last 3 bars for squeeze release
        lookback = min(3, len(df) - 1)
        recent = df.iloc[-lookback-1:]

        # Was squeeze released in last 1-3 bars?
        squeeze_released_recently = False
        release_bar_idx = -1

        for i in range(len(recent) - 1):
            if recent['squeeze_on'].iloc[i] and not recent['squeeze_on'].iloc[i + 1]:
                squeeze_released_recently = True
                release_bar_idx = i
                break

        if not squeeze_released_recently:
            return None

        # Check if price is near KC basis (pullback)
        distance_to_basis = abs(current['close'] - kc_basis) / atr
        if distance_to_basis > self.params['retest_tolerance']:
            return None

        # Get momentum direction to determine trade direction
        curr_momentum = current['ttm_momentum']

        # Determine direction
        # ONLY LONG TRADES - SHORT retests have 0% win rate
        if curr_momentum > 0:
            # Bullish retest
            direction = SignalDirection.LONG

            # Check for bullish confirmation (rejection of lows)
            if not self._is_bullish_rejection(current):
                return None

            entry = current['close']
            stop = entry - (self.params['initial_stop_atr'] * atr)
            risk = entry - stop
            target = entry + (risk * self.params['min_rrr'])
            setup_type = "SQUEEZE_RETEST_LONG"

        else:
            # Skip SHORT retest trades (0% win rate in backtests)
            return None

        return TradeSetup(
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            target=target,
            risk_per_share=risk,
            confidence=confidence,
            setup_type=setup_type,
            metadata={
                'kc_middle': kc_basis,
                'atr_14': atr,
                'ttm_momentum': curr_momentum,
                'regime': 'RANGE_COMPRESSION',
            }
        )

    def manage_exit(self,
                   df: pd.DataFrame,
                   position: Position) -> tuple[ExitAction, Optional[float]]:
        """
        Manage exit for squeeze breakout position.

        Breakout exits are FASTER than pullback exits because:
        - Breakouts are short-lived explosive moves
        - Energy dissipates quickly after initial expansion
        - Momentum reversals signal end of breakout phase
        - Tighter management captures the move before it fades

        Exit logic:
            1. Check for stop loss hit
            2. Check for target hit
            3. Check for momentum reversal (unique to breakouts)
            4. Move to breakeven at 1.0R (faster than pullback's 1.5R)
            5. Trail stop at 1.5R (faster than pullback's 2.5R)
            6. Exit if max bars exceeded (48 vs pullback's 120)

        Args:
            df: DataFrame with current price data and indicators
                Required columns: close, atr_14, ttm_momentum
            position: Active position to manage

        Returns:
            Tuple of (ExitAction, new_value):
                - ExitAction.HOLD: No change
                - ExitAction.STOP: Stop loss hit
                - ExitAction.TARGET: Target hit
                - ExitAction.BREAKEVEN: Move stop to breakeven
                - ExitAction.TRAIL: Trail stop
                - ExitAction.TIME_EXIT: Max bars exceeded

        Example:
            >>> action, value = strategy.manage_exit(df, position)
            >>> if action == ExitAction.BREAKEVEN:
            ...     print(f"Moving to breakeven at {value}")
        """
        # Validate required indicators
        self._validate_indicators_present(df, ['close', 'atr_14', 'ttm_momentum'])

        current = df.iloc[-1]
        current_price = current['close']
        atr = current['atr_14']

        # Calculate current R
        r = self.calculate_r_multiple(current_price, position)

        # Update max R
        if r > position.max_r:
            position.max_r = r

        # LONG position management
        if position.direction == SignalDirection.LONG:
            # Check stop loss
            if current_price <= position.stop_loss:
                return ExitAction.STOP, position.stop_loss

            # Check target
            if current_price >= position.target:
                return ExitAction.TARGET, position.target

            # Check momentum reversal (unique to breakout strategy)
            # If in profit and momentum reverses → exit
            if r > 0.5:  # In profit
                entry_momentum = position.metadata.get('ttm_momentum', 0) if position.metadata else 0
                curr_momentum = current['ttm_momentum']

                # Momentum reversal: was positive, now negative
                if entry_momentum > 0 and curr_momentum < 0:
                    return ExitAction.TARGET, current_price  # Exit at market

            # Move to breakeven at 1.0R (faster than pullback)
            if r >= self.params['breakeven_r'] and position.stop_loss < position.entry_price:
                return ExitAction.BREAKEVEN, position.entry_price

            # Trail stop at 1.5R (faster than pullback)
            if r >= self.params['trail_start_r']:
                new_stop = current_price - (self.params['trail_distance_atr'] * atr)
                if new_stop > position.stop_loss:
                    return ExitAction.TRAIL, new_stop

        # SHORT position management
        elif position.direction == SignalDirection.SHORT:
            # Check stop loss
            if current_price >= position.stop_loss:
                return ExitAction.STOP, position.stop_loss

            # Check target
            if current_price <= position.target:
                return ExitAction.TARGET, position.target

            # Check momentum reversal
            if r > 0.5:  # In profit
                entry_momentum = position.metadata.get('ttm_momentum', 0) if position.metadata else 0
                curr_momentum = current['ttm_momentum']

                # Momentum reversal: was negative, now positive
                if entry_momentum < 0 and curr_momentum > 0:
                    return ExitAction.TARGET, current_price  # Exit at market

            # Move to breakeven at 1.0R
            if r >= self.params['breakeven_r'] and position.stop_loss > position.entry_price:
                return ExitAction.BREAKEVEN, position.entry_price

            # Trail stop at 1.5R
            if r >= self.params['trail_start_r']:
                new_stop = current_price + (self.params['trail_distance_atr'] * atr)
                if new_stop < position.stop_loss:
                    return ExitAction.TRAIL, new_stop

        # Check time-based exit (shorter than pullback strategy)
        current_bar = len(df) - 1
        bars_in_trade = current_bar - position.entry_bar
        if bars_in_trade >= self.params['max_bars']:
            return ExitAction.TIME_EXIT, current_price

        return ExitAction.HOLD, None

    def _is_bullish_rejection(self, current: pd.Series) -> bool:
        """
        Check for bullish rejection of lows (hammer-like candle).

        Args:
            current: Current bar data

        Returns:
            True if bullish rejection found
        """
        candle_range = current['high'] - current['low']
        if candle_range == 0:
            return False

        # Calculate lower wick
        if current['close'] > current['open']:
            # Bullish candle
            lower_wick = current['open'] - current['low']
        else:
            # Bearish candle
            lower_wick = current['close'] - current['low']

        # Lower wick should be > 50% of total range (rejection)
        wick_ratio = lower_wick / candle_range
        return wick_ratio > 0.5

    def _is_bearish_rejection(self, current: pd.Series) -> bool:
        """
        Check for bearish rejection of highs (shooting star-like candle).

        Args:
            current: Current bar data

        Returns:
            True if bearish rejection found
        """
        candle_range = current['high'] - current['low']
        if candle_range == 0:
            return False

        # Calculate upper wick
        if current['close'] < current['open']:
            # Bearish candle
            upper_wick = current['high'] - current['open']
        else:
            # Bullish candle
            upper_wick = current['high'] - current['close']

        # Upper wick should be > 50% of total range (rejection)
        wick_ratio = upper_wick / candle_range
        return wick_ratio > 0.5


# Export
__all__ = ['TTMSqueezeStrategy']
