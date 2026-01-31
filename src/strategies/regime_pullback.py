"""
Regime Pullback Strategy - Trend Following Strategy

This is the MAIN strategy for trending markets, expected to handle ~70% of trades.

Strategy logic:
    - Trades pullbacks in strong trends (STRONG_TREND or WEAK_TREND regimes)
    - Waits for price to pull back to 20 EMA
    - Requires confirmation candle (engulfing or rejection wick)
    - Uses ATR-based stops and targets
    - Implements breakeven and trailing stops

Entry conditions:
    LONG:
        - Regime: STRONG_TREND or WEAK_TREND
        - Price pullback to 20 EMA (close near EMA)
        - Bullish confirmation: engulfing or rejection wick
        - ADX > 20 (trending)

    SHORT:
        - Regime: STRONG_TREND or WEAK_TREND
        - Price pullback to 20 EMA (close near EMA)
        - Bearish confirmation: engulfing or rejection wick
        - ADX > 20 (trending)

Exit management:
    - Move to breakeven at 1.5R
    - Trail stop at 2.5R (1.5 ATR trail distance)
    - Maximum hold time: 20 bars

Usage:
    from src.strategies.regime_pullback import RegimePullbackStrategy

    strategy = RegimePullbackStrategy(asset="US_TECH_100")
    df = detector.add_all_indicators(df)

    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
    if setup:
        # Execute trade
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


class RegimePullbackStrategy(BaseStrategy):
    """
    Trend following strategy that trades pullbacks to 20 EMA.

    This is the primary strategy for trending market regimes (STRONG_TREND, WEAK_TREND).

    The strategy waits for price to pull back to the 20 EMA in a trending market,
    then enters on confirmation candles (engulfing patterns or rejection wicks).

    Asset-specific parameters are tuned for different instrument types:
        - US Indices (NAS100, US30): Tighter stops, bigger targets
        - Gold: Medium stops and targets
        - Forex: Varies by pair volatility

    Example:
        >>> strategy = RegimePullbackStrategy(asset="US_TECH_100")
        >>> setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
        >>> if setup:
        ...     print(f"Entry at {setup.entry_price}, Stop: {setup.stop_loss}, Target: {setup.target}")
    """

    def _get_asset_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters for pullback strategy.

        Different assets have different volatility and behavior patterns,
        requiring customized parameters.

        Parameters:
            initial_stop_atr: Initial stop distance in ATR multiples
            min_rrr: Minimum reward-to-risk ratio
            breakeven_r: R level to move stop to breakeven
            trail_start_r: R level to start trailing stop
            trail_distance_atr: Trail distance in ATR multiples
            max_bars: Maximum bars to hold position
            pullback_tolerance: How close to EMA for pullback (ATR multiples)
            min_adx: Minimum ADX for trend confirmation

        Args:
            asset: Asset symbol (e.g., "US_TECH_100", "GOLD", "EUR_USD")

        Returns:
            Dict with asset-specific parameters

        Example:
            >>> strategy = RegimePullbackStrategy(asset="US_TECH_100")
            >>> print(strategy.params['initial_stop_atr'])  # 2.0
        """
        # US Indices (NAS100, US30, etc.) - Tighter stops, bigger targets
        if asset in ["US_TECH_100", "US_30", "US_SPX_500"]:
            return {
                'initial_stop_atr': 2.0,
                'min_rrr': 2.5,
                'breakeven_r': 1.5,
                'trail_start_r': 2.5,
                'trail_distance_atr': 1.5,
                'max_bars': 20,
                'pullback_tolerance': 0.5,
                'min_adx': 20,
            }

        # Gold - Medium parameters
        elif asset == "GOLD":
            return {
                'initial_stop_atr': 2.5,
                'min_rrr': 2.0,
                'breakeven_r': 1.5,
                'trail_start_r': 2.5,
                'trail_distance_atr': 1.5,
                'max_bars': 25,
                'pullback_tolerance': 0.6,
                'min_adx': 20,
            }

        # Forex - varies by pair
        elif asset in ["EUR_USD", "GBP_USD"]:
            return {
                'initial_stop_atr': 2.0,
                'min_rrr': 2.5,
                'breakeven_r': 1.5,
                'trail_start_r': 2.5,
                'trail_distance_atr': 1.5,
                'max_bars': 20,
                'pullback_tolerance': 0.5,
                'min_adx': 20,
            }

        # Default parameters for other assets
        else:
            return {
                'initial_stop_atr': 2.5,
                'min_rrr': 2.0,
                'breakeven_r': 1.5,
                'trail_start_r': 2.5,
                'trail_distance_atr': 1.5,
                'max_bars': 20,
                'pullback_tolerance': 0.5,
                'min_adx': 20,
            }

    def check_entry(self,
                   df: pd.DataFrame,
                   regime: str,
                   confidence: float) -> Optional[TradeSetup]:
        """
        Check for pullback entry setup.

        Looks for pullbacks to 20 EMA in trending markets with confirmation candles.

        Entry logic:
            1. Validate regime is STRONG_TREND or WEAK_TREND
            2. Check ADX > min_adx threshold
            3. Detect pullback to 20 EMA
            4. Check for confirmation candle (engulfing or rejection)
            5. Calculate entry, stop, and target
            6. Verify RRR meets minimum

        Args:
            df: DataFrame with price data and indicators
                Required columns: open, high, low, close, ema_20, atr_14, adx_14
            regime: Current market regime (from RegimeDetector)
            confidence: Regime confidence score (0-100)

        Returns:
            TradeSetup if valid setup found, None otherwise

        Example:
            >>> strategy = RegimePullbackStrategy(asset="US_TECH_100")
            >>> setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
            >>> if setup:
            ...     print(f"Found {setup.setup_type} at {setup.entry_price}")
        """
        # Validate required indicators
        self._validate_indicators_present(
            df,
            ['open', 'high', 'low', 'close', 'ema_20', 'atr_14', 'adx_14']
        )

        # Only trade in trending regimes
        if regime not in ['STRONG_TREND', 'WEAK_TREND']:
            return None

        # Need at least 3 bars for pattern detection
        if len(df) < 3:
            return None

        # Get current and previous bars
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check ADX threshold
        if current['adx_14'] < self.params['min_adx']:
            return None

        atr = current['atr_14']
        ema = current['ema_20']

        # Check for LONG setup
        if self._check_bullish_structure(df):
            # Price must be near EMA (pullback)
            distance_to_ema = abs(current['close'] - ema) / atr
            if distance_to_ema > self.params['pullback_tolerance']:
                return None

            # Check for bullish confirmation
            if not self._is_bullish_confirmation(current, prev):
                return None

            # Calculate entry, stop, target
            entry = current['close']
            stop = entry - (self.params['initial_stop_atr'] * atr)
            risk = entry - stop
            target = entry + (risk * self.params['min_rrr'])

            # Create setup
            return TradeSetup(
                direction=SignalDirection.LONG,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                risk_per_share=risk,
                confidence=confidence,
                setup_type="PULLBACK_LONG",
                metadata={
                    'ema_20': ema,
                    'atr_14': atr,
                    'adx_14': current['adx_14'],
                    'regime': regime,
                }
            )

        # Check for SHORT setup
        elif self._check_bearish_structure(df):
            # Price must be near EMA (pullback)
            distance_to_ema = abs(current['close'] - ema) / atr
            if distance_to_ema > self.params['pullback_tolerance']:
                return None

            # Check for bearish confirmation
            if not self._is_bearish_confirmation(current, prev):
                return None

            # Calculate entry, stop, target
            entry = current['close']
            stop = entry + (self.params['initial_stop_atr'] * atr)
            risk = stop - entry
            target = entry - (risk * self.params['min_rrr'])

            # Create setup
            return TradeSetup(
                direction=SignalDirection.SHORT,
                entry_price=entry,
                stop_loss=stop,
                target=target,
                risk_per_share=risk,
                confidence=confidence,
                setup_type="PULLBACK_SHORT",
                metadata={
                    'ema_20': ema,
                    'atr_14': atr,
                    'adx_14': current['adx_14'],
                    'regime': regime,
                }
            )

        return None

    def manage_exit(self,
                   df: pd.DataFrame,
                   position: Position) -> tuple[ExitAction, Optional[float]]:
        """
        Manage exit for pullback strategy position.

        Exit logic:
            1. Check for stop loss hit
            2. Check for target hit
            3. Move to breakeven at 1.5R
            4. Trail stop at 2.5R
            5. Exit if max bars exceeded

        Args:
            df: DataFrame with current price data and indicators
                Required columns: close, atr_14
            position: Active position to manage

        Returns:
            Tuple of (ExitAction, new_value):
                - ExitAction.HOLD: No change, keep position
                - ExitAction.STOP: Stop loss hit, exit at stop
                - ExitAction.TARGET: Target hit, exit at target
                - ExitAction.BREAKEVEN: Move stop to breakeven
                - ExitAction.TRAIL: Trail stop to new level
                - ExitAction.TIME_EXIT: Max bars exceeded

        Example:
            >>> action, value = strategy.manage_exit(df, position)
            >>> if action == ExitAction.STOP:
            ...     print(f"Stop loss hit at {value}")
            >>> elif action == ExitAction.TRAIL:
            ...     print(f"Trailing stop to {value}")
        """
        # Validate required indicators
        self._validate_indicators_present(df, ['close', 'atr_14'])

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

            # Move to breakeven at 1.5R
            if r >= self.params['breakeven_r'] and position.stop_loss < position.entry_price:
                return ExitAction.BREAKEVEN, position.entry_price

            # Trail stop at 2.5R
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

            # Move to breakeven at 1.5R
            if r >= self.params['breakeven_r'] and position.stop_loss > position.entry_price:
                return ExitAction.BREAKEVEN, position.entry_price

            # Trail stop at 2.5R
            if r >= self.params['trail_start_r']:
                new_stop = current_price + (self.params['trail_distance_atr'] * atr)
                if new_stop < position.stop_loss:
                    return ExitAction.TRAIL, new_stop

        # Check time-based exit
        current_bar = len(df) - 1
        bars_in_trade = current_bar - position.entry_bar
        if bars_in_trade >= self.params['max_bars']:
            return ExitAction.TIME_EXIT, current_price

        return ExitAction.HOLD, None

    def _check_bullish_structure(self, df: pd.DataFrame) -> bool:
        """
        Check if market structure is bullish (price above EMA).

        Args:
            df: DataFrame with price data and ema_20

        Returns:
            True if bullish structure (close > ema_20)

        Example:
            >>> if self._check_bullish_structure(df):
            ...     print("Market is in bullish trend")
        """
        current = df.iloc[-1]
        return current['close'] > current['ema_20']

    def _check_bearish_structure(self, df: pd.DataFrame) -> bool:
        """
        Check if market structure is bearish (price below EMA).

        Args:
            df: DataFrame with price data and ema_20

        Returns:
            True if bearish structure (close < ema_20)

        Example:
            >>> if self._check_bearish_structure(df):
            ...     print("Market is in bearish trend")
        """
        current = df.iloc[-1]
        return current['close'] < current['ema_20']

    def _is_bullish_confirmation(self, current: pd.Series, prev: pd.Series) -> bool:
        """
        Check for bullish confirmation candle.

        Looks for:
            1. Bullish engulfing: current candle engulfs previous
            2. Rejection wick: long lower wick (> 60% of range)

        Args:
            current: Current bar data (open, high, low, close)
            prev: Previous bar data

        Returns:
            True if bullish confirmation found

        Example:
            >>> if self._is_bullish_confirmation(current, prev):
            ...     print("Bullish confirmation candle detected")
        """
        # Bullish engulfing
        current_body = current['close'] - current['open']
        prev_body = prev['close'] - prev['open']

        if (current_body > 0 and prev_body < 0 and
            current['close'] > prev['open'] and
            current['open'] < prev['close']):
            return True

        # Rejection wick (hammer-like)
        candle_range = current['high'] - current['low']
        if candle_range > 0:
            lower_wick = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
            wick_ratio = lower_wick / candle_range
            if wick_ratio > 0.6 and current['close'] > current['open']:
                return True

        return False

    def _is_bearish_confirmation(self, current: pd.Series, prev: pd.Series) -> bool:
        """
        Check for bearish confirmation candle.

        Looks for:
            1. Bearish engulfing: current candle engulfs previous
            2. Rejection wick: long upper wick (> 60% of range)

        Args:
            current: Current bar data (open, high, low, close)
            prev: Previous bar data

        Returns:
            True if bearish confirmation found

        Example:
            >>> if self._is_bearish_confirmation(current, prev):
            ...     print("Bearish confirmation candle detected")
        """
        # Bearish engulfing
        current_body = current['close'] - current['open']
        prev_body = prev['close'] - prev['open']

        if (current_body < 0 and prev_body > 0 and
            current['close'] < prev['open'] and
            current['open'] > prev['close']):
            return True

        # Rejection wick (shooting star-like)
        candle_range = current['high'] - current['low']
        if candle_range > 0:
            upper_wick = current['high'] - current['open'] if current['close'] < current['open'] else current['high'] - current['close']
            wick_ratio = upper_wick / candle_range
            if wick_ratio > 0.6 and current['close'] < current['open']:
                return True

        return False


# Export
__all__ = ['RegimePullbackStrategy']
