"""
Validation Tests for RegimePullbackStrategy

Tests:
    1. Asset-specific parameters loading
    2. Bullish pullback entry detection
    3. Bearish pullback entry detection
    4. Confirmation candle detection
    5. Exit management (stops, targets, breakeven, trail)
    6. Invalid setups rejected
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.regime_pullback import RegimePullbackStrategy
from src.strategies.base import SignalDirection, ExitAction, Position


def test_asset_parameters():
    """Test 1: Asset-specific parameters loaded correctly."""
    print("\n" + "="*60)
    print("TEST 1: Asset-specific parameters")
    print("="*60)

    # US indices
    strategy_nas = RegimePullbackStrategy(asset="US_TECH_100")
    print(f"\nUS_TECH_100 params: {strategy_nas.params}")
    assert strategy_nas.params['initial_stop_atr'] == 2.0
    assert strategy_nas.params['min_rrr'] == 2.5
    assert strategy_nas.params['breakeven_r'] == 1.5
    print("✓ US_TECH_100 parameters correct")

    # Gold
    strategy_gold = RegimePullbackStrategy(asset="GOLD")
    print(f"\nGOLD params: {strategy_gold.params}")
    assert strategy_gold.params['initial_stop_atr'] == 2.5
    assert strategy_gold.params['min_rrr'] == 2.0
    print("✓ GOLD parameters correct")

    # Forex
    strategy_eur = RegimePullbackStrategy(asset="EUR_USD")
    print(f"\nEUR_USD params: {strategy_eur.params}")
    assert strategy_eur.params['initial_stop_atr'] == 2.0
    assert strategy_eur.params['min_adx'] == 20
    print("✓ EUR_USD parameters correct")

    print("\n✓ TEST 1 PASSED: All asset parameters loaded correctly")


def test_bullish_pullback_entry():
    """Test 2: Bullish pullback entry detection."""
    print("\n" + "="*60)
    print("TEST 2: Bullish pullback entry detection")
    print("="*60)

    strategy = RegimePullbackStrategy(asset="US_TECH_100")

    # Create bullish pullback scenario
    # Price pulled back to EMA, now bouncing with bullish hammer (rejection wick)
    df = pd.DataFrame({
        'open': [100, 102, 101, 101, 100],  # Last: opens at EMA
        'high': [101, 103, 102, 101.5, 101],
        'low': [99, 101, 100, 99, 97],  # Long lower wick on last candle (rejection)
        'close': [100.5, 102.5, 101, 100, 100.5],  # Last candle closes near EMA with long lower wick
        'ema_20': [100, 100, 100, 100, 100],  # Price at EMA
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25, 25, 25],  # Trending
    })

    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")
        print(f"Risk: {setup.risk_per_share}")
        print(f"RRR: {setup.reward_risk_ratio():.2f}")
        print(f"Confidence: {setup.confidence}")
        print(f"Setup type: {setup.setup_type}")

        assert setup.direction == SignalDirection.LONG
        assert setup.entry_price == 100.5  # Current close
        assert setup.stop_loss < setup.entry_price
        assert setup.target > setup.entry_price
        assert setup.reward_risk_ratio() >= 2.5  # Min RRR
        assert setup.setup_type == "PULLBACK_LONG"
        print("\n✓ Bullish pullback detected correctly")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected to find bullish pullback setup"

    print("\n✓ TEST 2 PASSED: Bullish pullback entry working")


def test_bearish_pullback_entry():
    """Test 3: Bearish pullback entry detection."""
    print("\n" + "="*60)
    print("TEST 3: Bearish pullback entry detection")
    print("="*60)

    strategy = RegimePullbackStrategy(asset="US_TECH_100")

    # Create bearish pullback scenario
    # Price pulled back to EMA, now rejecting with bearish shooting star (rejection wick)
    df = pd.DataFrame({
        'open': [100, 98, 99, 99, 100],  # Last: opens at EMA
        'high': [101, 99, 100, 100.5, 103],  # Long upper wick on last candle (rejection)
        'low': [99, 97, 98, 99, 99],
        'close': [99.5, 97.5, 99, 100, 99.5],  # Last candle closes near EMA with long upper wick
        'ema_20': [100, 100, 100, 100, 100],  # Price at EMA
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25, 25, 25],  # Trending
    })

    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=80.0)

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")
        print(f"Risk: {setup.risk_per_share}")
        print(f"RRR: {setup.reward_risk_ratio():.2f}")
        print(f"Confidence: {setup.confidence}")
        print(f"Setup type: {setup.setup_type}")

        assert setup.direction == SignalDirection.SHORT
        assert setup.entry_price == 99.5  # Current close
        assert setup.stop_loss > setup.entry_price
        assert setup.target < setup.entry_price
        assert setup.reward_risk_ratio() >= 2.5  # Min RRR
        assert setup.setup_type == "PULLBACK_SHORT"
        print("\n✓ Bearish pullback detected correctly")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected to find bearish pullback setup"

    print("\n✓ TEST 3 PASSED: Bearish pullback entry working")


def test_exit_management():
    """Test 4: Exit management (stops, targets, breakeven, trail)."""
    print("\n" + "="*60)
    print("TEST 4: Exit management")
    print("="*60)

    strategy = RegimePullbackStrategy(asset="US_TECH_100")

    # Create LONG position
    position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=96.0,  # 4.0 risk
        target=114.0,  # 3.5R target to allow testing trail at 2.5R
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="RegimePullbackStrategy"
    )

    # Test 1: HOLD when no conditions met
    df = pd.DataFrame({
        'close': [101.0],
        'atr_14': [2.0],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"\nAt 101.0 (0.25R): {action} - {value}")
    assert action == ExitAction.HOLD
    print("✓ HOLD action correct")

    # Test 2: BREAKEVEN at 1.5R
    df = pd.DataFrame({
        'close': [106.0],  # 1.5R profit
        'atr_14': [2.0],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 106.0 (1.5R): {action} - {value}")
    assert action == ExitAction.BREAKEVEN
    assert value == position.entry_price
    print("✓ BREAKEVEN action correct")

    # Update position to breakeven
    position.stop_loss = position.entry_price

    # Test 3: TRAIL at 2.5R
    df = pd.DataFrame({
        'close': [110.0],  # 2.5R profit
        'atr_14': [2.0],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 110.0 (2.5R): {action} - {value}")
    assert action == ExitAction.TRAIL
    assert value > position.stop_loss  # Trail should move up
    expected_trail = 110.0 - (1.5 * 2.0)  # current - (trail_distance_atr * atr)
    assert abs(value - expected_trail) < 0.01
    print(f"✓ TRAIL action correct (new stop: {value})")

    # Test 4: TARGET hit
    df = pd.DataFrame({
        'close': [114.5],  # Above target (114.0)
        'atr_14': [2.0],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 114.5 (target hit): {action} - {value}")
    assert action == ExitAction.TARGET
    assert value == position.target
    print("✓ TARGET action correct")

    # Test 5: STOP loss hit
    position2 = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=96.0,
        target=110.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="RegimePullbackStrategy"
    )
    df = pd.DataFrame({
        'close': [95.0],  # Below stop
        'atr_14': [2.0],
    })
    action, value = strategy.manage_exit(df, position2)
    print(f"At 95.0 (stop hit): {action} - {value}")
    assert action == ExitAction.STOP
    assert value == position2.stop_loss
    print("✓ STOP action correct")

    print("\n✓ TEST 4 PASSED: Exit management working correctly")


def test_invalid_setups_rejected():
    """Test 5: Invalid setups are rejected."""
    print("\n" + "="*60)
    print("TEST 5: Invalid setups rejected")
    print("="*60)

    strategy = RegimePullbackStrategy(asset="US_TECH_100")

    # Test 1: Wrong regime
    df = pd.DataFrame({
        'open': [100, 102, 101],
        'high': [101, 103, 102],
        'low': [99, 101, 100],
        'close': [100.5, 102.5, 101],
        'ema_20': [100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25],
    })
    setup = strategy.check_entry(df, regime="SIDEWAYS", confidence=85.0)
    print(f"\nWrong regime (SIDEWAYS): {setup is None}")
    assert setup is None
    print("✓ Wrong regime rejected")

    # Test 2: Low ADX (not trending)
    df = pd.DataFrame({
        'open': [100, 102, 101],
        'high': [101, 103, 102],
        'low': [99, 101, 100],
        'close': [100.5, 102.5, 101],
        'ema_20': [100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0],
        'adx_14': [15, 15, 15],  # Below min_adx threshold
    })
    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
    print(f"Low ADX (15): {setup is None}")
    assert setup is None
    print("✓ Low ADX rejected")

    # Test 3: Price too far from EMA
    df = pd.DataFrame({
        'open': [100, 102, 101],
        'high': [101, 103, 102],
        'low': [99, 101, 100],
        'close': [100.5, 102.5, 105],  # Far from EMA
        'ema_20': [100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25],
    })
    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
    print(f"Price too far from EMA: {setup is None}")
    assert setup is None
    print("✓ Price too far from EMA rejected")

    print("\n✓ TEST 5 PASSED: Invalid setups correctly rejected")


def test_confirmation_candles():
    """Test 6: Confirmation candle detection."""
    print("\n" + "="*60)
    print("TEST 6: Confirmation candle detection")
    print("="*60)

    strategy = RegimePullbackStrategy(asset="US_TECH_100")

    # Test bullish engulfing
    current = pd.Series({'open': 99, 'high': 102, 'low': 98, 'close': 101})
    prev = pd.Series({'open': 100, 'high': 101, 'low': 99, 'close': 99.5})
    is_bullish = strategy._is_bullish_confirmation(current, prev)
    print(f"\nBullish engulfing detected: {is_bullish}")
    assert is_bullish
    print("✓ Bullish engulfing works")

    # Test bearish engulfing
    current = pd.Series({'open': 101, 'high': 102, 'low': 98, 'close': 99})
    prev = pd.Series({'open': 100, 'high': 101, 'low': 99, 'close': 100.5})
    is_bearish = strategy._is_bearish_confirmation(current, prev)
    print(f"Bearish engulfing detected: {is_bearish}")
    assert is_bearish
    print("✓ Bearish engulfing works")

    # Test bullish rejection wick (hammer)
    current = pd.Series({'open': 100, 'high': 101, 'low': 97, 'close': 100.5})  # Long lower wick
    prev = pd.Series({'open': 100, 'high': 101, 'low': 99, 'close': 99.5})
    is_bullish = strategy._is_bullish_confirmation(current, prev)
    print(f"Bullish hammer detected: {is_bullish}")
    assert is_bullish
    print("✓ Bullish hammer works")

    # Test bearish rejection wick (shooting star)
    current = pd.Series({'open': 100, 'high': 103, 'low': 99, 'close': 99.5})  # Long upper wick
    prev = pd.Series({'open': 100, 'high': 101, 'low': 99, 'close': 100.5})
    is_bearish = strategy._is_bearish_confirmation(current, prev)
    print(f"Bearish shooting star detected: {is_bearish}")
    assert is_bearish
    print("✓ Bearish shooting star works")

    print("\n✓ TEST 6 PASSED: Confirmation candles detected correctly")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("REGIME PULLBACK STRATEGY VALIDATION")
    print("="*60)

    try:
        test_asset_parameters()
        test_bullish_pullback_entry()
        test_bearish_pullback_entry()
        test_exit_management()
        test_invalid_setups_rejected()
        test_confirmation_candles()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nRegimePullbackStrategy is production-ready!")
        print("- Asset-specific parameters: ✓")
        print("- Entry detection (LONG/SHORT): ✓")
        print("- Exit management: ✓")
        print("- Confirmation candles: ✓")
        print("- Invalid setup rejection: ✓")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
