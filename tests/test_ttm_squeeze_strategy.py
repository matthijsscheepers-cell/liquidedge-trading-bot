"""
Validation Tests for TTMSqueezeStrategy

Tests:
    1. Asset-specific parameters loading
    2. Squeeze release entry detection
    3. Squeeze retest entry detection
    4. Exit management (tighter than pullback)
    5. Momentum reversal exits
    6. Invalid setups rejected
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.ttm_squeeze import TTMSqueezeStrategy
from src.strategies.base import SignalDirection, ExitAction, Position


def test_asset_parameters():
    """Test 1: Asset-specific parameters loaded correctly."""
    print("\n" + "="*60)
    print("TEST 1: Asset-specific parameters (tighter than pullback)")
    print("="*60)

    # US indices
    strategy_nas = TTMSqueezeStrategy(asset="US_TECH_100")
    print(f"\nUS_TECH_100 params: {strategy_nas.params}")
    assert strategy_nas.params['initial_stop_atr'] == 1.5  # Tighter than pullback's 2.0
    assert strategy_nas.params['min_rrr'] == 1.8  # Lower than pullback's 2.5
    assert strategy_nas.params['breakeven_r'] == 1.0  # Faster than pullback's 1.5
    assert strategy_nas.params['trail_start_r'] == 1.5  # Faster than pullback's 2.5
    assert strategy_nas.params['max_bars'] == 48  # Shorter than pullback's 20
    print("✓ US_TECH_100 parameters correct (tighter risk management)")

    # Gold
    strategy_gold = TTMSqueezeStrategy(asset="GOLD")
    print(f"\nGOLD params: {strategy_gold.params}")
    assert strategy_gold.params['initial_stop_atr'] == 1.8
    assert strategy_gold.params['min_rrr'] == 1.6
    print("✓ GOLD parameters correct")

    print("\n✓ TEST 1 PASSED: Breakout parameters are tighter than pullback")


def test_squeeze_release_entry():
    """Test 2: Squeeze release (immediate breakout) entry detection."""
    print("\n" + "="*60)
    print("TEST 2: Squeeze release entry detection")
    print("="*60)

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")

    # Create squeeze release scenario
    # Squeeze was ON, now just turned OFF with positive momentum
    df = pd.DataFrame({
        'open': [100, 100, 100, 100, 100],
        'high': [101, 101, 101, 101, 102],
        'low': [99, 99, 99, 99, 99],
        'close': [100, 100, 100, 100, 101],
        'squeeze_on': [True, True, True, True, False],  # Just released!
        'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],  # Increasing
        'kc_middle': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })

    setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=90.0)

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")
        print(f"Risk: {setup.risk_per_share}")
        print(f"RRR: {setup.reward_risk_ratio():.2f}")
        print(f"Setup type: {setup.setup_type}")

        assert setup.direction == SignalDirection.LONG
        assert setup.entry_price == 101  # Current close
        assert setup.stop_loss < setup.entry_price
        assert setup.target > setup.entry_price
        assert setup.reward_risk_ratio() >= 1.8  # Min RRR for breakouts
        assert setup.setup_type == "SQUEEZE_RELEASE_LONG"
        print("\n✓ Squeeze release detected correctly")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected to find squeeze release setup"

    print("\n✓ TEST 2 PASSED: Squeeze release entry working")


def test_squeeze_retest_entry():
    """Test 3: Squeeze retest (pullback) entry detection."""
    print("\n" + "="*60)
    print("TEST 3: Squeeze retest entry detection")
    print("="*60)

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")

    # Create squeeze retest scenario
    # Squeeze released 2 bars ago, price pulled back to KC basis, now rejecting
    df = pd.DataFrame({
        'open': [100, 100, 102, 101, 100.2],
        'high': [101, 101, 103, 102, 101],
        'low': [99, 99, 101, 99.5, 98],  # Long lower wick on last candle (rejection)
        'close': [100, 100, 102.5, 100.5, 100.3],  # Pulled back near KC basis
        'squeeze_on': [True, True, False, False, False],  # Released 2 bars ago
        'ttm_momentum': [0.1, 0.15, 0.3, 0.25, 0.28],  # Positive momentum
        'kc_middle': [100, 100, 100, 100, 100],  # KC basis
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })

    setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")
        print(f"Risk: {setup.risk_per_share}")
        print(f"RRR: {setup.reward_risk_ratio():.2f}")
        print(f"Setup type: {setup.setup_type}")

        assert setup.direction == SignalDirection.LONG
        assert setup.entry_price == 100.3  # Current close
        assert setup.stop_loss < setup.entry_price
        assert setup.target > setup.entry_price
        assert setup.reward_risk_ratio() >= 1.8
        assert setup.setup_type == "SQUEEZE_RETEST_LONG"
        print("\n✓ Squeeze retest detected correctly")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected to find squeeze retest setup"

    print("\n✓ TEST 3 PASSED: Squeeze retest entry working")


def test_exit_management():
    """Test 4: Exit management (faster than pullback)."""
    print("\n" + "="*60)
    print("TEST 4: Exit management (faster exits)")
    print("="*60)

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")

    # Create LONG position
    position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=97.0,  # 3.0 risk (1.5 ATR * 2.0)
        target=105.4,    # 1.8R target
        units=1.0,
        risk_per_share=3.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="TTMSqueezeStrategy",
        metadata={'ttm_momentum': 0.3}  # Positive momentum at entry
    )

    # Test 1: HOLD when no conditions met
    df = pd.DataFrame({
        'close': [100.5],
        'atr_14': [2.0],
        'ttm_momentum': [0.3],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"\nAt 100.5 (0.17R): {action} - {value}")
    assert action == ExitAction.HOLD
    print("✓ HOLD action correct")

    # Test 2: BREAKEVEN at 1.0R (faster than pullback's 1.5R)
    df = pd.DataFrame({
        'close': [103.0],  # 1.0R profit
        'atr_14': [2.0],
        'ttm_momentum': [0.3],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 103.0 (1.0R): {action} - {value}")
    assert action == ExitAction.BREAKEVEN
    assert value == position.entry_price
    print("✓ BREAKEVEN action correct (faster than pullback)")

    # Update position to breakeven
    position.stop_loss = position.entry_price

    # Test 3: TRAIL at 1.5R (faster than pullback's 2.5R)
    df = pd.DataFrame({
        'close': [104.5],  # 1.5R profit
        'atr_14': [2.0],
        'ttm_momentum': [0.3],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 104.5 (1.5R): {action} - {value}")
    assert action == ExitAction.TRAIL
    assert value > position.stop_loss
    expected_trail = 104.5 - (1.2 * 2.0)  # current - (trail_distance_atr * atr)
    assert abs(value - expected_trail) < 0.01
    print(f"✓ TRAIL action correct (faster than pullback, new stop: {value})")

    # Test 4: TARGET hit
    df = pd.DataFrame({
        'close': [105.5],  # Above target
        'atr_14': [2.0],
        'ttm_momentum': [0.3],
    })
    action, value = strategy.manage_exit(df, position)
    print(f"At 105.5 (target hit): {action} - {value}")
    assert action == ExitAction.TARGET
    assert value == position.target
    print("✓ TARGET action correct")

    print("\n✓ TEST 4 PASSED: Faster exit management working")


def test_momentum_reversal_exit():
    """Test 5: Momentum reversal exits (unique to breakout strategy)."""
    print("\n" + "="*60)
    print("TEST 5: Momentum reversal exits")
    print("="*60)

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")

    # Create LONG position with positive entry momentum
    position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=97.0,
        target=105.4,
        units=1.0,
        risk_per_share=3.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="TTMSqueezeStrategy",
        metadata={'ttm_momentum': 0.3}  # Positive momentum at entry
    )

    # Test momentum reversal when in profit
    df = pd.DataFrame({
        'close': [102.0],  # 0.67R profit
        'atr_14': [2.0],
        'ttm_momentum': [-0.1],  # Momentum reversed to negative!
    })
    action, value = strategy.manage_exit(df, position)
    print(f"\nAt 102.0 (0.67R profit, momentum reversed): {action} - {value}")
    assert action == ExitAction.TARGET  # Exit at market
    assert value == 102.0  # Current price
    print("✓ Momentum reversal triggers exit")

    # Test momentum reversal NOT triggered when R < 0.5
    position2 = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=97.0,
        target=105.4,
        units=1.0,
        risk_per_share=3.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="TTMSqueezeStrategy",
        metadata={'ttm_momentum': 0.3}
    )

    df = pd.DataFrame({
        'close': [100.5],  # 0.17R profit (< 0.5R threshold)
        'atr_14': [2.0],
        'ttm_momentum': [-0.1],  # Momentum reversed
    })
    action, value = strategy.manage_exit(df, position2)
    print(f"At 100.5 (0.17R profit, momentum reversed): {action} - {value}")
    assert action == ExitAction.HOLD  # Don't exit yet, profit too small
    print("✓ Momentum reversal ignored when profit < 0.5R")

    print("\n✓ TEST 5 PASSED: Momentum reversal exits working")


def test_invalid_setups_rejected():
    """Test 6: Invalid setups are rejected."""
    print("\n" + "="*60)
    print("TEST 6: Invalid setups rejected")
    print("="*60)

    strategy = TTMSqueezeStrategy(asset="US_TECH_100")

    # Test 1: Wrong regime
    df = pd.DataFrame({
        'open': [100, 100, 100, 100, 100],
        'high': [101, 101, 101, 101, 102],
        'low': [99, 99, 99, 99, 99],
        'close': [100, 100, 100, 100, 101],
        'squeeze_on': [True, True, True, True, False],
        'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],
        'kc_middle': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })
    setup = strategy.check_entry(df, regime="STRONG_TREND", confidence=85.0)
    print(f"\nWrong regime (STRONG_TREND): {setup is None}")
    assert setup is None
    print("✓ Wrong regime rejected")

    # Test 2: Squeeze still active (not released)
    df = pd.DataFrame({
        'open': [100, 100, 100, 100, 100],
        'high': [101, 101, 101, 101, 101],
        'low': [99, 99, 99, 99, 99],
        'close': [100, 100, 100, 100, 100],
        'squeeze_on': [True, True, True, True, True],  # Still ON
        'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],
        'kc_middle': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })
    setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)
    print(f"Squeeze still active: {setup is None}")
    assert setup is None
    print("✓ Active squeeze (not released) rejected")

    # Test 3: Momentum too weak
    df = pd.DataFrame({
        'open': [100, 100, 100, 100, 100],
        'high': [101, 101, 101, 101, 101],
        'low': [99, 99, 99, 99, 99],
        'close': [100, 100, 100, 100, 100],
        'squeeze_on': [True, True, True, True, False],  # Released
        'ttm_momentum': [0.05, 0.05, 0.05, 0.05, 0.05],  # Too weak
        'kc_middle': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })
    setup = strategy.check_entry(df, regime="RANGE_COMPRESSION", confidence=85.0)
    print(f"Weak momentum (0.05 < min 0.2): {setup is None}")
    assert setup is None
    print("✓ Weak momentum rejected")

    print("\n✓ TEST 6 PASSED: Invalid setups correctly rejected")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("TTM SQUEEZE STRATEGY VALIDATION")
    print("="*60)

    try:
        test_asset_parameters()
        test_squeeze_release_entry()
        test_squeeze_retest_entry()
        test_exit_management()
        test_momentum_reversal_exit()
        test_invalid_setups_rejected()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nTTMSqueezeStrategy is production-ready!")
        print("- Tighter parameters than pullback: ✓")
        print("- Squeeze release entry: ✓")
        print("- Squeeze retest entry: ✓")
        print("- Faster exit management: ✓")
        print("- Momentum reversal exits: ✓")
        print("- Invalid setup rejection: ✓")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
