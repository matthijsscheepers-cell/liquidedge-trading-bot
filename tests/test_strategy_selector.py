"""
Validation Tests for StrategySelector

Tests:
    1. Initialization with strategies
    2. Routing to RegimePullbackStrategy
    3. Routing to TTMSqueezeStrategy
    4. Exit routing based on entry strategy
    5. Strategy stats and validation
    6. Unknown strategy handling
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.strategies.selector import StrategySelector
from src.strategies.base import SignalDirection, ExitAction, Position


def test_initialization():
    """Test 1: StrategySelector initialization."""
    print("\n" + "="*60)
    print("TEST 1: StrategySelector initialization")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    print(f"\nAsset: {selector.asset}")
    print(f"Regime strategy: {selector.regime_strategy}")
    print(f"TTM strategy: {selector.ttm_strategy}")

    assert selector.asset == "US_TECH_100"
    assert selector.regime_strategy is not None
    assert selector.ttm_strategy is not None
    assert selector.regime_strategy.asset == "US_TECH_100"
    assert selector.ttm_strategy.asset == "US_TECH_100"

    print("\n✓ Both strategies initialized correctly")
    print("\n✓ TEST 1 PASSED: Initialization working")


def test_routing_to_pullback():
    """Test 2: Routing to RegimePullbackStrategy."""
    print("\n" + "="*60)
    print("TEST 2: Routing to RegimePullbackStrategy")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Create pullback scenario
    df = pd.DataFrame({
        'open': [100, 102, 101, 101, 100],
        'high': [101, 103, 102, 101.5, 101],
        'low': [99, 101, 100, 99, 97],
        'close': [100.5, 102.5, 101, 100, 100.5],
        'ema_20': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25, 25, 25],
    })

    # Use selector with REGIME_PULLBACK recommendation
    setup = selector.check_entry(
        df=df,
        regime="STRONG_TREND",
        confidence=85.0,
        strategy_recommendation="REGIME_PULLBACK"
    )

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Setup type: {setup.setup_type}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")

        assert setup.direction == SignalDirection.LONG
        assert "PULLBACK" in setup.setup_type
        print("\n✓ Correctly routed to pullback strategy")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected pullback setup"

    print("\n✓ TEST 2 PASSED: Pullback routing working")


def test_routing_to_squeeze():
    """Test 3: Routing to TTMSqueezeStrategy."""
    print("\n" + "="*60)
    print("TEST 3: Routing to TTMSqueezeStrategy")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Create squeeze release scenario
    df = pd.DataFrame({
        'open': [100, 100, 100, 100, 100],
        'high': [101, 101, 101, 101, 102],
        'low': [99, 99, 99, 99, 99],
        'close': [100, 100, 100, 100, 101],
        'squeeze_on': [True, True, True, True, False],  # Released
        'ttm_momentum': [0.1, 0.15, 0.2, 0.25, 0.3],  # Increasing
        'kc_middle': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
    })

    # Use selector with TTM_SQUEEZE recommendation
    setup = selector.check_entry(
        df=df,
        regime="RANGE_COMPRESSION",
        confidence=90.0,
        strategy_recommendation="TTM_SQUEEZE"
    )

    print(f"\nSetup found: {setup is not None}")
    if setup:
        print(f"Direction: {setup.direction}")
        print(f"Setup type: {setup.setup_type}")
        print(f"Entry: {setup.entry_price}")
        print(f"Stop: {setup.stop_loss}")
        print(f"Target: {setup.target}")

        assert setup.direction == SignalDirection.LONG
        assert "SQUEEZE" in setup.setup_type
        print("\n✓ Correctly routed to squeeze strategy")
    else:
        print("\n✗ No setup found (unexpected)")
        assert False, "Expected squeeze setup"

    print("\n✓ TEST 3 PASSED: Squeeze routing working")


def test_exit_routing():
    """Test 4: Exit routing based on entry strategy."""
    print("\n" + "="*60)
    print("TEST 4: Exit routing based on entry strategy")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Test routing for PULLBACK position
    pullback_position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=96.0,
        target=114.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="PULLBACK_LONG"  # Key: identifies strategy
    )

    df = pd.DataFrame({
        'close': [101.0],
        'atr_14': [2.0],
        'ttm_momentum': [0.3],
    })

    action, value = selector.manage_exit(df, pullback_position)
    print(f"\nPullback position exit: {action} - {value}")
    assert action == ExitAction.HOLD
    print("✓ Pullback position routed to RegimePullbackStrategy")

    # Test routing for SQUEEZE position
    squeeze_position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=97.0,
        target=105.4,
        units=1.0,
        risk_per_share=3.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="SQUEEZE_RELEASE_LONG",  # Key: identifies strategy
        metadata={'ttm_momentum': 0.3}
    )

    action, value = selector.manage_exit(df, squeeze_position)
    print(f"Squeeze position exit: {action} - {value}")
    assert action == ExitAction.HOLD
    print("✓ Squeeze position routed to TTMSqueezeStrategy")

    print("\n✓ TEST 4 PASSED: Exit routing working correctly")


def test_strategy_stats():
    """Test 5: Strategy stats and validation."""
    print("\n" + "="*60)
    print("TEST 5: Strategy stats and validation")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Get stats
    stats = selector.get_strategy_stats()
    print(f"\nAsset: {stats['asset']}")
    print(f"\nRegime strategy params:")
    for key, value in stats['regime_params'].items():
        print(f"  {key}: {value}")

    print(f"\nTTM strategy params:")
    for key, value in stats['ttm_params'].items():
        print(f"  {key}: {value}")

    assert stats['asset'] == "US_TECH_100"
    assert 'initial_stop_atr' in stats['regime_params']
    assert 'initial_stop_atr' in stats['ttm_params']

    # Verify different parameters
    assert stats['regime_params']['initial_stop_atr'] == 2.0  # Pullback
    assert stats['ttm_params']['initial_stop_atr'] == 1.5  # Squeeze (tighter)
    print("\n✓ Different parameters confirmed")

    # Test get_strategy_for_regime
    regime_strat = selector.get_strategy_for_regime("STRONG_TREND")
    squeeze_strat = selector.get_strategy_for_regime("RANGE_COMPRESSION")
    none_strat = selector.get_strategy_for_regime("HIGH_VOLATILITY")

    assert regime_strat is not None
    assert squeeze_strat is not None
    assert none_strat is None
    print("✓ get_strategy_for_regime working")

    print("\n✓ TEST 5 PASSED: Stats and helpers working")


def test_unknown_strategy_handling():
    """Test 6: Unknown strategy handling."""
    print("\n" + "="*60)
    print("TEST 6: Unknown strategy handling")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Test with unknown recommendation
    df = pd.DataFrame({
        'open': [100],
        'high': [101],
        'low': [99],
        'close': [100],
    })

    setup = selector.check_entry(
        df=df,
        regime="NO_TRADE",
        confidence=0.0,
        strategy_recommendation="NONE"
    )

    print(f"\nSetup with 'NONE' recommendation: {setup}")
    assert setup is None
    print("✓ Returns None for unknown recommendation")

    # Test exit with unknown entry strategy
    unknown_position = Position(
        asset="US_TECH_100",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=96.0,
        target=110.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=0,
        entry_strategy="UNKNOWN_STRATEGY"  # Unknown
    )

    df = pd.DataFrame({
        'close': [101.0],
        'atr_14': [2.0],
    })

    action, value = selector.manage_exit(df, unknown_position)
    print(f"Unknown strategy fallback: {action} - {value}")
    assert action == ExitAction.HOLD  # Falls back to regime strategy
    print("✓ Falls back to RegimePullbackStrategy for unknown")

    print("\n✓ TEST 6 PASSED: Unknown strategy handling working")


def test_validate_setup():
    """Test 7: Setup validation."""
    print("\n" + "="*60)
    print("TEST 7: Setup validation")
    print("="*60)

    selector = StrategySelector(asset="US_TECH_100")

    # Create valid pullback setup
    df = pd.DataFrame({
        'open': [100, 102, 101, 101, 100],
        'high': [101, 103, 102, 101.5, 101],
        'low': [99, 101, 100, 99, 97],
        'close': [100.5, 102.5, 101, 100, 100.5],
        'ema_20': [100, 100, 100, 100, 100],
        'atr_14': [2.0, 2.0, 2.0, 2.0, 2.0],
        'adx_14': [25, 25, 25, 25, 25],
    })

    setup = selector.check_entry(
        df=df,
        regime="STRONG_TREND",
        confidence=85.0,
        strategy_recommendation="REGIME_PULLBACK"
    )

    if setup:
        is_valid = selector.validate_setup(setup)
        print(f"\nSetup valid: {is_valid}")
        print(f"Setup RRR: {setup.reward_risk_ratio():.2f}")
        assert is_valid
        print("✓ Valid setup passes validation")
    else:
        assert False, "Expected to find setup"

    print("\n✓ TEST 7 PASSED: Setup validation working")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("STRATEGY SELECTOR VALIDATION")
    print("="*60)

    try:
        test_initialization()
        test_routing_to_pullback()
        test_routing_to_squeeze()
        test_exit_routing()
        test_strategy_stats()
        test_unknown_strategy_handling()
        test_validate_setup()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nStrategySelector is production-ready!")
        print("- Initialization: ✓")
        print("- Pullback routing: ✓")
        print("- Squeeze routing: ✓")
        print("- Exit routing: ✓")
        print("- Strategy stats: ✓")
        print("- Unknown handling: ✓")
        print("- Setup validation: ✓")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
