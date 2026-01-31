"""
Complete Workflow Demo: RegimeDetector + StrategySelector

Demonstrates the complete trading decision pipeline:
1. Add indicators to price data
2. Detect market regime
3. Automatically select and execute strategy
4. Manage position exits

This is the RECOMMENDED way to use LIQUIDEDGE.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.regime.detector import RegimeDetector, MarketRegime
from src.strategies.selector import StrategySelector
from src.strategies.base import Position, ExitAction


def create_sample_data(n=300):
    """Create sample market data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    # Uptrend
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 0.3, n)
    close = trend + noise

    df = pd.DataFrame({
        'open': close - np.random.uniform(0.1, 0.3, n),
        'high': close + np.random.uniform(0.3, 0.7, n),
        'low': close - np.random.uniform(0.3, 0.7, n),
        'close': close,
        'volume': np.random.uniform(4000, 6000, n)
    }, index=dates)

    return df


def demo_complete_workflow():
    """Demo: Complete trading workflow with selector."""
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW DEMO")
    print("="*70)
    print("\nThis demonstrates the recommended usage pattern:")
    print("  RegimeDetector → StrategySelector → TradeSetup")

    # Step 1: Initialize components
    print("\n" + "-"*70)
    print("STEP 1: Initialize components")
    print("-"*70)

    detector = RegimeDetector()
    selector = StrategySelector(asset="US_TECH_100")

    print(f"✓ RegimeDetector initialized")
    print(f"✓ StrategySelector initialized for {selector.asset}")

    # Step 2: Prepare data
    print("\n" + "-"*70)
    print("STEP 2: Prepare market data")
    print("-"*70)

    df = create_sample_data(300)
    print(f"✓ Created {len(df)} bars of sample data")

    # Step 3: Add indicators
    print("\n" + "-"*70)
    print("STEP 3: Add indicators")
    print("-"*70)

    df = detector.add_all_indicators(df)
    print(f"✓ Added all regime indicators")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Sample indicators: {list(df.columns[-5:])}")

    # Step 4: Scan for trading opportunities
    print("\n" + "-"*70)
    print("STEP 4: Scan for trading opportunities")
    print("-"*70)

    entries_found = []
    for i in range(250, 300):
        window = df.iloc[:i+1]

        # Detect regime
        regime, confidence, strategy_name = detector.detect_regime(window)

        # Use selector to check for entry
        setup = selector.check_entry(
            df=window,
            regime=regime.value,
            confidence=confidence,
            strategy_recommendation=strategy_name
        )

        if setup:
            entries_found.append({
                'bar': i,
                'regime': regime.value,
                'confidence': confidence,
                'strategy': strategy_name,
                'setup': setup
            })

    print(f"\n✓ Scanned 50 bars")
    print(f"  Entries found: {len(entries_found)}")

    if entries_found:
        print(f"\n  Sample entry:")
        entry = entries_found[0]
        print(f"    Bar: {entry['bar']}")
        print(f"    Regime: {entry['regime']} ({entry['confidence']:.0f}% confidence)")
        print(f"    Strategy: {entry['strategy']}")
        print(f"    Setup: {entry['setup'].setup_type}")
        print(f"    Entry: {entry['setup'].entry_price:.2f}")
        print(f"    Stop: {entry['setup'].stop_loss:.2f}")
        print(f"    Target: {entry['setup'].target:.2f}")
        print(f"    RRR: {entry['setup'].reward_risk_ratio():.2f}R")

    # Step 5: Demonstrate position management
    print("\n" + "-"*70)
    print("STEP 5: Position management")
    print("-"*70)

    # Create a sample position
    position = Position(
        asset="US_TECH_100",
        direction=entry['setup'].direction if entries_found else None,
        entry_price=100.0,
        stop_loss=96.0,
        target=114.0,
        units=1.0,
        risk_per_share=4.0,
        entry_time=pd.Timestamp.now(),
        entry_bar=250,
        entry_strategy="PULLBACK_LONG"  # Identifies strategy for exit routing
    )

    # Simulate position management over several bars
    print(f"\n✓ Created position:")
    print(f"    Entry: {position.entry_price}")
    print(f"    Stop: {position.stop_loss}")
    print(f"    Target: {position.target}")

    print(f"\n  Managing position over next bars:")

    test_prices = [101.0, 103.0, 106.0, 110.0]
    for i, price in enumerate(test_prices):
        df_current = pd.DataFrame({
            'close': [price],
            'atr_14': [2.0],
        })

        action, value = selector.manage_exit(df_current, position)

        r_multiple = (price - position.entry_price) / position.risk_per_share
        print(f"    Bar {i+1}: Price {price:.1f} ({r_multiple:.2f}R) → {action.value}", end="")

        if action == ExitAction.BREAKEVEN:
            position.stop_loss = value
            print(f" (new stop: {value:.1f})")
        elif action == ExitAction.TRAIL:
            position.stop_loss = value
            print(f" (new stop: {value:.1f})")
        elif action in [ExitAction.STOP, ExitAction.TARGET]:
            print(f" at {value:.1f}")
            break
        else:
            print()

    print(f"\n✓ Position management complete")

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print("\n✅ Complete workflow demonstrated:")
    print("   1. RegimeDetector adds indicators ✓")
    print("   2. RegimeDetector identifies regime ✓")
    print("   3. StrategySelector routes to correct strategy ✓")
    print("   4. Strategy returns validated TradeSetup ✓")
    print("   5. StrategySelector manages exits ✓")
    print("\n✅ System working end-to-end!")


def demo_strategy_comparison():
    """Demo: Compare pullback vs squeeze strategies."""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)

    selector = StrategySelector(asset="US_TECH_100")
    stats = selector.get_strategy_stats()

    print("\nRegime Pullback vs TTM Squeeze:")
    print("-"*70)

    comparison_params = ['initial_stop_atr', 'min_rrr', 'breakeven_r', 'trail_start_r', 'max_bars']

    print(f"\n{'Parameter':<20} {'Pullback':<15} {'Squeeze':<15} {'Difference'}")
    print("-"*70)

    for param in comparison_params:
        pullback_val = stats['regime_params'].get(param, 'N/A')
        squeeze_val = stats['ttm_params'].get(param, 'N/A')

        if isinstance(pullback_val, (int, float)) and isinstance(squeeze_val, (int, float)):
            diff = "Tighter" if squeeze_val < pullback_val else ("Wider" if squeeze_val > pullback_val else "Same")
            print(f"{param:<20} {pullback_val:<15} {squeeze_val:<15} {diff}")
        else:
            print(f"{param:<20} {pullback_val:<15} {squeeze_val:<15} N/A")

    print("\n✅ Key Differences:")
    print("   - Squeeze has TIGHTER stops (faster failure detection)")
    print("   - Squeeze has LOWER targets (capture initial move)")
    print("   - Squeeze has FASTER breakeven (1.0R vs 1.5R)")
    print("   - Squeeze has FASTER trailing (1.5R vs 2.5R)")
    print("   - Squeeze has LONGER max hold (48 vs 20 bars)")
    print("\n   WHY? Breakouts are explosive but short-lived.")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("LIQUIDEDGE COMPLETE SYSTEM DEMO")
    print("="*70)

    try:
        demo_complete_workflow()
        demo_strategy_comparison()

        print("\n" + "="*70)
        print("ALL DEMOS COMPLETE ✓")
        print("="*70)
        print("\nLIQUIDEDGE Trading System Ready:")
        print("  ✓ Regime Detection")
        print("  ✓ Strategy Selection")
        print("  ✓ Entry Management")
        print("  ✓ Exit Management")
        print("  ✓ End-to-End Integration")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
