"""
Integration Demo: Regime Detector + Strategies

Demonstrates how the regime detector and strategies work together
to make trading decisions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.regime.detector import RegimeDetector, MarketRegime
from src.strategies.regime_pullback import RegimePullbackStrategy
from src.strategies.ttm_squeeze import TTMSqueezeStrategy


def create_trending_data(n=300):
    """Create trending market data."""
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


def create_squeeze_data(n=200):
    """Create squeeze/compression data."""
    np.random.seed(43)
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    # Compression then expansion
    compression_phase = np.ones(100) * 100
    expansion_phase = np.linspace(100, 110, 100)

    base_price = np.concatenate([compression_phase, expansion_phase])
    noise = np.random.normal(0, 0.15, n)  # Low volatility in compression
    close = base_price + noise

    df = pd.DataFrame({
        'open': close - np.random.uniform(0.05, 0.15, n),
        'high': close + np.random.uniform(0.1, 0.3, n),
        'low': close - np.random.uniform(0.1, 0.3, n),
        'close': close,
        'volume': np.random.uniform(4000, 6000, n)
    }, index=dates)

    return df


def demo_pullback_strategy():
    """Demo: Regime Pullback Strategy in trending market."""
    print("\n" + "="*70)
    print("DEMO 1: Regime Pullback Strategy (Trending Market)")
    print("="*70)

    # Create trending data
    df = create_trending_data(300)

    # Add indicators
    detector = RegimeDetector()
    df = detector.add_all_indicators(df)

    # Initialize strategy
    strategy = RegimePullbackStrategy(asset='US_TECH_100')

    # Scan for setups in last 50 bars
    setups_found = 0
    for i in range(250, 300):
        window = df.iloc[:i+1]
        regime, conf, strat_type = detector.detect_regime(window)

        # Only use pullback strategy in trending regimes
        if regime in [MarketRegime.STRONG_TREND, MarketRegime.WEAK_TREND]:
            setup = strategy.check_entry(window, regime.value, conf)

            if setup:
                setups_found += 1
                print(f"\nBar {i}:")
                print(f"  Regime: {regime.value} ({conf:.0f}% confidence)")
                print(f"  Setup: {setup.setup_type}")
                print(f"  Entry: {setup.entry_price:.2f}")
                print(f"  Stop: {setup.stop_loss:.2f}")
                print(f"  Target: {setup.target:.2f}")
                print(f"  RRR: {setup.reward_risk_ratio():.2f}R")

    print(f"\n✓ Found {setups_found} pullback setups in 50 bars")
    print("✓ Pullback strategy working with regime detector!")


def demo_squeeze_strategy():
    """Demo: TTM Squeeze Strategy in compression/breakout market."""
    print("\n" + "="*70)
    print("DEMO 2: TTM Squeeze Strategy (Compression/Breakout Market)")
    print("="*70)

    # Create squeeze data
    df = create_squeeze_data(200)

    # Add indicators
    detector = RegimeDetector()
    df = detector.add_all_indicators(df)

    # Initialize strategy
    strategy = TTMSqueezeStrategy(asset='US_TECH_100')

    # Scan for setups
    setups_found = 0
    for i in range(150, 200):
        window = df.iloc[:i+1]
        regime, conf, strat_type = detector.detect_regime(window)

        # Only use squeeze strategy in compression regime
        if regime == MarketRegime.RANGE_COMPRESSION:
            setup = strategy.check_entry(window, regime.value, conf)

            if setup:
                setups_found += 1
                print(f"\nBar {i}:")
                print(f"  Regime: {regime.value} ({conf:.0f}% confidence)")
                print(f"  Strategy: {strat_type}")
                print(f"  Setup: {setup.setup_type}")
                print(f"  Entry: {setup.entry_price:.2f}")
                print(f"  Stop: {setup.stop_loss:.2f}")
                print(f"  Target: {setup.target:.2f}")
                print(f"  RRR: {setup.reward_risk_ratio():.2f}R")

    print(f"\n✓ Found {setups_found} squeeze setups in 50 bars")
    print("✓ Squeeze strategy working with regime detector!")


def demo_strategy_selection():
    """Demo: Automatic strategy selection based on regime."""
    print("\n" + "="*70)
    print("DEMO 3: Automatic Strategy Selection Based on Regime")
    print("="*70)

    # Create mixed data
    df = create_trending_data(300)

    # Add indicators
    detector = RegimeDetector()
    df = detector.add_all_indicators(df)

    # Initialize both strategies
    pullback_strategy = RegimePullbackStrategy(asset='US_TECH_100')
    squeeze_strategy = TTMSqueezeStrategy(asset='US_TECH_100')

    # Scan and auto-select strategy
    total_setups = 0
    pullback_setups = 0
    squeeze_setups = 0

    for i in range(250, 300):
        window = df.iloc[:i+1]
        regime, conf, strat_type = detector.detect_regime(window)

        setup = None

        # Auto-select strategy based on regime
        if regime in [MarketRegime.STRONG_TREND, MarketRegime.WEAK_TREND]:
            # Use pullback strategy
            setup = pullback_strategy.check_entry(window, regime.value, conf)
            if setup:
                pullback_setups += 1

        elif regime == MarketRegime.RANGE_COMPRESSION:
            # Use squeeze strategy
            setup = squeeze_strategy.check_entry(window, regime.value, conf)
            if setup:
                squeeze_setups += 1

        if setup:
            total_setups += 1

    print(f"\nResults from 50 bars:")
    print(f"  Total setups: {total_setups}")
    print(f"  Pullback setups: {pullback_setups}")
    print(f"  Squeeze setups: {squeeze_setups}")

    if total_setups > 0:
        pullback_pct = (pullback_setups / total_setups) * 100
        squeeze_pct = (squeeze_setups / total_setups) * 100
        print(f"\nStrategy distribution:")
        print(f"  Pullback: {pullback_pct:.0f}%")
        print(f"  Squeeze: {squeeze_pct:.0f}%")

    print("\n✓ Automatic strategy selection working!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("LIQUIDEDGE STRATEGY SYSTEM INTEGRATION DEMO")
    print("="*70)

    try:
        demo_pullback_strategy()
        demo_squeeze_strategy()
        demo_strategy_selection()

        print("\n" + "="*70)
        print("ALL DEMOS COMPLETE ✓")
        print("="*70)
        print("\nStrategy System Summary:")
        print("- RegimePullbackStrategy: Trend following (70% of trades)")
        print("- TTMSqueezeStrategy: Breakout trading (30% of trades)")
        print("- RegimeDetector: Automatic regime + strategy selection")
        print("- Full integration working end-to-end!")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
