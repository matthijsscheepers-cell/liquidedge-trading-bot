#!/usr/bin/env python3
"""
Regime Detection for Trading Demo

Demonstrates the trading-focused regime detection API that provides:
- Confidence scores (0-100) for setup quality
- Strategy recommendations for each regime
- Priority-based regime detection (safety first)

This is the BRAIN of the trading bot - it decides WHEN and HOW to trade.

Usage:
    python scripts/demo_regime_trading.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.regime import RegimeDetector, MarketRegime, RegimeConfig


def create_sample_data(regime_type='trending', n=300):
    """
    Generate sample data for different market regimes.

    Args:
        regime_type: 'trending', 'ranging', 'squeeze', 'volatile'
        n: Number of bars
    """
    np.random.seed(42)

    if regime_type == 'trending':
        # Strong uptrend with consistent momentum
        trend = np.linspace(100, 150, n)
        noise = np.random.normal(0, 1.5, n)
        close = trend + noise

    elif regime_type == 'ranging':
        # Sideways movement
        base = 100
        noise = np.random.normal(0, 2.0, n)
        close = base + noise

    elif regime_type == 'squeeze':
        # Low volatility compression then breakout
        close = np.ones(n) * 100

        # First half: very low volatility
        close[:n//2] += np.random.normal(0, 0.3, n//2)

        # Second half: explosive breakout
        breakout_trend = np.linspace(0, 20, n - n//2)
        close[n//2:] += breakout_trend + np.random.normal(0, 1.0, n - n//2)

    elif regime_type == 'volatile':
        # High volatility, no clear direction
        close = 100 + np.random.normal(0, 5.0, n).cumsum()

    else:
        close = np.linspace(100, 120, n) + np.random.normal(0, 2, n)

    # Generate high/low
    high = close + np.abs(np.random.uniform(0.3, 1.5, n))
    low = close - np.abs(np.random.uniform(0.3, 1.5, n))

    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })

    return df


def demo_basic_usage():
    """Demo: Basic regime detection workflow."""
    print("\n" + "=" * 70)
    print("1. Basic Regime Detection Workflow")
    print("=" * 70)

    # Create detector
    detector = RegimeDetector()

    # Generate trending data
    df = create_sample_data('trending', n=300)

    print(f"\nData: {len(df)} bars of trending market")

    # Step 1: Add all indicators
    print("\nStep 1: Adding all indicators...")
    df_with_indicators = detector.add_all_indicators(df)

    print(f"  Original columns: {len(df.columns)}")
    print(f"  With indicators: {len(df_with_indicators.columns)}")
    print(f"\n  New indicator columns:")
    new_cols = [col for col in df_with_indicators.columns if col not in df.columns]
    for i, col in enumerate(new_cols[:10], 1):
        print(f"    {i:2d}. {col}")
    if len(new_cols) > 10:
        print(f"    ... and {len(new_cols) - 10} more")

    # Step 2: Detect regime
    print("\nStep 2: Detecting current market regime...")
    regime, confidence, strategy = detector.detect_regime(df_with_indicators)

    print(f"\n  Regime: {regime.value}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Strategy: {strategy}")

    # Interpretation
    print("\n" + "-" * 70)
    print("  INTERPRETATION:")
    if confidence > 70:
        print(f"  ✓ HIGH confidence setup - Consider trading {strategy}")
    elif confidence > 50:
        print(f"  ○ MODERATE confidence - {strategy} with caution")
    elif confidence > 0:
        print(f"  ○ LOW confidence - Wait for better setup")
    else:
        print(f"  ✗ No trade - Avoid trading in {regime.value}")
    print("-" * 70)


def demo_different_regimes():
    """Demo: Detect different market regimes."""
    print("\n" + "=" * 70)
    print("2. Different Market Regimes")
    print("=" * 70)

    detector = RegimeDetector()

    regimes_to_test = [
        ('trending', 'Strong Uptrend'),
        ('ranging', 'Sideways Range'),
        ('squeeze', 'TTM Squeeze Setup'),
        ('volatile', 'High Volatility'),
    ]

    for regime_type, description in regimes_to_test:
        print(f"\n{'-' * 70}")
        print(f"{description} Market")
        print(f"{'-' * 70}")

        df = create_sample_data(regime_type, n=300)
        df_with_indicators = detector.add_all_indicators(df)

        regime, confidence, strategy = detector.detect_regime(df_with_indicators)

        print(f"  Detected Regime: {regime.value}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Strategy: {strategy}")

        # Show key indicators
        current = df_with_indicators.iloc[-1]
        adx_col = [col for col in df_with_indicators.columns if col.startswith('adx_')][0]

        print(f"\n  Key Indicators:")
        print(f"    ADX: {current[adx_col]:.1f}")
        print(f"    ATR%: {current['atr_pct']:.2f}%")
        print(f"    EMA Alignment: {current.get('ema_alignment', 'N/A')}")
        print(f"    Squeeze On: {current['squeeze_on']}")
        if current['squeeze_on']:
            print(f"    Squeeze Duration: {current['squeeze_duration']:.0f} bars")
            print(f"    Compression Score: {current['compression_score']:.1f}")


def demo_confidence_scoring():
    """Demo: How confidence scores work."""
    print("\n" + "=" * 70)
    print("3. Confidence Score Mechanics")
    print("=" * 70)

    print("\nConfidence scores (0-100) indicate setup quality:")
    print("  90-100: Exceptional setup, high probability")
    print("  70-89:  Strong setup, favorable conditions")
    print("  50-69:  Moderate setup, acceptable risk/reward")
    print("  1-49:   Weak setup, poor conditions")
    print("  0:      No trade, avoid trading")

    detector = RegimeDetector()

    # Test squeeze with different durations
    print(f"\n{'-' * 70}")
    print("Example: TTM Squeeze Confidence vs Duration")
    print(f"{'-' * 70}")

    df = create_sample_data('squeeze', n=200)
    df_with_indicators = detector.add_all_indicators(df)

    # Find squeeze periods
    squeeze_bars = df_with_indicators[df_with_indicators['squeeze_on']]

    if len(squeeze_bars) > 0:
        print(f"\nFound {len(squeeze_bars)} bars with squeeze active")

        # Sample a few different durations
        for idx in [len(squeeze_bars)//4, len(squeeze_bars)//2, len(squeeze_bars)-1]:
            if idx < len(squeeze_bars):
                sample_df = df_with_indicators.iloc[:squeeze_bars.index[idx] + 1]
                regime, conf, strat = detector.detect_regime(sample_df)

                if regime == MarketRegime.RANGE_COMPRESSION:
                    duration = sample_df.iloc[-1]['squeeze_duration']
                    compression = sample_df.iloc[-1]['compression_score']

                    print(f"\n  Duration: {duration:.0f} bars")
                    print(f"  Compression: {compression:.1f}")
                    print(f"  Confidence: {conf:.1f}%")
                    print(f"  → {'TRADE' if conf > 60 else 'WAIT'}")


def demo_priority_system():
    """Demo: Priority-based regime detection."""
    print("\n" + "=" * 70)
    print("4. Priority-Based Detection System")
    print("=" * 70)

    print("\nDetection priority (checked in order):")
    print("  1. HIGH_VOLATILITY - Safety check (confidence = 0)")
    print("  2. RANGE_COMPRESSION - TTM Squeeze setups")
    print("  3. STRONG_TREND - High ADX trending")
    print("  4. WEAK_TREND - Lower ADX trending")
    print("  5. NO_TRADE - No clear opportunity")

    print("\nWhy this order?")
    print("  • Safety first: Avoid volatile conditions")
    print("  • Squeeze next: Explosive breakout potential")
    print("  • Trends: Reliable but slower-developing setups")
    print("  • Default: Wait for better conditions")

    detector = RegimeDetector()

    # Create volatile market
    df = create_sample_data('volatile', n=200)
    df_with_indicators = detector.add_all_indicators(df)

    regime, conf, strat = detector.detect_regime(df_with_indicators)

    print(f"\n{'-' * 70}")
    print("Example: High Volatility Detection")
    print(f"{'-' * 70}")
    print(f"  Regime: {regime.value}")
    print(f"  Confidence: {conf}%")
    print(f"  Strategy: {strat}")
    print(f"  → System prioritizes SAFETY and avoids trading")


def demo_strategy_mapping():
    """Demo: Regime to strategy mapping."""
    print("\n" + "=" * 70)
    print("5. Regime → Strategy Mapping")
    print("=" * 70)

    print("\nEach regime maps to a specific trading strategy:")
    print()
    print("  RANGE_COMPRESSION:")
    print("    Strategy: TTM_SQUEEZE or TTM_BREAKOUT")
    print("    Approach: Enter on squeeze release with momentum")
    print("    Risk: Low (tight stop-loss possible)")
    print()
    print("  STRONG_TREND:")
    print("    Strategy: REGIME_PULLBACK")
    print("    Approach: Wait for pullback, enter in trend direction")
    print("    Risk: Moderate (wider stops needed)")
    print()
    print("  WEAK_TREND:")
    print("    Strategy: REGIME_PULLBACK")
    print("    Approach: Same as strong trend, smaller position size")
    print("    Risk: Moderate (less reliable)")
    print()
    print("  HIGH_VOLATILITY:")
    print("    Strategy: NONE")
    print("    Approach: Stand aside, wait for volatility to normalize")
    print("    Risk: High (unpredictable)")
    print()
    print("  NO_TRADE:")
    print("    Strategy: NONE")
    print("    Approach: No clear edge, wait for setup")
    print("    Risk: N/A")


def demo_custom_config():
    """Demo: Custom configuration."""
    print("\n" + "=" * 70)
    print("6. Custom Configuration")
    print("=" * 70)

    # Create custom config
    custom_config = RegimeConfig(
        adx_period=20,                # Longer ADX period
        adx_strong_threshold=30.0,    # Higher threshold for "strong"
        adx_weak_threshold=22.0,      # Higher threshold for "weak"
        min_squeeze_duration=8,       # Wait for longer squeeze
        volatility_high_threshold=1.5  # More lenient volatility filter
    )

    print("\nCustom Configuration:")
    print(f"  ADX Period: {custom_config.adx_period}")
    print(f"  Strong Trend Threshold: {custom_config.adx_strong_threshold}")
    print(f"  Min Squeeze Duration: {custom_config.min_squeeze_duration}")

    detector_default = RegimeDetector()
    detector_custom = RegimeDetector(config=custom_config)

    df = create_sample_data('trending', n=300)

    print("\n" + "-" * 70)
    print("Comparing Default vs Custom Config:")
    print("-" * 70)

    # Default config
    df_default = detector_default.add_all_indicators(df)
    regime_def, conf_def, strat_def = detector_default.detect_regime(df_default)

    print(f"\nDefault Config:")
    print(f"  Regime: {regime_def.value}")
    print(f"  Confidence: {conf_def:.1f}%")

    # Custom config
    df_custom = detector_custom.add_all_indicators(df)
    regime_cust, conf_cust, strat_cust = detector_custom.detect_regime(df_custom)

    print(f"\nCustom Config:")
    print(f"  Regime: {regime_cust.value}")
    print(f"  Confidence: {conf_cust:.1f}%")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" Regime Detection for Trading - Comprehensive Demo")
    print("=" * 70)
    print("\nThis demo shows the TRADING-FOCUSED regime detection API.")
    print("Unlike historical analysis, this API provides:")
    print("  • Real-time regime classification")
    print("  • Confidence scores for position sizing")
    print("  • Strategy recommendations")
    print("  • Priority-based decision making")

    demo_basic_usage()
    demo_different_regimes()
    demo_confidence_scoring()
    demo_priority_system()
    demo_strategy_mapping()
    demo_custom_config()

    print("\n" + "=" * 70)
    print("✓ All demos completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use add_all_indicators() to calculate everything at once")
    print("  2. Use detect_regime() to get (regime, confidence, strategy)")
    print("  3. Confidence > 70 = high quality setup")
    print("  4. Each regime maps to a specific trading strategy")
    print("  5. System prioritizes safety (volatility checks first)")
    print("\nNext Steps:")
    print("  • Implement strategy executors for each regime")
    print("  • Add position sizing based on confidence")
    print("  • Backtest strategies on historical data")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
