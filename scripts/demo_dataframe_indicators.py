#!/usr/bin/env python3
"""
DataFrame-Based Indicator Demo

Demonstrates the new DataFrame-based convenience functions for trend indicators.
These functions make it easier to work with OHLC DataFrames.

Usage:
    python scripts/demo_dataframe_indicators.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.indicators import (
    calculate_adx_df,
    calculate_ema_slope,
    calculate_multiple_emas,
    check_ema_alignment,
    add_trend_indicators,
)


def create_sample_data(n=200):
    """Generate sample OHLC data."""
    np.random.seed(42)

    # Create trending data
    trend = np.linspace(100, 140, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise

    # Generate high/low
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)

    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })

    return df


def demo_adx_df():
    """Demo: Calculate ADX from DataFrame."""
    print("\n" + "=" * 70)
    print("1. Calculate ADX from DataFrame")
    print("=" * 70)

    df = create_sample_data(100)
    adx_df = calculate_adx_df(df, period=14)

    print(f"\nInput DataFrame shape: {df.shape}")
    print(f"ADX DataFrame shape: {adx_df.shape}")
    print(f"Columns: {list(adx_df.columns)}")

    print("\nLast 5 values:")
    print(adx_df.tail())

    # Analysis
    current_adx = adx_df['adx'].iloc[-1]
    current_plus_di = adx_df['plus_di'].iloc[-1]
    current_minus_di = adx_df['minus_di'].iloc[-1]

    print(f"\nCurrent ADX: {current_adx:.2f}")
    print(f"Current +DI: {current_plus_di:.2f}")
    print(f"Current -DI: {current_minus_di:.2f}")

    if current_adx > 25:
        trend = "BULLISH" if current_plus_di > current_minus_di else "BEARISH"
        print(f"✓ Strong {trend} trend detected (ADX > 25)")
    elif current_adx > 20:
        print("✓ Moderate trend detected (ADX > 20)")
    else:
        print("○ Weak/no trend (ADX < 20)")


def demo_multiple_emas():
    """Demo: Calculate multiple EMAs."""
    print("\n" + "=" * 70)
    print("2. Calculate Multiple EMAs")
    print("=" * 70)

    df = create_sample_data(100)
    emas = calculate_multiple_emas(df, periods=[20, 50, 200])

    print(f"\nEMA DataFrame columns: {list(emas.columns)}")
    print("\nLast 5 values:")
    print(emas.tail())

    # Check current values
    current_close = df['close'].iloc[-1]
    print(f"\nCurrent close: ${current_close:.2f}")
    print(f"EMA 20: ${emas['ema_20'].iloc[-1]:.2f}")
    print(f"EMA 50: ${emas['ema_50'].iloc[-1]:.2f}")
    print(f"EMA 200: ${emas['ema_200'].iloc[-1]:.2f}")


def demo_ema_alignment():
    """Demo: Check EMA alignment."""
    print("\n" + "=" * 70)
    print("3. Check EMA Alignment")
    print("=" * 70)

    df = create_sample_data(200)
    alignment = check_ema_alignment(df, periods=[20, 50, 200])

    # Count alignments
    bullish_count = (alignment == 'bullish').sum()
    bearish_count = (alignment == 'bearish').sum()
    mixed_count = (alignment == 'mixed').sum()

    total = len(alignment)

    print(f"\nAlignment Distribution ({total} bars):")
    print(f"  Bullish: {bullish_count:3d} bars ({bullish_count/total*100:.1f}%)")
    print(f"  Bearish: {bearish_count:3d} bars ({bearish_count/total*100:.1f}%)")
    print(f"  Mixed:   {mixed_count:3d} bars ({mixed_count/total*100:.1f}%)")

    print(f"\nCurrent alignment: {alignment.iloc[-1].upper()}")

    # Find longest bullish streak
    bullish_mask = alignment == 'bullish'
    groups = (bullish_mask != bullish_mask.shift()).cumsum()
    streaks = bullish_mask.groupby(groups).sum()

    if streaks.max() > 0:
        longest_streak = streaks.max()
        print(f"Longest bullish streak: {longest_streak} bars")


def demo_ema_slope():
    """Demo: Calculate EMA slope."""
    print("\n" + "=" * 70)
    print("4. Calculate EMA Slope (Momentum)")
    print("=" * 70)

    df = create_sample_data(100)
    slope = calculate_ema_slope(df, ema_period=20, slope_lookback=20)

    print(f"\nSlope series length: {len(slope)}")
    print("\nLast 10 slope values:")
    print(slope.tail(10))

    current_slope = slope.iloc[-1]
    avg_slope = slope.mean()

    print(f"\nCurrent slope: {current_slope:.4f}%")
    print(f"Average slope: {avg_slope:.4f}%")

    if current_slope > 0.1:
        print("✓ Strong upward momentum (slope > 0.1%)")
    elif current_slope > 0:
        print("✓ Weak upward momentum")
    elif current_slope < -0.1:
        print("✗ Strong downward momentum (slope < -0.1%)")
    else:
        print("○ Sideways (slope ≈ 0)")


def demo_all_indicators():
    """Demo: Add all trend indicators at once."""
    print("\n" + "=" * 70)
    print("5. Add All Trend Indicators (One-Liner)")
    print("=" * 70)

    df = create_sample_data(100)

    print(f"\nOriginal DataFrame columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Add all indicators
    df_enriched = add_trend_indicators(df)

    print(f"\nEnriched DataFrame columns: {list(df_enriched.columns)}")
    print(f"Shape: {df_enriched.shape}")

    print("\nLast row:")
    print(df_enriched.iloc[-1])

    # Quick analysis
    current = df_enriched.iloc[-1]

    print("\n" + "-" * 70)
    print("TREND ANALYSIS")
    print("-" * 70)
    print(f"ADX:         {current['adx']:.2f}")
    print(f"+DI:         {current['plus_di']:.2f}")
    print(f"-DI:         {current['minus_di']:.2f}")
    print(f"EMA 20:      ${current['ema_20']:.2f}")
    print(f"EMA 50:      ${current['ema_50']:.2f}")
    print(f"EMA 200:     ${current['ema_200']:.2f}")
    print(f"Alignment:   {current['ema_alignment'].upper()}")
    print("-" * 70)


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" DataFrame-Based Trend Indicators Demo")
    print("=" * 70)
    print("\nThese convenience functions make it easy to work with OHLC DataFrames.")
    print("They wrap the underlying Series-based functions used by the regime detector.")

    demo_adx_df()
    demo_multiple_emas()
    demo_ema_alignment()
    demo_ema_slope()
    demo_all_indicators()

    print("\n" + "=" * 70)
    print("✓ All demos completed successfully!")
    print("=" * 70)
    print("\nTip: Use add_trend_indicators(df) to add all indicators in one line!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
