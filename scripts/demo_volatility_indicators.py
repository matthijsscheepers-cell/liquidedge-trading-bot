#!/usr/bin/env python3
"""
DataFrame-Based Volatility Indicator Demo

Demonstrates the DataFrame-based convenience functions for volatility indicators.
These functions make it easier to work with OHLC DataFrames for volatility analysis.

Usage:
    python scripts/demo_volatility_indicators.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.indicators import (
    calculate_atr_df,
    calculate_atr_percentage,
    calculate_atr_percentile,
    calculate_bollinger_bands_df,
    calculate_bollinger_width,
    calculate_keltner_channels_df,
    calculate_keltner_width,
    add_volatility_indicators,
)


def create_sample_data(n=200, volatility_regime='normal'):
    """Generate sample OHLC data with different volatility regimes.

    Args:
        n: Number of bars
        volatility_regime: 'low', 'normal', 'high', or 'mixed'
    """
    np.random.seed(42)

    # Create base trend
    trend = np.linspace(100, 120, n)

    # Apply volatility regime
    if volatility_regime == 'low':
        noise = np.random.normal(0, 0.5, n)
        range_mult = 0.3
    elif volatility_regime == 'high':
        noise = np.random.normal(0, 3.0, n)
        range_mult = 2.0
    elif volatility_regime == 'mixed':
        # Low volatility first half, high volatility second half
        noise1 = np.random.normal(0, 0.5, n//2)
        noise2 = np.random.normal(0, 3.0, n - n//2)
        noise = np.concatenate([noise1, noise2])
        range_mult = np.concatenate([
            np.ones(n//2) * 0.3,
            np.ones(n - n//2) * 2.0
        ])
    else:  # normal
        noise = np.random.normal(0, 1.5, n)
        range_mult = 1.0

    close = trend + noise

    # Generate high/low
    if isinstance(range_mult, np.ndarray):
        high = close + np.random.uniform(0.5, 2.0, n) * range_mult
        low = close - np.random.uniform(0.5, 2.0, n) * range_mult
    else:
        high = close + np.random.uniform(0.5, 2.0, n) * range_mult
        low = close - np.random.uniform(0.5, 2.0, n) * range_mult

    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    })

    return df


def demo_atr():
    """Demo: Calculate ATR and ATR percentage."""
    print("\n" + "=" * 70)
    print("1. Average True Range (ATR) & ATR Percentage")
    print("=" * 70)

    df = create_sample_data(100, volatility_regime='mixed')

    # Calculate ATR
    atr = calculate_atr_df(df, period=14)
    atr_pct = calculate_atr_percentage(df, period=14)

    print(f"\nData points: {len(df)}")
    print(f"ATR series name: {atr.name}")
    print(f"ATR% series name: {atr_pct.name}")

    # Show first half (low volatility) vs second half (high volatility)
    mid_point = len(df) // 2

    print("\n" + "-" * 70)
    print("Low Volatility Period (first half):")
    print("-" * 70)
    print(f"Average ATR: ${atr.iloc[:mid_point].mean():.2f}")
    print(f"Average ATR%: {atr_pct.iloc[:mid_point].mean():.2f}%")

    print("\n" + "-" * 70)
    print("High Volatility Period (second half):")
    print("-" * 70)
    print(f"Average ATR: ${atr.iloc[mid_point:].mean():.2f}")
    print(f"Average ATR%: {atr_pct.iloc[mid_point:].mean():.2f}%")

    print("\n" + "-" * 70)
    print("Current Values:")
    print("-" * 70)
    print(f"Current ATR: ${atr.iloc[-1]:.2f}")
    print(f"Current ATR%: {atr_pct.iloc[-1]:.2f}%")
    print(f"Current price: ${df['close'].iloc[-1]:.2f}")

    # Volatility assessment
    if atr_pct.iloc[-1] > 2.5:
        print("✓ HIGH volatility environment")
    elif atr_pct.iloc[-1] > 1.5:
        print("○ NORMAL volatility environment")
    else:
        print("✓ LOW volatility environment")


def demo_atr_percentile():
    """Demo: Calculate ATR percentile."""
    print("\n" + "=" * 70)
    print("2. ATR Percentile Ranking")
    print("=" * 70)

    df = create_sample_data(300, volatility_regime='mixed')

    # Calculate ATR percentile
    atr_percentile = calculate_atr_percentile(
        df,
        atr_period=14,
        percentile_period=100
    )

    print(f"\nCalculating ATR percentile over 100-bar lookback")
    print(f"Series name: {atr_percentile.name}")

    # Show percentile progression
    print("\nATR Percentile Over Time:")
    print("-" * 70)

    quartiles = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    for i, idx in enumerate(quartiles):
        if not pd.isna(atr_percentile.iloc[idx]):
            print(f"Bar {idx:3d}: {atr_percentile.iloc[idx]:5.1f}th percentile")

    current_percentile = atr_percentile.iloc[-1]
    print("\n" + "-" * 70)
    print(f"Current ATR Percentile: {current_percentile:.1f}th")
    print("-" * 70)

    if current_percentile > 80:
        print("✓ Volatility is in top 20% of recent range (HIGH)")
    elif current_percentile > 60:
        print("○ Volatility is above average")
    elif current_percentile > 40:
        print("○ Volatility is near average")
    elif current_percentile > 20:
        print("○ Volatility is below average")
    else:
        print("✓ Volatility is in bottom 20% of recent range (LOW)")


def demo_bollinger_bands():
    """Demo: Bollinger Bands."""
    print("\n" + "=" * 70)
    print("3. Bollinger Bands")
    print("=" * 70)

    df = create_sample_data(100, volatility_regime='high')

    # Calculate Bollinger Bands
    bb = calculate_bollinger_bands_df(df, period=20, std_dev=2.0)
    bb_width = calculate_bollinger_width(df, period=20, std_dev=2.0)

    print(f"\nBollinger Bands columns: {list(bb.columns)}")
    print(f"BB Width series name: {bb_width.name}")

    print("\nLast 5 values:")
    print(bb.tail())

    # Current analysis
    current_close = df['close'].iloc[-1]
    current_upper = bb['bb_upper'].iloc[-1]
    current_middle = bb['bb_middle'].iloc[-1]
    current_lower = bb['bb_lower'].iloc[-1]
    current_width = bb_width.iloc[-1]

    print("\n" + "-" * 70)
    print("Current Bollinger Band Status:")
    print("-" * 70)
    print(f"Upper Band:  ${current_upper:.2f}")
    print(f"Middle (MA): ${current_middle:.2f}")
    print(f"Lower Band:  ${current_lower:.2f}")
    print(f"Current:     ${current_close:.2f}")
    print(f"BB Width:    {current_width:.4f}")

    # Position within bands
    band_range = current_upper - current_lower
    position = (current_close - current_lower) / band_range if band_range > 0 else 0.5

    print(f"\nPrice position: {position*100:.1f}% of band range")

    if position > 0.95:
        print("✓ Price near UPPER band (potential overbought)")
    elif position > 0.75:
        print("○ Price in upper half of bands")
    elif position > 0.25:
        print("○ Price in middle of bands")
    elif position > 0.05:
        print("○ Price in lower half of bands")
    else:
        print("✓ Price near LOWER band (potential oversold)")

    # Width analysis
    avg_width = bb_width.mean()
    if current_width < avg_width * 0.7:
        print(f"✓ BB SQUEEZE detected (width {current_width:.4f} < avg {avg_width:.4f})")
    elif current_width > avg_width * 1.3:
        print(f"✓ BB EXPANSION detected (width {current_width:.4f} > avg {avg_width:.4f})")


def demo_keltner_channels():
    """Demo: Keltner Channels."""
    print("\n" + "=" * 70)
    print("4. Keltner Channels")
    print("=" * 70)

    df = create_sample_data(100, volatility_regime='normal')

    # Calculate Keltner Channels
    kc = calculate_keltner_channels_df(
        df,
        ema_period=20,
        atr_period=20,
        atr_multiplier=1.5
    )
    kc_width = calculate_keltner_width(
        df,
        ema_period=20,
        atr_period=20,
        atr_multiplier=1.5
    )

    print(f"\nKeltner Channels columns: {list(kc.columns)}")
    print(f"KC Width series name: {kc_width.name}")

    print("\nLast 5 values:")
    print(kc.tail())

    # Current analysis
    current_close = df['close'].iloc[-1]
    current_upper = kc['kc_upper'].iloc[-1]
    current_middle = kc['kc_middle'].iloc[-1]
    current_lower = kc['kc_lower'].iloc[-1]
    current_width = kc_width.iloc[-1]

    print("\n" + "-" * 70)
    print("Current Keltner Channel Status:")
    print("-" * 70)
    print(f"Upper Channel: ${current_upper:.2f}")
    print(f"Middle (EMA):  ${current_middle:.2f}")
    print(f"Lower Channel: ${current_lower:.2f}")
    print(f"Current:       ${current_close:.2f}")
    print(f"KC Width:      {current_width:.4f}")

    # Position within channels
    if current_close > current_upper:
        print("✓ Price ABOVE upper channel (strong uptrend)")
    elif current_close < current_lower:
        print("✓ Price BELOW lower channel (strong downtrend)")
    else:
        print("○ Price within channels (normal range)")


def demo_bb_kc_squeeze():
    """Demo: Detect squeeze using BB and KC together."""
    print("\n" + "=" * 70)
    print("5. TTM Squeeze Detection (BB inside KC)")
    print("=" * 70)

    df = create_sample_data(150, volatility_regime='mixed')

    # Calculate both indicators
    bb = calculate_bollinger_bands_df(df, period=20, std_dev=2.0)
    kc = calculate_keltner_channels_df(df, ema_period=20, atr_period=20, atr_multiplier=1.5)

    # Detect squeeze: BB inside KC
    squeeze = (bb['bb_upper'] < kc['kc_upper']) & (bb['bb_lower'] > kc['kc_lower'])

    print(f"\nTotal bars analyzed: {len(df)}")
    print(f"Squeeze bars: {squeeze.sum()}")
    print(f"Squeeze percentage: {squeeze.sum()/len(df)*100:.1f}%")

    # Find squeeze periods
    squeeze_changes = squeeze.astype(int).diff()
    squeeze_starts = squeeze_changes[squeeze_changes == 1].index
    squeeze_ends = squeeze_changes[squeeze_changes == -1].index

    print(f"\nNumber of squeeze periods: {len(squeeze_starts)}")

    if len(squeeze_starts) > 0:
        print("\nSqueeze Periods:")
        print("-" * 70)
        for i, start in enumerate(squeeze_starts[:5]):  # Show first 5
            if i < len(squeeze_ends):
                end = squeeze_ends[i]
                duration = end - start
                print(f"  Period {i+1}: Bars {start}-{end} (duration: {duration} bars)")
            else:
                print(f"  Period {i+1}: Bars {start}-present (ongoing)")

    # Current status
    current_squeeze = squeeze.iloc[-1]
    print("\n" + "-" * 70)
    if current_squeeze:
        print("✓ SQUEEZE IS ON - Volatility contracting, breakout imminent")
    else:
        print("○ Squeeze is off - Normal volatility")


def demo_all_indicators():
    """Demo: Add all volatility indicators at once."""
    print("\n" + "=" * 70)
    print("6. Add All Volatility Indicators (One-Liner)")
    print("=" * 70)

    df = create_sample_data(100)

    print(f"\nOriginal DataFrame columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Add all indicators
    df_enriched = add_volatility_indicators(df)

    print(f"\nEnriched DataFrame columns: {list(df_enriched.columns)}")
    print(f"Shape: {df_enriched.shape}")

    print("\nLast row (all indicators):")
    print("-" * 70)
    print(df_enriched.iloc[-1])

    # Quick volatility report
    current = df_enriched.iloc[-1]

    print("\n" + "=" * 70)
    print(" VOLATILITY REPORT")
    print("=" * 70)
    print(f"\nPrice: ${current['close']:.2f}")
    print(f"\nATR Metrics:")
    print(f"  ATR (14):       ${current['atr_14']:.2f}")
    print(f"  ATR%:           {current['atr_pct_14']:.2f}%")

    print(f"\nBollinger Bands (20, 2.0):")
    print(f"  Upper:  ${current['bb_upper']:.2f}")
    print(f"  Middle: ${current['bb_middle']:.2f}")
    print(f"  Lower:  ${current['bb_lower']:.2f}")
    print(f"  Width:  {current['bb_width']:.4f}")

    print(f"\nKeltner Channels (20, 20, 1.5):")
    print(f"  Upper:  ${current['kc_upper']:.2f}")
    print(f"  Middle: ${current['kc_middle']:.2f}")
    print(f"  Lower:  ${current['kc_lower']:.2f}")

    # Calculate KC width manually (not included in add_volatility_indicators by default)
    kc_width = (current['kc_upper'] - current['kc_lower']) / current['kc_middle']
    print(f"  Width:  {kc_width:.4f}")

    # Squeeze detection
    bb_in_kc = (current['bb_upper'] < current['kc_upper'] and
                current['bb_lower'] > current['kc_lower'])

    print(f"\nSqueeze Status: {'ON ✓' if bb_in_kc else 'OFF ○'}")

    print("\nNote: ATR percentile requires long historical data (252 bars)")
    print("      and is calculated separately using calculate_atr_percentile()")
    print("=" * 70)


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" DataFrame-Based Volatility Indicators Demo")
    print("=" * 70)
    print("\nThese convenience functions make it easy to analyze volatility")
    print("using OHLC DataFrames.")

    demo_atr()
    demo_atr_percentile()
    demo_bollinger_bands()
    demo_keltner_channels()
    demo_bb_kc_squeeze()
    demo_all_indicators()

    print("\n" + "=" * 70)
    print("✓ All demos completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Use add_volatility_indicators(df) for a complete volatility suite")
    print("  • ATR percentage normalizes volatility across different price levels")
    print("  • ATR percentile shows relative volatility (high/low vs history)")
    print("  • Bollinger Bands show price extremes and squeeze/expansion")
    print("  • Keltner Channels provide ATR-based volatility bands")
    print("  • TTM Squeeze = BB inside KC (volatility contraction)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
