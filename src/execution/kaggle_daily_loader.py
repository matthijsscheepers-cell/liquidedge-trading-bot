"""
Kaggle Daily Gold Data Loader

Loads recent daily gold prices from Kaggle dataset.
Dataset: zkskhurram/gold-price-analysis-last-1-year

Provides:
- Daily OHLCV data (251 days)
- Feb 2025 - Jan 2026
- Pre-calculated MA7, MA30, Daily_Return
"""

import pandas as pd
import kagglehub
import os
from typing import Optional


class KaggleDailyGoldLoader:
    """
    Load and process Kaggle daily gold price dataset.

    Dataset: zkskhurram/gold-price-analysis-last-1-year
    - Daily OHLC data
    - ~1 year of data (251 trading days)
    - Feb 2025 - Jan 2026
    """

    def __init__(self):
        """Initialize loader and download dataset if needed."""
        self.dataset_path = None
        self.df = None

    def download_dataset(self) -> str:
        """
        Download Kaggle dataset to cache.

        Returns:
            Path to downloaded dataset
        """
        print("Downloading Kaggle daily gold dataset...")
        path = kagglehub.dataset_download("zkskhurram/gold-price-analysis-last-1-year")
        self.dataset_path = path
        print(f"✓ Dataset cached at: {path}")
        return path

    def load_data(self) -> pd.DataFrame:
        """
        Load daily gold prices.

        Returns:
            DataFrame with OHLCV data and technical indicators

        Columns:
            - Date (index): Trading date
            - open, high, low, close: OHLC prices
            - volume: Trading volume
            - MA7, MA30: Moving averages
            - Daily_Return: Daily return percentage

        Example:
            >>> loader = KaggleDailyGoldLoader()
            >>> df = loader.load_data()
            >>> print(df.tail())
        """
        if self.dataset_path is None:
            self.download_dataset()

        # Load CSV
        csv_path = os.path.join(self.dataset_path, 'gold_rates_1y.csv')
        df = pd.read_csv(csv_path)

        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Rename columns to lowercase for consistency
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }, inplace=True)

        # Keep only essential columns
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        optional_cols = ['MA7', 'MA30', 'Daily_Return']

        for col in optional_cols:
            if col in df.columns:
                columns_to_keep.append(col)

        df = df[columns_to_keep]

        self.df = df
        return df

    def get_ohlc(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get OHLC data for date range.

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLC data

        Example:
            >>> loader = KaggleDailyGoldLoader()
            >>> df = loader.get_ohlc(
            ...     start_date='2025-06-01',
            ...     end_date='2026-01-31'
            ... )
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def summary_stats(self) -> dict:
        """
        Get summary statistics for the dataset.

        Returns:
            Dict with statistics
        """
        if self.df is None:
            self.load_data()

        return {
            'start_date': self.df.index[0],
            'end_date': self.df.index[-1],
            'total_days': len(self.df),
            'min_price': self.df['close'].min(),
            'max_price': self.df['close'].max(),
            'mean_price': self.df['close'].mean(),
            'std_price': self.df['close'].std(),
            'min_volume': self.df['volume'].min(),
            'max_volume': self.df['volume'].max(),
            'mean_volume': self.df['volume'].mean(),
        }

    def compare_with_capital_com(self, capital_df: pd.DataFrame):
        """
        Compare Kaggle daily data with Capital.com 15m data.

        Args:
            capital_df: DataFrame from Capital.com API

        Prints comparison table.
        """
        print("=" * 70)
        print("DATA SOURCE COMPARISON")
        print("=" * 70)
        print()

        if self.df is None:
            self.load_data()

        # Resample Capital.com to daily for comparison
        capital_daily = capital_df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        print(f"Kaggle Daily Data:")
        print(f"  Bars: {len(self.df)}")
        print(f"  Date range: {self.df.index[0]} to {self.df.index[-1]}")
        print(f"  Timeframe: 1D")
        print(f"  Price range: ${self.df['close'].min():.2f} - ${self.df['close'].max():.2f}")
        print()

        print(f"Capital.com Data (15m → Daily):")
        print(f"  Bars (15m): {len(capital_df)}")
        print(f"  Bars (1D): {len(capital_daily)}")
        print(f"  Date range: {capital_df.index[0]} to {capital_df.index[-1]}")
        print(f"  Timeframe: 15m")
        print(f"  Price range: ${capital_df['close'].min():.2f} - ${capital_df['close'].max():.2f}")
        print()

        # Find overlapping dates
        overlap = self.df.index.intersection(capital_daily.index)
        if len(overlap) > 0:
            print(f"Overlapping days: {len(overlap)}")
            print()
            print("Price comparison on overlapping days:")
            print("(showing last 5 overlapping days)")
            print()

            comparison = pd.DataFrame({
                'Kaggle_Close': self.df.loc[overlap, 'close'],
                'CapitalCom_Close': capital_daily.loc[overlap, 'close'],
            })
            comparison['Difference'] = comparison['Kaggle_Close'] - comparison['CapitalCom_Close']
            comparison['Diff_%'] = (comparison['Difference'] / comparison['Kaggle_Close']) * 100

            print(comparison.tail())
        else:
            print("No overlapping dates found")


def compare_all_data_sources():
    """
    Compare all three data sources for gold trading.
    """
    print("=" * 70)
    print("GOLD DATA SOURCES COMPARISON")
    print("=" * 70)
    print()

    sources = [
        {
            'name': 'Kaggle Monthly (tunguz)',
            'timeframe': 'Monthly',
            'duration': '70 years (1950-2020)',
            'bars': '847',
            'granularity': '1 month',
            'ohlcv': 'Close only',
            'pros': [
                'Very long history',
                'Free and cached',
                'Good for regime analysis'
            ],
            'cons': [
                'Monthly only',
                'Outdated (2020)',
                'No OHLCV'
            ],
            'use_for': 'Long-term trend analysis, regime validation'
        },
        {
            'name': 'Kaggle Daily (zkskhurram)',
            'timeframe': 'Daily',
            'duration': '1 year (Feb 2025 - Jan 2026)',
            'bars': '251',
            'granularity': '1 day',
            'ohlcv': 'Full OHLCV',
            'pros': [
                'Daily OHLC data',
                'Very recent (2026)',
                'Includes volume',
                'Pre-calculated indicators'
            ],
            'cons': [
                'Only 1 year history',
                'Daily only (no intraday)',
                'Limited to gold'
            ],
            'use_for': 'Daily backtesting, trend analysis, strategy validation'
        },
        {
            'name': 'Capital.com API',
            'timeframe': '15m, 1H, 1D',
            'duration': '~7 days (500 bars @ 15m)',
            'bars': '500 (per request)',
            'granularity': '15 minutes',
            'ohlcv': 'Full OHLCV',
            'pros': [
                'Intraday data (15m)',
                'Real-time current',
                'Multiple assets',
                'Live trading ready'
            ],
            'cons': [
                'Limited history (500 bars)',
                'Requires API key',
                'Rate limits'
            ],
            'use_for': '15m intraday trading (YOUR STRATEGY)'
        },
    ]

    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['name']}")
        print(f"   Timeframe: {source['timeframe']}")
        print(f"   Duration: {source['duration']}")
        print(f"   Bars: {source['bars']}")
        print(f"   Granularity: {source['granularity']}")
        print(f"   OHLCV: {source['ohlcv']}")
        print()
        print("   Pros:")
        for pro in source['pros']:
            print(f"     ✓ {pro}")
        print()
        print("   Cons:")
        for con in source['cons']:
            print(f"     ✗ {con}")
        print()
        print(f"   Best for: {source['use_for']}")
        print()
        print("-" * 70)
        print()

    print("RECOMMENDATION:")
    print()
    print("  PRIMARY: Capital.com API")
    print("    → Your 15m intraday strategy requires intraday data")
    print()
    print("  SECONDARY: Kaggle Daily (zkskhurram)")
    print("    → Test strategy on daily timeframe")
    print("    → Validate approach with more history")
    print()
    print("  TERTIARY: Kaggle Monthly (tunguz)")
    print("    → Long-term regime analysis")
    print("    → Historical context")
    print()


if __name__ == "__main__":
    # Example usage
    loader = KaggleDailyGoldLoader()

    # Load data
    df = loader.load_data()
    print(f"Loaded {len(df)} days of gold prices")
    print()

    # Summary stats
    stats = loader.summary_stats()
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Show recent data
    print("Most recent 10 days:")
    print(df[['open', 'high', 'low', 'close', 'volume']].tail(10))
    print()

    # Compare all sources
    compare_all_data_sources()
