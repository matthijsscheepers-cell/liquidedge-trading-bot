"""
Kaggle Gold Dataset Loader

Loads historical gold prices from Kaggle dataset (tunguz/gold-prices).
Useful for long-term backtesting on daily/monthly timeframes.

Note: This dataset provides monthly data (1950-2020), which is NOT suitable
for intraday 15m trading. Use Capital.com API for intraday data.
"""

import pandas as pd
import kagglehub
import os
from typing import Optional


class KaggleGoldLoader:
    """
    Load and process Kaggle gold price dataset.

    Dataset: tunguz/gold-prices
    - Monthly data: 1950-2020 (847 months)
    - Annual data: 1950-2019 (70 years)
    """

    def __init__(self):
        """Initialize loader and download dataset if needed."""
        self.dataset_path = None
        self.monthly_df = None
        self.annual_df = None

    def download_dataset(self) -> str:
        """
        Download Kaggle dataset to cache.

        Returns:
            Path to downloaded dataset
        """
        print("Downloading Kaggle gold dataset...")
        path = kagglehub.dataset_download("tunguz/gold-prices")
        self.dataset_path = path
        print(f"✓ Dataset cached at: {path}")
        return path

    def load_monthly_data(self) -> pd.DataFrame:
        """
        Load monthly gold prices (1950-2020).

        Returns:
            DataFrame with columns: Date (index), Price

        Example:
            >>> loader = KaggleGoldLoader()
            >>> df = loader.load_monthly_data()
            >>> print(df.head())
        """
        if self.dataset_path is None:
            self.download_dataset()

        # Load CSV
        csv_path = os.path.join(self.dataset_path, 'monthly_csv.csv')
        df = pd.read_csv(csv_path)

        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
        df.set_index('Date', inplace=True)

        # Rename for consistency
        df.rename(columns={'Price': 'close'}, inplace=True)

        self.monthly_df = df
        return df

    def load_annual_data(self) -> pd.DataFrame:
        """
        Load annual gold prices (1950-2019).

        Returns:
            DataFrame with columns: Date (index), Price
        """
        if self.dataset_path is None:
            self.download_dataset()

        # Load CSV
        csv_path = os.path.join(self.dataset_path, 'annual_csv.csv')
        df = pd.read_csv(csv_path)

        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m')
        df.set_index('Date', inplace=True)

        # Rename for consistency
        df.rename(columns={'Price': 'close'}, inplace=True)

        self.annual_df = df
        return df

    def create_ohlc_from_monthly(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create synthetic OHLC data from monthly prices.

        Since we only have close prices, we'll create synthetic OHLC:
        - Open: Previous close
        - High: Close * 1.02 (assume 2% intramonth high)
        - Low: Close * 0.98 (assume 2% intramonth low)
        - Close: Actual price
        - Volume: Set to 1000 (placeholder)

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLC columns

        Example:
            >>> loader = KaggleGoldLoader()
            >>> df = loader.create_ohlc_from_monthly(
            ...     start_date='2010-01-01',
            ...     end_date='2020-12-31'
            ... )
        """
        if self.monthly_df is None:
            self.load_monthly_data()

        df = self.monthly_df.copy()

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Create OHLC
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.02  # Synthetic 2% high
        df['low'] = df['close'] * 0.98   # Synthetic 2% low
        df['volume'] = 1000              # Placeholder

        # First bar: use close as open
        df.loc[df.index[0], 'open'] = df.loc[df.index[0], 'close']

        # Reorder columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def resample_to_daily(self, interpolation: str = 'linear') -> pd.DataFrame:
        """
        Resample monthly data to approximate daily prices using interpolation.

        WARNING: This creates SYNTHETIC daily data from monthly prices.
        Use only for testing regime detection logic, NOT for actual backtesting.

        Args:
            interpolation: Method ('linear', 'quadratic', 'cubic')

        Returns:
            DataFrame with synthetic daily OHLC
        """
        if self.monthly_df is None:
            self.load_monthly_data()

        # Resample to daily and interpolate
        daily = self.monthly_df.resample('D').interpolate(method=interpolation)

        # Create synthetic OHLC
        daily['open'] = daily['close'].shift(1)
        daily['high'] = daily['close'] * 1.005  # 0.5% daily high
        daily['low'] = daily['close'] * 0.995   # 0.5% daily low
        daily['volume'] = 1000

        # First bar
        daily.loc[daily.index[0], 'open'] = daily.loc[daily.index[0], 'close']

        # Reorder
        daily = daily[['open', 'high', 'low', 'close', 'volume']]

        return daily

    def get_price_at_date(self, date: str) -> Optional[float]:
        """
        Get gold price at specific date (monthly granularity).

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Price or None if not found
        """
        if self.monthly_df is None:
            self.load_monthly_data()

        target_date = pd.to_datetime(date)

        # Find closest month
        closest_idx = self.monthly_df.index.get_indexer([target_date], method='nearest')[0]
        return self.monthly_df.iloc[closest_idx]['close']

    def summary_stats(self) -> dict:
        """
        Get summary statistics for the dataset.

        Returns:
            Dict with statistics
        """
        if self.monthly_df is None:
            self.load_monthly_data()

        return {
            'start_date': self.monthly_df.index[0],
            'end_date': self.monthly_df.index[-1],
            'total_months': len(self.monthly_df),
            'min_price': self.monthly_df['close'].min(),
            'max_price': self.monthly_df['close'].max(),
            'mean_price': self.monthly_df['close'].mean(),
            'std_price': self.monthly_df['close'].std(),
        }


def compare_data_sources():
    """
    Compare Kaggle data vs Capital.com data.

    This helps understand which data source to use for different strategies.
    """
    print("=" * 70)
    print("DATA SOURCE COMPARISON")
    print("=" * 70)
    print()

    print("KAGGLE DATASET (tunguz/gold-prices)")
    print("  Timeframe: Monthly (1950-2020)")
    print("  Duration: 70 years")
    print("  Granularity: 1 month")
    print("  Pros:")
    print("    + Very long history (70 years)")
    print("    + Free and always available")
    print("    + Good for regime analysis")
    print("  Cons:")
    print("    - Monthly only (no intraday)")
    print("    - Last update: 2020")
    print("    - No volume data")
    print("  Use for:")
    print("    - Long-term trend analysis")
    print("    - Regime detection validation")
    print("    - Educational purposes")
    print()

    print("CAPITAL.COM API")
    print("  Timeframe: 15m, 1H, 1D (configurable)")
    print("  Duration: ~500 bars (limited history)")
    print("  Granularity: 15 minutes minimum")
    print("  Pros:")
    print("    + Intraday data (15m, 1H)")
    print("    + Real-time and current")
    print("    + Includes volume")
    print("    + Multiple assets (GOLD, SILVER, US100, US500)")
    print("  Cons:")
    print("    - Limited history (500 bars)")
    print("    - Requires API credentials")
    print("    - Rate limits")
    print("  Use for:")
    print("    - Intraday backtesting (15m strategy)")
    print("    - Live trading")
    print("    - Recent market conditions")
    print()

    print("RECOMMENDATION:")
    print("  For 15m intraday trading: Use Capital.com API ✓")
    print("  For daily/monthly backtesting: Use Kaggle dataset")
    print("  For regime validation: Use Kaggle dataset")
    print()


if __name__ == "__main__":
    # Example usage
    loader = KaggleGoldLoader()

    # Load monthly data
    monthly = loader.load_monthly_data()
    print(f"Loaded {len(monthly)} months of gold prices")
    print()

    # Summary stats
    stats = loader.summary_stats()
    print("Summary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Create OHLC for last 10 years
    ohlc = loader.create_ohlc_from_monthly(
        start_date='2010-01-01',
        end_date='2020-12-31'
    )
    print(f"Created OHLC data: {len(ohlc)} bars")
    print(ohlc.head())
    print()

    # Compare data sources
    compare_data_sources()
