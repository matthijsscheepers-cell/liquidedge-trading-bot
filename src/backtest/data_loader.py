from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import pandas as pd
from datetime import datetime


class DataLoader(ABC):
    """
    Abstract base class for loading historical data.

    Supports multiple sources:
    - CSV files
    - Broker APIs
    - Yahoo Finance
    - Custom sources
    """

    def __init__(self, data_dir: str = 'data/historical'):
        """
        Initialize data loader

        Args:
            data_dir: Directory for cached data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}

    @abstractmethod
    def load_data(self,
                  asset: str,
                  start_date: datetime,
                  end_date: datetime,
                  timeframe: str = '1H') -> pd.DataFrame:
        """
        Load historical OHLCV data

        Args:
            asset: Instrument identifier
            start_date: Start date
            end_date: End date
            timeframe: Bar interval (1H, 4H, 1D)

        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index: DatetimeIndex
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required structure

        Args:
            df: DataFrame to validate

        Returns:
            True if valid
        """

        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return False

        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return False

        # Check no NaN in OHLC
        if df[['open', 'high', 'low', 'close']].isna().any().any():
            return False

        # Check high >= low
        if (df['high'] < df['low']).any():
            return False

        # Check high >= open, close
        if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
            return False

        # Check low <= open, close
        if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
            return False

        return True

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare data

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Sort by date
        df = df.sort_index()

        # Forward fill volume (some sources have 0)
        df['volume'] = df['volume'].replace(0, pd.NA).ffill().fillna(1000)

        # Remove rows with any NaN in OHLC
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        return df

    def cache_data(self, asset: str, timeframe: str, df: pd.DataFrame):
        """
        Cache data for faster access

        Args:
            asset: Instrument
            timeframe: Timeframe
            df: DataFrame to cache
        """
        cache_key = f"{asset}_{timeframe}"
        self.cache[cache_key] = df.copy()

    def get_cached_data(self, asset: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get cached data if available

        Args:
            asset: Instrument
            timeframe: Timeframe

        Returns:
            Cached DataFrame or None
        """
        cache_key = f"{asset}_{timeframe}"
        return self.cache.get(cache_key)


class CSVDataLoader(DataLoader):
    """
    Load data from CSV files.

    Expected CSV format:
    - Columns: timestamp, open, high, low, close, volume
    - timestamp: ISO format or unix timestamp

    Example:
        loader = CSVDataLoader('data/historical')
        df = loader.load_data('US_TECH_100', start, end, '1H')
    """

    def load_data(self,
                  asset: str,
                  start_date: datetime,
                  end_date: datetime,
                  timeframe: str = '1H') -> pd.DataFrame:
        """Load from CSV file"""

        # Check cache first
        cached = self.get_cached_data(asset, timeframe)
        if cached is not None:
            return cached[(cached.index >= start_date) & (cached.index <= end_date)]

        # Build filename
        filename = f"{asset}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {filepath}\n"
                f"Expected format: timestamp,open,high,low,close,volume"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            raise ValueError("CSV must have 'timestamp' or 'date' column")

        # Rename columns if needed (handle different formats)
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)

        # Clean
        df = self.clean_data(df)

        # Validate
        if not self.validate_dataframe(df):
            raise ValueError("Invalid DataFrame structure")

        # Cache
        self.cache_data(asset, timeframe, df)

        # Filter date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        return df


class CapitalDataLoader(DataLoader):
    """
    Load historical data from Capital.com API.

    Provides access to:
    - CFDs on indices (US100, US500, etc.)
    - Commodities (GOLD, SILVER, OIL, etc.)
    - Forex pairs
    - Cryptocurrencies

    Timeframes supported:
    - 1m, 5m, 15m, 1H, 4H, 1D

    Example:
        >>> loader = CapitalDataLoader(api_key='xxx', password='yyy')
        >>> df = loader.load_data(
        ...     asset='GOLD',
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     timeframe='15m'
        ... )
    """

    def __init__(self, api_key: str = None, password: str = None,
                 identifier: str = None, demo: bool = True):
        """
        Initialize Capital.com data loader.

        Args:
            api_key: Capital.com API key
            password: Capital.com password
            identifier: Capital.com email/login
            demo: Use demo account (default: True)
        """
        super().__init__()

        # Import here to avoid requiring capital.com for other loaders
        from src.execution.capital_connector import CapitalConnector

        self.connector = CapitalConnector({
            'api_key': api_key or 'demo',
            'password': password or 'demo',
            'identifier': identifier or 'demo@demo.com',
            'demo': demo
        })

    def load_data(self,
                 asset: str,
                 start_date: datetime,
                 end_date: datetime,
                 timeframe: str = '15m') -> pd.DataFrame:
        """
        Load data from Capital.com.

        Args:
            asset: Capital.com epic (e.g., 'GOLD', 'SILVER', 'US100', 'US500')
            start_date: Start date
            end_date: End date
            timeframe: Timeframe ('1m', '5m', '15m', '1H', '4H', '1D')

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading {asset} from Capital.com...")

        # Calculate number of bars needed
        days = (end_date - start_date).days

        # Estimate bars based on timeframe (assuming 24-hour markets)
        bars_per_day = {
            '1m': 1440,    # 24 * 60
            '5m': 288,     # 24 * 12
            '15m': 96,     # 24 * 4
            '1H': 24,
            '4H': 6,
            '1D': 1
        }

        daily_bars = bars_per_day.get(timeframe, 24)
        total_bars = min(days * daily_bars, 5000)  # Cap at 5000 (API limit)

        print(f"  Requesting {total_bars} bars ({days} days × {daily_bars} bars/day)")

        try:
            # Connect if not already connected
            if not self.connector.is_connected:
                self.connector.connect()

            # Fetch historical data
            df = self.connector.get_historical_data(
                asset=asset,
                timeframe=timeframe,
                count=total_bars
            )

            print(f"  Received {len(df)} bars from API")

            if df.empty:
                raise ValueError(f"No data returned for {asset}")

            # Filter by date range
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

            # If no data in requested range, use all available data
            if df_filtered.empty:
                print(f"⚠ No data in requested range {start_date} to {end_date}")
                print(f"  Using available data: {df.index[0]} to {df.index[-1]}")
                df_filtered = df

            # Validate and clean
            if not self.validate_dataframe(df_filtered):
                raise ValueError(f"Invalid data for {asset}")

            df_filtered = self.clean_data(df_filtered)

            print(f"✓ Downloaded {len(df_filtered)} bars")

            return df_filtered

        except Exception as e:
            print(f"✗ Failed to load {asset}: {e}")
            raise


class YahooFinanceLoader(DataLoader):
    """
    Load data from Yahoo Finance (free).

    Note: Yahoo Finance uses different symbols:
    - US_TECH_100 → ^NDX or NQ=F
    - US_SPX_500 → ^GSPC or ES=F
    - GOLD → GC=F

    Example:
        loader = YahooFinanceLoader()
        df = loader.load_data('^NDX', start, end, '1D')
    """

    def load_data(self,
                  asset: str,
                  start_date: datetime,
                  end_date: datetime,
                  timeframe: str = '1D') -> pd.DataFrame:
        """Load from Yahoo Finance"""

        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed. Install with: pip install yfinance"
            )

        # Check cache
        cached = self.get_cached_data(asset, timeframe)
        if cached is not None:
            return cached[(cached.index >= start_date) & (cached.index <= end_date)]

        # Map timeframe to yfinance interval
        interval_map = {
            '1H': '1h',
            '4H': '4h',
            '1D': '1d',
            '1W': '1wk'
        }

        interval = interval_map.get(timeframe, '1d')

        # Download
        print(f"Downloading {asset} from Yahoo Finance...")
        ticker = yf.Ticker(asset)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )

        if df.empty:
            raise ValueError(f"No data returned for {asset}")

        # Rename columns
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Clean
        df = self.clean_data(df)

        # Validate
        if not self.validate_dataframe(df):
            raise ValueError("Invalid data from Yahoo Finance")

        # Cache
        self.cache_data(asset, timeframe, df)

        print(f"✓ Downloaded {len(df)} bars")

        return df
