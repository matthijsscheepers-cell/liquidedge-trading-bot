"""
Databento Micro Futures Data Loader

Loads CME Globex micro futures data from Databento DBN files.

Symbols:
- MGC.FUT = Micro Gold futures
- MNQ.FUT = Micro Nasdaq-100 futures
- MES.FUT = Micro E-mini S&P 500 futures
- SIL.FUT = Micro Silver futures

Data: 1-minute OHLCV bars from 2010-2026 (16 years)
"""

import databento as db
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone


class DatabentoMicroFuturesLoader:
    """
    Load and process Databento micro futures data.

    Provides 1-minute OHLCV bars for:
    - Micro Gold (MGC)
    - Micro Nasdaq-100 (MNQ)
    - Micro S&P 500 (MES)
    - Micro Silver (SIL)

    Data range: 2010-09-12 to 2026-01-29
    """

    # Symbol mapping: Databento -> LiquidEdge naming
    SYMBOL_MAP = {
        'MGC.FUT': 'GOLD',
        'MNQ.FUT': 'US100',
        'MES.FUT': 'US500',
        'SIL.FUT': 'SILVER',
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing DBN files
                     (default: GLBX-20260201-B83UY6MA47)
        """
        if data_dir is None:
            # Default to the GLBX folder in project root
            project_root = Path(__file__).parent.parent.parent
            # Try both folder names (with and without copy suffix)
            data_dir = project_root / "GLBX-20260201-B83UY6MA47"
            if not data_dir.exists():
                data_dir = project_root / "GLBX-20260201-B83UY6MA47 (1)"

        self.data_dir = Path(data_dir)
        self.dbn_file = None
        self._find_dbn_file()

        # Cache for loaded data
        self._cache: Dict[str, pd.DataFrame] = {}

    def _find_dbn_file(self):
        """Find the DBN file in the data directory."""
        dbn_files = list(self.data_dir.glob("*.dbn.zst"))

        if not dbn_files:
            raise FileNotFoundError(
                f"No DBN files found in {self.data_dir}. "
                f"Expected .dbn.zst file."
            )

        self.dbn_file = dbn_files[0]
        print(f"Found DBN file: {self.dbn_file.name}")

    def load_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for a single symbol.

        Args:
            symbol: Symbol name (GOLD, US100, US500, SILVER)
                   or Databento symbol (MGC.FUT, MNQ.FUT, etc.)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            resample: Optional resample frequency ('15T', '1H', '1D')

        Returns:
            DataFrame with OHLCV data (1-minute or resampled)

        Example:
            >>> loader = DatabentoMicroFuturesLoader()
            >>> gold_15m = loader.load_symbol('GOLD', resample='15T')
            >>> gold_daily = loader.load_symbol('GOLD', resample='1D')
        """
        # Convert LiquidEdge symbol to Databento symbol
        databento_symbol = self._to_databento_symbol(symbol)

        # Check cache
        cache_key = f"{databento_symbol}_{start_date}_{end_date}_{resample}"
        if cache_key in self._cache:
            print(f"✓ Using cached data for {symbol}")
            return self._cache[cache_key]

        print(f"Loading {symbol} from DBN file...")

        # Read DBN file
        store = db.DBNStore.from_file(self.dbn_file)

        # Convert to pandas DataFrame
        print("  Converting DBN to DataFrame (16M+ rows, may take a minute)...")
        df = store.to_df()
        print(f"  ✓ Loaded {len(df)} total rows")

        # Filter by symbol prefix
        # Databento returns individual futures contracts (MGCZ0, MGCG1, etc.)
        # We need to filter by prefix and exclude spreads
        symbol_prefix = databento_symbol.replace('.FUT', '')

        print(f"  Filtering for {symbol_prefix}* symbols...")

        # Filter:  starts with prefix AND doesn't contain '-' (exclude spreads)
        mask = (
            df['symbol'].str.startswith(symbol_prefix) &
            ~df['symbol'].str.contains('-')
        )
        df = df[mask]

        if df.empty:
            raise ValueError(
                f"No data found for symbol {databento_symbol}. "
                f"Searched for prefix: {symbol_prefix}*"
            )

        print(f"  ✓ Found {len(df)} rows for {symbol_prefix}")

        # Convert to OHLCV format
        df = self._convert_to_ohlcv(df)

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date, utc=True)]

        # Resample if requested
        if resample:
            df = self._resample_data(df, resample)

        # Cache the result
        self._cache[cache_key] = df

        print(f"✓ Loaded {len(df)} bars for {symbol}")
        return df

    def _to_databento_symbol(self, symbol: str) -> str:
        """Convert LiquidEdge symbol to Databento symbol."""
        # If already Databento format, return as-is
        if symbol.endswith('.FUT'):
            return symbol

        # Map from LiquidEdge naming
        reverse_map = {v: k for k, v in self.SYMBOL_MAP.items()}
        if symbol in reverse_map:
            return reverse_map[symbol]

        raise ValueError(
            f"Unknown symbol: {symbol}. "
            f"Supported: {list(self.SYMBOL_MAP.values())}"
        )

    def _filter_by_instrument_id(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Filter DataFrame by symbol using symbology mapping.

        This is needed when DBN files use instrument_id instead of symbols.
        """
        # Load symbology.json to map instrument_id -> symbol
        symbology_file = self.data_dir / "symbology.json"

        if not symbology_file.exists():
            raise FileNotFoundError(f"Symbology file not found: {symbology_file}")

        import json
        with open(symbology_file, 'r') as f:
            symbology = json.load(f)

        # Build mapping from instrument_id to symbol
        id_to_symbol = {}
        for entry in symbology.get('result', []):
            if entry.get('stype_out_symbol') and entry.get('instrument_id'):
                id_to_symbol[entry['instrument_id']] = entry['stype_out_symbol']

        # Find instrument IDs for our symbol
        target_ids = [
            iid for iid, sym in id_to_symbol.items()
            if sym.startswith(symbol.replace('.FUT', ''))
        ]

        if not target_ids:
            raise ValueError(f"No instrument IDs found for symbol {symbol}")

        # Filter by instrument_id
        df = df[df['instrument_id'].isin(target_ids)]

        return df

    def _convert_to_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Databento DataFrame to standard OHLCV format.

        Databento columns: ts_event, open, high, low, close, volume
        LiquidEdge format: datetime index, open, high, low, close, volume
        """
        # Ensure we have the required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert timestamps to datetime index
        if 'ts_event' in df.columns:
            df['datetime'] = pd.to_datetime(df['ts_event'], utc=True)
            df.set_index('datetime', inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            # Already has datetime index
            pass
        else:
            raise ValueError("No timestamp column found (ts_event or datetime index)")

        # Keep only OHLCV columns
        df = df[required].copy()

        # Sort by time
        df.sort_index(inplace=True)

        return df

    def _resample_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Resample 1-minute data to higher timeframe.

        Args:
            df: DataFrame with 1-minute OHLCV data
            freq: Resample frequency ('15T', '1H', '1D')

        Returns:
            Resampled DataFrame
        """
        resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    def load_all_symbols(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample: Optional[str] = '15T'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for all symbols.

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            resample: Resample frequency ('15T', '1H', '1D')

        Returns:
            Dict of {symbol: DataFrame}

        Example:
            >>> loader = DatabentoMicroFuturesLoader()
            >>> data = loader.load_all_symbols(
            ...     start_date='2025-01-01',
            ...     resample='15T'
            ... )
            >>> gold = data['GOLD']
        """
        symbols = list(self.SYMBOL_MAP.values())
        result = {}

        for symbol in symbols:
            try:
                df = self.load_symbol(
                    symbol,
                    start_date=start_date,
                    end_date=end_date,
                    resample=resample
                )
                result[symbol] = df
            except Exception as e:
                print(f"⚠ Failed to load {symbol}: {e}")

        return result

    def summary_stats(self) -> Dict:
        """
        Get summary statistics for the dataset.

        Returns:
            Dict with dataset information
        """
        # Read metadata
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Convert nanosecond timestamps to datetime
            start_ns = metadata['query']['start']
            end_ns = metadata['query']['end']

            start_dt = pd.Timestamp(start_ns, unit='ns', tz='UTC')
            end_dt = pd.Timestamp(end_ns, unit='ns', tz='UTC')

            return {
                'dataset': metadata['query']['dataset'],
                'schema': metadata['query']['schema'],
                'symbols': metadata['query']['symbols'],
                'start_date': start_dt,
                'end_date': end_dt,
                'duration_years': (end_dt - start_dt).days / 365.25,
                'file_size_mb': self.dbn_file.stat().st_size / (1024 * 1024),
            }

        return {}


def test_loader():
    """Test the Databento loader."""
    print("=" * 70)
    print("DATABENTO MICRO FUTURES LOADER TEST")
    print("=" * 70)
    print()

    # Initialize loader
    loader = DatabentoMicroFuturesLoader()

    # Show stats
    stats = loader.summary_stats()
    print("Dataset Information:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

    # Load GOLD data (last 30 days, 15-minute bars)
    print("Loading GOLD 15-minute data (last 30 days)...")
    end_date = '2026-01-29'
    start_date = '2025-12-30'

    gold_15m = loader.load_symbol(
        'GOLD',
        start_date=start_date,
        end_date=end_date,
        resample='15T'
    )

    print()
    print(f"GOLD 15m data: {len(gold_15m)} bars")
    print()
    print("First 5 bars:")
    print(gold_15m.head())
    print()
    print("Last 5 bars:")
    print(gold_15m.tail())
    print()

    # Load all symbols
    print("=" * 70)
    print("Loading all symbols (last 7 days, 15m bars)...")
    print("=" * 70)
    print()

    all_data = loader.load_all_symbols(
        start_date='2026-01-23',
        end_date='2026-01-29',
        resample='15T'
    )

    for symbol, df in all_data.items():
        print(f"{symbol}: {len(df)} bars, ${df['close'].iloc[-1]:.2f} current price")

    print()
    print("=" * 70)
    print("DATABENTO DATA READY FOR BACKTESTING!")
    print("=" * 70)


if __name__ == "__main__":
    test_loader()
