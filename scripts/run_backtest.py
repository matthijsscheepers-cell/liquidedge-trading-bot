"""
Quick backtest runner for LiquidEdge
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.backtest.data_loader import YahooFinanceLoader, CSVDataLoader, CapitalDataLoader
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceCalculator
from src.backtest.visualizer import BacktestVisualizer


def generate_sample_data():
    """Generate sample trending data for testing"""

    print("Generating sample data (for testing)...")

    dates = pd.date_range('2024-01-01', '2024-03-31', freq='1h')

    # Strong uptrend with pullbacks
    trend = np.linspace(17000, 20000, len(dates))

    # Add realistic market behavior
    noise = np.random.normal(0, 50, len(dates))
    cycles = 200 * np.sin(np.linspace(0, 20, len(dates)))  # Pullback cycles

    close = trend + noise + cycles

    df = pd.DataFrame({
        'open': close - np.random.uniform(0, 30, len(dates)),
        'high': close + np.random.uniform(20, 100, len(dates)),
        'low': close - np.random.uniform(20, 100, len(dates)),
        'close': close,
        'volume': np.random.randint(5000, 15000, len(dates))
    }, index=dates)

    # Ensure high >= low
    df['high'] = df[['high', 'open', 'close']].max(axis=1) + np.random.uniform(0, 20, len(dates))
    df['low'] = df[['low', 'open', 'close']].min(axis=1) - np.random.uniform(0, 20, len(dates))

    print(f"✓ Generated {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    return df


def run_backtest_sample():
    """Run backtest on sample data"""

    print("\n" + "="*60)
    print("LIQUIDEDGE BACKTEST - SAMPLE DATA")
    print("="*60)

    # Generate data
    df = generate_sample_data()

    # Prepare data dict
    data = {'US_TECH_100': df}

    # Create engine
    print("\nInitializing backtest engine...")
    engine = BacktestEngine(
        data=data,
        initial_capital=10000,
        assets=['US_TECH_100'],
        commission_pct=0.0,
        slippage_pct=0.001
    )

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run()

    # Print results
    calc = PerformanceCalculator(
        results['equity_curve'],
        results['trades'],
        10000
    )
    calc.print_summary(results)

    # Visualize
    print("\nGenerating visualizations...")
    viz = BacktestVisualizer(results)
    viz.plot_all(save_path='data/backtest_results/sample_backtest.png')

    print("\n✓ Backtest complete!")
    print("✓ Chart saved to: data/backtest_results/sample_backtest.png")

    plt.show()


def run_backtest_multi_asset(symbols: list = None,
                             start_date: str = '2023-01-01',
                             end_date: str = '2024-12-31',
                             timeframe: str = '1h'):
    """
    Run backtest on multiple assets with hourly data

    Args:
        symbols: List of Yahoo Finance symbols (default: QQQ, SPY, IWM)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        timeframe: Data timeframe ('1h' for hourly, '1d' for daily)
    """
    if symbols is None:
        symbols = ['^NDX', 'QQQ', 'SPY']  # Nasdaq 100, QQQ ETF, S&P 500

    print("\n" + "="*60)
    print(f"LIQUIDEDGE MULTI-ASSET BACKTEST")
    print(f"Assets: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print("="*60)

    # Load data for all assets
    print(f"\nLoading data from Yahoo Finance...")
    loader = YahooFinanceLoader()
    data = {}

    for symbol in symbols:
        try:
            print(f"  Loading {symbol}...")
            df = loader.load_data(
                asset=symbol,
                start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                timeframe=timeframe
            )
            data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {symbol}: Failed to load - {e}")
            continue

    if not data:
        print("❌ No data loaded. Exiting.")
        return

    # Create engine with all assets
    print(f"\nInitializing backtest engine with {len(data)} assets...")
    engine = BacktestEngine(
        data=data,
        initial_capital=10000,
        assets=list(data.keys()),
        commission_pct=0.0,
        slippage_pct=0.001
    )

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run()

    # Print results
    calc = PerformanceCalculator(
        results['equity_curve'],
        results['trades'],
        10000
    )
    calc.print_summary(results)

    # Visualize
    print("\nGenerating visualizations...")
    viz = BacktestVisualizer(results)
    filename = f"multi_asset_{timeframe}_backtest.png"
    viz.plot_all(save_path=f'data/backtest_results/{filename}')

    print(f"\n✓ Backtest complete!")
    print(f"✓ Chart saved to: data/backtest_results/{filename}")

    plt.show()


def run_backtest_yahoo(symbol: str = '^NDX',
                       start_date: str = '2023-01-01',
                       end_date: str = '2024-12-31'):
    """
    Run backtest on Yahoo Finance data (single asset, daily)

    Args:
        symbol: Yahoo Finance symbol (^NDX for Nasdaq 100)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
    """

    print("\n" + "="*60)
    print(f"LIQUIDEDGE BACKTEST - {symbol}")
    print("="*60)

    # Load data
    print(f"\nLoading data from Yahoo Finance...")
    loader = YahooFinanceLoader()

    df = loader.load_data(
        asset=symbol,
        start_date=datetime.strptime(start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(end_date, '%Y-%m-%d'),
        timeframe='1d'  # Daily data
    )

    # Prepare data
    data = {symbol: df}

    # Create engine
    print("\nInitializing backtest engine...")
    engine = BacktestEngine(
        data=data,
        initial_capital=10000,
        assets=[symbol],
        commission_pct=0.0,
        slippage_pct=0.001
    )

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run()

    # Print results
    calc = PerformanceCalculator(
        results['equity_curve'],
        results['trades'],
        10000
    )
    calc.print_summary(results)

    # Visualize
    print("\nGenerating visualizations...")
    viz = BacktestVisualizer(results)
    viz.plot_all(save_path=f'data/backtest_results/{symbol}_backtest.png')

    print("\n✓ Backtest complete!")
    print(f"✓ Chart saved to: data/backtest_results/{symbol}_backtest.png")

    plt.show()


def run_backtest_capital(symbols: list = None,
                         start_date: str = '2024-01-01',
                         end_date: str = '2024-12-31',
                         timeframe: str = '15m',
                         api_key: str = None,
                         password: str = None,
                         identifier: str = None):
    """
    Run backtest using Capital.com data (intraday optimized)

    Args:
        symbols: List of Capital.com epics (default: GOLD, SILVER, US100, US500)
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        timeframe: Data timeframe ('5m', '15m', '1H', etc.)
        api_key: Capital.com API key (optional, uses demo if not provided)
        password: Capital.com password (optional)
        identifier: Capital.com email/login (optional, uses demo if not provided)
    """
    if symbols is None:
        symbols = ['GOLD', 'SILVER', 'US100', 'US500']

    print("\n" + "="*60)
    print(f"LIQUIDEDGE CAPITAL.COM INTRADAY BACKTEST")
    print(f"Assets: {', '.join(symbols)}")
    print(f"Timeframe: {timeframe}")
    print("="*60)

    # Load data for all assets
    print(f"\nLoading data from Capital.com...")
    loader = CapitalDataLoader(
        api_key=api_key,
        password=password,
        identifier=identifier,
        demo=True if not api_key else False
    )
    data = {}

    for symbol in symbols:
        try:
            print(f"  Loading {symbol}...")
            df = loader.load_data(
                asset=symbol,
                start_date=datetime.strptime(start_date, '%Y-%m-%d'),
                end_date=datetime.strptime(end_date, '%Y-%m-%d'),
                timeframe=timeframe
            )
            data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {symbol}: Failed to load - {e}")
            continue

    if not data:
        print("❌ No data loaded. Exiting.")
        return

    # Create engine with all assets
    print(f"\nInitializing backtest engine with {len(data)} assets...")
    engine = BacktestEngine(
        data=data,
        initial_capital=10000,
        assets=list(data.keys()),
        commission_pct=0.001,  # 0.1% commission for CFDs
        slippage_pct=0.001
    )

    # Run backtest
    print("\nRunning backtest...")
    results = engine.run()

    # Print results
    calc = PerformanceCalculator(
        results['equity_curve'],
        results['trades'],
        10000
    )
    calc.print_summary(results)

    # Visualize
    print("\nGenerating visualizations...")
    viz = BacktestVisualizer(results)
    filename = f"capital_{timeframe}_backtest.png"
    viz.plot_all(save_path=f'data/backtest_results/{filename}')

    print(f"\n✓ Backtest complete!")
    print(f"✓ Chart saved to: data/backtest_results/{filename}")

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run LiquidEdge backtest')
    parser.add_argument('--mode', choices=['sample', 'yahoo', 'multi', 'capital'], default='sample',
                       help='Data source (sample, yahoo, multi, capital)')
    parser.add_argument('--symbol', default='^NDX',
                       help='Yahoo Finance symbol for single-asset mode (default: ^NDX)')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Multiple symbols (e.g., --symbols GOLD SILVER US100 US500)')
    parser.add_argument('--start', default='2024-01-01',
                       help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default='2024-12-31',
                       help='End date YYYY-MM-DD')
    parser.add_argument('--timeframe', default='15m', choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Data timeframe (default: 15m for intraday)')
    parser.add_argument('--api-key', default=None,
                       help='Capital.com API key (optional, uses demo if not provided)')
    parser.add_argument('--password', default=None,
                       help='Capital.com password (optional)')
    parser.add_argument('--identifier', default=None,
                       help='Capital.com email/login (optional, uses demo if not provided)')

    args = parser.parse_args()

    # Create results directory
    from pathlib import Path
    Path('data/backtest_results').mkdir(parents=True, exist_ok=True)

    if args.mode == 'sample':
        run_backtest_sample()
    elif args.mode == 'multi':
        run_backtest_multi_asset(args.symbols, args.start, args.end, args.timeframe)
    elif args.mode == 'capital':
        run_backtest_capital(args.symbols, args.start, args.end, args.timeframe,
                           args.api_key, args.password, args.identifier)
    else:
        run_backtest_yahoo(args.symbol, args.start, args.end)
