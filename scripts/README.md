# Scripts Directory

Organized collection of development and testing scripts.

## Directory Structure

### 📊 backtests/
Backtest scripts for strategy validation and optimization.
- `backtest_progressive_cap.py` - Progressive Risk Cap strategy (WINNER)
- `backtest_asset_weighted.py` - Asset-weighted position sizing
- `backtest_kelly_criterion.py` - Kelly Criterion position sizing
- `backtest_stress_test.py` - Slippage and commission stress testing
- `backtest_user_strategy_2yr.py` - Full 15-year historical backtest
- `backtest_multi_asset_ttm.py` - Multi-asset TTM strategy testing
- `backtest_ttm_squeeze_pure.py` - Pure TTM Squeeze indicator testing
- `backtest_pinescript_vs_calibrated.py` - Compare indicator implementations

### 🎯 calibration/
Scripts for calibrating indicators and strategy parameters.
- `calibrate_ttm_squeeze.py` - TTM Squeeze indicator calibration
- `calibrate_ttm_squeeze_v2.py` - Enhanced TTM calibration

### 🔍 debugging/
Diagnostic and analysis scripts for troubleshooting.
- `diagnose_user_strategy.py` - Strategy performance diagnosis
- `diagnose_databento_strategy.py` - Data source diagnosis
- `diagnose_capital.py` - Capital.com API diagnosis
- `diagnose_ttm_squeeze.py` - TTM indicator diagnosis
- `debug_capital_api.py` - API debugging
- `inspect_trades.py` - Trade-by-trade inspection
- `analyze_entry_frequency.py` - Entry signal frequency analysis

### 📁 data/
Data acquisition and exploration scripts.
- `download_kaggle_gold.py` - Download GOLD data from Kaggle
- `download_kaggle_gold_1year.py` - Download recent GOLD data
- `explore_dbn.py` - Explore Databento DBN files
- `check_data.py` - Verify data integrity
- `compare_all_data_sources.py` - Compare different data sources

### 🧪 testing/
API connection and integration tests.
- `test_full_connection.py` - Complete Capital.com connection test
- `test_capital_api.py` - Basic API connectivity
- `test_capital_connect.py` - Connection verification
- `test_api_key.py` - API key validation
- `run_databento_backtest.py` - Databento integration test
- `run_databento_backtest_quick.py` - Quick Databento test
- `run_capital_backtest_simple.py` - Simple Capital.com backtest

## Usage

All scripts can be run from the project root:

```bash
# Example: Run stress test
python scripts/backtests/backtest_stress_test.py

# Example: Test Capital.com connection
python scripts/testing/test_full_connection.py
```

## Production Scripts

Production scripts remain in the project root:
- `paper_trading_engine.py` - Live paper trading engine (PRODUCTION)
