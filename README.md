# LIQUIDEDGE Trading Bot

A hybrid trading bot that combines technical analysis, regime detection, and risk management for automated CFD trading via Capital.com API.

## Features

- **Multi-Regime Detection**: Automatically detects market conditions (trending, ranging, volatile)
- **Technical Indicators**: Comprehensive suite of technical analysis indicators
- **Risk Management**: Position sizing, stop-loss, and portfolio risk controls
- **Backtesting Engine**: Test strategies on historical data before live deployment
- **Capital.com Integration**: Direct integration with Capital.com's API for live trading
- **Mac Compatible**: Works on macOS, Windows, and Linux
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Project Structure

```
LIQUIDEDGE/
├── config/              # Configuration files and parameters
├── src/                 # Source code
│   ├── indicators/      # Technical indicators
│   ├── regime/          # Market regime detection
│   ├── strategies/      # Trading strategies
│   ├── risk/            # Risk management
│   ├── execution/       # Order execution
│   ├── backtest/        # Backtesting engine
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for analysis
├── data/                # Data storage
│   ├── historical/      # Historical price data
│   ├── backtest/        # Backtest results
│   └── live/            # Live trading data
├── logs/                # Log files
└── scripts/             # Utility scripts
```

## Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Project LIQUIDEDGE"
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

## Quick Start with Make

This project includes a Makefile for convenient command shortcuts. Run `make help` to see all available commands.

**Common Commands:**

```bash
# View help
make help

# TODO Management
make todos              # List all todos with beautiful formatting
make todos-week W=4     # View todos for specific week
make todo-done ID=1     # Mark todo as complete
make todo-add           # Add new todo interactively

# Utilities
make instruments        # List available trading instruments
make test-connection    # Test Capital.com API connection
make install            # Install all dependencies
make clean              # Remove cache files

# Testing
make test               # Run test suite
make lint               # Run linting checks
```

**Weekly Planning Workflow:**

Every Friday, check your upcoming tasks:
```bash
make todos-week W=<next_week>
```

## Broker Setup - Capital.com

### Why Capital.com?

- ✅ **Mac Compatible**: REST API works on all platforms (no Windows-only limitations)
- ✅ **CFD Trading**: Access to forex, indices, commodities, stocks, and crypto
- ✅ **Low Minimum**: Start with as little as $20 (demo account available)
- ✅ **Demo Account**: Test strategies risk-free with virtual funds
- ✅ **Good API**: Well-documented REST API with unofficial Python library

### Step 1: Create Capital.com Account

1. Visit [Capital.com](https://capital.com) and sign up
2. Choose account type:
   - **Demo Account**: Virtual money for testing (recommended to start)
   - **Live Account**: Real money trading (after thorough testing)
3. Complete verification (for live account)

### Step 2: Get API Credentials

1. Log into Capital.com web platform
2. Navigate to: **Settings → API**
3. Click **"Generate API Key"**
4. Save your credentials:
   - **API Key**: Long alphanumeric string
   - **Email**: Your Capital.com login email
   - **Password**: Your Capital.com password

⚠️ **Security**: Keep these credentials secure and never commit them to version control!

### Step 3: Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` file with your credentials:
   ```bash
   # Broker Configuration
   BROKER=capital.com

   # Capital.com API Credentials
   CAPITAL_API_KEY=your_api_key_here
   CAPITAL_IDENTIFIER=your_email@example.com
   CAPITAL_PASSWORD=your_password_here
   CAPITAL_ENVIRONMENT=demo  # Use 'demo' for testing, 'live' for real trading

   # Risk Management
   MAX_POSITION_SIZE_PCT=0.25  # Maximum 25% of portfolio per position
   EMERGENCY_STOP_DD_PCT=0.15  # Stop trading at 15% drawdown
   ```

3. Replace placeholder values with your actual credentials

### Step 4: Test Connection

Run the connection test script:

```bash
python scripts/hello_world.py
```

Expected output:
```
🚀 Testing Capital.com Connection...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAPITAL.COM CONNECTION TEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Broker:              Capital.com
Environment:         DEMO
Account ID:          XXXXX
Balance:             EUR 1,000.00
Available:           EUR 1,000.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ ALL SYSTEMS OPERATIONAL!
```

### Capital.com Asset EPICs (Instrument Codes)

Common instruments and their EPIC codes:

**Indices:**
- `US_30` - Dow Jones Industrial Average
- `US_500` - S&P 500
- `US_TECH_100` - Nasdaq 100
- `UK_100` - FTSE 100
- `GERMANY_40` - DAX 40
- `JAPAN_225` - Nikkei 225

**Forex:**
- `EUR_USD` - Euro / US Dollar
- `GBP_USD` - British Pound / US Dollar
- `USD_JPY` - US Dollar / Japanese Yen
- `AUD_USD` - Australian Dollar / US Dollar
- `EUR_GBP` - Euro / British Pound

**Commodities:**
- `GOLD` - Gold (XAU/USD)
- `SILVER` - Silver (XAG/USD)
- `OIL_CRUDE` - Crude Oil (WTI)
- `NATURAL_GAS` - Natural Gas

**Cryptocurrencies:**
- `BITCOIN` - Bitcoin / USD
- `ETHEREUM` - Ethereum / USD

To find more EPICs, use the market search in Capital.com platform or use the connector's `get_markets()` method.

### Capital.com Resources

- **API Documentation**: [capital.com/api-development-guide](https://capital.com/api-development-guide)
- **Support**: [help.capital.com](https://help.capital.com)
- **Python Library**: [capitalcom-python on PyPI](https://pypi.org/project/capitalcom-python/)

### Troubleshooting

**Connection Failed:**
- Verify API key is correct (regenerate if needed)
- Check email and password match your Capital.com account
- Ensure you're using the correct environment (demo vs live)
- Check your internet connection

**Invalid Credentials:**
- Regenerate API key at: Settings → API
- Make sure `.env` file has no extra spaces or quotes
- Verify CAPITAL_ENVIRONMENT is set to "demo" or "live"

**Rate Limiting:**
- Capital.com has API rate limits
- Implement delays between requests if hitting limits
- Use demo account for development/testing

## Usage

### Test Connection

Verify your Capital.com connection:

```bash
python scripts/hello_world.py
# or
make test-connection
```

### Task Management

Track your development progress with the built-in TODO tracker:

```bash
# View all tasks
make todos
# or
./venv/bin/python3 scripts/todo_tracker.py list

# View specific week
make todos-week W=4

# Mark task complete
make todo-done ID=1

# Add new task
make todo-add
```

The TODO tracker features:
- Week-based organization (current focus: weeks 3-6)
- Color-coded priorities (critical, important, nice-to-have)
- Beautiful rich terminal formatting with tables and panels
- Time estimates for each task
- Progress tracking with completion timestamps

### Backtesting

Test strategies on historical data:

```bash
python scripts/run_backtest.py --strategy momentum --start 2023-01-01 --end 2024-12-31
```

### Live Trading

Run live trading (always start with demo/paper trading):

```bash
python scripts/run_live.py --strategy momentum --demo
```

### Data Collection

Fetch historical data for analysis:

```bash
python scripts/fetch_data.py --epic EUR_USD --start 2023-01-01
```

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=src
```

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run formatting and checks:

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## Risk Warning

**Trading forex carries significant risk. This bot is for educational and research purposes. Always test thoroughly with paper trading before using real capital. Past performance does not guarantee future results.**

## License

[Add your license here]

## Contact

[Add your contact information here]
