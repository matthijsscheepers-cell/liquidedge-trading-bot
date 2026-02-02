# Capital.com API Integration Guide

## Overview

LiquidEdge supports intraday backtesting using historical data from Capital.com API. This enables high-frequency trading strategies on CFDs for indices, commodities, forex, and cryptocurrencies.

## Prerequisites

### 1. Capital.com Account

You need an active Capital.com account. You can use either:
- **Demo Account**: For testing (free, no real money)
- **Live Account**: For real trading and full historical data access

### 2. API Credentials

To use the Capital.com API, you need **three pieces of information**:

1. **API Key**: Generated from your Capital.com account settings
2. **Email/Identifier**: The email address you used to register with Capital.com
3. **Password**: Your Capital.com account password

### How to Get Your API Key

1. Log into [Capital.com](https://capital.com)
2. Go to **Settings** → **API Keys**
3. Click **Generate New API Key**
4. Copy and save your API key securely

⚠️ **Important**: Keep your API key and password secure. Never commit them to version control.

## Testing Your Credentials

Before running a backtest, verify your credentials work correctly:

```bash
python3 test_capital_credentials.py
```

This interactive script will:
1. Prompt you for your API Key, Email, and Password
2. Test the connection to Capital.com
3. Verify data retrieval is working
4. Provide the exact command to run your backtest

## Running Backtests

### Basic Usage

```bash
python3 scripts/run_backtest.py --mode capital \
  --symbols GOLD SILVER US100 US500 \
  --timeframe 15m \
  --start 2024-10-01 --end 2024-12-31 \
  --api-key "YOUR_API_KEY" \
  --password "YOUR_PASSWORD" \
  --identifier "your.email@example.com"
```

### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--symbols` | Space-separated list of Capital.com epics | `GOLD SILVER US100 US500` |
| `--timeframe` | Data resolution | `5m`, `15m`, `1H`, `4H`, `1D` |
| `--start` | Start date (YYYY-MM-DD) | `2024-01-01` |
| `--end` | End date (YYYY-MM-DD) | `2024-12-31` |
| `--api-key` | Your Capital.com API key | `Hh6exwe6rrnYDgQe` |
| `--password` | Your Capital.com password | `MyPassword123` |
| `--identifier` | Your Capital.com email | `trader@example.com` |

### Available Assets

Capital.com provides CFDs on:

**Indices**:
- `US100` - Nasdaq 100
- `US500` - S&P 500
- `US30` - Dow Jones
- `UK100` - FTSE 100
- `GER40` - DAX 40

**Commodities**:
- `GOLD` - Gold
- `SILVER` - Silver
- `OIL` - Crude Oil (WTI)
- `BRENT` - Brent Crude

**Forex**:
- `EUR/USD` - Euro / US Dollar
- `GBP/USD` - British Pound / US Dollar
- `USD/JPY` - US Dollar / Japanese Yen

**Cryptocurrencies**:
- `BITCOIN` - Bitcoin
- `ETHEREUM` - Ethereum

### Timeframes

Supported intervals:
- `1m` - 1 minute (very short term)
- `5m` - 5 minutes (scalping)
- `15m` - 15 minutes (intraday)
- `1H` - 1 hour (swing)
- `4H` - 4 hours (positional)
- `1D` - 1 day (long term)

**Recommended for intraday trading**: `15m` or `1H`

## Troubleshooting

### Error: "Authentication failed: Invalid email/identifier or password"

**Solution**: Double-check your credentials:
- Use the **exact email** you registered with Capital.com
- Ensure your password is correct (no extra spaces)
- Verify your API key is active and not expired

### Error: "Invalid API key"

**Solution**:
- Generate a new API key from Capital.com settings
- Ensure you're using the correct environment (demo vs live)

### Error: "No data returned"

**Solution**:
- Check that the asset epic is correct (e.g., `GOLD` not `XAU/USD`)
- Verify the date range has market data available
- Some assets may have limited historical data

### Rate Limiting

Capital.com API has rate limits:
- **Maximum**: ~100 requests per minute
- **Automatic retry**: Built-in with exponential backoff
- **Best practice**: Don't fetch data more frequently than every 500ms

## Example Workflows

### 1. Test Demo Account

```bash
# First test with demo credentials
python3 test_capital_credentials.py
# Enter: demo API key, demo password, demo@example.com
```

### 2. Intraday Gold Trading (15-minute bars)

```bash
python3 scripts/run_backtest.py --mode capital \
  --symbols GOLD \
  --timeframe 15m \
  --start 2024-12-01 --end 2024-12-31 \
  --api-key "YOUR_API_KEY" \
  --password "YOUR_PASSWORD" \
  --identifier "your.email@example.com"
```

### 3. Multi-Asset Portfolio (Hourly)

```bash
python3 scripts/run_backtest.py --mode capital \
  --symbols GOLD SILVER US100 US500 EUR/USD \
  --timeframe 1H \
  --start 2024-01-01 --end 2024-12-31 \
  --api-key "YOUR_API_KEY" \
  --password "YOUR_PASSWORD" \
  --identifier "your.email@example.com"
```

### 4. Scalping Strategy (5-minute bars)

```bash
python3 scripts/run_backtest.py --mode capital \
  --symbols US100 \
  --timeframe 5m \
  --start 2024-12-15 --end 2024-12-31 \
  --api-key "YOUR_API_KEY" \
  --password "YOUR_PASSWORD" \
  --identifier "your.email@example.com"
```

## API Limits

| Limit Type | Value | Impact |
|------------|-------|--------|
| Historical Bars | 5,000 per request | Automatically handled |
| Rate Limit | ~100 req/min | Built-in retry logic |
| Concurrent Requests | 1 recommended | Sequential processing |

## Security Best Practices

1. **Never commit credentials** to Git
2. **Use environment variables** for sensitive data
3. **Rotate API keys** periodically
4. **Use demo account** for testing
5. **Monitor API usage** to avoid rate limits

### Using Environment Variables (Recommended)

```bash
# Set credentials as environment variables
export CAPITAL_API_KEY="your_api_key"
export CAPITAL_PASSWORD="your_password"
export CAPITAL_EMAIL="your.email@example.com"

# Run backtest without exposing credentials in command
python3 scripts/run_backtest.py --mode capital \
  --symbols GOLD SILVER \
  --timeframe 15m \
  --start 2024-11-01 --end 2024-12-31 \
  --api-key "$CAPITAL_API_KEY" \
  --password "$CAPITAL_PASSWORD" \
  --identifier "$CAPITAL_EMAIL"
```

## Support

For Capital.com API documentation and support:
- [Capital.com API Docs](https://open-api.capital.com/)
- [Python SDK](https://github.com/capitalcom/capitalcom-python)

For LiquidEdge issues:
- Check `test_capital_credentials.py` output for diagnostics
- Review error messages for specific failure reasons
- Verify internet connection and firewall settings
