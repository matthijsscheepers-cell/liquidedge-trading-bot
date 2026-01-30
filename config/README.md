# Configuration Module

Secure secrets management system for the LIQUIDEDGE trading bot.

## Quick Start

### 1. Setup Environment File

Copy the template and add your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your OANDA credentials:

```bash
# Required
OANDA_ACCOUNT_ID=101-001-12345678-001
OANDA_API_TOKEN=your_actual_api_token_here
OANDA_ENVIRONMENT=practice

# Risk Management
MAX_POSITION_SIZE_PCT=0.25
EMERGENCY_STOP_DD_PCT=0.15
```

**Important:** Never commit `.env` to version control!

### 2. Use in Your Code

```python
from config import settings

# Access configuration values
account_id = settings.OANDA_ACCOUNT_ID
api_url = settings.OANDA_API_URL
max_position = settings.MAX_POSITION_SIZE_PCT

# Check environment
if settings.is_practice_mode():
    print("Running in PRACTICE mode - safe for testing")

# All sensitive values are masked in logging
print(settings)  # API token is masked as ***MASKED***
```

## Configuration Values

### Required Variables

| Variable | Type | Description |
|----------|------|-------------|
| `OANDA_ACCOUNT_ID` | string | Your OANDA account ID |
| `OANDA_API_TOKEN` | string | Your OANDA API token (kept secret) |
| `OANDA_ENVIRONMENT` | string | `'practice'` or `'live'` |
| `MAX_POSITION_SIZE_PCT` | float | Max position size (0-1, e.g., 0.25 = 25%) |
| `EMERGENCY_STOP_DD_PCT` | float | Emergency stop drawdown (0-1, e.g., 0.15 = 15%) |

### Optional Variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `OANDA_API_URL` | Auto-set | API endpoint (set based on environment) |
| `DEFAULT_CURRENCY` | EUR_USD | Default trading pair |
| `DEFAULT_UNITS` | 1000 | Default trade size |
| `RISK_PER_TRADE` | 0.02 | Risk per trade (2%) |
| `LOG_LEVEL` | INFO | Logging level |
| `BACKTEST_INITIAL_CAPITAL` | 10000 | Starting capital for backtests |

## Validation

The configuration system validates all values:

- **API Token**: Must not be empty or placeholder value
- **Environment**: Must be 'practice' or 'live'
- **Risk Parameters**: Must be within safe ranges (0-1)
- **Position Size**: Maximum 50% recommended
- **Emergency Stop**: Minimum 5% recommended
- **Log Level**: Must be valid Python logging level

If validation fails, a `ConfigurationError` is raised with a clear error message.

## Security Features

### Never Logged

Sensitive values are **never** logged or printed:

```python
print(settings)
# Output: Settings(
#   environment=practice,
#   account_id=***-001,        # Last 4 chars only
#   api_token=***MASKED***,    # Fully masked
#   ...
# )
```

### Environment Isolation

- `.env` file is excluded from git (via `.gitignore`)
- `.env.example` is committed as a safe template
- All secrets stay local to your machine

### Validation on Load

Configuration is validated immediately on import:

```python
try:
    from config import settings
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

## Error Handling

### Missing .env File

```
ConfigurationError: Configuration file not found: /path/to/.env

Please create a .env file:
  1. Copy the template: cp .env.example .env
  2. Edit .env and add your OANDA credentials
  3. Never commit .env to version control!
```

### Invalid API Token

```
ConfigurationError: Invalid OANDA_API_TOKEN in .env file.
Please set a real API token from your OANDA account.
Get your token at: https://www.oanda.com/account/tpa/personal_token
```

### Invalid Environment

```
ConfigurationError: Invalid OANDA_ENVIRONMENT: 'test'
Must be either 'practice' or 'live'
```

## Best Practices

1. **Always use practice mode first**: Test your bot with paper trading before going live
2. **Set conservative risk limits**: Start with lower position sizes and emergency stops
3. **Keep secrets secure**: Never commit `.env`, share API tokens, or log sensitive data
4. **Validate early**: Import settings at the start of your application to catch errors immediately
5. **Use environment checks**: Check `settings.is_practice_mode()` before risky operations

## Example: Safe Trading Bot Startup

```python
#!/usr/bin/env python3
"""Trading bot entry point with safe configuration loading."""

import sys
from loguru import logger

try:
    from config import settings, ConfigurationError
except ConfigurationError as e:
    print(f"❌ Configuration Error: {e}")
    sys.exit(1)

# Log startup (secrets are automatically masked)
logger.info("Starting LIQUIDEDGE Trading Bot")
logger.info(f"Configuration: {settings}")

# Safety check
if settings.is_live_mode():
    response = input("⚠️  LIVE MODE - Real money at risk! Continue? (yes/no): ")
    if response.lower() != 'yes':
        logger.info("User cancelled live trading")
        sys.exit(0)

# Start trading
logger.info("Bot started successfully")
# ... rest of bot code ...
```

## Troubleshooting

**Problem**: `ConfigurationError: Configuration file not found`
**Solution**: Create `.env` file: `cp .env.example .env`

**Problem**: `Invalid OANDA_API_TOKEN`
**Solution**: Get real API token from OANDA account settings

**Problem**: `MAX_POSITION_SIZE_PCT is too high`
**Solution**: Reduce to 0.5 (50%) or less for safety

**Problem**: Settings not updating after editing `.env`
**Solution**: Restart your application (settings load once at import)
