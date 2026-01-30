"""
Configuration Module

This module manages application configuration, settings, and parameters.

The primary way to access configuration is through the settings singleton:

    from config import settings

    # Access values
    account_id = settings.OANDA_ACCOUNT_ID
    max_position = settings.MAX_POSITION_SIZE_PCT

    # Check environment
    if settings.is_practice_mode():
        print("Running in practice mode")

Setup Instructions:
    1. Copy .env.example to .env:
       $ cp .env.example .env

    2. Edit .env and add your OANDA credentials

    3. Adjust risk parameters as needed

Configuration Files:
    - .env: Environment variables (API keys, secrets, parameters)
    - Trading parameters (timeframes, instruments, strategy settings)
    - Risk parameters (position sizing, stop-loss levels)
    - API configuration (OANDA credentials, endpoints)
    - Logging configuration (log levels, formats)
    - Backtesting parameters (date ranges, initial capital)

Configuration Sources (in order of priority):
    1. Environment variables (.env file)
    2. Default values (in settings.py)

Security:
    - Never commit .env file to version control
    - API tokens are never logged or printed
    - Sensitive values are masked in string representations
"""

from typing import List

from config.settings import Settings, settings, ConfigurationError

__all__: List[str] = ["Settings", "settings", "ConfigurationError"]
