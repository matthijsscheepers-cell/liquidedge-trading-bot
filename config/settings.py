"""
Configuration Settings Module

This module handles loading and validating environment variables for the trading bot.
It provides a centralized Settings class with type-safe access to configuration values.

Setup Instructions:
    1. Copy .env.example to .env in the project root:
       $ cp .env.example .env

    2. Edit .env and add your Capital.com credentials:
       - CAPITAL_API_KEY: Your Capital.com API key (from Settings → API)
       - CAPITAL_IDENTIFIER: Your Capital.com email address
       - CAPITAL_PASSWORD: Your Capital.com password
       - CAPITAL_ENVIRONMENT: 'demo' for testing, 'live' for real trading

    3. Adjust risk parameters as needed:
       - MAX_POSITION_SIZE_PCT: Maximum portfolio % per position (default: 0.25)
       - EMERGENCY_STOP_DD_PCT: Stop trading if drawdown exceeds this % (default: 0.15)

    4. Never commit the .env file to version control!

Security Notes:
    - API keys, passwords and credentials are NEVER logged or printed
    - All sensitive data is kept in environment variables
    - Use Capital.com demo account for testing before live trading
    - Sensitive values are masked in string representations

Example Usage:
    from config import settings

    # Access configuration values
    broker = settings.broker  # "capital.com"
    api_key = settings.capital_api_key
    max_position = settings.max_position_size_pct

    # Check if running in demo mode
    if settings.is_demo():
        print("Running in DEMO mode - safe for testing")

    # Settings are automatically loaded on import
    # Validation happens immediately - errors are raised if config is invalid
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


@dataclass
class Settings:
    """
    Application settings loaded from environment variables.

    This class loads configuration from a .env file and validates that all
    required variables are present and valid. It provides type-safe access
    to configuration values.

    Attributes:
        broker: Broker name (always "capital.com")
        capital_api_key: Capital.com API key (never logged)
        capital_identifier: Capital.com email/username
        capital_password: Capital.com password (never logged)
        capital_environment: Trading environment ('demo' or 'live')
        default_currency: Default currency pair for trading
        default_units: Default trade size in units
        risk_per_trade: Risk percentage per trade (0-1)
        max_position_size_pct: Maximum position size as % of portfolio (0-1)
        emergency_stop_dd_pct: Emergency stop drawdown threshold (0-1)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        db_path: Database file path
        backtest_start_date: Backtesting start date
        backtest_end_date: Backtesting end date
        backtest_initial_capital: Initial capital for backtesting
        enable_notifications: Whether notifications are enabled
        telegram_bot_token: Telegram bot token (optional)
        telegram_chat_id: Telegram chat ID (optional)

    Example:
        >>> from config import settings
        >>> print(settings.broker)
        'capital.com'
        >>> if settings.is_demo():
        ...     print("Safe to test!")
        'Safe to test!'
    """

    # Broker Configuration
    broker: str = field(default="capital.com")

    # Capital.com API Credentials (REQUIRED)
    capital_api_key: str = field(default="", repr=False)
    capital_identifier: str = field(default="")
    capital_password: str = field(default="", repr=False)
    capital_environment: str = field(default="demo")

    # Trading Configuration
    default_currency: str = field(default="EUR_USD")
    default_units: int = field(default=1000)
    risk_per_trade: float = field(default=0.02)

    # Risk Management (REQUIRED)
    max_position_size_pct: float = field(default=0.25)
    emergency_stop_dd_pct: float = field(default=0.15)

    # Logging Configuration
    log_level: str = field(default="INFO")
    log_to_file: bool = field(default=True)
    log_to_console: bool = field(default=True)

    # Database Configuration
    db_path: str = field(default="data/trading.db")

    # Backtesting Configuration
    backtest_start_date: str = field(default="2023-01-01")
    backtest_end_date: str = field(default="2024-12-31")
    backtest_initial_capital: float = field(default=10000.0)

    # Notification Configuration (Optional)
    enable_notifications: bool = field(default=False)
    telegram_bot_token: Optional[str] = field(default=None, repr=False)
    telegram_chat_id: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """
        Validate all configuration values after initialization.

        Raises:
            ConfigurationError: If required variables are missing or invalid
        """
        self._validate_credentials()
        self._validate_environment()
        self._validate_risk_parameters()
        self._validate_logging()
        self._validate_notifications()

    def _validate_credentials(self) -> None:
        """
        Validate Capital.com API credentials.

        Raises:
            ConfigurationError: If credentials are missing or invalid
        """
        # Check API key
        if not self.capital_api_key:
            raise ConfigurationError(
                "CAPITAL_API_KEY is required.\n"
                "Get your API key from Capital.com:\n"
                "  1. Log into https://capital.com\n"
                "  2. Go to Settings → API\n"
                "  3. Generate new API key\n"
                "  4. Add to .env file"
            )

        if self.capital_api_key in ["your_api_key_here", ""]:
            raise ConfigurationError(
                "Invalid CAPITAL_API_KEY in .env file.\n"
                "Please replace 'your_api_key_here' with your actual API key."
            )

        if len(self.capital_api_key) < 10:
            raise ConfigurationError(
                "CAPITAL_API_KEY appears to be invalid (too short).\n"
                "API keys are typically longer alphanumeric strings."
            )

        # Check identifier (email)
        if not self.capital_identifier:
            raise ConfigurationError(
                "CAPITAL_IDENTIFIER is required.\n"
                "This should be your Capital.com email address."
            )

        if self.capital_identifier in ["your_email@example.com", "your_email_here"]:
            raise ConfigurationError(
                "Invalid CAPITAL_IDENTIFIER in .env file.\n"
                "Please replace with your actual Capital.com email address."
            )

        if "@" not in self.capital_identifier:
            raise ConfigurationError(
                "CAPITAL_IDENTIFIER should be a valid email address."
            )

        # Check password
        if not self.capital_password:
            raise ConfigurationError(
                "CAPITAL_PASSWORD is required.\n"
                "This should be your Capital.com password."
            )

        if self.capital_password in ["your_password_here", "password"]:
            raise ConfigurationError(
                "Invalid CAPITAL_PASSWORD in .env file.\n"
                "Please replace with your actual Capital.com password."
            )

    def _validate_environment(self) -> None:
        """
        Validate trading environment setting.

        Raises:
            ConfigurationError: If environment is invalid
        """
        valid_environments = ["demo", "live"]
        if self.capital_environment not in valid_environments:
            raise ConfigurationError(
                f"Invalid CAPITAL_ENVIRONMENT: '{self.capital_environment}'\n"
                f"Must be either 'demo' (for testing) or 'live' (real money)"
            )

    def _validate_risk_parameters(self) -> None:
        """
        Validate risk management parameters.

        Raises:
            ConfigurationError: If risk parameters are invalid
        """
        # Validate max position size
        if not 0.0 < self.max_position_size_pct <= 1.0:
            raise ConfigurationError(
                f"MAX_POSITION_SIZE_PCT ({self.max_position_size_pct}) must be between 0 and 1.\n"
                f"Example: 0.25 = 25% of portfolio"
            )

        if self.max_position_size_pct > 0.5:
            raise ConfigurationError(
                f"MAX_POSITION_SIZE_PCT ({self.max_position_size_pct}) is too high.\n"
                f"Recommended maximum is 0.5 (50% of portfolio) for safety."
            )

        # Validate emergency stop
        if not 0.0 < self.emergency_stop_dd_pct <= 1.0:
            raise ConfigurationError(
                f"EMERGENCY_STOP_DD_PCT ({self.emergency_stop_dd_pct}) must be between 0 and 1.\n"
                f"Example: 0.15 = stop at 15% drawdown"
            )

        if self.emergency_stop_dd_pct < 0.05:
            raise ConfigurationError(
                f"EMERGENCY_STOP_DD_PCT ({self.emergency_stop_dd_pct}) is too low.\n"
                f"Minimum recommended value is 0.05 (5%)."
            )

        # Validate risk per trade
        if not 0.0 < self.risk_per_trade <= 0.1:
            raise ConfigurationError(
                f"RISK_PER_TRADE ({self.risk_per_trade}) should be between 0 and 0.1.\n"
                f"Example: 0.02 = 2% risk per trade"
            )

    def _validate_logging(self) -> None:
        """
        Validate logging configuration.

        Raises:
            ConfigurationError: If logging config is invalid
        """
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid LOG_LEVEL: '{self.log_level}'\n"
                f"Must be one of: {', '.join(valid_log_levels)}"
            )
        self.log_level = self.log_level.upper()

    def _validate_notifications(self) -> None:
        """
        Validate notification configuration.

        Raises:
            ConfigurationError: If notifications are enabled but credentials missing
        """
        if self.enable_notifications:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                raise ConfigurationError(
                    "ENABLE_NOTIFICATIONS is true, but TELEGRAM_BOT_TOKEN "
                    "or TELEGRAM_CHAT_ID is missing"
                )

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Load settings from environment variables.

        This method loads the .env file and creates a Settings instance
        with all values populated from environment variables.

        Returns:
            Settings instance with values from .env file

        Raises:
            ConfigurationError: If .env file is missing or configuration is invalid

        Example:
            >>> settings = Settings.from_env()
            >>> print(settings.broker)
            'capital.com'
        """
        # Find project root (where .env should be)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        env_path = project_root / ".env"

        if not env_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {env_path}\n\n"
                f"Please create a .env file:\n"
                f"  1. Copy the template: cp .env.example .env\n"
                f"  2. Edit .env and add your Capital.com credentials\n"
                f"  3. Never commit .env to version control!\n\n"
                f"See .env.example for the required format."
            )

        # Load environment variables
        load_dotenv(env_path)

        # Create Settings instance from environment
        return cls(
            broker=os.getenv("BROKER", "capital.com"),
            capital_api_key=os.getenv("CAPITAL_API_KEY", ""),
            capital_identifier=os.getenv("CAPITAL_IDENTIFIER", ""),
            capital_password=os.getenv("CAPITAL_PASSWORD", ""),
            capital_environment=os.getenv("CAPITAL_ENVIRONMENT", "demo"),
            default_currency=os.getenv("DEFAULT_CURRENCY", "EUR_USD"),
            default_units=int(os.getenv("DEFAULT_UNITS", "1000")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
            max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "0.25")),
            emergency_stop_dd_pct=float(os.getenv("EMERGENCY_STOP_DD_PCT", "0.15")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() in ["true", "1", "yes"],
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() in ["true", "1", "yes"],
            db_path=os.getenv("DB_PATH", "data/trading.db"),
            backtest_start_date=os.getenv("BACKTEST_START_DATE", "2023-01-01"),
            backtest_end_date=os.getenv("BACKTEST_END_DATE", "2024-12-31"),
            backtest_initial_capital=float(os.getenv("BACKTEST_INITIAL_CAPITAL", "10000")),
            enable_notifications=os.getenv("ENABLE_NOTIFICATIONS", "false").lower() in ["true", "1", "yes"],
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        )

    def is_demo(self) -> bool:
        """
        Check if running in demo mode.

        Returns:
            True if in demo mode, False if in live mode

        Example:
            >>> if settings.is_demo():
            ...     print("Safe for testing!")
        """
        return self.capital_environment == "demo"

    def is_live(self) -> bool:
        """
        Check if running in live mode.

        Returns:
            True if in live mode, False if in demo mode

        Example:
            >>> if settings.is_live():
            ...     print("⚠️  REAL MONEY AT RISK!")
        """
        return self.capital_environment == "live"

    def __repr__(self) -> str:
        """
        String representation of settings (with sensitive data masked).

        Sensitive values (API key, password) are masked for security.

        Returns:
            Safe string representation

        Example:
            >>> print(settings)
            Settings(
              broker=capital.com,
              environment=demo,
              identifier=user@***,
              api_key=***MASKED***,
              password=***MASKED***,
              ...
            )
        """
        # Mask email (show first part only)
        email_parts = self.capital_identifier.split("@")
        masked_email = f"{email_parts[0]}@***" if len(email_parts) == 2 else "***"

        return (
            f"Settings(\n"
            f"  broker={self.broker},\n"
            f"  environment={self.capital_environment},\n"
            f"  identifier={masked_email},\n"
            f"  api_key=***MASKED***,\n"
            f"  password=***MASKED***,\n"
            f"  max_position_size_pct={self.max_position_size_pct},\n"
            f"  emergency_stop_dd_pct={self.emergency_stop_dd_pct},\n"
            f"  log_level={self.log_level}\n"
            f")"
        )


# Create a singleton instance
# This is loaded automatically when the module is imported
settings = Settings.from_env()
