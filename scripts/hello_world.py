#!/usr/bin/env python3
"""
Capital.com Connection Test - Hello World

This script tests the connection to Capital.com API and displays
account and market information in a formatted table.

Usage:
    python scripts/hello_world.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config import settings, ConfigurationError
from src.execution import CapitalConnector, ConnectionError, APIError, NotConnectedError


# ANSI color codes for terminal output
class Colors:
    """Terminal color codes."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'


def print_banner():
    """Print welcome banner."""
    print("\n" + Colors.CYAN + Colors.BOLD)
    print("🚀 Testing Capital.com Connection...")
    print(Colors.RESET)


def print_divider(char="━", length=50):
    """Print a divider line."""
    print(Colors.BLUE + char * length + Colors.RESET)


def print_section_header(title: str):
    """Print a section header."""
    print("\n" + Colors.BOLD + Colors.CYAN + title + Colors.RESET)
    print_divider()


def print_field(label: str, value: str, color: str = Colors.GREEN):
    """Print a formatted field."""
    print(f"{Colors.BOLD}{label:<20}{Colors.RESET} {color}{value}{Colors.RESET}")


def print_success(message: str):
    """Print success message."""
    print(Colors.GREEN + "✅ " + message + Colors.RESET)


def print_error(message: str):
    """Print error message."""
    print(Colors.RED + "❌ " + message + Colors.RESET)


def print_warning(message: str):
    """Print warning message."""
    print(Colors.YELLOW + "⚠️  " + message + Colors.RESET)


def print_info(message: str):
    """Print info message."""
    print(Colors.BLUE + "ℹ️  " + message + Colors.RESET)


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount."""
    return f"{currency} {amount:,.2f}"


def print_connection_table(
    environment: str,
    account_id: str,
    balance: float,
    available: float,
    currency: str,
    market_name: str = None,
    market_status: str = None,
    market_bid: float = None,
    market_ask: float = None,
):
    """Print formatted connection test results table."""

    print_section_header("CAPITAL.COM CONNECTION TEST")

    # Account Information
    env_color = Colors.GREEN if environment.lower() == "demo" else Colors.YELLOW
    print_field("Broker:", "Capital.com", Colors.CYAN)
    print_field("Environment:", environment.upper(), env_color)
    print_field("Account ID:", account_id, Colors.MAGENTA)
    print_field("Balance:", format_currency(balance, currency), Colors.GREEN)
    print_field("Available:", format_currency(available, currency), Colors.GREEN)
    print_field("Currency:", currency, Colors.BLUE)

    # Market Information (if provided)
    if market_name:
        print("\n" + Colors.BOLD + "Sample Market (US_TECH_100):" + Colors.RESET)
        print_field("  Name:", market_name, Colors.CYAN)

        status_color = Colors.GREEN if market_status == "TRADEABLE" else Colors.RED
        print_field("  Status:", market_status or "N/A", status_color)

        if market_bid is not None:
            print_field("  Bid:", f"{market_bid:,.2f}", Colors.BLUE)
        if market_ask is not None:
            print_field("  Ask:", f"{market_ask:,.2f}", Colors.BLUE)

        if market_bid and market_ask:
            spread = market_ask - market_bid
            print_field("  Spread:", f"{spread:.2f}", Colors.YELLOW)

    print_divider()


def print_troubleshooting(error_type: str, error_message: str):
    """Print troubleshooting information."""

    print("\n" + Colors.RED + Colors.BOLD + "❌ CONNECTION FAILED" + Colors.RESET)
    print_divider("─")

    print_error(f"Error Type: {error_type}")
    print_error(f"Message: {error_message}")

    print("\n" + Colors.BOLD + "🔧 Troubleshooting:" + Colors.RESET)

    if "credential" in error_message.lower() or "authentication" in error_message.lower():
        print_info("Wrong credentials detected")
        print("  1. Check your CAPITAL_API_KEY in .env file")
        print("  2. Verify CAPITAL_IDENTIFIER (email) is correct")
        print("  3. Confirm CAPITAL_PASSWORD is correct")
        print("  4. Regenerate API key at: https://capital.com → Settings → API")

    elif "network" in error_message.lower() or "connection" in error_message.lower():
        print_info("Network error detected")
        print("  1. Check your internet connection")
        print("  2. Verify Capital.com is accessible")
        print("  3. Check if firewall is blocking connection")
        print("  4. Try again in a few moments")

    elif "rate limit" in error_message.lower():
        print_info("Rate limit error")
        print("  1. Too many API requests")
        print("  2. Wait a few minutes before retrying")
        print("  3. Review Capital.com rate limits")

    elif "configuration" in error_message.lower():
        print_info("Configuration error")
        print("  1. Ensure .env file exists")
        print("  2. Copy from template: cp .env.example .env")
        print("  3. Fill in your actual Capital.com credentials")
        print("  4. Check all required fields are set")

    else:
        print_info("General troubleshooting steps")
        print("  1. Verify .env file has correct credentials")
        print("  2. Check Capital.com API status")
        print("  3. Review logs for detailed error information")
        print("  4. Ensure capitalcom-python library is installed")

    print_divider("─")


def main():
    """Main test function."""

    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    print_banner()

    # Step 1: Load Settings
    logger.info("Loading configuration...")
    try:
        # Settings are already loaded as singleton
        logger.success(f"✓ Configuration loaded ({settings.capital_environment} mode)")
    except ConfigurationError as e:
        print_error(f"Configuration Error: {e}")
        print_troubleshooting("ConfigurationError", str(e))
        return 1

    # Step 2: Create Connector
    logger.info("Initializing Capital.com connector...")
    try:
        connector = CapitalConnector(settings)
        logger.success("✓ Connector initialized")
    except Exception as e:
        print_error(f"Failed to initialize connector: {e}")
        return 1

    # Step 3: Connect to API
    logger.info("Connecting to Capital.com API...")
    try:
        if not connector.connect():
            print_error("Connection failed (returned False)")
            print_troubleshooting("ConnectionError", "Connection attempt returned False")
            return 1

        logger.success("✓ Connected to Capital.com API")

    except ConnectionError as e:
        print_error(f"Connection failed: {e}")
        print_troubleshooting("ConnectionError", str(e))
        return 1
    except Exception as e:
        print_error(f"Unexpected error during connection: {e}")
        print_troubleshooting("UnexpectedError", str(e))
        return 1

    # Main operations (wrapped in try-finally for cleanup)
    try:
        # Step 4: Get Account Info
        logger.info("Fetching account information...")
        try:
            account_info = connector.get_account_info()
            logger.success("✓ Account info retrieved")
        except (APIError, NotConnectedError) as e:
            print_error(f"Failed to get account info: {e}")
            print_troubleshooting("APIError", str(e))
            return 1

        # Step 5: Get Session Info
        logger.info("Fetching session information...")
        try:
            session_info = connector.get_session_info()
            logger.success("✓ Session info retrieved")
        except (APIError, NotConnectedError) as e:
            print_warning(f"Could not get session info: {e}")
            session_info = {}

        # Step 6: Get Sample Market Data
        logger.info("Fetching sample market data (US_TECH_100)...")
        market_data = None
        try:
            markets = connector.get_markets(["US_TECH_100"])
            if "US_TECH_100" in markets:
                market_data = markets["US_TECH_100"]
                logger.success("✓ Market data retrieved")
            else:
                logger.warning("US_TECH_100 not found in results")
        except (APIError, NotConnectedError) as e:
            logger.warning(f"Could not get market data: {e}")

        # Step 7: Display Results
        print("\n")
        print_connection_table(
            environment=account_info['account_type'],
            account_id=account_info['account_id'],
            balance=account_info['balance'],
            available=account_info['available'],
            currency=account_info['currency'],
            market_name=market_data['instrumentName'] if market_data else None,
            market_status=market_data['marketStatus'] if market_data else None,
            market_bid=market_data['bid'] if market_data else None,
            market_ask=market_data['offer'] if market_data else None,
        )

        # Additional info
        print("\n" + Colors.BOLD + "Session Details:" + Colors.RESET)
        if session_info:
            for key, value in list(session_info.items())[:5]:  # Show first 5 items
                if key not in ['password', 'apiKey']:  # Don't show sensitive data
                    print(f"  {key}: {value}")

        print("\n" + Colors.GREEN + Colors.BOLD)
        print("✅ ALL SYSTEMS OPERATIONAL!")
        print(Colors.RESET)

        logger.info(f"Test completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    finally:
        # Step 8: Disconnect
        logger.info("Disconnecting from Capital.com...")
        try:
            connector.disconnect()
            logger.success("✓ Disconnected successfully")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n" + Colors.YELLOW + "⚠️  Test interrupted by user" + Colors.RESET)
        sys.exit(130)

    except Exception as e:
        print_error(f"Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        sys.exit(1)
