#!/usr/bin/env python3
"""
Test script for Capital.com connector.

This script demonstrates basic usage of the CapitalConnector class.
Run with valid credentials in .env file.

Usage:
    python scripts/test_capital_connector.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config import settings, ConfigurationError
from src.execution import CapitalConnector, ConnectionError, APIError


def main():
    """Test Capital.com connector functionality."""

    logger.info("=== Capital.com Connector Test ===\n")

    # Display configuration (masked)
    logger.info(f"Configuration: {settings}")
    logger.info(f"Environment: {settings.capital_environment}")
    logger.info(f"Demo Mode: {settings.is_demo()}\n")

    # Test 1: Create connector
    logger.info("Test 1: Creating connector...")
    try:
        connector = CapitalConnector(settings)
        logger.success(f"✓ Connector created: {connector}")
    except Exception as e:
        logger.error(f"✗ Failed to create connector: {e}")
        return 1

    # Test 2: Connect to API
    logger.info("\nTest 2: Connecting to Capital.com API...")
    try:
        if connector.connect():
            logger.success("✓ Connected successfully!")
        else:
            logger.error("✗ Connection failed")
            return 1
    except ConnectionError as e:
        logger.error(f"✗ Connection error: {e}")
        return 1
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return 1

    try:
        # Test 3: Check connection status
        logger.info("\nTest 3: Checking connection status...")
        if connector.is_connected():
            logger.success("✓ Connection is active")
        else:
            logger.error("✗ Connection is not active")
            return 1

        # Test 4: Get session info
        logger.info("\nTest 4: Getting session information...")
        try:
            session = connector.get_session_info()
            logger.success(f"✓ Session info retrieved")
            logger.info(f"   Account ID: {session.get('accountId', 'N/A')}")
            logger.info(f"   Currency: {session.get('currency', 'N/A')}")
        except Exception as e:
            logger.error(f"✗ Failed to get session info: {e}")

        # Test 5: Get account info
        logger.info("\nTest 5: Getting account information...")
        try:
            account = connector.get_account_info()
            logger.success("✓ Account info retrieved:")
            logger.info(f"   Balance: {account['balance']} {account['currency']}")
            logger.info(f"   Available: {account['available']} {account['currency']}")
            logger.info(f"   Account Type: {account['account_type']}")
            logger.info(f"   Account ID: {account['account_id']}")
        except Exception as e:
            logger.error(f"✗ Failed to get account info: {e}")

        # Test 6: Get market data
        logger.info("\nTest 6: Getting market data...")
        try:
            # Test with common instruments
            epics = ["US_TECH_100", "EUR_USD"]
            markets = connector.get_markets(epics)

            logger.success(f"✓ Market data retrieved for {len(markets)} instruments:")
            for epic, data in markets.items():
                logger.info(
                    f"   {epic} ({data['instrumentName']}): "
                    f"Bid={data['bid']}, Ask={data['offer']}, "
                    f"Change={data['percentageChange']:.2f}%"
                )
        except Exception as e:
            logger.error(f"✗ Failed to get market data: {e}")

        # Test 7: Get single market details
        logger.info("\nTest 7: Getting detailed market info...")
        try:
            market = connector.get_single_market("US_TECH_100")
            logger.success("✓ Detailed market info retrieved for US_TECH_100")
            if 'instrument' in market:
                logger.info(f"   Name: {market['instrument'].get('name', 'N/A')}")
        except Exception as e:
            logger.error(f"✗ Failed to get single market: {e}")

    finally:
        # Test 8: Disconnect
        logger.info("\nTest 8: Disconnecting...")
        try:
            connector.disconnect()
            logger.success("✓ Disconnected successfully")
        except Exception as e:
            logger.error(f"✗ Disconnect error: {e}")

    logger.info("\n=== All Tests Completed ===")
    return 0


def test_context_manager():
    """Test context manager usage."""
    logger.info("\n=== Testing Context Manager ===\n")

    try:
        with CapitalConnector(settings) as connector:
            connector.connect()
            account = connector.get_account_info()
            logger.info(f"Balance: {account['balance']} {account['currency']}")

        logger.success("✓ Context manager test passed")
        return 0

    except Exception as e:
        logger.error(f"✗ Context manager test failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        # Run main tests
        exit_code = main()

        if exit_code == 0:
            # Run context manager test
            exit_code = test_context_manager()

        sys.exit(exit_code)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Please check your .env file")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nTest interrupted by user")
        sys.exit(130)
