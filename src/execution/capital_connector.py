"""
Capital.com API Connector

This module provides a connector class for interacting with the Capital.com API
using the unofficial capitalcom-python library.

The CapitalConnector class handles:
    - Authentication and session management
    - Account information retrieval
    - Market data access
    - Connection state management
    - Error handling and logging

Security:
    - Credentials are never logged
    - Sensitive session data is masked in logs
    - Automatic disconnect on context manager exit

Example:
    Basic usage with context manager:

    >>> from src.execution.capital_connector import CapitalConnector
    >>> from config import settings
    >>>
    >>> with CapitalConnector(settings) as connector:
    ...     if connector.connect():
    ...         account = connector.get_account_info()
    ...         print(f"Balance: {account['balance']}")

    Manual connection management:

    >>> connector = CapitalConnector(settings)
    >>> try:
    ...     connector.connect()
    ...     info = connector.get_account_info()
    ... finally:
    ...     connector.disconnect()
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import capitalcom
from config.settings import Settings


class ConnectionError(Exception):
    """Raised when connection to Capital.com API fails."""
    pass


class APIError(Exception):
    """Raised when Capital.com API returns an error."""
    pass


class NotConnectedError(Exception):
    """Raised when attempting operations without an active connection."""
    pass


@dataclass
class AccountInfo:
    """
    Account information from Capital.com.

    Attributes:
        balance: Current account balance
        available: Available funds for trading
        currency: Account currency (e.g., "USD", "EUR")
        account_type: Type of account ("demo" or "live")
        account_id: Capital.com account ID
    """
    balance: float
    available: float
    currency: str
    account_type: str
    account_id: str


class CapitalConnector:
    """
    Connector for Capital.com API integration.

    This class provides a high-level interface to the Capital.com trading API,
    handling authentication, session management, and common trading operations.

    The connector supports both demo and live trading environments, as configured
    in the settings.

    Attributes:
        settings: Application settings containing API credentials
        client: Capital.com API client instance (None until connected)
        is_connected: Connection status flag

    Example:
        >>> from config import settings
        >>> connector = CapitalConnector(settings)
        >>> if connector.connect():
        ...     print("Connected to Capital.com")
        ...     account = connector.get_account_info()
        ...     connector.disconnect()

        Using as context manager (recommended):
        >>> with CapitalConnector(settings) as conn:
        ...     conn.connect()
        ...     markets = conn.get_markets(["US_TECH_100"])
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the Capital.com connector.

        Args:
            settings: Application settings with Capital.com credentials

        Note:
            This does not establish a connection. Call connect() to authenticate.
        """
        self.settings = settings
        self.client: Optional[capitalcom.Client] = None
        self._connected: bool = False
        self._session_info: Optional[Dict[str, Any]] = None

        logger.info(
            f"Initialized Capital.com connector "
            f"(environment: {settings.capital_environment})"
        )

    def connect(self) -> bool:
        """
        Connect to Capital.com API and authenticate.

        Creates a client instance with credentials from settings and attempts
        to authenticate. The session is stored for subsequent API calls.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If authentication fails
            APIError: If API returns an error

        Example:
            >>> connector = CapitalConnector(settings)
            >>> if connector.connect():
            ...     print("Successfully connected")
            ... else:
            ...     print("Connection failed")
        """
        if self._connected:
            logger.warning("Already connected to Capital.com")
            return True

        try:
            logger.info(
                f"Connecting to Capital.com API "
                f"({self.settings.capital_environment} environment)..."
            )

            # Create client instance
            # Note: The library uses 'log' (login/email), 'pas' (password), 'api_key'
            # The Client class is from client_demo module (demo environment)
            self.client = capitalcom.Client(
                log=self.settings.capital_identifier,
                pas=self.settings.capital_password,
                api_key=self.settings.capital_api_key
            )

            # Verify connection by getting session details
            # Note: Library has typo "get_sesion_details"
            self._session_info = self.client.get_sesion_details()

            if self._session_info:
                self._connected = True
                logger.success(
                    f"✓ Connected to Capital.com "
                    f"({self.settings.capital_environment} mode)"
                )
                return True
            else:
                logger.error("Failed to retrieve session details")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Capital.com: {e}")
            self._connected = False
            self.client = None
            raise ConnectionError(f"Connection failed: {str(e)}") from e

    def disconnect(self) -> None:
        """
        Disconnect from Capital.com API.

        Logs out from the API and clears the session. Safe to call multiple times.

        Example:
            >>> connector = CapitalConnector(settings)
            >>> connector.connect()
            >>> # ... do trading operations ...
            >>> connector.disconnect()
        """
        if not self._connected or self.client is None:
            logger.debug("Not connected, nothing to disconnect")
            return

        try:
            logger.info("Disconnecting from Capital.com...")
            self.client.log_out_account()
            logger.success("✓ Disconnected from Capital.com")

        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

        finally:
            self._connected = False
            self.client = None
            self._session_info = None

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.

        Retrieves account details including balance, available funds, currency,
        and account type from the Capital.com API.

        Returns:
            Dictionary containing:
                - balance: Current account balance (float)
                - available: Available funds for trading (float)
                - currency: Account currency (str)
                - account_type: "demo" or "live" (str)
                - account_id: Capital.com account ID (str)
                - accounts: List of all accounts (list)

        Raises:
            NotConnectedError: If not connected to API
            APIError: If API call fails

        Example:
            >>> info = connector.get_account_info()
            >>> print(f"Balance: {info['balance']} {info['currency']}")
            >>> print(f"Available: {info['available']} {info['currency']}")
        """
        self._ensure_connected()

        try:
            logger.debug("Fetching account information...")

            # Get all accounts
            accounts_data = self.client.all_accounts()

            if not accounts_data or 'accounts' not in accounts_data:
                raise APIError("No account data returned from API")

            accounts = accounts_data['accounts']

            if not accounts:
                raise APIError("No accounts found")

            # Get the first (primary) account
            primary_account = accounts[0]

            # Extract balance info (nested structure)
            balance_data = primary_account.get('balance', {})

            # Extract account info
            account_info = {
                'balance': float(balance_data.get('balance', 0.0)),
                'available': float(balance_data.get('available', 0.0)),
                'deposit': float(balance_data.get('deposit', 0.0)),
                'profit_loss': float(balance_data.get('profitLoss', 0.0)),
                'currency': primary_account.get('currency', 'USD'),
                'account_type': self.settings.capital_environment,
                'account_id': primary_account.get('accountId', ''),
                'account_name': primary_account.get('accountName', ''),
                'accounts': accounts  # All accounts for reference
            }

            logger.debug(
                f"Account info retrieved: "
                f"Balance={account_info['balance']} {account_info['currency']}, "
                f"Available={account_info['available']}"
            )

            return account_info

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise APIError(f"Failed to retrieve account info: {str(e)}") from e

    def get_markets(self, epic_list: List[str]) -> Dict[str, Any]:
        """
        Get market information for specified instruments.

        Retrieves current market data including prices, status, and trading
        information for the specified EPICs (instrument identifiers).

        Args:
            epic_list: List of Capital.com EPIC codes
                      Examples: ["US_TECH_100", "EUR_USD", "GOLD"]

        Returns:
            Dictionary mapping EPIC codes to market information:
                - epic: Instrument identifier
                - instrumentName: Display name
                - bid: Current bid price
                - offer: Current offer price (ask)
                - high: Day's high price
                - low: Day's low price
                - percentageChange: Daily % change
                - updateTime: Last update timestamp
                - marketStatus: "TRADEABLE", "CLOSED", etc.

        Raises:
            NotConnectedError: If not connected to API
            APIError: If API call fails
            ValueError: If epic_list is empty

        Example:
            >>> markets = connector.get_markets(["US_TECH_100", "EUR_USD"])
            >>> for epic, data in markets.items():
            ...     print(f"{epic}: Bid={data['bid']}, Ask={data['offer']}")
        """
        self._ensure_connected()

        if not epic_list:
            raise ValueError("epic_list cannot be empty")

        try:
            logger.debug(f"Fetching market data for {len(epic_list)} instruments...")

            # Build comma-separated EPIC string
            epics_str = ",".join(epic_list)

            # Call API
            markets_data = self.client.searching_market(epics=epics_str)

            if not markets_data or 'markets' not in markets_data:
                raise APIError("No market data returned from API")

            # Parse results into dict keyed by EPIC
            markets = {}
            for market in markets_data['markets']:
                epic = market.get('epic')
                if epic:
                    markets[epic] = {
                        'epic': market.get('epic'),
                        'instrumentName': market.get('instrumentName'),
                        'bid': float(market.get('bid', 0.0)),
                        'offer': float(market.get('offer', 0.0)),
                        'high': float(market.get('high', 0.0)),
                        'low': float(market.get('low', 0.0)),
                        'percentageChange': float(market.get('percentageChange', 0.0)),
                        'updateTime': market.get('updateTime'),
                        'marketStatus': market.get('marketStatus'),
                        'raw': market  # Keep full data for reference
                    }

            logger.debug(f"Retrieved market data for {len(markets)} instruments")

            return markets

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            raise APIError(f"Failed to retrieve market data: {str(e)}") from e

    def get_single_market(self, epic: str) -> Dict[str, Any]:
        """
        Get detailed information for a single market.

        Retrieves comprehensive market data for a specific instrument,
        including trading hours, margin requirements, and contract details.

        Args:
            epic: Capital.com EPIC code (e.g., "US_TECH_100")

        Returns:
            Dictionary with detailed market information

        Raises:
            NotConnectedError: If not connected to API
            APIError: If API call fails

        Example:
            >>> market = connector.get_single_market("US_TECH_100")
            >>> print(f"Instrument: {market['instrument']['name']}")
        """
        self._ensure_connected()

        try:
            logger.debug(f"Fetching detailed market data for {epic}...")
            market_data = self.client.single_market(epic=epic)

            if not market_data:
                raise APIError(f"No data returned for EPIC: {epic}")

            logger.debug(f"Retrieved detailed market data for {epic}")
            return market_data

        except Exception as e:
            logger.error(f"Failed to get single market data for {epic}: {e}")
            raise APIError(f"Failed to retrieve market {epic}: {str(e)}") from e

    def is_connected(self) -> bool:
        """
        Check if currently connected to Capital.com API.

        Returns:
            True if connected, False otherwise

        Example:
            >>> if connector.is_connected():
            ...     print("Ready to trade")
            ... else:
            ...     connector.connect()
        """
        return self._connected and self.client is not None

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get current session information.

        Returns details about the current API session including account details,
        currency, and session status.

        Returns:
            Dictionary containing session details

        Raises:
            NotConnectedError: If not connected to API

        Example:
            >>> session = connector.get_session_info()
            >>> print(f"Account: {session.get('accountId')}")
            >>> print(f"Currency: {session.get('currency')}")
        """
        self._ensure_connected()

        if self._session_info is None:
            logger.warning("Session info not available, fetching...")
            try:
                self._session_info = self.client.get_sesion_details()
            except Exception as e:
                raise APIError(f"Failed to get session info: {str(e)}") from e

        return self._session_info

    def _ensure_connected(self) -> None:
        """
        Ensure connection is active.

        Raises:
            NotConnectedError: If not connected to API
        """
        if not self.is_connected():
            raise NotConnectedError(
                "Not connected to Capital.com API. Call connect() first."
            )

    def __enter__(self) -> "CapitalConnector":
        """
        Enter context manager.

        Returns:
            Self for use in with statement

        Example:
            >>> with CapitalConnector(settings) as conn:
            ...     conn.connect()
            ...     # Use connector
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context manager and cleanup.

        Automatically disconnects when exiting the context.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        self.disconnect()

    def __repr__(self) -> str:
        """
        String representation of connector.

        Returns:
            String showing connection status and environment
        """
        status = "connected" if self._connected else "disconnected"
        return (
            f"CapitalConnector("
            f"environment={self.settings.capital_environment}, "
            f"status={status})"
        )
