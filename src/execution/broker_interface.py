from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class OrderResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str]
    fill_price: Optional[float]
    filled_units: Optional[float]
    message: str
    timestamp: datetime
    metadata: Dict = None


@dataclass
class AccountInfo:
    """Broker account information"""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    unrealized_pnl: float
    currency: str
    account_id: str


@dataclass
class Position:
    """Open position information"""
    asset: str
    direction: str  # 'LONG' or 'SHORT'
    units: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    opened_at: datetime


class BrokerInterface(ABC):
    """
    Abstract interface for all brokers.

    This allows easy switching between:
    - Capital.com (live/demo)
    - Paper trader (simulation)
    - Other brokers (future)

    All broker implementations must implement these methods.
    """

    def __init__(self, config: dict):
        """
        Initialize broker connection

        Args:
            config: Broker-specific configuration
        """
        self.config = config
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker"""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get current account information

        Returns:
            AccountInfo with balance, equity, etc
        """
        pass

    @abstractmethod
    def get_current_price(self, asset: str) -> Tuple[float, float]:
        """
        Get current bid/ask price

        Args:
            asset: Instrument (e.g., "US_TECH_100")

        Returns:
            Tuple of (bid, ask)
        """
        pass

    @abstractmethod
    def place_market_order(self,
                          asset: str,
                          direction: str,
                          units: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> OrderResult:
        """
        Place market order

        Args:
            asset: Instrument
            direction: 'LONG' or 'SHORT'
            units: Position size
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)

        Returns:
            OrderResult with execution details
        """
        pass

    @abstractmethod
    def modify_position(self,
                       asset: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """
        Modify existing position

        Args:
            asset: Instrument
            stop_loss: New stop loss (optional)
            take_profit: New take profit (optional)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def close_position(self, asset: str, units: Optional[float] = None) -> OrderResult:
        """
        Close position (partial or full)

        Args:
            asset: Instrument
            units: Units to close (None = close all)

        Returns:
            OrderResult with close details
        """
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions

        Returns:
            List of Position objects
        """
        pass

    @abstractmethod
    def get_historical_data(self,
                           asset: str,
                           timeframe: str,
                           count: int) -> pd.DataFrame:
        """
        Get historical price data

        Args:
            asset: Instrument
            timeframe: e.g., '1H', '15m'
            count: Number of bars

        Returns:
            DataFrame with OHLCV data
        """
        pass

    def is_market_open(self, asset: str) -> bool:
        """
        Check if market is open for trading

        Args:
            asset: Instrument

        Returns:
            True if market is open
        """
        # Default implementation - override if broker provides this
        return True

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
