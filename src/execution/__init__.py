"""
Order Execution Module

This module handles order placement, management, and execution
via the Capital.com API.

Execution Components:
    - Order Types: Market, Limit, Stop, Stop-Limit, Trailing Stop
    - Order Management: Place, Modify, Cancel orders
    - Position Management: Track open positions, P&L calculation
    - Fill Management: Handle partial fills, slippage
    - Connection Management: API connection, error handling, retry logic

Capital.com Integration:
    - Real-time market data
    - Order execution (positions and working orders)
    - Position monitoring
    - Account information
    - Transaction history
    - Market navigation and search
"""

from typing import List
from src.execution.capital_connector import (
    CapitalConnector,
    ConnectionError,
    APIError,
    NotConnectedError,
    AccountInfo,
)

__all__: List[str] = [
    "CapitalConnector",
    "ConnectionError",
    "APIError",
    "NotConnectedError",
    "AccountInfo",
]
