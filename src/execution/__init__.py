"""
Execution layer - broker interfaces and order management.
"""

from .broker_interface import (
    BrokerInterface,
    OrderResult,
    AccountInfo,
    Position as BrokerPosition
)
from .capital_connector import CapitalConnector
from .paper_trader import PaperTrader

__all__ = [
    'BrokerInterface',
    'OrderResult',
    'AccountInfo',
    'BrokerPosition',
    'CapitalConnector',
    'PaperTrader'
]
