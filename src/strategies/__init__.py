"""
Trading strategies.

Available strategies:
- RegimePullbackStrategy: Trend following via pullbacks
- TTMSqueezeStrategy: Breakout trading on squeeze release
- StrategySelector: Intelligent routing to appropriate strategy
"""

from .base import (
    BaseStrategy,
    TradeSetup,
    Position,
    SignalDirection,
    ExitAction
)
from .regime_pullback import RegimePullbackStrategy
from .ttm_squeeze import TTMSqueezeStrategy
from .selector import StrategySelector

__all__ = [
    'BaseStrategy',
    'TradeSetup',
    'Position',
    'SignalDirection',
    'ExitAction',
    'RegimePullbackStrategy',
    'TTMSqueezeStrategy',
    'StrategySelector'
]
