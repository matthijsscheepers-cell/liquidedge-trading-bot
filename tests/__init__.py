"""
Test Suite

This module contains unit tests, integration tests, and test utilities
for the LIQUIDEDGE trading bot.

Test Structure:
    - test_indicators/: Tests for technical indicators
    - test_regime/: Tests for regime detection
    - test_strategies/: Tests for trading strategies
    - test_risk/: Tests for risk management
    - test_execution/: Tests for order execution
    - test_backtest/: Tests for backtesting engine
    - test_utils/: Tests for utility functions

Test Types:
    - Unit tests: Individual component testing
    - Integration tests: Component interaction testing
    - Property-based tests: Hypothesis testing
    - Mock tests: External API simulation
    - Regression tests: Historical bug prevention

Run tests with:
    pytest tests/ -v --cov=src
"""

from typing import List

__all__: List[str] = []
