"""
Utilities Module

This module provides common utility functions used throughout the application.

Utilities Include:
    - Instruments: Capital.com EPIC code mappings and rate limits
    - Data Processing: Cleaning, normalization, resampling
    - Time Management: Timezone handling, market hours
    - Logging: Structured logging, log rotation
    - Configuration: Config loading, validation
    - Validation: Input validation, type checking
    - File I/O: Data loading/saving, path management
    - Notifications: Email, Telegram alerts
    - Database: Data persistence, query helpers
"""

from typing import List
from src.utils.instruments import (
    INSTRUMENTS,
    EPIC_TO_NAME,
    RATE_LIMITS,
    get_epic,
    get_name,
    list_instruments,
    get_rate_limit,
    is_valid_epic,
)

__all__: List[str] = [
    "INSTRUMENTS",
    "EPIC_TO_NAME",
    "RATE_LIMITS",
    "get_epic",
    "get_name",
    "list_instruments",
    "get_rate_limit",
    "is_valid_epic",
]
