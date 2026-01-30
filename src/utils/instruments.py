"""
Capital.com Instrument Mappings

This module provides mappings between common instrument names and Capital.com EPIC codes.
It also includes rate limit information and helper functions for instrument handling.

Usage:
    from src.utils.instruments import INSTRUMENTS, get_epic

    # Get EPIC code for Nasdaq 100
    epic = INSTRUMENTS['NAS100']  # Returns 'US_TECH_100'

    # Or use helper function
    epic = get_epic('NAS100')  # Returns 'US_TECH_100'
"""

from typing import Dict, Optional, List


# Instrument mappings: Common name -> Capital.com EPIC
INSTRUMENTS: Dict[str, str] = {
    # === INDICES ===
    'NAS100': 'US_TECH_100',         # Nasdaq 100
    'NASDAQ': 'US_TECH_100',         # Nasdaq 100 (alias)
    'US500': 'US_SPX_500',           # S&P 500
    'SPX': 'US_SPX_500',             # S&P 500 (alias)
    'US30': 'US_30',                 # Dow Jones Industrial Average
    'DOW': 'US_30',                  # Dow Jones (alias)
    'GER40': 'GERMANY_40',           # DAX 40
    'DAX': 'GERMANY_40',             # DAX (alias)
    'UK100': 'UK_100',               # FTSE 100
    'FTSE': 'UK_100',                # FTSE (alias)
    'JPN225': 'JAPAN_225',           # Nikkei 225
    'NIKKEI': 'JAPAN_225',           # Nikkei (alias)

    # === COMMODITIES ===
    'XAUUSD': 'GOLD',                # Gold vs USD
    'GOLD': 'GOLD',                  # Gold (direct)
    'XAGUSD': 'SILVER',              # Silver vs USD
    'SILVER': 'SILVER',              # Silver (direct)
    'WTIUSD': 'CRUDE_OIL_WTI',       # WTI Crude Oil
    'OIL': 'CRUDE_OIL_WTI',          # Oil (alias)
    'BRENT': 'CRUDE_OIL_BRENT',      # Brent Crude Oil
    'NATGAS': 'NATURAL_GAS',         # Natural Gas

    # === FOREX ===
    'EURUSD': 'EUR_USD',             # Euro / US Dollar
    'GBPUSD': 'GBP_USD',             # British Pound / US Dollar
    'USDJPY': 'USD_JPY',             # US Dollar / Japanese Yen
    'AUDUSD': 'AUD_USD',             # Australian Dollar / US Dollar
    'USDCAD': 'USD_CAD',             # US Dollar / Canadian Dollar
    'USDCHF': 'USD_CHF',             # US Dollar / Swiss Franc
    'NZDUSD': 'NZD_USD',             # New Zealand Dollar / US Dollar
    'EURGBP': 'EUR_GBP',             # Euro / British Pound
    'EURJPY': 'EUR_JPY',             # Euro / Japanese Yen
    'GBPJPY': 'GBP_JPY',             # British Pound / Japanese Yen

    # === CRYPTOCURRENCIES ===
    'BTCUSD': 'BITCOIN',             # Bitcoin / USD
    'BTC': 'BITCOIN',                # Bitcoin (alias)
    'ETHUSD': 'ETHEREUM',            # Ethereum / USD
    'ETH': 'ETHEREUM',               # Ethereum (alias)
}


# Reverse mapping: EPIC -> Common name
EPIC_TO_NAME: Dict[str, str] = {epic: name for name, epic in INSTRUMENTS.items()}


# Rate limits for Capital.com API
RATE_LIMITS = {
    'demo': {
        'requests_per_minute': 60,
        'description': 'Demo account - 60 requests per minute',
        'recommended_interval': 1.0,  # seconds between requests
    },
    'live': {
        'requests_per_minute': 60,  # Conservative estimate
        'description': 'Live account - Check documentation for current limits',
        'recommended_interval': 1.0,  # seconds between requests
    }
}


def get_epic(instrument: str) -> Optional[str]:
    """
    Get Capital.com EPIC code for a given instrument name.

    Args:
        instrument: Common instrument name (e.g., 'NAS100', 'EURUSD')

    Returns:
        Capital.com EPIC code, or None if not found

    Example:
        >>> get_epic('NAS100')
        'US_TECH_100'
        >>> get_epic('GOLD')
        'GOLD'
        >>> get_epic('UNKNOWN')
        None
    """
    return INSTRUMENTS.get(instrument.upper())


def get_name(epic: str) -> Optional[str]:
    """
    Get common name for a Capital.com EPIC code.

    Args:
        epic: Capital.com EPIC code (e.g., 'US_TECH_100')

    Returns:
        Common instrument name, or None if not found

    Example:
        >>> get_name('US_TECH_100')
        'NAS100'
        >>> get_name('GOLD')
        'GOLD'
    """
    return EPIC_TO_NAME.get(epic.upper())


def list_instruments(category: Optional[str] = None) -> List[str]:
    """
    List all available instruments, optionally filtered by category.

    Args:
        category: Optional category filter ('indices', 'commodities', 'forex', 'crypto')

    Returns:
        List of instrument names

    Example:
        >>> list_instruments('forex')
        ['EURUSD', 'GBPUSD', 'USDJPY', ...]
    """
    if category is None:
        return list(INSTRUMENTS.keys())

    category = category.lower()
    if category == 'indices':
        return [k for k in INSTRUMENTS.keys() if k in [
            'NAS100', 'NASDAQ', 'US500', 'SPX', 'US30', 'DOW',
            'GER40', 'DAX', 'UK100', 'FTSE', 'JPN225', 'NIKKEI'
        ]]
    elif category == 'commodities':
        return [k for k in INSTRUMENTS.keys() if k in [
            'XAUUSD', 'GOLD', 'XAGUSD', 'SILVER', 'WTIUSD', 'OIL', 'BRENT', 'NATGAS'
        ]]
    elif category == 'forex':
        return [k for k in INSTRUMENTS.keys() if 'USD' in k or 'EUR' in k or 'GBP' in k or 'JPY' in k]
    elif category == 'crypto':
        return [k for k in INSTRUMENTS.keys() if k in ['BTCUSD', 'BTC', 'ETHUSD', 'ETH']]
    else:
        return []


def get_rate_limit(environment: str = 'demo') -> Dict:
    """
    Get rate limit information for the specified environment.

    Args:
        environment: 'demo' or 'live'

    Returns:
        Dictionary with rate limit information

    Example:
        >>> limits = get_rate_limit('demo')
        >>> print(limits['requests_per_minute'])
        60
    """
    return RATE_LIMITS.get(environment, RATE_LIMITS['demo'])


def is_valid_epic(epic: str) -> bool:
    """
    Check if an EPIC code is valid.

    Args:
        epic: Capital.com EPIC code to validate

    Returns:
        True if valid, False otherwise

    Example:
        >>> is_valid_epic('US_TECH_100')
        True
        >>> is_valid_epic('INVALID_CODE')
        False
    """
    return epic.upper() in EPIC_TO_NAME or epic.upper() in INSTRUMENTS


# Export all mappings
__all__ = [
    'INSTRUMENTS',
    'EPIC_TO_NAME',
    'RATE_LIMITS',
    'get_epic',
    'get_name',
    'list_instruments',
    'get_rate_limit',
    'is_valid_epic',
]
