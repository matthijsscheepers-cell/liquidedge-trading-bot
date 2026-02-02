"""
Session filtering for liquid trading hours.

Based on QuantConnect HourlyGoldMomentumAlpha strategy - only trade during
peak liquidity hours when spreads are tightest and momentum is most reliable.
"""

from datetime import time
import pandas as pd


def is_liquid_session(timestamp: pd.Timestamp, asset: str = 'GOLD') -> bool:
    """
    Check if timestamp falls within liquid trading session.

    Args:
        timestamp: The bar timestamp (UTC assumed)
        asset: Asset symbol (GOLD, SILVER, US100, US500)

    Returns:
        True if within liquid session, False otherwise

    Notes:
        GOLD/SILVER: London/NY overlap (13:00-17:00 UTC / 8 AM - 12 PM EST)
                     + NY afternoon (17:00-21:00 UTC / 1 PM - 5 PM EST)
        US100/US500: Regular trading hours (14:30-21:00 UTC / 9:30 AM - 4 PM EST)
    """

    if asset in ['GOLD', 'SILVER']:
        # Gold trades best during London/NY overlap and NY afternoon
        # Peak liquidity: 13:00-21:00 UTC (8 AM - 5 PM EST)
        hour = timestamp.hour

        # Avoid Asian session (low liquidity, wide spreads)
        # Trade only during London/NY sessions
        return 13 <= hour < 21

    elif asset in ['US100', 'US500']:
        # US indices: Regular trading hours
        # 14:30-21:00 UTC (9:30 AM - 4 PM EST)
        hour = timestamp.hour
        minute = timestamp.minute

        # Market opens at 14:30 UTC
        if hour == 14 and minute < 30:
            return False

        # Market closes at 21:00 UTC
        return 14 <= hour < 21

    else:
        # Default: trade during NY session
        return 13 <= timestamp.hour < 21


def get_session_name(timestamp: pd.Timestamp) -> str:
    """
    Get the name of the current trading session.

    Args:
        timestamp: The bar timestamp (UTC assumed)

    Returns:
        Session name: 'ASIAN', 'LONDON', 'NY_OVERLAP', 'NY_AFTERNOON', 'CLOSED'
    """
    hour = timestamp.hour

    # UTC session times:
    # Asian: 00:00-08:00 (Tokyo/Sydney)
    # London: 08:00-13:00 (London open, before NY)
    # NY Overlap: 13:00-17:00 (London + NY - MOST LIQUID)
    # NY Afternoon: 17:00-21:00 (NY only)
    # Closed: 21:00-00:00

    if 0 <= hour < 8:
        return 'ASIAN'
    elif 8 <= hour < 13:
        return 'LONDON'
    elif 13 <= hour < 17:
        return 'NY_OVERLAP'  # Peak liquidity
    elif 17 <= hour < 21:
        return 'NY_AFTERNOON'
    else:
        return 'CLOSED'


def is_news_blackout(timestamp: pd.Timestamp) -> bool:
    """
    Check if timestamp falls within news blackout period.

    Major economic news releases (NFP, FOMC, CPI) cause erratic price action.
    Avoid trading 30 min before and after major announcements.

    Args:
        timestamp: The bar timestamp (UTC assumed)

    Returns:
        True if within blackout period, False otherwise

    Notes:
        Common release times (EST → UTC):
        - 8:30 AM EST = 13:30 UTC (NFP, CPI, Jobless Claims)
        - 10:00 AM EST = 15:00 UTC (ISM, Consumer Sentiment)
        - 2:00 PM EST = 19:00 UTC (FOMC announcements)

        Blackout windows:
        - 13:00-14:00 UTC (8:00-9:00 AM EST)
        - 14:30-15:30 UTC (9:30-10:30 AM EST)
        - 18:30-19:30 UTC (1:30-2:30 PM EST)
    """
    hour = timestamp.hour
    minute = timestamp.minute

    # 8:30 AM EST releases (13:30 UTC) - blackout 13:00-14:00
    if hour == 13:
        return True

    # 10:00 AM EST releases (15:00 UTC) - blackout 14:30-15:30
    if hour == 14 and minute >= 30:
        return True
    if hour == 15 and minute < 30:
        return True

    # 2:00 PM EST releases (19:00 UTC) - blackout 18:30-19:30
    if hour == 18 and minute >= 30:
        return True
    if hour == 19 and minute < 30:
        return True

    return False
