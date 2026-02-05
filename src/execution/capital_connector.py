import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from functools import wraps

from src.execution.broker_interface import (
    BrokerInterface, OrderResult, AccountInfo, Position
)


def retry_on_failure(max_attempts=3, delay=2, backoff=2):
    """
    Decorator for retry logic with exponential backoff

    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            attempt = 0
            current_delay = delay
            last_exception = None

            while attempt < max_attempts:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    attempt += 1
                    last_exception = e

                    if attempt >= max_attempts:
                        self._log_error(f"Failed after {max_attempts} attempts: {e}")
                        raise

                    self._log_warning(f"Attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        return wrapper
    return decorator


class CapitalConnector(BrokerInterface):
    """
    Capital.com broker implementation with robust error handling.

    Features:
    - Automatic retry on failures
    - Rate limiting
    - Connection health monitoring
    - Detailed logging

    Example:
        config = {
            'api_key': 'xxx',
            'identifier': 'email@example.com',
            'password': 'xxx',
            'environment': 'demo'
        }

        with CapitalConnector(config) as broker:
            account = broker.get_account_info()
            print(f"Balance: {account.balance}")
    """

    def __init__(self, config: dict):
        """
        Initialize Capital.com connector

        Args:
            config: Dict with api_key, identifier, password, environment
        """
        super().__init__(config)

        # API client (will be initialized on connect)
        self.client = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

        # Connection health
        self.connection_errors = 0
        self.max_connection_errors = 5

        # Logging
        self.log_events = []

    @retry_on_failure(max_attempts=1, delay=1)  # Reduced retries since auth errors won't change
    def connect(self) -> bool:
        """
        Connect to Capital.com API

        Returns:
            True if successful
        """
        try:
            # Import here to avoid import errors if library not installed
            from capitalcom import Client

            # Initialize client with credentials
            # Parameters: log (email/login), pas (password), api_key
            # Note: Client.__init__ automatically attempts login
            self.client = Client(
                log=self.config.get('identifier', 'demo@demo.com'),
                pas=self.config.get('password', 'demo'),
                api_key=self.config.get('api_key', 'demo')
            )

            # Check if login was successful by verifying cst token exists
            if not hasattr(self.client, 'cst') or not self.client.cst:
                # Login failed - check response for error details
                if hasattr(self.client, 'response'):
                    status = self.client.response.status_code
                    error_msg = 'Unknown error'

                    try:
                        error_data = self.client.response.json()
                        error_msg = error_data.get('errorCode', 'Unknown error')
                    except:
                        pass

                    if error_msg == 'error.invalid.details':
                        raise ConnectionError(
                            "Authentication failed: Invalid email/identifier or password.\n"
                            f"Please verify your Capital.com credentials:\n"
                            f"  - Email/Identifier: {self.config.get('identifier', 'NOT PROVIDED')}\n"
                            f"  - Password: {'*' * 10}\n"
                            f"  - API Key: {self.config.get('api_key', 'NOT PROVIDED')[:10]}...\n"
                            "\nMake sure you're using the correct email address associated with your Capital.com account."
                        )
                    elif error_msg == 'error.invalid.api.key':
                        raise ConnectionError(
                            "Authentication failed: Invalid API key.\n"
                            "Please verify your Capital.com API key is correct."
                        )
                    else:
                        raise ConnectionError(f"Authentication failed: {error_msg} (status {status})")
                else:
                    raise ConnectionError("Authentication failed: No session token received")

            # Login successful - test by getting accounts
            accounts = self.client.all_accounts()

            if accounts:
                self.is_connected = True
                self.connection_errors = 0
                self._log_info("✓ Connected to Capital.com successfully")

                # Log account info (safely)
                if 'accounts' in accounts and accounts['accounts']:
                    acc = accounts['accounts'][0]
                    self._log_info(f"  Account ID: {acc.get('accountId', 'N/A')}")
                    self._log_info(f"  Currency: {acc.get('currency', 'N/A')}")
                    self._log_info(f"  Balance: {acc.get('balance', 'N/A')}")

                return True
            else:
                raise ConnectionError("Connected but failed to get account information")

        except ConnectionError:
            # Re-raise connection errors as-is
            raise
        except Exception as e:
            self.connection_errors += 1
            self._log_error(f"Connection failed: {e}")
            raise ConnectionError(f"Failed to connect to Capital.com: {e}")

    def reconnect(self) -> bool:
        """
        Re-establish connection to Capital.com API.
        Creates a fresh Client with new session tokens.

        Returns:
            True if reconnection successful
        """
        self._log_info("Attempting to reconnect to Capital.com...")

        # Try to cleanly disconnect first
        if self.client:
            try:
                self.client.log_out_account()
            except:
                pass

        self.client = None
        self.is_connected = False
        self.connection_errors = 0

        try:
            return self.connect()
        except Exception as e:
            self._log_error(f"Reconnection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Capital.com"""
        if self.client:
            try:
                self.client.log_out_account()
                self._log_info("Disconnected from Capital.com")
            except:
                pass

        self.is_connected = False
        self.client = None

    def _check_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _check_connection_health(self):
        """Check if too many errors occurred"""
        if self.connection_errors >= self.max_connection_errors:
            raise ConnectionError(
                f"Too many connection errors ({self.connection_errors}). "
                "Connection may be unstable."
            )

    @retry_on_failure(max_attempts=2, delay=1)
    def get_account_info(self) -> AccountInfo:
        """Get Capital.com account information"""
        self._check_rate_limit()
        self._check_connection_health()

        try:
            accounts = self.client.all_accounts()

            # Get first account (usually the active one)
            if accounts and 'accounts' in accounts:
                info = accounts['accounts'][0] if accounts['accounts'] else {}
            else:
                info = accounts if isinstance(accounts, dict) else {}

            # Extract balance - it can be a dict or a float
            balance_data = info.get('balance', {})
            if isinstance(balance_data, dict):
                balance = float(balance_data.get('balance', 0))
                available = float(balance_data.get('available', 0))
                deposit = float(balance_data.get('deposit', 0))
                profit_loss = float(balance_data.get('profitLoss', 0))
            else:
                balance = float(balance_data)
                available = float(info.get('available', balance))
                deposit = float(info.get('deposit', 0))
                profit_loss = float(info.get('profitLoss', 0))

            return AccountInfo(
                balance=balance,
                equity=available,
                margin_used=deposit,
                margin_available=available,
                unrealized_pnl=profit_loss,
                currency=info.get('currency', 'USD'),
                account_id=info.get('accountId', '')
            )

        except Exception as e:
            self.connection_errors += 1
            self._log_error(f"Failed to get account info: {e}")
            raise

    @retry_on_failure(max_attempts=2, delay=1)
    def get_current_price(self, asset: str) -> Tuple[float, float]:
        """Get current bid/ask for asset"""
        self._check_rate_limit()

        try:
            market = self.client.single_market(asset)

            # Extract bid/ask from snapshot
            snapshot = market.get('snapshot', {})
            bid = float(snapshot.get('bid', 0))
            ask = float(snapshot.get('offer', 0))  # Capital.com uses 'offer'

            return bid, ask

        except Exception as e:
            self.connection_errors += 1
            self._log_error(f"Failed to get price for {asset}: {e}")
            raise

    @retry_on_failure(max_attempts=3, delay=2)
    def place_market_order(self,
                          asset: str,
                          direction: str,
                          units: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> OrderResult:
        """
        Place market order on Capital.com

        Args:
            asset: Instrument epic (e.g., "US_TECH_100")
            direction: 'LONG' or 'SHORT'
            units: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            OrderResult with execution details
        """
        self._check_rate_limit()

        try:
            # Import DirectionType enum
            from capitalcom.client_demo import DirectionType

            # Capital.com uses DirectionType.BUY/SELL
            cap_direction = DirectionType.BUY if direction == 'LONG' else DirectionType.SELL

            # Build order parameters
            kwargs = {
                'direction': cap_direction,
                'epic': asset,
                'size': abs(units),
                'gsl': False,
                'tsl': False
            }

            # Add stop loss if provided
            if stop_loss:
                kwargs['stop_level'] = stop_loss

            # Add take profit if provided
            if take_profit:
                kwargs['profit_level'] = take_profit

            # Execute order
            response = self.client.place_the_position(**kwargs)

            # Parse response - initial response only contains dealReference
            deal_ref = response.get('dealReference') if response else None

            if deal_ref:
                # Get confirmation (may need short delay)
                time.sleep(0.5)
                confirmation = self.client.position_order_confirmation(deal_ref)

                deal_status = confirmation.get('dealStatus', '')

                if deal_status == 'ACCEPTED':
                    return OrderResult(
                        success=True,
                        order_id=confirmation.get('dealId'),
                        fill_price=float(confirmation.get('level', 0)),
                        filled_units=float(confirmation.get('size', abs(units))),
                        message="Order filled",
                        timestamp=datetime.now(),
                        metadata={
                            'dealReference': deal_ref,
                            'direction': direction,
                            'asset': asset,
                            'stopLevel': confirmation.get('stopLevel'),
                            'profitLevel': confirmation.get('profitLevel')
                        }
                    )
                else:
                    reason = confirmation.get('reason', 'Unknown')
                    return OrderResult(
                        success=False,
                        order_id=None,
                        fill_price=None,
                        filled_units=0,
                        message=f"Order rejected: {reason}",
                        timestamp=datetime.now(),
                        metadata=confirmation
                    )
            else:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    filled_units=0,
                    message=f"Order rejected: No deal reference received",
                    timestamp=datetime.now(),
                    metadata=response
                )

        except Exception as e:
            self.connection_errors += 1
            self._log_error(f"Order placement failed: {e}")

            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                filled_units=0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                metadata=None
            )

    def place_limit_order(self,
                         asset: str,
                         direction: str,
                         units: float,
                         limit_price: float,
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None,
                         good_till_date: Optional[str] = None) -> OrderResult:
        """
        Place a limit order on Capital.com

        Args:
            asset: Instrument epic
            direction: 'LONG' or 'SHORT'
            units: Position size
            limit_price: The limit price to enter at
            stop_loss: Stop loss price
            take_profit: Take profit price
            good_till_date: Expiry datetime string (ISO format)

        Returns:
            OrderResult with order details
        """
        self._check_rate_limit()

        try:
            from capitalcom.client_demo import DirectionType, OrderType

            cap_direction = DirectionType.BUY if direction == 'LONG' else DirectionType.SELL

            kwargs = {
                'direction': cap_direction,
                'epic': asset,
                'size': abs(units),
                'level': limit_price,
                'type': OrderType.LIMIT,
                'gsl': False,
                'tsl': False
            }

            if stop_loss:
                kwargs['stop_level'] = stop_loss
            if take_profit:
                kwargs['profit_level'] = take_profit
            if good_till_date:
                kwargs['good_till_date'] = good_till_date

            response = self.client.place_the_order(**kwargs)
            deal_ref = response.get('dealReference') if response else None

            if deal_ref:
                time.sleep(0.5)
                confirmation = self.client.position_order_confirmation(deal_ref)
                deal_status = confirmation.get('dealStatus', '')

                if deal_status == 'ACCEPTED':
                    return OrderResult(
                        success=True,
                        order_id=confirmation.get('dealId'),
                        fill_price=limit_price,
                        filled_units=abs(units),
                        message="Limit order placed",
                        timestamp=datetime.now(),
                        metadata={
                            'dealReference': deal_ref,
                            'type': 'LIMIT',
                            'level': limit_price,
                            'direction': direction,
                            'asset': asset,
                            'goodTillDate': good_till_date,
                            'stopLevel': confirmation.get('stopLevel'),
                            'profitLevel': confirmation.get('profitLevel')
                        }
                    )
                else:
                    reason = confirmation.get('reason', 'Unknown')
                    return OrderResult(
                        success=False,
                        order_id=None,
                        fill_price=None,
                        filled_units=0,
                        message=f"Limit order rejected: {reason}",
                        timestamp=datetime.now(),
                        metadata=confirmation
                    )
            else:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    filled_units=0,
                    message="Limit order rejected: No deal reference",
                    timestamp=datetime.now(),
                    metadata=response
                )

        except Exception as e:
            self.connection_errors += 1
            self._log_error(f"Limit order placement failed: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                filled_units=0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                metadata=None
            )

    def cancel_order(self, deal_id: str) -> bool:
        """Cancel a pending order"""
        try:
            self.client.close_order(deal_id)
            self._log_info(f"Cancelled order {deal_id}")
            return True
        except Exception as e:
            self._log_error(f"Failed to cancel order {deal_id}: {e}")
            return False

    @retry_on_failure(max_attempts=2, delay=1)
    def modify_position(self,
                       asset: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """Modify existing position's stop/target"""
        self._check_rate_limit()

        try:
            # Get current position to find deal ID
            positions_data = self.client.all_positions()
            positions = positions_data.get('positions', [])

            position = next(
                (p for p in positions if p.get('market', {}).get('epic') == asset),
                None
            )

            if not position:
                self._log_warning(f"No open position found for {asset}")
                return False

            deal_id = position.get('dealId')

            # Build update kwargs
            kwargs = {'dealid': deal_id, 'gsl': False, 'tsl': False}

            if stop_loss is not None:
                kwargs['stop_level'] = stop_loss

            if take_profit is not None:
                kwargs['profit_level'] = take_profit

            # Update via API
            response = self.client.update_the_position(**kwargs)

            return response.get('status') == 'SUCCESS'

        except Exception as e:
            self._log_error(f"Position modification failed: {e}")
            return False

    @retry_on_failure(max_attempts=3, delay=1)
    def close_position(self, asset: str, units: Optional[float] = None) -> OrderResult:
        """Close position (full or partial)"""
        self._check_rate_limit()

        try:
            # Get current position
            positions_data = self.client.all_positions()
            positions = positions_data.get('positions', [])

            position = next(
                (p for p in positions if p.get('market', {}).get('epic') == asset),
                None
            )

            if not position:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    filled_units=0,
                    message=f"No position found for {asset}",
                    timestamp=datetime.now(),
                    metadata=None
                )

            deal_id = position.get('dealId')

            # Close the position using deal ID
            response = self.client.close_position(deal_id)

            if response and response.get('dealStatus') == 'ACCEPTED':
                deal_ref = response.get('dealReference')
                time.sleep(0.5)
                confirmation = self.client.position_order_confirmation(deal_ref)

                return OrderResult(
                    success=True,
                    order_id=confirmation.get('dealId'),
                    fill_price=float(confirmation.get('level', 0)),
                    filled_units=float(position.get('size', 0)),
                    message="Position closed",
                    timestamp=datetime.now(),
                    metadata=confirmation
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    filled_units=0,
                    message=f"Close failed: {response.get('reason')}",
                    timestamp=datetime.now(),
                    metadata=response
                )

        except Exception as e:
            self._log_error(f"Close position failed: {e}")
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                filled_units=0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                metadata=None
            )

    @retry_on_failure(max_attempts=2, delay=1)
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        self._check_rate_limit()

        try:
            positions_data = self.client.all_positions()

            positions = []
            for pos in positions_data.get('positions', []):
                positions.append(Position(
                    asset=pos.get('market', {}).get('epic', ''),
                    direction='LONG' if pos.get('direction') == 'BUY' else 'SHORT',
                    units=float(pos.get('size', 0)),
                    entry_price=float(pos.get('level', 0)),
                    current_price=float(pos.get('market', {}).get('bid', 0)),
                    unrealized_pnl=float(pos.get('profit', 0)),
                    opened_at=datetime.fromisoformat(pos.get('createdDate', ''))
                ))

            return positions

        except Exception as e:
            self._log_error(f"Failed to get positions: {e}")
            return []

    @retry_on_failure(max_attempts=2, delay=1)
    def get_historical_data(self,
                           asset: str,
                           timeframe: str,
                           count: int) -> pd.DataFrame:
        """
        Get historical price data

        Args:
            asset: Epic
            timeframe: '1H', '15m', etc
            count: Number of bars

        Returns:
            DataFrame with OHLCV
        """
        self._check_rate_limit()

        try:
            # Import ResolutionType enum
            from capitalcom.client_demo import ResolutionType

            # Capital.com timeframe mapping to ResolutionType
            tf_map = {
                '1m': ResolutionType.MINUTE,
                '5m': ResolutionType.MINUTE_5,
                '15m': ResolutionType.MINUTE_15,
                '1H': ResolutionType.HOUR,
                '4H': ResolutionType.HOUR_4,
                '1D': ResolutionType.DAY
            }

            resolution = tf_map.get(timeframe, ResolutionType.HOUR)

            # Fetch data using historical_price method
            data = self.client.historical_price(
                epic=asset,
                resolution=resolution,
                max=count
            )

            # Reset error counter on successful data fetch
            self.connection_errors = 0

            # Convert to DataFrame
            df = pd.DataFrame(data.get('prices', []))

            if df.empty:
                return pd.DataFrame()

            # Extract prices from bid/ask dictionaries
            # Capital.com returns prices as {'bid': X, 'ask': Y}
            # We'll use the mid price (average of bid and ask)
            if 'openPrice' in df.columns and isinstance(df['openPrice'].iloc[0], dict):
                df['open'] = df['openPrice'].apply(lambda x: (x['bid'] + x['ask']) / 2 if isinstance(x, dict) else x)
                df['high'] = df['highPrice'].apply(lambda x: (x['bid'] + x['ask']) / 2 if isinstance(x, dict) else x)
                df['low'] = df['lowPrice'].apply(lambda x: (x['bid'] + x['ask']) / 2 if isinstance(x, dict) else x)
                df['close'] = df['closePrice'].apply(lambda x: (x['bid'] + x['ask']) / 2 if isinstance(x, dict) else x)
                df['volume'] = df['lastTradedVolume']
            else:
                # Fallback for non-dict format
                df = df.rename(columns={
                    'openPrice': 'open',
                    'highPrice': 'high',
                    'lowPrice': 'low',
                    'closePrice': 'close',
                    'lastTradedVolume': 'volume'
                })

            # Determine which time column to use
            time_col = 'snapshotTimeUTC' if 'snapshotTimeUTC' in df.columns else 'snapshotTime'
            df['time'] = pd.to_datetime(df[time_col])

            # Remove duplicates (keep last entry for each timestamp)
            df = df.drop_duplicates(subset=['time'], keep='last')

            # Set index
            df = df.set_index('time')

            # Select OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # Ensure all are float (already extracted from dicts, so should be clean)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col].astype(float)
            df['volume'] = df['volume'].astype(int)

            # Sort index (ensure chronological order)
            df = df.sort_index()

            return df

        except Exception as e:
            self._log_error(f"Failed to get historical data: {e}")
            return pd.DataFrame()

    def is_market_open(self, asset: str) -> bool:
        """Check if market is open"""
        try:
            market_info = self.client.single_market(asset)
            status = market_info.get('snapshot', {}).get('marketStatus')
            return status == 'TRADEABLE'
        except:
            return True  # Default to assume open

    def _log_info(self, message: str):
        """Log info message"""
        self.log_events.append({
            'level': 'INFO',
            'message': message,
            'timestamp': datetime.now()
        })
        print(f"[INFO] {message}")

    def _log_warning(self, message: str):
        """Log warning message"""
        self.log_events.append({
            'level': 'WARNING',
            'message': message,
            'timestamp': datetime.now()
        })
        print(f"[WARNING] {message}")

    def _log_error(self, message: str):
        """Log error message"""
        self.log_events.append({
            'level': 'ERROR',
            'message': message,
            'timestamp': datetime.now()
        })
        print(f"[ERROR] {message}")
