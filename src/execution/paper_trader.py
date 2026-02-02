from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.execution.broker_interface import (
    BrokerInterface, OrderResult, AccountInfo, Position
)


class PaperTrader(BrokerInterface):
    """
    Paper trading simulator.

    Simulates realistic trading without real money:
    - Slippage modeling
    - Spread costs
    - Fill delays
    - Market hours

    Perfect for testing strategies before live trading.

    Example:
        config = {
            'initial_balance': 10000,
            'currency': 'USD',
            'slippage_pct': 0.001,  # 0.1% slippage
            'spread_multiplier': 1.5  # 1.5x typical spread
        }

        trader = PaperTrader(config)
        trader.connect()
    """

    def __init__(self, config: dict):
        """
        Initialize paper trader

        Args:
            config: Configuration dict
        """
        super().__init__(config)

        # Account state
        self.balance = config.get('initial_balance', 10000)
        self.initial_balance = self.balance
        self.currency = config.get('currency', 'USD')

        # Open positions
        self.positions: Dict[str, dict] = {}

        # Closed trades
        self.trade_history = []

        # Simulation parameters
        self.slippage_pct = config.get('slippage_pct', 0.001)  # 0.1%
        self.spread_multiplier = config.get('spread_multiplier', 1.5)

        # Market data (for historical simulation)
        self.market_data: Dict[str, pd.DataFrame] = {}

        # Typical spreads (in points)
        self.typical_spreads = {
            'US_TECH_100': 2.0,
            'US_SPX_500': 0.7,
            'GOLD': 0.3,
            'EUR_USD': 0.00015,
            'GBP_USD': 0.00020
        }

    def connect(self) -> bool:
        """Connect (always succeeds for paper trading)"""
        self.is_connected = True
        print(f"[PAPER] Connected with ${self.balance:.2f} balance")
        return True

    def disconnect(self) -> None:
        """Disconnect"""
        self.is_connected = False
        print(f"[PAPER] Disconnected. Final balance: ${self.balance:.2f}")

    def get_account_info(self) -> AccountInfo:
        """Get simulated account info"""

        # Calculate equity (balance + unrealized P&L)
        unrealized_pnl = sum(
            pos['unrealized_pnl'] for pos in self.positions.values()
        )
        equity = self.balance + unrealized_pnl

        # Simple margin calculation (not exact)
        margin_used = sum(
            abs(pos['units'] * pos['current_price']) * 0.05  # 5% margin
            for pos in self.positions.values()
        )

        return AccountInfo(
            balance=self.balance,
            equity=equity,
            margin_used=margin_used,
            margin_available=equity - margin_used,
            unrealized_pnl=unrealized_pnl,
            currency=self.currency,
            account_id='PAPER_ACCOUNT'
        )

    def get_current_price(self, asset: str) -> Tuple[float, float]:
        """
        Get simulated current price

        Returns:
            (bid, ask) with realistic spread
        """

        # Get mid price from market data or generate
        if asset in self.market_data and not self.market_data[asset].empty:
            mid_price = self.market_data[asset]['close'].iloc[-1]
        else:
            # Generate random price (for testing)
            mid_price = 100.0 + np.random.randn() * 10

        # Add spread
        spread = self._get_spread(asset)
        bid = mid_price - (spread / 2)
        ask = mid_price + (spread / 2)

        return bid, ask

    def place_market_order(self,
                          asset: str,
                          direction: str,
                          units: float,
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> OrderResult:
        """
        Simulate market order placement

        Includes:
        - Realistic fill price (with slippage)
        - Spread cost
        - Fill confirmation
        """

        try:
            # Get current price
            bid, ask = self.get_current_price(asset)

            # Determine fill price (LONG buys at ask, SHORT sells at bid)
            if direction == 'LONG':
                base_price = ask
            else:
                base_price = bid

            # Add slippage
            slippage = base_price * self.slippage_pct
            if direction == 'LONG':
                fill_price = base_price + slippage
            else:
                fill_price = base_price - slippage

            # Check if position already exists
            if asset in self.positions:
                return OrderResult(
                    success=False,
                    order_id=None,
                    fill_price=None,
                    filled_units=0,
                    message=f"Position already open for {asset}",
                    timestamp=datetime.now(),
                    metadata=None
                )

            # Create position
            position = {
                'asset': asset,
                'direction': direction,
                'units': abs(units),
                'entry_price': fill_price,
                'current_price': fill_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'opened_at': datetime.now(),
                'unrealized_pnl': 0.0
            }

            self.positions[asset] = position

            order_id = f"PAPER_{asset}_{int(datetime.now().timestamp())}"

            print(f"[PAPER] {direction} {units} {asset} @ {fill_price:.2f}")

            return OrderResult(
                success=True,
                order_id=order_id,
                fill_price=fill_price,
                filled_units=abs(units),
                message="Order filled (simulated)",
                timestamp=datetime.now(),
                metadata={
                    'slippage': slippage,
                    'spread': ask - bid,
                    'direction': direction
                }
            )

        except Exception as e:
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                filled_units=0,
                message=f"Error: {str(e)}",
                timestamp=datetime.now(),
                metadata=None
            )

    def modify_position(self,
                       asset: str,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> bool:
        """Modify existing position"""

        if asset not in self.positions:
            print(f"[PAPER] No position found for {asset}")
            return False

        if stop_loss is not None:
            self.positions[asset]['stop_loss'] = stop_loss
            print(f"[PAPER] Updated SL for {asset}: {stop_loss:.2f}")

        if take_profit is not None:
            self.positions[asset]['take_profit'] = take_profit
            print(f"[PAPER] Updated TP for {asset}: {take_profit:.2f}")

        return True

    def close_position(self, asset: str, units: Optional[float] = None) -> OrderResult:
        """Close position (simulated)"""

        if asset not in self.positions:
            return OrderResult(
                success=False,
                order_id=None,
                fill_price=None,
                filled_units=0,
                message=f"No position found for {asset}",
                timestamp=datetime.now(),
                metadata=None
            )

        position = self.positions[asset]

        # Get close price (opposite of entry)
        bid, ask = self.get_current_price(asset)

        if position['direction'] == 'LONG':
            close_price = bid  # Sell at bid
        else:
            close_price = ask  # Buy to cover at ask

        # Add slippage
        slippage = close_price * self.slippage_pct
        if position['direction'] == 'LONG':
            close_price -= slippage
        else:
            close_price += slippage

        # Calculate P&L
        if position['direction'] == 'LONG':
            pnl = (close_price - position['entry_price']) * position['units']
        else:
            pnl = (position['entry_price'] - close_price) * position['units']

        # Update balance
        self.balance += pnl

        # Record trade
        self.trade_history.append({
            'asset': asset,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': close_price,
            'units': position['units'],
            'pnl': pnl,
            'opened_at': position['opened_at'],
            'closed_at': datetime.now()
        })

        # Remove position
        del self.positions[asset]

        print(f"[PAPER] Closed {asset} @ {close_price:.2f}, P&L: ${pnl:.2f}")

        return OrderResult(
            success=True,
            order_id=f"CLOSE_{asset}_{int(datetime.now().timestamp())}",
            fill_price=close_price,
            filled_units=position['units'],
            message=f"Position closed, P&L: ${pnl:.2f}",
            timestamp=datetime.now(),
            metadata={'pnl': pnl}
        )

    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""

        positions = []

        for asset, pos in self.positions.items():
            # Update current price and unrealized P&L
            bid, ask = self.get_current_price(asset)
            current_price = (bid + ask) / 2

            if pos['direction'] == 'LONG':
                unrealized_pnl = (current_price - pos['entry_price']) * pos['units']
            else:
                unrealized_pnl = (pos['entry_price'] - current_price) * pos['units']

            pos['current_price'] = current_price
            pos['unrealized_pnl'] = unrealized_pnl

            positions.append(Position(
                asset=asset,
                direction=pos['direction'],
                units=pos['units'],
                entry_price=pos['entry_price'],
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                opened_at=pos['opened_at']
            ))

        return positions

    def get_historical_data(self,
                           asset: str,
                           timeframe: str,
                           count: int) -> pd.DataFrame:
        """
        Get historical data (from stored data or generate)

        For real testing, you should load actual historical data.
        This generates random data for demonstration.
        """

        if asset in self.market_data:
            return self.market_data[asset].tail(count)

        # Generate random data (for testing only!)
        dates = pd.date_range(end=datetime.now(), periods=count, freq='1H')

        # Random walk
        returns = np.random.normal(0.0001, 0.01, count)
        prices = 100 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, count)),
            'high': prices * (1 + np.random.uniform(0, 0.01, count)),
            'low': prices * (1 - np.random.uniform(0, 0.01, count)),
            'close': prices,
            'volume': np.random.randint(1000, 10000, count)
        }, index=dates)

        return df

    def load_historical_data(self, asset: str, df: pd.DataFrame):
        """
        Load historical data for backtesting

        Args:
            asset: Instrument
            df: DataFrame with OHLCV data
        """
        self.market_data[asset] = df
        print(f"[PAPER] Loaded {len(df)} bars for {asset}")

    def _get_spread(self, asset: str) -> float:
        """Get simulated spread for asset"""

        # Find matching spread
        for key in self.typical_spreads:
            if key in asset or asset in key:
                base_spread = self.typical_spreads[key]
                return base_spread * self.spread_multiplier

        # Default
        return 1.0 * self.spread_multiplier

    def check_stop_loss_hit(self):
        """
        Check if any positions hit stop loss

        Call this periodically during simulation
        """

        for asset in list(self.positions.keys()):
            position = self.positions[asset]

            if position['stop_loss'] is None:
                continue

            bid, ask = self.get_current_price(asset)
            current_price = (bid + ask) / 2

            hit = False

            if position['direction'] == 'LONG':
                if current_price <= position['stop_loss']:
                    hit = True
            else:
                if current_price >= position['stop_loss']:
                    hit = True

            if hit:
                print(f"[PAPER] Stop loss hit for {asset}")
                self.close_position(asset)

    def get_performance_summary(self) -> dict:
        """Get trading performance summary"""

        if not self.trade_history:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'balance': self.balance
            }

        df = pd.DataFrame(self.trade_history)

        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]

        return {
            'total_trades': len(df),
            'total_pnl': df['pnl'].sum(),
            'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100
        }
