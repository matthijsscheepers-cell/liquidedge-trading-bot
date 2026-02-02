"""
Paper Trading Engine - Live TTM Pullback Strategy

Uses Capital.com demo account for paper trading with:
- Live market data (15min + 1H)
- TTM Squeeze Pullback Strategy
- Progressive Risk Cap position sizing
- Real-time monitoring and alerts

Usage:
    python paper_trading_engine.py
"""

import sys
import os
sys.path.insert(0, '.')

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from dotenv import load_dotenv

from src.execution.capital_connector import CapitalConnector
from src.strategies.ttm_pullback import TTMSqueezePullbackStrategy
from src.indicators.trend import calculate_ema
from src.indicators.volatility import calculate_atr
from src.indicators.ttm import calculate_ttm_squeeze_pinescript

# Load environment variables
load_dotenv()

# =====================================================
# CONFIGURATION
# =====================================================

# Capital.com API config (from .env)
CAPITAL_CONFIG = {
    'api_key': os.getenv('CAPITAL_API_KEY'),
    'identifier': os.getenv('CAPITAL_IDENTIFIER'),
    'password': os.getenv('CAPITAL_PASSWORD'),
    'environment': os.getenv('CAPITAL_ENVIRONMENT', 'demo')
}

# Trading config
INITIAL_CAPITAL = 300.0
MAX_POSITIONS = 4
LEVERAGE = 20
COMMISSION_PCT = 0.001
RISK_PER_TRADE = 0.02  # 2% base

# Progressive Risk Cap
def get_risk_cap(capital: float) -> float:
    """Return progressive risk cap based on capital"""
    if capital < 1000:
        return 50.0
    elif capital < 5000:
        return 100.0
    elif capital < 20000:
        return 200.0
    elif capital < 100000:
        return 500.0
    else:
        return 1000.0

# Asset mapping (Capital.com epics)
ASSETS = {
    'GOLD': 'GOLD',           # XAU/USD
    'SILVER': 'SILVER',       # XAG/USD
    'US100': 'US100',         # Nasdaq 100
    'US500': 'US500'          # S&P 500
}

# Monitoring intervals
CHECK_INTERVAL = 15  # Check for setups every 15 seconds
POSITION_CHECK_INTERVAL = 15  # Check positions every 15 seconds

# =====================================================
# PAPER TRADING ENGINE
# =====================================================

class PaperTradingEngine:
    """
    Paper trading engine for TTM Pullback Strategy
    """

    def __init__(self, broker: CapitalConnector):
        self.broker = broker
        self.capital = INITIAL_CAPITAL
        self.positions = {}  # {asset: position_dict}
        self.trades = []
        self.strategies = {
            asset: TTMSqueezePullbackStrategy(asset)
            for asset in ASSETS.keys()
        }

        # Runtime state
        self.is_running = False
        self.last_check_time = {}
        self.start_time = datetime.now()
        self.pending_entries = set()  # Assets with pending/active orders
        self.pending_orders = {}  # {asset: {'order_id': str, 'expires': datetime, 'setup': setup}}

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def get_live_data(self, asset: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
        """Fetch live historical data from Capital.com"""
        try:
            epic = ASSETS[asset]
            df = self.broker.get_historical_data(epic, timeframe, bars)

            if df.empty:
                self.log(f"No data received for {asset} {timeframe}", "WARNING")
                return pd.DataFrame()

            return df

        except Exception as e:
            self.log(f"Error fetching data for {asset} {timeframe}: {e}", "ERROR")
            return pd.DataFrame()

    def add_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add TTM indicators to dataframe"""
        if df.empty or len(df) < 50:
            return df

        # EMA and ATR
        df['ema_21'] = calculate_ema(df['close'], period=21)
        df['atr_20'] = calculate_atr(df['high'], df['low'], df['close'], period=20)

        # TTM Squeeze
        squeeze_on, momentum, color = calculate_ttm_squeeze_pinescript(
            df['high'], df['low'], df['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20, kc_multiplier=2.0,
            momentum_period=20
        )

        df['squeeze_on'] = squeeze_on
        df['ttm_momentum'] = momentum

        return df

    def check_for_setups(self):
        """Scan all assets for entry setups"""
        if len(self.positions) >= MAX_POSITIONS:
            return  # Portfolio full

        # Sync positions with broker to prevent duplicates
        self._sync_positions_with_broker()

        for asset, strategy in self.strategies.items():
            if asset in self.positions:
                continue  # Already in position
            if asset in self.pending_entries:
                continue  # Order already pending

            try:
                # Fetch live data
                df_15m = self.get_live_data(asset, '15m', 100)
                df_1h = self.get_live_data(asset, '1H', 100)

                if df_15m.empty or df_1h.empty:
                    # Market closed or no data available - skip silently
                    continue

                # Add indicators
                df_15m = self.add_indicators(df_15m, '15m')
                df_1h = self.add_indicators(df_1h, '1H')

                # Need minimum bars for indicators
                if len(df_15m) < 50 or len(df_1h) < 50:
                    continue

                # Verify data is fresh (not older than 30 minutes)
                # Capital.com returns UTC timestamps (tz-naive)
                latest_bar_time = df_15m.index[-1]
                now_utc = pd.Timestamp.now('UTC').tz_localize(None)
                data_age_minutes = (now_utc - latest_bar_time).total_seconds() / 60
                if data_age_minutes > 30:
                    self.log(f"Skipping {asset} - data is {data_age_minutes:.0f}min stale", "WARNING")
                    continue

                # Check for entry
                setup = strategy.check_entry(df_15m, df_1h, regime='ANY', confidence=80.0)

                if setup and setup.direction.value == 'LONG':
                    self.execute_entry(asset, setup, df_15m.iloc[-1])

            except Exception as e:
                self.log(f"Error scanning {asset}: {e}", "ERROR")
                continue

    def _sync_positions_with_broker(self):
        """Check broker for open positions to prevent duplicates"""
        try:
            broker_positions = self.broker.client.all_positions()
            if broker_positions and 'positions' in broker_positions:
                for pos in broker_positions['positions']:
                    epic = pos['market']['epic']
                    # Find which asset key maps to this epic
                    for asset_name, asset_epic in ASSETS.items():
                        if asset_epic == epic and asset_name not in self.positions:
                            p = pos['position']
                            self.positions[asset_name] = {
                                'entry_time': datetime.now(),
                                'entry_price': float(p.get('level', 0)),
                                'stop_loss': float(p.get('stopLevel', 0)) if p.get('stopLevel') else 0,
                                'target': float(p.get('profitLevel', 0)) if p.get('profitLevel') else 0,
                                'size': float(p.get('size', 0)),
                                'margin': 0,
                                'risk_cap': get_risk_cap(self.capital),
                                'order_id': p.get('dealId', '')
                            }
                            self.log(f"Synced broker position: {asset_name} @ ${p.get('level', 0)}")
                            break
        except Exception as e:
            self.log(f"Error syncing positions: {e}", "ERROR")

    def execute_entry(self, asset: str, setup, current_bar):
        """Execute entry using limit order at calculated pullback level"""
        # Lock this asset immediately to prevent duplicate orders
        self.pending_entries.add(asset)

        try:
            # Calculate position size with progressive cap
            current_risk_cap = get_risk_cap(self.capital)
            risk_amount = min(self.capital * RISK_PER_TRADE, current_risk_cap)
            stop_distance = setup.entry_price - setup.stop_loss

            if stop_distance <= 0:
                self.log(f"Invalid stop distance for {asset}: {stop_distance:.2f}", "ERROR")
                self.pending_entries.discard(asset)
                return

            position_size = risk_amount / stop_distance
            position_value = position_size * setup.entry_price
            margin_required = position_value / LEVERAGE

            # Check if we have enough margin
            if margin_required > self.capital:
                self.log(f"Insufficient margin for {asset} entry (need ${margin_required:.2f})", "WARNING")
                self.pending_entries.discard(asset)
                return

            # Get real-time price for smart order routing
            epic = ASSETS[asset]
            try:
                market_info = self.broker.client.single_market(epic)
                current_bid = float(market_info['snapshot']['bid'])
                current_offer = float(market_info['snapshot']['offer'])
                spread = current_offer - current_bid
                self.log(f"Real-time {asset}: Bid=${current_bid:.2f} Offer=${current_offer:.2f} Spread=${spread:.2f}")
            except Exception as e:
                self.log(f"Can't get real-time price for {asset}: {e}", "ERROR")
                self.pending_entries.discard(asset)
                return

            entry_price = setup.entry_price
            price_diff = current_offer - entry_price
            price_diff_pct = (price_diff / entry_price) * 100

            self.log(f"Setup: Entry=${entry_price:.2f} | Current=${current_offer:.2f} | Gap={price_diff_pct:.2f}%")

            # Decision: limit order vs market order
            if current_offer <= entry_price * 1.001:
                # Price is AT or BELOW entry level (within 0.1%) → market order
                self.log(f"Price at/below entry level → MARKET ORDER")
                result = self.broker.place_market_order(
                    asset=epic,
                    direction='LONG',
                    units=position_size,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.target
                )

                if result.success:
                    slippage_pct = abs(result.fill_price - entry_price) / entry_price * 100
                    self.positions[asset] = {
                        'entry_time': datetime.now(),
                        'entry_price': result.fill_price,
                        'stop_loss': setup.stop_loss,
                        'target': setup.target,
                        'size': position_size,
                        'margin': margin_required,
                        'risk_cap': current_risk_cap,
                        'order_id': result.order_id
                    }
                    self.log(f"✓ MARKET ENTRY: {asset} @ ${result.fill_price:.2f} (slippage: {slippage_pct:.2f}%)", "SUCCESS")
                    self.log(f"  Size: {position_size:.4f} | Stop: ${setup.stop_loss:.2f} | Target: ${setup.target:.2f}")
                else:
                    self.log(f"✗ MARKET ENTRY FAILED: {asset} - {result.message}", "ERROR")
                    self.pending_entries.discard(asset)

            else:
                # Price is ABOVE entry level → LIMIT ORDER (wait for pullback)
                # Set expiry to 15 minutes from now (one bar)
                expiry_time = datetime.utcnow() + timedelta(minutes=15)
                expiry_str = expiry_time.strftime('%Y-%m-%dT%H:%M:%S')

                self.log(f"Price above entry (+{price_diff_pct:.2f}%) → LIMIT ORDER @ ${entry_price:.2f}")
                self.log(f"  Expires: {expiry_str} UTC | Stop: ${setup.stop_loss:.2f} | Target: ${setup.target:.2f}")

                result = self.broker.place_limit_order(
                    asset=epic,
                    direction='LONG',
                    units=position_size,
                    limit_price=entry_price,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.target,
                    good_till_date=expiry_str
                )

                if result.success:
                    self.pending_orders[asset] = {
                        'order_id': result.order_id,
                        'entry_price': entry_price,
                        'stop_loss': setup.stop_loss,
                        'target': setup.target,
                        'size': position_size,
                        'expires': expiry_time,
                        'placed_at': datetime.now()
                    }
                    self.log(f"✓ LIMIT ORDER PLACED: {asset} @ ${entry_price:.2f} (waiting for pullback)", "SUCCESS")
                else:
                    self.log(f"✗ LIMIT ORDER FAILED: {asset} - {result.message}", "ERROR")
                    self.pending_entries.discard(asset)

        except Exception as e:
            self.log(f"Error executing entry for {asset}: {e}", "ERROR")
            self.pending_entries.discard(asset)

    def check_pending_orders(self):
        """Check if pending limit orders have been filled or need cancelling"""
        if not self.pending_orders:
            return

        orders_to_remove = []

        for asset, order in self.pending_orders.items():
            try:
                # Check if order has been filled (position now exists on broker)
                broker_data = self.broker.client.all_positions()
                epic = ASSETS[asset]
                filled = False

                if broker_data and 'positions' in broker_data:
                    for bp in broker_data['positions']:
                        if bp['market']['epic'] == epic:
                            # Order was filled!
                            p = bp['position']
                            fill_price = float(p.get('level', 0))
                            slippage_pct = abs(fill_price - order['entry_price']) / order['entry_price'] * 100

                            self.positions[asset] = {
                                'entry_time': datetime.now(),
                                'entry_price': fill_price,
                                'stop_loss': order['stop_loss'],
                                'target': order['target'],
                                'size': order['size'],
                                'margin': 0,
                                'risk_cap': get_risk_cap(self.capital),
                                'order_id': p.get('dealId', '')
                            }
                            self.log(f"✓ LIMIT ORDER FILLED: {asset} @ ${fill_price:.2f} (slippage: {slippage_pct:.3f}%)", "SUCCESS")
                            orders_to_remove.append(asset)
                            filled = True
                            break

                if filled:
                    continue

                # Check if order expired
                now_utc = datetime.utcnow()
                if now_utc > order['expires']:
                    self.log(f"Limit order expired: {asset} (price didn't reach ${order['entry_price']:.2f})", "INFO")
                    # Try to cancel on broker side
                    if order.get('order_id'):
                        self.broker.cancel_order(order['order_id'])
                    orders_to_remove.append(asset)
                    self.pending_entries.discard(asset)

            except Exception as e:
                self.log(f"Error checking pending order for {asset}: {e}", "ERROR")

        # Clean up
        for asset in orders_to_remove:
            if asset in self.pending_orders:
                del self.pending_orders[asset]

    def check_positions(self):
        """Check all open positions for exits"""
        if not self.positions:
            return

        # Get current positions from broker using raw API
        try:
            broker_data = self.broker.client.all_positions()
            broker_positions = {}
            if broker_data and 'positions' in broker_data:
                for bp in broker_data['positions']:
                    epic = bp['market']['epic']
                    broker_positions[epic] = bp

            positions_to_close = []

            for asset, pos in self.positions.items():
                epic = ASSETS[asset]

                # Check if position still exists on broker
                if epic not in broker_positions:
                    self.log(f"Position {asset} closed (stop/target hit or manual close)", "INFO")
                    positions_to_close.append(asset)
                    # Update capital from broker balance
                    try:
                        account = self.broker.get_account_info()
                        self.capital = account.balance
                    except:
                        pass
                    continue

                # Position still open - get current price
                bp = broker_positions[epic]
                market = bp['market']
                position = bp['position']
                current_bid = float(market.get('bid', 0))
                upl = float(position.get('upl', 0))

                # Log position status periodically (handled by status print)

            # Remove closed positions from tracking
            for asset in positions_to_close:
                if asset in self.positions:
                    del self.positions[asset]
                    self.pending_entries.discard(asset)

        except Exception as e:
            self.log(f"Error checking positions: {e}", "ERROR")

    def print_status(self):
        """Print current portfolio status"""
        print("\n" + "=" * 70)
        print("PORTFOLIO STATUS")
        print("=" * 70)

        # Get account info from broker
        try:
            account = self.broker.get_account_info()
            print(f"Balance:     ${account.balance:,.2f}")
            print(f"Equity:      ${account.equity:,.2f}")
            print(f"Margin Used: ${account.margin_used:,.2f}")
            print(f"P&L:         ${account.unrealized_pnl:,.2f}")
        except:
            print(f"Capital:     ${self.capital:,.2f}")

        print(f"Positions:   {len(self.positions)}/{MAX_POSITIONS}")
        print(f"Pending:     {len(self.pending_orders)} limit orders")
        print(f"Risk Cap:    ${get_risk_cap(self.capital):.0f}")
        print()

        # Open positions
        if self.positions:
            print("OPEN POSITIONS:")
            for asset, pos in self.positions.items():
                print(f"  {asset}:")
                print(f"    Entry: ${pos['entry_price']:.2f}")
                print(f"    Stop:  ${pos['stop_loss']:.2f}")
                print(f"    Target: ${pos['target']:.2f}")
                print()

        # Pending limit orders
        if self.pending_orders:
            print("PENDING LIMIT ORDERS:")
            for asset, order in self.pending_orders.items():
                remaining = (order['expires'] - datetime.utcnow()).total_seconds()
                print(f"  {asset}:")
                print(f"    Limit: ${order['entry_price']:.2f}")
                print(f"    Expires in: {max(0, remaining):.0f}s")
                print()

        # Runtime stats
        uptime = datetime.now() - self.start_time
        print(f"Uptime: {uptime}")
        print("=" * 70)
        print()

    def run(self):
        """Main trading loop"""
        self.log("=" * 70)
        self.log("PAPER TRADING ENGINE - STARTING")
        self.log("=" * 70)
        self.log(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
        self.log(f"Assets: {', '.join(ASSETS.keys())}")
        self.log(f"Environment: {CAPITAL_CONFIG['environment'].upper()}")
        self.log("")

        # Check which markets are currently operational
        self.log("Checking market availability...")
        operational_markets = []
        for asset in ASSETS.keys():
            df = self.get_live_data(asset, '15m', 10)
            if not df.empty:
                operational_markets.append(asset)
                self.log(f"  ✓ {asset} - OPERATIONAL")
            else:
                self.log(f"  ✗ {asset} - Closed (will monitor for open)")

        if not operational_markets:
            self.log("No markets currently operational - waiting for markets to open...", "WARNING")
        else:
            self.log(f"Monitoring {len(operational_markets)} operational markets")

        self.log("")

        self.is_running = True
        last_setup_check = time.time()
        last_position_check = time.time()
        last_status_print = time.time()

        try:
            while self.is_running:
                current_time = time.time()

                # Check for new setups
                if current_time - last_setup_check >= CHECK_INTERVAL:
                    self.log("Scanning for entry setups...")
                    self.check_for_setups()
                    last_setup_check = current_time

                # Check positions and pending orders
                if current_time - last_position_check >= POSITION_CHECK_INTERVAL:
                    self.check_pending_orders()
                    self.check_positions()
                    last_position_check = current_time

                # Print status every 5 minutes
                if current_time - last_status_print >= 300:
                    self.print_status()
                    last_status_print = current_time

                time.sleep(5)  # Sleep 5 seconds between cycles

        except KeyboardInterrupt:
            self.log("\nShutdown requested by user", "WARNING")
            self.stop()
        except Exception as e:
            self.log(f"Critical error in trading loop: {e}", "ERROR")
            self.stop()

    def stop(self):
        """Stop trading engine"""
        self.is_running = False
        self.log("=" * 70)
        self.log("PAPER TRADING ENGINE - STOPPING")
        self.log("=" * 70)
        self.print_status()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TTM PULLBACK PAPER TRADING ENGINE")
    print("=" * 70)
    print()

    # Validate config
    if not CAPITAL_CONFIG['api_key']:
        print("ERROR: CAPITAL_API_KEY not found in .env file")
        sys.exit(1)

    # Connect to Capital.com
    print("Connecting to Capital.com...")
    broker = CapitalConnector(CAPITAL_CONFIG)

    try:
        broker.connect()

        # Initialize engine
        engine = PaperTradingEngine(broker)

        # Print initial status
        engine.print_status()

        # Start trading
        print("\nPress Ctrl+C to stop\n")
        engine.run()

    except ConnectionError as e:
        print(f"\n✗ Connection Error: {e}")
        print("\nPlease verify your Capital.com credentials in .env file")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Disconnect
        if broker.is_connected:
            broker.disconnect()
            print("\nDisconnected from Capital.com")
