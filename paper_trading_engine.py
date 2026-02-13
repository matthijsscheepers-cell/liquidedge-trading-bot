"""
Paper Trading Engine - Live EMA Pullback Strategy

Uses Capital.com demo account for paper trading with:
- Live market data (15min + 1H)
- EMA Pullback Strategy (GOLD + SILVER + US500)
- 1H trend filter: Close > EMA(21)
- Asset-weighted risk: GOLD 2%, SILVER 1%, US500 1%
- Scale A tiers (2% -> 1% -> 0.5%)
- Real-time monitoring and alerts

Usage:
    python paper_trading_engine.py
"""

import sys
import os
sys.path.insert(0, '.')

import time
import pandas as pd
from datetime import datetime, timedelta, timezone
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
MAX_POSITIONS = 3  # One per asset
LEVERAGE = 20
COMMISSION_PCT = 0.001

# Scaling Risk % (Scale A)
# 2% until $1M, 1% until $10M, 0.5% above $10M
SCALING_TIERS = [
    (1_000_000, 0.02),
    (10_000_000, 0.01),
    (float('inf'), 0.005),
]

def get_scaling_risk_pct(capital: float) -> float:
    """Return risk percentage based on scaling tiers (uses GOLD's 2% base)"""
    for threshold, pct in SCALING_TIERS:
        if capital < threshold:
            return pct
    return SCALING_TIERS[-1][1]

def get_asset_risk_pct(asset: str, capital: float) -> float:
    """Return effective risk % for an asset, applying Scale A tiers.

    Combines per-asset base risk with Scale A tier caps.
    Example: SILVER base=1%, tier=2% → use 1%. tier=1% → use 1%. tier=0.5% → use 0.5%.
    """
    base_risk = ASSET_RISK.get(asset, 0.02)
    tier_cap = get_scaling_risk_pct(capital)
    return min(base_risk, tier_cap)

# Asset mapping (Capital.com epics)
# Close > EMA(21) filter, combined portfolio (2026-02-12):
#   GOLD: 89.0% WR, PF 24.94 | SILVER: 80.0% WR, PF 6.33 | US500: 70.5% WR, PF 7.06
#   Combined: 79.9% WR, -31.8% DD, $289B (backtest $3K start)
# US100 removed: 59% WR, blown account
ASSETS = {
    'GOLD': 'GOLD',           # XAU/USD
    'SILVER': 'SILVER',       # XAG/USD
    'US500': 'US500',         # S&P 500
}

# Per-asset base risk percentages (before Scale A tiers)
# GOLD at 2% (89% WR), SILVER/US500 at 1% (lower WR, higher DD at 2%)
ASSET_RISK = {
    'GOLD': 0.02,             # 2% risk per trade
    'SILVER': 0.01,           # 1% risk per trade
    'US500': 0.01,            # 1% risk per trade
}

# Monitoring intervals
CHECK_INTERVAL = 5  # Check for setups every 5 seconds
POSITION_CHECK_INTERVAL = 5  # Check positions every 5 seconds
BAR_CLOSE_BURST_WINDOW = 10  # Seconds after bar close to scan rapidly
BAR_CLOSE_BURST_INTERVAL = 2  # Scan every 2 seconds during burst
LIMIT_ORDER_EXPIRY_MINUTES = 5  # Limit order expiry (minutes)

# Spread filter thresholds (max spread as % of mid price)
MAX_SPREAD_PCT = {
    'GOLD': 0.20,    # Metals: max 0.20%
    'SILVER': 0.20,  # Metals: max 0.20%
    'US500': 0.05,   # Indices: max 0.05%
}

# Slippage tracking
BAD_EXECUTION_THRESHOLD = 0.15  # Flag fills with > 0.15% slippage

# Circuit breakers
CONSECUTIVE_STOP_LIMIT = 2       # Consecutive stops before cooldown
ASSET_COOLDOWN_HOURS = 4         # Hours to pause asset after consecutive stops
DAILY_LOSS_MULTIPLIER = 3        # Daily loss limit = multiplier × risk cap

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

        # Connection health tracking
        self.consecutive_data_failures = 0
        self.max_data_failures_before_reconnect = 3  # 3 full scan cycles with no data → reconnect
        self.last_reconnect_time = None
        self.reconnect_cooldown = 60  # Minimum seconds between reconnect attempts

        # Execution quality tracking (rolling window)
        self.execution_log = []  # [{asset, time, intended, actual, slippage_pct, bad}]

        # Circuit breakers
        self.consecutive_stops = {}   # {asset: count}
        self.asset_cooldowns = {}     # {asset: cooldown_until_datetime}
        self.daily_losses = 0.0
        self.daily_loss_date = datetime.now().date()
        self.trading_halted = False

        # Failed order cooldown (prevent retry spam)
        self.failed_order_cooldowns = {}  # {asset: cooldown_until_datetime}

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

        # TTM Squeeze (Beardy Squeeze Pro: 3 KC levels at 1.0, 1.5, 2.0)
        squeeze_on, momentum, color, squeeze_intensity = calculate_ttm_squeeze_pinescript(
            df['high'], df['low'], df['close'],
            bb_period=20, bb_std=2.0,
            kc_period=20,
            momentum_period=20
        )

        df['squeeze_on'] = squeeze_on
        df['ttm_momentum'] = momentum
        df['squeeze_intensity'] = squeeze_intensity

        return df

    def _attempt_reconnect(self):
        """Attempt to reconnect to Capital.com API"""
        now = time.time()
        if self.last_reconnect_time and (now - self.last_reconnect_time) < self.reconnect_cooldown:
            return False  # Too soon since last attempt

        self.last_reconnect_time = now
        self.log("API session appears expired - attempting reconnect...", "WARNING")

        success = self.broker.reconnect()
        if success:
            self.consecutive_data_failures = 0
            self.log("Reconnected to Capital.com successfully!", "INFO")
            return True
        else:
            self.log("Reconnection failed - will retry next cycle", "ERROR")
            return False

    def check_for_setups(self):
        """Scan all assets for entry setups"""
        if len(self.positions) >= MAX_POSITIONS:
            return  # Portfolio full

        # Daily loss limit reset on new day
        today = datetime.now().date()
        if today != self.daily_loss_date:
            self.daily_losses = 0.0
            self.daily_loss_date = today
            if self.trading_halted:
                self.trading_halted = False
                self.log("New trading day - daily loss limit reset", "INFO")

        # Daily loss limit circuit breaker
        if self.trading_halted:
            return

        # Sync positions with broker to prevent duplicates
        try:
            self._sync_positions_with_broker()
        except Exception as e:
            self.log(f"Broker sync failed: {e}", "ERROR")

        assets_with_data = 0
        assets_checked = 0

        for asset, strategy in self.strategies.items():
            if asset in self.positions:
                continue  # Already in position
            if asset in self.pending_entries:
                continue  # Order already pending

            # Consecutive stop cooldown check
            if asset in self.asset_cooldowns:
                if datetime.now() < self.asset_cooldowns[asset]:
                    continue  # Asset on cooldown
                else:
                    del self.asset_cooldowns[asset]
                    self.consecutive_stops.pop(asset, None)
                    self.log(f"COOLDOWN_EXPIRED: {asset} - resuming trading", "INFO")

            # Failed order cooldown (prevent retry spam)
            if asset in self.failed_order_cooldowns:
                if datetime.now() < self.failed_order_cooldowns[asset]:
                    continue
                else:
                    del self.failed_order_cooldowns[asset]

            assets_checked += 1

            try:
                # Fetch live data
                df_15m = self.get_live_data(asset, '15m', 100)
                df_1h = self.get_live_data(asset, '1H', 100)

                if df_15m.empty or df_1h.empty:
                    # Market closed or no data available - skip silently
                    continue

                assets_with_data += 1

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

        # Track data failures for auto-reconnect
        if assets_checked > 0 and assets_with_data == 0:
            self.consecutive_data_failures += 1
            if self.consecutive_data_failures >= self.max_data_failures_before_reconnect:
                self._attempt_reconnect()
        else:
            # Got data from at least one asset - connection is fine
            self.consecutive_data_failures = 0

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
                                'risk_pct': get_asset_risk_pct(asset_name, self.capital),
                                'order_id': p.get('dealId', '')
                            }
                            self.log(f"Synced broker position: {asset_name} @ ${p.get('level', 0)}")
                            break
        except Exception as e:
            self.log(f"Error syncing positions: {e}", "ERROR")

    def _log_execution(self, asset: str, intended_price: float, actual_price: float, order_type: str):
        """Log execution quality to rolling window"""
        slippage_pct = abs(actual_price - intended_price) / intended_price * 100
        is_bad = slippage_pct > BAD_EXECUTION_THRESHOLD

        entry = {
            'asset': asset,
            'time': datetime.now(),
            'intended': intended_price,
            'actual': actual_price,
            'slippage_pct': slippage_pct,
            'order_type': order_type,
            'bad_execution': is_bad
        }
        self.execution_log.append(entry)

        if is_bad:
            self.log(f"BAD_EXECUTION: {asset} {order_type} intended=${intended_price:.2f} actual=${actual_price:.2f} slippage={slippage_pct:.3f}%", "WARNING")

    def execute_entry(self, asset: str, setup, current_bar):
        """Execute entry using limit order at calculated pullback level"""
        # Lock this asset immediately to prevent duplicate orders
        self.pending_entries.add(asset)

        try:
            # Calculate position size with per-asset risk %
            current_risk_pct = get_asset_risk_pct(asset, self.capital)
            risk_amount = self.capital * current_risk_pct
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

            # Spread filter - block entry if spread is abnormally wide
            mid_price = (current_bid + current_offer) / 2
            spread_pct = (spread / mid_price) * 100
            max_spread = MAX_SPREAD_PCT.get(asset, 0.20)

            if spread_pct > max_spread:
                self.log(f"BLOCKED_SPREAD: {asset} spread={spread_pct:.3f}% > max {max_spread:.2f}% (${spread:.2f})", "WARNING")
                self.pending_entries.discard(asset)
                return

            entry_price = setup.entry_price
            price_diff = current_offer - entry_price
            price_diff_pct = (price_diff / entry_price) * 100

            # ATR-based threshold: market order only if price within 0.1 ATR of entry
            atr = current_bar.get('atr_20', 0) if hasattr(current_bar, 'get') else getattr(current_bar, 'atr_20', 0)
            market_order_threshold = entry_price + (0.1 * atr) if atr > 0 else entry_price

            self.log(f"Setup: Entry=${entry_price:.2f} | Current=${current_offer:.2f} | Gap={price_diff_pct:.2f}% | Spread={spread_pct:.3f}% | ATR=${atr:.2f}")

            # Decision: limit order vs market order
            if current_offer <= market_order_threshold:
                # Price is AT or BELOW entry level (within 0.1 ATR) → market order
                self.log(f"Price at/below entry level (within 0.1 ATR=${0.1*atr:.2f}) → MARKET ORDER")
                result = self.broker.place_market_order(
                    asset=epic,
                    direction='LONG',
                    units=position_size,
                    stop_loss=setup.stop_loss,
                    take_profit=setup.target
                )

                if result.success:
                    fill_price = result.fill_price
                    slippage_pct = abs(fill_price - entry_price) / entry_price * 100
                    is_bad = slippage_pct > BAD_EXECUTION_THRESHOLD

                    # Validate R:R after fill - reject if below 1.5:1
                    actual_risk = fill_price - setup.stop_loss
                    actual_reward = setup.target - fill_price
                    actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0

                    if actual_rr < 1.5:
                        self.log(f"R:R TOO LOW after fill: {actual_rr:.2f}:1 (risk=${actual_risk:.2f} reward=${actual_reward:.2f}) - closing immediately", "WARNING")
                        self.broker.close_position(result.order_id)
                        self.pending_entries.discard(asset)
                        self.failed_order_cooldowns[asset] = datetime.now() + timedelta(minutes=15)
                    else:
                        self.positions[asset] = {
                            'entry_time': datetime.now(),
                            'entry_price': fill_price,
                            'stop_loss': setup.stop_loss,
                            'target': setup.target,
                            'size': position_size,
                            'margin': margin_required,
                            'risk_pct': current_risk_pct,
                            'order_id': result.order_id
                        }
                        self._log_execution(asset, entry_price, fill_price, 'MARKET')
                        self.log(f"✓ MARKET ENTRY: {asset} @ ${fill_price:.2f} (slippage: {slippage_pct:.2f}%{'  BAD_EXECUTION' if is_bad else ''}) R:R={actual_rr:.1f}:1", "SUCCESS")
                        self.log(f"  Size: {position_size:.4f} | Stop: ${setup.stop_loss:.2f} | Target: ${setup.target:.2f}")
                else:
                    self.log(f"✗ MARKET ENTRY FAILED: {asset} - {result.message}", "ERROR")
                    self.pending_entries.discard(asset)
                    # Cooldown to prevent retry spam (e.g. market closed)
                    self.failed_order_cooldowns[asset] = datetime.now() + timedelta(minutes=15)
                    self.log(f"  {asset} order cooldown 15min (next retry after {self.failed_order_cooldowns[asset].strftime('%H:%M')})", "WARNING")

            else:
                # Price is ABOVE entry level → LIMIT ORDER (wait for pullback)
                expiry_time = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=LIMIT_ORDER_EXPIRY_MINUTES)
                expiry_str = expiry_time.strftime('%Y-%m-%dT%H:%M:%S')

                self.log(f"Price above entry (+{price_diff_pct:.2f}%) → LIMIT ORDER @ ${entry_price:.2f}")
                self.log(f"  Expires: {expiry_str} UTC ({LIMIT_ORDER_EXPIRY_MINUTES}min) | Stop: ${setup.stop_loss:.2f} | Target: ${setup.target:.2f}")

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
                    self.failed_order_cooldowns[asset] = datetime.now() + timedelta(minutes=15)
                    self.log(f"  {asset} order cooldown 15min (next retry after {self.failed_order_cooldowns[asset].strftime('%H:%M')})", "WARNING")

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

                            is_bad = slippage_pct > BAD_EXECUTION_THRESHOLD
                            self.positions[asset] = {
                                'entry_time': datetime.now(),
                                'entry_price': fill_price,
                                'stop_loss': order['stop_loss'],
                                'target': order['target'],
                                'size': order['size'],
                                'margin': 0,
                                'risk_pct': get_asset_risk_pct(asset, self.capital),
                                'order_id': p.get('dealId', '')
                            }
                            self._log_execution(asset, order['entry_price'], fill_price, 'LIMIT')
                            self.log(f"✓ LIMIT ORDER FILLED: {asset} @ ${fill_price:.2f} (slippage: {slippage_pct:.3f}%{'  BAD_EXECUTION' if is_bad else ''})", "SUCCESS")
                            orders_to_remove.append(asset)
                            filled = True
                            break

                if filled:
                    continue

                # Check if order expired
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
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
                    # Determine P&L to classify as stop or target
                    old_capital = self.capital
                    try:
                        account = self.broker.get_account_info()
                        self.capital = account.balance
                    except:
                        pass
                    pnl = self.capital - old_capital

                    if pnl < 0:
                        # Loss - likely stopped out
                        self.consecutive_stops[asset] = self.consecutive_stops.get(asset, 0) + 1
                        self.daily_losses += abs(pnl)
                        self.log(f"STOP_OUT: {asset} P&L=${pnl:.2f} | Consecutive stops: {self.consecutive_stops[asset]}/{CONSECUTIVE_STOP_LIMIT} | Daily losses: ${self.daily_losses:.2f}", "WARNING")

                        # Check consecutive stop cooldown
                        if self.consecutive_stops[asset] >= CONSECUTIVE_STOP_LIMIT:
                            cooldown_until = datetime.now() + timedelta(hours=ASSET_COOLDOWN_HOURS)
                            self.asset_cooldowns[asset] = cooldown_until
                            self.log(f"COOLDOWN_ACTIVE: {asset} - {self.consecutive_stops[asset]} consecutive stops, paused until {cooldown_until.strftime('%H:%M')}", "WARNING")

                        # Check daily loss limit (based on GOLD risk = highest)
                        daily_limit = (self.capital * get_asset_risk_pct('GOLD', self.capital)) * DAILY_LOSS_MULTIPLIER
                        if self.daily_losses >= daily_limit:
                            self.trading_halted = True
                            self.log(f"DAILY_LIMIT_REACHED: Losses ${self.daily_losses:.2f} >= limit ${daily_limit:.2f} - trading halted until tomorrow", "WARNING")
                    else:
                        # Win - reset consecutive stop counter
                        self.consecutive_stops[asset] = 0
                        self.log(f"TARGET_HIT: {asset} P&L=+${pnl:.2f}", "INFO")

                    positions_to_close.append(asset)
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
        print(f"Risk per trade:")
        for asset_name in ASSETS:
            rpct = get_asset_risk_pct(asset_name, self.capital)
            print(f"  {asset_name}: {rpct*100:.1f}% (${self.capital * rpct:,.0f})")
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
                remaining = (order['expires'] - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
                print(f"  {asset}:")
                print(f"    Limit: ${order['entry_price']:.2f}")
                print(f"    Expires in: {max(0, remaining):.0f}s")
                print()

        # Execution quality stats
        if self.execution_log:
            total = len(self.execution_log)
            bad = sum(1 for e in self.execution_log if e['bad_execution'])
            avg_slip = sum(e['slippage_pct'] for e in self.execution_log) / total
            print("EXECUTION QUALITY:")
            print(f"  Fills: {total} | Bad (>{BAD_EXECUTION_THRESHOLD}%): {bad} | Avg slippage: {avg_slip:.3f}%")
            for e in self.execution_log[-5:]:  # Show last 5
                flag = " BAD" if e['bad_execution'] else ""
                print(f"  {e['time'].strftime('%m-%d %H:%M')} {e['asset']} {e['order_type']}: {e['slippage_pct']:.3f}%{flag}")
            print()

        # Circuit breaker status (daily limit based on GOLD risk = highest)
        daily_limit = (self.capital * get_asset_risk_pct('GOLD', self.capital)) * DAILY_LOSS_MULTIPLIER
        print("CIRCUIT BREAKERS:")
        print(f"  Daily losses: ${self.daily_losses:.2f} / ${daily_limit:.2f} limit {'HALTED' if self.trading_halted else 'OK'}")
        if self.asset_cooldowns:
            for asset, until in self.asset_cooldowns.items():
                remaining = (until - datetime.now()).total_seconds() / 3600
                if remaining > 0:
                    print(f"  {asset}: COOLDOWN ({remaining:.1f}h remaining, {self.consecutive_stops.get(asset, 0)} consecutive stops)")
        if self.consecutive_stops:
            active_counts = {a: c for a, c in self.consecutive_stops.items() if c > 0 and a not in self.asset_cooldowns}
            if active_counts:
                for asset, count in active_counts.items():
                    print(f"  {asset}: {count}/{CONSECUTIVE_STOP_LIMIT} consecutive stops")
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
        for a, r in ASSET_RISK.items():
            self.log(f"  {a}: {r*100:.1f}% base risk")
        self.log(f"Environment: {CAPITAL_CONFIG['environment'].upper()}")
        self.log("")

        # Sync capital from broker (use actual balance, not initial)
        try:
            account = self.broker.get_account_info()
            self.capital = account.balance
            self.log(f"Capital synced from broker: ${self.capital:,.2f}")
        except Exception as e:
            self.log(f"Could not sync capital, using ${self.capital:,.2f}: {e}", "WARNING")

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
        last_bar_close_logged = None

        try:
            while self.is_running:
                current_time = time.time()

                # Bar-close alignment: detect proximity to 15-minute boundaries
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                seconds_into_bar = (now_utc.minute % 15) * 60 + now_utc.second
                is_bar_close_window = seconds_into_bar < BAR_CLOSE_BURST_WINDOW

                # Use burst interval near bar close, normal interval otherwise
                scan_interval = BAR_CLOSE_BURST_INTERVAL if is_bar_close_window else CHECK_INTERVAL

                # Log bar close detection once per boundary
                if is_bar_close_window:
                    bar_key = f"{now_utc.hour}:{(now_utc.minute // 15) * 15:02d}"
                    if bar_key != last_bar_close_logged:
                        self.log(f"15m bar closed ({bar_key} UTC) - burst scanning", "INFO")
                        last_bar_close_logged = bar_key

                # Check for new setups
                if current_time - last_setup_check >= scan_interval:
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

                time.sleep(1)  # Sleep 1 second between cycles for responsive bar detection

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
