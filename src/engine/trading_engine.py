import time
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd

from src.regime.detector import RegimeDetector, MarketRegime
from src.strategies.selector import StrategySelector
from src.strategies.base import Position as StrategyPosition, SignalDirection
from src.risk.governor import RiskGovernor
from src.execution.broker_interface import BrokerInterface
from src.engine.state_manager import StateManager


class TradingEngine:
    """
    Main trading engine - combines all components.

    Flow:
    1. Initialize all components
    2. Connect to broker
    3. Load saved state (if exists)
    4. Main loop:
       - Fetch market data
       - Calculate indicators
       - Detect regime
       - Scan for setups
       - Execute trades (if approved by risk)
       - Manage open positions
       - Save state
       - Sleep until next bar

    Example:
        from src.execution.capital_connector import CapitalConnector

        broker_config = {...}
        broker = CapitalConnector(broker_config)

        engine = TradingEngine(
            broker=broker,
            initial_capital=10000,
            assets=['US_TECH_100', 'GOLD']
        )

        engine.run(interval_minutes=60)
    """

    def __init__(self,
                 broker: BrokerInterface,
                 initial_capital: float,
                 assets: list[str],
                 state_dir: str = 'data/live'):
        """
        Initialize trading engine

        Args:
            broker: Broker interface (Capital.com, Paper, etc)
            initial_capital: Starting capital
            assets: List of instruments to trade
            state_dir: Directory for state persistence
        """

        print("="*60)
        print("INITIALIZING TRADING ENGINE")
        print("="*60)

        # Store config
        self.broker = broker
        self.assets = assets

        # Initialize components
        print("\n[1/5] Initializing components...")
        self.regime_detector = RegimeDetector()
        self.strategy_selector = StrategySelector(assets[0])  # Will create per asset
        self.risk_governor = RiskGovernor(initial_capital)
        self.state_manager = StateManager(state_dir)

        # Strategy selectors per asset
        self.strategies = {
            asset: StrategySelector(asset) for asset in assets
        }

        # State
        self.is_running = False
        self.current_positions: Dict[str, StrategyPosition] = {}

        # Data cache
        self.market_data: Dict[str, pd.DataFrame] = {}

        print("✓ Components initialized")

    def connect(self) -> bool:
        """
        Connect to broker and load state

        Returns:
            True if successful
        """

        print("\n[2/5] Connecting to broker...")

        if not self.broker.connect():
            print("✗ Failed to connect to broker")
            return False

        print("✓ Connected to broker")

        # Load previous state
        print("\n[3/5] Loading saved state...")
        state = self.state_manager.load_state()

        if state:
            self._restore_state(state)
            print("✓ State restored")
        else:
            print("✓ No previous state (fresh start)")

        # Verify account
        print("\n[4/5] Verifying account...")
        account = self.broker.get_account_info()
        print(f"✓ Account verified: {account.currency} {account.balance:.2f}")

        # Update capital from account
        self.risk_governor.current_capital = account.balance

        print("\n[5/5] Initialization complete")
        print("="*60)

        return True

    def run(self, interval_minutes: int = 60):
        """
        Main trading loop

        Args:
            interval_minutes: Time between scans (e.g., 60 = hourly)
        """

        print(f"\n🚀 STARTING TRADING ENGINE")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Assets: {', '.join(self.assets)}")
        print(f"   Capital: ${self.risk_governor.current_capital:.2f}")
        print("="*60)

        self.is_running = True

        try:
            while self.is_running:
                cycle_start = time.time()

                # Run one trading cycle
                self._run_cycle()

                # Save state
                self._save_state()

                # Sleep until next interval
                cycle_duration = time.time() - cycle_start
                sleep_time = (interval_minutes * 60) - cycle_duration

                if sleep_time > 0:
                    print(f"\n💤 Sleeping for {sleep_time/60:.1f} minutes...")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n⏸️  Bot stopped by user")
            self.stop()

        except Exception as e:
            print(f"\n💥 CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.stop()

    def _run_cycle(self):
        """
        Execute one complete trading cycle
        """

        print(f"\n{'='*60}")
        print(f"CYCLE START: {datetime.now()}")
        print(f"{'='*60}")

        # Reset daily counters if new day
        self.risk_governor.check_daily_reset(datetime.now())

        # Check if trading should be paused
        should_pause, reason, conditions = self.risk_governor.should_pause_trading()

        if should_pause:
            print(f"⚠️  TRADING PAUSED: {reason}")
            print(f"   Resume conditions: {conditions}")
            self._manage_existing_positions()
            return

        # Manage existing positions first
        print("\n[1] Managing existing positions...")
        self._manage_existing_positions()

        # Scan for new opportunities
        print("\n[2] Scanning for new setups...")
        self._scan_for_setups()

        # Display status
        print("\n[3] Current status:")
        self._display_status()

    def _manage_existing_positions(self):
        """
        Manage all open positions
        """

        if not self.current_positions:
            print("   No open positions")
            return

        for asset in list(self.current_positions.keys()):
            position = self.current_positions[asset]

            try:
                # Fetch current data
                df = self._get_market_data(asset, count=100)

                if df is None or df.empty:
                    print(f"   ⚠️  No data for {asset}, skipping")
                    continue

                # Add indicators
                df = self.regime_detector.add_all_indicators(df)

                # Get strategy for this asset
                strategy = self.strategies[asset]

                # Check exit conditions
                action, new_value = strategy.manage_exit(df, position)

                if action.value == 'HOLD':
                    # Update unrealized P&L
                    current_price = df.iloc[-1]['close']
                    current_r = self._calculate_r(position, current_price)
                    print(f"   {asset}: HOLD (R: {current_r:+.1f})")

                elif action.value == 'BREAKEVEN':
                    # Move to breakeven
                    self._update_stop_loss(asset, position, new_value)
                    print(f"   {asset}: BREAKEVEN moved to {new_value:.2f}")

                elif action.value == 'TRAIL':
                    # Update trailing stop
                    self._update_stop_loss(asset, position, new_value)
                    print(f"   {asset}: TRAIL updated to {new_value:.2f}")

                else:
                    # Exit position
                    self._close_position(asset, position, action.value)
                    print(f"   {asset}: EXIT ({action.value})")

            except Exception as e:
                print(f"   ✗ Error managing {asset}: {e}")

    def _scan_for_setups(self):
        """
        Scan all assets for entry setups
        """

        # Check if we can open more positions
        if len(self.current_positions) >= self.risk_governor.profile.max_concurrent_positions:
            print("   Max positions already open")
            return

        setups_found = []

        for asset in self.assets:

            # Skip if already in position
            if asset in self.current_positions:
                continue

            try:
                # Fetch data
                df = self._get_market_data(asset, count=300)

                if df is None or df.empty:
                    continue

                # Add indicators
                df = self.regime_detector.add_all_indicators(df)

                # Detect regime
                regime, confidence, strategy_rec = self.regime_detector.detect_regime(df)

                print(f"   {asset}: {regime.value} (conf: {confidence:.0f}%)")

                # Check for entry
                strategy = self.strategies[asset]
                setup = strategy.check_entry(df, regime.value, confidence, strategy_rec)

                if setup:
                    setups_found.append({
                        'asset': asset,
                        'setup': setup,
                        'regime': regime,
                        'confidence': confidence
                    })
                    print(f"      → SETUP: {setup.setup_type}")

            except Exception as e:
                print(f"   ✗ Error scanning {asset}: {e}")

        # Execute best setup (highest confidence)
        if setups_found:
            setups_found.sort(key=lambda x: x['confidence'], reverse=True)
            self._execute_setup(setups_found[0])

    def _execute_setup(self, setup_data: dict):
        """
        Execute a trading setup

        Args:
            setup_data: Dict with asset, setup, regime, confidence
        """

        asset = setup_data['asset']
        setup = setup_data['setup']
        regime = setup_data['regime']

        print(f"\n   🎯 EXECUTING: {asset} {setup.setup_type}")

        # Risk check
        allowed, reason, risk_pct = self.risk_governor.can_open_trade(
            asset=asset,
            regime=regime.value,
            setup={
                'confidence': setup.confidence,
                'risk_per_share': setup.risk_per_share,
                'entry_price': setup.entry_price
            }
        )

        if not allowed:
            print(f"   ✗ Trade rejected: {reason}")
            return

        print(f"   ✓ Risk approved: {risk_pct*100:.2f}%")

        # Calculate position size
        units = self.risk_governor.calculate_position_size(
            risk_pct=risk_pct,
            risk_per_share=setup.risk_per_share,
            price=setup.entry_price,
            asset=asset
        )

        if units == 0:
            print(f"   ✗ Position size too small")
            return

        print(f"   Units: {units}")

        # Place order
        try:
            order_result = self.broker.place_market_order(
                asset=asset,
                direction=setup.direction.value,
                units=units,
                stop_loss=setup.stop_loss,
                take_profit=setup.target
            )

            if order_result.success:
                print(f"   ✓ ORDER FILLED @ {order_result.fill_price:.2f}")

                # Create position tracking
                position = StrategyPosition(
                    asset=asset,
                    direction=setup.direction,
                    entry_price=order_result.fill_price,
                    stop_loss=setup.stop_loss,
                    target=setup.target,
                    units=units,
                    risk_per_share=setup.risk_per_share,
                    entry_time=pd.Timestamp.now(),
                    entry_bar=0,  # Will be updated
                    entry_strategy=setup.setup_type
                )

                self.current_positions[asset] = position
                self.risk_governor.add_position(asset, vars(position))

                print(f"   📊 Position opened successfully")
            else:
                print(f"   ✗ ORDER FAILED: {order_result.message}")

        except Exception as e:
            print(f"   ✗ Execution error: {e}")

    def _close_position(self, asset: str, position: StrategyPosition, reason: str):
        """
        Close a position

        Args:
            asset: Instrument
            position: Position to close
            reason: Exit reason
        """

        try:
            # Close via broker
            close_result = self.broker.close_position(asset)

            if close_result.success:

                # Calculate P&L
                if position.direction == SignalDirection.LONG:
                    pnl = (close_result.fill_price - position.entry_price) * position.units
                else:
                    pnl = (position.entry_price - close_result.fill_price) * position.units

                outcome = 'WIN' if pnl > 0 else 'LOSS'

                # Record trade
                self.risk_governor.record_trade(
                    asset=asset,
                    direction=position.direction.value,
                    entry_price=position.entry_price,
                    exit_price=close_result.fill_price,
                    units=position.units,
                    outcome=outcome,
                    setup_type=position.entry_strategy,
                    exit_reason=reason
                )

                # Remove from tracking
                del self.current_positions[asset]

                print(f"   💰 Closed {asset}: {outcome} P&L ${pnl:+.2f}")
            else:
                print(f"   ✗ Close failed: {close_result.message}")

        except Exception as e:
            print(f"   ✗ Error closing position: {e}")

    def _update_stop_loss(self, asset: str, position: StrategyPosition, new_stop: float):
        """Update position stop loss"""

        try:
            success = self.broker.modify_position(asset, stop_loss=new_stop)

            if success:
                position.stop_loss = new_stop
            else:
                print(f"   ⚠️  Failed to update stop for {asset}")

        except Exception as e:
            print(f"   ✗ Error updating stop: {e}")

    def _get_market_data(self, asset: str, count: int = 300) -> Optional[pd.DataFrame]:
        """
        Fetch market data (with caching)

        Args:
            asset: Instrument
            count: Number of bars

        Returns:
            DataFrame with OHLCV data
        """

        try:
            df = self.broker.get_historical_data(
                asset=asset,
                timeframe='1H',
                count=count
            )

            if not df.empty:
                self.market_data[asset] = df

            return df

        except Exception as e:
            print(f"   ✗ Data fetch error for {asset}: {e}")
            return None

    def _calculate_r(self, position: StrategyPosition, current_price: float) -> float:
        """Calculate current R multiple"""

        if position.direction == SignalDirection.LONG:
            pnl = current_price - position.entry_price
        else:
            pnl = position.entry_price - current_price

        return pnl / position.risk_per_share

    def _display_status(self):
        """Display current status"""

        health = self.risk_governor.get_health_status()

        print(f"   Health: {health['status']} ({health['score']}/100)")
        print(f"   Capital: ${health['current_capital']:.2f}")
        print(f"   DD: {health['dd_pct']:.1f}%")
        print(f"   Positions: {health['positions_open']}")
        print(f"   Streak: W{health['streak_w']}/L{health['streak_l']}")

        if health['warnings']:
            for warning in health['warnings']:
                print(f"   ⚠️  {warning}")

    def _save_state(self):
        """Save current state"""

        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'capital': self.risk_governor.current_capital,
                'positions': {
                    asset: vars(pos) for asset, pos in self.current_positions.items()
                },
                'daily_pnl': self.risk_governor.daily_pnl,
                'consecutive_wins': self.risk_governor.consecutive_wins,
                'consecutive_losses': self.risk_governor.consecutive_losses,
                'current_dd_pct': self.risk_governor.current_dd_pct
            }

            self.state_manager.save_state(state)

        except Exception as e:
            print(f"   ✗ Error saving state: {e}")

    def _restore_state(self, state: dict):
        """Restore state from saved data"""

        try:
            self.risk_governor.current_capital = state.get('capital', self.risk_governor.current_capital)
            self.risk_governor.daily_pnl = state.get('daily_pnl', 0)
            self.risk_governor.consecutive_wins = state.get('consecutive_wins', 0)
            self.risk_governor.consecutive_losses = state.get('consecutive_losses', 0)
            self.risk_governor.current_dd_pct = state.get('current_dd_pct', 0)

            # Note: Open positions would need to be reconciled with broker
            # For now, we'll verify them on next cycle

        except Exception as e:
            print(f"   ✗ Error restoring state: {e}")

    def stop(self):
        """Stop trading engine gracefully"""

        print("\n🛑 STOPPING TRADING ENGINE...")

        self.is_running = False

        # Save final state
        self._save_state()

        # Create backup
        self.state_manager.create_backup('shutdown')

        # Disconnect broker
        self.broker.disconnect()

        # Display final stats
        stats = self.risk_governor.get_statistics()
        if stats:
            print("\n📊 FINAL STATISTICS:")
            print(f"   Total Trades: {stats['total_trades']}")
            print(f"   Win Rate: {stats['win_rate']:.1f}%")
            print(f"   Total Return: {stats['total_return_pct']:+.2f}%")
            print(f"   Final Capital: ${stats['current_capital']:.2f}")

        print("\n✓ Engine stopped safely")
