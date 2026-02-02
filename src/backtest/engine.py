from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.regime.detector import RegimeDetector
from src.strategies.selector import StrategySelector
from src.risk.governor import RiskGovernor
from src.execution.paper_trader import PaperTrader


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates real trading bar-by-bar:
    1. Process each bar sequentially
    2. Update indicators
    3. Check for signals
    4. Execute trades
    5. Manage positions
    6. Track performance

    This avoids look-ahead bias and simulates realistic trading.

    Example:
        engine = BacktestEngine(
            data=df,
            initial_capital=10000,
            assets=['US_TECH_100']
        )

        results = engine.run()
        print(f"Return: {results['total_return']:.2f}%")
    """

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 initial_capital: float,
                 assets: List[str],
                 commission_pct: float = 0.0,
                 slippage_pct: float = 0.001):
        """
        Initialize backtest engine

        Args:
            data: Dict of {asset: DataFrame with OHLCV}
            initial_capital: Starting capital
            assets: List of assets to trade
            commission_pct: Commission per trade (e.g., 0.001 = 0.1%)
            slippage_pct: Slippage per trade (e.g., 0.001 = 0.1%)
        """

        self.data = data
        self.initial_capital = initial_capital
        self.assets = assets
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        # Initialize components
        self.regime_detector = RegimeDetector()
        self.strategies = {
            asset: StrategySelector(asset) for asset in assets
        }
        self.risk_governor = RiskGovernor(initial_capital)

        # Setup paper broker for simulation
        broker_config = {
            'initial_balance': initial_capital,
            'slippage_pct': slippage_pct,
            'spread_multiplier': 1.0
        }
        self.broker = PaperTrader(broker_config)
        self.broker.connect()

        # Load data into broker
        for asset, df in data.items():
            self.broker.load_historical_data(asset, df)

        # Tracking
        self.equity_curve = []
        self.trades = []
        self.current_bar = 0
        self.closed_this_bar = set()  # Track which assets were closed this bar

        print(f"[BACKTEST] Initialized with ${initial_capital:,.0f}")
        print(f"[BACKTEST] Assets: {', '.join(assets)}")
        print(f"[BACKTEST] Data range: {self._get_date_range()}")

    def run(self) -> Dict:
        """
        Run complete backtest

        Returns:
            Dict with results and statistics
        """

        print("\n" + "="*60)
        print("BACKTEST STARTING")
        print("="*60)

        # Get all timestamps (union of all assets)
        all_timestamps = self._get_all_timestamps()
        total_bars = len(all_timestamps)

        print(f"Total bars to process: {total_bars}")

        # Process bar by bar
        for i, timestamp in enumerate(all_timestamps):
            self.current_bar = i

            # Progress indicator
            if i % 100 == 0:
                progress = (i / total_bars) * 100
                print(f"Progress: {progress:.1f}% ({i}/{total_bars})")

            # Process this bar
            self._process_bar(timestamp)

            # Track equity
            account = self.broker.get_account_info()
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': account.equity,
                'balance': account.balance,
                'bar': i
            })

        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)

        # Calculate results
        results = self._calculate_results()

        return results

    def _process_bar(self, timestamp: datetime):
        """
        Process one bar

        Args:
            timestamp: Current bar timestamp
        """

        # Clear closed positions tracker for this bar
        self.closed_this_bar.clear()

        # Reset daily counters if new day
        self.risk_governor.check_daily_reset(timestamp)

        # Check if paused
        should_pause, reason, _ = self.risk_governor.should_pause_trading()
        if should_pause:
            # Only manage existing positions
            self._manage_positions(timestamp)
            return

        # Manage existing positions first
        self._manage_positions(timestamp)

        # Scan for new setups
        if len(self.broker.positions) < self.risk_governor.profile.max_concurrent_positions:
            self._scan_for_setups(timestamp)

    def _manage_positions(self, timestamp: datetime):
        """Manage open positions"""

        for asset in list(self.broker.positions.keys()):
            position = self.broker.positions[asset]

            # Get data up to current bar
            df = self._get_data_window(asset, timestamp, lookback=300)

            if df is None or len(df) < 50:
                continue

            # Add indicators
            df = self.regime_detector.add_all_indicators(df)

            # Create StrategyPosition object
            from src.strategies.base import Position as StrategyPosition, SignalDirection

            strat_position = StrategyPosition(
                asset=asset,
                direction=SignalDirection.LONG if position['direction'] == 'LONG' else SignalDirection.SHORT,
                entry_price=position['entry_price'],
                stop_loss=position['stop_loss'],
                target=position.get('take_profit', position['entry_price'] * 1.05),
                units=position['units'],
                risk_per_share=abs(position['entry_price'] - position['stop_loss']),
                entry_time=position['opened_at'],
                entry_bar=0,
                entry_strategy=position.get('strategy', 'UNKNOWN')
            )

            # Check exit
            strategy = self.strategies[asset]
            action, new_value = strategy.manage_exit(df, strat_position)

            if action.value == 'HOLD':
                # Check stop loss hit via broker
                self.broker.check_stop_loss_hit()

            elif action.value in ['BREAKEVEN', 'TRAIL']:
                # Update stop
                self.broker.modify_position(asset, stop_loss=new_value)

            else:
                # Exit
                result = self.broker.close_position(asset)
                if result.success:
                    self._record_trade(asset, strat_position, result, action.value, timestamp)
                    # Mark as closed this bar to prevent re-entry
                    self.closed_this_bar.add(asset)

    def _scan_for_setups(self, timestamp: datetime):
        """Scan for new entry setups"""

        for asset in self.assets:

            # Skip if already in position
            if asset in self.broker.positions:
                continue

            # Skip if position was just closed this bar (prevent immediate re-entry)
            if asset in self.closed_this_bar:
                continue

            # Get data
            df = self._get_data_window(asset, timestamp, lookback=300)

            if df is None or len(df) < 200:
                continue

            # Add indicators
            df = self.regime_detector.add_all_indicators(df)

            # Detect regime
            regime, confidence, strat_rec = self.regime_detector.detect_regime(df)

            # Check for setup
            strategy = self.strategies[asset]
            setup = strategy.check_entry(df, regime.value, confidence, strat_rec)

            if setup:
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

                if allowed:
                    # Calculate position size
                    units = self.risk_governor.calculate_position_size(
                        risk_pct=risk_pct,
                        risk_per_share=setup.risk_per_share,
                        price=setup.entry_price,
                        asset=asset
                    )

                    if units > 0:
                        # Execute
                        result = self.broker.place_market_order(
                            asset=asset,
                            direction=setup.direction.value,
                            units=units,
                            stop_loss=setup.stop_loss,
                            take_profit=setup.target
                        )

                        if result.success:
                            # Add strategy info to broker position
                            if asset in self.broker.positions:
                                self.broker.positions[asset]['strategy'] = setup.setup_type

                            # Track position in risk governor
                            self.risk_governor.add_position(asset, {
                                'units': units,
                                'entry': result.fill_price,
                                'strategy': setup.setup_type
                            })

    def _record_trade(self, asset, position, close_result, exit_reason, timestamp):
        """Record completed trade"""

        # Calculate P&L
        from src.strategies.base import SignalDirection

        if position.direction == SignalDirection.LONG:
            pnl = (close_result.fill_price - position.entry_price) * position.units
        else:
            pnl = (position.entry_price - close_result.fill_price) * position.units

        outcome = 'WIN' if pnl > 0 else 'LOSS'

        # Record in risk governor
        self.risk_governor.record_trade(
            asset=asset,
            direction=position.direction.value,
            entry_price=position.entry_price,
            exit_price=close_result.fill_price,
            units=position.units,
            outcome=outcome,
            setup_type=position.entry_strategy,
            exit_reason=exit_reason
        )

        # Track trade
        self.trades.append({
            'timestamp': timestamp,
            'asset': asset,
            'direction': position.direction.value,
            'entry_price': position.entry_price,
            'exit_price': close_result.fill_price,
            'units': position.units,
            'pnl': pnl,
            'outcome': outcome,
            'setup': position.entry_strategy,
            'exit_reason': exit_reason
        })

    def _get_data_window(self, asset: str, timestamp: datetime, lookback: int) -> Optional[pd.DataFrame]:
        """Get data window up to timestamp"""

        if asset not in self.data:
            return None

        df = self.data[asset]

        # Get data up to (but not including) current bar
        # This avoids look-ahead bias
        window = df[df.index < timestamp].tail(lookback)

        return window

    def _get_all_timestamps(self) -> pd.DatetimeIndex:
        """Get union of all timestamps across assets"""

        all_times = []
        for df in self.data.values():
            all_times.extend(df.index.tolist())

        # Remove duplicates and sort
        unique_times = sorted(set(all_times))

        return pd.DatetimeIndex(unique_times)

    def _get_date_range(self) -> str:
        """Get date range string"""

        all_times = self._get_all_timestamps()
        return f"{all_times[0].date()} to {all_times[-1].date()}"

    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""

        from src.backtest.performance import PerformanceCalculator

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df = equity_df.set_index('timestamp')

        # Get trade log from risk governor
        trades_df = pd.DataFrame([vars(t) for t in self.risk_governor.trade_log]) if self.risk_governor.trade_log else pd.DataFrame()

        # If no trades from risk governor, use self.trades
        if trades_df.empty and self.trades:
            trades_df = pd.DataFrame(self.trades)

        # Calculate performance metrics
        calc = PerformanceCalculator(
            equity_curve=equity_df,
            trades=trades_df,
            initial_capital=self.initial_capital
        )

        results = calc.calculate_all()

        # Add raw data
        results['equity_curve'] = equity_df
        results['trades'] = trades_df
        results['risk_governor'] = self.risk_governor
        results['broker'] = self.broker

        return results
