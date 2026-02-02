import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.execution.paper_trader import PaperTrader
from src.engine.trading_engine import TradingEngine


class TestIntegration:
    """
    End-to-end integration tests

    Tests complete flow from data → signal → execution
    """

    @pytest.fixture
    def paper_broker(self):
        """Create paper trading broker"""
        config = {
            'initial_balance': 10000,
            'currency': 'USD',
            'slippage_pct': 0.001,
            'spread_multiplier': 1.5
        }

        broker = PaperTrader(config)
        broker.connect()

        # Load some historical data
        dates = pd.date_range('2024-01-01', periods=500, freq='1H')
        trend = np.linspace(100, 120, 500)
        close = trend + np.random.normal(0, 0.4, 500)

        df = pd.DataFrame({
            'open': close - 0.2,
            'high': close + 0.6,
            'low': close - 0.6,
            'close': close,
            'volume': 5000
        }, index=dates)

        broker.load_historical_data('US_TECH_100', df)

        yield broker

        broker.disconnect()

    def test_engine_initialization(self, paper_broker):
        """Test engine initializes correctly"""

        engine = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir='data/test'
        )

        assert engine.broker == paper_broker
        assert engine.assets == ['US_TECH_100']
        assert engine.risk_governor is not None
        assert engine.regime_detector is not None

    def test_single_cycle_execution(self, paper_broker):
        """Test one complete trading cycle"""

        engine = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir='data/test'
        )

        # Initialize
        success = engine.connect()
        assert success

        # Run one cycle
        engine._run_cycle()

        # Should have scanned without errors
        assert True  # If we get here, cycle completed

    def test_position_lifecycle(self, paper_broker):
        """Test complete position lifecycle"""

        engine = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir='data/test'
        )

        engine.connect()

        # Run multiple cycles to potentially find setup
        for _ in range(10):
            engine._run_cycle()

            # If position opened, test management
            if engine.current_positions:
                asset = list(engine.current_positions.keys())[0]
                position = engine.current_positions[asset]

                # Position should have valid data
                assert position.entry_price > 0
                assert position.stop_loss > 0
                assert position.units > 0

                break

    def test_risk_limits_enforced(self, paper_broker):
        """Test risk limits are enforced"""

        engine = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir='data/test'
        )

        engine.connect()

        # Simulate max positions
        engine.risk_governor.open_positions = {
            f'asset_{i}': {} for i in range(5)
        }

        # Should not allow more positions
        allowed, reason, _ = engine.risk_governor.can_open_trade(
            'US_TECH_100',
            'STRONG_TREND',
            {'confidence': 85}
        )

        assert not allowed
        assert 'MAX_POSITIONS' in reason

    def test_state_persistence(self, paper_broker, tmp_path):
        """Test state saves and loads correctly"""

        state_dir = str(tmp_path / 'test_state')

        engine = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir=state_dir
        )

        engine.connect()

        # Save state
        engine._save_state()

        # Create new engine
        engine2 = TradingEngine(
            broker=paper_broker,
            initial_capital=10000,
            assets=['US_TECH_100'],
            state_dir=state_dir
        )

        # Load should work
        state = engine2.state_manager.load_state()
        assert state is not None
        assert 'capital' in state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
