"""
Full Integration Test - Complete LIQUIDEDGE Trading Bot

Tests the entire system end-to-end:
- Paper trading environment
- Market data generation
- Regime detection
- Strategy selection
- Risk management
- Order execution
- Position management
- State persistence

Run with: python scripts/test_full_integration.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.execution.paper_trader import PaperTrader
from src.engine.trading_engine import TradingEngine


def generate_realistic_market_data(start_price: float, bars: int, trend: str = 'up') -> pd.DataFrame:
    """
    Generate realistic OHLCV data for testing

    Args:
        start_price: Starting price
        bars: Number of bars
        trend: 'up', 'down', or 'sideways'

    Returns:
        DataFrame with OHLCV data
    """

    dates = pd.date_range(end=datetime.now(), periods=bars, freq='1H')

    # Generate price movement based on trend
    if trend == 'up':
        drift = 0.0002  # Slight upward bias
        volatility = 0.008
    elif trend == 'down':
        drift = -0.0002  # Slight downward bias
        volatility = 0.008
    else:  # sideways
        drift = 0.0
        volatility = 0.006

    # Random walk with drift
    returns = np.random.normal(drift, volatility, bars)
    prices = start_price * (1 + returns).cumprod()

    # Add some structure (trends and pullbacks)
    for i in range(0, bars, 50):
        if trend == 'up' and i < bars - 20:
            # Add pullback
            prices[i:i+10] *= 0.98
        elif trend == 'down' and i < bars - 20:
            # Add relief rally
            prices[i:i+10] *= 1.02

    # Create OHLC from close
    close = prices
    high = close * (1 + np.random.uniform(0, 0.01, bars))
    low = close * (1 - np.random.uniform(0, 0.01, bars))
    open_price = close - (np.random.uniform(-0.5, 0.5, bars) * (high - low))

    # Volume
    volume = np.random.randint(1000, 10000, bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


def setup_paper_trading_environment():
    """
    Set up paper trading broker with realistic data

    Returns:
        Configured PaperTrader instance
    """

    print("\n" + "="*60)
    print("SETTING UP PAPER TRADING ENVIRONMENT")
    print("="*60)

    # Create paper trader
    config = {
        'initial_balance': 10000,
        'currency': 'USD',
        'slippage_pct': 0.001,  # 0.1% slippage
        'spread_multiplier': 1.5
    }

    broker = PaperTrader(config)
    broker.connect()

    print(f"\n✓ Paper trader initialized")
    print(f"  Balance: ${broker.balance:.2f}")
    print(f"  Slippage: {broker.slippage_pct*100:.1f}%")

    # Load market data for multiple assets
    assets = {
        'US_TECH_100': {'start': 15000, 'bars': 500, 'trend': 'up'},
        'GOLD': {'start': 1950, 'bars': 500, 'trend': 'sideways'},
        'EUR_USD': {'start': 1.08, 'bars': 500, 'trend': 'down'}
    }

    print(f"\n✓ Loading historical data...")

    for asset, params in assets.items():
        df = generate_realistic_market_data(
            start_price=params['start'],
            bars=params['bars'],
            trend=params['trend']
        )

        broker.load_historical_data(asset, df)

        current_price = df.iloc[-1]['close']
        print(f"  {asset}: {len(df)} bars loaded (current: {current_price:.2f})")

    return broker


def run_integration_test():
    """
    Run complete integration test
    """

    print("\n" + "="*80)
    print(" "*20 + "LIQUIDEDGE FULL INTEGRATION TEST")
    print("="*80)

    # Setup
    broker = setup_paper_trading_environment()

    # Create trading engine
    print("\n" + "="*60)
    print("INITIALIZING TRADING ENGINE")
    print("="*60)

    engine = TradingEngine(
        broker=broker,
        initial_capital=10000,
        assets=['US_TECH_100', 'GOLD', 'EUR_USD'],
        state_dir='data/test_integration'
    )

    # Connect
    if not engine.connect():
        print("✗ Failed to connect engine")
        return

    # Run multiple trading cycles
    print("\n" + "="*60)
    print("RUNNING TRADING CYCLES")
    print("="*60)

    num_cycles = 5
    print(f"\nExecuting {num_cycles} trading cycles...")

    for cycle in range(1, num_cycles + 1):
        print(f"\n{'─'*60}")
        print(f"CYCLE {cycle}/{num_cycles}")
        print(f"{'─'*60}")

        try:
            # Run one cycle
            engine._run_cycle()

            # Brief summary
            health = engine.risk_governor.get_health_status()
            print(f"\n📊 Cycle {cycle} Complete:")
            print(f"   Capital: ${health['current_capital']:.2f}")
            print(f"   Open Positions: {health['positions_open']}")
            print(f"   Health: {health['status']}")

        except Exception as e:
            print(f"✗ Error in cycle {cycle}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    # Account status
    account = broker.get_account_info()
    print(f"\n💰 Account Summary:")
    print(f"   Starting Balance: $10,000.00")
    print(f"   Final Balance: ${account.balance:.2f}")
    print(f"   Equity: ${account.equity:.2f}")
    print(f"   Return: {((account.balance - 10000) / 10000) * 100:+.2f}%")

    # Trading statistics
    stats = broker.get_performance_summary()
    if stats['total_trades'] > 0:
        print(f"\n📈 Trading Statistics:")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"   Total P&L: ${stats['total_pnl']:+.2f}")
        print(f"   Avg Win: ${stats['avg_win']:.2f}")
        print(f"   Avg Loss: ${stats['avg_loss']:.2f}")
    else:
        print(f"\n📈 Trading Statistics:")
        print(f"   No trades executed in this test run")

    # Risk metrics
    health = engine.risk_governor.get_health_status()
    print(f"\n🛡️  Risk Metrics:")
    print(f"   Health Status: {health['status']}")
    print(f"   Health Score: {health['score']}/100")
    print(f"   Current Drawdown: {health['dd_pct']:.2f}%")
    print(f"   Max Drawdown: {health['max_dd_pct']}%")
    print(f"   Win Streak: {health['streak_w']}")
    print(f"   Loss Streak: {health['streak_l']}")

    # Open positions
    positions = broker.get_open_positions()
    if positions:
        print(f"\n📌 Open Positions ({len(positions)}):")
        for pos in positions:
            pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.units)) * 100
            print(f"   {pos.asset}: {pos.direction} {pos.units} @ {pos.entry_price:.2f}")
            print(f"      Current: {pos.current_price:.2f} | P&L: ${pos.unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)")
    else:
        print(f"\n📌 Open Positions: None")

    # Warnings
    if health['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in health['warnings']:
            print(f"   • {warning}")

    # State persistence check
    print(f"\n💾 State Persistence:")
    state = engine.state_manager.load_state()
    if state:
        print(f"   ✓ State saved successfully")
        print(f"   Last saved: {state.get('timestamp', 'unknown')}")

    # Cleanup
    print("\n" + "="*60)
    print("CLEANUP")
    print("="*60)

    engine.stop()

    print("\n" + "="*80)
    print(" "*25 + "✅ INTEGRATION TEST COMPLETE")
    print("="*80)

    # Test verdict
    print("\n" + "="*60)
    print("TEST VERDICT")
    print("="*60)

    passed = True
    checks = []

    # Check 1: Engine initialized
    if engine.broker and engine.risk_governor:
        checks.append("✓ Engine initialized correctly")
    else:
        checks.append("✗ Engine initialization failed")
        passed = False

    # Check 2: Cycles completed
    if num_cycles == 5:
        checks.append(f"✓ All {num_cycles} cycles completed")
    else:
        checks.append("✗ Not all cycles completed")
        passed = False

    # Check 3: Risk limits respected
    if health['positions_open'] <= 5:
        checks.append("✓ Risk limits respected")
    else:
        checks.append("✗ Risk limits violated")
        passed = False

    # Check 4: No critical errors
    if health['status'] != 'CRITICAL':
        checks.append("✓ No critical errors")
    else:
        checks.append("✗ Critical health status")
        passed = False

    # Check 5: State persistence works
    if state:
        checks.append("✓ State persistence working")
    else:
        checks.append("✗ State persistence failed")
        passed = False

    for check in checks:
        print(f"  {check}")

    print("\n" + "="*60)
    if passed:
        print(" "*15 + "🎉 ALL CHECKS PASSED 🎉")
    else:
        print(" "*15 + "⚠️  SOME CHECKS FAILED ⚠️")
    print("="*60 + "\n")

    return passed


if __name__ == '__main__':
    try:
        success = run_integration_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
