"""
Quick test to verify execution module imports and basic functionality
"""

import sys
import os

# Add the execution module directory directly to path to bypass src/__init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'execution'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import directly from files
from broker_interface import BrokerInterface, OrderResult, AccountInfo, Position
from paper_trader import PaperTrader

print("="*60)
print("EXECUTION MODULE TEST")
print("="*60)

# Test 1: Import verification
print("\n1. ✓ All imports successful")
print(f"   - BrokerInterface: {BrokerInterface}")
print(f"   - OrderResult: {OrderResult}")
print(f"   - AccountInfo: {AccountInfo}")
print(f"   - Position: {Position}")
print(f"   - PaperTrader: {PaperTrader}")

# Test 2: Create PaperTrader instance
print("\n2. Creating PaperTrader instance...")
config = {
    'initial_balance': 10000,
    'currency': 'USD',
    'slippage_pct': 0.001,
    'spread_multiplier': 1.5
}
trader = PaperTrader(config)
print(f"   ✓ PaperTrader created with ${trader.balance:.2f}")

# Test 3: Connect
print("\n3. Connecting to paper trader...")
connected = trader.connect()
assert connected, "Connection should succeed"
print(f"   ✓ Connected: {trader.is_connected}")

# Test 4: Get account info
print("\n4. Getting account info...")
account = trader.get_account_info()
print(f"   Balance: ${account.balance:.2f}")
print(f"   Equity: ${account.equity:.2f}")
print(f"   Currency: {account.currency}")
assert account.balance == 10000, "Initial balance should be 10000"
print("   ✓ Account info retrieved")

# Test 5: Get current price
print("\n5. Getting current price...")
bid, ask = trader.get_current_price('US_TECH_100')
print(f"   Bid: {bid:.2f}")
print(f"   Ask: {ask:.2f}")
print(f"   Spread: {ask - bid:.2f}")
assert bid > 0 and ask > 0, "Prices should be positive"
assert ask > bid, "Ask should be higher than bid"
print("   ✓ Price retrieval working")

# Test 6: Place market order
print("\n6. Placing market order...")
result = trader.place_market_order(
    asset='US_TECH_100',
    direction='LONG',
    units=10,
    stop_loss=None,
    take_profit=None
)
assert result.success, "Order should succeed"
print(f"   Order ID: {result.order_id}")
print(f"   Fill Price: {result.fill_price:.2f}")
print(f"   Units: {result.filled_units}")
print("   ✓ Order placed successfully")

# Test 7: Get open positions
print("\n7. Getting open positions...")
positions = trader.get_open_positions()
assert len(positions) == 1, "Should have 1 open position"
pos = positions[0]
print(f"   Asset: {pos.asset}")
print(f"   Direction: {pos.direction}")
print(f"   Units: {pos.units}")
print(f"   Entry: {pos.entry_price:.2f}")
print(f"   Current: {pos.current_price:.2f}")
print(f"   P&L: ${pos.unrealized_pnl:.2f}")
print("   ✓ Position tracking working")

# Test 8: Close position
print("\n8. Closing position...")
close_result = trader.close_position('US_TECH_100')
assert close_result.success, "Close should succeed"
print(f"   Close Price: {close_result.fill_price:.2f}")
print(f"   P&L: ${close_result.metadata['pnl']:.2f}")
print(f"   New Balance: ${trader.balance:.2f}")
print("   ✓ Position closed")

# Test 9: Performance summary
print("\n9. Getting performance summary...")
summary = trader.get_performance_summary()
print(f"   Total Trades: {summary['total_trades']}")
print(f"   Total P&L: ${summary['total_pnl']:.2f}")
print(f"   Win Rate: {summary['win_rate']*100:.1f}%")
print(f"   Return: {summary['return_pct']:.2f}%")
print("   ✓ Performance tracking working")

# Test 10: Disconnect
print("\n10. Disconnecting...")
trader.disconnect()
assert not trader.is_connected, "Should be disconnected"
print("   ✓ Disconnected")

print("\n" + "="*60)
print("✅ ALL EXECUTION MODULE TESTS PASSED!")
print("="*60)
