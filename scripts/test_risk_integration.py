"""
Quick integration test for risk management system
"""

import sys
sys.path.insert(0, '.')

from src.risk import RiskGovernor


def test_risk_system():
    """Test complete risk management flow"""

    print("="*60)
    print("RISK MANAGEMENT INTEGRATION TEST")
    print("="*60)

    # Initialize governor
    governor = RiskGovernor(initial_capital=10000)

    print(f"\n1. Initial State")
    print(f"   Profile: {governor.profile.profile_name}")
    print(f"   Base Risk: {governor.profile.base_risk_pct*100:.1f}%")
    print(f"   Max DD: {governor.profile.max_drawdown_pct*100:.0f}%")

    # Test normal trade
    print(f"\n2. Normal Trade Check")
    allowed, reason, risk = governor.can_open_trade(
        asset='US_TECH_100',
        regime='STRONG_TREND',
        setup={'confidence': 85, 'risk_per_share': 2.0, 'entry_price': 100.0}
    )
    print(f"   Allowed: {allowed}")
    print(f"   Reason: {reason}")
    print(f"   Risk: {risk*100:.2f}%")
    assert allowed, "Should allow normal trade"

    # Calculate position size
    print(f"\n3. Position Size Calculation")
    units = governor.calculate_position_size(
        risk_pct=risk,
        risk_per_share=2.0,
        price=100.0,
        asset='US_TECH_100'
    )
    print(f"   Units: {units}")
    print(f"   Position Value: €{units * 100:.0f}")
    assert units > 0, "Should calculate valid position size"

    # Simulate winning trade
    print(f"\n4. Record Winning Trade")
    governor.add_position('US_TECH_100', {'test': 'data'})
    governor.record_trade(
        asset='US_TECH_100',
        direction='LONG',
        entry_price=100.0,
        exit_price=105.0,
        units=units,
        outcome='WIN',
        setup_type='PULLBACK_LONG',
        exit_reason='TARGET'
    )
    print(f"   Capital After: €{governor.current_capital:.0f}")
    print(f"   Win Streak: {governor.consecutive_wins}")

    # Check health
    print(f"\n5. Health Status")
    health = governor.get_health_status()
    print(f"   Status: {health['status']}")
    print(f"   Score: {health['score']}/100")
    print(f"   Warnings: {len(health['warnings'])}")

    # Test drawdown scenario
    print(f"\n6. Drawdown Scenario")
    governor.current_capital = 9200
    governor.current_dd_pct = 0.08  # 8% DD (below 10% max)

    allowed, reason, risk_dd = governor.can_open_trade(
        asset='EUR_USD',
        regime='STRONG_TREND',
        setup={'confidence': 85, 'risk_per_share': 0.001, 'entry_price': 1.1}
    )
    print(f"   Allowed: {allowed}")
    print(f"   Risk (in DD): {risk_dd*100:.2f}%")
    print(f"   Risk Reduction: {((risk - risk_dd)/risk)*100:.0f}%")
    assert allowed, "Should still allow trade in moderate DD"
    assert risk_dd < risk, "Risk should be reduced in DD"

    # Test max DD block
    print(f"\n7. Max Drawdown Block")
    governor.current_dd_pct = 0.12

    allowed, reason, _ = governor.can_open_trade(
        asset='GOLD',
        regime='STRONG_TREND',
        setup={'confidence': 85}
    )
    print(f"   Allowed: {allowed}")
    print(f"   Reason: {reason}")
    assert not allowed, "Should block at max DD"
    assert reason == "MAX_DRAWDOWN_REACHED"

    print(f"\n{'='*60}")
    print("✅ ALL RISK INTEGRATION TESTS PASSED!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_risk_system()
