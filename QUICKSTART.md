# LIQUIDEDGE - Quick Start Guide

## Current Status

**Paper Trading Engine**: RUNNING
- Monitor: `tail -f /tmp/paper_trading_live.log`
- Stop: `pkill -f paper_trading_engine`
- Restart: `python paper_trading_engine.py`

**Strategy**: TTM Squeeze Pullback with Progressive Risk Cap
**Markets**: GOLD, SILVER (24/5) | US100, US500 (15:30-22:00 CET)
**Account**: Capital.com Demo
**Scan Interval**: 15 seconds

---

## Entry System: Smart Limit Orders

The bot uses intelligent order routing to minimize slippage:

1. **Setup detected** - TTM Squeeze fires + trend/momentum conditions met
2. **Real-time price check** - Gets live bid/ask from Capital.com
3. **Smart routing**:
   - Price at/below entry level (within 0.1%) -> **Market order** (pullback already happened)
   - Price above entry level -> **Limit order** at calculated pullback price (15-min expiry)
4. **Auto-cancel** - Unfilled limit orders expire after 15 minutes

### Why Limit Orders Matter
| Order Type | Entry Quality | Slippage |
|------------|--------------|----------|
| Market order | Wherever price is | 0.3-0.7% |
| Limit order | At calculated pullback | ~0% |

The strategy's edge depends on entering at the pullback level (EMA - 1xATR).
Market orders destroy this edge. Limit orders preserve it.

---

## Project Structure

### Production
- `paper_trading_engine.py` - Live paper trading bot (RUNNING)

### Development
- `scripts/backtests/` - Strategy backtesting (9 scripts)
- `scripts/testing/` - API connection tests (15 scripts)
- `scripts/debugging/` - Diagnostic tools (9 scripts)
- `scripts/calibration/` - Parameter optimization (2 scripts)
- `scripts/data/` - Data management (5 scripts)

### Documentation
- `docs/` - Complete strategy documentation (7 files)
- `results/` - Backtest results (3 files)
- `README.md` - Full project documentation

### Core
- `src/` - Source code (indicators, strategies, execution)
- `config/` - Configuration files
- `tests/` - Unit tests

---

## Key Results

### Backtest Performance (2010-2026, 15.5 years)
Starting Capital: **$300**
Final Capital: **$16,772,834**
Return: **+5,590,844%**
Max Drawdown: **3.1%**
Win Rate: **77%**

### Stress Test Results
| Slippage | Result | Status |
|----------|--------|--------|
| 0.1% | +3.4M% | Excellent |
| 0.2% | +1.5M% | Good |
| 0.5% | -100% | Fails |

**Critical**: Strategy requires spreads + slippage < 0.3% total.
This is NOT a scaling issue - it affects every trade from trade #1.
Limit orders are essential to keep slippage near zero.

---

## Common Commands

### Paper Trading
```bash
# Watch live
tail -f /tmp/paper_trading_live.log

# Watch entries/exits only
tail -f /tmp/paper_trading_live.log | grep -E "ENTRY|FILLED|STOP|TARGET|LIMIT"

# Stop
pkill -f paper_trading_engine

# Restart
cd "/Users/matthijs/Documents/Project LIQUIDEDGE"
nohup python3 -u paper_trading_engine.py > /tmp/paper_trading_live.log 2>&1 &
```

### Backtesting
```bash
python scripts/backtests/backtest_stress_test.py
python scripts/backtests/backtest_user_strategy_2yr.py
```

### Testing
```bash
python scripts/testing/test_full_connection.py
python scripts/testing/test_api_key.py
```

---

## Risk Management

### Progressive Risk Cap
| Capital | Max Risk/Trade |
|---------|---------------|
| < $1,000 | $50 |
| $1,000 - $5,000 | $100 |
| $5,000 - $20,000 | $200 |
| $20,000 - $100,000 | $500 |
| > $100,000 | $1,000 |

### Position Sizing
- Base risk: 2% of capital per trade
- Capped by progressive risk cap above
- CFD leverage: 1:20 (5% margin)
- Max 4 concurrent positions

### Safety Features
- Duplicate order prevention (broker sync + pending lock)
- Data freshness check (rejects data older than 30 min)
- Automatic stop loss and take profit on every order
- Limit order expiry (15 min) prevents stale orders

---

## Next Steps

1. **Monitor paper trading** for 1-2 weeks
2. **Track execution quality** - limit order fill rates, actual slippage
3. **Verify win rate** matches backtest (~77%)
4. **If successful**, consider small live account (300-500 EUR)

---

## Important Notes

- Strategy is **execution-sensitive** - needs tight spreads
- Focus trading during **US regular hours** (15:30-22:00 CET)
- GOLD/SILVER trade 24/5, US indices only during US hours
- Limit orders are critical for matching backtest performance
- Monitor first 10 trades closely for execution quality

---

## Documentation

- `docs/STRATEGY_OVERVIEW.md` - Strategy explanation
- `docs/TTM_SQUEEZE_EXECUTION_DETAIL.md` - Entry/exit logic
- `docs/CAPITAL_COM_SETUP.md` - Broker setup
- `scripts/README.md` - Script documentation
- `results/progressive_results.txt` - Detailed backtest results

---

**Built with**: Python 3.14, Capital.com API, Databento historical data
**Strategy**: TTM Squeeze Pullback (15min execution, 1H trend filter)
**Risk Management**: Progressive Risk Cap ($50-$1000)
**Order Execution**: Smart limit orders with real-time price lookahead
