# LIQUIDEDGE - Quick Start Guide

## 🚀 Current Status

**Paper Trading Engine**: RUNNING ✅
- Monitor: `tail -f /tmp/paper_trading_live.log`
- Stop: `pkill -f paper_trading_engine`
- Restart: `python paper_trading_engine.py`

**Strategy**: TTM Squeeze Pullback with Progressive Risk Cap
**Markets**: GOLD, SILVER (operational 24/5) | US100, US500 (15:30-22:00 CET)
**Account**: Capital.com Demo (€1,000)

---

## 📁 Project Structure

### Production
- `paper_trading_engine.py` - Live paper trading bot (RUNNING)

### Development
- `scripts/backtests/` - Strategy backtesting
- `scripts/testing/` - API connection tests
- `scripts/debugging/` - Diagnostic tools
- `scripts/calibration/` - Parameter optimization
- `scripts/data/` - Data management

### Documentation
- `docs/` - Complete strategy documentation
- `results/` - Backtest results
- `README.md` - Full project documentation

### Core
- `src/` - Source code (indicators, strategies, execution)
- `config/` - Configuration files
- `tests/` - Unit tests

---

## 🎯 Key Results

### Backtest Performance (2010-2026, 15.5 years)
Starting Capital: **$300**
Final Capital: **$16,772,834**
Return: **+5,590,844%**
Max Drawdown: **3.1%**
Win Rate: **77%**

### Stress Test Results
| Slippage | Result | Status |
|----------|--------|--------|
| 0.1% | +3.4M% | ✅ Excellent |
| 0.2% | +1.5M% | ✅ Good |
| 0.5% | -100% | ❌ Fails |

**Critical**: Strategy requires spreads + slippage < 0.3% total

---

## 🔧 Common Commands

### Paper Trading
```bash
# Watch live
tail -f /tmp/paper_trading_live.log

# Stop
pkill -f paper_trading_engine

# Restart
python paper_trading_engine.py
```

### Backtesting
```bash
# Run stress test
python scripts/backtests/backtest_stress_test.py

# Full historical backtest
python scripts/backtests/backtest_user_strategy_2yr.py
```

### Testing
```bash
# Test Capital.com connection
python scripts/testing/test_full_connection.py

# Test API key
python scripts/testing/test_api_key.py
```

---

## 📊 Next Steps

1. **Monitor paper trading** for 1-2 weeks
2. **Track execution quality** (slippage < 0.2% required)
3. **Verify win rate** matches backtest (~77%)
4. **If successful**, consider small live account (€300-500)

---

## ⚠️ Important Notes

- Strategy is **execution-sensitive** - needs tight spreads
- Focus trading during **US regular hours** (15:30-22:00 CET)
- Progressive Risk Cap: $50 → $100 → $200 → $500 → $1000
- Monitor first 10 trades closely for execution quality

---

## 📚 Documentation

- `docs/STRATEGY_OVERVIEW.md` - Strategy explanation
- `docs/TTM_SQUEEZE_EXECUTION_DETAIL.md` - Entry/exit logic
- `docs/CAPITAL_COM_SETUP.md` - Broker setup
- `scripts/README.md` - Script documentation
- `results/progressive_results.txt` - Detailed backtest results

---

**Built with**: Python 3.14, Capital.com API, Databento historical data
**Strategy**: TTM Squeeze Pullback (15min execution, 1H trend filter)
**Risk Management**: Progressive Risk Cap ($50-$1000)
