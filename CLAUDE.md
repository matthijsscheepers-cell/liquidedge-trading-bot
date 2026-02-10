# LIQUIDEDGE Trading Bot

## Project Overview
Automated trading bot running the TTM Squeeze Pullback strategy on Capital.com (CFD broker, demo account). Optimized for GOLD and SILVER on 15-minute charts with 1-hour trend confirmation.

## Current Strategy: TTM Squeeze Pullback (Optimized 2026-02-10)

### Assets
- **GOLD** (XAU/USD) — 94.8% backtest WR
- **SILVER** (XAG/USD) — 81.9% backtest WR
- US100/US500 removed after optimization (lower WR, higher drawdown, spread-sensitive)

### Indicators (15-minute chart)
- **21 EMA** — basis for entry/stop/target calculations
- **ATR(20)** — volatility measure, all levels expressed in ATR units
- **TTM Squeeze (Beardy Squeeze Pro)** — 3 Keltner Channel levels:
  - KC x1.0 (narrowest) = orange dots = tightest squeeze (intensity 3)
  - KC x1.5 = red dots = medium squeeze (intensity 2)
  - KC x2.0 (widest) = black dots = first level squeeze (intensity 1)
  - Green dots = no squeeze (intensity 0)
- **TTM Momentum** — trend direction (positive = bullish)
- **Bollinger Bands** (20, 2.0 std) — used internally by TTM Squeeze

### Indicators (1-hour chart)
- **TTM Momentum** — anchor timeframe trend confirmation

### Entry Criteria (ALL must be true)
1. **1H TTM momentum > 0** — bullish trend on higher timeframe
2. **15min Squeeze active** — BB inside KC x2.0 (black/red/orange dots)
3. **Price at -0.5 ATR level** — bar low within ±0.5 ATR of (EMA - 0.5 × ATR)

### Entry/Exit Levels
| Parameter | Value | Formula |
|-----------|-------|---------|
| Entry | -0.5 ATR | EMA(21) - 0.5 × ATR(20) |
| Stop Loss | -1.5 ATR | EMA(21) - 1.5 × ATR(20) |
| Take Profit | +1.5 ATR | EMA(21) + 1.5 × ATR(20) |
| Risk | 1.0 ATR | entry - stop |
| Reward | 2.0 ATR | target - entry |
| R:R | 2.0:1 | |
| Direction | LONG ONLY | |

### Order Execution
- Price at/below entry (within 0.1 ATR): **market order**
- Price above entry: **limit order** (expires after 5 minutes)
- Post-fill R:R check: reject if < 1.5:1
- Spread filter: max 0.20% for both GOLD and SILVER

### Position Sizing — Progressive Risk Cap
| Account Size | Max Risk/Trade |
|-------------|----------------|
| < $1,000 | $50 |
| < $5,000 | $100 |
| < $20,000 | $200 |
| < $100,000 | $500 |
| >= $100,000 | $1,000 |

Formula: `risk = min(capital × 2%, risk_cap)`, `size = risk / stop_distance`, `margin = size × price / 20`

### Circuit Breakers
- **2 consecutive stops** on same asset → 4 hour cooldown
- **Daily loss >= 3× risk cap** → all trading halted until next day
- **Failed order** → 15 minute cooldown on that asset

### Scanning
- Normal: every 5 seconds
- After 15min bar close: every 2 seconds for 10 seconds (burst mode)
- Uses current (forming) bar for both signal and price check
- Data freshness: skip if data older than 30 minutes
- Auto-reconnect after 3 consecutive data failures

### Backtest Performance (2010-2026, corrected)
| Metric | Value |
|--------|-------|
| Win Rate | 88.2% |
| Profit Factor | 7.44 |
| Max Drawdown | -12.7% |
| Total Trades | 9,759 |
| GOLD WR | 94.8% |
| SILVER WR | 81.9% |
| R:R | 2.0:1 |

With progressive risk cap: $300 → $14.5M over 15.5 years (backtest).

## Key Files
- `paper_trading_engine.py` — main bot, runs 24/7
- `src/strategies/ttm_pullback.py` — strategy logic (entry/exit)
- `src/indicators/ttm.py` — TTM Squeeze with 3 KC levels (Beardy Squeeze Pro)
- `src/indicators/trend.py` — EMA calculations
- `src/indicators/volatility.py` — ATR calculations
- `src/execution/capital_connector.py` — Capital.com API wrapper
- `src/execution/databento_loader.py` — historical data loader (Databento DBN files)
- `scripts/backtests/` — all backtest scripts
- `results/` — backtest results and trade logs

## Optimization History
- **2026-02-10**: Corrected backtest bugs (look-ahead bias, single-side commission, missing slippage). Optimized entry from -1.0 to -0.5 ATR. Removed US100/US500. Enabled circuit breakers. Result: 88.2% WR, PF 7.44. Tested partial exits (trailing stops) — rejected, hurts performance.

## Running the Bot
```bash
cd "/Users/matthijs/Documents/Project LIQUIDEDGE"
nohup python -u paper_trading_engine.py > paper_trading.log 2>&1 &
```
Requires `.env` file with Capital.com API credentials.
