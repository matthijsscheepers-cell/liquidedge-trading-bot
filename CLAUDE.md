# LIQUIDEDGE Trading Bot

## Project Overview
Automated trading bot running the EMA Pullback strategy on Capital.com (CFD broker, demo account). Trades GOLD, SILVER, and US500 on 15-minute charts with 1-hour trend confirmation.

## Current Strategy: EMA Pullback (Optimized 2026-02-12)

### Assets & Risk Allocation
- **GOLD** (XAU/USD) — 2% risk, 89.0% backtest WR, PF 24.94
- **SILVER** (XAG/USD) — 1% risk, 80.0% backtest WR, PF 6.33
- **US500** (S&P 500) — 1% risk, 70.5% backtest WR, PF 7.06
- US100 removed: 59% WR, blown account

Combined portfolio: 79.9% WR, -31.8% DD, $3K → $289B (backtest 2020-2026)

### Indicators (15-minute chart)
- **21 EMA** — basis for entry/stop/target calculations
- **ATR(20)** — volatility measure, all levels expressed in ATR units
- **TTM Momentum** — histogram color filter (reject red = momentum ≤ 0 AND falling)
- **Bollinger Bands** (20, 2.0 std) — used internally by TTM Squeeze

### Indicators (1-hour chart)
- **21 EMA** — anchor timeframe trend confirmation (Close must be > EMA)

### Entry Criteria (ALL must be true)
1. **1H Close > EMA(21)** — bullish trend on higher timeframe
2. **15min histogram color ≠ red** — momentum not negative AND falling
3. **Price at EMA level** — bar low reaches EMA(21) (limit order entry)
4. **NO squeeze requirement** — squeeze filter removed after optimization

### Entry/Exit Levels
| Parameter | Value | Formula |
|-----------|-------|---------|
| Entry | 0.0 ATR | EMA(21) |
| Stop Loss | -2.0 ATR | EMA(21) - 2.0 × ATR(20) |
| Take Profit | +2.0 ATR | EMA(21) + 2.0 × ATR(20) |
| Risk | 2.0 ATR | entry - stop |
| Reward | 2.0 ATR | target - entry |
| R:R | 1:1 | Edge comes from high win rate |
| Direction | LONG ONLY | |

### Order Execution
- Price at/below entry (within 0.1 ATR): **market order**
- Price above entry: **limit order** (expires after 5 minutes)
- Post-fill R:R check: reject if < 1.5:1
- Spread filter: max 0.20% for GOLD/SILVER, max 0.05% for US500
- Re-issue per bar (new limit order each 15-min bar if signal persists)

### Position Sizing — Asset-Weighted Risk with Scale A Tiers
**Base risk per asset:**
| Asset | Base Risk |
|-------|-----------|
| GOLD | 2.0% |
| SILVER | 1.0% |
| US500 | 1.0% |

**Scale A tier caps** (applied on top of base risk):
| Account Size | Max Risk |
|-------------|----------|
| < $1,000,000 | 2.0% |
| < $10,000,000 | 1.0% |
| ≥ $10,000,000 | 0.5% |

Effective risk = min(base_risk, tier_cap). Formula: `size = (capital × risk%) / stop_distance`

### Circuit Breakers
- **2 consecutive stops** on same asset → 4 hour cooldown
- **Daily loss >= 3× GOLD risk amount** → all trading halted until next day
- **Failed order** → 15 minute cooldown on that asset

### Scanning
- Normal: every 5 seconds
- After 15min bar close: every 2 seconds for 10 seconds (burst mode)
- Uses current (forming) bar for both signal and price check
- Data freshness: skip if data older than 30 minutes
- Auto-reconnect after 3 consecutive data failures

### Backtest Performance (2020-2026, 1-minute execution, combined portfolio)

**Combined Portfolio (GOLD 2% + SILVER 1% + US500 1%):**
| Metric | Value |
|--------|-------|
| Win Rate | 79.9% |
| Profit Factor | 7.55 |
| Max Drawdown | -31.8% |
| Total Trades | 6,705 |

**Per-asset breakdown:**
| Asset | Trades | WR | PF |
|-------|--------|-----|-----|
| GOLD | 1,667 | 89.7% | 19.43 |
| SILVER | 3,306 | 80.0% | 6.33 |
| US500 | 1,732 | 70.5% | 7.06 |

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
- **2026-02-10**: Corrected backtest bugs (look-ahead bias, single-side commission, missing slippage). Optimized entry from -1.0 to -0.5 ATR. Removed US100/US500. Enabled circuit breakers. Result: 88.2% WR, PF 7.44.
- **2026-02-11**: Full optimization suite with 1-minute execution. Entry moved to EMA(21) (0.0 ATR). Stop widened to -2.0 ATR, target to +2.0 ATR. Squeeze requirement removed, replaced with histogram color ≠ red. SILVER removed. Risk model changed from progressive cap to Scale A (2%→1%→0.5%). Result: 87.2% WR, PF 25.72 (Scale A), $3K→$69M.
- **2026-02-12**: Replaced 1H TTM momentum filter with Close > EMA(21). TTM momentum was negative 65-70% during gold's 2024-2025 bull run (measures rate-of-change, not direction). Close > EMA stays positive during consolidation phases. Added SILVER (1% risk) and US500 (1% risk) back — combined portfolio with asset-weighted risk: 79.9% WR, -31.8% DD, $289B. Aggressive start tested and rejected (hurts more than helps due to 2020-2021 weak periods).

## Running the Bot
```bash
cd "/Users/matthijs/Documents/Project LIQUIDEDGE"
nohup python -u paper_trading_engine.py > paper_trading.log 2>&1 &
```
Requires `.env` file with Capital.com API credentials.
