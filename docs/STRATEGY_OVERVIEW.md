# TTM Squeeze Strategy - Huidige Configuratie

## Overzicht

Na alle optimalisaties is dit de huidige TTM Squeeze strategy setup:

---

## Indicator Parameters

### Bollinger Bands
- Period: 20
- Standard Deviation: **2.5** (gecalibreerd voor intraday futures)
- Basis: SMA(close, 20)

### Keltner Channels
- Period: 20
- Multiplier: **0.5** (gecalibreerd voor intraday futures)
- Basis: EMA(close, 20)
- Range: Wilder's ATR(20)

### TTM Momentum
- Period: 12
- Type: Linear regression momentum

### Squeeze Detection
- **Squeeze ON**: Bollinger Bands INSIDE Keltner Channels (22.8% van bars)
- **Squeeze OFF**: Bollinger Bands OUTSIDE Keltner Channels
- Releases: ~50 per 1743 bars (goede frequentie)

---

## Entry Regels

### Filters
1. **Regime Filter**: ❌ DISABLED (was: only RANGE_COMPRESSION)
   - Reden: Te restrictief, miste 99% van opportunities

2. **Direction Filter**: ✅ ENABLED - ONLY LONG
   - Reden: SHORT trades hebben 0% win rate
   - LONG trades hebben 83% win rate

3. **Session Filter**: ❌ DISABLED (was: only NY session)
   - Reden: 24/7 futures trading

4. **News Blackout**: ❌ DISABLED
   - Reden: Te restrictief voor 24/7 trading

### Entry Types

#### 1. SQUEEZE_RELEASE_LONG
**Trigger:**
- Squeeze just turned OFF (prev bar ON, current OFF)
- Momentum > 0 (bullish)
- Momentum increasing OR momentum > min_threshold (0.15 for GOLD)

**Entry:**
- Price: Current close
- Stop: Entry - (1.8 * ATR) for GOLD
- Target: Entry + (Risk * 1.6) for GOLD

**Performance:**
- ✅ 80-83% win rate
- ✅ Beste setup type
- 5-6 trades per 3 maanden (GOLD only)

#### 2. SQUEEZE_RETEST_LONG
**Trigger:**
- Squeeze released 1-3 bars ago
- Price near KC middle line (within 0.4 ATR for GOLD)
- Momentum > 0 (bullish)
- Bullish rejection candle (lower wick > 50% of range)

**Entry:**
- Price: Current close
- Stop: Entry - (1.8 * ATR) for GOLD
- Target: Entry + (Risk * 1.6) for GOLD

**Performance:**
- Performance varies (need more data)
- Requires rejection confirmation
- 0-1 trades per 3 maanden (GOLD only)

---

## Asset-Specific Parameters (GOLD)

```python
{
    'initial_stop_atr': 1.8,        # Stop distance in ATR
    'min_rrr': 1.6,                 # Minimum reward-to-risk ratio
    'breakeven_r': 1.0,             # Move to breakeven at 1.0R
    'trail_start_r': 1.5,           # Start trailing at 1.5R
    'trail_distance_atr': 1.3,      # Trail distance
    'max_bars': 60,                 # Max hold time (15 bars)
    'retest_tolerance': 0.4,        # Distance to KC for retest
    'min_momentum': 0.15,           # Minimum momentum threshold
}
```

---

## Exit Management

### Exits (in order of priority):
1. **Stop Loss Hit**: Exit at stop price
2. **Target Hit**: Exit at target price
3. **Momentum Reversal**: If in profit (>0.5R) and momentum reverses direction
4. **Breakeven**: Move stop to entry at 1.0R profit
5. **Trailing Stop**: Start trailing at 1.5R profit
6. **Time Exit**: Max 60 bars for GOLD

---

## Current Performance (GOLD, 3 months)

### Metrics
- **Total Trades**: 6
- **Win Rate**: 83.3% (5 wins, 1 loss)
- **Return**: +7.07%
- **Profit Factor**: 7.74
- **Avg Win**: $162.35
- **Avg Loss**: -$104.94

### Trade Breakdown
- SQUEEZE_RELEASE_LONG: 6 trades, 83.3% win rate, +$706.82 P&L ✅
- SQUEEZE_RETEST_LONG: 0 trades (setup rarely occurs)

---

## Key Changes from Original

### ❌ Disabled (te restrictief):
1. Regime filtering (RANGE_COMPRESSION only)
2. Session hours filtering (NY session only)
3. News blackout filtering
4. SHORT trade direction

### ✅ Enabled (verbeterde performance):
1. LONG-only trading (83% win rate)
2. 24/7 trading (meer opportunities)
3. Gecalibreerde indicator parameters (BB 2.5, KC 0.5)
4. Momentum-based entry confirmation

---

## Next Steps

### Te Testen:
1. **Multiple Assets**: GOLD + SILVER + US100 + US500
2. **Longer Timeframe**: 6-12 maanden backtest
3. **Position Sizing**: Huidige 1% risk per trade
4. **Correlation**: Voorkomen van teveel correlated positions

### Mogelijke Verbeteringen:
1. Dynamic position sizing based on volatility
2. Pyramiding (add to winning positions)
3. Partial profit taking (exit 50% at 1.5R, rest trails)
4. Adaptive parameters based on market conditions
