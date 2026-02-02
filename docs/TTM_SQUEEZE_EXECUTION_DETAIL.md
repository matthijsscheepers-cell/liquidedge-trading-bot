# TTM Squeeze Strategy - Exacte Trade Execution

## Per Trade Flow (GOLD Example)

---

## ENTRY SCAN (Elke 15-min Bar)

### Stap 1: Indicator Check
```
Vereiste indicators:
- squeeze_on (boolean): Is BB binnen KC?
- ttm_momentum (float): Linear regression momentum
- kc_middle (float): Keltner Channel middellijn
- atr_14 (float): 14-period ATR
```

### Stap 2: Entry Type Detection

#### TYPE A: SQUEEZE_RELEASE_LONG

**Trigger Conditions:**
```python
# Squeeze moet NET vrijkomen
prev_bar['squeeze_on'] == True
current_bar['squeeze_on'] == False

# Momentum moet bullish zijn
current_bar['ttm_momentum'] > 0

# EN één van deze:
abs(current_momentum) > abs(prev_momentum)  # Momentum increasing
# OF
abs(current_momentum) >= 0.15  # Momentum strong enough (GOLD threshold)
```

**Entry Execution:**
```python
# GOLD parameters (from _get_asset_params)
initial_stop_atr = 1.8
min_rrr = 1.6

# Price levels
entry_price = current_bar['close']
atr = current_bar['atr_14']

# Stop loss (LONG)
stop_loss = entry_price - (1.8 * atr)
# Example: Entry $5000, ATR $50
# Stop = $5000 - (1.8 * $50) = $5000 - $90 = $4910

# Risk calculation
risk_per_share = entry_price - stop_loss
# Example: $5000 - $4910 = $90 risk per unit

# Target (based on RRR)
target = entry_price + (risk_per_share * 1.6)
# Example: $5000 + ($90 * 1.6) = $5000 + $144 = $5144

# Position sizing (1% account risk)
account_capital = $10,000
risk_per_trade = $10,000 * 0.01 = $100
position_size = $100 / $90 = 1.11 units

# Total position value
position_value = 1.11 * $5000 = $5,550
```

**Setup Object Created:**
```python
TradeSetup(
    direction = LONG,
    entry_price = $5000.00,
    stop_loss = $4910.00,
    target = $5144.00,
    risk_per_share = $90.00,
    confidence = 80.0,
    setup_type = "SQUEEZE_RELEASE_LONG",
    metadata = {
        'kc_middle': $4980.00,
        'atr_14': $50.00,
        'ttm_momentum': 0.25,
        'regime': 'ANY'
    }
)
```

---

#### TYPE B: SQUEEZE_RETEST_LONG

**Trigger Conditions:**
```python
# Squeeze released 1-3 bars ago
# Check last 3 bars for a release
for i in range(1, 4):
    if df.iloc[-i-1]['squeeze_on'] == True and df.iloc[-i]['squeeze_on'] == False:
        squeeze_released_recently = True
        break

# Price must be near KC middle line
distance_to_kc = abs(current['close'] - current['kc_middle']) / atr
distance_to_kc <= 0.4  # GOLD tolerance

# Momentum must be bullish
current['ttm_momentum'] > 0

# Bullish rejection candle required
lower_wick = min(open, close) - low
total_range = high - low
lower_wick / total_range > 0.5  # Rejection confirmed
```

**Entry Execution:**
```python
# Same parameters as RELEASE
entry_price = current_bar['close']
stop_loss = entry_price - (1.8 * atr)
risk_per_share = entry_price - stop_loss
target = entry_price + (risk_per_share * 1.6)

# Position sizing (same formula)
position_size = (account_capital * 0.01) / risk_per_share
```

---

## POSITION MANAGEMENT (Elke Bar)

### Exit Checks (in priority order):

#### 1. STOP LOSS HIT
```python
if current_bar['low'] <= position['stop_loss']:
    exit_price = position['stop_loss']
    exit_reason = "STOP"

    # P&L calculation
    pnl = (exit_price - entry_price) * size - (exit_price * size * 0.001)
    # Example: ($4910 - $5000) * 1.11 - ($4910 * 1.11 * 0.001)
    # = -$99.90 - $5.45 = -$105.35 (loss + commission)
```

#### 2. TARGET HIT
```python
if current_bar['high'] >= position['target']:
    exit_price = position['target']
    exit_reason = "TARGET"

    # P&L calculation
    pnl = (exit_price - entry_price) * size - (exit_price * size * 0.001)
    # Example: ($5144 - $5000) * 1.11 - ($5144 * 1.11 * 0.001)
    # = $159.84 - $5.71 = $154.13 (profit - commission)
```

#### 3. MOMENTUM REVERSAL (Squeeze Specific!)
```python
# Calculate R-multiple
current_r = (current_price - entry_price) / risk_per_share
# Example: ($5050 - $5000) / $90 = 0.56R

if current_r > 0.5:  # In profit
    entry_momentum = position.metadata['ttm_momentum']  # e.g., 0.25
    current_momentum = current_bar['ttm_momentum']       # e.g., -0.10

    # Momentum reversed from positive to negative
    if entry_momentum > 0 and current_momentum < 0:
        exit_price = current_bar['close']
        exit_reason = "MOMENTUM_REVERSAL"
```

#### 4. MOVE TO BREAKEVEN
```python
current_r = (current_price - entry_price) / risk_per_share

if current_r >= 1.0 and stop_loss < entry_price:  # At 1.0R profit
    new_stop = entry_price  # Move stop to breakeven
    exit_action = "BREAKEVEN"

    # Example:
    # Entry: $5000, Current: $5090, Risk: $90
    # R = ($5090 - $5000) / $90 = 1.0R
    # Move stop from $4910 to $5000 (breakeven)
```

#### 5. TRAILING STOP
```python
if current_r >= 1.5:  # At 1.5R profit, start trailing
    trail_distance_atr = 1.3  # GOLD parameter
    new_stop = current_price - (1.3 * atr)

    if new_stop > current_stop:
        stop_loss = new_stop
        exit_action = "TRAIL"

    # Example:
    # Entry: $5000, Current: $5135, ATR: $50, Risk: $90
    # R = ($5135 - $5000) / $90 = 1.5R
    # Trail stop = $5135 - (1.3 * $50) = $5135 - $65 = $5070
    # (This locks in $70 profit minimum)
```

#### 6. TIME EXIT
```python
bars_in_trade = current_bar_index - entry_bar_index

if bars_in_trade >= 60:  # GOLD max bars parameter
    exit_price = current_bar['close']
    exit_reason = "TIME_EXIT"

    # Example: After 60 bars (15 hours), force exit regardless of P&L
```

---

## COMPLETE TRADE EXAMPLE

### Entry Bar (2025-11-05 15:30)
```
Market State:
- GOLD price: $4144.10
- ATR: $236.07
- Squeeze just released (prev: ON, current: OFF)
- TTM Momentum: +0.35 (bullish, increasing)

Setup Calculation:
- Entry: $4144.10
- Stop: $4144.10 - (1.8 * $236.07) = $4144.10 - $424.93 = $3719.17
- Risk: $424.93 per unit
- Target: $4144.10 + ($424.93 * 1.6) = $4144.10 + $679.89 = $4823.99

Position Sizing:
- Account: $10,000
- Risk: 1% = $100
- Size: $100 / $424.93 = 0.235 units
- Position value: 0.235 * $4144.10 = $973.86

Entry Executed:
✓ LONG 0.235 units @ $4144.10
✓ Stop @ $3719.17 (-10.2%)
✓ Target @ $4823.99 (+16.4%)
✓ R:R Ratio = 1.6:1
```

### Exit Bar (2025-11-05 20:45) - 5.25 hours later
```
Market State:
- GOLD price: $4521.82 (target hit!)
- Bars held: 21 bars (5h 15min)

Exit Execution:
✓ Target hit @ $4521.82
✓ P&L = ($4521.82 - $4144.10) * 0.235 = $88.76
✓ Commission = $4521.82 * 0.235 * 0.001 = $1.06
✓ Net P&L = $88.76 - $1.06 = $87.70

Trade Result:
✓ +$87.70 profit (+0.88% account)
✓ 1.6R achieved
✓ Win rate contribution: 1 win
```

---

## ACTUAL PARAMETERS PER ASSET

### GOLD (Current Optimized)
```python
{
    'initial_stop_atr': 1.8,      # $50 ATR = $90 stop distance
    'min_rrr': 1.6,               # 1.6:1 reward-to-risk
    'breakeven_r': 1.0,           # Move to BE at +1R
    'trail_start_r': 1.5,         # Start trail at +1.5R
    'trail_distance_atr': 1.3,    # Trail 1.3 ATR below current price
    'max_bars': 60,               # Max 15 hours hold time
    'retest_tolerance': 0.4,      # 0.4 ATR from KC middle for retest
    'min_momentum': 0.15,         # Minimum momentum threshold
}
```

### SILVER (Default)
```python
{
    'initial_stop_atr': 1.8,
    'min_rrr': 1.6,
    'breakeven_r': 1.0,
    'trail_start_r': 1.5,
    'trail_distance_atr': 1.3,
    'max_bars': 60,
    'retest_tolerance': 0.4,
    'min_momentum': 0.15,
}
```

### US100/US500 (Index - Not Optimized Yet)
```python
{
    'initial_stop_atr': 1.5,      # Tighter stops for indices
    'min_rrr': 1.8,               # Higher RRR target
    'breakeven_r': 1.0,
    'trail_start_r': 1.5,
    'trail_distance_atr': 1.2,
    'max_bars': 48,               # Shorter hold time
    'retest_tolerance': 0.3,
    'min_momentum': 0.2,          # Higher momentum requirement
}
```

---

## RISK MANAGEMENT

### Per Trade Risk
```
- Fixed: 1% of account per trade
- Example: $10,000 account = $100 max risk
- Position size calculated to limit loss to $100 if stop hit
```

### Max Concurrent Positions
```
- Multi-asset: Max 4 positions
- Single-asset: Max 1 position
```

### Commission & Slippage
```
- Commission: 0.1% per trade ($10 on $10,000 position)
- Slippage: 0.1% per trade (built into backtest)
- Total cost: ~0.2% per round trip
```

---

## COMPARISON: SQUEEZE_RELEASE vs SQUEEZE_RETEST

| Metric | RELEASE | RETEST |
|--------|---------|--------|
| **Frequency** | More common (48/50 trades) | Rare (2/50 trades) |
| **Win Rate** | 41.7% | 50% |
| **Entry Trigger** | Squeeze just OFF | Near KC after release |
| **Confirmation** | Momentum only | Momentum + rejection candle |
| **Risk** | Higher (immediate entry) | Lower (pullback entry) |
| **Reward** | Higher (catches full move) | Lower (partial move) |

---

## KEY DIFFERENCES FROM OTHER STRATEGIES

### vs Regime Pullback Strategy:
1. **Tighter stops**: 1.8 ATR vs 2.0 ATR
2. **Lower targets**: 1.6 RRR vs 2.5 RRR
3. **Faster management**: Breakeven at 1.0R vs 1.5R
4. **Momentum exit**: Unique to Squeeze (reversal detection)
5. **Shorter holds**: 60 bars vs 120 bars

### Rationale:
- Squeeze breakouts are SHORT-LIVED explosive moves
- Need to capture profit QUICKLY before energy dissipates
- Tighter management prevents giving back gains
- Momentum reversal signals end of expansion phase
