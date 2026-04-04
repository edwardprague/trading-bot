# Version Backtest Instructions

This page contains guidelines for backtesting parameters and instructions.

## Backtesting Parameters

The backtesting parameters are all modifiable for each backtest.

| **Parameter** |
| ------------- |
| EMA Slow      |
| EMA Fast      |
| EMA Entry     |
| RRR           |
| Direction     |
| Instrument    |
| Interval      |

---

## Entry Instructions

**For each 5-minute bar, in order:**

**1. Check For Exits (if in a trade)**

- Look at the bar's high and low for intrabar stop/target hits
- If both SL and TP are hit on the same bar, SL takes priority (conservative)
- If neither intrabar level is hit, check the close price against SL/TP
- On exit: calculate P&L, update daily loss tracker, record trade details (including MAE)

**2. Check For Entry Conditions (if not in a trade)**

- **Daily loss limit check**: if cumulative closed P&L for the current UTC day has reached -$2,500, skip all bars entirely until the next trading day.

- Only one trade at a time (no overlapping positions).
- Position size = (cash × 1%) / stop distance

---

## Entry Conditions

### Long Entries

**1.Short Trend Bias**
EMA 8 > 20 > 40

**2. Candle Signal**
When an HL fractal forms, after an HH, with the fractal low point falling between the 8 and the 20 EMA.

**3. Trade Entry**
The fractal signal is a 5 candle pattern, the trade entry should occur at the close of the 5th candle.

**Stop Loss Level**
The stop loss level should be 15 points below the fractal low of the candle signal.

**Take Profit Level**
The take profit level should be determined by the R:R settings of the backtest run, which is based on the stop loss level.

### Short Entries

**1.Short Trend Bias**
EMA 8 < 20 < 40

**2. Candle Signal**
When an LH fractal forms, after an LL, with the fractal high point falling between the 8 EMA and the 20 EMA.

**3. Trade Entry**
The fractal signal is a 5 candle pattern, the trade entry should occur at the close of the 5th candle.

**Stop Loss Level**
The stop loss level should be 15 points above the fractal high of the candle signal.

**Take Profit Level**
The take profit level should be determined by the R:R settings of the backtest run, which is based on the stop loss level.
