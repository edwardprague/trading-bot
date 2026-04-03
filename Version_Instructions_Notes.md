# Version Instructions Purpose

This page contains guidelines for backtesting parameters and instructions.

## Backtesting Parameters

| **Parameter**    | **Value**  |
| ---------------- | ---------- |
| EMA Slow         | 200        |
| EMA Fast         | 50         |
| EMA Entry        | 20         |
| Swing Lookback   | 20 bars    |
| RRR              | 1:2        |
| Risk Per Trade   | 1.0%       |
| Min Stop         | 5 pips     |
| Max Stop         | 200 pips   |
| Direction        | short_only |
| Daily Loss Limit | $2,500     |
| Starting Cash    | $100,000   |
| Ticker           | EURUSD=X   |
| Interval         | 5m         |
| Days Back        | 730        |

# V1

## Entry Instructions

**For each 5-minute bar, in order:**

**1. Check Exits (if in a trade)**

- Look at the bar's high and low for intrabar stop/target hits
- If both SL and TP are hit on the same bar, SL takes priority (conservative)
- If neither intrabar level is hit, check the close price against SL/TP
- On exit: calculate P&L, update daily loss tracker, record trade details (including MAE)

**2. Check Entries (if not in a trade)**

- Calculate signals: `trend_down` = EMA50 < EMA200, `short_sig` = previous close was above EMA20 and current close crosses below EMA20
- Apply direction filter (currently short_only, so long signals are blocked and logged)
- **Daily loss limit check**: if cumulative closed P&L for the current UTC day has reached -$2,500, skip the bar entirely
- **Short entry**: if short signal fires and swing high (20-bar lookback) exists:
    - Stop distance = swing high minus current close
    - Validate stop is between 5–200 pips
    - **Entry fill on next bar's open** (not current close)
    - Target = fill price minus (stop distance × 2)
    - Position size = (cash × 1%) / stop distance

**Notable details:**

- Short entries fill on the next bar's open; long entries (currently disabled) fill on the signal bar's close
- Only one trade at a time (no overlapping positions)
- ADX is recorded at entry for diagnostics but doesn't affect entry logic
- Blocked signals are tracked for the Filter Impact analysis
