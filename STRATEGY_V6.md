# Strategy v6 — Fractal-Based Trend Following

## Overview

Trend following strategy on the 5-minute timeframe. Entries are triggered by fractal pivot patterns that confirm pullbacks within an established trend, filtered by EMA alignment.

---

## EMAs

Three exponential moving averages define trend bias:

- **EMA 8** (Short) — fastest, tracks immediate price action
- **EMA 20** (Mid) — intermediate, acts as the pullback zone boundary
- **EMA 40** (Long) — slowest, confirms broader trend direction

---

## Long Entries

1. **Trend Bias:** EMA 8 > EMA 20 > EMA 40 (bullish alignment)
2. **Candle Signal:** An HL (Higher Low) fractal forms, where the immediately preceding high pivot was classified as HH (Higher High). The fractal low point must fall between the EMA 8 and the EMA 20.
3. **Trade Entry:** A fractal is a 5-candle pattern (the pivot bar plus 2 bars on each side). The fractal is confirmed at the close of the 5th candle (i.e., 2 bars after the pivot bar). The trade entry occurs at the close of this confirming candle.
4. **Stop Loss:** 15 pips below the fractal low of the candle signal.
5. **Take Profit:** Determined by the R:R setting of the backtest run, calculated as stop distance × reward multiple.

## Short Entries

1. **Trend Bias:** EMA 8 < EMA 20 < EMA 40 (bearish alignment)
2. **Candle Signal:** An LH (Lower High) fractal forms, where the immediately preceding low pivot was classified as LL (Lower Low). The fractal high point must fall between the EMA 8 and the EMA 20.
3. **Trade Entry:** Same as longs — the trade entry occurs at the close of the 5th candle (the confirming bar, 2 bars after the pivot bar).
4. **Stop Loss:** 15 pips above the fractal high of the candle signal.
5. **Take Profit:** Determined by the R:R setting of the backtest run, calculated as stop distance × reward multiple.

---

## Fractal Detection

A fractal pivot high is a bar whose High is strictly greater than the Highs of the 2 bars before it and the 2 bars after it. A fractal pivot low is a bar whose Low is strictly less than the Lows of the 2 bars before it and the 2 bars after it. A single bar can be both a pivot high and a pivot low.

Fractals are detected on a rolling basis during the backtest. At bar `i`, the fractal at bar `i-2` is confirmed (because both right-side bars are now available).

---

## Fractal Classification

Each pivot is compared to the most recent previous same-type pivot (highs compared to highs, lows compared to lows — the two tracks are independent). The comparison uses a threshold of **0.5 × ATR(14)** at the current pivot's bar.

### Pivot Highs

- **HH (Higher High):** Current high − previous high > threshold (positive difference exceeds threshold)
- **LH (Lower High):** Previous high − current high > threshold (negative difference exceeds threshold)
- **CH (Consolidation High):** |current high − previous high| < threshold
- First pivot high of the dataset defaults to CH

### Pivot Lows

- **HL (Higher Low):** Current low − previous low > threshold (positive difference exceeds threshold)
- **LL (Lower Low):** Previous low − current low > threshold (negative difference exceeds threshold)
- **CL (Consolidation Low):** |current low − previous low| < threshold
- First pivot low of the dataset defaults to CL

---

## Entry Sequence Logic

The strategy requires a specific two-pivot sequence before entering:

- **Long:** The most recent high pivot must be HH, then an HL low pivot forms → long entry
- **Short:** The most recent low pivot must be LL, then an LH high pivot forms → short entry

This ensures the strategy only enters on pullbacks within a confirmed directional move, not on the first leg of a potential reversal.

---

## Filters

### Direction Filter
User-controlled setting: `both`, `long_only`, or `short_only`. Signals that don't match the selected direction are blocked and tracked for the Filter Impact Summary.

### Daily Loss Limit
$2,000 per UTC calendar day. If closed-trade losses for the current day reach this threshold, all further entries are skipped until the next UTC midnight.

### Min / Max Stop Distance
- Minimum: 5 pips (0.0005)
- Maximum: 200 pips (0.0200)
- Any signal where the stop distance falls outside this range is discarded.

### One Trade at a Time
Only one position can be open at any time. No new entries until the current trade closes via SL or TP.

---

## Position Sizing

Risk-based: 1% of current equity risked per trade. Position size = (equity × 0.01) / stop distance.

---

## Exit Logic

Intrabar SL/TP check on every bar while in a trade:

- If the bar's High/Low hits both SL and TP on the same bar, SL takes priority (conservative assumption).
- If only SL is hit, exit at SL.
- If only TP is hit, exit at TP.
- If neither intrabar level is hit, fall through to close-price check.

---

## User-Controlled Settings (via UI)

- **Instrument:** EURUSD or GBPUSD
- **Interval:** 5m (default), configurable
- **EMA Short / Mid / Long:** Default 8 / 20 / 40
- **Direction:** Both, Long only, or Short only
- **RRR:** Risk:Reward ratio (default 1:2)
