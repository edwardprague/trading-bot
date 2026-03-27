# Trading Bot — Project Strategy

## The Goal

Build a systematic algorithmic trading system and pass an FTMO $100,000 funded account challenge.

FTMO Rules:

- Max overall drawdown: 10% ($10,000)
- Max daily drawdown: 3% ($3,000)
- Profit target: 10% ($10,000) within challenge period

## The Philosophy

### RPF as Primary Architecture

The Rolling Profit Factor is not just a diagnostic — it is the primary risk management layer. The strategy framework sits inside the RPF system, not the other way around. A mediocre strategy with excellent risk management will outperform an excellent strategy with poor risk management every time.

The system never stops trading completely. Minimum risk keeps the feedback loop alive so the RPF can detect when conditions improve and scale back up.

### Conviction Score

Four-dimensional position sizing that scales risk between 0.25% and 1.0% per trade:

```
Position Size = Base Risk × RPF × Regime × Trade Quality × Market Context
```

Each dimension outputs 0-1. Multiplication means one weak dimension reduces the whole score — you cannot offset a terrible regime with a great entry.

### Entry Conditions vs Conviction Score

These are architecturally distinct:

- **Entry Conditions** — binary hard vetoes. Trade or don't trade.
- **Conviction Score** — continuous position sizing. How much to risk on trades that pass.

### Past Performance Philosophy

Two years of backtesting is sufficient for development. Older data introduces different market regimes that may be irrelevant and muddy the analysis. The Win Rate Trend diagnostic (3 equal periods) shows whether the strategy is improving or degrading over time — focus on the late period as the most relevant signal.

---

## Current State

### Infrastructure

Fully operational backtesting dashboard with:

- 5-minute EURUSD and GBPUSD data via Massive.io API
- Version control system (Add New Version)
- Date range testing system (Add Date Range) — same version, different periods
- Instrument selection per run
- Comprehensive diagnostics: RPF chart, RS diagnostic, monthly performance, time of day, regime classification, win rate trend, drawdown analysis

### Strategy (v6 — Current Baseline)

- EURUSD short only, 5-minute, EMA 20/50/200
- Time filter: UTC hours 1, 2, 16, 17, 18
- Stop: swing high over 20 bars
- Target: 2x stop distance (RRR 1:2)
- 1% risk per trade, $100k starting capital

### Key Results

- Full 730-day run: 382 trades, 34.3% WR, PF 1.03, +$7,413
- Jan-Mar 2026 EURUSD: 33 trades, 45.5% WR, PF 1.64, +$12,315
- Jan-Mar 2026 GBPUSD: 38 trades, 31.6% WR, PF 0.91, -$2,340

### Critical Finding

Average stop distance is only 10.9 pips on 5-minute EURUSD. This is the most important problem to solve — normal market noise clips stops before genuine trend invalidation. The swing high over 20 bars on a 5-minute chart produces structurally weak stop levels.

---

## What Has Been Tried

### Regime Filter (Removed in v5)

Built a LuxAlgo-inspired range detection filter. Findings:

- Binary filter too crude — removes trades entirely rather than scaling size
- Blocked trades were often the best trades (simulated outcomes showed 85%+ win rate on blocked signals)
- Needs to be a continuous Conviction Score dimension, not a binary veto
- The concept is valid but the foundation (entry/stop quality) needs fixing first
- Research notes preserved in CONTEXT.md for future reference

### Time Filter Refinement

Removing hours 03:00 and 04:00 UTC produced a $16,472 improvement in net profit. Those hours consistently underperformed across all tests. The current filter (01, 02, 16, 17, 18) is the established baseline.

### Direction Asymmetry

Short trades significantly outperform long trades on EURUSD over the test period. Running short_only. Long trades may be revisited with more data and better entry conditions.

### GBPUSD

Tested same strategy on GBPUSD. Edge confirmed but performance varies significantly by period. GBPUSD and EURUSD tend to be in opposite performance regimes — natural diversification opportunity when running multi-instrument.

---

## Chat Organisation

Each chat in the Project has a single focused domain. Start every chat with "Please read CONTEXT.md" if using Cowork, or paste the relevant section of PROJECT_STRATEGY.md for Claude context.

### Infrastructure Chat

Dashboard, Flask server, UI improvements, Cowork fixes. The system is largely complete — use this chat for maintenance and new dashboard features.

### cTrader Migration Chat (IMMEDIATE NEXT PRIORITY)

Translate current strategy to Python cBot in cTrader. Validate by running the same date range (Jan-Mar 2026) in both systems and comparing results. Target within 10-15% agreement. If aligned, build an export button in the dashboard that generates cBot code from the current version's entry conditions.

### Trend Following Strategy Chat

EURUSD and GBPUSD development. Primary focus areas:

1. Stop placement improvement — use higher timeframe swing highs, minimum stop distance
2. Entry quality — pullback depth, candle quality, EMA slope
3. Take profit improvement — structural levels rather than arbitrary 2x
4. Month-by-month analysis to identify which market conditions suit the strategy
5. RPF position sizing implementation once trade count is sufficient

### Counter Trend Strategy Chat

Separate development when trend following is stable. Different entry logic, different time filter, different instruments potentially.

### Range Trading Strategy Chat

Separate development. Different approach entirely — the regime filter research will be relevant here.

### RPF & Conviction Score Chat

Building the full position sizing architecture once enough trade history exists to calibrate it meaningfully. Requires 500+ trades for the RPF signal to be statistically robust.

---

## Hypothesis Queue

### Immediate

1. cTrader validation — does our Python backtest translate to live results?
2. Month-by-month EURUSD analysis — which months/conditions work?

### Entry Quality (Most Important)

- Minimum stop distance filter — reject entries with stops under 15 pips
- Higher timeframe stop placement — use 1h or 4h swing high instead of 5m
- Pullback depth requirement — price must retrace minimum distance before entry
- Candle quality — entry candle must close with conviction

### Take Profit

- Structural levels — next swing low below entry
- ATR-based target — 2x ATR from entry
- Partial exits — close 50% at 1x, remainder at 2x

### Multi-Instrument

- Run EURUSD and GBPUSD simultaneously when conditions warrant
- Risk scaling: 1% / N instruments running simultaneously
- Per-instrument RPF control — each instrument has its own RPF tier

### RPF System

- Fast RPF (5-trade window) — detects sharp performance drops
- Slow RPF (20-trade window) — detects gradual erosion
- Both dropping = serious regime shift, reduce to 0.25%
- Slow declining 3+ periods = flag for human review

---

## Key Principles Going Forward

**One change at a time.** Every version tests exactly one hypothesis. Multiple simultaneous changes make it impossible to know what caused an improvement or degradation.

**Trust the data over intuition.** When results are unexpected — investigate why rather than reverting immediately. The answer is usually informative.

**cTrader validation first.** All backtesting in Python is simulation. Until we confirm the strategy produces similar results in cTrader we don't know if the edge is real or an artifact of our simulation assumptions.

**The mid-period problem.** Nov 2024 to Jul 2025 is consistently weak across all versions. Understanding why is more valuable than optimising for the good periods.

**Sample size matters.** With 382 trades over 730 days, the RPF system cannot function at full resolution. Getting to 500+ trades through entry refinement or timeframe adjustment is a meaningful milestone.

---

## Workflow Reminders

- **Cowork** = Cowork (the agentic coding assistant)
- Keep Cowork sessions short — one task per session
- Always start Cowork with "Please read CONTEXT.md"
- After meaningful progress — update both CONTEXT.md and PROJECT_STRATEGY.md
- This chat = strategy thinking. Cowork = implementation. Terminal = execution.
- Update CONTEXT.md after every significant code change
- Update PROJECT_STRATEGY.md after every significant strategic decision
