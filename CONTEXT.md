# Trading Bot Project — Master Context & Memory

## Project Overview

Building a systematic algorithmic trading system targeting an FTMO $100k funded account challenge.
Core architecture: Conviction Score (4-dimensional position sizing) with Rolling Profit Factor as primary risk control.

## Terminology

- **Ada** — refers to Cowork (the agentic coding assistant). Always use "Ada" not "Cowork" going forward.
- **Claude** — the strategy thinking and analysis chat (this chat)
- **Terminal** — for running backtests, git operations, direct code inspection

## Three-Part Workflow

1. **Claude Chat** — strategy decisions, result interpretation, hypothesis formation, analysis
2. **Ada** — writing and editing Python files, implementing changes (keep sessions short, one task per session)
3. **Terminal** — running backtests, git operations, direct code fixes

Always begin Ada sessions with: _"Please read CONTEXT.md. [single focused task]."_

---

## Project Chat Structure (Claude Project)

Organised as separate focused chats:

- **Infrastructure** — dashboard, Flask server, UI, Ada fixes
- **cTrader Migration** — translating strategy to cBot, validation testing (NEXT PRIORITY)
- **Trend Following Strategy** — EURUSD/GBPUSD development, backtesting, parameter tuning
- **Counter Trend Strategy** — separate strategy development when ready
- **Range Trading Strategy** — separate strategy development when ready
- **RPF & Conviction Score** — building the position sizing architecture

---

## Technical Setup

### Machines

- **Desktop**: Edwards-iMac — `~/Documents/GitHub/trading-bot`
- **Laptop**: Edwards-MacBook-Air — `~/Documents/GitHub/trading-bot`

### Repository

- GitHub: `https://github.com/edwardprague/trading-bot` (public)
- GitHub Pages: `https://edwardprague.github.io/trading-bot/report.html`

### Starting a Session

Double-click `start.command` from Finder — auto-pulls latest, starts Flask server.
Open browser: `http://localhost:8080`

### Data Source

- **Active**: Massive.io Starter ($49/month) — C:EURUSD and C:GBPUSD, 5-minute bars, 730 days
- **Fallback**: Yahoo Finance (EURUSD=X, GBPUSD=X) hourly
- Massive API key stored in `.env` file — never commit
- Instrument mapping: EURUSD → C:EURUSD, GBPUSD → C:GBPUSD
- **CRITICAL FINDING**: Massive data has significant gaps — multiple months missing entirely (confirmed Dec 2025 = zero bars). Python backtest trade counts are systematically understated as a result. Python dashboard is reliable for relative version comparisons but not absolute trade frequency.
- **PLANNED REPLACEMENT**: cTrader Open API (OpenApiPy) — pulls IC Markets 5-minute bars directly, identical data source to live execution. Pending Spotware application approval (up to 3 business days from 2026-03-27).

### Key Files

- `strategy.py` — main backtest engine + HTML report template generator
- `server.py` — Flask web server with async backtest execution
- `start.command` — double-click startup (git pull + Flask)
- `report.html` — auto-generated dashboard (never edit directly — edit strategy.py instead)
- `style.css` — dashboard styling (safe to edit, never overwritten)
- `RESULTS_LOG.md` — master results table
- `.env` — API keys (gitignored, never commit)
- `results/` — versioned PNG charts and reports

### Critical Note

`report.html` is regenerated from the template in `strategy.py` on every backtest run. Always fix JavaScript and HTML in `strategy.py` not `report.html` directly.

---

## Dashboard Architecture

### Version Control vs Date Range Testing

Two fundamentally different operations:

**Version** = change to entry conditions or strategy logic

- Collaborative decision between user and Claude
- Triggered by **Add New Version** button
- Defaults to full 730-day run
- Increments from highest existing version number (never reuses deleted version numbers)

**Date Range Test** = same version, different time period

- User runs independently
- Triggered by **Add Date Range (vX)** button — label shows current version
- Uses selected start/end date pickers + selected instrument
- Added to whichever version is currently selected/viewed
- Does NOT increment version number

### Button Labels

- **Add New Version** — left side, green
- **Add Date Range (vX)** — right side, blue, X = current version number
- Both have instrument selector dropdowns immediately to their left
- Instruments available: EURUSD, GBPUSD

### Date Pickers

- From / To date pickers persist between runs until modified
- Located between instrument selector and Add Date Range button

### Sidebar Structure

```
v6
+$7,412.88
3-26-24 → 3-26-26
2.0 years

  +$3,854.12          ← date range iteration
  2-26-26 → 3-26-26
  1 month

v5 ►
v4 ►
```

Sidebar display rules:

- Version row shows: version number, P&L, date range (M-DD-YY format), duration
- Date range rows show: P&L, date range, duration
- No created-at timestamps, no run count, no Backtest History title
- Collapse/expand arrow on RIGHT side of version row — independent from row click
- Clicking version row loads full run report
- Clicking arrow toggles date range iterations

### Date/Duration Formatting

- Date format: M-DD-YY, single digit month, no leading zero (e.g. 3-26-26)
- Duration logic: under 2 months = weeks, 2-18 months = months (1 decimal), over 18 months = years (1 decimal)
- Duration calculated from actual start/end dates, not DAYS_BACK

### Copy Button (single context-aware button)

- Shows "Copy Version Report" on full version view → copies all iterations
- Shows "Copy Range Report" on date range view → copies that iteration only

### Delete Button (single context-aware button)

- Shows "Delete Version" on full version view → deletes version + all iterations
- Shows "Delete Date Range" on date range view → deletes only that iteration

### Version Header Area

- Row 1: Version number + Strategy badge (no blue date range badge)
- Row 2: Run date/time · Instrument (no =X suffix) · Date range · Duration
- No interval display

### Parameters Section

- Instrument displays without =X suffix (EURUSD not EURUSD=X)
- Min Stop, Max Stop included
- Avg Stop (pips) and Avg Target (pips) included

### Entry Conditions Table

| Condition | Rule | + | - |
Columns: Condition, Rule, + (version added), - (version removed)
No Purpose column.

---

## Current Strategy (v6 — Clean Baseline)

### Instrument & Timeframe

- Primary: EURUSD, 5-minute bars, 730 days
- Secondary: GBPUSD (available, tested)
- Starting capital: $100,000

### Parameters

| Parameter      | Value                           |
| -------------- | ------------------------------- |
| EMA Slow       | 200                             |
| EMA Fast       | 50                              |
| EMA Entry      | 20                              |
| Swing Lookback | 20 bars                         |
| RRR            | 1:2                             |
| Risk Per Trade | 1.0%                            |
| Min Stop       | 5 pips                          |
| Max Stop       | 200 pips                        |
| Direction      | short_only                      |
| Time Filter    | ON — UTC hours 1, 2, 16, 17, 18 |

### Entry Conditions

| Condition        | Rule                         | +   | -   |
| ---------------- | ---------------------------- | --- | --- |
| Trend Filter     | EMA50 < EMA200               | v1  | —   |
| Entry Signal     | Price crosses below EMA20    | v1  | —   |
| Stop Placement   | Swing high over 20 bars      | v1  | —   |
| Direction        | Short only                   | v1  | —   |
| Time Window      | UTC 01 02 16 17 18           | v1  | —   |
| Daily Loss Limit | Stop if daily loss >= $2,500 | v1  | —   |
| Regime Filter    | ATR range detection          | v2  | v5  |

### Key Results

| Version       | Period       | Instrument | Trades | Win Rate | PF   | Net P&L  |
| ------------- | ------------ | ---------- | ------ | -------- | ---- | -------- |
| v1 baseline   | 730 days     | EURUSD     | 382    | 34.3%    | 1.03 | +$7,413  |
| v1 date range | Jan-Mar 2026 | EURUSD     | 33     | 45.5%    | 1.64 | +$12,315 |
| v1 date range | Jan-Mar 2026 | GBPUSD     | 38     | 31.6%    | 0.91 | -$2,340  |
| v6 cTrader    | Jan-Mar 2026 | EURUSD     | 51     | 39.2%    | 1.28 | +$9,309  |

### FTMO $100k Alignment

- Max overall drawdown limit 10% — FAIL (current ~25%)
- Max daily drawdown limit 3% — borderline
- Main gaps: drawdown too high, entry quality needs improvement

---

## Critical Finding — Stop Distance

- Avg Stop: 10.9 pips on 5-minute EURUSD
- Avg Target: 21.8 pips
- 10.9 pip stops are too tight — normal 5-minute noise clips stops before genuine invalidation
- Swing high over 20 bars on 5-minute produces structurally weak levels
- **This is the primary area for strategy improvement**

---

## Key Strategic Insights

### Direction Asymmetry

Short trades significantly outperform long trades on EURUSD. Running short_only.

### Time of Day (UTC) — 5-minute

Best: 02:00, 16:00, 17:00, 18:00
Decent: 01:00
Removed: 03:00, 04:00

### Instrument Comparison (Jan-Mar 2026)

- EURUSD: PF 1.64, 45.5% WR — strong
- GBPUSD: PF 0.91, 31.6% WR — weak in this period

---

## Regime Filter — Research Notes (Removed in v5)

### What Was Built

LuxAlgo Range Detector inspired filter. Two conditions required for "ranging":

1. All closes over REGIME_LENGTH bars within ATR of SMA
2. Net displacement: abs(close_now - close_N_bars_ago) < ATR \* threshold

Parameters tested: REGIME_ATR_LENGTH=96, REGIME_LENGTH=20, REGIME_DISPLACEMENT_THRESHOLD=0.3

### Key Learnings

- Initial ATR calculation was wrong (close-to-close) — fixed to true ATR (high-low-close)
- Initial filter was inverted — blocking trends, allowing ranges
- Displacement condition fixed the inversion
- RS Diagnostic section added — shows filtered vs allowed trade stats with simulated outcomes
- On 730-day run: 13 trades filtered, results worse than baseline
- Binary filter too crude for this stage — removes trades entirely rather than scaling size

### Future Consideration

- Better as continuous Conviction Score dimension than binary filter
- Needs better entry/stop foundation first
- Consider position size reducer rather than hard veto

---

## Conviction Score Architecture (Future)

```
Position Size = Base Risk × RPF × Regime × Trade Quality × Market Context
```

Each dimension 0-1, multiplied. Never completely blocks — minimum 0.25% risk.

### RPF Tiers

| Rolling PF (10-trade window) | Risk           |
| ---------------------------- | -------------- |
| Above 1.3                    | 1.0% full risk |
| 1.0 to 1.3                   | 0.5% reduced   |
| Below 1.0                    | 0.25% minimum  |

### Two-Speed RPF

- Fast RPF (5-trade): detects sharp drops
- Slow RPF (20-trade): detects gradual erosion

---

## Dashboard Sections (Display Order)

1. Summary + Entry Conditions (side by side)
2. Results + Parameters (side by side)
3. Performance by Direction
4. Range Filter diagnostic + Regime Classification (side by side)
5. RPF section + RPF chart
6. Monthly Performance
7. Time of Day Performance
8. Main chart image (visual divider)
9. Streak Analysis + Stop vs Target (side by side)
10. Regime Classification + Win Rate Trend (side by side)
11. Trade Duration + Filter Impact (side by side)
12. RRR Sensitivity + Swing Lookback Sensitivity (side by side)

---

## cTrader Migration — Status: cBot Complete, Data Integration Pending

### What Was Built

- **TrendFollowerBot.cs** — C# cBot for cTrader Automate, faithfully translates v6 Python strategy
- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot, Python Open API is an external connection not suitable for prop firm use)
- **File location**: repository root

### cBot Architecture

- `OnStart()`: initialises EMA200, EMA50, EMA20 indicators, wires `Positions.Closed` event for daily P&L tracking, parses time filter hours string
- `OnBar()`: time filter → daily loss limit check → trend filter (EMA50 < EMA200) → EMA20 crossover → swing high calculation (`Last(2)` to `Last(SwingLookback+1)`) → stop distance validation → position guard (one trade at a time) → one-bar cooldown → `ExecuteMarketOrder`
- `OnPositionClosed()`: accumulates NetProfit into daily P&L tracker, resets at UTC midnight, label guard prevents manually opened positions corrupting the accumulator
- All parameters exposed as cTrader `Parameter` attributes — configurable without recompiling

### Key Implementation Decisions

- C# chosen over Python Open API — C# cBot runs natively inside cTrader, correct for prop firm execution
- `NetProfit` used (not `GrossProfit`) — includes spread and commission, matches Python pnl calculation
- `Symbol.NormalizeVolumeInUnits()` wraps raw position size — ensures valid lot steps for IC Markets
- `ExecuteMarketOrder` takes SL/TP in pips not price levels — `stopPips` and `stopPips * RRR`
- One-position guard: `Positions.Find(Symbol.Name)` returns immediately if trade already open
- One-bar cooldown: `_lastCloseBar` tracks bar count at close, new entries blocked for one bar after exit
- `TimeZone = TimeZones.UTC` on Robot attribute — `Server.Time.Hour` is UTC throughout
- Time filter hours passed as comma-separated string parameter — cTrader does not support array parameters

### Validation Results (Jan 2026 — Mar 2026, IC Markets demo)

| Metric         | Python (Massive) | cTrader (IC Markets) |
| -------------- | ---------------- | -------------------- |
| Total Trades   | 7                | 51                   |
| Win Rate       | 71.4%            | 39.2%                |
| Profit Factor  | 4.85             | 1.28                 |
| Net Profit     | +$8,211          | +$9,309              |
| Max Drawdown   | 1.0%             | 8.76%                |

Trade count discrepancy (7 vs 51) is primarily explained by Massive data gaps — not a logic error in the cBot. The cTrader result of 51 trades is the more reliable picture of actual strategy behaviour in this period.

### cTrader Open API — Data Integration Plan

Spotware maintains an official Python SDK (OpenApiPy) for the cTrader Open API. Plan is to replace the Massive API fetch in server.py with an OpenApiPy call pulling IC Markets 5-minute bars directly. This gives Python dashboard identical data to live execution, eliminating the data gap problem entirely.

**Setup requirements:**

- cTrader Open API application registered at openapi.ctrader.com — Client ID and Secret obtained 2026-03-27
- Application status: PENDING APPROVAL (up to 3 business days)
- Once approved: generate OAuth access token via playground, note ctidTraderAccountId for demo account
- Ada task: replace Massive fetch in server.py with OpenApiPy historical bars call
- Rate limit: 5 requests per second for historical data
- Bar data returned in relative price format — divide low by 100000, add deltas for OHLC

### Export Button (Future — after data integration)

Generates C# cBot file from current version's entry conditions. Entry Conditions table maps directly to cAlgo code structure.

---

## Next Development Priorities

### Immediate

1. cTrader Open API integration — replace Massive data source in server.py with OpenApiPy once application approved (Infrastructure task, instructions ready)
2. Run full available date range cTrader backtest on IC Markets to build richer comparison baseline
3. Month-by-month analysis on EURUSD to map which months work

### Entry Quality (Major Focus After cTrader)

- Avg stop of 10.9 pips is too tight for 5-minute bars
- Use higher timeframe swing highs for stop placement
- Minimum stop distance filter — reject entries with stops under 15 pips
- Pullback depth requirement
- Candle quality filter

### Take Profit Improvement

- Currently arbitrary 2x stop distance
- Better: next structural support level, ATR-based target, partial exits

### Multi-Instrument

- EURUSD primary
- GBPUSD secondary — same strategy, test when entry improvements done
- Risk scaling: 1% / N instruments simultaneously

---

## Known Bugs Fixed

- tod_net variable naming — net profit overwritten by time-of-day loop
- True ATR calculation — fixed to high-low-close based
- Delete button — correctly deletes version or date range based on context
- Copy button — context aware, single button
- Version numbering — always increments from highest, never reuses deleted numbers
- Date range runs — added to selected version not always latest
- Instrument selection — date range uses its own instrument selector independently
