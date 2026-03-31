## Project Overview

Building a systematic algorithmic trading system.

## Terminology

- **Ada** — refers to Cowork (the agentic coding assistant) for typing brevity. Always use "Ada" not "Cowork".

## Three-Part Workflow

1. **Claude (Chat)** — strategy decisions, result interpretation, hypothesis formation, analysis
2. **Ada (Cowork)** — writing and editing Python files, implementing changes (keep sessions short, one task per session)
3. **Edward (User)** — strategy decisions, result interpretation, hypothesis formation, analysis

When tasks are decided upon for Ada, Claude will provide instructions and Edward will copy and paste them to Ada.

- Always provide instructions for Ada in one paragraph of text, in itallics.

- Always start Ada instructions with "Please read CONTEXT.md"

---

## Technical Setup

#### Machines

- **Desktop**: Edwards-iMac — `~/Documents/GitHub/trading-bot`
- **Laptop**: Edwards-MacBook-Air — `~/Documents/GitHub/trading-bot`

#### Repository

- GitHub: `https://github.com/edwardprague/trading-bot` (public)
- GitHub Pages: `https://edwardprague.github.io/trading-bot/report.html`

#### Starting a Session

Double-click `start.command` from Finder — auto-pulls latest, starts Flask server.
Open browser: `http://localhost:8080`

#### Data Source

- **Active**: Massive.io Starter ($49/month) — C:EURUSD and C:GBPUSD, 5-minute bars, 730 days
- **Fallback**: Yahoo Finance (EURUSD=X, GBPUSD=X) hourly
- Massive API key stored in `.env` file — never commit

## Infrastructure

### Python Environment

- Python 3 (via Homebrew)
- Virtual environment (`venv`) in the trading-bot folder
- `pip` for package management

### Python Packages

- `flask` — web server framework
- `pandas` — data manipulation
- `numpy` — numerical calculations
- `yfinance` — Yahoo Finance data (fallback)
- `massive` — Massive.io API client (primary data source)
- `python-dotenv` — loads API keys from .env file
- `matplotlib` — chart generation
- `ta` / `pandas-ta` — technical indicators (ADX, EMA calculations)
- `requests` — HTTP requests

### Infrastructure Files

- `server.py` — Flask web server
- `start.command` — double-click startup script (runs git pull + Flask)
- `.env` — stores Massive API key (gitignored)
- `.gitignore` — excludes .env, venv, **pycache**

- `strategy.py` — main backtest engine + HTML report template generator
- `server.py` — Flask web server with async backtest execution
- `start.command` — double-click startup (git pull + Flask)
- `report.html` — auto-generated dashboard (never edit directly — edit strategy.py instead)
- `style.css` — dashboard styling (safe to edit, never overwritten)
- `RESULTS_LOG.md` — master results table
- `.env` — API keys (gitignored, never commit)
- `results/` — versioned PNG charts and reports
- `TrendFollowerBot.cs` — cBot script in C# for cTrader

### External Services

- GitHub — version control and auto-push after each backtest
- GitHub Pages — hosts report.html publicly
- Massive.io Starter — 5-minute forex data feed ($49/month)

### Desktop Setup Notes

- After cloning on a new machine run: `pip install flask pandas numpy yfinance massive python-dotenv matplotlib ta requests`
- Create `.env` manually with: `MASSIVE_API_KEY=yourkey`
- `.env` is gitignored and must be created on each machine manually

### Desktop Setup Checklist

After git clone on new machine:

1. cd trading-bot
2. python3 -m venv venv
3. source venv/bin/activate
4. pip install [packages]
5. Create .env with MASSIVE_API_KEY=yourkey
6. chmod +x start.command

#### Critical Note

`report.html` is regenerated from the template in `strategy.py` on every backtest run. Always fix JavaScript and HTML in `strategy.py` not `report.html` directly.

---

## cTrader Migration and Integration

The cTrader migration and integration has been completed, and the following notes are available for reference if needed:

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

---

## Dashboard Architecture

### Version Testing

New version testing denotes a change in entry conditions, which must be recorded

**Version Test** = change to entry conditions or strategy logic

---

#### Entry Conditions Section

**Entry Conditions Table Example**

| Condition        | Rule                         | +   | -   |
| ---------------- | ---------------------------- | --- | --- |
| Trend Filter     | EMA50 < EMA200               | v1  | —   |
| Entry Signal     | Price crosses below EMA20    | v1  | —   |
| Stop Placement   | Swing high over 20 bars      | v1  | —   |
| Direction        | Short only                   | v1  | —   |
| Time Window      | UTC 01 02 16 17 18           | v1  | —   |
| Daily Loss Limit | Stop if daily loss >= $2,500 | v1  | —   |

**Entry conditions modifications between versions**
Each time a new version is added the entry conditions table should be modified in the following ways:

1. New Condition
2. Rule
3. Added (+)
4. Removed (-)

- If adding a new condition, display which version it was added on in the + column.
- If removing a condition, display which version is was removed on in the - column.
- When a condition is removed it will remain in the list, with the version it was removed on displayed.

---

### Date Range Testing

New date range testing denotes the testing of date range iterations of a specific version.

**Date Range Test** = same version, different time period

- Edward (user) runs independently
- Triggered by **Add Date Range (vX)** button — label shows current version
- Uses selected from/to date pickers + selected instrument
- Added to whichever version is currently selected/viewed

---

### Sidebar Structure

The sidebar is where the versions and date ranges are navigated from.

#### Version Rows (parent) (top tier)

**Structure**

- Version #
- Instrument
- PL
- Date Range
- Duration

**Example**

- v6
- EURUSD
- +$7,412.88
- 3-26-24 → 3-26-26
- 2.0 years

#### Date Range Rows (children) (lower tier)

**Structure**

- Instrument
- PL
- Date Range
- Duration

**Example**

- EURUSD
- +$3,854.12
- 2-26-26 → 3-26-26
- 2.0 weeks

#### Duration Formatting

- over 12 months = years (1 decimal)
- 1-12 months = months (1 decimal)
- under 1 month = days

###

---

### Parameters

Strategy parameters are currently displayed in the following table example:

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

---

## Dashboard Sections (Display Order)

1. Summary + Entry Conditions (side by side)
2. Results + Parameters (side by side)
3. Performance by Direction
4. Range Filter diagnostic + Regime Classification (side by side)
5. RPF section
6. RPF chart image
7. Monthly Performance
8. Time of Day Performance
9. Daily Draw Down - Worst 5 Days
10. Main chart image
11. Streak Analysis + Stop vs Target (side by side)
12. Regime Classification
13. Win Rate Trend
14. Trade Duration + Filter Impact (side by side)
15. RRR Sensitivity + Swing Lookback Sensitivity (side by side)

---

# Trading Bot - Strategy Development

### Phase 1 - Strategy Development

**Capital Management**

- Starting capital: $100,000
- Max daily drawdown (2%): $2,000
- Max drawdown (6%): $6,000

**Strategy (Current Baseline)**

- Short only, 5-minute, EMA 20/50/200
- Time filter: UTC hours 1, 2, 16, 17, 18
- Stop: swing high over 20 bars
- Target: 2x stop distance (RRR 1:2)
- 1% risk per trade, $100k starting capital

**Entry Conditions**

Our next phase will be concerned with developing the nuances of our entry conditions, such as the following qualities to be initially explored:

**Entry Qualities (Current)**

- Pivot Points

**Entry Qualities (After Current)**

- Pullback depth
- Candle confirmation
- Volume at entry
- Stop distance quality

---

### Phase 2 - Conviction Score Architecture

Conviction Score (4-dimensional position sizing) with Rolling Profit Factor as primary risk control.

The conviction score is a functionality to be developed after strategy development.

The conviction score will aim to give a score which denotes the amount of risk on a given trade, and will be as important or more than the strategy.

**Conviction Score Calculation**

- Position Size = Base Risk × RPF × Regime × Trade Quality × Market Context
- Each dimension 0-1, multiplied. Never completely blocks — minimum 0.25% risk.

**Conviction Score Components**

These components are initial ideas to be explored at a later time.

1. Rotating Profit Factor

- Profit factor from last number of trades

2. Market Context

- Blocks of news time
- Session open
- Recent spike or strong move

3. Regime Score

- Bollinger bands
- ADX
- ATR

**Conviction Score Components Itemized**

**1. Rolling Profit Factor (RPF)**
There is already a panel in the dashboard concerning the RPF displaying the following information, but not currently being analized.

| Rolling PF (10-trade window) | Risk           |
| ---------------------------- | -------------- |
| Above 1.3                    | 1.0% full risk |
| 1.0 to 1.3                   | 0.5% reduced   |
| Below 1.0                    | 0.25% minimum  |

Items 2 - 4 have not been explored yet.
