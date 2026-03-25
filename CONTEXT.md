# Trading Bot Project — Context & Memory

## Project Overview
Building a systematic algorithmic trading system targeting an FTMO $100k funded account challenge.
The system is designed to be adaptive, with a Conviction Score as the primary risk management architecture.

## Three-Part Workflow
1. **Claude Chat (Project)** — Strategy decisions, result interpretation, hypothesis formation, analysis
2. **Cowork** — Writing and editing Python files, implementing changes (keep sessions short and focused)
3. **Terminal** — Running backtests with real data, viewing results

Always begin Cowork sessions with: *"Please read CONTEXT.md"*

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

### Running a Backtest
Click Run Backtest in browser — results auto-commit and push to GitHub.
Or run directly: `python3 strategy.py`

### Data Source
- Active: Massive.io Starter ($49/month) — C:EURUSD 5-minute bars, 2 years history
- Fallback: Yahoo Finance (EURUSD=X hourly) — free, used if Massive fails
- Massive API key stored in .env file (gitignored, never commit)

### Key Files
- strategy.py — main backtest engine
- server.py — Flask web server (localhost:8080) with async backtest execution
- start.command — double-click startup (auto-pulls + starts server)
- report.html — auto-generated dashboard (never edit directly)
- style.css — dashboard styling (safe to edit, never overwritten)
- RESULTS_LOG.md — master results comparison table
- .env — API keys (gitignored, never commit)
- results/ — versioned PNG charts and markdown reports

---

## Current Strategy

### Instrument & Timeframe
- Instrument: EURUSD (Massive API ticker: C:EURUSD)
- Timeframe: 5-minute bars
- History: 730 days (2 years)

### Parameters
- EMA Slow: 200, EMA Fast: 50, EMA Entry: 20
- Swing Lookback: 20 bars
- RRR: 1:2
- Risk Per Trade: 1.0%
- Min Stop: 5 pips, Max Stop: 200 pips
- Direction: short_only
- Time Filter: ON — UTC hours 1, 2, 16, 17, 18
- Starting Capital: $100,000

### Entry Conditions
| Condition | Rule | Purpose |
|-----------|------|---------|
| Trend Filter | EMA50 < EMA200 | Confirms downtrend — short only |
| Entry Signal | Price crosses below EMA20 | Pullback rejection in trend direction |
| Stop Placement | Swing high over 20 bars | Structural invalidation level |
| Direction | Short only | Asymmetric edge identified on EURUSD |
| Time Window | UTC 01, 02, 16, 17, 18 | Quality session hours |

### Current Baseline Results (v1 — 5-minute, 730 days)
- Total Trades: 382
- Win Rate: 34.3%
- Profit Factor: 1.03
- Net Profit: +$7,413 (+7.4%)
- Max Drawdown: -$29,243 (-25.36%)
- Max Daily DD: -$3,415 (-3.42%)
- Sharpe Ratio: 0.08

### FTMO $100k Alignment
- Max overall drawdown: 10% = $10,000 — FAIL (current 25.36%)
- Max daily drawdown: 3% = $3,000 — FAIL (current 3.42%)
- Profit target 10% = $10,000 — Achievable but needs consistency
- Main gaps: drawdown too high, daily DD breaching limit

---

## Key Strategic Insights

### Time Filter Development
Hours selected based on 5-minute performance analysis:
- Removed: 03:00 (-$8,850), 04:00 (-$7,299) — worst performers
- Kept: 01, 02, 16, 17, 18
- Next to test: Remove 17:00 (currently -$879, weakest remaining)

### Win Rate Trend (v1 baseline)
- Early (Mar-Nov 2024): 36.2%, PF 1.12
- Mid (Nov 2024-Jul 2025): 30.7%, PF 0.88 — weak period
- Late (Jul 2025-Mar 2026): 35.9%, PF 1.11
- Pattern: U-shaped — weak middle, recovering strongly

### Regime Classification
- Trending (ADX ≥25): PF 1.06
- Ranging (ADX <25): PF 1.01
- Both positive — good sign for robustness

### Daily Drawdown Issue
3 trades in one bad day can breach FTMO 3% daily limit.
Solution: MAX_TRADES_PER_DAY = 2 to be added in v2.

---

## Conviction Score Architecture (Future)

Position Size = Base Risk % × Conviction Score
Conviction Score = RPF × Regime × Quality × Context

### Four Dimensions
| Dimension | Measures | Status |
|-----------|---------|--------|
| RPF | Recent strategy health | Visualised, needs position sizing |
| Regime Score | Market conditions | Planned |
| Trade Quality | Entry quality | Planned |
| Market Context | News, session timing | Planned |

### RPF Tiers
- RPF > 1.3: 1.0% risk (full)
- RPF 1.0-1.3: 0.5% risk (reduced)
- RPF < 1.0: 0.25% risk (minimum)

Never stops completely — minimum risk keeps feedback loop alive.

---

## Dashboard Layout (per version)
1. Summary + Entry Conditions
2. Results + Parameters
3. Performance by Direction
4. Rolling Profit Factor data
5. RPF chart image
6. Monthly Performance
7. Time of Day Performance
8. Main chart (visual divider)
9. Streak Analysis + Stop vs Target
10. Regime Classification + Win Rate Trend
11. Trade Duration + Filter Impact Summary
12. RRR Sensitivity + Swing Lookback Sensitivity

---

## Hypothesis Log

### 5-minute Phase
| Version | Hypothesis | Result |
|---------|-----------|--------|
| v1 | Baseline 5min short only hours 01 02 16 17 18 | PF 1.03, +7.4%, DD 25.36% |
| v2 | Remove hour 17 + max 2 trades/day | TBD |

### Next to Test
1. Remove hour 17 + MAX_TRADES_PER_DAY=2
2. Scale EMA periods for 5-minute (200/50/20 may be too small)
3. RPF position sizing
4. Regime composite score
5. Walk-forward validation

---

## Known Bugs Fixed
- March 25 2026: P&L variable `net` overwritten by `tod_net` in time of day loop — caused Net Profit to show last hour P&L instead of overall result. Fixed via terminal sed commands.

## FTMO Parameters
- Account: $100,000
- Max overall drawdown: 10% ($10,000)
- Max daily drawdown: 3% ($3,000)
- Profit target: 10% ($10,000)
- Risk per trade: 1% = $1,000

## Multi-Instrument Plan
- EURUSD: Primary — currently best performing
- GBPUSD: Edge confirmed but late period degrading — revisit later
- Running N instruments: risk = 1% / N to control total exposure

## Next Session Priority
1. Confirm clean v1 in dashboard
2. Run v2: remove hour 17 + MAX_TRADES_PER_DAY=2
3. Analyse drawdown reduction
4. If drawdown still too high — consider EMA scaling for 5-minute
