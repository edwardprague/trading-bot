# Trading Bot Project Context

## Project Goal
Build a systematic trend following trading bot for indices and forex.
Develop, test and optimise in Python, then deploy as a cTrader Python cBot.

## Development Setup
- Python scripts live in this repository
- Test locally using Yahoo Finance data (free)
- Future: Massive.io Starter plan for 5-minute intraday data
- Deploy to cTrader using native Python cBot support

## Three-Part Workflow
1. **Claude Chat** — strategy decisions, result interpretation, next steps
2. **Cowork** — writing and editing Python files
3. **Browser** — run backtests and review results at http://localhost:8080

## Current Strategy
- Instrument: EURUSD hourly (Yahoo Finance ticker: EURUSD=X)
- Trend filter: EMA 50 vs EMA 200 (long above, short below)
- Entry signal: price crosses EMA 20 in trend direction
- Stop: structural swing high/low over 20 bar lookback
- Target: 1:2 Risk:Reward ratio
- Risk per trade: 1% of account
- Starting capital: $10,000
- Strategy tag: `STRATEGY = "Trend Following"` (used to group versions in the dashboard)

## Baseline Results (EURUSD hourly, 720 days)
- Not yet run on this setup — to be established

## Results Log
| Version | Change | Trades | Win Rate | Profit Factor | Max DD | Net P&L |
|---------|--------|--------|----------|---------------|--------|---------|
| v1 baseline | Initial strategy | TBD | TBD | TBD | TBD | TBD |

## Hypotheses Queue
- Add ADX filter to avoid ranging markets
- Test 15-minute timeframe for more trades
- Add time-of-day filter to avoid low liquidity periods

## Data Sources
- Yahoo Finance: EURUSD=X hourly, 720 days (free, working)
- Yahoo Finance: ^GDAXI hourly, 720 days (free, working)
- Massive.io Starter ($29/mo): 5-min intraday, 5 years history (pending)

## Files
- `strategy.py` — main backtest script; run directly or via the dashboard
- `server.py` — Flask dashboard server; serves report.html with a Run Backtest button
- `report.html` — auto-generated backtest report with version history (do not edit by hand)
- `RESULTS_LOG.md` — plain markdown table appended after every backtest run; columns: Version, Date, Strategy, Instrument, Timeframe, Notes, Trades, Win Rate, Profit Factor, Net P&L, Max Drawdown, Sharpe Ratio
- `results/` — versioned chart images, e.g. `results/v3_EURUSD_2026-03-22.png`
- `CONTEXT.md` — this file, project memory

## How to Start a Session
1. Read this file
2. Activate virtual environment: `source venv/bin/activate`
3. Start the dashboard: `python3 server.py`
4. Open http://localhost:8080 in your browser
5. Click **Run Backtest** to run strategy.py — the page refreshes automatically when done
6. Or run strategy directly in the terminal: `python3 strategy.py`

## Dashboard Notes
- `server.py` uses Flask; it installs Flask automatically via pip on first run if missing
- The Run Backtest button POSTs to `/run`, which runs `strategy.py` as a subprocess
- Backtests can take 1–2 minutes (data fetch + compute); a spinner shows while running
- Each run appends a new version to `report.html`; previous versions are never lost
- The server uses `sys.executable` so it always runs strategy.py with the same Python/venv

## Dashboard UI
- Strategy dropdown at top of sidebar: Trend Following / Counter Trend / Range Trading
- Version list filters to show only runs for the selected strategy
- Each version entry shows version name, date, and net P&L coloured green/red
- Development Log button at sidebar bottom: shows timeline table of all versions for the
  selected strategy with version, date, change notes, profit factor (with ▲/▼ arrows vs
  previous version), win rate, and net P&L
- `STRATEGY` variable in `strategy.py` tags each run; all 18 existing versions tagged as
  "Trend Following"
