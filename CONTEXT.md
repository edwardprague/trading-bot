# Project Overview

Building a backtesting dashboard for a systematic algorithmic trading system.

# Workflow

1. I will provide you with development tasks for you to complete.
2. Always ask questions and provide comments if needed.
3. ⁠When adding functionality, always put new styles into the style.css file and never use inline styles.
4. When you are finished with each task, let me know if it requires the following actions to view updates:

- Restart the flask server
- Run a backtest to regenerate report.html
- Or both

---

# Technical Setup

#### File Directory Path

`~/Documents/GitHub/trading-bot`

#### Virtual Environment

Terminal command - `source venv/bin/activate`

#### Repository

- GitHub: `https://github.com/edwardprague/trading-bot` (public)

#### Starting a Session

Double-click `start.command` from Finder — auto-pulls latest, starts Flask server.

#### Testing in Browser

- If it would be advantagous to connect to the dashboard in the browser, for testing updates and bug fixes, ask for permission and it will be allowed.
- Dashboard address: `http://localhost:8080`

#### Data Source

- **Active**: Massive.io Starter ($49/month) — C:EURUSD and C:GBPUSD, 5-minute bars, 730 days
- **Fallback**: Yahoo Finance (EURUSD=X, GBPUSD=X) hourly
- Massive API key stored in `.env` file — never commit

## Infrastructure

#### Python Environment

- Python 3 (via Homebrew)
- Virtual environment (`venv`) in the trading-bot folder
- `pip` for package management

#### Python Packages

- `flask` — web server framework
- `pandas` — data manipulation
- `numpy` — numerical calculations
- `yfinance` — Yahoo Finance data (fallback)
- `massive` — Massive.io API client (primary data source)
- `python-dotenv` — loads API keys from .env file
- `matplotlib` — chart generation
- `ta` / `pandas-ta` — technical indicators (ADX, EMA calculations)
- `requests` — HTTP requests

#### Infrastructure Files

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

#### Infrastructure Description

strategy.py is the backtest engine. It's the file that actually does the work — fetches price data (from Massive.io or Yahoo Finance as fallback), calculates indicators (EMAs, swing highs/lows, ADX), runs the bar-by-bar backtest simulation, generates charts, computes all the metrics (win rate, profit factor, drawdown, etc.), and then writes the results into report.html. It also handles the HTML template that defines the entire dashboard layout and JavaScript. When it runs, it's a standalone Python script executed as a subprocess — it reads configuration from environment variables (like TRADE_DIRECTION, INSTRUMENT, INTERVAL), does its thing, and exits.

server.py is the Flask web server that serves as the control layer between your browser and strategy.py. It does three things: it serves report.html to your browser at localhost:8080, it injects the run bar at the top of the page (the buttons for "Add New Version" and "Add Date Range", the date pickers, and the status indicator), and it handles the API endpoints (/run, /run_range, /status, /delete_version, /delete_run) that the browser calls when you click those buttons. When you click "Add New Version", the browser sends a POST to /run, server.py spawns strategy.py as a subprocess with the right environment variables set, polls for completion, and then reloads the page to show the new results.

In short: server.py is the middleman — it takes your UI selections (instrument, interval, direction, dates), packages them as environment variables, launches strategy.py to do the actual backtest, and serves the resulting report back to your browser. strategy.py is the engine — it crunches the data and produces the results.

#### Version Architecture

Versions have the following file structure.

| File             | Role                                                               |
| ---------------- | ------------------------------------------------------------------ |
| `strategy.py`    | Thin router — reads `VERSION` env var, delegates to versioned file |
| `strategy_v1.py` |                                                                    |
| `strategy_v2.py` |                                                                    |
| `strategy_v3.py` |                                                                    |
| `server.py`      | Flask server, passes VERSION to strategy subprocess                |

#### External Services

- GitHub — version control and auto-push after each backtest
- GitHub Pages — hosts report.html publicly
- Massive.io Starter — 5-minute forex data feed ($49/month)

### Desktop Setup Notes

- After cloning on a new machine run: `pip install flask pandas numpy yfinance massive python-dotenv matplotlib ta requests`
- Create `.env` manually with: `MASSIVE_API_KEY=yourkey`
- `.env` is gitignored and must be created on each machine manually

#### Desktop Setup Checklist

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

# cTrader Integration

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
