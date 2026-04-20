# Project Overview

We are working on building a backtesting dashboard to develop a cBot for cTrader.

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

## Infrastructure

#### Python Environment

- Python 3 (via Homebrew)
- Virtual environment (`venv`) in the trading-bot folder
- `pip` for package management

#### Python Packages

- `flask` — web server framework
- `pandas` — data manipulation
- `numpy` — numerical calculations
- `massive` — Massive.io API client (data source)
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

#### Version Architecture

Versions have the following file structure.

| File             | Role                                                               |
| ---------------- | ------------------------------------------------------------------ |
| `strategy.py`    | Thin router — reads `VERSION` env var, delegates to versioned file |
| `strategy_v1.py` |                                                                    |
| `strategy_v2.py` |                                                                    |
| `strategy_v3.py` |                                                                    |
| `server.py`      | Flask server, passes VERSION to strategy subprocess                |

---

# Strategy Logic

Both the Python backtest and the cTrader cBot follow the same execution flow. The cBot is generated from `cbot_templates.py` via the dashboard's **Create cBot** button.

#### v1 — Fractal Geometry Entries

1. **Fractal detection** — Williams N=2. A fractal is detected at bar `fi` only when bars `fi±1` and `fi±2` all have lower highs (for a fractal high) or higher lows (for a fractal low). Confirmation happens 2 bars later.
2. **Entry signal**
   - **Long** — new fractal low forms that is higher than the prior fractal low (higher-low).
   - **Short** — new fractal high forms that is lower than the prior fractal high (lower-high).
3. **Direction filter** — configurable: `both`, `long_only`, or `short_only`.
4. **Daily loss limit** — no new trades once the day's losing-trade count reaches `MaxDailyLosses` (default 3). Resets at UTC midnight.
5. **Time filter** — entries only fire during allowed UTC hours (dashboard uses a blocked-hours list; the cBot template converts it to allowed-hours).
6. **Stop validation** — stop-distance in pips must be between `MinStopPips` and `MaxStopPips`; signal is skipped if outside this range.
7. **Entry reference**
   - **Python backtest** — `next_open` (open of the bar after confirmation).
   - **cBot** — `Symbol.Ask` for longs, `Symbol.Bid` for shorts, read on the first tick of the bar after confirmation.
8. **Stop-loss** — fractal price ± `FractalStopPips` offset (default 7).
9. **Take-profit** — entry ± (stop distance × RRR), where RRR = `RrrReward / RrrRisk`.
10. **Position sizing** — `(equity × RiskPercent) / stop-distance-in-price`, normalised to the broker's volume step.
11. **Re-entry** — no cooldown. A new signal on the very next bar after a close is accepted (matches the Python backtest).

#### v2 — v1 + EMA Position Filter

All of the above, plus an additional filter evaluated on the confirmation bar's close:

- **Long** requires `close > EMA Long`.
- **Short** requires `close < EMA Long`.

The EMA comparison uses the confirmation bar's close (not the entry reference) to match the Python backtest.

---

# cTrader Integration

- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot, Python Open API is an external connection not suitable for prop firm use)
- **Code generation**: `cbot_templates.py` renders `FractalBot_v{N}.cs` on demand from the dashboard's **Create cBot** button. Changes to the bot's logic must be made in `cbot_templates.py` (not in the `.cs` output file, which is overwritten on each generation).
