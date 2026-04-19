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

# cTrader Integration

- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot, Python Open API is an external connection not suitable for prop firm use)
