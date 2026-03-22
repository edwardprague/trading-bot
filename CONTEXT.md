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
3. **Terminal** — running scripts with real data

## Current Strategy
- Instrument: EURUSD hourly (Yahoo Finance ticker: EURUSD=X)
- Trend filter: EMA 50 vs EMA 200 (long above, short below)
- Entry signal: price crosses EMA 20 in trend direction
- Stop: structural swing high/low over 20 bar lookback
- Target: 1:2 Risk:Reward ratio
- Risk per trade: 1% of account
- Starting capital: $10,000

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
- `strategy.py` — main backtest script
- `CONTEXT.md` — this file, project memory

## How to Start a Session
1. Read this file
2. Activate virtual environment: `source venv/bin/activate`
3. Run current strategy: `python3 strategy.py`
4. Review results and continue from Results Log above
