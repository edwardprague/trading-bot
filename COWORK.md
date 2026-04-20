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

# Most Recent Tasks

## 2026-04-20 — Reconciling Dashboard vs cTrader Jan-26 Discrepancy

### Background

The dashboard backtest for January 2026 was showing **-$8,916** while cTrader live execution showed **-$461** — a gap of roughly **$8,500** that needed to be reconciled. December was similarly off; February and March matched closely. The investigation focused on identifying and closing the gap.

### Root cause #1: SL slippage model (FIXED)

The dashboard was using a worst-case **"fill at bar extreme"** slippage model — when a stop-loss was hit, the exit price was set to the bar's high (for longs) or low (for shorts), producing artificially severe losses (some up to -$2,392) that didn't reflect real broker execution.

**Implementation (Option A):** Replaced bar-extreme slippage with a **pip-based slippage model**. SL exits now fill at:

- Long: `sl − SL_SLIPPAGE_PRICE`
- Short: `sl + SL_SLIPPAGE_PRICE`

Where `SL_SLIPPAGE_PIPS` defaults to **1.0 pip**, configurable via the Backtest Settings panel.

**Files changed:**

- `strategy_v1.py` — added `SL_SLIPPAGE_PIPS` / `SL_SLIPPAGE_PRICE` constants; updated both SL fill sites (df2 streaming loop ~line 317, main df loop ~line 527); added "SL Slippage" row to report metadata table; added `bs-sl-slippage-pips` input + persistence + event listener in both populated and empty-state report renderings; added `sl_slippage_pips` to report metadata dicts.
- `strategy_v2.py` — mirror of all v1 changes above.
- `server.py` — added `getSelectedSlSlippagePips()` getter (default `"1.0"`); added `sl_slippage_pips` to all 3 fetch bodies (`new_version`, `date_range`, `date_range_batch`); added backend wiring in all 3 routes (extract variable, set `env_overrides["SL_SLIPPAGE_PIPS"]`); batch route also adds to `shared_params`.
- `style.css` — no changes required (new input reuses existing classes).

All three Python files passed `python -m py_compile` cleanly.

### Test results: pip-based slippage across all four months

| Month  | Trades | P&L         | Worst Trade | Avg Loss  |
| ------ | ------ | ----------- | ----------- | --------- |
| Dec-25 | 68     | -$2,128.97  | -$1,295.13  | $1,167.78 |
| Jan-26 | 75     | -$3,743.87  | -$1,282.22  | $1,114.59 |
| Feb-26 | 77     | +$6,534.82  | -$1,326.92  | $1,208.90 |
| Mar-26 | 97     | +$21,234.48 | -$1,454.04  | $1,258.58 |

**Impact on Jan-26 specifically:** Gap closed from **~$8,500 → ~$3,300** (dashboard now -$3,744 vs cTrader -$461). Loss distribution is now tightly clustered ($985–$1,454 across all months), matching what a real broker's distribution looks like. The old model's tail (losses up to -$2,392) is gone.

### Root cause #2: Direction mismatch (UNDER INVESTIGATION)

After the slippage fix, residual analysis of the cTrader Jan CSV (uploaded by Edward) revealed:

```
cTrader Jan-26 totals:
  Total trades:     83
  Sells:            83  (100%)
  Buys:              0  (0%)
  Win rate:         50.6%   (42 wins, 41 losses)
  Avg win:        $1,047
  Avg loss:       $1,069
  Worst loss:    -$2,007   (single news event 13/01 13:15, 22-pip gap)
  Gross P&L:       +$114.62
  Net (after fees): -$461
```

**Key findings:**

- cTrader took **only sells** in January; the dashboard takes both directions by default (TRADE_DIRECTION=both). Edward confirmed both should be **shorts only**.
- cTrader's gross P&L is actually **+$114** (essentially flat); the -$461 includes ~$575 of commissions/swaps not currently modeled in the dashboard (~$7/trade across 83 trades).
- cTrader's worst loss (-$2,007) came from a single news-driven gap event — a real outlier, not systemic.

### Open questions awaiting Edward's answer

1. **What was TRADE_DIRECTION set to** for the Jan-26 dashboard backtest run that produced -$3,744? (Visible in the Backtest Settings panel of `report.html`.) If it was `"both"`, we need to rerun Jan with `short_only` before drawing further conclusions.
2. **Which strategy version** was Jan-26 run on — v1 or v2? v2 adds an EMA Long position filter (shorts only when price ≤ EMA Long) that would make it more restrictive than v1 or cTrader unless cTrader has the same filter.
3. **Does the cTrader bot use any EMA position filter** (e.g., trend filter blocking longs/shorts based on EMA), or is it pure fractal-shorts?

### Existing infrastructure relevant to next steps

- **TRADE_DIRECTION env var** is already wired end-to-end (UI → server → strategy). Default in localStorage is `"short_only"`. Supports `"both"`, `"long_only"`, `"short_only"`.
- v1 applies only `TRADE_DIRECTION` filter; v2 applies `TRADE_DIRECTION` **plus** EMA position filter (line ~400-405 in v2).

### Next steps when continuing in a new chat

1. Confirm TRADE_DIRECTION + strategy version that produced the -$3,744 Jan number.
2. If needed, **rerun Jan-26 with `short_only`** explicitly set, on the same version cTrader is running.
3. Compare the new dashboard trade list against the cTrader CSV at `/sessions/serene-pensive-pasteur/mnt/uploads/f7e7f7b5-11a9-4d4d-a849-0305ef017f57-1776699249499_jan.csv` trade-by-trade to identify any remaining fractal-detection drift between Massive.io mid bars and cTrader Ask/Bid ticks.
4. Decide whether to model commission/swap costs (~$7/trade) so dashboard P&L matches cTrader's net rather than gross.
5. Recommendation on `SL_SLIPPAGE_PIPS = 1.0`: leave as-is. Tuning further would fit noise — the dashboard's worst trades are already less severe than cTrader's worst (which includes real news-gap events that no fixed pip slippage can model).

### Diagnostic reference

- cTrader Jan CSV uploaded at: `/sessions/serene-pensive-pasteur/mnt/uploads/f7e7f7b5-11a9-4d4d-a849-0305ef017f57-1776699249499_jan.csv`
- Pre-compaction transcript (full conversation history): `/sessions/serene-pensive-pasteur/mnt/.claude/projects/-sessions-serene-pensive-pasteur/a76e75bc-5141-4184-ab6f-2f6c6e4034e9.jsonl`
