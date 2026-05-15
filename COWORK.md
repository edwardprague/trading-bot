# Project Overview

A backtesting dashboard for developing a cBot for cTrader. Python is the
research environment; the cBot is exported to C# via a code-generation
template so the live trading code matches the backtest line-for-line.

---

# Project Methodology

1. Development is task-driven — you receive tasks, then implement them.
2. Always ask clarifying questions when the request is ambiguous.
3. **New styling goes in `style.css`** — never inline `style="…"` attributes
   on new HTML. (The dashboard's `INJECT_HTML` in `server.py` has legacy
   inline styles; those are grandfathered, don't add more.)
4. When a task is done, state what the user needs to do to see the change:
   - Restart the Flask server
   - Run a backtest from the dashboard to regenerate `report.html`
   - Run `python3 regime_analysis.py` to regenerate `results/regime_analysis.html`
   - Or any combination of the above.

---

# Technical Setup

- **Working directory** — `~/Documents/GitHub/trading-bot`
- **Venv activation** — `source venv/bin/activate`
- **Repository** — `https://github.com/edwardprague/trading-bot` (public)
- **Session startup** — double-click `start.command` from Finder. It runs
  `cd ~/Documents/GitHub/trading-bot && git pull && source venv/bin/activate
  && python3 server.py`, opens the dashboard at `http://localhost:8080`.

## Python environment

Python 3 via Homebrew, with a project-local `venv/`. Key packages:

| Package        | Purpose                                       |
| -------------- | --------------------------------------------- |
| `flask`        | Dashboard + regime analysis web server         |
| `pandas`       | Data manipulation                             |
| `numpy`        | Numerical calculations                        |
| `pyarrow` / `fastparquet` | Parquet I/O (either works)         |
| `massive`      | Massive.io API client (data source)           |
| `python-dotenv`| Loads API keys from `.env`                    |
| `matplotlib`   | Chart generation                              |
| `ta` / `pandas-ta` | Technical indicators (ADX, EMA)           |
| `requests`     | HTTP requests                                 |

## Project files

| File / Folder           | Role                                                                          |
| ----------------------- | ----------------------------------------------------------------------------- |
| `server.py`             | Flask web server — routes the dashboard, regime analysis, and async backtests  |
| `strategy.py`           | Thin router — reads `STRATEGY_VERSION` env var, delegates to `strategy_vN.py` |
| `strategy_v1.py`        | Fractal-only entries                                                          |
| `strategy_v2.py`        | v1 + EMA position filter + regime gates (current)                             |
| `regime_analysis.py`     | Builds `regime_labels.parquet` + `results/regime_analysis.html`                |
| `cbot_templates.py`     | Renders `FractalBot_v{N}.cs` from a template for cTrader                      |
| `download_data.py`      | Fetches OHLC bars from Massive API → `data/<instrument>_<interval>.parquet`   |
| `start.command`         | Double-click startup (git pull + Flask)                                       |
| `report.html`           | Auto-generated dashboard — **never edit directly**; edit `strategy_vN.py`     |
| `style.css`             | Shared stylesheet for dashboard + regime analysis                              |
| `.env`                  | API keys (gitignored, never commit)                                           |
| `.gitignore`            | Excludes `.env`, `venv/`, `__pycache__/`                                      |
| `devlog.json`           | Free-form developer log; edited from the run bar's dev-log button             |
| `RESULTS_LOG.md`        | Manually-curated table of notable backtest results                            |
| `data/`                 | Cached OHLC parquets + regime labels (see below)                              |
| `results/`              | Generated reports + charts (see below)                                        |

## `data/` folder

| File                          | Contents                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `GBPUSD_5m.parquet`           | Cached GBPUSD 5-minute bars from Massive API (2009 → present)                     |
| `EURUSD_5m.parquet`           | Cached EURUSD 5-minute bars                                                       |
| `regime_labels.parquet`       | Per-fractal regime labels written by `regime_analysis.py`. Schema metadata key `regime_analysis` carries the macro_by_date JSON dict + tercile thresholds; the legacy `regime_labeler` key is also written for back-compat |
| `macro_by_date.json`          | Sidecar copy of macro labels — fallback when the parquet metadata is unavailable  |
| `regime_model.pkl`            | Legacy artifact (not currently used)                                              |

Refresh OHLC bars with `python3 download_data.py`.

## `results/` folder

| File / Folder                 | Contents                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `regime_analysis.html`         | Regime labeler page (regenerated by `python3 regime_analysis.py`)               |
| `regime_charts/YYYY-MM-DD.png`| Per-day price+trade preview charts (controlled by `GENERATE_DAILY_CHARTS`)     |
| Versioned PNGs                | Snapshots from past dashboard runs                                             |

The Flask `/results/<filename>` route serves any file in this folder over HTTP.
`/results/` (no filename) returns a directory listing.

## Version architecture

`strategy.py` reads `STRATEGY_VERSION` (default `v1`) and delegates to
`strategy_v{N}.py` via `runpy.run_path`. Server passes that env var to the
backtest subprocess. To add a new version, drop a `strategy_v3.py` alongside
the others and set `STRATEGY_VERSION=v3` on the next run.

---

# Strategy Logic

The Python backtest and the cTrader cBot follow the same execution flow. The
cBot is generated from `cbot_templates.py` via the dashboard's **Create cBot**
button. Changes to the bot's logic go in `cbot_templates.py`, not the
generated `.cs` file (which is overwritten on each generation).

## v1 — Fractal Geometry Entries

1. **Fractal detection** — Williams N=2. A fractal at bar `fi` requires bars
   `fi±1` and `fi±2` to all have lower highs (for a fractal high) or higher
   lows (for a fractal low). Confirmation happens 2 bars later.
2. **Entry signal**
   - **Long** — new fractal low forms higher than the prior fractal low.
   - **Short** — new fractal high forms lower than the prior fractal high.
3. **Direction filter** — `TRADE_DIRECTION` is `both` / `long_only` / `short_only`.
4. **Daily loss limit** — no new trades once the day's losing-trade count
   reaches `MAX_DAILY_LOSSES` (default **2**). Resets at UTC midnight.
5. **Time filter** — entries only fire outside `BLOCKED_HOURS_UTC` (the
   dashboard uses a blocked-hours list; the cBot template converts it to
   allowed-hours).
6. **Stop validation** — stop-distance must be between `MIN_STOP` (5 pips)
   and `MAX_STOP` (200 pips); signals outside this range are skipped.
7. **Entry reference**
   - Python — close of the bar after confirmation.
   - cBot — `Symbol.Ask` for longs, `Symbol.Bid` for shorts, on the first
     tick of the bar after confirmation.
8. **Stop-loss** — fractal price ± `FRACTAL_STOP_PIPS` offset (default **15**).
9. **Take-profit** — entry ± (stop distance × RRR), where RRR = `RRR_REWARD / RRR_RISK`.
10. **Position sizing** — `(equity × RISK_PCT) / stop-distance-in-price`,
    normalised to broker volume step.
11. **Re-entry** — no cooldown. A new signal on the bar after a close is
    accepted.

## v2 — adds EMA position filter, EMA buffer, regime gates

All of v1, plus four additional entry-time gates:

1. **EMA position filter** (toggle via `USE_EMA_FILTER`, default on) —
   evaluated on the confirmation bar's close. Long requires `close > EMA Long`;
   short requires `close < EMA Long`. `EMA_LONG` defaults to **40** bars.
2. **EMA buffer** (`EMA40_BUFFER_PIPS`, default **5**) — tightens the EMA
   filter so the close must beat EMA Long by at least the buffer:
   short requires `close < ema_long - buffer`; long requires `close > ema_long + buffer`.
   Setting `EMA40_BUFFER_PIPS=0` recovers v1-style strict-below.
3. **Macro regime gate** (`ALLOWED_MACRO_REGIMES`, default
   `"Staircase Down,Strong Down"`) — skips entries on days whose macro
   regime isn't in the allow-list. See **Regime Analysis** below.
4. **Micro regime gate** (`ALLOWED_MICRO_REGIMES`, default
   `"ranging-medium,ranging-wide"`) — skips entries whose micro regime at
   the entry timestamp isn't in the allow-list.

Both regime gates load their labels at module import from
`data/regime_labels.parquet`. If the parquet doesn't cover a day or
timestamp, the gate passes through silently — so a wider-range dashboard
backtest still produces trades.

## Entry gate pipeline (order matters)

Inside `run_backtest`, every candidate signal runs through these gates in
order. The **first** gate that rejects it stamps a `reason` on the blocked
signal — that ordering is what drives the counterfactual stats further down.

1. Signal generation (fractal + higher-low / lower-high test)
2. Direction filter (`TRADE_DIRECTION`)
3. EMA position filter (v2 only, if `USE_EMA_FILTER`)
4. Daily loss limit
5. Time filter (`BLOCKED_HOURS_UTC`)
6. Macro regime gate (v2 only)
7. Micro regime gate (v2 only)
8. Stop-distance validation (`MIN_STOP` / `MAX_STOP`)
9. Trade execution

## Current direction bias

`TRADE_DIRECTION` is set to `short_only`. No long trades are taken.

---

# Backtesting Dashboard vs. Regime Analysis

Both run on the same Flask server, share the same strategy code, but answer
different questions.

**Backtesting Dashboard** (`http://localhost:8080/`, served from `report.html`)
is the **iteration workbench**. From here you queue up new backtest runs —
change EMAs, RRR, blocked hours, date range, instrument, direction — and the
dashboard runs `strategy.py` as a subprocess to produce a new "version" of
results. Each run appears in the sidebar as a saved snapshot for comparison.
Answers: *"how does the strategy perform under this parameter set, and how
does that compare to my previous attempts?"*

**Regime Analysis** (`http://localhost:8080/results/regime_analysis.html`,
served from `regime_analysis.py`) is the **diagnostic lens**. It takes the
current strategy and breaks its behaviour down by **market regime** — both
day-level (macro) and intraday (micro). The interactive run bar lets you
toggle individual regimes on/off and immediately see how stats shift, plus
counterfactual stats for the locked regimes. Answers: *"which market
conditions should this strategy actually be trading in?"*

In short: the dashboard tunes parameters; the regime analysis tunes the
operating window. A typical workflow uses both.

---

# Regime Analysis

## Macro Regime

A per-day classification of overall daily character. Computed by combining
three signals: net displacement from open to close, EMA-40 slope across the
day, and N=18 fractal structure. One of five labels: `strong_down`,
`staircase_down`, `flat`, `staircase_up`, `strong_up`. Tunable constants
live at the top of `regime_analysis.py` (`LARGE_DISPLACEMENT_PIPS=30`,
`SMALL_DISPLACEMENT_PIPS=15`, `EMA_MACRO_PERIOD=40`, `N18_LOOKBACK=18`).

## Micro Regime

A per-fractal classification of local price structure. Built in two passes:
a coarse label (`trending_up`, `trending_down`, `ranging`, or `transitioning`)
from the rolling lookback of the last four same-kind fractals, then refined
into one of ten fine labels by per-period metrics: `trending_fast/medium/slow_down`,
`trending_fast/medium/slow_up`, `ranging_narrow/medium/wide`, `transitioning`.
The fast/medium/slow and narrow/medium/wide cuts are **quantile-based**
(terciles across all observed periods in the run), not absolute thresholds —
so a "fast" period in a quiet date range may not be fast by absolute standards.

## How labels and periods are produced

1. **Per-fractal coarse label** — Each Williams N=2 fractal looks at the
   last four same-kind fractals. Both highs and lows moving together →
   `trending_up` or `trending_down`. Neither directional → `ranging`.
   Conflicting → `transitioning`.
2. **Smoothing** — A confirmation rule (two same-kind fractals must agree
   on a new state) prevents flickering between regimes.
3. **Grouping** — Consecutive same-label fractals merge into a **period**
   with start/end timestamps and bar indices.
4. **Fine label via terciles** — Each period gets a metric: average
   pips-per-bar (trending) or channel width (ranging). The metric is
   bucketed into thirds across all periods, producing the fine labels.
5. **Push back** — The fine label is copied onto every fractal in the period.

So coarse labels make the periods; period-level metrics make the fine labels;
fine labels go back onto the fractals.

## What labels are used for

- `_check_micro_regime(ts)` does an `asof` lookup on per-fractal labels to
  allow/block trades at entry time.
- `_check_macro_regime(ts)` reads `macro_by_date` for the trade's day.
- Each trade gets its `regime` column from the fractal label at its entry timestamp.
- The hourly chips in the Daily Breakdown are coloured by fractal labels.
- The toggle-panel keys (`ranging_medium`, etc.) are the label keys.

## What periods are used for

- Tercile thresholds (pips-per-bar, width) — computed across periods.
- Threshold-distribution charts at the bottom of the report.
- Daily timeline cells and hourly chips — dominant regime per day/hour.
- Regime summary cards' "avg duration".
- The Regime periods table — chronological log of every detected period.

**Analogy:** fractal labels = diagnosis codes stamped on every office visit.
Periods = the "episode" — continuous stretches of the same diagnosis. Labels
drive decisions; periods are what you reason about for duration and trends.

## Counterfactual trades

The perf tables show stats for locked regimes too. Those stats are
**counterfactual** — what the strategy *would have* done on that regime if
its gate were unlocked.

**How they're computed.** Every signal `run_backtest` generates but blocks
goes into a `blocked_signals` list, each entry recording the would-be entry /
SL / TP / size and a `reason` tag (`ema_position`, `time`, `macro_regime`,
`micro_regime`, etc.). `_scan_outcome` walks forward bar-by-bar from the
would-be entry to see whether SL or TP would have hit first, and stamps a
pnl onto each blocked signal. At report time, locked-regime rows are built
from these blocked signals.

**Filter-specific.** Only signals blocked by the gate that's actually
locking the row are counted:
- Macro perf, locked row → `reason="macro_regime"` only.
- Micro perf, locked row → `reason="micro_regime"` only.

Signals blocked by other gates (EMA, time, etc.) wouldn't fire even if this
row's gate were unlocked, so they're excluded.

**Accuracy.** Trade count and win rate are exact. Aggregate P&L is
**approximate** — position sizing uses the cash level at the moment of
blocking (not what cash would be if the trade had fired), and the
daily-loss-limit isn't re-simulated. For low-frequency regimes the
approximation is close; for high-frequency regimes (Transitioning, Ranging
Narrow) the counterfactual P&L is a rough upper bound — a real re-run with
that gate unlocked will typically fire fewer trades and show smaller (often
negative) P&L due to position coupling and the daily-loss stop.

Locked rows are visually distinguished by a lock icon, a row-level dim, and
a counterfactual tooltip on the Total P&L cell.

## Toggle-state persistence

The regime analysis page saves the run bar's date range + toggle state to
`localStorage` under `regime_analysis.lastAnalysis.v1` and auto-runs the
analysis on page load. **Reset to Defaults** clears the saved state so the
next refresh falls back to the labeler's hardcoded defaults.

---

# Environment Variables Reference

| Variable                  | Default                       | Effect                                                 |
| ------------------------- | ----------------------------- | ------------------------------------------------------ |
| `STRATEGY_VERSION`        | `v1`                          | Routes `strategy.py` to `strategy_vN.py`               |
| `INSTRUMENT`              | `EURUSD`                      | One of `EURUSD`, `GBPUSD`                              |
| `INTERVAL`                | `5m`                          | Bar interval (`1m`, `5m`, `15m`, `60m`, etc.)          |
| `TRADE_DIRECTION`         | `both`                        | `both` / `long_only` / `short_only`                    |
| `EMA_SHORT`               | `8`                           | Short EMA period                                       |
| `EMA_MID`                 | `20`                          | Mid EMA period                                         |
| `EMA_LONG`                | `40`                          | Long EMA period (used by v2's EMA position filter)     |
| `RRR_RISK` / `RRR_REWARD` | `1` / `2`                     | RRR ratio                                              |
| `FRACTAL_STOP_PIPS`       | `15`                          | Pip offset for fractal-based SL                        |
| `MAX_DAILY_LOSSES`        | `2`                           | Daily-loss-stop threshold                              |
| `BLOCKED_HOURS_UTC`       | `4,5,6,8,10,11,14,17`         | UTC hours where entries are skipped                    |
| `USE_EMA_FILTER`          | `true`                        | v2 EMA position filter toggle                          |
| `EMA40_BUFFER_PIPS`       | `5`                           | Extra distance beyond EMA Long required for entry      |
| `ALLOWED_MACRO_REGIMES`   | `Staircase Down,Strong Down`  | Macro regime allow-list (empty = no macro gate)        |
| `ALLOWED_MICRO_REGIMES`   | `ranging-medium,ranging-wide` | Micro regime allow-list (empty = no micro gate)        |
| `APPLY_SLIPPAGE`          | `true`                        | Apply SL slippage in P&L                               |
| `SL_SLIPPAGE_PIPS`        | `1.0`                         | Pips of adverse slippage on SL fills                   |
| `SPREAD_PIPS`             | `1.0`                         | Round-trip spread cost per trade                       |
| `GENERATE_DAILY_CHARTS`   | `true`                        | `regime_analysis.py` per-day chart loop on/off          |

**Naming note:** Python uses `SCREAMING_SNAKE_CASE` (`MAX_DAILY_LOSSES`,
`EMA_LONG`). The cBot template uses `PascalCase` (`MaxDailyLosses`,
`EmaLong`). Same concepts, different conventions.

---

# Flask Endpoints

| Route                        | Method | Purpose                                                                    |
| ---------------------------- | ------ | -------------------------------------------------------------------------- |
| `/`                          | GET    | Dashboard — serves `report.html` with run-bar injected                     |
| `/style.css`                 | GET    | Shared stylesheet                                                          |
| `/run`                       | POST   | Kick off a backtest subprocess (new version, full date range)              |
| `/run_range`                 | POST   | Kick off a backtest over a specific date range                             |
| `/run_batch`                 | POST   | Queue multiple backtests                                                   |
| `/status`                    | GET    | Poll backtest progress (`{running, stage, progress, ok, no_data, error}`)  |
| `/delete_version`            | POST   | Delete a saved version                                                     |
| `/delete_run`                | POST   | Delete a single run within a version                                       |
| `/reorder_runs`              | POST   | Drag-and-drop reorder of runs                                              |
| `/generate_cbot`             | POST   | Render `FractalBot_v{N}.cs` from `cbot_templates.py`                       |
| `/devlog`                    | GET    | Return the contents of `devlog.json`                                       |
| `/devlog`                    | POST   | Save a new `devlog.json` array                                             |
| `/results/<path:filename>`   | GET    | Serve any file in `results/` (including nested paths like `regime_charts/`)|
| `/results` / `/results/`     | GET    | Directory listing of `results/`                                            |
| `/run_regime_analysis`       | POST   | Interactive regime-labeler update — refilters labels + reruns backtest stats, returns rendered HTML chunks. Does **not** re-run the labeler. |

---

# Regime Analysis — Operational Notes

- **Trigger a full regenerate**: `python3 regime_analysis.py`. Runs stages
  1–4, optionally generates per-day PNGs, writes `regime_labels.parquet` +
  `results/regime_analysis.html`, then opens it in the browser. Server URL is
  preferred over `file://` (so the Run Analysis button can call the Flask
  endpoint).
- **Skip chart generation**: `GENERATE_DAILY_CHARTS=false python3 regime_analysis.py`.
  Useful when iterating on labeler code — finishes in ~1 minute instead of
  5–10 minutes.
- **Widen the labeled range**: edit `START_DATE` / `END_DATE` at the top of
  `regime_analysis.py`. Currently set to `2025-01-01` → `2026-03-31`.
- **Run Analysis ≠ full regenerate**: the run bar's Run Analysis button only
  re-filters the *already-computed* parquet and reruns backtest stats.
  Generating new labels requires the terminal command above.

---

# cTrader Integration

- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot;
  the Python Open API is an external connection not suitable for prop firm use).
- **Code generation**: `cbot_templates.py` renders `FractalBot_v{N}.cs` on
  demand from the dashboard's **Create cBot** button. Logic changes go in
  `cbot_templates.py`, not the generated `.cs` (which is overwritten).

---

# Workflow Notes

- **`RESULTS_LOG.md`** — manually maintained. Append entries for notable runs
  (parameter set, date range, headline stats, takeaway). Not auto-updated.
- **`devlog.json`** — free-form developer log, edited from the dev-log icon
  on either page's run bar. Lives in the project root and is checked in.
- **Git hygiene** — `.env`, `venv/`, `__pycache__/` are gitignored. Commit
  `data/regime_labels.parquet` if you want the regime labels reproducible
  across machines; otherwise it's regenerated on first labeler run.
- **Adding a new strategy version** — drop a `strategy_v3.py` alongside the
  existing versions, set `STRATEGY_VERSION=v3` for backtests, and update the
  v2/v3 sections of this doc.
