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
4. **The strategy templates duplicate the BD HTML.** `strategy_v1.py` and
   `strategy_v2.py` each carry their own copy of `report.html`'s structure
   inside a `_build_html()` template. Any change to `report.html`'s
   inline JS or markup that should survive the next backtest regeneration
   must also be applied to **both** strategy templates. Failing to do so
   means the fix lives only until the next "Add Year" / "Add Date Range"
   click overwrites `report.html`.
5. When a task is done, state what the user needs to do to see the change:
    - Restart the Flask server (needed for `server.py` changes including `INJECT_HTML`)
    - Run a backtest from the dashboard to regenerate `report.html` (only if a strategy-template change should be visible immediately)
    - Run `python3 regime_analysis.py` to regenerate `results/regime_analysis.html` (RA page changes only)
    - Or any combination of the above.
6. **In-browser verification.** Where possible, exercise BD / RA / Versions / Discovery
   pages live via Claude-in-Chrome — read `[RA]`-prefixed console logs,
   inspect computed styles, fire change events — and report the
   observed end state. Logic-only changes can be sandbox-tested with
   `py_compile` and `node --check` on extracted JS regions, but visual
   bugs and event-flow bugs benefit from a real browser.

---

# Technical Setup

- **Working directory** — `~/Documents/GitHub/trading-bot`
- **Venv activation** — `source venv/bin/activate`
- **Repository** — `https://github.com/edwardprague/trading-bot` (public)
- **Session startup** — double-click `start.command` from Finder. It runs
  `cd ~/Documents/GitHub/trading-bot && git pull && source venv/bin/activate && python3 server.py`,
  opens the dashboard at `http://localhost:8080`.

## Python environment

Python 3 via Homebrew, with a project-local `venv/`. Key packages:

| Package                   | Purpose                                |
| ------------------------- | -------------------------------------- |
| `flask`                   | Dashboard + regime analysis web server |
| `pandas`                  | Data manipulation                      |
| `numpy`                   | Numerical calculations                 |
| `pyarrow` / `fastparquet` | Parquet I/O (either works)             |
| `massive`                 | Massive.io API client (data source)    |
| `python-dotenv`           | Loads API keys from `.env`             |
| `matplotlib`              | Chart generation                       |
| `ta` / `pandas-ta`        | Technical indicators (ADX, EMA)        |
| `requests`                | HTTP requests                          |

## Project files

| File / Folder        | Role                                                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server.py`          | Flask web server — routes BD, RA, Discovery, Versions, async backtests. Hosts `INJECT_HTML` (top-nav + run bar) injected on `/`                                 |
| `strategy.py`        | Thin router — reads `STRATEGY_VERSION` env var, delegates to `strategy_vN.py`                                                                                   |
| `strategy_v1.py`     | Fractal-only entries                                                                                                                                            |
| `strategy_v2.py`     | v1 + EMA position filter + regime gates (current base strategy for all profiles)                                                                                |
| `regime_analysis.py` | Builds `regime_labels.parquet` + `results/regime_analysis.html`                                                                                                 |
| `cbot_templates.py`  | Renders `FractalBot_v{N}.cs` from a template for cTrader                                                                                                        |
| `download_data.py`   | Fetches OHLC bars from Massive API → `data/<instrument>_<interval>.parquet`                                                                                     |
| `start.command`      | Double-click startup (git pull + Flask)                                                                                                                         |
| `report.html`        | Auto-generated dashboard — **never edit directly**; edit `strategy_vN.py`. Sidebar holds the run-history list; version + instrument selects live in the run bar |
| `style.css`          | Shared stylesheet for all pages. Run-bar selects + native date inputs styled via `.rb-select` / `.rb-date`                                                      |
| `.env`               | API keys (gitignored, never commit)                                                                                                                             |
| `.gitignore`         | Excludes `.env`, `venv/`, `__pycache__/`                                                                                                                        |
| `devlog.json`        | (Legacy) free-form developer log — superseded by per-version notes on the Versions page                                                                         |
| `RESULTS_LOG.md`     | (Legacy) manually-curated table of notable backtest results — superseded by per-version notes                                                                   |
| `data/`              | Cached OHLC parquets + regime labels + versions store (see below)                                                                                               |
| `results/`           | Generated reports + charts (see below)                                                                                                                          |

## `data/` folder

| File                    | Contents                                                                                                                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GBPUSD_5m.parquet`     | Cached GBPUSD 5-minute bars from Massive API (2009 → present)                                                                                                                                                              |
| `EURUSD_5m.parquet`     | Cached EURUSD 5-minute bars                                                                                                                                                                                                |
| `regime_labels.parquet` | Per-fractal regime labels written by `regime_analysis.py`. Schema metadata key `regime_analysis` carries the macro_by_date JSON dict + tercile thresholds; the legacy `regime_labeler` key is also written for back-compat |
| `macro_by_date.json`    | Sidecar copy of macro labels — fallback when the parquet metadata is unavailable                                                                                                                                           |
| `versions.json`         | Active-version id + per-version backtest params + regime allow-lists + notes. Source of truth for BD, RA, and Discovery — see **Versions store** below                                                                     |
| `regime_model.pkl`      | Legacy artifact (not currently used)                                                                                                                                                                                       |

Refresh OHLC bars with `python3 download_data.py`.

## `results/` folder

| File / Folder                  | Contents                                                                   |
| ------------------------------ | -------------------------------------------------------------------------- |
| `regime_analysis.html`         | Regime labeler page (regenerated by `python3 regime_analysis.py`)          |
| `regime_charts/YYYY-MM-DD.png` | Per-day price+trade preview charts (controlled by `GENERATE_DAILY_CHARTS`) |
| Versioned PNGs                 | Snapshots from past dashboard runs                                         |

The Flask `/results/<filename>` route serves any file in this folder over HTTP.
`/results/` (no filename) returns a directory listing.

---

# The Four-Page Platform

The platform lives at `http://localhost:8080` and has four pages accessible via the top navigation bar.

## 1. Backtesting (BD) — `http://localhost:8080/`

The **iteration workbench**. Queue up backtest runs against the active version's parameters and regime filters. Each run appears in the sidebar as a saved entry for comparison.

- **Version dropdown** — only shows versions with assigned `backtest_params`. Selecting a version pre-fills the run bar (EMA Long, stop loss, RRR, DLL) from those params.
- **Instrument dropdown** — GBPUSD or EURUSD. Sidebar filters to show only runs for the selected instrument.
- **Run bar** — date pickers + Add Year / Add Date Range buttons trigger new backtests with progress indicators.
- **Sidebar** — lists all saved runs for the active version + instrument, grouped by YEAR / MONTH / WEEKS / DAY. Hover a run to reveal an eye icon; hover the eye to preview its chart image.
- **Report area** — Results panel (net profit, win rate, PF, drawdown) + Parameters panel. General / Advanced tabs.
- **Active version** shown in top-right corner at all times.

## 2. Regimes (RA) — `http://localhost:8080/results/regime_analysis.html`

The **diagnostic lens**. Breaks strategy behaviour down by market regime — day-level (macro) and intraday (micro). Toggle panel lets you switch regimes on/off and re-run to see how stats shift, plus counterfactual stats for locked regimes.

- **Macro regimes** (5): `strong_down`, `staircase_down`, `flat`, `staircase_up`, `strong_up`
- **Micro regimes** (10): `trending_fast/medium/slow_down/up`, `ranging_narrow/medium/wide`, `transitioning`
- **Version dropdown** — same rule as BD: only assigned versions appear.
- **Run Analysis button** — triggers a full backtest re-run with the active toggle state (~30–100s). Synchronous; page updates on completion.
- **Per-version state** — each version's toggle state saved to `versions.json`. Switching versions loads that version's saved state.
- **The RA does NOT auto-run on page load.** Results shown are from the last saved run for that version.

## 3. Discovery — `http://localhost:8080/discovery`

The **optimisation engine**. Runs a random search across the v2 parameter space. Each trial samples a parameter combination, runs a full backtest over the configured date range, and scores it against the objective (PF ≥ 1.5, trades ≥ 50, max DD ≤ 15%).

- **Fixed constants across all trials:** 5m interval, GBPUSD, short only, blocked hours as per current defaults, slippage/spread at realistic values.
- **Sampled parameters per trial:** EMA Long, EMA filter on/off, stop loss pips, RRR reward, max daily losses, macro regime allow-list, micro regime allow-list.
- **Run Configuration** — start/end date, trial count, optional seed.
- **Results table** — shows all trials on completion, sortable by PF. "Show passing only" toggle filters to objective-passing trials.
- **Assign flow** — clicking Assign on a passing trial opens a modal with a select menu of currently unassigned versions. Selecting one and confirming writes the full trial params to that version's `backtest_params` and `regime_state` in `versions.json`. This is a one-time write; params are immutable after assignment.
- Results persist across server restarts within the same run session; a new Run Discovery starts fresh.

## 4. Versions — `http://localhost:8080/versions`

The **lifecycle manager**. Each version is a named profile that bundles backtest parameters (assigned from Discovery) with its own regime allow-list state and free-form notes.

- **Two states:** unassigned (blank slot, not visible in BD/RA dropdowns) and assigned (params locked, visible everywhere).
- **Add Version button** — creates the next numbered blank version (e.g. v4 if v3 is the latest). No selectors; the slot is empty until Discovery assigns params to it.
- **Version card** — shows name, unassigned placeholder or assigned indicator, notes field (editable, auto-saves on blur), and a delete button.
- **Deleting an assigned version** with existing BD runs shows a warning that run history will also be deleted.
- **Parameters are immutable** after assignment. To change params, delete the version and assign a new Discovery result to a fresh slot.
- `data/versions.json` is tracked in git — commit after any version changes.

---

# Versions store (`data/versions.json`)

The versions store is the single source of truth for BD, RA, and Discovery. It holds the active version id, and for each version: its backtest params, regime allow-list state, and notes.

```jsonc
{
    "active_version_id": "v3",
    "versions": [
        {
            "id": "v3",
            "name": "v3",
            "strategy_version": "v2",
            "backtest_params": {
                "ema_long": 50,
                "use_ema_filter": true,
                "fractal_stop_pips": 10,
                "rrr_risk": 1,
                "rrr_reward": 1,
                "max_daily_losses": 3,
                "blocked_hours": [4, 5, 6, 8, 10, 11, 14, 17],
                "instrument": "GBPUSD",
                "interval": "5m",
                "trade_direction": "short_only",
                "assigned_at": "2026-05-19T...",
            },
            "regime_state": {
                "allowed_macro_regimes": ["staircase_down", "strong_down"],
                "allowed_micro_regimes": ["ranging_medium", "ranging_wide", "transitioning", "trending_fast_down", "trending_slow_down"],
                "updated_at": "2026-05-19T...",
            },
            "notes": "",
            "notes_updated_at": "2026-05-19T...",
        },
    ],
}
```

**Key rules:**

- `backtest_params: null` means the version is unassigned. It will not appear in BD or RA dropdowns.
- `backtest_params` is written once by the Discovery assign flow and is never updated afterward.
- `regime_state` is written by RA's Run Analysis toggle flow and can change independently.
- BD's `/run`, `/run_range`, `/run_batch` read the active profile's `backtest_params` to pre-fill run bar defaults, and `regime_state` to populate `ALLOWED_MACRO_REGIMES` / `ALLOWED_MICRO_REGIMES` env vars on the backtest subprocess (`_apply_active_version_to_env`).
- RA's `/run_regime_analysis` writes its toggle state back into the active profile's `regime_state` so BD + RA stay in sync.
- Run-level metadata carries the version active at run time — every `new_run` written by `strategy_vN.py` includes `"active_version": <profile name>`. The sidebar reads this in preference to the bucket name.

---

# Strategy Logic

The Python backtest and the cTrader cBot follow the same execution flow. The
cBot is generated from `cbot_templates.py` via the dashboard's **Create cBot**
button. Changes to the bot's logic go in `cbot_templates.py`, not the
generated `.cs` file (which is overwritten on each generation).

## Strategy modules

Two strategy modules exist. Version **profiles** (v3, v4, etc.) are separate from strategy modules — a profile points at a module via its `strategy_version` field.

- `strategy_v1.py` — fractal-only entries, no EMA filter, no regime gates.
- `strategy_v2.py` — v1 + EMA position filter + EMA buffer + macro/micro regime gates. All current profiles use this module.

`strategy.py` reads `STRATEGY_VERSION` (default `v1`) and delegates to `strategy_vN.py` via `runpy.run_path`. The server passes that env var to the backtest subprocess.

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
5. **Time filter** — entries only fire outside `BLOCKED_HOURS_UTC`.
6. **Stop validation** — stop-distance must be between `MIN_STOP` (5 pips)
   and `MAX_STOP` (200 pips); signals outside this range are skipped.
7. **Entry reference**
    - Python — close of the bar after confirmation.
    - cBot — `Symbol.Ask` for longs, `Symbol.Bid` for shorts, on the first tick of the bar after confirmation.
8. **Stop-loss** — fractal price ± `FRACTAL_STOP_PIPS` offset (default **15**).
9. **Take-profit** — entry ± (stop distance × RRR), where RRR = `RRR_REWARD / RRR_RISK`.
10. **Position sizing** — `(equity × RISK_PCT) / stop-distance-in-price`, normalised to broker volume step.
11. **Re-entry** — no cooldown. A new signal on the bar after a close is accepted.

## v2 — adds EMA position filter, EMA buffer, regime gates

All of v1, plus four additional entry-time gates:

1. **EMA position filter** (toggle via `USE_EMA_FILTER`, default on) —
   evaluated on the confirmation bar's close. Long requires `close > EMA Long`;
   short requires `close < EMA Long`. `EMA_LONG` defaults to **40** bars.
2. **EMA buffer** (`EMA40_BUFFER_PIPS`, default **5**) — tightens the EMA
   filter: short requires `close < ema_long - buffer`; long requires `close > ema_long + buffer`.
   Setting `EMA40_BUFFER_PIPS=0` recovers strict-at-EMA behaviour.
3. **Macro regime gate** (`ALLOWED_MACRO_REGIMES`) — skips entries on days whose macro
   regime isn't in the allow-list.
4. **Micro regime gate** (`ALLOWED_MICRO_REGIMES`) — skips entries whose micro regime at
   the entry timestamp isn't in the allow-list.

Both regime gates load their labels at module import from
`data/regime_labels.parquet`. If the parquet doesn't cover a day or
timestamp, the gate passes through silently.

## Entry gate pipeline (order matters)

Inside `run_backtest`, every candidate signal runs through these gates in
order. The **first** gate that rejects it stamps a `reason` on the blocked
signal — that ordering drives the counterfactual stats.

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
is the **iteration workbench**. Answers: _"how does the strategy perform under this parameter set?"_

**Regime Analysis** (`http://localhost:8080/results/regime_analysis.html`)
is the **diagnostic lens**. Answers: _"which market conditions should this strategy actually be trading in?"_

In short: Discovery finds parameters; the dashboard validates them; regime analysis tunes the operating window.

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
(terciles across all observed periods in the run), not absolute thresholds.

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

## What labels are used for

- `_check_micro_regime(ts)` does an `asof` lookup on per-fractal labels to allow/block trades at entry time.
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

**Analogy:** fractal labels = diagnosis codes stamped on every office visit. Periods = the "episode" — continuous stretches of the same diagnosis. Labels drive decisions; periods are what you reason about for duration and trends.

## Counterfactual trades

The perf tables show stats for locked regimes too. Those stats are
**counterfactual** — what the strategy _would have_ done on that regime if
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

**Accuracy.** Trade count and win rate are exact. Aggregate P&L is
**approximate** — position sizing uses the cash level at the moment of
blocking, and the daily-loss-limit isn't re-simulated. For high-frequency
regimes the counterfactual P&L is a rough upper bound.

Locked rows are visually distinguished by a lock icon, a row-level dim, and
a counterfactual tooltip on the Total P&L cell.

## Toggle-state persistence

The RA page persists the run-bar state and rendered HTML chunks into
`localStorage` under a per-active-version key: regime_analysis.lastAnalysis.v3.<active_version_id>

On page load the cached HTML chunks are painted **only if** the saved
payload's allow-lists still match versions.json's current `regime_state`
(coherence check). If the allow-lists diverge, sections dim until the user
clicks **Run Analysis** to refresh.

While toggles diverge from the rendered state, the page adds a
`body.regime-stale` class — sections dim to 0.45 opacity and the Run
button gets a yellow halo + "← toggles changed" hint. (CSS lives in `style.css`.)

**Reset to Defaults** clears only the current profile's cache slot.
The RA page deliberately does **not** auto-run on load.

## Future work — RA caching

A first attempt at caching the unfiltered backtest result and post-
filtering by toggle state was reverted because the daily-loss-limit
interacts with the regime gates in a way that produces materially different
trade counts. The clean fix is to expose `run_backtest`'s **pre-DLL signal
stream** so the RA endpoint can re-simulate the DLL per toggle state on
a cached signal list. Tracked as a future engineering task; relevant comment
block lives at the top of the `/run_regime_analysis` section in `server.py`.

---

# Environment Variables Reference

| Variable                  | Default                       | Effect                                             |
| ------------------------- | ----------------------------- | -------------------------------------------------- |
| `STRATEGY_VERSION`        | `v1`                          | Routes `strategy.py` to `strategy_vN.py`           |
| `INSTRUMENT`              | `EURUSD`                      | One of `EURUSD`, `GBPUSD`                          |
| `INTERVAL`                | `5m`                          | Bar interval (`1m`, `5m`, `15m`, `60m`, etc.)      |
| `TRADE_DIRECTION`         | `both`                        | `both` / `long_only` / `short_only`                |
| `EMA_SHORT`               | `8`                           | Short EMA period                                   |
| `EMA_MID`                 | `20`                          | Mid EMA period                                     |
| `EMA_LONG`                | `40`                          | Long EMA period (used by v2's EMA position filter) |
| `RRR_RISK` / `RRR_REWARD` | `1` / `2`                     | RRR ratio                                          |
| `FRACTAL_STOP_PIPS`       | `15`                          | Pip offset for fractal-based SL                    |
| `MAX_DAILY_LOSSES`        | `2`                           | Daily-loss-stop threshold                          |
| `BLOCKED_HOURS_UTC`       | `4,5,6,8,10,11,14,17`         | UTC hours where entries are skipped                |
| `USE_EMA_FILTER`          | `true`                        | v2 EMA position filter toggle                      |
| `EMA40_BUFFER_PIPS`       | `5`                           | Extra distance beyond EMA Long required for entry  |
| `ALLOWED_MACRO_REGIMES`   | `Staircase Down,Strong Down`  | Macro regime allow-list (empty = no macro gate)    |
| `ALLOWED_MICRO_REGIMES`   | `ranging-medium,ranging-wide` | Micro regime allow-list (empty = no micro gate)    |
| `APPLY_SLIPPAGE`          | `true`                        | Apply SL slippage in P&L                           |
| `SL_SLIPPAGE_PIPS`        | `1.0`                         | Pips of adverse slippage on SL fills               |
| `SPREAD_PIPS`             | `1.0`                         | Round-trip spread cost per trade                   |
| `GENERATE_DAILY_CHARTS`   | `true`                        | `regime_analysis.py` per-day chart loop on/off     |

**Naming note:** Python uses `SCREAMING_SNAKE_CASE` (`MAX_DAILY_LOSSES`,
`EMA_LONG`). The cBot template uses `PascalCase` (`MaxDailyLosses`,
`EmaLong`). Same concepts, different conventions.

---

# Flask Endpoints

| Route                      | Method | Purpose                                                                                                                                                            |
| -------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/`                        | GET    | Dashboard — serves `report.html` with run-bar injected at start of `<body>` and `Cache-Control: no-store`                                                          |
| `/style.css`               | GET    | Shared stylesheet                                                                                                                                                  |
| `/versions`                | GET    | Versions-management page (add / delete / set active / edit notes)                                                                                                  |
| `/discovery`               | GET    | Discovery page                                                                                                                                                     |
| `/run`                     | POST   | Kick off a backtest subprocess (full date range)                                                                                                                   |
| `/run_range`               | POST   | Kick off a backtest over a specific date range                                                                                                                     |
| `/run_batch`               | POST   | Queue multiple backtests                                                                                                                                           |
| `/status`                  | GET    | Poll backtest progress (`{running, stage, progress, ok, no_data, error}`)                                                                                          |
| `/delete_version`          | POST   | Delete a saved version (from `report.html`'s VERSIONS array)                                                                                                       |
| `/delete_run`              | POST   | Delete a single run within a version                                                                                                                               |
| `/reorder_runs`            | POST   | Drag-and-drop reorder of runs                                                                                                                                      |
| `/generate_cbot`           | POST   | Render `FractalBot_v{N}.cs` from `cbot_templates.py`                                                                                                               |
| `/devlog`                  | GET    | Return the contents of `devlog.json` (legacy)                                                                                                                      |
| `/devlog`                  | POST   | Save a new `devlog.json` array (legacy)                                                                                                                            |
| `/api/versions`            | GET    | Return the full `versions.json` store                                                                                                                              |
| `/api/versions`            | POST   | Add a new blank version (no body required; name auto-assigned)                                                                                                     |
| `/api/versions/<id>`       | DELETE | Delete a profile and its run history. Refuses last one; auto-switches active if needed                                                                             |
| `/api/versions/<id>/notes` | POST   | Update a profile's `notes` field (body: `{notes: "…"}`). Auto-stamps `notes_updated_at`                                                                            |
| `/api/active_version`      | GET    | Return the currently-active profile dict                                                                                                                           |
| `/api/active_version`      | POST   | Switch the active profile (body: `{id}`)                                                                                                                           |
| `/api/discovery/run`       | POST   | Start a Discovery run (body: `{start, end, trials, seed?}`). Async; poll `/api/discovery/status` for progress                                                      |
| `/api/discovery/status`    | GET    | Poll Discovery run progress (`{running, trials_complete, trials_total, best, error, ...}`)                                                                         |
| `/api/discovery/results`   | GET    | Return all trial results from the current/last Discovery run                                                                                                       |
| `/api/discovery/assign`    | POST   | Assign a trial's params to an unassigned version (body: `{trial_id, version_id}`). Writes `backtest_params` + `regime_state` to versions.json. One-time write      |
| `/results/<path:filename>` | GET    | Serve any file in `results/`. `.html` responses get `Cache-Control: no-store`                                                                                      |
| `/results` / `/results/`   | GET    | Directory listing of `results/`                                                                                                                                    |
| `/run_regime_analysis`     | POST   | Synchronous RA backtest re-run — applies the requested allow-lists as in-backtest gates, returns rendered HTML chunks. Writes `regime_state` back to versions.json |

---

# Regime Analysis — Operational Notes

- **Trigger a full regenerate**: `python3 regime_analysis.py`. Runs stages
  1–4, optionally generates per-day PNGs, writes `regime_labels.parquet` +
  `results/regime_analysis.html`, then opens it in the browser. Server URL is
  preferred over `file://`.
- **Skip chart generation**: `GENERATE_DAILY_CHARTS=false python3 regime_analysis.py`.
  Useful when iterating on labeler code — finishes in ~1 minute instead of 5–10 minutes.
- **Widen the labeled range**: edit `START_DATE` / `END_DATE` at the top of
  `regime_analysis.py`. Currently set to `2025-01-01` → `2026-03-31`.
- **Run Analysis ≠ full regenerate**: the run bar's Run Analysis button only
  re-filters the _already-computed_ parquet and reruns backtest stats.
  Generating new labels requires the terminal command above.
- **Run Analysis is synchronous and ~30–100s per click.**

---

# Backtest Dashboard Testing by Claude

Chat (Claude in this session) is primarily responsible for browser testing of the platform via the Claude-in-Chrome extension.

## Backtest Dashboard — Keyboard Shortcuts

All single-key shortcuts ignore Ctrl / Cmd / Alt / Shift modifiers
(except where Shift is explicitly required) and are suppressed when
focus is in an input, textarea, select, or contenteditable element.

| Key                              | Action                                                                        |
| -------------------------------- | ----------------------------------------------------------------------------- |
| `↑` / `↓`                        | Sidebar navigation — move selection up/down                                   |
| `1`–`9`                          | Scroll to Nth visible section in the active tab                               |
| `0`                              | Scroll to bottom of report                                                    |
| `V` + `1`–`9`                    | Pick Nth version from the version dropdown (chord, ~800 ms window)            |
| `I` + `1`–`9`                    | Pick Nth instrument from the instrument dropdown (chord)                      |
| `D`                              | Add Date Range                                                                |
| `C`                              | Copy Report                                                                   |
| `L`                              | Toggle Development Log                                                        |
| `S`                              | Toggle Backtest Settings panel (shared key on Regimes + Discovery)            |
| `G`                              | General tab                                                                   |
| `A`                              | Advanced tab                                                                  |
| `Shift` + `1`–`4`                | Toggle sidebar section — Year / Month / Weeks / Day                           |
| `Shift` + `Delete` / `Backspace` | Delete (currently-selected version or run)                                    |
| `Enter`                          | Submit in the Add Version dialog and in in-place name/description edit fields |

## Regime Analysis — Keyboard Shortcuts

| Key | Action                                                  |
| --- | ------------------------------------------------------- |
| `1` | Scroll to top                                           |
| `2` | Stats cards                                             |
| `3` | Macro regime performance                                |
| `4` | Micro regime performance                                |
| `5` | Regime timeline                                         |
| `6` | Daily performance                                       |
| `7` | Macro Regime Profiles                                   |
| `8` | Regime summary cards                                    |
| `9` | Threshold distributions                                 |
| `0` | Regime periods (bottom)                                 |
| `S` | Toggle Regime Filters panel (shared key on BD + Discovery) |

## Discovery — Keyboard Shortcuts

| Key | Action                                                       |
| --- | ------------------------------------------------------------ |
| `S` | Toggle Discovery Settings panel (shared key on BD + Regimes) |

---

# cTrader Integration

- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot;
  the Python Open API is an external connection not suitable for prop firm use).
- **Code generation**: `cbot_templates.py` renders `FractalBot_v{N}.cs` on
  demand from the dashboard's **Create cBot** button. Logic changes go in
  `cbot_templates.py`, not the generated `.cs` (which is overwritten).

---

# Workflow Notes

- **Version lifecycle** — versions are created as blank slots on the Versions page, assigned parameters once from Discovery, and then immutable. To change parameters, delete the version and create a new slot.
- **Per-version notes** — the `/versions` page exposes a free-form notes textarea per profile. Notes persist to `data/versions.json`, auto-saving on blur via `POST /api/versions/<id>/notes`.
- **Run-level metadata carries the active version** — each `new_run` written by `strategy_vN.py` includes `"active_version": <profile name>`. The sidebar reads this in preference to the bucket name.
- **`RESULTS_LOG.md`** (legacy) — superseded by per-version notes.
- **`devlog.json`** (legacy) — superseded by per-version notes.
- **Git hygiene** — `.env`, `venv/`, `__pycache__/` are gitignored. Commit `data/regime_labels.parquet` if you want regime labels reproducible across machines. Always commit `data/versions.json` after version changes.
- **Adding a new strategy module** — drop a `strategy_v3.py` alongside the existing ones. A version profile can then point at it via its `strategy_version` field.
- **Adding a new version profile** — use the Versions page Add Version button to create a blank slot, then assign params from Discovery.
