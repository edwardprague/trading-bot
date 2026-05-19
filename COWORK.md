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
6. **In-browser verification.** Where possible, exercise BD / RA / Versions
   pages live via Claude-in-Chrome — read `[RA]`-prefixed console logs,
   inspect computed styles, fire change events — and report the
   observed end state. Logic-only changes can be sandbox-tested with
   `py_compile` and `node --check` on extracted JS regions, but visual
   bugs and event-flow bugs benefit from a real browser.

7. **Task completion notification.** At the end of every completed task, fire
   a macOS notification so the user is alerted without needing to check back
   manually:

```bash
   osascript -e 'display notification "Task complete" with title "Ada" sound name "Glass"'
```

Customise the message to reflect the task — e.g. `"Discovery page built"`,
`"Phase 1 complete"`. No setup required; `osascript` is built into macOS.
If macOS blocks the notification on first run, the user will need to allow
it once under **System Settings → Notifications**.

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

| File / Folder        | Role                                                                                                                                                                                                                                      |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `server.py`          | Flask web server — routes the dashboard, regime analysis, async backtests, and the `/versions` page. Hosts `INJECT_HTML` (top-nav + run bar) injected on `/`                                                                              |
| `strategy.py`        | Thin router — reads `STRATEGY_VERSION` env var, delegates to `strategy_vN.py`                                                                                                                                                             |
| `strategy_v1.py`     | Fractal-only entries                                                                                                                                                                                                                      |
| `strategy_v2.py`     | v1 + EMA position filter + regime gates (current)                                                                                                                                                                                         |
| `regime_analysis.py` | Builds `regime_labels.parquet` + `results/regime_analysis.html`                                                                                                                                                                           |
| `cbot_templates.py`  | Renders `FractalBot_v{N}.cs` from a template for cTrader                                                                                                                                                                                  |
| `download_data.py`   | Fetches OHLC bars from Massive API → `data/<instrument>_<interval>.parquet`                                                                                                                                                               |
| `start.command`      | Double-click startup (git pull + Flask)                                                                                                                                                                                                   |
| `report.html`        | Auto-generated dashboard — **never edit directly**; edit `strategy_vN.py`. Sidebar now holds only the run-history list — the version + instrument selects live in the run bar (rendered by `INJECT_HTML`) and are shared with the RA page |
| `style.css`          | Shared stylesheet for dashboard + regime analysis. Run-bar selects + native date inputs are styled via `.rb-select` / `.rb-date`                                                                                                          |
| `.env`               | API keys (gitignored, never commit)                                                                                                                                                                                                       |
| `.gitignore`         | Excludes `.env`, `venv/`, `__pycache__/`                                                                                                                                                                                                  |
| `devlog.json`        | (Legacy) free-form developer log — superseded by per-version notes on the Versions page                                                                                                                                                   |
| `RESULTS_LOG.md`     | (Legacy) manually-curated table of notable backtest results — superseded by per-version notes                                                                                                                                             |
| `data/`              | Cached OHLC parquets + regime labels + versions store (see below)                                                                                                                                                                         |
| `results/`           | Generated reports + charts (see below)                                                                                                                                                                                                    |

## `data/` folder

| File                    | Contents                                                                                                                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GBPUSD_5m.parquet`     | Cached GBPUSD 5-minute bars from Massive API (2009 → present)                                                                                                                                                              |
| `EURUSD_5m.parquet`     | Cached EURUSD 5-minute bars                                                                                                                                                                                                |
| `regime_labels.parquet` | Per-fractal regime labels written by `regime_analysis.py`. Schema metadata key `regime_analysis` carries the macro_by_date JSON dict + tercile thresholds; the legacy `regime_labeler` key is also written for back-compat |
| `macro_by_date.json`    | Sidecar copy of macro labels — fallback when the parquet metadata is unavailable                                                                                                                                           |
| `versions.json`         | Active-version id + per-version regime allow-lists + free-form notes. Source of truth for BD's "active version" context — see **Versions store** below                                                                     |
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

## Version architecture

`strategy.py` reads `STRATEGY_VERSION` (default `v1`) and delegates to
`strategy_v{N}.py` via `runpy.run_path`. Server passes that env var to the
backtest subprocess. To add a new version, drop a `strategy_v3.py` alongside
the others and set `STRATEGY_VERSION=v3` on the next run.

## Versions store (`data/versions.json`)

Distinct concept from the strategy modules above. A **version** here is a
user-facing **profile** — a name + the strategy module to use + a regime
allow-list + free-form notes. The active profile drives every new backtest
and the RA toggle scope.

```jsonc
{
  "active_version_id": "v3",
  "versions": [
    {
      "id": "v3",
      "name": "v3",
      "strategy_version": "v2", // which strategy_vN.py module
      "regime_state": {
        "allowed_macro_regimes": ["staircase_down", "strong_down"],
        "allowed_micro_regimes": ["ranging_medium", "ranging_wide"],
        "updated_at": "2026-05-17T11:39:45Z",
      },
      "notes": "free-form notes — see Versions page",
      "notes_updated_at": "2026-05-17T11:42:10Z",
    },
  ],
}
```

- The **seeded** entries `v1` and `v2` ship with the project; their `id` /
  `name` matches a strategy module (`strategy_v1.py` / `strategy_v2.py`).
- **User-added profiles** (e.g. `v3`) typically reuse a base strategy
  module — `strategy_version` points at it — and override only the regime
  state.
- BD's `/run`, `/run_range`, `/run_batch` read the active profile's
  `regime_state` to populate `ALLOWED_MACRO_REGIMES` / `ALLOWED_MICRO_REGIMES`
  env vars on the backtest subprocess (`_apply_active_version_to_env`).
- RA's `/run_regime_analysis` writes its toggle state back into the active
  profile's `regime_state` so BD + RA stay in sync.
- The `/versions` page is the UI for adding, deleting, switching active,
  and editing the notes field. Notes auto-save on blur.

Run-level metadata also carries the version that was active when each run
was performed — every `new_run` written by `strategy_vN.py` includes
`"active_version": <profile name>`. The sidebar uses this in preference to
the bucket name so display stays accurate even if the run was ever stored
in a fallback bucket.

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
Answers: _"how does the strategy perform under this parameter set, and how
does that compare to my previous attempts?"_

**Regime Analysis** (`http://localhost:8080/results/regime_analysis.html`,
served from `regime_analysis.py`) is the **diagnostic lens**. It takes the
current strategy and breaks its behaviour down by **market regime** — both
day-level (macro) and intraday (micro). The interactive run bar lets you
toggle individual regimes on/off and immediately see how stats shift, plus
counterfactual stats for the locked regimes. Answers: _"which market
conditions should this strategy actually be trading in?"_

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

The RA page persists the run-bar state (date range, instrument, toggle
selections) and the rendered HTML chunks into `localStorage` under a
per-active-version key:

```
regime_analysis.lastAnalysis.v3.<active_version_id>
```

So `regime_analysis.lastAnalysis.v3.v3` holds v3's last-run state,
`regime_analysis.lastAnalysis.v3.v2` holds v2's, etc. Switching the
active profile and returning restores the matching profile's state —
no bleed-through.

On page load the cached HTML chunks are painted **only if** the saved
payload's allow-lists still match versions.json's current `regime_state`
(coherence check). If a manual edit to versions.json changes the regime
allow-lists out from under the cache, the dimmed-but-readable static
sections stay until the user clicks **Run Analysis** to refresh.

While toggles diverge from the rendered state, the page adds a
`body.regime-stale` class — sections dim to 0.45 opacity and the Run
button gets a yellow halo + "← toggles changed" hint. Cleared on the
next successful Run Analysis. (CSS lives in `style.css`.)

**Reset to Defaults** clears only the current profile's cache slot.
The RA page deliberately does **not** auto-run on load — the user
clicks Run Analysis when they want fresh stats.

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

| Route                      | Method | Purpose                                                                                                                                                                                                         |
| -------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/`                        | GET    | Dashboard — serves `report.html` with run-bar injected at start of `<body>` and `Cache-Control: no-store` on the response                                                                                       |
| `/style.css`               | GET    | Shared stylesheet                                                                                                                                                                                               |
| `/versions`                | GET    | Versions-management page (add / delete / set active / edit notes)                                                                                                                                               |
| `/run`                     | POST   | Kick off a backtest subprocess (new version, full date range)                                                                                                                                                   |
| `/run_range`               | POST   | Kick off a backtest over a specific date range                                                                                                                                                                  |
| `/run_batch`               | POST   | Queue multiple backtests                                                                                                                                                                                        |
| `/status`                  | GET    | Poll backtest progress (`{running, stage, progress, ok, no_data, error}`) — BD only; RA's `/run_regime_analysis` is synchronous and doesn't write here                                                          |
| `/delete_version`          | POST   | Delete a saved version (from `report.html`'s VERSIONS array)                                                                                                                                                    |
| `/delete_run`              | POST   | Delete a single run within a version                                                                                                                                                                            |
| `/reorder_runs`            | POST   | Drag-and-drop reorder of runs                                                                                                                                                                                   |
| `/generate_cbot`           | POST   | Render `FractalBot_v{N}.cs` from `cbot_templates.py`                                                                                                                                                            |
| `/devlog`                  | GET    | Return the contents of `devlog.json` (legacy; superseded by per-version notes)                                                                                                                                  |
| `/devlog`                  | POST   | Save a new `devlog.json` array (legacy)                                                                                                                                                                         |
| `/api/versions`            | GET    | Return the full `versions.json` store                                                                                                                                                                           |
| `/api/versions`            | POST   | Add a new profile (body: `{strategy_version, base_id?}`)                                                                                                                                                        |
| `/api/versions/<id>`       | DELETE | Delete a profile. Refuses last one; auto-switches active if needed                                                                                                                                              |
| `/api/versions/<id>/notes` | POST   | Update a profile's `notes` field (body: `{notes: "…"}`). Auto-stamps `notes_updated_at`                                                                                                                         |
| `/api/active_version`      | GET    | Return the currently-active profile dict                                                                                                                                                                        |
| `/api/active_version`      | POST   | Switch the active profile (body: `{id}`)                                                                                                                                                                        |
| `/results/<path:filename>` | GET    | Serve any file in `results/`. `.html` responses get `Cache-Control: no-store` so dev-time edits to inline JS are always picked up on reload                                                                     |
| `/results` / `/results/`   | GET    | Directory listing of `results/`                                                                                                                                                                                 |
| `/run_regime_analysis`     | POST   | Synchronous RA backtest re-run — applies the requested allow-lists as in-backtest gates, returns rendered HTML chunks. Long-running (~30–100s on a 15-month range). Writes `regime_state` back to versions.json |

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
  re-filters the _already-computed_ parquet and reruns backtest stats.
  Generating new labels requires the terminal command above.
- **Run Analysis is synchronous and ~30–100s per click.** The endpoint
  runs the full backtest with the active gates each time — no cache.
  This guarantees trade counts match `run_backtest`'s gated semantics
  exactly. See "Future work" below for the planned caching path.

## Future work — RA caching

A first attempt at caching the unfiltered backtest result and post-
filtering by toggle state was reverted because the daily-loss-limit
inside `run_backtest` interacts with the regime gates in a way that
produces materially different trade counts when the gate is inactive
vs active. The clean fix is to expose `run_backtest`'s **pre-DLL signal
stream** so the RA endpoint can re-simulate the DLL per toggle state on
a cached signal list. With that, the cache is invariant under toggle
changes and post-filtering reproduces gated semantics exactly. Tracked
as a future engineering task; relevant comment block lives at the top
of the `/run_regime_analysis` section in `server.py`.

---

# Backtest Dashboard Testing by Claude

Claude will be primarily responsible for testing strategies

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
| `B`                              | Toggle Backtest Settings panel                                                |
| `G`                              | General tab                                                                   |
| `A`                              | Advanced tab                                                                  |
| `Shift` + `1`–`4`                | Toggle sidebar section — Year / Month / Weeks / Day                           |
| `Shift` + `Delete` / `Backspace` | Delete (currently-selected version or run)                                    |
| `Enter`                          | Submit in the Add Version dialog and in in-place name/description edit fields |

---

# cTrader Integration

- **Language**: C# (not Python — C# runs natively inside cTrader as a cBot;
  the Python Open API is an external connection not suitable for prop firm use).
- **Code generation**: `cbot_templates.py` renders `FractalBot_v{N}.cs` on
  demand from the dashboard's **Create cBot** button. Logic changes go in
  `cbot_templates.py`, not the generated `.cs` (which is overwritten).

---

# Workflow Notes

- **Per-version notes** — the `/versions` page now exposes a free-form
  notes textarea per profile. Notes persist to `data/versions.json` under
  the version's `notes` field, auto-saving on blur via
  `POST /api/versions/<id>/notes`. This replaces the manually-maintained
  `RESULTS_LOG.md` and the dev-log button's `devlog.json` for per-version
  notes. Those two files still exist but can be retired.
- **Run-level metadata carries the active version** — each `new_run` written
  by `strategy_vN.py` includes `"active_version": <TARGET_VERSION>`. The
  sidebar's per-run label reads this in preference to the bucket name, so
  even if a run was historically bucketed under a different name it will
  still display correctly.
- **`RESULTS_LOG.md`** (legacy) — was manually maintained for notable-run
  notes. Now superseded by per-version notes.
- **`devlog.json`** (legacy) — free-form developer log accessed via the
  dev-log icon on the run bar. Superseded by per-version notes.
- **Git hygiene** — `.env`, `venv/`, `__pycache__/` are gitignored. Commit
  `data/regime_labels.parquet` if you want the regime labels reproducible
  across machines; otherwise it's regenerated on first labeler run. Commit
  `data/versions.json` to share profile state across machines.
- **Adding a new strategy version** — drop a `strategy_v3.py` alongside the
  existing versions, set `STRATEGY_VERSION=v3` for backtests, and update the
  v2/v3 sections of this doc.
- **Adding a new profile** — use the `/versions` page **Add Version**
  form; pick a base strategy module and (optionally) copy regime state from
  an existing profile. No code changes needed.
