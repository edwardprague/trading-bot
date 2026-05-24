#!/usr/bin/env python3
"""
regime_analysis.py — Market regime detection + analysis report
==============================================================
Detects market regimes from fractal sequence structure (not statistical
clustering). For every Williams N=2 fractal on GBPUSD 5m, classifies the
prevailing regime using the last LOOKBACK_FRACTALS same-kind highs and lows:

  • trending_up | trending_down   — both H and L progressing same direction
  • ranging                       — neither side directionally consistent
  • transitioning                 — conflicting / unconfirmed

Trending periods are then sub-classified fast / medium / slow by data-driven
terciles of pips-per-bar. Ranging periods are sub-classified narrow / medium
/ wide by data-driven terciles of width.

Outputs
-------
  • results/regime_analysis.html   — full HTML report
  • data/regime_labels.parquet     — per-fractal labels + thresholds metadata

Usage
-----
    source venv/bin/activate
    python3 regime_analysis.py

    # Override the date windows (May 2026). --start / --end set both the
    # labels parquet AND the rendered report; --labels-* and --report-*
    # let those windows diverge:
    python3 regime_analysis.py --start 2009-09-25 --end 2026-05-11
    python3 regime_analysis.py --start 2009-09-25 --end 2026-05-11 \\
        --report-start 2025-01-01 --report-end 2026-03-31

    # Skip the per-day chart generation loop (much faster for label-only runs):
    GENERATE_DAILY_CHARTS=false python3 regime_analysis.py --start 2009-09-25 --end 2026-05-11
"""

import argparse
import os
import sys
import io
import json
import math
import base64
import shutil
import webbrowser
import contextlib
from pathlib import Path
from datetime import datetime, timedelta

# Force the active strategy version BEFORE importing strategy_v2 so the
# module-level globals (TICKER, INTERVAL, RRR, filters) match the user's
# current short-only v2 configuration.
os.environ.setdefault("STRATEGY_VERSION", "v2")
os.environ.setdefault("INSTRUMENT", "GBPUSD")
os.environ.setdefault("INTERVAL", "5m")
os.environ.setdefault("TRADE_DIRECTION", "short_only")


# Seed ALLOWED_MACRO_REGIMES / ALLOWED_MICRO_REGIMES from the active version
# in data/versions.json BEFORE importing strategy_v2. Without this the static
# HTML this script generates uses strategy_v2's hardcoded defaults, while the
# page's Run Analysis button uses the active version's regime_state — which
# can differ once the user has toggled anything. The result is a visible
# stats discrepancy on page load (cached static HTML) vs. after clicking Run.
# Honouring the active version here keeps both paths in sync.
def _seed_regime_env_from_active_version():
    import json as _json
    from pathlib import Path as _Path
    versions_file = _Path(__file__).parent / "data" / "versions.json"
    try:
        with open(versions_file, "r", encoding="utf-8") as _f:
            data = _json.load(_f)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    active_id = data.get("active_version_id")
    for v in data.get("versions", []):
        if v.get("id") != active_id:
            continue
        rs = v.get("regime_state") or {}
        macro = rs.get("allowed_macro_regimes")
        micro = rs.get("allowed_micro_regimes")
        # setdefault — explicit env-var overrides at the shell level still win,
        # which matches how STRATEGY_VERSION / INSTRUMENT / etc. behave above.
        if macro is not None:
            os.environ.setdefault("ALLOWED_MACRO_REGIMES", ",".join(macro))
        if micro is not None:
            os.environ.setdefault("ALLOWED_MICRO_REGIMES", ",".join(micro))
        break

_seed_regime_env_from_active_version()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _PARQUET_ENGINE = "pyarrow"
except ImportError:
    pa = None
    pq = None
    try:
        # Fallback engine — supports the same kv-metadata feature via the
        # `custom_metadata` kwarg on fastparquet.write.
        import fastparquet  # noqa: F401
        _PARQUET_ENGINE = "fastparquet"
    except ImportError:
        print("ERROR: Parquet output requires pyarrow or fastparquet. "
              "Install with:  pip install pyarrow  (or)  pip install fastparquet")
        sys.exit(1)

import strategy_v2 as strat


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

START_DATE        = "2025-01-01"
END_DATE          = "2026-03-31"
LOOKBACK_FRACTALS = 4

# How many bars (5m) to keep before START_DATE so the rolling lookback is
# already populated by the time we reach the first in-range fractal.
WARMUP_BARS = 30

# Report window — defaults to the labels window above, but can be narrowed
# independently via --report-start / --report-end (May 2026). Use case:
# regenerate labels for the full historical range without forcing the HTML
# report to render every day of it. Resolved by main() from CLI args; the
# rest of the module reads these names so callers don't have to thread
# explicit dates through every signature.
REPORT_START_DATE = START_DATE
REPORT_END_DATE   = END_DATE

# When True, generate_daily_charts() runs and writes one PNG per trading day
# to results/regime_charts/. When False, the chart-generation loop is skipped
# entirely and any existing PNGs in results/regime_charts/ are reused — the
# daily-breakdown table's hover-preview links keep working for days whose
# chart was generated by a prior run, while days that lack a chart show the
# greyed-out "Chart not available" icon.
#
# Set GENERATE_DAILY_CHARTS=false (or 0) in the environment to override the
# default at run time without editing the file.
GENERATE_DAILY_CHARTS = os.environ.get("GENERATE_DAILY_CHARTS", "true").strip().lower() in ("true", "1", "yes", "on")

ROOT_DIR           = Path(__file__).resolve().parent
DATA_DIR           = ROOT_DIR / "data"
RESULTS_DIR        = ROOT_DIR / "results"
REGIME_CHARTS_DIR  = RESULTS_DIR / "regime_charts"
REPORT_PATH        = RESULTS_DIR / "regime_analysis.html"
LABELS_PATH        = DATA_DIR / "regime_labels.parquet"

PIP = 10000  # non-JPY pip multiplier

# Dark theme palette — matches strategy_v2 charts / style.css
BG_DARK  = "#1a1a2e"
PANEL_BG = "#10101c"
TEXT     = "#d0d0e8"
GRID     = "#444"
YELLOW   = "#ffd93d"
ACCENT   = "#4cc9f0"

# Regime color palette — matches the spec:
#   downtrend = blue family (deep → pale by speed)
#   uptrend   = red  family (deep → pale by speed)
#   ranging   = green family (deep → pale by width)
#   transitioning = grey
REGIME_COLORS = {
    "trending_fast_down":   "#0d47a1",
    "trending_medium_down": "#1976d2",
    "trending_slow_down":   "#90caf9",
    "trending_fast_up":     "#b71c1c",
    "trending_medium_up":   "#e53935",
    "trending_slow_up":     "#ef9a9a",
    "ranging_narrow":       "#1b5e20",
    "ranging_medium":       "#388e3c",
    "ranging_wide":         "#81c784",
    "transitioning":        "#616161",
}

# Display order (May 2026, per Edward): up-trending → ranging (wide→narrow)
# → down-trending → transitioning. This dict's key insertion order doubles as
# the canonical REGIME_ORDER below, so editing one edits the other. Matches
# the Discovery trial page's micro ordering.
REGIME_DISPLAY = {
    "trending_fast_up":     "Trending — Fast Up",
    "trending_medium_up":   "Trending — Medium Up",
    "trending_slow_up":     "Trending — Slow Up",
    "ranging_wide":         "Ranging — Wide",
    "ranging_medium":       "Ranging — Medium",
    "ranging_narrow":       "Ranging — Narrow",
    "trending_fast_down":   "Trending — Fast Down",
    "trending_medium_down": "Trending — Medium Down",
    "trending_slow_down":   "Trending — Slow Down",
    "transitioning":        "Transitioning",
}

# Order used in tables / cards / timeline keys
REGIME_ORDER = list(REGIME_DISPLAY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Macro regime configuration
# ─────────────────────────────────────────────────────────────────────────────
# Tunable thresholds for the day-level macro classifier. All in pips.

LARGE_DISPLACEMENT_PIPS = 30   # |close − open| ≥ this → "large" displacement
SMALL_DISPLACEMENT_PIPS = 15   # |close − open| <  this → flat day (regardless of intraday)
EMA_MACRO_PERIOD        = 40   # span for the within-day EMA used for slope
N18_LOOKBACK            = 18   # bars each side for Williams N=18 fractal detection

# Display order (May 2026, per Edward): most-bullish at top, most-bearish
# at bottom. Matches the Discovery trial page's macro ordering. Used wherever
# the RA page iterates macros for display (perf table, summary cards,
# timeline legend, filter chips, toggle pills, etc.).
MACRO_REGIME_ORDER = [
    "staircase_up", "strong_up", "flat", "staircase_down", "strong_down",
]

MACRO_REGIME_DISPLAY = {
    "strong_down":    "Strong Down",
    "staircase_down": "Staircase Down",
    "flat":           "Flat",
    "staircase_up":   "Staircase Up",
    "strong_up":      "Strong Up",
}

# Color scheme per spec: deep→pale blue for down, deep→pale red for up, grey flat.
MACRO_REGIME_COLORS = {
    "strong_down":    "#0d47a1",
    "staircase_down": "#1976d2",
    "flat":           "#616161",
    "staircase_up":   "#e53935",
    "strong_up":      "#b71c1c",
}


def macro_class(label):
    """CSS class for a macro regime label (e.g. 'strong_down' → 'macro-color-strong-down')."""
    if not label:
        return "macro-color-flat"
    return "macro-color-" + label.replace("_", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Macro regime filter — exclude trades on certain macro-regime days from
# performance statistics. Filtered days still appear in the daily breakdown
# table and regime timeline (with a visual indicator) so the user retains
# the full context; only the aggregate stats, performance tables, win-rate
# and profit-factor calculations are affected.
#
# Use the human-readable display names ("Flat", "Staircase Up", "Strong Up").
# Internal keys ("flat", "staircase_up", "strong_up") are also accepted.
# ─────────────────────────────────────────────────────────────────────────────

BLOCKED_MACRO_REGIMES = ["Flat", "Staircase Up", "Strong Up"]
APPLY_MACRO_FILTER    = True

# Default blocked micro-regime keys (internal underscore form). The interactive
# toggle panel uses this list to set its initial state on page load.
BLOCKED_MICRO_REGIMES = [
    "trending_fast_down", "trending_medium_down", "trending_slow_down",
    "trending_fast_up",   "trending_medium_up",   "trending_slow_up",
    "ranging_narrow",     "transitioning",
]


def _blocked_macro_keys():
    """Convert BLOCKED_MACRO_REGIMES to the internal label-key set
    (e.g. 'Strong Up' → 'strong_up'). Empty set when filter is disabled."""
    if not APPLY_MACRO_FILTER:
        return set()
    display_to_key = {v: k for k, v in MACRO_REGIME_DISPLAY.items()}
    keys = set()
    for name in BLOCKED_MACRO_REGIMES:
        if name in display_to_key:
            keys.add(display_to_key[name])
        elif name in MACRO_REGIME_ORDER:
            keys.add(name)
    return keys


def _filter_trades_by_macro(trades_df, macro):
    """Drop trades whose entry day falls on a blocked macro regime.
    Returns (kept_trades_df, n_excluded, blocked_keys_set). If the filter is
    off or no trades match, the original frame is returned unmodified."""
    blocked = _blocked_macro_keys()
    if not blocked or trades_df.empty:
        return trades_df.copy(), 0, blocked
    td = trades_df.copy()
    ts = pd.to_datetime(td["entry_ts"])
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    entry_day = ts_utc.dt.strftime("%Y-%m-%d")
    macro_label = entry_day.map(lambda d: macro.get(d, {}).get("label"))
    blocked_mask = macro_label.isin(blocked)
    n_excluded = int(blocked_mask.sum())
    kept = td[~blocked_mask].reset_index(drop=True)
    return kept, n_excluded, blocked


def _compute_perf_df(trades_df):
    """Per-regime aggregate metrics — same shape produced by stage3_trade_outcomes,
    but rebuildable from any trades_df subset (e.g. after macro filtering)."""
    perf = []
    for label in REGIME_ORDER:
        sub = trades_df[trades_df["regime"] == label] if not trades_df.empty else pd.DataFrame()
        n = len(sub)
        if n == 0:
            perf.append({"regime": label, "trades": 0, "wins": 0, "win_rate": np.nan,
                         "profit_factor": np.nan, "avg_pnl": np.nan, "total_pnl": 0.0})
            continue
        wins = int(sub["win"].sum())
        win_rate = wins / n * 100
        gross_win  = float(sub.loc[sub["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-sub.loc[sub["pnl"] < 0, "pnl"].sum())
        pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        perf.append({
            "regime":   label, "trades": n, "wins": wins, "win_rate": win_rate,
            "profit_factor": pf, "avg_pnl": float(sub["pnl"].mean()),
            "total_pnl": float(sub["pnl"].sum()),
        })
    return pd.DataFrame(perf)


def _row_stats(sub_df):
    """Per-row stats for a perf table (Trades / Wins / Win-rate / PF / Avg / Total).
    Returns a dict with rendered HTML cell strings ready to drop into a row,
    sharing the same coloring rules used by the macro and micro perf tables.
    Used for both 'actual' rows (fired trades) and 'counterfactual' rows
    (fired + blocked-signal scan_outcomes on a locked regime).
    """
    n = int(len(sub_df))
    if n == 0:
        return {
            "n":           0,
            "wins_str":    "—",
            "wr_cell":     "<td class='regime-dim'>—</td>",
            "pf_cell":     "<td class='regime-dim'>—</td>",
            "avg_cell":    "<td class='regime-dim'>—</td>",
            "tot_inner":   "—",
            "tot_pnl":     0.0,
        }
    wins = int(sub_df["win"].sum())
    wr   = wins / n * 100
    gw   = float(sub_df.loc[sub_df["pnl"] > 0, "pnl"].sum())
    gl   = float(-sub_df.loc[sub_df["pnl"] < 0, "pnl"].sum())
    pf   = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    avg  = float(sub_df["pnl"].mean())
    tot  = float(sub_df["pnl"].sum())
    # May 2026: use BD's .pos / .neg text-colour classes everywhere so all
    # three pages share a single colouring convention. Thresholds match the
    # user-stated rules:
    #   • Win Rate ≥ 50 → green, else red
    #   • PF       ≥ 1.5 → green (∞ counts as ≥ 1.5), else red
    #   • Total P&L > 0  → green, < 0 → red, 0 → no class (default text)
    # The old win-good/win-bad background fill and the regime-pnl-pos/neg
    # variants were retired in favour of this.
    wr_cls      = "pos" if wr >= 50 else "neg"
    pf_cls      = "pos" if (math.isinf(pf) or pf >= 1.5) else "neg"
    tot_pnl_cls = "pos" if tot > 0 else ("neg" if tot < 0 else "")
    return {
        "n":         n,
        "wins_str":  str(wins),
        "wr_cell":   f"<td class='{wr_cls}'>{_fmt_pct(wr)}</td>",
        "pf_cell":   f"<td class='{pf_cls}'>{_fmt_pf(pf)}</td>",
        "avg_cell":  f"<td>{_fmt_money(avg)}</td>",
        "tot_inner": f"<span class='{tot_pnl_cls}'>{_fmt_money(tot)}</span>",
        "tot_pnl":   tot,
    }


def _compute_aggregate_stats(trades_df):
    """Top-line stats for the report header: total/wins/losses/win-rate/PF/total P&L."""
    n = int(len(trades_df))
    if n == 0:
        return {"total": 0, "wins": 0, "losses": 0,
                "win_rate": float("nan"), "pf": float("nan"), "total_pnl": 0.0}
    wins = int(trades_df["win"].sum())
    losses = n - wins
    win_rate = wins / n * 100
    gw = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
    gl = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
    pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    return {"total": n, "wins": wins, "losses": losses,
            "win_rate": win_rate, "pf": pf,
            "total_pnl": float(trades_df["pnl"].sum())}


# ─────────────────────────────────────────────────────────────────────────────
# Reusable section renderers — called by both build_report (static page render)
# and the server's /run_regime_analysis endpoint (interactive update path).
# ─────────────────────────────────────────────────────────────────────────────

def build_stats_bar_html(agg_stats, filter_state_label, filter_state_class):
    """Render the top-of-report summary stats bar (6 cards + filter label)."""
    _wr = agg_stats["win_rate"]
    if pd.isna(_wr) or agg_stats["total"] == 0:
        wr_cls = "regime-stat-neutral"
    elif _wr > 55:
        wr_cls = "regime-stat-good"
    elif _wr < 45:
        wr_cls = "regime-stat-bad"
    else:
        wr_cls = "regime-stat-neutral"
    pnl_total = agg_stats["total_pnl"]
    if agg_stats["total"] == 0:
        pnl_cls = "regime-stat-neutral"
    elif pnl_total > 0:
        pnl_cls = "regime-stat-good"
    elif pnl_total < 0:
        pnl_cls = "regime-stat-bad"
    else:
        pnl_cls = "regime-stat-neutral"
    return f"""
      <div class="regime-stats-cards">
        <div class="regime-stat-card">
          <span class="regime-stat-label">Total Trades</span>
          <span class="regime-stat-value">{agg_stats['total']}</span>
        </div>
        <div class="regime-stat-card">
          <span class="regime-stat-label">Wins</span>
          <span class="regime-stat-value">{agg_stats['wins']}</span>
        </div>
        <div class="regime-stat-card">
          <span class="regime-stat-label">Losses</span>
          <span class="regime-stat-value">{agg_stats['losses']}</span>
        </div>
        <div class="regime-stat-card">
          <span class="regime-stat-label">Win Rate</span>
          <span class="regime-stat-value {wr_cls}">{_fmt_pct(_wr)}</span>
        </div>
        <div class="regime-stat-card">
          <span class="regime-stat-label">Profit Factor</span>
          <span class="regime-stat-value">{_fmt_pf(agg_stats['pf'])}</span>
        </div>
        <div class="regime-stat-card">
          <span class="regime-stat-label">Total P&amp;L</span>
          <span class="regime-stat-value {pnl_cls}">{_fmt_money(pnl_total)}</span>
        </div>
      </div>
    """


def build_perf_table_html(perf_df, regime_count, blocked_micro_keys=None,
                           trades_df=None, blocked_signals_df=None,
                           allowed_macro_keys=None, unfiltered_trades_df=None):
    """Render the 'Micro regime performance' table.

    For ALLOWED regimes (not in `blocked_micro_keys`): stats come from
    `perf_df` (built earlier from the fired trades) — these match what the
    active strategy actually did.

    For LOCKED regimes: stats are counterfactual. Sources, in preference
    order:
      • `unfiltered_trades_df` — the FULL trade set from a backtest with
        the regime gate disabled (Task 1's cached pipeline). Most
        accurate because the other gates were still applied.
      • `trades_df + blocked_signals_df` (legacy) — combines active
        strategy trades with `reason='micro_regime'` rejections from a
        gate-active backtest. `allowed_macro_keys` further restricts the
        combined set to days whose macro regime is allowed.

    `trades_df` / `blocked_signals_df` are optional; if absent, locked rows
    fall back to em-dashes.
    """
    blocked_micro_keys = blocked_micro_keys or set()
    if trades_df is None: trades_df = pd.DataFrame()
    if blocked_signals_df is None: blocked_signals_df = pd.DataFrame()
    have_unfiltered = (unfiltered_trades_df is not None
                       and not unfiltered_trades_df.empty
                       and "regime" in unfiltered_trades_df.columns)

    counterfactual_tip = (
        "Counterfactual — would-be outcome of signals that were blocked "
        "specifically by the micro regime filter on this regime. Win rate "
        "and trade count are accurate; aggregate P&L is approximate due "
        "to position-sizing and daily-loss-limit interactions if these "
        "had actually fired."
    )

    # Only count signals with reason='micro_regime' for the counterfactual —
    # the strategy applies filters in order (EMA → daily-loss → time → macro
    # → micro), so signals that failed an earlier filter would still fail it
    # even if this micro regime were unlocked. Including all blocked signals
    # over-counts by ~10× because EMA / time / macro filters block far more
    # signals than the micro filter does.
    if not blocked_signals_df.empty and "reason" in blocked_signals_df.columns:
        micro_only_blocked = blocked_signals_df[blocked_signals_df["reason"] == "micro_regime"]
    else:
        micro_only_blocked = blocked_signals_df.iloc[0:0]

    # Build the combined frame once for reuse across locked rows.
    if not trades_df.empty and not micro_only_blocked.empty:
        combined = pd.concat([trades_df, micro_only_blocked], ignore_index=True)
    elif not trades_df.empty:
        combined = trades_df
    else:
        combined = micro_only_blocked

    if allowed_macro_keys is not None and not combined.empty and "macro_label" in combined.columns:
        combined = combined[combined["macro_label"].isin(allowed_macro_keys)]

    perf_rows = []
    for _, r in perf_df.iterrows():
        period_count = int(regime_count.get(r["regime"], 0))
        if period_count == 0:
            continue
        is_blocked = r["regime"] in blocked_micro_keys

        # Locked rows: prefer the unfiltered-trades source (Task 1's
        # cached pipeline runs the backtest with the micro gate disabled
        # so every signal that passed other gates lands here). Falls back
        # to the legacy combined frame if no unfiltered source provided.
        if is_blocked and have_unfiltered:
            sub = unfiltered_trades_df[unfiltered_trades_df["regime"] == r["regime"]]
            if allowed_macro_keys is not None and "macro_label" in sub.columns:
                sub = sub[sub["macro_label"].isin(allowed_macro_keys)]
            stats = _row_stats(sub)
            n_trades   = stats["n"]
            wins_cell  = stats["wins_str"]
            wr_cell    = stats["wr_cell"]
            pf_cell    = stats["pf_cell"]
            avg_cell   = stats["avg_cell"]
            total_cell = (
                f"<td class='regime-pnl-counterfactual' "
                f"title='{counterfactual_tip}'>{stats['tot_inner']}</td>"
            )
        elif is_blocked and not combined.empty and "regime" in combined.columns:
            sub = combined[combined["regime"] == r["regime"]]
            stats = _row_stats(sub)
            n_trades   = stats["n"]
            wins_cell  = stats["wins_str"]
            wr_cell    = stats["wr_cell"]
            pf_cell    = stats["pf_cell"]
            avg_cell   = stats["avg_cell"]
            total_cell = (
                f"<td class='regime-pnl-counterfactual' "
                f"title='{counterfactual_tip}'>{stats['tot_inner']}</td>"
            )
        else:
            n_trades = int(r["trades"])
            if n_trades > 0:
                # May 2026: BD-style binary text colours. Same thresholds as
                # _row_stats (the blocked-row helper), keeping macro/micro/non-
                # blocked/blocked rows visually identical across the table.
                pf_val      = r["profit_factor"]
                pf_is_good  = (pf_val is not None
                               and (math.isinf(pf_val) or pf_val >= 1.5))
                wr_cls      = "pos" if r["win_rate"] >= 50 else "neg"
                pf_cls      = "pos" if pf_is_good else "neg"
                tot_pnl     = float(r["total_pnl"])
                tot_pnl_cls = "pos" if tot_pnl > 0 else ("neg" if tot_pnl < 0 else "")
                wins_cell   = str(int(r["wins"]))
                wr_cell     = f"<td class='{wr_cls}'>{_fmt_pct(r['win_rate'])}</td>"
                pf_cell     = f"<td class='{pf_cls}'>{_fmt_pf(pf_val)}</td>"
                avg_cell    = f"<td>{_fmt_money(r['avg_pnl'])}</td>"
                total_cell  = f"<td><span class='{tot_pnl_cls}'>{_fmt_money(tot_pnl)}</span></td>"
            else:
                wins_cell   = "—"
                wr_cell     = "<td class='regime-dim'>—</td>"
                pf_cell     = "<td class='regime-dim'>—</td>"
                avg_cell    = "<td class='regime-dim'>—</td>"
                total_cell  = "<td class='regime-dim'>—</td>"

        if is_blocked:
            lock_html = (
                "<span class='regime-blocked-lock' "
                "title='Excluded from active strategy entries by the micro regime filter' "
                "aria-label='Filtered out'>"
                "<span class='material-symbols-outlined'>lock</span>"
                "</span>"
            )
        else:
            lock_html = ""
        row_class = "regime-day-blocked" if is_blocked else ""

        perf_rows.append(
            f"<tr class='{row_class}'>"
            f"<td><span class='regime-badge {regime_class(r['regime'])}'>"
            f"{REGIME_DISPLAY[r['regime']]}</span>"
            f"{lock_html}</td>"
            f"<td>{period_count}</td>"
            f"<td>{n_trades}</td>"
            f"<td>{wins_cell}</td>"
            f"{wr_cell}{pf_cell}{avg_cell}{total_cell}"
            f"</tr>"
        )
    body = ''.join(perf_rows) or "<tr><td colspan=8 class=regime-dim>No regimes observed in the requested range.</td></tr>"
    return f"""
      <table class='regime-table regime-perf-table'>
        <thead><tr>
          <th>Regime</th><th>Periods</th><th>Trades</th><th>Wins</th><th>Win rate</th>
          <th>Profit factor</th><th>Avg P&amp;L</th><th>Total P&amp;L</th>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
      <p class="regime-dim regime-small regime-counterfactual-note">
        Locked-regime rows show counterfactual stats — the would-be outcome
        of signals that were filtered out. Win rate and trade counts are
        accurate; aggregate P&amp;L is approximate.
      </p>
    """


def build_timeline_section_html(in_range_periods, macro, trades_per_day,
                                start_date, end_date, regime_count):
    """Render the full Regime Timeline section body — macro + micro strips,
    week labels along the bottom, and the two legends.
    """
    timeline = build_daily_timeline(in_range_periods, start_date, end_date)

    macro_tl_cells = []
    micro_tl_cells = []
    for entry in timeline:
        day_count = trades_per_day.get(entry["date"], 0)
        count_html = (
            f"<span class='regime-tl-cell-count'>{day_count}</span>"
            if day_count > 0 else ""
        )
        trade_suffix = (
            f" · {day_count} trade{'s' if day_count != 1 else ''}"
            if day_count > 0 else ""
        )
        title = f"{entry['date']} — {entry['label']}{trade_suffix}"
        micro_tl_cells.append(
            f"<div class='regime-tl-cell {regime_class(entry['regime'])}' "
            f"title='{title}'>{count_html}</div>"
        )
        mac_label = macro.get(entry["date"], {}).get("label") if macro else None
        mac_title = (
            f"{entry['date']} — "
            f"{MACRO_REGIME_DISPLAY.get(mac_label, 'No data')}{trade_suffix}"
        )
        macro_tl_cells.append(
            f"<div class='regime-tl-cell {macro_class(mac_label)}' "
            f"title='{mac_title}'>{count_html}</div>"
        )

    legend_items = []
    for label in REGIME_ORDER:
        if label in regime_count:
            legend_items.append(
                f"<span class='regime-chip'>"
                f"<span class='regime-swatch {regime_class(label)}'></span>"
                f"{REGIME_DISPLAY[label]}</span>"
            )
    legend_html = "".join(legend_items)

    macro_observed = {d["label"] for d in macro.values()} if macro else set()
    macro_legend_items = []
    for label in MACRO_REGIME_ORDER:
        if label in macro_observed:
            macro_legend_items.append(
                f"<span class='regime-chip'>"
                f"<span class='regime-swatch {macro_class(label)}'></span>"
                f"{MACRO_REGIME_DISPLAY[label]}</span>"
            )
    macro_legend_html = "".join(macro_legend_items)

    return f"""
      <h2>Regime timeline</h2>
      <div class="regime-tl-row">
        <span class="regime-tl-row-label">Macro</span>
        <div class="regime-tl-strip">{''.join(macro_tl_cells)}</div>
      </div>
      <div class="regime-tl-row">
        <span class="regime-tl-row-label">Micro</span>
        <div class="regime-tl-strip">{''.join(micro_tl_cells)}</div>
      </div>
      <div class="regime-legend"><strong class="regime-dim regime-small">Macro:</strong> {macro_legend_html}</div>
      <div class="regime-legend"><strong class="regime-dim regime-small">Micro:</strong> {legend_html}</div>
    """


def compute_filter_label(blocked_macro_keys, total_trades, n_excluded):
    """Return (filter_state_label, filter_state_class, macro_filter_note,
    macro_table_filter_note) for the current filter configuration."""
    blocked_display_names = [
        MACRO_REGIME_DISPLAY.get(k, k) for k in
        sorted(blocked_macro_keys,
               key=lambda x: MACRO_REGIME_ORDER.index(x)
               if x in MACRO_REGIME_ORDER else 99)
    ]
    if blocked_display_names:
        filter_state_label = "Filtered — excluding " + ", ".join(blocked_display_names) + " days"
        filter_state_class = "regime-filter-on"
    else:
        filter_state_label = "Unfiltered — all trades"
        filter_state_class = "regime-filter-off"

    if blocked_display_names and total_trades > 0:
        _pct_excl = n_excluded / total_trades * 100
        macro_filter_note = (
            f"<p class='regime-dim regime-small regime-perf-note regime-macro-filter-note'>"
            f"<span class='material-symbols-outlined regime-macro-filter-note-icon'>filter_alt</span>"
            f"{n_excluded} of {total_trades} trades ({_pct_excl:.1f}%) "
            f"were excluded by the macro filter. Blocked regimes: "
            f"<strong>{', '.join(blocked_display_names)}</strong>. "
            f"Statistics in this table reflect only trades from allowed macro regime days."
            f"</p>"
        )
        macro_table_filter_note = macro_filter_note
    elif blocked_display_names:
        macro_filter_note = (
            f"<p class='regime-dim regime-small regime-perf-note regime-macro-filter-note'>"
            f"<span class='material-symbols-outlined regime-macro-filter-note-icon'>filter_alt</span>"
            f"Macro filter is active. Blocked regimes: "
            f"<strong>{', '.join(blocked_display_names)}</strong>."
            f"</p>"
        )
        macro_table_filter_note = macro_filter_note
    else:
        macro_filter_note = ""
        macro_table_filter_note = ""

    return filter_state_label, filter_state_class, macro_filter_note, macro_table_filter_note


def compute_trades_per_day(trades_df):
    """{'YYYY-MM-DD': count} from trades_df['entry_ts']. Used by timeline overlay."""
    trades_per_day = {}
    if trades_df.empty:
        return trades_per_day
    _t = pd.to_datetime(trades_df["entry_ts"])
    _t = _t.dt.tz_convert("UTC") if _t.dt.tz is not None else _t.dt.tz_localize("UTC")
    for d in _t.dt.strftime("%Y-%m-%d"):
        trades_per_day[d] = trades_per_day.get(d, 0) + 1
    return trades_per_day


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Fractal extraction with rolling lookback metrics
# ─────────────────────────────────────────────────────────────────────────────

def stage1_extract_fractals():
    print("Stage 1: Extracting fractals from price data...")

    df = strat.fetch_data(
        strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
        start_date=START_DATE, end_date=END_DATE,
    )
    df = strat.add_indicators(df)

    # Window the working frame: warmup bars before START + 7 days after END
    # (the 7-day tail mirrors strategy_v2's exit-resolution buffer).
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(END_DATE,   tz="UTC") + pd.Timedelta(days=1)
    dts      = pd.to_datetime(df["Datetime"])
    dts_utc  = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts.dt.tz_localize("UTC")
    # 30 bars * 5 min = 150 min; we conservatively use 1d warmup so the
    # rolling lookback always has data even after fractal sparsity.
    bt_start = start_ts - pd.Timedelta(minutes=WARMUP_BARS * 5)
    bt_start = min(bt_start, start_ts - pd.Timedelta(days=1))
    bt_end   = end_ts + pd.Timedelta(days=7)
    df = df[(dts_utc >= bt_start) & (dts_utc < bt_end)].reset_index(drop=True)

    highs = df["High"].values
    lows  = df["Low"].values
    atrs  = df["atr14"].values
    adxs  = df["adx"].values

    rolling_H = []   # last LOOKBACK_FRACTALS dicts, oldest first
    rolling_L = []

    rows = []
    last_H_price = None
    last_H_bar   = None
    last_L_price = None
    last_L_bar   = None

    for fi in range(2, len(df) - 2):
        fh, fl = highs[fi], lows[fi]
        is_ph = (fh > highs[fi-1] and fh > highs[fi-2]
                 and fh > highs[fi+1] and fh > highs[fi+2])
        is_pl = (fl < lows[fi-1]  and fl < lows[fi-2]
                 and fl < lows[fi+1]  and fl < lows[fi+2])
        if not (is_ph or is_pl):
            continue

        ts_raw = pd.to_datetime(df["Datetime"].iloc[fi])
        ts_utc = ts_raw.tz_convert("UTC") if ts_raw.tzinfo else ts_raw.tz_localize("UTC")

        if fi + 2 < len(df):
            entry_ts_raw = pd.to_datetime(df["Datetime"].iloc[fi + 2])
            entry_ts_utc = entry_ts_raw.tz_convert("UTC") if entry_ts_raw.tzinfo else entry_ts_raw.tz_localize("UTC")
            entry_hour = int(entry_ts_utc.hour)
        else:
            entry_hour = int(ts_utc.hour)

        events = []
        if is_ph: events.append(("H", float(fh)))
        if is_pl: events.append(("L", float(fl)))

        for kind, price in events:
            if kind == "H":
                v_pips = abs(price - last_H_price) * PIP if last_H_price is not None else np.nan
                h_bars = float(fi - last_H_bar)         if last_H_bar   is not None else np.nan
            else:
                v_pips = abs(price - last_L_price) * PIP if last_L_price is not None else np.nan
                h_bars = float(fi - last_L_bar)         if last_L_bar   is not None else np.nan

            row = {
                "timestamp":    ts_utc,
                "fractal_bar":  fi,
                "kind":         kind,
                "price":        price,
                "adx":          float(adxs[fi]),
                "atr_pips":     float(atrs[fi]) * PIP,
                "v_dist_pips":  v_pips,
                "h_dist_bars":  h_bars,
                "entry_hour":   entry_hour,
            }

            # Update rolling list AFTER recording row context so this fractal's
            # row uses the *prior* same-kind rolling state.
            if kind == "H":
                rolling_H.append({"bar": fi, "price": price})
                if len(rolling_H) > LOOKBACK_FRACTALS:
                    rolling_H = rolling_H[-LOOKBACK_FRACTALS:]
                last_H_bar, last_H_price = fi, price
            else:
                rolling_L.append({"bar": fi, "price": price})
                if len(rolling_L) > LOOKBACK_FRACTALS:
                    rolling_L = rolling_L[-LOOKBACK_FRACTALS:]
                last_L_bar, last_L_price = fi, price

            # Snapshot of rolling lookback at this moment (after the update).
            # Stage 2 uses these to classify regime.
            row["_rolling_H"] = list(rolling_H)
            row["_rolling_L"] = list(rolling_L)
            rows.append(row)

    fractal_df = pd.DataFrame(rows)
    print(f"Stage 1 complete: {len(fractal_df)} fractals detected")
    return fractal_df, df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Regime classification (state machine + tercile refinement)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_raw(rolling_H, rolling_L):
    """Coarse regime classification from the rolling lookback alone.
    Returns one of: 'trending_up', 'trending_down', 'ranging', 'transitioning'.

    Interpretation of the spec ("at least three out of four consecutive pairs"):
    we require ceil((N-1) * 0.75) pairs to be sequentially in the same
    direction to call a side "consistent." With LOOKBACK_FRACTALS=4 that is
    ceil(3 * 0.75) = 3 → all 3 consecutive pairs must agree. With N=5 it is
    3 of 4 (matching the user's phrasing literally).
    """
    if len(rolling_H) < 2 or len(rolling_L) < 2:
        return "transitioning"

    pairs_H = len(rolling_H) - 1
    pairs_L = len(rolling_L) - 1
    threshold = max(2, math.ceil(pairs_H * 0.75))

    H_down = sum(1 for i in range(1, len(rolling_H))
                 if rolling_H[i]["price"] < rolling_H[i-1]["price"])
    H_up   = sum(1 for i in range(1, len(rolling_H))
                 if rolling_H[i]["price"] > rolling_H[i-1]["price"])
    L_down = sum(1 for i in range(1, len(rolling_L))
                 if rolling_L[i]["price"] < rolling_L[i-1]["price"])
    L_up   = sum(1 for i in range(1, len(rolling_L))
                 if rolling_L[i]["price"] > rolling_L[i-1]["price"])

    H_is_down = H_down >= threshold
    H_is_up   = H_up   >= threshold
    L_is_down = L_down >= threshold
    L_is_up   = L_up   >= threshold

    H_dir = "down" if H_is_down else ("up" if H_is_up else None)
    L_dir = "down" if L_is_down else ("up" if L_is_up else None)

    if H_dir == "down" and L_dir == "down":
        return "trending_down"
    if H_dir == "up" and L_dir == "up":
        return "trending_up"
    if (H_dir is not None and L_dir is not None) and H_dir != L_dir:
        return "transitioning"
    if H_dir is None and L_dir is None:
        return "ranging"
    # Only one side directional, the other not — conflicting / unconfirmed
    return "transitioning"


def _period_pips_per_bar(rolling_H, rolling_L):
    """Average v-distance / average h-distance across same-kind pairs.
    Combines the H-pair rate and L-pair rate (mean of the two)."""
    def rate(lst):
        if len(lst) < 2: return np.nan
        v = np.mean([abs(lst[i]["price"] - lst[i-1]["price"]) * PIP
                     for i in range(1, len(lst))])
        h = np.mean([lst[i]["bar"] - lst[i-1]["bar"]
                     for i in range(1, len(lst))])
        return v / h if h > 0 else np.nan
    rH = rate(rolling_H)
    rL = rate(rolling_L)
    vals = [x for x in (rH, rL) if not pd.isna(x)]
    return float(np.mean(vals)) if vals else np.nan


def _period_width_choppiness(rolling_H, rolling_L):
    """Width = (mean H − mean L) in pips. Choppiness = std(H prices)/width."""
    if not rolling_H or not rolling_L:
        return np.nan, np.nan
    h_prices = np.array([x["price"] for x in rolling_H])
    l_prices = np.array([x["price"] for x in rolling_L])
    width_price = h_prices.mean() - l_prices.mean()
    if width_price <= 0:
        return np.nan, np.nan
    width_pips = width_price * PIP
    chop_raw = float(h_prices.std(ddof=0) / width_price) if width_price > 0 else np.nan
    chop = float(min(1.0, max(0.0, chop_raw))) if not pd.isna(chop_raw) else np.nan
    return float(width_pips), chop


def _coarse_label(raw):
    """Map a raw 4-way classification to its 'coarse' regime key. Trending
    splits into up/down; ranging and transitioning remain as-is. This is the
    granularity periods are grouped at — sub-labels (fast/medium/slow etc.)
    are applied to the *period*, not per fractal."""
    return raw  # already in {trending_up, trending_down, ranging, transitioning}


def stage2_classify(fractal_df):
    print("Stage 2: Classifying market regimes...")

    # ── Pass 1: per-fractal raw classification + state machine ──────────────
    raws       = []
    pips_bar   = []
    widths     = []
    choppiness = []

    state              = "transitioning"
    state_start_bar    = None
    state_start_ts     = None
    last_H_class       = []   # last 2 raw classifications observed at H-fractals
    last_L_class       = []

    coarse_labels   = []
    regime_starts   = []
    candles_active  = []

    # Pull the heavy/object-typed columns out as numpy arrays / python lists
    # once so the hot loop avoids the per-row Series construction overhead of
    # iterrows. Over 16 years of data (~300K fractals) this turns a multi-
    # minute classification loop into ~25 seconds.
    #
    # IMPORTANT: use tolist() (not .values) for the timestamp column so we
    # keep tz-aware Timestamp objects rather than collapsing to tz-naive
    # numpy datetime64 — persist_labels relies on the tz survival downstream.
    rH_arr   = fractal_df["_rolling_H"].values
    rL_arr   = fractal_df["_rolling_L"].values
    kind_arr = fractal_df["kind"].values
    fb_arr   = fractal_df["fractal_bar"].values
    ts_arr   = fractal_df["timestamp"].tolist()
    N        = len(fractal_df)

    for i in range(N):
        rH = rH_arr[i]
        rL = rL_arr[i]
        raw = _classify_raw(rH, rL)
        raws.append(raw)
        pips_bar.append(_period_pips_per_bar(rH, rL))
        w, c = _period_width_choppiness(rH, rL)
        widths.append(w)
        choppiness.append(c)

        # Track per-kind class history for the confirmation rule
        if kind_arr[i] == "H":
            last_H_class.append(raw)
            if len(last_H_class) > 2: last_H_class = last_H_class[-2:]
        else:
            last_L_class.append(raw)
            if len(last_L_class) > 2: last_L_class = last_L_class[-2:]

        # Confirmation rule: state updates when EITHER last 2 H-fractals OR
        # last 2 L-fractals agree on a new, non-current, non-transitioning regime.
        new_committed = None
        for hist in (last_H_class, last_L_class):
            if len(hist) == 2 and hist[0] == hist[1] and hist[0] != "transitioning" \
                    and hist[0] != state:
                new_committed = hist[0]
                break

        if new_committed is not None:
            state = new_committed
            state_start_bar = int(fb_arr[i])
            state_start_ts  = ts_arr[i]
            committed_label = state
        else:
            if state_start_bar is None:
                state_start_bar = int(fb_arr[i])
                state_start_ts  = ts_arr[i]
            # If raw matches committed state, stay; otherwise we are mid-transition.
            if raw == state:
                committed_label = state
            else:
                # We don't move the committed state — we *report* transitioning
                # until confirmation.  Note: this also covers state==transitioning
                # initially, so early fractals get labeled transitioning.
                if state == "transitioning":
                    committed_label = "transitioning"
                else:
                    committed_label = "transitioning"
                # Reset state_start so reported transitioning periods reflect the
                # bar a transition was first observed.
                if not coarse_labels or coarse_labels[-1] != "transitioning":
                    state_start_bar = int(fb_arr[i])
                    state_start_ts  = ts_arr[i]
                    state = "transitioning"
                    # Clear class history so confirmation only counts the new
                    # post-transition signals.
                    if kind_arr[i] == "H": last_H_class = [raw]
                    else:                  last_L_class = [raw]

        coarse_labels.append(committed_label)
        regime_starts.append(state_start_ts)
        candles_active.append(int(fb_arr[i]) - int(state_start_bar))

    fractal_df = fractal_df.copy()
    fractal_df["raw_class"]       = raws
    fractal_df["pips_per_bar"]    = pips_bar
    fractal_df["width_pips"]      = widths
    fractal_df["choppiness"]      = choppiness
    fractal_df["coarse_label"]    = coarse_labels
    fractal_df["regime_start_ts"] = regime_starts
    fractal_df["candles_active"]  = candles_active

    # ── Group consecutive same-coarse-label fractals into periods ───────────
    # Numpy-direct loop over the now-populated columns — avoids a second
    # iterrows pass over the full fractal table. Timestamps are pulled via
    # tolist() so the tz-aware Timestamp objects survive (numpy .values
    # would strip the tz on this column).
    cl_arr  = fractal_df["coarse_label"].values
    ts_arr  = fractal_df["timestamp"].tolist()
    fb_arr  = fractal_df["fractal_bar"].values
    idx_arr = fractal_df.index.values

    periods = []
    cur = None
    for i in range(len(fractal_df)):
        label_i = cl_arr[i]
        if cur is None or label_i != cur["label"]:
            if cur is not None:
                periods.append(cur)
            cur = {
                "label":        label_i,
                "start_idx":    idx_arr[i],
                "end_idx":      idx_arr[i],
                "start_ts":     ts_arr[i],
                "end_ts":       ts_arr[i],
                "start_bar":    int(fb_arr[i]),
                "end_bar":      int(fb_arr[i]),
                "fractal_idxs": [idx_arr[i]],
            }
        else:
            cur["end_idx"]      = idx_arr[i]
            cur["end_ts"]       = ts_arr[i]
            cur["end_bar"]      = int(fb_arr[i])
            cur["fractal_idxs"].append(idx_arr[i])
    if cur is not None:
        periods.append(cur)

    # ── Per-period aggregate metrics ────────────────────────────────────────
    for p in periods:
        idxs = p["fractal_idxs"]
        sub = fractal_df.loc[idxs]
        if p["label"] in ("trending_up", "trending_down"):
            vals = sub["pips_per_bar"].dropna().values
            p["pips_per_bar"] = float(np.mean(vals)) if len(vals) else np.nan
            # Total pips covered across the period (sum of v_dist_pips of
            # same-kind successive pairs is approximated as pips/bar * duration_bars).
            duration_bars = p["end_bar"] - p["start_bar"]
            p["total_pips"] = float(p["pips_per_bar"] * duration_bars) if not pd.isna(p["pips_per_bar"]) else np.nan
        elif p["label"] == "ranging":
            w_vals = sub["width_pips"].dropna().values
            c_vals = sub["choppiness"].dropna().values
            p["width_pips"] = float(np.mean(w_vals)) if len(w_vals) else np.nan
            p["choppiness"] = float(np.mean(c_vals)) if len(c_vals) else np.nan

    # ── Tercile thresholds from period-level distributions ──────────────────
    trending_periods = [p for p in periods if p["label"] in ("trending_up", "trending_down")
                        and not pd.isna(p.get("pips_per_bar"))]
    ranging_periods  = [p for p in periods if p["label"] == "ranging"
                        and not pd.isna(p.get("width_pips"))]

    if len(trending_periods) >= 3:
        ppb_vals  = np.array([p["pips_per_bar"] for p in trending_periods])
        ppb_t1    = float(np.quantile(ppb_vals, 1/3))
        ppb_t2    = float(np.quantile(ppb_vals, 2/3))
    else:
        ppb_t1 = ppb_t2 = np.nan

    if len(ranging_periods) >= 3:
        w_vals = np.array([p["width_pips"] for p in ranging_periods])
        w_t1   = float(np.quantile(w_vals, 1/3))
        w_t2   = float(np.quantile(w_vals, 2/3))
    else:
        w_t1 = w_t2 = np.nan

    thresholds = {
        "pips_per_bar_t1": ppb_t1,  # slow ≤ t1 < medium ≤ t2 < fast
        "pips_per_bar_t2": ppb_t2,
        "width_pips_t1":   w_t1,    # narrow ≤ t1 < medium ≤ t2 < wide
        "width_pips_t2":   w_t2,
    }

    # ── Apply fine labels at the period level, then push down to fractals ──
    for p in periods:
        if p["label"] in ("trending_up", "trending_down"):
            direction = "up" if p["label"] == "trending_up" else "down"
            ppb = p.get("pips_per_bar")
            if pd.isna(ppb) or pd.isna(ppb_t1):
                speed = "medium"
            elif ppb >= ppb_t2:
                speed = "fast"
            elif ppb >= ppb_t1:
                speed = "medium"
            else:
                speed = "slow"
            p["regime"] = f"trending_{speed}_{direction}"
        elif p["label"] == "ranging":
            w = p.get("width_pips")
            if pd.isna(w) or pd.isna(w_t1):
                size = "medium"
            elif w >= w_t2:
                size = "wide"
            elif w >= w_t1:
                size = "medium"
            else:
                size = "narrow"
            p["regime"] = f"ranging_{size}"
        else:
            p["regime"] = "transitioning"

    # Project fine labels back onto each fractal row
    regime_col = [None] * len(fractal_df)
    for p in periods:
        for idx in p["fractal_idxs"]:
            regime_col[idx] = p["regime"]
    fractal_df["regime"] = regime_col

    # Drop the heavy rolling-list columns before returning — they're not needed
    # downstream and would bloat the parquet.
    fractal_df = fractal_df.drop(columns=["_rolling_H", "_rolling_L"])

    print(f"Stage 2 complete: {len(periods)} regime periods identified")
    return fractal_df, periods, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b — Macro regime detection (per-day classification)
# ─────────────────────────────────────────────────────────────────────────────

def classify_macro_regime(day_bars):
    """Classify a single trading day's macro regime using three signals:
      (1) Net displacement — first-bar open vs last-bar close, in pips.
      (2) EMA-40 slope — within-day EMA evaluated at Q1 vs Q3.
      (3) N18 fractal structure — sequentially lower (or higher) N=18 fractals.

    Returns (label, details) where label is one of MACRO_REGIME_ORDER and
    details is a dict carrying the raw signal values for display.

    Logic:
      • |displacement| < SMALL → flat (regardless of EMA/N18)
      • Otherwise direction = sign of displacement
      • Strong = displacement ≥ LARGE AND EMA slope agrees AND N18 confirms
      • Staircase = direction-aligned EMA slope (no strong confirmation needed)
      • Conflicting EMA slope vs displacement → flat (the day was indecisive)
    """
    details = {
        "displacement_pips": float("nan"),
        "ema_slope_pips":    float("nan"),
        "n18_high_count":    0,
        "n18_low_count":     0,
        "bars":              len(day_bars),
    }

    # Need enough bars for EMA stabilisation and for N18 to even attempt.
    # Strict-less-than-or-equal because q1 below can reach EMA_MACRO_PERIOD,
    # which would land one past the last valid index when n == EMA_MACRO_PERIOD.
    if len(day_bars) <= max(N18_LOOKBACK * 2 + 1, EMA_MACRO_PERIOD):
        return "flat", details

    bars = day_bars.reset_index(drop=True)

    # ── Signal 1: Net displacement ──────────────────────────────────────────
    displacement_price = float(bars["Close"].iloc[-1]) - float(bars["Open"].iloc[0])
    displacement_pips  = displacement_price * PIP
    details["displacement_pips"] = displacement_pips
    abs_disp = abs(displacement_pips)

    # ── Signal 2: EMA-40 slope from Q1 → Q3 ─────────────────────────────────
    ema = bars["Close"].ewm(span=EMA_MACRO_PERIOD, adjust=False).mean()
    n = len(bars)
    # Wait until the EMA has at least one full period of warmup before
    # sampling, so the slope reflects mature smoothing.
    q1 = max(EMA_MACRO_PERIOD, n // 4)
    q3 = max(q1 + 1, 3 * n // 4)
    q3 = min(q3, n - 1)
    ema_slope_pips = (float(ema.iloc[q3]) - float(ema.iloc[q1])) * PIP
    details["ema_slope_pips"] = ema_slope_pips

    # ── Signal 3: N18 fractal structure ─────────────────────────────────────
    highs = bars["High"].values
    lows  = bars["Low"].values
    n18_highs, n18_lows = [], []
    for i in range(N18_LOOKBACK, len(bars) - N18_LOOKBACK):
        is_ph = True
        is_pl = True
        for k in range(1, N18_LOOKBACK + 1):
            if is_ph and (highs[i] <= highs[i - k] or highs[i] <= highs[i + k]):
                is_ph = False
            if is_pl and (lows[i] >= lows[i - k] or lows[i] >= lows[i + k]):
                is_pl = False
            if not is_ph and not is_pl:
                break
        if is_ph: n18_highs.append(float(highs[i]))
        if is_pl: n18_lows.append(float(lows[i]))
    details["n18_high_count"] = len(n18_highs)
    details["n18_low_count"]  = len(n18_lows)

    def _all_dropping(seq):
        return len(seq) >= 2 and all(seq[i] < seq[i-1] for i in range(1, len(seq)))

    def _all_rising(seq):
        return len(seq) >= 2 and all(seq[i] > seq[i-1] for i in range(1, len(seq)))

    n18_down_confirms = _all_dropping(n18_highs) and _all_dropping(n18_lows)
    n18_up_confirms   = _all_rising(n18_highs)   and _all_rising(n18_lows)

    # ── Combine ─────────────────────────────────────────────────────────────
    if abs_disp < SMALL_DISPLACEMENT_PIPS:
        return "flat", details

    if displacement_pips < 0:
        ema_aligned = ema_slope_pips < 0
        if abs_disp >= LARGE_DISPLACEMENT_PIPS and ema_aligned and n18_down_confirms:
            return "strong_down", details
        if ema_aligned:
            return "staircase_down", details
        return "flat", details
    else:
        ema_aligned = ema_slope_pips > 0
        if abs_disp >= LARGE_DISPLACEMENT_PIPS and ema_aligned and n18_up_confirms:
            return "strong_up", details
        if ema_aligned:
            return "staircase_up", details
        return "flat", details


def stage2b_classify_macro(full_df):
    """Run classify_macro_regime over every trading day in [START_DATE, END_DATE].
    Returns dict keyed by YYYY-MM-DD with {'label': ..., 'details': {...}}."""
    print("Stage 2b: Classifying macro regimes...")

    trading_days = _trading_days_in_range(full_df)
    dts = pd.to_datetime(full_df["Datetime"])
    dts_utc = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts.dt.tz_localize("UTC")

    macro = {}
    for day in trading_days:
        day_start = pd.Timestamp(day, tz="UTC")
        day_end   = day_start + pd.Timedelta(days=1)
        day_bars  = full_df[(dts_utc >= day_start) & (dts_utc < day_end)]
        label, details = classify_macro_regime(day_bars)
        macro[day] = {"label": label, "details": details}

    # Per-label summary print
    counts = {}
    for d in macro.values():
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    print(f"Stage 2b complete: {len(macro)} days classified")
    for lbl in MACRO_REGIME_ORDER:
        if lbl in counts:
            print(f"  {MACRO_REGIME_DISPLAY[lbl]}: {counts[lbl]}")

    return macro


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Trade outcome mapping
# ─────────────────────────────────────────────────────────────────────────────

def stage3_trade_outcomes(fractal_df, full_df, macro=None):
    """Run the strategy backtest, attribute each trade + blocked signal to a
    macro day and a micro regime, and compute per-regime perf stats.

    Returns (trades, perf_df, blocked_df):
      • trades — DataFrame of trades that fired, with 'regime' (micro key)
        and 'macro_label' (macro key) columns added.
      • perf_df — per-micro-regime aggregate stats from the fired trades.
      • blocked_df — DataFrame of signals the strategy generated but didn't
        execute (any filter), with the would-be P&L from _scan_outcome.
        Same columns as trades plus a 'reason' column. Used downstream to
        populate counterfactual stats on locked-regime rows in the perf tables.
    """
    print("Stage 3: Mapping trade outcomes to regimes...")

    trades, _, blocked_signals = strat.run_backtest(full_df)

    # Trade attribution + per-regime stats are report-scoped (the perf tables
    # only show trades inside the report window). May 2026: switched from the
    # shared START_DATE/END_DATE to REPORT_START_DATE/REPORT_END_DATE so a
    # wider labels run doesn't pull in trades outside the report's window.
    start_ts = pd.Timestamp(REPORT_START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(REPORT_END_DATE,   tz="UTC") + pd.Timedelta(days=1)

    # ── Fresh per-fractal regime asof series (for micro attribution) ───────
    _frac_ts = pd.to_datetime(fractal_df["timestamp"])
    if _frac_ts.dt.tz is None:
        _frac_ts = _frac_ts.dt.tz_localize("UTC")
    else:
        _frac_ts = _frac_ts.dt.tz_convert("UTC")
    micro_series = (
        pd.Series(fractal_df["regime"].values, index=_frac_ts).dropna().sort_index()
    )

    def _normalise_entry_ts(df_):
        ts = pd.to_datetime(df_["entry_ts"])
        return ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")

    def _attribute(df_):
        """Add macro_label + regime columns by looking up each row's entry_ts."""
        if df_.empty:
            df_["regime"] = pd.Series([], dtype="object")
            df_["macro_label"] = pd.Series([], dtype="object")
            return df_
        ts_utc = _normalise_entry_ts(df_)
        # Micro — asof on the fresh fractal series.
        try:
            regimes = [micro_series.asof(t) for t in ts_utc]
        except Exception:
            regimes = [None] * len(df_)
        df_ = df_.copy()
        df_["regime"] = regimes
        # Macro — day-keyed lookup from the supplied macro dict.
        macro_dict = macro or {}
        df_["macro_label"] = ts_utc.dt.strftime("%Y-%m-%d").map(
            lambda d: (macro_dict.get(d) or {}).get("label")
        ).values
        return df_

    # ── Fired trades — keep prior in-range filter + add attribution ────────
    if not trades.empty:
        t_utc = _normalise_entry_ts(trades.rename(columns={"entry_ts": "entry_ts"}))
        in_range_mask = (t_utc >= start_ts) & (t_utc < end_ts)
        trades = trades[in_range_mask].copy()
    trades = _attribute(trades)

    # ── Blocked signals → DataFrame + in-range filter + attribution ────────
    if blocked_signals:
        blocked_df = pd.DataFrame(blocked_signals).rename(
            columns={"timestamp": "entry_ts"}
        )
        if "entry_ts" in blocked_df.columns:
            _bts = pd.to_datetime(blocked_df["entry_ts"])
            _bts = _bts.dt.tz_convert("UTC") if _bts.dt.tz is not None else _bts.dt.tz_localize("UTC")
            blocked_df = blocked_df[(_bts >= start_ts) & (_bts < end_ts)].reset_index(drop=True)
    else:
        blocked_df = pd.DataFrame(columns=["entry_ts", "win", "pnl", "reason", "direction"])
    blocked_df = _attribute(blocked_df)

    # ── Per-regime perf from FIRED trades only (active-strategy stats) ─────
    perf = []
    for label in REGIME_ORDER:
        sub = trades[trades["regime"] == label] if not trades.empty else pd.DataFrame()
        n = len(sub)
        if n == 0:
            perf.append({"regime": label, "trades": 0, "wins": 0, "win_rate": np.nan,
                         "profit_factor": np.nan, "avg_pnl": np.nan, "total_pnl": 0.0})
            continue
        wins = int(sub["win"].sum())
        win_rate = wins / n * 100
        gross_win  = float(sub.loc[sub["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-sub.loc[sub["pnl"] < 0, "pnl"].sum())
        pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
        perf.append({
            "regime":   label,
            "trades":   n,
            "wins":     wins,
            "win_rate": win_rate,
            "profit_factor": pf,
            "avg_pnl":  float(sub["pnl"].mean()),
            "total_pnl": float(sub["pnl"].sum()),
        })

    perf_df = pd.DataFrame(perf)
    print(f"Stage 3 complete — {len(trades)} fired trades · "
          f"{len(blocked_df)} blocked signals captured for counterfactual rows")
    return trades, perf_df, blocked_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Threshold print
# ─────────────────────────────────────────────────────────────────────────────

def stage4_thresholds(thresholds):
    print("Stage 4: Computing distribution thresholds...")
    def _fmt(v):
        return f"{v:.3f}" if not pd.isna(v) else "—"
    print(f"  Pips-per-bar terciles:  slow ≤ {_fmt(thresholds['pips_per_bar_t1'])}  "
          f"<  medium ≤ {_fmt(thresholds['pips_per_bar_t2'])}  <  fast")
    print(f"  Range-width terciles:   narrow ≤ {_fmt(thresholds['width_pips_t1'])} pips  "
          f"<  medium ≤ {_fmt(thresholds['width_pips_t2'])} pips  <  wide")
    print("Stage 4 complete")


# ─────────────────────────────────────────────────────────────────────────────
# Daily chart generation
# ─────────────────────────────────────────────────────────────────────────────

def _trading_days_in_range(full_df, start_date=None, end_date=None):
    """List of unique YYYY-MM-DD strings in `full_df` inside the requested
    [start_date, end_date] window. Defaults to the labels window
    (START_DATE / END_DATE) — pass the report window explicitly when the
    caller wants days bounded by the report range (e.g. build_report's
    low-activity stats or generate_daily_charts' per-day loop). May 2026:
    parameterised for the labels/report-window split."""
    sd = start_date if start_date is not None else START_DATE
    ed = end_date   if end_date   is not None else END_DATE
    start_ts = pd.Timestamp(sd, tz="UTC")
    end_ts   = pd.Timestamp(ed, tz="UTC") + pd.Timedelta(days=1)
    dts = pd.to_datetime(full_df["Datetime"])
    dts_utc = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts.dt.tz_localize("UTC")
    in_rng = full_df[(dts_utc >= start_ts) & (dts_utc < end_ts)]
    in_rng_dts = pd.to_datetime(in_rng["Datetime"])
    in_rng_utc = in_rng_dts.dt.tz_convert("UTC") if in_rng_dts.dt.tz is not None else in_rng_dts.dt.tz_localize("UTC")
    days = sorted(set(in_rng_utc.dt.strftime("%Y-%m-%d")))
    return days


def generate_daily_charts(full_df):
    """Generate one daily chart per trading day in the requested range and
    save to results/regime_charts/YYYY-MM-DD.png. Returns the set of dates
    for which a chart was successfully written.

    Strategy: run strategy_v2's per-day backtest pipeline (same as the
    dashboard's single-day mode), call strat.save_charts to produce the chart
    on disk, then shutil.move it into the regime_charts/ directory under a
    date-named filename. The companion equity/drawdown PNG is discarded —
    the hover preview only shows the main price/trade chart.

    Notes
    -----
    * save_charts builds its filename from `datetime.now()`, so each
      iteration overwrites `results/v2_GBPUSD_{today}.png`. We move the file
      immediately after each call to avoid collisions.
    * save_charts prints chart paths to stdout — we redirect that under a
      buffer so terminal output stays minimal.
    """
    REGIME_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    # Bound chart generation to the report window (May 2026). A wide labels
    # run shouldn't dump every labelled day's PNG to disk — only the days
    # the report will actually link to. If labels == report (the common
    # case), behaviour is unchanged.
    days = _trading_days_in_range(full_df, REPORT_START_DATE, REPORT_END_DATE)
    total = len(days)
    if total == 0:
        return set()

    available = set()
    for i, day in enumerate(days, start=1):
        print(f"Generating charts: day {i} of {total}...")
        try:
            # Mirror strategy_v2's date-range pipeline for a single day.
            df_day = strat.fetch_data(
                strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                start_date=day, end_date=day,
            )
            df_day = strat.add_indicators(df_day)

            # Apply v2's standard buffers: 1d pre / 7d post.
            r_start = pd.Timestamp(day, tz="UTC")
            r_end   = r_start + pd.Timedelta(days=1)
            dts     = pd.to_datetime(df_day["Datetime"])
            dts_utc = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts.dt.tz_localize("UTC")
            bt_start = r_start - pd.Timedelta(days=1)
            bt_end   = r_end   + pd.Timedelta(days=7)
            df_day = df_day[(dts_utc >= bt_start) & (dts_utc < bt_end)].reset_index(drop=True)

            # Run backtest, then trim to the requested day (mirrors v2 main).
            trades, equity, _ = strat.run_backtest(df_day)
            t_dts = pd.to_datetime(df_day["Datetime"])
            t_utc = t_dts.dt.tz_convert("UTC") if t_dts.dt.tz is not None else t_dts.dt.tz_localize("UTC")
            pre_buffer = int((t_utc < r_start).sum())
            day_mask   = (t_utc >= r_start) & (t_utc < r_end)
            df_view    = df_day[day_mask].reset_index(drop=True)
            if not trades.empty:
                t_entry = pd.to_datetime(trades["entry_ts"])
                t_entry_utc = t_entry.dt.tz_convert("UTC") if t_entry.dt.tz is not None else t_entry.dt.tz_localize("UTC")
                t_mask = (t_entry_utc >= r_start) & (t_entry_utc < r_end)
                trades = trades[t_mask].copy()
                trades["entry_idx"]   = trades["entry_idx"]   - pre_buffer
                trades["exit_idx"]    = trades["exit_idx"]    - pre_buffer
                trades["fractal_bar"] = trades["fractal_bar"] - pre_buffer
                trades = trades.reset_index(drop=True)

            # ── Pad df_view to a full 24h UTC grid (Issue 1) ────────────────
            # Build the canonical 5-min grid for the day (00:00 → 23:55 UTC,
            # 288 slots) and reindex df_view onto it. Forward/back-fill OHLC
            # and indicator columns so quiet hours render as flat segments
            # rather than being clipped from the chart x-axis. Then remap
            # each trade's positional indices via real timestamps so trade
            # markers and the equity curve still line up.
            #
            # Important: we must thread the tz-aware UTC Datetime Series
            # through end-to-end. Calling .values on a tz-aware Series
            # strips the tz (numpy datetime64 doesn't carry tz), which
            # would make reindex against the tz-aware full_grid silently
            # match nothing — every row would come back NaN.
            if not df_view.empty:
                # Normalise the Datetime column to a tz-aware UTC Series
                dt_col = pd.to_datetime(df_view["Datetime"])
                if dt_col.dt.tz is None:
                    dt_col = dt_col.dt.tz_localize("UTC")
                else:
                    dt_col = dt_col.dt.tz_convert("UTC")
                df_view = df_view.copy()
                df_view["Datetime"] = dt_col   # assign the Series — preserves tz

                # Snapshot the real timestamps each trade index points to
                # BEFORE padding so we can re-locate them in the padded frame.
                if not trades.empty:
                    def _ts_at(idx_series):
                        idx_clamped = idx_series.astype(int).clip(
                            lower=0, upper=len(df_view) - 1).values
                        return df_view["Datetime"].iloc[idx_clamped].tolist()
                    entry_ts_real = _ts_at(trades["entry_idx"])
                    exit_ts_real  = _ts_at(trades["exit_idx"])
                    fb_ts_real    = _ts_at(trades["fractal_bar"])

                # Build the canonical 24h grid (tz-aware UTC) and reindex
                full_grid = pd.date_range(
                    start=r_start, end=r_end - pd.Timedelta(minutes=5),
                    freq="5min", tz="UTC",
                )
                df_view_idx = df_view.set_index("Datetime")   # tz-aware index
                df_padded = df_view_idx.reindex(full_grid)
                for col in df_padded.columns:
                    df_padded[col] = df_padded[col].ffill().bfill()
                df_padded = df_padded.reset_index().rename(columns={"index": "Datetime"})
                df_view = df_padded

                # Re-locate trade indices in the padded grid
                if not trades.empty:
                    ts_to_pos = {pd.Timestamp(ts): i
                                 for i, ts in enumerate(df_view["Datetime"])}
                    def _remap(arr):
                        out = []
                        for ts in arr:
                            t = pd.Timestamp(ts)
                            if t.tzinfo is None: t = t.tz_localize("UTC")
                            out.append(ts_to_pos.get(t, 0))
                        return out
                    trades = trades.copy()
                    trades["entry_idx"]   = _remap(entry_ts_real)
                    trades["exit_idx"]    = _remap(exit_ts_real)
                    trades["fractal_bar"] = _remap(fb_ts_real)

            # Rebuild a day-scoped equity curve from trade exits inside the day.
            eq_exits = {}
            for _, t in trades.iterrows():
                eidx = int(t["exit_idx"])
                if eidx >= len(df_view):
                    eidx = len(df_view) - 1
                if 0 <= eidx:
                    eq_exits.setdefault(eidx, []).append(float(t["pnl"]))
            cash = strat.STARTING_CASH
            day_equity = [cash]
            for bi in range(1, len(df_view)):
                for p in eq_exits.get(bi, []):
                    cash += p
                day_equity.append(cash)

            # Call save_charts silently, then move the main chart into place.
            with contextlib.redirect_stdout(io.StringIO()):
                main_path, eq_dd_path = strat.save_charts(df_view, trades, day_equity)

            target = REGIME_CHARTS_DIR / f"{day}.png"
            if Path(main_path).exists():
                shutil.move(str(main_path), str(target))
                available.add(day)
            if Path(eq_dd_path).exists():
                # Discard companion eq/dd chart — not used by the hover preview.
                Path(eq_dd_path).unlink()
        except Exception as e:
            # Chart generation failure is non-fatal — the row just shows the
            # "Chart not available" greyed-out icon.
            print(f"  (day {day}: chart skipped — {type(e).__name__}: {e})")

    return available


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers — dark theme matplotlib → base64
# ─────────────────────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.4, linewidth=0.6)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG_DARK, bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def chart_distribution(values, t1, t2, title, xlabel):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.patch.set_facecolor(BG_DARK)
    if len(values) > 0:
        ax.hist(values, bins=min(20, max(5, len(values) // 2)),
                color=ACCENT, edgecolor=BG_DARK)
        if not pd.isna(t1):
            ax.axvline(t1, color=YELLOW, linestyle="--", linewidth=1.4,
                       label=f"t1 = {t1:.2f}")
        if not pd.isna(t2):
            ax.axvline(t2, color="#ff6b6b", linestyle="--", linewidth=1.4,
                       label=f"t2 = {t2:.2f}")
        leg = ax.legend(facecolor=BG_DARK, edgecolor=GRID, fontsize=9, labelcolor=TEXT)
    else:
        ax.text(0.5, 0.5, "No data", color=TEXT, ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Periods")
    _style_axes(ax)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — formatting / classes
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v):
    return f"{v:.1f}%" if not pd.isna(v) else "—"

def _fmt_num(v, dp=2):
    return f"{v:.{dp}f}" if not pd.isna(v) else "—"

def _fmt_pf(v):
    if pd.isna(v): return "—"
    if v == float("inf"): return "∞"
    return f"{v:.2f}"

def _fmt_money(v):
    if pd.isna(v): return "—"
    return f"${v:,.0f}"

def winrate_class(rate, trades):
    if trades == 0 or pd.isna(rate): return "win-neutral"
    if rate >= 55: return "win-good"
    if rate <= 45: return "win-bad"
    return "win-neutral"


def regime_class(regime_key):
    """Return the style.css class that paints this regime's color. Keeps the
    HTML free of inline styles even though colors are data-driven — there is
    one class per regime defined in style.css under '── Regime palette ──'."""
    if not regime_key:
        return "regime-color-none"
    return "regime-color-" + regime_key.replace("_", "-")


# ─────────────────────────────────────────────────────────────────────────────
# Timeline — daily dominant regime, color-coded
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_timeline(periods, start_date, end_date):
    """Return a list of {date, regime, label, color} for each calendar day
    in [start_date, end_date]. The dominant regime that day is whichever
    period covers the most minutes of that 24-hour window."""
    start = pd.Timestamp(start_date, tz="UTC").normalize()
    end   = pd.Timestamp(end_date,   tz="UTC").normalize()

    timeline = []
    day = start
    while day <= end:
        day_start = day
        day_end   = day + pd.Timedelta(days=1)
        # Tally minutes per regime by intersecting each period with the day.
        tally = {}
        for p in periods:
            p_start = pd.Timestamp(p["start_ts"]).tz_convert("UTC") if pd.Timestamp(p["start_ts"]).tzinfo else pd.Timestamp(p["start_ts"]).tz_localize("UTC")
            p_end   = pd.Timestamp(p["end_ts"]).tz_convert("UTC")   if pd.Timestamp(p["end_ts"]).tzinfo   else pd.Timestamp(p["end_ts"]).tz_localize("UTC")
            # Treat period as inclusive of last fractal — extend by 1 candle (5min)
            p_end = p_end + pd.Timedelta(minutes=5)
            lo = max(day_start, p_start)
            hi = min(day_end,   p_end)
            if hi > lo:
                mins = (hi - lo).total_seconds() / 60
                tally[p["regime"]] = tally.get(p["regime"], 0) + mins
        if tally:
            dominant = max(tally, key=tally.get)
        else:
            dominant = None
        timeline.append({
            "date":    day.strftime("%Y-%m-%d"),
            "regime":  dominant,
            "label":   REGIME_DISPLAY.get(dominant, "No data") if dominant else "No data",
            "color":   REGIME_COLORS.get(dominant, "#1a1a2e") if dominant else "#1a1a2e",
        })
        day = day + pd.Timedelta(days=1)
    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# Daily performance — one row per trading day with hourly chips + chart preview
# ─────────────────────────────────────────────────────────────────────────────

def _hour_dominant_regime(periods, hour_start, hour_end):
    """Return the regime label covering the most minutes of [hour_start, hour_end]."""
    tally = {}
    for p in periods:
        p_start = pd.Timestamp(p["start_ts"])
        p_end   = pd.Timestamp(p["end_ts"])
        if p_start.tzinfo is None: p_start = p_start.tz_localize("UTC")
        else:                       p_start = p_start.tz_convert("UTC")
        if p_end.tzinfo is None:   p_end   = p_end.tz_localize("UTC")
        else:                       p_end   = p_end.tz_convert("UTC")
        # Extend by 1 candle so the last fractal's minute is covered.
        p_end = p_end + pd.Timedelta(minutes=5)
        lo = max(hour_start, p_start)
        hi = min(hour_end, p_end)
        if hi > lo:
            mins = (hi - lo).total_seconds() / 60
            tally[p["regime"]] = tally.get(p["regime"], 0) + mins
    if not tally:
        return None
    return max(tally, key=tally.get)


LOW_ACTIVITY_FRACTAL_THRESHOLD = 3   # days with fewer than this many fractals
                                     # are flagged as "low activity"


def build_hourly_chips(day_str, day_fractals, day_trades_by_hour=None):
    """Return HTML for 24 hourly chips, one per UTC hour of day_str.

    `day_fractals` is a DataFrame of fractals that occurred on this UTC date
    (must have 'timestamp' and 'regime' columns). Hours containing at least
    one fractal show the most common regime in that hour. Hours with zero
    fractals show the regime-color-inactive (very dark) chip so it is visually
    obvious which hours had no classification at all — rather than letting
    forward-fill from prior periods make the day appear uniformly classified.

    `day_trades_by_hour` is an optional dict `{hour_int: [win_bool, ...]}`
    listing the trades that *entered* during each UTC hour of this day. Each
    listed trade is rendered as a small dot on its hour's chip — light green
    for a winning trade, red for a loss. Up to 3 dots are shown stacked
    vertically; for 4+ trades in the same hour the 3 dots are followed by a
    small "+N" indicator. On blocked-macro-regime rows the existing row-level
    opacity rule dims the dots along with the rest of the cell, so
    counterfactual entries are still visible at the same reduced opacity.
    """
    day_trades_by_hour = day_trades_by_hour or {}

    # Pre-group fractals by hour for this day
    by_hour = {}
    if not day_fractals.empty:
        ts = pd.to_datetime(day_fractals["timestamp"])
        ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
        hours = ts_utc.dt.hour.values
        for h, regime in zip(hours, day_fractals["regime"].values):
            by_hour.setdefault(int(h), []).append(regime)

    chips = []
    for h in range(24):
        regimes_in_hour = by_hour.get(h)
        trades_this_hour = day_trades_by_hour.get(h, [])

        # Determine chip class + base hover title
        if not regimes_in_hour:
            base_title = f"{day_str} {h:02d}:00 — No fractal activity"
            chip_cls   = "regime-color-inactive"
        else:
            # Most common regime in this hour (ties broken by first occurrence)
            counts = {}
            for r in regimes_in_hour:
                counts[r] = counts.get(r, 0) + 1
            top = max(counts, key=counts.get)
            n_frac = sum(counts.values())
            label = REGIME_DISPLAY.get(top, "Unknown")
            base_title = (
                f"{day_str} {h:02d}:00 — {label} "
                f"({n_frac} fractal{'s' if n_frac != 1 else ''})"
            )
            chip_cls = regime_class(top)

        # Trade-entry dots overlay
        if trades_this_hour:
            n_tr   = len(trades_this_hour)
            wins   = sum(1 for w in trades_this_hour if w)
            losses = n_tr - wins
            dots = []
            for w in trades_this_hour[:3]:
                dot_cls = ("regime-hour-chip-dot--win" if w
                           else "regime-hour-chip-dot--loss")
                dots.append(
                    f"<span class='regime-hour-chip-dot {dot_cls}'></span>"
                )
            extra = n_tr - 3
            if extra > 0:
                dots.append(
                    f"<span class='regime-hour-chip-dot-more'>+{extra}</span>"
                )
            trades_html = (
                f"<span class='regime-hour-chip-trades'>{''.join(dots)}</span>"
            )
            title = (
                f"{base_title} · {n_tr} trade{'s' if n_tr != 1 else ''} "
                f"({wins}W / {losses}L)"
            )
        else:
            trades_html = ""
            title = base_title

        chips.append(
            f"<span class='regime-hour-chip {chip_cls}' title='{title}'>"
            f"{trades_html}</span>"
        )
    return f"<div class='regime-hour-chips'>{''.join(chips)}</div>"


def _macro_trades_by_label(macro, trades_df):
    """Helper — group trades by the macro regime label of their entry day.
    Returns a DataFrame with an added `macro_label` column, plus a dict of
    {macro_label: trade_count_for_that_label}."""
    if trades_df.empty:
        return trades_df.copy(), {}
    td = trades_df.copy()
    ts = pd.to_datetime(td["entry_ts"])
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
    td["entry_day"] = ts_utc.dt.strftime("%Y-%m-%d")
    td["macro_label"] = td["entry_day"].map(
        lambda d: macro.get(d, {}).get("label")
    )
    counts = td.groupby("macro_label", dropna=False).size().to_dict()
    return td, counts


def build_macro_perf_table(macro, trades_df, blocked_macro_keys=None,
                            blocked_signals_df=None, unfiltered_trades_df=None):
    """Render the 'Macro regime performance' table.

    Columns: Regime / Periods / Trades / Wins / Win rate / Profit factor /
    Avg P&L / Total P&L. 'Periods' is the count of days classified into
    each macro regime.

    Allowed rows (macro not in blocked_macro_keys): stats come from
    `trades_df` (the active strategy's filtered trades).

    Locked rows: stats are counterfactual ("what would have happened if
    this regime were allowed"). Two sources are supported:
      • `unfiltered_trades_df` — the FULL trade set from a backtest with
        the regime gate disabled. When the RA endpoint runs in cached
        mode (Task 1), the backtest is executed with empty allow-lists so
        every signal that passed the non-regime gates fires; the locked
        row pulls trades on its regime directly from this set. Preferred
        because it accounts for all the other gates (EMA / time /
        daily-loss-limit) being applied to a no-regime-filter run.
      • `blocked_signals_df` — legacy fallback. Holds the `reason='macro_regime'`
        rejections from a backtest where the gate WAS active. Less
        accurate (position sizing + daily-loss-limit aren't re-simulated
        as if those signals had fired), but kept for back-compat with
        callers that haven't migrated to the cached pipeline.
    """
    days_per_label = {}
    for d in macro.values():
        days_per_label[d["label"]] = days_per_label.get(d["label"], 0) + 1

    blocked_macro_keys = blocked_macro_keys or set()
    if blocked_signals_df is None:
        blocked_signals_df = pd.DataFrame(columns=trades_df.columns)

    # Only count signals that were blocked *specifically* by the macro filter —
    # the strategy applies filters in order (EMA → daily-loss → time → macro →
    # micro), so a signal with reason='macro_regime' is the only kind that
    # would have fired if macro were unlocked on this row. Including
    # everything else (EMA-blocked, time-blocked, etc.) over-counts by an
    # order of magnitude — those signals stay blocked regardless of the
    # macro toggle.
    if not blocked_signals_df.empty and "reason" in blocked_signals_df.columns:
        macro_only_blocked = blocked_signals_df[blocked_signals_df["reason"] == "macro_regime"]
    else:
        macro_only_blocked = blocked_signals_df.iloc[0:0]

    # Counterfactual source for locked rows. Prefer the unfiltered trades
    # (from a no-regime-gate backtest) when provided; otherwise fall back
    # to the legacy fired-trades + macro_only_blocked combination.
    have_unfiltered = (unfiltered_trades_df is not None
                       and not unfiltered_trades_df.empty
                       and "macro_label" in unfiltered_trades_df.columns)

    counterfactual_tip = (
        "Counterfactual — would-be outcome of signals that were blocked "
        "specifically by the macro filter on this regime. Win rate and "
        "trade count are accurate; aggregate P&L is approximate due to "
        "position-sizing and daily-loss-limit interactions if these had "
        "actually fired."
    )

    rows = []
    for label in MACRO_REGIME_ORDER:
        days = days_per_label.get(label, 0)
        if days == 0:
            continue
        is_blocked = label in blocked_macro_keys

        # Pick the source frame:
        #   • Locked row + have_unfiltered → unfiltered trades (best counterfactual).
        #   • Locked row + legacy path → fired trades + macro_only_blocked.
        #   • Allowed row → fired trades only.
        if is_blocked and have_unfiltered:
            source = unfiltered_trades_df
        elif is_blocked and not macro_only_blocked.empty:
            source = pd.concat([trades_df, macro_only_blocked], ignore_index=True)
        else:
            source = trades_df

        if source.empty or "macro_label" not in source.columns:
            sub = source.iloc[0:0]
        else:
            sub = source[source["macro_label"] == label]
        stats = _row_stats(sub)

        # Total P&L cell — counterfactual tooltip on blocked rows.
        if is_blocked:
            tot_cell = (
                f"<td class='regime-pnl-counterfactual' title='{counterfactual_tip}'>"
                f"{stats['tot_inner']}</td>"
            )
        else:
            tot_cell = f"<td>{stats['tot_inner']}</td>"

        if is_blocked:
            lock_html = (
                "<span class='regime-blocked-lock' "
                "title='Excluded from active strategy statistics by the macro filter' "
                "aria-label='Filtered out'>"
                "<span class='material-symbols-outlined'>lock</span>"
                "</span>"
            )
        else:
            lock_html = ""
        row_class = "regime-day-blocked" if is_blocked else ""

        rows.append(
            f"<tr class='{row_class}'>"
            f"<td><span class='macro-badge {macro_class(label)}'>"
            f"{MACRO_REGIME_DISPLAY[label]}</span>"
            f"{lock_html}</td>"
            f"<td>{days}</td>"
            f"<td>{stats['n']}</td>"
            f"<td>{stats['wins_str']}</td>"
            f"{stats['wr_cell']}{stats['pf_cell']}{stats['avg_cell']}{tot_cell}"
            f"</tr>"
        )
    table = f"""
      <table class='regime-table regime-perf-table'>
        <thead><tr>
          <th>Regime</th><th>Periods</th><th>Trades</th><th>Wins</th><th>Win rate</th>
          <th>Profit factor</th><th>Avg P&amp;L</th><th>Total P&amp;L</th>
        </tr></thead>
        <tbody>{''.join(rows) or '<tr><td colspan=8 class=regime-dim>No macro regimes observed.</td></tr>'}</tbody>
      </table>
      <p class="regime-dim regime-small regime-counterfactual-note">
        Locked-regime rows show counterfactual stats — the would-be outcome
        of signals that were filtered out. Win rate and trade counts are
        accurate; aggregate P&amp;L is approximate.
      </p>
    """
    return table


def build_macro_summary_cards(macro, trades_df):
    """Render one summary card per observed macro regime. Same look as the
    micro summary cards but with macro-specific fields: days count, avg
    displacement, avg EMA slope, trade count, win rate, PF."""
    # Group days by label so we can aggregate signal averages
    by_label = {}
    for day, d in macro.items():
        by_label.setdefault(d["label"], []).append(d["details"])

    td, _ = _macro_trades_by_label(macro, trades_df)

    cards = []
    for label in MACRO_REGIME_ORDER:
        details_list = by_label.get(label, [])
        if not details_list:
            continue
        disp_vals = [d["displacement_pips"] for d in details_list
                     if not pd.isna(d["displacement_pips"])]
        slope_vals = [d["ema_slope_pips"] for d in details_list
                      if not pd.isna(d["ema_slope_pips"])]
        avg_disp  = float(np.mean(disp_vals))  if disp_vals  else float("nan")
        avg_slope = float(np.mean(slope_vals)) if slope_vals else float("nan")

        if td.empty:
            n_trades = 0; wins = 0; wr = float("nan"); pf = float("nan")
        else:
            sub = td[td["macro_label"] == label]
            n_trades = int(len(sub))
            if n_trades > 0:
                wins = int(sub["win"].sum())
                wr   = wins / n_trades * 100
                gw   = float(sub.loc[sub["pnl"] > 0, "pnl"].sum())
                gl   = float(-sub.loc[sub["pnl"] < 0, "pnl"].sum())
                pf   = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
            else:
                wins = 0; wr = float("nan"); pf = float("nan")

        slope_dir = "—" if pd.isna(avg_slope) else ("↗ positive" if avg_slope > 0 else "↘ negative")
        cards.append(f"""
          <div class='regime-card regime-summary-card'>
            <div class='regime-card-head'>
              <span class='macro-badge {macro_class(label)}'>
                {MACRO_REGIME_DISPLAY[label]}
              </span>
            </div>
            <div class='regime-card-grid'>
              <div><span class='regime-dim regime-small'>Days</span><strong>{len(details_list)}</strong></div>
              <div><span class='regime-dim regime-small'>Avg displacement</span><strong>{_fmt_num(avg_disp, 1)} pips</strong></div>
              <div><span class='regime-dim regime-small'>Avg EMA slope</span><strong>{_fmt_num(avg_slope, 2)} pips ({slope_dir})</strong></div>
              <div><span class='regime-dim regime-small'>Trades</span><strong>{n_trades}</strong></div>
              <div><span class='regime-dim regime-small'>Win rate</span><strong>{_fmt_pct(wr)}</strong></div>
              <div><span class='regime-dim regime-small'>Profit factor</span><strong>{_fmt_pf(pf)}</strong></div>
            </div>
          </div>
        """)
    return "".join(cards) if cards else "<p class='regime-dim'>No macro regimes observed.</p>"


def build_daily_breakdown(periods, trades_df, full_df, available_chart_days,
                          in_range_fractals, low_activity_days,
                          macro=None, blocked_macro_keys=None,
                          trading_days=None):
    """Build the sortable daily-breakdown table. Returns the full <table>…</table>
    HTML string.

    Parameters
    ----------
    available_chart_days : set of YYYY-MM-DD strings for which a chart was
        successfully generated; missing days get a greyed-out preview icon.
    in_range_fractals : the in-range fractal DataFrame with 'timestamp' and
        'regime' columns; used to drive the per-hour chip rendering so that
        hours with zero fractals show as inactive rather than forward-filled.
    low_activity_days : set of YYYY-MM-DD strings flagged as low-activity
        (fewer than LOW_ACTIVITY_FRACTAL_THRESHOLD fractals); a small amber
        warning dot is appended next to those dates.
    trading_days : optional pre-computed sorted list of YYYY-MM-DD trading
        days. When passed, skips the per-call `_trading_days_in_range` work
        which is O(N_bars) and dominates the post-cache RA endpoint time on
        wide date ranges. Falls back to deriving from `full_df` when None.
    """
    days = trading_days if trading_days is not None else _trading_days_in_range(full_df)
    if not days:
        return "<p class='regime-dim'>No trading days in range.</p>"

    # Index trades by their entry day for fast lookup
    if not trades_df.empty:
        td = trades_df.copy()
        ts = pd.to_datetime(td["entry_ts"])
        ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
        td["entry_day"] = ts_utc.dt.strftime("%Y-%m-%d")
    else:
        td = pd.DataFrame(columns=["entry_day", "win", "pnl"])

    # Pre-group trade-entry timestamps by (day, UTC hour). Each chip in the
    # hourly strip uses this to render small win/loss dots for any trade that
    # *entered* in that hour. We use the full (unfiltered) trades_df so the
    # hourly dots remain visible on blocked-macro-regime rows too — the
    # row-level opacity rule will dim them to match the rest of the row.
    trades_by_day_hour = {}
    if not trades_df.empty:
        _t = pd.to_datetime(trades_df["entry_ts"])
        _t_utc = _t.dt.tz_convert("UTC") if _t.dt.tz is not None else _t.dt.tz_localize("UTC")
        for _ts, _win in zip(_t_utc, trades_df["win"].values):
            _d = _ts.strftime("%Y-%m-%d")
            _h = int(_ts.hour)
            trades_by_day_hour.setdefault(_d, {}).setdefault(_h, []).append(bool(_win))

    # Pre-group fractals by day so each row's chip-build only sees its own day
    fractals_by_day = {}
    if not in_range_fractals.empty:
        ts = pd.to_datetime(in_range_fractals["timestamp"])
        ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC")
        date_keys = ts_utc.dt.strftime("%Y-%m-%d").values
        for k, sub_df in in_range_fractals.assign(_date=date_keys).groupby("_date"):
            fractals_by_day[k] = sub_df

    rows = []
    for day in days:
        day_trades = td[td["entry_day"] == day]
        n_trades = int(len(day_trades))
        n_wins   = int(day_trades["win"].sum()) if n_trades else 0
        pnl      = float(day_trades["pnl"].sum()) if n_trades else 0.0

        day_fractals = fractals_by_day.get(
            day, pd.DataFrame(columns=["timestamp", "regime"])
        )
        chips_html = build_hourly_chips(
            day, day_fractals,
            day_trades_by_hour=trades_by_day_hour.get(day, {}),
        )

        pnl_cls = "regime-pnl-pos" if pnl > 0 else ("regime-pnl-neg" if pnl < 0 else "regime-pnl-zero")
        pnl_html = f"<span class='{pnl_cls}'>{_fmt_money(pnl)}</span>"

        # Low activity flag — amber dot with tooltip
        if day in low_activity_days:
            n_frac = len(day_fractals)
            tip = ("Low activity day — fewer than 3 fractals detected. "
                   "Regime classification may not be reliable.")
            date_cell = (
                f"<span class='regime-low-activity-dot' title='{tip}'></span>"
                f"{day}"
            )
        else:
            date_cell = day

        if day in available_chart_days:
            chart_btn = (
                f"<button class='v-sub-preview-btn' type='button' "
                f"data-chart-src='regime_charts/{day}.png' "
                f"title='Preview chart' aria-label='Preview chart for {day}'>"
                f"<span class='material-symbols-outlined'>visibility</span>"
                f"</button>"
            )
        else:
            chart_btn = (
                f"<button class='v-sub-preview-btn disabled' type='button' "
                f"disabled aria-disabled='true' "
                f"title='Chart not available'>"
                f"<span class='material-symbols-outlined'>visibility_off</span>"
                f"</button>"
            )

        # Macro regime badge (new column) — colored badge for the day's
        # day-level macro classification. Falls back to a neutral '—' cell
        # if the macro layer wasn't computed.
        is_blocked = False
        if macro is not None and day in macro:
            macro_label = macro[day]["label"]
            if blocked_macro_keys and macro_label in blocked_macro_keys:
                is_blocked = True
                lock_html = (
                    "<span class='regime-blocked-lock' "
                    "title='Excluded from performance statistics by the macro filter' "
                    "aria-label='Filtered out'>"
                    "<span class='material-symbols-outlined'>lock</span>"
                    "</span>"
                )
            else:
                lock_html = ""
            macro_cell = (
                f"<span class='macro-badge {macro_class(macro_label)}'>"
                f"{MACRO_REGIME_DISPLAY.get(macro_label, '—')}</span>"
                f"{lock_html}"
            )
            macro_sort = MACRO_REGIME_ORDER.index(macro_label) if macro_label in MACRO_REGIME_ORDER else 99
        else:
            macro_cell = "<span class='regime-dim'>—</span>"
            macro_sort = 99

        row_class = "regime-day-blocked" if is_blocked else ""
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td data-sort-value='{day}'>{date_cell}</td>"
            f"<td data-sort-value='{macro_sort}'>{macro_cell}</td>"
            f"<td>{chips_html}</td>"
            f"<td data-sort-value='{n_trades}'>{n_trades}</td>"
            f"<td data-sort-value='{n_wins}'>{n_wins}</td>"
            f"<td data-sort-value='{pnl:.4f}'>{pnl_html}</td>"
            f"<td class='regime-chart-cell'>{chart_btn}</td>"
            f"</tr>"
        )

    table = f"""
      <table class='regime-table regime-daily-table' id='regime-daily-table'>
        <thead>
          <tr>
            <th class='regime-sort' data-sort-type='string'>Date</th>
            <th class='regime-sort' data-sort-type='number'>Macro Regime</th>
            <th>Regime by hour (UTC)</th>
            <th class='regime-sort' data-sort-type='number'>Trades</th>
            <th>Wins</th>
            <th class='regime-sort' data-sort-type='number'>P&amp;L</th>
            <th></th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    """
    return table


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(fractal_df, periods, thresholds, trades_df, perf_df,
                 full_df, available_chart_days, macro,
                 blocked_signals_df=None):
    """Render results/regime_analysis.html and return its path.

    `blocked_signals_df` is the DataFrame of signals that didn't fire
    (from stage3_trade_outcomes). When provided, the perf tables populate
    locked-regime rows with the counterfactual stats from those blocked
    signals' scan_outcome P&Ls.
    """
    if blocked_signals_df is None:
        blocked_signals_df = pd.DataFrame(columns=trades_df.columns)

    # Trim to requested range for display counts. May 2026: report-scoped —
    # the parquet may carry labels for a wider window (labels-pipeline), but
    # the report's display counts / cards / timeline only show the report
    # window. The labels parquet itself is unaffected.
    start_ts = pd.Timestamp(REPORT_START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(REPORT_END_DATE,   tz="UTC") + pd.Timedelta(days=1)
    in_range = fractal_df[
        (fractal_df["timestamp"] >= start_ts) & (fractal_df["timestamp"] < end_ts)
    ]
    in_range_periods = [
        p for p in periods
        if pd.Timestamp(p["end_ts"]) >= start_ts and pd.Timestamp(p["start_ts"]) < end_ts
    ]

    # Window the macro dict to the report range too (May 2026 follow-up to
    # the labels/report split). The unfiltered macro dict from stage2b spans
    # the full labels window; every downstream consumer in this function
    # (_filter_trades_by_macro, build_macro_perf_table, build_macro_summary_cards,
    # build_timeline_section_html) computes counts/labels from macro.values()
    # or macro.get(date) and would otherwise produce full-history totals.
    # Persist_labels in main() still gets the unfiltered macro — the parquet
    # carries the full labels window regardless of the report window.
    macro = {d: v for d, v in (macro or {}).items()
             if REPORT_START_DATE <= d <= REPORT_END_DATE}

    # Periods-per-regime count (used by the Micro regime performance table
    # and by the timeline legend).
    regime_count = {}
    for p in in_range_periods:
        regime_count[p["regime"]] = regime_count.get(p["regime"], 0) + 1

    # ── Low-activity day stats (Issue 3) ────────────────────────────────────
    # A day is "low activity" when fewer than LOW_ACTIVITY_FRACTAL_THRESHOLD
    # fractals were detected during it. Trades on these days are not excluded
    # from any statistics — we just flag them so the user knows the regime
    # labels assigned during those quiet stretches may be less reliable.
    trading_days_all = _trading_days_in_range(full_df, REPORT_START_DATE, REPORT_END_DATE)
    fractals_per_day = {}
    if not in_range.empty:
        _ts = pd.to_datetime(in_range["timestamp"])
        _ts_utc = _ts.dt.tz_convert("UTC") if _ts.dt.tz is not None else _ts.dt.tz_localize("UTC")
        for d in _ts_utc.dt.strftime("%Y-%m-%d"):
            fractals_per_day[d] = fractals_per_day.get(d, 0) + 1
    low_activity_days = {
        d for d in trading_days_all
        if fractals_per_day.get(d, 0) < LOW_ACTIVITY_FRACTAL_THRESHOLD
    }
    low_activity_count = len(low_activity_days)
    total_days         = len(trading_days_all)
    low_activity_pct   = (low_activity_count / total_days * 100) if total_days else 0.0

    # Trades that fired on low-activity days — for the perf-table note.
    trades_on_low_days = 0
    total_in_range_trades = 0
    if not trades_df.empty:
        _te = pd.to_datetime(trades_df["entry_ts"])
        _te_utc = _te.dt.tz_convert("UTC") if _te.dt.tz is not None else _te.dt.tz_localize("UTC")
        _t_date = _te_utc.dt.strftime("%Y-%m-%d")
        trades_on_low_days = int(_t_date.isin(low_activity_days).sum())
        total_in_range_trades = int(len(trades_df))

    # ── Macro-regime filter ─────────────────────────────────────────────────
    # When APPLY_MACRO_FILTER is True, exclude trades whose entry day falls
    # on a blocked macro regime from all aggregate stats, perf tables, win
    # rate, and profit factor. Daily performance + timeline still show those
    # days (marked with a lock icon) so the user can see them in context.
    filtered_trades_df, n_excluded_by_macro, blocked_macro_keys = \
        _filter_trades_by_macro(trades_df, macro)
    perf_df_filtered = _compute_perf_df(filtered_trades_df)
    # Use filtered perf_df for all aggregate displays. The unfiltered perf_df
    # is kept only as a fallback shape reference.
    perf_df = perf_df_filtered

    # Filter state labels + macro perf-table notes — shared helper so the
    # interactive endpoint can produce identical strings.
    filter_state_label, filter_state_class, _mfn_runtime, _mtfn_runtime = \
        compute_filter_label(blocked_macro_keys, total_in_range_trades, n_excluded_by_macro)
    blocked_display_names = [
        MACRO_REGIME_DISPLAY.get(k, k) for k in
        sorted(blocked_macro_keys,
               key=lambda x: MACRO_REGIME_ORDER.index(x)
               if x in MACRO_REGIME_ORDER else 99)
    ]

    # Aggregate stats for the top-of-report summary bar
    agg_stats = _compute_aggregate_stats(filtered_trades_df)

    # ── Regime period table ─────────────────────────────────────────────────
    period_rows = []
    for p in sorted(in_range_periods, key=lambda x: x["start_ts"]):
        start_str = pd.Timestamp(p["start_ts"]).strftime("%Y-%m-%d %H:%M")
        end_str   = pd.Timestamp(p["end_ts"]).strftime("%Y-%m-%d %H:%M")
        candles   = p["end_bar"] - p["start_bar"]
        hours     = candles * 5 / 60
        if p["regime"].startswith("trending"):
            metric_a = _fmt_num(p.get("pips_per_bar"), 3) + " pips/bar"
            metric_b = _fmt_num(p.get("total_pips"), 1) + " pips total"
        elif p["regime"].startswith("ranging"):
            metric_a = _fmt_num(p.get("width_pips"), 1) + " pips wide"
            metric_b = "chop " + _fmt_num(p.get("choppiness"), 2)
        else:
            metric_a = "—"
            metric_b = "—"
        period_rows.append(
            f"<tr>"
            f"<td>{start_str}</td>"
            f"<td>{end_str}</td>"
            f"<td><span class='regime-swatch {regime_class(p['regime'])}'></span>"
            f"{REGIME_DISPLAY[p['regime']]}</td>"
            f"<td>{candles}</td>"
            f"<td>{hours:.1f}</td>"
            f"<td>{metric_a}</td>"
            f"<td>{metric_b}</td>"
            f"</tr>"
        )
    period_table = f"""
      <table class='regime-table'>
        <thead><tr>
          <th>Start</th><th>End</th><th>Regime</th>
          <th>Candles</th><th>Hours</th><th>Rate / Width</th><th>Total / Chop</th>
        </tr></thead>
        <tbody>{''.join(period_rows) or '<tr><td colspan=7 class=regime-dim>No periods in range.</td></tr>'}</tbody>
      </table>
    """

    # ── Micro regime performance ────────────────────────────────────────────
    # Show every regime that occurred in the date range (period_count > 0), so
    # the per-regime period counts that used to live in the header chip strip
    # are preserved. Trade-stat cells show '—' when the regime had no trades.
    # Pass the current blocked-micro-keys set so blocked-regime rows render
    # with the same lock+dim styling as the macro perf table.
    _blocked_micro_static = set(BLOCKED_MICRO_REGIMES)
    # Allowed macro keys = complement of BLOCKED_MACRO_REGIMES — used by the
    # perf table to scope counterfactual locked-micro stats to the active
    # strategy's macro universe.
    _allowed_macro_static = set(MACRO_REGIME_ORDER) - _blocked_macro_keys()
    perf_table = build_perf_table_html(perf_df, regime_count,
                                       blocked_micro_keys=_blocked_micro_static,
                                       trades_df=trades_df,
                                       blocked_signals_df=blocked_signals_df,
                                       allowed_macro_keys=_allowed_macro_static)

    # ── Note on trades from low-activity days ───────────────────────────────
    # Trade attribution always maps a trade to a *real* fractal's regime label
    # (no forward-filling), but on low-activity days the few fractals that
    # exist may have less reliable labels because the rolling lookback reaches
    # far back in time. Surface the count so the user knows how much of the
    # table is driven by potentially noisier classifications.
    if trades_on_low_days > 0:
        _pct = trades_on_low_days / total_in_range_trades * 100 if total_in_range_trades else 0
        perf_low_note = (
            f"<p class='regime-dim regime-small regime-perf-note'>"
            f"<span class='regime-low-activity-dot regime-low-activity-dot--inline'></span>"
            f"{trades_on_low_days} of {total_in_range_trades} trades ({_pct:.1f}%) "
            f"fired on low-activity days (fewer than "
            f"{LOW_ACTIVITY_FRACTAL_THRESHOLD} fractals that day). Trade-to-regime "
            f"attribution is correct (each trade maps to a real classified fractal), "
            f"but those classifications were drawn from a sparse lookback and may be "
            f"less reliable."
            f"</p>"
        )
    else:
        perf_low_note = ""

    # Macro-filter notes (shared helper — also used by /run_regime_analysis).
    macro_filter_note = _mfn_runtime
    macro_table_filter_note = _mtfn_runtime

    # ── Summary cards (only observed regimes) ───────────────────────────────
    cards = []
    for label in REGIME_ORDER:
        ps = [p for p in in_range_periods if p["regime"] == label]
        if not ps:
            continue
        avg_dur_candles = float(np.mean([p["end_bar"] - p["start_bar"] for p in ps]))
        avg_dur_hours   = avg_dur_candles * 5 / 60
        perf_row = perf_df[perf_df["regime"] == label].iloc[0]
        if label.startswith("trending"):
            mag_label = "Avg pips/bar"
            mag_val   = _fmt_num(np.mean([p.get("pips_per_bar", np.nan) for p in ps]), 3)
        elif label.startswith("ranging"):
            mag_label = "Avg width (pips)"
            mag_val   = _fmt_num(np.mean([p.get("width_pips", np.nan) for p in ps]), 1)
        else:
            mag_label = "—"
            mag_val   = "—"
        cards.append(f"""
          <div class='regime-card regime-summary-card'>
            <div class='regime-card-head'>
              <span class='regime-swatch {regime_class(label)}'></span>
              <h3>{REGIME_DISPLAY[label]}</h3>
            </div>
            <div class='regime-card-grid'>
              <div><span class='regime-dim regime-small'>Periods</span><strong>{len(ps)}</strong></div>
              <div><span class='regime-dim regime-small'>Avg duration</span><strong>{avg_dur_candles:.0f} candles ({avg_dur_hours:.1f} h)</strong></div>
              <div><span class='regime-dim regime-small'>{mag_label}</span><strong>{mag_val}</strong></div>
              <div><span class='regime-dim regime-small'>Trades</span><strong>{int(perf_row['trades'])}</strong></div>
              <div><span class='regime-dim regime-small'>Win rate</span><strong>{_fmt_pct(perf_row['win_rate'])}</strong></div>
              <div><span class='regime-dim regime-small'>Profit factor</span><strong>{_fmt_pf(perf_row['profit_factor'])}</strong></div>
            </div>
          </div>
        """)

    # ── Daily performance table ─────────────────────────────────────────────
    # Daily performance keeps the *unfiltered* trades_df so every day still
    # shows its real trade counts and P&L. Blocked days are marked with a
    # row-level dim + lock icon via blocked_macro_keys.
    daily_table = build_daily_breakdown(in_range_periods, trades_df, full_df,
                                        available_chart_days,
                                        in_range, low_activity_days,
                                        macro=macro,
                                        blocked_macro_keys=blocked_macro_keys)

    # ── Macro layer pieces ──────────────────────────────────────────────────
    # The macro perf table receives the *unfiltered* trades plus the set of
    # blocked regime keys, so it can show every regime's actual (or
    # counterfactual) P&L. Blocked rows are visually dimmed and locked, with
    # a counterfactual tooltip on their Total P&L cell — see
    # build_macro_perf_table for details.
    macro_perf_table  = build_macro_perf_table(
        macro, trades_df,
        blocked_macro_keys=blocked_macro_keys,
        blocked_signals_df=blocked_signals_df,
    )
    # Macro summary cards (the profile cards section) keep the filtered view
    # so they reflect the active strategy's effective trade set.
    macro_cards_html  = build_macro_summary_cards(macro, filtered_trades_df)

    # Trade counts per day for the timeline cell overlay (unfiltered).
    trades_per_day = compute_trades_per_day(trades_df)
    # Timeline section inner HTML — shared helper. The outer <section> is
    # added in the page template below so JS can replace its innerHTML on
    # interactive updates.
    timeline_inner_html = build_timeline_section_html(
        in_range_periods, macro, trades_per_day,
        REPORT_START_DATE, REPORT_END_DATE, regime_count,
    )

    # Top-of-report summary stats bar — shared helper.
    stats_bar_inner = build_stats_bar_html(agg_stats, filter_state_label, filter_state_class)

    # ── Distribution charts ─────────────────────────────────────────────────
    ppb_vals = [p["pips_per_bar"] for p in periods if p["label"] in ("trending_up", "trending_down")
                and not pd.isna(p.get("pips_per_bar"))]
    w_vals   = [p["width_pips"]   for p in periods if p["label"] == "ranging"
                and not pd.isna(p.get("width_pips"))]
    ppb_chart = chart_distribution(
        np.array(ppb_vals), thresholds["pips_per_bar_t1"], thresholds["pips_per_bar_t2"],
        "Pips per bar — trending periods", "Pips per bar",
    )
    w_chart = chart_distribution(
        np.array(w_vals), thresholds["width_pips_t1"], thresholds["width_pips_t2"],
        "Range width — ranging periods", "Width (pips)",
    )

    # ── Run bar HTML ────────────────────────────────────────────────────────
    # Mirrors the dashboard's run bar (server.py INJECT_HTML) so the two pages
    # feel like part of the same system. Visual classes (.rb-btn, .rb-date,
    # etc.) live in style.css under "Run bar — shared with dashboard".
    run_bar_html = f"""
    <div id="run-bar" class="rb-runbar">
      <!-- Structural change (May 2026): both version-select and
           instrument-select dropdowns removed from the RA toolbar. Active
           version is set exclusively on /versions and surfaced read-only
           in the top-right "Active: vN" indicator. The title row also
           reads the active version's instrument + interval (handled by
           syncRegimePageTitle below).
           May 2026 update: Run Analysis moved to the RIGHT of the date
           pickers so the date inputs lead and the action follows them —
           consistent with reading "From X To Y, then run". -->
      <!-- Task 3: native date inputs (no Mon-DD-YY overlay). -->
      <label class="rb-label" for="rb-start">From</label>
      <input type="date" id="rb-start" class="rb-date" value="{REPORT_START_DATE}">
      <label class="rb-label" for="rb-end">To</label>
      <input type="date" id="rb-end" class="rb-date" value="{REPORT_END_DATE}">
      <button id="run-analysis-btn" class="rb-btn rb-btn-green" type="button">
        <span class="rb-btn-icon">&#9654;</span> Run Analysis
      </button>
      <span id="run-status" class="rb-status"></span>
      <div class="rb-action-group">
        <!-- May 2026: Development Log icon removed from the RA run bar — the
             devlog isn't surfaced anywhere meaningful on this page, and the
             icon was a dead-end click. The /devlog endpoint + the BD's
             devlog still work; the toggleDevlogPanel handler that used to
             live in this page's IIFE was retired alongside the button. -->
        <button id="regime-copy-btn" class="rb-btn rb-btn-copy" type="button"
                title="Copy report as markdown to clipboard">
          <span class="regime-copy-btn-label">Copy Report</span>
        </button>
      </div>
    </div>
    """

    # ── Regime toggle panel HTML ────────────────────────────────────────────
    # Default state derives from BLOCKED_MACRO_REGIMES + BLOCKED_MICRO_REGIMES.
    # data-regime-key holds the internal key the endpoint expects; data-default
    # records the default ON/OFF so the Reset to Defaults button can restore.
    def _macro_key_local(name):
        """Convert a display name ('Staircase Down') or internal key
        ('staircase_down') to its canonical internal-key form."""
        return name.strip().lower().replace(" ", "_").replace("-", "_")
    _blocked_macro_default = {_macro_key_local(n) for n in BLOCKED_MACRO_REGIMES}
    _blocked_micro_default = set(BLOCKED_MICRO_REGIMES)

    def _toggle(regime_key, label, color_class, blocked_default):
        on = (regime_key not in blocked_default)
        return (
            f"<label class='regime-toggle' data-regime-key='{regime_key}' "
            f"data-default='{1 if on else 0}'>"
            f"<input type='checkbox' class='regime-toggle-input' {'checked' if on else ''}>"
            f"<span class='regime-toggle-swatch {color_class}'></span>"
            f"<span class='regime-toggle-label'>{label}</span>"
            f"<span class='regime-toggle-switch'></span>"
            f"</label>"
        )

    macro_toggle_html = "".join(
        _toggle(key, MACRO_REGIME_DISPLAY[key], macro_class(key), _blocked_macro_default)
        for key in MACRO_REGIME_ORDER
    )
    micro_toggle_html = "".join(
        _toggle(key, REGIME_DISPLAY[key], regime_class(key), _blocked_micro_default)
        for key in REGIME_ORDER
    )

    control_panel_html = f"""
    <section class="regime-control-panel">
      <div class="regime-control-header">
        <h2>Regime filters</h2>
        <button id="regime-reset-btn" class="rb-btn rb-btn-ghost" type="button"
                title="Restore all toggles to their default blocked/allowed state">
          Reset to Defaults
        </button>
      </div>
      <div class="regime-control-grid">
        <div class="regime-control-col">
          <h3 class="regime-control-col-title">Macro Regimes</h3>
          <div class="regime-toggle-list" id="regime-macro-toggles">
            {macro_toggle_html}
          </div>
        </div>
        <div class="regime-control-col">
          <h3 class="regime-control-col-title">Micro Regimes</h3>
          <div class="regime-toggle-list" id="regime-micro-toggles">
            {micro_toggle_html}
          </div>
        </div>
      </div>
    </section>
    """

    html = f"""<!doctype html>
<html lang="en" id="main">
<head>
  <meta charset="utf-8">
  <title>Regime Analysis — GBPUSD 5m</title>
  <link rel="stylesheet" href="../style.css">
  <!-- Material Symbols — same icon font the dashboard sidebar uses for the
       preview eye icon. Variable axes (FILL/wght/GRAD/opsz) enabled so the
       icon can morph from outlined → filled on hover. -->
  <link rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
</head>
<body class="regime-report">
  <nav class="top-nav" id="top-nav">
    <ul class="top-nav-items">
      <li><a class="top-nav-link" href="/">Backtesting</a></li>
      <li><a class="top-nav-link top-nav-link-active" href="/results/regime_analysis.html">Regimes</a></li>
      <li><a class="top-nav-link" href="/discovery">Discovery</a></li>
      <li><a class="top-nav-link" href="/versions">Versions</a></li>
      <li><a class="top-nav-link" href="/docs/" target="_blank" rel="noopener noreferrer">Docs</a></li>
    </ul>
    <span class="top-nav-active-version" id="top-nav-active-version"></span>
  </nav>

  {run_bar_html}

  <div class="regime-container">

    <header class="regime-header">
      <div class="regime-header-top">
        <h1 id="regime-page-title">Regime Analysis — GBPUSD 5m</h1>
        <button class="bs-toggle-btn" id="regime-filters-toggle-btn"
                title="Show / hide regime filters" type="button">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <circle cx="8" cy="8" r="7" stroke="currentColor" stroke-width="1.5"/>
            <path d="M5.5 7L8 9.5L10.5 7" stroke="currentColor" stroke-width="1.5"
                  stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <div class="regime-header-meta">
        <span><strong>Range:</strong> {REPORT_START_DATE} → {REPORT_END_DATE}</span>
        <span><strong>Fractals:</strong> {len(in_range)}</span>
        <span><strong>Periods:</strong> {len(in_range_periods)}</span>
        <span><strong>Lookback:</strong> {LOOKBACK_FRACTALS}</span>
        <span><strong>Low activity:</strong>
          <span class='regime-low-activity-dot regime-low-activity-dot--inline'></span>
          {low_activity_count} of {total_days} days ({low_activity_pct:.0f}%)
        </span>
        <span class="regime-dim">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
      </div>
    </header>

    <div class="bs-collapsible" id="regime-filters-collapsible">
      {control_panel_html}
    </div>

    <section class="regime-stats-bar" id="regime-stats-section">
      {stats_bar_inner}
    </section>

    <section class="regime-card" id="regime-macro-perf-section">
      <h2>Macro regime performance <span class="regime-dim regime-small">(daily context)</span></h2>

      {macro_perf_table}
    </section>

    <section class="regime-card" id="regime-perf-section">
      <h2>Micro regime performance <span class="regime-dim regime-small">(v2 short-only)</span></h2>
      {perf_table}
      {perf_low_note}
    </section>

    <section class="regime-card" id="regime-timeline-section">
      {timeline_inner_html}
    </section>

    <section class="regime-card" id="regime-daily-section">
      <h2>Daily performance</h2>
      {daily_table}
      <p class="regime-dim regime-small regime-breakdown-note">
        <span class="regime-hour-chip regime-color-inactive regime-hour-chip--inline"></span>
        Dark chips indicate hours where <strong>no fractal was detected</strong> —
        not hours covered by a forward-filled regime label.
        &nbsp;&nbsp;
        <span class="regime-low-activity-dot regime-low-activity-dot--inline"></span>
        Indicates a low-activity day (fewer than {LOW_ACTIVITY_FRACTAL_THRESHOLD}
        fractals across the whole 24-hour period); regime labels on those days
        may be less reliable.
      </p>
    </section>

    <section id="regime-macro-profiles-section">
      <h2>Macro Regime Profiles</h2>
      <div class="regime-summary-grid">{macro_cards_html}</div>
    </section>

    <section id="regime-summary-cards-section">
      <h2>Regime summary cards</h2>
      <div class="regime-summary-grid">{''.join(cards) or '<p class=regime-dim>No regimes observed.</p>'}</div>
    </section>

    <section id="regime-thresholds-section" class="regime-card">
      <h2>Threshold distributions</h2>
      <p class="regime-dim regime-small">
        Tercile cuts (dashed lines) are taken from the actual observed distributions
        of pips-per-bar across all trending periods and range-width across all
        ranging periods.
      </p>
      <div class="regime-chart-row">
        <img alt="Pips per bar distribution" src="data:image/png;base64,{ppb_chart}" />
        <img alt="Range width distribution" src="data:image/png;base64,{w_chart}" />
      </div>
    </section>

    <section id="regime-periods-section" class="regime-card">
      <h2>Regime periods</h2>
      {period_table}
    </section>

  </div>

  <!-- Hover preview overlay — mirrors the dashboard's #chart-preview-overlay
       (styles already in style.css). Image src is set dynamically by JS to
       the regime_charts/YYYY-MM-DD.png file for the hovered row.
       The #chart-preview-chips slot above the image gets the hovered row's
       hour-chip strip cloned into it, stretched to the full card width so
       the chips line up visually with the chart's x-axis. -->
  <div id="chart-preview-overlay" aria-hidden="true">
    <div id="chart-preview-card">
      <div id="chart-preview-chips"></div>
      <img id="chart-preview-img" alt="Daily chart preview"/>
    </div>
  </div>

  <script>
  (function() {{
    // ── Chart hover preview ────────────────────────────────────────────────
    var overlay      = document.getElementById("chart-preview-overlay");
    var img          = document.getElementById("chart-preview-img");
    var chipsSlot    = document.getElementById("chart-preview-chips");

    function showChartPreview(src, chipsHtml) {{
      if (!overlay || !img) return;
      if (!src) {{
        img.removeAttribute("src");
        overlay.classList.add("no-chart");
      }} else {{
        img.src = src;
        overlay.classList.remove("no-chart");
      }}
      if (chipsSlot) chipsSlot.innerHTML = chipsHtml || "";
      overlay.classList.add("visible");
      overlay.setAttribute("aria-hidden", "false");
    }}

    function hideChartPreview() {{
      if (!overlay) return;
      overlay.classList.remove("visible");
      overlay.setAttribute("aria-hidden", "true");
      if (img) img.removeAttribute("src");
      if (chipsSlot) chipsSlot.innerHTML = "";
    }}

    // (Chart-preview + sortable-table handlers moved to attachDailyHandlers
    // below so they can be re-attached after a Run Analysis innerHTML swap.)

    // ── Copy Report — DOM → markdown ───────────────────────────────────────
    // Walks the page in document order and emits markdown for every section
    // from the header through "Threshold distributions" (inclusive). The
    // "Regime periods" section that follows is intentionally skipped.
    var copyBtn = document.getElementById("regime-copy-btn");
    if (copyBtn) {{
      copyBtn.addEventListener("click", function () {{
        var md = buildRegimeMarkdown();
        if (!navigator.clipboard || !navigator.clipboard.writeText) {{
          // Fallback for non-secure contexts
          var ta = document.createElement("textarea");
          ta.value = md;
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          try {{ document.execCommand("copy"); }} catch (e) {{}}
          document.body.removeChild(ta);
          flashCopied();
          return;
        }}
        navigator.clipboard.writeText(md).then(flashCopied).catch(function () {{
          var lbl = copyBtn.querySelector(".regime-copy-btn-label");
          if (lbl) lbl.textContent = "Failed";
          setTimeout(function () {{
            if (lbl) lbl.textContent = "Copy Report";
          }}, 2000);
        }});
      }});
    }}

    function flashCopied() {{
      var lbl = copyBtn.querySelector(".regime-copy-btn-label");
      copyBtn.classList.add("copied");
      if (lbl) lbl.textContent = "Copied!";
      setTimeout(function () {{
        copyBtn.classList.remove("copied");
        if (lbl) lbl.textContent = "Copy Report";
      }}, 2000);
    }}

    function tidy(s) {{
      return (s || "").replace(/\\s+/g, " ").trim();
    }}

    function renderTableMd(table, skipCols, lines) {{
      skipCols = skipCols || [];
      var headerRow = table.querySelector("thead tr");
      if (!headerRow) return;
      var ths = Array.prototype.slice.call(headerRow.querySelectorAll("th"));
      var keptIdx = [];
      var headerCells = [];
      ths.forEach(function (th, idx) {{
        if (skipCols.indexOf(idx) !== -1) return;
        var t = tidy(th.textContent);
        if (!t) return;
        keptIdx.push(idx);
        headerCells.push(t);
      }});
      if (headerCells.length === 0) return;
      lines.push("| " + headerCells.join(" | ") + " |");
      lines.push("|" + headerCells.map(function () {{ return "---"; }}).join("|") + "|");

      var bodyRows = table.querySelectorAll("tbody tr");
      bodyRows.forEach(function (tr) {{
        var tds = tr.querySelectorAll("td");
        // Empty placeholder rows (single colspan cell) — skip
        if (tds.length === 1 && tds[0].getAttribute("colspan")) return;
        var cells = keptIdx.map(function (i) {{
          if (i >= tds.length) return "";
          return tidy(tds[i].textContent).replace(/\\|/g, "\\\\|");
        }});
        if (cells.every(function (c) {{ return c === ""; }})) return;
        lines.push("| " + cells.join(" | ") + " |");
      }});
    }}

    function renderCardsMd(cards, lines) {{
      if (!cards || cards.length === 0) return;
      cards.forEach(function (card) {{
        var head = card.querySelector(".regime-card-head");
        var title = head ? tidy(head.textContent) : "";
        if (title) lines.push("### " + title);
        var grid = card.querySelectorAll(".regime-card-grid > div");
        grid.forEach(function (div) {{
          var kids = div.children;
          if (kids.length >= 2) {{
            var lbl = tidy(kids[0].textContent);
            var val = tidy(kids[1].textContent);
            lines.push("- **" + lbl + ":** " + val);
          }}
        }});
        lines.push("");
      }});
    }}

    function buildRegimeMarkdown() {{
      var lines = [];

      // ── Title ──────────────────────────────────────────────────────────
      var h1 = document.querySelector(".regime-header h1");
      if (h1) {{
        lines.push("# " + tidy(h1.textContent));
        lines.push("");
      }}

      // ── Header meta (Range / Fractals / Periods / Lookback / Low / Generated)
      var metaSpans = document.querySelectorAll(".regime-header-meta > span");
      metaSpans.forEach(function (s) {{
        var t = tidy(s.textContent);
        if (t) lines.push("- " + t);
      }});
      if (metaSpans.length) lines.push("");

      // ── Summary Stats bar ──────────────────────────────────────────────
      var statsBar = document.querySelector(".regime-stats-bar");
      if (statsBar) {{
        lines.push("## Summary Stats");
        lines.push("");
        var statCards = statsBar.querySelectorAll(".regime-stat-card");
        if (statCards.length) {{
          lines.push("| Metric | Value |");
          lines.push("|--------|-------|");
          statCards.forEach(function (card) {{
            var lbl = card.querySelector(".regime-stat-label");
            var val = card.querySelector(".regime-stat-value");
            if (lbl && val) {{
              lines.push("| " + tidy(lbl.textContent) + " | " + tidy(val.textContent) + " |");
            }}
          }});
          lines.push("");
        }}
        var filterLbl = statsBar.querySelector(".regime-stats-filter-label");
        if (filterLbl) {{
          lines.push("_" + tidy(filterLbl.textContent) + "_");
          lines.push("");
        }}
      }}

      // ── Walk sections in order, stop after "Threshold distributions" ──
      var sections = document.querySelectorAll(".regime-container > section");
      for (var i = 0; i < sections.length; i++) {{
        var section = sections[i];
        var h2 = section.querySelector("h2");
        if (!h2) continue;  // Stats bar (no h2) is handled above

        var title = tidy(h2.textContent);
        lines.push("## " + title);
        lines.push("");

        // Intro paragraph (first <p> directly under the section, not a note)
        var introP = section.querySelector(":scope > p:not(.regime-perf-note):not(.regime-breakdown-note)");
        if (introP) {{
          var introText = tidy(introP.textContent);
          if (introText) {{
            lines.push(introText);
            lines.push("");
          }}
        }}

        // Tables — skip non-textual columns for the daily-breakdown table
        var tables = section.querySelectorAll("table.regime-table");
        tables.forEach(function (table) {{
          var isDaily = table.classList.contains("regime-daily-table");
          // Daily table columns: 0=Date 1=Macro 2=Hour chips 3=Trades 4=Wins 5=P&L 6=Chart
          var skipCols = isDaily ? [2, 6] : [];
          renderTableMd(table, skipCols, lines);
          lines.push("");
        }});

        // Summary cards
        var grid = section.querySelector(".regime-summary-grid");
        if (grid) {{
          renderCardsMd(grid.querySelectorAll(".regime-summary-card"), lines);
        }}

        // Notes (filter note, low-activity note, breakdown note)
        var notes = section.querySelectorAll("p.regime-perf-note, p.regime-breakdown-note");
        notes.forEach(function (p) {{
          var t = tidy(p.textContent);
          if (t) {{
            lines.push("_" + t + "_");
            lines.push("");
          }}
        }});

        if (/Threshold\\s+distributions/i.test(title)) break;
      }}

      return lines.join("\\n").replace(/\\n{{3,}}/g, "\\n\\n").trim() + "\\n";
    }}

    // ── Title sync: reflect the active version's instrument + interval ─────
    // Structural change (May 2026): instrument + interval are now stored on
    // each version (versions.json → v.params.instrument / v.params.interval)
    // instead of being driven by a free-floating instrument dropdown. The
    // RA title row mirrors the BD title row format: "v7 · GBPUSD · 5m".
    // window.__activeVersion is populated by the run-bar select wiring once
    // /api/versions resolves; until then we fall back to localStorage hints
    // and finally to GBPUSD / 5m defaults so the page never renders blank.
    function syncRegimePageTitle() {{
      var v = window.__activeVersion || null;
      var vp = (v && v.params) || {{}};
      var vName = (v && (v.name || v.id)) || "";
      var inst = (vp.instrument || localStorage.getItem("rb_instrument") || "GBPUSD").trim() || "GBPUSD";
      var interval = (vp.interval || localStorage.getItem("bs_interval") || "5m").trim() || "5m";
      var parts = [];
      if (vName) parts.push(vName);
      parts.push(inst);
      parts.push(interval);
      var headline = "Regime Analysis — " + parts.join(" · ");
      document.title = headline;
      var h1 = document.getElementById("regime-page-title");
      if (h1) h1.textContent = headline;
    }}
    syncRegimePageTitle();
    window.addEventListener("storage", function (e) {{
      if (e && (e.key === "rb_instrument" || e.key === "bs_interval")) {{
        syncRegimePageTitle();
      }}
    }});

    // ── Toggle panel: collect allow-lists + Reset to Defaults ──────────────
    function collectAllowed(containerId) {{
      var container = document.getElementById(containerId);
      if (!container) return [];
      var allowed = [];
      container.querySelectorAll(".regime-toggle").forEach(function (toggle) {{
        var input = toggle.querySelector(".regime-toggle-input");
        if (input && input.checked) {{
          allowed.push(toggle.getAttribute("data-regime-key"));
        }}
      }});
      return allowed;
    }}

    function applyToggleVisual(toggle) {{
      var input = toggle.querySelector(".regime-toggle-input");
      if (input && input.checked) {{
        toggle.classList.remove("regime-toggle-off");
      }} else {{
        toggle.classList.add("regime-toggle-off");
      }}
    }}

    // Task 5: track when the toggle UI diverges from the last-rendered
    // state so the lock icons in the tables don't appear to lie. A user
    // can toggle a checkbox without clicking Run Analysis; the visible
    // lock icons still reflect the LAST run's blocked_macro_keys until
    // the next Run lands. We add `.regime-stale` to the body to dim the
    // sections + show a "click Run to refresh" hint, cleared on the next
    // successful run.
    function markStale() {{
      document.body.classList.add("regime-stale");
    }}
    function clearStale() {{
      document.body.classList.remove("regime-stale");
    }}
    document.querySelectorAll(".regime-toggle").forEach(function (toggle) {{
      applyToggleVisual(toggle);
      var input = toggle.querySelector(".regime-toggle-input");
      if (input) {{
        input.addEventListener("change", function () {{
          applyToggleVisual(toggle);
          markStale();
        }});
      }}
    }});

    // Bug fix (May 2026 — Issue 4): Reset to Defaults used to read the
    // `data-default` attribute baked into each toggle at HTML render
    // time. Those defaults come from the file-level BLOCKED_MACRO_REGIMES
    // and BLOCKED_MICRO_REGIMES constants — the labeler's hardcoded
    // defaults, which have nothing to do with the active version's
    // Discovery-assigned allow-lists. Symptom: after running with
    // modified toggles, clicking Reset snapped the toggles to the file
    // defaults (e.g. macro=staircase_down+strong_down, micro=ranging_
    // medium+ranging_wide) instead of restoring the version's Discovery
    // params (which is what users mean by "defaults" once a version is
    // bound to a Discovery trial).
    //
    // Fix: Reset reads `_versionParamsDefaults` (populated below from
    // `activeV.params.allowed_*` — the IMMUTABLE assignment record, not
    // the mutable `regime_state` that gets overwritten on every Run). If
    // the active version has no params (e.g. a freshly-created v1 with
    // no Discovery assignment), we fall back to `data-default` to
    // preserve the previous behaviour for unassigned versions.
    var _versionParamsDefaults = null;  // {{macro: [...], micro: [...]}}

    var resetBtn = document.getElementById("regime-reset-btn");
    if (resetBtn) {{
      resetBtn.addEventListener("click", function () {{
        if (_versionParamsDefaults
            && _versionParamsDefaults.macro
            && _versionParamsDefaults.micro) {{
          // Discovery-assigned version: reset to the immutable params.
          var mAllowed = {{}};
          _versionParamsDefaults.macro.forEach(function (k) {{ mAllowed[k] = true; }});
          var uAllowed = {{}};
          _versionParamsDefaults.micro.forEach(function (k) {{ uAllowed[k] = true; }});
          document.querySelectorAll("#regime-macro-toggles .regime-toggle").forEach(function (toggle) {{
            var k = toggle.getAttribute("data-regime-key");
            var input = toggle.querySelector(".regime-toggle-input");
            if (input) input.checked = !!mAllowed[k];
            applyToggleVisual(toggle);
          }});
          document.querySelectorAll("#regime-micro-toggles .regime-toggle").forEach(function (toggle) {{
            var k = toggle.getAttribute("data-regime-key");
            var input = toggle.querySelector(".regime-toggle-input");
            if (input) input.checked = !!uAllowed[k];
            applyToggleVisual(toggle);
          }});
        }} else {{
          // Unassigned version or /api/versions not yet resolved: fall
          // back to the file-level defaults baked into data-default.
          document.querySelectorAll(".regime-toggle").forEach(function (toggle) {{
            var def = toggle.getAttribute("data-default") === "1";
            var input = toggle.querySelector(".regime-toggle-input");
            if (input) input.checked = def;
            applyToggleVisual(toggle);
          }});
        }}
        // Toggles now diverge from the last-rendered stats — match the
        // change-handler at line ~2970 so the report sections dim and
        // prompt for a fresh Run.
        markStale();
      }});
    }}

    // ── Run Analysis: POST to /run_regime_analysis, swap section innerHTML ─
    var runBtn = document.getElementById("run-analysis-btn");
    var runStatus = document.getElementById("run-status");

    // Safety watchdog — if setRunning(true) is called and nothing ever
    // calls setRunning(false) (e.g. the promise chain hangs on a giant
    // response or the browser stalls during innerHTML rendering), this
    // timer forcibly resets the button after a hard cap so the user can
    // always retry. Cleared on every clean transition.
    var _runWatchdog = null;
    function setRunning(yes, silent) {{
      if (!runBtn) return;
      if (_runWatchdog) {{ clearTimeout(_runWatchdog); _runWatchdog = null; }}
      if (yes) {{
        runBtn.disabled = true;
        // Silent auto-load: keep the button label calm — the user didn't
        // click anything, so a spinning "Running…" label would be misleading.
        if (!silent) {{
          runBtn.innerHTML = "<span class='rb-spin'></span> Running…";
        }}
        if (runStatus) runStatus.textContent = silent ? "" : "";
        _runWatchdog = setTimeout(function () {{
          runBtn.disabled = false;
          runBtn.innerHTML = "<span class='rb-btn-icon'>&#9654;</span> Run Analysis";
          if (runStatus) runStatus.textContent = "Timed out — click to retry";
          _runWatchdog = null;
        }}, 5 * 60 * 1000);
      }} else {{
        runBtn.disabled = false;
        runBtn.innerHTML = "<span class='rb-btn-icon'>&#9654;</span> Run Analysis";
      }}
    }}

    function runAnalysis(opts) {{
      if (!runBtn) return;
      // If a Run is already in progress (button disabled), ignore further
      // clicks. The simplest way to prevent the race conditions we kept
      // hitting — no AbortController, no token comparison, no superseded
      // response logic. The user just waits for the current run to finish.
      if (runBtn.disabled) {{
        if (window.console) console.log("[RA] ignoring click — run already in progress");
        return;
      }}
      opts = opts || {{}};
      var isAutoLoad = !!opts.silent;
      var startEl = document.getElementById("rb-start");
      var endEl   = document.getElementById("rb-end");
      var startVal = startEl ? startEl.value : "";
      var endVal   = endEl   ? endEl.value   : "";
      if (!startVal || !endVal) {{
        if (runStatus) runStatus.textContent = "Set start + end dates";
        return;
      }}

      // Instrument follows the BD's selection. The BD persists it to
      // localStorage `rb_instrument`; we read the same key here so a
      // backtest re-run from the RA page targets the same instrument data.
      var instVal = (localStorage.getItem("rb_instrument") || "GBPUSD").trim() || "GBPUSD";
      var payload = {{
        instrument: instVal,
        start_date: startVal,
        end_date:   endVal,
        allowed_macro_regimes: collectAllowed("regime-macro-toggles"),
        allowed_micro_regimes: collectAllowed("regime-micro-toggles"),
      }};

      var scrollY = window.scrollY;
      setRunning(true, isAutoLoad);

      // Diagnostic prefix — filter dev-tools console by "[RA]" to see
      // exactly which step the request reached if it ever appears stuck.
      var _log = (window.console && console.log)
        ? function (msg, extra) {{ console.log("[RA] " + msg, extra === undefined ? "" : extra); }}
        : function () {{}};

      // ── Single-shot finish guard ─────────────────────────────────────
      // `done` ensures the button is re-enabled exactly once regardless
      // of which promise branch settles first (or if both somehow fire,
      // or if the watchdog fires before the response). No AbortController,
      // no token bookkeeping — just a flag. The orphaned fetch (if any)
      // will eventually complete server-side but its handlers will see
      // done=true and bail.
      var done = false;
      function finish(statusText) {{
        if (done) return;
        done = true;
        clearTimeout(safetyTimer);
        setRunning(false);
        if (statusText !== undefined && runStatus) runStatus.textContent = statusText;
        _log("finished  statusText=" + JSON.stringify(statusText));
      }}

      // 5-minute safety timer — last-resort guarantee that the button
      // re-enables even if every other code path fails. /run_regime_analysis
      // typically completes in 30-60s, so 5 min is well above the worst
      // case — but covers genuine server hangs.
      var safetyTimer = setTimeout(function () {{
        finish("Timed out — server didn't respond within 5 minutes");
      }}, 5 * 60 * 1000);

      _log("fetch start  payload=", payload);
      if (runStatus) runStatus.textContent = "Sending…";

      var fetchOpts = {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload),
      }};
      // No AbortController. Some browsers have quirky AbortController
      // behavior around POST with JSON bodies; relying on the safety
      // timer + button-disabled guard is simpler and more reliable.

      fetch("/run_regime_analysis", fetchOpts).then(function (r) {{
        _log("response  status=" + r.status + "  ok=" + r.ok);
        if (done) return null;  // safety timer already fired
        if (runStatus) runStatus.textContent = "Parsing…";
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      }}).then(function (data) {{
        if (done || data === null) return;  // either superseded by timeout or skipped above
        _log("json parsed  keys=", data ? Object.keys(data) : data);
        if (data && data.error) throw new Error(data.error);
        if (!data || typeof data !== "object") throw new Error("Empty response");
        var htmlChunks = {{
          stats_bar:   data.stats_bar,
          macro_perf:  data.macro_perf,
          regime_perf: data.regime_perf,
          timeline:    data.timeline,
          daily:       data.daily,
        }};
        // Save first (cheap, just JSON.stringify + localStorage write).
        if (runStatus) runStatus.textContent = "Saving…";
        try {{ saveRegimeAnalysisState(payload, htmlChunks); _log("save ok"); }}
        catch (e) {{ console.warn("[RA] save failed:", e); }}

        // Render in a deferred microtask so the browser repaints the
        // "Rendering…" status text before the (potentially slow)
        // synchronous innerHTML assignment locks the main thread. The
        // button stays disabled during render — we call finish() from
        // INSIDE the deferred callback so the button only re-enables
        // after the user can actually see the new stats.
        if (runStatus) runStatus.textContent = "Rendering…";
        setTimeout(function () {{
          if (done) return;
          try {{
            applyResponseHtml(htmlChunks);
            window.scrollTo(0, scrollY);
            // Task 5: tables now reflect current toggle state — clear stale flag.
            if (typeof clearStale === "function") clearStale();
            _log("render ok");
            finish("");
          }} catch (e) {{
            console.error("[RA] render failed:", e);
            finish("Render failed: " + (e.message || e));
          }}
        }}, 0);
      }}).catch(function (err) {{
        console.warn("[RA] fetch failed:", err);
        finish("Failed: " + ((err && err.message) || String(err)));
      }});
      // No .finally — `finish()` is the single, idempotent point of
      // truth. .finally + .catch + .then all calling setRunning(false)
      // in various conditional branches was the source of the "stuck
      // button" bug because one branch could quietly skip the reset.
    }}

    if (runBtn) runBtn.addEventListener("click", function () {{ runAnalysis(); }});

    // ── localStorage persistence (per active version) ─────────────────────
    // Each active version gets its OWN cache slot under a versioned key:
    //
    //   regime_analysis.lastAnalysis.v3.<active_version_id>
    //     → {{payload, html}}
    //
    // This way, switching the active version in the BD and returning here
    // restores that version's last-run state (date range + toggles + cached
    // html) without bleed-through from another version. The legacy
    // single-key schema (regime_analysis.lastAnalysis.v2) is still read as
    // a one-time migration fallback and re-saved under the new per-version
    // key the first time we find it.
    //
    // Older `regime_analysis.lastAnalysis.v1` / `regime_labeler.lastAnalysis.v1`
    // keys are no longer migrated — they predate per-version semantics and
    // weren't stamped with an active version id, so we have no way to
    // assign them to a slot.
    var REGIME_LS_PREFIX = "regime_analysis.lastAnalysis.v3.";
    var REGIME_LS_KEY_V2 = "regime_analysis.lastAnalysis.v2";   // legacy schema-v2

    // Set once /api/active_version resolves below. All save/load/clear
    // helpers key off this id so each active version has its own slot.
    var _currentActiveVersionId = null;

    function regimeLSKey(versionId) {{
      return REGIME_LS_PREFIX + (versionId || "");
    }}

    function saveRegimeAnalysisState(payload, htmlChunks) {{
      if (!_currentActiveVersionId) {{
        // Active version hasn't been resolved yet — usually means the
        // user clicked Run before /api/active_version returned. Warn so
        // the failure isn't silent and the user knows the run worked
        // but the per-version cache slot wasn't written.
        if (window.console && console.warn) {{
          console.warn("RA save skipped: active version id not yet known");
        }}
        return;
      }}
      try {{
        var state = {{payload: payload, html: htmlChunks || null}};
        localStorage.setItem(regimeLSKey(_currentActiveVersionId), JSON.stringify(state));
      }} catch (e) {{
        if (window.console && console.warn) {{
          console.warn("RA save failed (quota / privacy?):", e);
        }}
      }}
    }}

    function loadRegimeAnalysisState(versionId) {{
      if (!versionId) return null;
      try {{
        // Canonical per-version slot.
        var raw = localStorage.getItem(regimeLSKey(versionId));
        if (raw) {{
          var s = JSON.parse(raw);
          if (s && s.payload && s.payload.start_date && s.payload.end_date) {{
            return s;
          }}
        }}
        // Legacy schema-v2 fallback: a single key with active_version_id
        // stored INSIDE the value. Restore only if the stamp matches the
        // requested version, then migrate onto the per-version key so we
        // don't fall through to legacy on every load.
        var legacy = localStorage.getItem(REGIME_LS_KEY_V2);
        if (legacy) {{
          var ls = JSON.parse(legacy);
          if (ls && ls.payload && ls.payload.start_date && ls.payload.end_date
              && ls.active_version_id === versionId) {{
            var migrated = {{payload: ls.payload, html: ls.html || null}};
            try {{
              localStorage.setItem(regimeLSKey(versionId), JSON.stringify(migrated));
            }} catch (e) {{}}
            return migrated;
          }}
        }}
      }} catch (e) {{}}
      return null;
    }}

    // Reset to Defaults: clear only the current active version's slot.
    // Other versions' cached state is preserved.
    function clearRegimeAnalysisState() {{
      if (!_currentActiveVersionId) return;
      try {{
        localStorage.removeItem(regimeLSKey(_currentActiveVersionId));
      }} catch (e) {{}}
    }}

    // Swap each known section's innerHTML from a response object.
    // Synchronous and simple — the previous chunked-callback variant
    // had too many silent-failure paths and didn't actually reduce the
    // browser hang (each section's innerHTML is still synchronous; only
    // the gap between sections yielded, which is negligible for the
    // typical response size). If a single section's html is genuinely
    // pathological, the try/catch isolates the failure so the others
    // still render.
    var _SECTION_PAIRS = [
      ["stats_bar",   "regime-stats-section"],
      ["macro_perf",  "regime-macro-perf-section"],
      ["regime_perf", "regime-perf-section"],
      ["timeline",    "regime-timeline-section"],
      ["daily",       "regime-daily-section"],
    ];
    function applyResponseHtml(html) {{
      for (var s = 0; s < _SECTION_PAIRS.length; s++) {{
        var k = _SECTION_PAIRS[s][0], id = _SECTION_PAIRS[s][1];
        try {{
          if (typeof html[k] === "string") {{
            var el = document.getElementById(id);
            if (el) el.innerHTML = html[k];
          }}
        }} catch (e) {{
          if (window.console && console.warn) {{
            console.warn("RA section render failed:", k, e);
          }}
        }}
      }}
      try {{ attachDailyHandlers(); }} catch (e) {{
        if (window.console && console.warn) console.warn("attachDailyHandlers failed:", e);
      }}
    }}

    function applySavedStateToControls(saved) {{
      // Date pickers — restored from the last manually-set range so
      // navigating away and returning resumes the same window the user
      // last ran. Paired with painting the cached html chunks further
      // down (deferred until /api/active_version confirms the cache
      // belongs to the currently-active version), the displayed stats
      // stay consistent with the restored date range — no silent
      // disagreement between pickers and the numbers on screen.
      //
      // Task 3: native date inputs — straight `.value` assignment now
      // works because there's no overlay to keep in sync.
      var _s = document.getElementById("rb-start");
      var _e = document.getElementById("rb-end");
      if (_s && saved && saved.start_date) _s.value = saved.start_date;
      if (_e && saved && saved.end_date)   _e.value = saved.end_date;
      //
      // Instrument — restored to localStorage `rb_instrument` so any
      // subsequent runs (and the page title) reflect what this version
      // was last run against. If the BD's run-bar has an instrument
      // dropdown rendered, sync its visible value too.
      if (saved && saved.instrument) {{
        try {{ localStorage.setItem("rb_instrument", saved.instrument); }} catch (e) {{}}
        var instSel = document.getElementById("instrument-select");
        if (instSel) instSel.value = saved.instrument;
        if (typeof syncRegimePageTitle === "function") syncRegimePageTitle();
      }}
      //
      // Toggle checkboxes — drive from saved allowed-lists. (These then
      // get authoritatively overridden by the active-version-sync block
      // shortly after, so they're effectively a no-op visually — but we
      // keep this for backwards-compat with the saved-state schema.)
      function applyAllowList(containerId, allowed) {{
        var container = document.getElementById(containerId);
        if (!container) return;
        var allowedSet = {{}};
        (allowed || []).forEach(function (k) {{ allowedSet[k] = true; }});
        container.querySelectorAll(".regime-toggle").forEach(function (toggle) {{
          var key = toggle.getAttribute("data-regime-key");
          var input = toggle.querySelector(".regime-toggle-input");
          if (!input) return;
          input.checked = !!allowedSet[key];
          applyToggleVisual(toggle);
        }});
      }}
      applyAllowList("regime-macro-toggles", saved.allowed_macro_regimes);
      applyAllowList("regime-micro-toggles", saved.allowed_micro_regimes);
    }}

    // Saved state is loaded INSIDE the active-version handler below — not
    // here — because we need to know the active version id first to pick
    // the right per-version localStorage slot. Brief consequence: the date
    // pickers show the server-rendered full-range defaults until the
    // /api/active_version round-trip resolves (<100ms on localhost), then
    // they snap to the last-saved range for the now-known active version.
    //
    // We still deliberately do NOT auto-fire a runAnalysis on page load:
    // it would disable the Run button for the duration of a heavy backend
    // call, which feels like the page is broken when the user just
    // wanted to view or tweak toggles. The user clicks Run Analysis when
    // they want fresh stats.
    var _savedState = null;

    // ── Active-version sync ────────────────────────────────────────────────
    // versions.json (server-side) is the source of truth for toggle state.
    // On load we fetch the active version, populate the top-nav indicator,
    // restore THIS version's cached state (date range + toggles + html)
    // from its own per-version localStorage slot, then override the
    // toggle checkboxes from versions.json's regime_state (authoritative).
    // Each active version has its own cache slot, so switching versions
    // in the BD and returning here restores the matching version's state
    // — no bleed-through, no stale-html clear pass needed.
    //
    // Deliberately NOT firing a silent analysis run here: doing so would
    // disable the Run button for the duration of a heavy backend call,
    // which feels broken when the user just wanted to view or tweak
    // toggles. Clicking Run Analysis pulls fresh stats.
    fetch("/api/active_version").then(function (r) {{ return r.json(); }})
      .then(function (resp) {{
        var indEl = document.getElementById("top-nav-active-version");
        if (indEl && resp && resp.ok && resp.active) {{
          indEl.textContent = "Active: " + resp.active.name;
        }}
        if (resp && resp.ok && resp.active && resp.active.id) {{
          _currentActiveVersionId = resp.active.id;
        }}
        // Load this version's cached state and restore date pickers +
        // toggles. The cached html is painted ONLY if its saved
        // allow-lists still match versions.json's current regime_state —
        // otherwise (e.g. the user manually edited versions.json or
        // another tab wrote different toggles) the cached lock icons and
        // stats would reflect a stale toggle state, visually disagreeing
        // with the freshly-set toggle UI we're about to render below.
        // That mismatch was Bug 4's "lock icons inverted from toggles"
        // and contributed to Bug 1's "same filter set, different stats".
        _savedState = loadRegimeAnalysisState(_currentActiveVersionId);
        if (_savedState) {{
          applySavedStateToControls(_savedState.payload);
          if (_savedState.html && resp && resp.active && resp.active.regime_state) {{
            var _curMacro = ((resp.active.regime_state.allowed_macro_regimes || []).slice().sort()).join(",");
            var _curMicro = ((resp.active.regime_state.allowed_micro_regimes || []).slice().sort()).join(",");
            var _savMacro = ((_savedState.payload.allowed_macro_regimes || []).slice().sort()).join(",");
            var _savMicro = ((_savedState.payload.allowed_micro_regimes || []).slice().sort()).join(",");
            if (_savMacro === _curMacro && _savMicro === _curMicro) {{
              applyResponseHtml(_savedState.html);
            }}
            // else: leave the server-rendered static sections in place;
            // user must click Run Analysis to refresh against current toggles.
          }} else if (_savedState.html) {{
            // No regime_state to compare against — paint optimistically.
            applyResponseHtml(_savedState.html);
          }}
        }}
        if (!resp || !resp.ok || !resp.active) return;
        // Issue 5 fix: page-init toggles now reflect the version's
        // Discovery defaults (params.allowed_*), NOT the mutable
        // regime_state. Edward's expectation is that the page always
        // opens at the version's Discovery baseline — opening at the
        // last-run state (regime_state) was confusing because every Run
        // drifts it away from the assignment.
        // Fallback chain matches the _versionParamsDefaults populator:
        //   1. params.allowed_*  (immutable Discovery snapshot)
        //   2. regime_state.allowed_*  (legacy, for unmigrated versions)
        //   3. nothing — leave the static-render checked state untouched
        //      (preserves the file-level defaults baked by build_report)
        var ap  = resp.active.params       || {{}};
        var ars = resp.active.regime_state || {{}};
        var initMacro = (ap.allowed_macro_regimes !== undefined)
                          ? ap.allowed_macro_regimes
                          : ars.allowed_macro_regimes;
        var initMicro = (ap.allowed_micro_regimes !== undefined)
                          ? ap.allowed_micro_regimes
                          : ars.allowed_micro_regimes;
        function setAllow(containerId, allowed) {{
          var container = document.getElementById(containerId);
          if (!container) return;
          var allowedSet = {{}};
          (allowed || []).forEach(function (k) {{ allowedSet[k] = true; }});
          container.querySelectorAll(".regime-toggle").forEach(function (toggle) {{
            var key = toggle.getAttribute("data-regime-key");
            var input = toggle.querySelector(".regime-toggle-input");
            if (!input) return;
            input.checked = !!allowedSet[key];
            applyToggleVisual(toggle);
          }});
        }}
        if (initMacro !== undefined) setAllow("regime-macro-toggles", initMacro);
        if (initMicro !== undefined) setAllow("regime-micro-toggles", initMicro);
      }})
      .catch(function () {{}});

    // ── Active-version sync ────────────────────────────────────────────────
    // Structural change (May 2026): the version dropdown was removed. The
    // RA reads the active version from /api/versions on every load and
    // surfaces it read-only via the top-nav "Active: vN" indicator. To
    // change the active version, the user goes to /versions and picks a
    // different radio button — the next load of the RA reflects it.
    //
    // We also stash the active version on window.__activeVersion so
    // syncRegimePageTitle can build "v7 · GBPUSD · 5m" from
    // v.params.instrument / v.params.interval.
    (function () {{
      fetch("/api/versions").then(function (r) {{ return r.json(); }})
        .then(function (store) {{
          var all = (store && store.versions) || [];
          var versions = all.filter(function (v) {{ return v && v.params; }});
          var activeId = store && store.active_version_id;
          var activeV = null;
          for (var i = 0; i < versions.length; i++) {{
            if (versions[i].id === activeId) {{ activeV = versions[i]; break; }}
          }}
          if (!activeV && versions.length) activeV = versions[0];
          window.__activeVersion = activeV;
          window.__activeVersionId = activeV ? activeV.id : "";
          // Issue 4/5 fix: capture the version's IMMUTABLE Discovery
          // allow-lists for the Reset-to-Defaults button + page-init.
          // Source preference:
          //   1. params.allowed_macro/micro (set at assignment time by
          //      the May-2026 schema fix; immutable snapshot).
          //   2. regime_state.allowed_macro/micro (mutable working copy,
          //      written on every Run). Used as a fallback ONLY for
          //      versions that pre-date the schema fix AND that the
          //      _backfill_params_allow_lists migration couldn't match
          //      uniquely to a Discovery trial.
          // Skip entirely (leave defaults null → file-level fallback in
          // the Reset handler) if neither source has the lists — that's
          // an unassigned version like a freshly-created empty v1.
          if (activeV) {{
            var p  = activeV.params || {{}};
            var rs = activeV.regime_state || {{}};
            var macro = p.allowed_macro_regimes;
            var micro = p.allowed_micro_regimes;
            if (macro === undefined && rs.allowed_macro_regimes !== undefined) {{
              macro = rs.allowed_macro_regimes;
            }}
            if (micro === undefined && rs.allowed_micro_regimes !== undefined) {{
              micro = rs.allowed_micro_regimes;
            }}
            if (macro !== undefined && micro !== undefined) {{
              _versionParamsDefaults = {{
                macro: (macro || []).slice(),
                micro: (micro || []).slice(),
              }};
            }}
          }}
          if (typeof syncRegimePageTitle === "function") syncRegimePageTitle();
        }})
        .catch(function () {{}});

      var instSel = document.getElementById("instrument-select");
      if (instSel) {{
        var stored = (localStorage.getItem("rb_instrument") || "").trim();
        if (stored) instSel.value = stored;
        instSel.addEventListener("change", function () {{
          try {{ localStorage.setItem("rb_instrument", instSel.value); }} catch (e) {{}}
          if (typeof syncRegimePageTitle === "function") syncRegimePageTitle();
        }});
      }}
    }}());

    // Issue 5 fix (May 2026): the previous handler called
    // clearRegimeAnalysisState() on Reset, which wipes the entire
    // per-version localStorage cache — including the date range. Symptom:
    // clicking Reset and then refreshing the page snapped the date
    // pickers back to the full 15-month server-rendered defaults,
    // surprising the user. Reset's intent is "toggles → Discovery
    // baseline", nothing else. Cache invalidation now happens naturally
    // on the next Run (which writes a fresh entry). If the user reloads
    // before clicking Run, applySavedStateToControls's allow-list
    // comparison detects the mismatch and skips painting the stale
    // cached HTML — toggles still snap to the new baseline via the
    // active-version sync block above, so nothing is visually wrong.
    // (Intentionally a no-op block — the handler #1 above already
    // resets the toggle state.) Kept here as a comment for the next
    // person who wonders why the second resetBtn handler is gone.

    // ── Keyboard shortcuts: 1, 2-9, 0 jump to RA sections ─────────────────
    // Mirrors the BD's pattern. `#main` is `<html>` (it carries id="main"
    // — the RA page uses the document as its scroller). Suppressed when
    // focus is in a form field.
    //
    //   1 → top of page
    //   2 → Stats cards (regime-stats-section) — also surfaces the regime
    //       filters panel just below
    //   3 → Macro regime performance table
    //   4 → Micro regime performance table
    //   5 → Regime timeline
    //   6 → Daily performance
    //   7 → Macro Regime Profiles
    //   8 → Regime summary cards
    //   9 → Threshold distributions
    //   0 → Regime periods (bottom of page)
    document.addEventListener("keydown", function (e) {{
      if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
      var m = (e.code || "").match(/^Digit(\d)$/);
      if (!m) return;
      var num = parseInt(m[1], 10);

      var mainEl = document.getElementById("main");
      if (!mainEl) return;
      if (num === 1) {{
        e.preventDefault();
        mainEl.scrollTo({{ top: 0, behavior: "smooth" }});
        return;
      }}
      var sectionIds = {{
        2: "regime-stats-section",
        3: "regime-macro-perf-section",
        4: "regime-perf-section",
        5: "regime-timeline-section",
        6: "regime-daily-section",
        7: "regime-macro-profiles-section",
        8: "regime-summary-cards-section",
        9: "regime-thresholds-section",
        0: "regime-periods-section",
      }};
      if (!(num in sectionIds)) return;
      var sec = document.getElementById(sectionIds[num]);
      if (!sec) return;
      e.preventDefault();
      // Anchor on the section's <h2> heading when present so the title
      // lines up just below the fixed chrome. Falls back to the section
      // element for the stats-cards section (which has no h2).
      var anchor = sec.querySelector("h2") || sec;
      var topNav = document.getElementById("top-nav");
      var runBar = document.getElementById("run-bar");
      var chromeH = (topNav ? topNav.offsetHeight : 0) + (runBar ? runBar.offsetHeight : 0);
      if (chromeH === 0) chromeH = parseInt(getComputedStyle(document.body).paddingTop, 10) || 92;
      var top = anchor.offsetTop - mainEl.offsetTop - chromeH - 12;
      mainEl.scrollTo({{ top: Math.max(0, top), behavior: "smooth" }});
    }});

    // ── Regime-filters show/hide toggle ────────────────────────────────────
    // Mirrors the dashboard's `bs-toggle-btn` + `bs-collapsible` pattern.
    // Default state: collapsed. Click the chevron in the page-title row to
    // expand/collapse the regime-filters panel.
    (function () {{
      var btn = document.getElementById("regime-filters-toggle-btn");
      var panel = document.getElementById("regime-filters-collapsible");
      if (!btn || !panel) return;
      btn.addEventListener("click", function () {{
        var isOpen = panel.classList.toggle("open");
        btn.classList.toggle("open", isOpen);
        // Set max-height to the panel's scrollHeight so it animates to the
        // exact content height without clipping (mirrors the dashboard).
        panel.style.maxHeight = isOpen ? panel.scrollHeight + "px" : "0";
      }});
    }}());

    // ── Keyboard shortcut: S — Toggle Regime Filters panel ──────────────────
    // Mirrors the BD's S shortcut so the same key opens the equivalent
    // settings panel on every page (BD / RA / Discovery). Suppressed when
    // focus is in a form field, matching the digit-shortcut conventions
    // above.
    document.addEventListener("keydown", function (e) {{
      if (e.key !== "s" && e.key !== "S") return;
      if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
      var btn = document.getElementById("regime-filters-toggle-btn");
      if (!btn) return;
      e.preventDefault();
      btn.click();
    }});

    // ── Hover-preview + daily-table sort handlers (extracted so they can be
    //    re-attached after a Run Analysis innerHTML swap) ──
    function attachDailyHandlers() {{
      document.querySelectorAll(".v-sub-preview-btn:not(.disabled)").forEach(function (btn) {{
        if (btn.dataset.attached === "1") return;
        btn.dataset.attached = "1";
        btn.addEventListener("mouseenter", function () {{
          var row = btn.closest("tr");
          var chipsEl = row ? row.querySelector(".regime-hour-chips") : null;
          var chipsHtml = chipsEl ? chipsEl.outerHTML : "";
          showChartPreview(btn.getAttribute("data-chart-src"), chipsHtml);
        }});
        btn.addEventListener("mouseleave", hideChartPreview);
        btn.addEventListener("click", function (e) {{ e.preventDefault(); e.stopPropagation(); }});
      }});

      var dailyTable = document.getElementById("regime-daily-table");
      if (!dailyTable || dailyTable.dataset.sortAttached === "1") return;
      dailyTable.dataset.sortAttached = "1";
      var sortHeaders = dailyTable.querySelectorAll("th.regime-sort");
      sortHeaders.forEach(function (th) {{
        th.addEventListener("click", function () {{
          var tbody = dailyTable.tBodies[0];
          var rows  = Array.prototype.slice.call(tbody.rows);
          var idx   = th.cellIndex;
          var type  = th.getAttribute("data-sort-type") || "string";
          var dir = th.getAttribute("data-sort-dir") === "asc" ? "desc" : "asc";
          sortHeaders.forEach(function (h) {{
            if (h !== th) {{
              h.removeAttribute("data-sort-dir");
              h.classList.remove("regime-sort-asc", "regime-sort-desc");
            }}
          }});
          th.setAttribute("data-sort-dir", dir);
          th.classList.remove("regime-sort-asc", "regime-sort-desc");
          th.classList.add(dir === "asc" ? "regime-sort-asc" : "regime-sort-desc");
          rows.sort(function (a, b) {{
            var av = a.cells[idx].getAttribute("data-sort-value");
            var bv = b.cells[idx].getAttribute("data-sort-value");
            if (av === null) av = a.cells[idx].textContent.trim();
            if (bv === null) bv = b.cells[idx].textContent.trim();
            if (type === "number") {{
              av = parseFloat(av) || 0;
              bv = parseFloat(bv) || 0;
            }}
            return (av < bv ? -1 : av > bv ? 1 : 0) * (dir === "asc" ? 1 : -1);
          }});
          rows.forEach(function (r) {{ tbody.appendChild(r); }});
        }});
      }});
    }}
    attachDailyHandlers();

    // The Devlog panel (toggleDevlogPanel + #devlog-btn handler + the
    // dynamically-created #regime-devlog-panel/textarea/save flow) was
    // removed May 2026 along with the run bar's Development Log icon —
    // the panel had no practical use on the Regimes page.
  }})();
  </script>
</body>
</html>
"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    return REPORT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — labels parquet with thresholds in schema metadata
# ─────────────────────────────────────────────────────────────────────────────

def persist_labels(fractal_df, thresholds, periods, macro=None):
    """Save per-fractal labels + thresholds to data/regime_labels.parquet.

    Tercile thresholds and per-period summaries are tucked into the parquet's
    schema-level custom metadata (JSON-encoded) so the file remains a single
    self-contained artifact.

    `macro` is the optional dict returned by `stage2b_classify_macro`
    ({YYYY-MM-DD: {label, details}}). When provided, a flat
    `macro_by_date` mapping is added to the schema metadata so downstream
    consumers (e.g. strategy_v2.py's regime gates) can look up each day's
    macro classification without re-running the labeler.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out = fractal_df[[
        "timestamp", "fractal_bar", "kind", "price",
        "adx", "atr_pips", "v_dist_pips", "h_dist_bars", "entry_hour",
        "pips_per_bar", "width_pips", "choppiness",
        "coarse_label", "regime", "regime_start_ts", "candles_active",
    ]].copy()
    out["timestamp"]       = pd.to_datetime(out["timestamp"]).dt.tz_convert("UTC")
    out["regime_start_ts"] = pd.to_datetime(out["regime_start_ts"]).dt.tz_convert("UTC")

    macro_by_date = {}
    if macro:
        for day, info in macro.items():
            macro_by_date[day] = info.get("label") if isinstance(info, dict) else None

    meta_payload = {
        "thresholds": thresholds,
        "lookback_fractals": LOOKBACK_FRACTALS,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "period_count": len(periods),
        "generated":  datetime.utcnow().isoformat() + "Z",
        "macro_by_date": macro_by_date,
    }
    meta_json = json.dumps(meta_payload)

    # Write the metadata blob under both `regime_analysis` (new canonical key)
    # and `regime_labeler` (legacy, kept for back-compat with readers that
    # haven't been updated). Strategy_v2's loader and the Flask endpoint both
    # accept either key.
    if _PARQUET_ENGINE == "pyarrow":
        table = pa.Table.from_pandas(out, preserve_index=False)
        existing = table.schema.metadata or {}
        new_meta = {
            **dict(existing),
            b"regime_analysis": meta_json.encode("utf-8"),
            b"regime_labeler":  meta_json.encode("utf-8"),
        }
        table = table.replace_schema_metadata(new_meta)
        pq.write_table(table, LABELS_PATH)
    else:
        # fastparquet path — supports the same KV metadata via custom_metadata.
        from fastparquet import write as _fp_write
        _fp_write(str(LABELS_PATH), out, custom_metadata={
            "regime_analysis": meta_json,
            "regime_labeler":  meta_json,
        })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_iso_date(s, field):
    """Validate a YYYY-MM-DD string; abort with a clear error otherwise."""
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"ERROR: {field} must be YYYY-MM-DD (got: {s!r})")
    return s


def _build_arg_parser():
    """CLI for regime_analysis.py. May 2026 — added windowed flags so the
    labels parquet and report window can diverge (use case: regenerate
    labels for the full historical range, render the report on a narrow
    verification window). Resolution order:
      • --labels-start / --labels-end → labels window
      • --report-start / --report-end → report window
      • --start / --end fall through to anything not explicitly set above
      • module constants (START_DATE / END_DATE) are the final fallback
    With no flags, behaviour matches pre-May-2026 (both windows = constants)."""
    p = argparse.ArgumentParser(
        description="Regenerate regime labels parquet + HTML report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", type=str, default=None,
                   help="Set BOTH labels and report start dates (YYYY-MM-DD). "
                        "Overridden per-window by --labels-start / --report-start.")
    p.add_argument("--end", type=str, default=None,
                   help="Set BOTH labels and report end dates (YYYY-MM-DD). "
                        "Overridden per-window by --labels-end / --report-end.")
    p.add_argument("--labels-start", type=str, default=None, dest="labels_start",
                   help="Labels parquet start date (overrides --start).")
    p.add_argument("--labels-end", type=str, default=None, dest="labels_end",
                   help="Labels parquet end date (overrides --end).")
    p.add_argument("--report-start", type=str, default=None, dest="report_start",
                   help="HTML report start date (overrides --start). Must lie "
                        "within the labels window.")
    p.add_argument("--report-end", type=str, default=None, dest="report_end",
                   help="HTML report end date (overrides --end). Must lie "
                        "within the labels window.")
    return p


def _resolve_windows(args):
    """Resolve the labels + report windows from CLI args, validate that the
    report window is a subset of the labels window, and rebind the module
    globals so the stages downstream see the resolved values. Returns
    ((labels_start, labels_end), (report_start, report_end))."""
    global START_DATE, END_DATE, REPORT_START_DATE, REPORT_END_DATE
    labels_start = args.labels_start or args.start or START_DATE
    labels_end   = args.labels_end   or args.end   or END_DATE
    # Report defaults to labels when not explicitly set. This keeps the
    # default behaviour identical to pre-May-2026: one window, one report.
    report_start = args.report_start or args.start or labels_start
    report_end   = args.report_end   or args.end   or labels_end

    for s, name in [(labels_start, "--labels-start (or --start)"),
                    (labels_end,   "--labels-end (or --end)"),
                    (report_start, "--report-start (or --start)"),
                    (report_end,   "--report-end (or --end)")]:
        _parse_iso_date(s, name)

    if labels_start > labels_end:
        raise SystemExit(f"ERROR: labels-start ({labels_start}) is after "
                         f"labels-end ({labels_end})")
    if report_start > report_end:
        raise SystemExit(f"ERROR: report-start ({report_start}) is after "
                         f"report-end ({report_end})")
    if report_start < labels_start or report_end > labels_end:
        raise SystemExit(
            f"ERROR: report window [{report_start}, {report_end}] must lie "
            f"within labels window [{labels_start}, {labels_end}]. "
            "Widen --labels-start/--labels-end or narrow --report-start/--report-end.")

    START_DATE        = labels_start
    END_DATE          = labels_end
    REPORT_START_DATE = report_start
    REPORT_END_DATE   = report_end
    return (labels_start, labels_end), (report_start, report_end)


def main():
    args = _build_arg_parser().parse_args()
    (labels_start, labels_end), (report_start, report_end) = _resolve_windows(args)
    if (labels_start, labels_end) == (report_start, report_end):
        print(f"Windows: labels = report = {labels_start} → {labels_end}")
    else:
        print(f"Windows: labels {labels_start} → {labels_end}  "
              f"|  report {report_start} → {report_end}")

    fractal_df, full_df = stage1_extract_fractals()
    if fractal_df.empty:
        print("No fractals detected — aborting.")
        return

    fractal_df, periods, thresholds = stage2_classify(fractal_df)
    macro                           = stage2b_classify_macro(full_df)
    trades_df, perf_df, blocked_df  = stage3_trade_outcomes(fractal_df, full_df, macro=macro)
    stage4_thresholds(thresholds)

    # Skip the chart-generation loop when GENERATE_DAILY_CHARTS is off. The
    # hover-preview still works for any pre-existing PNGs in regime_charts/
    # because build_daily_breakdown inspects the directory itself rather
    # than trusting our return value.
    if GENERATE_DAILY_CHARTS:
        available_chart_days = generate_daily_charts(full_df)
    else:
        # Build the same set the chart-gen loop would have built from any
        # pre-existing PNGs, so the daily-breakdown rows light up the
        # preview button for days that already have a chart on disk.
        available_chart_days = set()
        if REGIME_CHARTS_DIR.exists():
            for p in REGIME_CHARTS_DIR.glob("*.png"):
                available_chart_days.add(p.stem)
        print(f"Skipping daily chart generation (GENERATE_DAILY_CHARTS=False); "
              f"{len(available_chart_days)} existing chart(s) reused")

    report_path = build_report(fractal_df, periods, thresholds, trades_df, perf_df,
                               full_df, available_chart_days, macro,
                               blocked_signals_df=blocked_df)
    persist_labels(fractal_df, thresholds, periods, macro=macro)

    # Prefer the Flask server URL when reachable — the page's Run Analysis
    # button calls /run_regime_analysis via relative fetch, which only works
    # under an http origin. Fall back to file:// if the server isn't running
    # so the page still opens (Run Analysis will just say "Failed to fetch"
    # in that case, with the same expectation as before).
    import socket
    def _server_alive(host="127.0.0.1", port=8080, timeout=0.4):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    if _server_alive():
        url = f"http://localhost:8080/results/{REPORT_PATH.name}"
    else:
        print("  (Flask server not detected on :8080 — opening file:// URL; "
              "start `python3 server.py` and reload at "
              f"http://localhost:8080/results/{REPORT_PATH.name} to enable "
              "the Run Analysis button.)")
        url = f"file://{report_path}"
    webbrowser.open(url)
    print(f"Report saved and opened: {url}")


if __name__ == "__main__":
    main()
