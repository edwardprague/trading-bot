#!/usr/bin/env python3
"""
regime_labeler.py — Market regime detection and labeling
=========================================================
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
  • results/regime_labeler.html   — full HTML report
  • data/regime_labels.parquet    — per-fractal labels + thresholds metadata

Usage
-----
    source venv/bin/activate
    python3 regime_labeler.py
"""

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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow is required for Parquet output. Install with:  pip install pyarrow")
    sys.exit(1)

import strategy_v2 as strat


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

START_DATE        = "2026-01-01"
END_DATE          = "2026-03-31"
LOOKBACK_FRACTALS = 4

# How many bars (5m) to keep before START_DATE so the rolling lookback is
# already populated by the time we reach the first in-range fractal.
WARMUP_BARS = 30

ROOT_DIR           = Path(__file__).resolve().parent
DATA_DIR           = ROOT_DIR / "data"
RESULTS_DIR        = ROOT_DIR / "results"
REGIME_CHARTS_DIR  = RESULTS_DIR / "regime_charts"
REPORT_PATH        = RESULTS_DIR / "regime_labeler.html"
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

REGIME_DISPLAY = {
    "trending_fast_down":   "Trending — Fast Down",
    "trending_medium_down": "Trending — Medium Down",
    "trending_slow_down":   "Trending — Slow Down",
    "trending_fast_up":     "Trending — Fast Up",
    "trending_medium_up":   "Trending — Medium Up",
    "trending_slow_up":     "Trending — Slow Up",
    "ranging_narrow":       "Ranging — Narrow",
    "ranging_medium":       "Ranging — Medium",
    "ranging_wide":         "Ranging — Wide",
    "transitioning":        "Transitioning",
}

# Order used in tables / cards / timeline keys
REGIME_ORDER = list(REGIME_DISPLAY.keys())


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

    for _, fr in fractal_df.iterrows():
        rH = fr["_rolling_H"]
        rL = fr["_rolling_L"]
        raw = _classify_raw(rH, rL)
        raws.append(raw)
        pips_bar.append(_period_pips_per_bar(rH, rL))
        w, c = _period_width_choppiness(rH, rL)
        widths.append(w)
        choppiness.append(c)

        # Track per-kind class history for the confirmation rule
        if fr["kind"] == "H":
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
            state_start_bar = int(fr["fractal_bar"])
            state_start_ts  = fr["timestamp"]
            committed_label = state
        else:
            if state_start_bar is None:
                state_start_bar = int(fr["fractal_bar"])
                state_start_ts  = fr["timestamp"]
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
                    state_start_bar = int(fr["fractal_bar"])
                    state_start_ts  = fr["timestamp"]
                    state = "transitioning"
                    # Clear class history so confirmation only counts the new
                    # post-transition signals.
                    if fr["kind"] == "H": last_H_class = [raw]
                    else:                 last_L_class = [raw]

        coarse_labels.append(committed_label)
        regime_starts.append(state_start_ts)
        candles_active.append(int(fr["fractal_bar"]) - int(state_start_bar))

    fractal_df = fractal_df.copy()
    fractal_df["raw_class"]       = raws
    fractal_df["pips_per_bar"]    = pips_bar
    fractal_df["width_pips"]      = widths
    fractal_df["choppiness"]      = choppiness
    fractal_df["coarse_label"]    = coarse_labels
    fractal_df["regime_start_ts"] = regime_starts
    fractal_df["candles_active"]  = candles_active

    # ── Group consecutive same-coarse-label fractals into periods ───────────
    periods = []
    cur = None
    for i, r in fractal_df.iterrows():
        if cur is None or r["coarse_label"] != cur["label"]:
            if cur is not None:
                periods.append(cur)
            cur = {
                "label":        r["coarse_label"],
                "start_idx":    i,
                "end_idx":      i,
                "start_ts":     r["timestamp"],
                "end_ts":       r["timestamp"],
                "start_bar":    int(r["fractal_bar"]),
                "end_bar":      int(r["fractal_bar"]),
                "fractal_idxs": [i],
            }
        else:
            cur["end_idx"]      = i
            cur["end_ts"]       = r["timestamp"]
            cur["end_bar"]      = int(r["fractal_bar"])
            cur["fractal_idxs"].append(i)
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
# Stage 3 — Trade outcome mapping
# ─────────────────────────────────────────────────────────────────────────────

def stage3_trade_outcomes(fractal_df, full_df):
    print("Stage 3: Mapping trade outcomes to regimes...")

    trades, _, _ = strat.run_backtest(full_df)

    # Filter to requested range by entry timestamp
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(END_DATE,   tz="UTC") + pd.Timedelta(days=1)
    if not trades.empty:
        t_entry = pd.to_datetime(trades["entry_ts"])
        t_utc   = t_entry.dt.tz_convert("UTC") if t_entry.dt.tz is not None else t_entry.dt.tz_localize("UTC")
        trades  = trades[(t_utc >= start_ts) & (t_utc < end_ts)].copy()

    # Map fractal_bar -> regime label (only fractals in range)
    in_range = fractal_df[
        (fractal_df["timestamp"] >= start_ts) & (fractal_df["timestamp"] < end_ts)
    ]
    bar_to_regime = dict(zip(in_range["fractal_bar"].astype(int), in_range["regime"]))

    if trades.empty:
        trades["regime"] = []
    else:
        trades["regime"] = trades["fractal_bar"].astype(int).map(bar_to_regime)

    # Aggregate metrics per regime
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
    print("Stage 3 complete")
    return trades, perf_df


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

def _trading_days_in_range(full_df):
    """List of unique YYYY-MM-DD strings inside [START_DATE, END_DATE] that
    have at least one bar in `full_df`."""
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(END_DATE,   tz="UTC") + pd.Timedelta(days=1)
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
    days = _trading_days_in_range(full_df)
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
            if not df_view.empty:
                # Snapshot the real timestamps that each trade index points to
                # BEFORE padding, so we can re-locate them in the padded frame.
                pre_padding_dts = pd.to_datetime(df_view["Datetime"])
                pre_padding_dts = (pre_padding_dts.dt.tz_convert("UTC")
                                   if pre_padding_dts.dt.tz is not None
                                   else pre_padding_dts.dt.tz_localize("UTC"))
                if not trades.empty:
                    def _ts_at(idx_series):
                        idx_clamped = idx_series.astype(int).clip(
                            lower=0, upper=len(df_view) - 1).values
                        return pre_padding_dts.iloc[idx_clamped].values
                    entry_ts_real = _ts_at(trades["entry_idx"])
                    exit_ts_real  = _ts_at(trades["exit_idx"])
                    fb_ts_real    = _ts_at(trades["fractal_bar"])

                # Build the canonical 24h grid and reindex
                full_grid = pd.date_range(
                    start=r_start, end=r_end - pd.Timedelta(minutes=5),
                    freq="5min", tz="UTC",
                )
                df_view_idx = df_view.copy()
                df_view_idx["Datetime"] = pre_padding_dts.values
                df_view_idx = df_view_idx.set_index("Datetime")
                df_padded = df_view_idx.reindex(full_grid)
                for col in df_padded.columns:
                    df_padded[col] = df_padded[col].ffill().bfill()
                df_padded = df_padded.reset_index().rename(columns={"index": "Datetime"})
                df_view = df_padded

                # Re-locate trade indices in the padded grid
                if not trades.empty:
                    padded_ts = pd.to_datetime(df_view["Datetime"])
                    padded_ts = (padded_ts.dt.tz_convert("UTC")
                                 if padded_ts.dt.tz is not None
                                 else padded_ts.dt.tz_localize("UTC"))
                    ts_to_pos = {pd.Timestamp(ts): i for i, ts in enumerate(padded_ts)}
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
# Daily breakdown — one row per trading day with hourly chips + chart preview
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


def build_hourly_chips(day_str, day_fractals):
    """Return HTML for 24 hourly chips, one per UTC hour of day_str.

    `day_fractals` is a DataFrame of fractals that occurred on this UTC date
    (must have 'timestamp' and 'regime' columns). Hours containing at least
    one fractal show the most common regime in that hour. Hours with zero
    fractals show the regime-color-inactive (very dark) chip so it is visually
    obvious which hours had no classification at all — rather than letting
    forward-fill from prior periods make the day appear uniformly classified.
    """
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
        if not regimes_in_hour:
            title = f"{day_str} {h:02d}:00 — No fractal activity"
            chips.append(
                f"<span class='regime-hour-chip regime-color-inactive' title='{title}'></span>"
            )
        else:
            # Most common regime in this hour (ties broken by first occurrence)
            counts = {}
            for r in regimes_in_hour:
                counts[r] = counts.get(r, 0) + 1
            top = max(counts, key=counts.get)
            n = sum(counts.values())
            label = REGIME_DISPLAY.get(top, "Unknown")
            title = f"{day_str} {h:02d}:00 — {label} ({n} fractal{'s' if n != 1 else ''})"
            chips.append(
                f"<span class='regime-hour-chip {regime_class(top)}' title='{title}'></span>"
            )
    return f"<div class='regime-hour-chips'>{''.join(chips)}</div>"


def build_daily_breakdown(periods, trades_df, full_df, available_chart_days,
                          in_range_fractals, low_activity_days):
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
    """
    days = _trading_days_in_range(full_df)
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
        chips_html = build_hourly_chips(day, day_fractals)

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

        rows.append(
            f"<tr>"
            f"<td data-sort-value='{day}'>{date_cell}</td>"
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
                 full_df, available_chart_days):
    """Render results/regime_labeler.html and return its path."""

    # Trim to requested range for display counts
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(END_DATE,   tz="UTC") + pd.Timedelta(days=1)
    in_range = fractal_df[
        (fractal_df["timestamp"] >= start_ts) & (fractal_df["timestamp"] < end_ts)
    ]
    in_range_periods = [
        p for p in periods
        if pd.Timestamp(p["end_ts"]) >= start_ts and pd.Timestamp(p["start_ts"]) < end_ts
    ]

    # Periods-per-regime count (used by the Trade performance by regime table
    # and by the timeline legend).
    regime_count = {}
    for p in in_range_periods:
        regime_count[p["regime"]] = regime_count.get(p["regime"], 0) + 1

    # ── Low-activity day stats (Issue 3) ────────────────────────────────────
    # A day is "low activity" when fewer than LOW_ACTIVITY_FRACTAL_THRESHOLD
    # fractals were detected during it. Trades on these days are not excluded
    # from any statistics — we just flag them so the user knows the regime
    # labels assigned during those quiet stretches may be less reliable.
    trading_days_all = _trading_days_in_range(full_df)
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

    # ── Timeline ────────────────────────────────────────────────────────────
    timeline = build_daily_timeline(in_range_periods, START_DATE, END_DATE)
    timeline_cells = []
    for entry in timeline:
        title = f"{entry['date']} — {entry['label']}"
        timeline_cells.append(
            f"<div class='regime-tl-cell {regime_class(entry['regime'])}' "
            f"title='{title}'></div>"
        )
    # Week labels
    week_labels = []
    for i, entry in enumerate(timeline):
        if i == 0 or (pd.Timestamp(entry["date"]).weekday() == 0):
            week_labels.append(f"<span class='regime-tl-week'>{entry['date'][5:]}</span>")
    legend_items = []
    for label in REGIME_ORDER:
        if label in regime_count:
            legend_items.append(
                f"<span class='regime-chip'>"
                f"<span class='regime-swatch {regime_class(label)}'></span>"
                f"{REGIME_DISPLAY[label]}</span>"
            )
    legend_html = "".join(legend_items)

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

    # ── Trade performance by regime ─────────────────────────────────────────
    # Show every regime that occurred in the date range (period_count > 0), so
    # the per-regime period counts that used to live in the header chip strip
    # are preserved. Trade-stat cells show '—' when the regime had no trades.
    perf_rows = []
    for _, r in perf_df.iterrows():
        period_count = int(regime_count.get(r["regime"], 0))
        if period_count == 0:
            continue   # regime never occurred in the requested range
        n_trades = int(r["trades"])
        if n_trades > 0:
            cls          = winrate_class(r["win_rate"], n_trades)
            wins_cell    = str(int(r["wins"]))
            wr_cell      = f"<td class='{cls}'>{_fmt_pct(r['win_rate'])}</td>"
            pf_cell      = f"<td>{_fmt_pf(r['profit_factor'])}</td>"
            avg_cell     = f"<td>{_fmt_money(r['avg_pnl'])}</td>"
            total_cell   = f"<td>{_fmt_money(r['total_pnl'])}</td>"
        else:
            wins_cell    = "—"
            wr_cell      = "<td class='regime-dim'>—</td>"
            pf_cell      = "<td class='regime-dim'>—</td>"
            avg_cell     = "<td class='regime-dim'>—</td>"
            total_cell   = "<td class='regime-dim'>—</td>"
        perf_rows.append(
            f"<tr>"
            f"<td><span class='regime-swatch {regime_class(r['regime'])}'></span>"
            f"{REGIME_DISPLAY[r['regime']]}</td>"
            f"<td>{period_count}</td>"
            f"<td>{n_trades}</td>"
            f"<td>{wins_cell}</td>"
            f"{wr_cell}{pf_cell}{avg_cell}{total_cell}"
            f"</tr>"
        )
    perf_table = f"""
      <table class='regime-table'>
        <thead><tr>
          <th>Regime</th><th>Periods</th><th>Trades</th><th>Wins</th><th>Win rate</th>
          <th>Profit factor</th><th>Avg P&amp;L</th><th>Total P&amp;L</th>
        </tr></thead>
        <tbody>{''.join(perf_rows) or '<tr><td colspan=8 class=regime-dim>No regimes observed in the requested range.</td></tr>'}</tbody>
      </table>
    """

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

    # ── Daily breakdown table ───────────────────────────────────────────────
    daily_table = build_daily_breakdown(in_range_periods, trades_df, full_df,
                                        available_chart_days,
                                        in_range, low_activity_days)

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

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Regime Labeler — GBPUSD 5m</title>
  <link rel="stylesheet" href="../style.css">
  <!-- Material Symbols — same icon font the dashboard sidebar uses for the
       preview eye icon. Variable axes (FILL/wght/GRAD/opsz) enabled so the
       icon can morph from outlined → filled on hover. -->
  <link rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
</head>
<body class="regime-report">
  <div class="regime-container">

    <header class="regime-header">
      <h1>Regime Labeler — GBPUSD 5m</h1>
      <div class="regime-header-meta">
        <span><strong>Range:</strong> {START_DATE} → {END_DATE}</span>
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

    <section class="regime-card">
      <h2>Trade performance by regime <span class="regime-dim regime-small">(v2 short-only)</span></h2>
      <p class="regime-dim regime-small">
        How the active strategy performed in each market condition. Green ≥ 55%, red ≤ 45%.
      </p>
      {perf_table}
      {perf_low_note}
    </section>

    <section class="regime-card">
      <h2>Regime timeline</h2>
      <p class="regime-dim regime-small">
        Each cell is one calendar day, coloured by the dominant regime that day.
        Hover for details.
      </p>
      <div class="regime-tl-strip">{''.join(timeline_cells)}</div>
      <div class="regime-tl-weeks">{''.join(week_labels)}</div>
      <div class="regime-legend">{legend_html}</div>
    </section>

    <section class="regime-card">
      <h2>Daily breakdown</h2>
      <p class="regime-dim regime-small">
        One row per trading day. Each chip is a UTC hour, coloured by the
        dominant regime among fractals detected in that hour. Hover the chart
        icon for a price/trade preview of that day. Click <strong>Date</strong>,
        <strong>Trades</strong>, or <strong>P&amp;L</strong> headers to sort.
      </p>
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

    <section>
      <h2>Regime summary cards</h2>
      <div class="regime-summary-grid">{''.join(cards) or '<p class=regime-dim>No regimes observed.</p>'}</div>
    </section>

    <section class="regime-card">
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

    <section class="regime-card">
      <h2>Regime periods</h2>
      {period_table}
    </section>

  </div>

  <!-- Hover preview overlay — mirrors the dashboard's #chart-preview-overlay
       (styles already in style.css). Image src is set dynamically by JS to
       the regime_charts/YYYY-MM-DD.png file for the hovered row. -->
  <div id="chart-preview-overlay" aria-hidden="true">
    <div id="chart-preview-card">
      <img id="chart-preview-img" alt="Daily chart preview"/>
    </div>
  </div>

  <script>
  (function() {{
    // ── Chart hover preview ────────────────────────────────────────────────
    var overlay = document.getElementById("chart-preview-overlay");
    var img     = document.getElementById("chart-preview-img");

    function showChartPreview(src) {{
      if (!overlay || !img) return;
      if (!src) {{
        img.removeAttribute("src");
        overlay.classList.add("no-chart");
      }} else {{
        img.src = src;
        overlay.classList.remove("no-chart");
      }}
      overlay.classList.add("visible");
      overlay.setAttribute("aria-hidden", "false");
    }}

    function hideChartPreview() {{
      if (!overlay) return;
      overlay.classList.remove("visible");
      overlay.setAttribute("aria-hidden", "true");
      if (img) img.removeAttribute("src");
    }}

    document.querySelectorAll(".v-sub-preview-btn:not(.disabled)").forEach(function (btn) {{
      btn.addEventListener("mouseenter", function () {{
        showChartPreview(btn.getAttribute("data-chart-src"));
      }});
      btn.addEventListener("mouseleave", hideChartPreview);
      btn.addEventListener("click", function (e) {{ e.preventDefault(); e.stopPropagation(); }});
    }});

    // ── Sortable daily-breakdown table ─────────────────────────────────────
    var table = document.getElementById("regime-daily-table");
    if (!table) return;
    var headers = table.querySelectorAll("th.regime-sort");

    headers.forEach(function (th) {{
      th.addEventListener("click", function () {{
        var tbody = table.tBodies[0];
        var rows  = Array.prototype.slice.call(tbody.rows);
        var idx   = th.cellIndex;
        var type  = th.getAttribute("data-sort-type") || "string";

        // Toggle direction; clear arrows on other headers
        var dir = th.getAttribute("data-sort-dir") === "asc" ? "desc" : "asc";
        headers.forEach(function (h) {{
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

def persist_labels(fractal_df, thresholds, periods):
    """Save per-fractal labels + thresholds to data/regime_labels.parquet.

    Tercile thresholds and per-period summaries are tucked into the parquet's
    schema-level custom metadata (JSON-encoded) so the file remains a single
    self-contained artifact."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out = fractal_df[[
        "timestamp", "fractal_bar", "kind", "price",
        "adx", "atr_pips", "v_dist_pips", "h_dist_bars", "entry_hour",
        "pips_per_bar", "width_pips", "choppiness",
        "coarse_label", "regime", "regime_start_ts", "candles_active",
    ]].copy()
    out["timestamp"]       = pd.to_datetime(out["timestamp"]).dt.tz_convert("UTC")
    out["regime_start_ts"] = pd.to_datetime(out["regime_start_ts"]).dt.tz_convert("UTC")

    table = pa.Table.from_pandas(out, preserve_index=False)

    meta_payload = {
        "thresholds": thresholds,
        "lookback_fractals": LOOKBACK_FRACTALS,
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "period_count": len(periods),
        "generated":  datetime.utcnow().isoformat() + "Z",
    }
    existing = table.schema.metadata or {}
    new_meta = {**dict(existing), b"regime_labeler": json.dumps(meta_payload).encode("utf-8")}
    table = table.replace_schema_metadata(new_meta)

    pq.write_table(table, LABELS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fractal_df, full_df = stage1_extract_fractals()
    if fractal_df.empty:
        print("No fractals detected — aborting.")
        return

    fractal_df, periods, thresholds = stage2_classify(fractal_df)
    trades_df, perf_df              = stage3_trade_outcomes(fractal_df, full_df)
    stage4_thresholds(thresholds)

    available_chart_days = generate_daily_charts(full_df)

    report_path = build_report(fractal_df, periods, thresholds, trades_df, perf_df,
                               full_df, available_chart_days)
    persist_labels(fractal_df, thresholds, periods)

    webbrowser.open(f"file://{report_path}")
    print(f"Report saved and opened: {report_path}")


if __name__ == "__main__":
    main()
