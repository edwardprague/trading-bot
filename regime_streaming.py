"""
regime_streaming.py — Streaming regime classifier (cBot port-ready)
====================================================================
A self-contained, deterministic re-implementation of the macro + micro
regime classification logic from regime_analysis.py — designed to be
translated to C# for the cTrader cBot.

Two modes
---------
  "parity":  exactly mirrors what data/regime_labels.parquet contains.
             - Macro: same-day label (look-ahead — uses the full day's
               close-to-open range, even bars after the entry timestamp).
             - Micro: fine sub-label is computed once per period from the
               period's FINAL aggregate, then projected back onto every
               fractal in the period (retroactive look-ahead).
             Used for validating that the rule logic is correct against
             the current Python backtest.

  "live":    cBot-faithful semantics — no look-ahead.
             - Macro: prior-day label (today's label is only committed at
               day close, applied to the *next* day's signals).
             - Micro: fine sub-label uses the running per-fractal rolling
               aggregate (pips_per_bar / width_pips) at the moment of
               classification — never reaches into future fractals.

Both modes share identical:
  • N=2 Williams fractal detection (confirmed at bar i-2 from bar i)
  • 4-deep rolling H / L lookback (same-kind only)
  • Coarse 4-way raw classifier
  • 2-fractal state-machine confirmation
  • Frozen tercile thresholds (defaults supplied below; pass your own to
    override).

The KMeans pickle in data/regime_model.pkl is NOT used here — that
artefact belongs to a separate research pipeline (regime_discovery.py)
and is not part of the production gate.

Source of truth
---------------
This file mirrors regime_analysis.py at:
  • classify_macro_regime (lines 1165-1259)
  • _classify_raw         (lines 849-892)
  • _period_pips_per_bar  (lines 895-908)
  • _period_width_choppiness (lines 911-923)
  • stage2_classify state machine (lines 968-1028)
  • period grouping + tercile sub-labels (lines 1049-1145)
"""

from __future__ import annotations

import bisect
import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants (mirror regime_analysis.py) ────────────────────────────────────

PIP = 10000  # non-JPY pip multiplier

# Macro constants
EMA_MACRO_PERIOD        = 40
N18_LOOKBACK            = 18
LARGE_DISPLACEMENT_PIPS = 30
SMALL_DISPLACEMENT_PIPS = 15

# Micro constants
LOOKBACK_FRACTALS = 4

# Frozen tercile thresholds — read from regime_labels.parquet metadata,
# computed once over the training window. Used as cBot parameters; refresh
# only when retraining over a new window.
DEFAULT_THRESHOLDS = {
    "pips_per_bar_t1": 0.7248903508771977,
    "pips_per_bar_t2": 1.3061385836386332,
    "width_pips_t1":   10.875,
    "width_pips_t2":   18.11799450549377,
}


# ── Raw classification helpers (verbatim from regime_analysis.py) ────────────

def _classify_raw(rolling_H, rolling_L) -> str:
    """Coarse regime from rolling H/L lookbacks alone.
    Returns one of: 'trending_up', 'trending_down', 'ranging', 'transitioning'.
    """
    if len(rolling_H) < 2 or len(rolling_L) < 2:
        return "transitioning"

    pairs_H = len(rolling_H) - 1
    threshold = max(2, math.ceil(pairs_H * 0.75))

    H_down = sum(1 for i in range(1, len(rolling_H))
                 if rolling_H[i]["price"] < rolling_H[i - 1]["price"])
    H_up   = sum(1 for i in range(1, len(rolling_H))
                 if rolling_H[i]["price"] > rolling_H[i - 1]["price"])
    L_down = sum(1 for i in range(1, len(rolling_L))
                 if rolling_L[i]["price"] < rolling_L[i - 1]["price"])
    L_up   = sum(1 for i in range(1, len(rolling_L))
                 if rolling_L[i]["price"] > rolling_L[i - 1]["price"])

    H_dir = "down" if H_down >= threshold else ("up" if H_up >= threshold else None)
    L_dir = "down" if L_down >= threshold else ("up" if L_up >= threshold else None)

    if H_dir == "down" and L_dir == "down":
        return "trending_down"
    if H_dir == "up" and L_dir == "up":
        return "trending_up"
    if (H_dir is not None and L_dir is not None) and H_dir != L_dir:
        return "transitioning"
    if H_dir is None and L_dir is None:
        return "ranging"
    return "transitioning"


def _period_pips_per_bar(rolling_H, rolling_L) -> float:
    """Average vertical price displacement (pips) / horizontal duration (bars),
    across same-kind successive pairs. Combines H- and L-rates as a mean.
    Returns NaN if neither side has at least 2 fractals."""
    def rate(lst):
        if len(lst) < 2:
            return float("nan")
        v = sum(abs(lst[i]["price"] - lst[i - 1]["price"]) * PIP
                for i in range(1, len(lst))) / (len(lst) - 1)
        h = sum(lst[i]["bar"] - lst[i - 1]["bar"]
                for i in range(1, len(lst))) / (len(lst) - 1)
        return v / h if h > 0 else float("nan")

    rH = rate(rolling_H)
    rL = rate(rolling_L)
    vals = [x for x in (rH, rL) if not math.isnan(x)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _period_width_choppiness(rolling_H, rolling_L):
    """Width = mean(H prices) − mean(L prices) in pips. Returns (width_pips, chop)."""
    if not rolling_H or not rolling_L:
        return float("nan"), float("nan")
    h_prices = np.array([x["price"] for x in rolling_H])
    l_prices = np.array([x["price"] for x in rolling_L])
    width_price = h_prices.mean() - l_prices.mean()
    if width_price <= 0:
        return float("nan"), float("nan")
    width_pips = float(width_price * PIP)
    chop_raw = float(h_prices.std(ddof=0) / width_price) if width_price > 0 else float("nan")
    chop = float(min(1.0, max(0.0, chop_raw))) if not math.isnan(chop_raw) else float("nan")
    return width_pips, chop


# ── Macro: per-day accumulator + classifier ─────────────────────────────────

class _MacroDayState:
    """Accumulates one UTC day's bars; classifies on finalize()."""

    __slots__ = ("date", "opens", "highs", "lows", "closes")

    def __init__(self, day: date):
        self.date = day
        self.opens, self.highs, self.lows, self.closes = [], [], [], []

    def add_bar(self, o, h, l, c):
        self.opens.append(o)
        self.highs.append(h)
        self.lows.append(l)
        self.closes.append(c)

    def finalize(self) -> str:
        """Mirrors classify_macro_regime in regime_analysis.py."""
        n = len(self.closes)
        if n <= max(N18_LOOKBACK * 2 + 1, EMA_MACRO_PERIOD):
            return "flat"

        # 1) Net displacement
        displacement_pips = (self.closes[-1] - self.opens[0]) * PIP
        abs_disp = abs(displacement_pips)

        # 2) EMA-40 slope from Q1 → Q3.
        # Uses pandas ewm(span=40, adjust=False), which equals an iterative
        # alpha=2/(span+1) recurrence seeded with the first close.
        alpha = 2.0 / (EMA_MACRO_PERIOD + 1)
        ema_val = self.closes[0]
        emas = [ema_val]
        for c in self.closes[1:]:
            ema_val = (c - ema_val) * alpha + ema_val
            emas.append(ema_val)

        q1 = max(EMA_MACRO_PERIOD, n // 4)
        q3 = max(q1 + 1, 3 * n // 4)
        q3 = min(q3, n - 1)
        ema_slope_pips = (emas[q3] - emas[q1]) * PIP

        # 3) N18 fractal structure
        H, L = self.highs, self.lows
        n18_highs, n18_lows = [], []
        for i in range(N18_LOOKBACK, n - N18_LOOKBACK):
            is_ph = True
            is_pl = True
            for k in range(1, N18_LOOKBACK + 1):
                if is_ph and (H[i] <= H[i - k] or H[i] <= H[i + k]):
                    is_ph = False
                if is_pl and (L[i] >= L[i - k] or L[i] >= L[i + k]):
                    is_pl = False
                if not is_ph and not is_pl:
                    break
            if is_ph:
                n18_highs.append(H[i])
            if is_pl:
                n18_lows.append(L[i])

        def _dropping(seq):
            return len(seq) >= 2 and all(seq[i] < seq[i - 1] for i in range(1, len(seq)))

        def _rising(seq):
            return len(seq) >= 2 and all(seq[i] > seq[i - 1] for i in range(1, len(seq)))

        down_confirms = _dropping(n18_highs) and _dropping(n18_lows)
        up_confirms   = _rising(n18_highs)   and _rising(n18_lows)

        # Decision tree
        if abs_disp < SMALL_DISPLACEMENT_PIPS:
            return "flat"
        if displacement_pips < 0:
            ema_aligned = ema_slope_pips < 0
            if abs_disp >= LARGE_DISPLACEMENT_PIPS and ema_aligned and down_confirms:
                return "strong_down"
            if ema_aligned:
                return "staircase_down"
            return "flat"
        else:
            ema_aligned = ema_slope_pips > 0
            if abs_disp >= LARGE_DISPLACEMENT_PIPS and ema_aligned and up_confirms:
                return "strong_up"
            if ema_aligned:
                return "staircase_up"
            return "flat"


# ── Main classifier ─────────────────────────────────────────────────────────

class StreamingRegimeClassifier:
    """Causal regime classifier with two modes; see module docstring."""

    def __init__(self, mode: str = "parity", thresholds: Optional[dict] = None):
        if mode not in ("parity", "live"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode
        self.t = dict(thresholds or DEFAULT_THRESHOLDS)

        # Macro outputs: {date -> label}
        self._macro_by_date: dict = {}

        # Sorted ascending; each entry: {"ts": Timestamp, "bar": int,
        # "kind": "H"|"L", "price": float, "coarse": str, "fine": str}
        self._fractals: list = []
        self._fractal_ts_list: list = []  # parallel array for bisect

    # ── Ingestion ────────────────────────────────────────────────────────

    def ingest(self, df: pd.DataFrame) -> None:
        """Bulk feed bars (UTC-indexed; columns Open/High/Low/Close).

        Each fractal's coarse label is committed using ONLY information from
        bars with timestamp ≤ that fractal's timestamp (causal). In "parity"
        mode the fine sub-label is back-filled from each period's final
        aggregate (matching regime_analysis.py); in "live" mode the fine
        sub-label uses the per-fractal rolling aggregate.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("df must have a DatetimeIndex")
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC") if df.index.tz != pd.Timestamp.utcnow().tz else df

        opens  = df["Open"].to_numpy()
        highs  = df["High"].to_numpy()
        lows   = df["Low"].to_numpy()
        closes = df["Close"].to_numpy()
        ts     = df.index.to_pydatetime()
        n      = len(df)

        # ── Macro pass ───────────────────────────────────────────────────
        cur_day = None
        cur_state: Optional[_MacroDayState] = None
        for i in range(n):
            day = ts[i].date()
            if cur_day is None:
                cur_day, cur_state = day, _MacroDayState(day)
            elif day != cur_day:
                self._macro_by_date[cur_day] = cur_state.finalize()
                cur_day, cur_state = day, _MacroDayState(day)
            cur_state.add_bar(opens[i], highs[i], lows[i], closes[i])
        if cur_state is not None:
            self._macro_by_date[cur_day] = cur_state.finalize()

        # ── Micro pass: detect N=2 fractals + state-machine commit ───────
        rolling_H: list = []
        rolling_L: list = []
        last_H_class: list = []
        last_L_class: list = []
        state = "transitioning"

        records: list = []   # one entry per processed (kind, price) event
        last_coarse_seen: Optional[str] = None  # mirrors `coarse_labels[-1]` in source

        for fi in range(2, n - 2):
            fh = highs[fi]
            fl = lows[fi]
            is_ph = (fh > highs[fi - 1] and fh > highs[fi - 2]
                     and fh > highs[fi + 1] and fh > highs[fi + 2])
            is_pl = (fl < lows[fi - 1] and fl < lows[fi - 2]
                     and fl < lows[fi + 1] and fl < lows[fi + 2])
            if not (is_ph or is_pl):
                continue

            events = []
            if is_ph:
                events.append(("H", float(fh)))
            if is_pl:
                events.append(("L", float(fl)))

            for kind, price in events:
                # Update rolling lookback for this kind BEFORE classification —
                # matches stage1's "row context uses pre-update, but stage2
                # reads the post-update snapshot stored on the row".
                if kind == "H":
                    rolling_H.append({"bar": fi, "price": price})
                    if len(rolling_H) > LOOKBACK_FRACTALS:
                        rolling_H = rolling_H[-LOOKBACK_FRACTALS:]
                else:
                    rolling_L.append({"bar": fi, "price": price})
                    if len(rolling_L) > LOOKBACK_FRACTALS:
                        rolling_L = rolling_L[-LOOKBACK_FRACTALS:]

                raw = _classify_raw(rolling_H, rolling_L)

                if kind == "H":
                    last_H_class.append(raw)
                    if len(last_H_class) > 2:
                        last_H_class = last_H_class[-2:]
                else:
                    last_L_class.append(raw)
                    if len(last_L_class) > 2:
                        last_L_class = last_L_class[-2:]

                # State-machine commit (mirrors regime_analysis.py stage2)
                new_committed = None
                for hist in (last_H_class, last_L_class):
                    if (len(hist) == 2
                            and hist[0] == hist[1]
                            and hist[0] != "transitioning"
                            and hist[0] != state):
                        new_committed = hist[0]
                        break

                if new_committed is not None:
                    state = new_committed
                    committed = state
                else:
                    if raw == state:
                        committed = state
                    else:
                        committed = "transitioning"
                        if last_coarse_seen != "transitioning":
                            state = "transitioning"
                            if kind == "H":
                                last_H_class = [raw]
                            else:
                                last_L_class = [raw]

                last_coarse_seen = committed

                records.append({
                    "ts":   pd.Timestamp(ts[fi]),
                    "bar":  fi,
                    "kind": kind,
                    "price": price,
                    "rolling_H": list(rolling_H),
                    "rolling_L": list(rolling_L),
                    "coarse": committed,
                })

        # ── Group consecutive same-coarse fractals into periods ──────────
        periods: list = []
        cur = None
        for rec in records:
            if cur is None or rec["coarse"] != cur["label"]:
                if cur is not None:
                    periods.append(cur)
                cur = {"label": rec["coarse"], "members": [rec]}
            else:
                cur["members"].append(rec)
        if cur is not None:
            periods.append(cur)

        # ── Assign fine sub-labels per fractal, by mode ───────────────────
        if self.mode == "parity":
            for p in periods:
                fine = self._final_period_fine_label(p)
                for rec in p["members"]:
                    rec["fine"] = fine
        else:  # live
            for p in periods:
                for rec in p["members"]:
                    rec["fine"] = self._running_fine_label(rec, p["label"])

        # Persist (drop the heavy rolling lists from public surface)
        self._fractals = [
            {"ts": r["ts"], "bar": r["bar"], "kind": r["kind"],
             "price": r["price"], "coarse": r["coarse"], "fine": r["fine"]}
            for r in records
        ]
        self._fractal_ts_list = [r["ts"] for r in self._fractals]

    # ── Fine-label helpers ───────────────────────────────────────────────

    def _final_period_fine_label(self, period: dict) -> str:
        """Parity mode: mirror regime_analysis.py — aggregate over the
        period's members, then bucket via terciles."""
        label = period["label"]
        if label == "transitioning":
            return "transitioning"

        if label in ("trending_up", "trending_down"):
            vals = []
            for rec in period["members"]:
                ppb = _period_pips_per_bar(rec["rolling_H"], rec["rolling_L"])
                if not math.isnan(ppb):
                    vals.append(ppb)
            ppb_period = (sum(vals) / len(vals)) if vals else float("nan")
            direction = "up" if label == "trending_up" else "down"
            return f"trending_{self._speed_for_ppb(ppb_period)}_{direction}"

        if label == "ranging":
            ws = []
            for rec in period["members"]:
                w, _ = _period_width_choppiness(rec["rolling_H"], rec["rolling_L"])
                if not math.isnan(w):
                    ws.append(w)
            width_period = (sum(ws) / len(ws)) if ws else float("nan")
            return f"ranging_{self._size_for_width(width_period)}"

        return "transitioning"

    def _running_fine_label(self, rec: dict, period_label: str) -> str:
        """Live mode: use the rolling-lookback aggregate at this single
        fractal — never the period's future."""
        if period_label == "transitioning":
            return "transitioning"
        if period_label in ("trending_up", "trending_down"):
            ppb = _period_pips_per_bar(rec["rolling_H"], rec["rolling_L"])
            direction = "up" if period_label == "trending_up" else "down"
            return f"trending_{self._speed_for_ppb(ppb)}_{direction}"
        if period_label == "ranging":
            w, _ = _period_width_choppiness(rec["rolling_H"], rec["rolling_L"])
            return f"ranging_{self._size_for_width(w)}"
        return "transitioning"

    def _speed_for_ppb(self, ppb: float) -> str:
        t1, t2 = self.t["pips_per_bar_t1"], self.t["pips_per_bar_t2"]
        if math.isnan(ppb) or math.isnan(t1):
            return "medium"
        if ppb >= t2:
            return "fast"
        if ppb >= t1:
            return "medium"
        return "slow"

    def _size_for_width(self, w: float) -> str:
        t1, t2 = self.t["width_pips_t1"], self.t["width_pips_t2"]
        if math.isnan(w) or math.isnan(t1):
            return "medium"
        if w >= t2:
            return "wide"
        if w >= t1:
            return "medium"
        return "narrow"

    # ── Query API ────────────────────────────────────────────────────────

    def macro_label_for_entry(self, ts) -> Optional[str]:
        """Return the macro label to apply for an entry at timestamp `ts`.

        parity: same-day label (matches parquet).
        live:   most recent prior-day label (≤ 7 days lookback for weekends).
        """
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        day = ts.date()
        if self.mode == "parity":
            return self._macro_by_date.get(day)
        # live
        for offset in range(1, 8):
            d = day - timedelta(days=offset)
            if d in self._macro_by_date:
                return self._macro_by_date[d]
        return None

    def micro_label_for_entry(self, ts) -> Optional[str]:
        """Return the micro fine label as-of `ts` (most recent fractal whose
        timestamp ≤ ts). Returns None if no fractals committed yet."""
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if not self._fractal_ts_list:
            return None
        idx = bisect.bisect_right(self._fractal_ts_list, ts) - 1
        if idx < 0:
            return None
        return self._fractals[idx]["fine"]

    # ── Introspection (used by validator) ────────────────────────────────

    @property
    def macro_by_date(self) -> dict:
        return dict(self._macro_by_date)

    @property
    def fractals(self) -> list:
        """List of {ts, bar, kind, price, coarse, fine} sorted ascending."""
        return list(self._fractals)
