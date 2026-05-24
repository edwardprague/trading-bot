"""
analyse_pre_session.py — pre-London-open fractal metrics, good days vs bad days
================================================================================
For each trading day in 2025 GBPUSD (v1 parity backtest), look at the last
6 N=2 fractals confirmed before 8:00 UTC and compute the per-day metrics
that mirror the BD page's fractal table:

  • lower-high % among consecutive H-fractal pairs
  • lower-low  % among consecutive L-fractal pairs
  • mean ADX
  • H1 / H3 / H6 of the most recent fractal (current swing height
    + 3-fractal and 6-fractal rolling means of height)
  • H1-vs-H3-vs-H6 trend (expanding / stable / contracting)
  • mean VD high / mean VD low (vert_dist of H / L fractals)

Then compare distributions between Good days (entry-day net P&L > 0), Bad
days (net P&L < 0), and Neutral days (no trades). Report the strongest
separating metric(s) and search for a 70%+ good-day-capture combination.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/sessions/laughing-elegant-shannon/mnt/trading-bot")
OUTPUTS_DIR  = Path("/sessions/laughing-elegant-shannon/mnt/outputs")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OUTPUTS_DIR) not in sys.path:
    sys.path.insert(0, str(OUTPUTS_DIR))
os.chdir(str(PROJECT_ROOT))


# ── 1. Run the 2025 parity backtest, group P&L by entry day ─────────────────

def run_backtest_2025() -> pd.DataFrame:
    """Reuse validate_streaming.run_backtest_with_gate for the parity gate.
    Returns DataFrame of trades."""
    # Force parity-mode env and v1 settings exactly as the dashboard uses
    for k, v in {
        "REGIME_MODE":          "parity",
        "STRATEGY_VERSION":     "v2",
        "INSTRUMENT":           "GBPUSD",
        "INTERVAL":             "5m",
        "TRADE_DIRECTION":      "short_only",
        "EMA_LONG":             "133",
        "USE_EMA_FILTER":       "false",
        "FRACTAL_STOP_PIPS":    "30",
        "RRR_RISK":             "1",
        "RRR_REWARD":           "1",
        "MAX_DAILY_LOSSES":     "2",
        "BLOCKED_HOURS":        "4,5,6,8,10,11,14,17",
        "ALLOWED_MACRO_REGIMES": "strong_down,staircase_down",
        "ALLOWED_MICRO_REGIMES":
            "trending_fast_down,trending_medium_down,trending_slow_down,"
            "ranging_narrow,ranging_medium,ranging_wide,transitioning",
    }.items():
        os.environ[k] = v

    # Re-import strategy_v2 cleanly so env changes take effect
    if "strategy_v2" in sys.modules:
        del sys.modules["strategy_v2"]
    import strategy_v2 as strat
    df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                           start_date="2025-01-01", end_date="2025-12-31")
    df = strat.add_indicators(df)
    trades_df, _eq, _meta = strat.run_backtest(df)
    return trades_df


def bucket_trades_by_day(trades: pd.DataFrame) -> pd.DataFrame:
    """Group trades by entry-day UTC, sum P&L per day. Returns DataFrame
    indexed by date with columns: n_trades, gross_pnl, n_wins, n_losses."""
    trades = trades.copy()
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["entry_day"] = trades["entry_ts"].dt.date
    grouped = trades.groupby("entry_day").agg(
        n_trades = ("pnl", "size"),
        gross_pnl= ("pnl", "sum"),
        n_wins   = ("win", "sum"),
        n_losses = ("win", lambda s: int((~s.astype(bool)).sum())),
    )
    return grouped


def classify_days(days_in_window: list[date],
                  daily_pnl: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame keyed by date for every calendar day in the window.
    Days with trades → 'good' (pnl > 0) or 'bad' (pnl < 0); days without
    trades → 'neutral'."""
    rows = []
    for d in days_in_window:
        if d in daily_pnl.index:
            r = daily_pnl.loc[d]
            cat = "good" if r["gross_pnl"] > 0 else (
                  "bad"  if r["gross_pnl"] < 0 else "neutral")
            rows.append({
                "date": d, "category": cat,
                "n_trades": int(r["n_trades"]), "pnl": float(r["gross_pnl"]),
            })
        else:
            rows.append({"date": d, "category": "neutral",
                          "n_trades": 0, "pnl": 0.0})
    return pd.DataFrame(rows)


# ── 2. Build fractal table with H1/H3/H6 ─────────────────────────────────────

def build_fractal_table() -> pd.DataFrame:
    """Load regime_labels.parquet and compute height + h3 + h6 per fractal,
    mirroring strategy_v2.py:1874-1889."""
    table = pd.read_parquet(PROJECT_ROOT / "data" / "regime_labels.parquet")
    table = table.sort_values(["timestamp", "kind"]).reset_index(drop=True)
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)

    PIP = 10000

    # height = distance from current pivot to nearest opposite-kind pivot
    # in the last 5 entries of the classified list (mirrors strategy_v2)
    heights: list[float] = [np.nan] * len(table)
    classified: list[dict] = []  # rolling history of {price, kind}
    for i, row in table.iterrows():
        h = np.nan
        for back in reversed(classified[-5:]):
            if back["kind"] != row["kind"]:
                h = round(abs(row["price"] - back["price"]) * PIP, 1)
                break
        heights[i] = h
        classified.append({"price": float(row["price"]), "kind": row["kind"]})

    table["height"] = heights

    # h3 / h6 = rolling means of the last N non-null heights including current
    h6_vals: list[float] = [np.nan] * len(table)
    h3_vals: list[float] = [np.nan] * len(table)
    recent: list[float] = []
    for i, h in enumerate(heights):
        if not np.isnan(h):
            recent.append(h)
        if recent:
            window6 = recent[-6:]
            window3 = recent[-3:]
            h6_vals[i] = round(sum(window6) / len(window6), 1)
            h3_vals[i] = round(sum(window3) / len(window3), 1)
    table["h3"] = h3_vals
    table["h6"] = h6_vals

    # vert_dist column rename for clarity — already in parquet as v_dist_pips
    return table


# ── 3. Pre-session window per day ────────────────────────────────────────────

def pre_session_window(fr: pd.DataFrame, day: date,
                       n_lookback: int = 6) -> pd.DataFrame:
    """Return up to `n_lookback` last fractals before 08:00 UTC on `day`."""
    cutoff = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=8)
    sub    = fr[fr["timestamp"] < cutoff].tail(n_lookback)
    return sub


def pre_session_metrics(window: pd.DataFrame) -> dict:
    """Compute the per-day pre-session metrics."""
    if window.empty:
        return {"n_pre": 0}

    # Highs and lows within the window
    H = window[window["kind"] == "H"].sort_values("timestamp")
    L = window[window["kind"] == "L"].sort_values("timestamp")

    # Lower-high % among consecutive H pairs
    if len(H) >= 2:
        hp = H["price"].to_numpy()
        lh_pairs   = int((hp[1:] < hp[:-1]).sum())
        n_h_pairs  = len(hp) - 1
        lh_pct     = 100.0 * lh_pairs / n_h_pairs
    else:
        lh_pct = np.nan
        n_h_pairs = 0

    # Lower-low % among consecutive L pairs
    if len(L) >= 2:
        lp = L["price"].to_numpy()
        ll_pairs   = int((lp[1:] < lp[:-1]).sum())
        n_l_pairs  = len(lp) - 1
        ll_pct     = 100.0 * ll_pairs / n_l_pairs
    else:
        ll_pct = np.nan
        n_l_pairs = 0

    # Mean ADX over the window
    mean_adx = float(window["adx"].mean())

    # Most recent fractal's H1 / H3 / H6 and the resulting trend
    latest = window.iloc[-1]
    h1 = float(latest["height"]) if pd.notna(latest["height"]) else np.nan
    h3 = float(latest["h3"]) if pd.notna(latest["h3"]) else np.nan
    h6 = float(latest["h6"]) if pd.notna(latest["h6"]) else np.nan

    h_trend = "n/a"
    if not (np.isnan(h1) or np.isnan(h3) or np.isnan(h6)):
        if h1 > h3 > h6:
            h_trend = "expanding"
        elif h1 < h3 < h6:
            h_trend = "contracting"
        else:
            h_trend = "stable"

    # Mean VD high / low (vert_dist of H / L fractals = v_dist_pips)
    vd_high = float(H["v_dist_pips"].mean()) if not H.empty else np.nan
    vd_low  = float(L["v_dist_pips"].mean()) if not L.empty else np.nan

    # Also: the coarse classifier output on this window (for context)
    # Use the existing _classify_raw via a quick wrapper to also surface the
    # raw 4-way label — useful as a fallback signal.
    coarse = "transitioning"
    if len(H) >= 2 and len(L) >= 2:
        # Compare with regime_analysis._classify_raw expectation; reuse logic
        h_dir = "down" if all(hp[i] < hp[i-1] for i in range(1, len(hp))) else \
                ("up"   if all(hp[i] > hp[i-1] for i in range(1, len(hp))) else None)
        l_dir = "down" if all(lp[i] < lp[i-1] for i in range(1, len(lp))) else \
                ("up"   if all(lp[i] > lp[i-1] for i in range(1, len(lp))) else None)
        if h_dir == "down" and l_dir == "down":
            coarse = "trending_down"
        elif h_dir == "up" and l_dir == "up":
            coarse = "trending_up"
        elif h_dir and l_dir and h_dir != l_dir:
            coarse = "transitioning"
        elif h_dir is None and l_dir is None:
            coarse = "ranging"
        else:
            coarse = "transitioning"

    return {
        "n_pre":     len(window),
        "n_h":       len(H),
        "n_l":       len(L),
        "lh_pct":    lh_pct,
        "ll_pct":    ll_pct,
        "mean_adx":  mean_adx,
        "h1":        h1,
        "h3":        h3,
        "h6":        h6,
        "h_trend":   h_trend,
        "vd_high":   vd_high,
        "vd_low":    vd_low,
        "coarse":    coarse,
    }


# ── 4. Aggregate comparison ─────────────────────────────────────────────────

def compare_metrics(daily: pd.DataFrame) -> None:
    """Print per-category means + medians + simple separation diagnostics."""
    good = daily[daily["category"] == "good"]
    bad  = daily[daily["category"] == "bad"]
    neut = daily[daily["category"] == "neutral"]

    print(f"\n{'='*72}")
    print(f"DAY CATEGORY COUNTS")
    print(f"{'='*72}")
    print(f"  Good days     : {len(good):>4d}  (total P&L +${good['pnl'].sum():,.0f})")
    print(f"  Bad days      : {len(bad):>4d}  (total P&L  ${bad['pnl'].sum():,.0f})")
    print(f"  Neutral days  : {len(neut):>4d}")

    # ── Headline metric distributions ──
    print(f"\n{'='*72}")
    print(f"METRIC DISTRIBUTIONS  (Good vs Bad, pre-08:00 UTC window of 6 fractals)")
    print(f"{'='*72}")
    rows = []
    for metric in ("lh_pct", "ll_pct", "mean_adx", "h1", "h3", "h6",
                   "vd_high", "vd_low"):
        g = good[metric].dropna()
        b = bad[metric].dropna()
        if len(g) == 0 or len(b) == 0:
            continue
        rows.append({
            "metric":   metric,
            "good_n":   len(g),
            "good_mean":float(g.mean()),
            "good_med": float(g.median()),
            "bad_n":    len(b),
            "bad_mean": float(b.mean()),
            "bad_med":  float(b.median()),
            "abs_mean_delta": abs(g.mean() - b.mean()),
            # Pooled std for an effect-size feel (Cohen's d numerator only)
            "cohens_d": (g.mean() - b.mean()) / np.sqrt(
                (g.var() + b.var()) / 2) if (g.var() > 0 and b.var() > 0) else float("nan"),
        })
    df_rows = pd.DataFrame(rows).sort_values(
        "cohens_d", key=lambda s: s.abs(), ascending=False)
    print(df_rows.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    # ── H-trend distribution ──
    print(f"\n{'='*72}")
    print(f"H1/H3/H6 SWING-TREND DISTRIBUTION")
    print(f"{'='*72}")
    for cat, df_cat in (("good", good), ("bad", bad)):
        if df_cat.empty:
            continue
        cnt = df_cat["h_trend"].value_counts(normalize=True) * 100
        print(f"  {cat:<8s}  expanding={cnt.get('expanding', 0):5.1f}%  "
              f"stable={cnt.get('stable', 0):5.1f}%  "
              f"contracting={cnt.get('contracting', 0):5.1f}%  "
              f"n/a={cnt.get('n/a', 0):5.1f}%")

    # ── Coarse pre-session regime ──
    print(f"\n{'='*72}")
    print(f"COARSE PRE-SESSION REGIME LABEL (from the last 6 pre-08:00 fractals)")
    print(f"{'='*72}")
    for cat, df_cat in (("good", good), ("bad", bad)):
        if df_cat.empty:
            continue
        cnt = df_cat["coarse"].value_counts(normalize=True) * 100
        ordered_keys = ["trending_down", "ranging", "transitioning", "trending_up"]
        bits = "  ".join(f"{k}={cnt.get(k, 0):5.1f}%" for k in ordered_keys)
        print(f"  {cat:<8s}  {bits}")

    return df_rows


def search_classifier(daily: pd.DataFrame, top_metrics: pd.DataFrame) -> None:
    """Try simple AND/OR threshold combinations on the top metrics and
    report the best good-day capture rate alongside bad-day exclusion."""
    print(f"\n{'='*72}")
    print(f"SIMPLE CLASSIFIER SEARCH  (goal: ≥70% good-day capture)")
    print(f"{'='*72}")
    good = daily[daily["category"] == "good"]
    bad  = daily[daily["category"] == "bad"]
    n_good, n_bad = len(good), len(bad)
    if not n_good or not n_bad:
        print("  Not enough sample to compare."); return

    # Single-threshold sweep on each candidate metric
    candidates = []
    for metric in ("lh_pct", "ll_pct", "mean_adx", "vd_high", "vd_low",
                   "h1", "h3", "h6"):
        g_vals = good[metric].dropna()
        b_vals = bad[metric].dropna()
        if g_vals.empty or b_vals.empty:
            continue
        # Try both "gate when >= t" and "gate when <= t" directions
        combined = pd.concat([g_vals, b_vals])
        candidate_thresholds = np.unique(combined.values)
        for direction in (">=", "<="):
            for t in candidate_thresholds:
                if direction == ">=":
                    g_pass = (g_vals >= t).mean()
                    b_pass = (b_vals >= t).mean()
                else:
                    g_pass = (g_vals <= t).mean()
                    b_pass = (b_vals <= t).mean()
                if g_pass >= 0.70 and (g_pass - b_pass) >= 0.10:
                    candidates.append({
                        "metric":      metric,
                        "direction":   direction,
                        "threshold":   t,
                        "good_pass":   g_pass * 100,
                        "bad_pass":    b_pass * 100,
                        "selectivity": (g_pass - b_pass) * 100,
                    })
    if candidates:
        cdf = pd.DataFrame(candidates).sort_values("selectivity", ascending=False).head(10)
        print("  TOP SINGLE-METRIC RULES (good_pass ≥ 70% AND good−bad ≥ 10pp):")
        print(cdf.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    else:
        print("  No single-metric rule clears good_pass ≥ 70% AND good−bad ≥ 10pp.")

    # Categorical: coarse pre-session label
    print(f"\n  H-TREND + COARSE-LABEL AND-COMBINATIONS:")
    rules = []
    for h_state in ("expanding", "stable", "contracting"):
        for coarse in ("trending_down", "ranging", "transitioning", "trending_up"):
            g_pass = ((good["h_trend"] == h_state) &
                      (good["coarse"]  == coarse)).mean()
            b_pass = ((bad["h_trend"] == h_state) &
                      (bad["coarse"]  == coarse)).mean()
            rules.append({
                "h_trend": h_state, "coarse": coarse,
                "good_pass": g_pass * 100, "bad_pass": b_pass * 100,
                "selectivity": (g_pass - b_pass) * 100,
            })
    rdf = pd.DataFrame(rules).sort_values("selectivity", ascending=False).head(8)
    print(rdf.to_string(index=False, float_format=lambda x: f"{x:6.1f}"))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*72}")
    print(f"PRE-SESSION FRACTAL METRICS — GOOD DAYS vs BAD DAYS")
    print(f"Window: 2025-01-01 → 2025-12-31  GBPUSD 5m  v1 parity")
    print(f"{'='*72}\n")

    print("[1] Running 2025 parity backtest …")
    trades = run_backtest_2025()
    daily_pnl = bucket_trades_by_day(trades)
    print(f"    trades: {len(trades)}  trading-days-with-trades: {len(daily_pnl)}")
    print(f"    total P&L: +${trades['pnl'].sum():,.2f}\n")

    # Calendar trading days = days that have bars in the parquet
    print("[2] Building fractal table with H1/H3/H6 …")
    fr_table = build_fractal_table()
    # Filter to 2025 fractals plus a Dec-2024 warm-up tail so the 6-fractal
    # pre-session lookbacks on early January are populated.
    fr_2025 = fr_table[
        (fr_table["timestamp"] >= pd.Timestamp("2024-12-25", tz="UTC")) &
        (fr_table["timestamp"] <  pd.Timestamp("2026-01-01", tz="UTC"))
    ].reset_index(drop=True)
    print(f"    fractals: {len(fr_2025)} (incl. late-Dec 2024 warm-up)\n")

    # Calendar days = unique dates with at least one fractal in 2025
    days_with_fractals = sorted({
        ts.date() for ts in fr_2025["timestamp"]
        if ts >= pd.Timestamp("2025-01-01", tz="UTC")
    })
    print(f"[3] Trading days in 2025: {len(days_with_fractals)}")

    day_class = classify_days(days_with_fractals, daily_pnl)

    # Compute pre-session metrics per day
    print("[4] Computing pre-session metrics per day …")
    rows = []
    for d in days_with_fractals:
        w = pre_session_window(fr_2025, d, n_lookback=6)
        m = pre_session_metrics(w)
        m["date"] = d
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    merged = day_class.merge(metrics_df, on="date", how="left")
    # Drop days where pre-session window was empty
    merged = merged[merged["n_pre"] > 0].reset_index(drop=True)

    print(f"    days with non-empty pre-session window: {len(merged)}")
    print(f"    by category: ")
    print(merged["category"].value_counts().to_string().replace("\n", "\n      "))

    # Aggregate comparison
    top_metrics = compare_metrics(merged)

    # Classifier search
    search_classifier(merged, top_metrics)

    # Save for later inspection
    merged.to_csv(OUTPUTS_DIR / "pre_session_metrics_2025.csv", index=False)
    print(f"\n  full per-day metrics written → "
          f"pre_session_metrics_2025.csv")

    return merged


if __name__ == "__main__":
    main()
