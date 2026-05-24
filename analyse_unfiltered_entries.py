"""
analyse_unfiltered_entries.py — honest per-entry pre-fractal metrics
=====================================================================
Replaces the parity-conditioned pre-session analysis. Instead of looking at
*parity-filtered trades* and asking which would have survived H1+ADX, this
generates the universe of N=2 entry signals that would fire if the macro
look-ahead gate were ABSENT, then asks whether H1 + ADX at the triggering
fractal can correctly separate winners from losers.

This is the question the live_v2 macro gate actually answers:
  "given a candidate entry signal, should we let it through?"

Strategy
--------
  1. Run a 2025 GBPUSD backtest with ALLOWED_MACRO_REGIMES="" (gate
     disabled). Micro stays at its v1 default. All other filters (direction,
     time, EMA, SL bounds, daily-loss cap) remain active. The resulting
     trade list is the universe of would-be entries.
  2. Independently re-walk the same bars to detect every N=2 fractal and
     compute its H1 / H3 / H6 swing-height metrics + ADX at the fractal
     bar. This mirrors what macro_classifier_v2 sees at decision time.
  3. Join trades to fractals by `fractal_bar` so each entry carries the
     metrics the live classifier would have seen for it.
  4. Split good / bad by trade.result (TP vs SL), compute distributions,
     sweep thresholds.

The output answers two distinct questions:
  • Distribution separation (Cohen's d) — do H1 and ADX actually separate
    winning from losing entries?
  • Net P&L curves — for any (T_h, T_adx) combination, what trade list
    and net P&L do we get?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/sessions/laughing-elegant-shannon/mnt/trading-bot")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

PIP = 10000


# ── Helper: detect fractals and compute H1/H3/H6 per fractal ────────────────

def build_fractal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """For each N=2 fractal in df, return a row with the fractal bar
    index, kind, price, ADX, H1, H3, H6. Mirrors strategy_v2.py:1874-1889."""
    highs = df["High"].to_numpy()
    lows  = df["Low"].to_numpy()
    adxs  = df["adx"].to_numpy()

    last_h_price: float | None = None
    last_l_price: float | None = None
    heights: list[float] = []
    rows: list[dict] = []

    for fi in range(2, len(df) - 2):
        fh, fl = highs[fi], lows[fi]
        is_ph = (fh > highs[fi-1] and fh > highs[fi-2]
                 and fh > highs[fi+1] and fh > highs[fi+2])
        is_pl = (fl < lows[fi-1] and fl < lows[fi-2]
                 and fl < lows[fi+1] and fl < lows[fi+2])
        if not (is_ph or is_pl):
            continue

        adx_fi = float(adxs[fi]) if not np.isnan(adxs[fi]) else None
        events = []
        if is_ph: events.append(("H", float(fh)))
        if is_pl: events.append(("L", float(fl)))

        for kind, price in events:
            height: float | None = None
            if kind == "H" and last_l_price is not None:
                height = abs(price - last_l_price) * PIP
            elif kind == "L" and last_h_price is not None:
                height = abs(price - last_h_price) * PIP

            if height is not None:
                heights.append(height)
                h1 = height
                h3 = sum(heights[-3:]) / min(3, len(heights))
                h6 = sum(heights[-6:]) / min(6, len(heights))
            else:
                h1 = h3 = h6 = None

            if kind == "H":
                last_h_price = price
            else:
                last_l_price = price

            rows.append({
                "fractal_bar": fi,
                "kind":        kind,
                "price":       price,
                "adx":         adx_fi,
                "h1":          h1,
                "h3":          h3,
                "h6":          h6,
            })
    return pd.DataFrame(rows)


# ── Run the unfiltered backtest ─────────────────────────────────────────────

def run_unfiltered_backtest(start: str = "2025-01-01",
                             end: str   = "2025-12-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trades_df, bars_df). Trades_df is the unfiltered entry list
    (macro disabled); bars_df is the OHLC + indicator frame used to fit."""
    os.environ.update({
        "REGIME_MODE":       "parity",   # parity gate honors empty allow-list as pass-through
        "STRATEGY_VERSION":  "v2",
        "INSTRUMENT":        "GBPUSD",
        "INTERVAL":          "5m",
        "TRADE_DIRECTION":   "short_only",
        "EMA_LONG":          "133",
        "USE_EMA_FILTER":    "false",
        "FRACTAL_STOP_PIPS": "30",
        "RRR_RISK":          "1",
        "RRR_REWARD":        "1",
        "MAX_DAILY_LOSSES":  "2",
        "BLOCKED_HOURS":     "4,5,6,8,10,11,14,17",
        # Macro gate DISABLED — every otherwise-eligible entry survives.
        "ALLOWED_MACRO_REGIMES": "",
        # Micro stays at v1 default (unchanged in the live_v2 build).
        "ALLOWED_MICRO_REGIMES":
            "trending_fast_down,trending_medium_down,trending_slow_down,"
            "ranging_narrow,ranging_medium,ranging_wide,transitioning",
    })
    if "strategy_v2" in sys.modules:
        del sys.modules["strategy_v2"]
    import strategy_v2 as strat

    df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                           start_date=start, end_date=end)
    df = strat.add_indicators(df)
    trades_df, _eq, _meta = strat.run_backtest(df)
    return trades_df, df


# ── Join trades to fractal metrics ─────────────────────────────────────────

def join_trades_to_fractals(trades: pd.DataFrame,
                             fractal_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-entry join. Shorts trigger off H-kind fractals (lower-high);
    longs off L-kind (higher-low). Use a plain merge for clarity."""
    t = trades.copy()
    t["frk_kind"] = t["direction"].map({"short": "H", "long": "L"})
    t["fractal_bar"] = t["fractal_bar"].astype(int)
    fm = fractal_metrics.rename(columns={"kind": "frk_kind", "adx": "adx_fr"})
    fm = fm[["fractal_bar", "frk_kind", "h1", "h3", "h6", "adx_fr"]]
    return t.merge(fm, on=["fractal_bar", "frk_kind"], how="left")


# ── Analysis + sweep ────────────────────────────────────────────────────────

def signal_separation(joined: pd.DataFrame) -> None:
    """Cohen's d per metric, good (TP) vs bad (SL)."""
    good = joined[joined["result"] == "TP"]
    bad  = joined[joined["result"] == "SL"]
    n_good, n_bad = len(good), len(bad)
    print(f"\n{'='*72}\nSIGNAL SEPARATION — winners (TP) vs losers (SL)")
    print(f"{'='*72}")
    print(f"  TP trades: {n_good} (+${good['pnl'].sum():,.0f})")
    print(f"  SL trades: {n_bad}  ({bad['pnl'].sum():+,.0f} = ${bad['pnl'].sum():,.0f})")

    print(f"\n  {'metric':>8s}  {'good_mean':>10s}  {'good_med':>9s}  "
          f"{'bad_mean':>9s}  {'bad_med':>8s}  {'cohens_d':>8s}")
    print(f"  " + "-" * 65)
    for col in ("h1", "h3", "h6", "adx_fr"):
        g = good[col].dropna()
        b = bad[col].dropna()
        if g.empty or b.empty:
            continue
        gm, bm = float(g.mean()), float(b.mean())
        gv, bv = float(g.var()), float(b.var())
        d = (gm - bm) / np.sqrt((gv + bv) / 2) if (gv > 0 and bv > 0) else float("nan")
        gmed = float(g.median()); bmed = float(b.median())
        print(f"  {col:>8s}  {gm:>10.2f}  {gmed:>9.2f}  {bm:>9.2f}  "
              f"{bmed:>8.2f}  {d:>8.3f}")


def threshold_sweep(joined: pd.DataFrame) -> pd.DataFrame:
    """For a grid of (T_h, T_adx, strict), report kept count + P&L + win rate."""
    print(f"\n{'='*72}\nTHRESHOLD SWEEP — net P&L for each operating point")
    print(f"{'='*72}")
    n_total = len(joined)
    total_pnl = float(joined["pnl"].sum())
    print(f"  unfiltered: {n_total} trades, net=${total_pnl:+,.0f}, "
          f"wr={100.0 * joined['win'].sum() / n_total:.1f}%\n")

    rows = []
    for th in (0, 5, 8, 10, 12, 15, 18, 20, 25, 30):
        for ta in (0, 15, 18, 20, 22, 25, 28, 30, 33):
            for strict in (False, True):
                mask = pd.Series(True, index=joined.index)
                if th > 0:
                    mask &= (joined["h1"].fillna(0) >= th)
                if ta > 0:
                    mask &= (joined["adx_fr"].fillna(0) >= ta)
                if strict:
                    mask &= (joined["h1"].fillna(-1) >= joined["h3"].fillna(0)) & \
                            (joined["h3"].fillna(-1) >= joined["h6"].fillna(0))
                kept = joined[mask]
                n = len(kept)
                if n == 0:
                    continue
                net = float(kept["pnl"].sum())
                wr  = 100.0 * kept["win"].sum() / n
                rows.append({
                    "T_h": th, "T_adx": ta, "strict": strict,
                    "trades": n, "net": net, "wr": wr,
                    "pct_retained": 100.0 * n / n_total,
                })
    df = pd.DataFrame(rows)
    return df


def print_top_combos(sweep: pd.DataFrame, k: int = 12) -> None:
    print(f"\n  TOP-{k} BY NET P&L:")
    top = sweep.sort_values("net", ascending=False).head(k)
    print(f"  {'T_h':>4s} {'T_adx':>5s} {'strict':>6s}  "
          f"{'trades':>6s}  {'net':>11s}  {'wr':>5s}  {'kept%':>5s}")
    print(f"  " + "-" * 64)
    for _, r in top.iterrows():
        print(f"  {r['T_h']:>4.0f} {r['T_adx']:>5.0f} "
              f"{('ON' if r['strict'] else 'off'):>6s}  "
              f"{r['trades']:>6d}  ${r['net']:>+10,.0f}  "
              f"{r['wr']:>4.1f}%  {r['pct_retained']:>4.1f}%")

    # Combos closest to break-even-and-positive with reasonable trade counts
    print(f"\n  POSITIVE OPERATING POINTS (net > 0, trades ≥ 50):")
    viable = sweep[(sweep["net"] > 0) & (sweep["trades"] >= 50)].sort_values("net", ascending=False).head(10)
    if viable.empty:
        print(f"    NONE — no (T_h, T_adx) combination produces net > 0 with ≥50 trades.")
    else:
        for _, r in viable.iterrows():
            print(f"  {r['T_h']:>4.0f} {r['T_adx']:>5.0f} "
                  f"{('ON' if r['strict'] else 'off'):>6s}  "
                  f"{r['trades']:>6d}  ${r['net']:>+10,.0f}  "
                  f"{r['wr']:>4.1f}%  {r['pct_retained']:>4.1f}%")


def main():
    print("[1] Running unfiltered backtest (macro disabled, micro v1 default) …")
    trades, bars = run_unfiltered_backtest()
    print(f"    unfiltered trades: {len(trades)}")
    print(f"    net P&L:           ${trades['pnl'].sum():+,.2f}")
    print(f"    win rate:          {100.0 * trades['win'].sum() / len(trades):.1f}%")

    print("\n[2] Computing per-fractal H1/H3/H6 from bars …")
    fr = build_fractal_metrics(bars)
    print(f"    fractals:          {len(fr):,}")

    print("\n[3] Joining trades to fractals …")
    joined = join_trades_to_fractals(trades, fr)
    n_attached = int(joined["h1"].notna().sum())
    print(f"    trades with attached fractal metrics: {n_attached}/{len(joined)}")

    signal_separation(joined)

    sweep = threshold_sweep(joined)
    print_top_combos(sweep)

    # Persist for inspection
    out_dir = Path("/sessions/laughing-elegant-shannon/mnt/outputs")
    joined.to_csv(out_dir / "unfiltered_entries_2025.csv", index=False)
    sweep.to_csv(out_dir / "threshold_sweep_2025.csv", index=False)
    print(f"\n  wrote {out_dir / 'unfiltered_entries_2025.csv'} ({len(joined)} rows)")
    print(f"  wrote {out_dir / 'threshold_sweep_2025.csv'} ({len(sweep)} combos)")


if __name__ == "__main__":
    main()
