"""
sweep_live_v2.py — focused live-mode threshold sweep
=====================================================
Replicates a Discovery run on 2025 GBPUSD in live mode with v1 fixed
settings, sampling T_h ∈ [5, 25] and T_adx ∈ [15, 35] uniformly per
trial. 60 trials by default. Records the same metrics Discovery records
plus the trial's T_h / T_adx for direct comparison.

Why bypass the official Discovery subprocess pipeline:
  • Each Discovery trial pays the streaming-classifier reset cost (~3s
    of warmup) inside its own subprocess plus subprocess startup
    (~0.5s). The CLI runs trials sequentially, so 60 trials = ~5 min.
  • The official pipeline writes results to discovery_results.json,
    which is the right destination once we want results in the
    dashboard. This script keeps the same params surface so its rows
    are trivially round-trippable into an official Discovery run later.

Strategy_v2 is re-imported per trial because the macro-classifier
thresholds are read at module load via env vars. Cost is small
because the bars cache is hot.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/sessions/laughing-elegant-shannon/mnt/trading-bot")
OUTPUTS_DIR  = Path("/sessions/laughing-elegant-shannon/mnt/outputs")
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))


BASE_ENV = {
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
    "REGIME_MODE":       "live",
    "MACRO_STRICT_SWINGS": "false",
    # Macro regime allow-list is ignored in live_v2 mode (the new
    # classifier replaces it). Micro stays at v1 default.
    "ALLOWED_MACRO_REGIMES": "",
    "ALLOWED_MICRO_REGIMES":
        "trending_fast_down,trending_medium_down,trending_slow_down,"
        "ranging_narrow,ranging_medium,ranging_wide,transitioning",
}


_STRAT_CACHE = {"strat": None, "df": None}


def _load_strategy_once():
    """One-time strategy_v2 import + bar fetch. Thresholds get mutated on
    the classifier instance per trial instead of re-importing, which cuts
    per-trial overhead from ~3s (subprocess startup + module init) to
    near-zero."""
    if _STRAT_CACHE["strat"] is not None:
        return _STRAT_CACHE["strat"], _STRAT_CACHE["df"]
    os.environ.update(BASE_ENV)
    # Initial thresholds don't matter — we'll overwrite them per trial.
    os.environ["MACRO_T_HEIGHT"] = "10"
    os.environ["MACRO_T_ADX"]    = "20"
    os.environ["MACRO_STRICT_SWINGS"] = "false"
    if "strategy_v2" in sys.modules:
        del sys.modules["strategy_v2"]
    import strategy_v2 as strat
    df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                           start_date="2025-01-01", end_date="2025-12-31")
    df = strat.add_indicators(df)
    _STRAT_CACHE["strat"] = strat
    _STRAT_CACHE["df"]    = df
    return strat, df


def run_one(t_h: float, t_adx: float, strict: bool = False) -> dict:
    """Run one live-mode 2025 backtest. Returns a row of metrics.

    Mutates the live classifier in-place so we don't pay the per-trial
    re-import cost. run_backtest() calls .reset() at its start so state
    is always cleared before each trial."""
    strat, df = _load_strategy_once()
    if strat._MACRO_CLF_V2 is None:
        raise RuntimeError("live-mode classifier not initialised — "
                            "REGIME_MODE env var was wrong at import")
    strat._MACRO_CLF_V2.t_height      = float(t_h)
    strat._MACRO_CLF_V2.t_adx         = float(t_adx)
    strat._MACRO_CLF_V2.strict_swings = bool(strict)
    trades_df, equity, _meta = strat.run_backtest(df)
    n = len(trades_df)
    if n == 0:
        return {"t_height": t_h, "t_adx": t_adx, "strict": strict,
                "trades": 0, "net": 0.0, "win_rate": 0.0, "pf": float("nan"),
                "max_dd_pct": 0.0}
    net = float(trades_df["pnl"].sum())
    wr  = 100.0 * trades_df["win"].sum() / n
    gross_pos = float(trades_df[trades_df["pnl"] > 0]["pnl"].sum())
    gross_neg = float(-trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    pf  = (gross_pos / gross_neg) if gross_neg > 0 else float("inf")
    # Max DD from equity curve, in %
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_pct = float(((peak - eq) / peak * 100).max())
    return {
        "t_height": round(t_h, 1),
        "t_adx":    round(t_adx, 1),
        "strict":   strict,
        "trades":   n,
        "net":      net,
        "win_rate": wr,
        "pf":       pf,
        "max_dd_pct": dd_pct,
    }


def grid_pairs() -> list[tuple[float, float]]:
    """Deterministic grid: T_h ∈ {6,9,12,15,18,21,24} × T_adx ∈ {16,20,24,28,32}.
    35 combinations, ordered consistently across runs."""
    th_vals  = [6, 9, 12, 15, 18, 21, 24]
    tadx_vals = [16, 20, 24, 28, 32]
    return [(float(h), float(a)) for h in th_vals for a in tadx_vals]


def main():
    grid       = grid_pairs()
    n_total    = len(grid)
    batch_from = int(os.environ.get("BATCH_FROM", 0))
    batch_to   = int(os.environ.get("BATCH_TO",   n_total))

    out_csv = OUTPUTS_DIR / "discovery_live_v2_sweep_2025.csv"
    # Append-mode: load existing rows so re-runs continue rather than restart.
    # Treat empty file as no existing data (avoids EmptyDataError on a fresh
    # truncate).
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    print(f"\n{'='*72}")
    print(f"LIVE-MODE THRESHOLD GRID — 2025 GBPUSD  [batch {batch_from}:{batch_to} of {n_total}]")
    print(f"  T_height: {{6,9,12,15,18,21,24}}   T_adx: {{16,20,24,28,32}}")
    print(f"  strict_swings: off (per LIVE_MACRO_V2_HONEST_REPORT.md)")
    print(f"{'='*72}\n")

    rows = existing.to_dict("records") if not existing.empty else []
    done = {(r["t_height"], r["t_adx"]) for r in rows}
    t0 = time.time()
    new_count = 0
    for idx in range(batch_from, min(batch_to, n_total)):
        th, ta = grid[idx]
        if (th, ta) in done:
            continue
        r  = run_one(th, ta, strict=False)
        rows.append(r)
        new_count += 1
        net_s = f"${r['net']:+,.0f}"
        print(f"  [{idx+1:>2d}/{n_total}] T_h={th:>5.1f} T_adx={ta:>5.1f}  "
              f"trades={r['trades']:>4d}  net={net_s:>12s}  "
              f"wr={r['win_rate']:>4.1f}%  pf={r['pf']:>5.2f}  dd={r['max_dd_pct']:.1f}%")

    elapsed = time.time() - t0
    print(f"\n  {new_count} new trials in {elapsed:.0f}s "
          f"({elapsed/max(1,new_count):.1f}s/trial)\n")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv} ({len(sweep)} total rows)\n")

    # Headline aggregations
    print(f"{'='*72}")
    print(f"RESULTS")
    print(f"{'='*72}")
    print(f"  unique combos sampled:   {len(sweep)}")
    print(f"  trials with net > 0:     {int((sweep['net'] > 0).sum())}")
    print(f"  median net P&L:          ${sweep['net'].median():+,.0f}")
    print(f"  best net P&L:            ${sweep['net'].max():+,.0f}")
    print(f"  worst net P&L:           ${sweep['net'].min():+,.0f}")

    print(f"\n  TOP 10 BY NET P&L:")
    top = sweep.sort_values("net", ascending=False).head(10)
    print(top[["t_height","t_adx","trades","net","win_rate","pf","max_dd_pct"]]
            .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))

    # Passing criteria (Discovery defaults): PF≥1.5, trades≥50, DD1≤10%
    passing = sweep[
        (sweep["pf"] >= 1.5) &
        (sweep["trades"] >= 50) &
        (sweep["max_dd_pct"] <= 10.0)
    ].sort_values("net", ascending=False)
    print(f"\n  PASSING DISCOVERY DEFAULTS (PF≥1.5, trades≥50, DD1≤10%):  "
          f"{len(passing)} of {len(sweep)}")
    if not passing.empty:
        print(passing[["t_height","t_adx","trades","net","win_rate","pf","max_dd_pct"]]
                .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    else:
        print(f"    NONE passed Discovery's default criteria.")


if __name__ == "__main__":
    main()
