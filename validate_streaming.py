"""
validate_streaming.py — Parity + live-mode diff for the streaming classifier
=============================================================================
Run from inside the trading-bot directory so strategy_v2 imports cleanly:

    cd trading-bot
    python3 -c "import sys; sys.path.insert(0, '/path/to/outputs'); \
        import validate_streaming as v; v.main()"

Two stages:
  1. Label parity — streaming(parity) vs data/regime_labels.parquet
     - macro_by_date: should be identical for every dated key.
     - micro per fractal: streaming's fractal list (ts, kind, coarse, fine)
       should match the parquet's per-row label.
  2. Trade-list parity — backtest with streaming-gate monkey patch vs
     backtest with parquet-gate (the default). Trade lists must match.
  3. Live-vs-parity diff — same backtest harness, mode="live" instead of
     "parity". Reports how many trades shift under no-look-ahead semantics.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date as _date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Validator runs from outputs/ but imports trading-bot's strategy_v2 and
# regime_streaming. Caller is responsible for setting sys.path; the bash
# harness does so explicitly.

PROJECT_ROOT = Path(__file__).resolve().parent  # outputs/
# trading-bot is mounted alongside outputs/
TRADING_BOT = None  # filled by main()


# ── Stage 1: load parquet ground truth ──────────────────────────────────────

def load_parquet_truth(parquet_path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    # Per-fractal micro labels
    micro_rows = df[["timestamp", "kind", "fractal_bar", "coarse_label", "regime"]].copy()
    micro_rows["timestamp"] = pd.to_datetime(micro_rows["timestamp"], utc=True)
    micro_rows = micro_rows.sort_values("timestamp").reset_index(drop=True)

    # Macro from parquet metadata
    meta = table.schema.metadata or {}
    macro_by_date: dict = {}
    for k, v in meta.items():
        kd = k.decode() if isinstance(k, (bytes, bytearray)) else k
        if kd in ("regime_analysis", "regime_labeler"):
            payload = json.loads(v.decode() if isinstance(v, (bytes, bytearray)) else v)
            macro_by_date = payload.get("macro_by_date") or macro_by_date
            thresholds = payload.get("thresholds")
    if not macro_by_date:
        sidecar = parquet_path.parent / "macro_by_date.json"
        if sidecar.exists():
            macro_by_date = json.loads(sidecar.read_text())

    # macro_by_date values may be {"label": "...", "details": {...}} or plain strings.
    macro_clean = {}
    for k, v in macro_by_date.items():
        if isinstance(v, dict):
            macro_clean[k] = v.get("label")
        else:
            macro_clean[k] = v

    return micro_rows, macro_clean, thresholds


# ── Stage 2: build a streaming classifier over the same window ───────────────

def build_streaming(df_bars: pd.DataFrame, mode: str, thresholds: dict):
    from regime_streaming import StreamingRegimeClassifier
    clf = StreamingRegimeClassifier(mode=mode, thresholds=thresholds)
    clf.ingest(df_bars)
    return clf


# ── Stage 3: label parity comparison ─────────────────────────────────────────

def compare_macro(parquet_macro: dict, streaming_macro: dict) -> dict:
    """Compare {date_str -> label} dicts. Returns summary + sample mismatches."""
    p_keys = set(parquet_macro.keys())
    # streaming_macro is keyed by datetime.date objects; convert to str
    s_macro = {d.isoformat(): lbl for d, lbl in streaming_macro.items()}
    s_keys = set(s_macro.keys())

    common = sorted(p_keys & s_keys)
    mismatches = [(d, parquet_macro[d], s_macro[d]) for d in common
                  if parquet_macro[d] != s_macro[d]]
    return {
        "parquet_only_days": sorted(p_keys - s_keys)[:5],
        "streaming_only_days": sorted(s_keys - p_keys)[:5],
        "common_days":      len(common),
        "matches":          len(common) - len(mismatches),
        "mismatches":       len(mismatches),
        "sample_mismatches": mismatches[:10],
    }


def compare_micro(parquet_rows: pd.DataFrame, streaming_clf) -> dict:
    """Compare per-fractal labels. We align by (timestamp, kind) since the
    parquet has one row per (fractal_bar, kind) and the streaming module
    emits one record per same."""
    s_records = streaming_clf.fractals  # list of dicts

    # Index streaming by (ts, kind) for fast lookup
    s_index = {}
    for r in s_records:
        s_index[(r["ts"], r["kind"])] = r

    fine_mismatches = []
    coarse_mismatches = []
    missing_in_streaming = 0
    missing_in_parquet = 0

    p_keys = set()
    for _, row in parquet_rows.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "tz_convert"):
            ts = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
        key = (pd.Timestamp(ts), row["kind"])
        p_keys.add(key)
        s = s_index.get(key)
        if s is None:
            missing_in_streaming += 1
            continue
        if s["coarse"] != row["coarse_label"]:
            coarse_mismatches.append({
                "ts": str(ts), "kind": row["kind"],
                "parquet": row["coarse_label"], "streaming": s["coarse"],
            })
        if s["fine"] != row["regime"]:
            fine_mismatches.append({
                "ts": str(ts), "kind": row["kind"],
                "parquet": row["regime"], "streaming": s["fine"],
            })

    s_keys = set(s_index.keys())
    missing_in_parquet = len(s_keys - p_keys)

    return {
        "parquet_rows":          len(parquet_rows),
        "streaming_records":     len(s_records),
        "missing_in_streaming":  missing_in_streaming,
        "missing_in_parquet":    missing_in_parquet,
        "coarse_mismatches":     len(coarse_mismatches),
        "fine_mismatches":       len(fine_mismatches),
        "sample_coarse":         coarse_mismatches[:10],
        "sample_fine":           fine_mismatches[:10],
    }


# ── Stage 4: trade-list parity ──────────────────────────────────────────────

def run_backtest_with_gate(start_date: str, end_date: str,
                            macro_fn, micro_fn, allowed_macro: set,
                            allowed_micro: set) -> list:
    """Run strategy_v2's backtest with the regime gate monkey-patched.

    macro_fn / micro_fn are callables taking a timestamp and returning a label.
    """
    import strategy_v2 as strat

    # Force the active-version config
    strat.TRADE_DIRECTION = "short_only"

    # Replace the module-level gate hooks
    def _patched_macro(ts):
        if not allowed_macro:
            return True, ""
        label = macro_fn(ts)
        if label is None:
            return True, ""
        if label in allowed_macro:
            return True, ""
        return False, "macro_regime"

    def _patched_micro(ts):
        if not allowed_micro:
            return True, ""
        label = micro_fn(ts)
        if label is None:
            return True, ""
        if label in allowed_micro:
            return True, ""
        return False, "micro_regime"

    strat._check_macro_regime = _patched_macro
    strat._check_micro_regime = _patched_micro
    strat.ALLOWED_MACRO_KEYS = set(allowed_macro)
    strat.ALLOWED_MICRO_KEYS = set(allowed_micro)

    df = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                           start_date=start_date, end_date=end_date)
    df = strat.add_indicators(df)
    trades_df, _equity, _meta = strat.run_backtest(df)
    # run_backtest returns a DataFrame; normalise to list-of-dicts for comparison
    trades = trades_df.to_dict("records") if hasattr(trades_df, "to_dict") else list(trades_df)
    return trades, df


def trade_key(t):
    return (str(t["entry_ts"]), t["direction"], round(t["entry"], 6),
            round(t["exit"], 6), t["result"])


def compare_trade_lists(a: list, b: list) -> dict:
    a_keys = [trade_key(t) for t in a]
    b_keys = [trade_key(t) for t in b]
    same = sum(1 for x, y in zip(a_keys, b_keys) if x == y)
    diff_count = max(len(a), len(b)) - same

    # First divergence
    first_div = None
    for i, (x, y) in enumerate(zip(a_keys, b_keys)):
        if x != y:
            first_div = {"index": i, "a": x, "b": y}
            break

    pnl_a = sum(t["pnl"] for t in a)
    pnl_b = sum(t["pnl"] for t in b)
    return {
        "a_count": len(a),
        "b_count": len(b),
        "exact_match": a_keys == b_keys,
        "matched_in_order": same,
        "first_divergence": first_div,
        "pnl_a": pnl_a,
        "pnl_b": pnl_b,
        "pnl_diff": pnl_b - pnl_a,
    }


# ── Main driver ──────────────────────────────────────────────────────────────

def main(start_date: str = "2025-01-01", end_date: str = "2025-12-31",
         trading_bot_dir: str = "/sessions/laughing-elegant-shannon/mnt/trading-bot"):
    global TRADING_BOT
    TRADING_BOT = Path(trading_bot_dir)
    os.chdir(TRADING_BOT)
    if str(TRADING_BOT) not in sys.path:
        sys.path.insert(0, str(TRADING_BOT))

    # Env to match active version v1 (strategy_version=v2, short-only, etc.)
    os.environ["STRATEGY_VERSION"] = "v2"
    os.environ["INSTRUMENT"]        = "GBPUSD"
    os.environ["INTERVAL"]          = "5m"
    os.environ["TRADE_DIRECTION"]   = "short_only"
    os.environ["EMA_LONG"]          = "133"
    os.environ["USE_EMA_FILTER"]    = "false"
    os.environ["FRACTAL_STOP_PIPS"] = "30"
    os.environ["RRR_RISK"]          = "1"
    os.environ["RRR_REWARD"]        = "1"
    os.environ["MAX_DAILY_LOSSES"]  = "2"
    os.environ["BLOCKED_HOURS"]     = "4,5,6,8,10,11,14,17"
    os.environ["ALLOWED_MACRO_REGIMES"] = "strong_down,staircase_down"
    os.environ["ALLOWED_MICRO_REGIMES"] = ("trending_fast_down,trending_medium_down,"
                                            "trending_slow_down,ranging_narrow,"
                                            "ranging_medium,ranging_wide,transitioning")

    print(f"\n{'='*72}")
    print(f"VALIDATION  ::  GBPUSD 5m  ::  {start_date} → {end_date}")
    print(f"{'='*72}\n")

    # ── Load parquet truth (covers the full historical range) ────────────
    parquet_path = TRADING_BOT / "data" / "regime_labels.parquet"
    print("[1] Loading parquet truth …")
    p_micro_rows, p_macro_by_date, p_thresholds = load_parquet_truth(parquet_path)
    print(f"    parquet macro days: {len(p_macro_by_date)}")
    print(f"    parquet micro rows: {len(p_micro_rows)}")
    print(f"    parquet thresholds: {p_thresholds}")

    # ── Filter parquet truth to the validation window ────────────────────
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts   = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=1)
    p_micro_window = p_micro_rows[
        (p_micro_rows["timestamp"] >= start_ts) &
        (p_micro_rows["timestamp"] <  end_ts)
    ].reset_index(drop=True)
    p_macro_window = {d: v for d, v in p_macro_by_date.items()
                      if start_date <= d <= end_date}
    print(f"    in-window parquet macro days: {len(p_macro_window)}")
    print(f"    in-window parquet micro rows: {len(p_micro_window)}")

    # ── Fetch bars (same way strategy_v2 does it) ───────────────────────
    import strategy_v2 as strat
    df_bars = strat.fetch_data(strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
                                start_date=start_date, end_date=end_date)
    print(f"    bars loaded: {len(df_bars)} | "
          f"{df_bars.index[0]} → {df_bars.index[-1]}")

    # ── Build streaming classifier (parity mode) ─────────────────────────
    print("\n[2] Building streaming classifier (parity mode) …")
    clf_parity = build_streaming(df_bars, mode="parity", thresholds=p_thresholds)
    print(f"    streaming macro days: {len(clf_parity.macro_by_date)}")
    print(f"    streaming fractals:   {len(clf_parity.fractals)}")

    # ── Label parity ──────────────────────────────────────────────────────
    print("\n[3] Label parity (parity-mode streaming vs parquet) …")
    s_macro_in_window = {d: v for d, v in clf_parity.macro_by_date.items()
                          if start_date <= d.isoformat() <= end_date}
    macro_cmp = compare_macro(p_macro_window, s_macro_in_window)
    print(f"    macro common days:   {macro_cmp['common_days']}")
    print(f"    macro matches:       {macro_cmp['matches']}")
    print(f"    macro mismatches:    {macro_cmp['mismatches']}")
    if macro_cmp["sample_mismatches"]:
        print("    sample macro mismatches:")
        for d, pv, sv in macro_cmp["sample_mismatches"]:
            print(f"      {d}: parquet={pv!r:20s} streaming={sv!r}")

    micro_cmp = compare_micro(p_micro_window, clf_parity)
    print(f"    micro parquet rows:        {micro_cmp['parquet_rows']}")
    print(f"    micro streaming records:   {micro_cmp['streaming_records']}")
    print(f"    coarse mismatches:         {micro_cmp['coarse_mismatches']}")
    print(f"    fine   mismatches:         {micro_cmp['fine_mismatches']}")
    print(f"    parquet rows w/o streaming: {micro_cmp['missing_in_streaming']}")
    print(f"    streaming rows w/o parquet: {micro_cmp['missing_in_parquet']}")
    if micro_cmp["sample_fine"]:
        print("    sample fine mismatches:")
        for s in micro_cmp["sample_fine"]:
            print(f"      {s}")

    # ── Trade-list parity ────────────────────────────────────────────────
    print("\n[4] Trade-list comparison: parquet-gate vs streaming-parity-gate …")
    allowed_macro = {"strong_down", "staircase_down"}
    allowed_micro = {"trending_fast_down", "trending_medium_down", "trending_slow_down",
                     "ranging_narrow", "ranging_medium", "ranging_wide", "transitioning"}

    # Run A: parquet-backed (as the dashboard does)
    import importlib
    if "strategy_v2" in sys.modules:
        importlib.reload(sys.modules["strategy_v2"])
    import strategy_v2 as strat
    trades_parquet, _ = run_backtest_with_gate(
        start_date, end_date,
        macro_fn=lambda ts: strat._REGIME_MACRO_BY_DATE.get(
            pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d")
            if pd.Timestamp(ts).tzinfo
            else pd.Timestamp(ts).tz_localize("UTC").strftime("%Y-%m-%d")
        ),
        micro_fn=lambda ts: (
            strat._REGIME_MICRO_SERIES.asof(
                pd.Timestamp(ts).tz_convert("UTC")
                if pd.Timestamp(ts).tzinfo
                else pd.Timestamp(ts).tz_localize("UTC")
            ) if strat._REGIME_MICRO_SERIES is not None else None
        ),
        allowed_macro=allowed_macro,
        allowed_micro=allowed_micro,
    )

    # Run B: streaming parity-mode
    if "strategy_v2" in sys.modules:
        importlib.reload(sys.modules["strategy_v2"])
    import strategy_v2 as strat
    trades_stream_parity, _ = run_backtest_with_gate(
        start_date, end_date,
        macro_fn=clf_parity.macro_label_for_entry,
        micro_fn=clf_parity.micro_label_for_entry,
        allowed_macro=allowed_macro,
        allowed_micro=allowed_micro,
    )

    cmp_pq_vs_stream = compare_trade_lists(trades_parquet, trades_stream_parity)
    print(f"    parquet trades:      {cmp_pq_vs_stream['a_count']}")
    print(f"    streaming(parity):   {cmp_pq_vs_stream['b_count']}")
    print(f"    exact match:         {cmp_pq_vs_stream['exact_match']}")
    print(f"    matched in order:    {cmp_pq_vs_stream['matched_in_order']}")
    print(f"    parquet  P&L:        {cmp_pq_vs_stream['pnl_a']:+,.2f}")
    print(f"    streaming P&L:       {cmp_pq_vs_stream['pnl_b']:+,.2f}")
    print(f"    P&L diff:            {cmp_pq_vs_stream['pnl_diff']:+,.2f}")
    if cmp_pq_vs_stream["first_divergence"]:
        print(f"    first divergence:    {cmp_pq_vs_stream['first_divergence']}")

    # ── Live-mode diff (informational) ────────────────────────────────────
    print("\n[5] Live-mode diff (cBot-faithful streaming, no look-ahead) …")
    clf_live = build_streaming(df_bars, mode="live", thresholds=p_thresholds)
    if "strategy_v2" in sys.modules:
        importlib.reload(sys.modules["strategy_v2"])
    import strategy_v2 as strat
    trades_stream_live, _ = run_backtest_with_gate(
        start_date, end_date,
        macro_fn=clf_live.macro_label_for_entry,
        micro_fn=clf_live.micro_label_for_entry,
        allowed_macro=allowed_macro,
        allowed_micro=allowed_micro,
    )
    cmp_parity_vs_live = compare_trade_lists(trades_stream_parity, trades_stream_live)
    print(f"    streaming(parity):   {cmp_parity_vs_live['a_count']} trades, "
          f"P&L {cmp_parity_vs_live['pnl_a']:+,.2f}")
    print(f"    streaming(live):     {cmp_parity_vs_live['b_count']} trades, "
          f"P&L {cmp_parity_vs_live['pnl_b']:+,.2f}")
    print(f"    delta (live − parity): "
          f"{cmp_parity_vs_live['b_count'] - cmp_parity_vs_live['a_count']:+d} trades, "
          f"{cmp_parity_vs_live['pnl_diff']:+,.2f} P&L")

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    parity_ok = (macro_cmp["mismatches"] == 0
                 and micro_cmp["fine_mismatches"] == 0
                 and micro_cmp["coarse_mismatches"] == 0
                 and cmp_pq_vs_stream["exact_match"])
    print(f"PARITY MODE LOGIC: {'PASS ✓' if parity_ok else 'FAIL ✗'}")
    print(f"  - Macro labels:   {'match' if macro_cmp['mismatches'] == 0 else 'DIFFER'}")
    print(f"  - Micro coarse:   {'match' if micro_cmp['coarse_mismatches'] == 0 else 'DIFFER'}")
    print(f"  - Micro fine:     {'match' if micro_cmp['fine_mismatches'] == 0 else 'DIFFER'}")
    print(f"  - Trade list:     {'match' if cmp_pq_vs_stream['exact_match'] else 'DIFFER'}")
    print(f"{'='*72}\n")

    return {
        "macro_cmp":           macro_cmp,
        "micro_cmp":           micro_cmp,
        "pq_vs_stream":        cmp_pq_vs_stream,
        "parity_vs_live":      cmp_parity_vs_live,
        "parity_pass":         parity_ok,
    }


if __name__ == "__main__":
    main()
