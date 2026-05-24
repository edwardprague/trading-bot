"""
validate_london_open.py — 4h-shifted macro label predictor analysis
====================================================================
Hypothesis: a regime label computed from a 24-hour window ending at the
8:00 UTC London open is a meaningfully better predictor of "today's"
actual completed daily macro label than the prior-day's daily label.

Methodology
-----------
For each trading day D in 2023-2025 GBPUSD:
  Ground truth  = classify_macro_regime(5m bars of calendar day D)
                  i.e. midnight UTC of D to midnight UTC of D+1.
                  This is what parity-mode Discovery sees and what
                  regime_labels.parquet stores.

  Prior-day     = classify_macro_regime(5m bars of calendar day D-1)
                  i.e. the label live mode currently uses to gate
                  today's entries.

  London-open   = classify_macro_regime(5m bars from 8:00 UTC on D-1
                  to 8:00 UTC on D). The existing classifier, unchanged,
                  applied to a 24h window that ends at London session
                  open instead of midnight. Uses no information from
                  after 8:00 UTC on D, so a live cBot could compute it.

The user's "4h" framing aligns to the 4h-boundary at 8:00 UTC; the
classifier itself runs on 5m bars (where its EMA-40 / N18 / displacement
thresholds remain at original calibration). We additionally sample the
shifted-window label at every 4h boundary (0/4/8/12/16/20 UTC) so the
time-of-day curve shows whether 8:00 UTC is the right sampling point.

Output
------
Headline match% per approach, per-regime breakdown, daily transition
rate (theoretical ceiling for prior-day baseline), time-of-day curve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/sessions/laughing-elegant-shannon/mnt/trading-bot")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import regime_analysis as ra  # noqa: E402

PIP = 10000  # GBPUSD


# ── Bar-loading helpers ─────────────────────────────────────────────────────

def load_5m_bars(start: str, end: str) -> pd.DataFrame:
    """Load GBPUSD 5m bars in [start, end). Index is UTC tz-aware."""
    df = pd.read_parquet(PROJECT_ROOT / "data" / "GBPUSD_5m.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end,   tz="UTC")
    return df[(df.index >= lo) & (df.index < hi)].copy()


def classify_window(bars: pd.DataFrame) -> str | None:
    """Run regime_analysis.classify_macro_regime on a window of bars.
    Returns the label or None if the window is too short."""
    if len(bars) <= max(ra.N18_LOOKBACK * 2 + 1, ra.EMA_MACRO_PERIOD):
        return None
    label, _ = ra.classify_macro_regime(bars)
    return label


# ── Per-day computation ─────────────────────────────────────────────────────

def compute_labels(df_5m: pd.DataFrame, hour_anchor: int) -> pd.DataFrame:
    """For each calendar day in df_5m, compute:
       ground_truth, prior_day, shifted_at_anchor.

    `hour_anchor` is the UTC hour at which the shifted 24h window ends
    (e.g., 8 → window is 08:00 D-1 to 08:00 D).
    """
    # Trading days = days with any bars in df_5m. UTC.
    days = sorted({ts.date() for ts in df_5m.index})

    rows = []
    for i in range(1, len(days)):
        D       = days[i]
        D_prev  = days[i - 1]

        # Ground-truth daily window: midnight UTC of D to midnight UTC of D+1
        day_start = pd.Timestamp(D, tz="UTC")
        day_end   = day_start + pd.Timedelta(days=1)
        bars_D    = df_5m[(df_5m.index >= day_start) & (df_5m.index < day_end)]
        gt        = classify_window(bars_D)

        # Prior-day daily window
        day_prev_start = pd.Timestamp(D_prev, tz="UTC")
        day_prev_end   = day_prev_start + pd.Timedelta(days=1)
        bars_prev      = df_5m[(df_5m.index >= day_prev_start) &
                               (df_5m.index <  day_prev_end)]
        pd_label       = classify_window(bars_prev)

        # Shifted window ending at `hour_anchor` on day D
        end_ts   = day_start + pd.Timedelta(hours=hour_anchor)
        start_ts = end_ts - pd.Timedelta(days=1)
        bars_sh  = df_5m[(df_5m.index >= start_ts) & (df_5m.index < end_ts)]
        sh_label = classify_window(bars_sh)

        rows.append({
            "date":         D,
            "ground_truth": gt,
            "prior_day":    pd_label,
            "shifted":      sh_label,
        })

    return pd.DataFrame(rows)


# ── Metrics ─────────────────────────────────────────────────────────────────

def match_pct(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Match percentage where both labels are non-null."""
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return 0, 0, 0.0
    matches = int((a[mask] == b[mask]).sum())
    return matches, n, 100.0 * matches / n


def per_regime_breakdown(df: pd.DataFrame, pred_col: str, gt_col: str) -> pd.DataFrame:
    """For each ground-truth regime, % of days where `pred_col` matched
    plus the most common confusion target."""
    out = []
    for regime, sub in df.dropna(subset=[gt_col, pred_col]).groupby(gt_col):
        total = len(sub)
        correct = int((sub[pred_col] == sub[gt_col]).sum())
        confusions = sub.loc[sub[pred_col] != sub[gt_col], pred_col].value_counts()
        top_conf = confusions.index[0] if len(confusions) else None
        top_conf_n = int(confusions.iloc[0]) if len(confusions) else 0
        out.append({
            "ground_truth": regime,
            "n_days":       total,
            "matched":      correct,
            "match_pct":    100.0 * correct / total,
            "top_confusion": top_conf,
            "top_confusion_pct": (100.0 * top_conf_n / total) if total else 0.0,
        })
    return pd.DataFrame(out).sort_values("ground_truth")


def transition_rate(df: pd.DataFrame) -> tuple[int, int, float]:
    """How often ground_truth[D] != ground_truth[D-1]."""
    s = df["ground_truth"].dropna().reset_index(drop=True)
    if len(s) < 2:
        return 0, 0, 0.0
    transitions = int((s.iloc[1:].values != s.iloc[:-1].values).sum())
    pairs = len(s) - 1
    return transitions, pairs, 100.0 * transitions / pairs


def regime_priors(df: pd.DataFrame, col: str) -> dict:
    counts = df[col].dropna().value_counts()
    total = counts.sum()
    return {k: 100.0 * v / total for k, v in counts.items()}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'=' * 72}")
    print(f"4H-SHIFTED MACRO LABEL VALIDATION")
    print(f"GBPUSD 5m, 2023-01-01 → 2025-12-31")
    print(f"{'=' * 72}\n")

    print("[1] Loading 5m bars …")
    df = load_5m_bars("2023-01-01", "2026-01-01")
    print(f"    bars: {len(df):,}  span: {df.index[0].date()} → {df.index[-1].date()}\n")

    print("[2] Computing labels per trading day (hour_anchor=8 UTC) …")
    labels = compute_labels(df, hour_anchor=8)
    print(f"    days: {len(labels):,}")
    print(f"    ground-truth coverage:   {labels['ground_truth'].notna().sum():,}")
    print(f"    prior-day coverage:      {labels['prior_day'].notna().sum():,}")
    print(f"    shifted (8 UTC) coverage:{labels['shifted'].notna().sum():,}\n")

    # ── Sanity check: prior_day on day D should equal ground_truth on day D-1.
    # Compute shift BEFORE dropna (post-dropna shift misaligns rows across gaps).
    labels = labels.reset_index(drop=True)
    gt_prev_row = labels["ground_truth"].shift(1)
    both = labels["prior_day"].notna() & gt_prev_row.notna()
    sanity_match = ((labels["prior_day"] == gt_prev_row) & both)
    total = int(both.sum())
    matches = int(sanity_match.sum())
    print(f"[sanity] prior_day vs yesterday's ground_truth: "
          f"{matches}/{total} ({100.0 * matches / total:.1f}%) — "
          f"100% expected (proves alignment).\n")

    # ── Headline match rates ──
    m_pd, n_pd, pct_pd = match_pct(labels["prior_day"],   labels["ground_truth"])
    m_sh, n_sh, pct_sh = match_pct(labels["shifted"],     labels["ground_truth"])
    print(f"[3] HEADLINE MATCH RATES (vs ground truth)")
    print(f"    prior-day  : {m_pd:>5d} / {n_pd:>5d}  =  {pct_pd:5.1f}%   ← live mode today")
    print(f"    shifted-8UTC: {m_sh:>5d} / {n_sh:>5d}  =  {pct_sh:5.1f}%   ← proposed 4h-shifted")
    print(f"    delta       : {pct_sh - pct_pd:+.1f} pp\n")

    # ── Daily transition rate ──
    t, p, pct_t = transition_rate(labels)
    print(f"[4] DAY-TO-DAY TRANSITION RATE (= ceiling for prior-day match if "
          f"yesterday's label is just yesterday's regime):")
    print(f"    transitions: {t:,} / {p:,} pairs  ({pct_t:.1f}% of days see a "
          f"regime change vs prior day)\n")
    print(f"    Theoretical max prior-day accuracy = {100.0 - pct_t:.1f}% "
          f"(prior-day label is automatically wrong on every transition day).\n")

    # ── Per-regime breakdown ──
    print("[5] PER-REGIME BREAKDOWN (rows = ground-truth regime)")
    print(f"\n    PRIOR-DAY label:")
    pd_bd = per_regime_breakdown(labels, "prior_day", "ground_truth")
    for _, r in pd_bd.iterrows():
        confuse = (f"  top confusion → {r['top_confusion']} "
                   f"({r['top_confusion_pct']:.1f}%)") if r["top_confusion"] else ""
        print(f"      {r['ground_truth']:<18s}  n={r['n_days']:>3d}  "
              f"match={r['match_pct']:5.1f}%{confuse}")
    print(f"\n    SHIFTED-8UTC label:")
    sh_bd = per_regime_breakdown(labels, "shifted", "ground_truth")
    for _, r in sh_bd.iterrows():
        confuse = (f"  top confusion → {r['top_confusion']} "
                   f"({r['top_confusion_pct']:.1f}%)") if r["top_confusion"] else ""
        print(f"      {r['ground_truth']:<18s}  n={r['n_days']:>3d}  "
              f"match={r['match_pct']:5.1f}%{confuse}")

    # ── Regime priors (for context) ──
    print("\n[6] REGIME PRIORS (ground-truth distribution):")
    priors = regime_priors(labels, "ground_truth")
    for k in ["strong_down", "staircase_down", "flat", "staircase_up", "strong_up"]:
        if k in priors:
            print(f"      {k:<18s} {priors[k]:5.1f}%")

    # ── Time-of-day curve ──
    print("\n[7] TIME-OF-DAY ACCURACY (shifted 24h window ending at hour X UTC):")
    print(f"      hour  | match% | n_days  | delta vs prior-day")
    print(f"      ------+--------+---------+------------------")
    for h in (0, 4, 8, 12, 16, 20):
        lbl_h = compute_labels(df, hour_anchor=h)
        m, n, pct = match_pct(lbl_h["shifted"], lbl_h["ground_truth"])
        m_pd2, n_pd2, pct_pd2 = match_pct(lbl_h["prior_day"], lbl_h["ground_truth"])
        delta = pct - pct_pd2
        print(f"      {h:>4d}  |  {pct:5.1f}% |  {n:>5,d}  |  {delta:+5.1f} pp")

    # ── Confusion: when 4h misses, what does it say instead? ──
    miss = labels.dropna(subset=["ground_truth", "shifted"])
    miss = miss[miss["shifted"] != miss["ground_truth"]]
    if len(miss):
        print(f"\n[8] SHIFTED-8UTC MISCLASSIFICATION PATTERNS ({len(miss)} miss days):")
        # Pairs of (gt, predicted)
        confusion = miss.groupby(["ground_truth", "shifted"]).size().sort_values(ascending=False)
        print(f"      ground_truth      → shifted_predicted      count")
        print(f"      ------------------+------------------------+-----")
        for (gt, sh), cnt in confusion.head(12).items():
            print(f"      {gt:<18s}→ {sh:<22s}  {cnt:>4d}")

    print(f"\n{'=' * 72}\n")
    return labels


if __name__ == "__main__":
    main()
