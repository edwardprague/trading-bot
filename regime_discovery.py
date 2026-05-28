#!/usr/bin/env python3
"""
regime_discovery.py — Regime Discovery Pipeline
================================================
Mines the local GBPUSD 5-minute price cache for Williams N=2 fractals,
builds a feature row per fractal (ADX, ATR pips, prior-fractal geometry,
pullback %, time-of-day, recent kind sequence), and clusters them with
KMeans to surface market regimes. Each cluster is validated against the
active strategy's (v2) actual trade outcomes — win rate per regime tells
us whether the clusters are predictive.

Stages
------
  1. Fractal extraction (with v2 backtest outcome labels)
  2. Feature engineering (scale + sin/cos hour + one-hot sequence)
  3. KMeans sweep k=3..8, silhouette-selected, model saved
  4. Cluster-level win-rate validation
  5. HTML report → results/regime_discovery.html (auto-opens)

Usage
-----
    source venv/bin/activate
    python3 regime_discovery.py

Requires
--------
    scikit-learn, joblib, matplotlib (already in venv: numpy, pandas, dotenv)
"""

import os
import sys
import base64
import io
import webbrowser
from pathlib import Path
from datetime import datetime

# ── Force the active strategy version BEFORE importing strategy_v2 ───────────
# This way the module-level globals (TICKER, INTERVAL, RRR, filters, etc.)
# resolve to the v2 short-only defaults the user is currently testing against.
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import joblib
except ImportError as e:
    print(f"ERROR: Missing dependency — {e.name}")
    print("Install with:  pip install scikit-learn joblib")
    sys.exit(1)

# Import the active strategy module to reuse fetch_data, add_indicators,
# run_backtest. strategy_v2.py is script-style but guarded by
# `if __name__ == "__main__":`, so the import has no side effects.
import strategy_v2 as strat


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

START_DATE = "2026-01-01"
END_DATE   = "2026-03-31"

ROOT_DIR    = Path(__file__).resolve().parent
# $DATA_DIR override (see server.py for the convention) so this CLI
# reads/writes the regime_model.pkl on the same volume the dashboard uses.
DATA_DIR    = Path(os.environ.get("DATA_DIR") or (ROOT_DIR / "data"))
RESULTS_DIR = ROOT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "regime_discovery.html"
MODEL_PATH  = DATA_DIR / "regime_model.pkl"

PIP = 10000  # non-JPY pairs

# Dark theme palette — matches strategy_v2 chart colors / style.css
BG_DARK    = "#1a1a2e"
PANEL_BG   = "#10101c"
TEXT       = "#d0d0e8"
TEXT_DIM   = "#9090b0"
GRID       = "#444"
ACCENT     = "#4cc9f0"
ACCENT_2   = "#7c5cbf"
GREEN      = "#6bcb77"
RED        = "#ef5350"
YELLOW     = "#ffd93d"

# Features used for clustering (excluding hour and sequence — handled separately)
NUMERIC_FEATURES = ["adx", "atr_pips", "v_dist_pips", "h_dist_bars", "pullback_pct"]
DISPLAY_NAMES = {
    "adx":          "ADX",
    "atr_pips":     "ATR (pips)",
    "v_dist_pips":  "Δ price vs prior same-kind (pips)",
    "h_dist_bars":  "Δ bars vs prior same-kind",
    "pullback_pct": "Pullback %",
    "entry_hour":   "Entry hour (UTC)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Fractal extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage1_extract_fractals():
    print("Stage 1: Extracting fractals from price data...")

    # Load 5m bars from cache via strategy_v2.fetch_data (transparently reads
    # the Parquet under data/), with the standard 30-day warmup buffer.
    df = strat.fetch_data(
        strat.TICKER, strat.INTERVAL, strat.DAYS_BACK,
        start_date=START_DATE, end_date=END_DATE,
    )
    df = strat.add_indicators(df)

    # Window: requested range ±1d pre, ±7d post — same buffers v2 uses so
    # trade resolution near the edges is realistic.
    start_ts = pd.Timestamp(START_DATE, tz="UTC")
    end_ts   = pd.Timestamp(END_DATE,   tz="UTC") + pd.Timedelta(days=1)
    dts      = pd.to_datetime(df["Datetime"])
    dts_utc  = dts.dt.tz_convert("UTC") if dts.dt.tz is not None else dts.dt.tz_localize("UTC")
    bt_start = start_ts - pd.Timedelta(days=1)
    bt_end   = end_ts   + pd.Timedelta(days=7)
    df = df[(dts_utc >= bt_start) & (dts_utc < bt_end)].reset_index(drop=True)

    # Run the active v2 backtest to harvest trade outcomes. Each trade row
    # carries fractal_bar — the bar index of the fractal that fired it.
    trades, _, _ = strat.run_backtest(df)

    if not trades.empty:
        t_entry = pd.to_datetime(trades["entry_ts"])
        t_utc   = t_entry.dt.tz_convert("UTC") if t_entry.dt.tz is not None else t_entry.dt.tz_localize("UTC")
        trades  = trades[(t_utc >= start_ts) & (t_utc < end_ts)].copy()

    # Map fractal_bar -> win/loss for fast lookup during the scan
    trade_outcome = {}
    if not trades.empty:
        for _, t in trades.iterrows():
            fb = int(t["fractal_bar"])
            trade_outcome[fb] = "W" if bool(t["win"]) else "L"

    # Scan every bar for Williams N=2 fractals using the identical predicate
    # from strategy_v2.run_backtest (lines 617-621):
    #   pivot high:  fh > highs[fi-1] and fh > highs[fi-2]
    #                and fh > highs[fi+1] and fh > highs[fi+2]
    #   pivot low:   fl < lows[fi-1]  and fl < lows[fi-2]
    #                and fl < lows[fi+1]  and fl < lows[fi+2]
    highs    = df["High"].values
    lows     = df["Low"].values
    atr_vals = df["atr14"].values
    adx_vals = df["adx"].values

    rows         = []
    history_kind = []          # rolling list of recent fractal kinds
    last_H_bar   = None
    last_H_price = None
    last_L_bar   = None
    last_L_price = None

    for fi in range(2, len(df) - 2):
        fh = highs[fi]
        fl = lows[fi]
        is_ph = (fh > highs[fi-1] and fh > highs[fi-2]
                 and fh > highs[fi+1] and fh > highs[fi+2])
        is_pl = (fl < lows[fi-1]  and fl < lows[fi-2]
                 and fl < lows[fi+1]  and fl < lows[fi+2])
        if not (is_ph or is_pl):
            continue

        # If a single bar is both (rare on 5m), record both rows in order.
        events = []
        if is_ph:
            events.append(("H", float(fh), last_H_bar, last_H_price))
        if is_pl:
            events.append(("L", float(fl), last_L_bar, last_L_price))

        ts_raw = pd.to_datetime(df["Datetime"].iloc[fi])
        ts_utc = ts_raw.tz_convert("UTC") if ts_raw.tzinfo else ts_raw.tz_localize("UTC")

        # Entry hour = hour of the confirmation bar (fi + 2) — this is the
        # candle the v2 engine would actually enter on.
        if fi + 2 < len(df):
            entry_ts_raw = pd.to_datetime(df["Datetime"].iloc[fi + 2])
            entry_ts_utc = entry_ts_raw.tz_convert("UTC") if entry_ts_raw.tzinfo else entry_ts_raw.tz_localize("UTC")
            entry_hour = int(entry_ts_utc.hour)
        else:
            entry_hour = int(ts_utc.hour)

        # Outcome: v2 entry fires at the confirmation bar. A trade with
        # fractal_bar == fi means that fractal produced a trade. We also
        # accept ±10 bars of slack in case any timing edge case occurs.
        outcome = None
        for offset in range(-1, 11):
            if (fi + offset) in trade_outcome:
                outcome = trade_outcome[fi + offset]
                break

        for kind, price, prev_bar, prev_price in events:
            # Distance to prior same-kind fractal
            if prev_bar is not None:
                v_dist_pips = abs(price - prev_price) * PIP
                h_dist_bars = float(fi - prev_bar)
            else:
                v_dist_pips = np.nan
                h_dist_bars = np.nan

            # Pullback % — mirrors strategy_v1 pullback formula (lines 1546-1577):
            #   H: (this_H − prev_L) / (prev_H − prev_L) × 100
            #   L: (prev_H − this_L) / (prev_H − prev_L) × 100
            pullback_pct = np.nan
            if last_H_price is not None and last_L_price is not None:
                rng = last_H_price - last_L_price
                if rng > 0:
                    if kind == "H":
                        pullback_pct = (price - last_L_price) / rng * 100
                    else:
                        pullback_pct = (last_H_price - price) / rng * 100
                    if pullback_pct < 0:
                        pullback_pct = np.nan

            # Sequence: last 4 kinds including this one
            seq_window = (history_kind + [kind])[-4:]
            sequence = "".join(seq_window) if len(seq_window) == 4 else None

            rows.append({
                "timestamp":    ts_utc,
                "fractal_bar":  fi,
                "kind":         kind,
                "price":        price,
                "adx":          float(adx_vals[fi]),
                "atr_pips":     float(atr_vals[fi]) * PIP,
                "v_dist_pips":  v_dist_pips,
                "h_dist_bars":  h_dist_bars,
                "pullback_pct": pullback_pct,
                "entry_hour":   entry_hour,
                "sequence":     sequence,
                "outcome":      outcome,
            })

            # Update rolling state AFTER recording so prior-same-kind refers
            # to the previous same-kind fractal, not this one.
            history_kind.append(kind)
            if len(history_kind) > 12:
                history_kind = history_kind[-12:]
            if kind == "H":
                last_H_bar, last_H_price = fi, price
            else:
                last_L_bar, last_L_price = fi, price

    fractal_df = pd.DataFrame(rows)

    # Restrict to fractals whose timestamp is inside the requested range.
    # We kept the buffer purely so prior-same-kind context is valid near the
    # left edge.
    fractal_df = fractal_df[
        (fractal_df["timestamp"] >= start_ts)
        & (fractal_df["timestamp"] < end_ts)
    ].reset_index(drop=True)

    print(f"Stage 1 complete: {len(fractal_df)} fractals detected")
    return fractal_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def stage2_prepare_features(fractal_df):
    print("Stage 2: Preparing features...")

    # Drop rows missing any clustering input (first H/L have no prior-same-kind;
    # first 3 fractals have no full sequence; pullback needs both H and L seen).
    needed = NUMERIC_FEATURES + ["entry_hour", "sequence"]
    clean = fractal_df.dropna(subset=needed).reset_index(drop=True)

    # Standard-scale the numeric features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clean[NUMERIC_FEATURES].values)
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_z" for c in NUMERIC_FEATURES])

    # Cyclic hour encoding
    hour = clean["entry_hour"].astype(float).values
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # One-hot encode the 4-kind sequence (e.g. HHLL, HLHL, ...)
    seq_dummies = pd.get_dummies(clean["sequence"], prefix="seq").astype(float)

    feature_matrix = pd.concat(
        [
            scaled_df,
            pd.DataFrame({"hour_sin": hour_sin, "hour_cos": hour_cos}),
            seq_dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    print("Stage 2 complete")
    return clean, feature_matrix, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Clustering
# ─────────────────────────────────────────────────────────────────────────────

def stage3_cluster(feature_matrix):
    print("Stage 3: Running clustering algorithm...")

    X = feature_matrix.values
    k_range = list(range(3, 9))   # 3..8 inclusive
    diagnostics = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else float("nan")
        diagnostics.append({"k": k, "inertia": inertia, "silhouette": sil})

    diag_df = pd.DataFrame(diagnostics)

    # Pick the k that maximises silhouette (primary selector per spec).
    best_k = int(diag_df.loc[diag_df["silhouette"].idxmax(), "k"])
    final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    final_labels = final.fit_predict(X)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    print(f"Stage 3 complete: {best_k} clusters identified")
    return diag_df, best_k, final_labels, final


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Validation
# ─────────────────────────────────────────────────────────────────────────────

def stage4_validate(clean_df, labels):
    print("Stage 4: Validating clusters against trade outcomes...")

    df = clean_df.copy()
    df["cluster"] = labels

    labelled = df[df["outcome"].isin(["W", "L"])].copy()
    if labelled.empty:
        validation = pd.DataFrame(columns=["cluster", "trades", "wins", "win_rate"])
    else:
        grouped = labelled.groupby("cluster").agg(
            trades=("outcome", "count"),
            wins=("outcome", lambda s: int((s == "W").sum())),
        )
        grouped["win_rate"] = grouped["wins"] / grouped["trades"] * 100.0
        validation = grouped.reset_index()

    print("Stage 4 complete")
    return df, validation


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers — render to base64 with the dark palette
# ─────────────────────────────────────────────────────────────────────────────

def _style_axes(ax):
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
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


def chart_elbow(diag_df, best_k):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    fig.patch.set_facecolor(BG_DARK)
    ax.plot(diag_df["k"], diag_df["inertia"], color=ACCENT, marker="o", linewidth=2)
    ax.axvline(best_k, color=YELLOW, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Elbow curve — inertia vs k")
    ax.set_xlabel("k clusters")
    ax.set_ylabel("Inertia")
    _style_axes(ax)
    return _fig_to_b64(fig)


def chart_silhouette(diag_df, best_k):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    fig.patch.set_facecolor(BG_DARK)
    colors = [GREEN if k == best_k else ACCENT_2 for k in diag_df["k"]]
    ax.bar(diag_df["k"], diag_df["silhouette"], color=colors)
    ax.set_title("Silhouette score vs k  (best highlighted)")
    ax.set_xlabel("k clusters")
    ax.set_ylabel("Silhouette")
    _style_axes(ax)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap bucket helper (warm = high, cool = low)
# ─────────────────────────────────────────────────────────────────────────────

def heat_class(z):
    """Map a z-score to a heatmap CSS class defined in style.css."""
    if pd.isna(z):
        return "heat-zero"
    if z >= 1.5:    return "heat-pos-5"
    if z >= 1.0:    return "heat-pos-4"
    if z >= 0.5:    return "heat-pos-3"
    if z >= 0.2:    return "heat-pos-2"
    if z >  -0.2:   return "heat-zero"
    if z >  -0.5:   return "heat-neg-2"
    if z >  -1.0:   return "heat-neg-3"
    if z >  -1.5:   return "heat-neg-4"
    return "heat-neg-5"


def winrate_class(rate, trades):
    if trades == 0:
        return "win-neutral"
    if rate >= 55: return "win-good"
    if rate <= 45: return "win-bad"
    return "win-neutral"


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(fractal_count, clusters_df, validation_df,
                 diag_df, best_k, elbow_b64, silhouette_b64):
    """Render the regime_discovery.html report and write to disk."""

    # ── Cluster summary table (mean per feature, z-coloured) ────────────────
    cluster_means = clusters_df.groupby("cluster")[NUMERIC_FEATURES + ["entry_hour"]].mean()
    cluster_means["count"] = clusters_df.groupby("cluster").size()
    overall_mean = clusters_df[NUMERIC_FEATURES + ["entry_hour"]].mean()
    overall_std  = clusters_df[NUMERIC_FEATURES + ["entry_hour"]].std(ddof=0).replace(0, np.nan)

    # Header
    range_str = f"{START_DATE} → {END_DATE}"

    # Build cluster summary HTML
    feat_cols = NUMERIC_FEATURES + ["entry_hour"]
    header_cells = "".join(
        f"<th>{DISPLAY_NAMES.get(f, f)}</th>" for f in feat_cols
    )
    summary_rows = []
    for c in sorted(cluster_means.index):
        cells = [f"<td><strong>Cluster {c}</strong></td>",
                 f"<td>{int(cluster_means.loc[c, 'count'])}</td>"]
        for f in feat_cols:
            v = cluster_means.loc[c, f]
            z = (v - overall_mean[f]) / overall_std[f] if not pd.isna(overall_std[f]) else 0
            cls = heat_class(z)
            cells.append(f"<td class='{cls}'>{v:.2f}</td>")
        summary_rows.append("<tr>" + "".join(cells) + "</tr>")

    summary_table = f"""
      <table class="regime-table">
        <thead>
          <tr>
            <th>Cluster</th>
            <th>N</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    """

    # Win-rate validation table
    val_rows = []
    if not validation_df.empty:
        for _, r in validation_df.sort_values("cluster").iterrows():
            cls = winrate_class(r["win_rate"], r["trades"])
            val_rows.append(
                f"<tr><td><strong>Cluster {int(r['cluster'])}</strong></td>"
                f"<td>{int(r['trades'])}</td>"
                f"<td>{int(r['wins'])}</td>"
                f"<td class='{cls}'>{r['win_rate']:.1f}%</td></tr>"
            )
    else:
        val_rows.append(
            "<tr><td colspan='4' class='win-neutral'>"
            "No trades fired against any fractal in the requested window — "
            "stage-4 validation skipped."
            "</td></tr>"
        )
    validation_table = f"""
      <table class="regime-table">
        <thead>
          <tr><th>Cluster</th><th>Trades</th><th>Wins</th><th>Win rate</th></tr>
        </thead>
        <tbody>{''.join(val_rows)}</tbody>
      </table>
    """

    # Profile cards — top 3 features by absolute z-score deviation per cluster
    profile_cards = []
    for c in sorted(cluster_means.index):
        z_devs = []
        for f in feat_cols:
            if pd.isna(overall_std[f]) or overall_std[f] == 0:
                continue
            z = (cluster_means.loc[c, f] - overall_mean[f]) / overall_std[f]
            z_devs.append((f, z, cluster_means.loc[c, f]))
        top3 = sorted(z_devs, key=lambda t: abs(t[1]), reverse=True)[:3]

        feat_lines = []
        for f, z, v in top3:
            arrow = "▲" if z > 0 else "▼"
            sign_cls = "heat-pos-3" if z > 0 else "heat-neg-3"
            feat_lines.append(
                f"<li><span class='{sign_cls} regime-pill'>{arrow}</span> "
                f"<strong>{DISPLAY_NAMES.get(f, f)}</strong>: "
                f"{v:.2f} <span class='regime-dim'>(z = {z:+.2f})</span></li>"
            )

        profile_cards.append(f"""
          <div class="regime-card regime-profile">
            <h3>Cluster {c}</h3>
            <div class="regime-dim regime-small">Fractals: {int(cluster_means.loc[c, 'count'])}</div>
            <ul class="regime-feature-list">
              {''.join(feat_lines)}
            </ul>
          </div>
        """)

    # Diagnostics line under chart row
    diag_lines = []
    for _, r in diag_df.iterrows():
        marker = " ◀ best" if int(r["k"]) == best_k else ""
        diag_lines.append(
            f"k={int(r['k'])}  silhouette={r['silhouette']:.3f}  "
            f"inertia={r['inertia']:.1f}{marker}"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Regime Discovery — GBPUSD 5m</title>
  <link rel="stylesheet" href="../style.css">
</head>
<body class="regime-report">
  <div class="regime-container">

    <header class="regime-header">
      <h1>Regime Discovery — GBPUSD 5m</h1>
      <div class="regime-header-meta">
        <span><strong>Range:</strong> {range_str}</span>
        <span><strong>Fractals:</strong> {fractal_count}</span>
        <span><strong>Clusters:</strong> {best_k}</span>
        <span class="regime-dim">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
      </div>
    </header>

    <section class="regime-card">
      <h2>Cluster selection diagnostics</h2>
      <div class="regime-chart-row">
        <img alt="Elbow curve" src="data:image/png;base64,{elbow_b64}" />
        <img alt="Silhouette" src="data:image/png;base64,{silhouette_b64}" />
      </div>
      <pre class="regime-diag">{chr(10).join(diag_lines)}</pre>
    </section>

    <section class="regime-card">
      <h2>Cluster feature summary</h2>
      <p class="regime-dim regime-small">
        Each cell shows the cluster's mean for that feature. Warm = above overall
        average, cool = below — z-scored against the population.
      </p>
      {summary_table}
    </section>

    <section class="regime-card">
      <h2>Win-rate validation (v2 short-only)</h2>
      <p class="regime-dim regime-small">
        Win rate per cluster computed only on fractals that actually produced a
        trade under the active strategy filters. Green ≥ 55%, red ≤ 45%.
      </p>
      {validation_table}
    </section>

    <section>
      <h2>Cluster profiles</h2>
      <div class="regime-profile-grid">
        {''.join(profile_cards)}
      </div>
    </section>

  </div>
</body>
</html>
"""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    return REPORT_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fractal_df = stage1_extract_fractals()
    if fractal_df.empty:
        print("No fractals detected in the requested range — aborting.")
        return

    clean_df, feature_matrix, _scaler = stage2_prepare_features(fractal_df)
    if len(clean_df) < 30:
        print(f"Too few fractals for clustering ({len(clean_df)}) — aborting.")
        return

    diag_df, best_k, labels, _model = stage3_cluster(feature_matrix)
    clusters_df, validation_df = stage4_validate(clean_df, labels)

    elbow_b64      = chart_elbow(diag_df, best_k)
    silhouette_b64 = chart_silhouette(diag_df, best_k)

    report_path = build_report(
        fractal_count=len(clean_df),
        clusters_df=clusters_df,
        validation_df=validation_df,
        diag_df=diag_df,
        best_k=best_k,
        elbow_b64=elbow_b64,
        silhouette_b64=silhouette_b64,
    )

    webbrowser.open(f"file://{report_path}")
    print(f"Report saved and opened: {report_path}")


if __name__ == "__main__":
    main()
