"""
Trading Bot — EMA Trend Following Strategy
==========================================
Instrument : EURUSD hourly (Yahoo Finance)
Strategy   : EMA 20/50/200 crossover with swing stop
RRR        : 1:2
Risk/Trade : 1% of equity

Usage:
    source venv/bin/activate
    python3 strategy.py [optional: "notes about this run"]
"""

import sys
import re
import json
import base64
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves chart to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

TICKER          = "EURUSD=X"
INTERVAL        = "1h"
DAYS_BACK       = 720
STARTING_CASH   = 10_000.0

EMA_SLOW        = 200
EMA_FAST        = 50
EMA_ENTRY       = 20
SWING_LOOKBACK  = 20
RRR             = 2.0
RISK_PCT        = 0.01
MIN_STOP        = 0.0005     # 5 pips minimum stop
MAX_STOP        = 0.0200     # 200 pips maximum stop

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(ticker, interval, days_back):
    try:
        import yfinance as yf
        end   = datetime.now()
        start = end - timedelta(days=days_back)
        print(f"\nFetching {ticker} {interval} data ({days_back} days)...")
        df = yf.download(ticker, start=start, end=end,
                         interval=interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty dataframe returned")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        print(f"  {len(df)} bars loaded | {df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"  Data fetch failed: {e}")
        sys.exit(1)

# ── Indicators ────────────────────────────────────────────────────────────────

def add_indicators(df):
    df = df.copy()
    df["ema_slow"]  = df.Close.ewm(span=EMA_SLOW,  adjust=False).mean()
    df["ema_fast"]  = df.Close.ewm(span=EMA_FAST,  adjust=False).mean()
    df["ema_entry"] = df.Close.ewm(span=EMA_ENTRY, adjust=False).mean()
    df["s_low"]     = df.Low.shift(1).rolling(SWING_LOOKBACK).min()
    df["s_high"]    = df.High.shift(1).rolling(SWING_LOOKBACK).max()
    return df.dropna().reset_index()

# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df):
    cash      = STARTING_CASH
    equity    = [cash]
    trades    = []
    in_trade  = False
    entry_p   = sl = tp = size = 0
    direction = None
    entry_idx = 0

    for i in range(1, len(df)):
        c     = float(df.Close.iloc[i])
        cp    = float(df.Close.iloc[i-1])
        en    = float(df.ema_entry.iloc[i])
        enp   = float(df.ema_entry.iloc[i-1])
        fast  = float(df.ema_fast.iloc[i])
        slow  = float(df.ema_slow.iloc[i])
        s_lo  = float(df.s_low.iloc[i])
        s_hi  = float(df.s_high.iloc[i])
        ts    = df.index[i] if not hasattr(df, 'Datetime') else df.Datetime.iloc[i]

        # ── Check exits ───────────────────────────────────────────────────────
        if in_trade:
            hit_sl = (direction == "long"  and c <= sl) or \
                     (direction == "short" and c >= sl)
            hit_tp = (direction == "long"  and c >= tp) or \
                     (direction == "short" and c <= tp)

            if hit_sl or hit_tp:
                exit_p  = sl if hit_sl else tp
                pnl     = (exit_p - entry_p) * size if direction == "long" \
                          else (entry_p - exit_p) * size
                cash   += pnl
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx":  i,
                    "direction": direction,
                    "entry":     entry_p,
                    "exit":      exit_p,
                    "stop":      sl,
                    "target":    tp,
                    "pnl":       pnl,
                    "win":       pnl > 0,
                    "result":    "TP" if hit_tp else "SL",
                    "timestamp": ts
                })
                in_trade = False

        equity.append(cash)

        # ── Check entries ─────────────────────────────────────────────────────
        if not in_trade:
            trend_up   = fast > slow
            trend_down = fast < slow
            long_sig   = trend_up   and cp < enp and c > en
            short_sig  = trend_down and cp > enp and c < en

            if long_sig and not np.isnan(s_lo):
                dist = c - s_lo
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "long"
                    entry_p   = c
                    sl        = s_lo
                    tp        = c + dist * RRR
                    size      = (cash * RISK_PCT) / dist
                    in_trade  = True
                    entry_idx = i

            elif short_sig and not np.isnan(s_hi):
                dist = s_hi - c
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "short"
                    entry_p   = c
                    sl        = s_hi
                    tp        = c - dist * RRR
                    size      = (cash * RISK_PCT) / dist
                    in_trade  = True
                    entry_idx = i

    return pd.DataFrame(trades), equity

# ── Results ───────────────────────────────────────────────────────────────────

def print_results(trades, equity):
    if trades.empty:
        print("\n  No trades generated.")
        return

    wins  = trades[trades.win]
    loss  = trades[~trades.win]
    net   = trades.pnl.sum()
    pf    = abs(wins.pnl.sum() / loss.pnl.sum()) if not loss.empty else float("inf")

    eq    = pd.Series(equity)
    peak  = eq.cummax()
    dd    = ((eq - peak) / peak * 100).min()

    # Sharpe (annualised, assuming hourly bars)
    returns = eq.pct_change().dropna()
    sharpe  = (returns.mean() / returns.std() * np.sqrt(24 * 252)
               if returns.std() > 0 else 0)

    print(f"\n{'═'*52}")
    print(f"  BACKTEST RESULTS")
    print(f"{'═'*52}")
    print(f"  Instrument     : {TICKER} {INTERVAL}")
    print(f"  Strategy       : EMA {EMA_ENTRY}/{EMA_FAST}/{EMA_SLOW} | RRR 1:{RRR}")
    print(f"{'─'*52}")
    print(f"  Total Trades   : {len(trades)}")
    print(f"  Winning Trades : {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
    print(f"  Losing Trades  : {len(loss)}")
    print(f"  Win Rate       : {len(wins)/len(trades)*100:.1f}%")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"{'─'*52}")
    print(f"  Avg Win        : ${wins.pnl.mean():.2f}" if not wins.empty else "  Avg Win        : N/A")
    print(f"  Avg Loss       : ${loss.pnl.mean():.2f}" if not loss.empty else "  Avg Loss       : N/A")
    print(f"  Best Trade     : ${trades.pnl.max():.2f}")
    print(f"  Worst Trade    : ${trades.pnl.min():.2f}")
    print(f"{'─'*52}")
    print(f"  Net Profit     : ${net:+.2f} ({net/STARTING_CASH*100:+.1f}%)")
    print(f"  Final Equity   : ${STARTING_CASH + net:.2f}")
    print(f"  Max Drawdown   : {dd:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"{'═'*52}")

    print(f"\n  Last 10 trades:")
    print(f"  {'Dir':<6} {'Entry':>8} {'Exit':>8} {'P&L':>10}  Result")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*10}  {'─'*6}")
    for _, t in trades.tail(10).iterrows():
        print(f"  {t.direction.capitalize():<6} "
              f"{t.entry:>8.5f} {t.exit:>8.5f} "
              f"{t.pnl:>+10.2f}  {'WIN' if t.win else 'LOSS'}")
    print(f"{'═'*52}\n")

# ── Charts ────────────────────────────────────────────────────────────────────

def save_charts(df, trades, equity):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["top"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["right"].set_color("#444")

    dates = pd.to_datetime(df.index if "Datetime" not in df.columns
                           else df.Datetime)

    # ── Price chart with EMAs ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, df.Close,     color="#e0e0e0", linewidth=0.8, label="Price")
    ax1.plot(dates, df.ema_slow,  color="#ff6b6b", linewidth=1.2, label=f"EMA {EMA_SLOW}")
    ax1.plot(dates, df.ema_fast,  color="#ffd93d", linewidth=1.0, label=f"EMA {EMA_FAST}")
    ax1.plot(dates, df.ema_entry, color="#6bcb77", linewidth=0.8, label=f"EMA {EMA_ENTRY}")

    # Mark trades
    if not trades.empty:
        for _, t in trades.iterrows():
            idx  = int(t.entry_idx)
            eidx = int(t.exit_idx)
            if idx < len(dates) and eidx < len(dates):
                color  = "#6bcb77" if t.win else "#ff6b6b"
                marker = "^" if t.direction == "long" else "v"
                ax1.scatter(dates.iloc[idx],  t.entry, color=color,
                           marker=marker, s=60, zorder=5)
                ax1.scatter(dates.iloc[eidx], t.exit,  color=color,
                           marker="x", s=40, zorder=5)

    ax1.set_title(f"{TICKER} — EMA Trend Following Backtest",
                  color="white", fontsize=13, pad=10)
    ax1.legend(loc="upper left", facecolor="#1a1a2e",
               labelcolor="white", fontsize=8)
    ax1.set_ylabel("Price", color="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Equity curve ──────────────────────────────────────────────────────────
    ax2 = axes[1]
    eq_dates = dates.iloc[:len(equity)] if len(equity) <= len(dates) else dates
    eq_series = pd.Series(equity[:len(eq_dates)])
    ax2.plot(eq_dates[:len(eq_series)], eq_series,
             color="#4cc9f0", linewidth=1.2)
    ax2.axhline(STARTING_CASH, color="#666", linestyle="--", linewidth=0.8)
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series >= STARTING_CASH,
                     alpha=0.2, color="#6bcb77")
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series < STARTING_CASH,
                     alpha=0.2, color="#ff6b6b")
    ax2.set_ylabel("Equity ($)", color="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax3 = axes[2]
    eq_s  = pd.Series(equity)
    peak  = eq_s.cummax()
    dd    = (eq_s - peak) / peak * 100
    ax3.fill_between(range(len(dd)), dd, 0, color="#ff6b6b", alpha=0.6)
    ax3.set_ylabel("Drawdown %", color="white")
    ax3.set_xlabel("Bar", color="white")

    plt.tight_layout(pad=2.0)
    chart_path = "backtest_chart.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  Chart saved → {chart_path}")
    print(f"  Open with:    open {chart_path}\n")

# ── HTML Report ───────────────────────────────────────────────────────────────

def compute_metrics(trades, equity):
    """Compute all backtest metrics and return as a JSON-serialisable dict."""
    if trades.empty:
        return None

    wins = trades[trades.win]
    loss = trades[~trades.win]
    net  = float(trades.pnl.sum())

    if not loss.empty and loss.pnl.sum() != 0:
        pf = round(abs(float(wins.pnl.sum()) / abs(float(loss.pnl.sum()))), 2)
    else:
        pf = None   # infinite — stored as null, displayed as ∞

    eq      = pd.Series(equity)
    peak    = eq.cummax()
    dd      = float(((eq - peak) / peak * 100).min())
    returns = eq.pct_change().dropna()
    sharpe  = float(returns.mean() / returns.std() * np.sqrt(24 * 252)
                    if returns.std() > 0 else 0)

    n = len(trades)
    return {
        "total_trades":   n,
        "winning_trades": int(len(wins)),
        "losing_trades":  int(len(loss)),
        "win_rate":       round(len(wins) / n * 100, 1),
        "profit_factor":  pf,
        "avg_win":        round(float(wins.pnl.mean()), 2) if not wins.empty else None,
        "avg_loss":       round(float(loss.pnl.mean()), 2) if not loss.empty else None,
        "best_trade":     round(float(trades.pnl.max()), 2),
        "worst_trade":    round(float(trades.pnl.min()), 2),
        "net_profit":     round(net, 2),
        "net_profit_pct": round(net / STARTING_CASH * 100, 2),
        "final_equity":   round(STARTING_CASH + net, 2),
        "max_drawdown":   round(dd, 2),
        "sharpe":         round(sharpe, 2),
    }


def _build_html(versions_json):
    """Return the complete self-contained HTML string."""
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backtest Report</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      background: #0c0c14;
      color: #d0d0e8;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    #sidebar {
      width: 228px;
      min-width: 228px;
      background: #10101c;
      border-right: 1px solid #1e1e30;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    #sidebar-header {
      padding: 20px 16px 14px;
      border-bottom: 1px solid #1e1e30;
      flex-shrink: 0;
    }
    #sidebar-header h1 {
      font-size: 10px;
      font-weight: 700;
      color: #505080;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    #run-count {
      font-size: 11px;
      color: #404060;
      margin-top: 4px;
    }
    #version-list { overflow-y: auto; flex: 1; }
    .v-item {
      padding: 12px 16px;
      cursor: pointer;
      border-bottom: 1px solid #181826;
      border-left: 3px solid transparent;
      transition: background 0.1s;
    }
    .v-item:hover  { background: #161624; }
    .v-item.active { background: #182040; border-left-color: #4cc9f0; }
    .v-name { font-size: 14px; font-weight: 600; color: #c8c8e8; }
    .v-date { font-size: 11px; color: #505070; margin-top: 3px; }
    .v-pnl  { font-size: 12px; font-weight: 600; margin-top: 5px; }

    /* ── Main ─────────────────────────────────────────────── */
    #main { flex: 1; overflow-y: auto; padding: 30px 38px 52px; }

    /* ── Version header ───────────────────────────────────── */
    #v-header { margin-bottom: 28px; }
    #v-header h2 { font-size: 24px; font-weight: 700; color: #e4e4ff; }
    .v-meta { font-size: 12px; color: #505070; margin-top: 6px; line-height: 1.7; }
    .v-notes {
      margin-top: 14px;
      padding: 11px 16px;
      background: #13132a;
      border-radius: 6px;
      border-left: 3px solid #4cc9f0;
      font-size: 13px;
      color: #9090c0;
      line-height: 1.6;
    }

    /* ── Section ──────────────────────────────────────────── */
    .section { margin-bottom: 30px; }
    .section-title {
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: #484868;
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid #1a1a2c;
    }

    /* ── Chart ────────────────────────────────────────────── */
    #chart-img {
      width: 100%;
      border-radius: 8px;
      display: block;
      border: 1px solid #1a1a2c;
    }

    /* ── Tables ───────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    thead th {
      background: #131326;
      color: #505080;
      font-weight: 700;
      padding: 9px 14px;
      text-align: left;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    tbody td { padding: 9px 14px; border-bottom: 1px solid #161626; }
    tbody tr:last-child td { border-bottom: none; }
    tbody tr:hover td { background: #131322; }
    .lbl { color: #707090; }

    /* ── Colour helpers ───────────────────────────────────── */
    .pos { color: #6bcb77; }
    .neg { color: #ff6b6b; }
    .neu { color: #ffd93d; }

    /* ── Badge ────────────────────────────────────────────── */
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .badge-win  { background: rgba(107,203,119,.16); color: #6bcb77; }
    .badge-loss { background: rgba(255,107,107,.16); color: #ff6b6b; }

    /* ── Two-column grid ──────────────────────────────────── */
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin-bottom: 30px; }
    @media (max-width: 860px) { .two-col { grid-template-columns: 1fr; } #main { padding: 20px; } }

    /* ── Empty state ──────────────────────────────────────── */
    #empty-state {
      display: flex; flex-direction: column; align-items: center;
      justify-content: center; height: 80vh;
      color: #303050; font-size: 15px; gap: 10px;
    }

    /* ── Scrollbars ───────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #252540; border-radius: 3px; }
  </style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>Backtest History</h1>
    <div id="run-count"></div>
  </div>
  <div id="version-list"></div>
</div>

<div id="main">
  <div id="content">
    <div id="empty-state">
      <span style="font-size:36px">&#128202;</span>
      <span>Select a version to view results</span>
    </div>
  </div>
</div>

<script type="application/json" id="versions-data">
__VERSIONS_JSON__
</script>

<script>
(function () {
  "use strict";

  var VERSIONS = JSON.parse(document.getElementById("versions-data").textContent);

  /* ── Helpers ──────────────────────────────────────────────── */
  function fmt(n, d) {
    if (n === null || n === undefined || (typeof n === "number" && isNaN(n))) return "&#8212;";
    return Number(n).toFixed(d !== undefined ? d : 2);
  }

  function fmtMoney(n) {
    if (n === null || n === undefined) return "&#8212;";
    var s = n >= 0 ? "+" : "";
    return s + "$" + Math.abs(n).toFixed(2);
  }

  function pClass(n) {
    if (n === null || n === undefined) return "";
    return n >= 0 ? "pos" : "neg";
  }

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function row(label, value) {
    var v = (value === null || value === undefined) ? "&#8212;" : value;
    return "<tr><td class='lbl'>" + label + "</td><td>" + v + "</td></tr>";
  }

  /* ── Sidebar ──────────────────────────────────────────────── */
  function renderSidebar() {
    document.getElementById("run-count").textContent =
      VERSIONS.length + " run" + (VERSIONS.length !== 1 ? "s" : "");

    var list = document.getElementById("version-list");
    list.innerHTML = "";

    for (var ri = VERSIONS.length - 1; ri >= 0; ri--) {
      var v    = VERSIONS[ri];
      var pnl  = v.metrics ? v.metrics.net_profit : null;
      var pc   = pnl === null ? "" : (pnl >= 0 ? "pos" : "neg");
      var ptxt = pnl === null ? "" :
        (pnl >= 0 ? "+" : "") + "$" + Math.abs(pnl).toFixed(2);

      var item = document.createElement("div");
      item.className   = "v-item";
      item.dataset.idx = ri;
      item.innerHTML   =
        "<div class='v-name'>" + esc(v.name) + "</div>" +
        "<div class='v-date'>" + esc(v.date) + "</div>" +
        (pnl !== null ? "<div class='v-pnl " + pc + "'>" + ptxt + "</div>" : "");

      (function (el, idx) {
        el.addEventListener("click", function () {
          document.querySelectorAll(".v-item").forEach(function (e) {
            e.classList.remove("active");
          });
          el.classList.add("active");
          renderContent(idx);
        });
      })(item, ri);

      list.appendChild(item);
    }
  }

  /* ── Content ──────────────────────────────────────────────── */
  function renderContent(idx) {
    var v = VERSIONS[idx];
    if (!v) return;

    var m = v.metrics    || {};
    var p = v.params     || {};
    var t = v.last_trades || [];

    /* profit factor display */
    var pfTxt = (m.profit_factor === null || m.profit_factor === undefined)
      ? "&#8734;" : fmt(m.profit_factor);
    var pfCls = (m.profit_factor === null || m.profit_factor === undefined || m.profit_factor >= 1.5)
      ? "pos" : (m.profit_factor < 1.0 ? "neg" : "neu");

    /* last 10 trades rows */
    var tRows = "";
    t.forEach(function (tr) {
      var pn = parseFloat(tr.pnl);
      var tc = pn >= 0 ? "pos" : "neg";
      tRows +=
        "<tr>" +
        "<td>" + esc(tr.direction) + "</td>" +
        "<td style='font-variant-numeric:tabular-nums'>" + esc(tr.entry) + "</td>" +
        "<td style='font-variant-numeric:tabular-nums'>" + esc(tr.exit)  + "</td>" +
        "<td class='" + tc + "' style='font-variant-numeric:tabular-nums'>" + esc(tr.pnl) + "</td>" +
        "<td><span class='badge " + (tr.result === "WIN" ? "badge-win" : "badge-loss") + "'>" + tr.result + "</span></td>" +
        "</tr>";
    });
    if (!tRows) {
      tRows = "<tr><td colspan='5' style='color:#404060;text-align:center;padding:22px'>No trades recorded</td></tr>";
    }

    var chartHtml = v.chart_b64
      ? "<div class='section'><div class='section-title'>Chart</div>" +
        "<img id='chart-img' src='data:image/png;base64," + v.chart_b64 + "' alt='Backtest Chart'/></div>"
      : "";

    var notesHtml = (v.notes && v.notes !== "&#8212;" && v.notes !== "\u2014" && v.notes !== "—")
      ? "<div class='v-notes'>&#128221;&nbsp; " + esc(v.notes) + "</div>"
      : "";

    var winRateCls = m.win_rate >= 50 ? "pos" : "neg";
    var sharpeCls  = m.sharpe >= 1 ? "pos" : (m.sharpe < 0 ? "neg" : "neu");

    var avgWinHtml  = m.avg_win  !== null && m.avg_win  !== undefined
      ? "<span class='pos'>$" + fmt(m.avg_win)  + "</span>" : "&#8212;";
    var avgLossHtml = m.avg_loss !== null && m.avg_loss !== undefined
      ? "<span class='neg'>" + fmt(m.avg_loss) + "</span>" : "&#8212;";

    document.getElementById("content").innerHTML =
      /* header */
      "<div id='v-header'>" +
        "<h2>" + esc(v.name) + "</h2>" +
        "<div class='v-meta'>Run on " + esc(v.date) +
          " &nbsp;&middot;&nbsp; " + esc(p.ticker || "") +
          " " + esc(p.interval || "") +
          " &nbsp;&middot;&nbsp; " + (p.days_back || "") + " days</div>" +
        notesHtml +
      "</div>" +

      /* chart */
      chartHtml +

      /* results + params */
      "<div class='two-col'>" +

        "<div class='section'>" +
          "<div class='section-title'>Results</div>" +
          "<table><tbody>" +
          row("Total Trades",  m.total_trades) +
          row("Winning Trades", m.winning_trades) +
          row("Losing Trades",  m.losing_trades) +
          row("Win Rate",      "<span class='" + winRateCls + "'>" + fmt(m.win_rate, 1) + "%</span>") +
          row("Profit Factor", "<span class='" + pfCls + "'>" + pfTxt + "</span>") +
          row("Net Profit",    "<span class='" + pClass(m.net_profit) + "'>" +
            fmtMoney(m.net_profit) + " (" + (m.net_profit_pct >= 0 ? "+" : "") + fmt(m.net_profit_pct, 1) + "%)</span>") +
          row("Final Equity",  "$" + fmt(m.final_equity)) +
          row("Max Drawdown",  "<span class='neg'>" + fmt(m.max_drawdown) + "%</span>") +
          row("Sharpe Ratio",  "<span class='" + sharpeCls + "'>" + fmt(m.sharpe) + "</span>") +
          row("Avg Win",       avgWinHtml) +
          row("Avg Loss",      avgLossHtml) +
          row("Best Trade",    "<span class='pos'>" + fmtMoney(m.best_trade)  + "</span>") +
          row("Worst Trade",   "<span class='neg'>" + fmtMoney(m.worst_trade) + "</span>") +
          "</tbody></table>" +
        "</div>" +

        "<div class='section'>" +
          "<div class='section-title'>Parameters</div>" +
          "<table><tbody>" +
          row("Instrument",     esc(p.ticker || "")) +
          row("Interval",       esc(p.interval || "")) +
          row("History",        (p.days_back || "") + " days") +
          row("Starting Cash",  "$" + (p.starting_cash || 0).toLocaleString()) +
          row("EMA Slow",       p.ema_slow) +
          row("EMA Fast",       p.ema_fast) +
          row("EMA Entry",      p.ema_entry) +
          row("Swing Lookback", (p.swing_lookback || "") + " bars") +
          row("RRR",            "1&thinsp;:&thinsp;" + (p.rrr || "")) +
          row("Risk / Trade",   ((p.risk_pct || 0) * 100).toFixed(1) + "%") +
          row("Min Stop",       ((p.min_stop || 0) * 10000).toFixed(0) + " pips") +
          row("Max Stop",       ((p.max_stop || 0) * 10000).toFixed(0) + " pips") +
          "</tbody></table>" +
        "</div>" +

      "</div>" +

      /* last 10 trades */
      "<div class='section'>" +
        "<div class='section-title'>Last 10 Trades</div>" +
        "<table>" +
          "<thead><tr>" +
          "<th>Direction</th><th>Entry</th><th>Exit</th><th>P&amp;L</th><th>Result</th>" +
          "</tr></thead>" +
          "<tbody>" + tRows + "</tbody>" +
        "</table>" +
      "</div>";
  }

  /* ── Init ──────────────────────────────────────────────────── */
  renderSidebar();
  if (VERSIONS.length > 0) {
    var first = document.querySelector(".v-item");
    if (first) first.classList.add("active");
    renderContent(VERSIONS.length - 1);
  }
})();
</script>

</body>
</html>"""
    return template.replace("__VERSIONS_JSON__", versions_json)


def generate_html_report(trades, equity, chart_path="backtest_chart.png", notes=""):
    """Create or update report.html with the new backtest version appended."""
    report_path = "report.html"

    metrics = compute_metrics(trades, equity)
    if metrics is None:
        print("  No trades generated — skipping HTML report.")
        return

    # ── Load chart as base64 ───────────────────────────────────────────────────
    chart_b64 = ""
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            chart_b64 = base64.b64encode(fh.read()).decode("utf-8")

    # ── Build last-10-trades list ──────────────────────────────────────────────
    last_trades = []
    for _, t in trades.tail(10).iterrows():
        last_trades.append({
            "direction": t.direction.capitalize(),
            "entry":     f"{float(t.entry):.5f}",
            "exit":      f"{float(t.exit):.5f}",
            "pnl":       f"{float(t.pnl):+.2f}",
            "result":    "WIN" if t.win else "LOSS",
        })

    # ── Read existing versions from report.html if it exists ──────────────────
    existing_versions = []
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        match = re.search(
            r'<script[^>]+id=["\']versions-data["\'][^>]*>([\s\S]*?)</script>',
            content
        )
        if match:
            try:
                existing_versions = json.loads(match.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                existing_versions = []

    # ── Assemble new version entry ─────────────────────────────────────────────
    version_num = len(existing_versions) + 1
    new_version = {
        "name":        f"v{version_num}",
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notes":       notes.strip() if notes else "—",
        "chart_b64":   chart_b64,
        "metrics":     metrics,
        "params": {
            "ticker":         TICKER,
            "interval":       INTERVAL,
            "days_back":      DAYS_BACK,
            "starting_cash":  STARTING_CASH,
            "ema_slow":       EMA_SLOW,
            "ema_fast":       EMA_FAST,
            "ema_entry":      EMA_ENTRY,
            "swing_lookback": SWING_LOOKBACK,
            "rrr":            RRR,
            "risk_pct":       RISK_PCT,
            "min_stop":       MIN_STOP,
            "max_stop":       MAX_STOP,
        },
        "last_trades": last_trades,
    }

    existing_versions.append(new_version)

    # ── Write HTML ─────────────────────────────────────────────────────────────
    versions_json = json.dumps(existing_versions, indent=2, ensure_ascii=False)
    html = _build_html(versions_json)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    action = "Updated" if version_num > 1 else "Created"
    print(f"  Report {action} → {report_path}  ({version_num} version{'s' if version_num > 1 else ''})")
    print(f"  Open with:    open {report_path}\n")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    notes           = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    df              = fetch_data(TICKER, INTERVAL, DAYS_BACK)
    df              = add_indicators(df)
    trades, equity  = run_backtest(df)
    print_results(trades, equity)
    save_charts(df, trades, equity)
    generate_html_report(trades, equity, notes=notes)
