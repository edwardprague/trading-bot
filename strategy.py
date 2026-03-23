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
import subprocess
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

TRADE_DIRECTION = "short_only"   # "both" | "long_only" | "short_only"

VERSION         = "v5"
NOTES           = "Short only — testing direction asymmetry identified in v4 diagnostics"
STRATEGY        = "Trend Following"

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
    # ── ADX (14-period, Wilder smoothing) ────────────────────────────────────
    _adx_n   = 14
    _alpha   = 1.0 / _adx_n
    _cp      = df.Close.shift(1)
    _tr      = pd.concat([(df.High - df.Low),
                           (df.High - _cp).abs(),
                           (df.Low  - _cp).abs()], axis=1).max(axis=1)
    _hd      = df.High.diff()
    _ld      = (-df.Low.diff())
    _pdm_raw = np.where((_hd > _ld) & (_hd > 0), _hd, 0.0)
    _mdm_raw = np.where((_ld > _hd) & (_ld > 0), _ld, 0.0)
    _pdm     = pd.Series(_pdm_raw, index=df.index)
    _mdm     = pd.Series(_mdm_raw, index=df.index)
    _atr14   = _tr.ewm(alpha=_alpha,  adjust=False).mean()
    _pdm14   = _pdm.ewm(alpha=_alpha, adjust=False).mean()
    _mdm14   = _mdm.ewm(alpha=_alpha, adjust=False).mean()
    _pdi     = 100.0 * _pdm14 / _atr14
    _mdi     = 100.0 * _mdm14 / _atr14
    _denom   = (_pdi + _mdi).replace(0, np.nan)
    _dx      = 100.0 * (_pdi - _mdi).abs() / _denom
    df["adx"] = _dx.ewm(alpha=_alpha, adjust=False).mean()
    return df.dropna().reset_index()

# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df):
    cash          = STARTING_CASH
    equity        = [cash]
    trades        = []
    in_trade      = False
    entry_p       = sl = tp = size = 0
    direction     = None
    entry_idx     = 0
    worst_adverse = 0.0   # tracks furthest adverse price during a trade
    entry_adx     = 0.0   # ADX value at entry bar

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
            # Update MAE: track worst adverse intra-bar price before exit check
            if direction == "long":
                worst_adverse = min(worst_adverse, float(df.Low.iloc[i]))
            else:
                worst_adverse = max(worst_adverse, float(df.High.iloc[i]))

            hit_sl = (direction == "long"  and c <= sl) or \
                     (direction == "short" and c >= sl)
            hit_tp = (direction == "long"  and c >= tp) or \
                     (direction == "short" and c <= tp)

            if hit_sl or hit_tp:
                exit_p  = sl if hit_sl else tp
                pnl     = (exit_p - entry_p) * size if direction == "long" \
                          else (entry_p - exit_p) * size
                mae     = abs(entry_p - worst_adverse)
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
                    "result":       "TP" if hit_tp else "SL",
                    "mae":          mae,
                    "adx_at_entry": entry_adx,
                    "timestamp":    ts
                })
                in_trade = False

        equity.append(cash)

        # ── Check entries ─────────────────────────────────────────────────────
        if not in_trade:
            trend_up   = fast > slow
            trend_down = fast < slow
            long_sig   = trend_up   and cp < enp and c > en
            short_sig  = trend_down and cp > enp and c < en

            # Apply direction filter
            if TRADE_DIRECTION == "long_only":
                short_sig = False
            elif TRADE_DIRECTION == "short_only":
                long_sig = False

            if long_sig and not np.isnan(s_lo):
                dist = c - s_lo
                if MIN_STOP <= dist <= MAX_STOP:
                    direction     = "long"
                    entry_p       = c
                    sl            = s_lo
                    tp            = c + dist * RRR
                    size          = (cash * RISK_PCT) / dist
                    in_trade      = True
                    entry_idx     = i
                    worst_adverse = c   # reset MAE tracker to entry price
                    entry_adx     = float(df.adx.iloc[i])

            elif short_sig and not np.isnan(s_hi):
                dist = s_hi - c
                if MIN_STOP <= dist <= MAX_STOP:
                    direction     = "short"
                    entry_p       = c
                    sl            = s_hi
                    tp            = c - dist * RRR
                    size          = (cash * RISK_PCT) / dist
                    in_trade      = True
                    entry_idx     = i
                    worst_adverse = c   # reset MAE tracker to entry price
                    entry_adx     = float(df.adx.iloc[i])

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
    ax3.fill_between(eq_dates[:len(dd)], dd, 0, color="#ff6b6b", alpha=0.6)
    ax3.set_ylabel("Drawdown %", color="white")
    ax3.set_xlabel("Date", color="white")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2.0)

    # Build a unique filename: results/{version}_{ticker}_{date}.png
    os.makedirs("results", exist_ok=True)
    ticker_clean = TICKER.split("=")[0].replace("^", "")
    date_str     = datetime.now().strftime("%Y-%m-%d")
    chart_path   = os.path.join("results", f"{VERSION}_{ticker_clean}_{date_str}.png")

    plt.savefig(chart_path, dpi=130, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close()
    print(f"  Chart saved → {chart_path}")
    print(f"  Open with:    open {chart_path}\n")
    return chart_path

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

    # ── By direction ──────────────────────────────────────────────────────────
    by_dir = {}
    for d in ["long", "short"]:
        sub = trades[trades.direction == d]
        if sub.empty:
            by_dir[d] = None
            continue
        sub_w  = sub[sub.win]
        sub_l  = sub[~sub.win]
        sub_pf = None
        if not sub_l.empty and sub_l.pnl.sum() != 0:
            sub_pf = round(abs(float(sub_w.pnl.sum()) / abs(float(sub_l.pnl.sum()))), 2)
        by_dir[d] = {
            "count":         int(len(sub)),
            "win_rate":      round(len(sub_w) / len(sub) * 100, 1),
            "profit_factor": sub_pf,
            "avg_win":       round(float(sub_w.pnl.mean()), 2) if not sub_w.empty else None,
            "avg_loss":      round(float(sub_l.pnl.mean()), 2) if not sub_l.empty else None,
            "net_pnl":       round(float(sub.pnl.sum()), 2),
        }

    # ── Monthly performance ────────────────────────────────────────────────────
    monthly = []
    try:
        t2 = trades.copy()
        t2["_month"] = pd.to_datetime(t2["timestamp"]).dt.to_period("M")
        for period, grp in t2.groupby("_month"):
            grp_w = grp[grp.win]
            grp_l = grp[~grp.win]
            monthly.append({
                "month":    str(period),
                "trades":   int(len(grp)),
                "wins":     int(len(grp_w)),
                "losses":   int(len(grp_l)),
                "win_rate": round(len(grp_w) / len(grp) * 100, 1),
                "net_pnl":  round(float(grp.pnl.sum()), 2),
            })
    except Exception:
        monthly = []

    # ── Streak analysis ────────────────────────────────────────────────────────
    results = trades["win"].tolist()
    max_win_streak = max_loss_streak = 0
    cur_w = cur_l = 0
    win_lens = []
    loss_lens = []
    for r in results:
        if r:
            cur_w += 1
            if cur_l > 0:
                loss_lens.append(cur_l)
                cur_l = 0
            max_win_streak = max(max_win_streak, cur_w)
        else:
            cur_l += 1
            if cur_w > 0:
                win_lens.append(cur_w)
                cur_w = 0
            max_loss_streak = max(max_loss_streak, cur_l)
    if cur_w > 0:
        win_lens.append(cur_w)
    if cur_l > 0:
        loss_lens.append(cur_l)
    avg_win_streak  = round(sum(win_lens)  / len(win_lens),  1) if win_lens  else 0.0
    avg_loss_streak = round(sum(loss_lens) / len(loss_lens), 1) if loss_lens else 0.0
    current_streak  = (f"{cur_w}W" if results[-1] else f"{cur_l}L") if results else "—"

    # ── Stop / target analysis ─────────────────────────────────────────────────
    pct_sl  = round(len(trades[trades.result == "SL"]) / n * 100, 1) \
              if "result" in trades.columns else None
    pct_tp  = round(len(trades[trades.result == "TP"]) / n * 100, 1) \
              if "result" in trades.columns else None
    avg_mae = round(float(trades["mae"].mean()) * 10000, 1) \
              if "mae" in trades.columns else None   # pips (×10 000)

    # ── Regime classification (ADX ≥ 25 = Trending, < 25 = Ranging) ──────────
    regime = []
    if "adx_at_entry" in trades.columns:
        for label, mask in [
            ("Trending (ADX \u226525)", trades["adx_at_entry"] >= 25),
            ("Ranging (ADX <25)",        trades["adx_at_entry"] <  25),
        ]:
            sub = trades[mask]
            if sub.empty:
                regime.append({"regime": label, "count": 0,
                                "win_rate": 0.0, "profit_factor": None, "net_pnl": 0.0})
                continue
            sub_w  = sub[sub.win]
            sub_l  = sub[~sub.win]
            sub_pf = None
            if not sub_l.empty and sub_l.pnl.sum() != 0:
                sub_pf = round(abs(float(sub_w.pnl.sum()) / abs(float(sub_l.pnl.sum()))), 2)
            regime.append({
                "regime":        label,
                "count":         int(len(sub)),
                "win_rate":      round(len(sub_w) / len(sub) * 100, 1),
                "profit_factor": sub_pf,
                "net_pnl":       round(float(sub.pnl.sum()), 2),
            })

    # ── Time of day performance (UTC hour) ────────────────────────────────────
    time_of_day = {"rows": [], "best_hour": None, "worst_hour": None}
    try:
        t3 = trades.copy()
        t3["_hour"] = pd.to_datetime(t3["timestamp"]).dt.hour
        tod_rows  = []
        best_pnl  = float("-inf")
        worst_pnl = float("inf")
        best_hour = worst_hour = None
        for hour, grp in t3.groupby("_hour"):
            grp_w = grp[grp.win]
            net   = round(float(grp.pnl.sum()), 2)
            tod_rows.append({
                "hour":     int(hour),
                "trades":   int(len(grp)),
                "win_rate": round(len(grp_w) / len(grp) * 100, 1),
                "net_pnl":  net,
            })
            if net > best_pnl:
                best_pnl  = net
                best_hour = int(hour)
            if net < worst_pnl:
                worst_pnl  = net
                worst_hour = int(hour)
        time_of_day = {"rows": tod_rows, "best_hour": best_hour, "worst_hour": worst_hour}
    except Exception:
        pass

    # ── Win rate trend (three equal time segments) ────────────────────────────
    win_rate_trend = []
    try:
        if n >= 3:
            seg_size = n // 3
            segments = [
                trades.iloc[:seg_size],
                trades.iloc[seg_size : 2 * seg_size],
                trades.iloc[2 * seg_size :],
            ]
            for seg in segments:
                if seg.empty:
                    continue
                seg_w  = seg[seg.win]
                seg_l  = seg[~seg.win]
                seg_pf = None
                if not seg_l.empty and seg_l.pnl.sum() != 0:
                    seg_pf = round(abs(float(seg_w.pnl.sum()) / abs(float(seg_l.pnl.sum()))), 2)
                ts0 = pd.to_datetime(seg["timestamp"].iloc[0]).strftime("%Y-%m-%d")
                ts1 = pd.to_datetime(seg["timestamp"].iloc[-1]).strftime("%Y-%m-%d")
                win_rate_trend.append({
                    "period":        f"{ts0} to {ts1}",
                    "trades":        int(len(seg)),
                    "win_rate":      round(len(seg_w) / len(seg) * 100, 1),
                    "profit_factor": seg_pf,
                    "net_pnl":       round(float(seg.pnl.sum()), 2),
                })
    except Exception:
        win_rate_trend = []

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
        "by_direction":   by_dir,
        "monthly":        monthly,
        "streaks": {
            "max_win_streak":  max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_win_streak":  avg_win_streak,
            "avg_loss_streak": avg_loss_streak,
            "current_streak":  current_streak,
        },
        "stop_target": {
            "pct_sl":  pct_sl,
            "pct_tp":  pct_tp,
            "avg_mae": avg_mae,
        },
        "regime":          regime,
        "time_of_day":     time_of_day,
        "win_rate_trend":  win_rate_trend,
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
      padding: 18px 16px 14px;
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
      margin-bottom: 12px;
    }
    .sb-label {
      font-size: 10px;
      font-weight: 700;
      color: #484868;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin-bottom: 6px;
    }
    #strategy-select {
      width: 100%;
      background: #181828;
      color: #c0c0e0;
      border: 1px solid #2a2a44;
      border-radius: 5px;
      padding: 6px 10px;
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      outline: none;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23505080'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
    }
    #strategy-select:focus { border-color: #4cc9f0; }

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

    #sidebar-footer {
      border-top: 1px solid #1e1e30;
      padding: 12px 16px;
      flex-shrink: 0;
    }
    #devlog-btn {
      width: 100%;
      background: #181828;
      color: #9090c0;
      border: 1px solid #2a2a44;
      border-radius: 5px;
      padding: 8px 12px;
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      text-align: left;
      transition: background 0.15s, color 0.15s;
    }
    #devlog-btn:hover { background: #1e1e38; color: #c0c0e0; }
    #devlog-btn.active { background: #182040; border-color: #4cc9f0; color: #4cc9f0; }

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

    /* ── Dev log ──────────────────────────────────────────── */
    #devlog-header { margin-bottom: 28px; }
    #devlog-header h2 { font-size: 24px; font-weight: 700; color: #e4e4ff; }
    #devlog-header .v-meta { font-size: 12px; color: #505070; margin-top: 6px; }
    .devlog-table thead th { white-space: nowrap; }
    .devlog-table .dl-notes { color: #9090c0; font-size: 12px; max-width: 300px; }
    .arrow-up { color: #6bcb77; font-weight: 700; margin-left: 5px; }
    .arrow-dn { color: #ff6b6b; font-weight: 700; margin-left: 5px; }
    .arrow-nc { color: #505070; margin-left: 5px; }

    /* ── Copy button ──────────────────────────────────────── */
    #v-header-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }
    #v-header-top h2 { flex: 1; min-width: 0; }
    .copy-btn {
      flex-shrink: 0;
      background: #181828;
      color: #7070a0;
      border: 1px solid #2a2a44;
      border-radius: 5px;
      padding: 6px 14px;
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.15s, color 0.15s, border-color 0.15s;
      margin-top: 4px;
    }
    .copy-btn:hover { background: #1e1e38; color: #c0c0e0; border-color: #3a3a60; }
    .copy-btn.copied { background: rgba(107,203,119,.12); border-color: #6bcb77; color: #6bcb77; }

    /* ── Toast ────────────────────────────────────────────── */
    #copy-toast {
      position: fixed;
      bottom: 28px;
      right: 32px;
      background: #0e2a1a;
      border: 1px solid #6bcb77;
      color: #6bcb77;
      border-radius: 7px;
      padding: 10px 20px;
      font-size: 13px;
      font-weight: 600;
      pointer-events: none;
      opacity: 0;
      transform: translateY(10px);
      transition: opacity 0.18s ease, transform 0.18s ease;
      z-index: 999;
    }
    #copy-toast.show { opacity: 1; transform: translateY(0); }

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
    <div class="sb-label">Strategy</div>
    <select id="strategy-select">
      <option value="Trend Following">Trend Following</option>
      <option value="Counter Trend">Counter Trend</option>
      <option value="Range Trading">Range Trading</option>
    </select>
  </div>
  <div id="version-list"></div>
  <div id="sidebar-footer">
    <button id="devlog-btn">&#128203;&nbsp; Development Log</button>
  </div>
</div>

<div id="main">
  <div id="content">
    <div id="empty-state">
      <span style="font-size:36px">&#128202;</span>
      <span>Select a version to view results</span>
    </div>
  </div>
</div>

<div id="copy-toast">&#10003;&nbsp; Copied to clipboard!</div>

<script type="application/json" id="versions-data">
__VERSIONS_JSON__
</script>

<script>
(function () {
  "use strict";

  var VERSIONS = JSON.parse(document.getElementById("versions-data").textContent);
  var currentStrategy = "Trend Following";
  var devLogOpen = false;

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

  function fmtMonth(s) {
    var names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    var parts = String(s).split("-");
    if (parts.length < 2) return s;
    var idx = parseInt(parts[1], 10) - 1;
    return names[idx] + " " + parts[0];
  }

  /* ── Toast ──────────────────────────────────────────────── */
  function showToast() {
    var toast = document.getElementById("copy-toast");
    toast.classList.add("show");
    setTimeout(function () { toast.classList.remove("show"); }, 2200);
  }

  /* ── Markdown builder ───────────────────────────────────── */
  function buildMarkdown(ver) {
    var m = ver.metrics || {};
    var p = ver.params  || {};

    function mf(n, d) {
      if (n === null || n === undefined || (typeof n === "number" && isNaN(n))) return "—";
      return Number(n).toFixed(d !== undefined ? d : 2);
    }
    function mfMoney(n) {
      if (n === null || n === undefined) return "—";
      var f = Number(n);
      return (f >= 0 ? "+" : "-") + "$" + Math.abs(f).toFixed(2);
    }

    var pf    = (m.profit_factor === null || m.profit_factor === undefined) ? "\u221e" : mf(m.profit_factor);
    var npStr = mfMoney(m.net_profit);
    if (m.net_profit_pct !== null && m.net_profit_pct !== undefined) {
      npStr += " (" + (m.net_profit_pct >= 0 ? "+" : "") + mf(m.net_profit_pct, 1) + "%)";
    }

    var lines = [];
    lines.push("## Backtest Report \u2014 " + (ver.name || "?"));
    lines.push("");
    lines.push("**Strategy:** " + (ver.strategy || "\u2014"));
    lines.push("**Instrument:** " + (p.ticker   || "\u2014") + " \u00b7 " + (p.interval || "\u2014") + " \u00b7 " + (p.days_back || "\u2014") + " days");
    lines.push("**Date:** "     + (ver.date    || "\u2014"));
    if (ver.notes && ver.notes !== "\u2014" && ver.notes !== "\u2014") {
      lines.push("**Notes:** " + ver.notes);
    }
    lines.push("");

    lines.push("### Results");
    lines.push("");
    lines.push("| Metric | Value |");
    lines.push("|--------|-------|");
    lines.push("| Total Trades | "    + (m.total_trades   !== undefined ? m.total_trades   : "\u2014") + " |");
    lines.push("| Winning Trades | "  + (m.winning_trades !== undefined ? m.winning_trades : "\u2014") + " |");
    lines.push("| Losing Trades | "   + (m.losing_trades  !== undefined ? m.losing_trades  : "\u2014") + " |");
    lines.push("| Win Rate | "        + mf(m.win_rate, 1) + "% |");
    lines.push("| Profit Factor | "   + pf + " |");
    lines.push("| Net Profit | "      + npStr + " |");
    lines.push("| Final Equity | $"   + mf(m.final_equity) + " |");
    lines.push("| Max Drawdown | "    + mf(m.max_drawdown) + "% |");
    lines.push("| Sharpe Ratio | "    + mf(m.sharpe) + " |");
    lines.push("| Avg Win | "         + (m.avg_win  !== null && m.avg_win  !== undefined ? "$"  + mf(m.avg_win)  : "\u2014") + " |");
    lines.push("| Avg Loss | "        + (m.avg_loss !== null && m.avg_loss !== undefined ? "$"  + mf(m.avg_loss) : "\u2014") + " |");
    lines.push("| Best Trade | "      + mfMoney(m.best_trade) + " |");
    lines.push("| Worst Trade | "     + mfMoney(m.worst_trade) + " |");
    lines.push("");

    lines.push("### Parameters");
    lines.push("");
    lines.push("| Parameter | Value |");
    lines.push("|-----------|-------|");
    lines.push("| Instrument | "    + (p.ticker    || "\u2014") + " |");
    lines.push("| Interval | "      + (p.interval  || "\u2014") + " |");
    lines.push("| History | "       + (p.days_back || "\u2014") + " days |");
    lines.push("| Starting Cash | $" + (p.starting_cash || 0).toLocaleString() + " |");
    lines.push("| EMA Slow | "      + (p.ema_slow      || "\u2014") + " |");
    lines.push("| EMA Fast | "      + (p.ema_fast      || "\u2014") + " |");
    lines.push("| EMA Entry | "     + (p.ema_entry     || "\u2014") + " |");
    lines.push("| Swing Lookback | " + (p.swing_lookback || "\u2014") + " bars |");
    lines.push("| RRR | 1:"         + (p.rrr || "\u2014") + " |");
    lines.push("| Risk / Trade | "  + ((p.risk_pct || 0) * 100).toFixed(1) + "% |");
    lines.push("| Min Stop | "      + ((p.min_stop || 0) * 10000).toFixed(0) + " pips |");
    lines.push("| Max Stop | "      + ((p.max_stop || 0) * 10000).toFixed(0) + " pips |");
    lines.push("| Direction | "     + (p.trade_direction || "both") + " |");
    lines.push("");

    /* ── Performance by Direction ──────────────────── */
    lines.push("### Performance by Direction");
    lines.push("");
    lines.push("| Direction | Trades | Win Rate | Profit Factor | Avg Win | Avg Loss | Net P&L |");
    lines.push("|-----------|--------|----------|---------------|---------|----------|---------|");
    var bd = m.by_direction || {};
    ["long", "short"].forEach(function (d) {
      var data = bd[d];
      if (!data) {
        lines.push("| " + d.charAt(0).toUpperCase() + d.slice(1) + " | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |");
        return;
      }
      var dpf = (data.profit_factor === null || data.profit_factor === undefined) ? "\u221e" : mf(data.profit_factor);
      lines.push("| " + d.charAt(0).toUpperCase() + d.slice(1) +
        " | " + data.count +
        " | " + mf(data.win_rate, 1) + "%" +
        " | " + dpf +
        " | " + (data.avg_win  !== null && data.avg_win  !== undefined ? "$" + mf(data.avg_win)  : "\u2014") +
        " | " + (data.avg_loss !== null && data.avg_loss !== undefined ? "$" + mf(Math.abs(data.avg_loss)) : "\u2014") +
        " | " + mfMoney(data.net_pnl) + " |");
    });
    lines.push("");

    /* ── Monthly Performance ────────────────────────── */
    lines.push("### Monthly Performance");
    lines.push("");
    lines.push("| Month | Trades | Wins | Losses | Win Rate | Net P&L |");
    lines.push("|-------|--------|------|--------|----------|---------|");
    var monthly = m.monthly || [];
    if (monthly.length === 0) {
      lines.push("| \u2014 | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |");
    } else {
      monthly.forEach(function (mo) {
        var mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
        var parts  = String(mo.month).split("-");
        var mLabel = parts.length >= 2 ? mnames[parseInt(parts[1], 10) - 1] + " " + parts[0] : mo.month;
        lines.push("| " + mLabel +
          " | " + mo.trades +
          " | " + mo.wins +
          " | " + mo.losses +
          " | " + mf(mo.win_rate, 1) + "%" +
          " | " + mfMoney(mo.net_pnl) + " |");
      });
    }
    lines.push("");

    /* ── Streak Analysis ────────────────────────────── */
    var str = m.streaks || {};
    lines.push("### Streak Analysis");
    lines.push("");
    lines.push("| Metric | Value |");
    lines.push("|--------|-------|");
    lines.push("| Max Win Streak | "  + (str.max_win_streak  !== undefined ? str.max_win_streak  + " trades" : "\u2014") + " |");
    lines.push("| Max Loss Streak | " + (str.max_loss_streak !== undefined ? str.max_loss_streak + " trades" : "\u2014") + " |");
    lines.push("| Avg Win Streak | "  + (str.avg_win_streak  !== undefined ? str.avg_win_streak               : "\u2014") + " |");
    lines.push("| Avg Loss Streak | " + (str.avg_loss_streak !== undefined ? str.avg_loss_streak               : "\u2014") + " |");
    lines.push("| Current Streak | "  + (str.current_streak  || "\u2014") + " |");
    lines.push("");

    /* ── Stop vs Target ─────────────────────────────── */
    var st = m.stop_target || {};
    lines.push("### Stop vs Target");
    lines.push("");
    lines.push("| Metric | Value |");
    lines.push("|--------|-------|");
    lines.push("| Hit Stop Loss | "   + (st.pct_sl  !== null && st.pct_sl  !== undefined ? st.pct_sl  + "%" : "\u2014") + " |");
    lines.push("| Hit Take Profit | " + (st.pct_tp  !== null && st.pct_tp  !== undefined ? st.pct_tp  + "%" : "\u2014") + " |");
    lines.push("| Avg MAE | "         + (st.avg_mae !== null && st.avg_mae !== undefined ? st.avg_mae + " pips" : "\u2014") + " |");
    lines.push("");

    /* ── Regime Classification ───────────────────────── */
    lines.push("### Regime Classification");
    lines.push("");
    lines.push("| Regime | Trades | Win Rate | Profit Factor | Net P&L |");
    lines.push("|--------|--------|----------|---------------|---------|");
    var mRegime = m.regime || [];
    if (mRegime.length === 0) {
      lines.push("| \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |");
    } else {
      mRegime.forEach(function (r) {
        var rpf = (r.profit_factor === null || r.profit_factor === undefined) ? "\u221e" : mf(r.profit_factor);
        lines.push("| " + r.regime + " | " + r.count + " | " + mf(r.win_rate, 1) + "% | " + rpf + " | " + mfMoney(r.net_pnl) + " |");
      });
    }
    lines.push("");

    /* ── Time of Day Performance ─────────────────────── */
    lines.push("### Time of Day Performance (UTC)");
    lines.push("");
    lines.push("| Hour | Trades | Win Rate | Net P&L |");
    lines.push("|------|--------|----------|---------|");
    var mTod     = m.time_of_day || { rows: [], best_hour: null, worst_hour: null };
    var mTodRows = mTod.rows || [];
    if (mTodRows.length === 0) {
      lines.push("| \u2014 | \u2014 | \u2014 | \u2014 |");
    } else {
      mTodRows.forEach(function (r) {
        var hStr = (r.hour < 10 ? "0" : "") + r.hour + ":00";
        if (mTod.best_hour  !== null && r.hour === mTod.best_hour)  hStr += " \u2605";
        if (mTod.worst_hour !== null && r.hour === mTod.worst_hour) hStr += " \u25bc";
        lines.push("| " + hStr + " | " + r.trades + " | " + mf(r.win_rate, 1) + "% | " + mfMoney(r.net_pnl) + " |");
      });
    }
    lines.push("");

    /* ── Win Rate Trend ──────────────────────────────── */
    lines.push("### Win Rate Trend (3 Equal Periods)");
    lines.push("");
    lines.push("| Period | Trades | Win Rate | Profit Factor | Net P&L |");
    lines.push("|--------|--------|----------|---------------|---------|");
    var mWrt = m.win_rate_trend || [];
    if (mWrt.length === 0) {
      lines.push("| \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |");
    } else {
      var segLabels = ["Early", "Mid", "Late"];
      mWrt.forEach(function (seg, idx) {
        var spf   = (seg.profit_factor === null || seg.profit_factor === undefined) ? "\u221e" : mf(seg.profit_factor);
        var label = (segLabels[idx] || ("Seg " + (idx + 1))) + " (" + seg.period + ")";
        lines.push("| " + label + " | " + seg.trades + " | " + mf(seg.win_rate, 1) + "% | " + spf + " | " + mfMoney(seg.net_pnl) + " |");
      });
    }
    lines.push("");

    return lines.join("\n");
  }

  /* ── Strategy filter ─────────────────────────────────────── */
  function getStrategyVersions() {
    var result = [];
    for (var i = 0; i < VERSIONS.length; i++) {
      var strat = VERSIONS[i].strategy || "Trend Following";
      if (strat === currentStrategy) {
        result.push({ v: VERSIONS[i], idx: i });
      }
    }
    return result;
  }

  /* ── Sidebar ──────────────────────────────────────────────── */
  function renderSidebar() {
    var svs  = getStrategyVersions();
    document.getElementById("run-count").textContent =
      svs.length + " run" + (svs.length !== 1 ? "s" : "");

    var list = document.getElementById("version-list");
    list.innerHTML = "";

    if (svs.length === 0) {
      list.innerHTML = "<div style='padding:20px 16px;color:#404060;font-size:12px'>No runs for this strategy yet.</div>";
      return;
    }

    for (var ri = svs.length - 1; ri >= 0; ri--) {
      var entry = svs[ri];
      var v    = entry.v;
      var idx  = entry.idx;
      var pnl  = v.metrics ? v.metrics.net_profit : null;
      var pc   = pnl === null ? "" : (pnl >= 0 ? "pos" : "neg");
      var ptxt = pnl === null ? "" :
        (pnl >= 0 ? "+" : "") + "$" + Math.abs(pnl).toFixed(2);

      var item = document.createElement("div");
      item.className     = "v-item";
      item.dataset.idx   = idx;
      item.innerHTML     =
        "<div class='v-name'>" + esc(v.name) + "</div>" +
        "<div class='v-date'>" + esc(v.date) + "</div>" +
        (pnl !== null ? "<div class='v-pnl " + pc + "'>" + ptxt + "</div>" : "");

      (function (el, origIdx) {
        el.addEventListener("click", function () {
          devLogOpen = false;
          document.getElementById("devlog-btn").classList.remove("active");
          document.querySelectorAll(".v-item").forEach(function (e) {
            e.classList.remove("active");
          });
          el.classList.add("active");
          renderContent(origIdx);
        });
      })(item, idx);

      list.appendChild(item);
    }
  }

  /* ── Content ──────────────────────────────────────────────── */
  function renderContent(idx) {
    var v = VERSIONS[idx];
    if (!v) return;

    var m = v.metrics || {};
    var p = v.params  || {};

    /* profit factor display */
    var pfTxt = (m.profit_factor === null || m.profit_factor === undefined)
      ? "&#8734;" : fmt(m.profit_factor);
    var pfCls = (m.profit_factor === null || m.profit_factor === undefined || m.profit_factor >= 1.5)
      ? "pos" : (m.profit_factor < 1.0 ? "neg" : "neu");

    /* ── Analytical section builders ──────────────────────────── */

    /* 1. Performance by Direction */
    var bd    = m.by_direction || {};
    var bdLng = bd.long  || null;
    var bdSht = bd.short || null;
    function dirRow(label, data) {
      if (!data) {
        return "<tr><td><strong>" + label + "</strong></td>" +
               "<td colspan='6' style='color:#404060'>No data</td></tr>";
      }
      var dpf    = (data.profit_factor === null || data.profit_factor === undefined) ? "\u221e" : fmt(data.profit_factor);
      var dpfCls = (data.profit_factor === null || data.profit_factor === undefined || data.profit_factor >= 1.5) ? "pos" : (data.profit_factor < 1.0 ? "neg" : "neu");
      return "<tr>" +
        "<td><strong>" + label + "</strong></td>" +
        "<td>" + data.count + "</td>" +
        "<td class='" + (data.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(data.win_rate, 1) + "%</td>" +
        "<td class='" + dpfCls + "'>" + dpf + "</td>" +
        "<td class='pos'>" + (data.avg_win  !== null && data.avg_win  !== undefined ? "$" + fmt(data.avg_win)  : "\u2014") + "</td>" +
        "<td class='neg'>" + (data.avg_loss !== null && data.avg_loss !== undefined ? "$" + fmt(Math.abs(data.avg_loss)) : "\u2014") + "</td>" +
        "<td class='" + pClass(data.net_pnl) + "'>" + fmtMoney(data.net_pnl) + "</td>" +
        "</tr>";
    }
    var dirHtml =
      "<div class='section'>" +
        "<div class='section-title'>Performance by Direction</div>" +
        "<table><thead><tr>" +
        "<th>Direction</th><th>Trades</th><th>Win Rate</th>" +
        "<th>Profit Factor</th><th>Avg Win</th><th>Avg Loss</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" +
        dirRow("Long", bdLng) + dirRow("Short", bdSht) +
        "</tbody></table></div>";

    /* 2. Monthly Performance */
    var monthly = m.monthly || [];
    var mRows = "";
    monthly.forEach(function (mo) {
      var pc = mo.net_pnl >= 0 ? "#6bcb77" : "#ff6b6b";
      var bg = mo.net_pnl >= 0 ? "rgba(107,203,119,.08)" : "rgba(255,107,107,.08)";
      mRows +=
        "<tr>" +
        "<td>" + fmtMonth(mo.month) + "</td>" +
        "<td>" + mo.trades + "</td>" +
        "<td class='pos'>" + mo.wins + "</td>" +
        "<td class='neg'>" + mo.losses + "</td>" +
        "<td class='" + (mo.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(mo.win_rate, 1) + "%</td>" +
        "<td style='color:" + pc + ";background:" + bg + ";font-weight:600'>" + fmtMoney(mo.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!mRows) {
      mRows = "<tr><td colspan='6' style='color:#404060;text-align:center;padding:20px'>No data</td></tr>";
    }
    var monthHtml =
      "<div class='section'>" +
        "<div class='section-title'>Monthly Performance</div>" +
        "<table><thead><tr>" +
        "<th>Month</th><th>Trades</th><th>Wins</th><th>Losses</th>" +
        "<th>Win Rate</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + mRows + "</tbody></table></div>";

    /* 3. Streak Analysis */
    var str = m.streaks || {};
    var streakHtml =
      "<div class='section'>" +
        "<div class='section-title'>Streak Analysis</div>" +
        "<table><tbody>" +
        row("Max Win Streak",  str.max_win_streak  !== undefined ? str.max_win_streak  + " trades" : "\u2014") +
        row("Max Loss Streak", str.max_loss_streak !== undefined ? str.max_loss_streak + " trades" : "\u2014") +
        row("Avg Win Streak",  str.avg_win_streak  !== undefined ? str.avg_win_streak               : "\u2014") +
        row("Avg Loss Streak", str.avg_loss_streak !== undefined ? str.avg_loss_streak               : "\u2014") +
        row("Current Streak",  str.current_streak  || "\u2014") +
        "</tbody></table></div>";

    /* 4. Stop vs Target */
    var st = m.stop_target || {};
    var stopHtml =
      "<div class='section'>" +
        "<div class='section-title'>Stop vs Target</div>" +
        "<table><tbody>" +
        row("Hit Stop Loss",   st.pct_sl  !== null && st.pct_sl  !== undefined ? "<span class='neg'>"  + st.pct_sl  + "%</span>" : "\u2014") +
        row("Hit Take Profit", st.pct_tp  !== null && st.pct_tp  !== undefined ? "<span class='pos'>"  + st.pct_tp  + "%</span>" : "\u2014") +
        row("Avg MAE",         st.avg_mae !== null && st.avg_mae !== undefined ? st.avg_mae + " pips"                             : "\u2014") +
        "</tbody></table></div>";

    /* 5. Regime Classification */
    var regime  = m.regime || [];
    var regRows = "";
    regime.forEach(function (r) {
      var rpf    = (r.profit_factor === null || r.profit_factor === undefined) ? "\u221e" : fmt(r.profit_factor);
      var rpfCls = (r.profit_factor === null || r.profit_factor === undefined || r.profit_factor >= 1.5) ? "pos" : (r.profit_factor < 1.0 ? "neg" : "neu");
      regRows +=
        "<tr>" +
        "<td><strong>" + esc(r.regime) + "</strong></td>" +
        "<td>" + r.count + "</td>" +
        "<td class='" + (r.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(r.win_rate, 1) + "%</td>" +
        "<td class='" + rpfCls + "'>" + rpf + "</td>" +
        "<td class='" + pClass(r.net_pnl) + "'>" + fmtMoney(r.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!regRows) regRows = "<tr><td colspan='5' style='color:#404060;text-align:center;padding:20px'>No data</td></tr>";
    var regimeHtml =
      "<div class='section'>" +
        "<div class='section-title'>Regime Classification</div>" +
        "<table><thead><tr>" +
        "<th>Regime</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + regRows + "</tbody></table></div>";

    /* 6. Time of Day Performance */
    var tod     = m.time_of_day || { rows: [], best_hour: null, worst_hour: null };
    var todList = tod.rows || [];
    var todRows = "";
    todList.forEach(function (r) {
      var isBest  = tod.best_hour  !== null && r.hour === tod.best_hour;
      var isWorst = tod.worst_hour !== null && r.hour === tod.worst_hour;
      var rowBg   = isBest  ? "background:rgba(107,203,119,.07);" :
                    isWorst ? "background:rgba(255,107,107,.07);" : "";
      var suffix  = isBest ? " <span style='color:#6bcb77;font-size:11px'>\u2605 best</span>" :
                    isWorst ? " <span style='color:#ff6b6b;font-size:11px'>\u25bc worst</span>" : "";
      var hStr    = (r.hour < 10 ? "0" : "") + r.hour + ":00 UTC";
      todRows +=
        "<tr" + (rowBg ? " style='" + rowBg + "'" : "") + ">" +
        "<td>" + hStr + suffix + "</td>" +
        "<td>" + r.trades + "</td>" +
        "<td class='" + (r.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(r.win_rate, 1) + "%</td>" +
        "<td class='" + pClass(r.net_pnl) + "'>" + fmtMoney(r.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!todRows) todRows = "<tr><td colspan='4' style='color:#404060;text-align:center;padding:20px'>No data</td></tr>";
    var timeOfDayHtml =
      "<div class='section'>" +
        "<div class='section-title'>Time of Day Performance (UTC)</div>" +
        "<table><thead><tr>" +
        "<th>Hour</th><th>Trades</th><th>Win Rate</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + todRows + "</tbody></table></div>";

    /* 7. Win Rate Trend */
    var wrt     = m.win_rate_trend || [];
    var wrtRows = "";
    wrt.forEach(function (seg, idx) {
      var spf    = (seg.profit_factor === null || seg.profit_factor === undefined) ? "\u221e" : fmt(seg.profit_factor);
      var spfCls = (seg.profit_factor === null || seg.profit_factor === undefined || seg.profit_factor >= 1.5) ? "pos" : (seg.profit_factor < 1.0 ? "neg" : "neu");
      var labels = ["Early", "Mid", "Late"];
      wrtRows +=
        "<tr>" +
        "<td><strong style='color:#7070a0'>" + (labels[idx] || ("Seg " + (idx + 1))) + "</strong><br>" +
        "<span style='font-size:11px;color:#505070'>" + esc(seg.period) + "</span></td>" +
        "<td>" + seg.trades + "</td>" +
        "<td class='" + (seg.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(seg.win_rate, 1) + "%</td>" +
        "<td class='" + spfCls + "'>" + spf + "</td>" +
        "<td class='" + pClass(seg.net_pnl) + "'>" + fmtMoney(seg.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!wrtRows) wrtRows = "<tr><td colspan='5' style='color:#404060;text-align:center;padding:20px'>No data</td></tr>";
    var winRateTrendHtml =
      "<div class='section'>" +
        "<div class='section-title'>Win Rate Trend (3 equal periods)</div>" +
        "<table><thead><tr>" +
        "<th>Period</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + wrtRows + "</tbody></table></div>";

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

    var stratBadge = (v.strategy)
      ? "<span style='font-size:11px;background:#1a1a30;border:1px solid #2a2a44;border-radius:4px;padding:2px 8px;color:#6060a0;margin-left:8px'>" + esc(v.strategy) + "</span>"
      : "";

    document.getElementById("content").innerHTML =
      /* header */
      "<div id='v-header'>" +
        "<div id='v-header-top'>" +
          "<h2>" + esc(v.name) + stratBadge + "</h2>" +
          "<button id='copy-btn' class='copy-btn'>Copy Report</button>" +
        "</div>" +
        "<div class='v-meta'>Run on " + esc(v.date) +
          " &nbsp;&middot;&nbsp; " + esc(p.ticker || "") +
          " " + esc(p.interval || "") +
          " &nbsp;&middot;&nbsp; " + (p.days_back || "") + " days</div>" +
        notesHtml +
      "</div>" +

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
          row("Direction",      "<span style='color:#ffd93d;font-weight:600'>" + esc(p.trade_direction || "both") + "</span>") +
          "</tbody></table>" +
        "</div>" +

      "</div>" +

      /* chart */
      chartHtml +

      /* analytical sections */
      dirHtml +
      monthHtml +
      "<div class='two-col'>" + streakHtml + stopHtml + "</div>" +
      "<div class='two-col'>" + regimeHtml + winRateTrendHtml + "</div>" +
      timeOfDayHtml;

    /* Wire copy button */
    (function (ver) {
      var btn = document.getElementById("copy-btn");
      if (!btn) return;
      btn.addEventListener("click", function () {
        var md = buildMarkdown(ver);
        navigator.clipboard.writeText(md).then(function () {
          btn.textContent = "\u2713 Copied!";
          btn.classList.add("copied");
          showToast();
          setTimeout(function () {
            btn.textContent = "Copy Report";
            btn.classList.remove("copied");
          }, 2200);
        }).catch(function () {
          btn.textContent = "Failed";
          setTimeout(function () { btn.textContent = "Copy Report"; }, 2500);
        });
      });
    }(v));
  }

  /* ── Dev Log ──────────────────────────────────────────────── */
  function showDevLog() {
    var svs = getStrategyVersions();

    if (svs.length === 0) {
      document.getElementById("content").innerHTML =
        "<div id='devlog-header'>" +
          "<h2>Development Log</h2>" +
          "<div class='v-meta'>" + esc(currentStrategy) + " &mdash; no runs yet</div>" +
        "</div>";
      return;
    }

    var dlRows = "";
    var prevPF = null;

    for (var i = 0; i < svs.length; i++) {
      var entry = svs[i];
      var v = entry.v;
      var m = v.metrics || {};

      var pf = (m.profit_factor !== null && m.profit_factor !== undefined) ? m.profit_factor : null;
      var wr = (m.win_rate !== null && m.win_rate !== undefined) ? m.win_rate : null;
      var np = (m.net_profit !== null && m.net_profit !== undefined) ? m.net_profit : null;

      /* PF change arrow */
      var arrowHtml = "";
      if (prevPF !== null && pf !== null) {
        if (pf > prevPF + 0.005) {
          arrowHtml = "<span class='arrow-up'>&#9650;</span>";
        } else if (pf < prevPF - 0.005) {
          arrowHtml = "<span class='arrow-dn'>&#9660;</span>";
        } else {
          arrowHtml = "<span class='arrow-nc'>&#8213;</span>";
        }
      }
      if (pf !== null) prevPF = pf;

      var pfDisp = (pf === null) ? "&#8212;" :
        "<span class='" + (pf >= 1.5 ? "pos" : (pf < 1.0 ? "neg" : "neu")) + "'>" + pf.toFixed(2) + "</span>" + arrowHtml;

      var wrDisp = (wr === null) ? "&#8212;" :
        "<span class='" + (wr >= 50 ? "pos" : "neg") + "'>" + wr.toFixed(1) + "%</span>";

      var npDisp = (np === null) ? "&#8212;" :
        "<span class='" + (np >= 0 ? "pos" : "neg") + "'>" + fmtMoney(np) + "</span>";

      var notes = (v.notes && v.notes !== "—" && v.notes !== "\u2014") ? esc(v.notes) : "<span style='color:#404060'>—</span>";

      dlRows +=
        "<tr>" +
        "<td style='white-space:nowrap;font-weight:600;color:#c8c8e8'>" + esc(v.name) + "</td>" +
        "<td style='white-space:nowrap;color:#505070'>" + esc(v.date) + "</td>" +
        "<td class='dl-notes'>" + notes + "</td>" +
        "<td style='text-align:right;white-space:nowrap'>" + pfDisp + "</td>" +
        "<td style='text-align:right;white-space:nowrap'>" + wrDisp + "</td>" +
        "<td style='text-align:right;white-space:nowrap'>" + npDisp + "</td>" +
        "</tr>";
    }

    document.getElementById("content").innerHTML =
      "<div id='devlog-header'>" +
        "<h2>Development Log</h2>" +
        "<div class='v-meta'>" + esc(currentStrategy) + " &mdash; " + svs.length + " run" + (svs.length !== 1 ? "s" : "") + "</div>" +
      "</div>" +
      "<div class='section'>" +
        "<table class='devlog-table'>" +
          "<thead><tr>" +
          "<th>Version</th><th>Date</th><th>Change</th>" +
          "<th style='text-align:right'>Profit Factor</th>" +
          "<th style='text-align:right'>Win Rate</th>" +
          "<th style='text-align:right'>Net P&amp;L</th>" +
          "</tr></thead>" +
          "<tbody>" + dlRows + "</tbody>" +
        "</table>" +
      "</div>";
  }

  /* ── Strategy change ─────────────────────────────────────── */
  function onStrategyChange() {
    currentStrategy = document.getElementById("strategy-select").value;
    devLogOpen = false;
    document.getElementById("devlog-btn").classList.remove("active");
    renderSidebar();

    /* auto-select most recent version for new strategy */
    var svs = getStrategyVersions();
    if (svs.length > 0) {
      var lastIdx = svs[svs.length - 1].idx;
      var firstItem = document.querySelector(".v-item");
      if (firstItem) firstItem.classList.add("active");
      renderContent(lastIdx);
    } else {
      document.getElementById("content").innerHTML =
        "<div id='empty-state'>" +
          "<span style='font-size:36px'>&#128202;</span>" +
          "<span>No runs for <strong>" + esc(currentStrategy) + "</strong> yet</span>" +
        "</div>";
    }
  }

  /* ── Init ──────────────────────────────────────────────────── */
  document.getElementById("strategy-select").addEventListener("change", onStrategyChange);

  document.getElementById("devlog-btn").addEventListener("click", function () {
    devLogOpen = !devLogOpen;
    this.classList.toggle("active", devLogOpen);
    document.querySelectorAll(".v-item").forEach(function (e) {
      e.classList.remove("active");
    });
    if (devLogOpen) {
      showDevLog();
    } else {
      /* re-select most recent version */
      var svs = getStrategyVersions();
      if (svs.length > 0) {
        var lastIdx = svs[svs.length - 1].idx;
        var firstItem = document.querySelector(".v-item");
        if (firstItem) firstItem.classList.add("active");
        renderContent(lastIdx);
      }
    }
  });

  renderSidebar();
  var svs = getStrategyVersions();
  if (svs.length > 0) {
    var lastIdx = svs[svs.length - 1].idx;
    var firstItem = document.querySelector(".v-item");
    if (firstItem) firstItem.classList.add("active");
    renderContent(lastIdx);
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
            "risk_pct":        RISK_PCT,
            "min_stop":        MIN_STOP,
            "max_stop":        MAX_STOP,
            "trade_direction": TRADE_DIRECTION,
        },
        "last_trades": last_trades,
        "strategy":    STRATEGY,
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

# ── Results log ───────────────────────────────────────────────────────────────

_RESULTS_LOG_PATH = "RESULTS_LOG.md"

_RESULTS_LOG_HEADER = (
    "# Results Log\n\n"
    "| Version | Date | Strategy | Instrument | Timeframe | Notes "
    "| Trades | Win Rate | Profit Factor | Net P&L | Max Drawdown | Sharpe Ratio |\n"
    "|---------|------|----------|------------|-----------|-------"
    "|--------|----------|---------------|---------|--------------|-------------|\n"
)


def _fmt_money(n):
    if n is None:
        return "—"
    try:
        f = float(n)
        return f"+${f:.2f}" if f >= 0 else f"-${abs(f):.2f}"
    except (TypeError, ValueError):
        return "—"


def _safe_cell(s):
    """Strip pipe chars that would break a markdown table."""
    return str(s).replace("|", "\\|").strip() if s else "—"


def update_results_log(metrics, notes=""):
    """Append one row to RESULTS_LOG.md (creates the file with header if absent)."""
    if metrics is None:
        print("  No metrics — skipping results log update.")
        return

    m = metrics

    pf_raw = m.get("profit_factor")
    pf     = "∞" if pf_raw is None else f"{float(pf_raw):.2f}"
    wr     = f"{float(m['win_rate']):.1f}%" if m.get("win_rate") is not None else "—"
    dd     = f"{float(m['max_drawdown']):.2f}%" if m.get("max_drawdown") is not None else "—"
    sharpe = f"{float(m['sharpe']):.2f}" if m.get("sharpe") is not None else "—"
    trades = str(m["total_trades"]) if m.get("total_trades") is not None else "—"
    pnl    = _fmt_money(m.get("net_profit"))
    date   = datetime.now().strftime("%Y-%m-%d %H:%M")
    note   = _safe_cell(notes) if notes else "—"

    row = (
        f"| {VERSION} | {date} | {STRATEGY} | {TICKER} | {INTERVAL} | {note} "
        f"| {trades} | {wr} | {pf} | {pnl} | {dd} | {sharpe} |\n"
    )

    if not os.path.exists(_RESULTS_LOG_PATH):
        with open(_RESULTS_LOG_PATH, "w", encoding="utf-8") as fh:
            fh.write(_RESULTS_LOG_HEADER)
        print(f"  Created {_RESULTS_LOG_PATH}")

    with open(_RESULTS_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(row)

    print(f"  Results log updated → {_RESULTS_LOG_PATH}")


# ── Git auto-commit ───────────────────────────────────────────────────────────

def git_commit_and_push(metrics, version, ticker, interval):
    """Stage all changes, commit with a metrics summary, then push.
    A failed push prints a warning but never raises — results are already saved."""

    if metrics is None:
        print("  No metrics — skipping git commit.")
        return

    pf  = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] is not None else "∞"
    wr  = f"{metrics['win_rate']:.1f}"
    dd  = f"{metrics['max_drawdown']:.2f}"
    pnl = f"{metrics['net_profit_pct']:+.1f}"
    date_str = datetime.now().strftime("%Y-%m-%d")

    msg = (
        f"backtest {version} {ticker} {interval} {date_str} | "
        f"WR {wr}% PF {pf} DD {dd}% P&L {pnl}%"
    )

    repo_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True)
        print("  git add -A ... done")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠  git add failed: {e}")
        return

    try:
        subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir, check=True)
        print(f"  git commit ... done")
        print(f"  Commit message: {msg}")
    except subprocess.CalledProcessError as e:
        # Exit code 1 means nothing to commit — not a real error
        print(f"  git commit — nothing new to commit (or error: {e})")
        return

    try:
        subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, check=True)
        print("  git push origin main ... done\n")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠  git push failed (results saved locally): {e}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # NOTES constant is used by default; a CLI argument overrides it if supplied
    cli_notes       = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    run_notes       = cli_notes if cli_notes else NOTES
    df              = fetch_data(TICKER, INTERVAL, DAYS_BACK)
    df              = add_indicators(df)
    trades, equity  = run_backtest(df)
    print_results(trades, equity)
    chart_path = save_charts(df, trades, equity)
    generate_html_report(trades, equity, chart_path=chart_path, notes=run_notes)
    metrics = compute_metrics(trades, equity)
    update_results_log(metrics, notes=run_notes)
    git_commit_and_push(metrics, VERSION, TICKER, INTERVAL)
