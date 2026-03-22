"""
Trading Bot — EMA Trend Following Strategy
==========================================
Instrument : EURUSD hourly (Yahoo Finance)
Strategy   : EMA 20/50/200 crossover with swing stop
RRR        : 1:2
Risk/Trade : 1% of equity

Usage:
    source venv/bin/activate
    python3 strategy.py
"""

import sys
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

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df              = fetch_data(TICKER, INTERVAL, DAYS_BACK)
    df              = add_indicators(df)
    trades, equity  = run_backtest(df)
    print_results(trades, equity)
    save_charts(df, trades, equity)
