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
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env (API keys, secrets — never commit .env)
load_dotenv()
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY")

# ── Configuration ─────────────────────────────────────────────────────────────

# Instrument mapping: INSTRUMENT env var → (Yahoo ticker, Massive ticker)
_INSTRUMENT_MAP = {
    "EURUSD": ("EURUSD=X", "C:EURUSD"),
    "GBPUSD": ("GBPUSD=X", "C:GBPUSD"),
}
_INSTRUMENT     = os.environ.get("INSTRUMENT", "EURUSD")
TICKER          = _INSTRUMENT_MAP.get(_INSTRUMENT, _INSTRUMENT_MAP["EURUSD"])[0]
MASSIVE_TICKER  = _INSTRUMENT_MAP.get(_INSTRUMENT, _INSTRUMENT_MAP["EURUSD"])[1]
INTERVAL        = "5m"            # bar interval — used by Massive (primary) and Yahoo (fallback)
DAYS_BACK       = 730             # Full 730-day run
STARTING_CASH   = 100_000.0

EMA_SLOW        = 200
EMA_FAST        = 50
EMA_ENTRY       = 20
SWING_LOOKBACK  = 20
RRR             = 2.0
RISK_PCT        = 0.01
MIN_STOP        = 0.0005     # 5 pips minimum stop
MAX_STOP        = 0.0200     # 200 pips maximum stop

TRADE_DIRECTION   = "short_only"   # "both" | "long_only" | "short_only"

TIME_FILTER       = True
TIME_FILTER_HOURS = [1, 2, 16, 17, 18]           # UTC hours allowed

MAX_DAILY_LOSS  = 2500.0            # stop trading if day's loss reaches $2,500 (2.5% of capital)

ROLLING_PF_WINDOW = 10              # window size for rolling profit factor

VERSION = "v5"
NOTES = "Removed regime filter — clean baseline for date range UI rebuild"
STRATEGY        = "Trend Following"

ENTRY_CONDITIONS = [
    {
        "condition":       "Trend Filter",
        "rule":            f"EMA{EMA_FAST} < EMA{EMA_SLOW}",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Entry Signal",
        "rule":            f"Price crosses below EMA{EMA_ENTRY}",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Stop Placement",
        "rule":            f"Swing high over {SWING_LOOKBACK} bars",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Direction",
        "rule":            "Short only",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Time Window",
        "rule":            f"UTC {' '.join(f'{h:02d}' for h in sorted(TIME_FILTER_HOURS))}",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Daily Loss Limit",
        "rule":            f"Stop trading if daily loss >= ${MAX_DAILY_LOSS:,.0f}",

        "since_version":   "v1",
        "removed_version": None,
    },
    {
        "condition":       "Regime Filter",
        "rule":            "ATR range detection — removed",

        "since_version":   "v2",
        "removed_version": "v5",
    },
]

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(ticker, interval, days_back, start_date=None, end_date=None):
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        days_back = (end - start).days
    else:
        end   = datetime.now()
        start = end - timedelta(days=days_back)

    # ── Primary: Massive API ───────────────────────────────────────────────────
    if MASSIVE_API_KEY:
        try:
            from massive import RESTClient
            print(f"\nFetching {MASSIVE_TICKER} {interval} data from Massive ({days_back} days)...")
            client = RESTClient(api_key=MASSIVE_API_KEY)
            bars = list(client.list_aggs(
                ticker    = MASSIVE_TICKER,
                multiplier= 5,
                timespan  = "minute",
                from_     = start.strftime("%Y-%m-%d"),
                to        = end.strftime("%Y-%m-%d"),
                sort      = "asc",
                limit     = 50000,
            ))
            if not bars:
                raise ValueError("No bars returned from Massive API")
            # Build DataFrame: timestamp is Unix milliseconds UTC
            df = pd.DataFrame(
                {
                    "Open":   [b.open   for b in bars],
                    "High":   [b.high   for b in bars],
                    "Low":    [b.low    for b in bars],
                    "Close":  [b.close  for b in bars],
                    "Volume": [b.volume if b.volume is not None else 0.0 for b in bars],
                },
                index=pd.DatetimeIndex(
                    [pd.Timestamp(b.timestamp, unit="ms", tz="UTC") for b in bars],
                    name="Datetime",
                ),
            )
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            print(f"  {len(df)} bars loaded | {df.index[0].date()} → {df.index[-1].date()}")
            print(f"  Source: Massive API ({MASSIVE_TICKER}, 5m)")
            return df
        except Exception as e:
            print(f"  Massive API fetch failed: {e}")
            print(f"  Falling back to Yahoo Finance...")
    else:
        print(f"\nNo MASSIVE_API_KEY found — using Yahoo Finance fallback.")

    # ── Fallback: Yahoo Finance ────────────────────────────────────────────────
    # Yahoo free tier: 5m data is capped at 60 days; use 1h for longer history
    try:
        import yfinance as yf
        yf_interval = "1h"
        yf_days     = min(days_back, 720)
        yf_start    = end - timedelta(days=yf_days)
        print(f"\nFetching {ticker} {yf_interval} data from Yahoo ({yf_days} days)...")
        df = yf.download(ticker, start=yf_start, end=end,
                         interval=yf_interval, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty dataframe returned")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        print(f"  {len(df)} bars loaded | {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Source: Yahoo Finance ({ticker}, 1h fallback)")
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
    df = df.dropna().reset_index()
    # Normalise the datetime column to 'Datetime' regardless of yfinance version
    for _col in ("Datetime", "Date", "index"):
        if _col in df.columns and _col != "Datetime":
            df = df.rename(columns={_col: "Datetime"})
            break
    return df

# ── Blocked-signal outcome simulation ────────────────────────────────────────

def _scan_outcome(df, entry_idx, direction, entry_p, sl, tp, size, ts, reason):
    """Forward-scan to estimate the TP/SL outcome of a signal that was filtered out.
    Scans up to 1 000 bars ahead; uses last-bar close if neither level is hit."""
    pnl  = None
    end  = min(entry_idx + 1000, len(df))
    for j in range(entry_idx + 1, end):
        c_j = float(df.Close.iloc[j])
        if direction == "long":
            if c_j <= sl:   pnl = (sl - entry_p) * size;  break
            elif c_j >= tp: pnl = (tp - entry_p) * size;  break
        else:
            if c_j >= sl:   pnl = (entry_p - sl) * size;  break
            elif c_j <= tp: pnl = (entry_p - tp) * size;  break
    if pnl is None:
        c_last = float(df.Close.iloc[min(end - 1, len(df) - 1)])
        pnl = (c_last - entry_p) * size if direction == "long" \
              else (entry_p - c_last) * size
    return {"reason": reason, "direction": direction,
            "pnl": pnl, "win": bool(pnl > 0), "timestamp": ts}


# ── Sensitivity helper ────────────────────────────────────────────────────────

def _sensitivity_run(df, rrr, swing_lookback):
    """Stripped-down backtest with a specific RRR and swing lookback.
    Recomputes swing stops; all other params use globals.
    Returns condensed metrics dict, or None if no trades."""
    df2 = df.copy()
    df2["s_low"]  = df2["Low"].shift(1).rolling(swing_lookback).min()
    df2["s_high"] = df2["High"].shift(1).rolling(swing_lookback).max()
    df2 = df2.dropna(subset=["s_low", "s_high"]).reset_index(drop=True)

    cash      = STARTING_CASH
    equity_s  = [cash]
    trades_s  = []
    in_trade  = False
    entry_p = sl = tp = size = 0.0
    direction = None

    for i in range(1, len(df2)):
        c    = float(df2["Close"].iloc[i])
        cp   = float(df2["Close"].iloc[i-1])
        en   = float(df2["ema_entry"].iloc[i])
        enp  = float(df2["ema_entry"].iloc[i-1])
        fast = float(df2["ema_fast"].iloc[i])
        slow = float(df2["ema_slow"].iloc[i])
        s_lo = float(df2["s_low"].iloc[i])
        s_hi = float(df2["s_high"].iloc[i])
        ts   = df2["Datetime"].iloc[i]

        if in_trade:
            hit_sl = (direction == "long"  and c <= sl) or \
                     (direction == "short" and c >= sl)
            hit_tp = (direction == "long"  and c >= tp) or \
                     (direction == "short" and c <= tp)
            if hit_sl or hit_tp:
                exit_p = sl if hit_sl else tp
                pnl    = (exit_p - entry_p) * size if direction == "long" \
                         else (entry_p - exit_p) * size
                cash  += pnl
                trades_s.append({"pnl": pnl, "win": pnl > 0})
                in_trade = False

        equity_s.append(cash)

        if not in_trade:
            trend_up   = fast > slow
            trend_down = fast < slow
            long_sig   = trend_up   and cp < enp and c > en and (TRADE_DIRECTION != "short_only")
            short_sig  = trend_down and cp > enp and c < en and (TRADE_DIRECTION != "long_only")

            if TIME_FILTER:
                _ts_u = pd.to_datetime(ts)
                if _ts_u.tzinfo is not None:
                    _ts_u = _ts_u.tz_convert("UTC")
                else:
                    _ts_u = _ts_u.tz_localize("UTC")
                if _ts_u.hour not in TIME_FILTER_HOURS:
                    continue

            if long_sig and not np.isnan(s_lo):
                dist = c - s_lo
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "long";  entry_p = c;  sl = s_lo
                    tp = c + dist * rrr; size = (cash * RISK_PCT) / dist; in_trade = True

            elif short_sig and not np.isnan(s_hi):
                dist = s_hi - c
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "short"; entry_p = c;  sl = s_hi
                    tp = c - dist * rrr; size = (cash * RISK_PCT) / dist; in_trade = True

    if not trades_s:
        return None

    tdf  = pd.DataFrame(trades_s)
    wins = tdf[tdf["win"]]
    loss = tdf[~tdf["win"]]
    net  = float(equity_s[-1]) - STARTING_CASH   # equity curve — consistent with compute_metrics
    wr   = round(len(wins) / len(tdf) * 100, 1)
    pf   = round(abs(float(wins["pnl"].sum()) / float(loss["pnl"].sum())), 2) \
           if not loss.empty and float(loss["pnl"].sum()) != 0 else None
    eq   = pd.Series(equity_s)
    peak = eq.cummax()
    dd   = round(float(((eq - peak) / peak * 100).min()), 2)

    return {"trades": len(tdf), "win_rate": wr,
            "profit_factor": pf, "net_pnl": round(net, 2), "max_drawdown": dd}


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(df):
    cash          = STARTING_CASH
    equity        = [cash]
    trades        = []
    in_trade      = False
    entry_p       = sl = tp = size = 0
    direction     = None
    entry_idx     = 0
    worst_adverse    = 0.0   # tracks furthest adverse price during a trade
    entry_adx        = 0.0   # ADX value at entry bar
    entry_ts         = None  # timestamp of the entry bar (for time-of-day diagnostics)
    blocked_signals  = []    # signals that were filtered out (for Filter Impact Summary)
    _debug_entries   = 0     # counter for entry timezone debug prints
    _daily_loss_day  = None  # current calendar day (UTC date) for daily loss tracking
    _daily_loss_pnl  = 0.0   # cumulative closed-trade P&L for _daily_loss_day

    for i in range(1, len(df)):
        c     = float(df.Close.iloc[i])
        cp    = float(df.Close.iloc[i-1])
        en    = float(df.ema_entry.iloc[i])
        enp   = float(df.ema_entry.iloc[i-1])
        fast  = float(df.ema_fast.iloc[i])
        slow  = float(df.ema_slow.iloc[i])
        s_lo  = float(df.s_low.iloc[i])
        s_hi  = float(df.s_high.iloc[i])
        ts    = df['Datetime'].iloc[i]

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
                # ── Update daily loss tracker on exit ─────────────────────────
                _exit_utc = pd.to_datetime(ts)
                _exit_utc = _exit_utc.tz_convert('UTC') if _exit_utc.tzinfo else _exit_utc.tz_localize('UTC')
                _exit_day = _exit_utc.date()
                if _exit_day != _daily_loss_day:
                    _daily_loss_day = _exit_day
                    _daily_loss_pnl = 0.0
                _daily_loss_pnl += pnl
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx":  i,
                    "direction": direction,
                    "entry":     entry_p,
                    "exit":      exit_p,
                    "stop":      sl,
                    "target":    tp,
                    "size":      size,
                    "pnl":       pnl,
                    "win":       pnl > 0,
                    "result":       "TP" if hit_tp else "SL",
                    "mae":          mae,
                    "adx_at_entry": entry_adx,
                    "timestamp":    ts,
                    "entry_ts":     entry_ts
                })
                in_trade = False

        equity.append(cash)

        # ── Check entries ─────────────────────────────────────────────────────
        if not in_trade:
            trend_up      = fast > slow
            trend_down    = fast < slow
            long_sig_raw  = trend_up   and cp < enp and c > en
            short_sig_raw = trend_down and cp > enp and c < en

            # ── Track direction-blocked signals ────────────────────────────────
            if TRADE_DIRECTION == "short_only" and long_sig_raw and not np.isnan(s_lo):
                dist_b = c - s_lo
                if MIN_STOP <= dist_b <= MAX_STOP:
                    blocked_signals.append(_scan_outcome(
                        df, i, "long", c, s_lo, c + dist_b * RRR,
                        (cash * RISK_PCT) / dist_b, ts, "direction"))
            if TRADE_DIRECTION == "long_only" and short_sig_raw and not np.isnan(s_hi):
                dist_b = s_hi - c
                if MIN_STOP <= dist_b <= MAX_STOP:
                    blocked_signals.append(_scan_outcome(
                        df, i, "short", c, s_hi, c - dist_b * RRR,
                        (cash * RISK_PCT) / dist_b, ts, "direction"))

            # Apply direction filter
            long_sig  = long_sig_raw  and (TRADE_DIRECTION != "short_only")
            short_sig = short_sig_raw and (TRADE_DIRECTION != "long_only")

            # ── Apply time filter and track blocked signals ────────────────────
            if TIME_FILTER:
                _ts_utc    = pd.to_datetime(ts)
                if _ts_utc.tzinfo is not None:
                    _ts_utc = _ts_utc.tz_convert('UTC')
                else:
                    _ts_utc = _ts_utc.tz_localize('UTC')
                entry_hour = _ts_utc.hour
                if entry_hour not in TIME_FILTER_HOURS:
                    if long_sig and not np.isnan(s_lo):
                        dist_b = c - s_lo
                        if MIN_STOP <= dist_b <= MAX_STOP:
                            blocked_signals.append(_scan_outcome(
                                df, i, "long", c, s_lo, c + dist_b * RRR,
                                (cash * RISK_PCT) / dist_b, ts, "time"))
                    if short_sig and not np.isnan(s_hi):
                        dist_b = s_hi - c
                        if MIN_STOP <= dist_b <= MAX_STOP:
                            blocked_signals.append(_scan_outcome(
                                df, i, "short", c, s_hi, c - dist_b * RRR,
                                (cash * RISK_PCT) / dist_b, ts, "time"))
                    long_sig  = False
                    short_sig = False
                    continue

            # ── Daily loss limit ──────────────────────────────────────────────
            _ts_day = pd.to_datetime(ts)
            _ts_day_utc = _ts_day.tz_convert('UTC') if _ts_day.tzinfo else _ts_day.tz_localize('UTC')
            _today = _ts_day_utc.date()
            if _today != _daily_loss_day:
                _daily_loss_day = _today
                _daily_loss_pnl = 0.0
            if _daily_loss_pnl <= -MAX_DAILY_LOSS:
                continue

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
                    entry_ts      = ts
                    worst_adverse = c   # reset MAE tracker to entry price
                    entry_adx     = float(df.adx.iloc[i])
                    if _debug_entries < 5:
                        _ts_dbg = pd.to_datetime(ts)
                        _ts_utc_dbg = _ts_dbg.tz_convert('UTC') if _ts_dbg.tzinfo else _ts_dbg.tz_localize('UTC')
                        print(f"  [DBG entry {_debug_entries+1}] raw={ts}  tz={getattr(ts,'tzinfo',None)}  utc_hour={_ts_utc_dbg.hour}  direction=long")
                        _debug_entries += 1

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
                    entry_ts      = ts
                    worst_adverse = c   # reset MAE tracker to entry price
                    entry_adx     = float(df.adx.iloc[i])
                    if _debug_entries < 5:
                        _ts_dbg = pd.to_datetime(ts)
                        _ts_utc_dbg = _ts_dbg.tz_convert('UTC') if _ts_dbg.tzinfo else _ts_dbg.tz_localize('UTC')
                        print(f"  [DBG entry {_debug_entries+1}] raw={ts}  tz={getattr(ts,'tzinfo',None)}  utc_hour={_ts_utc_dbg.hour}  direction=short")
                        _debug_entries += 1

    return pd.DataFrame(trades), equity, blocked_signals

# ── Results ───────────────────────────────────────────────────────────────────

def print_results(trades, equity):
    if trades.empty:
        print("\n  No trades generated.")
        return

    wins  = trades[trades.win]
    loss  = trades[~trades.win]
    net   = float(equity[-1]) - STARTING_CASH
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
    avg_stop_pips_print = abs(trades["entry"] - trades["stop"]).mean() / 0.0001
    avg_target_pips_print = abs(trades["entry"] - trades["target"]).mean() / 0.0001
    print(f"  Avg Stop       : {avg_stop_pips_print:.1f} pips")
    print(f"  Avg Target     : {avg_target_pips_print:.1f} pips")
    print(f"  Best Trade     : ${trades.pnl.max():.2f}")
    print(f"  Worst Trade    : ${trades.pnl.min():.2f}")
    print(f"{'─'*52}")
    print(f"  Net Profit     : ${net:+.2f} ({net/STARTING_CASH*100:+.1f}%)")
    print(f"  Final Equity   : ${float(equity[-1]):.2f}")
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
    """Save two chart images and return (main_path, rpf_path)."""
    os.makedirs("results", exist_ok=True)
    ticker_clean = TICKER.split("=")[0].replace("^", "")
    date_str     = datetime.now().strftime("%Y-%m-%d")

    def _style_ax(ax):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["top"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["right"].set_color("#444")

    dates = pd.to_datetime(df['Datetime'])

    # ── Main chart: Price / Equity / Drawdown (3 panels) ──────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        _style_ax(ax)

    # Price chart with EMAs
    ax1 = axes[0]
    ax1.plot(dates, df.Close,     color="#e0e0e0", linewidth=0.8, label="Price")
    ax1.plot(dates, df.ema_slow,  color="#ff6b6b", linewidth=1.2, label=f"EMA {EMA_SLOW}")
    ax1.plot(dates, df.ema_fast,  color="#ffd93d", linewidth=1.0, label=f"EMA {EMA_FAST}")
    ax1.plot(dates, df.ema_entry, color="#6bcb77", linewidth=0.8, label=f"EMA {EMA_ENTRY}")

    # ── Range detection shading (subtle grey background when ranging) ──────
    if "regime_ranging" in df.columns:
        ranging_vals = df["regime_ranging"].values
        in_range = False
        range_start = None
        for ri in range(len(ranging_vals)):
            if ranging_vals[ri] and not in_range:
                in_range = True
                range_start = dates.iloc[ri]
            elif not ranging_vals[ri] and in_range:
                in_range = False
                ax1.axvspan(range_start, dates.iloc[ri], alpha=0.08,
                            facecolor="#aaaaaa", edgecolor="none", zorder=0)
        if in_range and range_start is not None:
            ax1.axvspan(range_start, dates.iloc[-1], alpha=0.08,
                        facecolor="#aaaaaa", edgecolor="none", zorder=0)

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

    # Equity curve
    ax2 = axes[1]
    eq_dates  = dates.iloc[:len(equity)] if len(equity) <= len(dates) else dates
    eq_series = pd.Series(equity[:len(eq_dates)])
    ax2.plot(eq_dates[:len(eq_series)], eq_series, color="#4cc9f0", linewidth=1.2)
    ax2.axhline(STARTING_CASH, color="#666", linestyle="--", linewidth=0.8)
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series >= STARTING_CASH,
                     alpha=0.2, color="#6bcb77")
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series < STARTING_CASH,
                     alpha=0.2, color="#ff6b6b")
    ax2.set_ylabel("Equity ($)", color="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Drawdown
    ax3 = axes[2]
    eq_s = pd.Series(equity)
    peak = eq_s.cummax()
    dd   = (eq_s - peak) / peak * 100
    ax3.fill_between(eq_dates[:len(dd)], dd, 0, color="#ff6b6b", alpha=0.6)
    ax3.set_ylabel("Drawdown %", color="white")
    ax3.set_xlabel("Date", color="white")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=2.0)
    main_path = os.path.join("results", f"{VERSION}_{ticker_clean}_{date_str}.png")
    plt.savefig(main_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Main chart saved → {main_path}")

    # ── RPF chart: Rolling Profit Factor (1 panel) ────────────────────────────
    fig2, ax4 = plt.subplots(1, 1, figsize=(16, 4))
    fig2.patch.set_facecolor("#1a1a2e")
    _style_ax(ax4)

    if not trades.empty and len(trades) >= ROLLING_PF_WINDOW:
        rpf_vals  = []
        rpf_dates = []
        for _i in range(ROLLING_PF_WINDOW - 1, len(trades)):
            _win = trades.iloc[_i - ROLLING_PF_WINDOW + 1 : _i + 1]
            _ww  = _win[_win.win]
            _wl  = _win[~_win.win]
            if not _wl.empty and _wl.pnl.sum() != 0:
                _pf = abs(float(_ww.pnl.sum()) / abs(float(_wl.pnl.sum())))
            elif not _ww.empty:
                _pf = 5.0
            else:
                _pf = 0.0
            rpf_vals.append(min(_pf, 5.0))
            rpf_dates.append(pd.to_datetime(trades.iloc[_i]["timestamp"]))

        ax4.plot(rpf_dates, rpf_vals, color="#c77dff", linewidth=1.2,
                 label=f"Rolling PF ({ROLLING_PF_WINDOW} trades)")
        ax4.axhline(1.0, color="#ff6b6b", linestyle="--", linewidth=1.0,
                    label="Break Even (1.0)")
        ax4.axhline(1.3, color="#ffd93d", linestyle="--", linewidth=0.8,
                    label="Full Risk (1.3)")
        ax4.fill_between(rpf_dates, rpf_vals, 1.0,
                         where=[v >= 1.0 for v in rpf_vals],
                         alpha=0.12, color="#6bcb77")
        ax4.fill_between(rpf_dates, rpf_vals, 1.0,
                         where=[v < 1.0 for v in rpf_vals],
                         alpha=0.18, color="#ff6b6b")
        ax4.set_ylim(bottom=0)
        ax4.legend(loc="upper left", facecolor="#1a1a2e",
                   labelcolor="white", fontsize=8)
    else:
        ax4.text(0.5, 0.5, f"Insufficient trades for {ROLLING_PF_WINDOW}-trade rolling window",
                 ha="center", va="center", transform=ax4.transAxes, color="#666", fontsize=10)

    ax4.set_title(f"Rolling Profit Factor ({ROLLING_PF_WINDOW}-trade window)",
                  color="white", fontsize=12, pad=8)
    ax4.set_ylabel(f"Rolling PF ({ROLLING_PF_WINDOW})", color="white")
    ax4.set_xlabel("Date", color="white")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(pad=1.5)
    rpf_path = os.path.join("results", f"{VERSION}_{ticker_clean}_{date_str}_rpf.png")
    plt.savefig(rpf_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  RPF chart saved  → {rpf_path}\n")

    return main_path, rpf_path

# ── HTML Report ───────────────────────────────────────────────────────────────

def compute_metrics(trades, equity, blocked_signals=None, df=None):
    """Compute all backtest metrics and return as a JSON-serialisable dict."""
    if trades.empty:
        return None

    wins = trades[trades.win]
    loss = trades[~trades.win]
    # ── Authoritative net profit: equity curve is the ground truth ─────────────
    # Using equity[-1] - STARTING_CASH (not trades.pnl.sum()) ensures Net Profit
    # reflects exactly what the compounding simulation produced.  Direction P&Ls
    # (sub.pnl.sum()) are each trade's own P&L sized against equity at entry time;
    # the reconciliation step below forces their total to match this figure exactly.
    net  = float(equity[-1]) - STARTING_CASH

    if not loss.empty and loss.pnl.sum() != 0:
        pf = round(abs(float(wins.pnl.sum()) / abs(float(loss.pnl.sum()))), 2)
    else:
        pf = None   # infinite — stored as null, displayed as ∞

    eq        = pd.Series(equity)
    peak      = eq.cummax()
    dd        = float(((eq - peak) / peak * 100).min())
    dd_dollar = round(float((eq - peak).min()), 2)    # peak-to-trough in $
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
            "count":               int(len(sub)),
            "win_rate":            round(len(sub_w) / len(sub) * 100, 1),
            "profit_factor":       sub_pf,
            "avg_win":             round(float(sub_w.pnl.mean()), 2) if not sub_w.empty else None,
            "avg_loss":            round(float(sub_l.pnl.mean()), 2) if not sub_l.empty else None,
            "avg_pnl_per_trade":   round(float(sub.pnl.mean()), 2),
            "pct_of_total_trades": round(len(sub) / n * 100, 1),
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
        # Use entry_ts (entry bar) if available; fall back to timestamp (exit bar)
        _ts_col = "entry_ts" if "entry_ts" in t3.columns else "timestamp"
        _ts_s   = pd.to_datetime(t3[_ts_col])
        if _ts_s.dt.tz is not None:
            _ts_s = _ts_s.dt.tz_convert('UTC')
        else:
            _ts_s = _ts_s.dt.tz_localize('UTC')
        t3["_hour"] = _ts_s.dt.hour
        tod_rows  = []
        best_pnl  = float("-inf")
        worst_pnl = float("inf")
        best_hour = worst_hour = None
        for hour, grp in t3.groupby("_hour"):
            grp_w = grp[grp.win]
            tod_net = round(float(grp.pnl.sum()), 2)
            tod_rows.append({
                "hour":     int(hour),
                "trades":   int(len(grp)),
                "win_rate": round(len(grp_w) / len(grp) * 100, 1),
                "net_pnl":  tod_net,
            })
            if tod_net > best_pnl:
                best_pnl  = tod_net
                best_hour = int(hour)
            if tod_net < worst_pnl:
                worst_pnl  = tod_net
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

    # ── Trade Duration Analysis ────────────────────────────────────────────────
    duration_analysis = {}
    try:
        if "entry_idx" in trades.columns and "exit_idx" in trades.columns:
            td = trades.copy()
            td["_dur"] = (td["exit_idx"] - td["entry_idx"]).astype(float)
            for label, mask in [
                ("winners", td["win"]),
                ("losers",  ~td["win"]),
                ("all",     pd.Series(True, index=td.index)),
            ]:
                sub = td[mask]
                if sub.empty:
                    duration_analysis[label] = {"avg": None, "min": None, "max": None}
                else:
                    duration_analysis[label] = {
                        "avg": round(float(sub["_dur"].mean()), 1),
                        "min": int(sub["_dur"].min()),
                        "max": int(sub["_dur"].max()),
                    }
    except Exception:
        duration_analysis = {}

    # ── Daily drawdown analysis ────────────────────────────────────────────────
    max_daily_drawdown = {"dollar": None, "pct": None}
    daily_drawdown     = []
    try:
        td2 = trades.copy()
        _ts_col = "entry_ts" if "entry_ts" in td2.columns else "timestamp"
        _ts_s   = pd.to_datetime(td2[_ts_col])
        if _ts_s.dt.tz is not None:
            _ts_s = _ts_s.dt.tz_convert("UTC")
        else:
            _ts_s = _ts_s.dt.tz_localize("UTC")
        td2["_date"] = _ts_s.dt.date
        daily = td2.groupby("_date")["pnl"].agg(["sum", "count"]).reset_index()
        daily.columns = ["date", "pnl", "trades"]
        daily = daily.sort_values("pnl")       # worst first
        if not daily.empty:
            worst = daily.iloc[0]
            max_daily_drawdown = {
                "dollar": round(float(worst["pnl"]), 2),
                "pct":    round(float(worst["pnl"]) / STARTING_CASH * 100, 2),
            }
        for _, dr in daily.head(5).iterrows():
            daily_drawdown.append({
                "date":         str(dr["date"]),
                "trades":       int(dr["trades"]),
                "pnl":          round(float(dr["pnl"]), 2),
                "drawdown_pct": round(float(dr["pnl"]) / STARTING_CASH * 100, 2),
            })
    except Exception:
        pass

    # ── Filter Impact Summary ──────────────────────────────────────────────────
    filter_impact = []
    if blocked_signals:
        try:
            bs = pd.DataFrame(blocked_signals)
            for reason in sorted(bs["reason"].unique()):
                sub   = bs[bs["reason"] == reason]
                sub_w = sub[sub["win"]]
                filter_impact.append({
                    "filter":   reason.capitalize() + " Filter",
                    "removed":  int(len(sub)),
                    "win_rate": round(len(sub_w) / len(sub) * 100, 1) if len(sub) > 0 else 0.0,
                    "net_pnl":  round(float(sub["pnl"].sum()), 2),
                })
        except Exception:
            filter_impact = []

    # ── RS Diagnostic (Regime Score development) ────────────────────────────────
    rs_diagnostic = {}
    if blocked_signals:
        try:
            bs = pd.DataFrame(blocked_signals)
            regime_blocked = bs[bs["reason"] == "regime"]
            regime_filtered_count = len(regime_blocked)
            regime_allowed_count  = n   # all executed trades passed the regime filter
            total_signals = regime_filtered_count + regime_allowed_count
            filter_rate = round(regime_filtered_count / total_signals * 100, 1) if total_signals > 0 else 0.0

            # Stats for filtered (blocked) signals
            filtered_wins = regime_blocked[regime_blocked["win"]] if not regime_blocked.empty else pd.DataFrame()
            filtered_losses = regime_blocked[~regime_blocked["win"]] if not regime_blocked.empty else pd.DataFrame()
            filtered_wr = round(len(filtered_wins) / len(regime_blocked) * 100, 1) if len(regime_blocked) > 0 else 0.0
            filtered_pf = None
            if not filtered_losses.empty and filtered_losses["pnl"].sum() != 0:
                filtered_pf = round(abs(float(filtered_wins["pnl"].sum()) / abs(float(filtered_losses["pnl"].sum()))), 2)

            # Stats for allowed (executed) trades
            allowed_wr = round(len(wins) / n * 100, 1) if n > 0 else 0.0
            allowed_pf = pf   # already computed above

            rs_diagnostic = {
                "trades_filtered":  regime_filtered_count,
                "trades_allowed":   regime_allowed_count,
                "total_signals":    total_signals,
                "filter_rate":      filter_rate,
                "filtered_wr":      filtered_wr,
                "filtered_pf":      filtered_pf,
                "allowed_wr":       allowed_wr,
                "allowed_pf":       allowed_pf,
            }
        except Exception:
            rs_diagnostic = {}

    # ── Rolling Profit Factor ──────────────────────────────────────────────────
    rolling_pf_stats = {
        "window": ROLLING_PF_WINDOW,
        "current": None, "min": None, "max": None,
        "pct_above_1_3": None, "pct_1_0_to_1_3": None, "pct_below_1_0": None,
    }
    if n >= ROLLING_PF_WINDOW:
        try:
            rpf_vals = []
            for _i in range(ROLLING_PF_WINDOW - 1, n):
                _win = trades.iloc[_i - ROLLING_PF_WINDOW + 1 : _i + 1]
                _ww  = _win[_win.win]
                _wl  = _win[~_win.win]
                if not _wl.empty and _wl.pnl.sum() != 0:
                    _pf = abs(float(_ww.pnl.sum()) / abs(float(_wl.pnl.sum())))
                elif not _ww.empty:
                    _pf = 99.0   # cap infinity for storage
                else:
                    _pf = 0.0
                rpf_vals.append(_pf)
            _total = len(rpf_vals)
            # ── Look-ahead-free tier classification ────────────────────────────
            # For each trade j >= N, classify by PF of the PRECEDING N trades
            # (no look-ahead: trade j itself is NOT in the window that classifies it)
            _tier_full    = {"name": "Full Risk",    "threshold": "\u2265 1.3",   "count": 0, "net_pnl": 0.0}
            _tier_reduced = {"name": "Reduced Risk", "threshold": "1.0\u20131.3", "count": 0, "net_pnl": 0.0}
            _tier_minimum = {"name": "Minimum Risk", "threshold": "< 1.0",        "count": 0, "net_pnl": 0.0}
            for _j in range(ROLLING_PF_WINDOW, n):
                _prev = trades.iloc[_j - ROLLING_PF_WINDOW : _j]
                _pw   = _prev[_prev.win]
                _pl   = _prev[~_prev.win]
                if not _pl.empty and _pl.pnl.sum() != 0:
                    _ppf = abs(float(_pw.pnl.sum()) / abs(float(_pl.pnl.sum())))
                elif not _pw.empty:
                    _ppf = 99.0
                else:
                    _ppf = 0.0
                _tpnl = float(trades.iloc[_j]["pnl"])
                if _ppf >= 1.3:
                    _tier_full["count"]    += 1
                    _tier_full["net_pnl"]  += _tpnl
                elif _ppf >= 1.0:
                    _tier_reduced["count"]   += 1
                    _tier_reduced["net_pnl"] += _tpnl
                else:
                    _tier_minimum["count"]   += 1
                    _tier_minimum["net_pnl"] += _tpnl
            _n_class = _tier_full["count"] + _tier_reduced["count"] + _tier_minimum["count"]
            _tiers = []
            for _t in [_tier_full, _tier_reduced, _tier_minimum]:
                _tiers.append({
                    "name":      _t["name"],
                    "threshold": _t["threshold"],
                    "pct_time":  round(_t["count"] / _n_class * 100, 1) if _n_class > 0 else 0.0,
                    "trades":    _t["count"],
                    "net_pnl":   round(_t["net_pnl"], 2),
                })
            rolling_pf_stats = {
                "window":          ROLLING_PF_WINDOW,
                "current":         round(rpf_vals[-1], 2),
                "min":             round(min(rpf_vals), 2),
                "max":             round(min(max(rpf_vals), 99.0), 2),
                "pct_above_1_3":   round(sum(1 for v in rpf_vals if v >= 1.3)  / _total * 100, 1),
                "pct_1_0_to_1_3":  round(sum(1 for v in rpf_vals if 1.0 <= v < 1.3) / _total * 100, 1),
                "pct_below_1_0":   round(sum(1 for v in rpf_vals if v < 1.0)   / _total * 100, 1),
                "tiers":           _tiers,
            }
        except Exception:
            pass

    # ── Position size metrics ──────────────────────────────────────────────────
    avg_pos_size = None
    min_pos_size = None
    max_pos_size = None
    if "size" in trades.columns:
        avg_pos_size = round(float(trades["size"].mean()), 0)
        min_pos_size = round(float(trades["size"].min()), 0)
        max_pos_size = round(float(trades["size"].max()), 0)

    result = {
        "total_trades":   n,
        "winning_trades": int(len(wins)),
        "losing_trades":  int(len(loss)),
        "win_rate":       round(len(wins) / n * 100, 1),
        "profit_factor":  pf,
        "avg_win":        round(float(wins.pnl.mean()), 2) if not wins.empty else None,
        "avg_loss":       round(float(loss.pnl.mean()), 2) if not loss.empty else None,
        "avg_stop_pips":  round(float(abs(trades["entry"] - trades["stop"]).mean() / 0.0001), 1),
        "avg_target_pips": round(float(abs(trades["entry"] - trades["target"]).mean() / 0.0001), 1),
        "best_trade":     round(float(trades.pnl.max()), 2),
        "worst_trade":    round(float(trades.pnl.min()), 2),
        "net_profit":          round(net, 2),
        "net_profit_pct":      round(net / STARTING_CASH * 100, 2),
        "final_equity":        round(float(equity[-1]), 2),  # equity curve; equals STARTING_CASH + net
        "max_drawdown":        round(dd, 2),
        "max_drawdown_dollar": dd_dollar,
        "max_daily_drawdown":  max_daily_drawdown,
        "daily_drawdown":      daily_drawdown,
        "sharpe":         round(sharpe, 2),
        "avg_position_size": avg_pos_size,
        "min_position_size": min_pos_size,
        "max_position_size": max_pos_size,
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
        "duration_analysis": duration_analysis,
        "filter_impact":     filter_impact,
        "rolling_pf":        rolling_pf_stats,
        "rs_diagnostic":     rs_diagnostic,
        "rrr_sensitivity":   [],
        "swing_sensitivity": [],
    }

    # ── Sensitivity sweeps (requires original df) ──────────────────────────────
    if df is not None:
        try:
            rrr_rows, swing_rows = compute_sensitivity(df)
            result["rrr_sensitivity"]   = rrr_rows
            result["swing_sensitivity"] = swing_rows
        except Exception:
            pass

    return result


def compute_sensitivity(df):
    """Sweep RRR and swing lookback; return (rrr_rows, swing_rows) lists."""
    rrr_rows = []
    for val in [1.5, 2.0, 2.5, 3.0]:
        r = _sensitivity_run(df, rrr=val, swing_lookback=SWING_LOOKBACK)
        if r:
            r["param"] = val
            rrr_rows.append(r)

    swing_rows = []
    for val in [10, 15, 20, 25, 30]:
        r = _sensitivity_run(df, rrr=RRR, swing_lookback=val)
        if r:
            r["param"] = val
            swing_rows.append(r)

    return rrr_rows, swing_rows


def _build_html(versions_json):
    """Return the complete self-contained HTML string."""
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backtest Report</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
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
      <span class="empty-icon">&#128202;</span>
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
  var expandedVersions = {};  /* track which versions are expanded in sidebar */
  var activeVersionIdx = -1;
  var activeRunIdx     = 0;

  /* ── Helpers ──────────────────────────────────────────────── */
  function fmt(n, d) {
    if (n === null || n === undefined || (typeof n === "number" && isNaN(n))) return "&#8212;";
    return Number(n).toFixed(d !== undefined ? d : 2);
  }

  function commaFmt(n) {
    var parts = Math.abs(Number(n)).toFixed(2).split(".");
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    return parts.join(".");
  }

  function fmtMoney(n) {
    if (n === null || n === undefined) return "&#8212;";
    var s = n >= 0 ? "+" : "";
    return s + "$" + commaFmt(n);
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

  /* ── Compat: get run data from version (handles legacy + new format) ──── */
  function getRuns(v) {
    if (v.runs && v.runs.length > 0) return v.runs;
    /* legacy single-run format */
    return [{
      date: v.date || "", start_date: "", end_date: "",
      days_back: (v.params || {}).days_back || 0, label: "",
      notes: v.notes || "\u2014", chart_b64: v.chart_b64 || "",
      rpf_chart_b64: v.rpf_chart_b64 || "",
      metrics: v.metrics || {}, last_trades: v.last_trades || []
    }];
  }

  /* ── Toast ──────────────────────────────────────────────── */
  function showToast(msg) {
    var toast = document.getElementById("copy-toast");
    if (msg) toast.innerHTML = "&#10003;&nbsp; " + msg;
    toast.classList.add("show");
    setTimeout(function () { toast.classList.remove("show"); }, 2200);
  }

  /* ── Markdown builder (per run) ────────────────────────────── */
  function buildRunMarkdown(ver, run) {
    var m = run.metrics || {};
    var p = ver.params  || {};

    function mf(n, d) {
      if (n === null || n === undefined || (typeof n === "number" && isNaN(n))) return "\u2014";
      return Number(n).toFixed(d !== undefined ? d : 2);
    }
    function mfMoney(n) {
      if (n === null || n === undefined) return "\u2014";
      var f = Number(n);
      return (f >= 0 ? "+" : "-") + "$" + commaFmt(f);
    }

    var pf    = (m.profit_factor === null || m.profit_factor === undefined) ? "\u221e" : mf(m.profit_factor);
    var npStr = mfMoney(m.net_profit);
    if (m.net_profit_pct !== null && m.net_profit_pct !== undefined) {
      npStr += " (" + (m.net_profit_pct >= 0 ? "+" : "") + mf(m.net_profit_pct, 1) + "%)";
    }

    var lines = [];
    var runLabel = run.label ? " \u2014 " + run.label : "";
    lines.push("## Backtest Report \u2014 " + (ver.name || "?") + runLabel);
    lines.push("");
    lines.push("**Strategy:** " + (ver.strategy || "\u2014"));
    lines.push("**Instrument:** " + (p.ticker || "\u2014") + " \u00b7 " + (p.interval || "\u2014") + " \u00b7 " + (run.days_back || p.days_back || "\u2014") + " days");
    lines.push("**Date:** " + (run.date || "\u2014"));
    if (run.start_date && run.end_date) {
      lines.push("**Range:** " + run.start_date + " \u2192 " + run.end_date);
    }
    if (run.notes && run.notes !== "\u2014" && run.notes !== "\u2014") {
      lines.push("**Notes:** " + run.notes);
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
    lines.push("| Final Equity | $"   + commaFmt(m.final_equity) + " |");
    var mdDollar = m.max_drawdown_dollar;
    var mdStr    = (mdDollar !== null && mdDollar !== undefined)
      ? "-$" + commaFmt(mdDollar) + " (" + mf(m.max_drawdown) + "%)"
      : mf(m.max_drawdown) + "%";
    lines.push("| Max Drawdown | "    + mdStr + " |");
    var mddMd = m.max_daily_drawdown || {};
    var mddStr = (mddMd.dollar !== null && mddMd.dollar !== undefined)
      ? "-$" + commaFmt(mddMd.dollar) + " (" + mf(mddMd.pct, 2) + "%)"
      : "\u2014";
    lines.push("| Max Daily DD | "   + mddStr + " |");
    lines.push("| Sharpe Ratio | "    + mf(m.sharpe) + " |");
    lines.push("| Avg Win | "         + (m.avg_win  !== null && m.avg_win  !== undefined ? "$"  + commaFmt(m.avg_win)  : "\u2014") + " |");
    lines.push("| Avg Loss | "        + (m.avg_loss !== null && m.avg_loss !== undefined ? "$"  + commaFmt(m.avg_loss) : "\u2014") + " |");
    lines.push("| Best Trade | "      + mfMoney(m.best_trade) + " |");
    lines.push("| Worst Trade | "     + mfMoney(m.worst_trade) + " |");
    lines.push("");

    lines.push("### Parameters");
    lines.push("");
    lines.push("| Parameter | Value |");
    lines.push("|-----------|-------|");
    lines.push("| Instrument | "    + (p.ticker    || "\u2014") + " |");
    lines.push("| Interval | "      + (p.interval  || "\u2014") + " |");
    lines.push("| History | "       + (run.days_back || p.days_back || "\u2014") + " days |");
    lines.push("| Starting Cash | $" + (p.starting_cash || 0).toLocaleString() + " |");
    lines.push("| EMA Slow | "      + (p.ema_slow      || "\u2014") + " |");
    lines.push("| EMA Fast | "      + (p.ema_fast      || "\u2014") + " |");
    lines.push("| EMA Entry | "     + (p.ema_entry     || "\u2014") + " |");
    lines.push("| Swing Lookback | " + (p.swing_lookback || "\u2014") + " bars |");
    lines.push("| RRR | 1:"         + (p.rrr || "\u2014") + " |");
    lines.push("| Risk / Trade | "  + ((p.risk_pct || 0) * 100).toFixed(1) + "% |");
    lines.push("| Direction | "     + (p.trade_direction || "both") + " |");
    var tfHours = (p.time_filter_hours || []).join(", ");
    lines.push("| Time Filter | "   + (p.time_filter ? "ON \u2014 hours " + tfHours : "OFF") + " |");
    lines.push("");

    /* ── Performance by Direction ──────────────────── */
    lines.push("### Performance by Direction");
    lines.push("");
    lines.push("| Direction | Trades | Win Rate | Profit Factor | Net P&L |");
    lines.push("|-----------|--------|----------|---------------|---------|");
    var bd = m.by_direction || {};
    ["long", "short"].forEach(function (d) {
      var data = bd[d];
      if (!data) { lines.push("| " + d.charAt(0).toUpperCase() + d.slice(1) + " | \u2014 | \u2014 | \u2014 | \u2014 |"); return; }
      var dpf = (data.profit_factor === null || data.profit_factor === undefined) ? "\u221e" : mf(data.profit_factor);
      lines.push("| " + d.charAt(0).toUpperCase() + d.slice(1) + " | " + data.count + " | " + mf(data.win_rate, 1) + "% | " + dpf + " | " + mfMoney(data.avg_pnl_per_trade) + " |");
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
        lines.push("| " + mLabel + " | " + mo.trades + " | " + mo.wins + " | " + mo.losses + " | " + mf(mo.win_rate, 1) + "% | " + mfMoney(mo.net_pnl) + " |");
      });
    }
    lines.push("");

    return lines.join("\n");
  }

  /* ── Markdown builder (full version = all runs) ───────────── */
  function buildVersionMarkdown(ver) {
    var runs = getRuns(ver);
    var parts = [];
    for (var i = 0; i < runs.length; i++) {
      parts.push(buildRunMarkdown(ver, runs[i]));
    }
    return parts.join("\n---\n\n");
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

  /* ── Sidebar helpers ──────────────────────────────────────── */
  function fmtSbDate(s) {
    if (!s) return "";
    var p = s.slice(0, 10).split("-");
    return parseInt(p[1]) + "-" + p[2] + "-" + p[0].slice(2);
  }
  function calcDuration(startStr, endStr) {
    if (!startStr || !endStr) return "";
    var s = new Date(startStr.slice(0,10) + "T00:00:00");
    var e = new Date(endStr.slice(0,10) + "T00:00:00");
    var days = Math.round((e - s) / 86400000);
    if (days < 0) days = 0;
    var months = days / 30.44;
    if (months < 2) {
      var weeks = Math.round(days / 7);
      return weeks + (weeks === 1 ? " week" : " weeks");
    } else if (months <= 18) {
      return months.toFixed(1) + " months";
    } else {
      return (months / 12).toFixed(1) + " years";
    }
  }
  function fullRunRange(run) {
    var endStr = run.date ? run.date.slice(0,10) : "";
    var daysBack = run.days_back || 730;
    if (!endStr) return { start: "", end: "" };
    var endDt = new Date(endStr + "T00:00:00");
    var startDt = new Date(endDt.getTime() - daysBack * 86400000);
    var sy = startDt.getFullYear(), sm = String(startDt.getMonth()+1).padStart(2,"0"), sd = String(startDt.getDate()).padStart(2,"0");
    return { start: sy + "-" + sm + "-" + sd, end: endStr };
  }

  /* ── Sidebar ──────────────────────────────────────────────── */
  function renderSidebar() {
    var svs  = getStrategyVersions();
    var list = document.getElementById("version-list");
    list.innerHTML = "";

    if (svs.length === 0) {
      list.innerHTML = "<div class='sb-no-runs'>No runs for this strategy yet.</div>";
      return;
    }

    /* render newest first */
    for (var ri = svs.length - 1; ri >= 0; ri--) {
      var entry = svs[ri];
      var v     = entry.v;
      var idx   = entry.idx;
      var runs  = getRuns(v);
      var firstRun = runs[0] || {};
      var pnl  = firstRun.metrics ? firstRun.metrics.net_profit : null;
      var pc   = pnl === null ? "" : (pnl >= 0 ? "pos" : "neg");
      var ptxt = pnl === null ? "" : (pnl >= 0 ? "+" : "") + "$" + commaFmt(pnl);

      /* version header row */
      var vItem = document.createElement("div");
      vItem.className = "v-item" + (idx === activeVersionIdx && activeRunIdx === 0 ? " active" : "");
      vItem.dataset.idx = idx;

      var hasSubRuns = runs.length > 1;

      var arrowHtml = hasSubRuns
        ? "<span class='v-expand-arrow" + (expandedVersions[idx] ? " expanded" : "") + "' data-idx='" + idx + "'>\u25B6</span>"
        : "";

      var vRange = firstRun.start_date && firstRun.end_date
        ? { start: firstRun.start_date, end: firstRun.end_date }
        : fullRunRange(firstRun);
      var vDateRange = vRange.start && vRange.end
        ? fmtSbDate(vRange.start) + " \u2192 " + fmtSbDate(vRange.end) : "";
      var vDur = calcDuration(vRange.start, vRange.end);

      vItem.innerHTML =
        "<div class='v-item-row'>" +
          "<div class='v-item-content'>" +
            "<div class='v-name'>" + esc(v.name) + "</div>" +
            (pnl !== null ? "<div class='v-pnl " + pc + "'>" + ptxt + "</div>" : "") +
            (vDateRange ? "<div class='v-date'>" + esc(vDateRange) + "</div>" : "") +
            (vDur ? "<div class='v-duration'>" + esc(vDur) + "</div>" : "") +
          "</div>" +
          arrowHtml +
        "</div>";

      /* Clicking the version row navigates to the full run report */
      (function (el, vIdx) {
        el.addEventListener("click", function (e) {
          /* ignore clicks on the expand arrow */
          if (e.target.classList.contains("v-expand-arrow")) return;
          devLogOpen = false;
          document.getElementById("devlog-btn").classList.remove("active");
          activeVersionIdx = vIdx;
          activeRunIdx = 0;
          renderSidebar();
          renderContent(vIdx, 0);
        });
      })(vItem, idx);

      /* Clicking the arrow icon toggles expand/collapse independently */
      if (hasSubRuns) {
        var arrowEl = vItem.querySelector(".v-expand-arrow");
        if (arrowEl) {
          (function (arrow, vIdx) {
            arrow.addEventListener("click", function (e) {
              e.stopPropagation();
              expandedVersions[vIdx] = !expandedVersions[vIdx];
              renderSidebar();
            });
          })(arrowEl, idx);
        }
      }

      list.appendChild(vItem);

      /* ── Sub-runs (expanded) — skip index 0 (full run, already the version row) */
      if (expandedVersions[idx] && runs.length > 1) {
        for (var si = 1; si < runs.length; si++) {
          var run = runs[si];
          var runPnl = run.metrics ? run.metrics.net_profit : null;
          var runPc  = runPnl === null ? "" : (runPnl >= 0 ? "pos" : "neg");
          var runPtxt = runPnl === null ? "" : (runPnl >= 0 ? "+" : "") + "$" + commaFmt(runPnl);
          var runLabel = run.label || ("Run " + si);

          var subItem = document.createElement("div");
          subItem.className = "v-item v-sub-item" +
            (idx === activeVersionIdx && si === activeRunIdx ? " active" : "");
          subItem.dataset.idx = idx;
          subItem.dataset.runIdx = si;

          var subRange = run.start_date && run.end_date
            ? { start: run.start_date, end: run.end_date }
            : fullRunRange(run);
          var subDateRange = subRange.start && subRange.end
            ? fmtSbDate(subRange.start) + " \u2192 " + fmtSbDate(subRange.end) : "";
          var subDur = calcDuration(subRange.start, subRange.end);

          subItem.innerHTML =
            (runPnl !== null ? "<div class='v-pnl " + runPc + "'>" + runPtxt + "</div>" : "") +
            (subDateRange ? "<div class='v-date v-sub-name'>" + esc(subDateRange) + "</div>" : "") +
            (subDur ? "<div class='v-duration'>" + esc(subDur) + "</div>" : "");

          (function (el, vIdx, rIdx) {
            el.addEventListener("click", function (e) {
              e.stopPropagation();
              devLogOpen = false;
              document.getElementById("devlog-btn").classList.remove("active");
              activeVersionIdx = vIdx;
              activeRunIdx = rIdx;
              renderSidebar();
              renderContent(vIdx, rIdx);
            });
          })(subItem, idx, si);

          list.appendChild(subItem);
        }
      }
    }
  }

  /* ── Content ──────────────────────────────────────────────── */
  function renderContent(vIdx, runIdx) {
    var v = VERSIONS[vIdx];
    if (!v) return;

    var runs = getRuns(v);
    var run  = runs[runIdx] || runs[0];
    var m    = run.metrics || {};
    var p    = v.params  || {};

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
               "<td colspan='7' class='text-dim'>No data</td></tr>";
      }
      var dpf    = (data.profit_factor === null || data.profit_factor === undefined) ? "\u221e" : fmt(data.profit_factor);
      var dpfCls = (data.profit_factor === null || data.profit_factor === undefined || data.profit_factor >= 1.5) ? "pos" : (data.profit_factor < 1.0 ? "neg" : "neu");
      return "<tr>" +
        "<td><strong>" + label + "</strong></td>" +
        "<td>" + data.count + "</td>" +
        "<td>" + fmt(data.pct_of_total_trades, 1) + "%</td>" +
        "<td class='" + (data.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(data.win_rate, 1) + "%</td>" +
        "<td class='" + dpfCls + "'>" + dpf + "</td>" +
        "<td class='pos'>" + (data.avg_win  !== null && data.avg_win  !== undefined ? "$" + commaFmt(data.avg_win)  : "\u2014") + "</td>" +
        "<td class='neg'>" + (data.avg_loss !== null && data.avg_loss !== undefined ? "$" + commaFmt(Math.abs(data.avg_loss)) : "\u2014") + "</td>" +
        "<td class='" + pClass(data.avg_pnl_per_trade) + "'>" + fmtMoney(data.avg_pnl_per_trade) + "</td>" +
        "</tr>";
    }
    var dirHtml =
      "<div class='section'>" +
        "<div class='section-title'>Performance by Direction</div>" +
        "<table><thead><tr>" +
        "<th>Direction</th><th>Trades</th><th>% of Trades</th><th>Win Rate</th>" +
        "<th>Profit Factor</th><th>Avg Win</th><th>Avg Loss</th><th>Avg P&amp;L/Trade</th>" +
        "</tr></thead><tbody>" +
        dirRow("Long", bdLng) + dirRow("Short", bdSht) +
        "</tbody></table>" +
      "</div>";

    /* 2. Monthly Performance */
    var monthly = m.monthly || [];
    var mRows = "";
    monthly.forEach(function (mo) {
      var mPnlCls = mo.net_pnl >= 0 ? "mo-pnl-pos" : "mo-pnl-neg";
      mRows +=
        "<tr>" +
        "<td>" + fmtMonth(mo.month) + "</td>" +
        "<td>" + mo.trades + "</td>" +
        "<td class='pos'>" + mo.wins + "</td>" +
        "<td class='neg'>" + mo.losses + "</td>" +
        "<td class='" + (mo.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(mo.win_rate, 1) + "%</td>" +
        "<td class='" + mPnlCls + "'>" + fmtMoney(mo.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!mRows) {
      mRows = "<tr><td colspan='6' class='td-empty'>No data</td></tr>";
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
    if (!regRows) regRows = "<tr><td colspan='5' class='td-empty'>No data</td></tr>";
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
      var rowCls  = isBest  ? " class='tod-row-best'"  :
                    isWorst ? " class='tod-row-worst'" : "";
      var suffix  = isBest  ? " <span class='badge-pos'>\u2605 best</span>"  :
                    isWorst ? " <span class='badge-neg'>\u25bc worst</span>" : "";
      var hStr    = (r.hour < 10 ? "0" : "") + r.hour + ":00 UTC";
      todRows +=
        "<tr" + rowCls + ">" +
        "<td>" + hStr + suffix + "</td>" +
        "<td>" + r.trades + "</td>" +
        "<td class='" + (r.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(r.win_rate, 1) + "%</td>" +
        "<td class='" + pClass(r.net_pnl) + "'>" + fmtMoney(r.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!todRows) todRows = "<tr><td colspan='4' class='td-empty'>No data</td></tr>";
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
        "<td><strong class='seg-label'>" + (labels[idx] || ("Seg " + (idx + 1))) + "</strong><br>" +
        "<span class='seg-dates'>" + esc(seg.period) + "</span></td>" +
        "<td>" + seg.trades + "</td>" +
        "<td class='" + (seg.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(seg.win_rate, 1) + "%</td>" +
        "<td class='" + spfCls + "'>" + spf + "</td>" +
        "<td class='" + pClass(seg.net_pnl) + "'>" + fmtMoney(seg.net_pnl) + "</td>" +
        "</tr>";
    });
    if (!wrtRows) wrtRows = "<tr><td colspan='5' class='td-empty'>No data</td></tr>";
    var winRateTrendHtml =
      "<div class='section'>" +
        "<div class='section-title'>Win Rate Trend (3 equal periods)</div>" +
        "<table><thead><tr>" +
        "<th>Period</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + wrtRows + "</tbody></table></div>";

    /* 8. Trade Duration Analysis */
    var dur     = m.duration_analysis || {};
    var durRows = "";
    [["Winners", dur.winners], ["Losers", dur.losers], ["All Trades", dur.all]].forEach(function (pair) {
      var label = pair[0], d = pair[1];
      if (!d || d.avg === null) {
        durRows += "<tr><td><strong>" + label + "</strong></td><td colspan='3' class='text-dim'>No data</td></tr>";
        return;
      }
      durRows +=
        "<tr>" +
        "<td><strong>" + label + "</strong></td>" +
        "<td>" + d.avg + " bars</td>" +
        "<td>" + d.min + " bars</td>" +
        "<td>" + d.max + " bars</td>" +
        "</tr>";
    });
    var durationHtml =
      "<div class='section'>" +
        "<div class='section-title'>Trade Duration (bars = hours on 1h)</div>" +
        "<table><thead><tr>" +
        "<th>Group</th><th>Avg</th><th>Min</th><th>Max</th>" +
        "</tr></thead><tbody>" + durRows + "</tbody></table></div>";

    /* 9. Filter Impact Summary */
    var fi     = m.filter_impact || [];
    var fiRows = "";
    fi.forEach(function (f) {
      var noteCol;
      if (f.net_pnl < 0) {
        noteCol = "<span class='badge-pos'>\u2713 filter saved $" + commaFmt(-f.net_pnl) + "</span>";
      } else if (f.net_pnl > 0) {
        noteCol = "<span class='badge-neg'>\u26a0 blocked +$" + commaFmt(f.net_pnl) + "</span>";
      } else {
        noteCol = "<span class='badge-neu'>neutral</span>";
      }
      fiRows +=
        "<tr>" +
        "<td><strong>" + esc(f.filter) + "</strong></td>" +
        "<td>" + f.removed + "</td>" +
        "<td class='" + (f.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(f.win_rate, 1) + "%</td>" +
        "<td class='" + pClass(f.net_pnl) + "'>" + fmtMoney(f.net_pnl) + " " + noteCol + "</td>" +
        "</tr>";
    });
    if (!fiRows) fiRows = "<tr><td colspan='4' class='td-empty'>No filters active or no signals blocked</td></tr>";
    var filterImpactHtml =
      "<div class='section'>" +
        "<div class='section-title'>Filter Impact Summary</div>" +
        "<table><thead><tr>" +
        "<th>Filter</th><th>Trades Removed</th><th>Win Rate (if kept)</th><th>Net P&amp;L (if kept)</th>" +
        "</tr></thead><tbody>" + fiRows + "</tbody></table></div>";

    /* 9b. Range Filter development */
    var rsd    = m.rs_diagnostic || {};
    var rsdHtml = "";
    if (rsd.total_signals && rsd.total_signals > 0) {
      var filteredPfTxt = (rsd.filtered_pf === null || rsd.filtered_pf === undefined)
        ? "&#8734;" : fmt(rsd.filtered_pf);
      var allowedPfTxt  = (rsd.allowed_pf === null || rsd.allowed_pf === undefined)
        ? "&#8734;" : fmt(rsd.allowed_pf);
      var filteredPfCls = (rsd.filtered_pf === null || rsd.filtered_pf === undefined || rsd.filtered_pf >= 1.5)
        ? "pos" : (rsd.filtered_pf < 1.0 ? "neg" : "neu");
      var allowedPfCls  = (rsd.allowed_pf === null || rsd.allowed_pf === undefined || rsd.allowed_pf >= 1.5)
        ? "pos" : (rsd.allowed_pf < 1.0 ? "neg" : "neu");
      rsdHtml =
        "<div class='section'>" +
          "<div class='section-title'>Range Filter</div>" +
          "<table><thead><tr>" +
          "<th>Metric</th><th>Filtered (Blocked)</th><th>Allowed (Executed)</th>" +
          "</tr></thead><tbody>" +
          "<tr><td>Trades</td><td>" + rsd.trades_filtered + "</td><td>" + rsd.trades_allowed + "</td></tr>" +
          "<tr><td>Win Rate</td>" +
            "<td class='" + (rsd.filtered_wr >= 50 ? "pos" : "neg") + "'>" + fmt(rsd.filtered_wr, 1) + "%</td>" +
            "<td class='" + (rsd.allowed_wr >= 50 ? "pos" : "neg") + "'>" + fmt(rsd.allowed_wr, 1) + "%</td></tr>" +
          "<tr><td>Profit Factor</td>" +
            "<td class='" + filteredPfCls + "'>" + filteredPfTxt + "</td>" +
            "<td class='" + allowedPfCls + "'>" + allowedPfTxt + "</td></tr>" +
          "<tr><td colspan='3'><strong>Filter Rate: " + fmt(rsd.filter_rate, 1) + "%</strong> (" +
            rsd.trades_filtered + " of " + rsd.total_signals + " signals blocked)</td></tr>" +
          "</tbody></table></div>";
    }

    /* 10. Daily Drawdown — worst 5 days */
    var ddDays     = m.daily_drawdown || [];
    var ddDayRows  = "";
    ddDays.forEach(function (d) {
      ddDayRows +=
        "<tr>" +
        "<td class='nowrap'>" + esc(d.date) + "</td>" +
        "<td>" + d.trades + "</td>" +
        "<td class='neg'>" + "-$" + commaFmt(d.pnl) + "</td>" +
        "<td class='neg'>" + fmt(d.drawdown_pct, 2) + "%</td>" +
        "</tr>";
    });
    if (!ddDayRows) ddDayRows = "<tr><td colspan='4' class='td-empty-sm'>No data</td></tr>";
    var dailyDDHtml =
      "<div class='section'>" +
        "<div class='section-title'>Daily Drawdown \u2014 Worst 5 Days</div>" +
        "<table><thead><tr>" +
        "<th>Date (UTC)</th><th>Trades</th><th>P&amp;L</th><th>Drawdown %</th>" +
        "</tr></thead><tbody>" + ddDayRows + "</tbody></table></div>";

    /* 11. Rolling Profit Factor */
    var rpf     = m.rolling_pf || {};
    var rpfWin  = rpf.window || 10;
    var rpfCur  = rpf.current !== null && rpf.current !== undefined ? fmt(rpf.current) : "&#8212;";
    var rpfMin  = rpf.min     !== null && rpf.min     !== undefined ? fmt(rpf.min)     : "&#8212;";
    var rpfMax  = rpf.max     !== null && rpf.max     !== undefined ? fmt(rpf.max)     : "&#8212;";
    var rpfCurClass = (rpf.current !== null && rpf.current !== undefined)
      ? (rpf.current >= 1.3 ? "pos" : (rpf.current < 1.0 ? "neg" : "neu")) : "";
    var rpfTierRows = "";
    var rpfTiers = rpf.tiers || [];
    if (rpfTiers.length === 0) {
      rpfTierRows =
        "<tr><td><span class='pos'>&#9679; Full Risk</span></td><td>&ge; 1.3</td>" +
        "<td><span class='pos'>" + (rpf.pct_above_1_3  !== null && rpf.pct_above_1_3  !== undefined ? fmt(rpf.pct_above_1_3, 1)  + "%" : "&#8212;") + "</span></td>" +
        "<td>&#8212;</td><td>&#8212;</td></tr>" +
        "<tr><td><span class='neu'>&#9679; Reduced Risk</span></td><td>1.0 &ndash; 1.3</td>" +
        "<td><span class='neu'>" + (rpf.pct_1_0_to_1_3 !== null && rpf.pct_1_0_to_1_3 !== undefined ? fmt(rpf.pct_1_0_to_1_3, 1) + "%" : "&#8212;") + "</span></td>" +
        "<td>&#8212;</td><td>&#8212;</td></tr>" +
        "<tr><td><span class='neg'>&#9679; Minimum Risk</span></td><td>&lt; 1.0</td>" +
        "<td><span class='neg'>" + (rpf.pct_below_1_0  !== null && rpf.pct_below_1_0  !== undefined ? fmt(rpf.pct_below_1_0, 1)  + "%" : "&#8212;") + "</span></td>" +
        "<td>&#8212;</td><td>&#8212;</td></tr>";
    } else {
      var tierClasses = ["pos", "neu", "neg"];
      rpfTiers.forEach(function (t, i) {
        var cls  = tierClasses[i] || "";
        var pnlC = pClass(t.net_pnl);
        rpfTierRows +=
          "<tr>" +
          "<td><span class='" + cls + "'>&#9679; " + esc(t.name) + "</span></td>" +
          "<td>" + esc(t.threshold) + "</td>" +
          "<td><span class='" + cls + "'>" + fmt(t.pct_time, 1) + "%</span></td>" +
          "<td>" + t.trades + "</td>" +
          "<td class='" + pnlC + "'>" + fmtMoney(t.net_pnl) + "</td>" +
          "</tr>";
      });
    }
    var rpfNoteHtml = rpfTiers.length > 0
      ? "<div class='rpf-note'>" +
          "Tier classification is look-ahead-free: each trade is classified by the RPF of the " +
          rpfWin + " trades that preceded it." +
        "</div>"
      : "";
    var rollingPfHtml =
      "<div class='section'>" +
        "<div class='section-title'>Rolling Profit Factor (" + rpfWin + "-trade window)</div>" +
        "<table><tbody>" +
        "<tr><th>Metric</th><th>Value</th></tr>" +
        "<tr><td>Current Rolling PF</td><td><span class='" + rpfCurClass + "'>" + rpfCur + "</span></td></tr>" +
        "<tr><td>Min Rolling PF</td><td><span class='neg'>" + rpfMin + "</span></td></tr>" +
        "<tr><td>Max Rolling PF</td><td><span class='pos'>" + rpfMax + "</span></td></tr>" +
        "</tbody></table>" +
        "<table class='mt-table'><thead><tr>" +
        "<th>Tier</th><th>Threshold</th><th>% of Time</th><th>Trades</th><th>Net P&amp;L</th>" +
        "</tr></thead><tbody>" + rpfTierRows + "</tbody></table>" +
        rpfNoteHtml +
      "</div>";

    /* 12. RPF Chart image */
    var rpfChartHtml = run.rpf_chart_b64
      ? "<div class='section'><div class='section-title'>Rolling Profit Factor Chart</div>" +
        "<img id='rpf-chart-img' src='data:image/png;base64," + run.rpf_chart_b64 + "' alt='RPF Chart'/></div>"
      : "";

    /* 13. RRR Sensitivity */
    var rrrSens     = m.rrr_sensitivity || [];
    var rrrSensRows = "";
    rrrSens.forEach(function (r) {
      var isCurrent = (p.rrr !== undefined && Math.abs(r.param - p.rrr) < 0.001);
      var rowStyle  = isCurrent ? " class='sens-current'" : "";
      var pfTxt2    = r.profit_factor !== null && r.profit_factor !== undefined
        ? "<span class='" + (r.profit_factor >= 1.5 ? "pos" : (r.profit_factor < 1.0 ? "neg" : "neu")) + "'>" + fmt(r.profit_factor) + "</span>"
        : "&#8734;";
      rrrSensRows +=
        "<tr" + rowStyle + ">" +
        "<td class='nowrap'>" + (isCurrent ? "<strong>" : "") + "1&thinsp;:&thinsp;" + r.param.toFixed(1) + (isCurrent ? " &#9654;</strong>" : "") + "</td>" +
        "<td>" + r.trades + "</td>" +
        "<td class='" + (r.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(r.win_rate, 1) + "%</td>" +
        "<td>" + pfTxt2 + "</td>" +
        "<td class='" + pClass(r.net_pnl) + "'>" + fmtMoney(r.net_pnl) + "</td>" +
        "<td class='neg'>" + fmt(r.max_drawdown, 1) + "%</td>" +
        "</tr>";
    });
    if (!rrrSensRows) rrrSensRows = "<tr><td colspan='6' class='td-empty-sm'>No data</td></tr>";
    var rrrSensHtml =
      "<div class='section'>" +
        "<div class='section-title'>RRR Sensitivity</div>" +
        "<table><thead><tr>" +
        "<th>RRR</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th><th>Net P&amp;L</th><th>Max DD</th>" +
        "</tr></thead><tbody>" + rrrSensRows + "</tbody></table></div>";

    /* 14. Swing Lookback Sensitivity */
    var swingSens     = m.swing_sensitivity || [];
    var swingSensRows = "";
    swingSens.forEach(function (r) {
      var isCurrent = (p.swing_lookback !== undefined && r.param === p.swing_lookback);
      var rowStyle  = isCurrent ? " class='sens-current'" : "";
      var pfTxt3    = r.profit_factor !== null && r.profit_factor !== undefined
        ? "<span class='" + (r.profit_factor >= 1.5 ? "pos" : (r.profit_factor < 1.0 ? "neg" : "neu")) + "'>" + fmt(r.profit_factor) + "</span>"
        : "&#8734;";
      swingSensRows +=
        "<tr" + rowStyle + ">" +
        "<td class='nowrap'>" + (isCurrent ? "<strong>" : "") + r.param + " bars" + (isCurrent ? " &#9654;</strong>" : "") + "</td>" +
        "<td>" + r.trades + "</td>" +
        "<td class='" + (r.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(r.win_rate, 1) + "%</td>" +
        "<td>" + pfTxt3 + "</td>" +
        "<td class='" + pClass(r.net_pnl) + "'>" + fmtMoney(r.net_pnl) + "</td>" +
        "<td class='neg'>" + fmt(r.max_drawdown, 1) + "%</td>" +
        "</tr>";
    });
    if (!swingSensRows) swingSensRows = "<tr><td colspan='6' class='td-empty-sm'>No data</td></tr>";
    var swingSensHtml =
      "<div class='section'>" +
        "<div class='section-title'>Swing Lookback Sensitivity</div>" +
        "<table><thead><tr>" +
        "<th>Swing Bars</th><th>Trades</th><th>Win Rate</th><th>Profit Factor</th><th>Net P&amp;L</th><th>Max DD</th>" +
        "</tr></thead><tbody>" + swingSensRows + "</tbody></table></div>";

    var chartHtml = run.chart_b64
      ? "<div class='section'><div class='section-title'>Chart</div>" +
        "<img id='chart-img' src='data:image/png;base64," + run.chart_b64 + "' alt='Backtest Chart'/></div>"
      : "";

    var notesHtml = (run.notes && run.notes !== "&#8212;" && run.notes !== "\u2014" && run.notes !== "\u2014")
      ? "<div class='v-notes'>&#128221;&nbsp; " + esc(run.notes) + "</div>"
      : "";

    var winRateCls = m.win_rate >= 50 ? "pos" : "neg";
    var sharpeCls  = m.sharpe >= 1 ? "pos" : (m.sharpe < 0 ? "neg" : "neu");

    var avgWinHtml  = m.avg_win  !== null && m.avg_win  !== undefined
      ? "<span class='pos'>$" + commaFmt(m.avg_win)  + "</span>" : "&#8212;";
    var avgLossHtml = m.avg_loss !== null && m.avg_loss !== undefined
      ? "<span class='neg'>-$" + commaFmt(Math.abs(m.avg_loss)) + "</span>" : "&#8212;";
    var avgStopPipsHtml = m.avg_stop_pips !== null && m.avg_stop_pips !== undefined
      ? m.avg_stop_pips.toFixed(1) : "&#8212;";
    var avgTargetPipsHtml = m.avg_target_pips !== null && m.avg_target_pips !== undefined
      ? m.avg_target_pips.toFixed(1) : "&#8212;";

    var stratBadge = (v.strategy)
      ? "<span class='strat-badge'>" + esc(v.strategy) + "</span>"
      : "";



    /* ── Summary panel ─────────────────────────────────────── */
    var mddS = m.max_daily_drawdown || {};
    var mddSStr = (mddS.dollar !== null && mddS.dollar !== undefined)
      ? "<span class='neg'>-$" + commaFmt(mddS.dollar) + " (" + fmt(mddS.pct, 2) + "%)</span>"
      : "&#8212;";
    var ddSStr = (m.max_drawdown_dollar !== null && m.max_drawdown_dollar !== undefined)
      ? "<span class='neg'>-$" + commaFmt(m.max_drawdown_dollar) + " (" + fmt(m.max_drawdown) + "%)</span>"
      : "<span class='neg'>" + fmt(m.max_drawdown) + "%</span>";
    var summaryHtml =
      "<div class='section'>" +
        "<div class='section-title'>Summary</div>" +
        "<table><tbody>" +
        "<tr><td class='lbl'>Total P&amp;L</td><td><span class='" + pClass(m.net_profit) + "'>" +
          fmtMoney(m.net_profit) + " (" + (m.net_profit_pct >= 0 ? "+" : "") + fmt(m.net_profit_pct, 1) + "%)</span></td></tr>" +
        "<tr><td class='lbl'>Win Rate</td><td><span class='" + (m.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(m.win_rate, 1) + "%</span></td></tr>" +
        "<tr><td class='lbl'>Profit Factor</td><td><span class='" + pfCls + "'>" + pfTxt + "</span></td></tr>" +
        "<tr><td class='lbl'>Max Drawdown</td><td>" + ddSStr + "</td></tr>" +
        "<tr><td class='lbl'>Max Daily DD</td><td>" + mddSStr + "</td></tr>" +
        "</tbody></table>" +
      "</div>";

    /* ── Entry Conditions panel ──────────────────────────────────────────────── */
    var ecData = v.entry_conditions || null;
    var ecThStyle = "class='ec-th'";
    var entryCondHtml;
    if (ecData && ecData.length > 0) {
      var ecRows = ecData.map(function(ec) {
        var addedVal = ec.since_version || "v1";
        var removedDisp = ec.removed_version
          ? "<span class='ec-removed-val'>" + esc(ec.removed_version) + "</span>"
          : "<span class='text-dim'>\u2014</span>";
        var rowOpacity = ec.removed_version ? " class='ec-row-removed'" : "";
        return "<tr" + rowOpacity + ">" +
          "<td class='ec-td-cond'>" + esc(ec.condition) + "</td>" +
          "<td class='ec-td-rule'>" + esc(ec.rule) + "</td>" +
          "<td class='ec-td-purpose'>" + esc(ec.purpose) + "</td>" +
          "<td class='ec-td-ver'><span class='ec-since-val'>" + esc(addedVal) + "</span></td>" +
          "<td class='ec-td-ver'>" + removedDisp + "</td>" +
          "</tr>";
      }).join("");
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Entry Conditions</div>" +
          "<table>" +
            "<thead><tr>" +
              "<th " + ecThStyle + ">Condition</th>" +
              "<th " + ecThStyle + ">Rule</th>" +
              "<th " + ecThStyle + "></th>" +
              "<th " + ecThStyle + ">+</th>" +
              "<th " + ecThStyle + ">\u2212</th>" +
            "</tr></thead>" +
            "<tbody>" + ecRows + "</tbody>" +
          "</table>" +
        "</div>";
    } else {
      var tfHoursStr = (p.time_filter_hours || []).sort(function(a,b){return a-b;}).map(function(h){return (h<10?"0":"")+h;}).join(" ");
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Entry Conditions</div>" +
          "<table>" +
            "<thead><tr>" +
              "<th " + ecThStyle + ">Condition</th>" +
              "<th " + ecThStyle + ">Rule</th>" +
              "<th " + ecThStyle + "></th>" +
              "<th " + ecThStyle + ">+</th>" +
              "<th " + ecThStyle + ">\u2212</th>" +
            "</tr></thead>" +
            "<tbody>" +
            "<tr><td class='ec-td-cond'>Trend Filter</td><td class='ec-td-rule'>EMA" + (p.ema_fast||"50") + " &lt; EMA" + (p.ema_slow||"200") + "</td><td class='ec-td-purpose'>Confirms downtrend \u2014 short only</td><td class='ec-td-ver'><span class='ec-since-val'>v1</span></td><td class='ec-td-ver'><span class='text-dim'>\u2014</span></td></tr>" +
            "<tr><td class='ec-td-cond'>Entry Signal</td><td class='ec-td-rule'>Price crosses below EMA" + (p.ema_entry||"20") + "</td><td class='ec-td-purpose'>Pullback rejection in trend direction</td><td class='ec-td-ver'><span class='ec-since-val'>v1</span></td><td class='ec-td-ver'><span class='text-dim'>\u2014</span></td></tr>" +
            "<tr><td class='ec-td-cond'>Stop Placement</td><td class='ec-td-rule'>Swing high over " + (p.swing_lookback||"20") + " bars</td><td class='ec-td-purpose'>Structural invalidation level</td><td class='ec-td-ver'><span class='ec-since-val'>v1</span></td><td class='ec-td-ver'><span class='text-dim'>\u2014</span></td></tr>" +
            "<tr><td class='ec-td-cond'>Direction</td><td class='ec-td-rule'><span class='val-highlight'>Short only</span></td><td class='ec-td-purpose'>Asymmetric edge identified on EURUSD</td><td class='ec-td-ver'><span class='ec-since-val'>v1</span></td><td class='ec-td-ver'><span class='text-dim'>\u2014</span></td></tr>" +
            "<tr><td class='ec-td-cond'>Time Window</td><td class='ec-td-rule'>UTC " + (p.time_filter ? tfHoursStr : "<span class='text-dim'>OFF</span>") + "</td><td class='ec-td-purpose'>High quality session hours</td><td class='ec-td-ver'><span class='ec-since-val'>v1</span></td><td class='ec-td-ver'><span class='text-dim'>\u2014</span></td></tr>" +
            "</tbody>" +
          "</table>" +
        "</div>";
    }

    document.getElementById("content").innerHTML =
      /* header */
      "<div id='v-header'>" +
        "<div id='v-header-top'>" +
          "<h2>" + esc(v.name) + stratBadge + "</h2>" +
          "<div class='btn-group'>" +
            "<button id='copy-btn' class='copy-btn'>Copy Version Report</button>" +
            "<button id='delete-btn' class='delete-btn'>Delete Version</button>" +
          "</div>" +
        "</div>" +
        (function () {
          var metaTicker = (p.ticker || "").replace(/=X$/i, "");
          var metaRange = run.start_date && run.end_date
            ? { start: run.start_date, end: run.end_date }
            : fullRunRange(run);
          var metaDateStr = metaRange.start && metaRange.end
            ? fmtSbDate(metaRange.start) + " \u2192 " + fmtSbDate(metaRange.end) : "";
          var metaDur = calcDuration(metaRange.start, metaRange.end);
          return "<div class='v-meta'>Run on " + esc(run.date || "") +
            (metaTicker ? " &nbsp;&middot;&nbsp; " + esc(metaTicker) : "") +
            (metaDateStr ? " &nbsp;&middot;&nbsp; " + metaDateStr : "") +
            (metaDur ? " &nbsp;&middot;&nbsp; " + esc(metaDur) : "") +
            "</div>";
        }()) +
        notesHtml +
      "</div>" +

      /* ── Section 1: Summary + Entry Conditions ────────────────────────────── */
      "<div class='two-col'>" + summaryHtml + entryCondHtml + "</div>" +

      /* ── Section 2: Results + Parameters ──────────────────────────────────── */
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
          row("Final Equity",  "$" + commaFmt(m.final_equity)) +
          row("Max Drawdown",  (function () {
            var ddD = m.max_drawdown_dollar;
            var ddDStr = (ddD !== null && ddD !== undefined) ? "-$" + commaFmt(ddD) + " (" + fmt(m.max_drawdown) + "%)" : fmt(m.max_drawdown) + "%";
            return "<span class='neg'>" + ddDStr + "</span>";
          }()) ) +
          row("Max Daily DD",  (function () {
            var mdd = m.max_daily_drawdown || {};
            if (mdd.dollar === null || mdd.dollar === undefined) return "&#8212;";
            return "<span class='neg'>-$" + commaFmt(mdd.dollar) + " (" + fmt(mdd.pct, 2) + "%)</span>";
          }()) ) +
          row("Sharpe Ratio",  "<span class='" + sharpeCls + "'>" + fmt(m.sharpe) + "</span>") +
          row("Avg Win",       avgWinHtml) +
          row("Avg Loss",      avgLossHtml) +
          row("Avg Stop (pips)",   avgStopPipsHtml) +
          row("Avg Target (pips)", avgTargetPipsHtml) +
          row("Best Trade",    "<span class='pos'>" + fmtMoney(m.best_trade)  + "</span>") +
          row("Worst Trade",   "<span class='neg'>" + fmtMoney(m.worst_trade) + "</span>") +
          (m.avg_position_size !== null && m.avg_position_size !== undefined
            ? row("Avg Position Size", Math.round(m.avg_position_size).toLocaleString() + " units")
            : "") +
          (m.min_position_size !== null && m.min_position_size !== undefined
            ? row("Min Position Size", Math.round(m.min_position_size).toLocaleString() + " units")
            : "") +
          (m.max_position_size !== null && m.max_position_size !== undefined
            ? row("Max Position Size", Math.round(m.max_position_size).toLocaleString() + " units")
            : "") +
          "</tbody></table>" +
        "</div>" +

        "<div class='section'>" +
          "<div class='section-title'>Parameters</div>" +
          "<table><tbody>" +
          row("Instrument",     esc((p.ticker || "").replace(/=X$/i, ""))) +
          row("Interval",       esc(p.interval || "")) +
          row("History",        (run.days_back || p.days_back || "") + " days") +
          row("Starting Capital", "$" + (p.starting_cash || 0).toLocaleString()) +
          row("EMA Slow",       p.ema_slow) +
          row("EMA Fast",       p.ema_fast) +
          row("EMA Entry",      p.ema_entry) +
          row("Swing Lookback", (p.swing_lookback || "") + " bars") +
          row("RRR",            "1&thinsp;:&thinsp;" + (p.rrr || "")) +
          row("Risk / Trade",   ((p.risk_pct || 0) * 100).toFixed(1) + "% = $" + ((p.starting_cash || 0) * (p.risk_pct || 0)).toLocaleString()) +
          (m.avg_position_size !== null && m.avg_position_size !== undefined
            ? row("Avg Position Size", Math.round(m.avg_position_size).toLocaleString() + " units")
            : "") +
          row("Min Stop",       ((p.min_stop || 0) * 10000).toFixed(0) + " pips") +
          row("Max Stop",       ((p.max_stop || 0) * 10000).toFixed(0) + " pips") +
          row("Direction",      "<span class='val-highlight'>" + esc(p.trade_direction || "both") + "</span>") +
          row("Time Filter",    p.time_filter
            ? "<span class='pos'>ON</span> &mdash; " + esc((p.time_filter_hours || []).join(", ")) + " UTC"
            : "<span class='text-dim'>OFF</span>") +
          "</tbody></table>" +
        "</div>" +

      "</div>" +

      /* ── Section 3: Performance by Direction ──────────────────────────────── */
      dirHtml +

      /* ── Range Filter + Regime Classification (side by side) ────────────── */
      "<div class='two-col'>" + rsdHtml + regimeHtml + "</div>" +

      /* ── Section 4: Rolling Profit Factor data ────────────────────────────── */
      rollingPfHtml +

      /* ── Section 5: RPF chart image ───────────────────────────────────────── */
      rpfChartHtml +

      /* ── Section 6: Monthly Performance ──────────────────────────────────── */
      monthHtml +

      /* ── Section 7: Time of Day Performance ──────────────────────────────── */
      timeOfDayHtml +

      /* ── Section 8: Main chart (visual divider) ───────────────────────────── */
      chartHtml +

      /* ── Section 9: Streak Analysis + Stop vs Target ──────────────────────── */
      "<div class='two-col'>" + streakHtml + stopHtml + "</div>" +

      /* ── Win Rate Trend ──────────────────────────────────────────────────── */
      winRateTrendHtml +

      /* ── Section 11: Trade Duration + Filter Impact Summary ───────────────── */
      "<div class='two-col'>" + durationHtml + filterImpactHtml + "</div>" +

      /* ── Daily Drawdown ────────────────────────────────────────────────────── */
      dailyDDHtml +

      /* ── Section 12: RRR Sensitivity + Swing Lookback Sensitivity ─────────── */
      "<div class='two-col'>" + rrrSensHtml + swingSensHtml + "</div>";

    /* Wire copy button — context aware */
    (function (ver, runData) {
      var btn = document.getElementById("copy-btn");
      if (!btn) return;
      var isDateRange = activeRunIdx > 0;
      btn.textContent = isDateRange ? "Copy Range Report" : "Copy Version Report";
      btn.addEventListener("click", function () {
        var isRange = activeRunIdx > 0;
        var md = isRange ? buildRunMarkdown(ver, runData) : buildVersionMarkdown(ver);
        var label = isRange ? "Copy Range Report" : "Copy Version Report";
        navigator.clipboard.writeText(md).then(function () {
          btn.textContent = "\u2713 Copied!";
          btn.classList.add("copied");
          showToast(isRange ? "Range report copied" : "Version report copied");
          setTimeout(function () {
            btn.textContent = label;
            btn.classList.remove("copied");
          }, 2200);
        }).catch(function () {
          btn.textContent = "Failed";
          setTimeout(function () { btn.textContent = label; }, 2500);
        });
      });
    }(v, run));

    /* Wire delete button */
    (function (ver) {
      var delBtn = document.getElementById("delete-btn");
      if (!delBtn) return;
      var isDateRange = activeRunIdx > 0;
      delBtn.textContent = isDateRange ? "Delete Date Range" : "Delete Version";
      delBtn.addEventListener("click", function () {
        var isRange = activeRunIdx > 0;
        var msg = isRange
          ? "Are you sure you want to delete this date range run? This cannot be undone."
          : "Are you sure you want to delete this version and all its date range runs? This cannot be undone.";
        if (!confirm(msg)) return;
        delBtn.disabled = true;
        delBtn.textContent = "Deleting\u2026";
        var url = isRange ? "/delete_run" : "/delete_version";
        var body = isRange
          ? JSON.stringify({ name: ver.name, run_idx: activeRunIdx })
          : JSON.stringify({ name: ver.name });
        fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: body
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (data.ok) {
            window.location.reload();
          } else {
            delBtn.disabled = false;
            delBtn.textContent = isRange ? "Delete Date Range" : "Delete Version";
            alert("Delete failed: " + (data.error || "Unknown error"));
          }
        })
        .catch(function () {
          delBtn.disabled = false;
          delBtn.textContent = isRange ? "Delete Date Range" : "Delete Version";
          alert("Delete failed \u2014 is the server running?");
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

    var dlRowArr = [];
    var prevPF = null;

    for (var i = 0; i < svs.length; i++) {
      var entry = svs[i];
      var v = entry.v;
      var runs = getRuns(v);
      var firstRun = runs[0] || {};
      var m = firstRun.metrics || {};

      var pf = (m.profit_factor !== null && m.profit_factor !== undefined) ? m.profit_factor : null;
      var wr = (m.win_rate !== null && m.win_rate !== undefined) ? m.win_rate : null;
      var np = (m.net_profit !== null && m.net_profit !== undefined) ? m.net_profit : null;

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

      var notes = (firstRun.notes && firstRun.notes !== "\u2014" && firstRun.notes !== "\u2014") ? esc(firstRun.notes) : "<span class='text-dim'>\u2014</span>";

      dlRowArr.push(
        "<tr>" +
        "<td class='dl-version'>" + esc(v.name) + (runs.length > 1 ? " <span class='text-dim'>(" + runs.length + " runs)</span>" : "") + "</td>" +
        "<td class='dl-date'>" + esc(firstRun.date || "") + "</td>" +
        "<td class='dl-notes'>" + notes + "</td>" +
        "<td class='dl-td-right'>" + pfDisp + "</td>" +
        "<td class='dl-td-right'>" + wrDisp + "</td>" +
        "<td class='dl-td-right'>" + npDisp + "</td>" +
        "</tr>"
      );
    }
    var dlRows = dlRowArr.slice().reverse().join("");

    document.getElementById("content").innerHTML =
      "<div id='devlog-header'>" +
        "<h2>Development Log</h2>" +
        "<div class='v-meta'>" + esc(currentStrategy) + " &mdash; " + svs.length + " version" + (svs.length !== 1 ? "s" : "") + "</div>" +
      "</div>" +
      "<div class='section'>" +
        "<table class='devlog-table'>" +
          "<thead><tr>" +
          "<th>Version</th><th>Date</th><th>Change</th>" +
          "<th class='dl-th-right'>Profit Factor</th>" +
          "<th class='dl-th-right'>Win Rate</th>" +
          "<th class='dl-th-right'>Net P&amp;L</th>" +
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

    var svs = getStrategyVersions();
    if (svs.length > 0) {
      var lastIdx = svs[svs.length - 1].idx;
      activeVersionIdx = lastIdx;
      activeRunIdx = 0;
      expandedVersions[lastIdx] = true;
      renderSidebar();
      renderContent(lastIdx, 0);
    } else {
      document.getElementById("content").innerHTML =
        "<div id='empty-state'>" +
          "<span class='empty-icon'>&#128202;</span>" +
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
      var svs = getStrategyVersions();
      if (svs.length > 0) {
        var lastIdx = svs[svs.length - 1].idx;
        activeVersionIdx = lastIdx;
        activeRunIdx = 0;
        renderSidebar();
        renderContent(lastIdx, 0);
      }
    }
  });

  renderSidebar();
  var svs = getStrategyVersions();
  if (svs.length > 0) {
    var lastIdx = svs[svs.length - 1].idx;
    activeVersionIdx = lastIdx;
    activeRunIdx = 0;
    expandedVersions[lastIdx] = true;
    renderSidebar();
    renderContent(lastIdx, 0);
  }
})();
</script>

</body>
</html>"""
    return template.replace("__VERSIONS_JSON__", versions_json)


def generate_html_report(trades, equity, chart_path="backtest_chart.png", notes="",
                         blocked_signals=None, df=None, rpf_chart_path=None,
                         run_mode="new_version", run_start_date="", run_end_date=""):
    """Create or update report.html.

    run_mode="new_version" → increment version, create new entry with first run
    run_mode="date_range"  → append a run to the most recent version
    """
    global VERSION
    report_path = "report.html"

    metrics = compute_metrics(trades, equity, blocked_signals=blocked_signals, df=df)
    if metrics is None:
        print("  No trades generated — skipping HTML report.")
        return

    # ── Load main chart as base64 ──────────────────────────────────────────────
    chart_b64 = ""
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            chart_b64 = base64.b64encode(fh.read()).decode("utf-8")

    # ── Load RPF chart as base64 ───────────────────────────────────────────────
    rpf_chart_b64 = ""
    if rpf_chart_path and os.path.exists(rpf_chart_path):
        with open(rpf_chart_path, "rb") as fh:
            rpf_chart_b64 = base64.b64encode(fh.read()).decode("utf-8")

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

    # ── Migrate old versions: ensure every version has a runs[] array ─────────
    for v in existing_versions:
        if "runs" not in v:
            # Migrate legacy single-run format into runs[]
            v["runs"] = [{
                "date":          v.get("date", ""),
                "start_date":    "",
                "end_date":      "",
                "days_back":     v.get("params", {}).get("days_back", DAYS_BACK),
                "notes":         v.get("notes", "—"),
                "chart_b64":     v.get("chart_b64", ""),
                "rpf_chart_b64": v.get("rpf_chart_b64", ""),
                "metrics":       v.get("metrics", {}),
                "last_trades":   v.get("last_trades", []),
            }]
            # Remove legacy top-level run data (keep name, params, strategy, etc.)
            for key in ["date", "notes", "chart_b64", "rpf_chart_b64",
                        "metrics", "last_trades"]:
                v.pop(key, None)

    # ── Build a run object ─────────────────────────────────────────────────────
    run_label = ""
    if run_mode == "date_range" and run_start_date and run_end_date:
        run_label = f"{run_start_date} → {run_end_date}"
    new_run = {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
        "start_date":    run_start_date if run_mode == "date_range" else "",
        "end_date":      run_end_date   if run_mode == "date_range" else "",
        "days_back":     DAYS_BACK if run_mode != "date_range" else 0,
        "label":         run_label,
        "notes":         notes.strip() if notes else "—",
        "chart_b64":     chart_b64,
        "rpf_chart_b64": rpf_chart_b64,
        "metrics":       metrics,
        "last_trades":   last_trades,
    }

    params_dict = {
        "ticker":         TICKER,
        "interval":       INTERVAL,
        "days_back":      DAYS_BACK,
        "starting_cash":  STARTING_CASH,
        "ema_slow":       EMA_SLOW,
        "ema_fast":       EMA_FAST,
        "ema_entry":      EMA_ENTRY,
        "swing_lookback": SWING_LOOKBACK,
        "rrr":            RRR,
        "risk_pct":          RISK_PCT,
        "min_stop":          MIN_STOP,
        "max_stop":          MAX_STOP,
        "trade_direction":   TRADE_DIRECTION,
        "time_filter":       TIME_FILTER,
        "time_filter_hours": TIME_FILTER_HOURS if TIME_FILTER else [],
        "max_daily_loss": MAX_DAILY_LOSS,
    }

    if run_mode == "date_range" and existing_versions:
        # Append run to the most recent version (same strategy)
        latest = existing_versions[-1]
        latest["runs"].append(new_run)
        version_num = len(existing_versions)
        action = "Added date range run to"
    else:
        # New version — increment from the highest existing version number
        # so deleted versions are never reused (e.g. v1,v2,v5 → next is v6)
        max_ver = 0
        for v in existing_versions:
            m = re.match(r'^v(\d+)$', v.get("name", ""))
            if m:
                max_ver = max(max_ver, int(m.group(1)))
        next_ver = max_ver + 1
        VERSION = f"v{next_ver}"
        version_num = len(existing_versions) + 1
        new_version = {
            "name":             VERSION,
            "strategy":         STRATEGY,
            "entry_conditions": ENTRY_CONDITIONS,
            "params":           params_dict,
            "runs":             [new_run],
        }
        existing_versions.append(new_version)
        action = "Created" if version_num == 1 else "Updated"

    # ── Write HTML ─────────────────────────────────────────────────────────────
    versions_json = json.dumps(existing_versions, indent=2, ensure_ascii=False)
    html = _build_html(versions_json)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(html)

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

    # ── Run mode (set by server.py via environment variables) ─────────────────
    run_mode       = os.environ.get("RUN_MODE", "new_version")
    run_start_date = os.environ.get("RUN_START_DATE", "")
    run_end_date   = os.environ.get("RUN_END_DATE", "")

    if run_mode == "date_range" and run_start_date and run_end_date:
        df = fetch_data(TICKER, INTERVAL, DAYS_BACK,
                        start_date=run_start_date, end_date=run_end_date)
    else:
        df = fetch_data(TICKER, INTERVAL, DAYS_BACK)

    df              = add_indicators(df)
    trades, equity, blocked_signals = run_backtest(df)
    print_results(trades, equity)
    time_blocked = sum(1 for s in blocked_signals if s["reason"] == "time")
    print(f"  Time filter blocked : {time_blocked} signal(s)")
    chart_path, rpf_chart_path = save_charts(df, trades, equity)
    generate_html_report(trades, equity, chart_path=chart_path, notes=run_notes,
                         blocked_signals=blocked_signals, df=df,
                         rpf_chart_path=rpf_chart_path,
                         run_mode=run_mode,
                         run_start_date=run_start_date,
                         run_end_date=run_end_date)
    metrics = compute_metrics(trades, equity, blocked_signals=blocked_signals, df=df)
    update_results_log(metrics, notes=run_notes)
    git_commit_and_push(metrics, VERSION, TICKER, INTERVAL)
