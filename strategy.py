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
INTERVAL        = os.environ.get("INTERVAL", "5m")  # bar interval — used by Massive (primary) and Yahoo (fallback)
DAYS_BACK       = 365             # Full 365-day run
STARTING_CASH   = 100_000.0

EMA_SHORT       = int(os.environ.get("EMA_SHORT", 8))
EMA_MID         = int(os.environ.get("EMA_MID", 20))
EMA_LONG        = int(os.environ.get("EMA_LONG", 40))
RRR_RISK        = int(os.environ.get("RRR_RISK", 1))
RRR_REWARD      = int(os.environ.get("RRR_REWARD", 2))
RRR             = float(RRR_REWARD) / float(RRR_RISK)
RISK_PCT        = 0.01
MIN_STOP        = 0.0005     # 5 pips minimum stop
MAX_STOP        = 0.0200     # 200 pips maximum stop
FRACTAL_STOP_PIPS = float(os.environ.get("FRACTAL_STOP_PIPS", 15)) / 10000  # pips → price; default 15 pips

TRADE_DIRECTION   = os.environ.get("TRADE_DIRECTION", "both")   # "both" | "long_only" | "short_only"

MAX_DAILY_LOSS  = 2000.0            # stop trading if day's loss reaches $2,000 (2% of capital)

VERSION = "v6"
NOTES = "Fractal-based entries with EMA 8/20/40 alignment"
STRATEGY        = "Trend Following"

ENTRY_CONDITIONS = [
    {
        "condition":       "Instrument",
        "rule":            _INSTRUMENT,
    },
    {
        "condition":       "Interval",
        "rule":            INTERVAL,
    },
    {
        "condition":       "EMA Short",
        "rule":            str(EMA_SHORT),
    },
    {
        "condition":       "EMA Mid",
        "rule":            str(EMA_MID),
    },
    {
        "condition":       "EMA Long",
        "rule":            str(EMA_LONG),
    },
    {
        "condition":       "Stop Loss Level",
        "rule":            str(int(FRACTAL_STOP_PIPS * 10000)),
    },
    {
        "condition":       "Direction",
        "rule":            TRADE_DIRECTION.replace("_", " ").title(),
    },
    {
        "condition":       "RRR",
        "rule":            f"{RRR_RISK}:{RRR_REWARD}",
    },
]

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_data(ticker, interval, days_back, start_date=None, end_date=None):
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date,   "%Y-%m-%d")
        # Fetch extra history so indicators (EMA200 etc.) are properly warmed up;
        # the caller's post-filter trims to the exact requested range afterwards.
        # Also fetch 7 days AFTER end so trades entered near the end of the range
        # have enough future bars to hit their SL/TP and resolve properly.
        start = start - timedelta(days=30)
        end = end + timedelta(days=7)
        days_back = max(1, (end - start).days + 1)
    else:
        end   = datetime.now()
        start = end - timedelta(days=days_back)

    # ── Parse interval string (e.g. "5m", "1m", "60m") into multiplier + timespan
    import re as _re
    _iv_match = _re.match(r'^(\d+)(m|h)$', interval)
    if _iv_match:
        _multiplier = int(_iv_match.group(1))
        _timespan   = "minute" if _iv_match.group(2) == "m" else "hour"
    else:
        _multiplier = 5
        _timespan   = "minute"

    # ── Primary: Massive API ───────────────────────────────────────────────────
    if MASSIVE_API_KEY:
        try:
            from massive import RESTClient
            print(f"\nFetching {MASSIVE_TICKER} {interval} data from Massive ({days_back} days)...")
            client = RESTClient(api_key=MASSIVE_API_KEY)
            bars = list(client.list_aggs(
                ticker    = MASSIVE_TICKER,
                multiplier= _multiplier,
                timespan  = _timespan,
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
            print(f"  Source: Massive API ({MASSIVE_TICKER}, {interval})")
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
    df["ema_short"] = df.Close.ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_mid"]   = df.Close.ewm(span=EMA_MID,   adjust=False).mean()
    df["ema_long"]  = df.Close.ewm(span=EMA_LONG,  adjust=False).mean()
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
    df["adx"]   = _dx.ewm(alpha=_alpha, adjust=False).mean()
    df["atr14"] = _atr14
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
    """Stripped-down backtest with a specific RRR.
    Uses fractal-based entries; swing_lookback is unused but kept for API compat.
    Returns condensed metrics dict, or None if no trades."""
    df2 = df.copy()

    highs_s = df2['High'].values
    lows_s  = df2['Low'].values
    atr_s   = df2['atr14'].values

    cash      = STARTING_CASH
    equity_s  = [cash]
    trades_s  = []
    in_trade  = False
    entry_p = sl = tp = size = 0.0
    direction = None

    prev_high_price_s = None
    prev_low_price_s  = None
    last_high_label_s = None
    last_low_label_s  = None
    last_fractal_low_s  = None
    last_fractal_high_s = None

    for i in range(1, len(df2)):
        c    = float(df2["Close"].iloc[i])

        if in_trade:
            bar_hi = float(df2["High"].iloc[i])
            bar_lo = float(df2["Low"].iloc[i])
            if direction == "long":
                intra_sl = bar_lo <= sl
                intra_tp = bar_hi >= tp
            else:
                intra_sl = bar_hi >= sl
                intra_tp = bar_lo <= tp
            if intra_sl and intra_tp:
                hit_sl, hit_tp = True, False
            elif intra_sl:
                hit_sl, hit_tp = True, False
            elif intra_tp:
                hit_sl, hit_tp = False, True
            else:
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

        # Rolling fractal detection
        if i >= 4:
            fi = i - 2
            fh = highs_s[fi]; fl = lows_s[fi]
            thr = 0.5 * float(atr_s[fi])
            is_ph = (fh > highs_s[fi-1] and fh > highs_s[fi-2] and
                     fh > highs_s[fi+1] and fh > highs_s[fi+2])
            is_pl = (fl < lows_s[fi-1]  and fl < lows_s[fi-2]  and
                     fl < lows_s[fi+1]  and fl < lows_s[fi+2])
            if is_ph:
                if prev_high_price_s is None:
                    last_high_label_s = 'CH'
                else:
                    d = fh - prev_high_price_s
                    last_high_label_s = 'CH' if abs(d) < thr else ('LH' if d < 0 else 'HH')
                prev_high_price_s = float(fh)
                last_fractal_high_s = float(fh)
            if is_pl:
                if prev_low_price_s is None:
                    last_low_label_s = 'CL'
                else:
                    d = fl - prev_low_price_s
                    last_low_label_s = 'CL' if abs(d) < thr else ('HL' if d > 0 else 'LL')
                prev_low_price_s = float(fl)
                last_fractal_low_s = float(fl)

        if not in_trade and i >= 4:
            es = float(df2["ema_short"].iloc[i])
            em = float(df2["ema_mid"].iloc[i])
            el = float(df2["ema_long"].iloc[i])

            long_sig = (last_low_label_s == 'HL' and last_high_label_s == 'HH'
                        and es > em > el and last_fractal_low_s is not None
                        and min(es, em) <= last_fractal_low_s <= max(es, em)
                        and TRADE_DIRECTION != "short_only")
            short_sig = (last_high_label_s == 'LH' and last_low_label_s == 'LL'
                         and es < em < el and last_fractal_high_s is not None
                         and min(es, em) <= last_fractal_high_s <= max(es, em)
                         and TRADE_DIRECTION != "long_only")

            if long_sig:
                sl_p = last_fractal_low_s - FRACTAL_STOP_PIPS
                dist = c - sl_p
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "long"; entry_p = c; sl = sl_p
                    tp = c + dist * rrr; size = (cash * RISK_PCT) / dist; in_trade = True
            elif short_sig:
                sl_p = last_fractal_high_s + FRACTAL_STOP_PIPS
                dist = sl_p - c
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "short"; entry_p = c; sl = sl_p
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
    entry_atr        = 0.0   # ATR value at entry bar
    entry_ts         = None  # timestamp of the entry bar (for time-of-day diagnostics)
    blocked_signals  = []    # signals that were filtered out (for Filter Impact Summary)
    _debug_entries   = 0     # counter for entry timezone debug prints
    _daily_loss_day  = None  # current calendar day (UTC date) for daily loss tracking
    _daily_loss_pnl  = 0.0   # cumulative closed-trade P&L for _daily_loss_day

    # ── Rolling fractal state ─────────────────────────────────────────────────
    highs = df['High'].values
    lows  = df['Low'].values
    atr_vals = df['atr14'].values

    prev_high_price = None   # price of the most recent pivot high
    prev_low_price  = None   # price of the most recent pivot low
    last_high_label = None   # classification of last pivot high (HH, LH, CH)
    last_low_label  = None   # classification of last pivot low  (HL, LL, CL)
    last_fractal_low_price  = None   # price of the most recent confirmed pivot low
    last_fractal_high_price = None   # price of the most recent confirmed pivot high

    for i in range(1, len(df)):
        c     = float(df.Close.iloc[i])
        ts    = df['Datetime'].iloc[i]

        # ── Check exits ───────────────────────────────────────────────────────
        if in_trade:
            bar_hi = float(df.High.iloc[i])
            bar_lo = float(df.Low.iloc[i])

            # Update MAE: track worst adverse intra-bar price before exit check
            if direction == "long":
                worst_adverse = min(worst_adverse, bar_lo)
            else:
                worst_adverse = max(worst_adverse, bar_hi)

            # Intrabar check: high/low hit SL or TP before close is evaluated
            if direction == "long":
                intra_sl = bar_lo <= sl
                intra_tp = bar_hi >= tp
            else:
                intra_sl = bar_hi >= sl
                intra_tp = bar_lo <= tp

            # If both hit on same bar, SL takes priority (conservative)
            if intra_sl and intra_tp:
                hit_sl, hit_tp = True, False
            elif intra_sl:
                hit_sl, hit_tp = True, False
            elif intra_tp:
                hit_sl, hit_tp = False, True
            else:
                # Fall through to close-price logic
                hit_sl = (direction == "long"  and c <= sl) or \
                         (direction == "short" and c >= sl)
                hit_tp = (direction == "long"  and c >= tp) or \
                         (direction == "short" and c <= tp)

            # ── Determine exit type ───────────────────────────────────────────
            force_close = False
            if not (hit_sl or hit_tp):
                # Check daily loss limit with unrealised P&L
                _unrealised = (c - entry_p) * size if direction == "long" \
                              else (entry_p - c) * size
                _bar_utc = pd.to_datetime(ts)
                _bar_utc = _bar_utc.tz_convert('UTC') if _bar_utc.tzinfo else _bar_utc.tz_localize('UTC')
                _bar_day = _bar_utc.date()
                _day_pnl = _daily_loss_pnl if _bar_day == _daily_loss_day else 0.0
                if (_day_pnl + _unrealised) <= -MAX_DAILY_LOSS:
                    force_close = True

            if hit_sl or hit_tp or force_close:
                if force_close:
                    exit_p = c          # force-close at current bar's close
                else:
                    exit_p = sl if hit_sl else tp
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
                result_label = "DD" if force_close else ("TP" if hit_tp else "SL")
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
                    "result":       result_label,
                    "mae":          mae,
                    "adx_at_entry": entry_adx,
                    "atr_at_entry": entry_atr,
                    "timestamp":    ts,
                    "entry_ts":     entry_ts
                })
                in_trade = False

        equity.append(cash)

        # ── Rolling fractal detection (confirmed at bar i, formed at bar i-2) ─
        # A fractal needs 2 bars on each side, so the earliest confirmable
        # pivot is at index 2.  At bar i we confirm the pivot at bar i-2.
        if i >= 4:
            fi = i - 2   # fractal bar index
            fh = highs[fi]
            fl = lows[fi]
            atr_fi = float(atr_vals[fi])
            threshold = 0.5 * atr_fi

            # Pivot high?
            is_ph = (fh > highs[fi-1] and fh > highs[fi-2] and
                     fh > highs[fi+1] and fh > highs[fi+2])
            # Pivot low?
            is_pl = (fl < lows[fi-1]  and fl < lows[fi-2]  and
                     fl < lows[fi+1]  and fl < lows[fi+2])

            if is_ph:
                if prev_high_price is None:
                    last_high_label = 'CH'
                else:
                    diff = fh - prev_high_price
                    if abs(diff) < threshold:
                        last_high_label = 'CH'
                    elif diff < 0:
                        last_high_label = 'LH'
                    else:
                        last_high_label = 'HH'
                prev_high_price = float(fh)
                last_fractal_high_price = float(fh)

            if is_pl:
                if prev_low_price is None:
                    last_low_label = 'CL'
                else:
                    diff = fl - prev_low_price
                    if abs(diff) < threshold:
                        last_low_label = 'CL'
                    elif diff > 0:
                        last_low_label = 'HL'
                    else:
                        last_low_label = 'LL'
                prev_low_price = float(fl)
                last_fractal_low_price = float(fl)

        # ── Check entries ─────────────────────────────────────────────────────
        if not in_trade:
            ema_s = float(df.ema_short.iloc[i])
            ema_m = float(df.ema_mid.iloc[i])
            ema_l = float(df.ema_long.iloc[i])

            ema_bullish = ema_s > ema_m > ema_l
            ema_bearish = ema_s < ema_m < ema_l

            # ── Long signal: HL fractal just confirmed, after an HH,
            #    fractal low between EMA short and EMA mid ─────────────────────
            long_sig_raw = False
            long_fractal_price = None
            if (i >= 4 and last_low_label == 'HL' and last_high_label == 'HH'
                    and ema_bullish and last_fractal_low_price is not None):
                fp = last_fractal_low_price
                # Fractal low must sit between EMA 8 and EMA 20
                if min(ema_s, ema_m) <= fp <= max(ema_s, ema_m):
                    long_sig_raw = True
                    long_fractal_price = fp

            # ── Short signal: LH fractal just confirmed, after an LL,
            #    fractal high between EMA short and EMA mid ────────────────────
            short_sig_raw = False
            short_fractal_price = None
            if (i >= 4 and last_high_label == 'LH' and last_low_label == 'LL'
                    and ema_bearish and last_fractal_high_price is not None):
                fp = last_fractal_high_price
                # Fractal high must sit between EMA 8 and EMA 20
                if min(ema_s, ema_m) <= fp <= max(ema_s, ema_m):
                    short_sig_raw = True
                    short_fractal_price = fp

            # ── Track direction-blocked signals ───────────────────────────────
            if TRADE_DIRECTION == "short_only" and long_sig_raw:
                _sl_b = long_fractal_price - FRACTAL_STOP_PIPS
                dist_b = c - _sl_b
                if MIN_STOP <= dist_b <= MAX_STOP:
                    blocked_signals.append(_scan_outcome(
                        df, i, "long", c, _sl_b,
                        c + dist_b * RRR,
                        (cash * RISK_PCT) / dist_b, ts, "direction"))
            if TRADE_DIRECTION == "long_only" and short_sig_raw:
                _sl_b = short_fractal_price + FRACTAL_STOP_PIPS
                dist_b = _sl_b - c
                if MIN_STOP <= dist_b <= MAX_STOP:
                    blocked_signals.append(_scan_outcome(
                        df, i, "short", c, _sl_b,
                        c - dist_b * RRR,
                        (cash * RISK_PCT) / dist_b, ts, "direction"))

            # Apply direction filter
            long_sig  = long_sig_raw  and (TRADE_DIRECTION != "short_only")
            short_sig = short_sig_raw and (TRADE_DIRECTION != "long_only")

            # ── Daily loss limit ──────────────────────────────────────────────
            _ts_day = pd.to_datetime(ts)
            _ts_day_utc = _ts_day.tz_convert('UTC') if _ts_day.tzinfo else _ts_day.tz_localize('UTC')
            _today = _ts_day_utc.date()
            if _today != _daily_loss_day:
                _daily_loss_day = _today
                _daily_loss_pnl = 0.0
            if _daily_loss_pnl <= -MAX_DAILY_LOSS:
                continue

            if long_sig:
                sl_price      = long_fractal_price - FRACTAL_STOP_PIPS
                dist          = c - sl_price                          # actual distance: entry close → stop level
                if MIN_STOP <= dist <= MAX_STOP:
                    direction     = "long"
                    entry_p       = c                                 # enter at close of confirming candle
                    sl            = sl_price
                    tp            = c + dist * RRR
                    size          = (cash * RISK_PCT) / dist
                    in_trade      = True
                    entry_idx     = i
                    entry_ts      = ts
                    worst_adverse = c
                    entry_adx     = float(df.adx.iloc[i])
                    entry_atr     = float(df.atr14.iloc[i])
                    if _debug_entries < 5:
                        _ts_dbg = pd.to_datetime(ts)
                        _ts_utc_dbg = _ts_dbg.tz_convert('UTC') if _ts_dbg.tzinfo else _ts_dbg.tz_localize('UTC')
                        print(f"  [DBG entry {_debug_entries+1}] raw={ts}  utc_hour={_ts_utc_dbg.hour}  direction=long  fractal_low={long_fractal_price:.5f}")
                        _debug_entries += 1

            elif short_sig:
                sl_price      = short_fractal_price + FRACTAL_STOP_PIPS
                dist          = sl_price - c                          # actual distance: stop level → entry close
                if MIN_STOP <= dist <= MAX_STOP:
                    direction     = "short"
                    entry_p       = c                                 # enter at close of confirming candle
                    sl            = sl_price
                    tp            = c - dist * RRR
                    size          = (cash * RISK_PCT) / dist
                    in_trade      = True
                    entry_idx     = i
                    entry_ts      = ts
                    worst_adverse = c
                    entry_adx     = float(df.adx.iloc[i])
                    entry_atr     = float(df.atr14.iloc[i])
                    if _debug_entries < 5:
                        _ts_dbg = pd.to_datetime(ts)
                        _ts_utc_dbg = _ts_dbg.tz_convert('UTC') if _ts_dbg.tzinfo else _ts_dbg.tz_localize('UTC')
                        print(f"  [DBG entry {_debug_entries+1}] raw={ts}  utc_hour={_ts_utc_dbg.hour}  direction=short  fractal_high={short_fractal_price:.5f}")
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
    print(f"  Strategy       : EMA {EMA_SHORT}/{EMA_MID}/{EMA_LONG} | RRR 1:{RRR}")
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
    """Save two chart images and return (main_path, eq_dd_path)."""
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

    # ── Date range detection for chart style selection ────────────────────────
    _date_min  = dates.min().date()
    _date_max  = dates.max().date()
    _day_span  = (_date_max - _date_min).days
    is_one_day     = _day_span == 0
    is_short_range = 0 < _day_span < 7

    def _set_x_fmt(ax):
        """Apply appropriate x-axis date format based on span of date range."""
        if is_one_day:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        elif is_short_range:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Downsample helper for smooth chart rendering ──────────────────────────
    # With 150k+ 5-minute bars, plotting every point creates jagged lines at
    # chart resolution.  Downsample to ~3000 points and interpolate for smooth
    # EMA curves that look realistic.
    MAX_CHART_PTS = 3000
    step = max(1, len(dates) // MAX_CHART_PTS)

    ds_idx   = np.arange(0, len(dates), step)          # sampled indices
    ds_dates = dates.iloc[ds_idx]
    ds_close = df.Close.values[ds_idx]
    ds_slow  = df.ema_long.values[ds_idx]
    ds_fast  = df.ema_mid.values[ds_idx]
    ds_entry = df.ema_short.values[ds_idx]

    # Cubic interpolation for silky-smooth EMA curves
    try:
        from scipy.interpolate import make_interp_spline
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    _numx      = mdates.date2num(ds_dates)
    _numx_fine = np.linspace(_numx[0], _numx[-1], MAX_CHART_PTS * 2)
    _fine_dates = mdates.num2date(_numx_fine)

    def _smooth(y_sampled):
        """Cubic B-spline through downsampled points → dense smooth curve."""
        mask = ~np.isnan(y_sampled)
        if not _has_scipy or mask.sum() < 4:
            # Fallback: linear interpolation (still smoother than raw 5-min)
            return _fine_dates, np.interp(_numx_fine, _numx, np.nan_to_num(y_sampled))
        spl = make_interp_spline(_numx[mask], y_sampled[mask], k=3)
        return _fine_dates, spl(_numx_fine)

    sm_dates_slow,  sm_slow  = _smooth(ds_slow)
    sm_dates_fast,  sm_fast  = _smooth(ds_fast)
    sm_dates_entry, sm_entry = _smooth(ds_entry)

    # ── Main chart: Price only (1 panel) ───────────────────────────────────────
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    _style_ax(ax1)

    if is_one_day:
        # ── 1-day: full-resolution OHLC candlestick chart ────────────────────
        # Plot EMA lines first to initialise the datetime x-axis scale
        ax1.plot(dates, df.ema_long.values,  color="#ff6b6b", linewidth=1.4,
                 label=f"EMA {EMA_LONG}", zorder=4)
        ax1.plot(dates, df.ema_mid.values,  color="#ffd93d", linewidth=1.2,
                 label=f"EMA {EMA_MID}", zorder=4)
        ax1.plot(dates, df.ema_short.values, color="#6bcb77", linewidth=1.0,
                 label=f"EMA {EMA_SHORT}", zorder=4)
        # Draw OHLC candlesticks using date-number coordinates
        _dt_nums = mdates.date2num(dates)
        _bw      = ((_dt_nums[1] - _dt_nums[0]) * 0.8) if len(_dt_nums) > 1 else (5 / 1440 * 0.8)
        _opens   = df.Open.values
        _highs   = df.High.values
        _lows    = df.Low.values
        _closes  = df.Close.values
        for _i in range(len(_dt_nums)):
            _o, _h, _l, _c = _opens[_i], _highs[_i], _lows[_i], _closes[_i]
            _clr = "#26a69a" if _c >= _o else "#ef5350"   # green up / red down
            # Wick: high–low line
            ax1.plot([_dt_nums[_i], _dt_nums[_i]], [_l, _h],
                     color=_clr, linewidth=0.6, zorder=2)
            # Body: open–close rectangle
            ax1.add_patch(plt.Rectangle(
                (_dt_nums[_i] - _bw / 2, min(_o, _c)),
                _bw, max(abs(_c - _o), 1e-7),
                facecolor=_clr, edgecolor=_clr, linewidth=0.0, zorder=3
            ))
        # ── Pivot point markers (1-day only) ─────────────────────────────────
        _pvd = compute_pivot_diagnostics(df)
        if _pvd and _pvd.get('is_single_day') and _pvd.get('pivots'):
            _price_range   = _highs.max() - _lows.min()
            _pivot_offset  = max(_price_range * 0.008, 1e-5)
            _pivot_colors  = {
                'CH': '#ef5350', 'CL': '#ef5350',   # red   — consolidation
                'HH': '#6bcb77', 'HL': '#6bcb77',   # green — uptrend structure
                'LH': '#6bcb77', 'LL': '#6bcb77',   # green — downtrend structure
            }
            _label_offset = max(_price_range * 0.018, 2e-5)
            for _pv_idx, _pv in enumerate(_pvd['pivots']):
                _bar_i  = _pv['bar']
                _pv_clr = _pivot_colors.get(_pv['label'], '#ffffff')
                if _pv['kind'] == 'H':
                    _pv_y = _pv['price'] + _pivot_offset
                    _lbl_y = _pv_y + _label_offset
                else:
                    _pv_y = _pv['price'] - _pivot_offset
                    _lbl_y = _pv_y - _label_offset
                ax1.scatter(
                    _dt_nums[_bar_i], _pv_y,
                    color=_pv_clr, marker='o', s=18, zorder=6,
                    edgecolors='none',
                )
                ax1.text(
                    _dt_nums[_bar_i], _lbl_y, str(_pv_idx + 1),
                    color=_pv_clr, fontsize=6, fontweight='bold',
                    ha='center', va='bottom' if _pv['kind'] == 'H' else 'top',
                    zorder=7,
                )

        ax1.set_xlim(
            mdates.num2date(_dt_nums[0]  - _bw),
            mdates.num2date(_dt_nums[-1] + _bw)
        )
    else:
        ax1.plot(ds_dates, ds_close, color="#e0e0e0", linewidth=0.5, label="Price", alpha=0.7)
        ax1.plot(sm_dates_slow,  sm_slow,  color="#ff6b6b", linewidth=1.4, label=f"EMA {EMA_LONG}")
        ax1.plot(sm_dates_fast,  sm_fast,  color="#ffd93d", linewidth=1.2, label=f"EMA {EMA_MID}")
        ax1.plot(sm_dates_entry, sm_entry, color="#6bcb77", linewidth=1.0, label=f"EMA {EMA_SHORT}")

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
            if idx < 0 or idx >= len(dates):
                continue
            color  = "#6bcb77" if t.win else "#ff6b6b"
            marker = "^" if t.direction == "long" else "v"
            entry_date = dates.iloc[idx]
            exit_date  = dates.iloc[eidx] if 0 <= eidx < len(dates) else dates.iloc[-1]
            ax1.scatter(entry_date, t.entry, color=color,
                       marker=marker, s=60, zorder=5)
            ax1.scatter(exit_date, t.exit, color=color,
                       marker="x", s=40, zorder=5)
            ax1.plot([entry_date, exit_date], [t.entry, t.exit],
                     color="#ffffff", linewidth=0.8, alpha=1.0, zorder=4)
            # Stop loss level marker (white em-dash)
            if is_one_day:
                ax1.plot([entry_date, exit_date], [t.stop, t.stop],
                         color="#ffffff", linewidth=1.0, linestyle=(0, (1, 1)),
                         alpha=0.7, zorder=4)

    ax1.set_title(f"{TICKER} — EMA Trend Following Backtest",
                  color="white", fontsize=13, pad=10)
    ax1.legend(loc="upper left", facecolor="#1a1a2e",
               labelcolor="white", fontsize=8)
    ax1.set_ylabel("Price", color="white")
    ax1.set_xlabel("Date", color="white")
    _set_x_fmt(ax1)

    plt.tight_layout(pad=2.0)
    main_path = os.path.join("results", f"{VERSION}_{ticker_clean}_{date_str}.png")
    plt.savefig(main_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Main chart saved → {main_path}")

    # ── Equity & Drawdown chart (2 panels) ───────────────────────────────────
    eq_dates  = dates.iloc[:len(equity)] if len(equity) <= len(dates) else dates
    eq_series = pd.Series(equity[:len(eq_dates)])

    fig_eq, (ax2, ax3) = plt.subplots(2, 1, figsize=(16, 6),
                                       gridspec_kw={"height_ratios": [1, 1]})
    fig_eq.patch.set_facecolor("#1a1a2e")
    _style_ax(ax2)
    _style_ax(ax3)

    # Equity curve
    ax2.plot(eq_dates[:len(eq_series)], eq_series, color="#4cc9f0", linewidth=1.2)
    ax2.axhline(STARTING_CASH, color="#666", linestyle="--", linewidth=0.8)
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series >= STARTING_CASH,
                     alpha=0.2, color="#6bcb77")
    ax2.fill_between(eq_dates[:len(eq_series)], STARTING_CASH,
                     eq_series, where=eq_series < STARTING_CASH,
                     alpha=0.2, color="#ff6b6b")
    ax2.set_title("Equity Curve", color="white", fontsize=12, pad=8)
    ax2.set_ylabel("Equity ($)", color="white")
    _set_x_fmt(ax2)

    # Drawdown
    eq_s = pd.Series(equity)
    peak = eq_s.cummax()
    dd   = (eq_s - peak) / peak * 100
    ax3.fill_between(eq_dates[:len(dd)], dd, 0, color="#ff6b6b", alpha=0.6)
    ax3.set_title("Drawdown", color="white", fontsize=12, pad=8)
    ax3.set_ylabel("Drawdown %", color="white")
    ax3.set_xlabel("Date", color="white")
    _set_x_fmt(ax3)

    plt.tight_layout(pad=2.0)
    eq_dd_path = os.path.join("results", f"{VERSION}_{ticker_clean}_{date_str}_eq_dd.png")
    plt.savefig(eq_dd_path, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  Equity/DD chart saved → {eq_dd_path}")

    return main_path, eq_dd_path

# ── Fractal Diagnostics ───────────────────────────────────────────────────────

def compute_pivot_diagnostics(df):
    """Detect fractal pivot highs/lows and classify market structure.
    Returns None if df is None or spans more than 1 calendar day.
    Only intended for single-day (intraday) date range runs."""
    if df is None or len(df) < 10:
        return None

    # ── Determine if the data spans exactly 1 calendar day ────────────────────
    try:
        dts = pd.to_datetime(df['Datetime'])
        if dts.dt.tz is not None:
            dts_utc = dts.dt.tz_convert('UTC')
        else:
            dts_utc = dts.dt.tz_localize('UTC')
        dates = dts_utc.dt.date
        times_str = dts_utc.dt.strftime('%H:%M')
    except Exception:
        return None

    if dates.min() != dates.max():
        return {"is_single_day": False, "pivots": [], "structure": None}

    # ── Compute ATR(14) using Wilder's smoothing ───────────────────────────────
    _n     = 14
    _alpha = 1.0 / _n
    _cp    = df['Close'].shift(1)
    _tr    = pd.concat([
        (df['High'] - df['Low']),
        (df['High'] - _cp).abs(),
        (df['Low']  - _cp).abs(),
    ], axis=1).max(axis=1)
    atr14  = _tr.ewm(alpha=_alpha, adjust=False).mean()

    # ── Compute EMA(200) for pullback % EMA filter ────────────────────────────
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()

    highs = df['High'].values
    lows  = df['Low'].values
    n_bars = len(df)

    # ── Detect fractal pivots (need 2 bars on each side) ─────────────────────
    raw_pivots = []
    for i in range(2, n_bars - 2):
        h_i = highs[i]
        l_i = lows[i]
        t_str  = times_str.iloc[i]
        atr_i  = float(atr14.iloc[i])
        adx_i  = float(df['adx'].iloc[i]) if 'adx' in df.columns else None

        is_ph = (h_i > highs[i-1] and h_i > highs[i-2] and
                 h_i > highs[i+1] and h_i > highs[i+2])
        is_pl = (l_i < lows[i-1]  and l_i < lows[i-2]  and
                 l_i < lows[i+1]  and l_i < lows[i+2])

        if is_ph:
            raw_pivots.append({'kind': 'H', 'price': float(h_i),
                                'time': t_str, 'bar': i, 'atr': atr_i, 'adx': adx_i})
        if is_pl:
            raw_pivots.append({'kind': 'L', 'price': float(l_i),
                                'time': t_str, 'bar': i, 'atr': atr_i, 'adx': adx_i})

    # Sort by bar index so they appear in chronological order
    raw_pivots.sort(key=lambda x: x['bar'])

    # ── Classify each pivot vs previous same-type pivot ───────────────────────
    classified = []
    prev_high = None
    prev_low  = None

    for pv in raw_pivots:
        price     = pv['price']
        threshold = 0.5 * pv['atr']

        if pv['kind'] == 'H':
            if prev_high is None:
                label      = 'CH'         # first pivot high — no prior to compare, treat as consolidating
                vert_dist  = None
                horiz_dist = None
            else:
                diff = price - prev_high['price']
                if abs(diff) < threshold:
                    label = 'CH'
                elif diff < 0:
                    label = 'LH'
                else:
                    label = 'HH'
                vert_dist  = round(abs(diff) * 10000, 1)   # pips
                horiz_dist = pv['bar'] - prev_high['bar']

            classified.append({
                'label':      label,
                'price':      price,
                'time':       pv['time'],
                'vert_dist':  vert_dist,
                'horiz_dist': horiz_dist,
                'bar':        pv['bar'],
                'kind':       pv['kind'],
                'atr':        round(pv['atr'] * 10000, 1),  # ATR in pips
                'adx':        round(pv['adx'], 1) if pv['adx'] is not None else None,
            })
            prev_high = pv

        else:  # kind == 'L'
            if prev_low is None:
                label      = 'CL'         # first pivot low — no prior to compare, treat as consolidating
                vert_dist  = None
                horiz_dist = None
            else:
                diff = price - prev_low['price']
                if abs(diff) < threshold:
                    label = 'CL'
                elif diff > 0:
                    label = 'HL'
                else:
                    label = 'LL'
                vert_dist  = round(abs(diff) * 10000, 1)
                horiz_dist = pv['bar'] - prev_low['bar']

            classified.append({
                'label':      label,
                'price':      price,
                'time':       pv['time'],
                'vert_dist':  vert_dist,
                'horiz_dist': horiz_dist,
                'bar':        pv['bar'],
                'kind':       pv['kind'],
                'atr':        round(pv['atr'] * 10000, 1),  # ATR in pips
                'adx':        round(pv['adx'], 1) if pv['adx'] is not None else None,
            })
            prev_low = pv

    # ── Compute pullback % for each classified pivot ──────────────────────────
    # Helper: scan backwards through history to find the most recent pivot of a
    # given label, returning its price or None.  No ordering assumptions are made
    # about how high and low pivots interleave chronologically.
    def _last_price(label, history):
        for p in reversed(history):
            if p['label'] == label:
                return p['price']
        return None

    for i, pv in enumerate(classified):
        lbl   = pv['label']
        price = pv['price']
        bidx  = pv['bar']
        prior = classified[:i]   # all pivots that preceded this one

        pb = None  # default → "—" in UI

        if lbl == 'LH':
            # Prior range = most recent LH − most recent LL
            # Pullback    = this LH − most recent LL
            ref_lh = _last_price('LH', prior)
            ref_ll = _last_price('LL', prior)
            if ref_lh is not None and ref_ll is not None:
                prior_range = ref_lh - ref_ll
                pullback    = price  - ref_ll
                if abs(prior_range) > 1e-10:
                    pb = pullback / prior_range * 100

        elif lbl == 'HL':
            # Prior range = most recent HH − most recent HL
            # Pullback    = most recent HH − this HL
            ref_hh = _last_price('HH', prior)
            ref_hl = _last_price('HL', prior)
            if ref_hh is not None and ref_hl is not None:
                prior_range = ref_hh - ref_hl
                pullback    = ref_hh - price
                if abs(prior_range) > 1e-10:
                    pb = pullback / prior_range * 100

        elif lbl == 'HH':
            pass   # continuation pivot — always dash

        elif lbl == 'LL':
            pass   # continuation pivot — always dash

        elif lbl == 'CH':
            # Only calculate when bar's close is BELOW EMA200
            # Prior range = most recent CH − most recent CL
            # Pullback    = this CH − most recent CL
            try:
                close_val = float(df['Close'].iloc[bidx])
                ema_val   = float(ema200.iloc[bidx])
                if close_val < ema_val:
                    ref_ch = _last_price('CH', prior)
                    ref_cl = _last_price('CL', prior)
                    if ref_ch is not None and ref_cl is not None:
                        prior_range = ref_ch - ref_cl
                        pullback    = price  - ref_cl
                        if abs(prior_range) > 1e-10:
                            pb = pullback / prior_range * 100
            except Exception:
                pass

        elif lbl == 'CL':
            # Only calculate when bar's close is ABOVE EMA200
            # Prior range = most recent CH − most recent CL
            # Pullback    = most recent CH − this CL
            try:
                close_val = float(df['Close'].iloc[bidx])
                ema_val   = float(ema200.iloc[bidx])
                if close_val > ema_val:
                    ref_ch = _last_price('CH', prior)
                    ref_cl = _last_price('CL', prior)
                    if ref_ch is not None and ref_cl is not None:
                        prior_range = ref_ch - ref_cl
                        pullback    = ref_ch - price
                        if abs(prior_range) > 1e-10:
                            pb = pullback / prior_range * 100
            except Exception:
                pass

        # Cap to valid range 0–150%; outside this range indicates bad prior data
        if pb is not None and pb < 0:
            pb = None

        pv['pullback_pct'] = round(pb, 1) if pb is not None else None

    # ── Determine market structure from last 3 classified pivots ──────────────
    last_3 = classified[-3:] if len(classified) >= 3 else classified

    structure = "Consolidating"
    if len(last_3) >= 2:
        hh_hl = sum(1 for p in last_3 if p['label'] in ('HH', 'HL'))
        lh_ll = sum(1 for p in last_3 if p['label'] in ('LH', 'LL'))
        ch_cl = sum(1 for p in last_3 if p['label'] in ('CH', 'CL'))
        if hh_hl > lh_ll and hh_hl > ch_cl:
            structure = "Trending Up"
        elif lh_ll > hh_hl and lh_ll > ch_cl:
            structure = "Trending Down"

    return {
        "is_single_day": True,
        "pivots":         classified,
        "structure":      structure,
    }


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

    # ── Daily performance ──────────────────────────────────────────────────────
    daily_perf = []
    try:
        t_daily = trades.copy()
        _td_ts = pd.to_datetime(t_daily["entry_ts"])
        if _td_ts.dt.tz is not None:
            _td_ts = _td_ts.dt.tz_convert("UTC")
        else:
            _td_ts = _td_ts.dt.tz_localize("UTC")
        t_daily["_date"] = _td_ts.dt.strftime("%Y-%m-%d")
        for date_str, grp in t_daily.groupby("_date"):
            grp_w = grp[grp.win]
            grp_l = grp[~grp.win]
            daily_perf.append({
                "date":     date_str,
                "trades":   int(len(grp)),
                "wins":     int(len(grp_w)),
                "losses":   int(len(grp_l)),
                "win_rate": round(len(grp_w) / len(grp) * 100, 1),
                "net_pnl":  round(float(grp.pnl.sum()), 2),
            })
    except Exception:
        daily_perf = []

    # ── Intraday trade list (for single-day ranges) ─────────────────────────────
    intraday_trades = []
    try:
        t_intra = trades.copy()
        _ti_entry = pd.to_datetime(t_intra["entry_ts"])
        _ti_exit  = pd.to_datetime(t_intra["timestamp"])
        if _ti_entry.dt.tz is not None:
            _ti_entry = _ti_entry.dt.tz_convert("UTC")
        else:
            _ti_entry = _ti_entry.dt.tz_localize("UTC")
        if _ti_exit.dt.tz is not None:
            _ti_exit = _ti_exit.dt.tz_convert("UTC")
        else:
            _ti_exit = _ti_exit.dt.tz_localize("UTC")
        t_intra["_date"]       = _ti_entry.dt.strftime("%Y-%m-%d")
        t_intra["_entry_time"] = _ti_entry.dt.strftime("%H:%M")
        t_intra["_exit_time"]  = _ti_exit.dt.strftime("%H:%M")
        t_intra["_duration"]   = ((_ti_exit - _ti_entry).dt.total_seconds() / 60).round(0).astype(int)
        unique_dates = t_intra["_date"].unique()
        if len(unique_dates) == 1:
            for _, row_t in t_intra.iterrows():
                _stop_pips = round(abs(float(row_t["entry"]) - float(row_t["stop"])) * 10000, 1)
                _target_pips = round(abs(float(row_t["entry"]) - float(row_t["target"])) * 10000, 1)
                intraday_trades.append({
                    "date":       row_t["_date"],
                    "entry_time": row_t["_entry_time"],
                    "exit_time":  row_t["_exit_time"],
                    "duration":   int(row_t["_duration"]),
                    "direction":  row_t["direction"],
                    "stop_pips":  _stop_pips,
                    "target_pips": _target_pips,
                    "atr_pips":   round(float(row_t["atr_at_entry"]) * 10000, 1),
                    "adx":        round(float(row_t["adx_at_entry"]), 1),
                    "pnl":        round(float(row_t["pnl"]), 2),
                })
    except Exception:
        intraday_trades = []

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
                ts0 = pd.to_datetime(seg["timestamp"].iloc[0]).strftime("%b-%d-%y")
                ts1 = pd.to_datetime(seg["timestamp"].iloc[-1]).strftime("%b-%d-%y")
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
        "daily":          daily_perf,
        "intraday":       intraday_trades,
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
        "rs_diagnostic":     rs_diagnostic,
        "rrr_sensitivity":   [],
        "swing_sensitivity": [],
        "pivot_diagnostics": None,
    }

    # ── Sensitivity sweeps (requires original df) ──────────────────────────────
    if df is not None:
        try:
            print(f"  Running sensitivity sweeps ({len(df)} bars)...")
            rrr_rows, swing_rows = compute_sensitivity(df)
            result["rrr_sensitivity"]   = rrr_rows
            result["swing_sensitivity"] = swing_rows
            print(f"  Sensitivity sweeps complete.")
        except Exception:
            pass

    # ── Pivot structure diagnostics (single-day ranges only) ──────────────────
    if df is not None:
        try:
            result["pivot_diagnostics"] = compute_pivot_diagnostics(df)
        except Exception:
            pass

    return result


def compute_sensitivity(df):
    """Sweep RRR; return (rrr_rows, swing_rows) lists.
    swing_rows is always empty — swing lookback is no longer used."""
    rrr_rows = []
    for val in [1.5, 2.0, 2.5, 3.0]:
        r = _sensitivity_run(df, rrr=val, swing_lookback=0)
        if r:
            r["param"] = val
            rrr_rows.append(r)

    swing_rows = []   # no longer applicable with fractal-based entries

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
  </div>
</div>

<div id="copy-toast">&#10003;&nbsp; Copied to clipboard!</div>

<!-- Action buttons: hidden by default, moved into the run bar by server.py -->
<span class="rb-sep" id="rb-act-sep" style="display:none;"></span>
<button id="copy-btn" style="display:none;">Copy Version Report</button>
<button id="delete-btn" style="display:none;">Delete Version</button>

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
  var _dragSub         = null; /* drag-and-drop state for sub-item reorder */

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
    return names[idx] + "-" + parts[0].slice(2);
  }

  /* Convert "YYYY-MM-DD HH:MM" to "Mon-DD-YY HH:MM" */
  function fmtRunDate(s) {
    if (!s) return "";
    var _mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    var parts = String(s).split(" ");
    var d = parts[0].split("-");
    if (d.length !== 3) return s;
    return _mn[parseInt(d[1], 10) - 1] + "-" + String(d[2]).padStart(2, "0") + "-" + d[0].slice(2) + (parts[1] ? " " + parts[1] : "");
  }

  /* ── Compat: get run data from version (handles legacy + new format) ──── */
  function getRuns(v) {
    if (v.runs && v.runs.length > 0) return v.runs;
    /* legacy single-run format */
    return [{
      date: v.date || "", start_date: "", end_date: "",
      days_back: (v.params || {}).days_back || 0, label: "",
      notes: v.notes || "\u2014", chart_b64: v.chart_b64 || "",
      eq_dd_chart_b64: v.eq_dd_chart_b64 || "",
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
    var runLabel = run.label ? " \u2014 " + run.label.split(" \u2192 ").map(function(d){return fmtSbDate(d.trim());}).join(" \u2192 ") : "";
    lines.push("## Backtest Report \u2014 " + (ver.name || "?") + runLabel);
    lines.push("");
    lines.push("**Strategy:** " + (ver.strategy || "\u2014"));
    lines.push("**Instrument:** " + (p.ticker || "\u2014") + " \u00b7 " + (p.interval || "\u2014") + " \u00b7 " + (run.days_back || p.days_back || "\u2014") + " days");
    lines.push("**Date:** " + (fmtRunDate(run.date) || "\u2014"));
    if (run.start_date && run.end_date) {
      lines.push("**Range:** " + run.start_date + " \u2192 " + run.end_date);
    }
    if (run.notes && run.notes !== "\u2014" && run.notes !== "\u2014") {
      lines.push("**Notes:** " + run.notes);
    }
    lines.push("");

    /* ════════════════════════════════════════════════════════════
       GENERAL
       ════════════════════════════════════════════════════════════ */
    lines.push("# General");
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
    var _dr = run.start_date && run.end_date ? { start: run.start_date, end: run.end_date } : fullRunRange(run);
    var _drDur = calcDuration(_dr.start, _dr.end);
    var _drDates = _dr.start && _dr.end ? fmtSbDate(_dr.start) + " \u2192 " + fmtSbDate(_dr.end) : "";
    lines.push("| Date Range | " + (_drDur ? _drDur : "") + (_drDur && _drDates ? " \u00b7 " : "") + _drDates + " |");
    lines.push("| Starting Cash | $" + (p.starting_cash || 0).toLocaleString() + " |");
    lines.push("| EMA Short | "     + (p.ema_short     || "\u2014") + " |");
    lines.push("| EMA Mid | "       + (p.ema_mid       || "\u2014") + " |");
    lines.push("| EMA Long | "      + (p.ema_long      || "\u2014") + " |");
    lines.push("| Stop Loss Level | " + (p.stop_loss_pips || "\u2014") + " pips |");
    lines.push("| RRR | 1:"         + (p.rrr || "\u2014") + " |");
    lines.push("| Risk / Trade | "  + ((p.risk_pct || 0) * 100).toFixed(1) + "% |");
    lines.push("| Direction | "     + (p.trade_direction || "both") + " |");
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
        var mLabel = parts.length >= 2 ? mnames[parseInt(parts[1], 10) - 1] + "-" + parts[0].slice(2) : mo.month;
        lines.push("| " + mLabel + " | " + mo.trades + " | " + mo.wins + " | " + mo.losses + " | " + mf(mo.win_rate, 1) + "% | " + mfMoney(mo.net_pnl) + " |");
      });
    }
    lines.push("");

    /* ── Daily Performance (≤ 31 day ranges only) ──────────────── */
    (function () {
      var sd = run.start_date;
      var ed = run.end_date;
      if (!sd || !ed) return;
      var startDt  = new Date(sd + "T00:00:00Z");
      var endDt    = new Date(ed + "T00:00:00Z");
      var totalDays = Math.round((endDt - startDt) / 86400000) + 1;
      if (totalDays > 31) return;

      var dailyData = m.daily || [];
      var dailyLookup = {};
      dailyData.forEach(function (d) { dailyLookup[d.date] = d; });

      lines.push("### Daily Performance");
      lines.push("");
      lines.push("| Date | Trades | Wins | Losses | Win Rate | Net P&L |");
      lines.push("|------|--------|------|--------|----------|---------|");

      var cur = new Date(startDt.getTime());
      while (cur <= endDt) {
        var uy = cur.getUTCFullYear();
        var um = cur.getUTCMonth() + 1;
        var ud = cur.getUTCDate();
        var ds = uy + "-" + (um < 10 ? "0" : "") + um + "-" + (ud < 10 ? "0" : "") + ud;
        var mdMnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
        var dateLabel = mdMnames[um - 1] + "-" + String(ud).padStart(2, "0") + "-" + String(uy).slice(2);
        var d = dailyLookup[ds];
        if (d) {
          lines.push("| " + dateLabel + " | " + d.trades + " | " + d.wins + " | " + d.losses + " | " + mf(d.win_rate, 1) + "% | " + mfMoney(d.net_pnl) + " |");
        } else {
          lines.push("| " + dateLabel + " | 0 | \u2014 | \u2014 | \u2014 | \u2014 |");
        }
        cur.setUTCDate(cur.getUTCDate() + 1);
      }
      lines.push("");
    }());

    /* ── Intraday Performance (single-day ranges only) ──── */
    (function () {
      var intradayData = m.intraday || [];
      if (intradayData.length === 0) return;
      lines.push("### Intraday Performance");
      lines.push("");
      lines.push("| Date | Entry Time | Exit Time | Duration | Trade Direction | Stop (pips) | Target (pips) | P&L |");
      lines.push("|------|------------|-----------|----------|-----------------|-------------|---------------|-----|");
      var mdIMnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      function mdFmtDur(mins) {
        if (mins < 60) return mins + " min";
        return (mins / 60).toFixed(1).replace(".", ",") + " hrs";
      }
      intradayData.forEach(function (t) {
        var dir = t.direction.charAt(0).toUpperCase() + t.direction.slice(1);
        var idp = t.date.split("-");
        var iDateLabel = mdIMnames[parseInt(idp[1], 10) - 1] + "-" + String(idp[2]).padStart(2, "0") + "-" + idp[0].slice(2);
        var stopD   = (t.stop_pips   !== null && t.stop_pips   !== undefined) ? mf(t.stop_pips, 1)   : "\u2014";
        var targetD = (t.target_pips !== null && t.target_pips !== undefined) ? mf(t.target_pips, 1) : "\u2014";
        lines.push("| " + iDateLabel + " | " + t.entry_time + " UTC | " + t.exit_time + " UTC | " + mdFmtDur(t.duration) + " | " + dir + " | " + stopD + " | " + targetD + " | " + mfMoney(t.pnl) + " |");
      });
      lines.push("");
    }());

    /* ── Time of Day Performance ──── */
    (function () {
      var tod = m.time_of_day || { rows: [] };
      var todList = tod.rows || [];
      if (todList.length === 0) return;
      lines.push("### Time of Day Performance");
      lines.push("");
      lines.push("| Hour (UTC) | Trades | Win Rate | Net P&L |");
      lines.push("|------------|--------|----------|---------|");
      todList.forEach(function (r) {
        lines.push("| " + r.hour + ":00 | " + r.trades + " | " + mf(r.win_rate, 1) + "% | " + mfMoney(r.net_pnl) + " |");
      });
      lines.push("");
    }());

    /* ── Fractal Diagnostics (single-day ranges only) ──── */
    var pvd = m.pivot_diagnostics;
    if (pvd && pvd.is_single_day) {
      lines.push("### Fractal Diagnostics");
      lines.push("");
      var pvList = pvd.pivots || [];
      if (pvList.length === 0) {
        lines.push("No fractal pivot points detected in this date range.");
      } else {
        lines.push("| # | Type | Price | Time | ATR (pips) | ADX | Vert Distance (pips) | Horiz Distance (bars) | Pullback % |");
        lines.push("|---|------|-------|------|------------|-----|----------------------|-----------------------|------------|");
        pvList.forEach(function (pv, idx) {
          var vertD    = (pv.vert_dist    !== null && pv.vert_dist    !== undefined) ? mf(pv.vert_dist, 1) : "\u2014";
          var horizD   = (pv.horiz_dist   !== null && pv.horiz_dist   !== undefined) ? String(pv.horiz_dist) : "\u2014";
          var pullbackD = (pv.pullback_pct !== null && pv.pullback_pct !== undefined) ? mf(pv.pullback_pct, 1) + "%" : "\u2014";
          var atrD     = (pv.atr          !== null && pv.atr          !== undefined) ? mf(pv.atr, 1) : "\u2014";
          var adxD     = (pv.adx          !== null && pv.adx          !== undefined) ? mf(pv.adx, 1) : "\u2014";
          lines.push("| " + (idx + 1) + " | " + (pv.label || "\u2014") + " | " +
            mf(pv.price, 5) + " | " + (pv.time || "\u2014") + " | " + atrD + " | " + adxD + " | " + vertD + " | " + horizD + " | " + pullbackD + " |");
        });
      }
      lines.push("");
    }

    /* ════════════════════════════════════════════════════════════
       ADVANCED
       ════════════════════════════════════════════════════════════ */
    lines.push("# Advanced");
    lines.push("");

    /* ── Range Filter ──── */
    (function () {
      var rsd = m.rs_diagnostic || {};
      if (!rsd.total_signals || rsd.total_signals <= 0) return;
      var filteredPfTxt = (rsd.filtered_pf === null || rsd.filtered_pf === undefined) ? "\u221e" : mf(rsd.filtered_pf);
      var allowedPfTxt  = (rsd.allowed_pf === null || rsd.allowed_pf === undefined) ? "\u221e" : mf(rsd.allowed_pf);
      lines.push("### Range Filter");
      lines.push("");
      lines.push("| Metric | Filtered (Blocked) | Allowed (Executed) |");
      lines.push("|--------|--------------------|--------------------|");
      lines.push("| Trades | " + rsd.trades_filtered + " | " + rsd.trades_allowed + " |");
      lines.push("| Win Rate | " + mf(rsd.filtered_wr, 1) + "% | " + mf(rsd.allowed_wr, 1) + "% |");
      lines.push("| Profit Factor | " + filteredPfTxt + " | " + allowedPfTxt + " |");
      lines.push("");
      lines.push("**Filter Rate:** " + mf(rsd.filter_rate, 1) + "% (" + rsd.trades_filtered + " of " + rsd.total_signals + " signals blocked)");
      lines.push("");
    }());

    /* ── Regime Classification ──── */
    (function () {
      var regime = m.regime || [];
      if (regime.length === 0) return;
      lines.push("### Regime Classification");
      lines.push("");
      lines.push("| Regime | Trades | Win Rate | Profit Factor | Net P&L |");
      lines.push("|--------|--------|----------|---------------|---------|");
      regime.forEach(function (r) {
        var rpf = (r.profit_factor === null || r.profit_factor === undefined) ? "\u221e" : mf(r.profit_factor);
        lines.push("| " + r.regime + " | " + r.count + " | " + mf(r.win_rate, 1) + "% | " + rpf + " | " + mfMoney(r.net_pnl) + " |");
      });
      lines.push("");
    }());

    /* ── Streak Analysis ──── */
    (function () {
      var str = m.streaks || {};
      lines.push("### Streak Analysis");
      lines.push("");
      lines.push("| Metric | Value |");
      lines.push("|--------|-------|");
      lines.push("| Max Win Streak | "  + (str.max_win_streak  !== undefined ? str.max_win_streak  + " trades" : "\u2014") + " |");
      lines.push("| Max Loss Streak | " + (str.max_loss_streak !== undefined ? str.max_loss_streak + " trades" : "\u2014") + " |");
      lines.push("| Avg Win Streak | "  + (str.avg_win_streak  !== undefined ? str.avg_win_streak  : "\u2014") + " |");
      lines.push("| Avg Loss Streak | " + (str.avg_loss_streak !== undefined ? str.avg_loss_streak : "\u2014") + " |");
      lines.push("| Current Streak | "  + (str.current_streak  || "\u2014") + " |");
      lines.push("");
    }());

    /* ── Stop vs Target ──── */
    (function () {
      var st = m.stop_target || {};
      lines.push("### Stop vs Target");
      lines.push("");
      lines.push("| Metric | Value |");
      lines.push("|--------|-------|");
      lines.push("| Hit Stop Loss | "   + ((st.pct_sl  !== null && st.pct_sl  !== undefined) ? st.pct_sl  + "%" : "\u2014") + " |");
      lines.push("| Hit Take Profit | " + ((st.pct_tp  !== null && st.pct_tp  !== undefined) ? st.pct_tp  + "%" : "\u2014") + " |");
      lines.push("| Avg MAE | "         + ((st.avg_mae !== null && st.avg_mae !== undefined) ? st.avg_mae + " pips" : "\u2014") + " |");
      lines.push("");
    }());

    /* ── Win Rate Trend ──── */
    (function () {
      var wrt = m.win_rate_trend || [];
      if (wrt.length === 0) return;
      lines.push("### Win Rate Trend (3 equal periods)");
      lines.push("");
      lines.push("| Period | Trades | Win Rate | Profit Factor | Net P&L |");
      lines.push("|--------|--------|----------|---------------|---------|");
      wrt.forEach(function (w) {
        var wpf = (w.profit_factor === null || w.profit_factor === undefined) ? "\u221e" : mf(w.profit_factor);
        lines.push("| " + w.label + " | " + w.trades + " | " + mf(w.win_rate, 1) + "% | " + wpf + " | " + mfMoney(w.net_pnl) + " |");
      });
      lines.push("");
    }());

    /* ── Trade Duration ──── */
    (function () {
      var dur = m.duration_analysis || {};
      lines.push("### Trade Duration");
      lines.push("");
      lines.push("| Group | Avg | Min | Max |");
      lines.push("|-------|-----|-----|-----|");
      [["Winners", dur.winners], ["Losers", dur.losers], ["All Trades", dur.all]].forEach(function (pair) {
        var label = pair[0], d = pair[1];
        if (!d || d.avg === null) {
          lines.push("| " + label + " | \u2014 | \u2014 | \u2014 |");
        } else {
          lines.push("| " + label + " | " + d.avg + " bars | " + d.min + " bars | " + d.max + " bars |");
        }
      });
      lines.push("");
    }());

    /* ── Filter Impact Summary ──── */
    (function () {
      var fi = m.filter_impact || [];
      if (fi.length === 0) return;
      lines.push("### Filter Impact Summary");
      lines.push("");
      lines.push("| Filter | Trades Removed | Win Rate (if kept) | Net P&L (if kept) |");
      lines.push("|--------|----------------|--------------------|--------------------|");
      fi.forEach(function (f) {
        lines.push("| " + f.filter + " | " + f.removed + " | " + mf(f.win_rate, 1) + "% | " + mfMoney(f.net_pnl) + " |");
      });
      lines.push("");
    }());

    /* ── Daily Drawdown — Worst 5 Days ──── */
    (function () {
      var ddDays = m.daily_drawdown || [];
      if (ddDays.length === 0) return;
      lines.push("### Daily Drawdown \u2014 Worst 5 Days");
      lines.push("");
      lines.push("| Date (UTC) | Trades | P&L | Drawdown % |");
      lines.push("|------------|--------|-----|------------|");
      ddDays.forEach(function (d) {
        lines.push("| " + d.date + " | " + d.trades + " | -$" + commaFmt(d.pnl) + " | " + mf(d.drawdown_pct, 2) + "% |");
      });
      lines.push("");
    }());

    /* ── RRR Sensitivity ──── */
    (function () {
      var rrrSens = m.rrr_sensitivity || [];
      if (rrrSens.length === 0) return;
      lines.push("### RRR Sensitivity");
      lines.push("");
      lines.push("| RRR | Trades | Win Rate | Profit Factor | Net P&L | Max DD |");
      lines.push("|-----|--------|----------|---------------|---------|--------|");
      rrrSens.forEach(function (r) {
        var rpfTxt = (r.profit_factor !== null && r.profit_factor !== undefined) ? mf(r.profit_factor) : "\u221e";
        lines.push("| 1:" + r.param.toFixed(1) + " | " + r.trades + " | " + mf(r.win_rate, 1) + "% | " + rpfTxt + " | " + mfMoney(r.net_pnl) + " | " + mf(r.max_drawdown, 1) + "% |");
      });
      lines.push("");
    }());

    /* ── Swing Lookback Sensitivity ──── */
    (function () {
      var swingSens = m.swing_sensitivity || [];
      if (swingSens.length === 0) return;
      lines.push("### Swing Lookback Sensitivity");
      lines.push("");
      lines.push("| Swing Bars | Trades | Win Rate | Profit Factor | Net P&L | Max DD |");
      lines.push("|------------|--------|----------|---------------|---------|--------|");
      swingSens.forEach(function (r) {
        var rpfTxt = (r.profit_factor !== null && r.profit_factor !== undefined) ? mf(r.profit_factor) : "\u221e";
        lines.push("| " + r.param + " bars | " + r.trades + " | " + mf(r.win_rate, 1) + "% | " + rpfTxt + " | " + mfMoney(r.net_pnl) + " | " + mf(r.max_drawdown, 1) + "% |");
      });
      lines.push("");
    }());

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
    var _mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    var p = s.slice(0, 10).split("-");
    if (p.length < 3) return s;
    return _mn[parseInt(p[1], 10) - 1] + "-" + String(p[2]).padStart(2, "0") + "-" + p[0].slice(2);
  }
  /* ── Set date pickers from clickable table cells ─────────── */
  window.setDatePicker = function (startDate, endDate) {
    var startEl = document.getElementById("rb-start");
    var endEl   = document.getElementById("rb-end");
    if (!startEl || !endEl) return;
    startEl.value = startDate;
    endEl.value   = endDate;
    startEl.dispatchEvent(new Event("change"));
    endEl.dispatchEvent(new Event("change"));
  };

  function calcDuration(startStr, endStr) {
    if (!startStr || !endStr) return "";
    var s = new Date(startStr.slice(0,10) + "T00:00:00");
    var e = new Date(endStr.slice(0,10) + "T00:00:00");
    var days = Math.round((e - s) / 86400000) + 1;
    if (days < 1) days = 1;
    var months = days / 30.44;
    if (months >= 12) {
      var yrs = months / 12;
      return yrs.toFixed(1) + " years";
    } else if (months >= 1) {
      return months.toFixed(1) + " months";
    } else {
      return days + (days === 1 ? " day" : " days");
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

      var vInstrument = firstRun.instrument || (v.params && v.params.ticker ? v.params.ticker.replace(/=X$/i, "") : "");
      vItem.innerHTML =
        "<div class='v-item-row'>" +
          "<div class='v-item-content'>" +
            "<div class='v-name'>" + esc(v.name) + "</div>" +
            (vInstrument ? "<div class='v-instrument'>" + esc(vInstrument) + "</div>" : "") +
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
          subItem.draggable = true;

          var subRange = run.start_date && run.end_date
            ? { start: run.start_date, end: run.end_date }
            : fullRunRange(run);
          var subDateRange = subRange.start && subRange.end
            ? fmtSbDate(subRange.start) + " \u2192 " + fmtSbDate(subRange.end) : "";
          var subDur = calcDuration(subRange.start, subRange.end);

          var runInstrument = run.instrument || "";
          subItem.innerHTML =
            "<div class='v-sub-top-row'>" +
              (runInstrument ? "<span class='v-instrument'>" + esc(runInstrument) + "</span>" : "") +
              "<button class='v-sub-delete-btn' title='Delete date range'>&times;</button>" +
            "</div>" +
            (runPnl !== null ? "<div class='v-pnl " + runPc + "'>" + runPtxt + "</div>" : "") +
            (subDateRange ? "<div class='v-date v-sub-name'>" + esc(subDateRange) + "</div>" : "") +
            (subDur ? "<div class='v-duration'>" + esc(subDur) + "</div>" : "");

          (function (el, vIdx, rIdx, verName) {
            el.addEventListener("click", function (e) {
              e.stopPropagation();
              devLogOpen = false;
              document.getElementById("devlog-btn").classList.remove("active");
              activeVersionIdx = vIdx;
              activeRunIdx = rIdx;
              renderSidebar();
              renderContent(vIdx, rIdx);
            });
            /* Wire inline delete button */
            var delBtn = el.querySelector(".v-sub-delete-btn");
            if (delBtn) delBtn.addEventListener("click", function (e) {
              e.stopPropagation();
              if (!confirm("Delete this date range run?")) return;
              delBtn.disabled = true;
              fetch("/delete_run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: verName, run_idx: rIdx })
              })
              .then(function (r) { return r.json(); })
              .then(function (data) {
                if (data.ok) {
                  localStorage.setItem("rb_pending_delete_version", verName);
                  var focusIdx = rIdx - 1;
                  localStorage.setItem("rb_pending_delete_run_idx", String(focusIdx < 1 ? 0 : focusIdx));
                  window.location.reload();
                } else {
                  delBtn.disabled = false;
                  alert("Delete failed: " + (data.error || "Unknown error"));
                }
              })
              .catch(function () {
                delBtn.disabled = false;
                alert("Delete failed — is the server running?");
              });
            });

            /* ── Drag-and-drop reorder ── */
            el.addEventListener("dragstart", function (e) {
              e.stopPropagation();
              _dragSub = { vIdx: vIdx, runIdx: rIdx, verName: verName };
              el.classList.add("dragging");
              e.dataTransfer.effectAllowed = "move";
              e.dataTransfer.setData("text/plain", ""); /* required for Firefox */
            });
            el.addEventListener("dragend", function () {
              el.classList.remove("dragging");
              _dragSub = null;
              /* Clear any lingering drop indicators */
              document.querySelectorAll(".v-sub-item.drag-over-above, .v-sub-item.drag-over-below").forEach(function (x) {
                x.classList.remove("drag-over-above", "drag-over-below");
              });
            });
            el.addEventListener("dragover", function (e) {
              if (!_dragSub || _dragSub.vIdx !== vIdx) return;
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
              /* Show indicator above or below based on cursor position */
              var rect = el.getBoundingClientRect();
              var mid = rect.top + rect.height / 2;
              if (e.clientY < mid) {
                el.classList.add("drag-over-above");
                el.classList.remove("drag-over-below");
              } else {
                el.classList.add("drag-over-below");
                el.classList.remove("drag-over-above");
              }
            });
            el.addEventListener("dragleave", function () {
              el.classList.remove("drag-over-above", "drag-over-below");
            });
            el.addEventListener("drop", function (e) {
              e.preventDefault();
              el.classList.remove("drag-over-above", "drag-over-below");
              if (!_dragSub || _dragSub.vIdx !== vIdx) return;
              var fromIdx = _dragSub.runIdx;
              var rect = el.getBoundingClientRect();
              var mid = rect.top + rect.height / 2;
              var toIdx = e.clientY < mid ? rIdx : rIdx + 1;
              if (toIdx > fromIdx) toIdx--;  /* adjust for removal */
              if (fromIdx === toIdx) return;

              /* Build new order: index 0 (full run) stays fixed, sub-runs reorder */
              var runs = getRuns(VERSIONS[vIdx]);
              var order = [0]; /* index 0 is always the full run */
              var subOrder = [];
              for (var oi = 1; oi < runs.length; oi++) subOrder.push(oi);
              /* Move within subOrder */
              var subFrom = fromIdx - 1;
              var subTo = toIdx - 1;
              var moved = subOrder.splice(subFrom, 1)[0];
              subOrder.splice(subTo, 0, moved);
              for (var oj = 0; oj < subOrder.length; oj++) order.push(subOrder[oj]);

              /* Persist to server */
              fetch("/reorder_runs", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: verName, order: order })
              })
              .then(function (r) { return r.json(); })
              .then(function (data) {
                if (data.ok) {
                  /* Update local data and re-render without full reload */
                  var oldRuns = VERSIONS[vIdx].runs;
                  VERSIONS[vIdx].runs = [];
                  for (var ri = 0; ri < order.length; ri++) VERSIONS[vIdx].runs.push(oldRuns[order[ri]]);
                  /* Keep the dragged item active */
                  activeVersionIdx = vIdx;
                  activeRunIdx = toIdx; /* toIdx is already 1-based (sub indices start at 1) */
                  renderSidebar();
                  renderContent(vIdx, activeRunIdx);
                } else {
                  alert("Reorder failed: " + (data.error || "Unknown error"));
                }
              })
              .catch(function () {
                alert("Reorder failed — is the server running?");
              });
            });
          })(subItem, idx, si, VERSIONS[idx].name);

          list.appendChild(subItem);
        }
      }
    }

    /* Expose the currently active version name globally for the run-bar */
    var curV = VERSIONS[activeVersionIdx];
    window._currentVersionName = curV ? curV.name : "";
    if (typeof updateRangeButtonLabel === "function") updateRangeButtonLabel();
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
      var mParts = String(mo.month).split("-");
      var mY = parseInt(mParts[0], 10);
      var mM = parseInt(mParts[1], 10);
      var mStart = mo.month + "-01";
      var mLast  = new Date(Date.UTC(mY, mM, 0)).getUTCDate();
      var mEnd   = mo.month + "-" + String(mLast).padStart(2, "0");
      mRows +=
        "<tr>" +
        "<td><span class='date-link' onclick=\"setDatePicker('" + mStart + "','" + mEnd + "')\">" + fmtMonth(mo.month) + "</span></td>" +
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

    /* 2b. Daily Performance (≤ 31 day ranges only) */
    var dailyPerfHtml = "";
    (function () {
      var sd = run.start_date;
      var ed = run.end_date;
      if (!sd || !ed) return;
      var startDt = new Date(sd + "T00:00:00Z");
      var endDt   = new Date(ed + "T00:00:00Z");
      var totalDays = Math.round((endDt - startDt) / 86400000) + 1;
      if (totalDays > 31) return;

      var dailyData = m.daily || [];
      var dailyLookup = {};
      dailyData.forEach(function (d) { dailyLookup[d.date] = d; });

      var dRows = "";
      var cur = new Date(startDt.getTime());
      while (cur <= endDt) {
        var uy  = cur.getUTCFullYear();
        var um  = cur.getUTCMonth() + 1;
        var ud  = cur.getUTCDate();
        var ds  = uy + "-" + (um < 10 ? "0" : "") + um + "-" + (ud < 10 ? "0" : "") + ud;
        var dMnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
        var dateLabel = dMnames[um - 1] + "-" + String(ud).padStart(2, "0") + "-" + String(uy).slice(2);
        var dateLinkTd = "<td><span class='date-link' onclick=\"setDatePicker('" + ds + "','" + ds + "')\">" + dateLabel + "</span></td>";
        var d   = dailyLookup[ds];
        if (d) {
          var dPnlCls = d.net_pnl >= 0 ? "mo-pnl-pos" : "mo-pnl-neg";
          dRows +=
            "<tr>" +
            dateLinkTd +
            "<td>" + d.trades + "</td>" +
            "<td class='pos'>" + d.wins + "</td>" +
            "<td class='neg'>" + d.losses + "</td>" +
            "<td class='" + (d.win_rate >= 50 ? "pos" : "neg") + "'>" + fmt(d.win_rate, 1) + "%</td>" +
            "<td class='" + dPnlCls + "'>" + fmtMoney(d.net_pnl) + "</td>" +
            "</tr>";
        } else {
          dRows +=
            "<tr>" +
            dateLinkTd +
            "<td>0</td>" +
            "<td>\u2014</td>" +
            "<td>\u2014</td>" +
            "<td>\u2014</td>" +
            "<td>\u2014</td>" +
            "</tr>";
        }
        cur.setUTCDate(cur.getUTCDate() + 1);
      }

      dailyPerfHtml =
        "<div class='section' id='anchor-daily-perf'>" +
          "<div class='section-title'>Daily Performance</div>" +
          "<table><thead><tr>" +
          "<th>Date</th><th>Trades</th><th>Wins</th><th>Losses</th>" +
          "<th>Win Rate</th><th>Net P&amp;L</th>" +
          "</tr></thead><tbody>" + dRows + "</tbody></table></div>";
    }());

    /* 2c. Intraday Performance (single-day ranges only) */
    var intradayPerfHtml = "";
    (function () {
      var intradayData = m.intraday || [];
      if (intradayData.length === 0) return;

      var iMnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      function fmtDur(mins) {
        if (mins < 60) return mins + " min";
        return (mins / 60).toFixed(1).replace(".", ",") + " hrs";
      }
      var iRows = "";
      intradayData.forEach(function (t) {
        var pnlCls = t.pnl >= 0 ? "mo-pnl-pos" : "mo-pnl-neg";
        var dirCls = t.direction === "short" ? "intraday-dir-short" : "intraday-dir-long";
        var dp = t.date.split("-");
        var iDateLabel = iMnames[parseInt(dp[1], 10) - 1] + "-" + String(dp[2]).padStart(2, "0") + "-" + dp[0].slice(2);
        var stopD   = (t.stop_pips   !== null && t.stop_pips   !== undefined) ? fmt(t.stop_pips, 1)   : "\u2014";
        var targetD = (t.target_pips !== null && t.target_pips !== undefined) ? fmt(t.target_pips, 1) : "\u2014";
        var atrD = (t.atr_pips !== null && t.atr_pips !== undefined) ? fmt(t.atr_pips, 1) : "\u2014";
        var adxD = (t.adx     !== null && t.adx     !== undefined) ? fmt(t.adx, 1)      : "\u2014";
        iRows +=
          "<tr>" +
          "<td>" + esc(iDateLabel) + "</td>" +
          "<td>" + esc(t.entry_time) + " UTC</td>" +
          "<td>" + esc(t.exit_time) + " UTC</td>" +
          "<td>" + fmtDur(t.duration) + "</td>" +
          "<td class='" + dirCls + "'>" + esc(t.direction.charAt(0).toUpperCase() + t.direction.slice(1)) + "</td>" +
          "<td>" + stopD + "</td>" +
          "<td>" + targetD + "</td>" +
          "<td>" + atrD + "</td>" +
          "<td>" + adxD + "</td>" +
          "<td class='" + pnlCls + "'>" + fmtMoney(t.pnl) + "</td>" +
          "</tr>";
      });

      intradayPerfHtml =
        "<div class='section' id='anchor-intraday-perf'>" +
          "<div class='section-title'>Intraday Performance</div>" +
          "<table><thead><tr>" +
          "<th>Date</th><th>Entry Time</th><th>Exit Time</th><th>Duration</th><th>Trade Direction</th><th>Stop (pips)</th><th>Target (pips)</th><th>ATR (pips)</th><th>ADX</th><th>P&amp;L</th>" +
          "</tr></thead><tbody>" + iRows + "</tbody></table></div>";
    }());

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

    /* ── Fractal Diagnostics (single-day date ranges only) ──────────── */
    var pivotDiagHtml = "";
    (function () {
      var pvd = m.pivot_diagnostics;
      if (!pvd || !pvd.is_single_day) return;

      var pvRows = "";
      var pivotList = pvd.pivots || [];

      if (pivotList.length === 0) {
        pivotDiagHtml =
          "<div class='section' id='anchor-fractal-diag'>" +
            "<div class='section-title'>Fractal Diagnostics</div>" +
            "<div style='padding:10px;color:var(--text-dim,#888);font-size:13px;'>" +
              "No fractal pivot points detected in this date range." +
            "</div>" +
          "</div>";
        return;
      }

      pivotList.forEach(function (pv, idx) {
        var lbl  = pv.label || "\u2014";
        var bgClass = "";
        if (lbl === "CH" || lbl === "CL") {
          bgClass = " class='fractal-row-consolidation'";
        } else if (lbl === "LH" || lbl === "LL" || lbl === "HH" || lbl === "HL") {
          bgClass = " class='fractal-row-directional'";
        }
        var vertD    = (pv.vert_dist    !== null && pv.vert_dist    !== undefined) ? fmt(pv.vert_dist, 1)  : "\u2014";
        var horizD   = (pv.horiz_dist   !== null && pv.horiz_dist   !== undefined) ? pv.horiz_dist : "\u2014";
        var pullbackD = (pv.pullback_pct !== null && pv.pullback_pct !== undefined) ? fmt(pv.pullback_pct, 1) + "%" : "\u2014";
        var atrD     = (pv.atr          !== null && pv.atr          !== undefined) ? fmt(pv.atr, 1) : "\u2014";
        var adxD     = (pv.adx          !== null && pv.adx          !== undefined) ? fmt(pv.adx, 1) : "\u2014";
        pvRows +=
          "<tr" + bgClass + ">" +
          "<td>" + (idx + 1) + "</td>" +
          "<td><strong>" + esc(lbl) + "</strong></td>" +
          "<td class='nowrap'>" + fmt(pv.price, 5) + "</td>" +
          "<td class='nowrap'>" + esc(pv.time || "\u2014") + "</td>" +
          "<td>" + atrD + "</td>" +
          "<td>" + adxD + "</td>" +
          "<td>" + vertD + "</td>" +
          "<td>" + horizD + "</td>" +
          "<td>" + pullbackD + "</td>" +
          "</tr>";
      });

      var structure  = pvd.structure || "Consolidating";
      var structCls  = structure === "Trending Up"   ? "pos"
                     : structure === "Trending Down" ? "neg"
                     : "neu";

      pivotDiagHtml =
        "<div class='section' id='anchor-fractal-diag'>" +
          "<div class='section-title'>Fractal Diagnostics</div>" +
          "<table><thead><tr>" +
          "<th style='width:36px'>#</th>" +
          "<th>Type</th>" +
          "<th>Price</th>" +
          "<th>Time</th>" +
          "<th>ATR (pips)</th>" +
          "<th>ADX</th>" +
          "<th>Vert Distance (pips)</th>" +
          "<th>Horiz Distance (bars)</th>" +
          "<th>Pullback %</th>" +
          "</tr></thead><tbody>" + pvRows + "</tbody></table>" +
        "</div>";
    }());

    var chartHtml = run.chart_b64
      ? "<div class='section' id='anchor-chart'><div class='section-title'>Chart</div>" +
        "<img id='chart-img' src='data:image/png;base64," + run.chart_b64 + "' alt='Backtest Chart'/></div>"
      : "";

    var eqDdChartHtml = run.eq_dd_chart_b64
      ? "<div class='section'><div class='section-title'>Equity and Drawdown</div>" +
        "<img id='eq-dd-chart-img' src='data:image/png;base64," + run.eq_dd_chart_b64 + "' alt='Equity and Drawdown'/></div>"
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
    var dirOptions = [
      { value: "short_only", label: "Short only" },
      { value: "long_only",  label: "Long only" },
      { value: "both",       label: "Both" }
    ];
    var savedDir = run.trade_direction || p.trade_direction || "short_only";
    var dirSelectHtml = "<select id='ec-direction-select' class='ec-select'>" +
      dirOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === savedDir ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var instrOptions = [
      { value: "EURUSD", label: "EURUSD" },
      { value: "GBPUSD", label: "GBPUSD" }
    ];
    var savedInstr = (run.instrument || p.ticker || "EURUSD").replace(/=X$/i, "");
    var instrSelectHtml = "<select id='ec-instrument-select' class='ec-select'>" +
      instrOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === savedInstr ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var intervalOptions = [
      { value: "1m", label: "1m" },
      { value: "5m", label: "5m" },
      { value: "15m", label: "15m" },
      { value: "30m", label: "30m" },
      { value: "60m", label: "60m" }
    ];
    var savedInterval = run.interval || p.interval || "5m";
    var intervalSelectHtml = "<select id='ec-interval-select' class='ec-select'>" +
      intervalOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === savedInterval ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var savedEmaShort = run.ema_short || p.ema_short || 8;
    var savedEmaMid   = run.ema_mid   || p.ema_mid   || 20;
    var savedEmaLong  = run.ema_long  || p.ema_long  || 40;
    var emaShortHtml = "<input id='ec-ema-short' type='number' class='ec-input' value='" + savedEmaShort + "' min='1' step='1'>";
    var emaMidHtml   = "<input id='ec-ema-mid'   type='number' class='ec-input' value='" + savedEmaMid   + "' min='1' step='1'>";
    var emaLongHtml  = "<input id='ec-ema-long'  type='number' class='ec-input' value='" + savedEmaLong  + "' min='1' step='1'>";

    var savedStopPips  = run.stop_loss_pips || p.stop_loss_pips || 15;
    var stopPipsHtml   = "<input id='ec-stop-pips' type='number' class='ec-input' value='" + savedStopPips + "' min='1' step='1'>";

    var savedRrrRisk   = run.rrr_risk   || p.rrr_risk   || 1;
    var savedRrrReward = run.rrr_reward || p.rrr_reward || 2;
    var rrrOpts = [1, 2, 3, 4, 5];
    var rrrRiskHtml = "<select id='ec-rrr-risk' class='ec-select ec-select-narrow'>" +
      rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (n === savedRrrRisk ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var rrrRewardHtml = "<select id='ec-rrr-reward' class='ec-select ec-select-narrow'>" +
      rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (n === savedRrrReward ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var rrrSelectHtml = rrrRiskHtml + "<span class='ec-rrr-colon'>:</span>" + rrrRewardHtml;

    if (ecData && ecData.length > 0) {
      var ecRows = ecData.map(function(ec) {
        var ruleCell = ec.condition === "Direction"
          ? dirSelectHtml
          : ec.condition === "Instrument"
          ? instrSelectHtml
          : ec.condition === "Interval"
          ? intervalSelectHtml
          : ec.condition === "EMA Short"
          ? emaShortHtml
          : ec.condition === "EMA Mid"
          ? emaMidHtml
          : ec.condition === "EMA Long"
          ? emaLongHtml
          : ec.condition === "Stop Loss Level"
          ? stopPipsHtml
          : ec.condition === "RRR"
          ? rrrSelectHtml
          : esc(ec.rule);
        return "<tr>" +
          "<td class='ec-td-cond'>" + esc(ec.condition) + "</td>" +
          "<td class='ec-td-rule'>" + ruleCell + "</td>" +
          "</tr>";
      }).join("");
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Entry Conditions</div>" +
          "<table>" +
            "<tbody>" + ecRows + "</tbody>" +
          "</table>" +
        "</div>";
    } else {
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Entry Conditions</div>" +
          "<table>" +
            "<tbody>" +
            "<tr><td class='ec-td-cond'>Instrument</td><td class='ec-td-rule'>" + instrSelectHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>Interval</td><td class='ec-td-rule'>" + intervalSelectHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>EMA Short</td><td class='ec-td-rule'>" + emaShortHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>EMA Mid</td><td class='ec-td-rule'>" + emaMidHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>EMA Long</td><td class='ec-td-rule'>" + emaLongHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>Stop Loss Level</td><td class='ec-td-rule'>" + stopPipsHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>Direction</td><td class='ec-td-rule'>" + dirSelectHtml + "</td></tr>" +
            "<tr><td class='ec-td-cond'>RRR</td><td class='ec-td-rule'>" + rrrSelectHtml + "</td></tr>" +
            "</tbody>" +
          "</table>" +
        "</div>";
    }

    document.getElementById("content").innerHTML =
      /* header */
      "<div id='v-header'>" +
        "<div id='v-header-top'>" +
          "<h2>" + esc(v.name) + stratBadge +
            "<span class='report-tabs'>" +
              "<button class='report-tab active' data-tab='general'>General</button>" +
              "<button class='report-tab' data-tab='advanced'>Advanced</button>" +
            "</span>" +
            "<button class='ec-toggle-btn' id='ec-toggle-btn' title='Entry Conditions'>" +
              "<svg width='16' height='16' viewBox='0 0 16 16' fill='none'>" +
                "<circle cx='8' cy='8' r='7' stroke='currentColor' stroke-width='1.5'/>" +
                "<path d='M5.5 7L8 9.5L10.5 7' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>" +
              "</svg>" +
            "</button>" +
          "</h2>" +
        "</div>" +
        "<div class='ec-collapsible' id='ec-collapsible'>" + entryCondHtml + "</div>" +
      "</div>" +

      /* ── TAB: General ──────────────────────────────────────────────────────── */
      "<div class='tab-content active' data-tab-content='general'>" +

      /* ── Section 1: Results + Parameters ──────────────────────────────────── */
      "<div class='two-col'>" +

        "<div class='section'>" +
          "<div class='section-title'>Results</div>" +
          "<table><tbody>" +
          row("Net Profit",    "<span class='" + pClass(m.net_profit) + "'>" +
            fmtMoney(m.net_profit) + " (" + (m.net_profit_pct >= 0 ? "+" : "") + fmt(m.net_profit_pct, 1) + "%)</span>") +
          row("Total Trades",  m.total_trades) +
          row("Winning Trades", m.winning_trades) +
          row("Losing Trades",  m.losing_trades) +
          row("Win Rate",      "<span class='" + winRateCls + "'>" + fmt(m.win_rate, 1) + "%</span>") +
          row("Profit Factor", "<span class='" + pfCls + "'>" + pfTxt + "</span>") +
          row("Avg Win",       avgWinHtml) +
          row("Avg Loss",      avgLossHtml) +
          row("Avg Stop (pips)",   avgStopPipsHtml) +
          row("Avg Target (pips)", avgTargetPipsHtml) +
          row("Best Trade",    "<span class='pos'>" + fmtMoney(m.best_trade)  + "</span>") +
          row("Worst Trade",   "<span class='neg'>" + fmtMoney(m.worst_trade) + "</span>") +
          row("Win Streak",    (function () { var s = m.streaks || {}; return s.max_win_streak !== undefined ? s.max_win_streak + " trades" : "&#8212;"; }())) +
          row("Loss Streak",   (function () { var s = m.streaks || {}; return s.max_loss_streak !== undefined ? s.max_loss_streak + " trades" : "&#8212;"; }())) +
          row("Max Daily DD",  (function () {
            var mdd = m.max_daily_drawdown || {};
            if (mdd.dollar === null || mdd.dollar === undefined) return "&#8212;";
            return "<span class='neg'>-$" + commaFmt(mdd.dollar) + " (" + fmt(mdd.pct, 2) + "%)</span>";
          }()) ) +
          row("Max Drawdown",  (function () {
            var ddD = m.max_drawdown_dollar;
            var ddDStr = (ddD !== null && ddD !== undefined) ? "-$" + commaFmt(ddD) + " (" + fmt(m.max_drawdown) + "%)" : fmt(m.max_drawdown) + "%";
            return "<span class='neg'>" + ddDStr + "</span>";
          }()) ) +
          row("Sharpe Ratio",  "<span class='" + sharpeCls + "'>" + fmt(m.sharpe) + "</span>") +
          row("Final Equity",  "$" + commaFmt(m.final_equity)) +
          "</tbody></table>" +
        "</div>" +

        "<div class='section'>" +
          "<div class='section-title'>Parameters</div>" +
          "<table><tbody>" +
          row("Date Range",     (function () {
            var dr = run.start_date && run.end_date
              ? { start: run.start_date, end: run.end_date }
              : fullRunRange(run);
            var dur = calcDuration(dr.start, dr.end);
            var dates = dr.start && dr.end
              ? fmtSbDate(dr.start) + " \u2192 " + fmtSbDate(dr.end) : "";
            return (dur ? dur : "") + (dur && dates ? " &middot; " : "") + dates;
          }())) +
          row("Instrument",     "<span class='val-highlight'>" + esc(savedInstr) + "</span>") +
          row("Interval",       "<span class='val-highlight'>" + esc(savedInterval) + "</span>") +
          row("EMA Short",      "<span class='val-highlight'>" + esc(savedEmaShort) + "</span>") +
          row("EMA Mid",        "<span class='val-highlight'>" + esc(savedEmaMid) + "</span>") +
          row("EMA Long",       "<span class='val-highlight'>" + esc(savedEmaLong) + "</span>") +
          row("Stop Loss Level", "<span class='val-highlight'>" + esc(savedStopPips) + " pips</span>") +
          row("Direction",      "<span class='val-highlight'>" + esc(dirOptions.filter(function(o){return o.value===savedDir;})[0].label) + "</span>") +
          row("RRR",            (run.rrr_risk || p.rrr_risk || 1) + "&thinsp;:&thinsp;" + (run.rrr_reward || p.rrr_reward || 2)) +
          row("Run on",         esc(fmtRunDate(run.date || ""))) +
          "</tbody></table>" +
        "</div>" +

      "</div>" +

      /* ── Section 3: Performance by Direction ──────────────────────────────── */
      dirHtml +

      /* ── Section 6: Monthly Performance ──────────────────────────────────── */
      monthHtml +

      /* ── Section 6b: Daily Performance (≤ 31 day ranges only) ────────────── */
      dailyPerfHtml +

      /* ── Section 6c: Intraday Performance (single-day ranges only) ─────── */
      intradayPerfHtml +

      /* ── Main chart ───────────────────────────────────────────────────────── */
      chartHtml +

      /* ── Fractal Diagnostics (single-day date ranges only) ──────── */
      pivotDiagHtml +

      "</div>" + /* end General tab */

      /* ── TAB: Advanced ───────────────────────────────────────────────────── */
      "<div class='tab-content' data-tab-content='advanced'>" +

      /* ── Time of Day Performance ──────────────────────────────────────── */
      timeOfDayHtml +

      /* ── Range Filter + Regime Classification (side by side) ────────────── */
      "<div class='two-col'>" + rsdHtml + regimeHtml + "</div>" +

      /* ── Streak Analysis + Stop vs Target ─────────────────────────────────── */
      "<div class='two-col'>" + streakHtml + stopHtml + "</div>" +

      /* ── Win Rate Trend ──────────────────────────────────────────────────── */
      winRateTrendHtml +

      /* ── Trade Duration + Filter Impact Summary ───────────────────────────── */
      "<div class='two-col'>" + durationHtml + filterImpactHtml + "</div>" +

      /* ── Daily Drawdown ────────────────────────────────────────────────────── */
      dailyDDHtml +

      /* ── RRR Sensitivity + Swing Lookback Sensitivity ─────────────────────── */
      "<div class='two-col'>" + rrrSensHtml + swingSensHtml + "</div>" +

      /* ── Equity and Drawdown ──────────────────────────────────────────────── */
      eqDdChartHtml +

      "</div>"; /* end Advanced tab */


    /* Wire entry conditions toggle button */
    (function () {
      var toggleBtn = document.getElementById("ec-toggle-btn");
      var panel = document.getElementById("ec-collapsible");
      if (!toggleBtn || !panel) return;
      toggleBtn.addEventListener("click", function () {
        var isOpen = panel.classList.toggle("open");
        toggleBtn.classList.toggle("open", isOpen);
      });
    }());

    /* Wire report tabs */
    (function () {
      var tabs = document.querySelectorAll(".report-tab");
      var panels = document.querySelectorAll(".tab-content");
      var quickNav = document.getElementById("quick-nav-bar");
      tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
          var target = tab.dataset.tab;
          tabs.forEach(function (t) { t.classList.remove("active"); });
          panels.forEach(function (p) { p.classList.remove("active"); });
          tab.classList.add("active");
          var panel = document.querySelector(".tab-content[data-tab-content='" + target + "']");
          if (panel) panel.classList.add("active");
          if (quickNav) quickNav.style.display = (target === "general") ? "" : "none";
          if (mainEl) mainEl.scrollTop = 0;
        });
      });
    }());

    /* Wire direction select — persist to localStorage on change */
    (function () {
      var dirEl = document.getElementById("ec-direction-select");
      if (!dirEl) return;
      var stored = localStorage.getItem("ec_direction");
      if (stored) dirEl.value = stored;
      dirEl.addEventListener("change", function () {
        localStorage.setItem("ec_direction", dirEl.value);
      });
    }());

    /* Wire instrument select — persist to localStorage on change */
    (function () {
      var instrEl = document.getElementById("ec-instrument-select");
      if (!instrEl) return;
      var stored = localStorage.getItem("ec_instrument");
      if (stored) instrEl.value = stored;
      instrEl.addEventListener("change", function () {
        localStorage.setItem("ec_instrument", instrEl.value);
      });
    }());

    /* Wire interval select — persist to localStorage on change */
    (function () {
      var intEl = document.getElementById("ec-interval-select");
      if (!intEl) return;
      var stored = localStorage.getItem("ec_interval");
      if (stored) intEl.value = stored;
      intEl.addEventListener("change", function () {
        localStorage.setItem("ec_interval", intEl.value);
      });
    }());

    /* Wire EMA inputs — persist to localStorage on change */
    (function () {
      var ids = [
        { id: "ec-ema-short",  key: "ec_ema_short" },
        { id: "ec-ema-mid",    key: "ec_ema_mid" },
        { id: "ec-ema-long",   key: "ec_ema_long" },
        { id: "ec-stop-pips",  key: "ec_stop_pips" }
      ];
      ids.forEach(function (item) {
        var el = document.getElementById(item.id);
        if (!el) return;
        var stored = localStorage.getItem(item.key);
        if (stored) el.value = stored;
        el.addEventListener("change", function () {
          localStorage.setItem(item.key, el.value);
        });
      });
    }());

    /* Wire RRR selects — persist to localStorage on change */
    (function () {
      var riskEl   = document.getElementById("ec-rrr-risk");
      var rewardEl = document.getElementById("ec-rrr-reward");
      if (riskEl) {
        var storedRisk = localStorage.getItem("ec_rrr_risk");
        if (storedRisk) riskEl.value = storedRisk;
        riskEl.addEventListener("change", function () {
          localStorage.setItem("ec_rrr_risk", riskEl.value);
        });
      }
      if (rewardEl) {
        var storedReward = localStorage.getItem("ec_rrr_reward");
        if (storedReward) rewardEl.value = storedReward;
        rewardEl.addEventListener("change", function () {
          localStorage.setItem("ec_rrr_reward", rewardEl.value);
        });
      }
    }());

    /* Wire copy button — context aware (lives in the run bar) */
    (function (ver, runData) {
      var btn = document.getElementById("copy-btn");
      var sep = document.getElementById("rb-act-sep");
      if (!btn) return;
      var isDateRange = activeRunIdx > 0;
      btn.style.display = "";
      btn.disabled = false;
      btn.classList.remove("copied");
      if (sep) sep.style.display = "";
      btn.textContent = isDateRange ? "Copy Range Report" : "Copy Version Report";
      btn.onclick = function () {
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
      };
    }(v, run));

    /* Wire delete button (lives in the run bar) */
    (function (ver) {
      var delBtn = document.getElementById("delete-btn");
      if (!delBtn) return;
      var isDateRange = activeRunIdx > 0;
      delBtn.style.display = "";
      delBtn.disabled = false;
      delBtn.textContent = isDateRange ? "Delete Date Range" : "Delete Version";
      delBtn.onclick = function () {
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
            if (isRange) {
              localStorage.setItem("rb_pending_delete_version", ver.name);
              var focusIdx = activeRunIdx - 1;
              localStorage.setItem("rb_pending_delete_run_idx", String(focusIdx < 1 ? 0 : focusIdx));
            }
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
      };
    }(v));
  }

  /* ── Hide run-bar action buttons (Copy/Delete) ───────────── */
  function hideActionButtons() {
    var btn    = document.getElementById("copy-btn");
    var delBtn = document.getElementById("delete-btn");
    var sep    = document.getElementById("rb-act-sep");
    if (btn)    { btn.style.display = "none";    btn.onclick = null; }
    if (delBtn) { delBtn.style.display = "none"; delBtn.onclick = null; }
    if (sep)    { sep.style.display = "none"; }
  }

  /* ── Dev Log ──────────────────────────────────────────────── */
  function showDevLog() {
    hideActionButtons();
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
        "<td class='dl-date'>" + esc(fmtRunDate(firstRun.date || "")) + "</td>" +
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

  /* ── Empty state — show Entry Conditions selects when no versions exist ── */
  function renderEmptyState() {
    hideActionButtons();
    var ecThStyle = "class='ec-th'";

    var _dirOptions = [
      { value: "short_only", label: "Short only" },
      { value: "long_only",  label: "Long only" },
      { value: "both",       label: "Both" }
    ];
    var _savedDir = localStorage.getItem("ec_direction") || "short_only";
    var _dirSelectHtml = "<select id='ec-direction-select' class='ec-select'>" +
      _dirOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === _savedDir ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var _instrOptions = [
      { value: "EURUSD", label: "EURUSD" },
      { value: "GBPUSD", label: "GBPUSD" }
    ];
    var _savedInstr = localStorage.getItem("ec_instrument") || "EURUSD";
    var _instrSelectHtml = "<select id='ec-instrument-select' class='ec-select'>" +
      _instrOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === _savedInstr ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var _intervalOptions = [
      { value: "1m", label: "1m" },
      { value: "5m", label: "5m" },
      { value: "15m", label: "15m" },
      { value: "30m", label: "30m" },
      { value: "60m", label: "60m" }
    ];
    var _savedInterval = localStorage.getItem("ec_interval") || "5m";
    var _intervalSelectHtml = "<select id='ec-interval-select' class='ec-select'>" +
      _intervalOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === _savedInterval ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var _savedEmaShort = localStorage.getItem("ec_ema_short") || "8";
    var _savedEmaMid   = localStorage.getItem("ec_ema_mid")   || "20";
    var _savedEmaLong  = localStorage.getItem("ec_ema_long")  || "40";
    var _emaShortHtml = "<input id='ec-ema-short' type='number' class='ec-input' value='" + _savedEmaShort + "' min='1' step='1'>";
    var _emaMidHtml   = "<input id='ec-ema-mid'   type='number' class='ec-input' value='" + _savedEmaMid   + "' min='1' step='1'>";
    var _emaLongHtml  = "<input id='ec-ema-long'  type='number' class='ec-input' value='" + _savedEmaLong  + "' min='1' step='1'>";

    var _savedStopPips  = localStorage.getItem("ec_stop_pips") || "15";
    var _stopPipsHtml   = "<input id='ec-stop-pips' type='number' class='ec-input' value='" + _savedStopPips + "' min='1' step='1'>";

    var _savedRrrRisk   = localStorage.getItem("ec_rrr_risk")   || "1";
    var _savedRrrReward = localStorage.getItem("ec_rrr_reward") || "2";
    var _rrrOpts = [1, 2, 3, 4, 5];
    var _rrrRiskHtml = "<select id='ec-rrr-risk' class='ec-select ec-select-narrow'>" +
      _rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (String(n) === _savedRrrRisk ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var _rrrRewardHtml = "<select id='ec-rrr-reward' class='ec-select ec-select-narrow'>" +
      _rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (String(n) === _savedRrrReward ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var _rrrSelectHtml = _rrrRiskHtml + "<span class='ec-rrr-colon'>:</span>" + _rrrRewardHtml;

    document.getElementById("content").innerHTML =
      "<div class='section'>" +
        "<div class='section-title'>Entry Conditions</div>" +
        "<table>" +
          "<tbody>" +
          "<tr><td class='ec-td-cond'>Instrument</td><td class='ec-td-rule'>" + _instrSelectHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>Interval</td><td class='ec-td-rule'>" + _intervalSelectHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>EMA Short</td><td class='ec-td-rule'>" + _emaShortHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>EMA Mid</td><td class='ec-td-rule'>" + _emaMidHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>EMA Long</td><td class='ec-td-rule'>" + _emaLongHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>Stop Loss Level</td><td class='ec-td-rule'>" + _stopPipsHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>Direction</td><td class='ec-td-rule'>" + _dirSelectHtml + "</td></tr>" +
          "<tr><td class='ec-td-cond'>RRR</td><td class='ec-td-rule'>" + _rrrSelectHtml + "</td></tr>" +
          "</tbody>" +
        "</table>" +
      "</div>";

    /* Wire localStorage persistence for the empty-state selects */
    var _dirEl = document.getElementById("ec-direction-select");
    if (_dirEl) _dirEl.addEventListener("change", function () { localStorage.setItem("ec_direction", _dirEl.value); });
    var _instrEl = document.getElementById("ec-instrument-select");
    if (_instrEl) _instrEl.addEventListener("change", function () { localStorage.setItem("ec_instrument", _instrEl.value); });
    var _intEl = document.getElementById("ec-interval-select");
    if (_intEl) _intEl.addEventListener("change", function () { localStorage.setItem("ec_interval", _intEl.value); });
    var _rrrRiskEl = document.getElementById("ec-rrr-risk");
    if (_rrrRiskEl) _rrrRiskEl.addEventListener("change", function () { localStorage.setItem("ec_rrr_risk", _rrrRiskEl.value); });
    var _rrrRewardEl = document.getElementById("ec-rrr-reward");
    if (_rrrRewardEl) _rrrRewardEl.addEventListener("change", function () { localStorage.setItem("ec_rrr_reward", _rrrRewardEl.value); });
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
      renderEmptyState();
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

    /* If we just came back from a date-range run, jump to the newest sub-item */
    var pendingRunType    = localStorage.getItem("rb_pending_run_type");
    var pendingRunVersion = localStorage.getItem("rb_pending_run_version");
    if (pendingRunType === "date_range") {
      localStorage.removeItem("rb_pending_run_type");
      localStorage.removeItem("rb_pending_run_version");
      /* Find the version the run was added to */
      var targetIdx = lastIdx;
      if (pendingRunVersion) {
        for (var ti = 0; ti < VERSIONS.length; ti++) {
          if (VERSIONS[ti].name === pendingRunVersion) { targetIdx = ti; break; }
        }
      }
      activeVersionIdx = targetIdx;
      expandedVersions[targetIdx] = true;
      var targetRuns = getRuns(VERSIONS[targetIdx]);
      if (targetRuns.length > 1) {
        activeRunIdx = targetRuns.length - 1;
      }
    }

    /* If we just came back from a new version with auto date ranges */
    if (pendingRunType === "new_version_auto") {
      localStorage.removeItem("rb_pending_run_type");
      activeVersionIdx = lastIdx;
      activeRunIdx = 0;
      expandedVersions[lastIdx] = true;
    }

    /* If we just deleted a date-range sub-item, focus on the sub-item above it */
    var pendingDelVersion = localStorage.getItem("rb_pending_delete_version");
    var pendingDelRunIdx  = localStorage.getItem("rb_pending_delete_run_idx");
    if (pendingDelVersion) {
      localStorage.removeItem("rb_pending_delete_version");
      localStorage.removeItem("rb_pending_delete_run_idx");
      var delTargetIdx = lastIdx;
      for (var di = 0; di < VERSIONS.length; di++) {
        if (VERSIONS[di].name === pendingDelVersion) { delTargetIdx = di; break; }
      }
      activeVersionIdx = delTargetIdx;
      expandedVersions[delTargetIdx] = true;
      var delRunIdx = parseInt(pendingDelRunIdx, 10) || 0;
      var remainingRuns = getRuns(VERSIONS[delTargetIdx]);
      if (delRunIdx > 0 && delRunIdx < remainingRuns.length) {
        activeRunIdx = delRunIdx;
      } else {
        activeRunIdx = 0;
      }
    }

    expandedVersions[lastIdx] = true;
    renderSidebar();
    renderContent(activeVersionIdx, activeRunIdx);

    /* Scroll sidebar to bottom after date-range add */
    var vList = document.getElementById("version-list");
    if (vList) vList.scrollTop = vList.scrollHeight;
  } else {
    renderEmptyState();
  }

  /* ── Keyboard shortcuts: Shift+Up / Shift+Down sidebar navigation ── */
  document.addEventListener("keydown", function (e) {
    if (!e.shiftKey) return;
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    if (devLogOpen) return;

    /* Build flat list of visible sidebar items in render order (newest first) */
    var svs = getStrategyVersions();
    if (svs.length === 0) return;
    var items = []; /* { vIdx, runIdx } */
    for (var ri = svs.length - 1; ri >= 0; ri--) {
      var entry = svs[ri];
      var vIdx  = entry.idx;
      var runs  = getRuns(entry.v);
      items.push({ vIdx: vIdx, runIdx: 0 });
      if (expandedVersions[vIdx] && runs.length > 1) {
        for (var si = 1; si < runs.length; si++) {
          items.push({ vIdx: vIdx, runIdx: si });
        }
      }
    }
    if (items.length === 0) return;

    /* Find current position */
    var curPos = -1;
    for (var i = 0; i < items.length; i++) {
      if (items[i].vIdx === activeVersionIdx && items[i].runIdx === activeRunIdx) {
        curPos = i; break;
      }
    }

    var newPos = curPos;
    if (e.key === "ArrowUp")   newPos = curPos <= 0 ? 0 : curPos - 1;
    if (e.key === "ArrowDown") newPos = curPos >= items.length - 1 ? items.length - 1 : curPos + 1;
    if (newPos === curPos) return;

    e.preventDefault();
    var target = items[newPos];
    activeVersionIdx = target.vIdx;
    activeRunIdx     = target.runIdx;
    renderSidebar();
    renderContent(activeVersionIdx, activeRunIdx);

    /* Scroll active item into view */
    var activeEl = document.querySelector("#version-list .v-item.active");
    if (activeEl) activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });

  /* ── Keyboard shortcuts: 1-9 to scroll to section rows ── */
  document.addEventListener("keydown", function (e) {
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
    var m = (e.code || "").match(/^Digit(\d)$/);
    if (!m) return;
    var num = parseInt(m[1], 10);
    if (num < 1 || num > 9) return;
    var activeTab = document.querySelector(".tab-content.active");
    if (!activeTab) return;
    /* Collect visible direct-child section rows (divs with content) */
    var rows = [];
    for (var ci = 0; ci < activeTab.children.length; ci++) {
      var child = activeTab.children[ci];
      if (child.offsetHeight > 0) rows.push(child);
    }
    var idx = num - 1;
    if (idx >= rows.length) return;
    e.preventDefault();
    var mainEl = document.getElementById("main");
    if (!mainEl) return;
    if (idx === 0) {
      mainEl.scrollTo({ top: 0, behavior: "smooth" });
    } else {
      var top = rows[idx].offsetTop - mainEl.offsetTop - 16;
      mainEl.scrollTo({ top: top, behavior: "smooth" });
    }
  });

  /* ── Helper: check if focus is in a form field ──────────── */
  function isInputFocused(e) {
    var tag = (e.target.tagName || "").toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;
  }

  /* ── Keyboard shortcut: T or Shift+T to scroll to top ────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "t" && e.key !== "T") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    e.preventDefault();
    var main = document.getElementById("main");
    if (main) main.scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ── Keyboard shortcut: V or Shift+V — Add New Version ───── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "v" && e.key !== "V") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("run-new-btn");
    if (!btn || btn.disabled) return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: D or Shift+D — Add Date Range ────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "d" && e.key !== "D") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("run-range-btn");
    if (!btn || btn.disabled) return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: C or Shift+C — Copy Report ────────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "c" && e.key !== "C") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("copy-btn");
    if (!btn || btn.disabled || btn.style.display === "none") return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: Shift+Delete — Delete ──────────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Delete" && e.key !== "Backspace") return;
    if (!e.shiftKey) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("delete-btn");
    if (!btn || btn.disabled || btn.style.display === "none") return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: L or Shift+L — Development Log ───── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "l" && e.key !== "L") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("devlog-btn");
    if (!btn) return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: B — Toggle Entry Conditions ───────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "b" && e.key !== "B") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("ec-toggle-btn");
    if (!btn) return;
    e.preventDefault();
    btn.click();
  });
})();
</script>

<button id="scroll-top-btn" aria-label="Scroll to top">&#9650;</button>
<script>
(function () {
  var btn  = document.getElementById("scroll-top-btn");
  var main = document.getElementById("main");
  if (!btn || !main) return;
  main.addEventListener("scroll", function () {
    btn.classList.toggle("visible", main.scrollTop > 120);
  });
  btn.addEventListener("click", function () {
    main.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
</script>

</body>
</html>"""
    return template.replace("__VERSIONS_JSON__", versions_json)


def generate_html_report(trades, equity, chart_path="backtest_chart.png", notes="",
                         blocked_signals=None, df=None,
                         eq_dd_chart_path=None,
                         run_mode="new_version", run_start_date="", run_end_date=""):
    """Create or update report.html.

    run_mode="new_version" → increment version, create new entry with first run
    run_mode="date_range"  → append a run to the most recent version
    """
    global VERSION
    report_path = "report.html"

    print("  Computing metrics...")
    metrics = compute_metrics(trades, equity, blocked_signals=blocked_signals, df=df)
    if metrics is None:
        print("  No trades generated — skipping HTML report.")
        print("NO_DATA")
        return
    print("  Metrics complete. Building report...")

    # ── Load main chart as base64 ──────────────────────────────────────────────
    chart_b64 = ""
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as fh:
            chart_b64 = base64.b64encode(fh.read()).decode("utf-8")

    # ── Load Equity/Drawdown chart as base64 ─────────────────────────────────
    eq_dd_chart_b64 = ""
    if eq_dd_chart_path and os.path.exists(eq_dd_chart_path):
        with open(eq_dd_chart_path, "rb") as fh:
            eq_dd_chart_b64 = base64.b64encode(fh.read()).decode("utf-8")

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
                "eq_dd_chart_b64": v.get("eq_dd_chart_b64", ""),
                "metrics":       v.get("metrics", {}),
                "last_trades":   v.get("last_trades", []),
            }]
            # Remove legacy top-level run data (keep name, params, strategy, etc.)
            for key in ["date", "notes", "chart_b64",
                        "eq_dd_chart_b64", "metrics", "last_trades"]:
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
        "instrument":       _INSTRUMENT,
        "interval":         INTERVAL,
        "trade_direction":  TRADE_DIRECTION,
        "ema_short":        EMA_SHORT,
        "ema_mid":          EMA_MID,
        "ema_long":         EMA_LONG,
        "stop_loss_pips":   int(FRACTAL_STOP_PIPS * 10000),
        "rrr_risk":         RRR_RISK,
        "rrr_reward":       RRR_REWARD,
        "notes":         notes.strip() if notes else "—",
        "chart_b64":        chart_b64,
        "eq_dd_chart_b64":  eq_dd_chart_b64,
        "metrics":          metrics,
        "last_trades":      last_trades,
    }

    params_dict = {
        "ticker":         TICKER,
        "interval":       INTERVAL,
        "days_back":      DAYS_BACK,
        "starting_cash":  STARTING_CASH,
        "ema_short":      EMA_SHORT,
        "ema_mid":        EMA_MID,
        "ema_long":       EMA_LONG,
        "stop_loss_pips": int(FRACTAL_STOP_PIPS * 10000),
        "rrr":            RRR,
        "rrr_risk":       RRR_RISK,
        "rrr_reward":     RRR_REWARD,
        "risk_pct":          RISK_PCT,
        "min_stop":          MIN_STOP,
        "max_stop":          MAX_STOP,
        "trade_direction":   TRADE_DIRECTION,
        "max_daily_loss": MAX_DAILY_LOSS,
    }

    if run_mode == "date_range" and existing_versions:
        # Append run to the specified target version (or most recent as fallback)
        target_version_name = os.environ.get("TARGET_VERSION", "").strip()
        target = None
        if target_version_name:
            for v in existing_versions:
                if v.get("name") == target_version_name:
                    target = v
                    break
        if target is None:
            target = existing_versions[-1]
        target["runs"].append(new_run)
        target["entry_conditions"] = ENTRY_CONDITIONS
        version_num = len(existing_versions)
        action = f"Added date range run to {target.get('name', '?')}"
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
    print("  Building HTML template...")
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
        subprocess.run(["git", "add", "-A"], cwd=repo_dir, check=True, timeout=30)
        print("  git add -A ... done")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  ⚠  git add failed: {e}")
        return

    try:
        subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir, check=True, timeout=30)
        print(f"  git commit ... done")
        print(f"  Commit message: {msg}")
    except subprocess.CalledProcessError as e:
        # Exit code 1 means nothing to commit — not a real error
        print(f"  git commit — nothing new to commit (or error: {e})")
        return
    except subprocess.TimeoutExpired as e:
        print(f"  ⚠  git commit timed out: {e}")
        return

    try:
        subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, check=True, timeout=60)
        print("  git push origin main ... done\n")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
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

    print("PROGRESS:5:Fetching data…", flush=True)
    if run_mode == "date_range" and run_start_date and run_end_date:
        df = fetch_data(TICKER, INTERVAL, DAYS_BACK,
                        start_date=run_start_date, end_date=run_end_date)
    else:
        df = fetch_data(TICKER, INTERVAL, DAYS_BACK)

    print("PROGRESS:20:Computing indicators…", flush=True)
    df              = add_indicators(df)

    # ── Filter dataframe to requested date range (with buffers for state) ──────
    # Pre-start buffer (1 day): so the backtest carries forward any open-position
    # state from the prior session.
    # Post-end buffer (7 days): so trades entered near the end of the requested
    # range have enough future bars to hit their SL/TP and resolve properly.
    # Both buffers are trimmed after the backtest — only trades whose entry falls
    # within the requested range are kept in the final results.
    _range_start  = None
    _range_end    = None
    if run_mode == "date_range" and run_start_date and run_end_date:
        _range_start = pd.Timestamp(run_start_date, tz="UTC")
        _range_end   = pd.Timestamp(run_end_date,   tz="UTC") + pd.Timedelta(days=1)
        try:
            _dts = pd.to_datetime(df["Datetime"])
            if _dts.dt.tz is not None:
                _dts_utc = _dts.dt.tz_convert("UTC")
            else:
                _dts_utc = _dts.dt.tz_localize("UTC")
            _bt_start = _range_start - pd.Timedelta(days=1)
            _bt_end   = _range_end   + pd.Timedelta(days=7)
            _mask     = (_dts_utc >= _bt_start) & (_dts_utc < _bt_end)
            df        = df[_mask].reset_index(drop=True)
        except Exception as _e:
            print(f"  Date filter error: {_e}")
    print("PROGRESS:35:Running backtest…", flush=True)
    trades, equity, blocked_signals = run_backtest(df)

    # ── Trim buffer bars from date-range results ────────────────────────────
    # Both pre-start and post-end buffers are trimmed here so the df, trades,
    # equity curve, and charts reflect only the requested date range.
    if _range_start is not None and _range_end is not None:
        # Trim df to the exact requested range (remove pre-start and post-end buffers)
        _dts_trim = pd.to_datetime(df["Datetime"])
        _dts_trim_utc = (_dts_trim.dt.tz_convert("UTC")
                         if _dts_trim.dt.tz is not None
                         else _dts_trim.dt.tz_localize("UTC"))
        _range_mask = (_dts_trim_utc >= _range_start) & (_dts_trim_utc < _range_end)
        _pre_buffer = int((_dts_trim_utc < _range_start).sum())
        df = df[_range_mask].reset_index(drop=True)

        # Keep only trades whose entry falls within the requested range
        if not trades.empty:
            _t_entry = pd.to_datetime(trades["entry_ts"])
            _t_entry = (_t_entry.dt.tz_convert("UTC")
                        if _t_entry.dt.tz is not None
                        else _t_entry.dt.tz_localize("UTC"))
            _t_mask = (_t_entry >= _range_start) & (_t_entry < _range_end)
            trades  = trades[_t_mask].copy()
            trades["entry_idx"] = trades["entry_idx"] - _pre_buffer
            trades["exit_idx"]  = trades["exit_idx"]  - _pre_buffer
            trades = trades.reset_index(drop=True)

        # Filter blocked signals to the requested range
        _filtered_bs = []
        for _s in blocked_signals:
            _s_ts  = pd.Timestamp(_s["timestamp"])
            _s_utc = (_s_ts.tz_convert("UTC")
                      if _s_ts.tzinfo is not None
                      else _s_ts.tz_localize("UTC"))
            if _range_start <= _s_utc < _range_end:
                _filtered_bs.append(_s)
        blocked_signals = _filtered_bs

        # Recompute bar-by-bar equity from filtered trades so that the curve,
        # drawdown, and net-P&L reflect only the requested date range.
        # Trades whose exit falls in the post-end buffer (exit_idx >= len(df))
        # have their P&L added at the last visible bar so totals remain correct.
        _eq_exits = {}
        for _, _t in trades.iterrows():
            _eidx = int(_t.exit_idx)
            if _eidx >= len(df):
                _eidx = len(df) - 1   # clamp to last visible bar
            if 0 <= _eidx:
                _eq_exits.setdefault(_eidx, []).append(float(_t.pnl))
        _eq_cash = STARTING_CASH
        equity   = [_eq_cash]
        for _bi in range(1, len(df)):
            if _bi in _eq_exits:
                for _pnl in _eq_exits[_bi]:
                    _eq_cash += _pnl
            equity.append(_eq_cash)

    print("PROGRESS:55:Printing results…", flush=True)
    print_results(trades, equity)
    print("PROGRESS:60:Generating charts…", flush=True)
    chart_path, eq_dd_chart_path = save_charts(df, trades, equity)
    print("PROGRESS:75:Building report…", flush=True)
    generate_html_report(trades, equity, chart_path=chart_path, notes=run_notes,
                         blocked_signals=blocked_signals, df=df,
                         eq_dd_chart_path=eq_dd_chart_path,
                         run_mode=run_mode,
                         run_start_date=run_start_date,
                         run_end_date=run_end_date)
    print("PROGRESS:88:Computing metrics…", flush=True)
    metrics = compute_metrics(trades, equity, blocked_signals=blocked_signals, df=df)
    print("PROGRESS:92:Updating results log…", flush=True)
    update_results_log(metrics, notes=run_notes)
    print("PROGRESS:96:Pushing to git…", flush=True)
    git_commit_and_push(metrics, VERSION, TICKER, INTERVAL)
    print("PROGRESS:100:Complete", flush=True)
