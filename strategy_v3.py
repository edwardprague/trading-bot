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

# ── Time filter: skip entries during these UTC hours ─────────────────────────
BLOCKED_HOURS_UTC = [4, 5, 6, 8, 10, 11, 14, 17]

# ── Width filter: N18 must be Wide (≥ threshold × ATR) to allow entries ──────
N18_WIDTH_THRESHOLD = 1.0   # N18 is Wide if normalized width ≥ 1.0× ATR
N6_WIDTH_THRESHOLD  = 0.5   # N6  is Wide if normalized width ≥ 0.5× ATR

VERSION = "v6"
STRATEGY_VERSION_TAG = "v3"     # identifies which strategy file produced these results
NOTES = "Fractal geometry entries — no EMA alignment"
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
    df["ema_short"] = df.Close.ewm(span=EMA_SHORT, adjust=False).mean() if EMA_SHORT > 0 else df.Close
    df["ema_mid"]   = df.Close.ewm(span=EMA_MID,   adjust=False).mean() if EMA_MID   > 0 else df.Close
    df["ema_long"]  = df.Close.ewm(span=EMA_LONG,  adjust=False).mean() if EMA_LONG  > 0 else df.Close
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
    """Backtest with a specific RRR, using identical entry/filter logic to run_backtest.
    swing_lookback is unused but kept for API compat.
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

    # Daily loss limit state
    _daily_loss_day = None
    _daily_loss_pnl = 0.0

    # Rolling fractal state — mirrors run_backtest exactly
    prev_high_price_s = None
    prev_low_price_s  = None
    last_high_label_s = None
    last_low_label_s  = None
    last_fractal_low_s   = None
    last_fractal_high_s  = None
    prior_fractal_high_s = None
    prior_fractal_low_s  = None

    # N18/N6 rolling state for width filter
    _n18_hi_s = None
    _n18_lo_s = None
    _n6_hi_s  = None
    _n6_lo_s  = None

    for i in range(1, len(df2)):
        c  = float(df2["Close"].iloc[i])
        ts = df2['Datetime'].iloc[i]

        # ── Check exits ──────────────────────────────────────────────────────
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

            # Daily loss limit force-close check
            force_close = False
            if not (hit_sl or hit_tp):
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
                    exit_p = c
                else:
                    exit_p = sl if hit_sl else tp
                pnl = (exit_p - entry_p) * size if direction == "long" \
                      else (entry_p - exit_p) * size
                cash += pnl
                # Update daily loss tracker
                _exit_utc = pd.to_datetime(ts)
                _exit_utc = _exit_utc.tz_convert('UTC') if _exit_utc.tzinfo else _exit_utc.tz_localize('UTC')
                _exit_day = _exit_utc.date()
                if _exit_day != _daily_loss_day:
                    _daily_loss_day = _exit_day
                    _daily_loss_pnl = 0.0
                _daily_loss_pnl += pnl
                trades_s.append({"pnl": pnl, "win": pnl > 0})
                in_trade = False

        equity_s.append(cash)

        # ── Rolling fractal detection ────────────────────────────────────────
        _new_high_confirmed = False
        _new_low_confirmed  = False
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
                prior_fractal_high_s = last_fractal_high_s
                prev_high_price_s = float(fh)
                last_fractal_high_s = float(fh)
                _new_high_confirmed = True
            if is_pl:
                if prev_low_price_s is None:
                    last_low_label_s = 'CL'
                else:
                    d = fl - prev_low_price_s
                    last_low_label_s = 'CL' if abs(d) < thr else ('HL' if d > 0 else 'LL')
                prior_fractal_low_s = last_fractal_low_s
                prev_low_price_s = float(fl)
                last_fractal_low_s = float(fl)
                _new_low_confirmed = True

        # N18 fractal detection
        if i >= 36:
            fi18 = i - 18
            fh18 = highs_s[fi18]; fl18 = lows_s[fi18]
            if all(fh18 > highs_s[fi18 - k] for k in range(1, 19)) and \
               all(fh18 > highs_s[fi18 + k] for k in range(1, 19)):
                _n18_hi_s = float(fh18)
            if all(fl18 < lows_s[fi18 - k] for k in range(1, 19)) and \
               all(fl18 < lows_s[fi18 + k] for k in range(1, 19)):
                _n18_lo_s = float(fl18)

        # N6 fractal detection
        if i >= 12:
            fi6 = i - 6
            fh6 = highs_s[fi6]; fl6 = lows_s[fi6]
            if all(fh6 > highs_s[fi6 - k] for k in range(1, 7)) and \
               all(fh6 > highs_s[fi6 + k] for k in range(1, 7)):
                _n6_hi_s = float(fh6)
            if all(fl6 < lows_s[fi6 - k] for k in range(1, 7)) and \
               all(fl6 < lows_s[fi6 + k] for k in range(1, 7)):
                _n6_lo_s = float(fl6)

        # ── Check entries ────────────────────────────────────────────────────
        if not in_trade:
            # Long signal: newly confirmed HL whose low > prior low-type pivot
            long_sig = False
            long_fractal_price = None
            if (_new_low_confirmed and last_low_label_s == 'HL'
                    and last_fractal_low_s is not None
                    and prior_fractal_low_s is not None
                    and last_fractal_low_s > prior_fractal_low_s):
                long_sig = True
                long_fractal_price = last_fractal_low_s

            # Short signal: newly confirmed LH whose high < prior high-type pivot
            short_sig = False
            short_fractal_price = None
            if (_new_high_confirmed and last_high_label_s == 'LH'
                    and last_fractal_high_s is not None
                    and prior_fractal_high_s is not None
                    and last_fractal_high_s < prior_fractal_high_s):
                short_sig = True
                short_fractal_price = last_fractal_high_s

            # Direction filter
            long_sig  = long_sig  and (TRADE_DIRECTION != "short_only")
            short_sig = short_sig and (TRADE_DIRECTION != "long_only")

            # Daily loss limit
            _ts_day = pd.to_datetime(ts)
            _ts_day_utc = _ts_day.tz_convert('UTC') if _ts_day.tzinfo else _ts_day.tz_localize('UTC')
            _today = _ts_day_utc.date()
            if _today != _daily_loss_day:
                _daily_loss_day = _today
                _daily_loss_pnl = 0.0
            if _daily_loss_pnl <= -MAX_DAILY_LOSS:
                continue

            # Time filter
            _entry_hour_utc = _ts_day_utc.hour
            if _entry_hour_utc in BLOCKED_HOURS_UTC:
                continue

            # Width filter: block entries when N18 is Narrow
            _cur_atr = float(atr_s[i])
            if _cur_atr > 0 and _n18_hi_s is not None and _n18_lo_s is not None \
               and _n6_hi_s is not None and _n6_lo_s is not None:
                if abs(_n18_hi_s - _n18_lo_s) / _cur_atr < N18_WIDTH_THRESHOLD:
                    continue

            if long_sig:
                sl_p = long_fractal_price - FRACTAL_STOP_PIPS
                dist = c - sl_p
                if MIN_STOP <= dist <= MAX_STOP:
                    direction = "long"; entry_p = c; sl = sl_p
                    tp = c + dist * rrr; size = (cash * RISK_PCT) / dist; in_trade = True
            elif short_sig:
                sl_p = short_fractal_price + FRACTAL_STOP_PIPS
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
    entry_fractal_bar   = None  # bar index of the fractal that triggered entry
    entry_fractal_label = None  # label (HL, LH, etc.) of the triggering fractal
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
    last_fractal_low_bar    = None   # bar index of the most recent confirmed pivot low
    last_fractal_high_bar   = None   # bar index of the most recent confirmed pivot high
    prior_fractal_high_price = None  # price of the high-type pivot before the current one
    prior_fractal_low_price  = None  # price of the low-type pivot before the current one

    # ── N18/N6 rolling state for width filter ────────────────────────────────
    _n18_hi_price = None   # most recent N18 pivot high price
    _n18_lo_price = None   # most recent N18 pivot low price
    _n6_hi_price  = None   # most recent N6 pivot high price
    _n6_lo_price  = None   # most recent N6 pivot low price

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
                    "fractal_bar":   entry_fractal_bar,
                    "fractal_label": entry_fractal_label,
                    "timestamp":    ts,
                    "entry_ts":     entry_ts
                })
                in_trade = False

        equity.append(cash)

        # ── Rolling fractal detection (confirmed at bar i, formed at bar i-2) ─
        # A fractal needs 2 bars on each side, so the earliest confirmable
        # pivot is at index 2.  At bar i we confirm the pivot at bar i-2.
        _new_high_confirmed = False  # set True when a new pivot high confirms this bar
        _new_low_confirmed  = False  # set True when a new pivot low confirms this bar
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
                prior_fractal_high_price = last_fractal_high_price
                prev_high_price = float(fh)
                last_fractal_high_price = float(fh)
                last_fractal_high_bar   = fi
                _new_high_confirmed = True

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
                prior_fractal_low_price = last_fractal_low_price
                prev_low_price = float(fl)
                last_fractal_low_price = float(fl)
                last_fractal_low_bar   = fi
                _new_low_confirmed = True

        # ── N18 fractal detection (confirmed at bar i, formed at bar i-18) ───
        if i >= 36:  # need 18 bars on each side of fi18
            fi18 = i - 18
            fh18 = highs[fi18]
            fl18 = lows[fi18]
            if all(fh18 > highs[fi18 - k] for k in range(1, 19)) and \
               all(fh18 > highs[fi18 + k] for k in range(1, 19)):
                _n18_hi_price = float(fh18)
            if all(fl18 < lows[fi18 - k] for k in range(1, 19)) and \
               all(fl18 < lows[fi18 + k] for k in range(1, 19)):
                _n18_lo_price = float(fl18)

        # ── N6 fractal detection (confirmed at bar i, formed at bar i-6) ─────
        if i >= 12:  # need 6 bars on each side of fi6
            fi6 = i - 6
            fh6 = highs[fi6]
            fl6 = lows[fi6]
            if all(fh6 > highs[fi6 - k] for k in range(1, 7)) and \
               all(fh6 > highs[fi6 + k] for k in range(1, 7)):
                _n6_hi_price = float(fh6)
            if all(fl6 < lows[fi6 - k] for k in range(1, 7)) and \
               all(fl6 < lows[fi6 + k] for k in range(1, 7)):
                _n6_lo_price = float(fl6)

        # ── Check entries ─────────────────────────────────────────────────────
        if not in_trade:
            # Signal fires only on the confirmation bar (i == fractal bar + 2)
            # to ensure each fractal triggers at most one entry attempt.

            # ── Long signal: N2 HL whose low > prior low-type pivot's low
            #    Fires only on the bar the HL fractal confirms.
            long_sig_raw = False
            long_fractal_price = None
            if (_new_low_confirmed and last_low_label == 'HL'
                    and last_fractal_low_price is not None
                    and prior_fractal_low_price is not None
                    and last_fractal_low_price > prior_fractal_low_price):
                long_sig_raw = True
                long_fractal_price = last_fractal_low_price

            # ── Short signal: N2 LH whose high < prior high-type pivot's high
            #    Fires only on the bar the LH fractal confirms.
            short_sig_raw = False
            short_fractal_price = None
            if (_new_high_confirmed and last_high_label == 'LH'
                    and last_fractal_high_price is not None
                    and prior_fractal_high_price is not None
                    and last_fractal_high_price < prior_fractal_high_price):
                short_sig_raw = True
                short_fractal_price = last_fractal_high_price

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

            # ── Time filter: skip entries during blocked UTC hours ────────────
            _entry_hour_utc = _ts_day_utc.hour
            if _entry_hour_utc in BLOCKED_HOURS_UTC:
                if long_sig:
                    _sl_t = long_fractal_price - FRACTAL_STOP_PIPS
                    _dist_t = c - _sl_t
                    if MIN_STOP <= _dist_t <= MAX_STOP:
                        blocked_signals.append(_scan_outcome(
                            df, i, "long", c, _sl_t,
                            c + _dist_t * RRR,
                            (cash * RISK_PCT) / _dist_t, ts, "time"))
                elif short_sig:
                    _sl_t = short_fractal_price + FRACTAL_STOP_PIPS
                    _dist_t = _sl_t - c
                    if MIN_STOP <= _dist_t <= MAX_STOP:
                        blocked_signals.append(_scan_outcome(
                            df, i, "short", c, _sl_t,
                            c - _dist_t * RRR,
                            (cash * RISK_PCT) / _dist_t, ts, "time"))
                continue

            # ── Width filter: block entries when N18 is Narrow (score 1 or 2) ─
            _current_atr = float(atr_vals[i])
            if _current_atr > 0 and _n18_hi_price is not None and _n18_lo_price is not None \
               and _n6_hi_price is not None and _n6_lo_price is not None:
                _n18_w = abs(_n18_hi_price - _n18_lo_price) / _current_atr
                _n6_w  = abs(_n6_hi_price  - _n6_lo_price)  / _current_atr
                _n18_wide = _n18_w >= N18_WIDTH_THRESHOLD
                _width_score = 3 if _n18_wide else 1  # only N18 matters for filter
                if not _n18_wide:
                    # N18 is Narrow → score is 1 or 2 → block entry
                    if long_sig:
                        _sl_w = long_fractal_price - FRACTAL_STOP_PIPS
                        _dist_w = c - _sl_w
                        if MIN_STOP <= _dist_w <= MAX_STOP:
                            blocked_signals.append(_scan_outcome(
                                df, i, "long", c, _sl_w,
                                c + _dist_w * RRR,
                                (cash * RISK_PCT) / _dist_w, ts, "width"))
                    elif short_sig:
                        _sl_w = short_fractal_price + FRACTAL_STOP_PIPS
                        _dist_w = _sl_w - c
                        if MIN_STOP <= _dist_w <= MAX_STOP:
                            blocked_signals.append(_scan_outcome(
                                df, i, "short", c, _sl_w,
                                c - _dist_w * RRR,
                                (cash * RISK_PCT) / _dist_w, ts, "width"))
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
                    entry_fractal_bar   = last_fractal_low_bar
                    entry_fractal_label = last_low_label
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
                    entry_fractal_bar   = last_fractal_high_bar
                    entry_fractal_label = last_high_label
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
    is_mid_range   = 1 <= _day_span <= 30   # 2–31 calendar days
    is_long_range  = _day_span >= 31        # 32+ calendar days
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

    # Cubic interpolation for silky-smooth EMA curves (skip if no EMAs or no data)
    _any_ema = EMA_SHORT > 0 or EMA_MID > 0 or EMA_LONG > 0
    if _any_ema and len(ds_dates) > 1:
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
        if EMA_LONG > 0:
            ax1.plot(dates, df.ema_long.values,  color="#ff6b6b", linewidth=1.4,
                     label=f"EMA {EMA_LONG}", zorder=4)
        if EMA_MID > 0:
            ax1.plot(dates, df.ema_mid.values,  color="#ffd93d", linewidth=1.2,
                     label=f"EMA {EMA_MID}", zorder=4)
        if EMA_SHORT > 0:
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
            # Collect N=18 and N=6 pivot coordinates for trendlines
            _n18_highs_xy = []  # list of (dt_num, price_y) for N=18 high pivots
            _n18_lows_xy  = []  # list of (dt_num, price_y) for N=18 low pivots
            _n6_highs_xy  = []  # list of (dt_num, price_y) for N=6 high pivots
            _n6_lows_xy   = []  # list of (dt_num, price_y) for N=6 low pivots

            # ── L# counter state (consecutive lower-highs) ───────────────
            _lh_counter = 0
            _prev_high_price = None
            _lh_offset = max(_price_range * 0.045, 5e-5)  # above F# label

            for _pv_idx, _pv in enumerate(_pvd['pivots']):
                _bar_i  = _pv['bar']
                # Priority: N=18 yellow > N=6 white > N=2 red/green
                if _pv.get('n18'):
                    _pv_clr = '#ffd700'       # yellow — N=18
                elif _pv.get('n6'):
                    _pv_clr = '#ffffff'        # white  — N=6
                else:
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
                _fnum_clr = _pivot_colors.get(_pv['label'], '#ffffff')
                ax1.text(
                    _dt_nums[_bar_i], _lbl_y, str(_pv_idx + 1),
                    color=_fnum_clr, fontsize=6, fontweight='bold',
                    ha='center', va='bottom' if _pv['kind'] == 'H' else 'top',
                    zorder=7,
                )
                # ── L# label (consecutive lower-highs) ──────────────
                _is_high = _pv['label'] in ('LH', 'HH', 'CH')
                _lh_num_str = None
                if _is_high:
                    if _pv['label'] == 'LH' and _prev_high_price is not None and _pv['price'] < _prev_high_price:
                        _lh_counter += 1
                        _lh_num_str = str(_lh_counter)
                    elif _prev_high_price is not None and _pv['price'] > _prev_high_price:
                        _lh_counter = 0
                    _prev_high_price = _pv['price']
                if _lh_num_str and _pv['kind'] == 'H':
                    ax1.text(
                        _dt_nums[_bar_i], _pv_y + _lh_offset, _lh_num_str,
                        color='#ffffff', fontsize=6, fontweight='bold',
                        ha='center', va='bottom', zorder=7,
                    )
                # Track N=18 pivots for trendlines
                if _pv.get('n18'):
                    if _pv['kind'] == 'H':
                        _n18_highs_xy.append((_dt_nums[_bar_i], _pv_y))
                    else:
                        _n18_lows_xy.append((_dt_nums[_bar_i], _pv_y))
                # Track N=6 pivots for trendlines
                if _pv.get('n6'):
                    if _pv['kind'] == 'H':
                        _n6_highs_xy.append((_dt_nums[_bar_i], _pv_y))
                    else:
                        _n6_lows_xy.append((_dt_nums[_bar_i], _pv_y))

            # ── N=6 trendlines (white thin dashed) — disabled ─────────────
            # if len(_n6_highs_xy) >= 2:
            #     _hx = [p[0] for p in _n6_highs_xy]
            #     _hy = [p[1] for p in _n6_highs_xy]
            #     ax1.plot(_hx, _hy, color='#ffffff', linestyle='--',
            #              linewidth=0.5, alpha=0.5, zorder=4)
            # if len(_n6_lows_xy) >= 2:
            #     _lx = [p[0] for p in _n6_lows_xy]
            #     _ly = [p[1] for p in _n6_lows_xy]
            #     ax1.plot(_lx, _ly, color='#ffffff', linestyle='--',
            #              linewidth=0.5, alpha=0.5, zorder=4)

            # ── N=18 trendlines (yellow dashed) ──────────────────────────────
            if len(_n18_highs_xy) >= 2:
                _hx = [p[0] for p in _n18_highs_xy]
                _hy = [p[1] for p in _n18_highs_xy]
                ax1.plot(_hx, _hy, color='#ffd700', linestyle='--',
                         linewidth=0.9, alpha=0.7, zorder=5)
            if len(_n18_lows_xy) >= 2:
                _lx = [p[0] for p in _n18_lows_xy]
                _ly = [p[1] for p in _n18_lows_xy]
                ax1.plot(_lx, _ly, color='#ffd700', linestyle='--',
                         linewidth=0.9, alpha=0.7, zorder=5)

        ax1.set_xlim(
            mdates.num2date(_dt_nums[0]  - _bw),
            mdates.num2date(_dt_nums[-1] + _bw)
        )
    elif is_mid_range:
        # ── 2–31 days: price line + N18 dots & trendlines, no EMAs/N2/N6 dots ──
        ax1.plot(ds_dates, ds_close, color="#e0e0e0", linewidth=0.5, label="Price", alpha=0.7)
        # N18 (and optionally N6) pivot overlay
        _pvd_mid = compute_pivot_diagnostics(df)
        if _pvd_mid and _pvd_mid.get('pivots'):
            _all_dates_num = mdates.date2num(dates)
            _highs_arr = df['High'].values
            _lows_arr  = df['Low'].values
            _price_range_mid = _highs_arr.max() - _lows_arr.min()
            _pv_offset_mid   = max(_price_range_mid * 0.008, 1e-5)
            _n18_highs_xy_mid = []
            _n18_lows_xy_mid  = []
            _n6_highs_xy_mid  = []
            _n6_lows_xy_mid   = []
            for _pv in _pvd_mid['pivots']:
                _bar_i = _pv['bar']
                if _bar_i < 0 or _bar_i >= len(_all_dates_num):
                    continue
                if _pv['kind'] == 'H':
                    _pv_y = _pv['price'] + _pv_offset_mid
                else:
                    _pv_y = _pv['price'] - _pv_offset_mid
                # N18 dots (all mid-range)
                if _pv.get('n18'):
                    if _pv['kind'] == 'H':
                        _n18_highs_xy_mid.append((_all_dates_num[_bar_i], _pv_y))
                    else:
                        _n18_lows_xy_mid.append((_all_dates_num[_bar_i], _pv_y))
                    ax1.scatter(
                        _all_dates_num[_bar_i], _pv_y,
                        color='#ffd700', marker='o', s=18, zorder=6,
                        edgecolors='none',
                    )
                # Track N6 pivots for trendlines (2-day only)
                if _pv.get('n6') and _day_span == 1:
                    if _pv['kind'] == 'H':
                        _n6_highs_xy_mid.append((_all_dates_num[_bar_i], _pv_y))
                    else:
                        _n6_lows_xy_mid.append((_all_dates_num[_bar_i], _pv_y))
            # N6 trendlines (white thin dashed — 2-day only)
            if _day_span == 1:
                if len(_n6_highs_xy_mid) >= 2:
                    _hx = [p[0] for p in _n6_highs_xy_mid]
                    _hy = [p[1] for p in _n6_highs_xy_mid]
                    ax1.plot(_hx, _hy, color='#ffffff', linestyle='--',
                             linewidth=0.5, alpha=0.5, zorder=4)
                if len(_n6_lows_xy_mid) >= 2:
                    _lx = [p[0] for p in _n6_lows_xy_mid]
                    _ly = [p[1] for p in _n6_lows_xy_mid]
                    ax1.plot(_lx, _ly, color='#ffffff', linestyle='--',
                             linewidth=0.5, alpha=0.5, zorder=4)
            # N18 trendlines (yellow dashed)
            if len(_n18_highs_xy_mid) >= 2:
                _hx = [p[0] for p in _n18_highs_xy_mid]
                _hy = [p[1] for p in _n18_highs_xy_mid]
                ax1.plot(_hx, _hy, color='#ffd700', linestyle='--',
                         linewidth=0.9, alpha=0.7, zorder=5)
            if len(_n18_lows_xy_mid) >= 2:
                _lx = [p[0] for p in _n18_lows_xy_mid]
                _ly = [p[1] for p in _n18_lows_xy_mid]
                ax1.plot(_lx, _ly, color='#ffd700', linestyle='--',
                         linewidth=0.9, alpha=0.7, zorder=5)

    else:
        # ── 32+ days: price line + trade markers only ────────────────────────
        ax1.plot(ds_dates, ds_close, color="#e0e0e0", linewidth=0.5, label="Price", alpha=0.7)

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
            entry_date = dates.iloc[idx]
            color  = "#6bcb77" if t.win else "#ff6b6b"
            # Vertical line behind the entry candle (green win / red loss)
            ax1.axvline(entry_date, color=color, linewidth=0.5,
                        alpha=1.0, zorder=1)
            marker = "^" if t.direction == "long" else "v"
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

    _title_fmt = lambda d: d.strftime("%b-%d-%y")
    _chart_title = _title_fmt(_date_min) if is_one_day else f"{_title_fmt(_date_min)} → {_title_fmt(_date_max)}"
    ax1.set_title(_chart_title, color="white", fontsize=13, pad=10)
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
        _n_days = (dates.max() - dates.min()).days + 1
        if _n_days > 1:
            times_str = dts_utc.dt.strftime('%b-%d %H:%M')
        else:
            times_str = dts_utc.dt.strftime('%H:%M')
    except Exception:
        return None

    n_days = _n_days
    if n_days > 31:
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

    # ── Detect fractal pivots (N=2: need 2 bars on each side) ──────────────
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

    # ── Detect N=6 fractal pivots (need 6 bars on each side) ─────────────
    # Build a set of bar indices that have an N=6 fractal (high, low, or both)
    n6_high_bars = set()
    n6_low_bars  = set()
    for i in range(6, n_bars - 6):
        h_i = highs[i]
        l_i = lows[i]
        is_ph6 = all(h_i > highs[i - k] for k in range(1, 7)) and \
                 all(h_i > highs[i + k] for k in range(1, 7))
        is_pl6 = all(l_i < lows[i - k] for k in range(1, 7)) and \
                 all(l_i < lows[i + k] for k in range(1, 7))
        if is_ph6:
            n6_high_bars.add(i)
        if is_pl6:
            n6_low_bars.add(i)

    # ── Classify N=6 pivots independently ────────────────────────────────
    # Collect N=6 raw pivots sorted by bar, then classify with own state
    n6_raw = []
    for i in sorted(n6_high_bars | n6_low_bars):
        atr_i = float(atr14.iloc[i])
        if i in n6_high_bars:
            n6_raw.append({'kind': 'H', 'price': float(highs[i]), 'bar': i, 'atr': atr_i})
        if i in n6_low_bars:
            n6_raw.append({'kind': 'L', 'price': float(lows[i]),  'bar': i, 'atr': atr_i})
    n6_raw.sort(key=lambda x: x['bar'])

    n6_prev_high = None
    n6_prev_low  = None
    n6_labels    = {}  # bar → label string for N=6 classification
    for pv in n6_raw:
        price     = pv['price']
        threshold = 0.5 * pv['atr']
        if pv['kind'] == 'H':
            if n6_prev_high is None:
                lbl6 = 'CH'
            else:
                diff = price - n6_prev_high['price']
                lbl6 = 'CH' if abs(diff) < threshold else ('LH' if diff < 0 else 'HH')
            n6_prev_high = pv
        else:
            if n6_prev_low is None:
                lbl6 = 'CL'
            else:
                diff = price - n6_prev_low['price']
                lbl6 = 'CL' if abs(diff) < threshold else ('HL' if diff > 0 else 'LL')
            n6_prev_low = pv
        n6_labels[(pv['bar'], pv['kind'])] = lbl6

    # ── Compute Cycle state for each N=6 pivot ──────────────────────────────
    n6_ordered_labels = []  # running list of N6 labels in chronological order
    n6_cycle_at_bar = {}    # bar → cycle label (set only on N6 pivot bars)
    for pv in n6_raw:
        key = (pv['bar'], pv['kind'])
        lbl = n6_labels.get(key)
        if lbl:
            n6_ordered_labels.append(lbl)
        window = n6_ordered_labels[-10:]
        if len(window) < 5:
            n6_cyc = '\u2014'
        else:
            bullish_dir = sum(1 for l in window if l in ('HH', 'HL'))
            bearish_dir = sum(1 for l in window if l in ('LL', 'LH'))
            consol      = sum(1 for l in window if l in ('CH', 'CL'))
            most_recent = window[-1]
            prior_window = window[:-1]
            prior_bull = sum(1 for l in prior_window if l in ('HH', 'HL'))
            prior_bear = sum(1 for l in prior_window if l in ('LL', 'LH'))
            if prior_bull > prior_bear and prior_bull >= 4 and most_recent in ('LL', 'LH'):
                n6_cyc = 'Transitioning'
            elif prior_bear > prior_bull and prior_bear >= 4 and most_recent in ('HH', 'HL'):
                n6_cyc = 'Transitioning'
            elif bullish_dir >= 6:
                n6_cyc = 'Trending \u2191'
            elif bearish_dir >= 6:
                n6_cyc = 'Trending \u2193'
            else:
                n6_cyc = 'Consolidating'
        n6_cycle_at_bar[pv['bar']] = n6_cyc

    # ── Detect N=18 fractal pivots (need 18 bars on each side) ───────────
    n18_high_bars = set()
    n18_low_bars  = set()
    for i in range(18, n_bars - 18):
        h_i = highs[i]
        l_i = lows[i]
        is_ph18 = all(h_i > highs[i - k] for k in range(1, 19)) and \
                  all(h_i > highs[i + k] for k in range(1, 19))
        is_pl18 = all(l_i < lows[i - k] for k in range(1, 19)) and \
                  all(l_i < lows[i + k] for k in range(1, 19))
        if is_ph18:
            n18_high_bars.add(i)
        if is_pl18:
            n18_low_bars.add(i)

    # ── Classify N=18 pivots independently ───────────────────────────────
    n18_raw = []
    for i in sorted(n18_high_bars | n18_low_bars):
        atr_i = float(atr14.iloc[i])
        if i in n18_high_bars:
            n18_raw.append({'kind': 'H', 'price': float(highs[i]), 'bar': i, 'atr': atr_i})
        if i in n18_low_bars:
            n18_raw.append({'kind': 'L', 'price': float(lows[i]),  'bar': i, 'atr': atr_i})
    n18_raw.sort(key=lambda x: x['bar'])

    n18_prev_high = None
    n18_prev_low  = None
    n18_labels    = {}  # (bar, kind) → label string for N=18 classification
    for pv in n18_raw:
        price     = pv['price']
        threshold = 0.5 * pv['atr']
        if pv['kind'] == 'H':
            if n18_prev_high is None:
                lbl18 = 'CH'
            else:
                diff = price - n18_prev_high['price']
                lbl18 = 'CH' if abs(diff) < threshold else ('LH' if diff < 0 else 'HH')
            n18_prev_high = pv
        else:
            if n18_prev_low is None:
                lbl18 = 'CL'
            else:
                diff = price - n18_prev_low['price']
                lbl18 = 'CL' if abs(diff) < threshold else ('HL' if diff > 0 else 'LL')
            n18_prev_low = pv
        n18_labels[(pv['bar'], pv['kind'])] = lbl18

    # ── Compute Cycle state for each N=18 pivot ─────────────────────────────
    # Build ordered list of N18 labels, then classify cycle at each step
    n18_ordered_labels = []  # running list of N18 labels in chronological order
    n18_cycle_at_bar = {}    # bar → cycle label (set only on N18 pivot bars)
    for pv in n18_raw:
        key = (pv['bar'], pv['kind'])
        lbl = n18_labels.get(key)
        if lbl:
            n18_ordered_labels.append(lbl)
        window = n18_ordered_labels[-10:]  # last 10 N18 fractals
        if len(window) < 5:
            cycle_lbl = '\u2014'  # insufficient context
        else:
            # Score the window
            bullish_dir = sum(1 for l in window if l in ('HH', 'HL'))
            bearish_dir = sum(1 for l in window if l in ('LL', 'LH'))
            consol      = sum(1 for l in window if l in ('CH', 'CL'))
            # Transition: dominant direction exists but most recent pivot breaks it
            most_recent = window[-1]
            prior_window = window[:-1]
            prior_bull = sum(1 for l in prior_window if l in ('HH', 'HL'))
            prior_bear = sum(1 for l in prior_window if l in ('LL', 'LH'))
            if prior_bull > prior_bear and prior_bull >= 4 and most_recent in ('LL', 'LH'):
                cycle_lbl = 'Transitioning'
            elif prior_bear > prior_bull and prior_bear >= 4 and most_recent in ('HH', 'HL'):
                cycle_lbl = 'Transitioning'
            elif bullish_dir >= 6:
                cycle_lbl = 'Trending \u2191'
            elif bearish_dir >= 6:
                cycle_lbl = 'Trending \u2193'
            elif consol >= 6 or (bullish_dir < 6 and bearish_dir < 6 and consol < 6):
                cycle_lbl = 'Consolidating'
            else:
                cycle_lbl = 'Consolidating'
        n18_cycle_at_bar[pv['bar']] = cycle_lbl

    # ── Compute Width score at each N18 or N6 pivot bar ─────────────────────
    # Merge N18 and N6 pivots in bar order; at each, recalculate width score.
    _width_events = []
    for pv in n18_raw:
        _width_events.append(('n18', pv['kind'], pv['price'], pv['bar'], pv['atr']))
    for pv in n6_raw:
        _width_events.append(('n6', pv['kind'], pv['price'], pv['bar'], pv['atr']))
    _width_events.sort(key=lambda x: (x[3], 0 if x[0] == 'n18' else 1))

    _w_n18_hi = None    # most recent N18 high pivot price
    _w_n18_lo = None    # most recent N18 low pivot price
    _w_n6_hi  = None    # most recent N6 high pivot price
    _w_n6_lo  = None    # most recent N6 low pivot price
    width_at_bar = {}   # bar → width score (1-4) or None
    for (_src, _kind, _price, _bar, _atr_val) in _width_events:
        if _src == 'n18':
            if _kind == 'H':
                _w_n18_hi = _price
            else:
                _w_n18_lo = _price
        else:  # n6
            if _kind == 'H':
                _w_n6_hi = _price
            else:
                _w_n6_lo = _price
        # Compute score if both N18 and N6 have high+low pivots
        if _w_n18_hi is not None and _w_n18_lo is not None and \
           _w_n6_hi is not None and _w_n6_lo is not None and _atr_val > 0:
            n18_width = abs(_w_n18_hi - _w_n18_lo) / _atr_val
            n6_width  = abs(_w_n6_hi  - _w_n6_lo)  / _atr_val
            n18_wide  = n18_width >= N18_WIDTH_THRESHOLD
            n6_wide   = n6_width  >= N6_WIDTH_THRESHOLD
            if not n18_wide and not n6_wide:
                width_at_bar[_bar] = 1
            elif not n18_wide and n6_wide:
                width_at_bar[_bar] = 2
            elif n18_wide and not n6_wide:
                width_at_bar[_bar] = 3
            else:
                width_at_bar[_bar] = 4
        else:
            width_at_bar[_bar] = None  # insufficient history

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
                vert_dir   = None
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
                vert_dir   = 'up' if diff >= 0 else 'down'
                horiz_dist = pv['bar'] - prev_high['bar']

            _n6_key  = (pv['bar'], 'H')
            _n18_key = (pv['bar'], 'H')
            classified.append({
                'label':      label,
                'price':      price,
                'time':       pv['time'],
                'vert_dist':  vert_dist,
                'vert_dir':   vert_dir,
                'horiz_dist': horiz_dist,
                'bar':        pv['bar'],
                'kind':       pv['kind'],
                'atr':        round(pv['atr'] * 10000, 1),  # ATR in pips
                'adx':        round(pv['adx'], 1) if pv['adx'] is not None else None,
                'n6':         _n6_key in n6_labels,
                'n6_label':   n6_labels.get(_n6_key),
                'n18':        _n18_key in n18_labels,
                'n18_label':  n18_labels.get(_n18_key),
            })
            prev_high = pv

        else:  # kind == 'L'
            if prev_low is None:
                label      = 'CL'         # first pivot low — no prior to compare, treat as consolidating
                vert_dist  = None
                vert_dir   = None
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
                vert_dir   = 'up' if diff >= 0 else 'down'
                horiz_dist = pv['bar'] - prev_low['bar']

            _n6_key  = (pv['bar'], 'L')
            _n18_key = (pv['bar'], 'L')
            classified.append({
                'label':      label,
                'price':      price,
                'time':       pv['time'],
                'vert_dist':  vert_dist,
                'vert_dir':   vert_dir,
                'horiz_dist': horiz_dist,
                'bar':        pv['bar'],
                'kind':       pv['kind'],
                'atr':        round(pv['atr'] * 10000, 1),  # ATR in pips
                'adx':        round(pv['adx'], 1) if pv['adx'] is not None else None,
                'n6':         _n6_key in n6_labels,
                'n6_label':   n6_labels.get(_n6_key),
                'n18':        _n18_key in n18_labels,
                'n18_label':  n18_labels.get(_n18_key),
            })
            prev_low = pv

    # ── Attach carried-forward Cycle labels and Width score to each pivot ───────
    _carried_cycle = ''
    _carried_n6_cycle = ''
    _carried_width = None
    for pv in classified:
        bar = pv['bar']
        if bar in n18_cycle_at_bar:
            _carried_cycle = n18_cycle_at_bar[bar]
        pv['cycle_label'] = _carried_cycle if _carried_cycle else None
        if bar in n6_cycle_at_bar:
            _carried_n6_cycle = n6_cycle_at_bar[bar]
        pv['n6_cycle_label'] = _carried_n6_cycle if _carried_n6_cycle else None
        if bar in width_at_bar:
            _carried_width = width_at_bar[bar]
        pv['width_score'] = _carried_width

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
            # Prior range = most recent CH − most recent CL
            # Pullback    = this CH − most recent CL
            ref_ch = _last_price('CH', prior)
            ref_cl = _last_price('CL', prior)
            if ref_ch is not None and ref_cl is not None:
                prior_range = ref_ch - ref_cl
                pullback    = price  - ref_cl
                if abs(prior_range) > 1e-10:
                    pb = pullback / prior_range * 100

        elif lbl == 'CL':
            # Prior range = most recent CH − most recent CL
            # Pullback    = most recent CH − this CL
            ref_ch = _last_price('CH', prior)
            ref_cl = _last_price('CL', prior)
            if ref_ch is not None and ref_cl is not None:
                prior_range = ref_ch - ref_cl
                pullback    = ref_ch - price
                if abs(prior_range) > 1e-10:
                    pb = pullback / prior_range * 100

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
        if len(unique_dates) <= 31:
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
                    "fractal_bar":   int(row_t["fractal_bar"]) if row_t.get("fractal_bar") is not None else None,
                    "fractal_label": row_t.get("fractal_label"),
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
        "gross_profit":        round(float(wins.pnl.sum()), 2) if not wins.empty else 0,
        "gross_loss":          round(float(loss.pnl.sum()), 2) if not loss.empty else 0,
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
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <select id="version-select"></select>
    <select id="instrument-select" style="margin-top:8px;">
      <option value="EURUSD">EURUSD</option>
      <option value="GBPUSD">GBPUSD</option>
    </select>
  </div>
  <div id="version-list"></div>
</div>

<div id="main">
  <div id="content">
  </div>
</div>

<div id="copy-toast">&#10003;&nbsp; Copied to clipboard!</div>

<!-- Action buttons: hidden by default, moved into the run bar by server.py -->
<button id="devlog-btn" title="Development Log" style="display:none;"><span class="material-symbols-outlined">list</span></button>
<span class="rb-sep" id="rb-act-sep" style="display:none;"></span>
<button id="copy-btn" style="display:none;">Copy Report</button>

<script type="application/json" id="versions-data">
__VERSIONS_JSON__
</script>

<script>
(function () {
  "use strict";

  var VERSIONS = JSON.parse(document.getElementById("versions-data").textContent);
  var currentVersion = "";  /* selected version name, e.g. "v1" */
  var currentInstrument = "";  /* selected instrument, e.g. "EURUSD" */
  var devLogOpen = false;
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
    lines.push("## Backtest Report \u2014 " + (ver.strategy_version || ver.name || "?") + runLabel);
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
    lines.push("| Gross Profit | +$"  + commaFmt(m.gross_profit) + " |");
    lines.push("| Gross Loss | -$"    + commaFmt(Math.abs(m.gross_loss)) + " |");
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
    lines.push("| EMA Short | "     + (p.ema_short != null ? p.ema_short : "\u2014") + " |");
    lines.push("| EMA Mid | "       + (p.ema_mid   != null ? p.ema_mid   : "\u2014") + " |");
    lines.push("| EMA Long | "      + (p.ema_long  != null ? p.ema_long  : "\u2014") + " |");
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
      lines.push("| Date | Entry Time | Duration | Direction | Stop | Target | F # | F Type | ATR | ADX | VD | HD | PB % | P&L |");
      lines.push("|------|------------|----------|-----------|------|--------|-----|--------|-----|-----|-----|-----|------|-----|");
      /* Build bar→pivot# lookup from pivot diagnostics */
      var mdPvd = m.pivot_diagnostics || {};
      var mdPivotList = mdPvd.pivots || [];
      var mdBarToPivot = {};
      var mdBarToPullback = {};
      var mdBarToVert = {};
      var mdBarToHoriz = {};
      mdPivotList.forEach(function (pv, idx) {
        mdBarToPivot[pv.bar] = idx + 1;
        mdBarToPullback[pv.bar] = pv.pullback_pct;
        mdBarToVert[pv.bar] = pv.vert_dist;
        mdBarToHoriz[pv.bar] = pv.horiz_dist;
      });
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
        var atrD = (t.atr_pips !== null && t.atr_pips !== undefined) ? mf(t.atr_pips, 1) : "\u2014";
        var adxD = (t.adx     !== null && t.adx     !== undefined) ? mf(t.adx, 1)      : "\u2014";
        var fNum  = (t.fractal_bar !== null && t.fractal_bar !== undefined && mdBarToPivot[t.fractal_bar]) ? mdBarToPivot[t.fractal_bar] : "\u2014";
        var fType = t.fractal_label || "\u2014";
        var vdVal = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? mdBarToVert[t.fractal_bar] : null;
        var vdD = (vdVal !== null && vdVal !== undefined) ? mf(vdVal, 1) : "\u2014";
        var hdVal = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? mdBarToHoriz[t.fractal_bar] : null;
        var hdD = (hdVal !== null && hdVal !== undefined) ? hdVal : "\u2014";
        var pbVal = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? mdBarToPullback[t.fractal_bar] : null;
        var pbD = (pbVal !== null && pbVal !== undefined) ? mf(pbVal, 1) + "%" : "\u2014";
        lines.push("| " + iDateLabel + " | " + t.entry_time + " UTC | " + mdFmtDur(t.duration) + " | " + dir + " | " + stopD + " | " + targetD + " | " + fNum + " | " + fType + " | " + atrD + " | " + adxD + " | " + vdD + " | " + hdD + " | " + pbD + " | " + mfMoney(t.pnl) + " |");
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
      lines.push("### Fractals");
      lines.push("");
      var pvList = pvd.pivots || [];
      if (pvList.length === 0) {
        lines.push("No fractal pivot points detected in this date range.");
      } else {
        lines.push("| # | Type 2 | L# | Pullback % | Width | Price | Time | ATR (pips) | ADX | VD High | VD Low | Horiz Distance (bars) |");
        lines.push("|---|--------|----|-----------|----|-------|------|------------|-----|---------|--------|-----------------------|");
        var mdBarOutcome = {};
        (m.intraday || []).forEach(function (t) {
          if (t.fractal_bar !== null && t.fractal_bar !== undefined) {
            mdBarOutcome[t.fractal_bar] = t.pnl >= 0 ? "W" : "L";
          }
        });
        var mdCarriedN18 = "";
        var mdCarriedCycle = "";
        var mdCarriedN6Cycle = "";
        var _mdLhCounter = 0;
        var _mdPrevHighPrice = null;
        pvList.forEach(function (pv, idx) {
          var mdVertHigh = "";
          var mdVertLow  = "";
          if (pv.vert_dist !== null && pv.vert_dist !== undefined) {
            var _mdVArrow = pv.vert_dir === "up" ? "▲ " : "▼ ";
            if (pv.kind === "H") {
              mdVertHigh = _mdVArrow + mf(pv.vert_dist, 1);
            } else {
              mdVertLow  = _mdVArrow + mf(pv.vert_dist, 1);
            }
          }
          var horizD   = (pv.horiz_dist   !== null && pv.horiz_dist   !== undefined) ? String(pv.horiz_dist) : "";
          var pullbackD = (pv.pullback_pct !== null && pv.pullback_pct !== undefined) ? mf(pv.pullback_pct, 1) + "%" : "";
          var atrD     = (pv.atr          !== null && pv.atr          !== undefined) ? mf(pv.atr, 1) : "";
          var adxD     = (pv.adx          !== null && pv.adx          !== undefined) ? mf(pv.adx, 1) : "";
          var mdType1 = "";
          if (pv.n18 && pv.n18_label) {
            mdCarriedN18 = pv.n18_label;
            mdType1 = pv.n18_label + " \u2022";
          } else if (mdCarriedN18) {
            mdType1 = mdCarriedN18;
          }
          if (pv.cycle_label) {
            mdCarriedCycle = pv.cycle_label;
          }
          var mdCycle = mdCarriedCycle || "";
          if (pv.n6_cycle_label) {
            mdCarriedN6Cycle = pv.n6_cycle_label;
          }
          var mdN6Cycle = mdCarriedN6Cycle || "";
          var mdType2 = "";
          if (pv.label) {
            mdType2 = pv.label + (pv.n6 ? " \u2022" : "");
          }
          var mdWidth = (pv.width_score !== null && pv.width_score !== undefined) ? String(pv.width_score) : "";
          var mdLhNum = "";
          var _mdIsHigh = (pv.label === "LH" || pv.label === "HH" || pv.label === "CH");
          if (_mdIsHigh) {
            if (pv.label === "LH" && _mdPrevHighPrice !== null && pv.price < _mdPrevHighPrice) {
              _mdLhCounter++;
              mdLhNum = String(_mdLhCounter);
            } else if (_mdPrevHighPrice !== null && pv.price > _mdPrevHighPrice) {
              _mdLhCounter = 0;
            }
            _mdPrevHighPrice = pv.price;
          }
          var mdNum = String(idx + 1);
          var mdOc = mdBarOutcome[pv.bar];
          if (mdOc) mdNum += " " + mdOc;
          lines.push("| " + mdNum + " | " + (mdType2 || "") + " | " + mdLhNum + " | " + pullbackD + " | " + mdWidth + " | " +
            mf(pv.price, 5) + " | " + (pv.time || "") + " | " + atrD + " | " + adxD + " | " + mdVertHigh + " | " + mdVertLow + " | " + horizD + " |");
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
      var sv = VERSIONS[i].strategy_version || "v1";
      if (sv === currentVersion) {
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

  /* Track which section groups are expanded (persists across re-renders) */
  var _sectionState = (function () {
    try { return JSON.parse(localStorage.getItem("sb_section_state")) || {}; }
    catch (_) { return {}; }
  })();
  function _saveSectionState() {
    try { localStorage.setItem("sb_section_state", JSON.stringify(_sectionState)); } catch (_) {}
  }

  function _durationDays(startStr, endStr) {
    if (!startStr || !endStr) return 0;
    var s = new Date(startStr.slice(0,10) + "T00:00:00");
    var e = new Date(endStr.slice(0,10) + "T00:00:00");
    var d = Math.round((e - s) / 86400000) + 1;
    return d < 1 ? 1 : d;
  }

  function _sectionFor(days) {
    if (days >= 365) return "Year";
    if (days >= 28 && days <= 31) return "Month";
    if (days >= 2 && days <= 27) return "Weeks";
    if (days === 1) return "Day";
    /* 32-364 days that aren't exactly a month — group with Weeks as closest fit */
    if (days > 31 && days < 365) return "Weeks";
    return "Other";
  }

  /* Canonical order for sections */
  var _sectionOrder = ["Year", "Month", "Weeks", "Day", "Other"];

  function renderSidebar() {
    var svs  = getStrategyVersions();
    var list = document.getElementById("version-list");
    list.innerHTML = "";

    if (svs.length === 0) {
      list.innerHTML = "<div class='sb-no-runs'>No runs for this version yet.</div>";
      return;
    }

    /* Build flat list: every run from every matching version is a top-level item */
    var flatItems = []; /* { vIdx, runIdx, v, run } */
    for (var ri = 0; ri < svs.length; ri++) {
      var entry = svs[ri];
      var v     = entry.v;
      var idx   = entry.idx;
      var runs  = getRuns(v);
      for (var si = 0; si < runs.length; si++) {
        var _ri = runs[si];
        var _rinst = (_ri.instrument || (v.params && v.params.ticker ? v.params.ticker.replace(/=X$/i, "") : "")).toUpperCase();
        if (currentInstrument && _rinst && _rinst !== currentInstrument) continue;
        flatItems.push({ vIdx: idx, runIdx: si, v: v, run: _ri });
      }
    }

    if (flatItems.length === 0) {
      list.innerHTML = "<div class='sb-no-runs'>No runs for " + currentInstrument + " yet.</div>";
      return;
    }

    /* ── Group items by date-range section ── */
    var groups = {}; /* sectionLabel → [ item, ... ] */
    for (var fi = 0; fi < flatItems.length; fi++) {
      var item = flatItems[fi];
      var run  = item.run;
      var range = run.start_date && run.end_date
        ? { start: run.start_date, end: run.end_date }
        : fullRunRange(run);
      var days = _durationDays(range.start, range.end);
      var sec  = _sectionFor(days);
      if (!groups[sec]) groups[sec] = [];
      groups[sec].push(item);
    }

    /* Render sections in canonical order */
    for (var si = 0; si < _sectionOrder.length; si++) {
      var secLabel = _sectionOrder[si];
      var secItems = groups[secLabel];
      if (!secItems || secItems.length === 0) continue;

      var isExpanded = !!_sectionState[secLabel]; /* collapsed by default */

      /* ── Section header row ── */
      var secEl = document.createElement("div");
      secEl.className = "sb-section-header" + (isExpanded ? " expanded" : "");
      secEl.innerHTML =
        "<span class='sb-section-arrow'>\u25B6</span>" +
        "<span class='sb-section-label'>" + esc(secLabel) + "</span>" +
        "<span class='sb-section-count'>" + secItems.length + "</span>";

      (function (secLabel, secEl) {
        secEl.addEventListener("click", function () {
          _sectionState[secLabel] = !_sectionState[secLabel];
          _saveSectionState();
          renderSidebar();
        });
      })(secLabel, secEl);

      list.appendChild(secEl);

      /* ── Section items (hidden when collapsed) ── */
      if (!isExpanded) continue;

      for (var ii = 0; ii < secItems.length; ii++) {
        var item = secItems[ii];
        var run  = item.run;
        var pnl  = run.metrics ? run.metrics.net_profit : null;
        var pc   = pnl === null ? "" : (pnl >= 0 ? "pos" : "neg");
        var ptxt = pnl === null ? "" : (pnl >= 0 ? "+" : "") + "$" + commaFmt(pnl);

        var range = run.start_date && run.end_date
          ? { start: run.start_date, end: run.end_date }
          : fullRunRange(run);
        var dateRange = range.start && range.end
          ? fmtSbDate(range.start) + " \u2192 " + fmtSbDate(range.end) : "";
        var dur = calcDuration(range.start, range.end);

        var runInstrument = run.instrument || (item.v.params && item.v.params.ticker ? item.v.params.ticker.replace(/=X$/i, "") : "");

        var el = document.createElement("div");
        el.className = "v-item" + (item.vIdx === activeVersionIdx && item.runIdx === activeRunIdx ? " active" : "");
        el.dataset.idx = item.vIdx;
        el.dataset.runIdx = item.runIdx;
        el.draggable = true;

        var totalRuns = getRuns(item.v).length;

        el.innerHTML =
          "<div class='v-item-row'>" +
            "<div class='v-item-content'>" +
              "<div class='v-sub-top-row'>" +
                "<div class='v-name'>" + esc(item.v.strategy_version || item.v.name) + "</div>" +
                (runInstrument ? "<div class='v-instrument'>" + esc(runInstrument) + "</div>" : "") +
              "</div>" +
              "<div class='v-sub-metric-row'>" +
                (dur ? "<div class='v-duration'>" + esc(dur) + "</div>" : "") +
                (pnl !== null ? "<div class='v-pnl " + pc + "'>" + ptxt + "</div>" : "") +
              "</div>" +
              (dateRange ? "<div class='v-date date-link' data-start='" + esc(range.start) + "' data-end='" + esc(range.end) + "'>" + esc(dateRange) + "</div>" : "") +
            "</div>" +
            "<button class='v-sub-delete-btn' title='Delete run'>&times;</button>" +
          "</div>";

        (function (el, vIdx, rIdx, verName, totalRuns) {
          el.addEventListener("click", function (e) {
            if (e.target.closest(".v-sub-delete-btn")) return;
            var dl = e.target.closest(".date-link[data-start]");
            if (dl) { setDatePicker(dl.dataset.start, dl.dataset.end); return; }
            devLogOpen = false;
            document.getElementById("devlog-btn").classList.remove("active");
            activeVersionIdx = vIdx;
            activeRunIdx = rIdx;
            renderSidebar();
            renderContent(vIdx, rIdx);
          });
          /* Wire delete button */
          var delBtn = el.querySelector(".v-sub-delete-btn");
          if (delBtn) delBtn.addEventListener("click", function (e) {
            e.stopPropagation();
            if (totalRuns <= 1) {
              /* Only run in this version — delete the entire version */
              if (!confirm("Delete this version and its only run?")) return;
              delBtn.disabled = true;
              fetch("/delete_version", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: verName })
              })
              .then(function (r) { return r.json(); })
              .then(function (data) {
                if (data.ok) { window.location.reload(); }
                else { delBtn.disabled = false; alert("Delete failed: " + (data.error || "Unknown error")); }
              })
              .catch(function () { delBtn.disabled = false; alert("Delete failed — is the server running?"); });
            } else {
              /* Multiple runs — delete just this run */
              if (!confirm("Delete this run?")) return;
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
                  localStorage.setItem("rb_pending_delete_run_idx", String(focusIdx < 0 ? 0 : focusIdx));
                  window.location.reload();
                } else { delBtn.disabled = false; alert("Delete failed: " + (data.error || "Unknown error")); }
              })
              .catch(function () { delBtn.disabled = false; alert("Delete failed — is the server running?"); });
            }
          });

          /* ── Drag-and-drop reorder ── */
          el.addEventListener("dragstart", function (e) {
            e.stopPropagation();
            _dragSub = { vIdx: vIdx, runIdx: rIdx, verName: verName };
            /* Delay adding dragging class so the drag image captures the full item */
            setTimeout(function () { el.classList.add("dragging"); }, 0);
            e.dataTransfer.effectAllowed = "move";
            e.dataTransfer.setData("text/plain", "");
          });
          el.addEventListener("dragend", function () {
            el.classList.remove("dragging");
            _dragSub = null;
            document.querySelectorAll(".v-item.drag-over-above, .v-item.drag-over-below").forEach(function (x) {
              x.classList.remove("drag-over-above", "drag-over-below");
            });
          });
          el.addEventListener("dragover", function (e) {
            if (!_dragSub || _dragSub.vIdx !== vIdx) return;
            if (_dragSub.runIdx === rIdx) return; /* skip self */
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
            /* Clear indicators on all siblings first */
            document.querySelectorAll(".v-item.drag-over-above, .v-item.drag-over-below").forEach(function (x) {
              if (x !== el) x.classList.remove("drag-over-above", "drag-over-below");
            });
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
          el.addEventListener("dragleave", function (e) {
            /* Only clear if truly leaving this element (not entering a child) */
            if (!el.contains(e.relatedTarget)) {
              el.classList.remove("drag-over-above", "drag-over-below");
            }
          });
          el.addEventListener("drop", function (e) {
            e.preventDefault();
            el.classList.remove("drag-over-above", "drag-over-below");
            if (!_dragSub || _dragSub.vIdx !== vIdx) return;
            var fromIdx = _dragSub.runIdx;
            var rect = el.getBoundingClientRect();
            var mid = rect.top + rect.height / 2;
            var toIdx = e.clientY < mid ? rIdx : rIdx + 1;
            if (toIdx > fromIdx) toIdx--;
            if (fromIdx === toIdx) return;

            /* Build new order */
            var runs = getRuns(VERSIONS[vIdx]);
            var order = [];
            for (var oi = 0; oi < runs.length; oi++) order.push(oi);
            var moved = order.splice(fromIdx, 1)[0];
            order.splice(toIdx, 0, moved);

            fetch("/reorder_runs", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: verName, order: order })
            })
            .then(function (r) { return r.json(); })
            .then(function (data) {
              if (data.ok) {
                var oldRuns = VERSIONS[vIdx].runs;
                VERSIONS[vIdx].runs = [];
                for (var ri = 0; ri < order.length; ri++) VERSIONS[vIdx].runs.push(oldRuns[order[ri]]);
                activeVersionIdx = vIdx;
                activeRunIdx = toIdx;
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
        })(el, item.vIdx, item.runIdx, item.v.name, totalRuns);

        list.appendChild(el);
      }
    }

    /* Expose the currently active version name globally for the run-bar */
    var curV = VERSIONS[activeVersionIdx];
    window._currentVersionName = curV ? curV.name : "";
    window._currentVersionDisplayName = curV ? (curV.strategy_version || curV.name) : "";
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
      /* Build bar→pivot# lookup from pivot diagnostics */
      var pvdData = m.pivot_diagnostics || {};
      var pivotList = pvdData.pivots || [];
      var barToPivot = {};
      var barToPullback = {};
      var barToVert = {};
      var barToVertDir = {};
      var barToHoriz = {};
      pivotList.forEach(function (pv, idx) {
        barToPivot[pv.bar] = idx + 1;
        barToPullback[pv.bar] = pv.pullback_pct;
        barToVert[pv.bar] = pv.vert_dist;
        barToVertDir[pv.bar] = pv.vert_dir;
        barToHoriz[pv.bar] = pv.horiz_dist;
      });

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
        var fNum  = (t.fractal_bar !== null && t.fractal_bar !== undefined && barToPivot[t.fractal_bar]) ? barToPivot[t.fractal_bar] : "\u2014";
        var fType = t.fractal_label || "\u2014";
        iRows +=
          "<tr>" +
          "<td>" + esc(iDateLabel) + "</td>" +
          "<td>" + esc(t.entry_time) + " UTC</td>" +
          "<td>" + fmtDur(t.duration) + "</td>" +
          "<td class='" + dirCls + "'>" + esc(t.direction.charAt(0).toUpperCase() + t.direction.slice(1)) + "</td>" +
          "<td>" + stopD + "</td>" +
          "<td>" + targetD + "</td>" +
          "<td>" + fNum + "</td>" +
          "<td>" + esc(fType) + "</td>" +
          "<td>" + atrD + "</td>" +
          "<td>" + adxD + "</td>" +
          "<td>" + (function () { var v = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? barToVert[t.fractal_bar] : null; if (v === null || v === undefined) return "\u2014"; var vDir = barToVertDir[t.fractal_bar]; var vArrow = vDir === "up" ? "<span class='pos'>\u25B2</span> " : (vDir === "down" ? "<span class='neg'>\u25BC</span> " : ""); return vArrow + fmt(v, 1); }()) + "</td>" +
          "<td>" + (function () { var h = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? barToHoriz[t.fractal_bar] : null; return (h !== null && h !== undefined) ? h : "\u2014"; }()) + "</td>" +
          "<td>" + (function () { var pb = (t.fractal_bar !== null && t.fractal_bar !== undefined) ? barToPullback[t.fractal_bar] : null; return (pb !== null && pb !== undefined) ? fmt(pb, 1) + "%" : "\u2014"; }()) + "</td>" +
          "<td class='" + pnlCls + "'>" + fmtMoney(t.pnl) + "</td>" +
          "</tr>";
      });

      intradayPerfHtml =
        "<div class='section' id='anchor-intraday-perf'>" +
          "<div class='section-title'>Intraday Performance</div>" +
          "<table><thead><tr>" +
          "<th>Date</th><th>Entry Time</th><th>Duration</th><th>Direction</th><th>Stop</th><th>Target</th><th>F #</th><th>F Type</th><th>ATR</th><th>ADX</th><th>VD</th><th>HD</th><th>PB %</th><th>P&amp;L</th>" +
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
            "<div class='section-title'>Fractals</div>" +
            "<div style='padding:10px;color:var(--text-dim,#888);font-size:13px;'>" +
              "No fractal pivot points detected in this date range." +
            "</div>" +
          "</div>";
        return;
      }

      /* Build bar→trade outcome lookup from intraday data */
      var intradayData = m.intraday || [];
      var barToOutcome = {};  /* bar index → "win" | "loss" */
      intradayData.forEach(function (t) {
        if (t.fractal_bar !== null && t.fractal_bar !== undefined) {
          barToOutcome[t.fractal_bar] = t.pnl >= 0 ? "win" : "loss";
        }
      });

      /* Type 1 = N=18 cycle structure (carried forward).
         Cycle  = market cycle state derived from last 10 N=18 fractals (carried forward).
         Type 2 = N=2 entry texture + N=6 dot overlay. */
      var carriedN18 = "";  /* last-seen N=18 label, carried forward */
      var carriedCycle = "";  /* last-seen N18 cycle label, carried forward */
      var carriedN6Cycle = "";  /* last-seen N6 cycle label, carried forward */
      var _lhCounter = 0;            /* consecutive LH count (lower than prev high) */
      var _prevHighPrice = null;    /* price of previous high-type fractal (CH/HH/LH) */
      pivotList.forEach(function (pv, idx) {
        var lbl  = pv.label || "";
        var bgClass = "";
        if (lbl === "CH" || lbl === "CL") {
          bgClass = " class='fractal-row-consolidation'";
        } else if (lbl === "LH" || lbl === "LL" || lbl === "HH" || lbl === "HL") {
          bgClass = " class='fractal-row-directional'";
        }

        /* ── Type 1: N=18 cycle structure (carried forward) ── */
        var type1Html = "";
        if (pv.n18 && pv.n18_label) {
          carriedN18 = pv.n18_label;
          type1Html = "<strong>" + esc(pv.n18_label) + "</strong>" +
            " <span style='color:#ffd700;font-size:30px;line-height:1;vertical-align:-0.15em;' title='N=18 fractal'>\u2022</span>";
        } else if (carriedN18) {
          type1Html = "<strong>" + esc(carriedN18) + "</strong>";
        }

        /* ── Cycle: market cycle state (carried forward from N=18 pivots) ── */
        var cycleHtml = "";
        if (pv.cycle_label) {
          carriedCycle = pv.cycle_label;
        }
        if (carriedCycle) {
          if (carriedCycle === "" || carriedCycle === "\u2014") {
            cycleHtml = "";
          } else {
            var cycleCls = carriedCycle.indexOf("\u2191") >= 0 ? "pos"
                         : carriedCycle.indexOf("\u2193") >= 0 ? "neg"
                         : "neu";
            cycleHtml = "<span class='" + cycleCls + "'>" + esc(carriedCycle) + "</span>";
          }
        }

        /* ── N6 Cycle: market cycle state (carried forward from N=6 pivots) ── */
        var n6CycleHtml = "";
        if (pv.n6_cycle_label) {
          carriedN6Cycle = pv.n6_cycle_label;
        }
        if (carriedN6Cycle) {
          if (carriedN6Cycle === "" || carriedN6Cycle === "\u2014") {
            n6CycleHtml = "";
          } else {
            var n6CycleCls = carriedN6Cycle.indexOf("\u2191") >= 0 ? "pos"
                           : carriedN6Cycle.indexOf("\u2193") >= 0 ? "neg"
                           : "neu";
            n6CycleHtml = "<span class='" + n6CycleCls + "'>" + esc(carriedN6Cycle) + "</span>";
          }
        }

        /* ── Type 2: N=2 entry texture + optional N=6 dot ── */
        var type2Html = "";
        if (pv.label) {
          /* This row has an N=2 fractal (every row in pivotList is an N=2 pivot) */
          type2Html = "<strong>" + esc(pv.label) + "</strong>";
          if (pv.n6) type2Html += " <span style='color:#ffffff;font-size:30px;line-height:1;vertical-align:-0.15em;' title='N=6 fractal'>\u2022</span>";
        }

        var vertHigh = "";
        var vertLow  = "";
        if (pv.vert_dist !== null && pv.vert_dist !== undefined) {
          var _vArrow = pv.vert_dir === "up" ? "<span class='pos'>\u25B2</span> " : "<span class='neg'>\u25BC</span> ";
          if (pv.kind === "H") {
            vertHigh = _vArrow + fmt(pv.vert_dist, 1);
          } else {
            vertLow  = _vArrow + fmt(pv.vert_dist, 1);
          }
        }
        var horizD   = (pv.horiz_dist   !== null && pv.horiz_dist   !== undefined) ? pv.horiz_dist : "";
        var pullbackD = (pv.pullback_pct !== null && pv.pullback_pct !== undefined) ? fmt(pv.pullback_pct, 1) + "%" : "";
        var atrD     = (pv.atr          !== null && pv.atr          !== undefined) ? fmt(pv.atr, 1) : "";
        var adxD     = (pv.adx          !== null && pv.adx          !== undefined) ? fmt(pv.adx, 1) : "";
        var numHtml = String(idx + 1);
        var outcome = barToOutcome[pv.bar];
        if (outcome === "win") {
          numHtml += " <span class='fractal-outcome-win'>W</span>";
        } else if (outcome === "loss") {
          numHtml += " <span class='fractal-outcome-loss'>L</span>";
        }

        var widthHtml = (pv.width_score !== null && pv.width_score !== undefined) ? String(pv.width_score) : "";

        /* ── L#: consecutive lower-high count ── */
        var lhNumHtml = "";
        var _isHigh = (pv.label === "LH" || pv.label === "HH" || pv.label === "CH");
        if (_isHigh) {
          if (pv.label === "LH" && _prevHighPrice !== null && pv.price < _prevHighPrice) {
            _lhCounter++;
            lhNumHtml = String(_lhCounter);
          } else if (_prevHighPrice !== null && pv.price > _prevHighPrice) {
            _lhCounter = 0;
          }
          _prevHighPrice = pv.price;
        }

        pvRows +=
          "<tr" + bgClass + ">" +
          "<td class='nowrap'>" + numHtml + "</td>" +
          // "<td>" + type1Html + "</td>" +
          // "<td>" + cycleHtml + "</td>" +
          "<td>" + type2Html + "</td>" +
          "<td>" + lhNumHtml + "</td>" +
          "<td>" + pullbackD + "</td>" +
          // "<td>" + n6CycleHtml + "</td>" +
          "<td>" + widthHtml + "</td>" +
          "<td class='nowrap'>" + fmt(pv.price, 5) + "</td>" +
          "<td class='nowrap'>" + esc(pv.time || "") + "</td>" +
          "<td>" + atrD + "</td>" +
          "<td>" + adxD + "</td>" +
          "<td>" + vertHigh + "</td>" +
          "<td>" + vertLow + "</td>" +
          "<td>" + horizD + "</td>" +
          "</tr>";
      });

      var structure  = pvd.structure || "Consolidating";
      var structCls  = structure === "Trending Up"   ? "pos"
                     : structure === "Trending Down" ? "neg"
                     : "neu";

      pivotDiagHtml =
        "<div class='section' id='anchor-fractal-diag'>" +
          "<div class='section-title'>Fractals</div>" +
          "<table><thead><tr>" +
          "<th style='width:52px'>#</th>" +
          // "<th>Type 1</th>" +
          // "<th>Cycle 1</th>" +
          "<th>Type 2</th>" +
          "<th>L#</th>" +
          "<th>Pullback %</th>" +
          // "<th>Cycle 2</th>" +
          "<th>Width</th>" +
          "<th>Price</th>" +
          "<th>Time</th>" +
          "<th>ATR (pips)</th>" +
          "<th>ADX</th>" +
          "<th>VD High</th>" +
          "<th>VD Low</th>" +
          "<th>Horiz Distance (bars)</th>" +
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

    /* Build header: v1 · Jun-03-25 → Jun-03-25 · 1 day */
    var hdrRange = run.start_date && run.end_date
      ? { start: run.start_date, end: run.end_date }
      : fullRunRange(run);
    var hdrDateStr = hdrRange.start && hdrRange.end
      ? fmtSbDate(hdrRange.start) + " \u2192 " + fmtSbDate(hdrRange.end) : "";
    var hdrDur = calcDuration(hdrRange.start, hdrRange.end);
    var hdrParts = [esc(v.strategy_version || v.name)];
    if (hdrDateStr) hdrParts.push(hdrDateStr);
    if (hdrDur) hdrParts.push(hdrDur);
    var headerTitle = hdrParts.join(" \u00b7 ");



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

    /* ── Backtest Settings panel ──────────────────────────────────────────────── */
    var ecData = v.entry_conditions || null;
    var ecThStyle = "class='bs-th'";
    var entryCondHtml;
    var dirOptions = [
      { value: "short_only", label: "Short only" },
      { value: "long_only",  label: "Long only" },
      { value: "both",       label: "Both" }
    ];
    var savedDir = run.trade_direction || p.trade_direction || "short_only";
    var dirSelectHtml = "<select id='bs-direction-select' class='bs-select'>" +
      dirOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === savedDir ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var intervalOptions = [
      { value: "1m", label: "1m" },
      { value: "5m", label: "5m" },
      { value: "15m", label: "15m" },
      { value: "30m", label: "30m" },
      { value: "60m", label: "60m" }
    ];
    var savedInterval = run.interval || p.interval || "5m";
    var intervalSelectHtml = "<select id='bs-interval-select' class='bs-select'>" +
      intervalOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === savedInterval ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var savedEmaShort = run.ema_short != null ? run.ema_short : (p.ema_short != null ? p.ema_short : 8);
    var savedEmaMid   = run.ema_mid   != null ? run.ema_mid   : (p.ema_mid   != null ? p.ema_mid   : 20);
    var savedEmaLong  = run.ema_long  != null ? run.ema_long  : (p.ema_long  != null ? p.ema_long  : 40);
    var emaShortHtml = "<input id='bs-ema-short' type='number' class='bs-input' value='" + savedEmaShort + "' min='0' step='1'>";
    var emaMidHtml   = "<input id='bs-ema-mid'   type='number' class='bs-input' value='" + savedEmaMid   + "' min='0' step='1'>";
    var emaLongHtml  = "<input id='bs-ema-long'  type='number' class='bs-input' value='" + savedEmaLong  + "' min='0' step='1'>";

    var savedStopPips  = run.stop_loss_pips || p.stop_loss_pips || 15;
    var stopPipsHtml   = "<input id='bs-stop-pips' type='number' class='bs-input' value='" + savedStopPips + "' min='1' step='1'>";

    var savedRrrRisk   = run.rrr_risk   || p.rrr_risk   || 1;
    var savedRrrReward = run.rrr_reward || p.rrr_reward || 2;
    var rrrOpts = [1, 2, 3, 4, 5];
    var rrrRiskHtml = "<select id='bs-rrr-risk' class='bs-select bs-select-narrow'>" +
      rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (n === savedRrrRisk ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var rrrRewardHtml = "<select id='bs-rrr-reward' class='bs-select bs-select-narrow'>" +
      rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (n === savedRrrReward ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var rrrSelectHtml = rrrRiskHtml + "<span class='bs-rrr-colon'>:</span>" + rrrRewardHtml;

    if (ecData && ecData.length > 0) {
      var ecRows = ecData.filter(function(ec) {
        return ec.condition !== "Instrument";
      }).map(function(ec) {
        var ruleCell = ec.condition === "Direction"
          ? dirSelectHtml
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
          "<td class='bs-td-cond'>" + esc(ec.condition) + "</td>" +
          "<td class='bs-td-rule'>" + ruleCell + "</td>" +
          "</tr>";
      }).join("");
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Backtest Settings</div>" +
          "<table>" +
            "<tbody>" + ecRows + "</tbody>" +
          "</table>" +
        "</div>";
    } else {
      entryCondHtml =
        "<div class='section'>" +
          "<div class='section-title'>Backtest Settings</div>" +
          "<table>" +
            "<tbody>" +
            "<tr><td class='bs-td-cond'>Interval</td><td class='bs-td-rule'>" + intervalSelectHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>EMA Short</td><td class='bs-td-rule'>" + emaShortHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>EMA Mid</td><td class='bs-td-rule'>" + emaMidHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>EMA Long</td><td class='bs-td-rule'>" + emaLongHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>Stop Loss Level</td><td class='bs-td-rule'>" + stopPipsHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>Direction</td><td class='bs-td-rule'>" + dirSelectHtml + "</td></tr>" +
            "<tr><td class='bs-td-cond'>RRR</td><td class='bs-td-rule'>" + rrrSelectHtml + "</td></tr>" +
            "</tbody>" +
          "</table>" +
        "</div>";
    }

    document.getElementById("content").innerHTML =
      /* header */
      "<div id='v-header'>" +
        "<div id='v-header-top'>" +
          "<h2>" + headerTitle + "</h2>" +
          "<div class='v-header-actions'>" +
            "<span class='report-tabs'>" +
              "<button class='report-tab active' data-tab='general'>General</button>" +
              "<button class='report-tab' data-tab='advanced'>Advanced</button>" +
            "</span>" +
            "<button class='bs-toggle-btn' id='bs-toggle-btn' title='Backtest Settings'>" +
              "<svg width='16' height='16' viewBox='0 0 16 16' fill='none'>" +
                "<circle cx='8' cy='8' r='7' stroke='currentColor' stroke-width='1.5'/>" +
                "<path d='M5.5 7L8 9.5L10.5 7' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>" +
              "</svg>" +
            "</button>" +
          "</div>" +
        "</div>" +
        "<div class='bs-collapsible' id='bs-collapsible'>" + entryCondHtml + "</div>" +
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
          row("Gross Profit",  "<span class='pos'>+$" + commaFmt(m.gross_profit) + "</span>") +
          row("Gross Loss",    "<span class='neg'>-$" + commaFmt(Math.abs(m.gross_loss)) + "</span>") +
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
          row("Instrument",     "<span class='val-highlight'>" + esc((run.instrument || p.ticker || "EURUSD").replace(/=X$/i, "")) + "</span>") +
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
      var toggleBtn = document.getElementById("bs-toggle-btn");
      var panel = document.getElementById("bs-collapsible");
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
      var mainEl = document.getElementById("main");
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
      var dirEl = document.getElementById("bs-direction-select");
      if (!dirEl) return;
      var stored = localStorage.getItem("bs_direction");
      if (stored) dirEl.value = stored;
      dirEl.addEventListener("change", function () {
        localStorage.setItem("bs_direction", dirEl.value);
      });
    }());

    /* Wire interval select — persist to localStorage on change */
    (function () {
      var intEl = document.getElementById("bs-interval-select");
      if (!intEl) return;
      var stored = localStorage.getItem("bs_interval");
      if (stored) intEl.value = stored;
      intEl.addEventListener("change", function () {
        localStorage.setItem("bs_interval", intEl.value);
      });
    }());

    /* Wire EMA inputs — persist to localStorage on change */
    (function () {
      var ids = [
        { id: "bs-ema-short",  key: "bs_ema_short" },
        { id: "bs-ema-mid",    key: "bs_ema_mid" },
        { id: "bs-ema-long",   key: "bs_ema_long" },
        { id: "bs-stop-pips",  key: "bs_stop_pips" }
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
      var riskEl   = document.getElementById("bs-rrr-risk");
      var rewardEl = document.getElementById("bs-rrr-reward");
      if (riskEl) {
        var storedRisk = localStorage.getItem("bs_rrr_risk");
        if (storedRisk) riskEl.value = storedRisk;
        riskEl.addEventListener("change", function () {
          localStorage.setItem("bs_rrr_risk", riskEl.value);
        });
      }
      if (rewardEl) {
        var storedReward = localStorage.getItem("bs_rrr_reward");
        if (storedReward) rewardEl.value = storedReward;
        rewardEl.addEventListener("change", function () {
          localStorage.setItem("bs_rrr_reward", rewardEl.value);
        });
      }
    }());

    /* Wire copy button (lives in the run bar) */
    (function (ver, runData) {
      var btn = document.getElementById("copy-btn");
      var sep = document.getElementById("rb-act-sep");
      if (!btn) return;
      btn.style.display = "";
      btn.disabled = false;
      btn.classList.remove("copied");
      if (sep) sep.style.display = "";
      btn.textContent = "Copy Report";
      btn.onclick = function () {
        var md = buildRunMarkdown(ver, runData);
        navigator.clipboard.writeText(md).then(function () {
          btn.textContent = "\u2713 Copied!";
          btn.classList.add("copied");
          showToast("Report copied");
          setTimeout(function () {
            btn.textContent = "Copy Report";
            btn.classList.remove("copied");
          }, 2200);
        }).catch(function () {
          btn.textContent = "Failed";
          setTimeout(function () { btn.textContent = "Copy Report"; }, 2500);
        });
      };
    }(v, run));
  }

  /* ── Hide run-bar action buttons ──────────────────────────── */
  function hideActionButtons() {
    var btn = document.getElementById("copy-btn");
    var sep = document.getElementById("rb-act-sep");
    if (btn) { btn.style.display = "none"; btn.onclick = null; }
    if (sep) { sep.style.display = "none"; }
  }

  /* ── Dev Log ──────────────────────────────────────────────── */
  var _devlogData = null;  /* cached devlog array, loaded from /devlog */

  function _devlogSave(cb) {
    fetch("/devlog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(_devlogData)
    }).then(function (r) { return r.json(); })
      .then(function () { if (cb) cb(); })
      .catch(function () {
        if (cb) cb();
        var msg = document.createElement("div");
        msg.textContent = "Failed to save \\u2014 is the server running?";
        msg.style.cssText = "position:fixed;top:60px;left:50%;transform:translateX(-50%);background:#c0392b;color:#fff;padding:8px 18px;border-radius:6px;z-index:9999;font-size:13px;";
        document.body.appendChild(msg);
        setTimeout(function(){ msg.remove(); }, 3000);
      });
  }

  function _devlogRender() {
    var el = document.getElementById("content");
    if (!_devlogData) { _devlogData = []; }

    var versionsHtml = "";
    _devlogData.forEach(function (ver, vi) {
      var paramRows = "";
      (ver.params || []).forEach(function (p, pi) {
        paramRows +=
          "<tr class='dl-param-row' draggable='true' data-vi='" + vi + "' data-pi='" + pi + "'>" +
            "<td class='dl-drag-handle'><span class='material-symbols-outlined dl-drag-icon'>drag_indicator</span></td>" +
            "<td>" + esc(p.name) + "</td>" +
            "<td>" + esc(p.desc) + "</td>" +
            "<td class='dl-param-actions'>" +
              "<span class='dl-act' data-vi='" + vi + "' data-pi='" + pi + "' data-action='edit'>Edit</span>" +
              "<span class='dl-act dl-act-del' data-vi='" + vi + "' data-pi='" + pi + "' data-action='delete'>Delete</span>" +
            "</td>" +
          "</tr>";
      });

      versionsHtml +=
        "<div class='dl-version-block'>" +
          "<div class='dl-version-header'>" +
            "<h3>" + esc(ver.strategy_version || ver.name) + "</h3>" +
            "<div class='dl-add-param-row'>" +
              "<input type='text' class='dl-input dl-input-param' placeholder='Parameter' data-vi='" + vi + "' data-field='name'>" +
              "<input type='text' class='dl-input dl-input-desc' placeholder='Description' data-vi='" + vi + "' data-field='desc'>" +
              "<button class='dl-btn dl-btn-green dl-add-param-btn' data-vi='" + vi + "' title='Add Parameter'>+</button>" +
            "</div>" +
          "</div>" +
          "<table class='dl-param-table'>" +
            "<thead><tr><th class='dl-th-drag'></th><th>Parameter</th><th>Description</th><th></th></tr></thead>" +
            "<tbody>" + (paramRows || "<tr><td colspan='4' class='dl-empty'>No parameters yet <span class='dl-act dl-act-del dl-delete-version' data-vi='" + vi + "'>Delete Version</span></td></tr>") + "</tbody>" +
          "</table>" +
        "</div>";
    });

    el.innerHTML =
      "<div id='devlog-header'>" +
        "<h2>Development Log</h2>" +
        "<div class='dl-add-version-row'>" +
          "<input type='text' id='dl-new-version' class='dl-input' placeholder='V#'>" +
          "<button id='dl-add-version-btn' class='dl-btn dl-btn-green' title='Add Version'>+</button>" +
        "</div>" +
      "</div>" +
      versionsHtml;

    /* ── Wire Add Version ── */
    var addVerBtn = document.getElementById("dl-add-version-btn");
    var addVerInput = document.getElementById("dl-new-version");
    if (addVerBtn) {
      addVerBtn.addEventListener("click", function () {
        var name = addVerInput.value.trim();
        if (!name) return;
        _devlogData.unshift({ name: name, params: [] });
        _devlogSave(function () { _devlogRender(); });
      });
      addVerInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") addVerBtn.click();
      });
    }

    /* ── Wire Add Parameter buttons ── */
    document.querySelectorAll(".dl-add-param-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var vi = parseInt(btn.dataset.vi, 10);
        var block = btn.closest(".dl-version-header");
        var nameInput = block.querySelector(".dl-input-param");
        var descInput = block.querySelector(".dl-input-desc");
        var pName = nameInput.value.trim();
        var pDesc = descInput.value.trim();
        if (!pName) return;
        _devlogData[vi].params.push({ name: pName, desc: pDesc });
        _devlogSave(function () { _devlogRender(); });
      });
    });
    document.querySelectorAll(".dl-input-desc").forEach(function (inp) {
      inp.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          var vi = inp.dataset.vi;
          var btn = document.querySelector(".dl-add-param-btn[data-vi='" + vi + "']");
          if (btn) btn.click();
        }
      });
    });

    /* ── Wire Edit / Delete actions ── */
    document.querySelectorAll(".dl-act").forEach(function (span) {
      span.addEventListener("click", function () {
        var vi = parseInt(span.dataset.vi, 10);
        var pi = parseInt(span.dataset.pi, 10);
        var action = span.dataset.action;

        if (action === "delete") {
          _devlogData[vi].params.splice(pi, 1);
          _devlogSave(function () { _devlogRender(); });

        } else if (action === "edit") {
          var p = _devlogData[vi].params[pi];
          var row = span.closest("tr");
          row.draggable = false;
          row.innerHTML =
            "<td class='dl-drag-handle'></td>" +
            "<td><input type='text' class='dl-input dl-edit-name' value='" + esc(p.name).replace(/'/g, "&#39;") + "'></td>" +
            "<td><input type='text' class='dl-input dl-edit-desc' value='" + esc(p.desc).replace(/'/g, "&#39;") + "'></td>" +
            "<td class='dl-param-actions'>" +
              "<span class='dl-act dl-save-edit' data-vi='" + vi + "' data-pi='" + pi + "'>Save</span>" +
              "<span class='dl-act dl-cancel-edit'>Cancel</span>" +
            "</td>";

          var saveBtn = row.querySelector(".dl-save-edit");
          var cancelBtn = row.querySelector(".dl-cancel-edit");
          var editName = row.querySelector(".dl-edit-name");
          var editDesc = row.querySelector(".dl-edit-desc");

          function doSave() {
            p.name = editName.value.trim() || p.name;
            p.desc = editDesc.value.trim();
            _devlogSave(function () { _devlogRender(); });
          }
          saveBtn.addEventListener("click", doSave);
          editDesc.addEventListener("keydown", function (e) { if (e.key === "Enter") doSave(); });
          editName.addEventListener("keydown", function (e) { if (e.key === "Enter") doSave(); });
          cancelBtn.addEventListener("click", function () { _devlogRender(); });
          editName.focus();
        }
      });
    });

    /* ── Wire Delete Version buttons ── */
    document.querySelectorAll(".dl-delete-version").forEach(function (span) {
      span.addEventListener("click", function () {
        var vi = parseInt(span.dataset.vi, 10);
        _devlogData.splice(vi, 1);
        _devlogSave(function () { _devlogRender(); });
      });
    });

    /* ── Wire drag-to-sort on parameter rows ── */
    var _dragRow = null;
    document.querySelectorAll(".dl-param-row[draggable='true']").forEach(function (row) {
      row.addEventListener("dragstart", function (e) {
        _dragRow = row;
        row.classList.add("dl-dragging");
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", "");
      });
      row.addEventListener("dragend", function () {
        _dragRow = null;
        row.classList.remove("dl-dragging");
        document.querySelectorAll(".dl-drag-over").forEach(function (r) { r.classList.remove("dl-drag-over"); });
      });
      row.addEventListener("dragover", function (e) {
        if (!_dragRow || _dragRow === row) return;
        if (_dragRow.dataset.vi !== row.dataset.vi) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
        row.classList.add("dl-drag-over");
      });
      row.addEventListener("dragleave", function () {
        row.classList.remove("dl-drag-over");
      });
      row.addEventListener("drop", function (e) {
        e.preventDefault();
        if (!_dragRow || _dragRow === row) return;
        var vi = parseInt(row.dataset.vi, 10);
        var fromPi = parseInt(_dragRow.dataset.pi, 10);
        var toPi = parseInt(row.dataset.pi, 10);
        if (_dragRow.dataset.vi !== row.dataset.vi) return;
        var params = _devlogData[vi].params;
        var moved = params.splice(fromPi, 1)[0];
        params.splice(toPi, 0, moved);
        _devlogSave(function () { _devlogRender(); });
      });
    });
  }

  function showDevLog() {
    hideActionButtons();

    if (_devlogData !== null) {
      _devlogRender();
      return;
    }

    document.getElementById("content").innerHTML =
      "<div id='devlog-header'><h2>Development Log</h2>" +
      "<div class='v-meta'>Loading\u2026</div></div>";

    fetch("/devlog").then(function (r) { return r.json(); })
      .then(function (data) {
        _devlogData = Array.isArray(data) ? data : [];
        _devlogRender();
      })
      .catch(function () {
        _devlogData = [];
        _devlogRender();
      });
  }

  /* ── Empty state — show Backtest Settings selects when no versions exist ── */
  function renderEmptyState() {
    hideActionButtons();
    var ecThStyle = "class='bs-th'";

    var _dirOptions = [
      { value: "short_only", label: "Short only" },
      { value: "long_only",  label: "Long only" },
      { value: "both",       label: "Both" }
    ];
    var _savedDir = localStorage.getItem("bs_direction") || "short_only";
    var _dirSelectHtml = "<select id='bs-direction-select' class='bs-select'>" +
      _dirOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === _savedDir ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var _intervalOptions = [
      { value: "1m", label: "1m" },
      { value: "5m", label: "5m" },
      { value: "15m", label: "15m" },
      { value: "30m", label: "30m" },
      { value: "60m", label: "60m" }
    ];
    var _savedInterval = localStorage.getItem("bs_interval") || "5m";
    var _intervalSelectHtml = "<select id='bs-interval-select' class='bs-select'>" +
      _intervalOptions.map(function(o) {
        return "<option value='" + o.value + "'" + (o.value === _savedInterval ? " selected" : "") + ">" + o.label + "</option>";
      }).join("") + "</select>";

    var _savedEmaShort = localStorage.getItem("bs_ema_short") || "8";
    var _savedEmaMid   = localStorage.getItem("bs_ema_mid")   || "20";
    var _savedEmaLong  = localStorage.getItem("bs_ema_long")  || "40";
    var _emaShortHtml = "<input id='bs-ema-short' type='number' class='bs-input' value='" + _savedEmaShort + "' min='0' step='1'>";
    var _emaMidHtml   = "<input id='bs-ema-mid'   type='number' class='bs-input' value='" + _savedEmaMid   + "' min='0' step='1'>";
    var _emaLongHtml  = "<input id='bs-ema-long'  type='number' class='bs-input' value='" + _savedEmaLong  + "' min='0' step='1'>";

    var _savedStopPips  = localStorage.getItem("bs_stop_pips") || "15";
    var _stopPipsHtml   = "<input id='bs-stop-pips' type='number' class='bs-input' value='" + _savedStopPips + "' min='1' step='1'>";

    var _savedRrrRisk   = localStorage.getItem("bs_rrr_risk")   || "1";
    var _savedRrrReward = localStorage.getItem("bs_rrr_reward") || "2";
    var _rrrOpts = [1, 2, 3, 4, 5];
    var _rrrRiskHtml = "<select id='bs-rrr-risk' class='bs-select bs-select-narrow'>" +
      _rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (String(n) === _savedRrrRisk ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var _rrrRewardHtml = "<select id='bs-rrr-reward' class='bs-select bs-select-narrow'>" +
      _rrrOpts.map(function(n) {
        return "<option value='" + n + "'" + (String(n) === _savedRrrReward ? " selected" : "") + ">" + n + "</option>";
      }).join("") + "</select>";
    var _rrrSelectHtml = _rrrRiskHtml + "<span class='bs-rrr-colon'>:</span>" + _rrrRewardHtml;

    document.getElementById("content").innerHTML =
      "<div class='section'>" +
        "<div class='section-title'>Backtest Settings</div>" +
        "<table>" +
          "<tbody>" +
          "<tr><td class='bs-td-cond'>Interval</td><td class='bs-td-rule'>" + _intervalSelectHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>EMA Short</td><td class='bs-td-rule'>" + _emaShortHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>EMA Mid</td><td class='bs-td-rule'>" + _emaMidHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>EMA Long</td><td class='bs-td-rule'>" + _emaLongHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>Stop Loss Level</td><td class='bs-td-rule'>" + _stopPipsHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>Direction</td><td class='bs-td-rule'>" + _dirSelectHtml + "</td></tr>" +
          "<tr><td class='bs-td-cond'>RRR</td><td class='bs-td-rule'>" + _rrrSelectHtml + "</td></tr>" +
          "</tbody>" +
        "</table>" +
      "</div>";

    /* Wire localStorage persistence for the empty-state selects */
    var _dirEl = document.getElementById("bs-direction-select");
    if (_dirEl) _dirEl.addEventListener("change", function () { localStorage.setItem("bs_direction", _dirEl.value); });
    var _intEl = document.getElementById("bs-interval-select");
    if (_intEl) _intEl.addEventListener("change", function () { localStorage.setItem("bs_interval", _intEl.value); });
    var _rrrRiskEl = document.getElementById("bs-rrr-risk");
    if (_rrrRiskEl) _rrrRiskEl.addEventListener("change", function () { localStorage.setItem("bs_rrr_risk", _rrrRiskEl.value); });
    var _rrrRewardEl = document.getElementById("bs-rrr-reward");
    if (_rrrRewardEl) _rrrRewardEl.addEventListener("change", function () { localStorage.setItem("bs_rrr_reward", _rrrRewardEl.value); });
  }

  /* ── Version selector: populate and wire ─────────────────── */
  /* The selector shows strategy versions (v1, v2) — always both present.
     currentVersion holds the selected strategy version tag. */
  function populateVersionSelector() {
    var sel = document.getElementById("version-select");
    var stratVersions = ["v1", "v2", "v3"];
    sel.innerHTML = "";
    for (var j = 0; j < stratVersions.length; j++) {
      var opt = document.createElement("option");
      opt.value = stratVersions[j];
      opt.textContent = stratVersions[j];
      sel.appendChild(opt);
    }
    /* Restore last-used strategy version from localStorage, default to v1 */
    var stored = localStorage.getItem("rb_strategy_version");
    if (stored && stratVersions.indexOf(stored) >= 0) {
      currentVersion = stored;
    } else {
      currentVersion = "v1";
    }
    sel.value = currentVersion;
  }

  function onVersionChange() {
    currentVersion = document.getElementById("version-select").value;
    localStorage.setItem("rb_strategy_version", currentVersion);
    devLogOpen = false;
    document.getElementById("devlog-btn").classList.remove("active");
    renderSidebar();

    /* Jump to the last run that matches the current instrument */
    var svs = getStrategyVersions();
    var found = false;
    for (var si = svs.length - 1; si >= 0; si--) {
      var runs = getRuns(svs[si].v);
      for (var ri = runs.length - 1; ri >= 0; ri--) {
        var rinst = (runs[ri].instrument || (svs[si].v.params && svs[si].v.params.ticker ? svs[si].v.params.ticker.replace(/=X$/i, "") : "")).toUpperCase();
        if (!currentInstrument || rinst === currentInstrument) {
          activeVersionIdx = svs[si].idx;
          activeRunIdx = ri;
          renderSidebar();
          renderContent(svs[si].idx, ri);
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found) {
      renderEmptyState();
    }
  }

  /* ── Instrument selector: populate and wire ───────────────── */
  function populateInstrumentSelector() {
    var sel = document.getElementById("instrument-select");
    var stored = localStorage.getItem("rb_instrument");
    if (stored) {
      currentInstrument = stored;
    } else {
      /* Default to the first instrument that actually has runs */
      var firstWithData = "";
      for (var oi = 0; oi < sel.options.length; oi++) {
        var optVal = sel.options[oi].value;
        for (var vi = 0; vi < VERSIONS.length; vi++) {
          var runs = getRuns(VERSIONS[vi]);
          for (var ri = 0; ri < runs.length; ri++) {
            var rinst = (runs[ri].instrument || (VERSIONS[vi].params && VERSIONS[vi].params.ticker ? VERSIONS[vi].params.ticker.replace(/=X$/i, "") : "")).toUpperCase();
            if (rinst === optVal) { firstWithData = optVal; break; }
          }
          if (firstWithData) break;
        }
        if (firstWithData) break;
      }
      currentInstrument = firstWithData || (sel.options.length > 0 ? sel.options[0].value : "EURUSD");
    }
    sel.value = currentInstrument;
  }

  function onInstrumentChange() {
    currentInstrument = document.getElementById("instrument-select").value;
    localStorage.setItem("rb_instrument", currentInstrument);
    devLogOpen = false;
    document.getElementById("devlog-btn").classList.remove("active");
    renderSidebar();

    var svs = getStrategyVersions();
    /* Find the last run matching the instrument within current strategy versions */
    var found = false;
    for (var si = svs.length - 1; si >= 0; si--) {
      var runs = getRuns(svs[si].v);
      for (var ri = runs.length - 1; ri >= 0; ri--) {
        var rinst = (runs[ri].instrument || (svs[si].v.params && svs[si].v.params.ticker ? svs[si].v.params.ticker.replace(/=X$/i, "") : "")).toUpperCase();
        if (rinst === currentInstrument) {
          activeVersionIdx = svs[si].idx;
          activeRunIdx = ri;
          renderSidebar();
          renderContent(svs[si].idx, ri);
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found) {
      activeVersionIdx = -1;
      activeRunIdx = 0;
      renderSidebar();
      renderEmptyState();
    }
  }

  /* ── Init ──────────────────────────────────────────────────── */
  populateVersionSelector();
  populateInstrumentSelector();
  document.getElementById("version-select").addEventListener("change", onVersionChange);
  document.getElementById("instrument-select").addEventListener("change", onInstrumentChange);

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
    var lastRuns = getRuns(svs[svs.length - 1].v);
    activeVersionIdx = lastIdx;
    activeRunIdx = 0;

    /* If we just came back from a date-range run, jump to the newest item */
    var pendingRunType    = localStorage.getItem("rb_pending_run_type");
    var pendingRunVersion = localStorage.getItem("rb_pending_run_version");
    if (pendingRunType === "date_range") {
      localStorage.removeItem("rb_pending_run_type");
      localStorage.removeItem("rb_pending_run_version");
      var targetIdx = lastIdx;
      if (pendingRunVersion) {
        for (var ti = 0; ti < VERSIONS.length; ti++) {
          if (VERSIONS[ti].name === pendingRunVersion
              && (VERSIONS[ti].strategy_version || "v1") === currentVersion) {
            targetIdx = ti; break;
          }
        }
      }
      activeVersionIdx = targetIdx;
      var targetRuns = getRuns(VERSIONS[targetIdx]);
      activeRunIdx = targetRuns.length - 1;
    }

    /* If we just came back from a new version with auto date ranges */
    if (pendingRunType === "new_version_auto") {
      localStorage.removeItem("rb_pending_run_type");
      activeVersionIdx = lastIdx;
      activeRunIdx = 0;
    }

    /* If we just deleted a run, focus on the item above it */
    var pendingDelVersion = localStorage.getItem("rb_pending_delete_version");
    var pendingDelRunIdx  = localStorage.getItem("rb_pending_delete_run_idx");
    if (pendingDelVersion) {
      localStorage.removeItem("rb_pending_delete_version");
      localStorage.removeItem("rb_pending_delete_run_idx");
      var delTargetIdx = lastIdx;
      for (var di = 0; di < VERSIONS.length; di++) {
        if (VERSIONS[di].name === pendingDelVersion
            && (VERSIONS[di].strategy_version || "v1") === currentVersion) {
          delTargetIdx = di; break;
        }
      }
      activeVersionIdx = delTargetIdx;
      var delRunIdx = parseInt(pendingDelRunIdx, 10) || 0;
      var remainingRuns = getRuns(VERSIONS[delTargetIdx]);
      if (delRunIdx > 0 && delRunIdx < remainingRuns.length) {
        activeRunIdx = delRunIdx;
      } else {
        activeRunIdx = 0;
      }
    }

    renderSidebar();
    renderContent(activeVersionIdx, activeRunIdx);

    /* Scroll sidebar to bottom after date-range add */
    var vList = document.getElementById("version-list");
    if (vList) vList.scrollTop = vList.scrollHeight;
  } else {
    renderEmptyState();
  }

  /* ── Keyboard shortcuts: Up / Down arrow sidebar navigation ── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    if (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable) return;
    if (devLogOpen) return;

    /* Build flat list of sidebar items in render order (oldest first) */
    var svs = getStrategyVersions();
    if (svs.length === 0) return;
    var items = []; /* { vIdx, runIdx } */
    for (var ri = 0; ri < svs.length; ri++) {
      var entry = svs[ri];
      var vIdx  = entry.idx;
      var runs  = getRuns(entry.v);
      for (var si = 0; si < runs.length; si++) {
        items.push({ vIdx: vIdx, runIdx: si });
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

  /* ── Keyboard shortcut: V or Shift+V — Add Year ───── */
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

  /* ── Keyboard shortcut: B — Toggle Backtest Settings ───────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "b" && e.key !== "B") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var btn = document.getElementById("bs-toggle-btn");
    if (!btn) return;
    e.preventDefault();
    btn.click();
  });

  /* ── Keyboard shortcut: G — General tab ───────────────────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "g" && e.key !== "G") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var tab = document.querySelector(".report-tab[data-tab='general']");
    if (!tab) return;
    e.preventDefault();
    tab.click();
  });

  /* ── Keyboard shortcut: A — Advanced tab ──────────────────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "a" && e.key !== "A") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var tab = document.querySelector(".report-tab[data-tab='advanced']");
    if (!tab) return;
    e.preventDefault();
    tab.click();
  });

  /* ── Keyboard shortcut: 0 — Scroll to bottom ─────────────── */
  document.addEventListener("keydown", function (e) {
    if (e.key !== "0") return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isInputFocused(e)) return;
    var main = document.getElementById("main");
    if (!main) return;
    e.preventDefault();
    main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
  });

  /* ── Keyboard shortcuts: Shift+1‥4 — Toggle sidebar sections ── */
  (function () {
    var _shiftSectionMap = { Digit1: "Year", Digit2: "Month", Digit3: "Weeks", Digit4: "Day" };
    document.addEventListener("keydown", function (e) {
      if (!e.shiftKey) return;
      var sec = _shiftSectionMap[e.code];
      if (!sec) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isInputFocused(e)) return;
      e.preventDefault();
      _sectionState[sec] = !_sectionState[sec];
      _saveSectionState();
      renderSidebar();
    });
  })();
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
        # Filter by strategy_version so v1, v2, v3 strategies don't collide
        target_version_name = os.environ.get("TARGET_VERSION", "").strip()
        target = None
        if target_version_name:
            for v in existing_versions:
                if (v.get("name") == target_version_name
                        and v.get("strategy_version", "v1") == STRATEGY_VERSION_TAG):
                    target = v
                    break
        if target is None:
            # Fallback: most recent version belonging to this strategy
            for v in reversed(existing_versions):
                if v.get("strategy_version", "v1") == STRATEGY_VERSION_TAG:
                    target = v
                    break
        if target is None:
            target = existing_versions[-1]
        target["runs"].append(new_run)
        target["entry_conditions"] = ENTRY_CONDITIONS
        version_num = len(existing_versions)
        action = f"Added date range run to {target.get('name', '?')}"
    else:
        # New version — increment from highest version number across all strategies
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
            "strategy_version": STRATEGY_VERSION_TAG,
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
            trades["entry_idx"]  = trades["entry_idx"] - _pre_buffer
            trades["exit_idx"]   = trades["exit_idx"]  - _pre_buffer
            trades["fractal_bar"] = trades["fractal_bar"] - _pre_buffer
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
