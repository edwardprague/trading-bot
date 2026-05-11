#!/usr/bin/env python3
"""
download_data.py — Local data cache builder
============================================
Downloads the full available history of 5-minute bars from Massive.io
for the supported forex pairs and saves each as a Parquet file under
the project's ./data/ directory. The backtest engine's fetch_data()
will transparently load these files instead of hitting the API when
they're present.

Usage:
    source venv/bin/activate
    python3 download_data.py

Requires:
    - MASSIVE_API_KEY in .env
    - massive, pandas, python-dotenv, pyarrow

Output: data/GBPUSD_5m.parquet, data/EURUSD_5m.parquet
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv


# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"

INSTRUMENTS = [
    ("GBPUSD", "C:GBPUSD"),
    ("EURUSD", "C:EURUSD"),
]

# 5-minute bars. Encoded as (multiplier, timespan, label) where the label is
# used in the cache filename so fetch_data() can find the file by interval.
INTERVAL_MULT  = 5
INTERVAL_SPAN  = "minute"
INTERVAL_LABEL = "5m"

# Massive.io carries Forex history back roughly two decades; we ask for an
# even earlier date so we always get the full available window.
FROM_DATE = "2000-01-01"


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("ERROR: MASSIVE_API_KEY not found in .env", file=sys.stderr)
        print("  Add MASSIVE_API_KEY=<your-key> to the .env file in the project root.",
              file=sys.stderr)
        sys.exit(1)

    try:
        from massive import RESTClient
    except ImportError:
        print("ERROR: 'massive' package not installed.", file=sys.stderr)
        print("  Run: pip install massive", file=sys.stderr)
        sys.exit(1)

    # Probe Parquet support early so we fail fast with a clear message.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("ERROR: 'pyarrow' is required to write Parquet files.", file=sys.stderr)
        print("  Run: pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    client  = RESTClient(api_key=api_key)
    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for sym, ticker in INSTRUMENTS:
        out_path = DATA_DIR / f"{sym}_{INTERVAL_LABEL}.parquet"
        print(f"\nDownloading {ticker} {INTERVAL_LABEL} from {FROM_DATE} → {to_date} …")

        try:
            bars = list(client.list_aggs(
                ticker     = ticker,
                multiplier = INTERVAL_MULT,
                timespan   = INTERVAL_SPAN,
                from_      = FROM_DATE,
                to         = to_date,
                sort       = "asc",
                limit      = 50000,
            ))
        except Exception as e:
            print(f"  ERROR fetching {ticker}: {e}", file=sys.stderr)
            continue

        if not bars:
            print(f"  {sym}: no bars returned — skipping")
            continue

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
        # Same shape/dtype conventions as fetch_data(): drop NaNs, dedupe, sort.
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df[~df.index.duplicated(keep="first")].sort_index()

        try:
            df.to_parquet(out_path)
        except Exception as e:
            print(f"  ERROR writing {out_path}: {e}", file=sys.stderr)
            continue

        first = df.index[0]
        last  = df.index[-1]
        rel   = out_path.relative_to(ROOT_DIR)
        print(f"  {sym}: {len(df):,} bars  |  {first.date()} → {last.date()}  |  saved to {rel}")

    print("\nDone.")


if __name__ == "__main__":
    main()
