"""Decompose ADX vs H1 contribution. Batchable like sweep_live_v2.py."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_live_v2
import pandas as pd

OUT_CSV = Path("/sessions/laughing-elegant-shannon/mnt/outputs/decomposition_2025.csv")

TESTS = [
    ("Baseline (no macro gate)",     0.0,  0.0),
    ("ADX-only @ T_adx=32",          0.0, 32.0),
    ("ADX-only @ T_adx=28",          0.0, 28.0),
    ("ADX-only @ T_adx=24",          0.0, 24.0),
    ("H1-only @ T_h=12",            12.0,  0.0),
    ("H1-only @ T_h=18",            18.0,  0.0),
    ("Combined @ 12/32 (best)",     12.0, 32.0),
    ("Combined @ 18/32",            18.0, 32.0),
]

batch_from = int(os.environ.get("BATCH_FROM", 0))
batch_to   = int(os.environ.get("BATCH_TO",   len(TESTS)))

if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
    try:
        existing = pd.read_csv(OUT_CSV)
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
else:
    existing = pd.DataFrame()
rows = existing.to_dict("records") if not existing.empty else []
done = {r["label"] for r in rows}

for idx in range(batch_from, min(batch_to, len(TESTS))):
    label, th, ta = TESTS[idx]
    if label in done:
        continue
    r = sweep_live_v2.run_one(th, ta, strict=False)
    rows.append({"label": label, "t_height": th, "t_adx": ta, **r})
    print(f'  [{idx+1}/{len(TESTS)}] {label:<28s} '
          f'T_h={th:>5.1f}/T_adx={ta:>5.1f}  '
          f'trades={r["trades"]:>4d}  '
          f'${r["net"]:>+10,.0f}  '
          f'wr={r["win_rate"]:>4.1f}%  pf={r["pf"]:>5.2f}  dd={r["max_dd_pct"]:>4.1f}%')

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'\nwrote {OUT_CSV} ({len(rows)} rows)')
