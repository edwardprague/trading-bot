# Live Macro v2 — Honest Per-Entry Validation

**Date:** 2026-05-24
**Sample:** 2025-01-01 → 2025-12-31, GBPUSD 5m, v1 (short-only, EMA 133, SL 30 pips, RRR 1:1)
**Replaces:** the +$98k projection in `PRE_SESSION_METRICS_REPORT.md`

## What changed and why

The previous pre-session analysis projected the live macro v2 gate would preserve ~$98k of net P&L. The actual live_v2 backtest at the starting thresholds (T_h=10, T_adx=20) produced **−$20.3k**. That's a $118k gap — material enough that I stopped and re-examined the analysis before continuing.

The flaw was implicit conditioning. The previous analysis filtered the *output* of the parity backtest (207 trades already gated by the look-ahead macro filter) and asked which of those would survive H1+ADX. That answers "would H1+ADX preserve P&L if we kept the parity gate AND added H1+ADX on top?" — a question that's academic because we can't keep the look-ahead gate live. The real question is "what P&L does H1+ADX produce when it *replaces* the parity gate?" — and the answer requires a different population.

This report redoes the analysis on the right population: **every otherwise-eligible N=2 entry signal in 2025**, ignoring macro entirely (micro still on at the v1 defaults). 414 trades, of which 200 won and 214 lost. Net **−$27.8k**, win rate 48.3%. That's the universe live_v2 has to filter.

## Signal separation (per-entry, not per-day)

Cohen's d for winners (TP) vs losers (SL), measured at the fractal that triggered each entry:

| Metric | Winner mean | Loser mean | Winner median | Loser median | Cohen's d |
|---|---:|---:|---:|---:|---:|
| H1 | 17.8 pips | 14.9 pips | 13.0 | 12.0 | **0.17** |
| H3 | 26.5 pips | 20.8 pips | 19.1 | 18.3 | **0.15** |
| H6 | 25.7 pips | 19.8 pips | 18.0 | 17.6 | **0.12** |
| ADX (at fractal) | 30.8 | 28.9 | 28.4 | 26.6 | **0.16** |

These are all *small* effect sizes (Cohen's d < 0.2). The previous report had H1 at **0.58** and called swing height "dominant" — that number was an artifact of conditioning on parity macro labels, which were themselves correlated with the same instruments (price-action regimes that the daily classifier was already telling us about retrospectively).

The honest signal is weaker. There's still some separation — winning entries do happen on slightly larger swings and slightly higher ADX on average — but nothing close to the original Cohen's d ≈ 0.6 picture.

## Threshold sweep — net P&L per operating point

Sweep across `T_h ∈ {0, 5, 8, 10, 12, 15, 18, 20, 25, 30}` × `T_adx ∈ {0, 15, 18, 20, 22, 25, 28, 30, 33}` × `strict ∈ {off, on}`. Selected results:

| T_h | T_adx | strict | Trades retained | Net P&L | Win rate |
|---:|---:|:---:|---:|---:|---:|
| 0 | 0 | off | 414 (100%) | **−$27,848** | 48.3% |
| 10 | 20 | off | 370 (89%) | **−$20,311** | 49.2% |
| 12 | 22 | off | 155 (37%) | **+$13,668** | 56.8% |
| 12 | 30 | off | 99 (24%) | **+$12,895** | 59.6% |
| 12 | 33 | off | 74 (18%) | **+$14,873** | 63.5% |
| 15 | 33 | off | 57 (14%) | **+$7,962** | 59.6% |
| 20 | 25 | off | 63 (15%) | **+$6,252** | 57.1% |
| 25 | 22 | off | 49 (12%) | **+$6,500** | 59.2% |
| any | any | **ON** | varies | **all NEGATIVE** | — |

The full table is in `threshold_sweep_2025.csv` (180 combinations).

Three concrete findings:

1. **The starting defaults (T_h=10, T_adx=20) are too loose.** They pass 89% of trades through and barely improve on the unfiltered baseline.
2. **There's a sweet spot around T_h=12 with T_adx in the 22–33 range.** Net P&L lands between +$8k and +$15k on 60–170 trades, with win rates of 55–64%. The peak operating point in this sample is `T_h=12, T_adx=33 → +$14.9k on 74 trades`.
3. **Strict swings (H1 ≥ H3 ≥ H6) is harmful in every combination tested.** It cuts trade count so aggressively (down to 10–13 trades at typical thresholds) that the wins can't compensate for spread costs. Recommendation: leave the toggle in the UI but default OFF and document that current evidence is against turning it on.

## Realistic ceiling vs parity baseline

| Mode | Trades | Net P&L | Notes |
|---|---:|---:|---|
| Parity (look-ahead macro) | 207 | +$129k | Largely look-ahead premium; not reproducible live |
| **Best live_v2 in this sweep** | **74** | **+$14.9k** | T_h=12, T_adx=33 |
| Live_v2 at starting defaults (10, 20) | 370 | −$20.3k | Too loose |
| Unfiltered (no macro at all) | 414 | −$27.8k | Establishes baseline |

The honest ceiling on the live macro v2 gate is around **+$15k** for 2025 GBPUSD on the v1 parameter set — vs parity's +$129k. About **88% of parity's P&L is non-recoverable look-ahead premium**, with the remaining 12% available to a well-tuned live gate. This aligns with the broader findings from `LONDON_OPEN_VALIDATION_REPORT.md`: the macro classifier itself has near-zero next-day predictive power.

## Recommended updates to the build

1. **Update Discovery search ranges.** The current ranges (T_h: 5–30, T_adx: 15–35) are correct, but the *empirically interesting* region is narrower. Consider tightening to T_h: 10–25 and T_adx: 20–35 to focus the search where positive operating points actually exist. This is one-line in `discovery.py`.
2. **Update starting defaults in macro_classifier_v2.py.** Change `t_height=10.0 → 12.0` and `t_adx=20.0 → 22.0`. These are the documented "starting points" the UI / docs reference. Discovery will still search the full range; this just gives a sensible default for one-off manual runs.
3. **Keep strict_swings in the UI but mark it experimental.** Every sample point with strict on was negative in 2025 GBPUSD. We may find it helps in other instruments or other windows, but for now the toggle should default off and the UI hint should reflect this.
4. **Calibrate expectations.** The live gate is real but small. Net P&L on a year of 2025 GBPUSD is in the +$10k–$15k range, not +$100k. This is fine as a live-deployable strategy, but it should not be benchmarked against parity P&L.

## What about the full pipeline question?

This analysis isolates the macro gate. The bigger question — "should we ship a live macro gate at all, or just go macro-less in live mode?" — depends on the comparison:

| Setup | Net P&L (2025) |
|---|---:|
| No macro gate, parity micro | −$27.8k |
| **Live macro v2 at T_h=12, T_adx=33** | **+$14.9k** |
| Live macro v2 at T_h=10, T_adx=20 | −$20.3k |

A well-tuned live_v2 gate offers ~$43k of lift over no-macro-gate. That's meaningful — it makes the live strategy net-positive instead of net-negative. So the answer is: yes, ship it, but calibrate.

## Files

- [analyse_unfiltered_entries.py](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/analyse_unfiltered_entries.py) — the reproducible analysis script
- [unfiltered_entries_2025.csv](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/unfiltered_entries_2025.csv) — per-entry dataset (414 rows: trade record + fractal-time metrics)
- [threshold_sweep_2025.csv](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/threshold_sweep_2025.csv) — 180-combination sweep results
- [LIVE_MACRO_V2_HONEST_REPORT.md](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/LIVE_MACRO_V2_HONEST_REPORT.md) — this report
