# London-Open Macro Label — Pre-Build Validation

**Date:** 2026-05-24
**Instrument:** GBPUSD 5m, 2023-01-01 → 2025-12-31 (788 trading days with full coverage)
**Question:** Is a regime label computed at 8:00 UTC London open a meaningfully better predictor of today's actual completed macro label than yesterday's daily label?

## Recommendation: **Do not build live_v2 around this approach.**

The London-open shifted label gives **+5.4 percentage points** of lift over the prior-day baseline (34.5% vs 29.1%). That's well below the 70% threshold the task set as the bar for building, and only marginally above pure chance (28.4%). The deeper finding is that the daily macro classifier itself produces essentially i.i.d. labels day-to-day, which puts a structural ceiling on any "yesterday's information"-style approach.

## Methodology

Three labels per trading day D:

- **Ground truth** = `classify_macro_regime` applied to day D's calendar bars (midnight UTC → midnight UTC). This is what `regime_labels.parquet` stores and what parity mode sees.
- **Prior-day** = `classify_macro_regime` applied to day D-1's calendar bars. This is what live mode currently gates entries on.
- **Shifted-at-8UTC** = `classify_macro_regime` applied to bars from 8:00 UTC on D-1 to 8:00 UTC on D. The same classifier, unchanged, applied to a 24-hour window that ends at London session open.

The task's literal phrasing ("resample to 4h bars") would have required ≥40 4h bars (~6.7 days) just to satisfy the classifier's EMA-40 and N18 lookbacks, which means scaling its displacement thresholds well off-calibration. After confirming with you, I used 5m bars in a shifted 24-hour window — the 4h alignment shows up only as the boundary at 8:00 UTC (a 4h grid point). Sanity check: prior_day on day D matches ground_truth on day D-1 for 788 / 788 pairs (100%), proving the alignment is correct.

## Headline numbers

| Approach | Matches | Days | Match% |
|---|---:|---:|---:|
| Prior-day (live mode today) | 185 | 636 | **29.1%** |
| Shifted-at-8UTC (proposed)  | 267 | 775 | **34.5%** |
| Random baseline (∑ p²) | — | — | 28.4% |
| Theoretical max for prior-day | — | — | 30.3% |

The "theoretical max" for prior-day is the percentage of days where ground_truth doesn't change from the prior day — i.e., 1 − transition rate. The prior-day baseline of 29.1% is already at its ceiling.

## Day-to-day transition frequency

**549 / 788 day-pairs (69.7%)** see a regime change from the prior day. With ground-truth priors of:

| Regime | Share |
|---|---:|
| staircase_up | 32.1% |
| staircase_down | 30.2% |
| flat | 29.3% |
| strong_up | 4.8% |
| strong_down | 3.7% |

…the random next-day match rate is ∑ p² = 28.4%. The actual 30.3% same-as-yesterday rate (= 1 − 0.697) is only ~2 pp above pure chance, which means the daily macro classifier is producing essentially independent draws day to day. There's no meaningful persistence to exploit.

## Per-regime breakdown

**Prior-day** (rows = ground-truth regime):

| Ground truth | n | match% | Top confusion |
|---|---:|---:|---|
| flat | 179 | 28.5% | staircase_up (35.2%) |
| staircase_down | 205 | 30.2% | flat (32.7%) |
| staircase_up | 202 | 35.6% | staircase_down (30.7%) |
| strong_down | 22 | **0.0%** | staircase_up (40.9%) |
| strong_up | 28 | **0.0%** | staircase_down (35.7%) |

**Shifted-at-8UTC:**

| Ground truth | n | match% | Top confusion |
|---|---:|---:|---|
| flat | 220 | 43.2% | staircase_up (27.3%) |
| staircase_down | 237 | 30.4% | flat (40.1%) |
| staircase_up | 251 | 39.4% | flat (34.7%) |
| strong_down | 29 | **0.0%** | flat (48.3%) |
| strong_up | 38 | 2.6% | flat (42.1%) |

Two consistent patterns:

1. **Strong regimes are unpredictable from yesterday's data, regardless of method.** Both prior-day and shifted-at-8UTC achieve 0% match on strong_down and ~0% on strong_up. The strong-move classifier needs the full day's data — by definition it's about *this* day's price action, not a slow build-up that yesterday could have hinted at.
2. **Shifted-8UTC mostly helps "flat" days** (28.5% → 43.2%, the biggest improvement). Likely because eight hours of today's flat action visibly damps the noise that would otherwise classify yesterday differently.

## Time-of-day accuracy curve

Re-running the same comparison with the 24-hour window ending at different UTC hours:

| Window ends | Match% vs ground truth | Δ vs prior-day |
|---:|---:|---:|
| 0 UTC | 29.0% | −0.0 pp |
| 4 UTC | 29.4% | +0.3 pp |
| **8 UTC** (London open) | **34.5%** | **+5.4 pp** |
| 12 UTC | 37.5% | +8.4 pp |
| 16 UTC (NY close approaches) | 48.1% | +19.0 pp |
| 20 UTC | 73.6% | +44.5 pp |

The curve is monotone and intuitive: the closer the window's end approaches midnight, the more bars it shares with the calendar day, and the more it can "see" what ground truth will say. At 20 UTC the window contains 20 hours of today's bars and only 4 hours of yesterday's — the resulting label is mostly today's macro by construction.

The 70% match threshold the task set as the build bar **is not crossed until 20 UTC**, which isn't London open and uses ~83% of today's bars.

## Why the result is so weak

The macro classifier's three signals (close − open displacement, within-day EMA-40 slope, N18 fractal sequence) are all properties of the **completed day**. None of them tells you about the *next* day. A staircase_up day is a description of that one day's action — it doesn't imply tomorrow will also be staircase_up.

The earlier parity-vs-live divergence (+$129k parity → −$14k live) almost certainly reflects this same effect: parity-mode benefits because it knows today's actual classification at trade time, which is informational look-ahead. The macro gate's $259k "saved" P&L in the BD report is essentially the look-ahead premium — *not* a feature any look-back classifier can deliver.

## Possible directions if you still want a macro gate

If improving live-mode P&L through better regime gating is still the goal:

- **Use a fundamentally different macro signal** that has natural persistence (e.g., multi-day trend strength via ADX(14) on the daily timeframe, or a longer-EMA position filter). The current classifier's appeal — capturing intraday macro structure — is exactly what makes it useless for live look-ahead.
- **Drop the macro gate entirely** in live mode and rely on the micro classifier (which has a state machine and naturally persists across fractals) plus the entry signal. The earlier live-mode backtest with both gates active showed 164 trades / −$14k; running it with only the micro gate would isolate whether the macro gate is hurting or helping in live mode.
- **Optimise parameters under the new live mode** (Live Mode toggle on Discovery is already wired) and see what the search finds — maybe the optimal live-mode strategy doesn't gate on macro at all, or uses a much wider macro allow-list.

## Files

- [validate_london_open.py](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/validate_london_open.py) — reproducible analysis (`python3 validate_london_open.py` from project root)
- [LONDON_OPEN_VALIDATION_REPORT.md](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/LONDON_OPEN_VALIDATION_REPORT.md) — this report
