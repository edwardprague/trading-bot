# Pre-Session Fractal Metrics — Good Days vs Bad Days

**Date:** 2026-05-24
**Sample:** 2025-01-01 → 2025-12-31, GBPUSD 5m, v1 parity backtest
**Pre-session window:** last 6 N=2 fractals confirmed before 08:00 UTC each day

## Recommendation: build the new macro classifier around **pre-session swing height**, gated by **ADX ≥ 20**.

The combined rule `H1 ≥ 10 pips AND mean_ADX ≥ 20` catches **73.5%** of good days, passes only **31.2%** of bad days (selectivity gap = **+42.2 pp**), and preserves **83%** of the parity backtest's net P&L while removing **64%** of its losing-day exposure. That clears the 70%+ good-day-capture bar with material bad-day exclusion — a viable live macro gate.

## Sample

311 trading days in 2025 (every day with at least one N=2 fractal):

| Category | Days | Total day-P&L |
|---|---:|---:|
| Good (entry-day P&L > 0) | 49 | +$138,073 |
| Bad (entry-day P&L < 0) | 32 | −$19,761 |
| Neutral (no trades) | 230 | $0 |

The 207 parity-mode trades concentrated into 81 entry-days; the other 230 days were filtered out by existing gates (mostly the macro look-ahead). The good-vs-bad split is **49 : 32**, plenty of signal for ratio comparisons but small enough that thresholds should be treated as indicative rather than precision-tuned.

## Metric-by-metric separation (Cohen's d magnitude)

| Metric | Good mean | Bad mean | Δ (Good−Bad) | Cohen's d |
|---|---:|---:|---:|---:|
| **H3** (rolling-3 swing height) | 19.4 pips | 14.6 pips | +4.9 | **0.64** |
| **H1** (current swing height) | 24.2 pips | 14.0 pips | +10.3 | **0.58** |
| **H6** (rolling-6 swing height) | 16.6 pips | 13.6 pips | +3.0 | **0.56** |
| VD Low (mean v-dist of L pivots) | 10.6 pips | 7.4 pips | +3.2 | 0.38 |
| Lower-low % among L pairs | 58.9% | 48.1% | +10.8 | 0.28 |
| Mean ADX | 27.1 | 24.9 | +2.2 | 0.26 |
| VD High (mean v-dist of H pivots) | 8.4 pips | 6.9 pips | +1.5 | 0.19 |
| Lower-high % among H pairs | 55.7% | 55.2% | +0.5 | 0.01 |

**The dominant signal is swing height.** All three swing-height metrics (H1/H3/H6) show Cohen's d ≥ 0.56 — a moderate-to-large effect. On good days the pre-session swings are roughly **70% larger** than on bad days (H1 mean 24 vs 14 pips). This is the strongest single signal in the data.

Interestingly, the **lower-high %** metric — which one might naively expect to predict short-strategy success — has essentially zero separation (Cohen's d = 0.01). The "sequence of lower highs" is approximately equally common on good and bad days. This is consistent with the broader finding from the London-open analysis that day-to-day macro structure has limited persistence.

## Swing-trend distribution (H1 vs H3 vs H6 shape)

| Category | Expanding (H1>H3>H6) | Stable | Contracting (H1<H3<H6) |
|---|---:|---:|---:|
| Good | **49.0%** | 36.7% | 14.3% |
| Bad | 12.5% | 62.5% | 25.0% |

The "expanding" shape is **3.9× more common on good days**. This is the cleanest categorical signal in the analysis — and importantly, the "contracting" rate is **75% higher on bad days**, suggesting that decaying swings reliably correspond to deteriorating strategy conditions.

## Coarse pre-session regime (4-way label from last 6 fractals)

| Category | trending_down | ranging | transitioning | trending_up |
|---|---:|---:|---:|---:|
| Good | 22.4% | 12.2% | 53.1% | 12.2% |
| Bad | 9.4% | 12.5% | 65.6% | 12.5% |

Good days are 2.4× more likely to show `trending_down` pre-session — consistent with the short-only strategy's bias. But the absolute share is small (22%), and most days are `transitioning` regardless of outcome, so this signal alone is not strong enough to drive a classifier.

## Candidate rules — capture, exclusion, and P&L preservation

| Rule | Good-day pass | Bad-day pass | Selectivity | Good P&L kept | Bad P&L kept | Net |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (no gate) | 100% | 100% | — | +$138,073 | −$19,761 | +$118,312 |
| H1 ≥ 10 | 89.8% | 56.2% | +33.6 pp | +$127,313 | −$11,166 | +$116,147 |
| H1 ≥ 12 | 75.5% | 50.0% | +25.5 pp | +$101,523 | −$10,890 | +$90,634 |
| **H1 ≥ 10 AND ADX ≥ 20** | **73.5%** | **31.2%** | **+42.2 pp** | **+$105,631** | **−$7,077** | **+$98,554** |
| H3 ≥ 14 AND ADX ≥ 20 | 55.1% | 34.4% | +20.7 pp | +$75,543 | −$7,699 | +$67,844 |
| h_trend = expanding | 49.0% | 12.5% | +36.5 pp | +$59,112 | −$575 | +$58,537 |
| H1 ≥ 14 OR expanding | 67.3% | 40.6% | +26.7 pp | +$86,723 | −$6,993 | +$79,730 |

Two rules stand out for different reasons:

**`H1 ≥ 10 AND ADX ≥ 20` — the headline candidate.**
Highest selectivity that still clears the 70% good-day-capture bar. Excludes ~69% of bad days, keeps ~74% of good days, and preserves 83% of net P&L. This is what a viable live macro gate should look like — and the signal is cheap to compute: just the most recent fractal's swing height plus the 14-period ADX value already available in the indicator pipeline.

**`h_trend = expanding` — the highest-precision categorical filter.**
On its own it catches only half of good days, but it filters out **97%** of bad-day P&L (keeps only −$575 of the −$19,761 bad-day total). Useful as a confirmation signal layered on top of a primary rule rather than as a standalone gate. The mechanics are simple: at 08:00 UTC, check whether the most recent fractal's H1 > H3 > H6.

## Why this works (and why it works better than the London-open shift)

The pre-session signal exploits a different property than the macro classifier we validated last step. The daily macro classifier is essentially i.i.d. day-to-day (~30% next-day match, vs ~28% random) because its three signals — close-open displacement, within-day EMA-40 slope, N18 fractal structure — are all descriptions of *that one day's* completed price action. They don't extrapolate.

Swing height does extrapolate. The H1/H3/H6 features measure the **size of recent price moves**, and price-move size has well-documented short-horizon autocorrelation (volatility clustering). The pre-session swings before 08:00 UTC carry information about whether the rest of the day is likely to feature the kind of decisive, trade-worthy moves the strategy depends on, or the choppy small-move conditions that produce losing entries.

ADX adds a complementary signal: it measures trend strength rather than swing size. The combined rule effectively asks "are recent moves both *large* AND *directional* enough for the strategy to operate in?" — a question with material answer at 08:00 UTC, unlike "what will today's regime label be?"

## What to build

A new live macro classifier with this structure:

1. **At every new N=2 fractal**, the cBot computes H1 (height), H3 (rolling-3 mean of recent heights), and H6 (rolling-6 mean). These are the same calculations strategy_v2.py:1874-1889 already does — port directly.
2. **At entry time**, gate on a swing-height threshold (start with H1 ≥ 10) combined with the 14-period ADX (start with ADX ≥ 20). Both values are already available to a cTrader cBot via standard indicators.
3. **Optional second-stage confirmation**: require `H1 ≥ H3 ≥ H6` (the expanding-swings condition) for higher-precision entries. This collapses good-day capture but virtually eliminates bad-day exposure — useful as a "stricter mode" toggle.
4. **Calibrate the thresholds in Discovery's live mode.** The 10/20 numbers above are reasonable starting points but were not optimised — they should be search dimensions in the next Discovery sweep. The proper validation is then the Live Mode P&L on a 2026 out-of-sample window.

The macro-look-ahead classifier (the current parity gate) is contributing $259k of look-ahead P&L per the BD's Filter Impact Summary; the new gate won't recover all of that, but a 73.5%/31.2% rule preserving ~$98k of net P&L from the post-gate baseline of $118k is a clean improvement over live mode's −$14k floor.

## Files

- [analyse_pre_session.py](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/analyse_pre_session.py) — reproducible analysis script (`python3 analyse_pre_session.py` from project root)
- [pre_session_metrics_2025.csv](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/pre_session_metrics_2025.csv) — per-day metric values (one row per trading day)
- [PRE_SESSION_METRICS_REPORT.md](computer:///Users/edwardprimm/Documents/GitHub/trading-bot/PRE_SESSION_METRICS_REPORT.md) — this report
