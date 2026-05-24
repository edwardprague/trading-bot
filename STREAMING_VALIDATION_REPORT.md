# Streaming Classifier — Validation Report

**Date:** 2026-05-24
**Window:** GBPUSD 5m, 2025-01-01 → 2025-12-31
**Active version:** v1 (strategy_version=v2, short-only, regime gates on)

## Result: PARITY MODE LOGIC — PASS ✓

| Check                                       | Result |
|---------------------------------------------|:------:|
| Macro labels (312 trading days in window)   | 312 / 312 match, 0 mismatches |
| Micro coarse labels (15,529 fractal events) | 0 mismatches |
| Micro fine labels (15,529 fractal events)   | 0 mismatches |
| Backtest trade list (parity-gate vs parquet-gate) | 207 trades, exact match in order |
| Backtest P&L                                | +$129,440.91 both sides (delta $0.00) |

The streaming module, run in `"parity"` mode against the same bar data the parquet was built from, reproduces every regime label and every trade exactly. The rule logic in `regime_streaming.py` is a faithful re-implementation of `regime_analysis.py`'s classifier — ready to be translated to C#.

## Live-mode diff (informational, not a discrepancy)

Re-running the same backtest with `mode="live"` (cBot-faithful semantics — prior-day macro, running per-fractal sub-label, no look-ahead) on the same 2025 window:

| Mode      | Trades | P&L         |
|-----------|-------:|------------:|
| parity    |    207 | +$129,440.91 |
| live      |    164 |  -$14,399.35 |
| delta     |    −43 | −$143,840.26 |

This is the cost of removing the two look-ahead sources we identified earlier:

1. **Macro same-day lookup.** The parquet stores the day's macro label (computed using the day's full open-to-close range) and the backtest looks it up using the *entry day*. A live cBot only knows yesterday's label.
2. **Retroactive micro sub-label.** The parquet's fine label (`fast/medium/slow`, `narrow/medium/wide`) is computed once per period from the period's *final* aggregate and projected back onto every fractal in that period. Live, each fractal must use the running aggregate as of its own moment.

What this means practically: the v1 parameter set was Discovery-optimised against a backtest that includes both forms of look-ahead, so its absolute P&L is overstated relative to what's achievable live. Before forward-testing in cTrader, plan to either:

- Re-run Discovery with the streaming `"live"` mode as the gate, or
- Accept that v1's live performance will be materially below its backtest performance and treat the cBot test as a sanity check on mechanics rather than a P&L target.

## Files delivered

- `regime_streaming.py` — Self-contained classifier. Two modes (`"parity"` and `"live"`). Causal at the bar level. Frozen tercile thresholds default to the values stored in `regime_labels.parquet`. **This is the source for the C# port in step 2.**
- `validate_streaming.py` — Reproducible validation harness. Re-run any time with `python3 validate_streaming.py` from the project root.

## How to reproduce

```bash
cd trading-bot
source venv/bin/activate
python3 validate_streaming.py
```

The script:
1. Loads `data/regime_labels.parquet` (ground truth + frozen thresholds).
2. Builds a `StreamingRegimeClassifier(mode="parity")` over the same bar data.
3. Compares macro_by_date and per-fractal labels.
4. Runs the v2 backtest twice — once with the parquet gate, once with the streaming gate monkey-patched in — and compares trade lists.
5. Builds a second classifier in `mode="live"` and reports the live-vs-parity trade-list delta.

## Ready for step 2

The streaming module is the clean source for the C# port. Key surfaces a C# port must mirror:

- `_MacroDayState.finalize()` → end-of-day macro classifier (deterministic decision tree)
- `_classify_raw()` → 4-way coarse classifier (counts pairs in rolling H/L lookbacks)
- State-machine commit in `ingest()` → 2-consecutive-same-kind confirmation rule
- `_period_pips_per_bar()` / `_period_width_choppiness()` → period aggregate metrics
- `_speed_for_ppb()` / `_size_for_width()` → tercile bucketing against frozen constants

For the cBot, the `"live"` mode is the relevant target — the `"parity"` mode existed only to prove the rule logic against the parquet.
