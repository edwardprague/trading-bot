"""
macro_classifier_v2.py — Live-capable macro gate
=================================================
Replaces the parquet-label / streaming-classifier macro gate with a
classifier built on two real-time signals that are available at the
moment of fractal formation:

  • H1  — current N=2 fractal's swing height in pips (distance to the
          most recent opposite-kind pivot)
  • ADX — 14-period trend strength at the fractal bar

Optional strict mode additionally requires H1 ≥ H3 ≥ H6 (expanding
swings) where H3 / H6 are rolling means of the last 3 / 6 swing heights.

Empirical basis (see PRE_SESSION_METRICS_REPORT.md):
  • On 2025 GBPUSD parity backtest, swing height showed Cohen's d ≈ 0.6
    separating good-P&L days from bad-P&L days (vs ≈ 0 for the existing
    macro regime label).
  • H1 ≥ 10 AND ADX ≥ 20 captures 73.5% of good days, passes only 31.2%
    of bad days, preserves 83% of net P&L from the parity baseline.
  • Both inputs are zero-cost in a live cTrader cBot.

Public surface
--------------
  MacroClassifierV2(t_height, t_adx, strict_swings=False)
      Instantiate with thresholds. Stateful across fractals.

  .on_fractal(price, kind, adx)
      Call when an N=2 fractal is confirmed (price = fractal price,
      kind = "H" or "L", adx = ADX at the fractal bar).

  .check_gate(ts=None) → (passes: bool, reason: str)
      Same signature as strategy_v2._check_macro_regime. Pass-through
      while the classifier is warming up (no fractals seen yet).

  .reset()
      Clear state. Called at the start of each run_backtest so a
      single module-level instance can serve multiple backtests.

The gate reason is always "macro_regime" so the existing Filter Impact
Summary aggregation continues to work without changes. The classifier's
.last_fail_detail attribute carries the precise sub-reason
("height" / "adx" / "swings_contracting") for diagnostics.
"""

from __future__ import annotations

from typing import Optional


PIP = 10000  # non-JPY pip multiplier
HEIGHT_HISTORY_CAP = 100  # cap memory so a long backtest doesn't grow


class MacroClassifierV2:
    """Stateful live macro gate based on swing height + ADX."""

    __slots__ = (
        "t_height", "t_adx", "strict_swings",
        "_heights", "_last_h_price", "_last_l_price",
        "_h1", "_h3", "_h6", "_last_adx",
        "n_fractals", "n_checks", "n_blocked", "last_fail_detail",
    )

    # Starting defaults updated May 2026 from (10, 20) to (12, 32) after
    # the live-mode threshold sweep on 2025 GBPUSD (see
    # LIVE_MACRO_V2_HONEST_REPORT.md and discovery_live_v2_sweep_2025.csv).
    # The (10, 20) pair produced -$20k net P&L; (12, 32) is the best grid
    # point at +$18.3k net / PF 1.22 / 8.8% DD. The decomposition analysis
    # also confirmed both signals are required — neither H1 nor ADX is
    # profitable alone, but the conjunction is.
    def __init__(self, t_height: float = 12.0, t_adx: float = 32.0,
                 strict_swings: bool = False):
        self.t_height = float(t_height)
        self.t_adx = float(t_adx)
        self.strict_swings = bool(strict_swings)
        self.reset()

    def reset(self) -> None:
        """Clear all per-backtest state. Thresholds and strict-swings flag
        are preserved (they're set once per backtest run)."""
        self._heights: list = []
        self._last_h_price: Optional[float] = None
        self._last_l_price: Optional[float] = None
        self._h1: Optional[float] = None
        self._h3: Optional[float] = None
        self._h6: Optional[float] = None
        self._last_adx: Optional[float] = None
        self.n_fractals = 0
        self.n_checks = 0
        self.n_blocked = 0
        self.last_fail_detail = ""

    # ── Fractal stream input ─────────────────────────────────────────────

    def on_fractal(self, price: float, kind: str,
                   adx: Optional[float]) -> None:
        """Push a newly-confirmed N=2 fractal into the classifier.

        Mirrors strategy_v2.py:1874-1889 — height is the absolute price
        distance (in pips) to the most recent opposite-kind pivot, H3
        and H6 are rolling means of the last 3 / 6 non-null heights
        including the current one.
        """
        kind = (kind or "").upper()
        price = float(price)

        # Compute this fractal's height vs the most recent opposite-kind pivot.
        height: Optional[float] = None
        if kind == "H" and self._last_l_price is not None:
            height = abs(price - self._last_l_price) * PIP
        elif kind == "L" and self._last_h_price is not None:
            height = abs(price - self._last_h_price) * PIP

        if height is not None:
            self._heights.append(height)
            if len(self._heights) > HEIGHT_HISTORY_CAP:
                # Trim to keep memory bounded over long backtests; H3/H6
                # only look at the tail, so older values are irrelevant.
                self._heights = self._heights[-HEIGHT_HISTORY_CAP:]
            self._h1 = height
            tail3 = self._heights[-3:]
            tail6 = self._heights[-6:]
            self._h3 = sum(tail3) / len(tail3)
            self._h6 = sum(tail6) / len(tail6)

        # Update the most-recent pivot price for this kind (used to
        # measure the NEXT opposite-kind fractal's height).
        if kind == "H":
            self._last_h_price = price
        elif kind == "L":
            self._last_l_price = price

        # Record ADX at this fractal — this is the value the gate
        # consults until the next fractal arrives.
        try:
            self._last_adx = float(adx) if adx is not None else None
        except (TypeError, ValueError):
            self._last_adx = None

        self.n_fractals += 1

    # ── Gate decision ────────────────────────────────────────────────────

    def check_gate(self, ts=None) -> tuple[bool, str]:
        """Return (passes, reason).

        Returns (True, "") during warm-up (before the first H+L pair has
        produced a height). The "reason" string is unified to
        "macro_regime" so the existing Filter Impact Summary keeps its
        macro-vs-micro categorisation. The detailed cause of failure
        ("height" / "adx" / "swings_contracting") is exposed via
        .last_fail_detail for diagnostics.
        """
        self.n_checks += 1

        # Warmup: no fractals yet (or no opposite-kind pair to measure
        # height). Pass-through to avoid blocking every trade for the
        # first few bars of every run.
        if self._h1 is None or self._last_adx is None:
            self.last_fail_detail = ""
            return True, ""

        if self._h1 < self.t_height:
            self.n_blocked += 1
            self.last_fail_detail = "height"
            return False, "macro_regime"

        if self._last_adx < self.t_adx:
            self.n_blocked += 1
            self.last_fail_detail = "adx"
            return False, "macro_regime"

        if self.strict_swings:
            # Need all three values to evaluate the shape. _h3 and _h6
            # are only None when self._heights is empty — which is
            # already covered by the h1 None check above. Defensive
            # belt-and-braces:
            if self._h3 is None or self._h6 is None:
                self.last_fail_detail = ""
                return True, ""
            if not (self._h1 >= self._h3 >= self._h6):
                self.n_blocked += 1
                self.last_fail_detail = "swings_contracting"
                return False, "macro_regime"

        self.last_fail_detail = ""
        return True, ""

    # ── Introspection ────────────────────────────────────────────────────

    def state(self) -> dict:
        """Snapshot of current state — useful for tests and diagnostics."""
        return {
            "t_height":      self.t_height,
            "t_adx":         self.t_adx,
            "strict_swings": self.strict_swings,
            "h1":            self._h1,
            "h3":            self._h3,
            "h6":            self._h6,
            "adx":           self._last_adx,
            "n_fractals":    self.n_fractals,
            "n_checks":      self.n_checks,
            "n_blocked":     self.n_blocked,
        }
